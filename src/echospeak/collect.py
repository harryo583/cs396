"""One-command data collection: sound check, participant prompting, recording,
clip cutting and dataset bookkeeping — all on the laptop's own speaker + mic.

Run ``python collect.py`` (or ``echospeak collect``). One run = one session:

1. Operator: confirm participant / session (IDs are suggested automatically).
2. Sound check (~3 s): verifies every probe tone is captured, nothing clips,
   and the microphone hears the GO beep. Problems are explained before the
   participant starts.
3. Participant: one instruction screen, a 3-trial practice (not saved), then
   ``reps`` blocks with every class once per block in random order. Breaks
   between blocks continue automatically after a short countdown.
4. Probe-off control: one short extra block with the probe silent.
5. Automatically: save ``session.wav``/``cues.csv``/``session.json``, cut clips,
   rebuild ``data/manifest.csv`` and append to ``data/sessions.csv``.

Keys (operator or participant): SPACE = continue, x = redo the last trial,
p = pause/resume, q = save and stop early.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import re
import sys
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import parse_device, require_sounddevice, write_wav
from .config import Config
from .probe import beep, probe_block, tone_snr_db

INSTRUCTIONS = {
    "REST": "stay still and relaxed, mouth closed",
    "CHEWING": "chew naturally, as if chewing gum",
    "YAWNING": "yawn (or imitate a yawn)",
    "TALKING": "say a short sentence OUT LOUD (anything)",
}
PRACTICE = ["NEXT", "CHEWING", "TALKING"]
SESSION_LOG_FIELDS = ["participant", "session", "date", "clips", "probe_off_clips", "distance_cm",
                      "condition", "min_tone_snr_db", "peak_level", "notes"]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
YELLOW, GREEN, CYAN, RED = "\033[33m", "\033[32m", "\033[36m", "\033[31m"


def instruction(label: str) -> str:
    return INSTRUCTIONS.get(label, f"silently mouth \"{label.lower()}\" — lips move, NO sound, no whisper")


# ----------------------------------------------------------------------------
# IDs and bookkeeping
# ----------------------------------------------------------------------------

def next_participant(raw: Path) -> str:
    used = {p.name for p in raw.glob("P*") if p.is_dir()}
    n = 1
    while f"P{n:02d}" in used:
        n += 1
    return f"P{n:02d}"


def next_session(raw: Path, participant: str) -> str:
    existing = [int(m.group(1)) for p in (raw / participant).glob("S*")
                if (m := re.fullmatch(r"S(\d+)", p.name)) and (p / "cues.csv").exists()]
    return f"S{max(existing, default=0) + 1}"


def _append_session_log(data_root: Path, row: dict) -> None:
    path = data_root / "sessions.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SESSION_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SESSION_LOG_FIELDS})


# ----------------------------------------------------------------------------
# Terminal UI
# ----------------------------------------------------------------------------

class _Keys:
    """Non-blocking single-key reader (POSIX terminals). Disabled when stdin is not a TTY."""

    def __init__(self):
        self.pressed: list[str] = []
        self._old = None
        self._stop = False
        self.enabled = sys.stdin.isatty() and os.name == "posix"
        if self.enabled:
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import select
        while not self._stop:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                self.pressed.append(sys.stdin.read(1).lower())

    def take(self) -> set[str]:
        keys, self.pressed = set(self.pressed), []
        return keys

    def wait(self, accept: set[str], timeout: float | None = None) -> str | None:
        """Wait for one of ``accept`` (or the timeout). Returns immediately without a TTY."""
        if not self.enabled:
            return None
        end = None if timeout is None else time.time() + timeout
        while end is None or time.time() < end:
            hit = self.take() & accept
            if hit:
                return hit.pop()
            time.sleep(0.03)
        return None

    def close(self):
        self._stop = True
        if self._old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


def _show(*lines: str, big: str | None = None, color: str = "", header: str = "") -> None:
    out = ["\033[2J\033[H"]
    if header:
        out.append(f"  {DIM}{header}{RESET}\n")
    out.append("\n\n")
    if big:
        out.append(f"        {color}{BOLD}{'  '.join(big)}{RESET}\n\n")
    out += [f"        {line}\n" for line in lines]
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _progress(done: int, total: int, seconds_left: float) -> str:
    width = 30
    fill = int(width * done / max(total, 1))
    return f"[{'#' * fill}{'-' * (width - fill)}] {done}/{total}   about {max(1, round(seconds_left / 60))} min left"


# ----------------------------------------------------------------------------
# Audio engine: continuous full-duplex stream (probe out, microphone in)
# ----------------------------------------------------------------------------

class _Engine:
    def __init__(self, cfg: Config, input_device=None, output_device=None, channel: int = 0):
        self.sd = require_sounddevice()
        self.cfg, self.sr, self.channel = cfg, cfg.probe.sample_rate, channel
        self.pos = 0
        self.probe_on = True
        self.sounds: list[tuple[int, np.ndarray]] = []
        self.chunks: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.xruns = 0
        self.stream = self.sd.Stream(samplerate=self.sr, blocksize=1024, dtype="float32", latency="high",
                                     channels=(channel + 1, 1), callback=self._callback,
                                     device=(input_device, output_device))

    def _callback(self, indata, outdata, frames, _time, status):
        if status:
            self.xruns += 1
        start = self.pos
        out = probe_block(self.cfg.probe, start, frames) if self.probe_on else np.zeros(frames, np.float32)
        with self.lock:
            keep = []
            for at, snd in self.sounds:
                a, b = max(at, start), min(at + len(snd), start + frames)
                if a < b:
                    out[a - start: b - start] += snd[a - at: b - at]
                if at + len(snd) > start + frames:
                    keep.append((at, snd))
            self.sounds = keep
        outdata[:, 0] = np.clip(out, -1, 1)
        self.chunks.append(indata[:, self.channel].copy())
        self.pos = start + frames

    def schedule(self, sound: np.ndarray, at: int) -> None:
        with self.lock:
            self.sounds.append((at, sound))

    def wait_until(self, sample: int) -> None:
        while self.pos < sample:
            time.sleep(0.002)

    def seconds(self, s: float) -> int:
        return int(round(s * self.sr))

    def recording(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def sound_check(engine: _Engine) -> dict:
    """Record ~2.5 s with the probe and one beep; report tone SNR, peak and beep detection."""
    from .segment import find_beep
    start = engine.pos
    at = start + engine.seconds(1.0)
    engine.schedule(beep(engine.sr), at)
    engine.wait_until(start + engine.seconds(2.6))
    rec = engine.recording()
    seg = rec[start: start + engine.seconds(2.5)] if len(rec) >= start + engine.seconds(2.5) else rec[-engine.seconds(2.5):]
    snr = tone_snr_db(seg, engine.sr, engine.cfg.probe)
    _, beep_q = find_beep(rec, engine.sr, at, engine.seconds(0.3))
    return {"tone_snr_db": [round(s, 1) for s in snr], "min_tone_snr_db": round(min(snr), 1),
            "peak_level": round(float(np.abs(seg).max()), 3), "beep_heard": bool(beep_q > 4)}


def _sound_check_problems(result: dict) -> list[str]:
    problems = []
    if result["min_tone_snr_db"] < 20:
        problems.append(f"Probe tones are weak (min SNR {result['min_tone_snr_db']} dB, need >= 20). "
                        "Raise the laptop volume (~70-80%), use the built-in speaker (no Bluetooth/headphones), "
                        "disable mic effects (macOS Control Center > Mic Mode: Standard), or pick another "
                        "--input-device/--output-device (see 'echospeak devices').")
    if result["peak_level"] > 0.98:
        problems.append("The microphone is clipping. Lower the laptop volume or the input gain.")
    if not result["beep_heard"]:
        problems.append("The microphone did not hear the GO beep; clip timing will rely on logged cue times.")
    return problems


# ----------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------

def collect(cfg: Config, data_root: str | Path = "data", participant: str | None = None,
            session: str | None = None, *, reps: int = 5, probe_off_reps: int = 1, practice: bool = True,
            distance_cm: float = 10.0, condition: str = "normal", notes: str = "",
            input_device: str | None = None, output_device: str | None = None, channel: int = 0,
            labels: list[str] | None = None, confirm: bool = True, prep_s: float = 1.2,
            gap_s: float = 0.9, break_s: float = 10.0, log=print) -> Path:
    data_root = Path(data_root)
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    labels = labels or cfg.classes

    # --- operator setup -------------------------------------------------------
    if participant is None:
        suggestion = next_participant(raw)
        participant = (input(f"Participant ID [{suggestion}]: ").strip() if confirm else "") or suggestion
    session = session or next_session(raw, participant)
    out = raw / participant / session
    if (out / "cues.csv").exists():
        raise SystemExit(f"{out} already exists; pick another --session")
    per_trial = prep_s + cfg.data.clip_post_s + 0.1 + gap_s + 0.2
    n_main, n_off = reps * len(labels), probe_off_reps * len(labels)
    total_s = (n_main + n_off + (len(PRACTICE) if practice else 0)) * per_trial + max(reps - 1, 0) * break_s + 20
    log(f"\nSession {participant}/{session}: {n_main} trials + {n_off} probe-off trials, "
        f"about {total_s / 60:.0f} min.\nData folder: {out}\n"
        f"Seat the participant with their mouth ~{distance_cm:g} cm from the laptop microphone.")
    if confirm:
        input("Press Enter to start the sound check... ")

    rng = random.Random(zlib.crc32(f"{participant}/{session}".encode()))
    engine = _Engine(cfg, parse_device(input_device), parse_device(output_device), channel)
    meta = {
        "participant": participant, "session": session, "sample_rate": engine.sr,
        "probe": "on", "tones_hz": cfg.probe.tones_hz, "distance_cm": distance_cm, "condition": condition,
        "orientation": "laptop", "device": platform.node(), "notes": notes, "reps": reps,
        "probe_off_reps": probe_off_reps, "labels": labels, "clip_pre_s": cfg.data.clip_pre_s,
        "clip_post_s": cfg.data.clip_post_s, "created": datetime.now().isoformat(timespec="seconds"),
    }
    rows: list[dict] = []
    keys = None
    out.mkdir(parents=True, exist_ok=True)
    try:
        engine.stream.start()
        time.sleep(0.5)

        # --- sound check ------------------------------------------------------
        while True:
            log("Sound check (stay quiet for 3 s)...")
            check = sound_check(engine)
            problems = _sound_check_problems(check)
            log(f"  tone SNR {check['tone_snr_db']} dB, peak {check['peak_level']}, "
                f"beep heard: {check['beep_heard']}")
            if not problems:
                log("  OK")
                break
            for p in problems:
                log(f"  {RED}! {p}{RESET}")
            if not confirm:
                break
            ans = input("[r]etry check / [c]ontinue anyway / [q]uit: ").strip().lower()
            if ans.startswith("c"):
                break
            if ans.startswith("q"):
                raise SystemExit("aborted at sound check")
        meta["sound_check"] = check

        keys = _Keys()
        bp = beep(engine.sr)

        def run_trial(label: str, header: str, status: str) -> str:
            _show(f"Get ready:  {instruction(label)}", big=label, color=YELLOW, header=header)
            go = engine.pos + engine.seconds(prep_s)
            engine.schedule(bp, go)
            engine.wait_until(go)
            _show(instruction(label), big="GO", color=GREEN, header=header)
            engine.wait_until(go + engine.seconds(cfg.data.clip_post_s + 0.1))
            _show("", big="relax", color=DIM, header=header)
            time.sleep(gap_s + rng.uniform(0, 0.4))
            pressed = keys.take()
            if "x" in pressed and status == "ok":
                status = "discarded"
            rows.append({"trial": len(rows) + 1, "label": label, "go_sample": go,
                         "probe": "on" if engine.probe_on else "off", "status": status})
            if "p" in pressed:
                _show("Paused. Press p to continue.", big="PAUSE", header=header)
                keys.wait({"p"})
            if "q" in pressed:
                raise KeyboardInterrupt
            return status

        # --- participant instructions + practice --------------------------------
        _show(f"{BOLD}How this works{RESET}",
              f"- Sit comfortably, mouth about {distance_cm:g} cm (a hand's width) from the laptop microphone.",
              "- A word appears: get ready.  GO + beep: do it ONCE, then relax.",
              f"- Commands ({', '.join(cfg.data.commands)}): move your lips as if saying the word,",
              "  but make NO sound and do not whisper.",
              "- Other prompts (rest, chew, yawn, talk) say exactly what to do.",
              "- Keep your head still between prompts. Mistake? Press x and it will be repeated.",
              "", f"{CYAN}Press SPACE to start{' a short practice' if practice else ''}.{RESET}",
              big="Welcome", color=CYAN)
        keys.wait({" ", "\n"})
        if practice:
            for i, label in enumerate(PRACTICE):
                run_trial(label, f"Practice {i + 1}/{len(PRACTICE)} (not saved)", "practice")
            _show("Great! The real recording starts now.", "", f"{CYAN}Press SPACE when ready.{RESET}",
                  big="Ready?", color=CYAN)
            keys.wait({" ", "\n"})

        # --- main blocks -----------------------------------------------------------
        def run_blocks(n_blocks: int, part: str, total: int, done_before: int) -> None:
            done = done_before
            for block in range(n_blocks):
                queue = list(labels)
                rng.shuffle(queue)
                while queue:
                    label = queue.pop(0)
                    left = (total - done) * per_trial
                    header = f"{part}  {_progress(done, total, left)}   (x = redo, p = pause)"
                    if run_trial(label, header, "ok") == "discarded":
                        queue.append(label)
                    else:
                        done += 1
                if block < n_blocks - 1:
                    for remaining in range(int(break_s), 0, -1):
                        _show(f"Block {block + 1} of {n_blocks} done. Short break.",
                              f"{CYAN}Continuing in {remaining} s — press SPACE to continue now.{RESET}",
                              big="Break", color=CYAN, header=f"{part}  {_progress(done, total, (total - done) * per_trial)}")
                        if keys.wait({" ", "\n"}, timeout=1.0) or not keys.enabled:
                            break

        total = n_main + n_off
        run_blocks(reps, "Main part", total, 0)
        if probe_off_reps:
            _show("Almost done: one last short round (about 1 minute).", "",
                  f"{CYAN}Press SPACE to continue.{RESET}", big="Last part", color=CYAN)
            keys.wait({" ", "\n"}, timeout=break_s)
            engine.probe_on = False   # control condition: same task, no probe
            time.sleep(1.0)
            run_blocks(probe_off_reps, "Last part", total, n_main)
            engine.probe_on = True
        _show("Thank you, you're done!", big="Done", color=GREEN)
    except KeyboardInterrupt:
        log("\nStopped early; saving everything recorded so far.")
    finally:
        time.sleep(0.5)
        engine.stream.stop()
        engine.stream.close()
        if keys is not None:
            keys.close()

    # --- save, cut clips, update the dataset index ---------------------------------
    write_wav(out / "session.wav", engine.recording(), engine.sr, pcm16=True)
    with (out / "cues.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "label", "go_sample", "probe", "status"])
        w.writeheader()
        w.writerows(rows)
    meta["xruns"] = engine.xruns
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    from .data import index_dataset
    from .segment import segment_session
    clips = segment_session(out, cfg, log=log)
    index_dataset(raw, data_root / "manifest.csv")
    n_off_clips = sum(c["probe"] == "off" for c in clips)
    _append_session_log(data_root, {
        "participant": participant, "session": session, "date": meta["created"],
        "clips": len(clips) - n_off_clips, "probe_off_clips": n_off_clips, "distance_cm": distance_cm,
        "condition": condition, "min_tone_snr_db": meta.get("sound_check", {}).get("min_tone_snr_db", ""),
        "peak_level": meta.get("sound_check", {}).get("peak_level", ""), "notes": notes})
    if engine.xruns:
        log(f"{RED}WARNING: {engine.xruns} audio dropouts were reported; consider re-recording.{RESET}")
    log(f"Saved {len(clips) - n_off_clips} clips (+{n_off_clips} probe-off) to {out}; "
        f"updated {data_root / 'manifest.csv'} and {data_root / 'sessions.csv'}.")
    return out
