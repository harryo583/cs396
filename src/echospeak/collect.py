"""One-command, self-paced data collection on the laptop's own speaker + microphone.

Run ``python data_collection/collect.py`` (or ``echospeak collect``). One run = one session:

1. Operator: confirm participant / session (IDs are suggested automatically).
2. Sound check (~3 s): every probe tone captured, no clipping, GO beep heard.
3. Participant: one instruction screen and a short practice (not saved).
4. Takes. A *take* records ``group_size`` repetitions of one class after the
   participant presses SPACE (nothing is recorded while they get ready):
   * commands and YAWNING are *paced*: a beep every ``interval`` s (6 s for
     YAWNING); one repetition per beep, mouth closed in between. Each beep = one clip.
   * REST, CHEWING and TALKING are *continuous*: the activity lasts
     ``group_size`` x clip length and is cut into ``group_size`` clips.
   Each class's ``reps`` are split into several takes spread over the session
   (round r holds the r-th take of every class, shuffled), so slow drift in
   posture or distance is not confounded with the class.
5. Probe-off control: one short take per class with the probe silent.
6. Automatically: save ``session.wav``/``cues.csv``/``session.json``, cut clips,
   rebuild ``data/manifest.csv`` and append to ``data/sessions.csv``.

Keys: SPACE = start the take,  x = redo (the current take, or the previous one
when pressed on the "get ready" screen),  q = save and stop.

Clips of one take are strongly correlated, so every clip records its ``take``
and data splits keep whole takes together (see ``data.make_folds``).
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import parse_device, require_sounddevice, write_wav
from .config import Config
from .probe import beep, probe_block, tone_snr_db

CONTINUOUS = {"REST", "CHEWING", "TALKING"}
CLASS_INTERVAL_S = {"YAWNING": 6.0}
INSTRUCTIONS = {
    "REST": "stay still and relaxed, mouth closed, until the second beep",
    "CHEWING": "chew continuously (as if chewing gum) until the second beep",
    "TALKING": "talk OUT LOUD continuously (anything) until the second beep",
    "YAWNING": "yawn (or imitate a yawn) once at each beep",
}
PRACTICE = [("NEXT", 3), ("CHEWING", 2)]
COUNTDOWN_S = 1.5
SESSION_LOG_FIELDS = ["participant", "session", "date", "device", "clips", "probe_off_clips", "distance_cm",
                      "condition", "min_tone_snr_db", "peak_level", "notes"]
CUE_FIELDS = ["trial", "label", "go_sample", "probe", "status", "take", "beep"]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
YELLOW, GREEN, CYAN, RED = "\033[33m", "\033[32m", "\033[36m", "\033[31m"


def instruction(label: str) -> str:
    return INSTRUCTIONS.get(label, f"silently mouth \"{label.lower()}\" once at each beep — NO sound, "
                                   "no whisper; close your mouth in between")


# ----------------------------------------------------------------------------
# Repetition plan
# ----------------------------------------------------------------------------

@dataclass
class Take:
    label: str
    n: int                 # clips in this take
    continuous: bool
    interval_s: float      # spacing of clips (paced: beep interval; continuous: clip length)
    probe_on: bool = True

    def duration_s(self, clip_s: float) -> float:
        return COUNTDOWN_S + self.n * self.interval_s + (0.0 if self.continuous else clip_s - self.interval_s) + 0.5


def reps_plan(labels: list[str], reps: int, class_reps: dict[str, int] | None = None) -> dict[str, int]:
    """Repetitions per class: ``reps`` for every class, overridden per class by ``class_reps``."""
    class_reps = {k.upper(): int(v) for k, v in (class_reps or {}).items()}
    unknown = set(class_reps) - set(labels)
    if unknown:
        raise ValueError(f"class_reps has unknown classes {sorted(unknown)}; known: {labels}")
    if any(v < 0 for v in class_reps.values()) or reps < 0:
        raise ValueError("repetition counts must be >= 0")
    return {lab: class_reps.get(lab, reps) for lab in labels}


def parse_class_reps(items: list[str] | None) -> dict[str, int]:
    """Parse ``["NEXT=8", "yawning=2"]`` into ``{"NEXT": 8, "YAWNING": 2}``."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--class-reps expects LABEL=N, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip().upper()] = int(v)
    return out


def make_take(label: str, n: int, interval_s: float, clip_s: float, probe_on: bool = True) -> Take:
    if label in CONTINUOUS:
        return Take(label, n, True, clip_s, probe_on)
    return Take(label, n, False, CLASS_INTERVAL_S.get(label, interval_s), probe_on)


def build_takes(plan: dict[str, int], group_size: int, interval_s: float, clip_s: float,
                rng: random.Random, probe_on: bool = True) -> list[Take]:
    """Split each class's repetitions into takes of ``group_size`` and spread them:
    round r = the r-th take of every class, shuffled; no class twice in a row."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    per_class = {}
    for lab, reps in plan.items():
        sizes = [group_size] * (reps // group_size) + ([reps % group_size] if reps % group_size else [])
        per_class[lab] = [make_take(lab, s, interval_s, clip_s, probe_on) for s in sizes]
    takes: list[Take] = []
    for r in range(max((len(v) for v in per_class.values()), default=0)):
        round_ = [v[r] for v in per_class.values() if len(v) > r]
        rng.shuffle(round_)
        if takes and len(round_) > 1 and round_[0].label == takes[-1].label:
            round_[0], round_[1] = round_[1], round_[0]
        takes += round_
    return takes


def describe_plan(plan: dict[str, int]) -> str:
    return ", ".join(f"{lab} x{n}" for lab, n in plan.items() if n) + f"  = {sum(plan.values())} clips"


# ----------------------------------------------------------------------------
# IDs and bookkeeping
# ----------------------------------------------------------------------------

def laptop_model() -> str:
    """Best-effort hardware model string (e.g. 'Mac15,12'), used to group data by laptop."""
    import subprocess
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True,
                                  timeout=5).stdout.strip() or platform.node()
        if sys.platform.startswith("linux"):
            return Path("/sys/class/dmi/id/product_name").read_text().strip()
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_ComputerSystem).Model"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return out or platform.node()
    except Exception:  # pragma: no cover - platform dependent
        pass
    return platform.node()


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

    def wait(self, accept: set[str], timeout: float | None = None, default: str | None = None) -> str | None:
        """Wait for one of ``accept``; without a TTY return ``default`` immediately."""
        if not self.enabled:
            return default
        end = None if timeout is None else time.time() + timeout
        while end is None or time.time() < end:
            hit = self.take() & accept
            if hit:
                return sorted(hit)[0]
            time.sleep(0.03)
        return default

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
    return f"[{'#' * fill}{'-' * (width - fill)}] {done}/{total} clips   about {max(1, round(seconds_left / 60))} min left"


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
    seg = rec[start: start + engine.seconds(2.5)]
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
            session: str | None = None, *, reps: int = 10, class_reps: dict[str, int] | None = None,
            group_size: int = 5, interval_s: float = 3.0, probe_off_reps: int = 1, practice: bool = True,
            distance_cm: float = 10.0, condition: str = "normal", notes: str = "",
            input_device: str | None = None, output_device: str | None = None, channel: int = 0,
            labels: list[str] | None = None, confirm: bool = True, log=print) -> Path:
    data_root = Path(data_root)
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    labels = labels or cfg.classes
    clip_s = cfg.data.clip_pre_s + cfg.data.clip_post_s
    if interval_s < 2.0:
        raise ValueError("interval_s must be >= 2 s (one silent repetition takes ~1 s plus rest)")

    # --- operator setup -------------------------------------------------------
    if participant is None:
        suggestion = next_participant(raw)
        participant = (input(f"Participant ID [{suggestion}]: ").strip() if confirm else "") or suggestion
    session = session or next_session(raw, participant)
    out = raw / participant / session
    if (out / "cues.csv").exists():
        raise SystemExit(f"{out} already exists; pick another --session")
    plan = reps_plan(labels, reps, class_reps)
    rng = random.Random(zlib.crc32(f"{participant}/{session}".encode()))
    takes = build_takes(plan, group_size, interval_s, clip_s, rng)
    takes += build_takes({lab: probe_off_reps for lab in labels}, group_size, interval_s, clip_s, rng,
                         probe_on=False)
    n_main = sum(plan.values())
    n_off = probe_off_reps * len(labels)
    est = sum(t.duration_s(clip_s) + 3.0 for t in takes)   # +3 s per take for reading / pressing SPACE
    log(f"\nSession {participant}/{session}: {len(takes)} takes, about {est / 60:.0f} min.\n"
        f"  main part:  {describe_plan(plan)}  (groups of {group_size})\n"
        f"  probe-off:  {n_off} clips\n"
        f"  data folder: {out}\n"
        f"Seat the participant with their mouth ~{distance_cm:g} cm from the laptop microphone.")
    if confirm:
        input("Press Enter to start the sound check... ")

    engine = _Engine(cfg, parse_device(input_device), parse_device(output_device), channel)
    meta = {
        "participant": participant, "session": session, "sample_rate": engine.sr,
        "probe": "on", "tones_hz": cfg.probe.tones_hz, "distance_cm": distance_cm, "condition": condition,
        "orientation": "laptop", "device": laptop_model(), "hostname": platform.node(),
        "notes": notes, "reps_per_class": plan, "group_size": group_size, "interval_s": interval_s,
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
        runner = _TakeRunner(engine, cfg, keys, rows)

        # --- instructions + practice ----------------------------------------------
        _show(f"{BOLD}How this works{RESET}",
              f"- Sit comfortably, mouth about {distance_cm:g} cm (a hand's width) from the laptop microphone.",
              "- Each screen shows a word and how many times to do it. Get ready, then press SPACE.",
              "- After a 3-2-1 countdown, do it ONCE AT EACH BEEP. Close your mouth between beeps.",
              f"- Words ({', '.join(cfg.data.commands)}): move your lips as if saying the word,",
              "  but make NO sound and do not whisper.",
              "- Rest / chew / talk: keep doing it from the first beep until the second beep.",
              "- Take your time between screens. Mistake? Press x and that part is repeated.",
              "", f"{CYAN}Press SPACE to start{' a short practice' if practice else ''}.{RESET}",
              big="Welcome", color=CYAN)
        keys.wait({" ", "\n"}, default=" ")
        if practice:
            for label, n in PRACTICE:
                if label in labels:
                    runner.run(make_take(label, n, interval_s, clip_s), "Practice (not saved)", "practice")
            _show("Great! The real recording starts now.", "", f"{CYAN}Press SPACE when ready.{RESET}",
                  big="Ready?", color=CYAN)
            keys.wait({" ", "\n"}, default=" ")

        # --- takes ---------------------------------------------------------------------
        queue = list(takes)
        total = n_main + n_off
        done = 0
        last: tuple[Take, list[dict]] | None = None
        while queue:
            take = queue.pop(0)
            left = sum(t.duration_s(clip_s) + 3.0 for t in [take, *queue])
            part = "Main part" if take.probe_on else "Last part (almost done)"
            header = f"{part}  {_progress(done, total, left)}"
            result, take_rows = runner.run(take, header, "ok", can_redo_previous=last is not None)
            if result == "redo_previous" and last is not None:
                for r in last[1]:
                    r["status"] = "discarded"
                done -= len(last[1])
                queue[:0] = [last[0], take]
                last = None
                continue
            if result == "redo":
                queue.insert(0, take)
                continue
            done += len(take_rows)
            last = (take, take_rows)
            if result == "quit":
                raise KeyboardInterrupt
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
        w = csv.DictWriter(fh, fieldnames=CUE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    meta["xruns"] = engine.xruns
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    clips = finish_session(out, cfg, data_root, log=log)
    if engine.xruns:
        log(f"{RED}WARNING: {engine.xruns} audio dropouts were reported; consider re-recording.{RESET}")
    n_off_clips = sum(c["probe"] == "off" for c in clips)
    log(f"Saved {len(clips) - n_off_clips} clips (+{n_off_clips} probe-off) to {out}; "
        f"updated {data_root / 'manifest.csv'} and {data_root / 'sessions.csv'}.")
    return out


class _TakeRunner:
    """Runs one take: get-ready screen (waits for SPACE), countdown, beeps, cue logging."""

    def __init__(self, engine: _Engine, cfg: Config, keys: _Keys, rows: list[dict]):
        self.engine, self.cfg, self.keys, self.rows = engine, cfg, keys, rows
        self.beep = beep(engine.sr)
        self.n_takes = 0

    def run(self, take: Take, header: str, status: str, can_redo_previous: bool = False):
        e, cfg = self.engine, self.cfg
        what = (f"{take.label}  for {take.n * take.interval_s:.0f} s" if take.continuous
                else f"{take.label}  x {take.n}")
        extra = f"one beep every {take.interval_s:g} s" if not take.continuous else "starts and ends with a beep"
        redo_hint = "   x = redo the previous one" if can_redo_previous else ""
        self.keys.take()
        _show(f"{instruction(take.label)}  ({extra})", "",
              f"{CYAN}Get ready, then press SPACE.{RESET}{DIM}{redo_hint}   q = save and stop{RESET}",
              big=what, color=YELLOW, header=header)
        key = self.keys.wait({" ", "\n", "x", "q"} if can_redo_previous else {" ", "\n", "q"}, default=" ")
        if key == "x":
            return "redo_previous", []
        if key == "q":
            return "quit", []
        e.probe_on = take.probe_on
        t0 = e.pos
        start = t0 + e.seconds(COUNTDOWN_S)
        for i, c in enumerate(("3", "2", "1")):
            e.wait_until(t0 + e.seconds(i * COUNTDOWN_S / 3))
            _show(instruction(take.label), big=c, color=YELLOW, header=header)
        step = e.seconds(take.interval_s)
        if take.continuous:
            gos = [start + e.seconds(cfg.data.clip_pre_s) + k * step for k in range(take.n)]
            end = start + take.n * step
            e.schedule(self.beep, start)
            e.schedule(self.beep, end)
            e.wait_until(start)
            while e.pos < end:
                left = (end - e.pos) / e.sr
                _show(instruction(take.label), "", f"{left:.0f} s left", big=take.label, color=GREEN, header=header)
                e.wait_until(min(end, e.pos + e.seconds(1.0)))
            finish = end + e.seconds(0.3)
        else:
            gos = [start + k * step for k in range(take.n)]
            for g in gos:
                e.schedule(self.beep, g)
            for k, g in enumerate(gos):
                e.wait_until(g)
                _show(instruction(take.label), "", f"{k + 1} of {take.n}", big=take.label, color=GREEN,
                      header=header)
            finish = gos[-1] + e.seconds(cfg.data.clip_post_s + 0.1)
        e.wait_until(finish)
        e.probe_on = True
        pressed = self.keys.take()
        self.n_takes += 1
        take_id = f"t{self.n_takes:03d}"
        if "x" in pressed:
            status = "discarded"
        new = [{"trial": len(self.rows) + i + 1, "label": take.label, "go_sample": g,
                "probe": "on" if take.probe_on else "off", "status": status, "take": take_id,
                "beep": "0" if take.continuous else "1"} for i, g in enumerate(gos)]
        self.rows.extend(new)
        if "x" in pressed:
            return "redo", new
        return ("quit" if "q" in pressed else "ok"), new


def finish_session(out: Path, cfg: Config, data_root: Path, *, log=print, reindex: bool = True) -> list[dict]:
    """Cut clips for a saved session, log it in ``sessions.csv`` and rebuild the manifest."""
    from .data import index_dataset
    from .segment import segment_session
    meta = json.loads((out / "session.json").read_text())
    clips = segment_session(out, cfg, log=log)
    n_off = sum(c["probe"] == "off" for c in clips)
    check = meta.get("sound_check", {})
    _append_session_log(data_root, {
        "participant": meta["participant"], "session": meta["session"], "date": meta.get("created", ""),
        "device": meta.get("device", ""), "clips": len(clips) - n_off, "probe_off_clips": n_off,
        "distance_cm": meta.get("distance_cm", ""), "condition": meta.get("condition", ""),
        "min_tone_snr_db": check.get("min_tone_snr_db", ""), "peak_level": check.get("peak_level", ""),
        "notes": meta.get("notes", "")})
    if reindex:
        index_dataset(data_root / "raw", data_root / "manifest.csv")
    return clips


def merge_sessions(src: str | Path, cfg: Config, data_root: str | Path = "data", *, log=print) -> list[Path]:
    """Import sessions recorded with ``data_collection_portable`` (a folder or a .zip).

    Copies each session's ``session.json``, ``cues.csv`` and ``session.wav`` into
    ``data/raw/<participant>/<session>/``, re-cuts the clips with this repository's
    config (so clip timing is always consistent), logs them and rebuilds the manifest.
    Sessions already merged are skipped; conflicting IDs are reported and skipped.
    """
    import shutil
    import tempfile
    import zipfile
    data_root = Path(data_root)
    src = Path(src)
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp.name)
        src = Path(tmp.name)
    merged = []
    try:
        found = sorted(p.parent for p in src.rglob("session.json")
                       if (p.parent / "cues.csv").exists() and (p.parent / "session.wav").exists())
        if not found:
            log(f"no recorded sessions found under {src}")
        for sess in found:
            meta = json.loads((sess / "session.json").read_text())
            name = f"{meta['participant']}/{meta['session']}"
            tones = [float(f) for f in meta.get("tones_hz", [])]
            if tones != [float(f) for f in cfg.probe.tones_hz]:
                log(f"SKIP {name}: recorded with probe tones {tones}, config expects {cfg.probe.tones_hz}")
                continue
            dest = data_root / "raw" / meta["participant"] / meta["session"]
            if (dest / "session.json").exists():
                old = json.loads((dest / "session.json").read_text())
                same = (old.get("created"), old.get("device")) == (meta.get("created"), meta.get("device"))
                log(f"SKIP {name}: {'already merged' if same else 'a different session with this ID already exists'}")
                continue
            dest.mkdir(parents=True)
            for f in ("session.json", "cues.csv", "session.wav"):
                shutil.copy2(sess / f, dest / f)
            clips = finish_session(dest, cfg, data_root, log=lambda *_: None, reindex=False)
            log(f"merged {name} from {meta.get('device', '?')}: {len(clips)} clips")
            merged.append(dest)
    finally:
        if tmp is not None:
            tmp.cleanup()
    from .data import index_dataset
    index_dataset(data_root / "raw", data_root / "manifest.csv")
    return merged
