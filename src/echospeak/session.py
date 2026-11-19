"""Guided recording session (terminal prompter + sound card).

Modes
-----
duplex   The computer's sound card plays the probe AND records (laptop speaker+mic,
         or any audio interface). Fully automatic: clips are cut at the end.
prompt   The PHONE plays ``probe.wav`` and records audio itself; the computer only
         shows prompts and plays audible sync sounds (a chirp at the start, a beep
         at every GO). Afterwards run ``echospeak import`` with the phone's WAV.

Keys during a session:  x = discard the current/last trial (it is repeated later),
p = pause/resume,  q = save and quit.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import parse_device, require_sounddevice, write_wav
from .config import Config
from .probe import beep, probe_block, sync_chirp

INSTRUCTIONS = {
    "REST": "stay still, breathe normally, mouth closed",
    "CHEWING": "chew naturally, as if chewing gum",
    "YAWNING": "yawn (or imitate a yawn)",
    "TALKING": "say a short random sentence OUT LOUD",
}


def instruction(label: str) -> str:
    return INSTRUCTIONS.get(label, f"silently mouth '{label.lower()}' — no voice, no whisper")


class _Keys:
    """Non-blocking single-key reader (POSIX terminals); silently disabled elsewhere."""

    def __init__(self):
        self.pressed: list[str] = []
        self._old = None
        self._stop = False
        if sys.stdin.isatty() and os.name == "posix":
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

    def close(self):
        self._stop = True
        if self._old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


class _Engine:
    def __init__(self, cfg: Config, duplex: bool, probe_on: bool, sr: int,
                 input_device=None, output_device=None, channel: int = 0):
        self.sd = require_sounddevice()
        self.cfg, self.duplex, self.probe_on, self.sr = cfg, duplex, probe_on, sr
        self.channel = channel
        self.pos = 0
        self.sounds: list[tuple[int, np.ndarray]] = []
        self.chunks: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.status_flags = 0
        kw = dict(samplerate=sr, blocksize=1024, dtype="float32", callback=self._callback, latency="high")
        if duplex:
            self.stream = self.sd.Stream(device=(input_device, output_device), channels=(channel + 1, 1), **kw)
        else:
            self.stream = self.sd.OutputStream(device=output_device, channels=1, **kw)

    def _callback(self, *args):
        if self.duplex:
            indata, outdata, frames, _, status = args
        else:
            outdata, frames, _, status = args
            indata = None
        if status:
            self.status_flags += 1
        start = self.pos
        out = probe_block(self.cfg.probe, start, frames) if (self.duplex and self.probe_on) else np.zeros(frames, np.float32)
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
        if indata is not None:
            self.chunks.append(indata[:, self.channel].copy())
        self.pos = start + frames

    def schedule(self, sound: np.ndarray, at: int) -> None:
        with self.lock:
            self.sounds.append((at, sound))

    def wait_until(self, sample: int) -> None:
        while self.pos < sample:
            time.sleep(0.002)

    def recording(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def _show(*lines: str, big: str | None = None, color: str = "") -> None:
    sys.stdout.write("\033[2J\033[H")
    if big:
        spaced = "  ".join(big)
        sys.stdout.write(f"\n\n\n      {color}\033[1m{spaced}\033[0m\n\n")
    for line in lines:
        sys.stdout.write(f"      {line}\n")
    sys.stdout.flush()


def run_session(cfg: Config, out_root: str | Path, participant: str, session: str, *,
                mode: str = "duplex", reps: int = 5, probe_on: bool = True, distance_cm: float = 10.0,
                orientation: str = "front", condition: str = "normal", device_name: str = "",
                input_device: str | None = None, output_device: str | None = None, channel: int = 0,
                prep_s: float = 1.2, gap_s: float = 0.9, labels: list[str] | None = None,
                seed: int | None = None, log=print) -> Path:
    out = Path(out_root) / participant / session
    if (out / "cues.csv").exists():
        raise SystemExit(f"{out} already contains a session; choose another --session name")
    out.mkdir(parents=True, exist_ok=True)
    labels = labels or cfg.classes
    sr = cfg.probe.sample_rate
    rng = random.Random(seed if seed is not None else zlib.crc32(f"{participant}/{session}".encode()))
    engine = _Engine(cfg, mode == "duplex", probe_on, sr, parse_device(input_device),
                     parse_device(output_device), channel)
    meta = {
        "participant": participant, "session": session, "mode": mode, "sample_rate": sr,
        "probe": "on" if probe_on else "off", "tones_hz": cfg.probe.tones_hz,
        "distance_cm": distance_cm, "orientation": orientation, "condition": condition,
        "device": device_name or (platform.node() if mode == "duplex" else "phone"),
        "reps": reps, "labels": labels, "clip_pre_s": cfg.data.clip_pre_s, "clip_post_s": cfg.data.clip_post_s,
        "created": datetime.now().isoformat(timespec="seconds"),
        "audio": "session.wav" if mode == "duplex" else "phone.wav",
    }
    if mode == "prompt":
        _show("PHONE MODE",
              "1. On the phone: start playing probe.wav on loop (volume ~70%, no Bluetooth).",
              "2. Start the phone's recorder (uncompressed WAV, 48 kHz).",
              f"3. Hold the phone ~{distance_cm:g} cm from the mouth, mic toward the face.",
              "", "Press Enter when the phone is recording...")
    else:
        _show("LAPTOP (DUPLEX) MODE", f"Probe {'ON' if probe_on else 'OFF'}. Sit ~{distance_cm:g} cm from the mic/speaker.",
              "", "Press Enter to start...")
    input()
    keys = _Keys()
    rows: list[dict] = []
    try:
        engine.stream.start()
        time.sleep(1.0)
        if mode == "prompt":
            _show("Tap the phone firmly 3 times now", "(used to sync the IMU; harmless otherwise)", big="TAP x3")
            time.sleep(4.0)
            at = engine.pos + int(0.3 * sr)
            engine.schedule(sync_chirp(sr), at)
            meta["chirp_sample"] = at
            engine.wait_until(at + int(1.0 * sr))
        else:
            time.sleep(0.5)
        bp = beep(sr)
        trial = 0
        for rep in range(reps):
            queue = list(labels)
            rng.shuffle(queue)
            while queue:
                label = queue.pop(0)
                trial += 1
                _show(f"Block {rep + 1}/{reps}  ·  trial {trial}", "", f"Get ready: {instruction(label)}",
                      "", "\033[2m(x = redo this trial, p = pause, q = quit)\033[0m", big=label, color="\033[33m")
                go = engine.pos + int(prep_s * sr)
                engine.schedule(bp, go)
                engine.wait_until(go)
                _show(f"{instruction(label)}", big=f"GO  {label}", color="\033[32m")
                engine.wait_until(go + int((cfg.data.clip_post_s + 0.1) * sr))
                _show("relax", big="·")
                time.sleep(gap_s + rng.uniform(0, 0.4))
                pressed = keys.take()
                status = "discarded" if "x" in pressed else "ok"
                rows.append({"trial": trial, "label": label, "go_sample": go, "status": status})
                if status == "discarded":
                    queue.append(label)
                if "p" in pressed:
                    _show("PAUSED — press p to resume", big="PAUSE")
                    while "p" not in keys.take():
                        time.sleep(0.05)
                if "q" in pressed:
                    raise KeyboardInterrupt
            if rep < reps - 1:
                _show(f"Block {rep + 1}/{reps} done. Short break.", "Press Enter (or p) to continue...", big="BREAK")
                while not keys.take() & {"\n", "p", " "}:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        log("\nStopped early; saving what was recorded.")
    finally:
        time.sleep(0.5)
        engine.stream.stop()
        engine.stream.close()
        keys.close()

    with (out / "cues.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "label", "go_sample", "status"])
        w.writeheader()
        w.writerows(rows)
    meta["xruns"] = engine.status_flags
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    _show("Session finished.")
    if engine.status_flags:
        log(f"WARNING: {engine.status_flags} audio over/underruns were reported; check the recording.")
    if mode == "duplex":
        write_wav(out / "session.wav", engine.recording(), sr)
        from .segment import segment_session
        segment_session(out, cfg, log=log)
    else:
        log(f"Stop the phone recording and copy it to {out / 'phone.wav'}, then run:\n"
            f"  echospeak import {out}  [--imu-accel accel.csv --imu-gyro gyro.csv]")
    log(f"{sum(r['status'] == 'ok' for r in rows)} good trials saved in {out}")
    return out
