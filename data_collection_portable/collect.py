#!/usr/bin/env python3
"""EchoSpeak portable data collection: one participant per run, on any laptop.

Standalone: needs only numpy, scipy and sounddevice (see requirements.txt).
Copy this folder to a laptop, then:

    python collect.py --setup        # once per laptop: audio check + find where the participant should sit
    python collect.py                # one run = one participant session (asks for the participant ID)
    python collect.py --pack         # zip all recordings to send back (merge with `echospeak merge <zip>`)

Self-paced: nothing is recorded until the participant presses SPACE. One SPACE
records a *take* of GROUP_SIZE repetitions of one word (one per beep); each word's
repetitions are split into several takes spread over the session.

Repetitions per word/action: edit SETTINGS below, or per run:
    python collect.py --reps 6
    python collect.py --class-reps NEXT=15 YAWNING=5 TALKING=0

Recordings go to ./recordings/<participant>/<session>/ next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

# =============================== SETTINGS (edit freely) ===============================
REPS = 10                # repetitions (clips) of every word/action per run
CLASS_REPS: dict[str, int] = {}   # per-class overrides, e.g. {"NEXT": 15, "YAWNING": 5, "TALKING": 0}
GROUP_SIZE = 5           # repetitions recorded back-to-back after one SPACE press
INTERVAL_S = 3.0         # seconds between beeps within a take (YAWNING uses 6 s)
PROBE_OFF_REPS = 1       # probe-off control clips per class at the end (0 = none)
PRACTICE = True          # short unsaved practice before the real recording
DISTANCE_CM = 10         # default mouth-to-microphone distance (the setup can change it per laptop)
# ===== DO NOT CHANGE: must match configs/default.yaml of the training repository =====
COMMANDS = ["BACK", "MENU", "BANANA", "GO", "OKAY", "STOP", "NEXT", "APPLE"]
NON_COMMANDS = ["REST", "CHEWING", "YAWNING", "TALKING"]
TONES_HZ = [18000.0 + 250.0 * k for k in range(8)]
PROBE_AMPLITUDE = 0.6
CLIP_PRE_S, CLIP_POST_S = 0.75, 2.25
SAMPLE_RATES = (48000, 44100)    # first one the sound card accepts is used
BEEP_HZ, BEEP_S = 1000.0, 0.08
# =======================================================================================

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "laptop_profile.json"
CLASSES = COMMANDS + NON_COMMANDS
CONTINUOUS = {"REST", "CHEWING", "TALKING"}     # recorded as one stretch, cut into GROUP_SIZE clips
CLASS_INTERVAL_S = {"YAWNING": 6.0}
INSTRUCTIONS = {
    "REST": "stay still and relaxed, mouth closed, until the second beep",
    "CHEWING": "chew continuously (as if chewing gum) until the second beep",
    "TALKING": "talk OUT LOUD continuously (anything) until the second beep",
    "YAWNING": "yawn (or imitate a yawn) once at each beep",
}
PRACTICE_TAKES = [("NEXT", 3), ("CHEWING", 2)]
COUNTDOWN_S = 1.5
LOG_FIELDS = ["participant", "session", "date", "device", "collector", "clips", "probe_off_clips",
              "distance_cm", "condition", "min_tone_snr_db", "peak_level", "notes"]
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
YELLOW, GREEN, CYAN, RED = "\033[33m", "\033[32m", "\033[36m", "\033[31m"


def instruction(label: str) -> str:
    return INSTRUCTIONS.get(label, f"silently mouth \"{label.lower()}\" once at each beep — NO sound, "
                                   "no whisper; close your mouth in between")


# ----------------------------------------------------------------------------- signals

def _phases(n: int) -> np.ndarray:
    k = np.arange(1, n + 1)
    return -np.pi * k * (k - 1) / n  # Schroeder phases (low crest factor)


_SCALE: dict[int, float] = {}


def probe_block(sr: int, start: int, n: int) -> np.ndarray:
    tones = np.asarray(TONES_HZ)
    if sr not in _SCALE:
        t = np.arange(sr) / sr
        _SCALE[sr] = 1.0 / np.abs(np.cos(2 * np.pi * tones[:, None] * t + _phases(len(tones))[:, None]).sum(0)).max()
    t = (start + np.arange(n)) / sr
    x = np.cos(2 * np.pi * tones[:, None] * t + _phases(len(tones))[:, None]).sum(0)
    return (PROBE_AMPLITUDE * _SCALE[sr] * x).astype(np.float32)


def beep(sr: int) -> np.ndarray:
    n = int(BEEP_S * sr)
    return (0.4 * np.hanning(n) * np.sin(2 * np.pi * BEEP_HZ * np.arange(n) / sr)).astype(np.float32)


def tone_snr_db(x: np.ndarray, sr: int) -> list[float]:
    f, p = signal.welch(np.asarray(x, float), fs=sr, nperseg=8192)
    out = []
    for tone in TONES_HZ:
        peak = p[np.argmin(np.abs(f - tone))]
        mask = (np.abs(f - tone) > 40) & (np.abs(f - tone) < 120)
        out.append(round(float(10 * np.log10(max(peak, 1e-30) / max(np.median(p[mask]), 1e-30))), 1))
    return out


def motion_energy(x: np.ndarray, sr: int) -> float:
    """Median echo-motion energy (relative to the static path), averaged over tones."""
    t = np.arange(len(x)) / sr
    tones = np.asarray(TONES_HZ)
    mixed = x[None, :] * np.exp(-2j * np.pi * tones[:, None] * t[None, :])
    from math import gcd
    g = gcd(sr, 2000)
    mid = signal.resample_poly(mixed, 2000 // g, sr // g, axis=1)
    mid = signal.filtfilt(signal.firwin(255, 110, fs=2000), [1.0], mid, axis=1)[:, ::4]
    rel = mid / mid.mean(axis=1, keepdims=True) - 1
    d = np.abs(np.diff(rel[:, 50:-50], axis=1)) ** 2
    return float(np.median(d.mean(axis=0)))


def find_beep(audio: np.ndarray, sr: int, center: int, radius: int) -> tuple[int, float]:
    lo, hi = max(0, center - radius), min(len(audio), center + radius + int(BEEP_S * sr))
    if hi - lo < int(2 * BEEP_S * sr):
        return center, 0.0
    sos = signal.butter(4, [BEEP_HZ * 0.8, BEEP_HZ * 1.2], btype="band", fs=sr, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, audio[lo:hi])))
    n = int(BEEP_S * sr)
    smooth = np.convolve(env, np.hanning(n) / np.hanning(n).sum(), mode="same")
    k = int(np.argmax(smooth))
    return lo + k - n // 2, float(smooth[k] / (np.median(smooth) + 1e-12))


def write_wav16(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, np.round(np.clip(x, -1, 32767 / 32768) * 32768).astype(np.int16))


def cut_clips(session_dir: Path, audio: np.ndarray, sr: int, rows: list[dict]) -> list[dict]:
    """Cut one clip per good cue (latency from the recorded beeps). The training repo re-cuts on merge."""
    pred = np.array([float(r["go_sample"]) for r in rows])
    has_beep = np.array([r["beep"] == "1" for r in rows], bool)
    found = [find_beep(audio, sr, int(p), int(0.25 * sr)) if b else (int(p), 0.0) for p, b in zip(pred, has_beep)]
    good = (np.array([q for _, q in found]) > 4) & has_beep
    resid = np.array([p for p, _ in found]) - pred
    shift = float(np.median(resid[good])) if good.sum() >= 3 else 0.0
    clips = []
    for r, p in zip(rows, pred):
        if r["status"] != "ok":
            continue
        a = int(round(p + shift - CLIP_PRE_S * sr))
        b = a + int(round((CLIP_PRE_S + CLIP_POST_S) * sr))
        if a < 0 or b > len(audio):
            continue
        name = f"clips/{int(r['trial']):04d}_{r['label']}.wav"
        write_wav16(session_dir / name, audio[a:b], sr)
        clips.append({"clip": name, "trial": r["trial"], "label": r["label"], "probe": r["probe"], "take": r["take"]})
    with (session_dir / "clips.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "trial", "label", "probe", "take"])
        w.writeheader()
        w.writerows(clips)
    return clips


# ----------------------------------------------------------------------------- system info

def laptop_model() -> str:
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
    except Exception:
        pass
    return platform.node()


def sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError):
        raise SystemExit("Missing audio library. Run:  pip install -r requirements.txt")


# ----------------------------------------------------------------------------- terminal UI

class Keys:
    """Non-blocking single-key reader (macOS/Linux terminals and Windows consoles)."""

    def __init__(self):
        self.pressed: list[str] = []
        self._stop = False
        self._old = None
        self.enabled = sys.stdin.isatty()
        if not self.enabled:
            return
        if os.name == "posix":
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if os.name == "nt":
            import msvcrt
            while not self._stop:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    self.pressed.append("\n" if ch == "\r" else ch.lower())
                time.sleep(0.03)
        else:
            import select
            while not self._stop:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    self.pressed.append(sys.stdin.read(1).lower())

    def take(self) -> set[str]:
        keys, self.pressed = set(self.pressed), []
        return keys

    def wait(self, accept: set[str], timeout: float | None = None) -> str | None:
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


def show(*lines: str, big: str | None = None, color: str = "", header: str = "") -> None:
    out = ["\033[2J\033[H"]
    if header:
        out.append(f"  {DIM}{header}{RESET}\n")
    out.append("\n\n")
    if big:
        out.append(f"        {color}{BOLD}{'  '.join(big)}{RESET}\n\n")
    out += [f"        {line}\n" for line in lines]
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def progress(done: int, total: int, seconds_left: float) -> str:
    fill = int(30 * done / max(total, 1))
    return f"[{'#' * fill}{'-' * (30 - fill)}] {done}/{total}   about {max(1, round(seconds_left / 60))} min left"


# ----------------------------------------------------------------------------- audio engine

class Engine:
    """Continuous full-duplex stream: probe (+ scheduled beeps) out, microphone in."""

    def __init__(self, input_device=None, output_device=None):
        sd = sounddevice()
        self.pos, self.probe_on, self.xruns = 0, True, 0
        self.sounds: list[tuple[int, np.ndarray]] = []
        self.chunks: list[np.ndarray] = []
        self.lock = threading.Lock()
        last = None
        for sr in SAMPLE_RATES:
            try:
                self.sr = sr
                self.stream = sd.Stream(samplerate=sr, blocksize=1024, dtype="float32", latency="high",
                                        channels=(1, 1), callback=self._callback,
                                        device=(input_device, output_device))
                return
            except Exception as exc:  # try the next sample rate
                last = exc
        raise SystemExit(f"Could not open the audio devices at {SAMPLE_RATES} Hz: {last}")

    def _callback(self, indata, outdata, frames, _t, status):
        if status:
            self.xruns += 1
        start = self.pos
        out = probe_block(self.sr, start, frames) if self.probe_on else np.zeros(frames, np.float32)
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
        self.chunks.append(indata[:, 0].copy())
        self.pos = start + frames

    def schedule(self, sound: np.ndarray, at: int) -> None:
        with self.lock:
            self.sounds.append((at, sound))

    def wait_until(self, sample: int) -> None:
        while self.pos < sample:
            time.sleep(0.002)

    def n(self, seconds: float) -> int:
        return int(round(seconds * self.sr))

    def recording(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    def record_for(self, seconds: float) -> np.ndarray:
        start = self.pos
        self.wait_until(start + self.n(seconds))
        return self.recording()[start: start + self.n(seconds)]


def sound_check(engine: Engine) -> dict:
    start = engine.pos
    at = start + engine.n(1.0)
    engine.schedule(beep(engine.sr), at)
    engine.wait_until(start + engine.n(2.6))
    rec = engine.recording()
    seg = rec[start: start + engine.n(2.5)]
    snr = tone_snr_db(seg, engine.sr)
    _, q = find_beep(rec, engine.sr, at, engine.n(0.3))
    return {"tone_snr_db": snr, "min_tone_snr_db": min(snr), "peak_level": round(float(np.abs(seg).max()), 3),
            "beep_heard": bool(q > 4)}


def check_problems(c: dict) -> list[str]:
    out = []
    if c["min_tone_snr_db"] < 20:
        out.append(f"Probe tones are weak (min SNR {c['min_tone_snr_db']} dB, need >= 20): raise the volume to "
                   "~70-80%, use the built-in speaker (no Bluetooth/headphones), turn off mic effects "
                   "(macOS: Control Center > Mic Mode: Standard; Windows: Sound settings > disable audio "
                   "enhancements), or rerun --setup and pick other devices.")
    if c["peak_level"] > 0.98:
        out.append("The microphone is clipping: lower the volume or the input gain.")
    if not c["beep_heard"]:
        out.append("The microphone did not hear the GO beep: raise the volume.")
    return out


def run_sound_check(engine: Engine, interactive: bool) -> dict:
    while True:
        print("Sound check (stay quiet for 3 s)...")
        c = sound_check(engine)
        print(f"  tone SNR {c['tone_snr_db']} dB, peak {c['peak_level']}, beep heard: {c['beep_heard']}")
        problems = check_problems(c)
        if not problems:
            print("  OK")
            return c
        for p in problems:
            print(f"  {RED}! {p}{RESET}")
        if not interactive:
            return c
        ans = input("[r]etry / [c]ontinue anyway / [q]uit: ").strip().lower()
        if ans.startswith("c"):
            return c
        if ans.startswith("q"):
            raise SystemExit("aborted at the sound check")


# ----------------------------------------------------------------------------- setup (once per laptop)

def setup() -> dict:
    sd = sounddevice()
    print(f"\n{BOLD}EchoSpeak laptop setup{RESET}  ({laptop_model()}, {platform.system()})\n")
    print(sd.query_devices())
    default_in, default_out = sd.default.device
    print("\nUse the BUILT-IN microphone and speakers (not Bluetooth, not headphones).")
    inp = input(f"Input device number [Enter = default {default_in}]: ").strip()
    outp = input(f"Output device number [Enter = default {default_out}]: ").strip()
    tag = ""
    while not re.fullmatch(r"[A-Za-z0-9]{2,6}", tag):
        tag = input("Collector tag, 2-6 letters/digits, unique per person/laptop (e.g. your initials): ").strip()
    profile = {"collector": tag.upper(), "device": laptop_model(), "hostname": platform.node(),
               "input_device": int(inp) if inp else None, "output_device": int(outp) if outp else None}
    engine = Engine(profile["input_device"], profile["output_device"])
    engine.stream.start()
    try:
        time.sleep(0.5)
        profile["sample_rate"] = engine.sr
        profile["sound_check"] = run_sound_check(engine, interactive=True)
        print(f"\n{BOLD}Placement test{RESET}: find where the participant's mouth should be.")
        print("Look for the laptop's microphone (usually near the camera or beside the keyboard).")
        while True:
            dist = input(f"Mouth-to-microphone distance for this test in cm [{DISTANCE_CM}]: ").strip()
            dist = float(dist) if dist else float(DISTANCE_CM)
            input("Sit at that spot. Press Enter, then stay STILL for 4 s... ")
            still = engine.record_for(4.0)
            input("Press Enter, then silently mouth NEXT, BACK, STOP ... repeatedly for 4 s... ")
            moving = engine.record_for(4.0)
            ratio = motion_energy(moving, engine.sr) / max(motion_energy(still, engine.sr), 1e-15)
            verdict = ("good" if ratio >= 3 else "weak: move closer to the microphone" if ratio >= 1.5
                       else "too weak: find the microphone and sit closer")
            print(f"  mouthing / still motion ratio: {ratio:.1f}x  -> {verdict}")
            if input("Try another position? [y/N]: ").strip().lower() != "y":
                break
        profile.update({"distance_cm": dist, "motion_ratio": round(ratio, 2)})
        profile["seating"] = input("Describe where participants should sit (shown before every run), e.g. "
                                   "'mouth 10 cm above the left of the keyboard': ").strip()
    finally:
        engine.stream.stop()
        engine.stream.close()
    profile["created"] = datetime.now().isoformat(timespec="seconds")
    PROFILE.write_text(json.dumps(profile, indent=2))
    print(f"\nSaved {PROFILE.name}. Start a recording with:  python collect.py")
    return profile


# ----------------------------------------------------------------------------- one run = one participant

class Take:
    """One SPACE press: ``n`` repetitions of ``label`` (paced by beeps, or one continuous stretch)."""

    def __init__(self, label: str, n: int, interval_s: float, probe_on: bool = True):
        self.label, self.n, self.probe_on = label, n, probe_on
        self.continuous = label in CONTINUOUS
        self.interval_s = (CLIP_PRE_S + CLIP_POST_S) if self.continuous else CLASS_INTERVAL_S.get(label, interval_s)

    def duration_s(self) -> float:
        tail = 0.0 if self.continuous else CLIP_POST_S - self.interval_s
        return COUNTDOWN_S + self.n * self.interval_s + tail + 0.5 + 3.0   # +3 s to read and press SPACE


def reps_plan(reps: int, class_reps: dict[str, int]) -> dict[str, int]:
    class_reps = {k.upper(): int(v) for k, v in class_reps.items()}
    unknown = set(class_reps) - set(CLASSES)
    if unknown:
        raise SystemExit(f"Unknown classes in class reps: {sorted(unknown)}; valid: {CLASSES}")
    return {lab: class_reps.get(lab, reps) for lab in CLASSES}


def build_takes(plan: dict[str, int], group_size: int, interval_s: float, rng: random.Random,
                probe_on: bool = True) -> list[Take]:
    """Takes of ``group_size``; round r = the r-th take of every class, shuffled (spread over the session)."""
    per_class = {}
    for lab, reps in plan.items():
        sizes = [group_size] * (reps // group_size) + ([reps % group_size] if reps % group_size else [])
        per_class[lab] = [Take(lab, s, interval_s, probe_on) for s in sizes]
    takes: list[Take] = []
    for r in range(max((len(v) for v in per_class.values()), default=0)):
        round_ = [v[r] for v in per_class.values() if len(v) > r]
        rng.shuffle(round_)
        if takes and len(round_) > 1 and round_[0].label == takes[-1].label:
            round_[0], round_[1] = round_[1], round_[0]
        takes += round_
    return takes


def next_participant(root: Path, tag: str) -> str:
    n = 1
    while (root / f"{tag}-P{n:02d}").exists():
        n += 1
    return f"{tag}-P{n:02d}"


def next_session(root: Path, participant: str) -> str:
    nums = [int(m.group(1)) for p in (root / participant).glob("S*")
            if (m := re.fullmatch(r"S(\d+)", p.name)) and (p / "cues.csv").exists()]
    return f"S{max(nums, default=0) + 1}"


def run_take(engine: Engine, keys: Keys, rows: list[dict], take: Take, header: str, status: str,
             take_id: str, can_redo_previous: bool) -> str:
    """Get-ready screen (waits for SPACE), countdown, beeps; appends cue rows. Returns ok/redo/redo_previous/quit."""
    what = f"{take.label}  for {take.n * take.interval_s:.0f} s" if take.continuous else f"{take.label}  x {take.n}"
    extra = "starts and ends with a beep" if take.continuous else f"one beep every {take.interval_s:g} s"
    keys.take()
    show(f"{instruction(take.label)}  ({extra})", "",
         f"{CYAN}Get ready, then press SPACE.{RESET}{DIM}"
         f"{'   x = redo the previous one' if can_redo_previous else ''}   q = save and stop{RESET}",
         big=what, color=YELLOW, header=header)
    key = keys.wait({" ", "\n", "q", "x"} if can_redo_previous else {" ", "\n", "q"})
    if key == "x":
        return "redo_previous"
    if key == "q":
        return "quit"
    engine.probe_on = take.probe_on
    t0 = engine.pos
    start = t0 + engine.n(COUNTDOWN_S)
    for i, c in enumerate(("3", "2", "1")):
        engine.wait_until(t0 + engine.n(i * COUNTDOWN_S / 3))
        show(instruction(take.label), big=c, color=YELLOW, header=header)
    step = engine.n(take.interval_s)
    bp = beep(engine.sr)
    if take.continuous:
        gos = [start + engine.n(CLIP_PRE_S) + k * step for k in range(take.n)]
        end = start + take.n * step
        engine.schedule(bp, start)
        engine.schedule(bp, end)
        engine.wait_until(start)
        while engine.pos < end:
            show(instruction(take.label), "", f"{(end - engine.pos) / engine.sr:.0f} s left",
                 big=take.label, color=GREEN, header=header)
            engine.wait_until(min(end, engine.pos + engine.n(1.0)))
        finish = end + engine.n(0.3)
    else:
        gos = [start + k * step for k in range(take.n)]
        for g in gos:
            engine.schedule(bp, g)
        for k, g in enumerate(gos):
            engine.wait_until(g)
            show(instruction(take.label), "", f"{k + 1} of {take.n}", big=take.label, color=GREEN, header=header)
        finish = gos[-1] + engine.n(CLIP_POST_S + 0.1)
    engine.wait_until(finish)
    engine.probe_on = True
    pressed = keys.take()
    if "x" in pressed:
        status = "discarded"
    rows.extend({"trial": len(rows) + i + 1, "label": take.label, "go_sample": g,
                 "probe": "on" if take.probe_on else "off", "status": status, "take": take_id,
                 "beep": "0" if take.continuous else "1"} for i, g in enumerate(gos))
    if "x" in pressed:
        return "redo"
    return "quit" if "q" in pressed else "ok"


def run(args) -> Path:
    profile = json.loads(PROFILE.read_text()) if PROFILE.exists() else None
    if profile is None or profile.get("device") != laptop_model():
        print("This laptop has not been set up yet; running the one-time setup first.")
        profile = setup()
    root = Path(args.out)
    tag = profile["collector"]
    participant = args.participant or input(f"Participant ID [{next_participant(root, tag)}]: ").strip() \
        or next_participant(root, tag)
    if not participant.upper().startswith(f"{tag}-"):
        participant = f"{tag}-{participant}"   # IDs stay unique across laptops/collectors
    session = args.session or next_session(root, participant)
    out = root / participant / session
    if (out / "cues.csv").exists():
        raise SystemExit(f"{out} already exists")
    if args.interval < 2.0 or args.group_size < 1:
        raise SystemExit("--interval must be >= 2 s and --group-size >= 1")
    distance = args.distance_cm if args.distance_cm is not None else profile.get("distance_cm", DISTANCE_CM)
    plan = reps_plan(args.reps, {**CLASS_REPS, **parse_class_reps(args.class_reps)})
    rng = random.Random(zlib.crc32(f"{participant}/{session}".encode()))
    takes = build_takes(plan, args.group_size, args.interval, rng)
    takes += build_takes({lab: args.probe_off_reps for lab in CLASSES}, args.group_size, args.interval, rng,
                         probe_on=False)
    n_main, n_off = sum(plan.values()), args.probe_off_reps * len(CLASSES)
    practice = not args.no_practice and PRACTICE
    print(f"\n{BOLD}Run {participant}/{session}{RESET} on {profile['device']} (collector {tag}): "
          f"{len(takes)} takes, about {sum(t.duration_s() for t in takes) / 60:.0f} min")
    print("  main part:  " + ", ".join(f"{k} x{v}" for k, v in plan.items() if v)
          + f"  = {n_main} clips (groups of {args.group_size})")
    print(f"  probe-off:  {n_off} clips")
    print(f"  seating:    {profile.get('seating') or f'mouth ~{distance:g} cm from the microphone'}")
    input("Press Enter to start the sound check... ")

    engine = Engine(profile.get("input_device"), profile.get("output_device"))
    meta = {"participant": participant, "session": session, "sample_rate": engine.sr, "probe": "on",
            "tones_hz": TONES_HZ, "distance_cm": distance, "condition": args.condition, "orientation": "laptop",
            "device": profile["device"], "collector": tag, "hostname": platform.node(), "notes": args.notes,
            "reps_per_class": plan, "group_size": args.group_size, "interval_s": args.interval,
            "probe_off_reps": args.probe_off_reps, "labels": CLASSES, "clip_pre_s": CLIP_PRE_S,
            "clip_post_s": CLIP_POST_S, "source": "portable", "created": datetime.now().isoformat(timespec="seconds")}
    rows: list[dict] = []
    keys = None
    out.mkdir(parents=True, exist_ok=True)
    try:
        engine.stream.start()
        time.sleep(0.5)
        meta["sound_check"] = run_sound_check(engine, interactive=True)
        keys = Keys()
        show(f"{BOLD}How this works{RESET}",
             f"- Sit with your mouth about {distance:g} cm (a hand's width) from the laptop microphone.",
             "- Each screen shows a word and how many times to do it. Get ready, then press SPACE.",
             "- After a 3-2-1 countdown, do it ONCE AT EACH BEEP. Close your mouth between beeps.",
             f"- Words ({', '.join(COMMANDS)}): move your lips as if saying the word,",
             "  but make NO sound and do not whisper.",
             "- Rest / chew / talk: keep doing it from the first beep until the second beep.",
             "- Take your time between screens. Mistake? Press x and that part is repeated.",
             "", f"{CYAN}Press SPACE to start{' a short practice' if practice else ''}.{RESET}",
             big="Welcome", color=CYAN)
        keys.wait({" ", "\n"})
        if practice:
            for label, n in PRACTICE_TAKES:
                run_take(engine, keys, rows, Take(label, n, args.interval), "Practice (not saved)", "practice",
                         "practice", False)
            show("Great! The real recording starts now.", "", f"{CYAN}Press SPACE when ready.{RESET}",
                 big="Ready?", color=CYAN)
            keys.wait({" ", "\n"})
        queue, done, last, n_takes = list(takes), 0, None, 0
        total = n_main + n_off
        while queue:
            take = queue.pop(0)
            part = "Main part" if take.probe_on else "Last part (almost done)"
            header = f"{part}  {progress(done, total, sum(t.duration_s() for t in [take, *queue]))}"
            before = len(rows)
            n_takes += 1
            result = run_take(engine, keys, rows, take, header, "ok", f"t{n_takes:03d}", last is not None)
            new = rows[before:]
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
            done += len(new)
            if new:
                last = (take, new)
            if result == "quit":
                raise KeyboardInterrupt
        show("Thank you, you're done!", big="Done", color=GREEN)
    except KeyboardInterrupt:
        print("\nStopped early; saving what was recorded.")
    finally:
        time.sleep(0.5)
        engine.stream.stop()
        engine.stream.close()
        if keys is not None:
            keys.close()

    audio = engine.recording()
    write_wav16(out / "session.wav", audio, engine.sr)
    with (out / "cues.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "label", "go_sample", "probe", "status", "take", "beep"])
        w.writeheader()
        w.writerows(rows)
    meta["xruns"] = engine.xruns
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    clips = cut_clips(out, audio, engine.sr, rows)
    n_off_clips = sum(c["probe"] == "off" for c in clips)
    log = root / "sessions.csv"
    new_log = not log.exists()
    with log.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        w.writerow({"participant": participant, "session": session, "date": meta["created"],
                    "device": meta["device"], "collector": tag, "clips": len(clips) - n_off_clips,
                    "probe_off_clips": n_off_clips, "distance_cm": distance, "condition": args.condition,
                    "min_tone_snr_db": meta["sound_check"]["min_tone_snr_db"],
                    "peak_level": meta["sound_check"]["peak_level"], "notes": args.notes})
    if engine.xruns:
        print(f"{RED}WARNING: {engine.xruns} audio dropouts; consider re-recording this participant.{RESET}")
    print(f"Saved {len(clips) - n_off_clips} clips (+{n_off_clips} probe-off) in {out}")
    print("When all runs on this laptop are done:  python collect.py --pack")
    return out


def pack(root: Path) -> Path:
    """Zip every session (full recordings + logs; clips are re-cut on merge) for sending."""
    profile = json.loads(PROFILE.read_text()) if PROFILE.exists() else {}
    name = HERE / f"echospeak_{profile.get('collector', 'X')}_{datetime.now():%Y%m%d-%H%M}.zip"
    n = 0
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root.rglob("*")):
            if f.is_file() and "clips" not in f.relative_to(root).parts:
                zf.write(f, f.relative_to(root.parent))
                n += f.name == "session.json"
    print(f"Packed {n} session(s) into {name}\nSend it to the team; merge with:  echospeak merge {name.name}")
    return name


def parse_class_reps(items: list[str] | None) -> dict[str, int]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--class-reps expects LABEL=N, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip().upper()] = int(v)
    return out


def main(argv: list[str] | None = None) -> None:
    if os.name == "nt":
        os.system("")  # enable ANSI colours in the Windows console
    p = argparse.ArgumentParser(description="EchoSpeak portable data collection (one participant per run)")
    p.add_argument("--setup", action="store_true", help="one-time laptop setup (devices, sound check, placement)")
    p.add_argument("--pack", action="store_true", help="zip all recordings for sending")
    p.add_argument("--participant", help="participant ID (default: ask, suggesting <TAG>-P##)")
    p.add_argument("--session", help="session ID (default: next free S#)")
    p.add_argument("--reps", type=int, default=REPS, help=f"repetitions (clips) of every word/action (default {REPS})")
    p.add_argument("--class-reps", nargs="+", metavar="LABEL=N", help="per-class counts, e.g. NEXT=15 TALKING=0")
    p.add_argument("--group-size", type=int, default=GROUP_SIZE,
                   help=f"repetitions back-to-back per SPACE press (default {GROUP_SIZE})")
    p.add_argument("--interval", type=float, default=INTERVAL_S,
                   help=f"seconds between beeps in a take (default {INTERVAL_S:g}; YAWNING 6 s)")
    p.add_argument("--probe-off-reps", type=int, default=PROBE_OFF_REPS)
    p.add_argument("--no-practice", action="store_true")
    p.add_argument("--distance-cm", type=float, help="override the distance saved by --setup")
    p.add_argument("--condition", default="normal", help="free tag, e.g. normal, far, noisy")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default=str(HERE / "recordings"))
    args = p.parse_args(argv)
    if args.setup:
        setup()
    elif args.pack:
        pack(Path(args.out))
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
