"""Cut a continuous session recording into per-trial clips.

``collect.py`` records the whole session as one WAV and logs, for every clip, its
cue sample. Paced repetitions have a 1 kHz beep at exactly that sample, which the
microphone records too (``beep=1`` in ``cues.csv``). The beeps are located, and a
robust linear fit over them absorbs the fixed speaker-to-microphone latency; the
same correction is applied to every cue, including continuous takes (``beep=0``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import read_wav, write_wav
from .config import Config
from .probe import BEEP_HZ, BEEP_S

CLIP_FIELDS = ["clip", "trial", "label", "probe", "take"]


def _envelope(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


def find_beep(audio: np.ndarray, sr: int, center: int, radius: int) -> tuple[int, float]:
    """Locate a beep start near ``center``; returns ``(sample, quality)``."""
    lo, hi = max(0, center - radius), min(len(audio), center + radius + int(BEEP_S * sr))
    if hi - lo < int(2 * BEEP_S * sr):
        return center, 0.0
    env = _envelope(audio[lo:hi], sr, BEEP_HZ * 0.8, BEEP_HZ * 1.2)
    n = int(BEEP_S * sr)
    smooth = np.convolve(env, np.hanning(n) / np.hanning(n).sum(), mode="same")
    k = int(np.argmax(smooth))
    return lo + k - n // 2, float(smooth[k] / (np.median(smooth) + 1e-12))


def read_cues(session_dir: Path) -> list[dict]:
    with (session_dir / "cues.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def segment_session(session_dir: str | Path, cfg: Config, *, log=print) -> list[dict]:
    """(Re-)cut ``clips/`` and ``clips.csv`` from ``session.wav`` + ``cues.csv``."""
    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session.json").read_text())
    audio, sr = read_wav(session_dir / "session.wav")
    if sr != int(meta["sample_rate"]):
        raise ValueError("session.wav sample rate differs from session.json")
    cues = read_cues(session_dir)
    if not cues:
        log("no trials recorded")
        return []

    predicted = np.array([float(c["go_sample"]) for c in cues])
    has_beep = np.array([c.get("beep", "1") != "0" for c in cues])
    found = [find_beep(audio, sr, int(p), int(0.25 * sr)) if b else (int(p), 0.0)
             for p, b in zip(predicted, has_beep)]
    detected = np.array([f[0] for f in found], float)
    good = (np.array([f[1] for f in found]) > 4) & has_beep
    resid = detected - predicted
    if good.sum() >= 3:
        med = np.median(resid[good])
        inl = good & (np.abs(resid - med) < 0.02 * sr)
        slope, icpt = np.polyfit(predicted[inl], resid[inl], 1) if inl.sum() >= 3 else (0.0, med)
        go = predicted + icpt + slope * predicted
        log(f"beep alignment: {int(good.sum())}/{int(has_beep.sum())} beeps heard, "
            f"audio latency {icpt / sr * 1000:.0f} ms")
    else:
        log("WARNING: GO beeps not heard by the microphone; using the logged cue times "
            "(clips may be offset by the audio latency, typically < 0.1 s)")
        go = predicted

    pre, post = cfg.data.clip_pre_s, cfg.data.clip_post_s
    rows = []
    for cue, g in zip(cues, go):
        if cue.get("status", "ok") != "ok":
            continue
        a = int(round(g - pre * sr))
        b = a + int(round((pre + post) * sr))
        if a < 0 or b > len(audio):
            log(f"skipping trial {cue['trial']}: outside the recording")
            continue
        name = f"clips/{int(cue['trial']):04d}_{cue['label']}.wav"
        write_wav(session_dir / name, audio[a:b], sr, pcm16=True)
        rows.append({"clip": name, "trial": cue["trial"], "label": cue["label"], "probe": cue.get("probe", "on"),
                     "take": cue.get("take", "")})
    with (session_dir / "clips.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CLIP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows
