"""Cut a continuous session recording into per-trial clips.

The prompter logs, for every trial, the sample (in the laptop's audio clock) at
which the GO cue was shown, and plays a short 1 kHz beep at exactly that sample.

* ``duplex`` sessions were recorded by the same sound card, so the cue sample is
  already (up to the fixed I/O latency) a position in the recording.
* ``prompt`` sessions were recorded on the phone. The laptop played a sync chirp
  at the start; finding it in the phone recording maps laptop samples to phone
  samples.

In both cases every predicted cue is refined by locating its beep, and a robust
linear fit over all beeps absorbs latency and clock drift between devices.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import read_wav, write_wav
from .config import Config
from .probe import BEEP_HZ, BEEP_S, CHIRP_F0, CHIRP_F1, sync_chirp


def _envelope(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


def find_chirp(audio: np.ndarray, sr: int, search_s: float = 180.0) -> tuple[int, float]:
    """Return ``(sample, quality)`` of the sync chirp start (quality = peak / median)."""
    n = int(min(len(audio), search_s * sr))
    sos = signal.butter(4, [CHIRP_F0 * 0.8, CHIRP_F1 * 1.1], btype="band", fs=sr, output="sos")
    x = signal.sosfiltfilt(sos, audio[:n])
    tmpl = sync_chirp(sr)
    corr = np.abs(signal.hilbert(signal.fftconvolve(x, tmpl[::-1], mode="valid")))
    k = int(np.argmax(corr))
    return k, float(corr[k] / (np.median(corr) + 1e-12))


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


def segment_session(session_dir: str | Path, cfg: Config, audio_path: str | Path | None = None, *,
                    channel: int = 0, imu_accel: str | None = None, imu_gyro: str | None = None,
                    log=print) -> list[dict]:
    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session.json").read_text())
    audio_path = Path(audio_path) if audio_path else session_dir / meta.get("audio", "session.wav")
    audio, sr = read_wav(audio_path, channel=channel)
    if sr < 2 * max(cfg.probe.tones_hz) + 1000:
        raise ValueError(f"{audio_path}: sample rate {sr} Hz is too low for the probe tones")
    lsr = int(meta["sample_rate"])
    cues = read_cues(session_dir)
    scale = sr / lsr
    if meta["mode"] == "prompt":
        chirp_at, q = find_chirp(audio, sr)
        if q < 8:
            log(f"WARNING: sync chirp detection is weak (quality {q:.1f}); check the alignment plot/residuals")
        origin = chirp_at - int(meta["chirp_sample"]) * scale
        radius = int(0.35 * sr)
        log(f"sync chirp found at {chirp_at / sr:.2f}s (quality {q:.0f})")
    else:
        origin, radius = 0.0, int(0.25 * sr)

    predicted = np.array([origin + int(c["go_sample"]) * scale for c in cues])
    detected, quality = [], []
    for p in predicted:
        pos, q = find_beep(audio, sr, int(p), radius)
        detected.append(pos)
        quality.append(q)
    detected, quality = np.array(detected, float), np.array(quality)
    resid = detected - predicted
    good = quality > 4
    if good.sum() >= 3:
        med = np.median(resid[good])
        inl = good & (np.abs(resid - med) < 0.02 * sr)
        if inl.sum() >= 3:
            slope, icpt = np.polyfit(predicted[inl], resid[inl], 1)
        else:
            slope, icpt = 0.0, med
        go = predicted + icpt + slope * predicted
        spread = np.std(resid[inl] - (icpt + slope * predicted[inl])) / sr * 1000 if inl.sum() > 1 else 0.0
        log(f"beep alignment: {int(good.sum())}/{len(cues)} beeps found, offset {icpt / sr * 1000:.1f} ms, "
            f"drift {slope * 1e6:.0f} ppm, residual jitter {spread:.1f} ms")
    else:
        log("WARNING: beeps not detected; using predicted cue times without refinement")
        go = predicted

    pre, post = cfg.data.clip_pre_s, cfg.data.clip_post_s
    clip_s = pre + post
    imu_t = imu = None
    if imu_accel:
        from .imu import align_by_taps, read_session_imu
        imu_t, imu = read_session_imu(imu_accel, imu_gyro)
        first_cue_s = float(go.min() / sr) if len(go) else 60.0
        imu_offset = align_by_taps(audio, sr, imu_t, imu, search_s=min(first_cue_s, 120.0))
        log(f"IMU aligned by taps: imu_time = audio_time + {imu_offset:.3f}s")
        meta["imu_offset_s"] = imu_offset
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2))

    rows = []
    for cue, g in zip(cues, go):
        if cue.get("status", "ok") != "ok":
            continue
        a = int(round(g - pre * sr))
        b = a + int(round(clip_s * sr))
        if a < 0 or b > len(audio):
            log(f"skipping trial {cue['trial']}: outside the recording")
            continue
        name = f"clips/{int(cue['trial']):04d}_{cue['label']}.wav"
        write_wav(session_dir / name, audio[a:b], sr)
        row = {"clip": name, "trial": cue["trial"], "label": cue["label"], "imu": ""}
        if imu is not None:
            from .imu import write_clip_imu
            row["imu"] = f"imu/{int(cue['trial']):04d}_{cue['label']}.csv"
            write_clip_imu(session_dir / row["imu"], imu_t, imu, a / sr + meta["imu_offset_s"], clip_s)
        rows.append(row)
    with (session_dir / "clips.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "trial", "label", "imu"])
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {len(rows)} clips to {session_dir / 'clips'}")
    return rows
