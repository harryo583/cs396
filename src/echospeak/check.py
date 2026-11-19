"""Milestone 1: verify that the hardware emits/captures the probe and sees motion.

Reports per-tone SNR (tones missing => the sound card/OS filters >16 kHz,
or the speaker cannot play them), clipping, and the echo-motion
energy during "still" vs "moving" phases. Saves a PNG of the motion trace and
Doppler spectrogram when matplotlib is installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .audio import parse_device, read_wav, require_sounddevice, write_wav
from .config import Config
from .features import motion_trace
from .probe import demodulate, probe_signal, tone_snr_db

PHASES = [(0, 4, "hold still"),
          (4, 8, "wave a hand slowly 5-15 cm in front of the laptop mic"),
          (8, 12, "silently mouth NEXT, BACK, STOP ... repeatedly")]


def record_check(cfg: Config, out_wav: Path, input_device=None, output_device=None, channel: int = 0) -> Path:
    sd = require_sounddevice()
    sr = cfg.probe.sample_rate
    total = PHASES[-1][1]
    probe = probe_signal(cfg.probe, total + 0.5)
    print("Recording probe check. Follow the prompts:")
    rec = sd.playrec(probe[:, None], samplerate=sr, channels=channel + 1, dtype="float32",
                     device=(parse_device(input_device), parse_device(output_device)))
    t0 = time.time()
    for a, b, text in PHASES:
        while time.time() - t0 < a:
            time.sleep(0.01)
        print(f"  [{a:2d}-{b:2d}s] {text}")
    sd.wait()
    write_wav(out_wav, rec[:, channel], sr)
    return out_wav


def analyze(cfg: Config, wav: Path, out_dir: Path, phases=PHASES) -> dict:
    audio, sr = read_wav(wav)
    out_dir.mkdir(parents=True, exist_ok=True)
    snr = tone_snr_db(audio, sr, cfg.probe)
    bb = demodulate(audio, sr, cfg.probe, cfg.features)
    trace = motion_trace(torch.from_numpy(bb), cfg.features).numpy()
    fr = cfg.features.frame_rate
    report = {"file": str(wav), "sample_rate": sr, "peak": float(np.abs(audio).max()),
              "tone_snr_db": dict(zip([f"{f:.0f}" for f in cfg.probe.tones_hz], [round(s, 1) for s in snr]))}
    lines = [f"file: {wav}  ({len(audio) / sr:.1f}s @ {sr} Hz)",
             f"peak level: {report['peak']:.3f}" + ("  <-- CLIPPING, lower the volume" if report["peak"] > 0.98 else ""),
             "tone SNR (dB): " + ", ".join(f"{k}:{v}" for k, v in report["tone_snr_db"].items())]
    weak = [k for k, v in report["tone_snr_db"].items() if v < 20]
    if weak:
        lines.append(f"WARNING: weak/missing tones {weak} — check volume, speaker, sample rate, "
                     "and that the recorder saves uncompressed audio without voice processing.")
    if phases and len(trace) / fr >= phases[-1][1]:
        energies = {text: float(np.median(trace[int((a + 0.5) * fr): int((b - 0.5) * fr)])) for a, b, text in phases}
        still = energies[phases[0][2]]
        for text, e in energies.items():
            lines.append(f"motion energy '{text}': {e:.2e}  ({e / max(still, 1e-12):.1f}x still)")
        report["motion_energy"] = energies
    print("\n".join(lines))
    (out_dir / "check.txt").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(cfg.probe.tones_hz) // 2
        rel = bb[k] / bb[k].mean() - 1
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(np.arange(len(trace)) / fr, trace)
        ax[0].set_ylabel("echo motion energy")
        ax[0].set_yscale("log")
        rate = cfg.features.baseband_rate
        ax[1].specgram(rel, NFFT=cfg.features.doppler_nfft, Fs=rate, noverlap=cfg.features.doppler_nfft - 5,
                       sides="twosided", cmap="magma")
        ax[1].set_ylim(-cfg.features.doppler_max_hz, cfg.features.doppler_max_hz)
        ax[1].set_ylabel(f"Doppler (Hz) @ {cfg.probe.tones_hz[k]:.0f} Hz")
        ax[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / "check.png", dpi=120)
        print(f"plot: {out_dir / 'check.png'}")
    except ImportError:
        pass
    return report
