"""Physically-motivated synthetic recordings for pipeline testing.

Each clip = static paths + echo from a moving articulator whose displacement
follows a class-specific template, with per-participant, per-session and
per-trial variation. Synthetic accuracy says NOTHING about real sensing; it only
verifies that the code (demodulation -> features -> model -> evaluation) works.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .audio import write_wav
from .config import Config
from .probe import _probe_scale, _tone_phases

SPEED_OF_SOUND = 343.0

# (time s, displacement mm, width s) bumps after motion onset
WORDS = {
    "NEXT": [(0.15, 3.0, 0.08), (0.35, -1.5, 0.10)],
    "BACK": [(0.12, 4.0, 0.10), (0.30, -2.0, 0.06)],
    "SELECT": [(0.10, 2.0, 0.06), (0.35, 3.5, 0.09), (0.55, -1.0, 0.06)],
    "CANCEL": [(0.12, 4.0, 0.08), (0.40, 2.5, 0.10)],
    "START": [(0.10, -1.0, 0.05), (0.30, 4.0, 0.12)],
    "STOP": [(0.12, -1.5, 0.05), (0.30, 3.0, 0.08), (0.50, -2.5, 0.06)],
}


def _displacement(label: str, t: np.ndarray, rng: np.random.Generator, speed: float, amp: float) -> np.ndarray:
    onset = rng.uniform(0.3, 0.7)
    tau = (t - onset) * speed
    d = 0.3 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.3))  # breathing
    if label in WORDS:
        for c, a, w in WORDS[label]:
            c *= rng.uniform(0.85, 1.15)
            d += amp * a * rng.uniform(0.75, 1.25) * np.exp(-0.5 * ((tau - c) / w) ** 2)
    elif label == "CHEWING":
        f = rng.uniform(1.2, 1.9)
        d += amp * 3.0 * np.clip(np.sin(2 * np.pi * f * tau + rng.uniform(0, 6.3)), 0, None) * (tau > -0.2)
    elif label == "YAWNING":
        d += amp * 10.0 * np.clip(tau / 1.2, 0, 1) ** 2 * (tau < 2.0)
    elif label == "TALKING":
        for c in np.cumsum(rng.uniform(0.1, 0.3, 10)):
            d += amp * rng.uniform(-3, 4) * np.exp(-0.5 * ((tau - c) / rng.uniform(0.03, 0.08)) ** 2)
    elif label != "REST":
        raise ValueError(label)
    return d / 1000.0


def synth_clip(cfg: Config, label: str, rng: np.random.Generator, user: dict, probe_on: bool = True) -> np.ndarray:
    sr = cfg.probe.sample_rate
    dur = cfg.data.clip_pre_s + cfg.data.clip_post_s
    t = np.arange(int(dur * sr)) / sr
    tones = np.asarray(cfg.probe.tones_hz)
    phases = _tone_phases(len(tones))
    scale = cfg.probe.amplitude * _probe_scale(tuple(float(f) for f in tones), sr)
    x = np.zeros_like(t)
    if probe_on:
        disp = _displacement(label, t - cfg.data.clip_pre_s, rng, user["speed"], user["amp"])
        tremor = 0.0002 * np.sin(2 * np.pi * rng.uniform(4, 8) * t + rng.uniform(0, 6.3))
        for k, f in enumerate(tones):
            resp = user["response"][k]
            static = resp * np.cos(2 * np.pi * f * t + phases[k] + user["static_phase"][k])
            for refl, gain, lag in ((user["d0"], user["echo"], 0.0), (user["d0"] + 0.02, 0.5 * user["echo"], 0.03)):
                path = 2 * (refl + np.interp(t - lag, t, disp) + tremor)
                static += resp * gain * np.cos(2 * np.pi * f * (t - path / SPEED_OF_SOUND) + phases[k])
            x += scale * static / len(tones)
    x += user["noise"] * rng.standard_normal(len(t))
    if label == "TALKING":  # audible speech energy (far below the probe band)
        x += 0.02 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 4 * t) > 0)
    return x.astype(np.float32)


def make_user(rng: np.random.Generator, n_tones: int) -> dict:
    return {
        "speed": rng.uniform(0.8, 1.25), "amp": rng.uniform(0.7, 1.4),
        "d0": rng.uniform(0.07, 0.13), "echo": rng.uniform(0.01, 0.03),
        "response": rng.uniform(0.3, 1.0, n_tones), "static_phase": rng.uniform(0, 2 * np.pi, n_tones),
        "noise": 10 ** rng.uniform(-3.7, -3.2),
    }


def generate(cfg: Config, out_root: str | Path, participants: int = 6, sessions: int = 2, reps: int = 4,
             probe_off_reps: int = 1, seed: int = 0, log=print) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(out_root)
    n = 0
    for p in range(participants):
        base = make_user(rng, len(cfg.probe.tones_hz))
        for s in range(sessions + (1 if probe_off_reps else 0)):
            probe_on = s < sessions
            user = dict(base)
            user["d0"] = base["d0"] * rng.uniform(0.85, 1.15)  # re-holding the phone
            user["static_phase"] = rng.uniform(0, 2 * np.pi, len(base["response"]))
            sess = f"S{s + 1}" if probe_on else "S-probeoff"
            d = root / f"P{p + 1:02d}" / sess
            d.mkdir(parents=True, exist_ok=True)
            meta = {"participant": f"P{p + 1:02d}", "session": sess, "mode": "synthetic",
                    "sample_rate": cfg.probe.sample_rate, "probe": "on" if probe_on else "off",
                    "distance_cm": round(100 * user["d0"]), "orientation": "front",
                    "condition": "normal" if probe_on else "probe_off", "device": "synthetic"}
            (d / "session.json").write_text(json.dumps(meta, indent=2))
            rows, trial = [], 0
            for _ in range(reps if probe_on else probe_off_reps):
                for label in cfg.classes:
                    trial += 1
                    name = f"clips/{trial:04d}_{label}.wav"
                    write_wav(d / name, synth_clip(cfg, label, rng, user, probe_on), cfg.probe.sample_rate)
                    rows.append({"clip": name, "trial": trial, "label": label, "imu": ""})
                    n += 1
            with (d / "clips.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["clip", "trial", "label", "imu"])
                w.writeheader()
                w.writerows(rows)
    log(f"wrote {n} synthetic clips under {root}")
    return root
