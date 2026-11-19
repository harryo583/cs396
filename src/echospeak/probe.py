"""Probe synthesis, audible cue sounds, and coherent I/Q demodulation.

The probe is a sum of K continuous-wave tones (default 18.0-19.75 kHz). A moving
reflector (lips/jaw) adds an echo whose phase follows the round-trip path length
and whose frequency is Doppler-shifted. Demodulating each tone to complex
baseband recovers ``h_k(t)``: static paths (direct speaker->mic leakage, face,
hand) appear as a constant, and articulator motion as a time-varying residual.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from scipy import signal

from .config import FeatureConfig, ProbeConfig

BEEP_HZ = 1000.0
BEEP_S = 0.08
CHIRP_S = 0.4
CHIRP_F0, CHIRP_F1 = 800.0, 4000.0
INTERMEDIATE_RATE = 2000


# ----------------------------------------------------------------------------
# Probe synthesis
# ----------------------------------------------------------------------------

def _tone_phases(n_tones: int) -> np.ndarray:
    # Schroeder phases keep the crest factor of a multi-tone signal low.
    k = np.arange(1, n_tones + 1)
    return -np.pi * k * (k - 1) / n_tones


@lru_cache(maxsize=8)
def _probe_scale(tones: tuple[float, ...], sample_rate: int) -> float:
    t = np.arange(sample_rate) / sample_rate
    raw = _raw_probe(np.asarray(tones), t)
    return 1.0 / max(np.max(np.abs(raw)), 1e-9)


def _raw_probe(tones: np.ndarray, t: np.ndarray) -> np.ndarray:
    phases = _tone_phases(len(tones))
    return np.cos(2 * np.pi * tones[:, None] * t[None, :] + phases[:, None]).sum(axis=0)


def probe_block(cfg: ProbeConfig, start: int, n: int) -> np.ndarray:
    """Samples ``start .. start+n`` of the (infinite, phase-continuous) probe."""
    tones = tuple(float(f) for f in cfg.tones_hz)
    t = (start + np.arange(n)) / cfg.sample_rate
    scale = _probe_scale(tones, cfg.sample_rate)
    return (cfg.amplitude * scale * _raw_probe(np.asarray(tones), t)).astype(np.float32)


def probe_signal(cfg: ProbeConfig, seconds: float, fade_s: float = 0.05) -> np.ndarray:
    n = int(round(seconds * cfg.sample_rate))
    x = probe_block(cfg, 0, n)
    fade = min(int(fade_s * cfg.sample_rate), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


# ----------------------------------------------------------------------------
# Audible cue sounds (far below the probe band; removed by demodulation)
# ----------------------------------------------------------------------------

def beep(sample_rate: int, amplitude: float = 0.4) -> np.ndarray:
    n = int(BEEP_S * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.hanning(n) * np.sin(2 * np.pi * BEEP_HZ * t)).astype(np.float32)


def sync_chirp(sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(CHIRP_S * sample_rate)
    t = np.arange(n) / sample_rate
    x = signal.chirp(t, CHIRP_F0, CHIRP_S, CHIRP_F1)
    return (amplitude * signal.windows.tukey(n, 0.2) * x).astype(np.float32)


# ----------------------------------------------------------------------------
# Demodulation
# ----------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _baseband_filter(lowpass_hz: float) -> np.ndarray:
    return signal.firwin(255, lowpass_hz, fs=INTERMEDIATE_RATE)


def demodulate(audio: np.ndarray, sample_rate: int, probe: ProbeConfig, feat: FeatureConfig) -> np.ndarray:
    """Return complex baseband ``[K, N]`` at ``feat.baseband_rate`` (zero-phase filtered).

    Sample ``i`` of the output corresponds to time ``i / baseband_rate`` in the input.
    Works for any input rate whose Nyquist exceeds the highest tone (e.g. 44.1/48/96 kHz).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("demodulate expects mono audio")
    tones = np.asarray(probe.tones_hz, dtype=np.float64)
    if tones.max() >= sample_rate / 2:
        raise ValueError(f"sample rate {sample_rate} cannot represent the probe tones")
    t = np.arange(len(x)) / sample_rate
    mixed = x[None, :] * np.exp(-2j * np.pi * tones[:, None] * t[None, :])
    g = gcd(int(sample_rate), INTERMEDIATE_RATE)
    mid = signal.resample_poly(mixed, INTERMEDIATE_RATE // g, int(sample_rate) // g, axis=1)
    taps = _baseband_filter(float(feat.lowpass_hz))
    if mid.shape[1] > 3 * len(taps):
        mid = signal.filtfilt(taps, [1.0], mid, axis=1)
    else:
        mid = signal.lfilter(taps, [1.0], mid, axis=1)
    step = INTERMEDIATE_RATE // feat.baseband_rate
    if INTERMEDIATE_RATE % feat.baseband_rate:
        raise ValueError("baseband_rate must divide 2000")
    # x2: undo the 1/2 amplitude of mixing a real cosine down to baseband.
    return (2.0 * mid[:, ::step]).astype(np.complex64)


def tone_snr_db(audio: np.ndarray, sample_rate: int, probe: ProbeConfig) -> list[float]:
    """Per-tone power relative to the median spectral floor in the band between tones."""
    x = np.asarray(audio, dtype=np.float64)
    f, p = signal.welch(x, fs=sample_rate, nperseg=8192)
    tones = np.asarray(probe.tones_hz)
    out = []
    for tone in tones:
        peak = p[np.argmin(np.abs(f - tone))]
        mask = (np.abs(f - tone) > 40) & (np.abs(f - tone) < 120)
        floor = np.median(p[mask]) if mask.any() else np.median(p)
        out.append(float(10 * np.log10(max(peak, 1e-30) / max(floor, 1e-30))))
    return out
