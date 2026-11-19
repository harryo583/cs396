"""Signal processing: probe, demodulation, features, segmentation, IMU alignment."""

import csv
import json

import numpy as np
import pytest
import torch

from echospeak.audio import read_wav, write_wav
from echospeak.config import Config
from echospeak.features import compute_features, doppler_bins, frames_per_window, iq_channels
from echospeak.probe import beep, demodulate, probe_block, probe_signal, sync_chirp, tone_snr_db
from echospeak.segment import segment_session

C = 343.0


def test_probe_is_phase_continuous_and_bounded():
    cfg = Config()
    whole = probe_block(cfg.probe, 0, 4096)
    parts = np.concatenate([probe_block(cfg.probe, 0, 1000), probe_block(cfg.probe, 1000, 3096)])
    assert np.allclose(whole, parts, atol=1e-6)
    assert np.abs(probe_signal(cfg.probe, 1.0)).max() <= cfg.probe.amplitude + 1e-4
    assert min(tone_snr_db(probe_signal(cfg.probe, 2.0), 48000, cfg.probe)) > 40


@pytest.mark.parametrize("sr", [44100, 48000])
def test_demodulation_recovers_doppler_sign(sr):
    """An approaching reflector must appear at a positive baseband frequency 2vf/c."""
    cfg = Config()
    t = np.arange(int(1.5 * sr)) / sr
    v = 0.1  # m/s towards the phone
    dist = 0.10 - v * t
    tones = np.asarray(cfg.probe.tones_hz)
    x = sum(np.cos(2 * np.pi * f * t) + 0.05 * np.cos(2 * np.pi * f * (t - 2 * dist / C)) for f in tones)
    bb = demodulate(x / len(tones), sr, cfg.probe, cfg.features)
    assert bb.shape == (len(tones), int(1.5 * cfg.features.baseband_rate))
    rel = bb[3] / bb[3].mean() - 1
    rel = rel[50:-50]
    spec = np.fft.fftshift(np.abs(np.fft.fft(rel * np.hanning(len(rel)))))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(rel), 1 / cfg.features.baseband_rate))
    expected = 2 * v * tones[3] / C
    assert abs(freqs[np.argmax(spec)] - expected) < 2.0


def test_feature_shapes_and_rotation_invariance():
    cfg = Config()
    n = int(cfg.data.window_s * cfg.features.baseband_rate)
    k = len(cfg.probe.tones_hz)
    rng = np.random.default_rng(0)
    bb = (1 + 0.01 * (rng.standard_normal((2, k, n)) + 1j * rng.standard_normal((2, k, n)))).astype(np.complex64)
    bb = torch.from_numpy(bb)
    feats = compute_features(bb, cfg.features)
    t = frames_per_window(cfg.features, cfg.data.window_s)
    assert feats["doppler"].shape == (2, k, doppler_bins(cfg.features), t)
    assert feats["iq"].shape == (2, iq_channels(k, cfg.features), t)
    assert torch.isfinite(feats["doppler"]).all() and torch.isfinite(feats["iq"]).all()
    # Rotation-only augmentation leaves the Doppler view untouched.
    aug = cfg.augment
    aug.gain_log_std = aug.noise_std = aug.tone_dropout = 0.0
    aug.time_masks = 0
    rotated = compute_features(bb, cfg.features, aug=aug, gen=torch.Generator().manual_seed(1))
    assert torch.allclose(rotated["doppler"], feats["doppler"], atol=1e-3)
    assert not torch.allclose(rotated["iq"], feats["iq"])


def test_features_are_gain_invariant():
    cfg = Config()
    rng = np.random.default_rng(1)
    bb = torch.from_numpy((1 + 0.01 * rng.standard_normal((1, 8, 1000))).astype(np.complex64))
    a = compute_features(bb, cfg.features)
    b = compute_features(bb * (0.2 * np.exp(1j * 0.7)), cfg.features)
    assert torch.allclose(a["iq"], b["iq"], atol=1e-3)
    above_floor = a["doppler"] > -12  # bins near the log(1e-8) floor are float32 rounding noise
    assert torch.allclose(a["doppler"][above_floor], b["doppler"][above_floor], atol=1e-2)


@pytest.mark.parametrize("mode", ["duplex", "prompt"])
def test_segmentation_finds_cues_with_offset_and_drift(tmp_path, mode):
    cfg = Config()
    lsr = 48000
    sr = 48000
    drift = 1 + 80e-6 if mode == "prompt" else 1.0
    offset = 1.234 if mode == "prompt" else 0.012  # phone started earlier / I/O latency
    labels = ["NEXT", "REST", "STOP", "BACK"]
    go_l = [int((4.0 + 4.5 * i) * lsr) for i in range(len(labels))]  # laptop clock
    chirp_l = int(2.0 * lsr)
    dur = (go_l[-1] / lsr + 4) * drift + offset
    rng = np.random.default_rng(0)
    audio = 0.001 * rng.standard_normal(int(dur * sr)).astype(np.float32)
    audio += probe_block(cfg.probe, 0, len(audio)) * 0.3

    def place(snd, laptop_sample):
        pos = int(round((laptop_sample / lsr * drift + offset) * sr))
        audio[pos: pos + len(snd)] += 0.2 * snd

    if mode == "prompt":
        place(sync_chirp(sr), chirp_l)
    for g in go_l:
        place(beep(sr), g)
    d = tmp_path / "P01" / "S1"
    d.mkdir(parents=True)
    write_wav(d / ("phone.wav" if mode == "prompt" else "session.wav"), audio, sr)
    meta = {"participant": "P01", "session": "S1", "mode": mode, "sample_rate": lsr,
            "audio": "phone.wav" if mode == "prompt" else "session.wav"}
    if mode == "prompt":
        meta["chirp_sample"] = chirp_l
    (d / "session.json").write_text(json.dumps(meta))
    with (d / "cues.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "label", "go_sample", "status"])
        w.writeheader()
        for i, (lab, g) in enumerate(zip(labels, go_l)):
            w.writerow({"trial": i + 1, "label": lab, "go_sample": g, "status": "discarded" if i == 1 else "ok"})
    rows = segment_session(d, cfg, log=lambda *_: None)
    assert [r["label"] for r in rows] == ["NEXT", "STOP", "BACK"]
    # The beep must start exactly clip_pre_s into each clip (within 3 ms).
    for r in rows:
        clip, _ = read_wav(d / r["clip"])
        from echospeak.segment import find_beep
        pos, _ = find_beep(clip, sr, int(cfg.data.clip_pre_s * sr), int(0.2 * sr))
        assert abs(pos / sr - cfg.data.clip_pre_s) < 0.003


def test_imu_tap_alignment(tmp_path):
    from echospeak.imu import align_by_taps, load_clip_imu, write_clip_imu
    sr = 48000
    rng = np.random.default_rng(0)
    taps = [1.0, 1.45, 2.2]
    audio = 0.001 * rng.standard_normal(5 * sr)
    for tap in taps:
        i = int(tap * sr)
        audio[i: i + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    true_offset = 3.37  # imu_time = audio_time + offset
    imu_t = np.arange(0, 12, 0.01)
    imu = np.zeros((len(imu_t), 6))
    imu[:, 2] = 9.81 + 0.01 * rng.standard_normal(len(imu_t))
    for tap in taps:
        imu[int(round((tap + true_offset) * 100)), 2] += 3.0
    est = align_by_taps(audio, sr, imu_t, imu, search_s=5)
    assert abs(est - true_offset) < 0.02
    write_clip_imu(tmp_path / "c.csv", imu_t, imu, 2.0, 3.0)
    arr = load_clip_imu(tmp_path / "c.csv", 100, 3.0)
    assert arr.shape == (6, 300)
