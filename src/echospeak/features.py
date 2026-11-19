"""Model inputs computed from complex baseband windows (batched, in torch).

Pipeline for one window ``bb[k, n]`` (tone k, baseband sample n):

1. ``rel = bb / mean_n(bb) - 1``: the echo *relative to the static path*. This
   cancels each tone's speaker/microphone gain and arbitrary carrier phase, so the
   features do not depend on phone volume or hardware frequency response.
2. (train only) physically motivated augmentation of ``rel``: echo gain, global
   phase rotation (unknown mouth distance mod one wavelength), tone dropout,
   noise.
3. Two views:
   * ``doppler``  ``[B, K, F, T]``  log-power Doppler spectrogram per tone
     (+-``doppler_max_hz``), i.e. radial velocity of articulators over time;
   * ``iq``       ``[B, 5K(+6), T]`` frame-rate I/Q trajectory, its derivative and
     log-amplitude change per tone (+ optional IMU channels).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import AugmentConfig, FeatureConfig


def frames_per_window(feat: FeatureConfig, window_s: float) -> int:
    return int(round(window_s * feat.frame_rate))


def hop(feat: FeatureConfig) -> int:
    return feat.baseband_rate // feat.frame_rate


def doppler_bins(feat: FeatureConfig) -> int:
    res = feat.baseband_rate / feat.doppler_nfft
    return 2 * int(feat.doppler_max_hz // res) + 1


def iq_channels(n_tones: int, feat: FeatureConfig) -> int:
    return 5 * n_tones + (6 if feat.use_imu else 0)


def relative_echo(bb: torch.Tensor) -> torch.Tensor:
    static = bb.mean(dim=-1, keepdim=True)
    mag = static.abs().clamp_min(1e-12)
    static = torch.where(static.abs() < 1e-12, mag.to(bb.dtype), static)
    return bb / static - 1


def augment_echo(rel: torch.Tensor, aug: AugmentConfig, gen: torch.Generator | None = None) -> torch.Tensor:
    b, k, _ = rel.shape
    dev = rel.device

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=dev)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=dev)

    if aug.gain_log_std > 0:
        rel = rel * torch.exp(aug.gain_log_std * randn(b, 1, 1))
    if aug.noise_std > 0:
        rms = rel.abs().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        noise = torch.complex(randn(*rel.shape), randn(*rel.shape)) / math.sqrt(2)
        rel = rel + aug.noise_std * rand(b, 1, 1) * rms * noise
    if aug.tone_dropout > 0 and k > 1:
        drop = rand(b) < aug.tone_dropout
        which = torch.randint(0, k, (b,), generator=gen, device=dev)
        mask = torch.ones(b, k, 1, device=dev)
        mask[drop, which[drop], 0] = 0.0
        rel = rel * mask
    return rel


def rotate_echo(rel: torch.Tensor, aug: AugmentConfig, gen: torch.Generator | None = None) -> torch.Tensor:
    if not aug.rotation:
        return rel
    b, k, _ = rel.shape
    theta = 2 * math.pi * torch.rand(b, 1, 1, generator=gen, device=rel.device)
    theta = theta + aug.tone_phase_jitter * torch.randn(b, k, 1, generator=gen, device=rel.device)
    return rel * torch.polar(torch.ones_like(theta), theta)


def compute_features(
    bb: torch.Tensor,
    feat: FeatureConfig,
    imu: torch.Tensor | None = None,
    aug: AugmentConfig | None = None,
    gen: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """``bb``: complex ``[B, K, N]`` baseband window. Returns ``doppler`` and ``iq`` tensors."""
    if bb.dim() == 2:
        bb = bb[None]
    bb = bb.to(torch.complex64)
    b, k, n = bb.shape
    h = hop(feat)
    t_frames = n // h

    rel = relative_echo(bb)
    if aug is not None and aug.enabled:
        rel = augment_echo(rel, aug, gen)
    logamp = torch.log((1 + rel).abs().clamp_min(1e-6))
    if aug is not None and aug.enabled:
        rel = rotate_echo(rel, aug, gen)

    # --- Doppler spectrogram --------------------------------------------------
    nfft = feat.doppler_nfft
    window = torch.hann_window(nfft, device=bb.device)
    spec = torch.stft(rel.reshape(b * k, n), n_fft=nfft, hop_length=h, window=window,
                      center=True, return_complex=True, onesided=False)
    power = spec.abs().pow(2)
    power = torch.fft.fftshift(power, dim=1)
    centre = nfft // 2
    half = doppler_bins(feat) // 2
    power = power[:, centre - half: centre + half + 1, :t_frames]
    doppler = torch.log(power + 1e-8).reshape(b, k, 2 * half + 1, t_frames)

    # --- Frame-rate I/Q trajectory ---------------------------------------------
    n_used = t_frames * h
    re = F.avg_pool1d(rel.real[..., :n_used], h)
    im = F.avg_pool1d(rel.imag[..., :n_used], h)
    la = F.avg_pool1d(logamp[..., :n_used], h)
    d_re = torch.diff(re, dim=-1, prepend=re[..., :1])
    d_im = torch.diff(im, dim=-1, prepend=im[..., :1])
    s = feat.iq_scale
    iq = torch.cat([
        torch.asinh(re / s), torch.asinh(im / s),
        torch.asinh(d_re / (s / 4)), torch.asinh(d_im / (s / 4)),
        torch.asinh(la / s),
    ], dim=1)
    if feat.use_imu:
        if imu is None:
            raise ValueError("features.use_imu is true but no IMU data was supplied")
        iq = torch.cat([iq, imu.to(iq)[..., :t_frames]], dim=1)

    out = {"doppler": doppler, "iq": iq}
    if aug is not None and aug.enabled and aug.time_masks > 0:
        out = time_mask(out, aug, gen)
    return out


def time_mask(feats: dict[str, torch.Tensor], aug: AugmentConfig, gen: torch.Generator | None) -> dict[str, torch.Tensor]:
    doppler, iq = feats["doppler"].clone(), feats["iq"].clone()
    b, t = iq.shape[0], iq.shape[-1]
    max_w = max(1, int(aug.time_mask_frac * t))
    d_fill = doppler.mean(dim=-1, keepdim=True)
    i_fill = iq.mean(dim=-1, keepdim=True)
    for _ in range(aug.time_masks):
        widths = torch.randint(0, max_w + 1, (b,), generator=gen)
        starts = (torch.rand(b, generator=gen) * (t - widths).clamp_min(1)).long()
        idx = torch.arange(t)
        mask = (idx[None] >= starts[:, None]) & (idx[None] < (starts + widths)[:, None])  # [B, T]
        mask = mask.to(iq.device)
        doppler = torch.where(mask[:, None, None, :], d_fill, doppler)
        iq = torch.where(mask[:, None, :], i_fill, iq)
    return {"doppler": doppler, "iq": iq}


def motion_trace(bb: torch.Tensor, feat: FeatureConfig) -> torch.Tensor:
    """Scalar echo-motion energy per frame ``[T]`` (for plots and the live UI)."""
    rel = relative_echo(bb[None] if bb.dim() == 2 else bb)[0]
    h = hop(feat)
    n = (rel.shape[-1] // h) * h
    d = torch.diff(rel[..., :n], dim=-1, prepend=rel[..., :1])
    energy = d.abs().pow(2).mean(dim=0)
    return F.avg_pool1d(energy[None, None], h)[0, 0].sqrt()
