"""Typed configuration loaded from YAML, with ``--set section.key=value`` overrides.

Every experiment is fully described by one :class:`Config`. The ``probe`` and
``data`` sections describe the recordings and must be identical for all members of
an ensemble; ``features``, ``model``, ``train`` and ``augment`` may differ per member.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

COMMANDS = ["NEXT", "BACK", "SELECT", "CANCEL", "START", "STOP"]
NON_COMMANDS = ["REST", "CHEWING", "YAWNING", "TALKING"]


@dataclass
class ProbeConfig:
    sample_rate: int = 48000
    # Multi-tone continuous wave. 250 Hz spacing >> max lip Doppler (~±60 Hz at 19 kHz).
    tones_hz: list[float] = field(default_factory=lambda: [18000.0 + 250.0 * k for k in range(8)])
    amplitude: float = 0.6  # peak amplitude of the summed probe (full scale = 1.0)


@dataclass
class DataConfig:
    commands: list[str] = field(default_factory=lambda: list(COMMANDS))
    non_commands: list[str] = field(default_factory=lambda: list(NON_COMMANDS))
    clip_pre_s: float = 0.75   # audio kept before the GO cue in each clip
    clip_post_s: float = 2.25  # audio kept after the GO cue
    window_s: float = 2.0      # model input length (also the live sliding window)
    edge_margin_s: float = 0.1  # never crop within this distance of a clip edge (filter transients)
    eval_offset_s: float = 0.5  # deterministic crop start for evaluation (window = GO-0.25 .. GO+1.75)

    @property
    def classes(self) -> list[str]:
        return list(self.commands) + list(self.non_commands)


@dataclass
class FeatureConfig:
    baseband_rate: int = 500   # complex baseband rate after demodulation
    lowpass_hz: float = 110.0  # per-tone low-pass (must be < half the tone spacing)
    frame_rate: int = 100      # frame rate of the model input sequence
    doppler_nfft: int = 64     # STFT size on the 500 Hz baseband (7.8 Hz bins)
    doppler_max_hz: float = 80.0
    iq_scale: float = 3e-3     # asinh compression scale for relative I/Q channels
    use_imu: bool = False      # append 6 IMU channels to the sequence branch


@dataclass
class ModelConfig:
    branches: str = "both"        # both | doppler | iq
    temporal: str = "transformer"  # none | gru | transformer | tcn
    channels: int = 64             # conv width (first stage); later stages scale 1x/1.5x
    d_model: int = 96
    layers: int = 2                # temporal layers
    heads: int = 4
    dropout: float = 0.3


@dataclass
class AugmentConfig:
    enabled: bool = True
    time_stretch: float = 0.12     # uniform speed factor in [1-x, 1+x]
    gain_log_std: float = 0.3      # log-normal gain on the dynamic (echo) component
    rotation: bool = True          # random global phase rotation of the echo component
    tone_phase_jitter: float = 0.3  # extra per-tone phase jitter (rad)
    tone_dropout: float = 0.15     # probability of zeroing one tone
    noise_std: float = 0.1         # complex noise, relative to the clip's echo RMS
    time_masks: int = 2
    time_mask_frac: float = 0.08
    mixup_alpha: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.02
    warmup_frac: float = 0.05
    label_smoothing: float = 0.1
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"           # auto | cpu | cuda | mps
    num_workers: int = 0
    tta_shifts_s: list[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])


@dataclass
class GateConfig:
    threshold: float = 0.7         # minimum probability to trigger a command
    min_consecutive: int = 2       # live: consecutive windows agreeing on the command
    cooldown_s: float = 1.5        # live: dead time after a trigger
    hop_s: float = 0.1             # live: inference period
    target_false_trigger: float = 0.02  # used to recommend a threshold from validation data
    imu_max_gyro_rms: float = 0.3  # rad/s; trials above are rejected when IMU gating is on


@dataclass
class Config:
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "Config":
        values = values or {}
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            section_cls = type(getattr(cls(), f.name))
            section = values.get(f.name) or {}
            names = {g.name for g in dataclasses.fields(section_cls)}
            bad = set(section) - names
            if bad:
                raise ValueError(f"Unknown keys in '{f.name}': {sorted(bad)}")
            kwargs[f.name] = section_cls(**section)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        tones = sorted(self.probe.tones_hz)
        if len(tones) < 1:
            raise ValueError("probe.tones_hz must not be empty")
        if tones[-1] >= self.probe.sample_rate / 2:
            raise ValueError("probe tones must be below Nyquist")
        if len(tones) > 1:
            spacing = min(b - a for a, b in zip(tones, tones[1:]))
            if self.features.lowpass_hz >= spacing / 2:
                raise ValueError("features.lowpass_hz must be below half the tone spacing")
        if self.features.baseband_rate % self.features.frame_rate:
            raise ValueError("features.baseband_rate must be a multiple of features.frame_rate")
        if self.model.branches not in {"both", "doppler", "iq"}:
            raise ValueError("model.branches must be both | doppler | iq")
        if self.model.temporal not in {"none", "gru", "transformer", "tcn"}:
            raise ValueError("model.temporal must be none | gru | transformer | tcn")
        d = self.data
        if d.window_s + 2 * d.edge_margin_s > d.clip_pre_s + d.clip_post_s:
            raise ValueError("data.window_s plus margins exceeds the clip length")
        if not (d.edge_margin_s <= d.eval_offset_s <= d.clip_pre_s + d.clip_post_s - d.window_s - d.edge_margin_s):
            raise ValueError("data.eval_offset_s puts the evaluation window outside the clip margins")
        if len(set(d.classes)) != len(d.classes):
            raise ValueError("duplicate class names")

    # Convenience -----------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return self.data.classes


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(values: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    values = copy.deepcopy(values)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.key, got {key!r}")
        values.setdefault(parts[0], {})[parts[1]] = _parse_value(raw)
    return values


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
    return Config.from_dict(apply_overrides(values, overrides))


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
