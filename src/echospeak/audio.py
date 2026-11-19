"""WAV I/O and (optional) sound-card helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_wav(path: str | Path, channel: int = 0) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 in [-1, 1]. Multi-channel files use ``channel``."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x[:, min(channel, x.shape[1] - 1)]
    return x, int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, pcm16: bool = False) -> None:
    """Write a WAV: float32 by default, or 16-bit PCM (half the size; enough for a 16/24-bit mic)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    if pcm16:
        x = np.round(np.clip(x, -1.0, 32767 / 32768) * 32768).astype(np.int16)
    wavfile.write(str(path), int(sample_rate), x)


def require_sounddevice():
    try:
        import sounddevice as sd  # noqa: F401
    except (ImportError, OSError) as exc:  # pragma: no cover - hardware dependent
        raise SystemExit("This command needs the 'sounddevice' package and PortAudio: "
                         "pip install -e '.[audio]'") from exc
    return sd


def list_devices() -> str:  # pragma: no cover - hardware dependent
    sd = require_sounddevice()
    return str(sd.query_devices())


def parse_device(value: str | None):
    if value is None or value == "":
        return None
    return int(value) if value.lstrip("-").isdigit() else value
