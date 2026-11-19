"""Optional phone IMU (accelerometer + gyroscope) support.

Uses (plan: "IMU-assisted motion gating or feature fusion"):
* gating: reject trials whose gyroscope RMS exceeds ``gate.imu_max_gyro_rms``;
* fusion: ``features.use_imu: true`` appends 6 IMU channels to the I/Q branch.

Sensor apps export different CSV layouts. ``read_xyz_csv`` accepts a time column
(seconds, milliseconds, nanoseconds, or Unix time) and x/y/z columns. The phone
audio and IMU clocks are aligned by having the participant tap the phone three
times before the session's sync chirp: the taps are sharp transients in both.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import signal

IMU_RATE = 100.0
CLIP_COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]


def load_clip_imu(path: str | Path, frame_rate: int, clip_s: float) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    t, values = data[:, 0], data[:, 1:7]
    grid = np.arange(int(round(clip_s * frame_rate))) / frame_rate
    return np.stack([np.interp(grid, t, values[:, j]) for j in range(6)]).astype(np.float32)


def _to_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if np.nanmax(t) > 1e17:      # Unix ns
        t = t / 1e9
    elif np.nanmax(t) > 1e14:    # Unix us
        t = t / 1e6
    elif np.nanmax(t) > 1e11:    # Unix ms
        t = t / 1e3
    return t


def read_xyz_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_seconds, xyz[N,3])`` from a sensor CSV with a time column and x/y/z columns."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader)]
        rows = [r for r in reader if r]

    def find(cands):
        for c in cands:
            for i, h in enumerate(header):
                if h == c or h.startswith(c + " ") or h.startswith(c + "("):
                    return i
        return None

    ti = find(["seconds_elapsed", "time", "timestamp", "t", "time (s)"])
    xi, yi, zi = find(["x", "ax", "gx", "acceleration x", "gyroscope x"]), find(["y", "ay", "gy", "acceleration y", "gyroscope y"]), find(["z", "az", "gz", "acceleration z", "gyroscope z"])
    if None in (ti, xi, yi, zi):
        raise ValueError(f"{path}: need a time column and x/y/z columns, found {header}")
    arr = np.array([[float(r[ti]), float(r[xi]), float(r[yi]), float(r[zi])] for r in rows])
    order = np.argsort(arr[:, 0])
    return _to_seconds(arr[order, 0]), arr[order, 1:]


def read_session_imu(accel_csv: str | Path, gyro_csv: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, data[N,6])`` on a common 100 Hz grid. ``accel_csv`` may hold all six
    columns (ax..gz) if ``gyro_csv`` is None."""
    if gyro_csv is None:
        data = np.loadtxt(accel_csv, delimiter=",", skiprows=1, ndmin=2)
        t, vals = _to_seconds(data[:, 0]), data[:, 1:7]
        ta = tg = t
        acc, gyr = vals[:, :3], vals[:, 3:6]
    else:
        ta, acc = read_xyz_csv(accel_csv)
        tg, gyr = read_xyz_csv(gyro_csv)
    t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
    grid = np.arange(t0, t1, 1 / IMU_RATE)
    out = np.concatenate([np.stack([np.interp(grid, ta, acc[:, j]) for j in range(3)], 1),
                          np.stack([np.interp(grid, tg, gyr[:, j]) for j in range(3)], 1)], axis=1)
    return grid - grid[0], out


def _events(x: np.ndarray) -> np.ndarray:
    x = np.abs(x - np.median(x))
    x = x / (np.percentile(x, 99.5) + 1e-12)
    return np.clip(x, 0, 3)


def align_by_taps(audio: np.ndarray, sr: int, imu_t: np.ndarray, imu: np.ndarray, search_s: float) -> float:
    """Return ``offset`` such that ``imu_time = audio_time + offset``.

    Uses the first ``search_s`` seconds of audio (before the sync chirp)."""
    n = int(min(len(audio), search_s * sr))
    sos = signal.butter(4, [200, 6000], btype="band", fs=sr, output="sos")
    env = np.abs(signal.sosfiltfilt(sos, audio[:n]))
    hop = int(sr / IMU_RATE)
    env = env[: len(env) // hop * hop].reshape(-1, hop).max(axis=1)
    acc_mag = np.linalg.norm(imu[:, :3], axis=1)
    jerk = np.abs(np.diff(acc_mag, prepend=acc_mag[0]))
    a, b = _events(env), _events(jerk)
    corr = signal.correlate(b, a, mode="full")
    lag = np.argmax(corr) - (len(a) - 1)
    return float(lag / IMU_RATE + imu_t[0])


def write_clip_imu(path: str | Path, imu_t: np.ndarray, imu: np.ndarray, start_s: float, clip_s: float) -> None:
    grid = np.arange(int(round(clip_s * IMU_RATE))) / IMU_RATE
    vals = np.stack([np.interp(start_s + grid, imu_t, imu[:, j]) for j in range(6)], 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([grid, vals]), delimiter=",", header=",".join(CLIP_COLUMNS),
               comments="", fmt="%.5f")
