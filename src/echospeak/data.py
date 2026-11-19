"""Manifest, baseband cache, held-out splits, CV folds and the training dataset.

Dataset layout produced by ``echospeak session`` / ``import`` / ``synth``::

    data/raw/<participant>/<session>/
        session.json     participant, session, device, probe on/off, distance, ...
        cues.csv         one row per prompted trial (label, GO sample, status)
        session.wav      full continuous recording (duplex mode) or phone.wav
        clips/<trial>.wav  one clip per trial, GO cue at ``clip_pre_s``
        clips.csv        clip index (written by segmentation)
        imu/<trial>.csv  optional per-clip IMU (t, ax, ay, az, gx, gy, gz)

``echospeak index`` gathers every ``clips.csv`` into a single ``manifest.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import read_wav
from .config import Config
from .features import frames_per_window, hop
from .probe import demodulate

MANIFEST_FIELDS = [
    "clip", "participant", "session", "trial", "label", "probe", "distance_cm",
    "orientation", "condition", "device", "imu",
]


@dataclass
class Clip:
    clip: str            # path relative to the manifest directory (also the clip id)
    participant: str
    session: str
    trial: str
    label: str
    probe: str = "on"
    distance_cm: str = ""
    orientation: str = ""
    condition: str = ""
    device: str = ""
    imu: str = ""        # path relative to the manifest directory, or empty
    root: Path = field(default=Path("."), repr=False, compare=False)

    @property
    def path(self) -> Path:
        return self.root / self.clip

    @property
    def imu_path(self) -> Path | None:
        return self.root / self.imu if self.imu else None

    @property
    def group(self) -> str:
        return f"{self.participant}/{self.session}"


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def index_dataset(root: str | Path, out: str | Path) -> list[Clip]:
    """Collect all ``clips.csv`` files under ``root`` into one manifest at ``out``."""
    root, out = Path(root).resolve(), Path(out).resolve()
    rows: list[dict] = []
    for clips_csv in sorted(root.rglob("clips.csv")):
        session_dir = clips_csv.parent
        meta = json.loads((session_dir / "session.json").read_text()) if (session_dir / "session.json").exists() else {}
        with clips_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rec = {k: "" for k in MANIFEST_FIELDS}
                rec.update({
                    "participant": meta.get("participant", session_dir.parent.name),
                    "session": meta.get("session", session_dir.name),
                    "probe": meta.get("probe", "on"),
                    "distance_cm": str(meta.get("distance_cm", "")),
                    "orientation": meta.get("orientation", ""),
                    "condition": meta.get("condition", ""),
                    "device": meta.get("device", ""),
                })
                rec.update({k: v for k, v in row.items() if k in MANIFEST_FIELDS and v != ""})
                rec["clip"] = _rel(session_dir / row["clip"], out.parent)
                if row.get("imu"):
                    rec["imu"] = _rel(session_dir / row["imu"], out.parent)
                rows.append(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return read_manifest(out)


def _rel(path: Path, base: Path) -> str:
    import os
    return os.path.relpath(path.resolve(), base.resolve())


def read_manifest(path: str | Path, classes: Iterable[str] | None = None) -> list[Clip]:
    path = Path(path).resolve()
    keep = set(classes) if classes is not None else None
    clips = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            row = {k: (row.get(k) or "").strip() for k in MANIFEST_FIELDS}
            row["label"] = row["label"].upper()
            row["probe"] = row["probe"].lower() or "on"
            if keep is not None and row["label"] not in keep:
                continue
            clips.append(Clip(root=path.parent, **row))
    ids = [c.clip for c in clips]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate clip paths in manifest")
    return clips


def summarize(clips: list[Clip]) -> str:
    by_user = defaultdict(lambda: defaultdict(int))
    for c in clips:
        by_user[c.participant][c.label] += 1
    labels = sorted({c.label for c in clips})
    lines = ["participant | " + " | ".join(labels) + " | total", "---|" + "---|" * (len(labels) + 1)]
    for user in sorted(by_user):
        counts = [by_user[user][lab] for lab in labels]
        lines.append(f"{user} | " + " | ".join(map(str, counts)) + f" | {sum(counts)}")
    lines.append(f"\n{len(clips)} clips, {len(by_user)} participants, "
                 f"{len({c.group for c in clips})} sessions, probe-off clips: {sum(c.probe == 'off' for c in clips)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Held-out test split (made once, before any tuning)
# ----------------------------------------------------------------------------

def make_split(clips: list[Clip], *, holdout_participants: list[str] | None = None,
               n_holdout_participants: int = 0, known_fraction: float = 0.15,
               holdout_conditions: list[str] | None = None, seed: int = 0) -> dict:
    rng = random.Random(seed)
    users = sorted({c.participant for c in clips})
    test_users = list(holdout_participants or [])
    if not test_users and n_holdout_participants:
        test_users = sorted(rng.sample(users, n_holdout_participants))
    missing = set(test_users) - set(users)
    if missing:
        raise ValueError(f"Unknown participants: {sorted(missing)}")
    conditions = set(holdout_conditions or [])
    test_condition = sorted(c.clip for c in clips if c.participant not in test_users
                            and c.condition in conditions and c.probe == "on")
    pool = defaultdict(list)
    for c in clips:
        if c.participant in test_users or c.probe != "on" or c.clip in test_condition:
            continue
        pool[(c.participant, c.label)].append(c.clip)
    test_known = []
    for key in sorted(pool):
        items = sorted(pool[key])
        rng.shuffle(items)
        test_known += items[: int(round(known_fraction * len(items)))]
    return {"seed": seed, "test_users": test_users, "test_known": sorted(test_known),
            "test_condition": test_condition, "holdout_conditions": sorted(conditions)}


def apply_split(clips: list[Clip], split: dict | None) -> tuple[list[Clip], dict[str, list[Clip]]]:
    """Return ``(dev, tests)``. Probe-off clips never enter ``dev``."""
    split = split or {}
    users = set(split.get("test_users", []))
    known, cond = set(split.get("test_known", [])), set(split.get("test_condition", []))
    dev, tests = [], defaultdict(list)
    for c in clips:
        if c.probe != "on":
            tests["probe_off"].append(c)
        elif c.participant in users:
            tests["unseen_users"].append(c)
        elif c.clip in cond:
            tests["new_condition"].append(c)
        elif c.clip in known:
            tests["known_users"].append(c)
        else:
            dev.append(c)
    return dev, dict(tests)


def load_split(path: str | Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


# ----------------------------------------------------------------------------
# Cross-validation folds on the development set
# ----------------------------------------------------------------------------

def make_folds(clips: list[Clip], protocol: str, n_folds: int = 5, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Protocols: ``trial`` (known users, held-out trials), ``session`` (held-out
    recording session per participant), ``user`` (held-out participants)."""
    rng = random.Random(seed)
    n = len(clips)
    fold_of = [0] * n
    if protocol == "trial":
        by_key = defaultdict(list)
        for i, c in enumerate(clips):
            by_key[(c.participant, c.label)].append(i)
        offset = 0
        for key in sorted(by_key):
            idx = by_key[key]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                fold_of[i] = (offset + j) % n_folds
            offset += len(idx)
    elif protocol == "session":
        sessions = defaultdict(set)
        for c in clips:
            sessions[c.participant].add(c.session)
        if max(len(s) for s in sessions.values()) < 2:
            raise ValueError("session protocol needs >= 2 sessions for some participant")
        rank = {}
        for user, s in sessions.items():
            ordered = sorted(s)
            start = rng.randrange(len(ordered))
            for j, sess in enumerate(ordered):
                rank[(user, sess)] = (start + j) % len(ordered)
        n_folds = max(len(s) for s in sessions.values())
        fold_of = [rank[(c.participant, c.session)] for c in clips]
    elif protocol == "user":
        users = sorted({c.participant for c in clips})
        if len(users) < 2:
            raise ValueError("user protocol needs >= 2 participants")
        rng.shuffle(users)
        n_folds = min(n_folds, len(users)) if n_folds > 0 else len(users)
        user_fold = {u: j % n_folds for j, u in enumerate(users)}
        fold_of = [user_fold[c.participant] for c in clips]
    else:
        raise ValueError("protocol must be trial | session | user")
    folds = []
    for f in range(n_folds):
        val = [i for i in range(n) if fold_of[i] == f]
        train = [i for i in range(n) if fold_of[i] != f]
        if val and train:
            folds.append((train, val))
    return folds


# ----------------------------------------------------------------------------
# Baseband cache
# ----------------------------------------------------------------------------

def _cache_key(path: Path, cfg: Config) -> str:
    st = path.stat()
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(json.dumps([cfg.probe.tones_hz, cfg.features.baseband_rate, cfg.features.lowpass_hz]).encode())
    return h.hexdigest()


def load_basebands(clips: list[Clip], cfg: Config, cache_dir: str | Path = "cache",
                   log=print) -> dict[str, np.ndarray]:
    cache = Path(cache_dir) / "baseband"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    todo = 0
    for i, c in enumerate(clips):
        f = cache / f"{_cache_key(c.path, cfg)}.npy"
        if f.exists():
            out[c.clip] = np.load(f)
            continue
        audio, sr = read_wav(c.path)
        bb = demodulate(audio, sr, cfg.probe, cfg.features)
        np.save(f, bb)
        out[c.clip] = bb
        todo += 1
        if todo % 200 == 0:
            log(f"  demodulated {todo} new clips ({i + 1}/{len(clips)})")
    return out


def load_imu(clips: list[Clip], cfg: Config) -> dict[str, np.ndarray]:
    """Per-clip IMU resampled to the feature frame rate: ``[6, frames]`` float32."""
    from .imu import load_clip_imu
    if not cfg.features.use_imu:
        return {}
    clip_s = cfg.data.clip_pre_s + cfg.data.clip_post_s
    out = {}
    for c in clips:
        if c.imu_path is None:
            raise ValueError(f"features.use_imu is on but clip {c.clip} has no IMU file")
        out[c.clip] = load_clip_imu(c.imu_path, cfg.features.frame_rate, clip_s)
    return out


# ----------------------------------------------------------------------------
# Torch dataset
# ----------------------------------------------------------------------------

class ClipDataset(Dataset):
    """Yields cropped (and, for training, time-stretched) baseband windows."""

    def __init__(self, clips: list[Clip], basebands: dict[str, np.ndarray], cfg: Config, *,
                 train: bool, shift_s: float = 0.0, imu: dict[str, np.ndarray] | None = None,
                 seed: int = 0):
        self.clips, self.bb, self.cfg, self.train = clips, basebands, cfg, train
        self.shift_s, self.imu = shift_s, imu or {}
        self.index = {name: i for i, name in enumerate(cfg.classes)}
        self.rng = np.random.default_rng(seed)
        r = cfg.features.baseband_rate
        self.n_win = int(round(cfg.data.window_s * r))
        self.margin = int(round(cfg.data.edge_margin_s * r))

    def __len__(self) -> int:
        return len(self.clips)

    def _positions(self, length: int) -> tuple[np.ndarray, float]:
        r = self.cfg.features.baseband_rate
        n, m = self.n_win, self.margin
        if self.train:
            a = self.cfg.augment.time_stretch if self.cfg.augment.enabled else 0.0
            stretch = float(self.rng.uniform(1 - a, 1 + a)) if a > 0 else 1.0
            span = (n - 1) * stretch
            hi = length - 1 - m - span
            if hi < m:  # not enough room: shrink the stretch
                stretch = (length - 1 - 2 * m) / (n - 1)
                span, hi = (n - 1) * stretch, m
            start = float(self.rng.uniform(m, hi))
        else:
            stretch = 1.0
            start = (self.cfg.data.eval_offset_s + self.shift_s) * r
            start = float(np.clip(start, 0, length - n))
        return start + np.arange(n) * stretch, stretch

    def __getitem__(self, i: int):
        clip = self.clips[i]
        bb = self.bb[clip.clip]
        pos, _ = self._positions(bb.shape[1])
        if np.allclose(pos, np.round(pos)):
            a = int(round(pos[0]))
            win = bb[:, a: a + self.n_win]
        else:
            grid = np.arange(bb.shape[1])
            win = np.stack([np.interp(pos, grid, row.real) + 1j * np.interp(pos, grid, row.imag) for row in bb])
        win = win.astype(np.complex64)
        imu = None
        if self.cfg.features.use_imu:
            data = self.imu[clip.clip]
            h = hop(self.cfg.features)
            t = frames_per_window(self.cfg.features, self.cfg.data.window_s)
            fpos = pos[::h][:t] / h
            grid = np.arange(data.shape[1])
            imu = np.stack([np.interp(fpos, grid, row) for row in data]).astype(np.float32)
            imu[:3] -= imu[:3].mean(axis=1, keepdims=True)  # remove gravity / orientation offset
        return win, imu, self.index.get(clip.label, -1)


def collate(batch):
    bb = torch.from_numpy(np.stack([b[0] for b in batch]))
    imu = None if batch[0][1] is None else torch.from_numpy(np.stack([b[1] for b in batch]))
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return bb, imu, y


def clip_table(clips: list[Clip]) -> list[dict]:
    return [{k: v for k, v in asdict(c).items() if k != "root"} for c in clips]
