"""Manifest, baseband cache, held-out splits, CV folds and the training dataset.

Dataset layout produced by ``collect.py`` (and ``echospeak synth``)::

    data/
        manifest.csv     every clip (rebuilt automatically after each session)
        sessions.csv     one row per recorded session (log)
        raw/<participant>/<session>/
            session.json     participant, session, distance, condition, probe check, ...
            cues.csv         one row per prompted trial (label, GO sample, probe, status)
            session.wav      full continuous recording
            clips/<trial>.wav  one clip per trial, GO cue at ``clip_pre_s``
            clips.csv        clip index (written by segmentation)

``echospeak index`` gathers every ``clips.csv`` into ``manifest.csv``.
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
from .probe import demodulate

MANIFEST_FIELDS = [
    "clip", "participant", "session", "trial", "label", "probe", "distance_cm",
    "orientation", "condition", "device", "take",
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
    take: str = ""       # recording take (group of back-to-back repetitions) within the session
    root: Path = field(default=Path("."), repr=False, compare=False)

    @property
    def path(self) -> Path:
        return self.root / self.clip

    @property
    def group(self) -> str:
        return f"{self.participant}/{self.session}"

    @property
    def take_key(self) -> str:
        """Clips of one take are near-duplicates and must stay on the same side of any split."""
        return f"{self.participant}/{self.session}/{self.take}" if self.take else self.clip


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
    # Whole takes are held out, so near-duplicate repetitions never straddle train/test.
    pool = defaultdict(lambda: defaultdict(list))
    for c in clips:
        if c.participant in test_users or c.probe != "on" or c.clip in test_condition:
            continue
        pool[(c.participant, c.label)][c.take_key].append(c.clip)
    test_known = []
    for key in sorted(pool):
        takes = sorted(pool[key])
        rng.shuffle(takes)
        n = int(round(known_fraction * len(takes)))
        if n == 0 and known_fraction > 0 and len(takes) >= 3:
            n = 1  # e.g. 2 sessions x 2 takes per word -> hold out 1 take (25%)
        for t in takes[:n]:
            test_known += pool[key][t]
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
    recording session per participant), ``user`` (held-out participants),
    ``device`` (held-out laptop models: train on some laptops, test on another)."""
    rng = random.Random(seed)
    n = len(clips)
    fold_of = [0] * n
    if protocol == "trial":
        # Stratified by (participant, label); whole takes are assigned to a fold.
        by_key = defaultdict(lambda: defaultdict(list))
        for i, c in enumerate(clips):
            by_key[(c.participant, c.label)][c.take_key].append(i)
        offset = 0
        for key in sorted(by_key):
            takes = sorted(by_key[key])
            rng.shuffle(takes)
            for j, t in enumerate(takes):
                for i in by_key[key][t]:
                    fold_of[i] = (offset + j) % n_folds
            offset += len(takes)
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
    elif protocol in ("user", "device"):
        key = (lambda c: c.participant) if protocol == "user" else (lambda c: c.device or "unknown")
        groups = sorted({key(c) for c in clips})
        if len(groups) < 2:
            raise ValueError(f"{protocol} protocol needs >= 2 different {protocol}s")
        rng.shuffle(groups)
        n_folds = min(n_folds, len(groups)) if (n_folds > 0 and protocol == "user") else len(groups)
        group_fold = {g: j % n_folds for j, g in enumerate(groups)}
        fold_of = [group_fold[key(c)] for c in clips]
    else:
        raise ValueError("protocol must be trial | session | user | device")
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


# ----------------------------------------------------------------------------
# Torch dataset
# ----------------------------------------------------------------------------

class ClipDataset(Dataset):
    """Yields cropped (and, for training, time-stretched) baseband windows."""

    def __init__(self, clips: list[Clip], basebands: dict[str, np.ndarray], cfg: Config, *,
                 train: bool, shift_s: float = 0.0, seed: int = 0):
        self.clips, self.bb, self.cfg, self.train = clips, basebands, cfg, train
        self.shift_s = shift_s
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
        return win.astype(np.complex64), self.index.get(clip.label, -1)


def collate(batch):
    bb = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return bb, y


def clip_table(clips: list[Clip]) -> list[dict]:
    return [{k: v for k, v in asdict(c).items() if k != "root"} for c in clips]
