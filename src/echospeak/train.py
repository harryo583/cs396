"""Training (AdamW + warmup/cosine + label smoothing + EMA), prediction with
test-time crop averaging, and ensemble bundles."""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, GateConfig
from .data import Clip, ClipDataset, collate
from .features import compute_features
from .models import EchoNet, count_parameters

BUNDLE_VERSION = 2


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon: ~10x faster than CPU for this model
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    """Exponential moving average of weights *and* BatchNorm buffers, with the usual
    warm-up ``decay_t = min(decay, (1 + t) / (10 + t))`` so short runs are not dominated
    by the random initialisation."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            target = ema_state[name]
            if value.dtype.is_floating_point:
                target.mul_(d).add_(value.detach(), alpha=1 - d)
            else:
                target.copy_(value)


def _param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("pos") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def fit_member(cfg: Config, clips: list[Clip], basebands: dict, *,
               log=print, val: tuple[list[Clip], dict] | None = None) -> EchoNet:
    """Train one model and return its EMA weights (on CPU, eval mode)."""
    tc = cfg.train
    set_seed(tc.seed)
    device = select_device(tc.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ds = ClipDataset(clips, basebands, cfg, train=True, seed=tc.seed)
    if any(ds.index.get(c.label, -1) < 0 for c in clips):
        raise ValueError("training clips contain labels outside data.commands + data.non_commands")
    loader = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate,
                        drop_last=len(ds) > tc.batch_size, num_workers=tc.num_workers)
    model = EchoNet(cfg).to(device)
    ema = EMA(model, tc.ema_decay)
    opt = torch.optim.AdamW(_param_groups(model, tc.weight_decay), lr=tc.lr)
    total = max(1, tc.epochs * len(loader))
    warm = max(1, int(tc.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    gen = torch.Generator().manual_seed(tc.seed)
    log(f"  model: {cfg.model.temporal}/{cfg.model.branches}, {count_parameters(model) / 1e3:.0f}k params, "
        f"{len(clips)} clips, device={device}")
    t0 = time.time()
    for epoch in range(tc.epochs):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for bb, y in loader:
            feats = compute_features(bb, cfg.features, aug=cfg.augment, gen=gen)
            d, q, y = feats["doppler"].to(device), feats["iq"].to(device), y.to(device)
            if cfg.augment.enabled and cfg.augment.mixup_alpha > 0:
                lam = float(np.random.beta(cfg.augment.mixup_alpha, cfg.augment.mixup_alpha))
                perm = torch.randperm(len(y), device=device)
                d, q = lam * d + (1 - lam) * d[perm], lam * q + (1 - lam) * q[perm]
                logits = model(d, q)
                loss = lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[perm])
            else:
                logits = model(d, q)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            seen += len(y)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        last = epoch == tc.epochs - 1
        if val is not None and ((epoch + 1) % 10 == 0 or last):
            vprobs = predict_member(ema.module, cfg, *val)
            vy = np.array([ds.index.get(c.label, -1) for c in val[0]])
            vacc = float((vprobs.argmax(1) == vy).mean())
            log(f"  epoch {epoch + 1:3d}  loss {loss_sum / seen:.3f}  train acc {correct / seen:.3f}  "
                f"val acc (EMA) {vacc:.3f}  [{time.time() - t0:.0f}s]")
        elif (epoch + 1) % 10 == 0 or last:
            log(f"  epoch {epoch + 1:3d}  loss {loss_sum / seen:.3f}  train acc {correct / seen:.3f}  "
                f"[{time.time() - t0:.0f}s]")
    final = copy.deepcopy(ema.module).cpu().eval()
    return final


@torch.no_grad()
def predict_member(model: EchoNet, cfg: Config, clips: list[Clip], basebands: dict,
                   tta: bool = True) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    shifts = cfg.train.tta_shifts_s if tta else [0.0]
    total = np.zeros((len(clips), len(cfg.classes)))
    for shift in shifts:
        ds = ClipDataset(clips, basebands, cfg, train=False, shift_s=shift)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        out = []
        for bb, _ in loader:
            feats = compute_features(bb, cfg.features)
            out.append(torch.softmax(model(feats["doppler"].to(device), feats["iq"].to(device)), dim=1).cpu())
        total += torch.cat(out).numpy() if out else 0
    return total / len(shifts)


# ----------------------------------------------------------------------------
# Ensembles
# ----------------------------------------------------------------------------

class Bundle:
    """An ensemble of trained members that share probe/data settings."""

    def __init__(self, members: list[tuple[Config, EchoNet]], gate: GateConfig | None = None, meta: dict | None = None):
        if not members:
            raise ValueError("empty ensemble")
        ref = members[0][0]
        for cfg, _ in members[1:]:
            if cfg.probe != ref.probe or cfg.data != ref.data:
                raise ValueError("ensemble members must share probe and data config")
        self.members = members
        self.gate = gate or copy.deepcopy(ref.gate)
        self.meta = meta or {}

    @property
    def cfg(self) -> Config:
        return self.members[0][0]

    @property
    def classes(self) -> list[str]:
        return self.cfg.classes

    @property
    def n_commands(self) -> int:
        return len(self.cfg.data.commands)

    def predict_clips(self, clips: list[Clip], basebands: dict, tta: bool = True) -> np.ndarray:
        probs = [predict_member(m, c, clips, basebands, tta) for c, m in self.members]
        return np.mean(probs, axis=0)

    @torch.no_grad()
    def predict_windows(self, bb: np.ndarray | torch.Tensor) -> np.ndarray:
        """``bb``: complex ``[B,K,N]`` baseband windows of length ``window_s``."""
        bb = torch.as_tensor(bb)
        if bb.dim() == 2:
            bb = bb[None]
        out = []
        for cfg, model in self.members:
            feats = compute_features(bb, cfg.features)
            out.append(torch.softmax(model(feats["doppler"], feats["iq"]), dim=1).numpy())
        return np.mean(out, axis=0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": BUNDLE_VERSION,
            "members": [{"config": c.to_dict(), "state": m.state_dict()} for c, m in self.members],
            "gate": dataclasses.asdict(self.gate),
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("version") != BUNDLE_VERSION:
            raise ValueError(f"{path} is not an EchoSpeak v{BUNDLE_VERSION} model bundle")
        members = []
        for item in blob["members"]:
            cfg = Config.from_dict(item["config"])
            model = EchoNet(cfg)
            model.load_state_dict(item["state"])
            members.append((cfg, model.eval()))
        return cls(members, GateConfig(**blob["gate"]), blob.get("meta", {}))


def member_configs(configs: list[Config], seeds: int) -> list[Config]:
    out = []
    for cfg in configs:
        for s in range(seeds):
            c = copy.deepcopy(cfg)
            c.train.seed = cfg.train.seed + 1000 * s
            out.append(c)
    return out


def fit_ensemble(configs: list[Config], seeds: int, clips: list[Clip], basebands: dict,
                 *, log=print, val=None) -> Bundle:
    members = []
    plan = member_configs(configs, seeds)
    for i, cfg in enumerate(plan):
        log(f"member {i + 1}/{len(plan)} (seed {cfg.train.seed})")
        members.append((cfg, fit_member(cfg, clips, basebands, log=log, val=val)))
    return Bundle(members)
