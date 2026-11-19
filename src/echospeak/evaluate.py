"""Cross-validation on the development set and final evaluation on held-out sets."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config, save_config
from .data import Clip, load_basebands, make_folds
from .metrics import format_confusion, recommend_threshold, summarize
from .train import Bundle, fit_ensemble


def labels_of(clips: list[Clip], classes: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index.get(c.label, -1) for c in clips])


def _write_predictions(path: Path, clips: list[Clip], probs: np.ndarray, classes: list[str],
                       extra: dict[str, list] | None = None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        extra = extra or {}
        w.writerow(["clip", "participant", "session", "label", "pred", "conf", *extra, *[f"p_{c}" for c in classes]])
        for i, c in enumerate(clips):
            p = probs[i]
            w.writerow([c.clip, c.participant, c.session, c.label, classes[int(p.argmax())], f"{p.max():.4f}",
                        *[extra[k][i] for k in extra], *[f"{v:.4f}" for v in p]])


def _mean_std(values: list[float]) -> str:
    v = np.array(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.3f}"


def run_cv(configs: list[Config], clips: list[Clip], protocol: str, out_dir: str | Path, *,
           n_folds: int = 5, seeds: int = 1, cache_dir: str = "cache", log=print,
           fold_callback=None) -> dict:
    """K-fold CV. ``fold_callback(fold_index, fold_score)`` may raise to stop early (Optuna pruning)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = configs[0]
    classes, n_cmd = cfg.classes, len(cfg.data.commands)
    clips = [c for c in clips if c.label in set(classes)]
    folds = make_folds(clips, protocol, n_folds, seed=cfg.train.seed)
    log(f"CV protocol={protocol}: {len(folds)} folds over {len(clips)} clips")
    bb = load_basebands(clips, cfg, cache_dir, log=log)
    y = labels_of(clips, classes)
    oof = np.full((len(clips), len(classes)), np.nan)
    fold_id = np.full(len(clips), -1)
    fold_metrics = []
    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        log(f"\n=== fold {f + 1}/{len(folds)}: train {len(tr)}, validate {len(va)} "
            f"({', '.join(sorted({clips[i].participant if protocol == 'user' else clips[i].group for i in va}))[:120]})")
        bundle = fit_ensemble(configs, seeds, [clips[i] for i in tr], bb, log=log)
        probs = bundle.predict_clips([clips[i] for i in va], bb)
        oof[va] = probs
        fold_id[va] = f
        m = summarize(probs, y[va], classes, n_cmd, cfg.gate.threshold)
        fold_metrics.append(m)
        log(f"fold {f + 1}: accuracy {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
            f"[{time.time() - t0:.0f}s]")
        if fold_callback is not None:
            fold_callback(f, m)
    done = fold_id >= 0
    threshold = recommend_threshold(oof[done], y[done], n_cmd, cfg.gate.target_false_trigger)
    pooled = summarize(oof[done], y[done], classes, n_cmd, threshold)
    summary = {
        "protocol": protocol, "folds": len(folds), "seeds": seeds, "n_clips": len(clips),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "fold_metrics": [{k: m[k] for k in ("n", "accuracy", "macro_f1")} for m in fold_metrics],
        "recommended_threshold": threshold,
        "pooled": pooled,
        "configs": [c.to_dict() for c in configs],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_predictions(out / "oof_predictions.csv", [clips[i] for i in np.where(done)[0]], oof[done], classes,
                       {"fold": list(fold_id[done])})
    for i, c in enumerate(configs):
        save_config(c, out / f"config{i + 1}.yaml")
    report = [f"# Cross-validation ({protocol})", "",
              f"- clips: {len(clips)}, folds: {len(folds)}, ensemble: {len(configs)} config(s) × {seeds} seed(s)",
              f"- **accuracy (mean ± sd over folds): {_mean_std([m['accuracy'] for m in fold_metrics])}**",
              f"- macro-F1: {_mean_std([m['macro_f1'] for m in fold_metrics])}",
              f"- command vs non-command accuracy (pooled): {pooled['command_vs_noncommand_accuracy']:.3f}",
              f"- recommended threshold for ≤{cfg.gate.target_false_trigger:.0%} false triggers: **{threshold:.2f}**",
              "", _trigger_lines(pooled["trigger"]), "", "## Pooled confusion matrix", "",
              format_confusion(pooled["confusion"], classes), "",
              "## Per-class recall", "", *[f"- {k}: {v:.3f}" for k, v in pooled["per_class_recall"].items()]]
    (out / "report.md").write_text("\n".join(report) + "\n")
    log("\n" + "\n".join(report[:7]))
    return summary


def _trigger_lines(t: dict) -> str:
    keys = [("command_correct_trigger", "commands correctly triggered"),
            ("command_wrong_trigger", "commands triggering the WRONG action"),
            ("command_missed", "commands rejected (no action)"),
            ("false_trigger_rate", "non-command trials that triggered (false-trigger rate)")]
    rows = [f"- {label}: {t[k]:.3f}" for k, label in keys if k in t]
    return f"At threshold {t['threshold']:.2f}:\n" + "\n".join(rows)


def run_test(bundle: Bundle, groups: dict[str, list[Clip]], out_dir: str | Path, *,
             cache_dir: str = "cache", threshold: float | None = None, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = bundle.cfg
    classes, n_cmd = bundle.classes, bundle.n_commands
    thr = bundle.gate.threshold if threshold is None else threshold
    results = {"threshold": thr}
    report = ["# Test results", "", f"Ensemble of {len(bundle.members)} member(s); threshold {thr:.2f}.", ""]
    for name, clips in groups.items():
        clips = [c for c in clips if c.label in set(classes)]
        if not clips:
            continue
        t0 = time.time()
        bb = load_basebands(clips, cfg, cache_dir, log=log)
        probs = bundle.predict_clips(clips, bb)
        ms = 1000 * (time.time() - t0) / len(clips)
        y = labels_of(clips, classes)
        m = summarize(probs, y, classes, n_cmd, thr)
        m["breakdown"] = _breakdown(clips, probs, y)
        m["ms_per_clip_including_io"] = ms
        results[name] = m
        _write_predictions(out / f"predictions_{name}.csv", clips, probs, classes)
        report += [f"## {name} ({len(clips)} clips)", ""]
        if name == "probe_off":
            report += ["Control: the probe was off, so the model should NOT recognise commands here.",
                       f"- accuracy (should be near chance, {1 / len(classes):.2f}): {m['accuracy']:.3f}",
                       f"- trigger rate: {1 - m['trigger'].get('command_missed', 1.0):.3f} of command trials, "
                       f"{m['trigger'].get('false_trigger_rate', float('nan')):.3f} of non-command trials", ""]
        else:
            report += [f"- **accuracy: {m['accuracy']:.3f}**, macro-F1: {m['macro_f1']:.3f}, "
                       f"command-vs-non-command: {m['command_vs_noncommand_accuracy']:.3f}",
                       _trigger_lines(m["trigger"])]
            report += ["", format_confusion(m["confusion"], classes), ""]
            for key, rows in m["breakdown"].items():
                if len(rows) > 1:
                    report.append(f"Accuracy by {key}: " + ", ".join(f"{k}={v['accuracy']:.3f} (n={v['n']})"
                                                                     for k, v in rows.items()))
            report.append("")
        log(f"{name}: n={len(clips)} accuracy={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f}")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "report.md").write_text("\n".join(report) + "\n")
    return results


def _breakdown(clips: list[Clip], probs: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for key in ("participant", "distance_cm", "orientation", "condition", "session"):
        groups = defaultdict(list)
        for i, c in enumerate(clips):
            groups[getattr(c, key) or "-"].append(i)
        out[key] = {k: {"n": len(v), "accuracy": float((probs[v].argmax(1) == y[v]).mean())}
                    for k, v in sorted(groups.items())}
    return out
