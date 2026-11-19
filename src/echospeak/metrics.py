"""Classification and trigger metrics.

A *trigger* happens when the top class is a command and its probability is at
least the threshold. Non-command classes (REST, CHEWING, ...) never trigger.
"""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    present = cm.sum(axis=1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def trigger_metrics(probs: np.ndarray, y: np.ndarray, n_commands: int, threshold: float,
                    rejected: np.ndarray | None = None) -> dict:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    fires = (pred < n_commands) & (conf >= threshold)
    if rejected is not None:
        fires &= ~rejected
    is_cmd = y < n_commands
    out = {"threshold": float(threshold)}
    if is_cmd.any():
        out["command_correct_trigger"] = float((fires & (pred == y))[is_cmd].mean())
        out["command_wrong_trigger"] = float((fires & (pred != y))[is_cmd].mean())
        out["command_missed"] = float((~fires)[is_cmd].mean())
    if (~is_cmd).any():
        out["false_trigger_rate"] = float(fires[~is_cmd].mean())
    if rejected is not None:
        out["imu_rejected_fraction"] = float(rejected.mean())
    return out


def recommend_threshold(probs: np.ndarray, y: np.ndarray, n_commands: int, target_ftr: float) -> float:
    """Smallest threshold whose false-trigger rate on non-command trials is <= target."""
    is_cmd = y < n_commands
    if not (~is_cmd).any():
        return 0.5
    for thr in np.round(np.arange(0.30, 0.991, 0.01), 2):
        if trigger_metrics(probs, y, n_commands, thr)["false_trigger_rate"] <= target_ftr:
            return float(thr)
    return 0.99


def summarize(probs: np.ndarray, y: np.ndarray, classes: list[str], n_commands: int,
              threshold: float, rejected: np.ndarray | None = None) -> dict:
    labelled = y >= 0
    probs, y = probs[labelled], y[labelled]
    rejected = rejected[labelled] if rejected is not None else None
    pred = probs.argmax(axis=1)
    cm = confusion(y, pred, len(classes))
    # Binary command-vs-non-command accuracy (what matters for false triggers).
    cmd_true, cmd_pred = y < n_commands, pred < n_commands
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "macro_f1": macro_f1(cm),
        "command_vs_noncommand_accuracy": float((cmd_true == cmd_pred).mean()) if len(y) else float("nan"),
        "per_class_recall": {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes) if cm[i].sum()},
        "confusion": cm.tolist(),
        "trigger": trigger_metrics(probs, y, n_commands, threshold, rejected),
    }
    if (y < n_commands).any():
        m = y < n_commands
        out["command_only_accuracy"] = float((pred[m] == y[m]).mean())
    return out


def format_confusion(cm: list[list[int]], classes: list[str]) -> str:
    short = [c[:6] for c in classes]
    lines = ["true \\ pred | " + " | ".join(short), "---|" + "---|" * len(classes)]
    for c, row in zip(short, cm):
        lines.append(f"**{c}** | " + " | ".join(str(v) for v in row))
    return "\n".join(lines)
