"""Optuna hyperparameter search scored by cross-validation on the development set.

Only the development set is used; held-out test sets are never touched. Each
trial is a full K-fold CV; unpromising trials are pruned after each fold. The
study is stored in SQLite, so an interrupted search resumes with the same command.
Outputs: ``leaderboard.csv`` and ``best1.yaml`` .. ``bestN.yaml`` (ready for
``echospeak train --config``; several of them make a strong ensemble).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import Config, save_config
from .data import Clip
from .evaluate import run_cv

SEARCH_SPACE_HELP = """\
model.temporal      none | gru | transformer | tcn
model.branches      both | doppler | iq
model.channels      48 | 64 | 96
model.d_model       64 | 96 | 128
model.layers        1 .. 3
model.dropout       0.1 .. 0.5
train.lr            3e-4 .. 5e-3 (log)
train.weight_decay  1e-3 .. 1e-1 (log)
train.epochs        40 | 60 | 90
train.label_smoothing 0 .. 0.2
augment.time_stretch 0 .. 0.2
augment.gain_log_std 0 .. 0.5
augment.noise_std   0 .. 0.3
augment.time_masks  0 .. 3
augment.tone_dropout 0 .. 0.3
augment.mixup_alpha 0 | 0.2 | 0.4
"""


def suggest(trial, base: Config, fixed_temporal: str | None = None) -> Config:
    over = {
        "model": {
            "temporal": fixed_temporal or trial.suggest_categorical("temporal", ["none", "gru", "transformer", "tcn"]),
            "branches": trial.suggest_categorical("branches", ["both", "doppler", "iq"]),
            "channels": trial.suggest_categorical("channels", [48, 64, 96]),
            "d_model": trial.suggest_categorical("d_model", [64, 96, 128]),
            "layers": trial.suggest_int("layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        },
        "train": {
            "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "epochs": trial.suggest_categorical("epochs", [40, 60, 90]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        },
        "augment": {
            "time_stretch": trial.suggest_float("time_stretch", 0.0, 0.2),
            "gain_log_std": trial.suggest_float("gain_log_std", 0.0, 0.5),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.3),
            "time_masks": trial.suggest_int("time_masks", 0, 3),
            "tone_dropout": trial.suggest_float("tone_dropout", 0.0, 0.3),
            "mixup_alpha": trial.suggest_categorical("mixup_alpha", [0.0, 0.2, 0.4]),
        },
    }
    values = base.to_dict()
    for section, items in over.items():
        values[section].update(items)
    return Config.from_dict(values)


def run_tune(base: Config, clips: list[Clip], out_dir: str | Path, *, protocol: str = "user",
             n_folds: int = 5, n_trials: int = 40, metric: str = "accuracy", cache_dir: str = "cache",
             fixed_temporal: str | None = None, top: int = 3, log=print) -> list[dict]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("tuning needs optuna: pip install -e '.[tune]'") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out / 'study.db').resolve()}"
    study = optuna.create_study(study_name="echospeak", storage=storage, load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=base.train.seed, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    # Always evaluate the shipped defaults first so the search can only improve on them.
    if len(study.trials) == 0 and fixed_temporal is None:
        study.enqueue_trial({"temporal": base.model.temporal, "branches": base.model.branches,
                             "channels": base.model.channels, "d_model": base.model.d_model,
                             "layers": base.model.layers, "dropout": base.model.dropout, "lr": base.train.lr,
                             "weight_decay": base.train.weight_decay, "epochs": base.train.epochs,
                             "label_smoothing": base.train.label_smoothing,
                             "time_stretch": base.augment.time_stretch, "gain_log_std": base.augment.gain_log_std,
                             "noise_std": base.augment.noise_std, "time_masks": base.augment.time_masks,
                             "tone_dropout": base.augment.tone_dropout, "mixup_alpha": base.augment.mixup_alpha})

    def objective(trial):
        cfg = suggest(trial, base, fixed_temporal)
        scores = []

        def on_fold(i, m):
            scores.append(m[metric])
            trial.report(sum(scores) / len(scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        summary = run_cv([cfg], clips, protocol, out / f"trial{trial.number:03d}", n_folds=n_folds,
                         cache_dir=cache_dir, log=lambda *_: None, fold_callback=on_fold)
        save_config(cfg, out / f"trial{trial.number:03d}" / "config.yaml")
        log(f"trial {trial.number}: {metric}={summary[metric + '_mean']:.4f}  {trial.params}")
        return summary[f"{metric}_mean"]

    remaining = max(0, n_trials - len([t for t in study.trials if t.state.is_finished()]))
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    done = sorted([t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"],
                  key=lambda t: t.value, reverse=True)
    rows = [{"trial": t.number, metric: round(t.value, 4), **t.params} for t in done]
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out / "leaderboard.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for i, t in enumerate(done[:top]):
        cfg_path = out / f"trial{t.number:03d}" / "config.yaml"
        (out / f"best{i + 1}.yaml").write_text(cfg_path.read_text())
    log(f"\nbest {metric}: " + ", ".join(f"trial {r['trial']}={r[metric]}" for r in rows[:top]))
    log(f"configs written to {out}/best1.yaml .. best{min(top, len(rows))}.yaml")
    return rows
