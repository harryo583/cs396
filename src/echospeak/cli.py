"""Command-line interface: ``echospeak <command> --help`` for details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cfg(args):
    from .config import load_config
    return load_config(args.config, args.set)


def _configs(args):
    from .config import load_config
    paths = args.config if isinstance(args.config, list) else [args.config]
    return [load_config(p, args.set) for p in (paths or [None])]


def _dev_clips(args, cfg):
    from .data import apply_split, load_split, read_manifest
    clips = read_manifest(args.manifest, cfg.classes)
    dev, tests = apply_split(clips, load_split(args.splits))
    if args.splits:
        print(f"development set: {len(dev)} clips (held out: " +
              ", ".join(f"{k}={len(v)}" for k, v in tests.items()) + ")")
    else:
        print(f"WARNING: no --splits given; using all {len(dev)} probe-on clips as development data")
    return dev, tests


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def cmd_devices(args):
    from .audio import list_devices
    print(list_devices())


def cmd_check(args):
    from .check import analyze, record_check
    cfg = _cfg(args)
    out = Path(args.out)
    if args.wav:
        analyze(cfg, Path(args.wav), out, phases=None)
    else:
        wav = record_check(cfg, out / "check.wav", args.input_device, args.output_device, args.channel)
        analyze(cfg, wav, out)


def cmd_collect(args):
    from .collect import collect
    collect(_cfg(args), args.data, args.participant, args.session, reps=args.reps,
            probe_off_reps=args.probe_off_reps, practice=not args.no_practice, distance_cm=args.distance_cm,
            condition=args.condition, notes=args.notes, input_device=args.input_device,
            output_device=args.output_device, channel=args.channel, labels=args.labels, confirm=not args.yes)


def cmd_segment(args):
    from .segment import segment_session
    rows = segment_session(Path(args.session_dir), _cfg(args))
    print(f"wrote {len(rows)} clips; run 'echospeak index' to refresh the manifest")


def cmd_synth(args):
    from .synth import generate
    generate(_cfg(args), args.out, participants=args.participants, sessions=args.sessions, reps=args.reps,
             probe_off_reps=args.probe_off_reps, seed=args.seed)


def cmd_index(args):
    from .data import index_dataset, summarize
    clips = index_dataset(args.root, args.out)
    print(summarize(clips))
    print(f"manifest: {args.out}")


def cmd_split(args):
    from .data import make_split, read_manifest
    cfg = _cfg(args)
    clips = read_manifest(args.manifest, cfg.classes)
    split = make_split(clips, holdout_participants=args.holdout_participants,
                       n_holdout_participants=args.n_holdout_participants, known_fraction=args.known_fraction,
                       holdout_conditions=args.holdout_conditions, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2))
    print(f"test participants: {split['test_users']}; known-user test clips: {len(split['test_known'])}; "
          f"new-condition test clips: {len(split['test_condition'])}\nwrote {args.out}")


def cmd_cv(args):
    from .evaluate import run_cv
    configs = _configs(args)
    dev, _ = _dev_clips(args, configs[0])
    run_cv(configs, dev, args.protocol, args.out, n_folds=args.folds, seeds=args.seeds, cache_dir=args.cache)


def cmd_tune(args):
    from .tune import run_tune
    cfg = _cfg(args)
    dev, _ = _dev_clips(args, cfg)
    run_tune(cfg, dev, args.out, protocol=args.protocol, n_folds=args.folds, n_trials=args.trials,
             metric=args.metric, cache_dir=args.cache, fixed_temporal=args.temporal, top=args.top)


def cmd_train(args):
    from .config import save_config
    from .data import load_basebands
    from .train import fit_ensemble
    configs = _configs(args)
    dev, tests = _dev_clips(args, configs[0])
    if args.include_test:
        dev += [c for name, group in tests.items() if name != "probe_off" for c in group]
        print(f"--include-test: training on {len(dev)} clips (use only for the final demo model)")
    bb = load_basebands(dev, configs[0], args.cache)
    bundle = fit_ensemble(configs, args.seeds, dev, bb)
    if args.threshold is not None:
        bundle.gate.threshold = args.threshold
    bundle.meta = {"n_train": len(dev), "participants": sorted({c.participant for c in dev}),
                   "include_test": bool(args.include_test)}
    out = Path(args.out)
    bundle.save(out / "model.pt")
    for i, c in enumerate(configs):
        save_config(c, out / f"config{i + 1}.yaml")
    print(f"saved {out / 'model.pt'} ({len(bundle.members)} members, threshold {bundle.gate.threshold:.2f})")


def cmd_test(args):
    from .data import apply_split, load_split, read_manifest
    from .evaluate import run_test
    from .train import Bundle
    bundle = Bundle.load(args.model)
    clips = read_manifest(args.manifest, bundle.classes)
    _, tests = apply_split(clips, load_split(args.splits))
    if not tests:
        raise SystemExit("no held-out clips found (did you pass --splits?)")
    run_test(bundle, tests, args.out, cache_dir=args.cache, threshold=args.threshold)
    print((Path(args.out) / "report.md").read_text())


def cmd_predict(args):
    from .audio import read_wav
    from .live import Gate
    from .probe import demodulate
    from .train import Bundle
    bundle = Bundle.load(args.model)
    cfg = bundle.cfg
    thr = args.threshold if args.threshold is not None else bundle.gate.threshold
    for path in args.wav:
        audio, sr = read_wav(path)
        bb = demodulate(audio, sr, cfg.probe, cfg.features)
        rate = cfg.features.baseband_rate
        n_win = int(cfg.data.window_s * rate)
        clip_len = (cfg.data.clip_pre_s + cfg.data.clip_post_s) * rate
        if abs(bb.shape[1] - clip_len) < 0.05 * rate and not args.stream:
            starts = [int((cfg.data.eval_offset_s + s) * rate) for s in cfg.train.tta_shifts_s]
            probs = bundle.predict_windows(np.stack([bb[:, s: s + n_win] for s in starts])).mean(0)
            k = int(probs.argmax())
            fires = k < bundle.n_commands and probs[k] >= thr
            top = np.argsort(probs)[::-1][:3]
            print(f"{path}: {bundle.classes[k]} p={probs[k]:.2f} -> {'TRIGGER' if fires else 'no action'}  "
                  f"(top3: {', '.join(f'{bundle.classes[i]} {probs[i]:.2f}' for i in top)})")
            continue
        # Continuous recording: slide the window, apply the live gate, list events.
        margin = int(cfg.data.edge_margin_s * rate)
        hop = int(bundle.gate.hop_s * rate)
        starts = list(range(margin, bb.shape[1] - n_win - margin, hop))
        gate = Gate(bundle.n_commands, thr, bundle.gate.min_consecutive, bundle.gate.cooldown_s)
        events = []
        for i in range(0, len(starts), 64):
            batch = starts[i: i + 64]
            probs = bundle.predict_windows(np.stack([bb[:, s: s + n_win] for s in batch]))
            for s, p in zip(batch, probs):
                t_end = (s + n_win) / rate
                k = gate.update(p, t_end)
                if k is not None:
                    events.append((t_end, bundle.classes[k], float(p[k])))
        minutes = len(audio) / sr / 60
        print(f"{path}: {len(events)} command events in {minutes:.1f} min ({len(events) / max(minutes, 1e-9):.2f}/min)")
        for t, label, p in events:
            print(f"  {t:8.2f}s  {label:7s} p={p:.2f}")


def cmd_live(args):
    from .live import run_live
    from .train import Bundle
    run_live(Bundle.load(args.model), threshold=args.threshold, input_device=args.input_device,
             output_device=args.output_device, channel=args.channel, action=args.action, gui=not args.no_gui,
             seconds=args.seconds)


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="echospeak", description="EchoSpeak: silent command recognition from acoustic echoes")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, config=True, multi=False):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        if config:
            if multi:
                sp.add_argument("--config", nargs="+", default=["configs/default.yaml"],
                                help="one or more YAML configs (several = ensemble)")
            else:
                sp.add_argument("--config", default="configs/default.yaml")
            sp.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                            help="override config values, e.g. model.temporal=gru train.epochs=80")
        return sp

    def audio_args(sp):
        sp.add_argument("--input-device", help="sounddevice input device (index or name); see 'echospeak devices'")
        sp.add_argument("--output-device", help="sounddevice output device")
        sp.add_argument("--channel", type=int, default=0, help="input channel to use")

    def data_args(sp):
        sp.add_argument("--manifest", default="data/manifest.csv")
        sp.add_argument("--splits", default=None, help="splits.json from 'echospeak split'")
        sp.add_argument("--cache", default="cache", help="directory for cached demodulated clips")

    add("devices", cmd_devices, "list audio devices", config=False)

    sp = add("check", cmd_check, "milestone 1: verify the probe is captured and motion is visible")
    sp.add_argument("--wav", help="analyse an existing recording (e.g. a clip) instead of recording")
    sp.add_argument("--out", default="reports/check")
    audio_args(sp)

    sp = add("collect", cmd_collect, "record one participant session (sound check, prompts, clips, manifest)")
    sp.add_argument("--participant", help="participant ID (default: ask, suggesting the next free P##)")
    sp.add_argument("--session", help="session ID (default: next free S#)")
    sp.add_argument("--data", default="data", help="dataset folder")
    sp.add_argument("--reps", type=int, default=5, help="blocks; every class appears once per block")
    sp.add_argument("--probe-off-reps", type=int, default=1, help="probe-off control blocks at the end (0 = none)")
    sp.add_argument("--no-practice", action="store_true", help="skip the 3-trial practice")
    sp.add_argument("--distance-cm", type=float, default=10, help="mouth-to-microphone distance")
    sp.add_argument("--condition", default="normal", help="free tag, e.g. normal, far, noisy, moving")
    sp.add_argument("--notes", default="")
    sp.add_argument("--labels", nargs="+", help="subset of classes to prompt")
    sp.add_argument("--yes", action="store_true", help="no operator prompts (accept suggested IDs, ignore sound-check warnings)")
    audio_args(sp)

    sp = add("segment", cmd_segment, "re-cut the clips of a recorded session (e.g. after changing clip timing)")
    sp.add_argument("session_dir")

    sp = add("synth", cmd_synth, "generate a synthetic dataset to test the pipeline")
    sp.add_argument("--out", default="data/synth")
    sp.add_argument("--participants", type=int, default=6)
    sp.add_argument("--sessions", type=int, default=2)
    sp.add_argument("--reps", type=int, default=4)
    sp.add_argument("--probe-off-reps", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)

    sp = add("index", cmd_index, "build manifest.csv from recorded sessions", config=False)
    sp.add_argument("--root", default="data/raw")
    sp.add_argument("--out", default="data/manifest.csv")

    sp = add("split", cmd_split, "reserve held-out test sets (do this once, before tuning)")
    sp.add_argument("--manifest", default="data/manifest.csv")
    sp.add_argument("--out", default="data/splits.json")
    sp.add_argument("--holdout-participants", nargs="*", help="explicit unseen-user test participants")
    sp.add_argument("--n-holdout-participants", type=int, default=2)
    sp.add_argument("--known-fraction", type=float, default=0.15, help="fraction of each remaining user's trials held out")
    sp.add_argument("--holdout-conditions", nargs="*", help="session conditions reserved for the new-condition test")
    sp.add_argument("--seed", type=int, default=0)

    sp = add("cv", cmd_cv, "cross-validate config(s) on the development set", multi=True)
    data_args(sp)
    sp.add_argument("--protocol", choices=["trial", "session", "user"], default="user")
    sp.add_argument("--folds", type=int, default=5, help="folds for trial/user protocols (0 = leave-one-user-out)")
    sp.add_argument("--seeds", type=int, default=1)
    sp.add_argument("--out", required=True)

    sp = add("tune", cmd_tune, "Optuna hyperparameter search (CV on the development set)")
    data_args(sp)
    sp.add_argument("--protocol", choices=["trial", "session", "user"], default="user")
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--trials", type=int, default=40)
    sp.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    sp.add_argument("--temporal", choices=["none", "gru", "transformer", "tcn"], help="fix the temporal model")
    sp.add_argument("--top", type=int, default=3)
    sp.add_argument("--out", required=True)

    sp = add("train", cmd_train, "train the final model / ensemble on the development set", multi=True)
    data_args(sp)
    sp.add_argument("--seeds", type=int, default=3, help="seeds per config (ensemble size = configs x seeds)")
    sp.add_argument("--threshold", type=float, help="trigger threshold to store (use the CV recommendation)")
    sp.add_argument("--include-test", action="store_true", help="also train on held-out clips (demo model only)")
    sp.add_argument("--out", required=True)

    sp = add("test", cmd_test, "evaluate a trained model on the held-out test sets", config=False)
    data_args(sp)
    sp.add_argument("--model", required=True)
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--out", required=True)

    sp = add("predict", cmd_predict, "classify clip WAVs, or list events in a long recording", config=False)
    sp.add_argument("--model", required=True)
    sp.add_argument("wav", nargs="+")
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--stream", action="store_true", help="treat every file as a continuous recording")

    sp = add("live", cmd_live, "live demo with the computer's speaker and microphone", config=False)
    sp.add_argument("--model", required=True)
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--action", choices=["none", "keys"], default="none",
                    help="keys: NEXT/BACK press right/left arrow (macOS; grant Accessibility permission)")
    sp.add_argument("--no-gui", action="store_true")
    sp.add_argument("--seconds", type=float, default=0, help="stop after N seconds (no-gui mode)")
    audio_args(sp)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
