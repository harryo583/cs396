"""Models, data splits, metrics and an end-to-end run on a tiny synthetic dataset."""

import json

import numpy as np
import pytest
import torch

from echospeak.cli import main
from echospeak.config import Config, load_config
from echospeak.data import Clip, apply_split, make_folds, make_split, read_manifest
from echospeak.metrics import recommend_threshold, trigger_metrics
from echospeak.models import EchoNet


@pytest.mark.parametrize("temporal", ["none", "gru", "transformer", "tcn"])
@pytest.mark.parametrize("branches", ["both", "doppler", "iq"])
def test_models_forward_backward(temporal, branches):
    cfg = Config()
    cfg.model.temporal, cfg.model.branches = temporal, branches
    model = EchoNet(cfg)
    d = torch.randn(3, 8, 21, 200)
    q = torch.randn(3, 40, 200)
    out = model(d, q)
    assert out.shape == (3, len(cfg.classes))
    out.sum().backward()


def _clips():
    clips = []
    for p in ["P1", "P2", "P3", "P4"]:
        for s in ["S1", "S2"]:
            for lab in ["NEXT", "REST"]:
                for t in range(3):
                    clips.append(Clip(f"{p}/{s}/{lab}{t}.wav", p, s, str(t), lab))
    clips.append(Clip("P1/off/x.wav", "P1", "off", "0", "NEXT", probe="off"))
    return clips


@pytest.mark.parametrize("protocol", ["trial", "session", "user"])
def test_folds_do_not_leak(protocol):
    clips = [c for c in _clips() if c.probe == "on"]
    folds = make_folds(clips, protocol, n_folds=3)
    seen = set()
    for train, val in folds:
        assert not set(train) & set(val)
        seen |= set(val)
        if protocol == "user":
            assert not {clips[i].participant for i in train} & {clips[i].participant for i in val}
        if protocol == "session":
            assert not {clips[i].group for i in train} & {clips[i].group for i in val}
    assert seen == set(range(len(clips)))


def test_takes_never_straddle_folds_or_splits():
    clips = []
    for p in ["P1", "P2", "P3"]:
        for s in ["S1", "S2"]:
            for lab in ["NEXT", "REST"]:
                for t in range(2):
                    for r in range(5):
                        clips.append(Clip(f"{p}/{s}/{lab}{t}_{r}.wav", p, s, str(r), lab, take=f"t{lab}{t}"))
    for train, val in make_folds(clips, "trial", n_folds=4):
        assert not {clips[i].take_key for i in train} & {clips[i].take_key for i in val}
    split = make_split(clips, known_fraction=0.15)
    held = set(split["test_known"])
    for c in clips:
        same_take = [d.clip in held for d in clips if d.take_key == c.take_key]
        assert all(same_take) or not any(same_take)
    assert len(held) == 3 * 2 * 5          # 1 of 4 takes per (participant, label)


def test_split_is_disjoint_and_excludes_probe_off():
    clips = _clips()
    split = make_split(clips, holdout_participants=["P4"], known_fraction=0.34)
    dev, tests = apply_split(clips, split)
    assert {c.participant for c in tests["unseen_users"]} == {"P4"}
    assert all(c.probe == "on" for c in dev)
    assert len(tests["probe_off"]) == 1
    names = [c.clip for c in dev] + [c.clip for g in tests.values() for c in g]
    assert len(names) == len(set(names)) == len(clips)


def test_trigger_metrics():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    y = np.array([0, 1, 2, 2])  # classes 0,1 commands; 2 non-command
    m = trigger_metrics(probs, y, n_commands=2, threshold=0.7)
    assert m["command_correct_trigger"] == 0.5
    assert m["false_trigger_rate"] == 0.5
    assert recommend_threshold(probs, y, 2, 0.0) == 0.81


def test_config_overrides_and_validation(tmp_path):
    cfg = load_config(None, ["model.temporal=gru", "train.epochs=3"])
    assert cfg.model.temporal == "gru" and cfg.train.epochs == 3
    with pytest.raises(ValueError):
        load_config(None, ["model.temporal=lstm"])
    with pytest.raises(ValueError):
        load_config(None, ["model.nonsense=1"])


FAST = """
data: {commands: [NEXT, STOP], non_commands: [REST, CHEWING]}
model: {channels: 16, d_model: 32}
train: {epochs: 2, device: cpu}
"""


def test_end_to_end_synthetic(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(FAST)
    cfg = ["--config", "c.yaml"]
    main(["synth", "--out", "data/raw", "--participants", "3", "--sessions", "2", "--reps", "1", *cfg])
    main(["index", "--root", "data/raw", "--out", "data/manifest.csv"])
    clips = read_manifest("data/manifest.csv")
    assert len(clips) == 3 * (2 + 1) * 4
    main(["split", "--holdout-participants", "P03", *cfg])
    split = json.loads(open("data/splits.json").read())
    assert split["test_users"] == ["P03"]

    main(["cv", *cfg, "--splits", "data/splits.json", "--protocol", "user", "--out", "runs/cv"])
    summary = json.loads(open("runs/cv/summary.json").read())
    assert summary["folds"] == 2 and 0 <= summary["accuracy_mean"] <= 1

    # Two configs x 1 seed = 2-member ensemble with different temporal models.
    (tmp_path / "gru.yaml").write_text(FAST.replace("d_model: 32", "d_model: 32, temporal: gru"))
    main(["train", "--config", "c.yaml", "gru.yaml", "--seeds", "1", "--splits", "data/splits.json",
          "--out", "runs/final"])
    main(["test", "--model", "runs/final/model.pt", "--splits", "data/splits.json", "--out", "runs/final/test"])
    results = json.loads(open("runs/final/test/results.json").read())
    assert {"unseen_users", "probe_off"} <= set(results)

    main(["predict", "--model", "runs/final/model.pt", str(clips[0].path)])
    # A long recording made of concatenated clips goes through the streaming path.
    from echospeak.audio import read_wav, write_wav
    audio = np.concatenate([read_wav(c.path)[0] for c in clips[:6]])
    write_wav("long.wav", audio, 48000)
    main(["predict", "--model", "runs/final/model.pt", "long.wav"])
    assert "command events" in capsys.readouterr().out
