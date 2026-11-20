"""Collection flows against a fake loopback sound card (no hardware needed):
main-laptop collection, the portable collector, and merging its recordings."""

import csv
import importlib.util
import json
import random
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from echospeak.config import load_config
from echospeak.data import read_manifest

REPO = Path(__file__).resolve().parent.parent


class _FakeStream:
    """Calls the callback faster than real time; input = previous output block (loopback)."""

    def __init__(self, samplerate, blocksize, callback, channels, **_):
        self.bs, self.cb, self.in_ch = blocksize, callback, channels[0]
        self._stop = threading.Event()

    def _run(self):
        prev = np.zeros((self.bs, self.in_ch), np.float32)
        while not self._stop.is_set():
            out = np.zeros((self.bs, 1), np.float32)
            self.cb(prev, out, self.bs, None, None)
            prev = np.repeat(out, self.in_ch, axis=1) * 0.5
            time.sleep(self.bs / 48000 / 8)

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()

    def close(self):
        pass


def _fake_sd():
    sd = types.ModuleType("sounddevice")
    sd.Stream = lambda device=None, **kw: _FakeStream(**kw)
    return sd


def _fast_sleep(s):
    threading.Event().wait(s / 8)


def test_reps_plan_and_takes_are_spread():
    from echospeak.collect import build_takes, parse_class_reps, reps_plan
    labels = ["NEXT", "BACK", "CHEWING", "YAWNING"]
    plan = reps_plan(labels, 10, parse_class_reps(["back=7", "YAWNING=5"]))
    assert plan == {"NEXT": 10, "BACK": 7, "CHEWING": 10, "YAWNING": 5}
    takes = build_takes(plan, 5, 3.0, 3.0, random.Random(0))
    assert sorted((t.label, t.n) for t in takes) == sorted(
        [("NEXT", 5), ("NEXT", 5), ("BACK", 5), ("BACK", 2), ("CHEWING", 5), ("CHEWING", 5), ("YAWNING", 5)])
    # Spread: every class's first take comes before any class's second take; never twice in a row.
    first_round = {t.label for t in takes[:4]}
    assert first_round == set(labels)
    assert all(a.label != b.label for a, b in zip(takes, takes[1:]))
    kinds = {t.label: (t.continuous, t.interval_s) for t in takes}
    assert kinds["CHEWING"] == (True, 3.0) and kinds["YAWNING"] == (False, 6.0) and kinds["NEXT"] == (False, 3.0)
    with pytest.raises(ValueError):
        reps_plan(labels, 2, {"JUMP": 1})


def test_collect_session_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    monkeypatch.setattr("echospeak.collect.time.sleep", _fast_sleep)
    from echospeak.collect import collect, next_participant, next_session

    cfg = load_config(None)
    data = tmp_path / "data"
    assert next_participant(data / "raw") == "P01"
    out = collect(cfg, data, reps=4, group_size=2, probe_off_reps=1, labels=["NEXT", "REST"],
                  confirm=False, log=lambda *_: None)
    assert out == data / "raw" / "P01" / "S1"
    assert next_session(data / "raw", "P01") == "S2"
    assert next_participant(data / "raw") == "P02"

    with (out / "cues.csv").open() as fh:
        cues = list(csv.DictReader(fh))
    assert sum(c["status"] == "practice" for c in cues) == 3        # NEXT x3 (CHEWING is not in labels)
    clips = read_manifest(data / "manifest.csv")
    on = [c for c in clips if c.probe == "on"]
    assert sorted(c.label for c in on) == ["NEXT"] * 4 + ["REST"] * 4
    assert len({c.take_key for c in on}) == 4                       # 2 takes per class, 2 clips each
    assert sum(c.probe == "off" for c in clips) == 2
    assert all(c.path.exists() and c.device for c in clips)
    # Continuous clips tile each REST stretch without gaps or overlap (one clip length apart).
    rest = [c for c in cues if c["label"] == "REST" and c["status"] == "ok" and c["probe"] == "on"]
    for take in {c["take"] for c in rest}:
        starts = sorted(int(c["go_sample"]) for c in rest if c["take"] == take)
        assert np.diff(starts).tolist() == [int(3.0 * 48000)] * (len(starts) - 1)
    assert all(c["beep"] == "0" for c in rest)
    with (data / "sessions.csv").open() as fh:
        log = list(csv.DictReader(fh))
    assert log[0]["participant"] == "P01" and log[0]["clips"] == "8" and log[0]["probe_off_clips"] == "2"


def test_portable_collect_pack_and_merge(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("portable", REPO / "data_collection_portable" / "collect.py")
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    monkeypatch.setattr(portable, "sounddevice", _fake_sd)
    monkeypatch.setattr(portable.time, "sleep", _fast_sleep)
    monkeypatch.setattr(portable, "HERE", tmp_path)
    monkeypatch.setattr(portable, "PROFILE", tmp_path / "laptop_profile.json")
    monkeypatch.setattr("builtins.input", lambda *a: "c")   # accept defaults / continue past checks
    (tmp_path / "laptop_profile.json").write_text(json.dumps(
        {"collector": "JX", "device": portable.laptop_model(), "input_device": None, "output_device": None,
         "distance_cm": 12}))

    rec = tmp_path / "recordings"
    portable.main(["--participant", "P01", "--reps", "1", "--class-reps", "NEXT=2", "TALKING=0",
                   "--group-size", "2", "--probe-off-reps", "0", "--no-practice", "--out", str(rec)])
    session = rec / "JX-P01" / "S1"
    with (session / "clips.csv").open() as fh:
        labels = sorted(r["label"] for r in csv.DictReader(fh))
    assert labels == sorted(["NEXT", "NEXT"] + [c for c in portable.CLASSES if c not in ("NEXT", "TALKING")])
    meta = json.loads((session / "session.json").read_text())
    assert meta["collector"] == "JX" and meta["tones_hz"] == load_config(REPO / "configs" / "default.yaml").probe.tones_hz

    zip_path = portable.pack(rec)
    from echospeak.cli import main
    data = tmp_path / "data"
    main(["merge", str(zip_path), "--data", str(data), "--config", str(REPO / "configs" / "default.yaml")])
    clips = read_manifest(data / "manifest.csv")
    assert len(clips) == len(labels) and {c.participant for c in clips} == {"JX-P01"}
    assert all(c.path.exists() for c in clips)
    # Merging again is a no-op.
    main(["merge", str(zip_path), "--data", str(data), "--config", str(REPO / "configs" / "default.yaml")])
    assert len(read_manifest(data / "manifest.csv")) == len(labels)
