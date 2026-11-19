"""Recording session flow against a fake loopback sound card (no hardware needed)."""

import csv
import sys
import threading
import time
import types

import numpy as np

from echospeak.config import load_config


class _FakeStream:
    """Calls the callback in real time; duplex input = previous output block (loopback)."""

    def __init__(self, samplerate, blocksize, callback, channels, duplex, **_):
        self.sr, self.bs, self.cb, self.duplex = samplerate, blocksize, callback, duplex
        self.in_ch = channels[0] if duplex else 0
        self._stop = threading.Event()

    def _run(self):
        prev = np.zeros((self.bs, max(1, self.in_ch)), np.float32)
        while not self._stop.is_set():
            out = np.zeros((self.bs, 1), np.float32)
            if self.duplex:
                self.cb(prev, out, self.bs, None, None)
                prev = np.repeat(out, max(1, self.in_ch), axis=1) * 0.5
            else:
                self.cb(out, self.bs, None, None)
            time.sleep(self.bs / self.sr / 4)  # 4x faster than real time

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
    sd.Stream = lambda device=None, **kw: _FakeStream(duplex=True, **kw)
    sd.OutputStream = lambda device=None, **kw: _FakeStream(duplex=False, **kw)
    return sd


def test_duplex_session_records_and_segments(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr("echospeak.session.time.sleep", lambda s: threading.Event().wait(s / 4))
    from echospeak.session import run_session

    cfg = load_config(None, ["data.clip_pre_s=0.5", "data.clip_post_s=1.0", "data.window_s=1.2",
                             "data.eval_offset_s=0.1"])
    out = run_session(cfg, tmp_path, "P01", "S1", mode="duplex", reps=1, labels=["NEXT", "REST"],
                      prep_s=0.3, gap_s=0.1, log=lambda *_: None)
    with (out / "clips.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert sorted(r["label"] for r in rows) == ["NEXT", "REST"]
    assert all((out / r["clip"]).exists() for r in rows)
