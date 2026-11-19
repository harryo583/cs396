"""Full collection flow against a fake loopback sound card (no hardware needed)."""

import csv
import sys
import threading
import time
import types

import numpy as np

from echospeak.config import load_config
from echospeak.data import read_manifest


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


def test_collect_session_end_to_end(tmp_path, monkeypatch):
    sd = types.ModuleType("sounddevice")
    sd.Stream = lambda device=None, **kw: _FakeStream(**kw)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr("echospeak.collect.time.sleep", lambda s: threading.Event().wait(s / 8))
    from echospeak.collect import collect, next_participant, next_session

    cfg = load_config(None, ["data.clip_pre_s=0.5", "data.clip_post_s=1.0", "data.window_s=1.2",
                             "data.eval_offset_s=0.1"])
    data = tmp_path / "data"
    assert next_participant(data / "raw") == "P01"
    out = collect(cfg, data, reps=2, probe_off_reps=1, labels=["NEXT", "REST"], confirm=False,
                  prep_s=0.3, gap_s=0.1, break_s=1, log=lambda *_: None)
    assert out == data / "raw" / "P01" / "S1"
    assert next_session(data / "raw", "P01") == "S2"
    assert next_participant(data / "raw") == "P02"

    with (out / "cues.csv").open() as fh:
        cues = list(csv.DictReader(fh))
    assert sum(c["status"] == "practice" for c in cues) == 3       # practice is logged, not cut
    clips = read_manifest(data / "manifest.csv")
    assert len(clips) == 2 * 2 + 1 * 2
    assert sum(c.probe == "off" for c in clips) == 2
    assert all(c.path.exists() for c in clips)
    with (data / "sessions.csv").open() as fh:
        log = list(csv.DictReader(fh))
    assert log[0]["participant"] == "P01" and log[0]["clips"] == "4" and log[0]["probe_off_clips"] == "2"
