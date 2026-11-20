"""Live demo: play the probe, classify a sliding window every ``hop_s``, show the
motion trace / prediction / confidence, and optionally advance slides.

Gate: a command fires only if the same command is the top class with probability
>= threshold for ``min_consecutive`` consecutive windows, outside the cooldown.
Non-command classes (REST/CHEWING/YAWNING/TALKING) never fire.
"""

from __future__ import annotations

import collections
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from .audio import parse_device, require_sounddevice
from .features import motion_trace
from .probe import demodulate, probe_block
from .train import Bundle

MAC_KEYS = {"NEXT": 124, "GO": 124, "BACK": 123, "OKAY": 36, "STOP": 53}  # right, right, left, return, escape


def send_action(label: str, action: str) -> None:
    if action == "keys" and sys.platform == "darwin" and label in MAC_KEYS:
        subprocess.Popen(["osascript", "-e", f'tell application "System Events" to key code {MAC_KEYS[label]}'])


class Gate:
    def __init__(self, n_commands: int, threshold: float, min_consecutive: int, cooldown_s: float):
        self.n_commands, self.threshold = n_commands, threshold
        self.min_consecutive, self.cooldown_s = min_consecutive, cooldown_s
        self.run_label, self.run_len, self.last_fire = -1, 0, -1e9

    def update(self, probs: np.ndarray, now: float) -> int | None:
        k = int(probs.argmax())
        ok = k < self.n_commands and probs[k] >= self.threshold
        if ok and k == self.run_label:
            self.run_len += 1
        else:
            self.run_label, self.run_len = (k, 1) if ok else (-1, 0)
        if ok and self.run_len >= self.min_consecutive and now - self.last_fire >= self.cooldown_s:
            self.last_fire = now
            self.run_label, self.run_len = -1, 0
            return k
        return None


def run_live(bundle: Bundle, *, threshold: float | None = None, input_device=None, output_device=None,
             channel: int = 0, action: str = "none", gui: bool = True, seconds: float = 0.0) -> None:
    sd = require_sounddevice()
    cfg, gate_cfg = bundle.cfg, bundle.gate
    sr = cfg.probe.sample_rate
    margin = cfg.data.edge_margin_s
    buf_n = int((cfg.data.window_s + 2 * margin) * sr)
    ring = np.zeros(buf_n, np.float32)
    filled = [0]
    pos = [0]
    lock = threading.Lock()

    def callback(indata, outdata, frames, _t, _status):
        outdata[:, 0] = probe_block(cfg.probe, pos[0], frames)
        pos[0] += frames
        x = indata[:, channel]
        with lock:
            ring[:-frames] = ring[frames:]
            ring[-frames:] = x
            filled[0] = min(buf_n, filled[0] + frames)

    gate = Gate(bundle.n_commands, threshold if threshold is not None else gate_cfg.threshold,
                gate_cfg.min_consecutive, gate_cfg.cooldown_s)
    rate = cfg.features.baseband_rate
    a, n_win = int(margin * rate), int(cfg.data.window_s * rate)
    trace_hist = collections.deque(maxlen=int(6 * cfg.features.frame_rate))
    state = {"probs": np.zeros(len(bundle.classes)), "fired": "", "fired_at": 0.0, "ms": 0.0}

    def step():
        with lock:
            if filled[0] < buf_n:
                return
            audio = ring.copy()
        t0 = time.perf_counter()
        bb = demodulate(audio, sr, cfg.probe, cfg.features)[:, a: a + n_win]
        probs = bundle.predict_windows(bb)[0]
        state["ms"] = 1000 * (time.perf_counter() - t0)
        tr = motion_trace(torch.from_numpy(bb), cfg.features).numpy()
        trace_hist.extend(tr[-int(gate_cfg.hop_s * cfg.features.frame_rate):])
        state["probs"] = probs
        fired = gate.update(probs, time.time())
        if fired is not None:
            label = bundle.classes[fired]
            state["fired"], state["fired_at"] = label, time.time()
            print(f"\n>>> {label}  (p={probs[fired]:.2f}, inference {state['ms']:.0f} ms)")
            send_action(label, action)

    stream = sd.Stream(samplerate=sr, blocksize=512, channels=(channel + 1, 1), dtype="float32",
                       callback=callback, device=(parse_device(input_device), parse_device(output_device)))
    print(f"Live: threshold {gate.threshold:.2f}, classes {bundle.classes}. Ctrl+C to stop.")
    with stream:
        start = time.time()
        if gui:
            _gui_loop(bundle, state, trace_hist, step, gate_cfg.hop_s, gate.threshold)
        else:
            try:
                while not seconds or time.time() - start < seconds:
                    t0 = time.time()
                    step()
                    p = state["probs"]
                    k = int(p.argmax())
                    sys.stdout.write(f"\r{bundle.classes[k]:8s} {p[k]:.2f}  [{state['ms']:.0f} ms]   ")
                    sys.stdout.flush()
                    time.sleep(max(0.0, gate_cfg.hop_s - (time.time() - t0)))
            except KeyboardInterrupt:
                pass


def _gui_loop(bundle, state, trace_hist, step, hop_s, threshold):  # pragma: no cover - interactive
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1.3]})
    fig.canvas.manager.set_window_title("EchoSpeak live")
    (line,) = ax_t.plot([], [], lw=1.2)
    ax_t.set_title("acoustic motion trace (last 6 s)")
    ax_t.set_yscale("log")
    bars = ax_p.bar(bundle.classes, np.zeros(len(bundle.classes)),
                    color=["tab:blue"] * bundle.n_commands + ["tab:gray"] * (len(bundle.classes) - bundle.n_commands))
    ax_p.axhline(threshold, color="tab:red", ls="--", lw=1)
    ax_p.set_ylim(0, 1)
    ax_p.tick_params(axis="x", rotation=30)
    banner = fig.text(0.5, 0.94, "", ha="center", fontsize=22, weight="bold", color="tab:green")

    def update(_):
        step()
        y = np.asarray(trace_hist)
        if len(y):
            line.set_data(np.arange(len(y)) / 100.0, np.maximum(y, 1e-9))
            ax_t.set_xlim(0, max(6, len(y) / 100.0))
            ax_t.set_ylim(max(y.min(), 1e-7) * 0.5, y.max() * 2 + 1e-9)
        for b, p in zip(bars, state["probs"]):
            b.set_height(p)
        k = int(np.argmax(state["probs"]))
        recent = time.time() - state["fired_at"] < 1.5
        banner.set_text(f"{state['fired']} ✓" if recent else f"{bundle.classes[k]}  {state['probs'][k]:.2f}")
        banner.set_color("tab:green" if recent else "0.4")
        return [line, *bars, banner]

    _anim = FuncAnimation(fig, update, interval=int(hop_s * 1000), blit=False, cache_frame_data=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
