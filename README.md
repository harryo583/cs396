# EchoSpeak

Silent command recognition from acoustic echoes, using only a laptop's built-in
speaker and microphone. COMP_SCI 396, Group 5.

The speaker plays an inaudible multi-tone probe at 18–20 kHz while the microphone
records the echoes. Lip and jaw motion modulates those echoes. A CNN with a
temporal model reads the motion and recognises six silently mouthed commands
(NEXT, BACK, SELECT, CANCEL, START, STOP). It rejects four everyday mouth
movements (REST, CHEWING, YAWNING, TALKING), so those never trigger an action.

**See [plan.md](plan.md)** for the full project plan. It covers the recording
protocol, training, tuning, evaluation and the live demo.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
pytest -q                     # ~15 s, no hardware needed

echospeak check               # once: verify the laptop captures the probe and sees motion
python collect.py             # record one participant session (everything is automatic)
```

All data goes to `data/` (git-ignored): one folder per participant and session,
plus `manifest.csv` and `sessions.csv`, which are updated after every session.

| Module | Purpose |
|---|---|
| `collect.py` (top level) | one-command data collection |
| `probe.py`, `segment.py` | probe synthesis, I/Q demodulation, clip cutting |
| `features.py` | Doppler spectrogram and I/Q-trajectory features, augmentation |
| `models.py` | EchoNet: two-branch CNN + none/GRU/Transformer/TCN temporal model |
| `train.py`, `evaluate.py`, `tune.py` | training/ensembles, CV and test reports, Optuna tuning |
| `check.py`, `live.py`, `synth.py` | hardware check, live demo, synthetic data |
