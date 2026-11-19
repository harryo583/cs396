# EchoSpeak

Silent smartphone command recognition from acoustic echoes. COMP_SCI 396, Group 5.

The phone's speaker plays an inaudible multi-tone probe at 18–20 kHz while its
microphone records the echoes. Lip and jaw motion modulates those echoes. A
CNN with a temporal model reads the motion and recognises six silently mouthed
commands (NEXT, BACK, SELECT, CANCEL, START, STOP). It rejects four everyday
mouth movements (REST, CHEWING, YAWNING, TALKING), so those never trigger an
action.

**See [plan.md](plan.md)** for the full project plan. It covers the recording
protocol, data collection, training, tuning, evaluation and the live demo.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
pytest -q                                   # ~15 s, no hardware needed
echospeak synth --out data/synth            # fake dataset to try the pipeline
echospeak index --root data/synth --out data/synth_manifest.csv
echospeak cv --manifest data/synth_manifest.csv --protocol user --out runs/cv-synth
```

| Module | Purpose |
|---|---|
| `probe.py` | probe synthesis, cue sounds, coherent I/Q demodulation |
| `session.py`, `segment.py`, `imu.py` | guided recording, sync and clip cutting, optional IMU |
| `features.py` | Doppler spectrogram and I/Q-trajectory features, augmentation |
| `models.py` | EchoNet: two-branch CNN + none/GRU/Transformer/TCN temporal model |
| `train.py`, `evaluate.py`, `tune.py` | training/ensembles, CV and test reports, Optuna tuning |
| `check.py`, `live.py`, `synth.py` | hardware check, live demo, synthetic data |
