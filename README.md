# EchoSpeak

Silent command recognition from acoustic echoes, using only a laptop's built-in
speaker and microphone. COMP_SCI 396, Group 5.

The speaker plays an inaudible multi-tone probe at 18–20 kHz while the microphone
records the echoes. Lip and jaw motion modulates those echoes. A CNN with a
temporal model reads the motion and recognises silently mouthed commands
(BACK, MENU, BANANA, GO, OKAY, STOP, NEXT, APPLE). It rejects four everyday mouth
movements (REST, CHEWING, YAWNING, TALKING), so those never trigger an action.

**See [plan.md](plan.md)** for the full project plan. It covers the recording
protocol, training, tuning, evaluation and the live demo.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
pytest -q                     # ~15 s, no hardware needed

echospeak check                          # once: verify the laptop captures the probe and sees motion
python data_collection/collect.py        # record one participant on the main laptop (everything is automatic)
python data_collection/collect.py --reps 6 --class-reps NEXT=15   # choose clips per word/action (default 10)
```

| Folder | Use it for |
|---|---|
| [`data_collection/`](data_collection/README.md) | the main laptop; writes straight into `data/` (git-ignored) and updates `manifest.csv` and `sessions.csv` |
| [`data_collection_portable/`](data_collection_portable/README.md) | any other laptop; standalone (numpy, scipy, sounddevice only). One participant per run; send the zip and `echospeak merge` it |

| Module | Purpose |
|---|---|
| `collect.py` | collection sessions and participant UI, merging portable recordings |
| `probe.py`, `segment.py` | probe synthesis, I/Q demodulation, clip cutting |
| `features.py` | Doppler spectrogram and I/Q-trajectory features, augmentation |
| `models.py` | EchoNet: two-branch CNN + none/GRU/Transformer/TCN temporal model |
| `train.py`, `evaluate.py`, `tune.py` | training/ensembles, CV and test reports, Optuna tuning |
| `check.py`, `live.py`, `synth.py` | hardware check, live demo, synthetic data |
