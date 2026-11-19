# EchoSpeak: Silent Smartphone Command Recognition via Acoustic Echoes

**Team (Group 5):** Jerry Xiang, Harry Wang, Jacob Lyu, Yixuan Xiao, Chris Chen

This document is the project plan and the operating manual for this repository.
It keeps everything in the original proposal (`plan.pdf`, now removed) and adds
how the implementation works, how to record data, and the exact commands for
training, tuning, evaluating and demoing.

---

## 1. Goal

EchoSpeak turns an ordinary smartphone into a hands-free, camera-free interface
for silent commands. It detects changes in acoustic reflections caused by lip and
jaw movement. From these it recognises a small, predefined set of silently
mouthed commands, with no spoken audio, video or wearable sensors. Target
settings are libraries, meetings and noisy places where voice input is
inconvenient or unreliable.

The project is **command classification**, not open-ended transcription. Part
of the task is *rejecting* unrelated everyday mouth movements.

| | Classes |
|---|---|
| Commands (6) | `NEXT`, `BACK`, `SELECT`, `CANCEL`, `START`, `STOP` |
| Non-commands (4) | `REST`, `CHEWING`, `YAWNING`, `TALKING` |

The model outputs a class and a confidence. **Low-confidence predictions and
non-command predictions never trigger an action.** If reliability is limited,
start with four commands and expand to six (see §9).

**Primary metric:** test accuracy (10-class), reported for three settings:
1. known users, held-out trials;
2. new session or altered placement;
3. unseen participants.

**Also reported:** macro-F1, false-trigger rate, response latency, and (for
IMU-assisted rejection) the fraction of trials rejected.

---

## 2. Method

### 2.1 Sensing

The phone speaker plays a continuous high-frequency probe. The microphone
captures the direct signal plus reflected echoes as raw PCM audio.

* **Probe:** 8 continuous-wave tones, 18.00–19.75 kHz, 250 Hz apart, with
  Schroeder phases for a low crest factor. The 250 Hz spacing is far larger
  than the Doppler shift of lips and jaw (at most about ±60 Hz at 19 kHz), so the
  tones never interfere. Eight tones at different frequencies also make the
  system robust when one frequency fades because of multipath.
* **Coherent I/Q demodulation:** each tone is mixed down to complex baseband,
  low-pass filtered at 110 Hz and sampled at 500 Hz. Each tone's baseband
  `h_k(t)` is the sum of two parts:
  * the static paths (speaker-to-mic leakage, face, hand), which are constant;
  * the articulator echo, whose phase follows the round-trip path length
    (one wavelength ≈ 1.8 cm, so 1 mm of lip motion ≈ 0.7 rad).
* **Normalisation:** `rel_k(t) = h_k(t) / mean(h_k) − 1` is the echo
  *relative to the static path*. It cancels phone volume, the speaker and mic
  frequency response, and the arbitrary carrier phase. The same features
  therefore work across phones and sessions.
* **IMU (optional):** synchronized accelerometer and gyroscope data can flag
  motion-corrupted trials (gating) or be added as model input (fusion). The IMU
  measures phone movement and orientation. **It cannot measure the distance to
  the mouth.**

### 2.2 Features

Features are computed per 2.0 s window, in torch, identically for training,
evaluation and the live demo (`src/echospeak/features.py`).

| View | Shape | Content |
|---|---|---|
| `doppler` | 8 tones × 21 bins × 200 frames | log-power Doppler spectrogram of `rel_k` (±80 Hz, 7.8 Hz bins, 100 fps): radial velocity of lips and jaw over time |
| `iq` | 40 channels × 200 frames | per tone: Re/Im of `rel_k`, their derivatives, and log-amplitude change. This is the fine-grained displacement trajectory |

### 2.3 Model

The proposal planned a CNN, with a CNN-GRU as an extension. The implementation
generalises this into one architecture with a selectable temporal model, so all
variants can be compared under identical conditions.

```
doppler [8,21,200] ─ BN ─ 2D residual CNN (pools frequency) ─┐
                                                              ├─ concat ─ 1×1 conv ─ temporal model ─ attention+max pool ─ classifier (10)
iq      [40,200]   ─ BN ─ 1D residual CNN ───────────────────┘
```

`model.temporal` takes one of these values:

| Value | Model |
|---|---|
| `none` | CNN only: the proposal's primary baseline |
| `gru` | CNN-GRU: the proposal's extension (2-layer BiGRU) |
| `transformer` | CNN-Transformer (pre-norm, learned positions) |
| `tcn` | CNN with dilated temporal convolutions |

`model.branches` (`both` / `doppler` / `iq`) gives the feature ablation.

The default is a CNN-Transformer with both branches, at about 0.5 M parameters.
That size is deliberate: with roughly 1,000 labelled trials, a large network
would only memorise the data.

**What actually maximises test accuracy with little data** (all implemented):

1. *Physically motivated augmentation* (`augment:` section):
   * random crop and time-stretch: reaction time and speaking rate;
   * echo-gain jitter: phone distance;
   * random phase rotation: the unknown mouth distance modulo one wavelength;
   * tone dropout: multipath fades;
   * noise, time masks, and optional mixup.
2. *Regularisation:* AdamW with weight decay, label smoothing, dropout,
   warm-up plus a cosine schedule.
3. *EMA weights:* an exponential moving average of the parameters is what gets
   evaluated. It is more stable than the last epoch, so no early stopping on test
   data is needed.
4. *Test-time augmentation (TTA):* predictions are averaged over three crops
   shifted by −0.15, 0 and +0.15 s.
5. *Ensembles:* averaging several seeds and/or several tuned configurations
   (e.g. the top three from tuning, mixing GRU and Transformer) typically adds a
   few points of accuracy and gives better-calibrated confidences.
6. *Tuning by cross-validation* on development data only, with Optuna (§6).

### 2.4 Decision and rejection

A trial or window triggers an action only if:
* the top class is a **command**, and
* its probability is at least `gate.threshold`.

The threshold comes from cross-validation, as the smallest threshold whose
false-trigger rate on non-command trials is ≤ 2% (`gate.target_false_trigger`).
In the live demo, the same command must also win 2 consecutive 100 ms windows,
followed by a 1.5 s cooldown.

With IMU gating, a trial is rejected whenever the phone's gyroscope RMS exceeds
`gate.imu_max_gyro_rms` (0.3 rad/s).

---

## 3. Setup

Python ≥ 3.10. Training runs on Apple Silicon (MPS, used automatically), CUDA or
CPU.

```bash
cd cs396
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'      # torch, numpy, scipy, pyyaml, sounddevice, matplotlib, optuna, pytest
pytest -q                    # 28 tests, ~15 s, no hardware needed
echospeak --help
```

`echospeak <command> --help` documents every option. All experiment settings
live in `configs/default.yaml`. Any value can be overridden on the command line
with `--set section.key=value`, e.g. `--set model.temporal=gru train.epochs=80`.

### Try the whole pipeline on synthetic data first (≈15 min)

```bash
echospeak synth --out data/synth --participants 6 --sessions 2 --reps 4
echospeak index --root data/synth --out data/synth/manifest.csv
echospeak split --manifest data/synth/manifest.csv --out data/synth/splits.json --n-holdout-participants 1
echospeak cv    --manifest data/synth/manifest.csv --splits data/synth/splits.json --protocol user --out runs/synth-cv
echospeak train --manifest data/synth/manifest.csv --splits data/synth/splits.json --seeds 1 --out runs/synth-final
echospeak test  --manifest data/synth/manifest.csv --splits data/synth/splits.json --model runs/synth-final/model.pt --out runs/synth-final/test
```

The synthetic simulator only checks that the code works. **Its accuracy says
nothing about real sensing.**

---

## 4. Milestone 1: verify usable acoustic reflections

Do this before collecting any data, on the exact hardware you will use.

### 4.1 Choose the recording setup

| | **A. Laptop duplex** (`--mode duplex`) | **B. Phone** (`--mode prompt`) |
|---|---|---|
| Speaker / mic | the computer's (or a USB audio interface) | the phone's |
| What the computer does | plays the probe, records, prompts, cuts clips automatically | shows prompts and plays audible sync sounds; the phone records |
| Effort per session | lowest | copy the phone WAV over, then run `echospeak import` |
| Live demo | works directly (`echospeak live`) | needs the phone's audio to reach the computer (see §8) |
| Matches proposal | partially | fully (smartphone sensing) |

**Rule: train on data from the same hardware you will demo on.** Speaker and
mic placement change the echoes. The relative-echo normalisation helps, but
does not make devices interchangeable. A practical path:
1. Do the pilot on setup A, to debug the protocol quickly.
2. Collect the main dataset on setup B (the phone), as proposed.
3. Run the live demo with the phone playing the probe and its microphone
   streaming to the laptop (§8).

### 4.2 Run the check

**Setup A (laptop):**
```bash
echospeak devices                                   # find input/output device indices
echospeak check --out reports/check-laptop          # 12 s: still → hand waving → mouthing
```

**Setup B (phone):**
```bash
echospeak probe --out probe.wav --int16             # 10 min loopable probe; copy it to the phone
```
1. On the phone, play `probe.wav` on loop at about 70% volume. Use the
   built-in speaker; never Bluetooth.
2. At the same time, record with a recorder app set to **uncompressed WAV,
   48 kHz**. Follow the 12 s script: 4 s still, then 4 s waving a hand
   5–15 cm away, then 4 s silently mouthing words.
3. Copy the recording to the computer and analyse it:

```bash
echospeak check --wav phone_check.wav --phases --out reports/check-phone
```

### 4.3 Pass criteria

All of the following must hold:
* **Tone SNR ≥ 20 dB for all 8 tones.**
  * If tones are missing, the app or OS is low-pass filtering (voice
    processing, AAC/M4A compression), the sample rate is below 44.1 kHz, or the
    speaker cannot play them.
  * Fix by trying another recorder app, a "raw/unprocessed/measurement" mode,
    or WAV output. Lowering `probe.tones_hz` to e.g. 17–19 kHz is the last
    resort, since lower tones are more audible.
* **Peak level < 0.98:** no clipping. Lower the volume if it clips.
* **Motion energy while mouthing is ≥ 3–5× the "still" energy**, and the
  Doppler plot (`reports/check*/check.png`) shows clear streaks when the lips
  move.

Then repeat at **5, 10 and 15 cm**, in portrait and tilted orientations, to
pick the working range for data collection.

---

## 5. Data collection

### 5.1 Protocol

| Item | Plan |
|---|---|
| Participants | ≥ 10 (the proposal's target). More participants help unseen-user accuracy more than more repetitions do |
| Sessions | 2 per participant, on different days or after re-seating; phone re-positioned each time |
| Repetitions | `--reps 5` per class per session (proposal minimum: 3). 10 classes × 5 × 2 sessions × 10 people = **1,000 trials**, ~4 min per session |
| Placement | phone held ~10 cm from the face, mic toward the mouth, in a stable grip. Record `--distance-cm` and `--orientation` |
| Order | randomised within each block, so every block contains every class once (avoids drift confounds) |
| Robustness sessions | for a subset of people, extra sessions with `--condition far` (15 cm), `--condition tilted`, `--condition motion` (deliberate phone motion), and a quiet vs a noisy room |
| Probe-off control | one short session per participant with `--probe-off --reps 1` (laptop), or the phone simply not playing the probe |
| IMU (optional) | log the accelerometer and gyroscope with a sensor app during phone sessions; tap the phone 3× when prompted (used for sync) |

**Instructions shown to participants** (the prompter displays them):

| Class | Instruction |
|---|---|
| commands | silently mouth the word: lips and jaw move, but there is **no voice and no whisper** |
| `REST` | stay still, breathe normally, mouth closed |
| `CHEWING` | chew naturally, as if chewing gum |
| `YAWNING` | yawn, or imitate a yawn |
| `TALKING` | say a short random sentence out loud |

Each trial runs: "Get ready: LABEL" (1.2 s), then **GO** with a beep. The
participant performs the action after GO. The clip runs from GO −0.75 s to
GO +2.25 s.

**Prompter keys:**
* `x`: the participant made a mistake. The trial is discarded and repeated
  later.
* `p`: pause or resume.
* `q`: save and quit.

Participant IDs must be anonymous (`P01`, `P02`, …). Recordings stay on
team machines, and `data/` is git-ignored.

### 5.2 Recording commands

**Setup A, laptop duplex:** everything is automatic.

```bash
echospeak session --participant P01 --session S1 --mode duplex --reps 5 --distance-cm 10 \
                  [--input-device N --output-device M]
```

The session writes `data/raw/P01/S1/` containing:
* `session.wav`, the full recording;
* `cues.csv`;
* `session.json`;
* `clips/*.wav` and `clips.csv`, cut automatically.

**Setup B, phone:**

```bash
echospeak session --participant P01 --session S1 --mode prompt --reps 5 --distance-cm 10
```
1. Start `probe.wav` looping on the phone, then start the phone recorder (WAV,
   48 kHz). If you use IMU, also start the sensor logger. Press Enter.
2. When prompted, **tap the phone 3×**. The laptop then plays a sync chirp. Keep
   the laptop speaker audible to the phone: about 0.5–1 m away, volume ~50%.
3. Follow the prompts. Each GO plays a 1 kHz beep, which the phone also records.
4. Stop the phone recording, copy the WAV to the computer, and run:

```bash
echospeak import data/raw/P01/S1 --audio ~/Downloads/rec.wav \
      [--imu-accel Accelerometer.csv --imu-gyro Gyroscope.csv]
```

`import` works in four steps:
1. It finds the sync chirp in the phone recording.
2. It maps every logged cue into phone time.
3. It refines each cue on its recorded beep, correcting latency and clock drift
   with a robust linear fit.
4. It cuts the clips.

It prints the number of beeps found, the offset, the drift in ppm, and the
residual jitter in ms. **Expect the jitter to be < 5 ms.** If the IMU CSVs are
given, the tap transients align the IMU clock, and per-clip IMU files are
written. The IMU CSV needs a time column plus x/y/z columns, which most sensor
logger apps export.

**Probe-off control:**
```bash
echospeak session --participant P01 --session S-off --mode duplex --probe-off --reps 1
```
(Phone: run `--mode prompt --probe-off` and simply do not play `probe.wav`.)

### 5.3 After every session (quality check, 1 min)

* Read the printed summary: the number of good trials, and the beep alignment
  (phone mode).
* Spot-check two clips:
  `echospeak check --wav data/raw/P01/S1/clips/0003_NEXT.wav --out reports/qc`.
  The tones must be present, and the motion must be visible after GO.
* If audio over/underruns were reported, or the alignment looked bad, re-record.
* If the clip timing in the config ever changes, run
  `echospeak segment data/raw/P01/S1` to re-cut clips from the saved full
  recording.

### 5.4 Build the dataset and reserve the test sets (once)

```bash
echospeak index --root data/raw --out data/manifest.csv     # prints a per-participant × class count table
echospeak split --manifest data/manifest.csv --out data/splits.json \
                --n-holdout-participants 2 --known-fraction 0.15 \
                [--holdout-conditions far tilted]
```

`splits.json` defines the held-out **test** sets. **Create it once, before any
tuning, and never tune on it.**

| Test set | Contents | Proposal evaluation |
|---|---|---|
| `unseen_users` | all clips of 2 held-out participants | (3) unseen participants |
| `known_users` | 15% of each remaining participant's trials, stratified by class | (1) held-out trials from known users |
| `new_condition` | sessions tagged with the given `--condition`s (altered placement) | (2) new session or placement |
| `probe_off` | all probe-off clips (automatic) | probe-off control |

Everything else is the **development set**, used for CV, tuning and the final
training.

When more participants are recorded later, re-run `index`. Keep the same
`splits.json` if possible, or make a new one and re-tune.

---

## 6. Training, model selection and tuning

### 6.1 Cross-validation protocols (development set only)

`echospeak cv --protocol …` has three protocols:

| Protocol | Folds | Corresponds to |
|---|---|---|
| `trial` | stratified 5-fold over trials; every user appears in train and validation | known users |
| `session` | hold out one recording session per participant per fold | new session or placement |
| `user` | group k-fold by participant (`--folds 0` = leave-one-participant-out) | unseen users |

Each run writes the following to `--out`:
* `report.md`: mean ± sd accuracy, macro-F1, the pooled confusion matrix,
  per-class recall, the false-trigger rate, and the **recommended threshold**;
* `summary.json`;
* `oof_predictions.csv`.

### 6.2 Step-by-step experiment plan

```bash
M="--manifest data/manifest.csv --splits data/splits.json"

# 1) Baselines (proposal: CNN first, CNN-GRU as extension) + other temporal models
for T in none gru transformer tcn; do
  echospeak cv $M --protocol user --out runs/cv-user-$T --set model.temporal=$T
done
#    Repeat the best two with --protocol session and --protocol trial for the report table.

# 2) Feature ablation: Doppler-only vs I/Q-only vs both
echospeak cv $M --protocol user --out runs/cv-doppler --set model.branches=doppler
echospeak cv $M --protocol user --out runs/cv-iq      --set model.branches=iq

# 3) Hyperparameter search (Optuna, resumable; run overnight)
echospeak tune $M --protocol user --folds 5 --trials 40 --out runs/tune
#    -> runs/tune/leaderboard.csv, runs/tune/best1.yaml .. best3.yaml

# 4) Check that an ensemble of the top configs beats the single best (it usually does)
echospeak cv $M --protocol user --config runs/tune/best1.yaml --out runs/cv-best1
echospeak cv $M --protocol user --config runs/tune/best1.yaml runs/tune/best2.yaml runs/tune/best3.yaml --out runs/cv-top3

# 5) Final model: train on the whole development set (configs × seeds ensemble),
#    storing the threshold recommended by the CV report of the chosen setup
echospeak train $M --config runs/tune/best1.yaml runs/tune/best2.yaml runs/tune/best3.yaml \
                --seeds 3 --threshold <from runs/cv-top3/report.md> --out runs/final

# 6) Evaluate ONCE on the held-out test sets
echospeak test $M --model runs/final/model.pt --out runs/final/test
```

`runs/final/test/report.md` reports each test set: accuracy, macro-F1,
command-vs-non-command accuracy, trigger metrics, the confusion matrix, and
accuracy broken down by participant, distance, orientation, condition and
session. The probe-off set is reported as a control: accuracy should be near
chance, and there should be few triggers.

### 6.3 Hyperparameter tuning guide

**Choosing the tuning protocol.** `--protocol user` optimises generalisation to
new people, which is the hardest setting and the one that matters for a demo
with a new person. If fewer than 6 development participants exist, use
`--protocol session` instead.

**Choosing the metric.** `--metric accuracy` is the default. Use
`--metric macro_f1` if the classes are unbalanced.

**Search space** (`src/echospeak/tune.py`):
* temporal model and branches;
* CNN width, `d_model`, layers, dropout;
* learning rate, weight decay, epochs, label smoothing;
* time-stretch, gain, noise, time-mask, tone-dropout and mixup strength.

**How the search runs:**
* The first trial is always the shipped default, so the search can only
  improve on it.
* Optuna's TPE sampler proposes configurations.
* After each fold, trials worse than the median are pruned.
* The study is stored in `runs/tune/study.db`. Re-running the same command
  resumes it, and increasing `--trials` continues it.

**Budget** (measured on an Apple M4 with MPS, 60 epochs): about 0.8 s per epoch
per 300 training clips. For 1,000 development clips, one 5-fold CV is about
10–15 min, so 40 trials take about 5–8 h with pruning. To go faster:
* fix the temporal model (`--temporal gru`);
* use `--folds 3`;
* run two `tune` processes on different machines with different `--out`
  directories, and compare their leaderboards.

**Manual knobs worth trying first** if tuning time is short:

| Knob | Suggested values |
|---|---|
| `train.epochs` | 40 / 60 / 90 |
| `model.dropout` | 0.2–0.4 |
| `augment.time_stretch` | 0.1–0.2 |
| `augment.mixup_alpha` | 0 / 0.2 |
| `model.temporal` | gru vs transformer |

**Read the curves.** If train accuracy is far above CV accuracy, the model
overfits: increase dropout or augmentation, or collect more participants. If
both are low, there may be a sensing problem; revisit §4 and the clip timing.

**Never select anything using `runs/final/test`.** If the test numbers are used
to change the model, they become a validation set. In that case record new
participants for a fresh test set.

### 6.4 Results table to fill in for the report

| Model | Known users (trial) | New session | Unseen users | Macro-F1 (user) | False-trigger rate | Probe-off acc. |
|---|---|---|---|---|---|---|
| CNN (`none`) | | | | | | |
| CNN-GRU | | | | | | |
| CNN-Transformer | | | | | | |
| Tuned top-3 ensemble | | | | | | |
| + IMU fusion | | | | | | |
| + IMU gating | | | | | | |

CV numbers are mean ± sd over folds on development data. Final rows also
report the held-out test sets.

### 6.5 IMU comparison (optional)

This needs IMU files for every clip (recorded in phone mode with `--imu-*`).

```bash
echospeak cv $M --protocol user --out runs/cv-imu-fusion --set features.use_imu=true   # fusion
```

* **Gating** is reported automatically. When IMU files exist, `cv` and `test`
  also print the trigger metrics with the gate applied, plus
  `imu_rejected_fraction`.
* **Tuning the gate:** adjust `gate.imu_max_gyro_rms` so that it rejects most
  `--condition motion` trials and almost no normal trials.

---

## 7. Evaluation metrics

| Metric | Definition |
|---|---|
| Accuracy | argmax over the 10 classes equals the label |
| Macro-F1 | unweighted mean of per-class F1 |
| Command-vs-non-command accuracy | binary correctness of "is this a command?" |
| Correct trigger rate | command trials whose top class is correct and has probability ≥ threshold |
| Wrong trigger rate | command trials that fire a *different* command (the worst error) |
| **False-trigger rate** | non-command trials (REST / CHEWING / YAWNING / TALKING) that fire any command |
| False activations per minute | `echospeak predict --model M long_recording.wav` slides the live gate over a continuous recording (e.g. 5 min of reading aloud, eating, idle) and lists every event |
| IMU rejected fraction | trials removed by the motion gate |
| Latency | the live demo prints inference ms per window. End-to-end latency is the gesture end to the trigger: the window hop (0.1 s) plus 2 consecutive windows plus inference. Measure it with a stopwatch or a video during the demo |
| Probe-off control | accuracy and trigger rate on probe-off clips. These must be near chance and near zero; otherwise the model is using something other than the echo (e.g. audible speech or handling noise) |

---

## 8. Live demo

The proposal's demo: a participant holds the phone about 10 cm from the face and
silently mouths "NEXT". The screen shows an acoustic motion trace, the
predicted command and its confidence, and the slide advances. Chewing or yawning
must not trigger anything.

1. **Train the demo model.**
   * Include the demo participant's own data in training. Known-user accuracy is
     much higher than unseen-user accuracy.
   * After the test numbers are reported, a final demo model may use all data:
     ```bash
     echospeak train $M --config runs/tune/best1.yaml runs/tune/best2.yaml runs/tune/best3.yaml \
                     --seeds 2 --include-test --threshold <cv value> --out runs/demo
     ```
2. **Run it.**
   ```bash
   echospeak live --model runs/demo/model.pt --action keys [--input-device N --output-device M]
   ```
   * The window shows the motion trace, per-class probability bars with the
     threshold line, and the triggered command.
   * `--action keys` presses → for NEXT and ← for BACK (macOS: give the terminal
     Accessibility permission). Open the slides in front.
   * `--no-gui` prints predictions in the terminal instead.
3. **Match the hardware.** `live` uses the computer's audio devices for the
   probe and the microphone.
   * If the dataset was recorded in duplex mode, demo the same way.
   * For a phone-held demo, the phone's microphone must stream to the laptop.
     Two options:
     * a wired USB-C/Lightning audio interface, with the phone used only as the
       speaker playing `probe.wav`;
     * the phone as a macOS input device (Continuity microphone), if
       `echospeak check` shows all tones survive. Verify this before relying on
       it.
4. **Rehearse.**
   * Check that NEXT advances the slide and that chewing and yawning do not.
   * If false triggers appear, raise `--threshold`. If commands are missed,
     lower it.

---

## 9. Timeline and fallbacks

| Week | Milestone |
|---|---|
| 1 | Milestone 1 (§4): tones captured, motion visible at 5–15 cm and varied orientations; choose the hardware setup |
| 2 | Pilot: 2–3 team members, 2 sessions each; run `cv --protocol trial` to check the pipeline on real data; fix the protocol |
| 3–4 | Main collection: ≥ 10 participants × 2 sessions, plus robustness and probe-off sessions; optional IMU |
| 4 | `split`; baselines (CNN, CNN-GRU, CNN-Transformer); feature ablation |
| 5 | Tuning plus ensemble; IMU comparison; distance, orientation and motion tests |
| 6 | Final training, the one-time test evaluation, false activations per minute on continuous recordings, the live demo, and the report |

**Fallbacks:**
* **Low accuracy:** reduce to four commands first, choosing the most distinct
  ones by the pilot's confusion matrix (e.g. NEXT, BACK, SELECT, STOP).
  Restrict the vocabulary with
  `--set "data.commands=[NEXT, BACK, SELECT, STOP]"` for `cv`, `tune` and
  `train`; clips with other labels are ignored. Expand to six afterwards.
* **Weak echoes on the phone:**
  * increase the probe amplitude (`probe.amplitude`, without clipping);
  * hold the phone closer (5–8 cm);
  * use the bottom mic and speaker orientation found best in §4.
* **Large differences between users:** collect more participants rather than
  more repetitions. For the demo, include the demo user's data (§8).

---

## 10. Repository layout

```
configs/default.yaml      all settings (probe, clip timing, features, model, augmentation, training, gate)
src/echospeak/
  probe.py                probe synthesis, beep/chirp cues, I/Q demodulation, tone SNR
  session.py              guided recording prompter (duplex / phone mode)
  segment.py              chirp + beep alignment, drift correction, clip cutting
  imu.py                  IMU CSV parsing, tap-based clock alignment, per-clip IMU
  data.py                 manifest, held-out splits, CV folds, baseband cache, dataset
  features.py             Doppler and I/Q features, augmentation
  models.py               EchoNet (two-branch CNN + none/gru/transformer/tcn)
  train.py                training loop (AdamW, cosine, label smoothing, EMA), TTA, ensembles
  evaluate.py             CV and test reports, IMU gating, breakdowns
  tune.py                 Optuna search
  metrics.py              accuracy, macro-F1, trigger metrics, threshold recommendation
  check.py, live.py       hardware check, live demo
  synth.py                synthetic data for testing
  cli.py                  `echospeak` command
tests/                    unit + end-to-end tests (no hardware needed)
data/  cache/  runs/  reports/     generated, git-ignored
```

---

## 11. References

1. R. Zhang et al. *HPSpeech: Silent Speech Interface for Commodity
   Headphones.* ISWC 2023. Acoustic echo profiles and a CNN (ResNet-18) for
   silent speech.
2. J. Luo et al. *End-to-End Silent Speech Recognition with Acoustic Sensing.*
   IEEE SLT 2021. Smartphone speaker and mic, multiple CW tones, coherent I/Q,
   phase features.
3. W. Wang, A. X. Liu, K. Sun. *Device-free gesture tracking using acoustic
   signals (LLAP).* MobiCom 2016. Multi-tone CW phase tracking and static-vector
   removal.
4. D. S. Park et al. *SpecAugment.* Interspeech 2019. Time masking.
5. H. Zhang et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
6. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
   K. Cho et al. *GRU.* EMNLP 2014.
7. T. Akiba et al. *Optuna: A Next-generation Hyperparameter Optimization
   Framework.* KDD 2019.
