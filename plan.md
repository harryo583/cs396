# EchoSpeak: Silent Command Recognition via Acoustic Echoes

**Team (Group 5):** Jerry Xiang, Harry Wang, Jacob Lyu, Yixuan Xiao, Chris Chen

This document is the project plan and the operating manual for this repository.
It keeps the content of the original proposal (`plan.pdf`, now removed) and adds
how the implementation works, how to record data, and the exact commands for
training, tuning, evaluating and demoing.

**Change from the proposal:** everything runs on a **laptop's built-in speaker and
microphone**, so no phone, extra hardware or app is needed. The sensing
principle, classes and evaluation are unchanged. The phone IMU extension is
dropped, because a laptop has no motion sensor in the user's hand.

---

## 1. Goal

EchoSpeak turns an ordinary device into a hands-free, camera-free interface for
silent commands. It detects changes in acoustic reflections caused by lip and jaw
movement. From these it recognises a small, predefined set of silently mouthed
commands, with no spoken audio, video or wearable sensors. Target settings are
libraries, meetings and noisy places where voice input is inconvenient or
unreliable.

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
2. new session or altered seating/placement;
3. unseen participants.

**Also reported:** macro-F1, false-trigger rate and response latency.

---

## 2. Method

### 2.1 Sensing

The laptop speaker plays a continuous, inaudible high-frequency probe. The
microphone captures the direct signal plus the echoes from the face as raw PCM
audio.

* **Probe:** 8 continuous-wave tones, 18.00–19.75 kHz, 250 Hz apart, with
  Schroeder phases for a low crest factor. The 250 Hz spacing is far larger
  than the Doppler shift of lips and jaw (at most about ±60 Hz at 19 kHz), so the
  tones never interfere. Eight tones at different frequencies also make the
  system robust when one frequency fades because of multipath.
* **Coherent I/Q demodulation:** each tone is mixed down to complex baseband,
  low-pass filtered at 110 Hz and sampled at 500 Hz. Each tone's baseband
  `h_k(t)` is the sum of two parts:
  * the static paths (speaker-to-mic leakage, face, desk, screen), which are
    constant;
  * the articulator echo, whose phase follows the round-trip path length
    (one wavelength ≈ 1.8 cm, so 1 mm of lip motion ≈ 0.7 rad).
* **Normalisation:** `rel_k(t) = h_k(t) / mean(h_k) − 1` is the echo
  *relative to the static path*. It cancels the volume, the speaker and mic
  frequency response, and the arbitrary carrier phase, so features are
  comparable across laptops and sessions.

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
would only memorise the data. The default is a starting point; tuning (§6)
decides on real data.

**What actually maximises test accuracy with little data** (all implemented):

1. *Physically motivated augmentation* (`augment:` section):
   * random crop and time-stretch: reaction time and speaking rate;
   * echo-gain jitter: distance to the laptop;
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

---

## 3. Setup

Python ≥ 3.10 on a laptop with a built-in speaker and microphone. Training runs
on Apple Silicon (MPS, used automatically), CUDA or CPU.

```bash
cd cs396
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'      # torch, numpy, scipy, pyyaml, sounddevice, matplotlib, optuna, pytest
pytest -q                    # 26 tests, ~15 s, no hardware needed
echospeak --help
```

`echospeak <command> --help` documents every option. All experiment settings
live in `configs/default.yaml`. Any value can be overridden on the command line
with `--set section.key=value`, e.g. `--set model.temporal=gru train.epochs=80`.

`echospeak synth` creates a fake dataset in the same format, which is useful for
trying the training commands before real data exists. Its accuracy says nothing
about real sensing.

---

## 4. Milestone 1: verify usable acoustic reflections

Do this once per laptop model, before collecting data. Use the same laptop
(or the same model) for collection and for the demo: speaker and mic placement
change the echoes.

```bash
echospeak devices                        # optional: list audio devices
echospeak check --out reports/check      # 12 s: still → hand waving → silent mouthing
```

### Pass criteria

All of the following must hold:
* **Tone SNR ≥ 20 dB for all 8 tones.** If tones are missing:
  * raise the volume;
  * use the built-in speaker, never Bluetooth or headphones;
  * on macOS, set Control Center → Mic Mode to *Standard*;
  * try another device with `--input-device` / `--output-device`.
* **Peak level < 0.98:** no clipping.
* **Motion energy while mouthing is ≥ 3–5× the "still" energy**, and the
  Doppler plot (`reports/check/check.png`) shows clear streaks when the lips
  move.

Then try 5, 10 and 15 cm from the microphone to pick the working distance for
data collection. Find where the laptop's microphone is: usually near the camera
or beside the keyboard.

---

## 5. Data collection

### 5.1 One command per session

All collection runs through **one script**. It handles the sound check,
prompting the participant, recording, cutting clips and updating the dataset:

```bash
python collect.py                          # asks for the participant ID, suggesting the next free one (P01, P02, ...)
python collect.py --participant P03        # P03's next session (S1, S2, ...) is chosen automatically
```

What happens in one run (about 6 minutes):

| Step | Who | What |
|---|---|---|
| 1. Setup | operator | Confirm the participant ID. The script prints the plan, the estimated time and the data folder |
| 2. Sound check (3 s) | operator | Checks that all probe tones are captured, nothing clips, and the mic hears the GO beep. If something is wrong, it explains the fix and offers retry / continue / quit, *before* the participant starts |
| 3. Instructions | participant | One screen explaining the task. Press SPACE |
| 4. Practice | participant | 3 trials (NEXT, CHEWING, TALKING), not saved |
| 5. Main part | participant | 5 blocks; every class appears once per block, in random order. A progress bar shows how much time is left. Breaks between blocks continue automatically after 10 s, or press SPACE |
| 6. Last part (~1 min) | participant | One more block with the probe silently switched off: the probe-off control |
| 7. Saving | automatic | Saves the recording, cuts one clip per trial, rebuilds `data/manifest.csv`, and adds a row to `data/sessions.csv` |

**Each trial:**
1. The word appears ("Get ready: NEXT") for 1.2 s.
2. **GO** appears with a beep, and the participant does the action once.
3. "relax".

The saved clip runs from GO −0.75 s to GO +2.25 s. The GO beep is also recorded
by the microphone, which makes the clip timing exact.

**Keys during the session** (participant or operator):

| Key | Effect |
|---|---|
| `SPACE` | continue |
| `x` | redo the last trial (it is discarded and repeated later in the block) |
| `p` | pause / resume |
| `q` | stop early; everything recorded so far is still saved and cut |

**Useful options** (`python collect.py --help` lists all):

| Option | Meaning |
|---|---|
| `--reps 5` | number of blocks |
| `--probe-off-reps 1` | number of probe-off blocks (0 = none) |
| `--no-practice` | skip the practice, e.g. for a participant's second session |
| `--distance-cm 10` | mouth-to-microphone distance, recorded in the metadata |
| `--condition normal` | free tag, e.g. `far`, `noisy`, `moving` |
| `--notes "..."` | free text saved with the session |
| `--yes` | skip operator questions |

### 5.2 What to collect

| Item | Plan |
|---|---|
| Participants | ≥ 10 (the proposal's target). More participants help unseen-user accuracy more than more repetitions do |
| Sessions | 2 per participant, on different days or after re-seating (`python collect.py --participant P01` again creates S2) |
| Repetitions | 5 per class per session (default). 10 classes × 5 × 2 sessions × 10 people = **1,000 trials** |
| Placement | sit facing the laptop, mouth ~10 cm (a hand's width) from the microphone, head still between prompts |
| Robustness (optional) | for some participants, an extra session with `--condition far --distance-cm 20` and/or `--condition noisy` (a noisy room) |
| Probe-off control | automatic: the last block of every session |

**Instructions shown to participants:**

| Class | Instruction |
|---|---|
| commands | silently mouth the word: lips and jaw move, but there is **no sound and no whisper** |
| `REST` | stay still and relaxed, mouth closed |
| `CHEWING` | chew naturally, as if chewing gum |
| `YAWNING` | yawn, or imitate a yawn |
| `TALKING` | say a short sentence out loud (anything) |

**Tips for low hassle:**
* Use a quiet room, the same chair and the same laptop position for everyone.
* The operator sits next to the participant and can press `x` when a trial
  goes wrong.
* For a second session the practice can be skipped (`--no-practice`).

Participant IDs are anonymous (`P01`, …). All recordings stay in `data/`, which
is git-ignored.

### 5.3 Data folder

```
data/
  manifest.csv          every clip (participant, session, label, probe on/off, distance, condition); rebuilt after each session
  sessions.csv          one row per session: date, clip counts, sound-check SNR, notes
  raw/P01/S1/
    session.json        participant, session, distance, condition, sound-check result, settings
    cues.csv            every prompted trial: label, GO time, probe on/off, status (ok / discarded / practice)
    session.wav         the full continuous recording (16-bit, 48 kHz)
    clips/0004_NEXT.wav one 3 s clip per good trial
    clips.csv           clip index
```

About 50 MB per session.

**After a session**, glance at the printed summary. Re-record if it reports a
sound-check problem or audio dropouts.

To inspect a clip:
```bash
echospeak check --wav data/raw/P01/S1/clips/0004_NEXT.wav --out reports/qc
```

If the clip timing in `configs/default.yaml` is ever changed, run
`echospeak segment data/raw/P01/S1` (for each session) and then
`echospeak index` to re-cut every clip from the saved full recordings.

### 5.4 Reserve the test sets (once, after collection)

```bash
echospeak index     # prints a participant × class count table (the manifest is already up to date)
echospeak split --out data/splits.json --n-holdout-participants 2 --known-fraction 0.15 \
                [--holdout-conditions far noisy]
```

`splits.json` defines the held-out **test** sets. **Create it once, before any
tuning, and never tune on it.**

| Test set | Contents | Proposal evaluation |
|---|---|---|
| `unseen_users` | all clips of 2 held-out participants | (3) unseen participants |
| `known_users` | 15% of each remaining participant's trials, stratified by class | (1) held-out trials from known users |
| `new_condition` | sessions tagged with the given `--condition`s | (2) altered placement |
| `probe_off` | all probe-off clips (automatic) | probe-off control |

Everything else is the **development set**, used for CV, tuning and the final
training.

---

## 6. Training, model selection and tuning

### 6.1 Cross-validation protocols (development set only)

`echospeak cv --protocol …` has three protocols:

| Protocol | Folds | Corresponds to |
|---|---|---|
| `trial` | stratified 5-fold over trials; every user appears in train and validation | known users |
| `session` | hold out one recording session per participant per fold | new session |
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
accuracy broken down by participant, distance, condition and session. The
probe-off set is reported as a control: accuracy should be near chance, and
there should be few triggers.

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
both are low, there may be a sensing problem; revisit §4.

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

CV numbers are mean ± sd over folds on development data. Final rows also report
the held-out test sets.

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
| Latency | the live demo prints inference ms per window. End-to-end latency is the gesture end to the trigger: the window hop (0.1 s) plus 2 consecutive windows plus inference. Measure it with a stopwatch or a video during the demo |
| Probe-off control | accuracy and trigger rate on probe-off clips. These must be near chance and near zero; otherwise the model is using something other than the echo (e.g. audible speech or noise) |

---

## 8. Live demo

The demo: a participant sits about 10 cm from the laptop microphone and silently
mouths "NEXT". The screen shows an acoustic motion trace, the predicted command
and its confidence, and the slide advances. Chewing or yawning must not trigger
anything.

1. **Train the demo model.**
   * Include the demo participant's own data in training. Known-user accuracy is
     much higher than unseen-user accuracy.
   * After the test numbers are reported, a final demo model may use all data:
     ```bash
     echospeak train $M --config runs/tune/best1.yaml runs/tune/best2.yaml runs/tune/best3.yaml \
                     --seeds 2 --include-test --threshold <cv value> --out runs/demo
     ```
2. **Run it on the same laptop** used for collection:
   ```bash
   echospeak live --model runs/demo/model.pt --action keys
   ```
   * The window shows the motion trace, per-class probability bars with the
     threshold line, and the triggered command.
   * `--action keys` presses → for NEXT and ← for BACK (macOS: give the terminal
     Accessibility permission). Put the slides in front.
   * `--no-gui` prints predictions in the terminal instead.
3. **Rehearse.**
   * Check that NEXT advances the slide and that chewing and yawning do not.
   * If false triggers appear, raise `--threshold`. If commands are missed,
     lower it.

---

## 9. Timeline and fallbacks

| Week | Milestone |
|---|---|
| 1 | Milestone 1 (§4): tones captured, motion visible at 5–15 cm; choose the laptop and working distance |
| 2 | Pilot: 2–3 team members, 2 sessions each with `collect.py`; run `cv --protocol trial` to check the pipeline on real data; adjust the protocol |
| 3–4 | Main collection: ≥ 10 participants × 2 sessions (+ optional far/noisy sessions) |
| 4 | `split`; baselines (CNN, CNN-GRU, CNN-Transformer); feature ablation |
| 5 | Tuning plus ensemble; distance and noise tests |
| 6 | Final training, the one-time test evaluation, false activations per minute on continuous recordings, the live demo, and the report |

**Fallbacks:**
* **Low accuracy:** reduce to four commands first, choosing the most distinct
  ones by the pilot's confusion matrix (e.g. NEXT, BACK, SELECT, STOP).
  Restrict the vocabulary with
  `--set "data.commands=[NEXT, BACK, SELECT, STOP]"` for `cv`, `tune` and
  `train`; clips with other labels are ignored. Expand to six afterwards.
* **Weak echoes:**
  * raise the volume or `probe.amplitude` (without clipping);
  * sit closer (5–8 cm from the microphone);
  * keep hands and objects away from the speaker–mic path.
* **Large differences between users:** collect more participants rather than
  more repetitions. For the demo, include the demo user's data (§8).

---

## 10. Repository layout

```
collect.py                one-command data collection (sound check, prompts, clips, manifest)
configs/default.yaml      all settings (probe, clip timing, features, model, augmentation, training, gate)
src/echospeak/
  collect.py              collection session logic and participant UI
  probe.py                probe synthesis, GO beep, I/Q demodulation, tone SNR
  segment.py              beep-based alignment and clip cutting
  data.py                 manifest, held-out splits, CV folds, baseband cache, dataset
  features.py             Doppler and I/Q features, augmentation
  models.py               EchoNet (two-branch CNN + none/gru/transformer/tcn)
  train.py                training loop (AdamW, cosine, label smoothing, EMA), TTA, ensembles
  evaluate.py             CV and test reports, breakdowns
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
   IEEE SLT 2021. Speaker and mic sensing, multiple CW tones, coherent I/Q,
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
