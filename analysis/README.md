# Preliminary data analysis

**[`preliminary_analysis.ipynb`](preliminary_analysis.ipynb)** checks whether a recorded
session is usable and whether the classes can be told apart. To analyse another
recording, set `PARTICIPANT` / `SESSION` in the first cell. Figures are written to
[`figures/`](figures/).

```bash
pip install -e '.[all]'                    # includes scikit-learn and jupyter
jupyter lab analysis/preliminary_analysis.ipynb
```

## Results for P02 / S1 (2026-11-19, MacBook Mac16,1, 10 cm)

The session has 120 clips (12 classes × 10, recorded as 2 takes of 5) plus 12
probe-off clips.

### 1. The recording is technically sound

| Check | Result |
|---|---|
| Probe tone SNR | 62–66 dB on all 8 tones (need ≥ 20) |
| Clipping / dropouts | 82 clipped samples, all in the first 10 s (sound check and practice), before the first take at 104 s; 0 dropouts |
| Clip timing | GO beeps detected; clips are aligned to the beep |
| Probe-off control | tones drop by 86 dB; a classifier trained on probe-on data scores 0.08 on probe-off clips (chance 0.08), as it should |
| Artefact | constant amplitude-modulation sidebands at ±44 Hz (−37 dB) on every tone, from the playback/recording electronics. They don't change with motion ([fig](figures/P02_1_spectrum.png)) |

### 2. Mouthing produces a clear, time-locked echo response

After coherent demodulation, removing the static path (2 s moving average) and
low-passing at 15 Hz:
* Every word shows an echo-motion response 0.4–1.6 s after the beep
  ([fig](figures/P02_4_class_responses.png)).
* REST, CHEWING and TALKING stay flat relative to their own baseline. This is
  expected: they are continuous activities, so motion is present both before and
  after the cue.
* Different words have different temporal shapes ([fig](figures/P02_6_class_templates.png)):
  * BANANA: long, strong;
  * NEXT and APPLE: a single peak around 0.7–0.9 s;
  * STOP: late and strong;
  * GO: late and weak;
  * BACK and OKAY: the weakest.

| Class | Median motion increase after beep | Repetitions with > 1.5× increase |
|---|---|---|
| BANANA | 2.4× | 90% |
| STOP | 2.5× | 70% |
| NEXT | 2.1× | 80% |
| GO | 1.8× | 80% |
| APPLE | 1.7× | 60% |
| MENU | 1.6× | 60% |
| OKAY | 1.5× | 40% |
| BACK | 1.4× | 50% |
| REST | 0.9× | 0% |

**Detecting a single repetition:** words vs REST gives ROC AUC **0.79**. 66% of word
repetitions show more than a 1.5× motion increase, against 0% of REST
([fig](figures/P02_5_detectability.png)).

### 3. Telling the classes apart: above chance, but far from usable yet

Evaluated **across takes**: train on one take (5 repetitions) per class, test on the
other.

| Method | 12 classes (chance 0.08) | 8 words (chance 0.125) |
|---|---|---|
| Best hand-made features + classical model | 0.12–0.20 | up to 0.21–0.24 |
| CNN pipeline (`none` and `transformer`) | 0.15 | n/a |
| Echo *level* only (`static`, a posture-confound control) | 0.03–0.07, i.e. chance | 0.06–0.11 |
| Leaky random split (for comparison only) | 0.32 | n/a |

**How to read this:**
* **There is real signal.** Every method scores above chance. The `static` control
  is at chance, so the models aren't exploiting posture differences between takes.
* **Data quantity is the main limit right now.** Five training repetitions per class
  are far too few, especially for a CNN. The gap between the leaky split (0.32) and
  the cross-take split (0.15) shows that repetitions within a take resemble each
  other much more than repetitions in different takes. The model has to learn
  across sessions and takes, which needs many more of them.
* **Single repetitions are noisy.** About a third of word repetitions show no clear
  motion increase. BACK and OKAY are the weakest: small jaw and lip movement,
  little visible change.

## Recommendations before collecting everyone

1. **Fix the take-start disturbance.** Pressing SPACE causes a large echo
   disturbance (hand and body movement) that lasts about 1 s. With the current
   1.5 s countdown, it reaches into the first clip of each take
   ([fig](figures/P02_2_filtering_NEXT_take.png), panel b, at ~164.5 s). Increase
   `COUNTDOWN_S` to about 2.5 s in both collection scripts.
2. **Increase the echo from the mouth:**
   * sit closer: 5–8 cm from the microphone instead of 10 cm;
   * ask participants to articulate clearly, with visible lip and jaw movement
     (still silent);
   * re-run `echospeak check` at a few positions and choose the one with the
     largest mouthing/still ratio.
3. **Collect more takes and sessions per person,** not just more participants.
   The cross-take result is what generalises. Two sessions per participant is the
   minimum; more takes per class per session (e.g. `--group-size 3 --reps 12`, which
   gives 4 takes) helps the model learn the variation between takes.
4. **Prefer distinct words.** BANANA, STOP, NEXT, GO and APPLE give the clearest
   responses for this participant, BACK and OKAY the weakest. Re-check after more
   participants, before fixing the final command set.
5. **Optional pipeline tweak:** the ±44 Hz sidebands sit inside the Doppler range.
   Setting `features.doppler_max_hz: 40` excludes them. Lip motion is mostly below
   30 Hz.
6. **Re-run this notebook** for every new participant, and re-run
   `echospeak cv --protocol trial` (take-aware) once there are 3 or more sessions.
   That is the first point where the CNN accuracy becomes meaningful.
