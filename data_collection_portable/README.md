# EchoSpeak portable data collection

Use this folder to record participants on **any laptop** (macOS, Windows or
Linux) without installing the training repository. Each run records **one
participant**. The recordings are sent back as a zip and merged into the main
dataset.

## 1. Install (once per laptop, ~1 min)

Copy this whole folder to the laptop, then:

```bash
cd data_collection_portable
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # numpy, scipy, sounddevice
```

## 2. Set up the laptop (once per laptop, ~3 min)

```bash
python collect.py --setup
```

The setup asks for or does the following:

1. **Audio devices.** Press Enter to use the defaults. Use the built-in speaker
   and mic, never Bluetooth or headphones.
2. **Collector tag.** Your initials, e.g. `JX`. Every participant ID from this
   laptop gets this prefix (`JX-P01`), so IDs never collide between laptops.
3. **Sound check.** It checks that the speaker plays and the mic captures all 8
   inaudible tones, nothing clips, and the GO beep is heard.
4. **Placement test.** You sit at a spot, stay still for 4 s, then silently
   mouth words for 4 s. It reports a mouthing/still ratio:
   - **≥ 3×** is good;
   - lower means you should move closer to the microphone and try again.
5. **A one-line seating description,** e.g. "mouth 10 cm above the left of the
   keyboard". It is shown before every run, so all participants sit the same way.

Everything is saved in `laptop_profile.json`.

## 3. Record a participant (one run, ~10 min with the defaults)

```bash
python collect.py
```

It asks for the participant ID (suggesting the next one) and runs a quick sound
check. The participant then sees:

1. one instruction screen;
2. a short practice, which is not saved;
3. the **takes**, all self-paced.

**Each take:**
1. The screen shows, for example, `NEXT  x 5`. The participant gets ready and
   presses SPACE; nothing is recorded until then.
2. A countdown runs: 3 – 2 – 1.
3. The repetitions follow, depending on the class:
   * **Words:** silently mouth the word once at each beep (every 3 s), mouth
     closed in between.
   * **YAWNING:** one yawn per beep (every 6 s).
   * **REST, CHEWING, TALKING:** keep doing it from the first beep to the second
     (15 s).

Each word's repetitions are split into takes spread over the session. A short
probe-off round comes last.

**Keys:**
* `SPACE` starts a take.
* `x` during a take redoes it; on the get-ready screen, `x` redoes the previous
  take.
* `q` stops and saves.

### Choosing how many clips per run

The script prints the exact clip count and estimated time before starting.

| Goal | How (default) |
|---|---|
| Repetitions of every word/action | `python collect.py --reps 10` |
| Different numbers per word/action | `python collect.py --class-reps NEXT=15 YAWNING=5 TALKING=0` (unlisted classes use `--reps`; 0 skips) |
| Repetitions per SPACE press | `--group-size 5` |
| Seconds between beeps | `--interval 3.0` |
| Probe-off control clips per class | `--probe-off-reps 1` (0 = none) |
| Permanent defaults | edit `REPS`, `CLASS_REPS`, `GROUP_SIZE`, `INTERVAL_S` and `PROBE_OFF_REPS` under **SETTINGS** at the top of `collect.py` |

Defaults: 12 classes × 10 = 120 clips plus 12 probe-off clips, about 10 minutes.

Other options: `--no-practice` (for repeat participants), `--condition noisy`,
`--notes "..."`, `--distance-cm 15`. See `python collect.py --help`.

## 4. Send the data back

```bash
python collect.py --pack       # -> echospeak_<TAG>_<date>.zip
```

The zip holds every session's full recording and logs. Clips are re-cut on
merge, so they are left out to keep the zip small. On the main machine, in the
training repository:

```bash
echospeak merge path/to/echospeak_JX_20261120-1530.zip
```

## Output folder

```
recordings/
  sessions.csv                one row per run (date, laptop, clip counts, sound check)
  JX-P01/S1/
    session.json              participant, laptop model, collector, settings, sound check
    cues.csv                  every prompted trial (label, GO time, probe on/off, status)
    session.wav               full recording (16-bit)
    clips/0004_NEXT.wav ...   one 3 s clip per good trial
    clips.csv
```

**Do not change** the values under "DO NOT CHANGE" in `collect.py` (tones, clip
timing, class names). They must match the training configuration, and
`echospeak merge` rejects recordings made with different probe tones.
