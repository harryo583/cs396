# EchoSpeak data collection (main laptop)

Use this on the **main laptop**: the one used for training and the live demo.
Recordings go straight into the repository's `data/` folder. The dataset
manifest is updated automatically after every run.

To record on other laptops, use [`../data_collection_portable/`](../data_collection_portable/README.md)
and merge its zip with `echospeak merge`.

## Before the first participant

```bash
pip install -e '.[all]'          # from the repository root, once
echospeak check                  # verify the laptop captures the probe and sees lip motion (plan.md §4)
```

## One run = one participant session (~10 min with the defaults)

```bash
python data_collection/collect.py                     # asks for the participant ID (suggests P01, P02, ...)
python data_collection/collect.py --participant P03   # P03's next session (S1, S2, ...) is automatic
```

Recording is **self-paced**: nothing is recorded while the participant gets ready.
Each screen shows one **take**, for example `NEXT  x 5`:

1. The participant presses SPACE.
2. A countdown runs: 3 – 2 – 1.
3. The repetitions follow, depending on the class:
   * **Words:** a beep every 3 s; silently mouth the word once per beep, mouth
     closed in between.
   * **YAWNING:** one yawn per beep, every 6 s.
   * **REST, CHEWING, TALKING:** keep doing it for 15 s, from the first beep to
     the second.

Each beep, or each 3 s of the continuous activities, becomes one clip.

The run has three parts:
1. a sound check;
2. an instruction screen and a short practice (not saved);
3. the takes.

Each word's 10 repetitions are split into 2 takes spread across the session.
A short probe-off round comes last. The run then cuts the clips and updates
`data/manifest.csv` and `data/sessions.csv`.

**Keys:**
* `SPACE` starts a take.
* `x` during a take redoes it; on the get-ready screen, `x` redoes the previous
  take.
* `q` stops and saves.

## Choosing how many clips per run

The run prints the exact clip count and estimated time before it starts.

| Goal | Option (default) |
|---|---|
| Repetitions of every word/action | `--reps 10` |
| Different numbers per class | `--class-reps NEXT=15 YAWNING=5 TALKING=0` (unlisted classes use `--reps`; 0 skips) |
| Repetitions per SPACE press | `--group-size 5` |
| Seconds between beeps | `--interval 3.0` |
| Probe-off control clips per class | `--probe-off-reps 1` (0 = none) |

Defaults: 12 classes × 10 = 120 clips plus 12 probe-off clips, about 10 minutes.

Other options: `--no-practice`, `--distance-cm 10`, `--condition noisy`,
`--notes "..."`, `--input-device/--output-device` (see `echospeak devices`),
and `--help`.

## Output

```
data/
  manifest.csv      all clips (rebuilt after every run)
  sessions.csv      one row per run
  raw/P01/S1/       session.json, cues.csv, session.wav, clips/*.wav, clips.csv
```
