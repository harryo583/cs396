#!/usr/bin/env python3
"""Record one EchoSpeak data-collection session on this laptop.

    python collect.py                          # asks for the participant ID (suggests the next one)
    python collect.py --participant P03        # next session for P03 is chosen automatically
    python collect.py --help                   # all options (repetitions, distance, devices, ...)

Everything is saved under ./data (recordings, clips, manifest.csv, sessions.csv).
See plan.md section 5 for the full protocol.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from echospeak.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data" not in args:
        args += ["--data", str(ROOT / "data")]
    if "--config" not in args:
        args += ["--config", str(ROOT / "configs" / "default.yaml")]
    main(["collect", *args])
