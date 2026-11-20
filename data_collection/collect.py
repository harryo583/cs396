#!/usr/bin/env python3
"""Record one EchoSpeak session on the MAIN laptop (the one used for training and the demo).

    python data_collection/collect.py                         # asks for the participant ID (suggests the next one)
    python data_collection/collect.py --participant P03       # P03's next session is chosen automatically
    python data_collection/collect.py --reps 3                # 3 repetitions of every word/action
    python data_collection/collect.py --class-reps NEXT=8 TALKING=2   # per-class counts
    python data_collection/collect.py --help                  # all options

Clips go straight into <repo>/data (raw/, manifest.csv, sessions.csv).
For other laptops use data_collection_portable/ instead. See data_collection/README.md.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from echospeak.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data" not in args:
        args += ["--data", str(REPO / "data")]
    if "--config" not in args:
        args += ["--config", str(REPO / "configs" / "default.yaml")]
    main(["collect", *args])
