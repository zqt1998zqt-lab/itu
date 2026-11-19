#!/usr/bin/env python3
"""Run one pinned NS model in an isolated worker process."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.ns_model_worker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
