#!/usr/bin/env python3
"""Incrementally hardlink completed evaluation audio by SHA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from common import atomic_json, dedupe_completed_tree, load_json, utc_now


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args(argv)
    store = args.store.expanduser().resolve()
    status = {"schema_version": 1, "status": "running", "started_utc": utc_now(), "roots": [], "store": str(store)}
    atomic_json(args.status, status)
    total = 0
    for root in args.root:
        root = root.expanduser().resolve()
        completed_cases = 0
        completed_shards = 0
        files = 0
        for receipt_path in root.rglob("receipt.json"):
            try:
                document = load_json(receipt_path)
            except Exception:
                continue
            if document.get("status") != "complete" or int(document.get("failed", 1)) != 0:
                continue
            result = dedupe_completed_tree(receipt_path.parent, store)
            files += result["files"]
            completed_cases += 1
            total += result["files"]
        for status_path in root.rglob("score-shards/shard-*/steps/status.json"):
            try:
                document = load_json(status_path)
            except Exception:
                continue
            if document.get("status") != "ok":
                continue
            scored_dir = status_path.parents[1] / "scored"
            normalized = scored_dir / "cache/normalized"
            if not normalized.is_dir():
                continue
            result = dedupe_completed_tree(normalized, store)
            files += result["files"]
            completed_shards += 1
            total += result["files"]
        status["roots"].append({
            "root": str(root),
            "completed_cases": completed_cases,
            "completed_score_shards": completed_shards,
            "audio_files": files,
        })
        atomic_json(args.status, status)
    status.update({"status": "complete", "finished_utc": utc_now(), "audio_files": total})
    atomic_json(args.status, status)
    print(json.dumps({"status": "complete", "audio_files": total, "status_path": str(args.status)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

