#!/usr/bin/env python3
"""Run real, single-case batches on all 48 revised dataset identities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True


def main():
    from server.acoustic.dataset_revisions_v2 import (
        install_revision_profiles, revision_scene_profile, revision_stress_profile,
    )
    from server.acoustic.acceptance_scope import ACCEPTANCE_MODES
    from server.acoustic.acceptance_scene_condition_api import _load_dataset
    from server.acoustic.acoustic_stress_api import load_stress
    from server.acoustic.scene_condition_batch import start_batch, get_batch, _resource_preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("all", "scene", "stress"), default="all")
    args = parser.parse_args()
    _resource_preflight(PROJECT)
    args.output.mkdir(parents=True, exist_ok=False)
    install_revision_profiles(activate=True)  # Isolated process; no global receipt.
    tasks = []
    for mode in sorted(ACCEPTANCE_MODES):
        if args.kind in {"all", "scene"}:
            tasks.extend(("scene", revision_scene_profile(scene, mode)) for scene in ("sce1", "sce2", "sce3", "sce4"))
        if args.kind in {"all", "stress"}:
            tasks.extend(("stress", revision_stress_profile(scope, mode)) for scope in ("echo", "double_talk"))
    results = []
    for kind, profile in tasks:
        _resource_preflight(PROJECT)
        dataset = _load_dataset(profile.dataset_id) if kind == "scene" else load_stress(profile.dataset_id)
        if kind == "scene":
            # The second double-talk case is far-dominant in the new recipe.
            matches = [case for case in dataset["cases"] if case["provenance"]["conversation_mode"] == "double_talk"
                       and float(case["provenance"]["render_metadata"]["near_to_echo_overlap_ser_db_measured"]) < 0]
        else:
            category = "path_change" if profile.scope == "echo" else "short_overlap"
            matches = [case for case in dataset["cases"] if case["provenance"]["workload_category"] == category]
        assert matches
        case = matches[0]
        started = time.monotonic()
        snapshot = start_batch(profile.dataset_id, "vcp_unified", mode=profile.mode,
                               scene=profile.scene, case_id=case["case_id"])
        batch_id = snapshot["batch_id"]
        while snapshot["status"] in {"queued", "running"}:
            _resource_preflight(PROJECT)
            if time.monotonic() - started > 300:
                raise RuntimeError(f"batch remains active after 300 seconds: {batch_id}")
            time.sleep(0.5)
            snapshot = get_batch(batch_id)
            assert snapshot is not None
        entry = {
            "kind": kind, "mode": profile.mode,
            "scope": profile.scene if kind == "scene" else profile.scope,
            "dataset_id": profile.dataset_id, "case_id": case["case_id"],
            "batch_id": batch_id, "status": snapshot["status"],
            "elapsed_s": time.monotonic() - started,
            "cases": snapshot["cases"], "summary": snapshot.get("summary"),
            "cli_sha256": snapshot["cli_sha256"],
            "standard_compliance": False,
        }
        with (args.output / f"{kind}-{entry['scope']}-{profile.mode}.json").open("x", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        results.append({key: value for key, value in entry.items() if key != "cases"})
        print(f"{len(results)}/{len(tasks)} {entry['scope']}/{profile.mode} {snapshot['status']} {entry['elapsed_s']:.2f}s", flush=True)
        if snapshot["status"] != "completed" or any(row["status"] != "completed" for row in snapshot["cases"]):
            raise RuntimeError(f"real batch failed: {batch_id}; inspect preserved snapshot")
    report = {"cases_executed": len(results), "results": results,
              "scope": "one real VCP single-case batch per revised dataset; not all 4800 cases",
              "created_at": datetime.now(timezone.utc).isoformat()}
    with (args.output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
