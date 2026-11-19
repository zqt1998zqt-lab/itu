#!/usr/bin/env python3
"""Exercise the real batch case runner on one case per mode and scene.

Receipts live in a separate build directory. This is representative execution,
not a completed 400-case run and not formal ITU-T certification.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from server.acoustic import acceptance_batch, scene_condition_batch as batches, service
from server.acoustic.acceptance_scene_condition_api import _load_dataset
from server.acoustic.acceptance_scope import background_noise_scene_items
from server.acoustic.scene_conversation_profiles import CONVERSATION_MODES, CONVERSATION_PROFILES
from server.alignment_runner import resolve_algorithm_cluster, resolve_cli


def write_once(path: Path, value) -> None:
    batches._resource_preflight(path.parent)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    original_build = Path(service.BUILD)
    batches._resource_preflight(original_build)
    datasets = {scene: _load_dataset(profile.dataset_id)
                for scene, profile in CONVERSATION_PROFILES.items()}
    root = Path(tempfile.mkdtemp(prefix="conversation-vcp-verification-", dir=PROJECT / "build"))
    cli = resolve_cli()
    if cli is None:
        raise ValueError("VCP CLI unavailable")
    _, _, configuration, catalog_hash = resolve_algorithm_cluster("vcp_unified")
    cli_hash = hashlib.sha256(cli.read_bytes()).hexdigest()
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "one real VCP case per conversation mode and scene",
              "expected_cases": 20, "entire_400_case_batch": False,
              "cli_sha256": cli_hash, "cases": [], "root": str(root),
              "standard_compliance": False, "formal_test_status": "not_measured"}
    # This process owns its separate receipt store; no test fixtures or fake CLI.
    service.BUILD = root
    started = time.monotonic()
    resource_stopped = False
    try:
        for scene, dataset in datasets.items():
            profile = CONVERSATION_PROFILES[scene]
            identifier = uuid.uuid4().hex
            rows = {row["id"]: row for row in background_noise_scene_items("WB-wired", scene)}
            snapshot = batches._build_snapshot(dataset, rows, identifier,
                "vcp_unified", configuration, catalog_hash, cli_hash)
            batches._create_batch_directories(identifier)
            snapshot["status"] = "running"
            acceptance_batch._ACTIVE.add(identifier)
            batches._save(snapshot)
            index = {(row["case_id"], row["item_id"]): row for row in snapshot["items"]}
            for mode in CONVERSATION_MODES:
                batches._resource_preflight(root)
                record = next(case for case in snapshot["cases"]
                              if case["provenance"]["conversation_mode"] == mode)
                start = time.monotonic()
                batches._run_one_case(snapshot, dataset, record, index)
                batches.validate_snapshot(snapshot)
                items = [deepcopy(row) for row in snapshot["items"] if row["case_id"] == record["case_id"]]
                entry = {"scene": scene, "dataset_id": profile.dataset_id,
                         "case_id": record["case_id"], "mode": mode,
                         "status": record["status"], "error": record.get("error"),
                         "elapsed_s": round(time.monotonic() - start, 3),
                         "input_sha256": record["input_sha256"],
                         "output_sha256": record.get("output_sha256"),
                         "diagnostics": record.get("conversation_diagnostics"),
                         "items": [{key: row.get(key) for key in
                             ("item_id", "status", "metric", "value", "reason")} for row in items]}
                report["cases"].append(entry)
                write_once(root / f"{scene}-{mode}.json", entry)
                print(json.dumps({key: entry[key] for key in ("scene", "mode", "status", "elapsed_s", "error")}), flush=True)
                if record["status"] != "completed":
                    raise ValueError(f"actual VCP case failed: {record['case_id']}: {record.get('error')}")
                if mode in {"far_single", "transition"} and any(
                    row["value"] is not None or row["status"] != "not_measured" for row in items
                ):
                    raise ValueError("inapplicable MOS was produced")
                if mode in {"near_single", "double_talk", "noise_only"} and any(
                    row["value"] is None or row["status"] != "review_required" for row in items
                ):
                    raise ValueError("applicable engineering metric unavailable")
            acceptance_batch._ACTIVE.discard(identifier)
        report["status"] = "PASS"
        return 0
    except batches.ResourceSafetyStop:
        resource_stopped = True
        raise
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        service.BUILD = original_build
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        if resource_stopped:
            print("resource safety stop; existing evidence preserved at " + str(root), flush=True)
        else:
            write_once(root / "verification.json", report)
            print("receipt=" + str(root / "verification.json"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
