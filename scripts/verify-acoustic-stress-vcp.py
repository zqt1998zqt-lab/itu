#!/usr/bin/env python3
"""Run real VCP representatives for every echo/double-talk mode and category."""
from __future__ import annotations

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

from server.acoustic import acceptance_batch, scene_condition_batch as engine, service
from server.acoustic.acoustic_stress_profiles import CATEGORIES, PROFILES
from server.acoustic.acoustic_stress_store import load_stress_dataset
from server.alignment_runner import resolve_algorithm_cluster, resolve_cli


def save(path, data):
    engine._resource_preflight(path.parent)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    original_build = Path(service.BUILD)
    engine._resource_preflight(original_build)
    datasets = {key: load_stress_dataset(profile.dataset_id) for key, profile in PROFILES.items()}
    root = Path(tempfile.mkdtemp(prefix="stress-vcp-representatives-", dir=PROJECT / "build"))
    cli = resolve_cli()
    if cli is None:
        raise ValueError("real VCP CLI unavailable")
    _, _, configuration, catalog_sha = resolve_algorithm_cluster("vcp_unified")
    cli_sha = hashlib.sha256(cli.read_bytes()).hexdigest()
    sources = ["server/acoustic/acoustic_stress_models.py", "server/acoustic/acoustic_stress_batch.py",
               "server/acoustic/acoustic_stress_store.py", "server/acoustic/scene_condition_batch.py"]
    source_hashes = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in sources}
    report = {"root": str(root), "scope": "16 scope/mode profiles, one case per five categories",
              "all_1600_executed": False, "expected_cases": 80,
              "cli_sha256": cli_sha, "source_sha256": source_hashes,
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    started = time.monotonic()
    resource_stopped = False
    service.BUILD = root
    try:
        for (scope, mode), profile in PROFILES.items():
            dataset = datasets[(scope, mode)]
            identifier = uuid.uuid4().hex
            contracts = {row["id"]: row for row in profile.contract_rows}
            snapshot = engine._build_snapshot(dataset, contracts, identifier,
                "vcp_unified", configuration, catalog_sha, cli_sha)
            engine._create_batch_directories(identifier)
            acceptance_batch._ACTIVE.add(identifier)
            snapshot["status"] = "running"
            engine._save(snapshot)
            index = {(row["case_id"], row["item_id"]): row for row in snapshot["items"]}
            for category_index, category in enumerate(CATEGORIES[scope]):
                case_index = category_index % profile.contract_item_count + 1
                record = next(case for case in snapshot["cases"]
                    if case["provenance"]["workload_category"] == category
                    and case["provenance"]["category_index"] == case_index)
                start = time.monotonic()
                engine._resource_preflight(root)
                engine._run_one_case(snapshot, dataset, record, index)
                engine.validate_snapshot(snapshot)
                item = next(row for row in snapshot["items"] if row["case_id"] == record["case_id"])
                result = {"scope": scope, "mode": mode, "category": category,
                    "volume": record["conditions"]["volume"], "case_id": record["case_id"],
                    "status": record["status"], "error": record.get("error"),
                    "item_status": item["status"], "value": item["value"], "metric": item["metric"],
                    "elapsed_s": round(time.monotonic() - start, 3),
                    "input_sha256": record["input_sha256"], "output_sha256": record.get("output_sha256"),
                    "diagnostics": record.get("stress_diagnostics")}
                report["cases"].append(result)
                save(root / f"{scope}-{mode}-{category}.json", result)
                print(json.dumps({k: result[k] for k in ("scope", "mode", "category", "status", "item_status", "elapsed_s")}), flush=True)
                if record["status"] != "completed" or item["status"] != "review_required" or item["value"] is None:
                    raise ValueError(f"actual VCP stress representative failed: {result}")
            acceptance_batch._ACTIVE.discard(identifier)
        if source_hashes != {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in sources}:
            raise ValueError("measurement implementation changed during the representative run")
        report["status"] = "PASS"
        return 0
    except engine.ResourceSafetyStop:
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
            print("Resource floor reached; preserved " + str(root), flush=True)
        else:
            save(root / "verification.json", report)
            print("receipt=" + str(root / "verification.json"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
