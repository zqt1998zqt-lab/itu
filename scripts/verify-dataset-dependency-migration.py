#!/usr/bin/env python3
"""Verify active APIs and old report reads while all retired inputs are denied."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT, PROJECT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
sys.dont_write_bytecode = True


def main():
    from fastapi.testclient import TestClient
    from server.app import app
    from server.acoustic.dataset_dependency_archive import DatasetMetadataArchive
    from server.acoustic.dataset_revisions_v2 import REVISION_SCENE_PROFILES, REVISION_STRESS_PROFILES
    from server.acoustic import acceptance_batch, service

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    free = shutil.disk_usage(PROJECT).free
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if free < 100 * 1024**3 or available < 5 * 1024**3:
        raise RuntimeError("resource floor crossed; preserve all files")
    review = json.loads((PROJECT / "build/dataset-revision-20261117/cleanup-inventory-20261117-readonly/review-manifest.json").read_text())
    old_roots = tuple(str(Path(row["path"])) for row in review["entries"] if row.get("dataset_id"))
    archive = DatasetMetadataArchive()
    old_ids = set(archive.entries)
    assert len(old_roots) == len(old_ids) == 103
    contract_batches = []
    batch_root = service.BUILD / "acceptance-batches"
    # Read only the latest report surface per batch; avoid scanning its audio.
    for directory in sorted(batch_root.iterdir()):
        if not directory.is_dir() or len(directory.name) != 32:
            continue
        path = acceptance_batch._report_path(directory.name)
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        if value.get("dataset_id") in old_ids and value.get("status") in {"completed", "failed"}:
            contract_batches.append((directory.name, value.get("execution_scope") == "single_case"))
    result = {"retired_stores_denied": len(old_roots), "blocked_reads": [],
              "active_scene_apis": [], "active_stress_apis": [], "historical_contract_batches": [],
              "historical_scene_lists": [], "historical_outputs_checked": [],
              "retired_requests": [], "metadata_records_checked": 0}
    original_bytes, original_text, original_open = Path.read_bytes, Path.read_text, Path.open

    def reject(path):
        text = str(Path(path).absolute())
        if any(text == root or text.startswith(root + "/") for root in old_roots):
            result["blocked_reads"].append(text)
            raise AssertionError("retired source store was read: " + text)

    def read_bytes(path, *a, **kw):
        reject(path); return original_bytes(path, *a, **kw)

    def read_text(path, *a, **kw):
        reject(path); return original_text(path, *a, **kw)

    def open_path(path, *a, **kw):
        reject(path); return original_open(path, *a, **kw)

    with patch.object(Path, "read_bytes", read_bytes), patch.object(Path, "read_text", read_text), patch.object(Path, "open", open_path):
        client = TestClient(app)
        dataset_response = client.get("/api/alignment/acceptance/datasets")
        assert dataset_response.status_code == 200, dataset_response.text
        dataset_body = dataset_response.json()
        dataset_rows = dataset_body if isinstance(dataset_body, list) else dataset_body["datasets"]
        assert all(row["dataset_id"] not in old_ids for row in dataset_rows)
        result["current_contract_datasets"] = [row["dataset_id"] for row in dataset_rows]
        for identifier in old_ids:
            assert archive.load(identifier)["dataset"]["dataset_id"] == identifier
            result["metadata_records_checked"] += 1
        for (scene, mode), profile in REVISION_SCENE_PROFILES.items():
            response = client.get("/api/alignment/acceptance/scene-condition-datasets",
                                  params={"scope": "background_noise", "scene": scene, "mode": mode})
            assert response.status_code == 200, response.text
            assert response.json()["datasets"][0]["dataset_id"] == profile.dataset_id
            result["active_scene_apis"].append(profile.dataset_id)
        for (scope, mode), profile in REVISION_STRESS_PROFILES.items():
            response = client.get("/api/alignment/acceptance/stress-datasets", params={"scope": scope, "mode": mode})
            assert response.status_code == 200, response.text
            assert response.json()["dataset_id"] == profile.dataset_id
            result["active_stress_apis"].append(profile.dataset_id)
        for batch_id, single in contract_batches:
            response = client.get(f"/api/alignment/acceptance/batches/{batch_id}/report")
            assert response.status_code == 200, f"historical contract {batch_id}: {response.text}"
            result["historical_contract_batches"].append({"batch_id": batch_id, "single_case": single})
            report = response.json()
            for case in report.get("cases", []):
                hashes = case.get("execution", {}).get("output_sha256", {})
                if case.get("status") != "completed" or not hashes:
                    continue
                kind = "tx" if "tx" in hashes else "rx"
                output = client.get(f"/api/alignment/acceptance/batches/{batch_id}/cases/{case['case_id']}/outputs/{kind}")
                assert output.status_code == 200, output.text
                assert hashlib.sha256(output.content).hexdigest() == hashes[kind]
                result["historical_outputs_checked"].append({"batch_id": batch_id, "case_id": case["case_id"], "kind": kind})
                break
        for dataset_id in ("cf29475e343052eaffa73dfebad1b0b0", "0563b621e16a1a7702dd7994683ddda2"):
            response = client.get("/api/alignment/acceptance/scene-condition-batches", params={
                "dataset_id": dataset_id, "algorithm_cluster_id": "vcp_unified",
                "latest_per_case": "true", "limit": 20,
            })
            assert response.status_code == 200, response.text
            result["historical_scene_lists"].append({"dataset_id": dataset_id, "returned": len(response.json())})
        for dataset_id, record in archive.entries.items():
            payload = archive.load(dataset_id)
            if record["kind"] == "contract":
                url = f"/api/alignment/acceptance/datasets/{dataset_id}/cases"
                response = client.get(url, params={"test_scope": "basic_uplink", "mode": "WB-wired"})
            elif record["kind"] == "stress":
                response = client.get(f"/api/alignment/acceptance/stress-datasets/{dataset_id}/preview")
            else:
                response = client.get("/api/alignment/acceptance/scene-condition-preview", params={
                    "dataset_id": dataset_id, "mode": payload["dataset"]["mode"], "scene": payload["dataset"]["scene"],
                })
            assert response.status_code == 410, f"retired dataset still available: {dataset_id}"
            assert response.json()["replacement_dataset_id"] not in old_ids
            try:
                acceptance_batch.start_batch(dataset_id, "vcp_unified")
            except ValueError as error:
                assert "退役" in str(error)
            else:
                raise AssertionError("retired dataset started an execution")
            result["retired_requests"].append(dataset_id)
    assert result["blocked_reads"] == []
    assert result["historical_outputs_checked"], "history playback checks must exercise at least one real output"
    result.update(status="PASS", files_deleted=0, checked_at=datetime.now(timezone.utc).isoformat(),
                  scope="retired-input denial in an isolated process; no files moved or deleted")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": "PASS", "active_apis": len(result["active_scene_apis"]) + len(result["active_stress_apis"]),
                      "historical_contract_reports": len(result["historical_contract_batches"]),
                      "historical_single_reports": sum(row["single_case"] for row in result["historical_contract_batches"]),
                      "retired_410": len(result["retired_requests"]), "old_store_reads": 0,
                      "report": str(args.report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
