from __future__ import annotations

from copy import deepcopy
import threading

import pytest

from server.acoustic import scene_condition_batch


def _snapshot(case_ids: tuple[str, ...] = ("case-a", "case-b")) -> dict:
    return {
        "batch_id": "a" * 32,
        "dataset_id": "b" * 32,
        "dataset_name": "parallel-test",
        "algorithm_cluster_id": "localvqe_aec200k",
        "configuration": {"profile": "vcp"},
        "selected_mode": "WB-wired",
        "progress": {
            "completed_cases": 0,
            "total_cases": len(case_ids),
            "completed_items": 0,
            "total_items": len(case_ids),
        },
        "summary": {
            "pass": 0,
            "fail": 0,
            "review_required": 0,
            "not_measured": 0,
            "invalid": 0,
            "total": len(case_ids),
            "computed": 0,
        },
        "cases": [
            {
                "case_id": case_id,
                "status": "pending",
                "contract_item_ids": [f"item-{case_id}"],
            }
            for case_id in case_ids
        ],
        "items": [
            {
                "case_id": case_id,
                "item_id": f"item-{case_id}",
                "status": "pending",
            }
            for case_id in case_ids
        ],
    }


def test_case_worker_count_is_bounded_and_configurable(monkeypatch):
    monkeypatch.delenv(scene_condition_batch.CASE_WORKERS_ENV, raising=False)
    assert scene_condition_batch._case_worker_count(100) == 2
    assert scene_condition_batch._case_worker_count(1) == 1
    assert scene_condition_batch._case_worker_count(0) == 0

    monkeypatch.setenv(scene_condition_batch.CASE_WORKERS_ENV, "3")
    assert scene_condition_batch._case_worker_count(100) == 3
    assert scene_condition_batch._case_worker_count(2) == 2

    for invalid in ("0", "5", "not-a-number"):
        monkeypatch.setenv(scene_condition_batch.CASE_WORKERS_ENV, invalid)
        with pytest.raises(ValueError):
            scene_condition_batch._case_worker_count(100)


def test_case_work_snapshot_isolated_from_parent():
    parent = _snapshot()
    parent["cases"][1]["tx_visual"] = {"payload": "must-not-be-copied"}
    work = scene_condition_batch._case_work_snapshot(parent, "case-a")

    assert [case["case_id"] for case in work["cases"]] == ["case-a"]
    assert [row["item_id"] for row in work["items"]] == ["item-case-a"]
    work["cases"][0]["status"] = "running"
    work["items"][0]["status"] = "running"
    assert parent["cases"][0]["status"] == "pending"
    assert parent["items"][0]["status"] == "pending"
    assert "tx_visual" not in work["cases"][0]


def test_pending_cases_run_concurrently_and_merge_in_stable_order(
    monkeypatch,
):
    snapshot = _snapshot()
    original = deepcopy(snapshot)
    barrier = threading.Barrier(2)
    saves: list[dict] = []

    def fake_run(work, dataset):
        barrier.wait(timeout=5)
        case = work["cases"][0]
        case["status"] = "completed"
        for row in work["items"]:
            row["status"] = "not_measured"
        return deepcopy(case), deepcopy(work["items"])

    monkeypatch.setenv(scene_condition_batch.CASE_WORKERS_ENV, "2")
    monkeypatch.setattr(
        scene_condition_batch,
        "_run_case_snapshot",
        fake_run,
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "_save",
        lambda value: saves.append(deepcopy(value)),
    )

    scene_condition_batch._run_pending_cases(snapshot, {}, {})

    assert [case["case_id"] for case in snapshot["cases"]] == [
        "case-a",
        "case-b",
    ]
    assert [case["status"] for case in snapshot["cases"]] == [
        "completed",
        "completed",
    ]
    assert [row["item_id"] for row in snapshot["items"]] == [
        "item-case-a",
        "item-case-b",
    ]
    assert all(row["status"] == "not_measured" for row in snapshot["items"])
    assert snapshot["cases"][0]["contract_item_ids"] == [
        "item-case-a"
    ]
    assert original["cases"][0]["status"] == "pending"
    assert saves
