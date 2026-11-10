from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.acoustic import acceptance_batch, service
from server.app import app


def _report(
    batch_id: str,
    created_at: str,
    *,
    algorithm_cluster_id: str = "vcp_default",
    status: str = "completed",
) -> dict:
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "dataset_id": "d" * 32,
        "dataset_name": "历史测试集",
        "algorithm_cluster_id": algorithm_cluster_id,
        "status": status,
        "progress": {
            "completed_cases": 106,
            "total_cases": 106,
            "completed_items": 336,
            "total_items": 336,
        },
        "summary": {
            "total": 336,
            "computed": 335,
            "pass": 0,
            "fail": 0,
            "review_required": 335,
            "not_measured": 1,
            "invalid": 0,
        },
        "created_at": created_at,
        "items": [{"large": "must not be returned"}],
        "cases": [{"large": "must not be returned"}],
        "configuration": {"large": "must not be returned"},
    }


def _write_report(
    root: Path, report: dict | list, directory_id: str | None = None
) -> Path:
    batch_id = (
        directory_id
        if directory_id is not None
        else report["batch_id"]
        if isinstance(report, dict)
        else "f" * 32
    )
    path = root / batch_id / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def history_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    root = service.BUILD / "acceptance-batches"
    root.mkdir(parents=True)
    return root


def test_list_history_sorts_limits_filters_and_omits_large_payloads(history_root):
    a_id, b_id, c_id, d_id = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)
    reports = {
        a_id: _report(
            a_id,
            "2026-11-04T12:00:00+00:00",
            algorithm_cluster_id="other",
        ),
        b_id: _report(
            b_id,
            "2026-11-04T11:00:00+00:00",
            status="failed",
        ),
        c_id: _report(c_id, "2026-11-04T10:00:00+00:00"),
        d_id: _report(d_id, "2026-11-04T10:00:00+00:00"),
    }
    reports[b_id].update(
        finished_at="2026-11-04T11:05:00+00:00",
        updated_at="2026-11-04T11:04:00+00:00",
        error="历史失败",
    )
    for report in reports.values():
        _write_report(history_root, report)

    all_rows = acceptance_batch.list_batch_summaries(10)
    assert [row["batch_id"] for row in all_rows] == [
        a_id,
        b_id,
        d_id,
        c_id,
    ]
    rows = acceptance_batch.list_batch_summaries(2, "vcp_default")
    assert [row["batch_id"] for row in rows] == [b_id, d_id]
    assert set(rows[0]) == {
        "batch_id",
        "dataset_id",
        "dataset_name",
        "algorithm_cluster_id",
        "status",
        "progress",
        "summary",
        "created_at",
        "finished_at",
        "updated_at",
        "error",
    }
    assert all(
        "items" not in row and "cases" not in row and "configuration" not in row
        for row in rows
    )

    response = TestClient(app).get(
        "/api/alignment/acceptance/batches",
        params={"limit": 2, "algorithm_cluster_id": "vcp_default"},
    )
    assert response.status_code == 200
    assert response.json() == rows


def test_list_history_empty_store_is_success(history_root):
    assert acceptance_batch.list_batch_summaries() == []
    response = TestClient(app).get("/api/alignment/acceptance/batches")
    assert response.status_code == 200
    assert response.json() == []


def test_list_history_skips_corrupt_invalid_and_symlinked_reports(
    history_root, tmp_path
):
    valid_id = "1" * 32
    valid_path = _write_report(
        history_root,
        _report(valid_id, "2026-11-04T12:00:00Z"),
    )
    original = valid_path.read_bytes()

    corrupt_id = "2" * 32
    corrupt = history_root / corrupt_id / "report.json"
    corrupt.parent.mkdir()
    corrupt.write_text("{not-json", encoding="utf-8")

    not_object_id = "3" * 32
    not_object = history_root / not_object_id / "report.json"
    not_object.parent.mkdir()
    not_object.write_text("[]", encoding="utf-8")

    missing_summary = _report("4" * 32, "2026-11-04T11:00:00Z")
    missing_summary.pop("summary")
    _write_report(history_root, missing_summary)

    mismatched = _report("5" * 32, "2026-11-04T10:00:00Z")
    mismatched["batch_id"] = "6" * 32
    _write_report(history_root, mismatched, directory_id="5" * 32)

    invalid_time = _report("7" * 32, "not-a-time")
    _write_report(history_root, invalid_time)

    non_hex = history_root / "not-a-batch"
    non_hex.mkdir()
    (non_hex / "report.json").write_text(
        json.dumps(_report("9" * 32, "2026-11-04T09:00:00Z")),
        encoding="utf-8",
    )

    target = tmp_path / "linked-target"
    target.mkdir()
    target_report = _report("8" * 32, "2026-11-04T13:00:00Z")
    (target / "report.json").write_text(
        json.dumps(target_report),
        encoding="utf-8",
    )
    (history_root / ("8" * 32)).symlink_to(target, target_is_directory=True)

    rows = acceptance_batch.list_batch_summaries()
    assert [row["batch_id"] for row in rows] == [valid_id]
    assert valid_path.read_bytes() == original


def test_list_history_enforces_counter_contract_and_projects_only_counters(
    history_root,
):
    valid_id = "1" * 32
    valid = _report(valid_id, "2026-11-04T12:00:00Z")
    valid["progress"]["nested"] = {"unexpected": [1, 2, 3]}
    valid["summary"]["nested"] = {"unexpected": [1, 2, 3]}
    _write_report(history_root, valid)

    invalid_reports: list[dict] = []

    invalid_status = _report("2" * 32, "2026-11-04T11:00:00Z")
    invalid_status["status"] = "bogus"
    invalid_reports.append(invalid_status)

    missing_counter = _report("3" * 32, "2026-11-04T10:59:00Z")
    missing_counter["summary"].pop("pass")
    invalid_reports.append(missing_counter)

    boolean_counter = _report("4" * 32, "2026-11-04T10:58:00Z")
    boolean_counter["progress"]["completed_cases"] = True
    invalid_reports.append(boolean_counter)

    nan_counter = _report("5" * 32, "2026-11-04T10:57:00Z")
    nan_counter["summary"]["computed"] = float("nan")
    invalid_reports.append(nan_counter)

    inconsistent_progress = _report("6" * 32, "2026-11-04T10:56:00Z")
    inconsistent_progress["progress"]["completed_cases"] += 1
    invalid_reports.append(inconsistent_progress)

    inconsistent_summary = _report("7" * 32, "2026-11-04T10:55:00Z")
    inconsistent_summary["summary"]["computed"] = (
        inconsistent_summary["summary"]["total"] + 1
    )
    invalid_reports.append(inconsistent_summary)

    mismatched_total = _report("8" * 32, "2026-11-04T10:54:00Z")
    mismatched_total["summary"]["total"] -= 1
    invalid_reports.append(mismatched_total)

    invalid_dataset = _report("9" * 32, "2026-11-04T10:53:00Z")
    invalid_dataset["dataset_id"] = "not-a-dataset-id"
    invalid_reports.append(invalid_dataset)

    for report in invalid_reports:
        _write_report(history_root, report)

    rows = acceptance_batch.list_batch_summaries(100)
    assert [row["batch_id"] for row in rows] == [valid_id]
    assert set(rows[0]["progress"]) == {
        "completed_cases",
        "total_cases",
        "completed_items",
        "total_items",
    }
    assert set(rows[0]["summary"]) == {
        "total",
        "computed",
        "pass",
        "fail",
        "review_required",
        "not_measured",
        "invalid",
    }
    assert rows[0]["progress"] == {
        field: valid["progress"][field]
        for field in (
            "completed_cases",
            "total_cases",
            "completed_items",
            "total_items",
        )
    }
    assert rows[0]["summary"] == {
        field: valid["summary"][field]
        for field in (
            "total",
            "computed",
            "pass",
            "fail",
            "review_required",
            "not_measured",
            "invalid",
        )
    }

    response = TestClient(app).get("/api/alignment/acceptance/batches")
    assert response.status_code == 200
    assert response.json() == rows


def test_history_api_validates_limit_and_returns_503_for_unreadable_store(
    history_root, tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="1 到 100"):
        acceptance_batch.list_batch_summaries(0)
    client = TestClient(app)
    assert client.get(
        "/api/alignment/acceptance/batches", params={"limit": 0}
    ).status_code == 422
    assert client.get(
        "/api/alignment/acceptance/batches", params={"limit": 101}
    ).status_code == 422
    assert client.get(
        "/api/alignment/acceptance/batches", params={"limit": "invalid"}
    ).status_code == 422

    broken_build = tmp_path / "broken-build"
    broken_build.mkdir()
    (broken_build / "acceptance-batches").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(service, "BUILD", broken_build)
    response = client.get("/api/alignment/acceptance/batches")
    assert response.status_code == 503
    assert response.json()["detail"] == "批次历史目录不可访问"


def test_history_list_route_preserves_post_and_id_get_behavior(history_root):
    client = TestClient(app)
    assert client.post("/api/alignment/acceptance/batches", json={}).status_code == 422
    assert client.get(
        f"/api/alignment/acceptance/batches/{'0' * 32}"
    ).status_code == 404
