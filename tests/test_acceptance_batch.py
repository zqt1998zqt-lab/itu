from __future__ import annotations

import copy
import errno
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.acoustic import acceptance_batch, acceptance_linearity, service
from server.acoustic.acceptance_contract import (
    contract_items,
    required_conditions,
)
from server.acoustic.acceptance_dataset import import_dataset, template_manifest
from server.app import app


SIGNALS = ("microphone", "reference", "far_end", "noise")


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


def _wav(values: np.ndarray, rate: int = 16_000) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _small_dataset_archive() -> bytes:
    manifest = template_manifest()
    shared = {
        "audio/shared/microphone.wav": _wav(
            np.zeros((16_000, 2), dtype=np.float64)
        ),
        "audio/shared/reference.wav": _wav(np.zeros(16_000)),
        "audio/shared/far_end.wav": _wav(np.zeros(16_000)),
        "audio/shared/noise.wav": _wav(np.zeros(24_000)),
    }
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav" for role in SIGNALS
        }
        case["analysis_start_s"] = 0
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for name, payload in shared.items():
            archive.writestr(name, payload)
    return stream.getvalue()


@pytest.fixture
def batch_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(
        acceptance_linearity,
        "finish_linearity",
        lambda snapshot, dataset, destination: None,
    )
    imported = import_dataset(_small_dataset_archive())
    assert imported["case_count"] == 106
    assert imported["item_count"] == 336
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    return imported, executor


def _queue_batch(dataset_id: str, executor: _CapturingExecutor) -> dict:
    queued = acceptance_batch.start_batch(dataset_id, "vcp_default")
    assert queued["status"] == "queued"
    assert queued["progress"]["total_cases"] == 106
    assert queued["progress"]["total_items"] == 336
    assert len({row["item_id"] for row in queued["items"]}) == 336
    assert len(queued["cli_sha256"]) == 64
    assert len(queued["catalog_sha256"]) == 64
    assert queued["batch_id"] in acceptance_batch._ACTIVE
    assert len(executor.submissions) == 1
    function, args = executor.submissions[-1]
    assert function is acceptance_batch.run_batch
    assert args == (queued["batch_id"],)
    return queued


def _controlled_outputs(
    case: dict,
    destination: Path,
    *,
    cli_sha256: str,
    catalog_sha256: str,
    paths: dict[str, Path],
) -> tuple[dict, dict]:
    direction = case["conditions"]["direction"]
    kinds = (
        ("rx",)
        if direction == "receive"
        else ("tx", "rx")
        if direction == "round_trip"
        else ("tx",)
    )
    outputs = {kind: destination / f"{kind}.controlled.wav" for kind in kinds}
    for path in outputs.values():
        sf.write(path, np.zeros(160), 16000, subtype="PCM_16")
    execution = {
        "source": "controlled-executor",
        "cli_sha256": cli_sha256,
        "catalog_sha256": catalog_sha256,
        "input_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in paths.items()
        },
        "output_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in outputs.items()
        },
    }
    return outputs, execution


def _controlled_rows(
    case: dict,
    execution: dict,
    *,
    report_id: str,
    null_item_id: str | None = None,
) -> list[dict]:
    rows = []
    for item_id in case["item_ids"]:
        value = None if item_id == null_item_id else 1.0
        rows.append(
            {
                "item_id": item_id,
                "value": value,
                "status": "review_required" if value is not None else "not_measured",
                "reason": "controlled engineering result",
                "standard_compliance": False,
                "evidence_binding": {
                    **case["conditions"],
                    "case_id": case["id"],
                    "report_id": report_id,
                    "batch_id": execution["batch_id"],
                    "dataset_id": execution["dataset_id"],
                    "input_sha256": execution["input_sha256"],
                    "output_sha256": execution["output_sha256"],
                },
            }
        )
    return rows


def test_full_batch_keeps_case_report_and_contract_identity(
    batch_environment, monkeypatch
):
    imported, executor = batch_environment
    queued = _queue_batch(imported["dataset_id"], executor)
    dataset = acceptance_batch.get_dataset(imported["dataset_id"])
    assert dataset is not None
    expected_case = {
        item_id: case["id"]
        for case in dataset["manifest"]["cases"]
        for item_id in case["item_ids"]
    }
    null_item_id = next(iter(expected_case))
    run_calls = []
    calculate_calls = []

    def controlled_run_cli(case, paths, destination, cluster):
        assert cluster == "vcp_default"
        assert set(paths) == set(SIGNALS)
        assert all(path.is_file() for path in paths.values())
        outputs, execution = _controlled_outputs(
            case,
            destination,
            cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"],
            paths=paths,
        )
        run_calls.append(
            {
                "case": copy.deepcopy(case),
                "paths": dict(paths),
                "destination": destination,
                "outputs": dict(outputs),
            }
        )
        return outputs, execution

    def controlled_calculate(
        case, input_paths, outputs, execution, *, report_id
    ):
        calculate_calls.append(
            {
                "case": copy.deepcopy(case),
                "report_id": report_id,
                "execution": copy.deepcopy(execution),
                "outputs": dict(outputs),
            }
        )
        return _controlled_rows(
            case,
            execution,
            report_id=report_id,
            null_item_id=null_item_id,
        )

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert result["status"] == "completed"
    assert result["batch_id"] == queued["batch_id"]
    assert result["dataset_id"] == imported["dataset_id"]
    assert result["progress"] == {
        "completed_cases": 106,
        "total_cases": 106,
        "completed_items": 336,
        "total_items": 336,
    }
    assert len(run_calls) == len(calculate_calls) == 106
    assert {call["case"]["id"] for call in run_calls} == set(expected_case.values())
    assert [call["report_id"] for call in calculate_calls]
    assert len({call["report_id"] for call in calculate_calls}) == 106

    call_by_case = {call["case"]["id"]: call for call in calculate_calls}
    assert set(call_by_case) == set(expected_case.values())
    for call in calculate_calls:
        case = call["case"]
        execution = call["execution"]
        assert execution["batch_id"] == queued["batch_id"]
        assert execution["dataset_id"] == imported["dataset_id"]
        assert set(call["outputs"]) <= {"tx", "rx"}
        assert all(
            path.parent == Path(run_calls[calculate_calls.index(call)]["destination"])
            for path in call["outputs"].values()
        )
        report_path = (
            Path(service.BUILD)
            / "acceptance-batches"
            / queued["batch_id"]
            / "cases"
            / case["id"]
            / "assessment.json"
        )
        report = json.loads(report_path.read_text())
        assert report["case_id"] == case["id"]
        assert report["report_id"] == call["report_id"]
        assert {row["item_id"] for row in report["items"]} == set(case["item_ids"])
        assert len(report["items"]) == len(case["item_ids"])

    rows_by_id = {row["item_id"]: row for row in result["items"]}
    assert set(rows_by_id) == {row["id"] for row in contract_items()}
    assert len(rows_by_id) == 336
    report_for_case = {
        case_id: call_by_case[case_id]["report_id"] for case_id in call_by_case
    }
    for item_id, row in rows_by_id.items():
        binding = row["evidence_binding"]
        case_id = expected_case[item_id]
        assert binding["case_id"] == case_id
        assert binding["report_id"] == report_for_case[case_id]
        assert binding["batch_id"] == queued["batch_id"]
        assert binding["dataset_id"] == imported["dataset_id"]
        assert {
            key: binding[key] for key in required_conditions(rows_by_id[item_id])
        } == required_conditions(rows_by_id[item_id])

    assert result["summary"]["total"] == 336
    assert result["summary"]["computed"] == 335
    assert result["summary"]["review_required"] == 335
    assert result["summary"]["not_measured"] == 1
    assert result["summary"]["invalid"] == 0
    assert rows_by_id[null_item_id]["value"] is None
    assert rows_by_id[null_item_id]["status"] == "not_measured"
    assert result["standard_compliance"] is False
    assert result["formal_test_status"] == "not_measured"
    assert queued["batch_id"] not in acceptance_batch._ACTIVE


def test_failed_case_invalidates_only_its_items_and_fails_batch(
    batch_environment, monkeypatch
):
    imported, executor = batch_environment
    queued = _queue_batch(imported["dataset_id"], executor)
    dataset = acceptance_batch.get_dataset(imported["dataset_id"])
    assert dataset is not None
    failed_case = dataset["manifest"]["cases"][0]
    failed_ids = set(failed_case["item_ids"])
    calculate_calls = []

    def controlled_run_cli(case, paths, destination, cluster):
        if case["id"] == failed_case["id"]:
            raise RuntimeError("controlled case failure")
        outputs, execution = _controlled_outputs(
            case,
            destination,
            cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"],
            paths=paths,
        )
        return outputs, execution

    def controlled_calculate(
        case, input_paths, outputs, execution, *, report_id
    ):
        calculate_calls.append(case["id"])
        return _controlled_rows(
            case,
            execution,
            report_id=report_id,
        )

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    rows_by_id = {row["item_id"]: row for row in result["items"]}
    failed_rows = [rows_by_id[item_id] for item_id in failed_ids]
    passed_rows = [
        row for item_id, row in rows_by_id.items() if item_id not in failed_ids
    ]
    assert len(calculate_calls) == 105
    assert all(row["status"] == "invalid" for row in failed_rows)
    assert all(row["value"] is None for row in failed_rows)
    assert all("controlled case failure" in row["reason"] for row in failed_rows)
    failed_report_ids = {row.get("report_id") for row in failed_rows}
    assert len(failed_report_ids) == 1
    failed_report_id = failed_report_ids.pop()
    assert isinstance(failed_report_id, str) and len(failed_report_id) == 32
    assert all(
        row["evidence_binding"]["case_id"] == failed_case["id"]
        and row["evidence_binding"]["batch_id"] == queued["batch_id"]
        and row["evidence_binding"]["dataset_id"] == imported["dataset_id"]
        and row["evidence_binding"]["report_id"] == failed_report_id
        for row in failed_rows
    )
    assert all(row["status"] == "review_required" for row in passed_rows)
    assert result["summary"]["invalid"] == len(failed_ids)
    assert result["summary"]["computed"] == 336 - len(failed_ids)
    assert result["status"] == "failed"
    assert result["standard_compliance"] is False
    assert result["formal_test_status"] == "not_measured"


def test_changed_input_hash_is_rejected_before_executor(
    batch_environment, monkeypatch
):
    imported, executor = batch_environment
    queued = _queue_batch(imported["dataset_id"], executor)
    microphone = acceptance_batch.dataset_asset(
        imported["dataset_id"],
        "audio/shared/microphone.wav",
    )
    microphone.write_bytes(_wav(np.ones((16_000, 2), dtype=np.float64) * 0.01))
    run_calls = []
    calculate_calls = []

    def must_not_run(*args, **kwargs):
        run_calls.append(args)
        raise AssertionError("changed input must be rejected before VCP")

    def must_not_calculate(*args, **kwargs):
        calculate_calls.append(args)
        raise AssertionError("changed input must not reach calculation")

    monkeypatch.setattr(acceptance_batch, "_run_cli", must_not_run)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        must_not_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert run_calls == []
    assert calculate_calls == []
    assert len(result["items"]) == 336
    assert all(row["status"] == "invalid" for row in result["items"])
    assert all(row["value"] is None for row in result["items"])
    assert all("文件身份已改变" in row["reason"] for row in result["items"])
    assert result["summary"]["computed"] == 0
    assert result["status"] == "failed"


def test_concurrent_batch_is_rejected(batch_environment):
    imported, executor = batch_environment
    first = _queue_batch(imported["dataset_id"], executor)
    with pytest.raises(ValueError, match="已有|正在执行|并发"):
        acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    assert first["batch_id"] in acceptance_batch._ACTIVE
    assert len(executor.submissions) == 1


def test_enospc_stops_without_writing_failure_reports(batch_environment, monkeypatch):
    imported, executor = batch_environment
    queued = _queue_batch(imported["dataset_id"], executor)
    calls = []

    def full_disk(*args):
        calls.append(args[0]["id"])
        raise OSError(errno.ENOSPC, "disk full; preserve evidence")

    monkeypatch.setattr(acceptance_batch, "_run_cli", full_disk)
    with pytest.raises(OSError) as error:
        acceptance_batch.run_batch(queued["batch_id"])
    assert error.value.errno == errno.ENOSPC
    assert len(calls) == 1
    path = service.BUILD / "acceptance-batches" / queued["batch_id"] / "report.json"
    saved = json.loads(path.read_text())
    assert saved["status"] == "running"
    assert saved["cases"] == []
    assert saved["progress"]["completed_cases"] == 0


def test_single_scene_api_rejects_active_batch(batch_environment):
    imported, executor = batch_environment
    _queue_batch(imported["dataset_id"], executor)
    response = TestClient(app).post("/api/alignment/run", json={"scope": "contract"})
    assert response.status_code == 409


def test_batch_rejects_active_single_scene(batch_environment, monkeypatch):
    imported, executor = batch_environment
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", True)
    with pytest.raises(ValueError, match="正在执行"):
        acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    assert not executor.submissions


def test_queue_write_failure_releases_activity_without_cleanup(batch_environment, monkeypatch):
    imported, executor = batch_environment
    def cannot_save(snapshot):
        raise OSError(errno.ENOSPC, "disk full")
    monkeypatch.setattr(acceptance_batch, "_save", cannot_save)
    with pytest.raises(OSError):
        acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    assert not acceptance_batch._ACTIVE
    assert not executor.submissions


def test_initial_report_read_failure_releases_activity(monkeypatch):
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    def cannot_read(identifier):
        raise OSError("report unreadable")
    monkeypatch.setattr(acceptance_batch, "get_batch", cannot_read)
    with pytest.raises(OSError):
        acceptance_batch.run_batch("a" * 32)
    assert not acceptance_batch._ACTIVE


def test_restart_does_not_create_reports_for_unexecuted_modes(batch_environment):
    imported, executor = batch_environment
    queued = _queue_batch(imported["dataset_id"], executor)
    acceptance_batch._ACTIVE.discard(queued["batch_id"])
    result = acceptance_batch.get_batch(queued["batch_id"])
    assert result["status"] == "failed"
    assert result["progress"]["completed_cases"] == 0
    assert result["progress"]["completed_items"] == 0
    assert result["mode_reports"] == []
    assert result["summary"]["computed"] == 0


def test_acceptance_api_upload_template_and_missing_report(
    batch_environment, monkeypatch
):
    imported, _ = batch_environment
    client = TestClient(app)

    template_response = client.get("/api/alignment/acceptance/template")
    assert template_response.status_code == 200
    assert template_response.headers["content-disposition"] == (
        'attachment; filename="manifest.json"'
    )
    template = template_response.json()
    assert len(template["cases"]) == 106
    assert sum(len(case["item_ids"]) for case in template["cases"]) == 336

    upload = client.post(
        "/api/alignment/acceptance/datasets",
        files={
            "archive": (
                "acceptance.zip",
                _small_dataset_archive(),
                "application/zip",
            )
        },
    )
    assert upload.status_code == 201, upload.text
    uploaded = upload.json()
    assert uploaded["case_count"] == 106
    assert uploaded["item_count"] == 336
    assert uploaded["dataset_id"] != imported["dataset_id"]

    all_datasets = client.get("/api/alignment/acceptance/datasets")
    assert all_datasets.status_code == 200
    assert isinstance(all_datasets.json(), list)
    assert len(all_datasets.json()) >= 2

    scene_view = client.get(
        "/api/alignment/acceptance/datasets",
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": "sce1",
        },
    )
    assert scene_view.status_code == 200, scene_view.text
    payload = scene_view.json()
    assert payload["test_scope"] == "background_noise"
    assert payload["mode"] == "WB-wired"
    assert payload["scene"] == "sce1"
    assert payload["scene_item_count"] == 5
    assert payload["scene_case_count"] >= 1
    assert payload["total_dataset_count"] == len(all_datasets.json())
    assert len(payload["datasets"]) == 1
    assert payload["datasets"][0]["dataset_id"] in {
        row["dataset_id"] for row in all_datasets.json()
    }

    invalid_scene = client.get(
        "/api/alignment/acceptance/datasets",
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": "unknown",
        },
    )
    assert invalid_scene.status_code == 422

    bad_mode = client.get(
        "/api/alignment/acceptance/datasets",
        params={
            "scope": "background_noise",
            "mode": "not-a-mode",
            "scene": "sce1",
        },
    )
    assert bad_mode.status_code == 422

    missing = client.get(
        f"/api/alignment/acceptance/batches/{'0' * 32}/report"
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "批次不存在"
