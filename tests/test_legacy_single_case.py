from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from server.acoustic import acceptance_batch as engine, service
from server.acoustic.acceptance_dataset import get_dataset, import_dataset
from server.acoustic.acceptance_scope import select_batch_scope
from server.app import app
from test_basic_uplink import _complete_archive, _controlled_outputs, _controlled_rows


class CaptureExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))


@pytest.fixture
def single_store(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(engine, "_ACTIVE", set())
    monkeypatch.setattr(engine, "_LEGACY_ACTIVE", False)
    executor = CaptureExecutor()
    monkeypatch.setattr(engine, "_EXECUTOR", executor)
    imported = import_dataset(_complete_archive())
    return imported, executor, TestClient(app)


@pytest.mark.parametrize("scope", ["basic_uplink", "basic_downlink"])
def test_catalog_and_single_case_only_execute_selected_group(single_store, monkeypatch, scope):
    imported, executor, client = single_store
    mode = "WB-wired"
    catalog = client.get(
        f"/api/alignment/acceptance/datasets/{imported['dataset_id']}/cases",
        params={"test_scope": scope, "mode": mode},
    )
    assert catalog.status_code == 200, catalog.text
    value = catalog.json()
    assert value["schema_version"] == "acceptance-case-catalog-v1"
    assert value["observation_count"] == (8 if scope == "basic_uplink" else 9)
    assert "visuals" in value
    assert set(value["visuals"]["cases"]) == {case["case_id"] for case in value["cases"]}
    assert sum(len(case["item_ids"]) for case in value["cases"]) == value["observation_count"]
    selected = value["cases"][0]
    url = f"/api/alignment/acceptance/{scope.replace('_', '-')}/batches"
    response = client.post(url, json={
        "dataset_id": imported["dataset_id"], "algorithm_cluster_id": "vcp_default",
        "mode": mode, "case_id": selected["case_id"],
    })
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["execution_scope"] == "single_case"
    assert queued["selected_case_id"] == selected["case_id"]
    assert queued["dataset_case_count"] == value["case_count"]
    assert queued["dataset_observation_count"] == value["observation_count"]
    assert queued["preview_condition_group_count"] == value["case_count"]
    assert queued["progress"]["total_cases"] == 1
    assert queued["progress"]["total_items"] == len(selected["item_ids"])
    assert {row["item_id"] for row in queued["items"]} == set(selected["item_ids"])
    assert len(executor.calls) == 1
    calls = []

    def run_cli(case, paths, destination, cluster):
        calls.append(case["id"])
        outputs, execution = _controlled_outputs(
            case, destination, cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"], paths=paths,
        )
        execution["output_alignment"] = {kind: {"analysis_filename": output.name} for kind, output in outputs.items()}
        if scope == "basic_downlink":
            # Native RX already uses the analysis timeline and has no crop
            # metadata. Its well-defined filename remains hash-verified.
            output = destination / "rx.rx-spk.wav"
            output.write_bytes(outputs["tx"].read_bytes())
            outputs = {"rx": output}
            execution["output_sha256"] = {"rx": engine._sha(output)}
            execution["output_alignment"] = {}
        return outputs, execution

    monkeypatch.setattr(engine, "_run_cli", run_cli)
    def controlled_calculate(case, paths, outputs, execution, **kw):
        by_id = {row["item_id"]: row for row in queued["items"]}
        return [{**deepcopy(by_id[row["item_id"]]), **row}
                for row in _controlled_rows(case, execution, report_id=kw["report_id"])]
    monkeypatch.setattr(engine, "calculate_case_items", controlled_calculate)
    finished = engine.run_batch(queued["batch_id"])
    assert finished["status"] == "completed", finished.get("error")
    assert calls == [selected["case_id"]]
    assert finished["progress"]["completed_cases"] == 1
    assert finished["mode_reports"] == []
    engine._validate_single_snapshot(finished)
    tampered = deepcopy(finished)
    tampered["items"][0].update(status="pass", value=123.456)
    tampered["summary"] = engine._summary(tampered["items"])
    with pytest.raises(ValueError, match="报告身份"):
        engine._validate_single_snapshot(tampered)
    fetched = client.get(f"/api/alignment/acceptance/batches/{queued['batch_id']}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["selected_case_id"] == selected["case_id"]
    assert fetched.json()["status"] == "completed"
    history = client.get(url).json()
    assert history[0]["execution_scope"] == "single_case"
    snapshots = engine._directory(queued["batch_id"]) / "snapshots"
    assert len(list(snapshots.glob("*.json"))) >= 3
    assert not (engine._directory(queued["batch_id"]) / "report.json").exists()
    assert not any(path.name.endswith(".tmp") for path in snapshots.iterdir())


@pytest.mark.parametrize("case_id", ["", "../escape", "unknown-case", 10, "WB-wrong-mode"])
def test_invalid_single_case_rejected_before_enqueue(single_store, case_id):
    imported, executor, client = single_store
    response = client.post("/api/alignment/acceptance/basic-uplink/batches", json={
        "dataset_id": imported["dataset_id"], "algorithm_cluster_id": "vcp_default",
        "mode": "WB-wired", "case_id": case_id,
    })
    assert response.status_code == 422
    assert executor.calls == []
    assert not (service.BUILD / "acceptance-batches").exists()


def test_single_snapshot_forgery_is_rejected(single_store):
    imported, _, _ = single_store
    dataset = get_dataset(imported["dataset_id"])
    rows, cases = select_batch_scope(dataset["manifest"], test_scope="basic_uplink", mode="WB-wired")
    queued = engine.start_batch(imported["dataset_id"], "vcp_default", test_scope="basic_uplink",
                                mode="WB-wired", case_id=cases[0]["id"])
    engine._validate_single_snapshot(queued)
    for patch in (
        {"dataset_observation_count": 336},
        {"selected_case_id": None},
        {"selected_case_id": cases[-1]["id"]},
        {"status": "completed"},
        {"observation_count": len(rows)},
    ):
        bad = {**deepcopy(queued), **patch}
        with pytest.raises(ValueError):
            engine._validate_single_snapshot(bad)
    bad = deepcopy(queued)
    bad.pop("execution_scope")
    with pytest.raises(ValueError):
        engine._validate_single_snapshot(bad)


def test_full_request_remains_unselected(single_store):
    imported, _, client = single_store
    response = client.post("/api/alignment/acceptance/basic-uplink/batches", json={
        "dataset_id": imported["dataset_id"], "algorithm_cluster_id": "vcp_default", "mode": "WB-wired",
    })
    assert response.status_code == 202, response.text
    snapshot = response.json()
    assert "execution_scope" not in snapshot
    assert snapshot["progress"]["total_cases"] == 2
    assert snapshot["progress"]["total_items"] == 8


def test_single_corrupt_latest_falls_back_without_writing(single_store):
    imported, _, _ = single_store
    dataset = get_dataset(imported["dataset_id"])
    _, cases = select_batch_scope(dataset["manifest"], test_scope="basic_uplink", mode="WB-wired")
    queued = engine.start_batch(imported["dataset_id"], "vcp_default", test_scope="basic_uplink",
                                mode="WB-wired", case_id=cases[0]["id"])
    directory = engine._directory(queued["batch_id"]) / "snapshots"
    broken = directory / "00000002.json"
    broken.write_text('{"status":', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    engine._ACTIVE.discard(queued["batch_id"])
    fetched = engine.get_batch(queued["batch_id"])
    history = engine.list_batch_summaries(test_scope="basic_uplink")
    assert fetched["status"] == history[0]["status"] == "failed"
    assert fetched["summary"] == history[0]["summary"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_complete_without_execution_evidence_is_rejected(single_store):
    imported, _, _ = single_store
    dataset = get_dataset(imported["dataset_id"])
    _, cases = select_batch_scope(dataset["manifest"], test_scope="basic_uplink", mode="WB-wired")
    bad = engine.start_batch(imported["dataset_id"], "vcp_default", test_scope="basic_uplink",
                             mode="WB-wired", case_id=cases[0]["id"])
    bad = deepcopy(bad)
    bad["status"] = "completed"
    bad["cases"] = [{"case_id": cases[0]["id"], "status": "completed"}]
    for item in bad["items"]:
        item.update(status="not_measured", value=None)
    bad["summary"] = engine._summary(bad["items"])
    bad["progress"].update(completed_cases=1, completed_items=len(bad["items"]))
    with pytest.raises(ValueError, match="报告证据"):
        engine._validate_single_snapshot(bad)


def test_full_scope_single_cannot_appear_as_a_336_item_batch(single_store):
    imported, _, _ = single_store
    dataset = get_dataset(imported["dataset_id"])
    selected_case = dataset["manifest"]["cases"][0]
    queued = engine.start_batch(imported["dataset_id"], "vcp_default", case_id=selected_case["id"])
    assert queued["execution_scope"] == "single_case"
    assert queued["dataset_observation_count"] == 336
    assert queued["progress"]["total_cases"] == 1
    assert queued["mode_reports"] == []
    assert engine.list_batch_summaries(test_scope="full") == []
