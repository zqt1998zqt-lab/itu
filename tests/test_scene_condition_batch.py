from __future__ import annotations

import base64
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.acoustic import (
    acceptance_batch,
    acceptance_scene_condition_api,
    scene_condition_batch,
    scene_condition_batch_api,
    service,
)
from server.acoustic.acceptance_scope import background_noise_scene_items
from server.acoustic.acceptance_scene_conditions import (
    CONDITIONS,
    CONTRACT_ITEM_COUNT,
    MODE,
    ROLES,
    SCENE,
)
from server.app import app as product_app


DATASET_ID = "c" * 32
DATASET_NAME = "synthetic-scene-condition-batch-v1"
ALGORITHM_CLUSTER = "vcp_default"
CATALOG_SHA256 = "e" * 64
BATCH_PREFIX = "/api/alignment/acceptance/scene-condition-batches"


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


@dataclass
class _SceneBatchEnvironment:
    build: Path
    dataset: dict
    contract_by_id: dict[str, dict]
    executor: _CapturingExecutor
    cli: Path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_input_wav(path: Path, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 1_600
    if role == "microphone":
        payload = np.zeros((frames, 2), dtype=np.float32)
    else:
        payload = np.zeros(frames, dtype=np.float32)
    sf.write(path, payload, 16_000, subtype="PCM_16")


def _controlled_visual(path: Path, source_sha256: str) -> dict:
    assert path.is_file()
    return {
        "source_sha256": source_sha256,
        "duration_s": 0.1,
        "sample_rate": 16_000,
        "channels": 1,
        "peak": 0.0,
        "wave_scale": 1.0,
        "wave_min_i16": base64.b64encode(b"\0" * (720 * 2)).decode(),
        "wave_max_i16": base64.b64encode(b"\0" * (720 * 2)).decode(),
        "mel_i16": base64.b64encode(b"\0" * (40 * 240 * 2)).decode(),
        "mel_ceiling_db": -120.0,
    }


def _synthetic_dataset(build: Path) -> dict:
    root = build / "acoustic-scene-condition-datasets" / DATASET_ID
    signals: dict[str, str] = {}
    signals_sha256: dict[str, str] = {}
    for role in ROLES:
        relative = f"audio/{role}.wav"
        path = root / relative
        _write_input_wav(path, role)
        signals[role] = relative
        signals_sha256[role] = _sha256_file(path)

    contract_rows = background_noise_scene_items(MODE, SCENE)
    assert len(contract_rows) == CONTRACT_ITEM_COUNT == 5
    contract_by_id = {row["id"]: row for row in contract_rows}
    speech_item_ids = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    ]
    cng_item_ids = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"cng_level", "cng_spectrum"}
    ]
    assert len(speech_item_ids) == 3
    assert len(cng_item_ids) == 2

    cases = []
    for index in range(20):
        kind = "speech" if index < 10 else "cng"
        analysis_start_s = 0.5 if index % 2 == 0 else 4.0
        beam_delays_ms = [0.0625, float(index % 4) * 0.25]
        cases.append(
            {
                "case_id": f"synthetic-{kind}-{index + 1:02d}",
                "conditions": copy.deepcopy(CONDITIONS),
                "families": [
                    contract_by_id[item_id]["family"]
                    for item_id in (
                        speech_item_ids if kind == "speech" else cng_item_ids
                    )
                ],
                "contract_item_ids": list(
                    speech_item_ids if kind == "speech" else cng_item_ids
                ),
                "signals": copy.deepcopy(signals),
                "signals_sha256": copy.deepcopy(signals_sha256),
                "provenance": {
                    "analysis_start_s": analysis_start_s,
                    "render_metadata": {
                        "kind": kind,
                        "beam_delays_ms": beam_delays_ms,
                    },
                },
            }
        )

    return {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "root": root,
        "cases": cases,
        "dataset_manifest_sha256": "1" * 64,
        "dataset_assets_sha256": "2" * 64,
        "contract_item_count": CONTRACT_ITEM_COUNT,
    }


def _validated_loader(dataset: dict, contract_by_id: dict[str, dict]):
    def load(dataset_id: str):
        if dataset_id != DATASET_ID:
            raise ValueError("场景条件测试集不可用: unknown synthetic dataset")
        return copy.deepcopy(dataset), copy.deepcopy(contract_by_id)

    return load


@pytest.fixture
def scene_batch_environment(tmp_path, monkeypatch) -> _SceneBatchEnvironment:
    build = tmp_path / "build"
    dataset = _synthetic_dataset(build)
    contract_by_id = {
        row["id"]: row for row in background_noise_scene_items(MODE, SCENE)
    }
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        _validated_loader(dataset, contract_by_id),
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)

    cli = tmp_path / "vcp-cli"
    cli.write_bytes(b"fake-vcp-cli")

    import server.alignment_runner as alignment_runner

    def resolve_cluster(cluster_id: str):
        profile = (
            "vcp_default"
            if cluster_id.startswith("vcp")
            else "agc_default"
        )
        return (
            {"profile": profile},
            {},
            {"algorithm_cluster_id": cluster_id},
            CATALOG_SHA256,
        )

    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        resolve_cluster,
    )
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_visual_track",
        _controlled_visual,
    )
    return _SceneBatchEnvironment(
        build=build,
        dataset=dataset,
        contract_by_id=contract_by_id,
        executor=executor,
        cli=cli,
    )


def _queue_batch(environment: _SceneBatchEnvironment) -> dict:
    queued = scene_condition_batch.start_batch(
        DATASET_ID,
        ALGORITHM_CLUSTER,
    )
    assert queued["status"] == "queued"
    assert queued["batch_id"] in acceptance_batch._ACTIVE
    assert len(environment.executor.submissions) == 1
    function, args = environment.executor.submissions[-1]
    assert function is scene_condition_batch.run_batch
    assert args == (queued["batch_id"],)
    return queued


def _write_tx(destination: Path, case_id: str) -> Path:
    path = destination / "tx.after.wav"
    samples = np.zeros(1_600, dtype=np.float32)
    samples[0] = (sum(case_id.encode("utf-8")) % 101) / 1000.0
    sf.write(path, samples, 16_000, subtype="PCM_16")
    return path


def _controlled_calculation(
    case: dict,
    input_paths: dict[str, Path],
    outputs: dict[str, Path],
    execution: dict,
    *,
    report_id: str,
) -> list[dict]:
    assert report_id
    assert set(input_paths) == set(ROLES)
    assert set(outputs) == {"tx"}
    assert set(execution["input_sha256"]) == set(ROLES)
    return [
        {
            "item_id": item_id,
            "value": 1.25,
            "status": "review_required",
            "metric": "controlled_engineering_proxy",
            "reason": "controlled fake calculation",
        }
        for item_id in case["item_ids"]
    ]


def _install_successful_executor(
    monkeypatch,
    queued: dict,
    *,
    fail_case_id: str | None = None,
    mismatch: str | None = None,
) -> SimpleNamespace:
    run_calls: list[dict] = []
    calculate_calls: list[dict] = []
    mismatch_used = False

    def controlled_run_cli(
        case: dict,
        paths: dict[str, Path],
        destination: Path,
        cluster: str,
    ):
        nonlocal mismatch_used
        run_calls.append(
            {
                "case": copy.deepcopy(case),
                "paths": dict(paths),
                "destination": destination,
                "cluster": cluster,
            }
        )
        if fail_case_id is not None and case["id"] == fail_case_id:
            raise RuntimeError("controlled case failure")

        tx_path = _write_tx(destination, case["id"])
        input_sha256 = {
            role: _sha256_file(path) for role, path in paths.items()
        }
        output_sha256 = {"tx": _sha256_file(tx_path)}
        reported_inputs = dict(input_sha256)
        reported_outputs = dict(output_sha256)
        if mismatch is not None and not mismatch_used:
            mismatch_used = True
            if mismatch == "input":
                reported_inputs["microphone"] = "0" * 64
            elif mismatch == "output":
                reported_outputs["tx"] = "0" * 64
            else:
                raise AssertionError(f"unknown mismatch: {mismatch}")
        execution = {
            "source": "controlled-fake-cli",
            "algorithm_cluster_id": queued["algorithm_cluster_id"],
            "configuration": copy.deepcopy(queued["configuration"]),
            "catalog_sha256": queued["catalog_sha256"],
            "cli_sha256": queued["cli_sha256"],
            "input_sha256": reported_inputs,
            "output_sha256": reported_outputs,
            "output_alignment": {
                "tx": {"analysis_filename": tx_path.name}
            },
        }
        return {"tx": tx_path}, execution

    def controlled_calculate(*args, **kwargs):
        calculate_calls.append(
            {
                "case": copy.deepcopy(args[0]),
                "report_id": kwargs["report_id"],
            }
        )
        return _controlled_calculation(*args, **kwargs)

    monkeypatch.setattr(
        acceptance_batch,
        "_run_cli",
        controlled_run_cli,
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    return SimpleNamespace(
        run_calls=run_calls,
        calculate_calls=calculate_calls,
    )


def _isolated_batch_client() -> TestClient:
    isolated = FastAPI()
    isolated.include_router(scene_condition_batch_api.router)
    return TestClient(isolated)


def _new_mismatched_output_batch(
    completed: dict,
    environment: _SceneBatchEnvironment,
    *,
    assessment_mismatch: bool,
) -> tuple[str, str, bytes]:
    identifier = (
        "a" * 32 if assessment_mismatch else "b" * 32
    )
    scene_condition_batch._create_batch_directories(identifier)
    snapshot = copy.deepcopy(completed)
    snapshot["batch_id"] = identifier
    snapshot["status"] = "completed"
    snapshot.pop("error", None)
    for case in snapshot["cases"]:
        case["report_id"] = case.get("report_id") or "9" * 32
    for row in snapshot["items"]:
        row["evidence_binding"]["batch_id"] = identifier

    target_case = snapshot["cases"][0]
    case_id = target_case["case_id"]
    destination = (
        environment.build
        / scene_condition_batch.BATCH_ROOT_NAME
        / identifier
        / "cases"
        / case_id
    )
    destination.mkdir(parents=True, exist_ok=False)
    tx_path = destination / "tx.after.wav"
    payload = b"RIFF-controlled-mismatched-tx"
    tx_path.write_bytes(payload)
    actual_sha256 = _sha256(payload)
    expected_sha256 = (
        actual_sha256 if assessment_mismatch else "f" * 64
    )
    target_case["execution"]["output_alignment"] = {
        "tx": {"analysis_filename": tx_path.name}
    }
    target_case["execution"]["output_sha256"] = {"tx": expected_sha256}
    target_case["output_sha256"] = {"tx": expected_sha256}
    target_case["tx_visual"]["source_sha256"] = expected_sha256
    target_case["assessment_sha256"] = (
        "f" * 64 if assessment_mismatch else _sha256(b"unused")
    )
    (destination / "assessment.json").write_bytes(b"controlled assessment\n")
    for row in snapshot["items"]:
        if row["case_id"] != case_id:
            continue
        row["evidence_binding"]["report_id"] = target_case["report_id"]
        row["evidence_binding"]["output_sha256"] = {"tx": expected_sha256}
    scene_condition_batch._save(snapshot)
    return identifier, case_id, payload


def _new_assessment_input_mismatch_batch(
    completed: dict,
    environment: _SceneBatchEnvironment,
) -> tuple[str, str]:
    identifier = "9" * 32
    scene_condition_batch._create_batch_directories(identifier)
    snapshot = copy.deepcopy(completed)
    snapshot["batch_id"] = identifier
    snapshot["status"] = "completed"
    snapshot.pop("error", None)
    for row in snapshot["items"]:
        row["evidence_binding"]["batch_id"] = identifier

    target_case = snapshot["cases"][0]
    case_id = target_case["case_id"]
    destination = (
        environment.build
        / scene_condition_batch.BATCH_ROOT_NAME
        / identifier
        / "cases"
        / case_id
    )
    destination.mkdir(parents=True, exist_ok=False)
    tx_path = destination / "tx.after.wav"
    payload = b"RIFF-controlled-assessment-input-mismatch"
    tx_path.write_bytes(payload)
    output_sha256 = _sha256(payload)
    target_case["execution"]["output_alignment"] = {
        "tx": {"analysis_filename": tx_path.name}
    }
    target_case["execution"]["output_sha256"] = {"tx": output_sha256}
    target_case["output_sha256"] = {"tx": output_sha256}
    target_case["tx_visual"]["source_sha256"] = output_sha256
    for row in snapshot["items"]:
        if row["case_id"] == case_id:
            row["evidence_binding"]["output_sha256"] = {
                "tx": output_sha256
            }

    original_assessment_path = (
        environment.build
        / scene_condition_batch.BATCH_ROOT_NAME
        / completed["batch_id"]
        / "cases"
        / case_id
        / "assessment.json"
    )
    assessment = json.loads(original_assessment_path.read_text())
    assessment["batch_id"] = identifier
    assessment["input_sha256"] = copy.deepcopy(target_case["input_sha256"])
    assessment["input_sha256"]["microphone"] = "f" * 64
    assessment["output_sha256"] = {"tx": output_sha256}
    target_case["assessment_sha256"] = scene_condition_batch._write_once_json(
        destination / "assessment.json",
        assessment,
    )
    scene_condition_batch._save(snapshot)
    return identifier, case_id


def test_start_batch_creates_twenty_cases_fifty_unique_observations(
    scene_batch_environment: _SceneBatchEnvironment,
):
    queued = _queue_batch(scene_batch_environment)

    assert queued["contract_item_count"] == 5
    assert queued["observation_count"] == 50
    assert queued["progress"] == {
        "completed_cases": 0,
        "total_cases": 20,
        "completed_items": 0,
        "total_items": 50,
    }
    assert len(queued["cases"]) == 20
    assert len(queued["items"]) == 50
    assert len({row["observation_id"] for row in queued["items"]}) == 50
    assert all(
        row["observation_id"] == f"{row['case_id']}:{row['item_id']}"
        for row in queued["items"]
    )
    item_counts = Counter(row["item_id"] for row in queued["items"])
    assert set(item_counts) == set(scene_batch_environment.contract_by_id)
    assert set(item_counts.values()) == {10}
    case_item_counts = Counter(
        row["case_id"] for row in queued["items"]
    )
    assert set(case_item_counts.values()) == {2, 3}
    assert queued["standard_compliance"] is False
    assert queued["formal_test_status"] == "not_measured"


def test_case_input_hash_must_match_manifest_signals_even_when_items_match(
    scene_batch_environment: _SceneBatchEnvironment,
):
    queued = _queue_batch(scene_batch_environment)
    snapshot = copy.deepcopy(queued)
    target_case = snapshot["cases"][0]
    case_id = target_case["case_id"]
    forged = copy.deepcopy(target_case["input_sha256"])
    forged["microphone"] = "f" * 64
    target_case["input_sha256"] = forged
    for row in snapshot["items"]:
        if row["case_id"] == case_id:
            row["evidence_binding"]["input_sha256"] = copy.deepcopy(forged)

    with pytest.raises(ValueError, match="case.input_sha256"):
        scene_condition_batch.validate_snapshot(snapshot)


def test_snapshot_exposes_the_exported_v1_identity_contract(
    scene_batch_environment: _SceneBatchEnvironment,
):
    queued = _queue_batch(scene_batch_environment)

    assert queued["schema_version"] == scene_condition_batch.SCHEMA
    assert queued["dataset_name"] == DATASET_NAME
    for row in queued["items"]:
        case = next(
            item
            for item in queued["cases"]
            if item["case_id"] == row["case_id"]
        )
        binding = row["evidence_binding"]
        assert "conditions" not in binding
        assert {
            key: binding[key] for key in case["conditions"]
        } == case["conditions"]


def test_fake_twenty_case_execution_preserves_analysis_and_beam_delays(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    calls = _install_successful_executor(monkeypatch, queued)
    first_snapshot = (
        scene_batch_environment.build
        / scene_condition_batch.BATCH_ROOT_NAME
        / queued["batch_id"]
        / "snapshots"
        / "00000001.json"
    )
    first_payload = first_snapshot.read_bytes()

    result = scene_condition_batch.run_batch(queued["batch_id"])

    assert result["status"] == "completed"
    assert result["progress"] == {
        "completed_cases": 20,
        "total_cases": 20,
        "completed_items": 50,
        "total_items": 50,
    }
    assert result["summary"]["review_required"] == 50
    assert result["summary"]["computed"] == 50
    assert len(calls.run_calls) == 20
    assert len(calls.calculate_calls) == 20
    assert {call["case"]["analysis_start_s"] for call in calls.run_calls} == {
        0.5,
        4.0,
    }
    assert all(
        call["case"]["beam_delays_ms"] for call in calls.run_calls
    )
    assert all(
        row["status"] == "review_required"
        and row["value"] == 1.25
        and row["observation_id"] == f"{row['case_id']}:{row['item_id']}"
        for row in result["items"]
    )
    assert len(
        {case["report_id"] for case in result["cases"]}
    ) == 20
    assert not (
        scene_batch_environment.build / "acceptance-batches"
    ).exists()

    assert first_snapshot.read_bytes() == first_payload
    assert json.loads(first_payload)["status"] == "queued"
    versions = sorted(
        (
            scene_batch_environment.build
            / scene_condition_batch.BATCH_ROOT_NAME
            / queued["batch_id"]
            / "snapshots"
        ).glob("*.json")
    )
    assert len(versions) >= 22
    assert json.loads(versions[-1].read_text())["status"] == "completed"
    assert scene_condition_batch.get_batch(queued["batch_id"])["status"] == (
        "completed"
    )


def test_partial_case_failure_invalidates_only_its_items_and_fails_batch(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    failed_case = next(
        case for case in queued["cases"] if len(case["contract_item_ids"]) == 2
    )
    failed_ids = set(failed_case["contract_item_ids"])
    calls = _install_successful_executor(
        monkeypatch,
        queued,
        fail_case_id=failed_case["case_id"],
    )

    result = scene_condition_batch.run_batch(queued["batch_id"])

    assert result["status"] == "failed"
    assert len(calls.run_calls) == 20
    assert len(calls.calculate_calls) == 19
    failed_rows = [
        row for row in result["items"] if row["case_id"] == failed_case["case_id"]
    ]
    other_rows = [
        row for row in result["items"] if row["case_id"] != failed_case["case_id"]
    ]
    assert {row["item_id"] for row in failed_rows} == failed_ids
    assert all(row["status"] == "invalid" for row in failed_rows)
    assert all(row["value"] is None for row in failed_rows)
    assert all(
        "controlled case failure" in row["reason"] for row in failed_rows
    )
    assert len(
        {row["evidence_binding"]["report_id"] for row in failed_rows}
    ) == 1
    assert all(row["status"] == "review_required" for row in other_rows)
    assert result["summary"]["invalid"] == len(failed_ids)
    assert result["summary"]["review_required"] == 50 - len(failed_ids)
    assert result["summary"]["computed"] == 50 - len(failed_ids)


@pytest.mark.parametrize(
    ("mismatch", "expected_reason"),
    [
        ("input", "输入哈希"),
        ("output", "输出身份"),
    ],
)
def test_executor_input_or_output_hash_mismatch_is_rejected(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
    mismatch: str,
    expected_reason: str,
):
    queued = _queue_batch(scene_batch_environment)
    first_case = queued["cases"][0]
    calls = _install_successful_executor(
        monkeypatch,
        queued,
        mismatch=mismatch,
    )

    result = scene_condition_batch.run_batch(queued["batch_id"])

    assert result["status"] == "failed"
    assert len(calls.run_calls) == 20
    assert len(calls.calculate_calls) == 19
    rejected = [
        row for row in result["items"] if row["case_id"] == first_case["case_id"]
    ]
    assert all(row["status"] == "invalid" for row in rejected)
    assert all(expected_reason in row["reason"] for row in rejected)


def test_frozen_dataset_or_input_identity_mismatch_never_reaches_fake_cli(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    changed = copy.deepcopy(scene_batch_environment.dataset)
    changed["dataset_manifest_sha256"] = "f" * 64
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        _validated_loader(changed, scene_batch_environment.contract_by_id),
    )
    run_calls: list[object] = []

    def must_not_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        raise AssertionError("changed dataset must be rejected before CLI")

    monkeypatch.setattr(acceptance_batch, "_run_cli", must_not_run)
    result = scene_condition_batch.run_batch(queued["batch_id"])

    assert run_calls == []
    assert result["status"] == "failed"
    assert all(row["status"] == "invalid" for row in result["items"])
    assert all("测试集身份发生变化" in row["reason"] for row in result["items"])


def test_scene_and_legacy_batches_share_the_single_execution_lock(
    scene_batch_environment: _SceneBatchEnvironment,
):
    acceptance_batch._LEGACY_ACTIVE = True
    with pytest.raises(ValueError, match="已有测试任务正在执行"):
        scene_condition_batch.start_batch(DATASET_ID, ALGORITHM_CLUSTER)

    acceptance_batch._LEGACY_ACTIVE = False
    queued = _queue_batch(scene_batch_environment)
    acceptance_batch._LEGACY_ACTIVE = True
    with pytest.raises(ValueError, match="禁止并发"):
        scene_condition_batch.run_batch(queued["batch_id"])


def test_invalid_dataset_mode_scene_and_non_vcp_cluster_are_rejected(
    scene_batch_environment: _SceneBatchEnvironment,
):
    with pytest.raises(ValueError, match="测试集不可用"):
        scene_condition_batch.start_batch("not-a-dataset", ALGORITHM_CLUSTER)
    with pytest.raises(ValueError, match=MODE):
        scene_condition_batch.start_batch(
            DATASET_ID,
            ALGORITHM_CLUSTER,
            mode="NB-wired",
        )
    with pytest.raises(ValueError, match=SCENE):
        scene_condition_batch.start_batch(
            DATASET_ID,
            ALGORITHM_CLUSTER,
            scene="sce2",
        )
    with pytest.raises(ValueError, match="VCP"):
        scene_condition_batch.start_batch(DATASET_ID, "agc_default")
    assert scene_batch_environment.executor.submissions == []


def test_batch_api_rejects_non_wb_and_non_vcp_requests(
    scene_batch_environment: _SceneBatchEnvironment,
):
    client = _isolated_batch_client()
    base = {
        "dataset_id": DATASET_ID,
        "algorithm_cluster_id": ALGORITHM_CLUSTER,
        "mode": MODE,
        "scene": SCENE,
    }
    non_wb = client.post(BATCH_PREFIX, json={**base, "mode": "NB-wired"})
    assert non_wb.status_code == 422
    non_vcp = client.post(
        BATCH_PREFIX,
        json={**base, "algorithm_cluster_id": "agc_default"},
    )
    assert non_vcp.status_code == 422
    assert "VCP" in non_vcp.json()["detail"]
    assert scene_batch_environment.executor.submissions == []


def test_tx_endpoint_serves_only_the_bound_case_and_rejects_wrong_case(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    _install_successful_executor(monkeypatch, queued)
    completed = scene_condition_batch.run_batch(queued["batch_id"])
    case_id = completed["cases"][0]["case_id"]
    expected_path, expected_sha256 = scene_condition_batch.output_path(
        queued["batch_id"],
        case_id,
    )
    expected_payload = expected_path.read_bytes()
    client = _isolated_batch_client()

    response = client.get(
        f"{BATCH_PREFIX}/{queued['batch_id']}/cases/{case_id}/outputs/tx"
    )
    assert response.status_code == 200
    assert response.content == expected_payload
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["etag"] == f'"{expected_sha256}"'

    wrong_case = client.get(
        f"{BATCH_PREFIX}/{queued['batch_id']}/cases/not-a-case/outputs/tx"
    )
    assert wrong_case.status_code == 404


def test_tx_endpoint_rejects_mismatched_tx_and_report_hash(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    _install_successful_executor(monkeypatch, queued)
    completed = scene_condition_batch.run_batch(queued["batch_id"])
    client = _isolated_batch_client()

    tx_identifier, tx_case_id, _ = _new_mismatched_output_batch(
        completed,
        scene_batch_environment,
        assessment_mismatch=False,
    )
    tx_response = client.get(
        f"{BATCH_PREFIX}/{tx_identifier}/cases/{tx_case_id}/outputs/tx"
    )
    assert tx_response.status_code in {409, 422, 503}

    report_identifier, report_case_id, _ = _new_mismatched_output_batch(
        completed,
        scene_batch_environment,
        assessment_mismatch=True,
    )
    report_response = client.get(
        f"{BATCH_PREFIX}/{report_identifier}/cases/{report_case_id}/outputs/tx"
    )
    assert report_response.status_code in {409, 422, 503}


def test_tx_endpoint_rejects_assessment_input_hash_not_matching_case(
    scene_batch_environment: _SceneBatchEnvironment,
    monkeypatch,
):
    queued = _queue_batch(scene_batch_environment)
    _install_successful_executor(monkeypatch, queued)
    completed = scene_condition_batch.run_batch(queued["batch_id"])
    identifier, case_id = _new_assessment_input_mismatch_batch(
        completed,
        scene_batch_environment,
    )

    response = _isolated_batch_client().get(
        f"{BATCH_PREFIX}/{identifier}/cases/{case_id}/outputs/tx"
    )

    assert response.status_code == 409
    assert "assessment.json" in response.json()["detail"]
    assert "input" in response.json()["detail"]


def test_scene_condition_batch_routes_are_mounted_on_the_product_app():
    paths = {
        getattr(route, "path", "")
        for route in product_app.routes
    }
    assert BATCH_PREFIX in paths
