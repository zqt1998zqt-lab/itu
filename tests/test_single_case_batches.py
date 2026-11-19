from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from server.acoustic import (
    acceptance_batch,
    acceptance_scene_condition_api,
    acoustic_stress_batch,
    scene_condition_batch,
    scene_condition_batch_api,
    service,
)
from server.acoustic.acceptance_scope import background_noise_scene_items
from server.acoustic.acceptance_scene_conditions import MODE, ROLES, SCENE
from server.acoustic.acoustic_stress_profiles import (
    CATEGORIES,
    stress_profile_for_dataset,
)
from server.acoustic.scene_conversation_profiles import (
    CONVERSATION_MODES,
    profile_for_dataset as conversation_profile_for_dataset,
)


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_acoustic_stress_batch import _build_stress_dataset  # noqa: E402
from test_conversation_batches import (  # noqa: E402
    _build_conversation_dataset,
    _build_legacy_dataset,
)


ALGORITHM_CLUSTER = "vcp_default"
CATALOG_SHA256 = "e" * 64
BATCH_PREFIX = "/api/alignment/acceptance/scene-condition-batches"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract_by_id(dataset: dict) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in background_noise_scene_items(MODE, dataset["scene"])
    }


def _build_snapshot(
    dataset: dict,
    case_id: str | None = None,
) -> dict:
    return scene_condition_batch._build_snapshot(
        dataset,
        _contract_by_id(dataset),
        "a" * 32,
        ALGORITHM_CLUSTER,
        {"profile": ALGORITHM_CLUSTER},
        CATALOG_SHA256,
        "c" * 64,
        case_id=case_id,
    )


def _persist_snapshot(
    dataset: dict,
    identifier: str,
    case_id: str | None,
    created_at: str,
    *,
    algorithm_cluster_id: str = ALGORITHM_CLUSTER,
) -> dict:
    snapshot = scene_condition_batch._build_snapshot(
        dataset,
        _contract_by_id(dataset),
        identifier,
        algorithm_cluster_id,
        {"profile": algorithm_cluster_id},
        CATALOG_SHA256,
        "c" * 64,
        case_id=case_id,
    )
    snapshot["created_at"] = created_at
    scene_condition_batch._create_batch_directories(identifier)
    scene_condition_batch._save(snapshot)
    return snapshot


def _visual(source_sha256: str) -> dict:
    return {
        "source_sha256": source_sha256,
        "duration_s": 0.1,
        "sample_rate": 16_000,
        "channels": 1,
        "peak": 0.0,
        "wave_scale": 1.0,
        "wave_min_i16": base64.b64encode(b"\0" * 1440).decode(),
        "wave_max_i16": base64.b64encode(b"\0" * 1440).decode(),
        "mel_i16": base64.b64encode(b"\0" * (40 * 240 * 2)).decode(),
        "mel_ceiling_db": -120.0,
    }


class _Executor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))


def _install_start_environment(
    monkeypatch,
    tmp_path: Path,
    dataset: dict,
) -> _Executor:
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "BUILD", build)
    contract_by_id = _contract_by_id(dataset)
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        lambda dataset_id, requested_scene=None: (
            deepcopy(dataset),
            deepcopy(contract_by_id),
        ),
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _Executor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    cli = build / "fake-vcp"
    cli.write_bytes(b"fake-vcp")
    import server.alignment_runner as alignment_runner

    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster_id: (
            {"profile": "vcp_default"},
            {},
            {"algorithm_cluster_id": cluster_id},
            CATALOG_SHA256,
        ),
    )
    return executor


def test_background_single_snapshot_filters_dynamic_counts(tmp_path):
    legacy = _build_legacy_dataset(tmp_path / "legacy")
    speech_case = next(
        case
        for case in legacy["cases"]
        if case["provenance"]["render_metadata"]["kind"] == "speech"
    )

    full = _build_snapshot(legacy)
    single = _build_snapshot(legacy, speech_case["case_id"])

    assert not set(scene_condition_batch.SINGLE_CASE_SELECTION_FIELDS) & set(full)
    scene_condition_batch.validate_snapshot(full)
    scene_condition_batch.validate_snapshot(single)
    assert single["execution_scope"] == "single_case"
    assert single["selected_case_id"] == speech_case["case_id"]
    assert single["dataset_case_count"] == 20
    assert single["dataset_observation_count"] == 50
    assert single["case_count"] == 1
    assert single["observation_count"] == 3
    assert single["preview_condition_group_count"] == 20
    assert single["contract_item_count"] == 5
    assert single["speech_case_count"] == 1
    assert single["cng_case_count"] == 0
    assert single["conversation_counts"] == {}
    assert single["progress"] == {
        "completed_cases": 0,
        "total_cases": 1,
        "completed_items": 0,
        "total_items": 3,
    }
    assert [case["case_id"] for case in single["cases"]] == [
        speech_case["case_id"]
    ]
    assert {row["case_id"] for row in single["items"]} == {
        speech_case["case_id"]
    }

    conversation = _build_conversation_dataset(
        tmp_path / "conversation",
        "sce1",
    )
    near_case = next(
        case
        for case in conversation["cases"]
        if case["provenance"]["conversation_mode"] == "near_single"
    )
    conversation_single = _build_snapshot(
        conversation,
        near_case["case_id"],
    )
    scene_condition_batch.validate_snapshot(conversation_single)
    profile = conversation_profile_for_dataset(conversation["dataset_id"])

    assert conversation_single["dataset_case_count"] == 100
    assert conversation_single["dataset_observation_count"] == 280
    assert conversation_single["observation_count"] == 3
    assert conversation_single["preview_condition_group_count"] == 100
    assert conversation_single["speech_case_count"] == 1
    assert conversation_single["cng_case_count"] == 0
    assert conversation_single["conversation_counts"] == {
        mode: int(mode == "near_single")
        for mode in CONVERSATION_MODES
    }
    assert set(profile.conversation_counts) == set(CONVERSATION_MODES)


def test_stress_single_snapshot_keeps_all_category_keys(tmp_path):
    dataset = _build_stress_dataset(
        tmp_path / "stress",
        "echo",
        "WB-wired",
    )
    selected = dataset["cases"][0]
    category = selected["provenance"]["workload_category"]
    profile = stress_profile_for_dataset(dataset["dataset_id"])

    full = acoustic_stress_batch.build_snapshot(
        dataset,
        dataset["contract_by_id"],
        "b" * 32,
        ALGORITHM_CLUSTER,
        {"profile": ALGORITHM_CLUSTER},
        CATALOG_SHA256,
        "c" * 64,
    )
    single = acoustic_stress_batch.build_snapshot(
        dataset,
        dataset["contract_by_id"],
        "b" * 32,
        ALGORITHM_CLUSTER,
        {"profile": ALGORITHM_CLUSTER},
        CATALOG_SHA256,
        "c" * 64,
        case_id=selected["case_id"],
    )

    assert not set(scene_condition_batch.SINGLE_CASE_SELECTION_FIELDS) & set(full)
    acoustic_stress_batch.validate_snapshot(full)
    acoustic_stress_batch.validate_snapshot(single)
    assert single["execution_scope"] == "single_case"
    assert single["selected_case_id"] == selected["case_id"]
    assert single["dataset_case_count"] == 100
    assert single["dataset_observation_count"] == 100
    assert single["case_count"] == 1
    assert single["observation_count"] == 1
    assert single["preview_condition_group_count"] == 100
    assert single["contract_item_count"] == profile.contract_item_count
    assert set(single["category_counts"]) == set(CATEGORIES["echo"])
    assert single["category_counts"] == {
        key: int(key == category)
        for key in CATEGORIES["echo"]
    }
    assert single["selected_mode"] == "WB-wired"
    assert single["selected_scene"] == "baseline"
    assert single["standard_compliance"] is False
    assert single["formal_test_status"] == "not_measured"


def test_invalid_and_cross_dataset_case_ids_never_create_or_run(
    tmp_path,
    monkeypatch,
):
    legacy = _build_legacy_dataset(tmp_path / "legacy")
    conversation = _build_conversation_dataset(
        tmp_path / "conversation",
        "sce2",
    )
    executor = _install_start_environment(
        monkeypatch,
        tmp_path,
        legacy,
    )
    directory_calls: list[str] = []
    cli_calls: list[tuple] = []
    monkeypatch.setattr(
        scene_condition_batch,
        "_create_batch_directories",
        lambda identifier: directory_calls.append(identifier),
    )
    monkeypatch.setattr(
        acceptance_batch,
        "_run_cli",
        lambda *args, **kwargs: cli_calls.append((args, kwargs)),
    )

    invalid_ids = (
        "",
        "../escape",
        "missing-case",
        conversation["cases"][0]["case_id"],
    )
    for case_id in invalid_ids:
        with pytest.raises(ValueError, match="case_id"):
            scene_condition_batch.start_batch(
                legacy["dataset_id"],
                ALGORITHM_CLUSTER,
                case_id=case_id,
            )

    assert directory_calls == []
    assert executor.submissions == []
    assert cli_calls == []


def test_cross_stress_scope_case_never_creates_or_runs(
    tmp_path,
    monkeypatch,
):
    target = _build_stress_dataset(
        tmp_path / "target-stress",
        "echo",
        "WB-wired",
    )
    foreign = _build_stress_dataset(
        tmp_path / "foreign-stress",
        "double_talk",
        "WB-wired",
    )
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(
        acoustic_stress_batch,
        "validated_dataset",
        lambda dataset_id, requested_scene=None: (
            deepcopy(target),
            deepcopy(target["contract_by_id"]),
        ),
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _Executor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    cli = build / "fake-vcp"
    cli.write_bytes(b"fake-vcp")
    import server.alignment_runner as alignment_runner

    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster_id: (
            {"profile": "vcp_default"},
            {},
            {"algorithm_cluster_id": cluster_id},
            CATALOG_SHA256,
        ),
    )
    directory_calls: list[str] = []
    monkeypatch.setattr(
        scene_condition_batch,
        "_create_batch_directories",
        lambda identifier: directory_calls.append(identifier),
    )

    with pytest.raises(ValueError, match="case_id"):
        scene_condition_batch.start_batch(
            target["dataset_id"],
            ALGORITHM_CLUSTER,
            mode="WB-wired",
            scene="baseline",
            case_id=foreign["cases"][0]["case_id"],
        )

    assert directory_calls == []
    assert executor.submissions == []


def test_latest_per_case_recovers_complete_100_case_coverage(
    tmp_path,
    monkeypatch,
):
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "BUILD", build)
    dataset = _build_conversation_dataset(
        tmp_path / "conversation",
        "sce1",
    )
    contract_by_id = _contract_by_id(dataset)
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        lambda dataset_id, requested_scene=None: (
            deepcopy(dataset),
            deepcopy(contract_by_id),
        ),
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)

    def persist(
        identifier: str,
        case_id: str | None,
        second: int,
    ) -> None:
        snapshot = _persist_snapshot(
            dataset,
            identifier,
            case_id,
            f"2026-11-11T00:00:{second:02d}+00:00",
        )
        assert snapshot["batch_id"] == identifier
        acceptance_batch._ACTIVE.add(identifier)

    full_id = f"{1:032x}"
    persist(full_id, None, 0)
    original_first_id = f"{2:032x}"
    persist(original_first_id, dataset["cases"][0]["case_id"], 1)

    distinct_single_ids: list[str] = []
    for index in range(1, 25):
        identifier = f"{index + 2:032x}"
        distinct_single_ids.append(identifier)
        persist(
            identifier,
            dataset["cases"][index]["case_id"],
            index + 1,
        )

    duplicate_id = f"{27:032x}"
    persist(duplicate_id, dataset["cases"][0]["case_id"], 26)
    foreign_cluster_id = f"{28:032x}"
    _persist_snapshot(
        dataset,
        foreign_cluster_id,
        dataset["cases"][-1]["case_id"],
        "2026-11-11T00:00:27+00:00",
        algorithm_cluster_id="vcp_other",
    )
    acceptance_batch._ACTIVE.add(foreign_cluster_id)

    ordinary = scene_condition_batch.list_batches(
        limit=20,
        dataset_id=dataset["dataset_id"],
        algorithm_cluster_id=ALGORITHM_CLUSTER,
    )
    ordinary_case_ids = {
        case["case_id"]
        for snapshot in ordinary
        for case in snapshot["cases"]
    }
    assert len(ordinary) == 20
    assert len(ordinary_case_ids) < 100
    assert full_id not in {snapshot["batch_id"] for snapshot in ordinary}
    assert foreign_cluster_id not in {
        snapshot["batch_id"] for snapshot in ordinary
    }

    aggregated = scene_condition_batch.list_batches(
        limit=1,
        dataset_id=dataset["dataset_id"],
        algorithm_cluster_id=ALGORITHM_CLUSTER,
        latest_per_case=True,
    )
    aggregated_case_ids = {
        case["case_id"]
        for snapshot in aggregated
        for case in snapshot["cases"]
    }
    aggregated_ids = {
        snapshot["batch_id"]
        for snapshot in aggregated
    }

    assert aggregated_case_ids == {
        case["case_id"] for case in dataset["cases"]
    }
    assert len(aggregated) == 26
    assert len(aggregated_ids) == 26
    assert duplicate_id in aggregated_ids
    assert original_first_id not in aggregated_ids
    assert foreign_cluster_id not in aggregated_ids
    assert full_id in aggregated_ids
    assert set(distinct_single_ids) <= aggregated_ids
    assert aggregated[0]["batch_id"] == duplicate_id
    assert aggregated[-1]["batch_id"] == full_id
    assert sum(
        snapshot.get("execution_scope") == "single_case"
        for snapshot in aggregated
    ) == 25
    assert sum(
        "execution_scope" not in snapshot
        for snapshot in aggregated
    ) == 1
    assert not set(
        scene_condition_batch.SINGLE_CASE_SELECTION_FIELDS
    ) & set(aggregated[-1])
    for snapshot in aggregated:
        assert snapshot == scene_condition_batch.get_batch(
            snapshot["batch_id"]
        )


def test_latest_per_case_api_requires_both_filters():
    isolated = FastAPI()
    isolated.include_router(scene_condition_batch_api.router)
    client = TestClient(isolated)

    assert client.get(
        f"{BATCH_PREFIX}?latest_per_case=true"
    ).status_code == 422
    assert client.get(
        f"{BATCH_PREFIX}?latest_per_case=true&dataset_id={'a' * 32}"
    ).status_code == 422
    with pytest.raises(ValueError, match="布尔"):
        scene_condition_batch.list_batches(latest_per_case=1)


def test_latest_per_case_api_forwards_boolean_and_filters(
    monkeypatch,
):
    calls: list[dict] = []
    monkeypatch.setattr(
        scene_condition_batch_api,
        "list_batches",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    isolated = FastAPI()
    isolated.include_router(scene_condition_batch_api.router)
    client = TestClient(isolated)

    response = client.get(
        BATCH_PREFIX,
        params={
            "limit": 1,
            "dataset_id": "a" * 32,
            "algorithm_cluster_id": ALGORITHM_CLUSTER,
            "latest_per_case": "true",
        },
    )

    assert response.status_code == 200
    assert calls == [
        {
            "limit": 1,
            "dataset_id": "a" * 32,
            "algorithm_cluster_id": ALGORITHM_CLUSTER,
            "latest_per_case": True,
        }
    ]


def test_latest_per_case_covers_stress_single_plus_old_full(
    tmp_path,
    monkeypatch,
):
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "BUILD", build)
    dataset = _build_stress_dataset(
        tmp_path / "stress",
        "echo",
        "WB-wired",
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        lambda dataset_id, requested_scene=None: (
            deepcopy(dataset),
            deepcopy(dataset["contract_by_id"]),
        ),
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)

    def persist(
        identifier: str,
        case_id: str | None,
        created_at: str,
    ) -> None:
        snapshot = acoustic_stress_batch.build_snapshot(
            dataset,
            dataset["contract_by_id"],
            identifier,
            ALGORITHM_CLUSTER,
            {"profile": ALGORITHM_CLUSTER},
            CATALOG_SHA256,
            "c" * 64,
            case_id=case_id,
        )
        snapshot["created_at"] = created_at
        scene_condition_batch._create_batch_directories(identifier)
        scene_condition_batch._save(snapshot)
        acceptance_batch._ACTIVE.add(identifier)

    full_id = f"{101:032x}"
    single_id = f"{102:032x}"
    persist(full_id, None, "2026-11-11T00:01:00+00:00")
    persist(
        single_id,
        dataset["cases"][0]["case_id"],
        "2026-11-11T00:01:01+00:00",
    )

    aggregated = scene_condition_batch.list_batches(
        limit=1,
        dataset_id=dataset["dataset_id"],
        algorithm_cluster_id=ALGORITHM_CLUSTER,
        latest_per_case=True,
    )

    assert [snapshot["batch_id"] for snapshot in aggregated] == [
        single_id,
        full_id,
    ]
    assert aggregated[0]["execution_scope"] == "single_case"
    assert aggregated[0]["case_count"] == 1
    assert not set(
        scene_condition_batch.SINGLE_CASE_SELECTION_FIELDS
    ) & set(aggregated[1])
    assert aggregated[1]["case_count"] == 100
    assert {
        case["case_id"]
        for snapshot in aggregated
        for case in snapshot["cases"]
    } == {case["case_id"] for case in dataset["cases"]}


def test_batch_api_forwards_optional_case_id(
    monkeypatch,
):
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        scene_condition_batch_api,
        "start_batch",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
    )
    isolated = FastAPI()
    isolated.include_router(scene_condition_batch_api.router)
    client = TestClient(isolated)
    body = {
        "dataset_id": "a" * 32,
        "algorithm_cluster_id": ALGORITHM_CLUSTER,
        "mode": MODE,
        "scene": SCENE,
    }

    response = client.post(
        BATCH_PREFIX,
        json={**body, "case_id": "case-001"},
    )
    assert response.status_code == 202
    assert calls[-1][1]["case_id"] == "case-001"

    response = client.post(BATCH_PREFIX, json=body)
    assert response.status_code == 202
    assert calls[-1][1]["case_id"] is None


def test_pruned_snapshot_cannot_masquerade_as_full_batch(tmp_path):
    legacy = _build_legacy_dataset(tmp_path / "legacy")
    full = _build_snapshot(legacy)
    selected = full["cases"][0]["case_id"]
    pruned = deepcopy(full)
    pruned["cases"] = [
        case for case in pruned["cases"] if case["case_id"] == selected
    ]
    pruned["items"] = [
        row for row in pruned["items"] if row["case_id"] == selected
    ]
    pruned["case_count"] = 1
    pruned["observation_count"] = len(pruned["items"])
    pruned["progress"] = scene_condition_batch.progress_counts(pruned)
    pruned["summary"] = scene_condition_batch.summarize_items(pruned["items"])

    with pytest.raises(ValueError, match="数量|选择字段"):
        scene_condition_batch.validate_snapshot(pruned)

    single = _build_snapshot(legacy, selected)
    scene_condition_batch._verify_dataset_frozen(single, legacy)
    changed = deepcopy(legacy)
    target = next(
        case for case in changed["cases"] if case["case_id"] == selected
    )
    target["signals_sha256"]["microphone"] = "f" * 64
    with pytest.raises(ValueError, match="输入身份"):
        scene_condition_batch._verify_dataset_frozen(single, changed)


def test_full_and_single_batches_share_execution_history_and_output(
    tmp_path,
    monkeypatch,
):
    dataset = _build_legacy_dataset(tmp_path / "legacy")
    executor = _install_start_environment(monkeypatch, tmp_path, dataset)
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_visual_track",
        lambda path, source_sha256: _visual(source_sha256),
    )
    run_calls: list[str] = []
    calculate_calls: list[str] = []
    frozen: dict = {}

    def run_cli(case, paths, destination, cluster):
        run_calls.append(case["id"])
        tx = destination / "tx.after.wav"
        tx.write_bytes(f"tx:{case['id']}".encode())
        inputs = {
            role: _sha256(path.read_bytes())
            for role, path in paths.items()
        }
        output = {"tx": _sha256(tx.read_bytes())}
        return {"tx": tx}, {
            "source": "controlled-single-case-cli",
            "algorithm_cluster_id": frozen["algorithm_cluster_id"],
            "configuration": deepcopy(frozen["configuration"]),
            "catalog_sha256": frozen["catalog_sha256"],
            "cli_sha256": frozen["cli_sha256"],
            "input_sha256": inputs,
            "output_sha256": output,
            "output_alignment": {
                "tx": {"analysis_filename": tx.name}
            },
        }

    def calculate(case, paths, outputs, execution, *, report_id):
        calculate_calls.append(case["id"])
        assert report_id
        assert set(paths) == set(ROLES)
        return [
            {
                "item_id": item_id,
                "value": 1.0,
                "status": "review_required",
                "metric": "controlled_single_case_proxy",
                "reason": "controlled fake calculation",
            }
            for item_id in case["item_ids"]
        ]

    monkeypatch.setattr(acceptance_batch, "_run_cli", run_cli)
    monkeypatch.setattr(scene_condition_batch, "calculate_case_items", calculate)

    full_queued = scene_condition_batch.start_batch(
        dataset["dataset_id"],
        ALGORITHM_CLUSTER,
    )
    frozen.update(full_queued)
    full_completed = scene_condition_batch.run_batch(full_queued["batch_id"])
    assert full_completed["status"] == "completed"
    assert not set(scene_condition_batch.SINGLE_CASE_SELECTION_FIELDS) & set(
        full_completed
    )
    assert len(run_calls) == 20

    selected = dataset["cases"][4]["case_id"]
    single_queued = scene_condition_batch.start_batch(
        dataset["dataset_id"],
        ALGORITHM_CLUSTER,
        case_id=selected,
    )
    frozen.update(single_queued)
    before = len(run_calls)
    single_completed = scene_condition_batch.run_batch(
        single_queued["batch_id"]
    )

    assert single_completed["status"] == "completed"
    assert run_calls[before:] == [selected]
    assert calculate_calls[-1] == selected
    assert len(single_completed["cases"]) == 1
    assert single_completed["cases"][0]["case_id"] == selected
    assert single_completed["progress"] == {
        "completed_cases": 1,
        "total_cases": 1,
        "completed_items": len(single_completed["items"]),
        "total_items": len(single_completed["items"]),
    }
    history = scene_condition_batch.list_batches(limit=20)
    assert {
        full_queued["batch_id"],
        single_queued["batch_id"],
    } <= {row["batch_id"] for row in history}

    isolated = FastAPI()
    isolated.include_router(scene_condition_batch_api.router)
    client = TestClient(isolated)
    assert client.get(
        f"{BATCH_PREFIX}/{full_queued['batch_id']}"
    ).status_code == 200
    single_response = client.get(
        f"{BATCH_PREFIX}/{single_queued['batch_id']}"
    )
    assert single_response.status_code == 200
    assert single_response.json()["selected_case_id"] == selected
    output_response = client.get(
        f"{BATCH_PREFIX}/{single_queued['batch_id']}"
        f"/cases/{selected}/outputs/tx"
    )
    assert output_response.status_code == 200
    assert output_response.content == f"tx:{selected}".encode()
    assert len(executor.submissions) == 2
