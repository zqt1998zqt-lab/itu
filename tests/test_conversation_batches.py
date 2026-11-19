from __future__ import annotations

import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import (
    acceptance_batch,
    acceptance_scene_condition_api,
    acceptance_scene_condition_store,
    scene_condition_batch,
    service,
)
from server.acoustic.acceptance_scope import background_noise_scene_items
from server.acoustic.acceptance_scene_conditions import ROLES
from server.acoustic.conversation_diagnostics import (
    EXPECTED_RENDER_KIND,
    EXPECTED_SEGMENT_LABELS,
    diagnose_conversation_case,
    validate_activity_metadata,
    validate_diagnostics,
)
from server.acoustic.scene_condition_profiles import (
    profile_for_scene as legacy_profile_for_scene,
)
from server.acoustic.scene_conversation_profiles import (
    CONVERSATION_MODES,
    profile_for_dataset,
    profile_for_scene,
)


SCENES = ("sce1", "sce2", "sce3", "sce4")
ALGORITHM_CLUSTER = "vcp_default"
CATALOG_SHA256 = "e" * 64
RATE = 16_000
DURATION_S = 12.0
FRAMES = RATE * 12


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _activity_segments(mode: str) -> list[dict[str, object]]:
    if mode == "transition":
        bounds = (
            ("quiet", 0.0, 0.5),
            ("far_only", 0.5, 2.0),
            ("near_only", 2.0, 5.0),
            ("far_only", 5.0, 8.0),
            ("double_talk", 8.0, 11.0),
            ("noise_only", 11.0, 12.0),
        )
    elif mode == "noise_only":
        bounds = (
            ("cng_active", 0.0, 2.0),
            ("cng_settling", 2.0, 4.0),
            ("cng_noise_observation", 4.0, 12.0),
        )
    else:
        label = EXPECTED_SEGMENT_LABELS[mode][0]
        bounds = ((label, 0.0, 12.0),)
    return [
        {"label": label, "start_s": start, "end_s": end}
        for label, start, end in bounds
    ]


def _signal_files(root: Path, scene: str) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, object]],
]:
    scene_index = SCENES.index(scene) if scene in SCENES else len(SCENES)
    rng = np.random.default_rng(20261110 + scene_index)
    signals: dict[str, str] = {}
    hashes: dict[str, str] = {}
    assets: dict[str, dict[str, object]] = {}
    for role in ROLES:
        relative = f"assets/audio/{role}.wav"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        mono = rng.normal(scale=0.01, size=FRAMES)
        if role == "microphone":
            payload = np.column_stack((mono, mono * 0.5))
        else:
            payload = mono
        sf.write(path, payload, RATE, subtype="PCM_16")
        digest = _sha256_file(path)
        signals[role] = relative
        hashes[role] = digest
        assets[relative] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    return signals, hashes, assets


def _build_conversation_dataset(
    build: Path,
    scene: str,
) -> dict:
    profile = profile_for_scene(scene)
    root = (
        build
        / "acoustic-scene-condition-datasets"
        / profile.dataset_id
    )
    root.mkdir(parents=True, exist_ok=False)
    signals, hashes, assets = _signal_files(root, scene)
    contract_rows = background_noise_scene_items("WB-wired", scene)
    speech_items = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    ]
    cng_items = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"cng_level", "cng_spectrum"}
    ]
    family_by_id = {row["id"]: row["family"] for row in contract_rows}
    cases = []
    for mode in CONVERSATION_MODES:
        kind = EXPECTED_RENDER_KIND[mode]
        item_ids = speech_items if kind == "speech" else cng_items
        for index in range(20):
            cases.append(
                {
                    "case_id": f"{scene}-{mode}-{index:02d}",
                    "conditions": copy.deepcopy(profile.conditions),
                    "families": [family_by_id[item_id] for item_id in item_ids],
                    "contract_item_ids": list(item_ids),
                    "signals": copy.deepcopy(signals),
                    "signals_sha256": copy.deepcopy(hashes),
                    "provenance": {
                        "conversation_mode": mode,
                        "analysis_start_s": (
                            4.0 if mode == "noise_only" else 0.0
                        ),
                        "render_metadata": {
                            "kind": kind,
                            "duration_s": DURATION_S,
                            "sample_rate_hz": RATE,
                            "beam_delays_ms": [0.0625, 0.0],
                            "near_arrival_delays_samples": [0, 0],
                            "echo_paths": [],
                            "noise_spatial": {
                                "channel_1_recipe": {
                                    "rho": 0.0,
                                    "seed2": 7,
                                    "normalizer": 1.0,
                                    "noise_channel_scale_linear": 1.0,
                                }
                            },
                            "segments": _activity_segments(mode),
                        },
                    },
                }
            )
    manifest = {
        "schema_version": profile.schema_version,
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "test_scope": "background_noise",
        "mode": "WB-wired",
        "scene": scene,
        "contract_item_count": 5,
        "preview_condition_group_count": 100,
        "case_count": 100,
        "speech_case_count": 80,
        "cng_case_count": 20,
        "observation_count": 280,
        "conversation_counts": copy.deepcopy(profile.conversation_counts),
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "cases": cases,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "assets.json", assets)
    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "schema_version": profile.schema_version,
        "mode": "WB-wired",
        "scene": scene,
        "root": root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "dataset_manifest_sha256": "1" * 64,
        "dataset_assets_sha256": "2" * 64,
        "contract_item_count": 5,
        "contract_item_ids": profile.contract_item_ids,
        "preview_condition_group_count": 100,
        "preview_count": 100,
        "case_count": 100,
        "speech_case_count": 80,
        "cng_case_count": 20,
        "observation_count": 280,
        "conversation_counts": copy.deepcopy(profile.conversation_counts),
        "referenced_assets": set(signals.values()),
    }


def _build_legacy_dataset(build: Path) -> dict:
    profile = legacy_profile_for_scene("sce1")
    root = (
        build
        / "acoustic-scene-condition-datasets"
        / profile.dataset_id
    )
    root.mkdir(parents=True, exist_ok=False)
    signals, hashes, assets = _signal_files(root, "legacy-sce1")
    contract_rows = background_noise_scene_items("WB-wired", "sce1")
    speech_items = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    ]
    cng_items = [
        row["id"]
        for row in contract_rows
        if row["family"] in {"cng_level", "cng_spectrum"}
    ]
    family_by_id = {row["id"]: row["family"] for row in contract_rows}
    cases = []
    for kind, item_ids in (("speech", speech_items), ("cng", cng_items)):
        for index in range(10):
            cases.append(
                {
                    "case_id": f"legacy-{kind}-{index:02d}",
                    "conditions": copy.deepcopy(profile.conditions),
                    "families": [family_by_id[item_id] for item_id in item_ids],
                    "contract_item_ids": list(item_ids),
                    "signals": copy.deepcopy(signals),
                    "signals_sha256": copy.deepcopy(hashes),
                    "provenance": {
                        "analysis_start_s": 0.5 if kind == "speech" else 4.0,
                        "render_metadata": {
                            "kind": kind,
                            "beam_delays_ms": [0.0625, 0.0],
                        },
                    },
                }
            )
    manifest = {
        "schema_version": "scene-condition-preview-v1",
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "test_scope": "background_noise",
        "mode": "WB-wired",
        "scene": "sce1",
        "contract_item_count": 5,
        "preview_condition_group_count": 20,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "cases": cases,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "assets.json", assets)
    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "mode": "WB-wired",
        "scene": "sce1",
        "root": root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "dataset_manifest_sha256": "3" * 64,
        "dataset_assets_sha256": "4" * 64,
        "contract_item_count": 5,
        "contract_item_ids": profile.contract_item_ids,
        "preview_condition_group_count": 20,
        "observation_count": 50,
        "referenced_assets": set(signals.values()),
    }


@pytest.fixture(scope="module")
def conversation_build(tmp_path_factory) -> tuple[Path, dict[str, dict]]:
    build = tmp_path_factory.mktemp("conversation-build")
    datasets = {
        scene: _build_conversation_dataset(build, scene)
        for scene in SCENES
    }
    return build, datasets


def _snapshot(dataset: dict, identifier: str) -> dict:
    contract_by_id = {
        row["id"]: row
        for row in background_noise_scene_items(
            "WB-wired",
            dataset["scene"],
        )
    }
    return scene_condition_batch._build_snapshot(
        dataset,
        contract_by_id,
        identifier,
        ALGORITHM_CLUSTER,
        {"profile": "vcp_default"},
        CATALOG_SHA256,
        "c" * 64,
    )


@pytest.mark.parametrize("scene", SCENES)
def test_v2_store_normalizes_four_scene_100_case_280_observation(
    conversation_build,
    monkeypatch,
    scene: str,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    profile = profile_for_dataset(datasets[scene]["dataset_id"])

    loaded = acceptance_scene_condition_store.load_scene_condition_dataset(
        profile.dataset_id
    )

    assert loaded["schema_version"] == "scene-condition-preview-v2"
    assert loaded["dataset_kind"] == "background_noise_conversation_preview"
    assert loaded["mode"] == "WB-wired"
    assert loaded["scene"] == scene
    assert loaded["preview_condition_group_count"] == 100
    assert loaded["preview_count"] == 100
    assert loaded["case_count"] == 100
    assert loaded["speech_case_count"] == 80
    assert loaded["cng_case_count"] == 20
    assert loaded["observation_count"] == 280
    assert loaded["conversation_counts"] == profile.conversation_counts
    assert Counter(
        case["provenance"]["conversation_mode"]
        for case in loaded["cases"]
    ) == Counter({mode: 20 for mode in CONVERSATION_MODES})


@pytest.mark.parametrize("scene", SCENES)
def test_v2_snapshot_keeps_batch_v1_but_uses_profile_counts(
    conversation_build,
    scene: str,
) -> None:
    _, datasets = conversation_build
    snapshot = _snapshot(datasets[scene], f"{SCENES.index(scene) + 1:032x}")

    scene_condition_batch.validate_snapshot(snapshot)

    assert snapshot["schema_version"] == "scene-condition-batch-v1"
    assert snapshot["dataset_schema_version"] == "scene-condition-preview-v2"
    assert snapshot["selected_scene"] == scene
    assert snapshot["case_count"] == 100
    assert snapshot["speech_case_count"] == 80
    assert snapshot["cng_case_count"] == 20
    assert snapshot["observation_count"] == 280
    assert snapshot["progress"]["total_cases"] == 100
    assert snapshot["progress"]["total_items"] == 280
    assert Counter(
        case["provenance"]["conversation_mode"]
        for case in snapshot["cases"]
    ) == Counter({mode: 20 for mode in CONVERSATION_MODES})
    assert Counter(
        len(case["contract_item_ids"])
        for case in snapshot["cases"]
    ) == Counter({3: 80, 2: 20})


def test_legacy_store_and_snapshot_still_read_20_cases_50_observations(
    tmp_path,
    monkeypatch,
) -> None:
    build = tmp_path / "legacy-build"
    dataset = _build_legacy_dataset(build)
    monkeypatch.setattr(service, "BUILD", build)

    loaded = acceptance_scene_condition_store.load_scene_condition_dataset(
        dataset["dataset_id"]
    )
    snapshot = _snapshot(dataset, "a" * 32)
    scene_condition_batch.validate_snapshot(snapshot)

    assert loaded["schema_version"] == "scene-condition-preview-v1"
    assert loaded["preview_condition_group_count"] == 20
    assert loaded["observation_count"] == 50
    assert loaded["conversation_counts"] == {}
    assert snapshot["schema_version"] == "scene-condition-batch-v1"
    assert snapshot["observation_count"] == 50
    assert len(snapshot["cases"]) == 20
    assert len(snapshot["items"]) == 50


def test_v2_store_rejects_wrong_counts_hash_and_path(
    conversation_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    source = datasets["sce1"]["root"]

    wrong_counts_root = tmp_path / "wrong-counts"
    shutil.copytree(source, wrong_counts_root)
    manifest = json.loads(
        (wrong_counts_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["conversation_counts"]["near_single"] = 19
    _write_json(wrong_counts_root / "manifest.json", manifest)
    monkeypatch.setattr(
        acceptance_scene_condition_store,
        "dataset_root",
        lambda dataset_id: wrong_counts_root,
    )
    with pytest.raises(
        acceptance_scene_condition_store.SceneConditionDatasetError,
        match="conversation_counts",
    ):
        acceptance_scene_condition_store._load_scene_condition_dataset_uncached(
            datasets["sce1"]["dataset_id"]
        )

    wrong_path_root = tmp_path / "wrong-path"
    shutil.copytree(source, wrong_path_root)
    manifest = json.loads(
        (wrong_path_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["cases"][0]["signals"]["reference"] = "../escape.wav"
    _write_json(wrong_path_root / "manifest.json", manifest)
    monkeypatch.setattr(
        acceptance_scene_condition_store,
        "dataset_root",
        lambda dataset_id: wrong_path_root,
    )
    with pytest.raises(
        acceptance_scene_condition_store.SceneConditionDatasetError,
        match="unsafe|asset path",
    ):
        acceptance_scene_condition_store._load_scene_condition_dataset_uncached(
            datasets["sce1"]["dataset_id"]
        )

    wrong_hash_root = tmp_path / "wrong-hash"
    shutil.copytree(source, wrong_hash_root)
    assets = json.loads(
        (wrong_hash_root / "assets.json").read_text(encoding="utf-8")
    )
    assets["assets/audio/reference.wav"]["sha256"] = "0" * 64
    _write_json(wrong_hash_root / "assets.json", assets)
    monkeypatch.setattr(
        acceptance_scene_condition_store,
        "dataset_root",
        lambda dataset_id: wrong_hash_root,
    )
    with pytest.raises(
        acceptance_scene_condition_store.SceneConditionDatasetError,
        match="hash",
    ):
        acceptance_scene_condition_store._load_scene_condition_dataset_uncached(
            datasets["sce1"]["dataset_id"]
        )


def test_validated_dataset_rejects_wrong_scene_mode_and_conversation_counts(
    conversation_build,
    monkeypatch,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets["sce1"]
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_load_dataset",
        lambda dataset_id: copy.deepcopy(dataset),
    )

    with pytest.raises(ValueError, match="不一致"):
        scene_condition_batch._validated_dataset(
            dataset["dataset_id"],
            requested_scene="sce2",
        )

    wrong_mode = copy.deepcopy(dataset)
    wrong_mode["mode"] = "NB-wired"
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_load_dataset",
        lambda dataset_id: copy.deepcopy(wrong_mode),
    )
    with pytest.raises(ValueError, match="不一致"):
        scene_condition_batch._validated_dataset(dataset["dataset_id"])

    wrong_counts = copy.deepcopy(dataset)
    wrong_counts["conversation_counts"]["transition"] = 19
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_load_dataset",
        lambda dataset_id: copy.deepcopy(wrong_counts),
    )
    with pytest.raises(ValueError, match="conversation_counts"):
        scene_condition_batch._validated_dataset(dataset["dataset_id"])

    partial_tx = copy.deepcopy(dataset)
    near_case = next(
        case
        for case in partial_tx["cases"]
        if case["provenance"]["conversation_mode"] == "near_single"
    )
    near_case["provenance"]["analysis_start_s"] = 1.0
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_load_dataset",
        lambda dataset_id: copy.deepcopy(partial_tx),
    )
    with pytest.raises(ValueError, match="完整 12 秒"):
        scene_condition_batch._validated_dataset(dataset["dataset_id"])


def test_snapshot_rejects_cross_scene_row_and_identity_mutations(
    conversation_build,
) -> None:
    _, datasets = conversation_build
    snapshot = _snapshot(datasets["sce1"], "b" * 32)

    wrong_scene = copy.deepcopy(snapshot)
    wrong_scene["selected_scene"] = "sce2"
    with pytest.raises(ValueError):
        scene_condition_batch.validate_snapshot(wrong_scene)

    wrong_mode = copy.deepcopy(snapshot)
    wrong_mode["selected_mode"] = "NB-wired"
    with pytest.raises(ValueError):
        scene_condition_batch.validate_snapshot(wrong_mode)

    wrong_name = copy.deepcopy(snapshot)
    wrong_name["dataset_name"] = "cross-scene-dataset"
    with pytest.raises(ValueError):
        scene_condition_batch.validate_snapshot(wrong_name)

    wrong_conversations = copy.deepcopy(snapshot)
    wrong_conversations["conversation_counts"]["noise_only"] = 19
    with pytest.raises(ValueError, match="conversation_counts"):
        scene_condition_batch.validate_snapshot(wrong_conversations)

    wrong_row = copy.deepcopy(snapshot)
    wrong_row["items"][0]["family"] = "cng_level"
    wrong_row["items"][0]["source_cell"] = "B27"
    with pytest.raises(ValueError, match="合同行"):
        scene_condition_batch.validate_snapshot(wrong_row)

    wrong_kind = copy.deepcopy(snapshot)
    speech_case = next(
        case
        for case in wrong_kind["cases"]
        if case["provenance"]["render_metadata"]["kind"] == "speech"
    )
    speech_case["contract_item_ids"] = [
        "WB-wired:B27",
        "WB-wired:B28",
    ]
    with pytest.raises(ValueError):
        scene_condition_batch.validate_snapshot(wrong_kind)


def _persist_queued_batch(dataset: dict, identifier: str) -> dict:
    snapshot = _snapshot(dataset, identifier)
    scene_condition_batch._create_batch_directories(identifier)
    scene_condition_batch._save(snapshot)
    return snapshot


def _fixture_marker(
    identifier: str,
    snapshot_sha256: str,
    *,
    reason: str = "synthetic test fixture",
) -> dict:
    return {
        "schema": "scene-condition-test-fixture-v1",
        "batch_id": identifier,
        "reason": reason,
        "snapshot_sha256": snapshot_sha256,
    }


def test_fixture_marker_hides_only_valid_synthetic_batches(
    tmp_path,
    monkeypatch,
) -> None:
    build = tmp_path / "marker-build"
    dataset = _build_conversation_dataset(build, "sce1")
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)

    real_identifier = "1" * 32
    fixture_identifier = "2" * 32
    _persist_queued_batch(dataset, real_identifier)
    _persist_queued_batch(dataset, fixture_identifier)
    snapshot_path = (
        scene_condition_batch._snapshot_directory(fixture_identifier)
        / "00000001.json"
    )
    _write_json(
        scene_condition_batch._batch_directory(fixture_identifier)
        / scene_condition_batch.FIXTURE_MARKER_NAME,
        _fixture_marker(
            fixture_identifier,
            _sha256_file(snapshot_path),
        ),
    )

    visible = scene_condition_batch.list_batches(limit=20)

    assert [row["batch_id"] for row in visible] == [real_identifier]


@pytest.mark.parametrize(
    "marker_mutation",
    ("wrong_reason", "wrong_snapshot_sha256", "wrong_batch_id", "extra_field"),
)
def test_invalid_fixture_marker_is_not_silently_hidden(
    tmp_path,
    monkeypatch,
    marker_mutation: str,
) -> None:
    build = tmp_path / f"invalid-marker-{marker_mutation}"
    dataset = _build_conversation_dataset(build, "sce1")
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    identifier = "3" * 32
    _persist_queued_batch(dataset, identifier)
    snapshot_path = (
        scene_condition_batch._snapshot_directory(identifier)
        / "00000001.json"
    )
    marker = _fixture_marker(identifier, _sha256_file(snapshot_path))
    if marker_mutation == "wrong_reason":
        marker["reason"] = "not synthetic"
    elif marker_mutation == "wrong_snapshot_sha256":
        marker["snapshot_sha256"] = "0" * 64
    elif marker_mutation == "wrong_batch_id":
        marker["batch_id"] = "4" * 32
    elif marker_mutation == "extra_field":
        marker["unexpected"] = True
    _write_json(
        scene_condition_batch._batch_directory(identifier)
        / scene_condition_batch.FIXTURE_MARKER_NAME,
        marker,
    )

    with pytest.raises(ValueError, match="fixture marker"):
        scene_condition_batch.list_batches(limit=20)


def _write_diagnostic_audio(
    root: Path,
    *,
    tx_muted: bool = False,
    delay_samples: int = 80,
) -> tuple[dict[str, Path], Path]:
    rng = np.random.default_rng(19)
    reference = rng.normal(scale=0.08, size=RATE * 12)
    noise = rng.normal(scale=0.004, size=RATE * 12)
    far = rng.normal(scale=0.05, size=RATE * 12)
    microphone = reference + far * 0.3 + noise
    if tx_muted:
        tx = np.zeros_like(reference)
    else:
        tx = np.r_[
            np.zeros(delay_samples),
            reference[:-delay_samples],
        ] * 0.5
    paths: dict[str, Path] = {}
    payloads = {
        "microphone": np.column_stack((microphone, microphone * 0.5)),
        "reference": reference,
        "far_end": far,
        "noise": noise,
    }
    for role, payload in payloads.items():
        path = root / f"{role}.wav"
        sf.write(path, payload, RATE, subtype="FLOAT")
        paths[role] = path
    tx_path = root / "tx.wav"
    sf.write(tx_path, tx, RATE, subtype="FLOAT")
    return paths, tx_path


def _diagnostic_case(mode: str) -> dict:
    return {
        "case_id": f"diag-{mode}",
        "provenance": {
            "conversation_mode": mode,
            "analysis_start_s": 4.0 if mode == "noise_only" else 0.0,
            "render_metadata": {
                "kind": EXPECTED_RENDER_KIND[mode],
                "duration_s": DURATION_S,
                "sample_rate_hz": RATE,
                "segments": _activity_segments(mode),
            },
        },
    }


def test_diagnostics_near_alignment_and_projection_are_explicit(
    tmp_path,
) -> None:
    paths, tx = _write_diagnostic_audio(tmp_path)
    case = _diagnostic_case("near_single")
    activity = validate_activity_metadata(case["provenance"])
    diagnostics = diagnose_conversation_case(
        case,
        paths,
        tx,
    )
    validate_diagnostics(
        diagnostics,
        expected_mode="near_single",
        expected_case_id=case["case_id"],
        expected_tx_sha256=_sha256_file(tx),
        expected_activity=activity,
    )
    segment = diagnostics["segments"][0]

    assert diagnostics["evidence_grade"] == "engineering_proxy"
    assert diagnostics["formal_compliance"] is False
    assert diagnostics["status"] == "review_required"
    assert segment["start_s"] == 0.0
    assert segment["end_s"] == 12.0
    assert segment["mode"] == "near"
    assert segment["alignment"]["lag_samples"] == 80
    assert segment["metrics"]["near_correlation"] == pytest.approx(
        1.0,
        abs=1e-3,
    )
    assert segment["metrics"]["near_gain_db"] == pytest.approx(
        20 * np.log10(0.5),
        abs=0.02,
    )
    assert segment["metrics"]["echo_reduction_db"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_diagnostics_mute_is_not_a_pass_and_returns_null(tmp_path) -> None:
    paths, tx = _write_diagnostic_audio(tmp_path, tx_muted=True)
    near = diagnose_conversation_case(
        _diagnostic_case("near_single"),
        paths,
        tx,
    )
    near_metrics = near["segments"][0]["metrics"]

    assert near["status"] == "not_measurable"
    assert near["review_required"] is False
    assert near_metrics["near_correlation"] is None
    assert near_metrics["near_gain_db"] is None
    assert near_metrics["tx_level_dbfs"] is None
    assert near_metrics["echo_reduction_db"] is None

    far = diagnose_conversation_case(
        _diagnostic_case("far_single"),
        paths,
        tx,
    )
    far_segment = far["segments"][0]
    assert far_segment["mode"] == "far"
    assert far_segment["status"] == "not_measurable"
    assert far_segment["tx_muted"] is True
    assert far_segment["metrics"]["echo_reduction_db"] is None
    assert far["status"] == "not_measurable"
    json.dumps(far, allow_nan=False)


def test_diagnostics_short_tx_is_explicitly_not_measurable(tmp_path) -> None:
    paths, tx = _write_diagnostic_audio(tmp_path)
    samples, rate = sf.read(tx, dtype="float64")
    sf.write(tx, samples[:rate], rate, subtype="FLOAT")

    case = _diagnostic_case("near_single")
    activity = validate_activity_metadata(case["provenance"])
    diagnostics = diagnose_conversation_case(
        case,
        paths,
        tx,
    )
    validate_diagnostics(
        diagnostics,
        expected_mode="near_single",
        expected_case_id=case["case_id"],
        expected_tx_sha256=_sha256_file(tx),
        expected_activity=activity,
    )
    segment = diagnostics["segments"][0]

    assert segment["tx_overlap_ratio"] == pytest.approx(1 / 12)
    assert segment["status"] == "not_measurable"
    assert segment["metrics"]["near_gain_db"] is None
    assert segment["metrics"]["near_correlation"] is None
    assert "重叠不足 95%" in segment["reason"]
    assert diagnostics["tx_duration_s"] == pytest.approx(1.0)
    json.dumps(diagnostics, allow_nan=False)


def test_diagnostics_transition_records_all_segments_and_no_formal_claim(
    tmp_path,
) -> None:
    paths, tx = _write_diagnostic_audio(tmp_path)
    diagnostics = diagnose_conversation_case(
        _diagnostic_case("transition"),
        paths,
        tx,
    )
    validate_diagnostics(diagnostics, expected_mode="transition")

    assert [
        segment["label"] for segment in diagnostics["segments"]
    ] == list(EXPECTED_SEGMENT_LABELS["transition"])
    assert [
        segment["mode"] for segment in diagnostics["segments"]
    ] == ["not_applicable", "far", "near", "far", "double_talk", "noise"]
    assert diagnostics["segments"][0]["status"] == "not_applicable"
    assert diagnostics["formal_compliance"] is False
    assert diagnostics["formal_test_status"] == "not_measured"
    json.dumps(diagnostics, allow_nan=False)


def _visual(source_sha256: str) -> dict:
    return {
        "source_sha256": source_sha256,
        "duration_s": 0.1,
        "sample_rate": RATE,
        "channels": 1,
        "peak": 0.0,
        "wave_scale": 1.0,
        "wave_min_i16": base64.b64encode(b"\0" * (720 * 2)).decode(),
        "wave_max_i16": base64.b64encode(b"\0" * (720 * 2)).decode(),
        "mel_i16": base64.b64encode(b"\0" * (40 * 240 * 2)).decode(),
        "mel_ceiling_db": -120.0,
    }


def _execution_harness(
    monkeypatch,
    dataset: dict,
    *,
    identifier: str,
    fail_case_id: str | None = None,
) -> tuple[dict, dict, list[dict]]:
    snapshot = _snapshot(dataset, identifier)
    scene_condition_batch._create_batch_directories(identifier)
    calculate_calls: list[dict] = []

    def run_cli(case, paths, destination, cluster):
        if fail_case_id is not None and case["id"] == fail_case_id:
            raise RuntimeError("controlled conversation case failure")
        destination.mkdir(parents=True, exist_ok=True)
        tx_path = destination / "tx.after.wav"
        sf.write(
            tx_path,
            np.zeros(FRAMES, dtype=np.float32),
            RATE,
            subtype="PCM_16",
        )
        input_sha256 = {
            role: _sha256_file(path)
            for role, path in paths.items()
        }
        output_sha256 = {"tx": _sha256_file(tx_path)}
        execution = {
            "source": "controlled-conversation-cli",
            "algorithm_cluster_id": snapshot["algorithm_cluster_id"],
            "configuration": copy.deepcopy(snapshot["configuration"]),
            "catalog_sha256": snapshot["catalog_sha256"],
            "cli_sha256": snapshot["cli_sha256"],
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
            "output_alignment": {
                "tx": {"analysis_filename": tx_path.name}
            },
        }
        return {"tx": tx_path}, execution

    def calculate(case, paths, outputs, execution, *, report_id):
        calculate_calls.append(copy.deepcopy(case))
        if not case["item_ids"]:
            return []
        return [
            {
                "item_id": item_id,
                "value": 1.0,
                "status": "review_required",
                "metric": "controlled_conversation_proxy",
                "reason": "controlled fake calculation",
            }
            for item_id in case["item_ids"]
        ]

    monkeypatch.setattr(acceptance_batch, "_run_cli", run_cli)
    monkeypatch.setattr(scene_condition_batch, "calculate_case_items", calculate)
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_visual_track",
        lambda path, source_sha256: _visual(source_sha256),
    )
    item_by_key = {
        (row["case_id"], row["item_id"]): row
        for row in snapshot["items"]
    }
    return snapshot, item_by_key, calculate_calls


def test_far_and_transition_skip_dnsmos_but_save_tx_and_diagnostics(
    conversation_build,
    monkeypatch,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets["sce1"]
    snapshot, item_by_key, calculate_calls = _execution_harness(
        monkeypatch,
        dataset,
        identifier="d" * 32,
    )
    selected: dict[str, dict] = {}
    for mode in CONVERSATION_MODES:
        case_record = next(
            case
            for case in snapshot["cases"]
            if case["provenance"]["conversation_mode"] == mode
        )
        selected[mode] = case_record
        scene_condition_batch._run_one_case(
            snapshot,
            dataset,
            case_record,
            item_by_key,
        )

    scene_condition_batch.validate_snapshot(snapshot)
    by_mode = {
        call["conversation_mode"]: call["item_ids"]
        for call in calculate_calls
    }
    assert by_mode["far_single"] == []
    assert by_mode["transition"] == []
    assert len(by_mode["near_single"]) == 3
    assert len(by_mode["double_talk"]) == 3
    assert len(by_mode["noise_only"]) == 2

    for mode in ("far_single", "transition"):
        case_record = selected[mode]
        rows = [
            row
            for row in snapshot["items"]
            if row["case_id"] == case_record["case_id"]
        ]
        assert all(row["status"] == "not_measured" for row in rows)
        assert all(row["value"] is None for row in rows)
        assert all(row["metric"] is None for row in rows)
        assert all("DNSMOS" in row["reason"] for row in rows)
        assert isinstance(case_record["conversation_diagnostics"], dict)
        assessment_path = (
            scene_condition_batch._case_directory(
                snapshot["batch_id"],
                case_record["case_id"],
            )
            / "assessment.json"
        )
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        assert assessment["provenance"] == case_record["provenance"]
        assert (
            assessment["conversation_diagnostics"]
            == case_record["conversation_diagnostics"]
        )


def test_completed_v2_diagnostics_reject_case_tx_bounds_and_mode_tampering(
    conversation_build,
    monkeypatch,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets["sce1"]
    snapshot, item_by_key, _ = _execution_harness(
        monkeypatch,
        dataset,
        identifier="e" * 32,
    )
    case_record = next(
        case
        for case in snapshot["cases"]
        if case["provenance"]["conversation_mode"] == "near_single"
    )
    scene_condition_batch._run_one_case(
        snapshot,
        dataset,
        case_record,
        item_by_key,
    )
    scene_condition_batch.validate_snapshot(snapshot)

    mutations = (
        (
            "case_id",
            "case_id differs from its case",
            lambda diagnostics: diagnostics.update(case_id="other-case"),
        ),
        (
            "tx_sha256",
            "tx_sha256 differs from its output",
            lambda diagnostics: diagnostics.update(tx_sha256="0" * 64),
        ),
        (
            "segment_bounds",
            "segment bounds differ from provenance",
            lambda diagnostics: diagnostics["segments"][0].update(
                start_s=0.25
            ),
        ),
        (
            "segment_mode",
            "segment mode differs from its label",
            lambda diagnostics: diagnostics["segments"][0].update(
                mode="far"
            ),
        ),
        (
            "source_duration",
            "source/TX duration is invalid",
            lambda diagnostics: diagnostics.update(source_duration_s=11.5),
        ),
    )
    for _, expected_error, mutate in mutations:
        tampered = copy.deepcopy(snapshot)
        target = next(
            case
            for case in tampered["cases"]
            if case["case_id"] == case_record["case_id"]
        )
        mutate(target["conversation_diagnostics"])
        with pytest.raises(ValueError, match=expected_error):
            scene_condition_batch.validate_snapshot(tampered)

    assessment_case = copy.deepcopy(case_record)
    assessment_case["conversation_diagnostics"]["case_id"] = "other-case"
    with pytest.raises(
        scene_condition_batch.ArtifactIntegrityError,
        match="v2 诊断身份绑定无效",
    ):
        scene_condition_batch._verified_assessment(
            snapshot,
            assessment_case,
            scene_condition_batch._case_directory(
                snapshot["batch_id"],
                case_record["case_id"],
            ),
        )


def test_one_conversation_case_failure_does_not_invalidate_another_case(
    conversation_build,
    monkeypatch,
) -> None:
    build, datasets = conversation_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets["sce1"]
    failed_case = next(
        case
        for case in dataset["cases"]
        if case["provenance"]["conversation_mode"] == "far_single"
    )
    snapshot, item_by_key, _ = _execution_harness(
        monkeypatch,
        dataset,
        identifier="f" * 32,
        fail_case_id=failed_case["case_id"],
    )
    failed_record = next(
        case
        for case in snapshot["cases"]
        if case["case_id"] == failed_case["case_id"]
    )
    successful_record = next(
        case
        for case in snapshot["cases"]
        if case["provenance"]["conversation_mode"] == "near_single"
    )

    scene_condition_batch._run_one_case(
        snapshot,
        dataset,
        failed_record,
        item_by_key,
    )
    scene_condition_batch._run_one_case(
        snapshot,
        dataset,
        successful_record,
        item_by_key,
    )

    failed_rows = [
        row
        for row in snapshot["items"]
        if row["case_id"] == failed_record["case_id"]
    ]
    successful_rows = [
        row
        for row in snapshot["items"]
        if row["case_id"] == successful_record["case_id"]
    ]
    assert failed_record["status"] == "failed"
    assert successful_record["status"] == "completed"
    assert all(row["status"] == "invalid" for row in failed_rows)
    assert all(
        "controlled conversation case failure" in row["reason"]
        for row in failed_rows
    )
    assert all(
        row["status"] == "review_required" for row in successful_rows
    )
    scene_condition_batch.validate_snapshot(snapshot)
