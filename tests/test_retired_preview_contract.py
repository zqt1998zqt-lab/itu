from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import acceptance_scene_condition_api as scene_api
from server.acoustic.dataset_revisions_v2 import (
    DATASET_REVISION,
    revision_scene_profile,
    revision_stress_profile,
)
from server.acoustic.scene_condition_profiles import (
    profile_for_scene as retired_v1_profile_for_scene,
)


PROJECT = Path(__file__).resolve().parents[1]
RECEIPT = (
    PROJECT
    / "benchmarks"
    / "fixtures"
    / "retired-acoustic-metadata-v1"
    / "retirement.json"
)
INVENTORY = (
    PROJECT
    / "build"
    / "dataset-revision-20261117"
    / "cleanup-inventory-20261117-readonly"
    / "review-manifest.json"
)
SCENE_DATASETS = "/api/alignment/acceptance/scene-condition-datasets"
SCENE_PREVIEW = "/api/alignment/acceptance/scene-condition-preview"
STRESS_DATASETS = "/api/alignment/acceptance/stress-datasets"
OLD_DATASET_GROUPS = {
    "r1_scene",
    "r1_stress",
    "old_scene_v2",
    "old_scene_v1",
    "old_stress_v1",
}


def _old_entries() -> list[dict]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in inventory["entries"]
        if entry.get("group") in OLD_DATASET_GROUPS
    ]
    assert len(entries) == 100
    return entries


def _all_retired_entries() -> list[dict]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in inventory["entries"]
        if isinstance(entry.get("dataset_id"), str)
    ]
    assert len(entries) == 103
    return entries


def _controlled_visuals(dataset: dict, *, waveform_only: bool = False) -> dict:
    return {
        "schema": "controlled-current-r2-visual-v1",
        "waveform_only": waveform_only,
        "cases": {},
    }


def _replacement_for(entry: dict) -> str:
    if entry["group"] in {"r1_stress", "old_stress_v1"}:
        return revision_stress_profile(entry["condition"], entry["mode"]).dataset_id
    return revision_scene_profile(entry["condition"], entry["mode"]).dataset_id


def _assert_retired(
    response,
    *,
    retired_id: str,
    replacement_id: str,
) -> dict:
    assert response.status_code == 410, response.text
    payload = response.json()
    assert isinstance(payload, dict), payload
    assert payload["dataset_id"] == retired_id
    assert payload["replacement_dataset_id"] == replacement_id
    assert isinstance(payload["detail"], str) and payload["detail"]
    return payload


@pytest.fixture(scope="module")
def retirement_receipt() -> dict:
    assert RECEIPT.is_file(), (
        "retirement receipt is required; do not treat a legacy 200 as current: "
        f"{RECEIPT}"
    )
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "itu-dataset-retirement-v1"
    assert receipt["active_revision"] == DATASET_REVISION
    retired_ids = set(receipt["retired_dataset_ids"])
    old_entries = _old_entries()
    all_entries = _all_retired_entries()
    assert len(retired_ids) == 103
    assert retired_ids == {
        entry["dataset_id"] for entry in all_entries
    }
    for entry in old_entries:
        assert receipt["replacement_dataset_ids"][entry["dataset_id"]] == (
            _replacement_for(entry)
        )
    for entry in all_entries:
        if entry.get("group") is None:
            assert receipt["replacement_dataset_ids"][entry["dataset_id"]] == (
                "fa11d83bbeb6480c98a9eb3bae35045f"
            )
    return receipt


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(scene_api, "_preview_visuals", _controlled_visuals)
    return TestClient(app)


@pytest.mark.parametrize("scene", ("sce1", "sce2", "sce3", "sce4"))
def test_legacy_scene_query_is_retired_with_current_replacement(
    client: TestClient,
    retirement_receipt: dict,
    scene: str,
) -> None:
    assert retirement_receipt
    replacement = revision_scene_profile(scene, "WB-wired")
    response = client.get(
        SCENE_DATASETS,
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": scene,
            "version": "legacy",
        },
    )
    retired_id = retired_v1_profile_for_scene(scene).dataset_id
    payload = _assert_retired(
        response,
        retired_id=retired_id,
        replacement_id=replacement.dataset_id,
    )
    assert payload["replacement_dataset_id"] == replacement.dataset_id

    current = client.get(
        SCENE_DATASETS,
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": scene,
            "version": "current",
        },
    )
    assert current.status_code == 200, current.text
    assert current.json()["datasets"][0]["dataset_id"] == replacement.dataset_id
    assert current.json()["formal_test_status"] == "not_measured"


@pytest.mark.parametrize(
    "entry",
    [
        next(entry for entry in _old_entries() if entry["group"] == "old_scene_v1" and entry["mode"] == "WB-wired"),
        next(entry for entry in _old_entries() if entry["group"] == "r1_scene" and entry["mode"] == "WB-wired"),
        next(entry for entry in _old_entries() if entry["group"] == "old_scene_v2" and entry["mode"] == "WB-wired"),
        next(entry for entry in _old_entries() if entry["group"] == "old_stress_v1" and entry["mode"] == "WB-wired"),
        next(entry for entry in _old_entries() if entry["group"] == "r1_stress" and entry["mode"] == "WB-wired"),
    ],
    ids=lambda entry: f"{entry['group']}-{entry['dataset_id']}",
)
def test_retired_preview_and_asset_routes_never_read_old_stores(
    client: TestClient,
    retirement_receipt: dict,
    monkeypatch: pytest.MonkeyPatch,
    entry: dict,
) -> None:
    assert retirement_receipt
    old_root = Path(entry["path"])
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_stat = Path.stat
    denied: list[str] = []

    def reject(path: Path, operation: str) -> None:
        candidate = os.path.abspath(os.fspath(path))
        root = os.path.abspath(os.fspath(old_root))
        if candidate == root or candidate.startswith(root + os.sep):
            denied.append(f"{operation}:{path}")
            raise PermissionError(f"retired store access is forbidden: {path}")

    def guarded_read_bytes(path: Path, *args, **kwargs):
        reject(path, "read_bytes")
        return original_read_bytes(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        reject(path, "read_text")
        return original_read_text(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        reject(path, "stat")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    replacement_id = _replacement_for(entry)

    if entry["group"] in {"r1_stress", "old_stress_v1"}:
        preview_path = f"{STRESS_DATASETS}/{entry['dataset_id']}/preview"
        asset_path = (
            f"{STRESS_DATASETS}/{entry['dataset_id']}"
            "/cases/retired-case/inputs/microphone"
        )
    else:
        preview_path = SCENE_PREVIEW
        asset_path = (
            f"{SCENE_DATASETS}/{entry['dataset_id']}"
            "/cases/retired-case/inputs/microphone"
        )

    preview_params = (
        None
        if entry["group"] in {"r1_stress", "old_stress_v1"}
        else {
            "dataset_id": entry["dataset_id"],
            "mode": entry["mode"],
            "scene": entry["condition"],
        }
    )
    preview = client.get(preview_path, params=preview_params)
    _assert_retired(
        preview,
        retired_id=entry["dataset_id"],
        replacement_id=replacement_id,
    )
    asset = client.get(asset_path)
    _assert_retired(
        asset,
        retired_id=entry["dataset_id"],
        replacement_id=replacement_id,
    )
    assert denied == [], denied


def test_current_r2_preview_counts_assets_cross_scene_and_invalid_role(
    client: TestClient,
    retirement_receipt: dict,
) -> None:
    assert retirement_receipt
    scene_profile = revision_scene_profile("sce1", "WB-wired")
    preview = client.get(
        SCENE_PREVIEW,
        params={
            "dataset_id": scene_profile.dataset_id,
            "mode": "WB-wired",
            "scene": "sce1",
            "visual_detail": "waveform",
        },
    )
    assert preview.status_code == 200, preview.text
    payload = preview.json()
    assert payload["dataset_id"] == scene_profile.dataset_id
    assert len(payload["cases"]) == 100
    assert payload["observation_count"] == 280
    assert payload["formal_test_status"] == "not_measured"
    case = payload["cases"][0]
    microphone_sha = case["signals_sha256"]["microphone"]
    audio = client.get(
        f"{SCENE_DATASETS}/{scene_profile.dataset_id}"
        f"/cases/{case['case_id']}/inputs/microphone"
    )
    assert audio.status_code == 200, audio.text
    assert audio.content[:4] == b"RIFF"
    assert hashlib.sha256(audio.content).hexdigest() == microphone_sha

    invalid_role = client.get(
        f"{SCENE_DATASETS}/{scene_profile.dataset_id}"
        f"/cases/{case['case_id']}/inputs/not-a-role"
    )
    assert invalid_role.status_code == 404
    cross_scene = client.get(
        SCENE_PREVIEW,
        params={
            "dataset_id": scene_profile.dataset_id,
            "mode": "WB-wired",
            "scene": "sce2",
        },
    )
    assert cross_scene.status_code == 422

    stress_profile = revision_stress_profile("echo", "WB-wired")
    stress_preview = client.get(
        f"{STRESS_DATASETS}/{stress_profile.dataset_id}/preview"
    )
    assert stress_preview.status_code == 200, stress_preview.text
    stress_payload = stress_preview.json()
    assert len(stress_payload["cases"]) == 100
    assert stress_payload["observation_count"] == 100
    assert stress_payload["formal_test_status"] == "not_measured"
    assert stress_payload["category_counts"] == stress_profile.category_counts
    stress_case = stress_payload["cases"][0]
    stress_audio = client.get(
        f"{STRESS_DATASETS}/{stress_profile.dataset_id}"
        f"/cases/{stress_case['case_id']}/inputs/microphone"
    )
    assert stress_audio.status_code == 200, stress_audio.text
    assert hashlib.sha256(stress_audio.content).hexdigest() == (
        stress_case["signals_sha256"]["microphone"]
    )
    stress_invalid_role = client.get(
        f"{STRESS_DATASETS}/{stress_profile.dataset_id}"
        f"/cases/{stress_case['case_id']}/inputs/not-a-role"
    )
    assert stress_invalid_role.status_code == 404


@pytest.mark.parametrize(
    "source_name",
    [
        "server/acoustic/dataset_revisions.py",
        "server/acoustic/dataset_revisions_v2.py",
    ],
)
def test_current_r2_source_hash_drift_fails_before_preview(
    client: TestClient,
    retirement_receipt: dict,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    assert retirement_receipt
    source_path = PROJECT / source_name
    original_read_bytes = Path.read_bytes

    def drifted(path: Path, *args, **kwargs):
        value = original_read_bytes(path, *args, **kwargs)
        return value + b"\n# drift\n" if path == source_path else value

    monkeypatch.setattr(Path, "read_bytes", drifted)
    monkeypatch.setattr(
        scene_api,
        "_preview_visuals",
        lambda dataset, waveform_only=False: pytest.fail(
            "source drift reached visualization"
        ),
    )
    profile = revision_scene_profile("sce1", "WB-wired")
    response = client.get(
        SCENE_PREVIEW,
        params={
            "dataset_id": profile.dataset_id,
            "mode": "WB-wired",
            "scene": "sce1",
        },
    )
    assert response.status_code == 503, response.text
