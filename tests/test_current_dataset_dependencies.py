from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import acceptance_scene_condition_api as scene_api
from server.acoustic import service
from server.acoustic.dataset_revisions_v2 import (
    DATASET_REVISION,
    REVISION_SCENE_PROFILES,
    REVISION_STRESS_PROFILES,
)


PROJECT = Path(__file__).resolve().parents[1]
INVENTORY = (
    PROJECT
    / "build"
    / "dataset-revision-20261117"
    / "cleanup-inventory-20261117-readonly"
    / "review-manifest.json"
)
ACTIVATION = PROJECT / "build" / "dataset-revision-20261117" / "ACTIVATION-r2.json"
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
REQUIRED_PRESERVED_SOURCES = {
    "server/acoustic/dataset_revisions.py",
    "server/acoustic/dataset_revisions_v2.py",
    "scripts/create-revised-acoustic-datasets.py",
    "scripts/create-revised-acoustic-datasets-v2.py",
    "scripts/dataset_revision_observation.py",
    "scripts/dataset_revision_signal_design.py",
    "scripts/dataset_revision_sources.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_dataset_roots() -> list[Path]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    roots = [
        Path(entry["path"])
        for entry in inventory["entries"]
        if entry.get("group") in OLD_DATASET_GROUPS
    ]
    assert len(roots) == 100, len(roots)
    assert len(set(roots)) == 100
    return roots


def _is_under(path: Path, root: Path) -> bool:
    candidate = os.path.abspath(os.fspath(path))
    prefix = os.path.abspath(os.fspath(root))
    return candidate == prefix or candidate.startswith(prefix + os.sep)


def _controlled_visuals(dataset: dict, *, waveform_only: bool = False) -> dict:
    return {
        "schema": "controlled-current-r2-visual-v1",
        "waveform_only": waveform_only,
        "cases": {},
    }


@pytest.fixture()
def current_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    roots = _old_dataset_roots()
    denied: list[str] = []
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_stat = Path.stat

    def reject(path: Path, operation: str) -> None:
        if any(_is_under(path, root) for root in roots):
            denied.append(f"{operation}:{path}")
            raise PermissionError(f"retired dataset store access is forbidden: {path}")

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
    monkeypatch.setattr(scene_api, "_preview_visuals", _controlled_visuals)
    client = TestClient(app)
    yield client
    assert denied == [], denied


def test_current_six_pages_never_read_any_of_the_100_retired_roots(
    current_client: TestClient,
) -> None:
    assert DATASET_REVISION == "measurement-20261117-r2"
    assert len(REVISION_SCENE_PROFILES) == 32
    assert len(REVISION_STRESS_PROFILES) == 16

    for scene in ("sce1", "sce2", "sce3", "sce4"):
        profile = REVISION_SCENE_PROFILES[(scene, "WB-wired")]
        listed = current_client.get(
            SCENE_DATASETS,
            params={
                "scope": "background_noise",
                "mode": "WB-wired",
                "scene": scene,
            },
        )
        assert listed.status_code == 200, listed.text
        payload = listed.json()
        dataset = payload["datasets"][0]
        assert dataset["dataset_id"] == profile.dataset_id
        assert dataset["mode"] == "WB-wired"
        assert dataset["scene"] == scene
        assert dataset["preview_condition_group_count"] == 100
        assert dataset["observation_count"] == 280
        assert dataset["conversation_counts"] == profile.conversation_counts

        preview = current_client.get(
            SCENE_PREVIEW,
            params={
                "dataset_id": profile.dataset_id,
                "mode": "WB-wired",
                "scene": scene,
                "visual_detail": "waveform",
            },
        )
        assert preview.status_code == 200, preview.text
        preview_payload = preview.json()
        assert preview_payload["dataset_id"] == profile.dataset_id
        assert preview_payload["schema_version"] == "scene-condition-preview-v2"
        assert len(preview_payload["cases"]) == 100
        assert preview_payload["observation_count"] == 280

    for scope in ("echo", "double_talk"):
        profile = REVISION_STRESS_PROFILES[(scope, "WB-wired")]
        listed = current_client.get(
            STRESS_DATASETS,
            params={"scope": scope, "mode": "WB-wired"},
        )
        assert listed.status_code == 200, listed.text
        payload = listed.json()
        assert payload["dataset_id"] == profile.dataset_id
        assert payload["test_scope"] == scope
        assert payload["mode"] == "WB-wired"
        assert payload["preview_condition_group_count"] == 100
        assert payload["observation_count"] == 100
        assert payload["category_counts"] == profile.category_counts

        preview = current_client.get(
            f"{STRESS_DATASETS}/{profile.dataset_id}/preview"
        )
        assert preview.status_code == 200, preview.text
        preview_payload = preview.json()
        assert preview_payload["dataset_id"] == profile.dataset_id
        assert preview_payload["schema_version"] == "acoustic-stress-preview-v1"
        assert len(preview_payload["cases"]) == 100
        assert preview_payload["observation_count"] == 100


def test_all_r2_bindings_preserve_and_verify_legacy_source_filenames() -> None:
    activation = json.loads(ACTIVATION.read_text(encoding="utf-8"))
    assert activation["revision"] == DATASET_REVISION
    assert len(activation["datasets"]) == 48

    for row in activation["datasets"]:
        root_name = (
            "acoustic-scene-condition-datasets"
            if row["kind"] == "scene"
            else "acoustic-stress-datasets"
        )
        binding_path = (
            Path(service.BUILD)
            / root_name
            / row["dataset_id"]
            / "runtime-contract-binding.json"
        )
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        sources = binding["source_sha256"]
        assert REQUIRED_PRESERVED_SOURCES.issubset(sources)
        for relative in REQUIRED_PRESERVED_SOURCES:
            assert _sha256(PROJECT / relative) == sources[relative]
