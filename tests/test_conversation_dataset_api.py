from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import acceptance_scene_condition_api as api, service
from server.acoustic.scene_conversation_profiles import CONVERSATION_PROFILES

DATASETS = "/api/alignment/acceptance/scene-condition-datasets"
PREVIEW = "/api/alignment/acceptance/scene-condition-preview"


@pytest.fixture(scope="module")
def client():
    for profile in CONVERSATION_PROFILES.values():
        if not (Path(service.BUILD) / "acoustic-scene-condition-datasets" / profile.dataset_id / "manifest.json").is_file():
            pytest.skip("generated conversation datasets are required")
    return TestClient(app)


@pytest.mark.parametrize("scene", tuple(CONVERSATION_PROFILES))
def test_active_scene_exposes_100_audio_cases_with_five_conversation_modes(client, scene):
    profile = CONVERSATION_PROFILES[scene]
    response = client.get(DATASETS, params={"scope": "background_noise", "mode": "WB-wired", "scene": scene})
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["datasets"][0]["dataset_id"] == profile.dataset_id
    assert value["preview_condition_group_count"] == 100
    assert value["observation_count"] == 280
    assert value["conversation_counts"] == profile.conversation_counts
    response = client.get(PREVIEW, params={"dataset_id": profile.dataset_id, "mode": "WB-wired", "scene": scene})
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["schema_version"] == "scene-condition-preview-v2"
    assert preview["dataset_kind"] == "background_noise_conversation_preview"
    assert len(preview["cases"]) == 100
    assert len(preview["visuals"]["cases"]) == 100
    assert Counter(case["provenance"]["conversation_mode"] for case in preview["cases"]) == profile.conversation_counts
    assert sum(len(case["contract_item_ids"]) for case in preview["cases"]) == 280
    assert preview["standard_compliance"] is False
    for case in preview["cases"]:
        assert case["conditions"] == profile.conditions
        assert case["provenance"]["engineering_preview"] is True
        assert set(case["signals"]) == {"reference", "microphone", "noise", "far_end"}
        for role in case["signals"]:
            track = preview["visuals"]["cases"][case["case_id"]][role]
            assert track["source_sha256"] == case["signals_sha256"][role]


@pytest.mark.parametrize("scene", tuple(CONVERSATION_PROFILES))
def test_previous_twenty_case_dataset_is_still_available_explicitly(client, scene):
    response = client.get(DATASETS, params={"scope": "background_noise", "mode": "WB-wired", "scene": scene, "version": "legacy"})
    assert response.status_code == 200
    assert response.json()["preview_condition_group_count"] == 20
    assert response.json()["observation_count"] == 50


def test_v2_binding_source_drift_is_rejected_before_visualization(client, monkeypatch):
    profile = CONVERSATION_PROFILES["sce1"]
    original = Path.read_bytes
    def drifted(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return value + b"\n# drift\n" if path.name == "scene_conversation_profiles.py" else value
    monkeypatch.setattr(Path, "read_bytes", drifted)
    monkeypatch.setattr(api, "_preview_visuals", lambda _: pytest.fail("invalid source reached visualization"))
    response = client.get(PREVIEW, params={"dataset_id": profile.dataset_id, "mode": "WB-wired", "scene": "sce1"})
    assert response.status_code == 503


def test_v2_binding_cannot_be_borrowed_from_another_scene(client, monkeypatch):
    profile = CONVERSATION_PROFILES["sce1"]
    binding = Path(service.BUILD) / "acoustic-scene-condition-datasets" / profile.dataset_id / "runtime-contract-binding.json"
    original = Path.read_text
    def forged(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == binding:
            data = json.loads(value)
            data["dataset_id"] = CONVERSATION_PROFILES["sce2"].dataset_id
            return json.dumps(data)
        return value
    monkeypatch.setattr(Path, "read_text", forged)
    response = client.get(PREVIEW, params={"dataset_id": profile.dataset_id, "mode": "WB-wired", "scene": "sce1"})
    assert response.status_code == 503


# The current conversation contract is R2.  This local alias changes only the
# test module's view; the frozen V1 profile module and its source bytes remain
# untouched for the runtime source-hash bindings.
from server.acoustic.dataset_revisions_v2 import (
    SCENES as CURRENT_CONVERSATION_SCENES,
    revision_scene_profile,
)
from server.acoustic.scene_condition_profiles import (
    profile_for_scene as retired_v1_profile_for_scene,
)

CONVERSATION_PROFILES = {
    scene: revision_scene_profile(scene, "WB-wired")
    for scene in CURRENT_CONVERSATION_SCENES
}


@pytest.mark.parametrize("scene", tuple(CONVERSATION_PROFILES))
def test_previous_twenty_case_dataset_is_still_available_explicitly(
    client,
    scene,
):
    profile = CONVERSATION_PROFILES[scene]
    response = client.get(
        DATASETS,
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": scene,
            "version": "legacy",
        },
    )
    assert response.status_code == 410, response.text
    payload = response.json()
    assert isinstance(payload["detail"], str) and payload["detail"]
    assert payload["dataset_id"] == retired_v1_profile_for_scene(scene).dataset_id
    assert payload["replacement_dataset_id"] == profile.dataset_id

    replacement = client.get(
        DATASETS,
        params={
            "scope": "background_noise",
            "mode": "WB-wired",
            "scene": scene,
            "version": "current",
        },
    )
    assert replacement.status_code == 200, replacement.text
    replacement_payload = replacement.json()
    assert replacement_payload["datasets"][0]["dataset_id"] == profile.dataset_id
    assert replacement_payload["formal_test_status"] == "not_measured"
