from __future__ import annotations

from pathlib import Path

import pytest

from server.acoustic import scene_condition_batch, service
from server.acoustic.acceptance_scene_condition_store import (
    load_scene_condition_dataset,
)
from server.acoustic.acceptance_scope import background_noise_scene_items
from server.acoustic.scene_condition_profiles import (
    SCENE_CONDITION_PROFILES,
    profile_for_dataset,
    profile_for_scene,
)


EXPECTED_SCENES = ("sce1", "sce2", "sce3", "sce4")
EXPECTED_SNR_DB = {
    "sce1": 15.0,
    "sce2": 10.0,
    "sce3": 6.0,
    "sce4": 3.0,
}


def test_scene_profiles_match_the_real_wb_wired_contract_rows() -> None:
    for scene in EXPECTED_SCENES:
        profile = profile_for_scene(scene)
        rows = background_noise_scene_items("WB-wired", scene)
        assert profile.scene == scene
        assert profile.contract_item_ids == tuple(row["id"] for row in rows)
        assert profile.contract_item_count == 5
        assert profile.snr_db == EXPECTED_SNR_DB[scene]
        assert profile.conditions["scene"] == scene
        assert profile_for_dataset(profile.dataset_id) == profile

    dataset_ids = {
        profile.dataset_id for profile in SCENE_CONDITION_PROFILES.values()
    }
    dataset_names = {
        profile.dataset_name for profile in SCENE_CONDITION_PROFILES.values()
    }
    assert len(dataset_ids) == 4
    assert len(dataset_names) == 4


@pytest.mark.parametrize("scene", EXPECTED_SCENES)
def test_generated_scene_dataset_has_twenty_isolated_groups(scene: str) -> None:
    profile = profile_for_scene(scene)
    manifest = (
        Path(service.BUILD)
        / "acoustic-scene-condition-datasets"
        / profile.dataset_id
        / "manifest.json"
    )
    if not manifest.is_file():
        pytest.skip(f"{scene} engineering-preview dataset is not generated")

    dataset = load_scene_condition_dataset(profile.dataset_id)
    assert dataset["dataset_id"] == profile.dataset_id
    assert dataset["dataset_name"] == profile.dataset_name
    assert dataset["dataset_kind"] == profile.dataset_kind
    assert dataset["scene"] == scene
    assert dataset["mode"] == "WB-wired"
    assert len(dataset["cases"]) == 20
    assert dataset["preview_condition_group_count"] == 20

    covered = {
        item_id
        for case in dataset["cases"]
        for item_id in case["contract_item_ids"]
    }
    assert covered == set(profile.contract_item_ids)
    assert all(
        case["conditions"]["scene"] == scene for case in dataset["cases"]
    )
    assert {
        case["provenance"]["render_metadata"]["kind"]
        for case in dataset["cases"]
    } == {"speech", "cng"}

    if scene != "sce1":
        assert {
            case["provenance"]["snr_db"] for case in dataset["cases"]
        } == {EXPECTED_SNR_DB[scene]}
        assert all(
            case["case_id"].startswith(f"{scene}-wb-wired-")
            for case in dataset["cases"]
        )


@pytest.mark.parametrize("scene", EXPECTED_SCENES)
def test_batch_snapshot_uses_each_scene_dataset_contract(scene: str) -> None:
    profile = profile_for_scene(scene)
    manifest = (
        Path(service.BUILD)
        / "acoustic-scene-condition-datasets"
        / profile.dataset_id
        / "manifest.json"
    )
    if not manifest.is_file():
        pytest.skip(f"{scene} engineering-preview dataset is not generated")

    dataset, contract_by_id = scene_condition_batch._validated_dataset(
        profile.dataset_id,
        requested_scene=scene,
    )
    snapshot = scene_condition_batch._build_snapshot(
        dataset,
        contract_by_id,
        "a" * 32,
        "vcp_default",
        {"profile": "vcp_default"},
        "b" * 64,
        "c" * 64,
    )
    scene_condition_batch.validate_snapshot(snapshot)

    assert snapshot["selected_scene"] == scene
    assert snapshot["selected_mode"] == "WB-wired"
    assert snapshot["contract_item_count"] == 5
    assert len(snapshot["cases"]) == 20
    assert len(snapshot["items"]) == 50
    assert {
        item["item_id"] for item in snapshot["items"]
    } == set(profile.contract_item_ids)
