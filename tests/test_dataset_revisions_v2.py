from __future__ import annotations

import hashlib

import pytest

from server.acoustic import dataset_revisions as first
from server.acoustic import dataset_revisions_v2 as current
from server.acoustic import scene_mode_conversation_profiles as scenes
from server.acoustic import acoustic_stress_profiles as stress


@pytest.fixture(autouse=True)
def preserve_registries():
    registries = (scenes.MODE_CONVERSATION_PROFILES, scenes.PROFILE_BY_DATASET_ID,
                  stress.PROFILES, stress.BY_DATASET_ID)
    snapshots = [dict(registry) for registry in registries]
    yield
    for registry, prior in zip(registries, snapshots, strict=True):
        registry.clear()
        registry.update(prior)


def test_r2_namespace_is_distinct_while_r1_source_identity_stays_fixed():
    assert first.DATASET_REVISION == "measurement-20261117-r1"
    assert current.DATASET_REVISION == "measurement-20261117-r2"
    identifiers = set()
    for (scene, mode), profile in current.REVISION_SCENE_PROFILES.items():
        expected = hashlib.sha256(f"measurement-20261117-r2:scene:{scene}:{mode}".encode()).hexdigest()[:32]
        assert profile.dataset_id == expected
        assert profile.dataset_id != first.revision_scene_profile(scene, mode).dataset_id
        identifiers.add(profile.dataset_id)
    for (scope, mode), profile in current.REVISION_STRESS_PROFILES.items():
        assert profile.dataset_id != first.revision_stress_profile(scope, mode).dataset_id
        identifiers.add(profile.dataset_id)
    assert len(identifiers) == 48
    assert current.ACTIVATION_PATH.name == "ACTIVATION-r2.json"


def test_r2_register_and_activate_keep_old_and_r1_identifiers():
    old_scene = {key: value.dataset_id for key, value in scenes.MODE_CONVERSATION_PROFILES.items()}
    old_stress = {key: value.dataset_id for key, value in stress.PROFILES.items()}
    first.install_revision_profiles(activate=False)
    current.install_revision_profiles(activate=False)
    assert {key: value.dataset_id for key, value in scenes.MODE_CONVERSATION_PROFILES.items()} == old_scene
    current.install_revision_profiles(activate=True)
    for key, prior in old_scene.items():
        assert scenes.MODE_CONVERSATION_PROFILES[key] is current.REVISION_SCENE_PROFILES[key]
        assert scenes.profile_for_dataset(prior).dataset_id == prior
        r1_id = first.REVISION_SCENE_PROFILES[key].dataset_id
        assert scenes.profile_for_dataset(r1_id).dataset_id == r1_id
    for key, prior in old_stress.items():
        assert stress.PROFILES[key] is current.REVISION_STRESS_PROFILES[key]
        assert stress.stress_profile_for_dataset(prior).dataset_id == prior


def test_r2_missing_or_malformed_activation_preserves_defaults(tmp_path, monkeypatch):
    prior = dict(scenes.MODE_CONVERSATION_PROFILES)
    path = tmp_path / "ACTIVATION-r2.json"
    monkeypatch.setattr(current._implementation, "ACTIVATION_PATH", path)
    assert current.install_from_activation()["activated"] is False
    path.write_text('{"revision":"measurement-20261117-r1","datasets":[]}', encoding="utf-8")
    with pytest.raises(current.DatasetActivationError):
        current.install_from_activation()
    assert scenes.MODE_CONVERSATION_PROFILES == prior
