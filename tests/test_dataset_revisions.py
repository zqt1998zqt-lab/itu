from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from server.acoustic import acoustic_stress_profiles as stress_profiles
from server.acoustic import dataset_revisions
from server.acoustic import scene_mode_conversation_profiles as scene_profiles
from server.acoustic import service
from server.acoustic.acceptance_scope import ACCEPTANCE_MODES


PROJECT = Path(__file__).resolve().parents[1]
SCENES = ("sce1", "sce2", "sce3", "sce4")
MODES = tuple(sorted(ACCEPTANCE_MODES))
SCOPES = ("echo", "double_talk")
ARTIFACT_NAMES = (
    "manifest.json",
    "summary.json",
    "assets.json",
    "quality.json",
    "runtime-contract-binding.json",
)
LEGACY_SOURCE_SHA256 = {
    "server/acoustic/scene_mode_conversation_profiles.py": (
        "ab16b71d841b0b3c25a546e65dfe10d0926a08aa92f3da52f189d203d4074a72"
    ),
    "server/acoustic/acoustic_stress_profiles.py": (
        "c8f04cded4b24dac0adc24197aa4e8a7b779dd1ac07a0e5cf64366f558c48811"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def restore_profile_state():
    state = (
        scene_profiles.MODE_CONVERSATION_PROFILES,
        dict(scene_profiles.MODE_CONVERSATION_PROFILES),
        scene_profiles.PROFILE_BY_DATASET_ID,
        dict(scene_profiles.PROFILE_BY_DATASET_ID),
        stress_profiles.PROFILES,
        dict(stress_profiles.PROFILES),
        stress_profiles.BY_DATASET_ID,
        dict(stress_profiles.BY_DATASET_ID),
    )
    yield
    for live, snapshot in zip(state[::2], state[1::2]):
        live.clear()
        live.update(snapshot)


def _revision_source_hashes() -> dict[str, str]:
    required = (
        dataset_revisions._SCENE_REQUIRED_SOURCES
        | dataset_revisions._STRESS_REQUIRED_SOURCES
    )
    return {
        relative: _sha(PROJECT / relative)
        for relative in sorted(required)
    }


def _write_store(
    build_root: Path,
    *,
    kind: str,
    dataset_id: str,
    source_sha256: dict[str, str],
) -> dict[str, str]:
    store_name = (
        dataset_revisions.SCENE_ROOT
        if kind == "scene"
        else dataset_revisions.STRESS_ROOT
    )
    root = build_root / store_name / dataset_id
    payloads = {
        "manifest.json": {"dataset_id": dataset_id, "kind": kind},
        "summary.json": {"dataset_id": dataset_id, "summary": True},
        "assets.json": {"dataset_id": dataset_id, "assets": []},
        "quality.json": {"dataset_id": dataset_id, "quality": True},
    }
    for name, payload in payloads.items():
        _write_json(root / name, payload)
    artifact_sha256 = {
        name: _sha(root / name)
        for name in payloads
    }
    binding = {
        "schema_version": (
            "scene-condition-runtime-binding-v2"
            if kind == "scene"
            else "acoustic-stress-runtime-binding-v1"
        ),
        "dataset_id": dataset_id,
        "status": "reconciled",
        "source_sha256": dict(source_sha256),
        "artifact_sha256": artifact_sha256,
    }
    _write_json(root / "runtime-contract-binding.json", binding)
    return {
        name: _sha(root / name)
        for name in ARTIFACT_NAMES
    }


def _build_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], Path]:
    build_root = tmp_path / "acoustic-qualification"
    monkeypatch.setattr(service, "BUILD", build_root)
    activation_path = tmp_path / "dataset-revision-20261117" / "ACTIVATION.json"
    monkeypatch.setattr(
        dataset_revisions,
        "ACTIVATION_PATH",
        activation_path,
    )
    source_sha256 = _revision_source_hashes()
    datasets = []
    for scene in SCENES:
        for mode in MODES:
            profile = dataset_revisions.revision_scene_profile(scene, mode)
            artifacts = _write_store(
                build_root,
                kind="scene",
                dataset_id=profile.dataset_id,
                source_sha256=source_sha256,
            )
            datasets.append(
                {
                    "dataset_id": profile.dataset_id,
                    "kind": "scene",
                    "scene": scene,
                    "mode": mode,
                    "artifacts": artifacts,
                }
            )
    for scope in SCOPES:
        for mode in MODES:
            profile = dataset_revisions.revision_stress_profile(scope, mode)
            artifacts = _write_store(
                build_root,
                kind="stress",
                dataset_id=profile.dataset_id,
                source_sha256=source_sha256,
            )
            datasets.append(
                {
                    "dataset_id": profile.dataset_id,
                    "kind": "stress",
                    "scope": scope,
                    "mode": mode,
                    "artifacts": artifacts,
                }
            )
    receipt = {
        "revision": dataset_revisions.DATASET_REVISION,
        "datasets": datasets,
    }
    _write_json(activation_path, receipt)
    return activation_path, receipt, build_root


def test_revision_profiles_are_unique_stable_and_register_without_defaults():
    scene_defaults = scene_profiles.MODE_CONVERSATION_PROFILES
    stress_defaults = stress_profiles.PROFILES
    scene_default_values = dict(scene_defaults)
    stress_default_values = dict(stress_defaults)
    old_scene_ids = set(scene_profiles.PROFILE_BY_DATASET_ID)
    old_stress_ids = set(stress_profiles.BY_DATASET_ID)

    result = dataset_revisions.install_revision_profiles(activate=False)

    assert result == {
        "revision": "measurement-20261117-r1",
        "registered": 48,
        "activated": False,
    }
    assert scene_profiles.MODE_CONVERSATION_PROFILES is scene_defaults
    assert stress_profiles.PROFILES is stress_defaults
    assert scene_defaults == scene_default_values
    assert stress_defaults == stress_default_values
    assert old_scene_ids.issubset(scene_profiles.PROFILE_BY_DATASET_ID)
    assert old_stress_ids.issubset(stress_profiles.BY_DATASET_ID)

    scene_ids = set()
    for scene in SCENES:
        for mode in MODES:
            profile = dataset_revisions.revision_scene_profile(scene, mode)
            expected_id = hashlib.sha256(
                f"measurement-20261117-r1:scene:{scene}:{mode}".encode()
            ).hexdigest()[:32]
            assert isinstance(
                profile,
                scene_profiles.ModeConversationProfile,
            )
            assert profile.dataset_id == expected_id
            assert (
                profile.dataset_name
                == scene_default_values[(scene, mode)].dataset_name
            )
            assert (
                scene_profiles.PROFILE_BY_DATASET_ID[expected_id]
                is profile
            )
            assert scene_profiles.profile_for_dataset(expected_id) is profile
            scene_ids.add(profile.dataset_id)

    stress_ids = set()
    for scope in SCOPES:
        for mode in MODES:
            profile = dataset_revisions.revision_stress_profile(scope, mode)
            expected_id = hashlib.sha256(
                f"measurement-20261117-r1:stress:{scope}:{mode}".encode()
            ).hexdigest()[:32]
            assert isinstance(profile, stress_profiles.StressProfile)
            assert profile.dataset_id == expected_id
            assert (
                profile.dataset_name
                == stress_default_values[(scope, mode)].dataset_name
            )
            assert stress_profiles.BY_DATASET_ID[expected_id] is profile
            assert stress_profiles.stress_profile_for_dataset(expected_id) is profile
            stress_ids.add(profile.dataset_id)

    assert len(scene_ids) == 32
    assert len(stress_ids) == 16
    assert not scene_ids & stress_ids
    assert len(scene_ids | stress_ids) == 48


def test_activate_updates_the_imported_default_dictionaries_in_place():
    scene_defaults = scene_profiles.MODE_CONVERSATION_PROFILES
    stress_defaults = stress_profiles.PROFILES
    scene_default_id = id(scene_defaults)
    stress_default_id = id(stress_defaults)
    old_scene_registry = dict(scene_profiles.PROFILE_BY_DATASET_ID)
    old_stress_registry = dict(stress_profiles.BY_DATASET_ID)
    imported_scene_lookup = scene_profiles.profile_for_scene_and_mode
    imported_stress_lookup = stress_profiles.stress_profile

    result = dataset_revisions.install_revision_profiles(activate=True)

    assert result["activated"] is True
    assert id(scene_profiles.MODE_CONVERSATION_PROFILES) == scene_default_id
    assert id(stress_profiles.PROFILES) == stress_default_id
    for key, profile in dataset_revisions.REVISION_SCENE_PROFILES.items():
        assert scene_defaults[key] is profile
        assert imported_scene_lookup(*key) is profile
    for key, profile in dataset_revisions.REVISION_STRESS_PROFILES.items():
        assert stress_defaults[key] is profile
        assert imported_stress_lookup(*key) is profile
    assert old_scene_registry.items() <= scene_profiles.PROFILE_BY_DATASET_ID.items()
    assert old_stress_registry.items() <= stress_profiles.BY_DATASET_ID.items()


def test_missing_activation_only_registers_revision_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        dataset_revisions,
        "ACTIVATION_PATH",
        tmp_path / "missing" / "ACTIVATION.json",
    )
    scene_defaults = dict(scene_profiles.MODE_CONVERSATION_PROFILES)
    stress_defaults = dict(stress_profiles.PROFILES)

    result = dataset_revisions.install_from_activation()

    assert result == {
        "revision": "measurement-20261117-r1",
        "registered": 48,
        "activation": "absent",
        "activated": False,
    }
    assert scene_profiles.MODE_CONVERSATION_PROFILES == scene_defaults
    assert stress_profiles.PROFILES == stress_defaults
    for profile in dataset_revisions.REVISION_SCENE_PROFILES.values():
        assert scene_profiles.PROFILE_BY_DATASET_ID[profile.dataset_id] is profile
    for profile in dataset_revisions.REVISION_STRESS_PROFILES.values():
        assert stress_profiles.BY_DATASET_ID[profile.dataset_id] is profile


def test_valid_activation_switches_defaults_only_after_all_48_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _build_activation(tmp_path, monkeypatch)

    result = dataset_revisions.install_from_activation()

    assert result["activation"] == "active"
    assert result["activated"] is True
    assert scene_profiles.MODE_CONVERSATION_PROFILES == (
        dataset_revisions.REVISION_SCENE_PROFILES
    )
    assert stress_profiles.PROFILES == (
        dataset_revisions.REVISION_STRESS_PROFILES
    )


def test_incomplete_activation_is_rejected_without_switching_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    activation_path, receipt, _ = _build_activation(tmp_path, monkeypatch)
    receipt["datasets"] = receipt["datasets"][:-1]
    _write_json(activation_path, receipt)
    scene_defaults = dict(scene_profiles.MODE_CONVERSATION_PROFILES)
    stress_defaults = dict(stress_profiles.PROFILES)

    with pytest.raises(
        dataset_revisions.DatasetActivationError,
        match="48 datasets",
    ):
        dataset_revisions.install_from_activation()

    assert scene_profiles.MODE_CONVERSATION_PROFILES == scene_defaults
    assert stress_profiles.PROFILES == stress_defaults


def test_tampered_artifact_is_rejected_without_switching_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, receipt, build_root = _build_activation(tmp_path, monkeypatch)
    target = receipt["datasets"][0]
    root = (
        build_root
        / dataset_revisions.SCENE_ROOT
        / target["dataset_id"]
    )
    (root / "manifest.json").write_text(
        '{"dataset_id":"tampered"}\n',
        encoding="utf-8",
    )
    scene_defaults = dict(scene_profiles.MODE_CONVERSATION_PROFILES)
    stress_defaults = dict(stress_profiles.PROFILES)

    with pytest.raises(
        dataset_revisions.DatasetActivationError,
        match="artifact hash differs: manifest.json",
    ):
        dataset_revisions.install_from_activation()

    assert scene_profiles.MODE_CONVERSATION_PROFILES == scene_defaults
    assert stress_profiles.PROFILES == stress_defaults


def test_tampered_source_binding_is_rejected_even_if_receipt_is_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    activation_path, receipt, build_root = _build_activation(
        tmp_path,
        monkeypatch,
    )
    target = receipt["datasets"][0]
    root = (
        build_root
        / dataset_revisions.SCENE_ROOT
        / target["dataset_id"]
    )
    binding_path = root / "runtime-contract-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["source_sha256"]["server/acoustic/dataset_revisions.py"] = "0" * 64
    _write_json(binding_path, binding)
    target["artifacts"]["runtime-contract-binding.json"] = _sha(binding_path)
    _write_json(activation_path, receipt)
    scene_defaults = dict(scene_profiles.MODE_CONVERSATION_PROFILES)
    stress_defaults = dict(stress_profiles.PROFILES)

    with pytest.raises(
        dataset_revisions.DatasetActivationError,
        match="source hash differs",
    ):
        dataset_revisions.install_from_activation()

    assert scene_profiles.MODE_CONVERSATION_PROFILES == scene_defaults
    assert stress_profiles.PROFILES == stress_defaults


def test_legacy_profile_sources_remain_byte_identical():
    for relative, expected in LEGACY_SOURCE_SHA256.items():
        assert _sha(PROJECT / relative) == expected
