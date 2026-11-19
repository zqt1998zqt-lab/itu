from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import omlsa_tuning_fixture as fixture


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


current = _load_module(
    "itu_omlsa_tuning_current_under_test",
    SCRIPTS_DIR / "evaluate_omlsa_tuning_current.py",
)
prepare = _load_module(
    "itu_omlsa_tuning_prepare_under_test",
    SCRIPTS_DIR / "prepare-omlsa-tuning-fixture.py",
)


def _manifest_document() -> dict:
    return json.loads(fixture.MANIFEST_PATH.read_text(encoding="utf-8"))


def _copy_fixture_except(
    destination: Path,
    target_relative_path: str | None = None,
    replacement: bytes | str | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest_path.write_bytes(fixture.MANIFEST_PATH.read_bytes())
    lock = fixture.freeze_source_lock()
    for artifact in lock.artifacts:
        if artifact.relative_path == target_relative_path:
            if replacement is None:
                continue
            target = destination / artifact.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if replacement == "symlink":
                os.symlink(artifact.path, target)
            else:
                assert isinstance(replacement, bytes)
                target.write_bytes(replacement)
            continue
        target = destination / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact.path, target)
    return manifest_path


def _write_modified_manifest(destination: Path, mutate) -> Path:
    document = _manifest_document()
    mutate(document)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_fixed_fixture_contains_four_reference_wavs_and_three_assessments():
    lock = fixture.freeze_source_lock()
    references = [artifact for artifact in lock.artifacts if artifact.kind == "reference"]
    assessments = [
        artifact for artifact in lock.artifacts if artifact.kind == "assessment"
    ]

    assert lock.manifest_sha256 == fixture.sha256_file(fixture.MANIFEST_PATH)
    assert lock.manifest_sha256 == (
        "318c7098c8e069ac7ef6bc6bafa6c993bdb992f82c07f71cabfb2d69f595f847"
    )
    assert len(references) == 4
    assert len(assessments) == 3
    assert {artifact.sha256 for artifact in references} == fixture.REFERENCE_SHA256
    assert fixture.EXPECTED_MODEL_SHA256 not in fixture.REFERENCE_SHA256
    assert lock.upstream_script.sha256 == fixture.UPSTREAM_SCRIPT_SHA256
    assert lock.upstream_script.bytes == fixture.UPSTREAM_SCRIPT_BYTES

    for artifact in references:
        assert artifact.path.parent == fixture.FIXTURE_ROOT / "references"
        assert artifact.path.stat().st_size == 384_044
        assert fixture.sha256_file(artifact.path) == artifact.sha256
        assert Path(artifact.source_path).is_absolute()
        assert Path(artifact.source_path).name == f"{artifact.sha256}.wav"

    for artifact in assessments:
        assert artifact.path.name == "assessment.json"
        assert artifact.path.stat().st_size == artifact.bytes
        assert fixture.sha256_file(artifact.path) == artifact.sha256
        assert Path(artifact.source_path).is_absolute()
        assert artifact.reference_binding is not None
        assert artifact.reference_binding["fixture_path_comparison"] == "not_performed"

    assert not any(path.is_symlink() for path in fixture.FIXTURE_ROOT.rglob("*"))


def test_manifest_preserves_original_paths_and_missing_tmp_dev_manifest():
    lock = fixture.freeze_source_lock()

    assert lock.source_store["dataset_id"] == fixture.DATASET_ID
    assert lock.source_store["holdout_batch_id"] == fixture.HOLDOUT_BATCH_ID
    assert Path(lock.source_store["dataset_audio_dir"]).is_absolute()
    assert Path(lock.source_store["holdout_batch_dir"]).is_absolute()
    assert lock.dev_manifest_origin["source_path"] == str(
        fixture.DEV_MANIFEST_SOURCE_PATH
    )
    assert lock.dev_manifest_origin["present_at_preparation"] is False
    assert lock.dev_manifest_origin["sha256"] is None
    assert lock.dev_manifest_origin["bytes"] is None
    assert lock.dev_manifest_origin["frozen_metadata"]["noise_seed"] == 2026111401
    assert lock.dev_manifest_origin["frozen_metadata"]["target_snr_db"] == 12.0

    manifest = fixture.load_manifest()
    for artifact in manifest["artifacts"]:
        assert Path(artifact["source_path"]).is_absolute()
        assert artifact["relative_path"]
        assert artifact["sha256"]
        assert isinstance(artifact["bytes"], int)


def test_assessment_binding_uses_original_canonical_path_not_fixture_path():
    lock = fixture.freeze_source_lock()
    assessment = lock.artifact("assessment:02")
    document = json.loads(assessment.path.read_text(encoding="utf-8"))
    canonical = (
        Path(lock.source_store["dataset_audio_dir"])
        / f"{fixture.HOLDOUT_REFERENCES['02'][0]}.wav"
    )

    binding = fixture.assessment_reference_binding(
        document,
        fixture.HOLDOUT_REFERENCES["02"][0],
        canonical,
    )
    assert binding["explicit_path_present"] is False
    assert binding["explicit_reference_paths"] == []
    assert binding["fixture_path_comparison"] == "not_performed"

    original_path_document = copy.deepcopy(document)
    original_path_document["clean_reference_path"] = str(canonical)
    accepted = fixture.assessment_reference_binding(
        original_path_document,
        fixture.HOLDOUT_REFERENCES["02"][0],
        canonical,
    )
    assert accepted["explicit_path_present"] is True
    assert accepted["explicit_reference_paths"] == [str(canonical)]

    migrated_path_document = copy.deepcopy(document)
    migrated_path_document["clean_reference_path"] = str(
        lock.artifact(f"reference:{fixture.HOLDOUT_REFERENCES['02'][0]}").path
    )
    with pytest.raises(fixture.FixtureError, match="original canonical path"):
        fixture.assessment_reference_binding(
            migrated_path_document,
            fixture.HOLDOUT_REFERENCES["02"][0],
            canonical,
        )

    missing_hash_document = copy.deepcopy(document)
    del missing_hash_document["input_sha256"]["reference"]
    with pytest.raises(fixture.FixtureError, match="input_sha256.reference"):
        fixture.assessment_reference_binding(
            missing_hash_document,
            fixture.HOLDOUT_REFERENCES["02"][0],
            canonical,
        )


def test_current_loaders_preserve_case_specs_and_ignore_old_store_paths(monkeypatch):
    monkeypatch.setattr(
        current.base,
        "DATASET_AUDIO_DIR",
        Path("/definitely/not/readable/acoustic-scene-condition-datasets"),
    )
    monkeypatch.setattr(
        current.base,
        "HOLDOUT_BATCH_DIR",
        Path("/definitely/not/readable/scene-condition-batches"),
    )

    dev, record = current.base.load_dev_spec()
    holdouts = current.base.load_holdout_specs()
    lock = fixture.freeze_source_lock()
    by_case = {case.case_id: case for case in lock.cases}

    assert type(dev) is current.base.CaseSpec
    assert dev.case_id == "dev-near_single-01-white-seed2026111401-snr12"
    assert dev.role == "dev"
    assert dev.source_case_id == "sce1-wb-wired-near_single-01-conversation-v2"
    assert dev.noise_seed == 2026111401
    assert dev.target_snr_db == 12.0
    assert dev.assessment_path is None
    assert dev.reference_path.is_relative_to(fixture.FIXTURE_ROOT)
    assert record["path"] == str(fixture.MANIFEST_PATH)
    assert record["manifest"]["sha256"] == lock.manifest_sha256

    assert len(holdouts) == 6
    assert all(type(spec) is current.base.CaseSpec for spec in holdouts)
    for spec in holdouts:
        expected = by_case[spec.case_id]
        assert spec.role == "holdout"
        assert spec.source_case_id == expected.source_case_id
        assert spec.noise_seed == expected.noise_seed
        assert spec.target_snr_db == expected.target_snr_db
        assert spec.reference_path.is_relative_to(fixture.FIXTURE_ROOT)
        assert spec.assessment_path is not None
        assert spec.assessment_path.is_relative_to(fixture.FIXTURE_ROOT)

    assert [spec.case_id for spec in holdouts] == [
        case.case_id for case in lock.cases if case.role == "holdout"
    ]
    decoded = {}
    for spec in [dev, *holdouts]:
        if spec.reference_path not in decoded:
            audio, metadata = current.base.validate_reference(
                spec.reference_path,
                spec.reference_sha256,
            )
            assert audio.shape == (192_000,)
            assert metadata["sample_rate"] == 16_000
            assert metadata["decoded_samples"] == 192_000
            decoded[spec.reference_path] = metadata["sha256"]
    assert set(decoded.values()) == fixture.REFERENCE_SHA256
    assert sys.modules[current.LEGACY_MODULE_NAME] is current.base


def test_current_evaluate_injects_fixture_origin_without_changing_score_shape(monkeypatch):
    def fake_historical_evaluate(args):
        assert args == "sentinel-args"
        return {
            "schema": "itu_omlsa_tuning_evaluation.v1",
            "dev_manifest": {"path": "legacy"},
            "cases": [{"case_id": "unchanged"}],
            "verification": {"project_inputs_unchanged": True},
        }

    monkeypatch.setattr(current, "_ORIGINAL_BASE_EVALUATE", fake_historical_evaluate)
    evidence = current.evaluate("sentinel-args")

    assert evidence["schema"] == "itu_omlsa_tuning_evaluation.v1"
    assert evidence["cases"] == [{"case_id": "unchanged"}]
    assert evidence["verification"] == {"project_inputs_unchanged": True}
    assert evidence["fixture"]["fixture_id"] == fixture.FIXTURE_ID
    assert evidence["fixture"]["manifest"]["path"] == str(fixture.MANIFEST_PATH)
    assert evidence["fixed_reference_sha256"] == sorted(fixture.REFERENCE_SHA256)
    assert evidence["fixture_source_lock"]["source_paths_accessed_at_evaluation"] is False
    assert evidence["fixture_source_lock"]["upstream_script_sha256"] == (
        fixture.UPSTREAM_SCRIPT_SHA256
    )
    assert evidence["dev_manifest"]["fixture_manifest_sha256"] == (
        evidence["fixture"]["manifest"]["sha256"]
    )
    assert evidence["dev_manifest"]["original_dev_manifest_accessed_at_evaluation"] is False


def test_current_main_delegates_historical_cli_and_main(monkeypatch):
    parsed = current.base.parse_args(
        [
            "--probe",
            "candidate",
            "--original-probe",
            "control",
            "--output",
            "new-evidence",
            "--mode",
            "all",
            "--variants",
            '{"baseline":[1,0,60]}',
        ]
    )
    assert parsed.probe == "candidate"
    assert parsed.original_probe == "control"
    assert parsed.output == "new-evidence"
    assert parsed.mode == "all"
    assert parsed.variants == '{"baseline":[1,0,60]}'

    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        return 7

    monkeypatch.setattr(current.base, "main", fake_main)
    assert current.main(["--mode", "dev"]) == 7
    assert observed["argv"] == ["--mode", "dev"]


def test_prepare_fixture_is_create_or_byte_identical_only(tmp_path):
    destination = tmp_path / "fixture"
    first = prepare.prepare_fixture(destination)
    second = prepare.prepare_fixture(destination)
    checked = prepare.prepare_fixture(destination, check_only=True)

    assert first["status"] == "prepared_or_verified"
    assert {item["status"] for item in first["artifacts"]} == {"created"}
    assert second["status"] == "verified"
    assert {item["status"] for item in second["artifacts"]} == {"verified"}
    assert second["manifest"]["status"] == "verified"
    assert checked["status"] == "verified"
    assert first["manifest"]["sha256"] == second["manifest"]["sha256"]
    assert (destination / "manifest.json").read_bytes() == fixture.MANIFEST_PATH.read_bytes()


def test_prepare_check_only_does_not_access_original_sources(monkeypatch):
    missing = Path("/definitely/not/readable/omlsa-source")
    monkeypatch.setattr(prepare.fixture, "DATASET_AUDIO_DIR", missing / "audio")
    monkeypatch.setattr(prepare.fixture, "HOLDOUT_BATCH_DIR", missing / "batch")
    monkeypatch.setattr(prepare.fixture, "DEV_MANIFEST_SOURCE_PATH", missing / "evidence.json")

    checked = prepare.prepare_fixture(fixture.FIXTURE_ROOT, check_only=True)
    assert checked["status"] == "verified"
    assert checked["manifest"]["sha256"] == fixture.FIXTURE_MANIFEST_SHA256
    assert {item["status"] for item in checked["artifacts"]} == {"verified"}


def test_freeze_source_lock_rejects_missing_artifact(tmp_path):
    target = f"references/{fixture.DEV_REFERENCE_SHA256}.wav"
    manifest_path = _copy_fixture_except(
        tmp_path / "fixture",
        target,
        None,
    )
    with pytest.raises(fixture.FixtureError, match="missing"):
        fixture.freeze_source_lock(manifest_path)


def test_freeze_source_lock_rejects_changed_artifact_bytes(tmp_path):
    target = (
        f"references/{fixture.DEV_REFERENCE_SHA256}.wav"
    )
    manifest_path = _copy_fixture_except(
        tmp_path / "fixture",
        target,
        b"X" * fixture.REFERENCE_BYTES,
    )
    with pytest.raises(fixture.FixtureError, match="SHA-256 changed"):
        fixture.freeze_source_lock(manifest_path)


def test_freeze_source_lock_rejects_artifact_symlink(tmp_path):
    target = (
        f"references/{fixture.DEV_REFERENCE_SHA256}.wav"
    )
    manifest_path = _copy_fixture_except(
        tmp_path / "fixture",
        target,
        "symlink",
    )
    with pytest.raises(fixture.FixtureError, match="symlink component"):
        fixture.freeze_source_lock(manifest_path)


def test_freeze_source_lock_rejects_relative_path_escape(tmp_path):
    def mutate(document):
        document["artifacts"][0]["relative_path"] = "../outside.wav"

    manifest_path = _write_modified_manifest(tmp_path / "fixture", mutate)
    with pytest.raises(fixture.FixtureError, match="manifest SHA-256 changed"):
        fixture.freeze_source_lock(manifest_path)
    with pytest.raises(fixture.FixtureError, match="escapes the fixture root"):
        fixture._safe_relative_path("../outside.wav", "relative_path")


def test_load_manifest_rejects_symlink(tmp_path):
    symlink = tmp_path / "manifest.json"
    os.symlink(fixture.MANIFEST_PATH, symlink)
    with pytest.raises(fixture.FixtureError, match="symlink"):
        fixture.load_manifest(symlink)


def test_load_manifest_rejects_symlinked_parent(tmp_path):
    parent = tmp_path / "fixture-parent"
    os.symlink(fixture.FIXTURE_ROOT, parent)
    with pytest.raises(fixture.FixtureError, match="symlink component"):
        fixture.load_manifest(parent / "manifest.json")
