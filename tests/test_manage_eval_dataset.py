"""CRUD history, immutable parents and controlled sweeps must fail closed."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
import manage_eval_dataset as manager
import build_curated_sweeps as sweeps
import run_audio_eval as ev
import audit_dataset_quality as audit


def catalog(tmp_path):
    parent = tmp_path / "parent"
    sources = parent / "sources"
    sources.mkdir(parents=True)
    rng = np.random.default_rng(10)
    ev.write(sources / "voice.wav", rng.normal(0, .01, 384000))
    source = dict(path="voice.wav", sha256=ev.digest(sources / "voice.wav"), kind="clean_speech",
                  speaker="1", language="en", sample_rate=16000, channels=1)
    (sources / "manifest.json").write_bytes(manager.encoded(dict(sources=[source])))
    case = dict(id="original", family="ns", pair=0, near_speaker="1", far_speaker="2", seed=1,
                near_dbfs=-26, noise="white", snr_db=10, ser_db=0, rt60_s=.15,
                echo_delay_ms=80, stress="none", profile="expanded-v3", mics=1)
    scene = ev.make_case(case, {"1":rng.normal(size=384000), "2":rng.normal(size=384000)}, {},
                         parent / "cases/original", {})
    (parent / "inputs-manifest.json").write_bytes(manager.encoded(dict(scenes=[scene],
        source_manifest_sha256=ev.digest(sources / "manifest.json"))))
    root = tmp_path / "catalog"
    root.mkdir()
    state = manager.initialize(root, [parent])
    manager.save(root, state, "init", state["parents"])
    return root, state, parent


def test_crud_export_preserves_parent_and_excludes_removed(tmp_path):
    root, state, parent = catalog(tmp_path)
    before = ev.digest(parent / "inputs-manifest.json")
    record = next(iter(state["scenes"].values()))
    record["scene"].update(quality_track="quality", split="development")
    manager.save(root, state, "update", dict(reason="derived tags"))
    output = tmp_path / "export"
    result = manager.export(manager.load(root), root, output, storage="copy")
    assert result["scenes"] == 1
    exported = json.loads((output / "inputs-manifest.json").read_text())["scenes"][0]
    assert audit.audit_scene((output, exported))["errors"] == []
    assert json.loads((manager.folder(exported) / "scene.json").read_text()) == exported
    record.update(excluded=True, exclusion_reason="logical deletion")
    manager.save(root, state, "remove", dict(reason="logical deletion"))
    assert manager.select(manager.load(root)) == []
    with pytest.raises(ValueError, match="nonempty export"):
        manager.export(state, root, tmp_path / "removed")
    assert ev.digest(parent / "inputs-manifest.json") == before
    assert (parent / "cases/original/before.wav").exists()


def test_corrupt_catalog_history_negative_control(tmp_path):
    root, state, _ = catalog(tmp_path)
    state["revision"] += 1
    (root / "catalog.json").write_bytes(manager.encoded(state))
    with pytest.raises(ValueError, match="outside CRUD history"):
        manager.load(root)


def test_operation_evidence_tampering_negative_control(tmp_path):
    root, _, _ = catalog(tmp_path)
    path = root / "operations.jsonl"
    path.write_text(path.read_text().replace('"action": "init"', '"action": "tampered"'))
    with pytest.raises(ValueError, match="log content changed"):
        manager.load(root)


def test_wrong_hash_and_bad_window_are_detected(tmp_path):
    _, state, parent = catalog(tmp_path)
    scene = next(iter(state["scenes"].values()))["scene"]
    bad = copy.deepcopy(scene)
    bad["files"]["before.wav"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        manager.verify_scene(bad)
    bad = copy.deepcopy(scene)
    bad["windows"]["noise_only"] = [2, 3]
    assert "target_leaks_into:noise_only" in audit.audit_scene((parent, bad))["errors"]


def test_added_source_requires_provenance_and_rejects_duplicates(tmp_path):
    _, state, parent = catalog(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        manager.add_source(state, parent / "sources/voice.wav", "clean_speech", "1", "en", None, "CC0")
    with pytest.raises(ValueError, match="duplicate source"):
        manager.add_source(state, parent / "sources/voice.wav", "clean_speech", "1", "en", "test", "CC0")
    ev.write(tmp_path / "new.wav", np.sin(np.arange(16000) * .2) * .1)
    added = manager.add_source(state, tmp_path / "new.wav", "clean_speech", "3", "zh", "local test", "CC0")
    assert added["sample_rate"] == 16000 and added["language"] == "zh"


def test_candidate_sources_are_queryable_but_not_ground_truth_or_noise(tmp_path):
    root, state, _ = catalog(tmp_path)
    ev.write(tmp_path / "candidate.wav", np.sin(np.arange(16000) * .2) * .1)
    added = manager.add_source(state, tmp_path / "candidate.wav", "speech_candidate", "3", "zh",
                               "local test", "unknown_not_for_redistribution")
    manager.save(root, state, "add", added)
    command = [sys.executable, str(Path(manager.__file__)), "--workspace", str(root),
               "query", "--sources", "--where", "kind=speech_candidate", "--where", "language=zh"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    found = json.loads(result.stdout)
    assert found["count"] == 1 and found["sources"][0]["speaker"] == "3"
    output = tmp_path / "export"
    manager.export(state, root, output)
    people, noises = ev.sources(output)
    assert set(people) == {"1"} and noises == {}


def test_added_noise_requires_key_and_survives_source_export(tmp_path):
    root, state, _ = catalog(tmp_path)
    ev.write(tmp_path / "noise.wav", np.random.default_rng(5).normal(0, .01, 480000))
    with pytest.raises(ValueError, match="noise name"):
        manager.add_source(state, tmp_path / "noise.wav", "noise", None, None, "test", "CC0")
    added = manager.add_source(state, tmp_path / "noise.wav", "noise", None, None, "test", "CC0", "CAR_NEW")
    manager.save(root, state, "add", added)
    output = tmp_path / "export"
    manager.export(state, root, output)
    people, noises = ev.sources(output)
    assert set(people) == {"1"} and set(noises) == {"car_new"}
    assert len(noises["car_new"]) == 480000
    ev.write(tmp_path / "other-noise.wav", np.random.default_rng(6).normal(0, .01, 480000))
    with pytest.raises(ValueError, match="duplicate noise name"):
        manager.add_source(state, tmp_path / "other-noise.wav", "noise", None, None, "test", "CC0", "car_new")
    case = dict(next(iter(state["scenes"].values()))["scene"], noise="car_new")
    people["2"] = np.random.default_rng(3).normal(0, .01, 480000)
    generated = ev.make_case(case, people, noises, tmp_path / "cases/mixture", {})
    assert generated["noise_origin"] == "real recording"
    assert audit.audit_scene((tmp_path, dict(generated, source_dataset=str(tmp_path), source_case_id="mixture")))["errors"] == []
    with pytest.raises(ValueError, match="shorter than scene"):
        ev.noise_background(case, 384000, people, {"car_new": noises["car_new"][:16000]}, np.random.default_rng(4))


@pytest.mark.parametrize("rate", [24000, 48000])
def test_native_source_is_candidate_until_generator_supports_it(tmp_path, rate):
    root, state, _ = catalog(tmp_path)
    import soundfile as sf
    path = tmp_path / "native.wav"
    sf.write(path, np.sin(np.arange(rate) * .2) * .1, rate, subtype="PCM_16")
    for kind in ("clean_speech", "noise"):
        with pytest.raises(ValueError, match="generator requires 16 kHz"):
            manager.add_source(state, path, kind, "native", "zh", "test", "CC0", "native_noise")
    added = manager.add_source(state, path, "speech_candidate", "native", "zh", "test", "CC0")
    manager.save(root, state, "add", added)
    output = tmp_path / "export"
    manager.export(state, root, output)
    assert set(ev.sources(output)[0]) == {"1"}


def test_snr_sweep_keeps_identical_target_and_quantized_snr(tmp_path):
    _, state, _ = catalog(tmp_path)
    original = next(iter(state["scenes"].values()))["scene"]
    original.update(quality_track="quality", quality_flags=[], split="development")
    prepared = sweeps.prepare_group([original], -36, [-10, 0, 10, 20, 30])
    gain = min(1., .88 / prepared[-1])
    scenes = sweeps.render_group(prepared, [-10, 0, 10, 20, 30], gain, tmp_path / "sweeps")
    assert len(scenes) == 5
    assert len({s["files"]["clean.wav"]["sha256"] for s in scenes}) == 1
    assert max(abs(s["measured_input_snr_db"] - s["snr_db"]) for s in scenes) < .05
    wrong = copy.deepcopy(scenes[-1])
    wrong["snr_db"] -= 10
    assert "measured_snr_outside_0.05_db" in audit.audit_scene((tmp_path, wrong))["errors"]


def test_export_rejects_speaker_leakage(tmp_path):
    root, state, _ = catalog(tmp_path)
    first = next(iter(state["scenes"].values()))
    first["scene"].update(quality_track="quality", split="development")
    second = copy.deepcopy(first)
    second["scene"].update(id="other", split="holdout")
    state["scenes"]["other"] = second
    manager.save(root, state, "update", {})
    with pytest.raises(ValueError, match="speaker leakage"):
        manager.export(state, root, tmp_path / "leaking")


def test_metadata_cannot_relabel_speech_as_environmental_noise():
    rows = [dict(scene=dict(noise="car_passenger", playback_source="none"))]
    with pytest.raises(ValueError, match="contradicts"):
        manager.validate_patch(dict(interference_class="environmental_noise"), rows)
    with pytest.raises(ValueError, match="regenerate"):
        manager.validate_patch(dict(snr_db=20), rows)
    manager.validate_patch(dict(interference_class="interfering_speech"), rows)
