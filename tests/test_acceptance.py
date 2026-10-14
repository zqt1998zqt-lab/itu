"""Acceptance must reject distorted, missing, and vacuous measurements."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acceptance", ROOT / "cpp/tools/check_audio_acceptance.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
rules = json.loads(checker.RULES.read_text())

def sample():
    scene = dict(id="probe", family="ns", stress="none", near_dbfs=-26, windows={"near_only": [1, 3], "noise_only": [3, 4]})
    row = dict(mode="full", segments={"near_only": {"before": {"sisdr_db": 15},
        "after": {"sisdr_db": 18, "projection_gain_db": 0, "rms_dbfs": -26}, "stoi_improvement": 0},
        "noise_only": {"total_attenuation_db": 10, "before_dbfs": -31, "after_dbfs": -41}},
        engine={"process_p95_ms": .2, "process_max_ms": .4, "process_rtf": .02},
        process_frames_over_10ms=0, output_levels={"peak_dbfs": -3, "clipping_percent": 0})
    return scene, row

def test_frozen_rules_and_real_failure_controls():
    scene, row = sample()
    assert checker.check_scene(scene, row, rules)["passed"]
    noisy = deepcopy(row)
    noisy["segments"]["noise_only"]["total_attenuation_db"] = 0
    assert not checker.check_scene(scene, noisy, rules)["passed"]
    damaged = deepcopy(row)
    damaged["segments"]["near_only"]["after"]["sisdr_db"] = 13.99
    assert not checker.check_scene(scene, damaged, rules)["passed"]
    clipped = deepcopy(row)
    clipped["output_levels"]["clipping_percent"] = .01
    assert not checker.check_scene(scene, clipped, rules)["passed"]

def test_missing_numbers_and_missing_windows_never_pass():
    scene, row = sample()
    row["segments"]["near_only"]["after"]["sisdr_db"] = None
    report = checker.check_scene(scene, row, rules)
    assert not report["passed"] and report["counts"]["unavailable"] == 1
    del row["segments"]["near_only"]
    with pytest.raises(ValueError, match="missing measurement windows"):
        checker.check_scene(scene, row, rules)

def test_silent_condition_remains_unavailable_without_a_fake_pass():
    scene, row = sample()
    row["segments"]["noise_only"] = dict(total_attenuation_db=None, before_dbfs=-200, after_dbfs=-200)
    report = checker.check_scene(scene, row, rules)
    assert report["passed"]
    gate = next(g for g in report["gates"] if g["segment"] == "noise_only")
    assert gate["status"] == "unavailable" and gate["optional"]

def test_clean_ceiling_is_a_distortion_budget_not_an_exact_infinite_baseline():
    scene, row = sample()
    row["segments"]["near_only"]["before"] = {"sisdr_db": 120, "sisdr_censored": True}
    row["segments"]["near_only"]["after"]["sisdr_db"] = 19
    assert checker.check_scene(scene, row, rules)["passed"]
    row["segments"]["near_only"]["after"]["sisdr_db"] = 18.999
    assert not checker.check_scene(scene, row, rules)["passed"]

def test_frozen_inventory_rejects_deleted_or_changed_scenes():
    frozen = {"scenes": [dict(id="a", windows={"near_only": [1, 3]}, files={"before.wav": {"sha256": "x"}}),
                         dict(id="b", windows={"near_only": [1, 3]}, files={"before.wav": {"sha256": "y"}})]}
    checker.verify_manifest(deepcopy(frozen), frozen)
    reduced = deepcopy(frozen)
    reduced["scenes"].pop()
    with pytest.raises(ValueError, match="inventory"):
        checker.verify_manifest(reduced, frozen)
    changed = deepcopy(frozen)
    changed["scenes"][0]["windows"] = {}
    with pytest.raises(ValueError, match="conditions"):
        checker.verify_manifest(changed, frozen)

def test_recomputed_measurement_mismatch_is_rejected():
    checker.same_measurements({"metric": 1.2, "unavailable": None}, {"metric": 1.2, "unavailable": None})
    with pytest.raises(ValueError, match="disagrees with audio"):
        checker.same_measurements({"sisdr": 25.}, {"sisdr": -10.})
    with pytest.raises(ValueError, match="keys"):
        checker.same_measurements({}, {"metric": 1.})
    with pytest.raises(ValueError, match="disagrees with audio"):
        checker.same_measurements({"metric": True}, {"metric": 1.})

def test_weakened_rule_file_is_rejected_before_dataset_access(tmp_path):
    weaker = deepcopy(rules)
    weaker["quality"]["stoi_max_loss"] = 1
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(weaker))
    with pytest.raises(ValueError, match="thresholds differ"):
        checker.evaluate(tmp_path, rules_path=path)

def test_frozen_baseline_manifest_is_anchored(monkeypatch, tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text('{"scenes":[]}')
    monkeypatch.setattr(checker, "FROZEN_MANIFEST", path)
    with pytest.raises(ValueError, match="baseline manifest hash"):
        checker.evaluate(tmp_path)


def test_new_frozen_inventory_requires_an_explicit_correct_hash(tmp_path):
    path = tmp_path / "new-inputs.json"
    path.write_text('{"scenes":[{"id":"new"}]}')
    with pytest.raises(ValueError, match="pre-run SHA256"):
        checker.frozen_inventory(path)
    with pytest.raises(ValueError, match="baseline manifest hash"):
        checker.frozen_inventory(path, "0" * 64)
    inventory, _, sha = checker.frozen_inventory(path, checker.digest(path))
    assert inventory["scenes"][0]["id"] == "new" and sha == checker.digest(path)

def test_timing_comes_from_frames_with_nonzero_and_bad_controls():
    ev = checker.evaluator()
    rows = [{"frame": str(i), "process_ms": str(.1 + i / 100)} for i in range(100)]
    timing, late = ev.frame_timing(rows, 100)
    assert timing["process_p95_ms"] == pytest.approx(1.04)
    assert timing["process_max_ms"] == pytest.approx(1.09)
    assert timing["process_rtf"] == pytest.approx(.0595)
    assert late == 0
    rows[-1]["process_ms"] = "12"
    assert ev.frame_timing(rows, 100)[1] == 1
    with pytest.raises(ValueError, match="missing or unordered"):
        ev.frame_timing(rows[:99], 100)
    rows[1]["process_ms"] = "nan"
    with pytest.raises(ValueError, match="invalid processing"):
        ev.frame_timing(rows, 100)

def test_full_invocation_rejects_disabled_modules_and_another_reference(tmp_path):
    ev = checker.evaluator()
    scene = dict(family="aec", stress="none")
    argv = ev.invocation(scene, "full", "cli", tmp_path)
    record = dict(exit_code=0, cli_sha256="hash", argv=argv)
    checker.verify_invocation(record, scene, "full", "hash", tmp_path, ev)
    for flag in ("--no-ns", "--no-agc", "--no-limiter", "--internal-delay"):
        with pytest.raises(ValueError, match="exactly match"):
            checker.verify_invocation({**record, "argv": argv + [flag]}, scene, "full", "hash", tmp_path, ev)
    with pytest.raises(ValueError, match="exactly match"):
        checker.verify_invocation({**record, "argv": argv[:-1] + [str(tmp_path / "clean.wav")]}, scene, "full", "hash", tmp_path, ev)
    with pytest.raises(ValueError, match="oracle"):
        checker.verify_invocation({**record, "argv": argv + ["--external-delay-ms", "100"]}, scene, "full", "hash", tmp_path, ev)

def test_pcm16_zero_output_retains_unavailable_exact_and_proves_only_a_bound():
    ev = checker.evaluator()
    values = ev.attenuation_measurement(np.full(160, .1), np.zeros(160), pcm16=True)
    assert values["total_attenuation_db"] is None
    assert values["attenuation_lower_bound_db"] == pytest.approx(20 * np.log10(.1 / (.5 / 32768)))
    scene, row = sample()
    scene["noise"] = "ooffice"
    scene["windows"]["far_only"] = [4, 5]
    row["segments"]["far_only"] = values
    report = checker.check_scene(scene, row, rules)
    assert report["passed"]
    gates = [g for g in report["gates"] if g["segment"] == "far_only"]
    assert gates[0]["status"] == "unavailable" and gates[0]["optional"]
    assert gates[1]["status"] == "pass" and gates[1]["threshold"] == 15
    row["segments"]["far_only"]["attenuation_lower_bound_db"] = 14.99
    assert not checker.check_scene(scene, row, rules)["passed"]
    row["segments"]["far_only"] = ev.attenuation_measurement(np.zeros(160), np.zeros(160), pcm16=True)
    assert not checker.check_scene(scene, row, rules)["passed"]
    assert ev.attenuation_measurement(np.full(160, .1), np.zeros(160))["attenuation_lower_bound_db"] is None
    scene, row = sample()
    row["segments"]["noise_only"] = ev.attenuation_measurement(np.full(160, .1), np.zeros(160))
    assert not checker.check_scene(scene, row, rules)["passed"]
    row["segments"]["noise_only"] = values
    assert checker.check_scene(scene, row, rules)["passed"]

@pytest.mark.parametrize("profile", ["referenced_radio", "vehicle_stream"])
def test_named_processing_profile_is_exactly_checked(tmp_path, profile):
    ev = checker.evaluator()
    scene = dict(family="aec", stress="none")
    command = ev.invocation(scene, "full", "cli", tmp_path, profile)
    assert command[:3] == ["cli", "--profile", profile]
    record = dict(exit_code=0, cli_sha256="hash", argv=command)
    checker.verify_invocation(record, scene, "full", "hash", tmp_path, ev, profile)
    with pytest.raises(ValueError, match="exactly match"):
        checker.verify_invocation(record, scene, "full", "hash", tmp_path, ev)
    with pytest.raises(ValueError, match="exactly match"):
        checker.verify_invocation({**record, "argv": command + ["--no-ns"]}, scene, "full", "hash", tmp_path, ev, profile)
    with pytest.raises(ValueError, match="unknown processing profile"):
        ev.invocation(scene, "full", "cli", tmp_path, "unknown")


@pytest.mark.parametrize("profile", ["general", "referenced_radio", "vehicle_stream"])
def test_processing_profile_rejects_engine_fallback(profile):
    ev = checker.evaluator()
    engine = dict(processing_profile=profile, continuous_vehicle_policy=profile == "vehicle_stream")
    ev.verify_processing_profile(engine, profile)
    with pytest.raises(ValueError, match="processing profile differs"):
        ev.verify_processing_profile({**engine, "processing_profile": "unknown"}, profile)
    with pytest.raises(ValueError, match="vehicle policy differs"):
        ev.verify_processing_profile({**engine, "continuous_vehicle_policy": not engine["continuous_vehicle_policy"]}, profile)
    with pytest.raises(ValueError, match="processing profile differs"):
        ev.verify_processing_profile({}, profile)

def test_negative_controls_must_detect_nonzero_real_failures(tmp_path):
    scenes, rows = [], {}
    for family, mode in (("ns", "no_ns"), ("aec", "no_aec")):
        scene = dict(id=family, family=family, sample_rate=16000, windows={"noise_only": [0, 1], "far_only": [0, 1]})
        scenes.append(scene)
        folder = tmp_path / "cases" / family
        folder.mkdir(parents=True)
        checker.sf.write(folder / "before.wav", np.full(16000, .1), 16000, subtype="PCM_16")
        for name in (mode, "bypass"):
            target = folder / "runs" / name
            target.mkdir(parents=True)
            checker.sf.write(target / "result.after.wav", np.full(16000, .1), 16000, subtype="PCM_16")
            rows[(family, name)] = {"engine": {"dsp_delay_samples": 0}}
    checks = checker.negative_control_checks(tmp_path, scenes, rows, rules)
    assert all(check["verified"] for check in checks.values())
    assert checks["NS disabled"]["examples"][0]["value"] == 0
    checker.sf.write(tmp_path / "cases/ns/runs/no_ns/result.after.wav", np.full(16000, .001), 16000, subtype="PCM_16")
    checks = checker.negative_control_checks(tmp_path, scenes, rows, rules)
    assert not checks["NS disabled"]["verified"]
    assert checks["NS disabled"]["counts"] == {"unexpected_pass": 1}
    with pytest.raises(ValueError, match="negative control measurement missing"):
        checker.negative_control_checks(tmp_path, scenes, {}, rules)
