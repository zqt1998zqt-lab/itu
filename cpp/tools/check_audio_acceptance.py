#!/usr/bin/env python3
"""Frozen per-scene engineering gates. Missing measurements are never passes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import importlib.util
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

RULES = Path(__file__).resolve().parents[1] / "evaluation/acceptance-v1.json"
RULES_SHA256 = "84cda36a400ff565afe20dedf692d84608a72192d190f61c7c8428f44c1ac913"
FROZEN_MANIFEST = Path(__file__).resolve().parents[2] / "datasets/itu-eval-v1/manifest.json"
MANIFEST_SHA256 = "1b3ae045516872e8b68fc6cf5c2ef23907db066c5e22cd6d133163b9b5b391c8"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_manifest(manifest, frozen):
    scenes = manifest["scenes"]
    expected = {s["id"]: s for s in frozen["scenes"]}
    ids = [s["id"] for s in scenes]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("scene inventory differs from frozen baseline")
    for scene in scenes:
        original = expected[scene["id"]]
        # Only generated output identity may differ between paired runs.
        if {k:v for k,v in scene.items() if k != "denoise_sha256"} != {k:v for k,v in original.items() if k != "denoise_sha256"}:
            raise ValueError(f"frozen scene conditions changed: {scene['id']}")

def evaluator():
    spec = importlib.util.spec_from_file_location("itu_eval_recompute", RULES.parents[1] / "tools/run_audio_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_invocation(record, scene, mode, cli_sha, folder, ev, processing_profile="general"):
    if record["exit_code"] != 0 or record["cli_sha256"] != cli_sha:
        raise ValueError("invocation failed or used another binary")
    if mode not in ev.modes(scene) or not record["argv"]:
        raise ValueError("control invocation does not match requested configuration")
    if any(flag in record["argv"] for flag in ("--external-delay-ms", "--far-start", "--far-end", "--linear-output-diagnostic")):
        raise ValueError("diagnostic/oracle options cannot enter final acceptance")
    expected = ev.invocation(scene, mode, record["argv"][0], folder, processing_profile)
    if record["argv"] != expected:
        raise ValueError("control invocation does not exactly match requested configuration")

def same_measurements(recorded, actual, path=""):
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(actual):
            raise ValueError(f"measurement keys changed: {path}")
        for key, value in actual.items():
            same_measurements(recorded[key], value, f"{path}.{key}")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or not math.isfinite(recorded) or not math.isclose(recorded, actual, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"measurement disagrees with audio: {path}")
    elif recorded != actual:
        raise ValueError(f"measurement disagrees with audio: {path}")

def negative_control_checks(root, scenes, rows, rules):
    checks = {name: {"counts": {}, "examples": []} for name in ("NS disabled", "AEC disabled", "bypass")}
    totals = {name: Counter() for name in checks}
    for scene in scenes:
        folder = root / "cases" / scene["id"]
        before, rate = sf.read(folder / "before.wav", dtype="float64", always_2d=True)
        candidates = [("bypass", "bypass", "noise_only", rules["suppression"]["noise_only_min_attenuation_db"])]
        if scene["family"] == "ns":
            candidates.append(("NS disabled", "no_ns", "noise_only", rules["suppression"]["noise_only_min_attenuation_db"]))
        elif scene["family"] == "aec":
            candidates.append(("AEC disabled", "no_aec", "far_only", rules["suppression"]["far_only_min_attenuation_db"]))
        for name, mode, segment, threshold in candidates:
            if (scene["id"], mode) not in rows:
                raise ValueError("negative control measurement missing")
            output, output_rate = sf.read(folder / "runs" / mode / "result.after.wav", dtype="float64", always_2d=True)
            delay = rows[(scene["id"], mode)]["engine"]["dsp_delay_samples"]
            if rate != scene["sample_rate"] or output_rate != rate or output.shape[1] != 1 or len(output) != len(before) + delay or not np.isfinite(output).all():
                raise ValueError("invalid negative control output")
            start, end = (round(value * rate) for value in scene["windows"][segment])
            p = float(np.mean(before[start:end, 0] ** 2))
            q = float(np.mean(output[start + delay:end + delay, 0] ** 2))
            value = 10 * math.log10(p / q) if p > 1e-12 and q > 1e-12 else None
            status = "unavailable" if value is None else "expected_failure" if value < threshold else "unexpected_pass"
            totals[name][status] += 1
            if status == "expected_failure" and len(checks[name]["examples"]) < 3:
                checks[name]["examples"].append(dict(id=scene["id"], mode=mode, segment=segment, value=value, threshold=threshold))
    for name, check in checks.items():
        check["counts"] = dict(totals[name])
        check["verified"] = totals[name]["expected_failure"] > 0
    return checks

def check_scene(scene, row, rules):
    gates = []
    missing = set(scene["windows"]) - set(row["segments"])
    if missing:
        raise ValueError(f"missing measurement windows: {scene['id']} {sorted(missing)}")
    def gate(segment, metric, value, op, threshold, reason=None, optional=False):
        available = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        passed = available and (value >= threshold if op == ">=" else value <= threshold)
        gates.append(dict(segment=segment, metric=metric, value=value, operator=op,
                          threshold=threshold, status="pass" if passed else "fail" if available else "unavailable",
                          reason=reason if not available else None, optional=optional))
    q, s, l, t = (rules[k] for k in ("quality", "suppression", "level", "runtime"))
    wrong = scene["stress"] == "wrong_reference" and row["mode"] != "correct_reference"
    for name, segment in row["segments"].items():
        if name.startswith("near") or name == "double_talk":
            before, after = segment["before"], segment["after"]
            b, a = before.get("sisdr_db"), after.get("sisdr_db")
            if name == "double_talk":
                target = b - q["wrong_reference_sisdr_max_loss_db"] if wrong and b is not None else b + q["double_talk_sisdr_min_improvement_db"] if b is not None else None
            else:
                target = min(b, q["near_sisdr_clean_ceiling_db"]) - q["near_sisdr_max_loss_db"] if b is not None else None
            if target is None:
                gate(name, "raw_sisdr_db", None, ">=", None, "target speech or baseline unavailable")
            else:
                gate(name, "raw_sisdr_db", a, ">=", target, "zero projection / numerical boundary")
            gate(name, "stoi_change", segment.get("stoi_improvement"), ">=", -q["stoi_max_loss"], "STOI cannot be computed for this signal")
            gain = after.get("projection_gain_db")
            minimum = l["low_input_agc_min_projection_gain_db"] if scene["family"] == "agc" and scene["near_dbfs"] <= l["low_input_agc_threshold_dbfs"] else q["projection_gain_min_db"]
            gate(name, "projection_gain_min_db", gain, ">=", minimum, "zero target / projection")
            gate(name, "projection_gain_max_db", gain, "<=", q["projection_gain_max_db"], "zero target / projection")
            gate(name, "rms_min_dbfs", after.get("rms_dbfs"), ">=", l["near_rms_min_dbfs"])
            gate(name, "rms_max_dbfs", after.get("rms_dbfs"), "<=", l["near_rms_max_dbfs"])
        elif name == "noise_only":
            no_excitation = segment.get("before_dbfs", 0) <= -120
            bounded = segment.get("attenuation_lower_bound_db") is not None
            gate(name, "total_attenuation_db", segment.get("total_attenuation_db"), ">=", s["noise_only_min_attenuation_db"], "no measurable input excitation" if no_excitation else "exact attenuation censored; conservative bound required", optional=no_excitation or bounded)
            if bounded:
                gate(name, "attenuation_lower_bound_db", segment["attenuation_lower_bound_db"], ">=", s["noise_only_min_attenuation_db"])
        elif name == "far_only":
            minimum = s["wrong_reference_min_attenuation_db"] if wrong else s["far_only_min_attenuation_db"]
            bounded = segment.get("attenuation_lower_bound_db") is not None
            gate(name, "total_attenuation_db", segment.get("total_attenuation_db"), ">=", minimum, "exact attenuation censored; PCM16 conservative bound checked separately" if bounded else "digital silence / residual censored", optional=bounded)
            if bounded:
                gate(name, "attenuation_lower_bound_db", segment["attenuation_lower_bound_db"], ">=", minimum)
            if scene["noise"] == "none" and not wrong:
                gate(name, "linear_erle_engineering_db", segment.get("linear_erle_engineering_db"), ">=", s["linear_erle_min_db"], "linear tap unavailable or censored")
        elif name == "far_recovery" and scene["stress"] == "path_change":
            bounded = segment.get("attenuation_lower_bound_db") is not None
            gate(name, "total_attenuation_db", segment.get("total_attenuation_db"), ">=", s["path_change_recovery_min_attenuation_db"], "exact attenuation censored; conservative bound checked separately" if bounded else "residual censored", optional=bounded)
            if bounded:
                gate(name, "attenuation_lower_bound_db", segment["attenuation_lower_bound_db"], ">=", s["path_change_recovery_min_attenuation_db"])
    engine = row["engine"]
    if "model_nonfinite_output_hops" in engine:
        # A recovered callback must still fail backend integrity. This zero
        # fault invariant supplements quality gates without relaxing them.
        gate("execution", "model_nonfinite_output_hops", engine["model_nonfinite_output_hops"], "<=", 0)
    for key, rule in (("process_p95_ms", "process_p95_max_ms"), ("process_max_ms", "process_max_ms"), ("process_rtf", "process_rtf_max")):
        gate("runtime", key, engine.get(key), "<=", t[rule])
    gate("runtime", "frames_over_10ms", row.get("process_frames_over_10ms"), "<=", t["frames_over_10ms"])
    if engine.get("paced_10ms"):
        gate("runtime", "completion_deadline_misses", engine.get("completion_deadline_misses"), "<=", t["frames_over_10ms"], "paced deadline measurement missing")
    for key, rule in (("peak_dbfs", "output_peak_max_dbfs"), ("clipping_percent", "clipping_max_percent")):
        gate("whole_output", key, row.get("output_levels", {}).get(key), "<=", l[rule], "whole-output measurement missing")
    if wrong:
        gate("reference", "unreliable_frames", engine.get("reference_unreliable_frames"), ">=", 1, "reference reliability safeguard not implemented/reported")
    return dict(id=scene["id"], family=scene["family"], mode=row["mode"], passed=all(g["status"] == "pass" or (g["status"] == "unavailable" and g["optional"]) for g in gates),
                counts=dict(Counter(g["status"] for g in gates)), gates=gates)

def frozen_inventory(path=None, sha256=None):
    if path is None:
        path, sha256 = FROZEN_MANIFEST, MANIFEST_SHA256
    elif not sha256 or len(sha256) != 64:
        raise ValueError("an external frozen manifest requires its pre-run SHA256")
    if digest(path) != sha256:
        raise ValueError("frozen baseline manifest hash mismatch")
    return json.loads(path.read_text()), path, sha256


def recompute_scene(arguments):
    scene, mode, folder = arguments
    return evaluator().measure(scene, mode, folder, write_metrics=False)


def evaluate(root, mode="full", rules_path=RULES, diagnostic_no_controls=False, frozen_manifest=None, frozen_sha256=None, jobs=1):
    if digest(rules_path) != RULES_SHA256:
        raise ValueError("engineering thresholds differ from frozen acceptance v1")
    rules = json.loads(rules_path.read_text())
    frozen, frozen_path, frozen_sha = frozen_inventory(frozen_manifest, frozen_sha256)
    manifest = json.loads((root / "manifest.json").read_text())
    verify_manifest(manifest, frozen)
    source_sha = manifest.get("source_manifest_sha256")
    if source_sha is not None and digest(root / "sources/manifest.json") != source_sha:
        raise ValueError("source inventory hash mismatch")
    rows = json.loads((root / "results.json").read_text())
    keys = [(row["id"], row["mode"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/mode measurement")
    ev = evaluator()
    cli_sha = manifest["environment"]["cli_sha256"]
    artifact = root / "artifacts/itu_audio_cli"
    if not artifact.exists():
        artifact = FROZEN_MANIFEST.parents[1].parents[0] / "build/quality-repair/baseline-v1/build/webrtc-host/cpp/itu_audio_cli"
    if not artifact.exists() or digest(artifact) != cli_sha:
        raise ValueError("dataset binary missing or hash mismatch")
    expected_keys = {(scene["id"], name) for scene in manifest["scenes"] for name in ev.modes(scene)}
    if mode == "full" and not diagnostic_no_controls and set(keys) != expected_keys:
        raise ValueError("final acceptance requires the complete frozen control matrix")
    for scene in manifest["scenes"]:
        for name, record in scene["files"].items():
            if digest(root / "cases" / scene["id"] / name) != record["sha256"]:
                raise ValueError(f"input hash mismatch: {scene['id']} {name}")
    scene_map = {s["id"]:s for s in manifest["scenes"]}
    for row in rows:
        if row["id"] not in scene_map or row["family"] != scene_map[row["id"]]["family"]:
            raise ValueError("measurement scene/family mismatch")
        folder = root / "cases" / row["id"]
        for name, sha in row["files"].items():
            if digest(folder / name) != sha:
                raise ValueError(f"output/report hash mismatch: {row['id']} {name}")
        invocation = json.loads((folder / "runs" / row["mode"] / "invocation.json").read_text())
        verify_invocation(invocation, scene_map[row["id"]], row["mode"], cli_sha, folder, ev,
                          manifest.get("processing_profile", "general"))
    by_id = {row["id"]: row for row in rows if row["mode"] == mode}
    scenes = []
    selected_scenes = [scene for scene in manifest["scenes"] if scene["id"] in by_id]
    if mode == "full" and len(selected_scenes) != len(manifest["scenes"]):
        raise ValueError("missing full measurement")
    arguments = [(scene, mode, root / "cases" / scene["id"]) for scene in selected_scenes]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            recomputed = list(pool.map(recompute_scene, arguments, chunksize=4))
    else:
        recomputed = [ev.measure(scene, mode, folder, write_metrics=False) for scene, mode, folder in arguments]
    actual_by_id = {row["id"]: row for row in recomputed}
    for scene in manifest["scenes"]:
        if scene["id"] not in by_id:
            if mode == "full":
                raise ValueError(f"missing {mode} measurement: {scene['id']}")
            continue
        row = dict(by_id[scene["id"]])
        actual = actual_by_id[scene["id"]]
        for key in ("segments", "engine", "output_sha256", "process_frames_over_10ms"):
            same_measurements(row[key], actual[key], f"{scene['id']}.{key}")
        path = root / "cases" / scene["id"] / "runs" / mode / "result.after.wav"
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        if rate != scene["sample_rate"] or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError(f"invalid output: {path}")
        peak = float(np.max(np.abs(audio)))
        row["output_levels"] = dict(peak_dbfs=20 * math.log10(max(peak, 1e-10)), clipping_percent=float(np.mean(np.abs(audio) >= .999) * 100))
        scenes.append(check_scene(scene, row, rules))
    if not scenes:
        raise ValueError("empty acceptance run")
    bypass_errors = []
    if mode == "full" and not diagnostic_no_controls:
        for scene in manifest["scenes"]:
            folder = root / "cases" / scene["id"]
            before, _ = sf.read(folder / "before.wav", dtype="int16", always_2d=True)
            after, _ = sf.read(folder / "runs/bypass/result.after.wav", dtype="int16", always_2d=True)
            config = json.loads((folder / "runs/bypass/result.json").read_text())
            if any(config[k] for k in ("aec_enabled", "ns_enabled", "agc2_enabled")):
                raise ValueError("bypass processing modules must be disabled")
            if after.shape[1] != 1 or len(before) != len(after):
                raise ValueError("bypass length mismatch")
            bypass_errors.append(int(np.max(np.abs(before[:, 0].astype(np.int32) - after[:, 0].astype(np.int32)))))
    counts = Counter(g["status"] for scene in scenes for g in scene["gates"])
    by_metric = {}
    for scene in scenes:
        for g in scene["gates"]:
            key = g["segment"] + "." + g["metric"]
            by_metric.setdefault(key, Counter())[g["status"]] += 1
    bypass_max = max(bypass_errors) if bypass_errors else None
    bypass_passed = bypass_max is not None and bypass_max <= rules["runtime"]["bypass_max_pcm16_lsb_error"]
    controls = negative_control_checks(root, manifest["scenes"], {(row["id"], row["mode"]): row for row in rows}, rules) if mode == "full" and not diagnostic_no_controls else {}
    controls_verified = bypass_passed and bool(controls) and all(check["verified"] for check in controls.values())
    measurement_complete = all(scene["passed"] or not any(g["status"] == "unavailable" and not g["optional"] for g in scene["gates"]) for scene in scenes)
    report = dict(schema_version=1, standard_compliance=False, mode=mode, rules=rules,
                  processing_profile=manifest.get("processing_profile", "general"),
                  profile=manifest.get("profile", "baseline-v1"),
                  evaluation_scope=manifest.get("evaluation_scope", "full batch inventory"),
                  frozen_manifest_sha256=frozen_sha, frozen_manifest_path=str(frozen_path),
                  rules_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),
                  evidence_sha256={"manifest.json": digest(root / "manifest.json"),
                                   "results.json": digest(root / "results.json"),
                                   "checker": digest(Path(__file__)),
                                   "evaluator": digest(RULES.parents[1] / "tools/run_audio_eval.py"),
                                   "measurement_notes": digest(RULES.parent / "MEASUREMENT_NOTES.md")},
                  dataset_cli_sha256=manifest["environment"]["cli_sha256"], scenes=len(scenes),
                  passed_scenes=sum(s["passed"] for s in scenes), counts=dict(counts),
                  all_available_passed=mode == "full" and counts["fail"] == 0 and measurement_complete and controls_verified,
                  scope="final full matrix" if mode == "full" and not diagnostic_no_controls else "diagnostic subset; not final acceptance",
                  required_measurements_complete=measurement_complete, controls_verified=controls_verified, negative_controls=controls,
                  unavailable_capabilities={"DTD accuracy": "no labelled classifier output", "MOS/POLQA": "no licensed/validated estimator"},
                  bypass_max_pcm16_lsb_error=bypass_max, by_metric={k: dict(v) for k, v in by_metric.items()}, details=scenes)
    return report

def write_report(root, report):
    (root / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (root / "acceptance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "family", "mode", "segment", "metric", "value", "operator", "threshold", "status", "reason", "optional"])
        writer.writeheader()
        for scene in report["details"]:
            for g in scene["gates"]:
                writer.writerow({**{k: scene[k] for k in ("id", "family", "mode")}, **g})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--rules", type=Path, default=RULES)
    parser.add_argument("--diagnostic-no-controls", action="store_true", help="diagnostic sweep only; never reports final acceptance")
    parser.add_argument("--allow-failures", action="store_true", help="write an honest diagnostic report and exit zero")
    parser.add_argument("--frozen-manifest", type=Path, help="independent input inventory frozen before running this batch")
    parser.add_argument("--frozen-sha256", help="pre-run SHA256 of --frozen-manifest")
    parser.add_argument("--jobs", type=int, default=1, help="parallel offline metric recomputation; never changes C++ timing")
    args = parser.parse_args()
    report = evaluate(args.root.resolve(), args.mode, args.rules, args.diagnostic_no_controls,
                      args.frozen_manifest, args.frozen_sha256, args.jobs)
    write_report(args.root, report)
    print(json.dumps({k: report[k] for k in ("scenes", "passed_scenes", "counts", "all_available_passed", "bypass_max_pcm16_lsb_error")}, indent=2))
    raise SystemExit(0 if report["all_available_passed"] or args.allow_failures else 1)

if __name__ == "__main__":
    main()
