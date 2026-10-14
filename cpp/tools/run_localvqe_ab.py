#!/usr/bin/env python3
"""Identified serial mono streaming alternatives under frozen engineering gates.

LocalVQE receives only the real mic/render PCM. Independent clean speech is
used exclusively after inference. Multichannel scenes are explicitly excluded;
no array or multi-render downmix is substituted. This is a host diagnostic.
"""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import check_audio_acceptance as ac
import run_audio_eval as ev

MODELS = {
    "joint": ("localvqe-v1.2-1.3M-f32.gguf", "4856ecf5f522b23fb2bc5caeac81f323c0ef1c4c156a9c7d40a6adbe092ba9ce"),
    "frontend": ("localvqe-v1.4-aec-2.7K-f32.gguf", "d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4"),
    "compact": ("localvqe-pi-v1-49k-f32.gguf", "0e0c82a8e9703e818b64dedd0fc306394cf5bbb59fcec1ccca82099d352d0c26"),
}


def negative_control_summary(rows):
    negatives = [r for r in rows if r["negative_control"]]
    hits = [r for r in negatives if any(g["status"] == "fail" and not g["optional"] and
            g["metric"] in ("total_attenuation_db", "attenuation_lower_bound_db") for g in r["acceptance"]["gates"])]
    result = dict(rows=len(negatives), suppression_expected_failures=len(hits),
                  verified=bool(hits), scope="bypass must fail a measurable suppression gate")
    if negatives and not hits:
        raise ValueError("bypass negative control failed to exercise suppression gates")
    return result


def measure(scene, source, prefix, rate):
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = ev.read(source / "before.wav")[:, 0]
    clean = ev.read(source / "clean.wav")[:, 0]
    output, output_rate = sf.read(prefix.with_suffix(".after.wav"), dtype="float64")
    delay = engine["dsp_delay_samples"]
    if (output_rate != rate or output.ndim != 1 or
            len(output) != len(before) * rate // 16000 + delay or not np.isfinite(output).all()):
        raise ValueError("invalid output sample contract")
    aligned_io = output[delay:]
    before_io, before_rate = sf.read(prefix.parent.parent / "before.wav", dtype="float64")
    if before_rate != rate or before_io.shape != aligned_io.shape:
        raise ValueError("native-rate attenuation input contract mismatch")
    after = aligned_io
    gcd = math.gcd(rate, 16000)
    if rate != 16000:
        after = resample_poly(after, 16000 // gcd, rate // gcd)
    if len(after) != len(before):
        raise ValueError("invalid resampled output length")
    windows = dict(scene["windows"])
    if "cold_start" in windows:
        windows["near_cold_start"] = windows.pop("cold_start")
    segments = {}
    for name, bounds in windows.items():
        b, z = ev.window(before, bounds), ev.window(after, bounds)
        if name.startswith("near") or name == "double_talk":
            target = ev.window(clean, bounds)
            qb, qa = ev.quality(target, b), ev.quality(target, z)
            segments[name] = dict(before=qb, after=qa, stoi_improvement=
                qa["stoi"] - qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None)
        else:
            start, end = (round(t * rate) for t in bounds)
            # PCM16 half-LSB applies to actual decoded output, not polyphase
            # resampled samples. Energy ratios and censoring use native I/O.
            segments[name] = ev.attenuation_measurement(before_io[start:end], aligned_io[start:end], pcm16=True)
            if name.startswith("far") and engine.get("linear_aec_available"):
                tap, tap_rate = sf.read(prefix.with_suffix(".linear-aec.wav"), dtype="float64")
                if tap_rate != engine["linear_aec_sample_rate"]:
                    raise ValueError("linear tap rate mismatch")
                ld = engine["linear_aec_delay_samples"]
                residual = tap[ld:ld + len(before) * tap_rate // 16000]
                tap_gcd = math.gcd(tap_rate, 16000)
                if tap_rate != 16000:
                    residual = resample_poly(residual, 16000 // tap_gcd, tap_rate // tap_gcd)
                residual = ev.window(residual, bounds)
                if engine.get("linear_erle_uses_conditioned_input"):
                    conditioned, cr = sf.read(prefix.with_suffix(".aec-input.wav"), dtype="float64")
                    if cr != rate:
                        raise ValueError("AEC input tap rate mismatch")
                    cd = engine["aec_input_delay_samples"]
                    conditioned = conditioned[cd:cd + len(before) * rate // 16000]
                    if rate != 16000:
                        conditioned = resample_poly(conditioned, 16000 // gcd, rate // gcd)
                    b = ev.window(conditioned, bounds)
                segments[name]["linear_erle_engineering_db"] = (
                    ev.db(ev.power(b)) - ev.db(ev.power(residual))
                    if scene["noise"] == "none" and ev.power(b) > 1e-12 and ev.power(residual) > 1e-12 else None)
    with prefix.with_suffix(".frames.csv").open() as f:
        timing, over = ev.frame_timing(list(csv.DictReader(f)), math.ceil(len(before) / 160))
    for key, value in timing.items():
        if not np.isclose(engine[key], value, rtol=2e-5, atol=1e-8):
            raise ValueError("timing disagrees with actual frame CSV")
        engine[key] = value
    measured = dict(mode="full", segments=segments, engine=engine, process_frames_over_10ms=over,
        output_levels=dict(peak_dbfs=ev.db(float(np.max(np.abs(output))) ** 2),
                           clipping_percent=float(np.mean(np.abs(output) >= .999) * 100)))
    return dict(scene, windows=windows), measured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "root", "cli", "probe", "library", "models"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rates", type=int, nargs="+", choices=(16000, 24000, 48000), default=[16000, 24000, 48000])
    p.add_argument("--variants", nargs="+", choices=("control", "joint", "compact", "frontend", "frontend_webrtc", "frontend_ulunas", "frontend_webrtc_guard", "bypass"), default=["control", "joint", "frontend"])
    p.add_argument("--post-model", type=Path, help="UL-UNAS model; required for frontend_ulunas")
    p.add_argument("--scene", action="append", default=[])
    a = p.parse_args()
    root, dataset = a.root.resolve(), a.dataset.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = dataset / "inputs-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen engineering thresholds changed")
    rules = json.loads(ac.RULES.read_text())
    eligible = [s for s in manifest["scenes"] if s["mics"] == 1 and s.get("reference_channels", 1) == 1]
    scenes = [s for s in eligible if not a.scene or s["id"] in a.scene]
    if not scenes or set(a.scene) - {s["id"] for s in scenes}:
        p.error("requires nonempty, known mono/single-render scenes")
    artifacts = root / "artifacts"
    artifacts.mkdir()
    snapshot = {}
    for key, path in (("cli", a.cli), ("probe", a.probe), ("library", a.library)):
        dest = artifacts / path.name
        shutil.copy2(path.resolve(), dest)
        snapshot[key] = dest
    if "frontend_ulunas" in a.variants and (a.post_model is None or not a.post_model.is_file()):
        p.error("frontend_ulunas requires --post-model")
    model_variants = {"frontend" if v.startswith("frontend") else v for v in a.variants} & set(MODELS)
    if "compact" in model_variants:
        model_variants.add("frontend")  # explicit allowlisted companion DAF; auto-detected by pinned SDK
    for name in model_variants:
        filename, expected = MODELS[name]
        source = a.models / filename
        if ev.digest(source) != expected:
            raise ValueError("pinned model hash mismatch")
        dest = artifacts / filename
        shutil.copy2(source, dest)
        snapshot[name] = dest
    if "frontend_ulunas" in a.variants:
        dest = artifacts / a.post_model.name
        shutil.copy2(a.post_model.resolve(), dest)
        snapshot["post_model"] = dest
    shutil.copy2(Path(__file__), artifacts / Path(__file__).name)
    identity = dict(manifest_sha256=ev.digest(manifest_path), rules_sha256=ev.digest(ac.RULES),
        runner_sha256=ev.digest(Path(__file__)), measure_sha256=ev.digest(Path(ev.__file__)),
        acceptance_sha256=ev.digest(Path(ac.__file__)),
        artifacts={k: dict(path=str(v.relative_to(root)), sha256=ev.digest(v)) for k, v in snapshot.items()},
        rates=a.rates, variants=a.variants, scene_ids=[s["id"] for s in scenes],
        excluded_multichannel_ids=[s["id"] for s in manifest["scenes"] if s not in eligible],
        backend_core_rate=16000, native_highband=False, standard_compliance=False,
        scope="serial host mono/single-render diagnostic; actual 10ms callbacks; no clip priming, gate, whole-clip gain normalization or lag fitting",
        compact_normalization="pinned SDK causal current-hop RMS EMA, undone before overlap-add; no whole-clip statistics",
        frontend_scope="frontend alone is AEC-only; frontend_webrtc/ulunas add independent NS/AGC at 16k; joint/compact lack linear taps",
        target_scope="frozen channel-0 clean; explicit declared causal DSP delay only")
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    rows = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for scene in scenes:
        source = dataset / "cases" / scene["id"]
        for name, record in scene["files"].items():
            if ev.digest(source / name) != record["sha256"]:
                raise ValueError("frozen input hash mismatch")
        for rate in a.rates:
            folder = root / f"{scene['id']}-r{rate}"
            folder.mkdir()
            for name in ("before.wav", "ref.wav"):
                x = ev.read(source / name)
                if x.shape[1] != 1:
                    raise ValueError("multichannel input rejected")
                gcd = math.gcd(rate, 16000)
                if rate != 16000:
                    x = resample_poly(x, rate // gcd, 16000 // gcd, axis=0)
                sf.write(folder / name, x, rate, subtype="FLOAT")
            input_hashes = {name: ev.digest(folder / name) for name in ("before.wav", "ref.wav")}
            for variant in a.variants:
                run = folder / variant
                run.mkdir()
                prefix = run / "result"
                command = [str(snapshot["cli" if variant in ("control", "bypass") else "probe"])]
                if variant in ("control", "bypass"):
                    command += ["--profile", "vehicle_stream"]
                    if variant == "bypass":
                        command += ["--no-aec", "--no-ns", "--no-agc", "--no-limiter", "--no-low-cut"]
                    else:
                        command += ["--noise-transition-control"]
                else:
                    model_name = "frontend" if variant.startswith("frontend") else variant
                    command += ["--library", str(snapshot["library"]), "--model", str(snapshot[model_name])]
                    if variant.startswith("frontend"):
                        command += ["--frontend-only"]
                        if variant != "frontend":
                            command += ["--post-ns", variant.removeprefix("frontend_").removesuffix("_guard")]
                        if variant.endswith("_guard"):
                            command += ["--aec-divergence-guard"]
                        if variant == "frontend_ulunas":
                            command += ["--post-model", str(snapshot["post_model"])]
                command += ["--mic", str(folder / "before.wav"), "--echo", str(folder / "ref.wav"), "--out", str(prefix)]
                proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, timeout=180)
                (run / "process.log").write_text(proc.stdout)
                (run / "invocation.json").write_text(json.dumps(dict(argv=command, exit_code=proc.returncode,
                    input_sha256=input_hashes), indent=2) + "\n")
                if proc.returncode:
                    # A failing backend is a failed scenario, not a deleted
                    # case and not a zero-valued fabricated quality result.
                    row = dict(id=scene["id"], pair=scene["pair"], label=scene.get("field_label", scene["noise"]),
                        rate=rate, variant=variant, negative_control=variant=="bypass", execution_failed=True,
                        quality_passed=False, runtime_passed=False,
                        measurements=dict(unavailable_reason="backend failed; see retained process.log"),
                        acceptance=dict(passed=False, gates=[dict(segment="execution", metric="backend_execution",
                            value=proc.returncode, threshold=0, operator="==", status="fail", optional=False)]),
                        files={f.name: ev.digest(f) for f in sorted(run.iterdir()) if f.is_file()})
                    (run / "measurements.json").write_text(json.dumps(row,indent=2)+"\n")
                    rows.append(row)
                    print(json.dumps(dict(id=row["id"],rate=rate,variant=variant,execution_failed=True)),flush=True)
                    continue
                checked_scene, measured = measure(scene, source, prefix, rate)
                verdict = ac.check_scene(checked_scene, measured, rules)
                def passed(gates):
                    return all(g["status"] == "pass" or g["status"] == "unavailable" and g["optional"] for g in gates)
                row = dict(id=scene["id"], pair=scene["pair"], label=scene.get("field_label", scene["noise"]),
                    rate=rate, variant=variant, negative_control=variant == "bypass", measurements=measured,
                    acceptance=verdict, quality_passed=passed([g for g in verdict["gates"] if g["segment"] != "runtime"]),
                    runtime_passed=passed([g for g in verdict["gates"] if g["segment"] == "runtime"]),
                    files={f.name: ev.digest(f) for f in sorted(run.iterdir()) if f.is_file()})
                (run / "measurements.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("id", "rate", "variant", "quality_passed", "runtime_passed")}), flush=True)
                (root / "progress.json").write_text(json.dumps(dict(done=len(rows), planned=len(scenes)*len(a.rates)*len(a.variants))) + "\n")
    groups = {}
    for rate in a.rates:
        for variant in a.variants:
            selected = [r for r in rows if r["rate"] == rate and r["variant"] == variant]
            groups[f"{rate}-{variant}"] = dict(rows=len(selected), quality_pass=sum(r["quality_passed"] for r in selected),
                runtime_pass=sum(r["runtime_passed"] for r in selected), all_pass=sum(r["acceptance"]["passed"] for r in selected),
                failed_metrics=dict(Counter(g["metric"] for r in selected for g in r["acceptance"]["gates"] if g["status"] != "pass" and not g["optional"])))
    for key, path in snapshot.items():
        if ev.digest(path) != identity["artifacts"][key]["sha256"]:
            raise ValueError("artifact changed during run")
    if ev.digest(manifest_path) != identity["manifest_sha256"] or ev.digest(ac.RULES) != identity["rules_sha256"]:
        raise ValueError("manifest or rules changed during run")
    (root / "report.json").write_text(json.dumps(dict(identity=identity, groups=groups, rows=rows,
        negative_controls=negative_control_summary(rows)), indent=2, allow_nan=False) + "\n")
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
