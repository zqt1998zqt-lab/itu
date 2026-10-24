#!/usr/bin/env python3
"""Replay published-rate sources without corpus resampling through the VCP CLI.

This is a new, explicitly scoped extension, not the historical 20,298-case batch.
Engineering rules remain frozen. Missing required metrics do not count as passes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy import signal
import soundfile as sf

import run_audio_eval as ev
import check_audio_acceptance as acceptance

RULES_SHA = "84cda36a400ff565afe20dedf692d84608a72192d190f61c7c8428f44c1ac913"
POOLS = {}
RIRS = {}


def save(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_path(manifest, entry):
    return (manifest.parent / entry["path"]).resolve()


def source_pool(manifest, rate):
    key = (str(manifest), rate)
    if key in POOLS:
        return POOLS[key]
    records = json.loads(manifest.read_text())["sources"]
    speech = defaultdict(list)
    provenance = defaultdict(list)
    noise = {}
    for entry in records:
        if entry["sample_rate"] != rate:
            continue
        if entry["kind"] == "clean_speech":
            speaker = str(entry["speaker"])
            if sum(len(x) for x in speech[speaker]) >= 60 * rate:
                continue
        elif entry["kind"] == "noise":
            if entry["scene"].lower() not in ("tcar", "straffic"):
                continue
        else:
            continue
        path = source_path(manifest, entry)
        if ev.digest(path) != entry["sha256"]:
            raise ValueError(f"source hash changed: {path}")
        audio, actual = sf.read(path, dtype="float64", always_2d=True)
        if actual != rate or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"invalid native source: {path}")
        if entry["kind"] == "clean_speech":
            speech[speaker].append(audio[:, 0])
            provenance[speaker].append(dict(path=str(path), sha256=entry["sha256"],
                                            samples=len(audio), rate=actual))
        else:
            noise[entry["scene"].lower()] = audio[:, 0]
    result = ({k: np.concatenate(v) for k, v in speech.items()}, noise, provenance)
    POOLS[key] = result
    return result


def matrix(manifest):
    entries = json.loads(manifest.read_text())["sources"]
    cases = []
    for rate in (16000, 24000, 48000):
        names = sorted({str(r["speaker"]) for r in entries
                        if r["sample_rate"] == rate and r["kind"] == "clean_speech"})
        if len(names) < 2:
            raise ValueError(f"clean/reference speaker gap at {rate}")
        for pair in range(2):
            near = names[(2 * pair) % len(names)]
            far = names[(2 * pair + 1) % len(names)]
            if len(names) == 2 and pair:
                near, far = far, near
            for microphones in (1, 2, 4):
                base = dict(sample_rate=rate, pair=pair, near_speaker=near,
                    far_speaker=far, near_dbfs=-26, mics=microphones, noise="none",
                    snr_db=None, ser_db=-5, rt60_s=.15, echo_delay_ms=30,
                    stress="none", profile="native-rate-extension-v1", pcm_subtype="FLOAT",
                    room_dimensions_m=[3.6, 1.8, 1.5],
                    microphone_positions_m=[[1.4 + (ch - (microphones - 1) / 2) * .035, .9, 1.3]
                                            for ch in range(microphones)],
                    near_position_m=[.8, .65, 1.1],
                    render_positions_m=[[.2, .3, .8], [.2, 1.5, .8]])
                # The 24 kHz bank has no verified released-rate vehicle noise.
                noises = ("white", "pink", "burst", "hum") if rate == 24000 else ("white", "pink", "tcar", "straffic")
                for noise in noises:
                    for snr in range(-10, 31, 5):
                        cases.append(dict(base, family="ns", noise=noise, snr_db=snr,
                                          near_windows=[[0, 10], [12, 22]], field_label="cold_start"))
                for rt60 in (.15, .45):
                    for delay in (20, 80, 180):
                        cases.append(dict(base, family="aec", rt60_s=rt60, echo_delay_ms=delay))
                for noise in noises[-2:]:
                    for snr in (0, 15, 30):
                        cases.append(dict(base, family="combined", noise=noise, snr_db=snr))
                for stress in ("nonlinear_speaker", "clock_300ppm", "wrong_reference"):
                    cases.append(dict(base, family="stress", noise=noises[-1], snr_db=15,
                                      rt60_s=.3, stress=stress))
                for level in (-45, -30, -15):
                    cases.append(dict(base, family="agc", near_dbfs=level, noise=noises[-1], snr_db=20))
                cases.append(dict(base, family="ns", noise=noises[-1], snr_db=20,
                    noise_steps=[dict(time_s=11, gain_db=9)], field_label="noise_step"))
    for index, case in enumerate(cases):
        case["id"] = f"native-{case['sample_rate']}-m{case['mics']}-{index:04d}-{case['family']}"
        case["seed"] = 2026101600 + index
        case["noise_origin"] = ("published-rate real recording" if case["noise"] in ("tcar", "straffic")
                                else "synthetic at the requested sample rate" if case["noise"] != "none" else "none")
    return cases


def invoke(command, prefix, cli_sha):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    process = subprocess.run(command, capture_output=True, text=True, timeout=150)
    prefix.with_suffix(".log").write_text(process.stdout + process.stderr)
    save(prefix.parent / "invocation.json", dict(argv=command, cli_sha256=cli_sha,
         exit_code=process.returncode, wall_seconds=time.time() - start))
    if process.returncode:
        raise RuntimeError(f"CLI exit {process.returncode}: {prefix}; {process.stderr[-1000:]}")


def transport(playback, scene):
    """Apply the predeclared simulated clock/nonlinearity, never fit the output."""
    if scene["stress"] == "clock_300ppm":
        return np.interp(np.arange(len(playback)) * 1.0003, np.arange(len(playback)),
                         playback, left=0, right=0)
    if scene["stress"] == "nonlinear_speaker":
        common = scene["common_gain"]
        return np.tanh(playback / common * 6) / 6 * common
    return playback


def prepare(case, root, manifest, cli, cli_sha):
    rate = case["sample_rate"]
    ev.RATE = rate
    speakers, noises, provenance = source_pool(manifest, rate)
    folder = root / "cases" / case["id"]
    # The legacy generator assumes one RATE per process and omits fs from its
    # RIR cache key. Separate banks here: a 16 kHz RIR cannot enter a 24/48 kHz case.
    scene = ev.make_case(case, speakers, noises, folder, RIRS.setdefault(rate, {}))
    scene["source_speech_files"] = {s: provenance[s] for s in (case["near_speaker"], case["far_speaker"])}
    scene["sample_rate_conversion"] = False
    scene["simulated_clock_time_warp"] = scene["stress"] == "clock_300ppm"
    scene["geometry_scope"] = "existing synthetic vehicle-field geometry; not a measured SU7 RIR"
    acoustic = case["family"] not in ("ns", "agc")
    if acoustic:
        h_shape = np.load(folder / "echo-rir.npy", mmap_mode="r").shape
        if h_shape != (round(scene["rt60_s"] * rate), scene["mics"]):
            raise ValueError(f"RIR rate/shape mismatch: {h_shape} at {rate} Hz")
        scene["rir_sample_rate"] = rate
    if acoustic and case["stress"] != "wrong_reference":
        # Generate the echo from ACTUAL processed playback. A fresh duplex replay
        # must reproduce this tap bit-for-bit; raw RX is never treated as playback.
        oracle = folder / "oracle" / "rx"
        invoke([cli, "--profile", "vcp", "--rx-in", str(folder / "playback.wav"),
                "--out", str(oracle)], oracle, cli_sha)
        raw = ev.read(folder / "playback.wav")[:, 0]
        actual = ev.read(oracle.with_suffix(".rx-spk.wav"))[:, 0]
        h = np.load(folder / "echo-rir.npy")
        delay = round(scene["echo_delay_ms"] * rate / 1000)
        old_linear = np.column_stack([ev.shift(signal.fftconvolve(transport(raw, scene), h[:, ch])[:len(raw)], delay)
                                      for ch in range(scene["mics"])])
        old_echo = ev.read(folder / "echo-component.wav")
        denominator = float(np.sum(old_linear * old_linear))
        if denominator <= 1e-15:
            raise ValueError("echo calibration is unexcited")
        # Recover the generator's scalar RIR normalization from known components.
        # It is fixed before candidate processing and uses no candidate output.
        scale = float(np.sum(old_linear * old_echo) / denominator)
        if np.max(np.abs(old_linear * scale - old_echo)) > 2e-6:
            raise ValueError("generator echo normalization could not be verified")
        echo = scale * np.column_stack([ev.shift(signal.fftconvolve(transport(actual, scene), h[:, ch])[:len(raw)], delay)
                                       for ch in range(scene["mics"])])
        before = ev.read(folder / "clean-mics.wav") + echo + ev.read(folder / "noise-component.wav")
        ev.write(folder / "before.wav", before, "FLOAT")
        ev.write(folder / "echo-component.wav", echo, "FLOAT")
        scene["echo_rir_normalization"] = scale
        scene["reference_mode"] = "duplex_actual_post_rx_playback"
        scene["rx_input"] = str(folder / "playback.wav")
        scene["actual_playback_sha256"] = ev.digest(oracle.with_suffix(".rx-spk.wav"))
    else:
        scene["reference_mode"] = "deliberately_wrong_provided_reference" if acoustic else "no_playback"
    if scene.get("field_label") == "cold_start":
        scene["windows"]["near_start"] = [.05, .5]
    if scene.get("field_label") == "noise_step":
        scene["windows"]["noise_only_before_step"] = [10, 11]
        scene["windows"]["noise_only_after_step"] = [11, 12]
    scene["files"] = {p.name: dict(sha256=ev.digest(p), bytes=p.stat().st_size)
                      for p in sorted(folder.iterdir()) if p.is_file() and p.name != "scene.json"}
    save(folder / "scene.json", scene)
    return scene


def command_for(scene, folder, prefix, cli, mode, paced=False):
    command = [cli, "--profile", "vcp", "--mic", str(folder / "before.wav"), "--out", str(prefix)]
    if scene["family"] in ("ns", "agc"):
        command += ["--no-aec"]
    elif scene["reference_mode"] == "duplex_actual_post_rx_playback":
        command += ["--rx-in", scene["rx_input"]]
    else:
        command += ["--echo", str(folder / "ref.wav")]
    if scene["mics"] > 1:
        command += ["--beam-delays", ",".join(f"{d:.9f}" for d in scene["beam_delays_samples"])]
    if mode == "negative_no_ns":
        command += ["--no-ns", "--no-agc"]
    elif mode == "negative_no_aec":
        command += ["--no-aec", "--no-ns", "--no-agc"]
    if paced:
        command += ["--pace-10ms"]
    return command


def measure(scene, root, mode):
    rate = scene["sample_rate"]
    ev.RATE = rate
    folder = root / "cases" / scene["id"]
    prefix = folder / "runs" / mode / "result"
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = ev.read(folder / "before.wav")[:, 0]
    target = ev.read(folder / ("clean-beam.wav" if scene["mics"] > 1 else "clean.wav"))[:, 0]
    output = ev.read(prefix.with_suffix(".after.wav"))[:, 0]
    delay = engine["dsp_delay_samples"]
    if len(output) != len(before) + delay or not np.isfinite(output).all():
        raise ValueError("output length/delay/finite check failed")
    if engine["processing_profile"] != "vcp" or engine["external_vad_enabled"]:
        raise ValueError("requested profile/VAD policy differs")
    if scene["mics"] > 1 and (not engine["beamforming_before_aec"] or engine["aec_capture_channels"] != 1):
        raise ValueError("BF-before-AEC topology differs")
    if scene["reference_mode"] == "duplex_actual_post_rx_playback":
        oracle = folder / "oracle/rx.rx-spk.wav"
        if ev.digest(prefix.with_suffix(".rx-spk.wav")) != ev.digest(oracle):
            raise ValueError("live RX differs from the playback used to synthesize echo")
        if ev.digest(prefix.with_suffix(".rx-ref.wav")) != ev.digest(oracle):
            raise ValueError("reference differs from actual playback")
        if engine["rx_processing_frames"] != int(np.ceil(len(before) / (rate // 100))):
            raise ValueError("RX did not process exactly once per frame")
    aligned = output[delay:delay + len(before)]
    segments = {}
    linear = conditioned = None
    if engine["aec_enabled"] and engine["linear_aec_available"] and rate == 16000:
        linear = ev.read(prefix.with_suffix(".linear-aec.wav"))[:, 0]
        ld = engine["linear_aec_delay_samples"]
        linear = linear[ld:ld + len(before)]
        conditioned = ev.read(prefix.with_suffix(".aec-input.wav"))[:, 0]
        cd = engine["aec_input_delay_samples"]
        conditioned = conditioned[cd:cd + len(before)]
    for name, limits in scene["windows"].items():
        b, a, clean = (ev.window(x, limits, rate=rate) for x in (before, aligned, target))
        if name.startswith("near") or name == "double_talk":
            qb, qa = ev.quality(clean, b), ev.quality(clean, a)
            segments[name] = dict(before=qb, after=qa,
                sisdr_improvement_db=qa["sisdr_db"] - qb["sisdr_db"] if qa["sisdr_db"] is not None and qb["sisdr_db"] is not None else None,
                stoi_improvement=qa["stoi"] - qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None)
        else:
            segments[name] = ev.attenuation_measurement(b, a, pcm16=True)
            if name.startswith("far"):
                erle = None
                if linear is not None and scene["noise"] == "none":
                    lin = ev.window(linear, limits, rate=rate)
                    cond = ev.window(conditioned, limits, rate=rate)
                    if ev.power(lin) > 1e-12 and ev.power(cond) > 1e-12:
                        erle = ev.db(ev.power(cond)) - ev.db(ev.power(lin))
                segments[name]["linear_erle_engineering_db"] = erle
                segments[name]["linear_erle_scope"] = ("same-rate conditioned input and linear residual"
                    if rate == 16000 else "unavailable: engine linear tap is 16 kHz; no unequal-band power comparison")
    with prefix.with_suffix(".frames.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    timing, overruns = ev.frame_timing(frames, round(scene["seconds"] * 100))
    for key, value in timing.items():
        if not np.isclose(engine[key], value, rtol=2e-5, atol=1e-8):
            raise ValueError(f"timing report/CSV mismatch: {key}")
        engine[key] = value
    row = dict(id=scene["id"], family=scene["family"], sample_rate=rate, microphones=scene["mics"],
        mode=mode, engine=engine, segments=segments, process_frames_over_10ms=overruns,
        output_sha256=ev.digest(prefix.with_suffix(".after.wav")),
        output_levels=dict(peak_dbfs=ev.db(float(np.max(np.abs(output))) ** 2),
                           clipping_percent=float(np.mean(np.abs(output) >= .999) * 100)),
        alignment="only declared causal DSP delay removed; no per-case lag optimization",
        stoi_scope="native-rate PCM passed to standard STOI; its internal metric filtering is not corpus resampling",
        timing_scope="parallel offline callback timing; paced verification reported separately")
    gates = acceptance.check_scene(scene, row, json.loads(acceptance.RULES.read_text()))
    # Also apply the existing 6 dB noise-only rule to the two transition windows.
    for name in ("noise_only_before_step", "noise_only_after_step"):
        if name in segments:
            value = segments[name]["total_attenuation_db"]
            lower = segments[name]["attenuation_lower_bound_db"]
            if value is None:
                value = lower
            gates["gates"].append(dict(segment=name, metric="total_attenuation_db", value=value,
                operator=">=", threshold=6., status="unavailable" if value is None else "pass" if value >= 6 else "fail",
                reason="same frozen noise-only threshold applied to noise transition", optional=False))
    gates["counts"] = dict(Counter(g["status"] for g in gates["gates"]))
    gates["passed"] = all(g["status"] == "pass" or (g["status"] == "unavailable" and g["optional"]) for g in gates["gates"])
    gates["status"] = ("pass" if gates["passed"] else "fail" if any(g["status"] == "fail" for g in gates["gates"])
                         else "incomplete_required_measurement")
    save(prefix.parent / "metrics.json", row)
    save(prefix.parent / "acceptance.json", gates)
    return dict(scene=scene, metrics=row, acceptance=gates)


def run_one(arguments):
    case, root, manifest, cli, cli_sha = arguments
    try:
        scene = prepare(case, root, manifest, cli, cli_sha)
        prefix = root / "cases" / scene["id"] / "runs/full/result"
        invoke(command_for(scene, prefix.parents[2], prefix, cli, "full"), prefix, cli_sha)
        return measure(scene, root, "full")
    except Exception:
        return dict(id=case["id"], sample_rate=case["sample_rate"], error=traceback.format_exc())


def summary(rows):
    statuses = Counter()
    rates = defaultdict(Counter)
    failures = Counter()
    for row in rows:
        if "error" in row:
            statuses["execution_error"] += 1
            rates[row["sample_rate"]]["execution_error"] += 1
            continue
        status = row["acceptance"]["status"]
        statuses[status] += 1
        rates[row["scene"]["sample_rate"]][status] += 1
        for gate in row["acceptance"]["gates"]:
            if gate["status"] != "pass" and not gate.get("optional"):
                failures[(gate["status"], gate["segment"], gate["metric"])] += 1
    return dict(status_counts=dict(statuses), by_rate={str(k): dict(v) for k, v in rates.items()},
        nonpassing_gates=[dict(status=k[0], segment=k[1], metric=k[2], count=v) for k, v in failures.most_common()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--pilot", action="store_true", help="one combined scene per rate/mic combination")
    parser.add_argument("--paced", action="store_true", help="sequential isolated paced checks of all nine rate/mic combinations")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("jobs must be 1..8")
    root, manifest, cli = args.root.resolve(), args.sources.resolve(), args.cli.resolve()
    if root.exists():
        parser.error("use a new output directory; previous evidence is immutable")
    if ev.digest(acceptance.RULES) != RULES_SHA:
        raise ValueError("frozen engineering thresholds changed")
    root.mkdir(parents=True)
    (root / "artifacts").mkdir()
    artifact = root / "artifacts/itu_audio_cli"
    shutil.copy2(cli, artifact)
    cli_sha = ev.digest(artifact)
    dependencies = [Path(__file__), Path(ev.__file__), Path(acceptance.__file__), acceptance.RULES, manifest]
    identity = dict(cli_sha256=cli_sha, source_manifest=str(manifest), sample_rate_conversion=False,
        profile="vcp", rules_sha256=RULES_SHA, jobs=args.jobs, paced=args.paced,
        dependencies={str(p.resolve()): ev.digest(p) for p in dependencies})
    save(root / "identity.json", identity)
    cases = matrix(manifest)
    if args.pilot:
        cases = [next(c for c in cases if c["sample_rate"] == r and c["mics"] == m and c["family"] == "combined")
                 for r in (16000, 24000, 48000) for m in (1, 2, 4)]
    save(root / "plan.json", dict(scenes=cases, count=len(cases),
        scope="native published-rate extension; not the previous 20,298-case frozen regression",
        source_gaps=json.loads(manifest.read_text())["unresolved_gaps"],
        sources_may_have_publisher_resampling=True,
        snr_grid_db=list(range(-10, 31, 5)), rates=[16000, 24000, 48000], microphones=[1, 2, 4]))
    rows = []
    start = time.monotonic()
    save(root / "report.json", dict(state="running", completed=0, total=len(cases)))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, (case, root, manifest, str(artifact), cli_sha)) for case in cases]
        with (root / "results.jsonl").open("w") as stream:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                elapsed = time.monotonic() - start
                status = "execution_error" if "error" in row else row["acceptance"]["status"]
                print(json.dumps(dict(completed=len(rows), total=len(cases), status=status,
                    id=row.get("id", row.get("scene", {}).get("id")), elapsed_s=elapsed)), flush=True)
                if len(rows) % 10 == 0 or len(rows) == len(cases):
                    save(root / "report.json", dict(state="running", completed=len(rows), total=len(cases),
                        elapsed_s=elapsed, scenes_per_minute=len(rows) * 60 / elapsed, **summary(rows)))
    errors = [r for r in rows if "error" in r]
    controls = []
    paced = []
    if not errors:
        for rate in (16000, 24000, 48000):
            for microphones in (1, 2, 4):
                for family, mode, segment in (("ns", "negative_no_ns", "noise_only"), ("aec", "negative_no_aec", "far_only")):
                    selected = next((r for r in rows if r["scene"]["sample_rate"] == rate and r["scene"]["mics"] == microphones
                                     and r["scene"]["family"] == family), None)
                    if selected is None:
                        continue
                    scene = selected["scene"]
                    folder = root / "cases" / scene["id"]
                    prefix = folder / "runs" / mode / "result"
                    invoke(command_for(scene, folder, prefix, str(artifact), mode), prefix, cli_sha)
                    result = measure(scene, root, mode)
                    gate = next(g for g in result["acceptance"]["gates"] if g["segment"] == segment and g["metric"] == "total_attenuation_db")
                    controls.append(dict(id=scene["id"], mode=mode, sample_rate=rate, microphones=microphones,
                        attenuation_db=gate["value"], threshold=gate["threshold"], verified=gate["status"] == "fail"))
        if args.paced:
            save(root / "report.json", dict(state="paced_validation", completed=len(rows), total=len(cases), **summary(rows)))
            for rate in (16000, 24000, 48000):
                for microphones in (1, 2, 4):
                    selected = next(r for r in rows if r["scene"]["sample_rate"] == rate and
                                    r["scene"]["mics"] == microphones and r["scene"]["family"] == "combined")
                    scene = selected["scene"]
                    prefix = root / "cases" / scene["id"] / "runs/paced/result"
                    invoke(command_for(scene, prefix.parents[2], prefix, str(artifact), "full", paced=True), prefix, cli_sha)
                    result = measure(scene, root, "paced")
                    paced.append(result)
                    save(root / "paced-results.json", paced)
                    print(json.dumps(dict(paced_completed=len(paced), sample_rate=rate, microphones=microphones,
                        process_p95_ms=result["metrics"]["engine"]["process_p95_ms"],
                        completion_deadline_misses=result["metrics"]["engine"]["completion_deadline_misses"])), flush=True)
    for path, expected in identity["dependencies"].items():
        if ev.digest(path) != expected:
            raise ValueError(f"dependency changed during run: {path}")
    if ev.digest(artifact) != cli_sha:
        raise ValueError("executed binary changed during run")
    save(root / "controls.json", controls)
    save(root / "report.json", dict(state="completed_with_execution_errors" if errors else "completed",
        completed=len(rows), total=len(cases), elapsed_s=time.monotonic() - start,
        created_utc=datetime.now(timezone.utc).isoformat(), identity_verified=True,
        cli_sha256=cli_sha, standard_compliance=False, **summary(rows),
        negative_controls=controls, paced_cases=len(paced),
        paced_runtime_failures=sum(g["status"] != "pass" for r in paced for g in r["acceptance"]["gates"] if g["segment"] == "runtime"),
        source_gaps=json.loads(manifest.read_text())["unresolved_gaps"],
        unavailable_scope="24/48 kHz linear ERLE requires common-band evaluator; unavailable is not pass; no licensed POLQA or hardware ITU certification"))
    print(json.dumps(json.loads((root / "report.json").read_text())), flush=True)


if __name__ == "__main__":
    main()
