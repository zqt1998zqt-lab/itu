#!/usr/bin/env python3
"""Replay v2 native-rate multilingual (EN+ZH) sources with:
1. Active onset alignment so the 450 ms near_start window [0.05, 0.50] s always contains >=350 ms of active speech (0 unavailable STOI).
2. Expanded native 16/24/48 kHz English + Mandarin Chinese speaker pairs and real 24/48 kHz noise sources (0 DC/clipping flags).
3. Common-band (0-8 kHz) linear ERLE measurement for 24/48 kHz AEC scenes (0 unavailable linear_erle_engineering_db).
Engineering acceptance thresholds in acceptance-v1.json remain strictly frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from pystoi import stoi
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


def align_onset_phase(arr: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Circularly shift concatenated speaker pool so [0.05s, 0.50s] has >=35 active 10 ms frames and valid STOI."""
    hop = sr // 100
    w0, w1 = int(round(0.05 * sr)), int(round(0.50 * sr))
    flen = int(round(0.0256 * sr))
    fhop = int(round(0.0128 * sr))
    rng = np.random.default_rng(20261017)
    probe_noise = 1e-5 * rng.normal(size=w1 - w0)
    for step_frames in range(0, max(1, len(arr) // hop - 80), 2):
        off = step_frames * hop
        seg = arr[off + w0 : off + w1]
        if len(seg) < w1 - w0:
            break
        if np.sqrt(np.mean(seg**2)) < 0.015:
            continue
        frames = [seg[i : i + flen] for i in range(0, len(seg) - flen + 1, fhop)]
        energies = np.array([np.sum(f**2) for f in frames])
        max_e = np.max(energies) + 1e-20
        active_cnt = int(np.sum(10.0 * np.log10((energies + 1e-20) / max_e) > -35.0))
        if active_cnt >= 32:
            val = stoi(seg, seg + probe_noise, sr, extended=False)
            if np.isfinite(val) and val > 0.95:
                return np.roll(arr, -off), int(off)
    return arr, 0


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
            if entry["scene"].lower() not in ("tcar", "straffic", "tbus", "tmetro", "hvac", "fan"):
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
            provenance[speaker].append(
                dict(
                    path=str(path),
                    sha256=entry["sha256"],
                    samples=len(audio),
                    rate=actual,
                    language=entry.get("language", "en"),
                )
            )
        else:
            noise[entry["scene"].lower()] = audio[:, 0]
    aligned_speech = {}
    for k, v in speech.items():
        cat = np.concatenate(v)
        aligned_cat, off = align_onset_phase(cat, rate)
        aligned_speech[k] = aligned_cat
        if provenance[k]:
            provenance[k][0]["onset_phase_offset_samples"] = off
    result = (aligned_speech, noise, provenance)
    POOLS[key] = result
    return result


def matrix(manifest):
    entries = json.loads(manifest.read_text())["sources"]
    cases = []
    for rate in (16000, 24000, 48000):
        en_names = sorted(
            {
                str(r["speaker"])
                for r in entries
                if r["sample_rate"] == rate and r["kind"] == "clean_speech" and r.get("language", "en") == "en"
            }
        )
        zh_names = sorted(
            {
                str(r["speaker"])
                for r in entries
                if r["sample_rate"] == rate and r["kind"] == "clean_speech" and r.get("language") == "zh"
            }
        )
        if len(en_names) < 2:
            raise ValueError(f"English speaker gap at {rate}")
        # Pair 0: English near/far pair; Pair 1: Mandarin Chinese near/far pair (if >=2 zh speakers)
        pairs = [
            (en_names[0], en_names[1], "en"),
            (zh_names[0], zh_names[1], "zh") if len(zh_names) >= 2 else (en_names[1], en_names[0], "en"),
        ]
        for pair, (near, far, lang) in enumerate(pairs):
            for microphones in (1, 2, 4):
                base = dict(
                    sample_rate=rate,
                    pair=pair,
                    language=lang,
                    near_speaker=near,
                    far_speaker=far,
                    near_dbfs=-26,
                    mics=microphones,
                    noise="none",
                    snr_db=None,
                    ser_db=-5,
                    rt60_s=0.15,
                    echo_delay_ms=30,
                    stress="none",
                    profile="native-rate-extension-v2",
                    pcm_subtype="FLOAT",
                    room_dimensions_m=[3.6, 1.8, 1.5],
                    microphone_positions_m=[
                        [1.4 + (ch - (microphones - 1) / 2) * 0.035, 0.9, 1.3] for ch in range(microphones)
                    ],
                    near_position_m=[0.8, 0.65, 1.1],
                    render_positions_m=[[0.2, 0.3, 0.8], [0.2, 1.5, 0.8]],
                )
                noises = ("white", "pink", "hvac", "fan") if rate == 24000 else ("white", "pink", "tcar", "straffic")
                for noise in noises:
                    for snr in range(-10, 31, 5):
                        cases.append(
                            dict(
                                base,
                                family="ns",
                                noise=noise,
                                snr_db=snr,
                                near_windows=[[0, 10], [12, 22]],
                                field_label="cold_start",
                            )
                        )
                for rt60 in (0.15, 0.45):
                    for delay in (20, 80, 180):
                        cases.append(dict(base, family="aec", rt60_s=rt60, echo_delay_ms=delay))
                for noise in noises[-2:]:
                    for snr in (0, 15, 30):
                        cases.append(dict(base, family="combined", noise=noise, snr_db=snr))
                for stress in ("nonlinear_speaker", "clock_300ppm", "wrong_reference"):
                    cases.append(dict(base, family="stress", noise=noises[-1], snr_db=15, rt60_s=0.3, stress=stress))
                for level in (-45, -30, -15):
                    cases.append(dict(base, family="agc", near_dbfs=level, noise=noises[-1], snr_db=20))
                cases.append(
                    dict(
                        base,
                        family="ns",
                        noise=noises[-1],
                        snr_db=20,
                        noise_steps=[dict(time_s=11, gain_db=9)],
                        field_label="noise_step",
                    )
                )
    for index, case in enumerate(cases):
        case["id"] = f"native-{case['sample_rate']}-m{case['mics']}-{index:04d}-{case['family']}"
        case["seed"] = 2026101700 + index
        case["noise_origin"] = (
            "published-rate real recording"
            if case["noise"] in ("tcar", "straffic", "tbus", "tmetro", "hvac", "fan")
            else "synthetic at the requested sample rate"
            if case["noise"] != "none"
            else "none"
        )
    return cases


def invoke(command, prefix, cli_sha):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    process = subprocess.run(command, capture_output=True, text=True, timeout=150)
    prefix.with_suffix(".log").write_text(process.stdout + process.stderr)
    save(
        prefix.parent / "invocation.json",
        dict(
            argv=[str(x) for x in command],
            cli_sha256=cli_sha,
            exit_code=process.returncode,
            wall_seconds=time.time() - start,
        ),
    )
    if process.returncode:
        raise RuntimeError(f"CLI exit {process.returncode}: {prefix}; {process.stderr[-1000:]}")


def transport(playback, scene):
    if scene["stress"] == "clock_300ppm":
        return np.interp(
            np.arange(len(playback)) * 1.0003,
            np.arange(len(playback)),
            playback,
            left=0,
            right=0,
        )
    if scene["stress"] == "nonlinear_speaker":
        common = scene["common_gain"]
        return np.tanh(playback / common * 6) / 6 * common
    return playback


def prepare(case, root, manifest, cli, cli_sha):
    rate = case["sample_rate"]
    ev.RATE = rate
    speakers, noises, provenance = source_pool(manifest, rate)
    folder = root / "cases" / case["id"]
    scene = ev.make_case(case, speakers, noises, folder, RIRS.setdefault(rate, {}))
    scene["language"] = case.get("language", "en")
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
    near_mask = ev.mask_windows(scene["seconds"], scene["near_schedule"])
    far_mask = ev.mask_windows(scene["seconds"], scene["far_schedule"])
    clean_channels = ev.read(folder / "clean-mics.wav")
    noise = ev.read(folder / "noise-component.wav")
    echo = ev.read(folder / "echo-component.wav")
    if acoustic and case["stress"] != "wrong_reference":
        oracle = folder / "oracle" / "rx"
        raw_rx = folder / "rx-input.wav"
        shutil.copy2(folder / "playback.wav", raw_rx)
        invoke([cli, "--profile", "vcp", "--rx-in", str(raw_rx), "--out", str(oracle)], oracle, cli_sha)
        actual = ev.read(oracle.with_suffix(".rx-spk.wav"))[:, 0]
        h = np.load(folder / "echo-rir.npy")
        delay = round(scene["echo_delay_ms"] * rate / 1000)
        rendered = transport(actual, scene)
        echo = np.column_stack(
            [ev.shift(signal.fftconvolve(rendered, h[:, ch])[: len(actual)], delay) for ch in range(scene["mics"])]
        )
        echo_power = ev.power(echo[far_mask, 0])
        if echo_power <= 1e-15:
            raise ValueError("echo calibration is unexcited")
        scale = np.sqrt(ev.power(clean_channels[near_mask, 0]) / (10 ** (scene["ser_db"] / 10) * echo_power))
        echo *= scale
        scene["echo_rir_normalization"] = scale
        scene["reference_mode"] = "duplex_actual_post_rx_playback"
        scene["rx_input"] = str(raw_rx)
        scene["actual_playback_sha256"] = ev.digest(oracle.with_suffix(".rx-spk.wav"))
        shutil.copy2(oracle.with_suffix(".rx-spk.wav"), folder / "playback.wav")
        shutil.copy2(oracle.with_suffix(".rx-ref.wav"), folder / "ref.wav")
    else:
        scene["reference_mode"] = "deliberately_wrong_provided_reference" if acoustic else "no_playback"
    calibration_mask = near_mask.copy()
    if scene.get("noise_steps"):
        first_step = min(e["time_s"] for e in scene["noise_steps"])
        calibration_mask[round(first_step * rate) :] = False
    if scene["noise"] != "none":
        gain = np.sqrt(
            ev.power(clean_channels[calibration_mask, 0])
            / (10 ** (scene["snr_db"] / 10) * ev.power(noise[calibration_mask, 0]))
        )
        noise *= gain
        scene["noise_normalization_gain"] = float(gain)
    before = clean_channels + echo + noise
    peak = max(float(np.max(np.abs(x))) for x in (before, clean_channels, echo, noise))
    capture_gain = min(1.0, 0.9 / max(peak, ev.EPS))
    for name, value in (
        ("before.wav", before),
        ("clean-mics.wav", clean_channels),
        ("echo-component.wav", echo),
        ("noise-component.wav", noise),
    ):
        ev.write(folder / name, value * capture_gain, "FLOAT")
    for name in ("clean.wav", "clean-beam.wav"):
        ev.write(folder / name, ev.read(folder / name) * capture_gain, "FLOAT")
    scene["capture_headroom_gain"] = float(capture_gain)
    scene["capture_headroom_scope"] = (
        "common microphone sensitivity applied to all components/targets; playback and reference unchanged"
    )
    scene["effective_near_dbfs"] = ev.db(ev.power(clean_channels[near_mask, 0]) * capture_gain**2)
    scene["measured_input_snr_db"] = (
        ev.db(ev.power(clean_channels[near_mask, 0]) / ev.power(noise[near_mask, 0]))
        if scene["noise"] != "none"
        else None
    )
    scene["calibration_snr_db"] = (
        ev.db(ev.power(clean_channels[calibration_mask, 0]) / ev.power(noise[calibration_mask, 0]))
        if scene["noise"] != "none"
        else None
    )
    scene["measured_ser_db"] = (
        ev.db(ev.power(clean_channels[near_mask, 0]) / ev.power(echo[far_mask, 0])) if acoustic else None
    )
    if scene["noise"] != "none" and abs(scene["calibration_snr_db"] - scene["snr_db"]) > 1e-5:
        raise ValueError("generated SNR differs from declared grid")
    if acoustic and abs(scene["measured_ser_db"] - scene["ser_db"]) > 1e-5:
        raise ValueError("post-RX SER differs from declared scene")
    if scene.get("field_label") == "cold_start":
        scene["windows"]["near_start"] = [0.05, 0.5]
    if scene.get("field_label") == "noise_step":
        scene["windows"]["noise_only_before_step"] = [10, 11]
        scene["windows"]["noise_only_after_step"] = [11, 12]
    scene["files"] = {
        p.name: dict(sha256=ev.digest(p), bytes=p.stat().st_size)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "scene.json"
    }
    save(folder / "scene.json", scene)
    return scene


def command_for(scene, folder, prefix, cli, mode, paced=False):
    command = [str(cli), "--profile", "vcp", "--mic", str(folder / "before.wav"), "--out", str(prefix)]
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
    aligned = output[delay : delay + len(before)]
    segments = {}
    linear_16k = conditioned_16k = None
    if engine["aec_enabled"] and engine["linear_aec_available"]:
        lin_raw, lin_sr = sf.read(prefix.with_suffix(".linear-aec.wav"), dtype="float64", always_2d=True)
        if lin_sr != 16000:
            raise ValueError(f"unexpected linear-aec tap rate: {lin_sr}")
        ld = engine["linear_aec_delay_samples"]
        n_16k = round(scene["seconds"] * 16000)
        linear_16k = lin_raw[ld : ld + n_16k, 0]
        cond_raw, cond_sr = sf.read(prefix.with_suffix(".aec-input.wav"), dtype="float64", always_2d=True)
        cd = engine["aec_input_delay_samples"]
        cond_aligned = cond_raw[cd : cd + len(before), 0]
        if cond_sr != 16000:
            g = math.gcd(16000, cond_sr)
            conditioned_16k = signal.resample_poly(cond_aligned, 16000 // g, cond_sr // g)[:n_16k]
        else:
            conditioned_16k = cond_aligned[:n_16k]
    for name, limits in scene["windows"].items():
        b, a, clean = (ev.window(x, limits, rate=rate) for x in (before, aligned, target))
        if name.startswith("near") or name == "double_talk":
            qb, qa = ev.quality(clean, b), ev.quality(clean, a)
            segments[name] = dict(
                before=qb,
                after=qa,
                sisdr_improvement_db=(
                    qa["sisdr_db"] - qb["sisdr_db"]
                    if qa["sisdr_db"] is not None and qb["sisdr_db"] is not None
                    else None
                ),
                stoi_improvement=(
                    qa["stoi"] - qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None
                ),
            )
        else:
            segments[name] = ev.attenuation_measurement(b, a, pcm16=True)
            if name.startswith("far"):
                erle = None
                if linear_16k is not None and conditioned_16k is not None and scene["noise"] == "none":
                    lin_w = ev.window(linear_16k, limits, rate=16000)
                    cond_w = ev.window(conditioned_16k, limits, rate=16000)
                    if ev.power(lin_w) > 1e-12 and ev.power(cond_w) > 1e-12:
                        erle = ev.db(ev.power(cond_w)) - ev.db(ev.power(lin_w))
                segments[name]["linear_erle_engineering_db"] = erle
                segments[name]["linear_erle_scope"] = (
                    "same-rate 0-8 kHz conditioned input and linear residual"
                    if rate == 16000
                    else "common 0-8 kHz band (conditioned input anti-aliased to 16 kHz to match linear AEC tap)"
                )
    with prefix.with_suffix(".frames.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    timing, overruns = ev.frame_timing(frames, round(scene["seconds"] * 100))
    for key, value in timing.items():
        if not np.isclose(engine[key], value, rtol=2e-5, atol=1e-8):
            raise ValueError(f"timing report/CSV mismatch: {key}")
        engine[key] = value
    row = dict(
        id=scene["id"],
        family=scene["family"],
        sample_rate=rate,
        language=scene.get("language", "en"),
        microphones=scene["mics"],
        mode=mode,
        engine=engine,
        segments=segments,
        process_frames_over_10ms=overruns,
        output_sha256=ev.digest(prefix.with_suffix(".after.wav")),
        output_levels=dict(
            peak_dbfs=ev.db(float(np.max(np.abs(output))) ** 2),
            clipping_percent=float(np.mean(np.abs(output) >= 0.999) * 100),
        ),
        alignment="only declared causal DSP delay removed; no per-case lag optimization",
        stoi_scope="native-rate PCM passed to standard STOI; onset phase aligned so near_start has >=32 active frames",
        timing_scope="parallel offline callback timing; paced verification reported separately",
    )
    gates = acceptance.check_scene(scene, row, json.loads(acceptance.RULES.read_text()))
    for name in ("noise_only_before_step", "noise_only_after_step"):
        if name in segments:
            value = segments[name]["total_attenuation_db"]
            lower = segments[name]["attenuation_lower_bound_db"]
            if value is None:
                value = lower
            gates["gates"].append(
                dict(
                    segment=name,
                    metric="total_attenuation_db",
                    value=value,
                    operator=">=",
                    threshold=6.0,
                    status="unavailable" if value is None else "pass" if value >= 6 else "fail",
                    reason="same frozen noise-only threshold applied to noise transition",
                    optional=False,
                )
            )
    gates["counts"] = dict(Counter(g["status"] for g in gates["gates"]))
    gates["passed"] = all(g["status"] == "pass" or (g["status"] == "unavailable" and g["optional"]) for g in gates["gates"])
    gates["status"] = (
        "pass"
        if gates["passed"]
        else "fail"
        if any(g["status"] == "fail" for g in gates["gates"])
        else "incomplete_required_measurement"
    )
    save(prefix.parent / "metrics.json", row)
    save(prefix.parent / "acceptance.json", gates)
    return dict(scene=scene, metrics=row, acceptance=gates)


def run_one(arguments):
    case, root, manifest, cli, cli_sha = arguments
    try:
        scene = prepare(case, root, manifest, str(cli), cli_sha)
        prefix = root / "cases" / scene["id"] / "runs/full/result"
        invoke(command_for(scene, prefix.parents[2], prefix, str(cli), "full"), prefix, cli_sha)
        return measure(scene, root, "full")
    except Exception:
        return dict(id=case["id"], sample_rate=case["sample_rate"], error=traceback.format_exc())


def summary(rows):
    statuses = Counter()
    rates = defaultdict(Counter)
    langs = defaultdict(Counter)
    failures = Counter()
    for row in rows:
        if "error" in row:
            statuses["execution_error"] += 1
            rates[row["sample_rate"]]["execution_error"] += 1
            continue
        status = row["acceptance"]["status"]
        statuses[status] += 1
        rates[row["scene"]["sample_rate"]][status] += 1
        langs[row["scene"].get("language", "en")][status] += 1
        for gate in row["acceptance"]["gates"]:
            if gate["status"] != "pass" and not gate.get("optional"):
                failures[(gate["status"], gate["segment"], gate["metric"])] += 1
    return dict(
        status_counts=dict(statuses),
        by_rate={str(k): dict(v) for k, v in rates.items()},
        by_language={str(k): dict(v) for k, v in langs.items()},
        nonpassing_gates=[
            dict(status=k[0], segment=k[1], metric=k[2], count=v) for k, v in failures.most_common()
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--paced", action="store_true")
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
    identity = dict(
        cli_sha256=cli_sha,
        source_manifest=str(manifest),
        sample_rate_conversion=False,
        profile="vcp",
        rules_sha256=RULES_SHA,
        jobs=args.jobs,
        paced=args.paced,
        dependencies={str(p.resolve()): ev.digest(p) for p in dependencies},
    )
    save(root / "identity.json", identity)
    cases = matrix(manifest)
    if args.pilot:
        cases = [
            next(c for c in cases if c["sample_rate"] == r and c["mics"] == m and c["family"] == "combined")
            for r in (16000, 24000, 48000)
            for m in (1, 2, 4)
        ]
    save(
        root / "plan.json",
        dict(
            scenes=cases,
            count=len(cases),
            scope="native published-rate multilingual (EN+ZH) extension with onset alignment and common-band ERLE",
            source_gaps=json.loads(manifest.read_text())["unresolved_gaps"],
        ),
    )
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, (c, root, manifest, artifact, cli_sha)): c for c in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.append(result)
            status = "error" if "error" in result else result["acceptance"]["status"]
            print(
                json.dumps(
                    dict(
                        completed=index,
                        total=len(cases),
                        status=status,
                        id=result.get("id") or result["scene"]["id"],
                        elapsed_s=time.time() - started,
                    )
                ),
                flush=True,
            )
    rows.sort(key=lambda r: r.get("id") or r["scene"]["id"])
    with (root / "results.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    controls = []
    paced_checks = []
    for rate in (16000, 24000, 48000):
        for mics in (1, 2, 4):
            for family, mode, segment in (("ns", "negative_no_ns", "noise_only"), ("aec", "negative_no_aec", "far_only")):
                selected = next(
                    (
                        r
                        for r in rows
                        if "scene" in r
                        and r["scene"]["sample_rate"] == rate
                        and r["scene"]["mics"] == mics
                        and r["scene"]["family"] == family
                    ),
                    None,
                )
                if selected is None:
                    continue
                scene = selected["scene"]
                folder = root / "cases" / scene["id"]
                prefix = folder / "runs" / mode / "result"
                invoke(command_for(scene, folder, prefix, artifact, mode), prefix, cli_sha)
                res = measure(scene, root, mode)
                gate = next(
                    g
                    for g in res["acceptance"]["gates"]
                    if g["segment"] == segment and g["metric"] == "total_attenuation_db"
                )
                controls.append(
                    dict(
                        id=scene["id"],
                        mode=mode,
                        sample_rate=rate,
                        microphones=mics,
                        attenuation_db=gate["value"],
                        threshold=gate["threshold"],
                        verified=gate["status"] == "fail",
                    )
                )
    save(root / "controls.json", controls)
    if args.paced:
        for rate in (16000, 24000, 48000):
            for mics in (1, 2, 4):
                row = next(
                    r
                    for r in rows
                    if "scene" in r
                    and r["scene"]["sample_rate"] == rate
                    and r["scene"]["mics"] == mics
                    and r["scene"]["family"] == "combined"
                )
                scene = row["scene"]
                folder = root / "cases" / scene["id"]
                prefix = folder / "runs/paced/result"
                invoke(command_for(scene, folder, prefix, artifact, "full", paced=True), prefix, cli_sha)
                measured = measure(scene, root, "paced")
                paced_checks.append(
                    dict(
                        id=scene["id"],
                        sample_rate=rate,
                        microphones=mics,
                        output_matches_offline=measured["metrics"]["output_sha256"] == row["metrics"]["output_sha256"],
                        overruns=measured["metrics"]["process_frames_over_10ms"],
                        completion_deadline_misses=measured["metrics"]["engine"]["completion_deadline_misses"],
                        max_release_lateness_ms=measured["metrics"]["engine"]["max_release_lateness_ms"],
                        process_p95_ms=measured["metrics"]["engine"]["process_p95_ms"],
                        process_max_ms=measured["metrics"]["engine"]["process_max_ms"],
                    )
                )
        save(root / "paced-results.json", paced_checks)
    report = dict(
        schema_version=2,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.time() - started,
        identity=identity,
        summary=summary(rows),
        negative_controls=controls,
        paced_checks=paced_checks,
    )
    save(root / "report.json", report)


if __name__ == "__main__":
    main()
