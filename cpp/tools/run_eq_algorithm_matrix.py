#!/usr/bin/env python3
"""Paired EQ algorithm matrix on frozen speech and a known-response fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import freqz
from pystoi import stoi

import run_audio_eval as ev


PROJECT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("controlled-native-full-20261017", "itu-controlled-quality-v1-20261017"),
    ("controlled-cavemove-20261017", "itu-cavemove-quality-v1-20261017"),
)

VARIANTS = {
    "bypass": [],
    "low_shelf": ["--eq-band", "low_shelf,160,-3,0.7071"],
    "parametric_peak": [
        "--eq-preamp-db", "-3",
        "--eq-band", "peak,250,3,1",
        "--eq-band", "peak,1000,-3,1",
    ],
    "three_band": [
        "--eq-preamp-db", "-3",
        "--eq-band", "low_shelf,180,2,0.7071",
        "--eq-band", "peak,1000,-2,1",
        "--eq-band", "high_shelf,3500,1,0.7071",
    ],
    "calibrated_inverse": [
        "--eq-preamp-db", "-0.049",
        "--eq-band", "peak,120,0.006,1",
        "--eq-band", "peak,250,-5.985,1",
        "--eq-band", "peak,500,0.003,1",
        "--eq-band", "peak,1000,0.009,1",
        "--eq-band", "peak,2000,0.009,1",
        "--eq-band", "peak,4000,0.01,1",
        "--eq-band", "peak,6000,0.012,1",
    ],
    "linear_phase_three_band": [
        "--eq-phase", "linear",
        "--eq-preamp-db", "-3",
        "--eq-band", "low_shelf,180,2,0.7071",
        "--eq-band", "peak,1000,-2,1",
        "--eq-band", "high_shelf,3500,1,0.7071",
    ],
}

ENGINEERING_GATES = {
    "near_late_sisdr_delta_min_db": -0.1,
    "near_late_stoi_delta_min": -0.02,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response(frequencies: np.ndarray, flags: list[str], rate: int) -> np.ndarray:
    result = np.zeros_like(frequencies, dtype=np.float64)
    preamp = 0.0
    for index, flag in enumerate(flags):
        if flag == "--eq-preamp-db":
            preamp += float(flags[index + 1])
        if flag != "--eq-band":
            continue
        kind, hz, gain, q = flags[index + 1].split(",")
        hz, gain, q = float(hz), float(gain), float(q)
        a = 10 ** (gain / 40)
        w = 2 * np.pi * hz / rate
        cs, sn = np.cos(w), np.sin(w)
        alpha = sn / (2 * q)
        shelf = sn / math.sqrt(2)
        if kind == "peak":
            b = [1 + alpha * a, -2 * cs, 1 - alpha * a]
            aa = [1 + alpha / a, -2 * cs, 1 - alpha / a]
        elif kind == "low_shelf":
            t = 2 * math.sqrt(a) * shelf
            b = [a * ((a + 1) - (a - 1) * cs + t),
                 2 * a * ((a - 1) - (a + 1) * cs),
                 a * ((a + 1) - (a - 1) * cs - t)]
            aa = [(a + 1) + (a - 1) * cs + t,
                  -2 * ((a - 1) + (a + 1) * cs),
                  (a + 1) + (a - 1) * cs - t]
        else:
            t = 2 * math.sqrt(a) * shelf
            b = [a * ((a + 1) + (a - 1) * cs + t),
                 -2 * a * ((a - 1) + (a + 1) * cs),
                 a * ((a + 1) + (a - 1) * cs - t)]
            aa = [(a + 1) - (a - 1) * cs + t,
                  2 * ((a - 1) - (a + 1) * cs),
                  (a + 1) - (a - 1) * cs - t]
        _, h = freqz(b, aa, worN=2 * np.pi * frequencies / rate)
        result += 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return result + preamp


def linear_coefficients(flags: list[str], rate: int) -> np.ndarray:
    """Mirror the engine's symmetric LS design for the response diagnostic."""
    bands = []
    preamp = 0.0
    taps = 0
    for index, flag in enumerate(flags):
        if flag == "--eq-preamp-db":
            preamp = float(flags[index + 1])
        elif flag == "--eq-linear-taps":
            taps = int(flags[index + 1])
        elif flag == "--eq-band":
            kind, hz, gain, q = flags[index + 1].split(",")
            bands.append((kind, float(hz), float(gain), float(q)))
    if not taps:
        taps = ((rate + 49) // 50) | 1
    half = taps // 2
    bins = min(1024, rate // 16)
    normal = np.zeros((half + 1, half + 1), dtype=np.float64)
    rhs = np.zeros(half + 1, dtype=np.float64)
    for k in range(bins):
        hz = .5 + (rate * .49 - .5) * k / max(1, bins - 1)
        target = 10 ** (preamp / 20)
        for kind, center, gain, q in bands:
            a = 10 ** (gain / 40)
            w = 2 * np.pi * center / rate
            cs, sn = np.cos(w), np.sin(w)
            alpha = sn / (2 * q)
            shelf = sn / np.sqrt(2)
            if kind == "peak":
                b = [1 + alpha * a, -2 * cs, 1 - alpha * a]
                aa = [1 + alpha / a, -2 * cs, 1 - alpha / a]
            elif kind == "low_shelf":
                t = 2 * np.sqrt(a) * shelf
                b = [a * ((a + 1) - (a - 1) * cs + t),
                     2 * a * ((a - 1) - (a + 1) * cs),
                     a * ((a + 1) - (a - 1) * cs - t)]
                aa = [(a + 1) + (a - 1) * cs + t,
                      -2 * ((a - 1) + (a + 1) * cs),
                      (a + 1) + (a - 1) * cs - t]
            else:
                t = 2 * np.sqrt(a) * shelf
                b = [a * ((a + 1) + (a - 1) * cs + t),
                     -2 * a * ((a - 1) + (a + 1) * cs),
                     a * ((a + 1) - (a - 1) * cs - t)]
                aa = [(a + 1) - (a - 1) * cs + t,
                      2 * ((a - 1) - (a + 1) * cs),
                      (a + 1) - (a - 1) * cs - t]
            z = np.exp(-1j * w)
            target *= (b[0] + b[1] * z + b[2] * z * z) / (
                aa[0] + aa[1] * z + aa[2] * z * z)
        row = np.r_[1., 2 * np.cos(2 * np.pi * hz / rate *
                                   np.arange(1, half + 1))]
        normal += np.outer(row, row)
        rhs += row * target.real
    normal[np.diag_indices_from(normal)] += 1e-10 * bins
    solution = np.linalg.solve(normal, rhs)
    coefficients = np.zeros(taps)
    for i, value in enumerate(solution):
        coefficients[half - i] = coefficients[half + i] = value
    return coefficients


def select_cases():
    selected = []
    for run, manifest_name in SOURCES:
        manifest = json.loads((PROJECT / "datasets" / manifest_name / "manifest.json").read_text())
        core = {scene["id"] for scene in manifest["scenes"] if scene["track"] == "spectrum_screened_core"}
        for line in (PROJECT / "build" / run / "results.jsonl").open():
            row = json.loads(line)
            scene = row["scene"]
            if scene["family"] == "agc" and scene["id"] in core:
                selected.append((run, row))
    if len(selected) != 108 or len({(run, row["scene"]["id"]) for run, row in selected}) != len(selected):
        raise ValueError("EQ speech matrix requires 108 unique run-scoped AGC cases")
    return selected


def process_speech(task):
    root, run, original, cli, variant, flags = task
    scene = original["scene"]
    source = PROJECT / "build" / run / "cases" / scene["id"]
    case_key = f"{run}__{scene['id']}__{scene.get('noise', 'none')}"
    folder = root / variant / "cases" / case_key
    folder.mkdir(parents=True)
    for name, record in scene["files"].items():
        if digest(source / name) != record["sha256"]:
            raise ValueError(f"frozen input changed: {source / name}")
        (folder / name).symlink_to(source / name)
    prefix = folder / "runs/full/result"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli), "--profile", "vcp",
        "--mic", str(folder / "before.wav"),
        "--out", str(prefix),
        "--no-aec",
        "--beam-delays", ",".join(f"{value:.9f}" for value in scene["beam_delays_samples"]),
        "--repair-stage", "4",
        "--stage4-noise-release",
        "--no-tx-drc",
    ] + flags
    run_result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (prefix.parent / "invocation.json").write_text(json.dumps(dict(
        argv=command, cli_sha256=digest(cli), exit_code=run_result.returncode,
        stdout=run_result.stdout, stderr=run_result.stderr), indent=2) + "\n")
    run_result.check_returncode()
    row = measure_multirate(scene, prefix, folder)
    return dict(
        variant=variant, original_run=run, scene_id=scene["id"],
        noise=scene.get("noise"), sample_rate=scene["sample_rate"],
        near_dbfs=scene["near_dbfs"], measurements=row,
        eq=dict(preamp_db=row["engine"].get("output_eq_preamp_db"),
                bands=row["engine"].get("output_eq_bands")),
    )


def read_rate(path: Path, expected: int) -> np.ndarray:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != expected:
        raise ValueError(f"expected {expected} Hz, got {rate}: {path}")
    return data[:, 0]


def measure_multirate(scene, prefix: Path, folder: Path):
    rate = int(scene["sample_rate"])
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = read_rate(folder / "before.wav", rate)
    clean = read_rate(folder / "clean.wav", rate)
    output = read_rate(prefix.with_suffix(".after.wav"), rate)
    delay = int(engine["dsp_delay_samples"])
    if len(output) != len(before) + delay:
        raise ValueError(f"EQ output delay mismatch: {scene['id']}")
    after = output[delay:delay + len(before)]

    def q(target, observed):
        target = target - np.mean(target)
        observed = observed - np.mean(observed)
        alpha = float(np.dot(target, observed) / max(np.dot(target, target), 1e-20))
        projected = alpha * target
        error = observed - projected
        pproj, perr = ev.power(projected), ev.power(error)
        if pproj < 1e-15:
            return dict(sisdr_db=None, stoi=None, projection_gain_db=None)
        return dict(
            sisdr_db=min(120., ev.db(pproj / max(perr, 1e-20))),
            stoi=float(stoi(target, observed, rate, extended=False)),
            projection_gain_db=ev.db(alpha ** 2),
            correlation=float(np.corrcoef(target, observed)[0, 1]),
            rms_dbfs=ev.db(ev.power(observed)),
            peak_dbfs=ev.db(float(np.max(np.abs(observed))) ** 2),
            clipping_percent=float(np.mean(np.abs(observed) >= .999) * 100),
        )

    segments = {}
    for name in ("near_only", "near_late"):
        limits = scene["windows"][name]
        a = int(round(limits[0] * rate))
        b = int(round(limits[1] * rate))
        before_q = q(clean[a:b], before[a:b])
        after_q = q(clean[a:b], after[a:b])
        segments[name] = dict(
            before=before_q, after=after_q,
            sisdr_improvement_db=after_q["sisdr_db"] - before_q["sisdr_db"],
            stoi_improvement=after_q["stoi"] - before_q["stoi"],
        )
    limits = scene["windows"]["noise_only"]
    a = int(round(limits[0] * rate))
    b = int(round(limits[1] * rate))
    segments["noise_only"] = ev.attenuation_measurement(before[a:b], after[a:b])
    return dict(
        id=scene["id"], family=scene["family"], mode="full",
        sample_rate=rate, segments=segments, engine=engine,
        output_sha256=digest(prefix.with_suffix(".after.wav")),
        output_levels=dict(
            peak_dbfs=ev.db(float(np.max(np.abs(output))) ** 2),
            clipping_percent=float(np.mean(np.abs(output) >= .999) * 100),
        ),
    )


def target_response(root: Path, cli: Path):
    frequencies = np.unique(np.r_[np.geomspace(40, 7000, 120), [160, 250, 1000, 3500]])
    target = np.zeros_like(frequencies)
    target += 4 * np.exp(-0.5 * (np.log2(frequencies / 250) / 0.32) ** 2)
    target -= 3 * np.exp(-0.5 * (np.log2(frequencies / 1200) / 0.45) ** 2)
    rows = []
    for name, flags in VARIANTS.items():
        if "--eq-phase" in flags and flags[flags.index("--eq-phase") + 1] == "linear":
            coefficients = linear_coefficients(flags, 48000)
            predicted = 20 * np.log10(np.maximum(
                np.abs(np.exp(-2j * np.pi * frequencies[:, None] / 48000 *
                             np.arange(len(coefficients))[None, :]) @ coefficients),
                1e-12))
        else:
            predicted = response(frequencies, flags, 48000)
        rows.append(dict(
            variant=name,
            phase="linear" if "--eq-phase" in flags else "minimum",
            taps=len(linear_coefficients(flags, 48000))
            if "--eq-phase" in flags else 0,
            delay_samples=(len(linear_coefficients(flags, 48000)) - 1) // 2
            if "--eq-phase" in flags else 0,
            rms_error_db=float(np.sqrt(np.mean((predicted - target) ** 2))),
            max_error_db=float(np.max(np.abs(predicted - target))),
            max_boost_db=float(max(0, np.max(predicted))),
            max_cut_db=float(max(0, -np.min(predicted))),
            preamp_db=float(next((float(flags[i + 1]) for i, item in enumerate(flags)
                                  if item == "--eq-preamp-db"), 0)),
            band_count=sum(item == "--eq-band" for item in flags),
        ))
    source = root / "response-fixture"
    source.mkdir(parents=True)
    excitation = np.random.default_rng(20261021).normal(0, .02, 48000 * 4)
    sf.write(source / "excitation.wav", excitation, 48000, subtype="FLOAT")
    for name, flags in VARIANTS.items():
        prefix = source / name
        command = [str(cli), "--mic", str(source / "excitation.wav"), "--out", str(prefix),
                   "--no-aec", "--no-ns", "--no-agc", "--no-limiter", "--no-low-cut"] + flags
        run_result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        run_result.check_returncode()
        (prefix.with_suffix(".invocation.json")).write_text(json.dumps(dict(
            argv=command, exit_code=run_result.returncode, cli_sha256=digest(cli),
            stderr=run_result.stderr), indent=2) + "\n")
        data, rate = sf.read(prefix.with_suffix(".after.wav"), dtype="float64")
        if rate != 48000:
            raise ValueError("EQ response fixture rate changed")
        # The analytic response is a target diagnostic; the actual streamed
        # PCM and its declared delay are retained as separate evidence.
        row = next(item for item in rows if item["variant"] == name)
        row.update(audio_sha256=digest(prefix.with_suffix(".after.wav")),
                   audio_samples=len(data))
    return rows


def summarize(results):
    baseline = {
        (row["original_run"], row["scene_id"], row["noise"]): row
        for row in results if row["variant"] == "bypass"
    }
    variants = {}
    for name in VARIANTS:
        rows = [row for row in results if row["variant"] == name]
        if len(rows) != 108:
            raise ValueError(f"incomplete EQ variant: {name}")
        paired = []
        for row in rows:
            key = (row["original_run"], row["scene_id"], row["noise"])
            base = baseline[key]
            current = row["measurements"]
            reference = base["measurements"]
            paired_row = dict(
                original_run=row["original_run"],
                scene_id=row["scene_id"],
                noise=row["noise"],
                sample_rate=row["sample_rate"],
                near_dbfs=row["near_dbfs"],
                sisdr_delta=current["segments"]["near_late"]["after"]["sisdr_db"] -
                            reference["segments"]["near_late"]["after"]["sisdr_db"],
                stoi_delta=current["segments"]["near_late"]["after"]["stoi"] -
                           reference["segments"]["near_late"]["after"]["stoi"],
                output_changed=current["output_sha256"] != reference["output_sha256"],
            )
            paired_row["engineering_pass"] = (
                paired_row["sisdr_delta"]
                >= ENGINEERING_GATES["near_late_sisdr_delta_min_db"]
                and paired_row["stoi_delta"]
                >= ENGINEERING_GATES["near_late_stoi_delta_min"]
            )
            paired.append(paired_row)
        delta_sisdr = [row["sisdr_delta"] for row in paired]
        delta_stoi = [row["stoi_delta"] for row in paired]
        peaks = [row["measurements"]["output_levels"]["peak_dbfs"] for row in rows]
        clips = [row["measurements"]["output_levels"]["clipping_percent"] for row in rows]
        variants[name] = dict(
            cases=len(rows),
            mean_sisdr_delta_vs_bypass_db=float(np.mean(delta_sisdr)),
            median_sisdr_delta_vs_bypass_db=float(np.median(delta_sisdr)),
            mean_stoi_delta_vs_bypass=float(np.mean(delta_stoi)),
            median_stoi_delta_vs_bypass=float(np.median(delta_stoi)),
            stoi_regressions=sum(value < -0.02 for value in delta_stoi),
            sisdr_losses=sum(value < -0.1 for value in delta_sisdr),
            engineering_pass_cases=sum(row["engineering_pass"] for row in paired),
            engineering_fail_cases=sum(not row["engineering_pass"] for row in paired),
            outputs_changed_vs_bypass=sum(row["output_changed"] for row in paired),
            clipped_cases=sum(value > 0 for value in clips),
            worst_peak_dbfs=float(max(peaks)),
            bands=rows[0]["eq"]["bands"],
            preamp_db=rows[0]["eq"]["preamp_db"],
            paired_cases=paired,
        )
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/host/cpp/itu_audio_cli"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root, cli = args.root.resolve(), args.cli.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    shutil.copy2(cli, artifacts / "itu_audio_cli")
    cli = artifacts / "itu_audio_cli"
    selected = select_cases()
    tasks = [(root, run, row, cli, variant, flags)
             for variant, flags in VARIANTS.items() for run, row in selected]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (root / "results.jsonl").open("x") as stream:
        for future in as_completed([pool.submit(process_speech, task) for task in tasks]):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(results) % 108 == 0:
                print(json.dumps(dict(completed=len(results), total=len(tasks))), flush=True)
    speech = summarize(results)
    target = target_response(root, cli)
    report = dict(
        state="completed",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_cases=len(selected),
        scope="108 frozen AGC speech cases x six EQ forms; target-response diagnostic",
        cases=108,
        engineering_gates=ENGINEERING_GATES,
        engineering_gate_scope=(
            "EQ contribution relative to the paired bypass control on near_late; "
            "this is an EQ-specific diagnostic gate, not the full ITU acceptance gate."
        ),
        variants=speech,
        target_response=target,
        cli_sha256=digest(cli),
        results_sha256=digest(root / "results.jsonl"),
        elapsed_seconds=time.monotonic() - started,
        standard_compliance=False,
        limitation=(
            "The speech matrix is an engineering replay through the actual VCP chain. "
            "It is not an acoustic cabin calibration, ITU mask compliance, device test, "
            "or a claim that a filter shape is universally preferred."
        ),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(state="completed", cases=108, cli_sha256=report["cli_sha256"]), indent=2))


if __name__ == "__main__":
    main()
