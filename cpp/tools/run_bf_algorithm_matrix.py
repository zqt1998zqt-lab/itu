#!/usr/bin/env python3
"""Paired multi-microphone beamforming matrix on the frozen vehicle field set."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi

import run_audio_eval as ev
from evaluation_targets import calibrated_beam_target


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "datasets/itu-vehicle-fields-float-20261010"
MANIFEST = DATASET / "inputs-manifest.json"

VARIANTS = {
    "no_bf": [],
    "delay_sum": [],
    "mvdr_observed": ["--spatial-mvdr"],
    "mvdr_fixed": ["--spatial-mvdr", "--spatial-fixed-steering"],
    "mpdr_continuous": [
        "--spatial-mvdr",
        "--spatial-fixed-steering",
        "--spatial-continuous-covariance",
    ],
    "das_wind": ["--wind-control"],
    "mvdr_wind": ["--spatial-mvdr", "--wind-control"],
}

ENGINEERING_GATES = {
    "speech_window_sisdr_delta_min_db": -0.1,
    "speech_window_stoi_delta_min": -0.02,
    "noise_attenuation_gain_min_db": 0.0,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rate(path: Path, expected: int) -> np.ndarray:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != expected:
        raise ValueError(f"expected {expected} Hz, got {rate}: {path}")
    return data


def quality(target: np.ndarray, observed: np.ndarray, rate: int) -> dict:
    target = target - np.mean(target)
    observed = observed - np.mean(observed)
    alpha = float(np.dot(target, observed) / max(np.dot(target, target), 1e-20))
    projected = alpha * target
    error = observed - projected
    pproj, perr = ev.power(projected), ev.power(error)
    if pproj < 1e-15:
        return {
            "sisdr_db": None,
            "stoi": None,
            "projection_gain_db": None,
            "rms_dbfs": ev.db(ev.power(observed)),
            "peak_dbfs": ev.db(float(np.max(np.abs(observed))) ** 2),
            "clipping_percent": float(np.mean(np.abs(observed) >= 0.999) * 100),
        }
    return {
        "sisdr_db": min(120.0, ev.db(pproj / max(perr, 1e-20))),
        "stoi": float(stoi(target, observed, rate, extended=False)),
        "projection_gain_db": ev.db(alpha**2),
        "rms_dbfs": ev.db(ev.power(observed)),
        "peak_dbfs": ev.db(float(np.max(np.abs(observed))) ** 2),
        "clipping_percent": float(np.mean(np.abs(observed) >= 0.999) * 100),
    }


def process_scene(task):
    root, scene, cli, variant, flags = task
    source = DATASET / "cases" / scene["id"]
    case_key = scene["id"]
    folder = root / variant / "cases" / case_key
    folder.mkdir(parents=True)
    for name, record in scene["files"].items():
        if digest(source / name) != record["sha256"]:
            raise ValueError(f"frozen input changed: {source / name}")
        (folder / name).symlink_to(source / name)

    prefix = folder / "runs/full/result"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli),
        "--profile",
        "general",
        "--mic",
        str(folder / "before.wav"),
        "--out",
        str(prefix),
        "--no-aec",
        "--no-ns",
        "--no-agc",
        "--no-tx-drc",
        "--no-limiter",
        "--no-low-cut",
    ]
    if variant == "no_bf":
        command += ["--no-bf"]
    else:
        command += [
            "--beam-delays",
            ",".join(f"{value:.9f}" for value in scene["beam_delays_samples"]),
        ] + flags

    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (prefix.parent / "invocation.json").write_text(
        json.dumps(
            {
                "argv": command,
                "cli_sha256": digest(cli),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
        )
        + "\n"
    )
    completed.check_returncode()
    measured = measure_scene(scene, prefix, folder)
    return {
        "variant": variant,
        "scene_id": scene["id"],
        "mics": scene["mics"],
        "snr_db": scene["snr_db"],
        "spatial_noise_model": scene.get("spatial_noise_model"),
        "stress": scene.get("stress"),
        "measurements": measured,
    }


def measure_scene(scene, prefix: Path, folder: Path) -> dict:
    rate = int(scene["sample_rate"])
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = read_rate(folder / "before.wav", rate)
    clean_mics = read_rate(folder / "clean-mics.wav", rate)
    output = read_rate(prefix.with_suffix(".after.wav"), rate)[:, 0]
    delay = int(engine["dsp_delay_samples"])
    source_samples = len(before)
    if len(output) != source_samples + delay:
        raise ValueError(f"BF output delay mismatch: {scene['id']}")
    after = output[delay : delay + source_samples]

    if engine["beamforming_enabled"]:
        target, alignment = calibrated_beam_target(
            clean_mics,
            scene["beam_delays_samples"],
            rate,
            source_rate=16000,
            core_rate=rate,
        )
        target_kind = "calibrated_delay_sum_clean_beam"
    else:
        target = clean_mics[:, 0]
        alignment = {
            "method": "explicit channel 0; no BF delay",
            "core_rate": rate,
        }
        target_kind = "clean_channel_0"

    # Multirender scenes intentionally expose ``near_only`` but no
    # ``near_late`` because their near talker is continuous rather than the
    # two-interval schedule used by the other field families.
    speech_window = (
        "near_late" if "near_late" in scene["windows"] else "near_only"
    )
    segments = {}
    for name in ("near_only", "near_late"):
        if name not in scene["windows"]:
            continue
        limits = scene["windows"][name]
        a = int(round(limits[0] * rate))
        b = int(round(limits[1] * rate))
        before_q = quality(target[a:b], before[a:b, 0], rate)
        after_q = quality(target[a:b], after[a:b], rate)
        segments[name] = {
            "before": before_q,
            "after": after_q,
            "sisdr_improvement_db": after_q["sisdr_db"] - before_q["sisdr_db"],
            "stoi_improvement": after_q["stoi"] - before_q["stoi"],
        }

    limits = scene["windows"]["noise_only"]
    a = int(round(limits[0] * rate))
    b = int(round(limits[1] * rate))
    segments["noise_only"] = ev.attenuation_measurement(before[a:b, 0], after[a:b])
    return {
        "id": scene["id"],
        "sample_rate": rate,
        "segments": segments,
        "engine": engine,
        "target_kind": target_kind,
        "target_alignment": alignment,
        "speech_window": speech_window,
        "output_sha256": digest(prefix.with_suffix(".after.wav")),
        "output_levels": {
            "peak_dbfs": ev.db(float(np.max(np.abs(output))) ** 2),
            "clipping_percent": float(np.mean(np.abs(output) >= 0.999) * 100),
            "crest_db": ev.db(float(np.max(np.abs(output))) ** 2)
            - ev.db(ev.power(output)),
        },
    }


def summarize(results):
    baseline = {row["scene_id"]: row for row in results if row["variant"] == "no_bf"}
    variants = {}
    for name in VARIANTS:
        rows = [row for row in results if row["variant"] == name]
        if len(rows) != 72:
            raise ValueError(f"incomplete BF variant: {name}")
        paired = []
        for row in rows:
            reference = baseline[row["scene_id"]]
            current = row["measurements"]
            base = reference["measurements"]
            current_window = current["speech_window"]
            base_window = base["speech_window"]
            if current_window != base_window:
                raise ValueError(
                    f"paired BF speech-window mismatch for {row['scene_id']}: "
                    f"{base_window} != {current_window}"
                )
            sisdr_delta = (
                current["segments"][current_window]["after"]["sisdr_db"]
                - base["segments"][base_window]["after"]["sisdr_db"]
            )
            stoi_delta = (
                current["segments"][current_window]["after"]["stoi"]
                - base["segments"][base_window]["after"]["stoi"]
            )
            current_atten = current["segments"]["noise_only"]["total_attenuation_db"]
            base_atten = base["segments"]["noise_only"]["total_attenuation_db"]
            noise_gain = (
                current_atten - base_atten
                if current_atten is not None and base_atten is not None
                else None
            )
            pass_case = (
                sisdr_delta >= ENGINEERING_GATES["speech_window_sisdr_delta_min_db"]
                and stoi_delta >= ENGINEERING_GATES["speech_window_stoi_delta_min"]
                and noise_gain is not None
                and noise_gain >= ENGINEERING_GATES["noise_attenuation_gain_min_db"]
            )
            paired.append(
                {
                    "scene_id": row["scene_id"],
                    "mics": row["mics"],
                    "snr_db": row["snr_db"],
                    "spatial_noise_model": row["spatial_noise_model"],
                    "speech_window": current_window,
                    "sisdr_delta": sisdr_delta,
                    "stoi_delta": stoi_delta,
                    "noise_attenuation_gain_db": noise_gain,
                    "engineering_pass": pass_case,
                    "output_changed": (
                        current["output_sha256"] != base["output_sha256"]
                    ),
                }
            )

        mics_groups = {}
        for mics in (2, 4):
            subset = [row for row in paired if row["mics"] == mics]
            mics_groups[str(mics)] = {
                "pass": sum(row["engineering_pass"] for row in subset),
                "total": len(subset),
            }
        snr_groups = {}
        for snr in (0, 15, 30):
            subset = [row for row in paired if row["snr_db"] == snr]
            snr_groups[str(snr)] = {
                "pass": sum(row["engineering_pass"] for row in subset),
                "total": len(subset),
            }

        variants[name] = {
            "cases": len(rows),
            "engineering_pass_cases": sum(row["engineering_pass"] for row in paired),
            "engineering_fail_cases": sum(not row["engineering_pass"] for row in paired),
            "mean_sisdr_delta_vs_no_bf_db": float(
                np.mean([row["sisdr_delta"] for row in paired])
            ),
            "median_sisdr_delta_vs_no_bf_db": float(
                np.median([row["sisdr_delta"] for row in paired])
            ),
            "mean_stoi_delta_vs_no_bf": float(
                np.mean([row["stoi_delta"] for row in paired])
            ),
            "mean_noise_attenuation_gain_vs_no_bf_db": float(
                np.mean(
                    [
                        row["noise_attenuation_gain_db"]
                        for row in paired
                        if row["noise_attenuation_gain_db"] is not None
                    ]
                )
            ),
            "outputs_changed_vs_no_bf": sum(row["output_changed"] for row in paired),
            "clipped_cases": sum(
                row["measurements"]["output_levels"]["clipping_percent"] > 0
                for row in rows
            ),
            "by_mics": mics_groups,
            "by_snr": snr_groups,
            "paired_cases": paired,
        }
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/host/cpp/itu_audio_cli"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root, cli = args.root.resolve(), args.cli.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    shutil.copy2(cli, artifacts / "itu_audio_cli")
    cli = artifacts / "itu_audio_cli"

    manifest = json.loads(args.manifest.read_text())
    scenes = [scene for scene in manifest["scenes"] if scene["mics"] > 1]
    if len(scenes) != 72 or len({scene["id"] for scene in scenes}) != 72:
        raise ValueError("BF matrix requires 72 unique multi-microphone scenes")

    tasks = [
        (root, scene, cli, variant, flags)
        for variant, flags in VARIANTS.items()
        for scene in scenes
    ]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (
        root / "results.jsonl"
    ).open("x") as stream:
        for future in as_completed([pool.submit(process_scene, task) for task in tasks]):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(results) % 72 == 0:
                print(json.dumps({"completed": len(results), "total": len(tasks)}), flush=True)

    variants = summarize(results)
    report = {
        "state": "completed",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "72 frozen multi-microphone vehicle-field scenes x seven BF routes",
        "total_cases": 72,
        "engineering_gates": ENGINEERING_GATES,
        "engineering_gate_scope": (
            "BF contribution relative to explicit channel-0 no-BF; speech is measured "
            "against each route's declared clean target, and noise gain is relative "
            "to the channel-0 paired control."
        ),
        "target_scope": (
            "no-BF uses clean channel 0; BF uses calibrated delay-sum clean-beam target "
            "at the declared causal latency."
        ),
        "dataset_manifest_sha256": digest(args.manifest),
        "variants": variants,
        "mics": dict(Counter(scene["mics"] for scene in scenes)),
        "snr": dict(Counter(scene["snr_db"] for scene in scenes)),
        "spatial_noise_models": dict(
            Counter(str(scene.get("spatial_noise_model")) for scene in scenes)
        ),
        "cli_sha256": digest(cli),
        "results_sha256": digest(root / "results.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "standard_compliance": False,
        "limitation": (
            "Simulated spatial fields and deterministic geometry; not measured cabin "
            "array, steering calibration, paced device validation, or formal ITU acceptance."
        ),
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"state": "completed", "cases": 72, "cli_sha256": report["cli_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
