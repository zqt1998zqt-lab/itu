#!/usr/bin/env python3
"""Paired DRC configuration matrix on the frozen 108-case speech core."""
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


PROJECT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("controlled-native-full-20261017", "itu-controlled-quality-v1-20261017"),
    ("controlled-cavemove-20261017", "itu-cavemove-quality-v1-20261017"),
)

VARIANTS = {
    "bypass": ["--no-tx-drc", "--no-limiter"],
    "limiter_only": ["--no-tx-drc"],
    "default": [
        "--tx-drc",
        "--tx-drc-threshold",
        "-12",
        "--tx-drc-ratio",
        "4",
        "--tx-drc-knee",
        "6",
        "--tx-drc-attack-ms",
        "5",
        "--tx-drc-release-ms",
        "60",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
    ],
    "gentle": [
        "--tx-drc",
        "--tx-drc-threshold",
        "-12",
        "--tx-drc-ratio",
        "2",
        "--tx-drc-knee",
        "8",
        "--tx-drc-attack-ms",
        "8",
        "--tx-drc-release-ms",
        "100",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
    ],
    "aggressive": [
        "--tx-drc",
        "--tx-drc-threshold",
        "-18",
        "--tx-drc-ratio",
        "8",
        "--tx-drc-knee",
        "4",
        "--tx-drc-attack-ms",
        "3",
        "--tx-drc-release-ms",
        "80",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
    ],
    "fast": [
        "--tx-drc",
        "--tx-drc-threshold",
        "-12",
        "--tx-drc-ratio",
        "4",
        "--tx-drc-knee",
        "6",
        "--tx-drc-attack-ms",
        "1",
        "--tx-drc-release-ms",
        "20",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
    ],
    "lookahead_2ms": [
        "--tx-drc",
        "--tx-drc-threshold",
        "0",
        "--tx-drc-ratio",
        "1",
        "--tx-drc-knee",
        "0",
        "--tx-drc-attack-ms",
        "50",
        "--tx-drc-release-ms",
        "60",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
        "--tx-drc-lookahead-ms",
        "2",
    ],
    "lookahead_2ms_off": [
        "--tx-drc",
        "--tx-drc-threshold",
        "0",
        "--tx-drc-ratio",
        "1",
        "--tx-drc-knee",
        "0",
        "--tx-drc-attack-ms",
        "50",
        "--tx-drc-release-ms",
        "60",
        "--tx-drc-makeup-db",
        "0",
        "--tx-drc-ceiling-dbfs",
        "-0.5",
        "--tx-drc-lookahead-ms",
        "0",
    ],
}

ENGINEERING_GATES = {
    "near_late_sisdr_delta_min_db": -0.1,
    "near_late_stoi_delta_min": -0.02,
    "output_peak_max_dbfs": -0.5,
    "clipping_max_percent": 0.0,
    "max_extra_delay_ms": 3.0,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases():
    selected = []
    for run, manifest_name in SOURCES:
        manifest = json.loads(
            (PROJECT / "datasets" / manifest_name / "manifest.json").read_text()
        )
        core = {
            scene["id"]
            for scene in manifest["scenes"]
            if scene["track"] == "spectrum_screened_core"
        }
        for line in (PROJECT / "build" / run / "results.jsonl").open():
            row = json.loads(line)
            scene = row["scene"]
            if scene["family"] == "agc" and scene["id"] in core:
                selected.append((run, row))
    if len(selected) != 108 or len(
        {(run, row["scene"]["id"]) for run, row in selected}
    ) != len(selected):
        raise ValueError("DRC matrix requires 108 unique run-scoped speech cases")
    return selected


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
        return {"sisdr_db": None, "stoi": None, "projection_gain_db": None}
    rms = ev.db(ev.power(observed))
    peak = ev.db(float(np.max(np.abs(observed))) ** 2)
    return {
        "sisdr_db": min(120.0, ev.db(pproj / max(perr, 1e-20))),
        "stoi": float(stoi(target, observed, rate, extended=False)),
        "projection_gain_db": ev.db(alpha**2),
        "rms_dbfs": rms,
        "peak_dbfs": peak,
        "crest_db": peak - rms,
        "clipping_percent": float(np.mean(np.abs(observed) >= 0.999) * 100),
    }


def process_case(task):
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
        "--no-low-cut",
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
    measured = measure_case(scene, prefix, folder)
    return {
        "variant": variant,
        "original_run": run,
        "scene_id": scene["id"],
        "noise": scene.get("noise"),
        "sample_rate": scene["sample_rate"],
        "near_dbfs": scene["near_dbfs"],
        "measurements": measured,
        "drc": {
            "enabled": measured["engine"].get("tx_drc_enabled"),
            "config": measured["engine"].get("tx_drc_config"),
        },
    }


def measure_case(scene, prefix: Path, folder: Path) -> dict:
    rate = int(scene["sample_rate"])
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = read_rate(folder / "before.wav", rate)[:, 0]
    clean = read_rate(folder / "clean.wav", rate)[:, 0]
    output_file = read_rate(prefix.with_suffix(".after.wav"), rate)[:, 0]
    delay = int(engine["dsp_delay_samples"])
    if len(output_file) != len(before) + delay:
        raise ValueError(f"DRC output delay mismatch: {scene['id']}")
    # ``dsp_delay_samples`` is the complete physical offset. Keep the original
    # route timeline for peak/clipping, and remove only that declared offset for
    # quality. The look-ahead tail remains beyond the input duration.
    output = output_file
    extra_delay = int(engine.get("tx_drc_delay_samples", 0))
    after = output_file[delay : delay + len(before)]

    segments = {}
    for name in ("near_only", "near_late"):
        limits = scene["windows"][name]
        a = int(round(limits[0] * rate))
        b = int(round(limits[1] * rate))
        before_q = quality(clean[a:b], before[a:b], rate)
        after_q = quality(clean[a:b], after[a:b], rate)
        segments[name] = {
            "before": before_q,
            "after": after_q,
            "sisdr_improvement_db": after_q["sisdr_db"] - before_q["sisdr_db"],
            "stoi_improvement": after_q["stoi"] - before_q["stoi"],
        }
    limits = scene["windows"]["noise_only"]
    a = int(round(limits[0] * rate))
    b = int(round(limits[1] * rate))
    segments["noise_only"] = ev.attenuation_measurement(before[a:b], after[a:b])
    rms = ev.db(ev.power(output))
    peak = ev.db(float(np.max(np.abs(output))) ** 2)
    return {
        "id": scene["id"],
        "sample_rate": rate,
        "segments": segments,
        "engine": engine,
        "output_sha256": digest(prefix.with_suffix(".after.wav")),
        "evaluation_alignment": {
            "method": "physical zero-based output timeline",
            "declared_delay_samples": delay,
            "evaluated_samples": len(output),
        },
        "output_levels": {
            "peak_dbfs": peak,
            "rms_dbfs": rms,
            "crest_db": peak - rms,
            "clipping_percent": float(np.mean(np.abs(output) >= 0.999) * 100),
        },
        "evaluation_delay_samples": extra_delay,
    }


def summarize(results):
    bypass = {
        (row["original_run"], row["scene_id"], row["noise"]): row
        for row in results
        if row["variant"] == "bypass"
    }
    lookahead_control = {
        (row["original_run"], row["scene_id"], row["noise"]): row
        for row in results
        if row["variant"] == "lookahead_2ms_off"
    }
    variants = {}
    for name in VARIANTS:
        rows = [row for row in results if row["variant"] == name]
        if len(rows) != 108:
            raise ValueError(f"incomplete DRC variant: {name}")
        paired = []
        for row in rows:
            key = (row["original_run"], row["scene_id"], row["noise"])
            base = lookahead_control[key] if name == "lookahead_2ms" else bypass[key]
            current = row["measurements"]
            reference = base["measurements"]
            bypass_reference = bypass[key]["measurements"]
            sisdr_delta = (
                current["segments"]["near_late"]["after"]["sisdr_db"]
                - reference["segments"]["near_late"]["after"]["sisdr_db"]
            )
            stoi_delta = (
                current["segments"]["near_late"]["after"]["stoi"]
                - reference["segments"]["near_late"]["after"]["stoi"]
            )
            output_peak = current["output_levels"]["peak_dbfs"]
            clipping = current["output_levels"]["clipping_percent"]
            pass_case = (
                sisdr_delta >= ENGINEERING_GATES["near_late_sisdr_delta_min_db"]
                and stoi_delta >= ENGINEERING_GATES["near_late_stoi_delta_min"]
                and output_peak <= ENGINEERING_GATES["output_peak_max_dbfs"]
                and clipping <= ENGINEERING_GATES["clipping_max_percent"]
            )
            extra_delay_samples = (
                current["engine"].get("tx_drc_delay_samples", 0)
                - reference["engine"].get("tx_drc_delay_samples", 0)
            )
            extra_delay_ms = 1000.0 * extra_delay_samples / row["sample_rate"]
            paired.append(
                {
                    "original_run": row["original_run"],
                    "scene_id": row["scene_id"],
                    "noise": row["noise"],
                    "sample_rate": row["sample_rate"],
                    "near_dbfs": row["near_dbfs"],
                    "sisdr_delta": sisdr_delta,
                    "stoi_delta": stoi_delta,
                    "peak_reduction_db": (
                        bypass_reference["output_levels"]["peak_dbfs"] - output_peak
                    ),
                    "crest_reduction_db": (
                        bypass_reference["output_levels"]["crest_db"]
                        - current["output_levels"]["crest_db"]
                    ),
                    "engineering_pass": pass_case,
                    "output_changed": (
                        current["output_sha256"] != reference["output_sha256"]
                    ),
                    "extra_delay_samples": extra_delay_samples,
                    "extra_delay_ms": extra_delay_ms,
                }
            )
            paired[-1]["sisdr_delta_vs_bypass_db"] = (
                current["segments"]["near_late"]["after"]["sisdr_db"]
                - bypass[key]["measurements"]["segments"]["near_late"]["after"]["sisdr_db"]
            )
            paired[-1]["stoi_delta_vs_bypass"] = (
                current["segments"]["near_late"]["after"]["stoi"]
                - bypass[key]["measurements"]["segments"]["near_late"]["after"]["stoi"]
            )
            paired[-1]["engineering_pass"] = (
                paired[-1]["engineering_pass"]
                and extra_delay_ms <= ENGINEERING_GATES["max_extra_delay_ms"]
            )

        by_rate = {}
        for rate in (16000, 24000, 48000):
            subset = [row for row in paired if row["sample_rate"] == rate]
            by_rate[str(rate)] = {
                "pass": sum(row["engineering_pass"] for row in subset),
                "total": len(subset),
            }
        by_level = {}
        for level in (-45, -30, -15):
            subset = [row for row in paired if row["near_dbfs"] == level]
            by_level[str(level)] = {
                "pass": sum(row["engineering_pass"] for row in subset),
                "total": len(subset),
            }
        variants[name] = {
            "cases": len(rows),
            "engineering_pass_cases": sum(row["engineering_pass"] for row in paired),
            "engineering_fail_cases": sum(not row["engineering_pass"] for row in paired),
            "mean_sisdr_delta_vs_bypass_db": float(
                np.mean([row["sisdr_delta_vs_bypass_db"] for row in paired])
            ),
            "mean_sisdr_delta_vs_reference_db": float(
                np.mean([row["sisdr_delta"] for row in paired])
            ),
            "median_sisdr_delta_vs_bypass_db": float(
                np.median([row["sisdr_delta"] for row in paired])
            ),
            "mean_stoi_delta_vs_bypass": float(
                np.mean([row["stoi_delta_vs_bypass"] for row in paired])
            ),
            "mean_stoi_delta_vs_reference": float(
                np.mean([row["stoi_delta"] for row in paired])
            ),
            "mean_peak_reduction_db": float(
                np.mean([row["peak_reduction_db"] for row in paired])
            ),
            "median_crest_reduction_db": float(
                np.median([row["crest_reduction_db"] for row in paired])
            ),
            "outputs_changed_vs_bypass": sum(
                row["output_changed"] for row in paired
            ),
            "clipped_cases": sum(
                row["measurements"]["output_levels"]["clipping_percent"] > 0
                for row in rows
            ),
            "by_rate": by_rate,
            "by_level": by_level,
            "paired_cases": paired,
        }
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/host/cpp/itu_audio_cli"))
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    root, cli = args.root.resolve(), args.cli.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    shutil.copy2(cli, artifacts / "itu_audio_cli")
    cli = artifacts / "itu_audio_cli"
    selected = select_cases()
    tasks = [
        (root, run, row, cli, variant, flags)
        for variant, flags in VARIANTS.items()
        for run, row in selected
    ]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (
        root / "results.jsonl"
    ).open("x") as stream:
        for future in as_completed([pool.submit(process_case, task) for task in tasks]):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(results) % 108 == 0:
                print(json.dumps({"completed": len(results), "total": len(tasks)}), flush=True)

    variants = summarize(results)
    report = {
        "state": "completed",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "108 frozen speech cases x eight isolated DRC routes",
        "total_cases": 108,
        "engineering_gates": ENGINEERING_GATES,
        "engineering_gate_scope": (
            "Speech contribution relative to no-DRC/no-limiter bypass; peak ceiling "
            "and clipping are measured on the complete output."
        ),
        "variants": variants,
        "cli_sha256": digest(cli),
        "results_sha256": digest(root / "results.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "standard_compliance": False,
        "limitation": (
            "Post-stage PCM engineering replay; not acoustic playback, loudspeaker "
            "protection, true-peak compliance, device validation, or formal ITU acceptance."
        ),
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"state": "completed", "cases": 108, "cli_sha256": report["cli_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
