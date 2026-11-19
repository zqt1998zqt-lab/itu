#!/usr/bin/env python3
"""Run a four-route 48 kHz AEC x NS engineering comparison.

This script only creates a new evidence directory. It never deletes, moves, or
overwrites an existing artifact. The RX playback reference is frozen once per
fixture and shared byte-for-byte by all TX routes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from pystoi import stoi
from scipy import signal
import soundfile as sf


ROOT = Path("/home/zhangqingtian/project/itu")
CLI = ROOT / "build/host/cpp/itu_audio_cli"
CLI_SHA256 = "152a6b4e2b5e4be608be428232e91d385088ac2ec4e1482e9f595ff7c9715b10"
HYBRID_MODEL = ROOT / "cpp/models/hybrid-ns-gru32-v1/student-48000.bin"
HYBRID_MODEL_SHA256 = "d27e51dee54dc16390bb9e08a8b7897d8f94132169572ec2599694bc46934dec"
DNSMOS_ASSETS = ROOT / "build/ecnr-v2-20261011/perceptual-assets"
RATE = 48_000

FIXTURES = {
    "path_change": {
        "root": ROOT
        / "build/acoustic-qualification/acoustic-stress-datasets"
        / "7bad1929c761167b31758785e243c81c",
        "case_id": "echo-SWB-wired-path_change-01-stress-v1",
        "far_measure_s": (2.0, 5.5),
        "mixed_attenuation_windows_s": {
            "steady_before_switch": (2.0, 5.5),
            "first_second_after_switch": (6.0, 7.0),
            "recovered_after_switch": (7.3, 8.3),
        },
        "fidelity_windows_s": {},
        "dnsmos": False,
    },
    "double_talk": {
        "root": ROOT
        / "build/acoustic-qualification/acoustic-stress-datasets"
        / "0ce869dda093bab274d5e913b0875c04",
        "case_id": "double_talk-SWB-wired-transition-02-stress-v1",
        "far_measure_s": (5.0, 8.0),
        "mixed_attenuation_windows_s": {
            "far_only_pre_double_talk": (5.0, 8.0),
            "noise_only": (11.0, 12.0),
        },
        "fidelity_windows_s": {
            "near_only": (2.0, 5.0),
            "double_talk": (8.0, 11.0),
        },
        "dnsmos": True,
    },
}

ROUTES = {
    "aec3_webrtc_ns": {
        "aec": "webrtc",
        "ns": "webrtc",
        "expected_aec": "webrtc_aec3",
        "expected_ns": "webrtc",
        "hybrid": False,
    },
    "aec3_omlsa_gru32": {
        "aec": "webrtc",
        "ns": "omlsa_imcra",
        "expected_aec": "webrtc_aec3",
        "expected_ns": "omlsa_imcra",
        "hybrid": True,
    },
    "mdf_res_webrtc_ns": {
        "aec": "speex",
        "ns": "webrtc",
        "expected_aec": "speex_mdf",
        "expected_ns": "webrtc",
        "hybrid": False,
    },
    "mdf_res_omlsa_gru32": {
        "aec": "speex",
        "ns": "omlsa_imcra",
        "expected_aec": "speex_mdf",
        "expected_ns": "omlsa_imcra",
        "hybrid": True,
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_resources() -> None:
    mem_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    free = shutil.disk_usage(ROOT).free
    if free < 100 * 2**30 or mem_kib * 1024 < 5 * 2**30:
        raise RuntimeError(
            "resource floor reached; retain all files and stop: "
            f"free={free}, MemAvailable={mem_kib * 1024}"
        )


def verify_cli() -> None:
    actual = sha256(CLI)
    if actual != CLI_SHA256:
        raise RuntimeError(f"CLI SHA mismatch: expected {CLI_SHA256}, got {actual}")
    actual_model = sha256(HYBRID_MODEL)
    if actual_model != HYBRID_MODEL_SHA256:
        raise RuntimeError(
            f"hybrid model SHA mismatch: expected {HYBRID_MODEL_SHA256}, got {actual_model}"
        )


def run_cli(argv: list[str], log_path: Path, timeout_s: int = 240) -> dict[str, Any]:
    check_resources()
    before = sha256(CLI)
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.perf_counter() - started
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
        stream.write(completed.stderr)
    after = sha256(CLI)
    if before != after:
        raise RuntimeError("CLI changed during execution; evidence is invalid")
    if completed.returncode:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"CLI failed with exit {completed.returncode}: "
            + (tail[-1] if tail else "no diagnostic output")
        )
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
        "cli_sha256_before": before,
        "cli_sha256_after": after,
    }


def read_case(fixture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = json.loads((fixture["root"] / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    case = next(row for row in cases if row.get("case_id") == fixture["case_id"])
    paths = {
        role: (fixture["root"] / relative).resolve(strict=True)
        for role, relative in case["signals"].items()
    }
    expected = {"far_end", "microphone", "noise", "reference"}
    if set(paths) != expected:
        raise ValueError(f"unexpected fixture signals: {sorted(paths)}")
    for role, path in paths.items():
        info = sf.info(path)
        if info.samplerate != RATE:
            raise ValueError(f"{role} is not 48 kHz: {path}")
        if not info.frames or info.channels < 1:
            raise ValueError(f"{role} is empty or has no channels: {path}")
    return case, paths


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid audio: {path}")
    return values[:, 0], rate


def read_stage(
    stages_path: Path, stage_id: str
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    document = json.loads(stages_path.read_text(encoding="utf-8"))
    rows = [row for row in document["pipeline_stages"] if row["id"] == stage_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {stage_id} stage, found {len(rows)}")
    row = rows[0]
    values, rate = sf.read(row["file"], dtype="float64", always_2d=True)
    delay = int(row["fixed_delay_samples"])
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid stage audio: {row['file']}")
    # tx_aec preserves the physical microphone channels. With --no-bf the
    # downstream chain consumes channel zero, so all scalar metrics use that
    # same channel while the full stage file hash remains an identity control.
    samples = values[:, 0]
    if len(samples) < delay:
        raise ValueError(f"stage shorter than declared delay: {stage_id}")
    return samples[delay:], rate, delay, row


def db(values: np.ndarray) -> float:
    return 10.0 * math.log10(max(float(np.mean(values * values)), 1e-20))


def window(values: np.ndarray, bounds_s: tuple[float, float]) -> np.ndarray:
    begin, end = (int(round(bound * RATE)) for bound in bounds_s)
    if begin < 0 or end <= begin or end > len(values):
        raise ValueError(f"invalid window {bounds_s} for {len(values)} samples")
    return values[begin:end]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-20:
        raise ValueError("SI-SDR reference has no energy")
    scale = float(np.dot(estimate, reference) / energy)
    target = scale * reference
    residual = estimate - target
    return db(target) - db(residual)


def band_scores(
    reference: np.ndarray, estimate: np.ndarray, bounds_hz: tuple[float, float]
) -> dict[str, float]:
    sos = signal.butter(6, bounds_hz, fs=RATE, btype="bandpass", output="sos")
    reference_band = signal.sosfiltfilt(sos, reference)
    estimate_band = signal.sosfiltfilt(sos, estimate)
    energy = float(np.dot(reference_band, reference_band))
    coefficient = float(np.dot(estimate_band, reference_band) / energy)
    return {
        "projection_gain_db": 20.0 * math.log10(max(abs(coefficient), 1e-8)),
        "si_sdr_db": si_sdr(reference_band, estimate_band),
    }


def route_metrics(
    *,
    fixture: dict[str, Any],
    microphone: np.ndarray,
    reference: np.ndarray,
    output: np.ndarray,
) -> dict[str, Any]:
    if len(output) != len(microphone):
        raise ValueError(
            f"aligned output length {len(output)} != microphone {len(microphone)}"
        )
    attenuation = {
        label: db(window(microphone, bounds)) - db(window(output, bounds))
        for label, bounds in fixture["mixed_attenuation_windows_s"].items()
    }
    fidelity: dict[str, Any] = {}
    for label, bounds in fixture["fidelity_windows_s"].items():
        ref_window = window(reference, bounds)
        out_window = window(output, bounds)
        fidelity[label] = {
            "si_sdr_db": si_sdr(ref_window, out_window),
            "stoi": float(stoi(ref_window, out_window, RATE, extended=False)),
            "bands": {
                "full_0.2_23.8khz": band_scores(ref_window, out_window, (200, 23_800)),
                "speech_0.2_7.5khz": band_scores(ref_window, out_window, (200, 7_500)),
                "high_8.5_15.5khz": band_scores(ref_window, out_window, (8_500, 15_500)),
                "uhf_16.5_22.5khz": band_scores(ref_window, out_window, (16_500, 22_500)),
            },
        }
    return {
        "mixed_input_attenuation_db": attenuation,
        "fidelity": fidelity,
    }


def make_dnsmos():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cpp.tools.ecnr_perceptual import DNSMOS

    return DNSMOS(DNSMOS_ASSETS)


def score_dnsmos(scorer: Any, values: np.ndarray) -> dict[str, Any]:
    result = scorer.score_array(
        values,
        RATE,
        kind="double_talk",
        use_cache=False,
    )
    if not result.get("available"):
        raise RuntimeError(f"DNSMOS unavailable: {result.get('reason')}")
    return {
        "scope": result["scope"],
        "model_sha256": result["model_sha256"],
        "poly": result["poly"],
        "raw": result["raw"],
        "num_scored_hops": result["num_scored_hops"],
    }


def rx_command(far_end: Path, prefix: Path) -> list[str]:
    return [
        str(CLI),
        "--profile",
        "vcp",
        "--rx-in",
        str(far_end),
        "--rx-processing",
        "--out",
        str(prefix),
    ]


def tx_command(
    *,
    route: dict[str, Any],
    microphone: Path,
    echo_reference: Path,
    prefix: Path,
    far_window: tuple[float, float],
) -> list[str]:
    command = [
        str(CLI),
        "--profile",
        "vcp",
        "--aec-backend",
        route["aec"],
    ]
    if route["aec"] == "speex":
        command += [
            "--speex-coherence-residual",
            "--speex-native-conditioning",
            "--speex-post-context",
        ]
    command += [
        "--ns-backend",
        route["ns"],
    ]
    if route["hybrid"]:
        command += [
            "--ns-level",
            "2",
            "--hybrid-ns-model",
            str(HYBRID_MODEL),
        ]
    command += [
        "--no-bf",
        "--no-low-cut",
        "--no-agc",
        "--no-tx-drc",
        "--no-limiter",
        "--no-rx-processing",
        "--diagnostic-stages",
        "--profile-stages",
        "--far-start",
        str(far_window[0]),
        "--far-end",
        str(far_window[1]),
        "--mic",
        str(microphone),
        "--echo",
        str(echo_reference),
        "--out",
        str(prefix),
    ]
    return command


def validate_report(
    *,
    report: dict[str, Any],
    route_name: str,
    route: dict[str, Any],
) -> None:
    checks = {
        "sample_rate": RATE,
        "frame_size": 480,
        "aec_enabled": True,
        "ns_enabled": True,
        "aec_backend_effective": route["expected_aec"],
        "ns_backend": route["expected_ns"],
    }
    for key, expected in checks.items():
        if report.get(key) != expected:
            raise ValueError(
                f"{route_name}: {key}={report.get(key)!r}, expected {expected!r}"
            )
    if route["aec"] == "speex" and not report.get(
        "speex_coherence_residual_suppression"
    ):
        raise ValueError(f"{route_name}: Coherence RES was not active")
    if route["hybrid"]:
        if report.get("hybrid_ns_state") != 1:
            raise ValueError(f"{route_name}: hybrid NS state is not ready")
        if int(report.get("hybrid_ns_inferences") or 0) <= 0:
            raise ValueError(f"{route_name}: hybrid NS made no inference")
        if int(report.get("hybrid_ns_fallback_frames") or 0) != 0:
            raise ValueError(f"{route_name}: hybrid NS used fallback")


def run_fixture(
    *,
    label: str,
    fixture: dict[str, Any],
    output_root: Path,
    dnsmos_scorer: Any,
) -> dict[str, Any]:
    case, paths = read_case(fixture)
    destination = output_root / label
    destination.mkdir()
    rx_prefix = destination / "rx"
    rx_execution = run_cli(
        rx_command(paths["far_end"], rx_prefix), destination / "rx.log"
    )
    rx_report_path = Path(str(rx_prefix) + ".json")
    rx_report = json.loads(rx_report_path.read_text(encoding="utf-8"))
    if (
        rx_report.get("sample_rate") != RATE
        or rx_report.get("rx_processing") is not True
        or int(rx_report.get("rx_processing_frames") or 0) <= 0
    ):
        raise ValueError(f"{label}: RX processing was not proven")
    rx_speaker = Path(str(rx_prefix) + ".rx-spk.wav")
    rx_reference = Path(str(rx_prefix) + ".rx-ref.wav")
    if sha256(rx_speaker) != sha256(rx_reference):
        raise ValueError(f"{label}: RX speaker and echo-reference outputs differ")

    microphone, _ = read_mono(paths["microphone"])
    clean_reference, _ = read_mono(paths["reference"])
    if len(microphone) != len(clean_reference):
        raise ValueError(f"{label}: microphone and clean reference lengths differ")

    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "fixture_root": str(fixture["root"]),
        "evidence_scope": "synthetic acoustic stress engineering stimulus",
        "actual_vehicle_recording": False,
        "signals": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in paths.items()
        },
        "rx": {
            "execution": rx_execution,
            "report_path": str(rx_report_path),
            "report_sha256": sha256(rx_report_path),
            "speaker_path": str(rx_speaker),
            "speaker_sha256": sha256(rx_speaker),
            "reference_path": str(rx_reference),
            "reference_sha256": sha256(rx_reference),
            "processing_frames": rx_report["rx_processing_frames"],
        },
        "routes": {},
    }

    for route_name, route in ROUTES.items():
        prefix = destination / route_name
        execution = run_cli(
            tx_command(
                route=route,
                microphone=paths["microphone"],
                echo_reference=rx_speaker,
                prefix=prefix,
                far_window=fixture["far_measure_s"],
            ),
            destination / f"{route_name}.log",
        )
        report_path = Path(str(prefix) + ".json")
        stages_path = Path(str(prefix) + ".stages.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(report=report, route_name=route_name, route=route)

        final_output, final_rate, final_delay, final_stage = read_stage(
            stages_path, "tx_output"
        )
        ns_output, ns_rate, ns_delay, ns_stage = read_stage(stages_path, "tx_ns")
        aec_output, aec_rate, aec_delay, aec_stage = read_stage(
            stages_path, "tx_aec"
        )
        if final_rate != RATE or ns_rate != RATE or aec_rate != RATE:
            raise ValueError(f"{route_name}: expected native 48 kHz stage exports")
        if len(final_output) != len(microphone):
            raise ValueError(f"{route_name}: final aligned stage has wrong length")
        if len(ns_output) != len(microphone):
            raise ValueError(f"{route_name}: NS aligned stage has wrong length")
        if len(aec_output) != len(microphone):
            raise ValueError(f"{route_name}: AEC aligned stage has wrong length")

        metrics = route_metrics(
            fixture=fixture,
            microphone=microphone,
            reference=clean_reference,
            output=final_output,
        )
        ns_metrics = route_metrics(
            fixture=fixture,
            microphone=microphone,
            reference=clean_reference,
            output=ns_output,
        )
        route_result: dict[str, Any] = {
            "execution": execution,
            "report_path": str(report_path),
            "report_sha256": sha256(report_path),
            "stages_path": str(stages_path),
            "stages_sha256": sha256(stages_path),
            "stage_delay_samples": {
                "tx_aec": aec_delay,
                "tx_ns": ns_delay,
                "tx_output": final_delay,
            },
            "stage_file_sha256": {
                "tx_aec": sha256(Path(aec_stage["file"])),
                "tx_ns": sha256(Path(ns_stage["file"])),
                "tx_output": sha256(Path(final_stage["file"])),
            },
            "report": {
                key: report.get(key)
                for key in (
                    "aec_backend",
                    "aec_backend_requested",
                    "aec_backend_effective",
                    "aec_enabled",
                    "ns_backend",
                    "ns_enabled",
                    "hybrid_ns_model",
                    "hybrid_ns_state",
                    "hybrid_ns_inferences",
                    "hybrid_ns_fallback_frames",
                    "speex_coherence_residual_suppression",
                    "dsp_delay_samples",
                    "linear_erle_far_only_estimate_db",
                    "process_p95_ms",
                    "process_max_ms",
                    "process_rtf",
                    "process_frames_over_10ms",
                    "standard_compliance",
                )
            },
            "final_metrics": metrics,
            "ns_output_metrics": ns_metrics,
        }
        if fixture["dnsmos"]:
            route_result["dnsmos_final_whole_scene"] = score_dnsmos(
                dnsmos_scorer, final_output
            )
            route_result["dnsmos_ns_output_whole_scene"] = score_dnsmos(
                dnsmos_scorer, ns_output
            )
        result["routes"][route_name] = route_result
        print(
            label,
            route_name,
            "delay_ms",
            final_delay / RATE * 1000,
            "rtf",
            report["process_rtf"],
            flush=True,
        )

    ns_upstream_hashes: dict[str, str] = {}
    for route_name in ROUTES:
        stages_path = destination / f"{route_name}.stages.json"
        document = json.loads(stages_path.read_text(encoding="utf-8"))
        row = next(
            item
            for item in document["pipeline_stages"]
            if item["id"] == "tx_aec"
        )
        ns_upstream_hashes[route_name] = sha256(Path(row["file"]))
    if (
        ns_upstream_hashes["aec3_webrtc_ns"]
        != ns_upstream_hashes["aec3_omlsa_gru32"]
        or ns_upstream_hashes["mdf_res_webrtc_ns"]
        != ns_upstream_hashes["mdf_res_omlsa_gru32"]
    ):
        raise ValueError("NS-only pair did not receive byte-identical AEC output")
    result["ns_only_pair_control"] = {
        "ns_upstream_stage": "tx_aec",
        "tx_aec_sha256": ns_upstream_hashes,
        "aec3_pair_identical": ns_upstream_hashes["aec3_webrtc_ns"]
        == ns_upstream_hashes["aec3_omlsa_gru32"],
        "mdf_res_pair_identical": ns_upstream_hashes["mdf_res_webrtc_ns"]
        == ns_upstream_hashes["mdf_res_omlsa_gru32"],
    }
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: run_itu_48k_paired_ab.py <new-output-directory>")
    output_root = Path(argv[1]).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; choose a new directory: {output_root}"
        )
    check_resources()
    verify_cli()
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    result = {
        "schema_version": 1,
        "scope": (
            "48 kHz four-route AEC x NS engineering factorial on synthetic "
            "acoustic stress fixtures; not vehicle measurement or ITU-T compliance"
        ),
        "rate_hz": RATE,
        "cli": {
            "path": str(CLI),
            "sha256": sha256(CLI),
        },
        "hybrid_model": {
            "path": str(HYBRID_MODEL),
            "sha256": sha256(HYBRID_MODEL),
        },
        "routes": ROUTES,
        "fixtures": {},
        "resources_before": {
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "mem_available_kib": next(
                int(line.split()[1])
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            ),
        },
    }
    dnsmos = make_dnsmos()
    for label, fixture in FIXTURES.items():
        result["fixtures"][label] = run_fixture(
            label=label,
            fixture=fixture,
            output_root=output_root,
            dnsmos_scorer=dnsmos,
        )
    verify_cli()
    result["resources_after"] = {
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "mem_available_kib": next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        ),
    }
    result["elapsed_s"] = time.perf_counter() - started
    result["status"] = "completed"
    write_json(output_root / "result.json", result)
    print(output_root / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
