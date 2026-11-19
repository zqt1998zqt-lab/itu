#!/usr/bin/env python3
"""Run strict same-input 48 kHz NS-only comparisons.

The AEC stage is run once per family. Its actual tx_ns_input is frozen, then
WebRTC NS and OM-LSA/GRU32 consume the same bytes. The frozen 16 kHz DPDFNet2
reference consumes a fixed 48->16 kHz projection of those same bytes.
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
SOURCE = ROOT / "build/itu-48k-paired-factorial-20261118-v3"
CLI = ROOT / "build/itu-48k-ns-export-cli-20261118-v1/itu_audio_cli"
CLI_SHA256 = "ca8e5350295f8c876c6a51bb079a790f7919a19f9a7b8ab6146db568b2f1cf3a"
HYBRID_MODEL = ROOT / "cpp/models/hybrid-ns-gru32-v1/student-48000.bin"
HYBRID_SHA256 = "d27e51dee54dc16390bb9e08a8b7897d8f94132169572ec2599694bc46934dec"
DPDF_MODEL = ROOT / "build/ns-next-20261115/dpdfnet/assets/dpdfnet2.onnx"
DPDF_SHA256 = "4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c"
WORKER = ROOT / "scripts/run-ns-model.py"
RATE = 48_000
REFERENCE_RATE = 16_000

FIXTURES = {
    "path_change": {
        "case_id": "echo-SWB-wired-path_change-01-stress-v1",
        "far_measure_s": (2.0, 5.5),
        "attenuation_windows_s": {
            "steady_before_switch": (2.0, 5.5),
            "first_second_after_switch": (6.0, 7.0),
            "recovered_after_switch": (7.3, 8.3),
        },
        "fidelity_windows_s": {},
        "dnsmos": False,
    },
    "double_talk": {
        "case_id": "double_talk-SWB-wired-transition-02-stress-v1",
        "far_measure_s": (5.0, 8.0),
        "attenuation_windows_s": {
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

AEC_ROUTES = {
    "aec3": {
        "flags": ["--aec-backend", "webrtc"],
        "expected": "webrtc_aec3",
    },
    "mdf_res": {
        "flags": [
            "--aec-backend",
            "speex",
            "--speex-coherence-residual",
            "--speex-native-conditioning",
            "--speex-post-context",
        ],
        "expected": "speex_mdf",
    },
}

NS_ROUTES = {
    "webrtc": {
        "flags": ["--ns-backend", "webrtc"],
        "expected": "webrtc",
        "hybrid": False,
    },
    "omlsa_gru32": {
        "flags": [
            "--ns-backend",
            "omlsa_imcra",
            "--ns-level",
            "2",
            "--hybrid-ns-model",
            str(HYBRID_MODEL),
        ],
        "expected": "omlsa_imcra",
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


def verify_assets() -> None:
    if sha256(CLI) != CLI_SHA256:
        raise RuntimeError("new CLI SHA mismatch")
    if sha256(HYBRID_MODEL) != HYBRID_SHA256:
        raise RuntimeError("hybrid model SHA mismatch")
    if sha256(DPDF_MODEL) != DPDF_SHA256:
        raise RuntimeError("DPDFNet model SHA mismatch")


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
    if before != sha256(CLI):
        raise RuntimeError("CLI changed during execution")
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
        "cli_sha256_after": sha256(CLI),
    }


def load_fixture(label: str, source_result: dict[str, Any]) -> dict[str, Any]:
    fixture = FIXTURES[label]
    source_fixture = source_result["fixtures"][label]
    return {
        **fixture,
        "signals": source_fixture["signals"],
        "rx": source_fixture["rx"],
    }


def read_stage(path: Path, stage_id: str) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in document["pipeline_stages"] if row["id"] == stage_id]
    if len(rows) != 1:
        raise ValueError(f"expected one {stage_id}, found {len(rows)}")
    row = rows[0]
    values, rate = sf.read(row["file"], dtype="float64", always_2d=True)
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid stage: {row['file']}")
    delay = int(row["fixed_delay_samples"])
    if len(values) < delay:
        raise ValueError(f"stage shorter than delay: {stage_id}")
    return values[delay:, 0], rate, delay, row


def resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    gcd = math.gcd(source_rate, target_rate)
    return np.asarray(
        signal.resample_poly(
            values,
            up=target_rate // gcd,
            down=source_rate // gcd,
        ),
        dtype=np.float64,
    )


def quantize_pcm16_float(values: np.ndarray) -> np.ndarray:
    scaled = np.clip(values * 32768.0, -32768.0, 32767.0)
    rounded = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    return np.asarray(rounded / 32768.0, dtype=np.float32)


def db(values: np.ndarray) -> float:
    return 10.0 * math.log10(max(float(np.mean(values * values)), 1e-20))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-20:
        raise ValueError("reference has no energy")
    scale = float(np.dot(estimate, reference) / energy)
    target = scale * reference
    residual = estimate - target
    return db(target) - db(residual)


def window(values: np.ndarray, bounds: tuple[float, float], rate: int) -> np.ndarray:
    begin, end = (int(round(value * rate)) for value in bounds)
    if begin < 0 or end <= begin or end > len(values):
        raise ValueError(f"invalid window {bounds} for {len(values)} at {rate}")
    return values[begin:end]


def band_scores(
    reference: np.ndarray,
    estimate: np.ndarray,
    rate: int,
    bounds_hz: tuple[float, float],
) -> dict[str, float]:
    sos = signal.butter(6, bounds_hz, fs=rate, btype="bandpass", output="sos")
    reference_band = signal.sosfiltfilt(sos, reference)
    estimate_band = signal.sosfiltfilt(sos, estimate)
    energy = float(np.dot(reference_band, reference_band))
    coefficient = float(np.dot(estimate_band, reference_band) / energy)
    return {
        "projection_gain_db": 20.0 * math.log10(max(abs(coefficient), 1e-8)),
        "si_sdr_db": si_sdr(reference_band, estimate_band),
    }


def score_fidelity(
    reference: np.ndarray,
    estimate: np.ndarray,
    rate: int,
    windows: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, bounds in windows.items():
        ref = window(reference, bounds, rate)
        est = window(estimate, bounds, rate)
        result[label] = {
            "si_sdr_db": si_sdr(ref, est),
            "stoi": float(stoi(ref, est, rate, extended=False)),
            "bands": {
                "full_0.2_23.8khz": band_scores(ref, est, rate, (200, 23_800))
                if rate == RATE
                else None,
                "speech_0.2_7.5khz": band_scores(ref, est, rate, (200, 7_500)),
                "high_8.5_15.5khz": band_scores(ref, est, rate, (8_500, 15_500))
                if rate == RATE
                else None,
                "uhf_16.5_22.5khz": band_scores(ref, est, rate, (16_500, 22_500))
                if rate == RATE
                else None,
            },
        }
    return result


def score_dnsmos(values: np.ndarray, rate: int) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cpp.tools.ecnr_perceptual import DNSMOS

    scorer = DNSMOS(ROOT / "build/ecnr-v2-20261011/perceptual-assets")
    result = scorer.score_array(values, rate, kind="double_talk", use_cache=False)
    if not result.get("available"):
        raise RuntimeError(f"DNSMOS unavailable: {result.get('reason')}")
    return {
        "scope": result["scope"],
        "model_sha256": result["model_sha256"],
        "poly": result["poly"],
        "raw": result["raw"],
        "num_scored_hops": result["num_scored_hops"],
    }


def run_dpdfnet(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    check_resources()
    command = [
        sys.executable,
        str(WORKER),
        "--backend",
        "dpdfnet2",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--receipt",
        str(receipt_path),
    ]
    before = sha256(DPDF_MODEL)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"DPDFNet worker failed: {tail[-1] if tail else completed.returncode}"
        )
    if before != sha256(DPDF_MODEL):
        raise RuntimeError("DPDFNet model changed during execution")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model_sha256") != DPDF_SHA256:
        raise RuntimeError("DPDFNet receipt model SHA mismatch")
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "receipt": receipt,
    }


def upstream_command(
    fixture: dict[str, Any],
    aec: dict[str, Any],
    mic: Path,
    echo: Path,
    prefix: Path,
) -> list[str]:
    return [
        str(CLI),
        "--profile",
        "vcp",
        *aec["flags"],
        "--ns-backend",
        "webrtc",
        "--ns-level",
        "0",
        "--no-bf",
        "--no-low-cut",
        "--no-agc",
        "--no-tx-drc",
        "--no-limiter",
        "--no-rx-processing",
        "--diagnostic-stages",
        "--far-start",
        str(fixture["far_measure_s"][0]),
        "--far-end",
        str(fixture["far_measure_s"][1]),
        "--mic",
        str(mic),
        "--echo",
        str(echo),
        "--out",
        str(prefix),
    ]


def ns_command(ns_input: Path, ns: dict[str, Any], prefix: Path) -> list[str]:
    return [
        str(CLI),
        "--profile",
        "vcp",
        "--no-aec",
        "--no-bf",
        "--no-low-cut",
        "--no-agc",
        "--no-tx-drc",
        "--no-limiter",
        "--no-rx-processing",
        "--diagnostic-stages",
        *ns["flags"],
        "--mic",
        str(ns_input),
        "--out",
        str(prefix),
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: run_itu_48k_ns_only.py <new-output-directory>")
    output_root = Path(argv[1]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output exists; choose a new directory: {output_root}")
    check_resources()
    verify_assets()
    if not (SOURCE / "result.json").is_file():
        raise FileNotFoundError(SOURCE / "result.json")
    source_result = json.loads((SOURCE / "result.json").read_text(encoding="utf-8"))
    output_root.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": (
            "strict same-input 48 kHz NS-only comparison after one AEC pass; "
            "synthetic acoustic stress engineering evidence, not vehicle or ITU compliance"
        ),
        "source_result": str(SOURCE / "result.json"),
        "source_result_sha256": sha256(SOURCE / "result.json"),
        "cli": {"path": str(CLI), "sha256": sha256(CLI)},
        "hybrid_model": {"path": str(HYBRID_MODEL), "sha256": sha256(HYBRID_MODEL)},
        "dpdfnet_model": {"path": str(DPDF_MODEL), "sha256": sha256(DPDF_MODEL)},
        "fixtures": {},
    }
    for fixture_name in FIXTURES:
        fixture = load_fixture(fixture_name, source_result)
        destination = output_root / fixture_name
        destination.mkdir()
        mic = Path(fixture["signals"]["microphone"]["path"])
        clean = Path(fixture["signals"]["reference"]["path"])
        echo = Path(fixture["rx"]["speaker_path"])
        clean_values, clean_rate = sf.read(clean, dtype="float64", always_2d=True)
        if clean_rate != RATE or clean_values.shape[1] < 1:
            raise ValueError("clean reference is not 48 kHz mono")
        clean_48 = clean_values[:, 0]
        fixture_result: dict[str, Any] = {
            "case_id": fixture["case_id"],
            "signals": fixture["signals"],
            "rx": fixture["rx"],
            "aec_families": {},
        }
        for family, aec in AEC_ROUTES.items():
            family_dir = destination / family
            family_dir.mkdir()
            upstream_prefix = family_dir / "upstream"
            upstream_execution = run_cli(
                upstream_command(fixture, aec, mic, echo, upstream_prefix),
                family_dir / "upstream.log",
            )
            upstream_report_path = Path(str(upstream_prefix) + ".json")
            upstream_report = json.loads(upstream_report_path.read_text(encoding="utf-8"))
            if upstream_report.get("aec_backend_effective") != aec["expected"]:
                raise ValueError(f"{family}: AEC effective backend mismatch")
            if upstream_report.get("aec_enabled") is not True:
                raise ValueError(f"{family}: AEC was not enabled")
            ns_input_raw, ns_input_rate, ns_input_delay, ns_input_stage = read_stage(
                Path(str(upstream_prefix) + ".stages.json"), "tx_ns_input"
            )
            if ns_input_rate != RATE:
                raise ValueError(f"{family}: tx_ns_input is not 48 kHz")
            if len(ns_input_raw) != len(clean_48):
                raise ValueError(f"{family}: tx_ns_input length differs from fixture")
            ns_input_path = family_dir / "tx_ns_input.aligned.float.wav"
            sf.write(
                ns_input_path,
                np.asarray(ns_input_raw, dtype=np.float32),
                RATE,
                subtype="FLOAT",
                format="WAV",
            )
            ns_input_sha = sha256(ns_input_path)
            family_result: dict[str, Any] = {
                "upstream": {
                    "execution": upstream_execution,
                    "report_path": str(upstream_report_path),
                    "report_sha256": sha256(upstream_report_path),
                    "tx_ns_input_stage": {
                        "path": ns_input_stage["file"],
                        "raw_sha256": sha256(Path(ns_input_stage["file"])),
                        "fixed_delay_samples": ns_input_delay,
                        "sample_rate": ns_input_rate,
                    },
                    "aligned_ns_input": {
                        "path": str(ns_input_path),
                        "sha256": ns_input_sha,
                        "samples": int(len(ns_input_raw)),
                    },
                },
                "ns_routes": {},
            }
            for ns_name, ns in NS_ROUTES.items():
                ns_dir = family_dir / ns_name
                ns_dir.mkdir()
                prefix = ns_dir / "ns"
                execution = run_cli(
                    ns_command(ns_input_path, ns, prefix),
                    ns_dir / "ns.log",
                )
                report_path = Path(str(prefix) + ".json")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("ns_backend") != ns["expected"]:
                    raise ValueError(f"{family}/{ns_name}: NS backend mismatch")
                if report.get("aec_enabled") is not False:
                    raise ValueError(f"{family}/{ns_name}: NS-only pass enabled AEC")
                if ns["hybrid"]:
                    if report.get("hybrid_ns_state") != 1:
                        raise ValueError(f"{family}/{ns_name}: GRU not ready")
                    if int(report.get("hybrid_ns_inferences") or 0) <= 0:
                        raise ValueError(f"{family}/{ns_name}: GRU made no inference")
                    if int(report.get("hybrid_ns_fallback_frames") or 0) != 0:
                        raise ValueError(f"{family}/{ns_name}: GRU fallback")
                ns_output, ns_rate, ns_delay, ns_stage = read_stage(
                    Path(str(prefix) + ".stages.json"), "tx_ns"
                )
                ns_tap, tap_rate, tap_delay, tap_stage = read_stage(
                    Path(str(prefix) + ".stages.json"), "tx_ns_input"
                )
                if ns_rate != RATE or tap_rate != RATE:
                    raise ValueError(f"{family}/{ns_name}: NS stages not 48 kHz")
                tap_values, tap_values_rate = sf.read(
                    tap_stage["file"], dtype="float64", always_2d=True
                )
                if (
                    tap_values_rate != RATE
                    or tap_values.shape[1] != 1
                    or not np.array_equal(tap_values[:, 0], ns_input_raw)
                ):
                    raise ValueError(f"{family}/{ns_name}: NS input tap changed")
                if len(ns_output) != len(clean_48):
                    raise ValueError(f"{family}/{ns_name}: NS output length mismatch")
                fidelity_48 = score_fidelity(
                    clean_48,
                    ns_output,
                    RATE,
                    fixture["fidelity_windows_s"],
                )
                attenuation = {
                    label: db(window(ns_input_raw, bounds, RATE))
                    - db(window(ns_output, bounds, RATE))
                    for label, bounds in fixture["attenuation_windows_s"].items()
                }
                output_16 = resample(ns_output, RATE, REFERENCE_RATE)
                clean_16 = resample(clean_48, RATE, REFERENCE_RATE)
                fidelity_16 = score_fidelity(
                    clean_16,
                    output_16,
                    REFERENCE_RATE,
                    fixture["fidelity_windows_s"],
                )
                route_result: dict[str, Any] = {
                    "execution": execution,
                    "report_path": str(report_path),
                    "report_sha256": sha256(report_path),
                    "stages_path": str(Path(str(prefix) + ".stages.json")),
                    "stages_sha256": sha256(Path(str(prefix) + ".stages.json")),
                    "ns_input_tap_sha256": sha256(Path(tap_stage["file"])),
                    "ns_output_sha256": sha256(Path(ns_stage["file"])),
                    "ns_input_delay_samples": tap_delay,
                    "ns_output_delay_samples": ns_delay,
                    "report": {
                        key: report.get(key)
                        for key in (
                            "aec_enabled",
                            "ns_backend",
                            "ns_enabled",
                            "hybrid_ns_model",
                            "hybrid_ns_state",
                            "hybrid_ns_inferences",
                            "hybrid_ns_fallback_frames",
                            "dsp_delay_samples",
                            "process_p95_ms",
                            "process_max_ms",
                            "process_rtf",
                            "process_frames_over_10ms",
                            "standard_compliance",
                        )
                    },
                    "attenuation_db": attenuation,
                    "fidelity_48k": fidelity_48,
                    "fidelity_16k": fidelity_16,
                }
                if fixture["dnsmos"]:
                    route_result["dnsmos_48k"] = score_dnsmos(ns_output, RATE)
                    route_result["dnsmos_16k_projection"] = score_dnsmos(
                        output_16, REFERENCE_RATE
                    )
                if ns_name == "webrtc":
                    ns_input_16_path = ns_dir / "tx_ns_input.16k.float.wav"
                    ns_input_16 = resample(ns_input_raw, RATE, REFERENCE_RATE)
                    sf.write(
                        ns_input_16_path,
                        quantize_pcm16_float(ns_input_16),
                        REFERENCE_RATE,
                        subtype="FLOAT",
                        format="WAV",
                    )
                    dpdf_output = ns_dir / "dpdfnet2.16k.float.wav"
                    dpdf_receipt = ns_dir / "dpdfnet2.receipt.json"
                    dpdf_execution = run_dpdfnet(
                        ns_input_16_path, dpdf_output, dpdf_receipt
                    )
                    dpdf_values, dpdf_rate = sf.read(
                        dpdf_output, dtype="float64", always_2d=True
                    )
                    if dpdf_rate != REFERENCE_RATE or dpdf_values.shape[1] != 1:
                        raise ValueError("DPDFNet output is not mono 16 kHz")
                    dpdf_values = dpdf_values[:, 0]
                    route_result["dpdfnet2_16k_reference"] = {
                        "execution": dpdf_execution,
                        "input_path": str(ns_input_16_path),
                        "input_sha256": sha256(ns_input_16_path),
                        "output_path": str(dpdf_output),
                        "output_sha256": sha256(dpdf_output),
                        "fidelity_16k": score_fidelity(
                            clean_16,
                            dpdf_values,
                            REFERENCE_RATE,
                            fixture["fidelity_windows_s"],
                        ),
                    }
                    if fixture["dnsmos"]:
                        route_result["dpdfnet2_16k_reference"]["dnsmos_16k"] = (
                            score_dnsmos(dpdf_values, REFERENCE_RATE)
                        )
                family_result["ns_routes"][ns_name] = route_result
            fixture_result["aec_families"][family] = family_result
        result["fixtures"][fixture_name] = fixture_result
        print(fixture_name, "completed", flush=True)
    verify_assets()
    result["status"] = "completed"
    write_json(output_root / "result.json", result)
    print(output_root / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
