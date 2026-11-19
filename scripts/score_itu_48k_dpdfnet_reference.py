#!/usr/bin/env python3
"""Score the frozen 16 kHz DPDFNet2 reference on 48 kHz AEC outputs.

This is a diagnostic extension of run_itu_48k_paired_ab.py. It reads the
completed factorial evidence, resamples the actual AEC stage to 16 kHz, runs
the pinned DPDFNet2 worker, and writes a new reference-only report. No input
or existing result is modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
RESULT_ROOT = ROOT / "build/itu-48k-paired-factorial-20261118-v3"
WORKER = ROOT / "scripts/run-ns-model.py"
MODEL = ROOT / "build/ns-next-20261115/dpdfnet/assets/dpdfnet2.onnx"
MODEL_SHA256 = "4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c"
DPDF_REFERENCE_DELAY_SAMPLES = 640
RATE = 48_000
REFERENCE_RATE = 16_000


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


def load_run_module():
    path = ROOT / "scripts/run_itu_48k_paired_ab.py"
    spec = importlib.util.spec_from_file_location("itu_48k_paired_ab", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load 48 kHz paired runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_stage(path: Path, stage_id: str) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in document["pipeline_stages"] if row["id"] == stage_id]
    if len(rows) != 1:
        raise ValueError(f"expected one {stage_id}, found {len(rows)}")
    row = rows[0]
    values, rate = sf.read(row["file"], dtype="float64", always_2d=True)
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid stage audio: {row['file']}")
    delay = int(row["fixed_delay_samples"])
    if len(values) < delay:
        raise ValueError(f"stage is shorter than declared delay: {stage_id}")
    return values[delay:, 0], rate, delay, row


def resample_48_to_16(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        signal.resample_poly(values, up=1, down=3),
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


def window(values: np.ndarray, bounds_s: tuple[float, float], rate: int) -> np.ndarray:
    begin, end = (int(round(bound * rate)) for bound in bounds_s)
    if begin < 0 or end <= begin or end > len(values):
        raise ValueError(f"invalid window {bounds_s} for {len(values)} at {rate}")
    return values[begin:end]


def score_window(
    reference: np.ndarray,
    estimate: np.ndarray,
    bounds_s: tuple[float, float],
    rate: int,
) -> dict[str, float]:
    ref = window(reference, bounds_s, rate)
    est = window(estimate, bounds_s, rate)
    return {
        "si_sdr_db": si_sdr(ref, est),
        "stoi": float(stoi(ref, est, rate, extended=False)),
    }


def score_dnsmos(values: np.ndarray) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cpp.tools.ecnr_perceptual import DNSMOS

    scorer = DNSMOS(ROOT / "build/ecnr-v2-20261011/perceptual-assets")
    result = scorer.score_array(
        values,
        REFERENCE_RATE,
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


def run_worker(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    check_resources()
    before = sha256(MODEL)
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
    after = sha256(MODEL)
    if before != after:
        raise RuntimeError("DPDFNet model changed during execution")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("DPDFNet receipt model SHA mismatch")
    if int(receipt.get("delay_samples") or -1) != DPDF_REFERENCE_DELAY_SAMPLES:
        raise RuntimeError("DPDFNet fixed delay changed")
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "receipt": receipt,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: score_itu_48k_dpdfnet_reference.py")
    run_module = load_run_module()
    check_resources()
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("DPDFNet model SHA mismatch")
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(RESULT_ROOT)
    output_root = RESULT_ROOT / "dpdfnet-reference"
    if output_root.exists():
        raise FileExistsError(f"reference evidence already exists: {output_root}")
    output_root.mkdir()
    result_path = output_root / "result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": (
            "frozen DPDFNet2 16 kHz reference applied to the actual 48 kHz "
            "AEC stage; diagnostic comparison, not an ITU-T compliance result"
        ),
        "source_result": str(RESULT_ROOT / "result.json"),
        "source_result_sha256": sha256(RESULT_ROOT / "result.json"),
        "model": {"path": str(MODEL), "sha256": sha256(MODEL)},
        "reference_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
        "reference_rate_hz": REFERENCE_RATE,
        "fixtures": {},
    }

    fixture_doc = json.loads((RESULT_ROOT / "result.json").read_text(encoding="utf-8"))
    for fixture_name in ("path_change", "double_talk"):
        fixture = fixture_doc["fixtures"][fixture_name]
        destination = output_root / fixture_name
        destination.mkdir()
        routes: dict[str, Any] = {}
        for aec_family, route_name in (
            ("aec3", "aec3_webrtc_ns"),
            ("mdf_res", "mdf_res_webrtc_ns"),
        ):
            stages_path = RESULT_ROOT / fixture_name / f"{route_name}.stages.json"
            aec_values, aec_rate, aec_delay, aec_stage = read_stage(
                stages_path, "tx_aec"
            )
            if aec_rate != RATE:
                raise ValueError(f"{route_name}: AEC stage is not 48 kHz")
            clean_path = ROOT / fixture["signals"]["reference"]["path"]
            clean_48, clean_rate = sf.read(
                clean_path, dtype="float64", always_2d=True
            )
            if clean_rate != RATE or clean_48.shape[1] < 1:
                raise ValueError("clean reference is not 48 kHz mono")
            clean_48 = clean_48[:, 0]
            if len(clean_48) != len(aec_values):
                raise ValueError("AEC stage and clean reference lengths differ")

            aec_16 = quantize_pcm16_float(resample_48_to_16(aec_values))
            clean_16 = resample_48_to_16(clean_48)
            if len(aec_16) != len(clean_16):
                raise ValueError("16 kHz AEC/reference lengths differ")

            input_path = destination / f"{aec_family}.aec-16k.float.wav"
            output_path = destination / f"{aec_family}.dpdfnet2-16k.float.wav"
            receipt_path = destination / f"{aec_family}.receipt.json"
            sf.write(
                input_path,
                aec_16,
                REFERENCE_RATE,
                subtype="FLOAT",
                format="WAV",
            )
            execution = run_worker(input_path, output_path, receipt_path)
            enhanced, enhanced_rate = sf.read(
                output_path, dtype="float64", always_2d=True
            )
            if enhanced_rate != REFERENCE_RATE or enhanced.shape[1] != 1:
                raise ValueError("DPDFNet output is not mono 16 kHz")
            enhanced = enhanced[:, 0]
            if len(enhanced) != len(clean_16):
                raise ValueError("DPDFNet output and clean reference lengths differ")

            windows = (
                {"near_only": (2.0, 5.0), "double_talk": (8.0, 11.0)}
                if fixture_name == "double_talk"
                else {}
            )
            route_result: dict[str, Any] = {
                "aec_stage": {
                    "path": aec_stage["file"],
                    "sha256": sha256(Path(aec_stage["file"])),
                    "fixed_delay_samples": aec_delay,
                    "channels_used": 1,
                },
                "aec_16k_input": {
                    "path": str(input_path),
                    "sha256": sha256(input_path),
                    "samples": int(len(aec_16)),
                    "pcm16_quantized_float": True,
                },
                "dpdfnet_output": {
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                    "samples": int(len(enhanced)),
                },
                "worker": execution,
                "fidelity": {
                    label: score_window(clean_16, enhanced, bounds, REFERENCE_RATE)
                    for label, bounds in windows.items()
                },
            }
            if fixture_name == "double_talk":
                route_result["dnsmos_whole_scene"] = score_dnsmos(enhanced)
            routes[aec_family] = route_result
            print(fixture_name, aec_family, "completed", flush=True)
        result["fixtures"][fixture_name] = {"routes": routes}
    result["status"] = "completed"
    write_json(result_path, result)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
