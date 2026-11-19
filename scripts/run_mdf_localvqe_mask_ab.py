#!/usr/bin/env python3
"""Controlled A/B: external Speex MDF residual -> LocalVQE 203K mask.

This is an engineering experiment, not a production route or a formal ITU-T
measurement. It uses the pinned LocalVQE 200K weights and a separately built
mask-only library that skips LocalVQE's own DAF while reusing its graph and OLA.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build/aec-routes-host-20261111-v2/cpp/itu_audio_cli"
CLI_SHA256 = "75c9c2053bfd8dd5551122dac0e5838a586680c51ab1c07f4a61ce30b8d9daf1"
LIBRARY = ROOT / "build/localvqe-residual-build-jMdatk/liblocalvqe-residual.so"
MODEL = ROOT / "build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf"
MODEL_SHA256 = "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
STRESS = ROOT / "build/acoustic-qualification/acoustic-stress-datasets"
DATASETS = {
    "far_single": ("0563b621e16a1a7702dd7994683ddda2", "steady_echo"),
    "double_talk": ("01d5e9176ea5abbbead40185fc9b4546", "balanced"),
    "path_change": ("0563b621e16a1a7702dd7994683ddda2", "path_change"),
}
RATE = 16000
SECONDS = 4
SAMPLES = RATE * SECONDS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mono(path: Path, limit: int = SAMPLES) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE:
        raise ValueError(f"{path} must be 16 kHz")
    x = np.ascontiguousarray(audio[:limit, 0])
    if len(x) < 512:
        raise ValueError(f"{path} too short")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise ValueError(f"{path} contains invalid PCM")
    return x


def write_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(path, np.ascontiguousarray(audio, dtype=np.float32), RATE, subtype="FLOAT")


def frozen_cases(kind: str, count: int) -> list[dict]:
    dataset, category = DATASETS[kind]
    base = STRESS / dataset
    manifest = json.loads((base / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c["provenance"].get("workload_category") == category]
    if not cases:
        raise ValueError(f"no frozen {kind} cases")
    rows = []
    for case in cases[:count]:
        paths = {role: base / rel for role, rel in case["signals"].items()}
        if any(digest(path) != case["signals_sha256"][role] for role, path in paths.items()):
            raise ValueError(f"frozen dataset hash mismatch: {case['case_id']}")
        rows.append({"id": case["case_id"], "paths": paths, "hashes": case["signals_sha256"]})
    return rows


def mdf_inputs(case: dict, work: Path) -> dict[str, np.ndarray]:
    prefix = work / "mdf"
    mic = mono(case["paths"]["microphone"])
    ref = mono(case["paths"]["far_end"])
    clean = mono(case["paths"]["reference"])
    # The frozen generator uses a stereo microphone; channel 0 is explicitly
    # selected for this mono experiment and recorded in the report.
    write_wav(work / "mic.wav", mic)
    write_wav(work / "playback.wav", ref)
    command = [str(CLI), "--profile", "general", "--aec-backend", "speex", "--no-bf",
               "--mic", str(work / "mic.wav"), "--echo", str(work / "playback.wav"),
               "--out", str(prefix)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError(f"MDF CLI failed: {process.stderr[-1000:]}")
    report = json.loads(prefix.with_suffix(".json").read_text())
    if report.get("aec_backend") != "speex_mdf" or report.get("linear_aec_delay_samples") != 0:
        raise RuntimeError("unexpected MDF report contract")
    e = mono(prefix.with_suffix(".linear-aec.wav"))
    x = mono(prefix.with_suffix(".aec-input.wav"))
    if not (len(e) == len(x) == len(mic)):
        raise RuntimeError("MDF tap length mismatch")
    yhat = np.ascontiguousarray(x - e)
    render = mono(prefix.with_suffix(".aec-input.wav"))  # positive identity control
    return {"mic": mic, "playback": ref, "clean": clean, "e": e,
            "conditioned_capture": x, "yhat": yhat, "mdf_report": report}


class LocalVQE:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL(str(LIBRARY))
        u, f = ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)
        self.lib.localvqe_options_new.restype = u
        self.lib.localvqe_options_free.argtypes = [u]
        self.lib.localvqe_options_set_model_path.argtypes = [u, ctypes.c_char_p]
        self.lib.localvqe_options_set_model_path.restype = ctypes.c_int
        self.lib.localvqe_options_set_threads.argtypes = [u, ctypes.c_int]
        self.lib.localvqe_options_set_threads.restype = ctypes.c_int
        self.lib.localvqe_new_with_options.argtypes = [u]
        self.lib.localvqe_new_with_options.restype = u
        self.lib.localvqe_free.argtypes = [u]
        self.lib.localvqe_last_error.argtypes = [u]
        self.lib.localvqe_last_error.restype = ctypes.c_char_p
        self.lib.localvqe_sample_rate.argtypes = [u]
        self.lib.localvqe_sample_rate.restype = ctypes.c_int
        self.lib.localvqe_hop_length.argtypes = [u]
        self.lib.localvqe_hop_length.restype = ctypes.c_int
        self.lib.localvqe_process_f32.argtypes = [u, f, f, ctypes.c_int, f]
        self.lib.localvqe_process_f32.restype = ctypes.c_int
        self.lib.localvqe_process_residual_f32.argtypes = [u, f, f, ctypes.c_int, f]
        self.lib.localvqe_process_residual_f32.restype = ctypes.c_int
        o = self.lib.localvqe_options_new()
        if not o or self.lib.localvqe_options_set_model_path(o, str(MODEL).encode()) != 0:
            raise RuntimeError("LocalVQE model path rejected")
        if self.lib.localvqe_options_set_threads(o, 4) != 0:
            raise RuntimeError("LocalVQE thread setting rejected")
        self.ctx = self.lib.localvqe_new_with_options(o)
        self.lib.localvqe_options_free(o)
        if not self.ctx:
            raise RuntimeError("LocalVQE 200K model load failed")
        if (self.lib.localvqe_sample_rate(self.ctx), self.lib.localvqe_hop_length(self.ctx)) != (RATE, 256):
            raise RuntimeError("unexpected LocalVQE stream contract")

    def run(self, first: np.ndarray, second: np.ndarray, *, residual: bool) -> tuple[np.ndarray, float]:
        f = ctypes.POINTER(ctypes.c_float)
        n = min(len(first), len(second))
        out = np.zeros(n, dtype="float32")
        call = self.lib.localvqe_process_residual_f32 if residual else self.lib.localvqe_process_f32
        started = time.perf_counter()
        rc = call(self.ctx, np.ascontiguousarray(first[:n]).ctypes.data_as(f),
                  np.ascontiguousarray(second[:n]).ctypes.data_as(f), n,
                  out.ctypes.data_as(f))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if rc:
            raise RuntimeError(self.lib.localvqe_last_error(self.ctx).decode() or f"LocalVQE rc={rc}")
        if not np.isfinite(out).all():
            raise RuntimeError("LocalVQE produced nonfinite samples")
        return out, elapsed_ms

    def close(self) -> None:
        self.lib.localvqe_free(self.ctx)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def db_ratio(a: np.ndarray, b: np.ndarray) -> float | None:
    pa, pb = float(np.mean(np.asarray(a, dtype=np.float64) ** 2)), float(np.mean(np.asarray(b, dtype=np.float64) ** 2))
    if pa <= 0 or pb <= 0:
        return None
    return 10 * np.log10(pa / pb)


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float | None:
    e, t = np.asarray(estimate, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if np.dot(t, t) <= 1e-18:
        return None
    projected = np.dot(e, t) / np.dot(t, t) * t
    residual = e - projected
    denom = np.dot(residual, residual)
    return 10 * np.log10(np.dot(projected, projected) / denom) if denom > 1e-18 else None


def shifted_target(target: np.ndarray, lag: int) -> np.ndarray:
    aligned = np.zeros_like(target)
    if lag >= 0:
        aligned[lag:] = target[:len(target) - lag]
    else:
        aligned[:len(target) + lag] = target[-lag:]
    return aligned


def best_lag(estimate: np.ndarray, target: np.ndarray) -> int:
    correlation = signal.correlate(estimate, target, mode="full", method="fft")
    lags = signal.correlation_lags(len(estimate), len(target), mode="full")
    return int(lags[int(np.argmax(np.abs(correlation)))])


def measures(output: np.ndarray, inputs: dict, *, work: Path, name: str) -> dict:
    clean = inputs["clean"]
    mic, e = inputs["mic"], inputs["e"]
    write_wav(work / f"{name}.wav", output)
    # The 200K graph has a one-hop codec delay. Report both the raw pair and
    # the declared 256-sample alignment so a timing offset is not mistaken
    # for speech distortion. Only LocalVQE graph routes receive this offset.
    graph_route = name in {"localvqe_full", "mdf_residual_mask", "mdf_residual_wrong_ref"}
    declared_target = shifted_target(clean, 256) if graph_route else clean
    measured_lag = best_lag(output, clean) if graph_route else 0
    return {
        "rms": round(rms(output), 9),
        "echo_residual_rms": round(rms(output - declared_target), 9),
        "echo_suppression_db_vs_mic": None if db_ratio(mic - declared_target, output - declared_target) is None else round(db_ratio(mic - declared_target, output - declared_target), 4),
        "near_si_sdr_db": None if si_sdr(output, declared_target) is None else round(si_sdr(output, declared_target), 4),
        "near_si_sdr_db_unaligned": None if si_sdr(output, clean) is None else round(si_sdr(output, clean), 4),
        "declared_delay_samples": 256 if graph_route else 0,
        "measured_best_lag_samples": measured_lag,
        "delta_vs_mdf_rms": round(rms(output - e), 9),
        "output_sha256": hashlib.sha256(np.ascontiguousarray(output, dtype=np.float32).tobytes()).hexdigest(),
    }


def run_case(engine: LocalVQE, kind: str, case: dict, root: Path) -> dict:
    work = root / kind / case["id"]
    work.mkdir(parents=True, exist_ok=False)
    inputs = mdf_inputs(case, work)
    write_wav(work / "mic.wav", inputs["mic"])
    write_wav(work / "playback.wav", inputs["playback"])
    write_wav(work / "clean.wav", inputs["clean"])
    write_wav(work / "mdf-e.wav", inputs["e"])
    write_wav(work / "mdf-yhat.wav", inputs["yhat"])
    routes = {}
    # Full LocalVQE baseline: its own DAF and 203K mask.
    full, full_ms = engine.run(inputs["mic"], inputs["playback"], residual=False)
    routes["localvqe_full"] = {**measures(full, inputs, work=work, name="localvqe_full"), "process_ms": round(full_ms, 3)}
    # Experimental: external MDF residual + strict yhat = conditioned capture - e.
    external, ext_ms = engine.run(inputs["e"], inputs["yhat"], residual=True)
    routes["mdf_residual_mask"] = {**measures(external, inputs, work=work, name="mdf_residual_mask"), "process_ms": round(ext_ms, 3)}
    # Negative control: second input is conditioned playback, not yhat.
    bad, bad_ms = engine.run(inputs["e"], inputs["playback"], residual=True)
    routes["mdf_residual_wrong_ref"] = {**measures(bad, inputs, work=work, name="mdf_residual_wrong_ref"), "process_ms": round(bad_ms, 3)}
    # Known-bad input: zero residual must produce finite zero output.
    zeros = np.zeros(len(inputs["e"]), dtype="float32")
    zero, zero_ms = engine.run(zeros, zeros, residual=True)
    zero_ok = bool(np.isfinite(zero).all() and np.max(np.abs(zero)) == 0)
    routes["mdf_residual_only"] = measures(inputs["e"], inputs, work=work, name="mdf_residual_only")
    routes["input_mic"] = measures(inputs["mic"], inputs, work=work, name="input_mic")
    return {
        "case_id": case["id"],
        "category": kind,
        "sources": {"dataset": DATASETS[kind][0], "workload_category": DATASETS[kind][1],
                    "channel_selection": "microphone channel 0 only", "hashes": case["hashes"]},
        "mdf_report": {k: inputs["mdf_report"].get(k) for k in
                       ("aec_backend", "linear_aec_available", "linear_aec_delay_samples",
                        "aec_input_delay_samples", "reference_channels", "processing_profile")},
        "routes": routes,
        "negative_controls": {"zero_residual_output_is_zero": zero_ok,
                              "different_route_output_hashes": len({routes[k]["output_sha256"] for k in
                                  ("localvqe_full", "mdf_residual_mask", "mdf_residual_wrong_ref")}) >= 2},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--cases", type=int, default=3, help="frozen cases per category")
    args = parser.parse_args()
    if args.cases < 1 or args.cases > 5:
        parser.error("--cases must be 1..5")
    if digest(CLI) != CLI_SHA256 or digest(MODEL) != MODEL_SHA256:
        raise RuntimeError("pinned CLI or model hash mismatch")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output must be a new evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    engine = LocalVQE()
    rows = []
    try:
        for kind in DATASETS:
            for case in frozen_cases(kind, args.cases):
                rows.append(run_case(engine, kind, case, out))
                print(f"done {kind} {case['id']}", flush=True)
    finally:
        engine.close()
    summary = {
        "schema": "itu-mdf-localvqe-mask-ab-v1",
        "objective": "existing MDF residual -> LocalVQE 203K mask, single-talk/double-talk/path-change A/B",
        "cli": {"path": str(CLI), "sha256": CLI_SHA256},
        "library": {"path": str(LIBRARY), "sha256": digest(LIBRARY)},
        "model": {"path": str(MODEL), "sha256": MODEL_SHA256},
        "cases": rows,
        "boundary": "controlled engineering experiment; MDF e is PCM16-derived, yhat is conditioned_capture-e; no formal compliance or production-readiness claim",
        "formal_compliance": False,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(rows), "result": str(out / "result.json")}))


if __name__ == "__main__":
    sys.exit(main())
