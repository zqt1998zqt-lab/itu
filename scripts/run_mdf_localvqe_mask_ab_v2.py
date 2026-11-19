#!/usr/bin/env python3
"""Matched post-RX, full-duration MDF -> LocalVQE 203K mask comparison.

Runs near-only, far-only, double-talk and a real 6-second path switch. Every
model receives identical delivered microphone/playback PCM. Clean targets are
used only by the scorer. Graph outputs are flushed, then trimmed by the fixed
256-sample codec delay established by the independent codec control.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
from scipy import signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build/aec-routes-host-20261111-v2/cpp/itu_audio_cli"
CLI_SHA = "75c9c2053bfd8dd5551122dac0e5838a586680c51ab1c07f4a61ce30b8d9daf1"
MODEL = ROOT / "build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf"
MODEL_SHA = "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
ORIGINAL_LIBRARY = ROOT / "build/localvqe-host-20261010/liblocalvqe.so.0.1.0"
STRESS = ROOT / "build/acoustic-qualification/acoustic-stress-datasets"
ECHO_DATASET = "0563b621e16a1a7702dd7994683ddda2"
DT_DATASET = "01d5e9176ea5abbbead40185fc9b4546"
RATE, HOP, CODEC_DELAY = 16000, 256, 256
SCHEMA = "itu-mdf-localvqe-mask-ab-v2"
GROUPS = ("near_single", "far_single", "double_talk", "path_change")
DT_CATEGORIES = ("balanced", "near_dominant", "far_dominant", "transition", "short_overlap")
ROUTES = ("input_mic", "mdf_residual_only", "localvqe_full", "mdf_residual_mask", "mdf_residual_wrong_ref")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pcm_digest(samples: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(samples, dtype="<f4").tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def resource_check() -> None:
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if shutil.disk_usage(ROOT).free < 100 * 2**30 or available < 5 * 2**30:
        raise RuntimeError("resource stop: disk <100 GiB or MemAvailable <5 GiB; preserve all evidence")


def read_mono(path: Path, *, select_channel_zero: bool = False) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or (audio.shape[1] != 1 and not select_channel_zero):
        raise ValueError(f"16 kHz mono input required: {path}")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"invalid audio: {path}")
    return np.ascontiguousarray(audio[:, 0])


def write_audio(path: Path, samples: np.ndarray) -> dict:
    if path.exists():
        raise FileExistsError(f"preserve existing evidence: {path}")
    data = np.ascontiguousarray(samples, dtype=np.float32)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("invalid output PCM")
    sf.write(path, data, RATE, subtype="FLOAT")
    return {"path": path.name, "sha256": digest(path), "pcm_sha256": pcm_digest(data), "samples": len(data)}


def call_cli(command: list[str], log: Path) -> tuple[dict, float]:
    resource_check()
    started = time.perf_counter()
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    elapsed = (time.perf_counter() - started) * 1000
    with log.open("x") as stream:
        stream.write(json.dumps(command) + "\n" + run.stdout + "\n" + run.stderr)
    if run.returncode:
        raise RuntimeError(f"CLI exit {run.returncode}: {run.stderr[-1000:]}")
    prefix = Path(command[command.index("--out") + 1])
    return json.loads(Path(str(prefix) + ".json").read_text()), elapsed


def source_cases(group: str, count: int) -> list[dict]:
    dataset = DT_DATASET if group in {"near_single", "double_talk"} else ECHO_DATASET
    base = STRESS / dataset
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for index in range(count):
        category = (DT_CATEGORIES[index % len(DT_CATEGORIES)] if dataset == DT_DATASET
                    else "path_change" if group == "path_change" else "steady_echo")
        candidates = [c for c in manifest["cases"] if c["provenance"]["workload_category"] == category]
        chosen = candidates[index // len(DT_CATEGORIES) if dataset == DT_DATASET else index]
        paths = {role: base / relative for role, relative in chosen["signals"].items()}
        for role, path in paths.items():
            if digest(path) != chosen["signals_sha256"][role]:
                raise ValueError(f"source hash mismatch: {path}")
        rows.append({"id": f"{group}-{index + 1:02d}", "source_id": chosen["case_id"],
                     "dataset": dataset, "manifest_sha256": digest(manifest_path),
                     "category": category, "paths": paths, "hashes": chosen["signals_sha256"],
                     "render": chosen["provenance"]["render_metadata"]})
    return rows


def fir_component(playback: np.ndarray, taps: list[int], gains: list[float]) -> np.ndarray:
    if len(taps) != len(gains) or not taps or any(t < 0 for t in taps):
        raise ValueError("invalid echo FIR")
    result = np.zeros(len(playback), dtype=np.float64)
    for offset, gain in zip(taps, gains):
        if offset < len(result):
            result[offset:] += playback[:len(result) - offset] * gain
    return result


def reconstruct_echo(playback: np.ndarray, segments: list[dict], echo_gain: float) -> np.ndarray:
    result = np.zeros(len(playback), dtype=np.float64)
    cursor = 0
    for segment in segments:
        start, stop = round(segment["start_s"] * RATE), round(segment["end_s"] * RATE)
        if start != cursor or stop <= start or stop > len(playback):
            raise ValueError("FIR segments must cover the full observation timeline")
        path = segment["paths"][0]
        response = fir_component(playback, path["tap_samples"], path["tap_gains"])
        result[start:stop] = response[start:stop] * echo_gain
        cursor = stop
    if cursor != len(playback):
        raise ValueError("FIR path does not cover final samples")
    return result


def verify_path_coverage(samples: int, segments: list[dict]) -> list[float]:
    events = []
    for before, after in zip(segments, segments[1:]):
        moment = float(after["start_s"])
        if before["paths"][0] == after["paths"][0]:
            continue
        if not 0 < moment * RATE < samples or samples / RATE - moment < 2:
            raise ValueError("path switch is outside the observation or has <2 s recovery")
        events.append(moment)
    if not events:
        raise ValueError("path_change has no observable FIR change")
    return events


def prepare_case(case: dict, group: str, work: Path) -> tuple[dict[str, np.ndarray], dict]:
    near = read_mono(case["paths"]["reference"])
    pre_rx = read_mono(case["paths"]["far_end"])
    noise = read_mono(case["paths"]["noise"])
    if len(near) != len(pre_rx) or len(near) != len(noise) or len(near) != 12 * RATE:
        raise ValueError("each frozen scene must be full-duration 12 seconds")
    if group == "near_single":
        # A near-only transform of a frozen clean speech/noise pair. It is
        # explicitly a reconstructed control, never relabelled original mic.
        pre_rx = np.zeros_like(pre_rx)
    source_record = write_audio(work / "pre-rx.wav", pre_rx)
    rx_prefix = work / "rx"
    rx_command = [str(CLI), "--profile", "vcp", "--rx-in", str(work / "pre-rx.wav"),
                  "--rx-processing", "--out", str(rx_prefix)]
    rx, rx_ms = call_cli(rx_command, work / "rx.log")
    playback = read_mono(work / "rx.rx-spk.wav")
    if (rx.get("rx_processing") is not True or rx.get("rx_processing_frames", 0) <= 0
            or len(playback) != len(pre_rx)):
        raise ValueError("RX processing not proven")
    segments = case["render"]["echo_path_segments"]
    events = verify_path_coverage(len(near), segments) if group == "path_change" else []
    echo = reconstruct_echo(playback, segments, float(case["render"]["echo_gain_linear"]))
    mixture = near.astype(np.float64) + echo + noise
    common_gain = min(1.0, 0.94 / max(float(np.max(np.abs(mixture))), 1e-20))
    # Preserve exact post-RX playback. Any capture safety gain changes the
    # declared microphone/path gain equally for clean, echo and noise.
    near = np.asarray(near * common_gain, dtype=np.float32)
    noise = np.asarray(noise * common_gain, dtype=np.float32)
    echo = np.asarray(echo * common_gain, dtype=np.float32)
    mic = np.asarray(near.astype(np.float64) + echo + noise, dtype=np.float32)
    arrays = {"mic": mic, "playback": playback, "clean": near, "echo": echo, "noise": noise}
    artifacts = {role: write_audio(work / f"{role}.wav", data) for role, data in arrays.items()}
    if group == "near_single" and (np.any(playback) or np.any(echo)):
        raise ValueError("near-only control has active far-end/echo")
    if group in {"far_single", "path_change"} and np.any(near):
        raise ValueError("far-only case has active near speech")
    return arrays, {"source_id": case["source_id"], "source_dataset": case["dataset"],
                    "source_manifest_sha256": case["manifest_sha256"], "source_sha256": case["hashes"],
                    "source_category": case["category"], "geometry": "channel 0 FIR; no multi-mic/BF claim",
                    "kind": "synthetic_matched_post_rx_reconstruction", "duration_s": len(mic) / RATE,
                    "pre_rx": source_record, "rx_command": rx_command, "rx_wall_ms": rx_ms,
                    "rx_report_sha256": digest(work / "rx.json"),
                    "rx_speaker_sha256": digest(work / "rx.rx-spk.wav"),
                    "rx_speaker_pcm_sha256": pcm_digest(playback),
                    "tx_echo_pcm_sha256": pcm_digest(arrays["playback"]),
                    "echo_path_segments": segments,
                    "effective_echo_gain": float(case["render"]["echo_gain_linear"]) * common_gain,
                    "capture_common_gain": common_gain, "path_change_times_s": events,
                    "artifacts": artifacts,
                    "mix_max_abs_error": float(np.max(np.abs(mic.astype(np.float64) - near - echo - noise)))}


def mdf_inputs(arrays: dict, work: Path) -> tuple[dict, dict]:
    prefix = work / "mdf"
    command = [str(CLI), "--profile", "vcp", "--aec-backend", "speex", "--no-bf",
               "--no-speex-coherence-residual", "--no-speex-multirender-context", "--no-ns", "--no-agc",
               "--no-tx-drc", "--no-limiter", "--mic", str(work / "mic.wav"),
               "--echo", str(work / "playback.wav"), "--out", str(prefix)]
    report, wall_ms = call_cli(command, work / "mdf.log")
    required = {"aec_backend": "speex_mdf", "aec_enabled": True, "speex_echo_tail_ms": 500,
                "speex_native_conditioning": True, "speex_residual_suppression": False,
                "speex_coherence_residual_suppression": False, "linear_aec_available": True,
                "linear_aec_delay_samples": 0, "aec_input_delay_samples": 0}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError(f"VCP MDF contract mismatch: { {k: report.get(k) for k in required} }")
    residual = read_mono(work / "mdf.linear-aec.wav")
    capture = read_mono(work / "mdf.aec-input.wav")
    if not (len(residual) == len(capture) == len(arrays["mic"])):
        raise ValueError("MDF taps have unexpected lengths")
    # Both WAV taps use the same clamp/lround PCM16 writer as Speex capture.
    # sf.read float32 gives exact multiples of 1/32768, not prequantized PCM.
    for path, data in ((work / "mdf.aec-input.wav", capture), (work / "mdf.linear-aec.wav", residual)):
        if sf.info(path).subtype != "PCM_16" or not np.array_equal(data * 32768, np.rint(data * 32768)):
            raise ValueError("MDF capture/error are not exact PCM16-domain taps")
    yhat = np.asarray(capture - residual, dtype=np.float32)
    if not np.array_equal(capture, residual + yhat):
        raise ValueError("x != e + yhat in PCM16 input domain")
    artifacts = {"mdf_e": write_audio(work / "mdf-e.wav", residual),
                 "mdf_yhat": write_audio(work / "mdf-yhat.wav", yhat)}
    return {"e": residual, "yhat": yhat}, {"command": command, "wall_ms": wall_ms,
            "raw_tap_scope": "VCP 500 ms MDF before fusion/RES/NS; PCM16-derived residual",
            "report": {key: report[key] for key in required},
            "report_sha256": digest(work / "mdf.json"), "artifacts": artifacts,
            "capture_quantization": "PCM_16 / 32768", "yhat_rule": "quantized_conditioned_capture - raw_MDF_error",
            "capture_reconstruction_max_abs_error": float(np.max(np.abs(capture - residual - yhat)))}


class Engine:
    def __init__(self, library: Path, *, original=False):
        self.library = library
        self.lib = ctypes.CDLL(str(library))
        u, p = ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)
        self.lib.localvqe_options_new.restype = u
        self.lib.localvqe_options_free.argtypes = [u]
        for name in ("model_path", "backend"):
            fn = getattr(self.lib, f"localvqe_options_set_{name}")
            fn.argtypes, fn.restype = [u, ctypes.c_char_p], ctypes.c_int
        self.lib.localvqe_options_set_threads.argtypes = [u, ctypes.c_int]
        self.lib.localvqe_new_with_options.argtypes, self.lib.localvqe_new_with_options.restype = [u], u
        self.lib.localvqe_free.argtypes = [u]
        self.lib.localvqe_reset.argtypes = [u]
        self.lib.localvqe_last_error.argtypes, self.lib.localvqe_last_error.restype = [u], ctypes.c_char_p
        self.lib.localvqe_set_noise_gate.argtypes = [u, ctypes.c_int, ctypes.c_float]
        self.calls = {}
        names = ["localvqe_process_frame_f32"] if original else [
            "localvqe_process_frame_f32", "localvqe_external_mask_frame_f32", "localvqe_external_codec_frame_f32"]
        for name in names:
            fn = getattr(self.lib, name)
            fn.argtypes = [u, p, ctypes.c_int, p] if "codec" in name else [u, p, p, ctypes.c_int, p]
            fn.restype = ctypes.c_int
            self.calls[name] = fn
        if not original:
            self.lib.localvqe_external_bridge_version.restype = ctypes.c_int
            if self.lib.localvqe_external_bridge_version() != 1:
                raise ValueError("unsupported experiment bridge")
            fn = self.lib.localvqe_capture_daf_frame_f32
            fn.argtypes, fn.restype = [u, p, p, ctypes.c_int, p, p], ctypes.c_int
        options = self.lib.localvqe_options_new()
        if not options:
            raise RuntimeError("options allocation failed")
        try:
            for key, value in (("model_path", str(MODEL).encode()), ("backend", b"CPU")):
                if getattr(self.lib, f"localvqe_options_set_{key}")(options, value):
                    raise ValueError(f"LocalVQE {key} rejected")
            if self.lib.localvqe_options_set_threads(options, 4):
                raise ValueError("CPU threads rejected")
            self.ctx = self.lib.localvqe_new_with_options(options)
        finally:
            self.lib.localvqe_options_free(options)
        if not self.ctx:
            raise RuntimeError("pinned model did not load")
        self.lib.localvqe_set_noise_gate(self.ctx, 0, -90.0)

    def close(self):
        if self.ctx:
            self.lib.localvqe_free(self.ctx)
            self.ctx = 0

    @staticmethod
    def pad(samples: np.ndarray, *, flush: int = CODEC_DELAY) -> np.ndarray:
        count = math.ceil((len(samples) + flush) / HOP) * HOP
        return np.pad(np.ascontiguousarray(samples, dtype=np.float32), (0, count - len(samples)))

    def run(self, first, second, *, mode="full") -> tuple[np.ndarray, np.ndarray, dict]:
        if len(first) != len(second) or not len(first):
            raise ValueError("two inputs must share the complete timeline")
        a, b = self.pad(first), self.pad(second)
        raw = np.zeros_like(a)
        self.lib.localvqe_reset(self.ctx)
        call = self.calls[{"full": "localvqe_process_frame_f32", "mask": "localvqe_external_mask_frame_f32",
                           "codec": "localvqe_external_codec_frame_f32"}[mode]]
        ptr = ctypes.POINTER(ctypes.c_float)
        times = []
        for start in range(0, len(a), HOP):
            pa, pb, po = [v[start:start + HOP].ctypes.data_as(ptr) for v in (a, b, raw)]
            started = time.perf_counter()
            rc = call(self.ctx, pa, HOP, po) if mode == "codec" else call(self.ctx, pa, pb, HOP, po)
            times.append((time.perf_counter() - started) * 1000)
            if rc:
                raise RuntimeError(f"{mode} returned {rc}: {self.lib.localvqe_last_error(self.ctx)}")
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite model output")
        output = np.ascontiguousarray(raw[CODEC_DELAY:CODEC_DELAY + len(first)])
        if len(output) != len(first):
            raise ValueError("flush did not preserve the final input sample")
        return output, raw, {"process_ms": sum(times), "process_rtf": sum(times) / (len(first) / RATE * 1000),
              "hop_p50_ms": float(np.percentile(times, 50)), "hop_p99_ms": float(np.percentile(times, 99)),
              "hop_max_ms": max(times), "hops_over_16ms": sum(t > 16 for t in times),
              "processed_hops": len(times), "padding_samples": len(a) - len(first),
              "codec_delay_samples": CODEC_DELAY, "reset_sample_offsets": [0]}

    def capture_daf(self, first, second):
        a, b = self.pad(first), self.pad(second)
        e, yhat = np.zeros_like(a), np.zeros_like(a)
        self.lib.localvqe_reset(self.ctx)
        ptr = ctypes.POINTER(ctypes.c_float)
        call = self.lib.localvqe_capture_daf_frame_f32
        for start in range(0, len(a), HOP):
            pa, pb, pe, py = [v[start:start + HOP].ctypes.data_as(ptr) for v in (a, b, e, yhat)]
            rc = call(self.ctx, pa, pb, HOP, pe, py)
            if rc:
                raise RuntimeError(f"DAF capture failed: {rc}")
        return e, yhat


def activity_mask(samples: np.ndarray) -> np.ndarray:
    active = np.zeros(len(samples), dtype=bool)
    for start in range(0, len(samples) - 320 + 1, 160):
        if np.mean(samples[start:start + 320].astype(np.float64)**2) > 1e-7:
            active[start:start + 320] = True
    return active


def power(samples: np.ndarray) -> float:
    return float(np.mean(np.asarray(samples, dtype=np.float64)**2)) if len(samples) else 0.0


def ratio_db(numerator: float, denominator: float) -> float | None:
    return float(10 * np.log10(numerator / denominator)) if min(numerator, denominator) > 1e-20 else None


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float | None:
    e, t = np.asarray(estimate, dtype=np.float64), np.asarray(target, dtype=np.float64)
    tt, ee = float(np.sum(t*t)), float(np.sum(e*e))
    if min(tt, ee) < 1e-18:
        return None
    projection = float(np.sum(e*t)) / tt * t
    pp, error = float(np.sum(projection**2)), float(np.sum((e - projection)**2))
    return float(10 * np.log10(max(pp, 1e-30) / max(error, 1e-30)))


def measure(output, mic, clean, selected, *, near_present: bool) -> dict:
    if len(output) != len(mic) or len(output) != len(clean) or not np.any(selected):
        raise ValueError("invalid common measurement window")
    y, m, t = [np.asarray(v[selected], dtype=np.float64) for v in (output, mic, clean)]
    result = {"samples": len(y), "output_rms": math.sqrt(power(y)),
              "output_attenuation_db": ratio_db(power(m), power(y)) if not near_present else None,
              "near_si_sdr_db": si_sdr(y, t) if near_present else None,
              "near_total_error_rms": math.sqrt(power(y - t)) if near_present else None,
              "near_target_gain_db": None, "near_retention_probe": None}
    if near_present:
        target_energy = float(np.sum(t*t))
        alpha = float(np.sum(y*t)) / target_energy if target_energy > 1e-18 else 0.0
        gain = float(20 * np.log10(abs(alpha))) if abs(alpha) > 1e-15 else None
        result["near_target_gain_db"] = gain
        result["near_retention_probe"] = gain is not None and gain >= -6.0
    return result


def reference_codec(samples: np.ndarray) -> np.ndarray:
    """Decode the two pinned F32 GGUF matrices independently of the C++ API."""
    scalar_types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                    10: "Q", 11: "q", 12: "d"}
    with MODEL.open("rb") as stream:
        def read(fmt):
            data = stream.read(struct.calcsize("<" + fmt))
            return struct.unpack("<" + fmt, data)[0]

        def text():
            length = read("Q")
            if length > MODEL.stat().st_size:
                raise ValueError("invalid pinned GGUF string length")
            return stream.read(length).decode("utf-8")

        def value(kind):
            if kind in scalar_types:
                return read(scalar_types[kind])
            if kind == 8:
                return text()
            if kind == 9:
                element_type, count = read("I"), read("Q")
                return [value(element_type) for _ in range(count)]
            raise ValueError(f"unknown GGUF metadata type: {kind}")

        if stream.read(4) != b"GGUF" or read("I") != 3:
            raise ValueError("pinned model is not GGUF v3")
        tensor_count, metadata_count = read("Q"), read("Q")
        metadata = {}
        for _ in range(metadata_count):
            key, kind = text(), read("I")
            metadata[key] = value(kind)
        tensors = {}
        for _ in range(tensor_count):
            name, rank = text(), read("I")
            shape = tuple(read("Q") for _ in range(rank))
            kind, offset = read("I"), read("Q")
            tensors[name] = (shape, kind, offset)
        alignment = int(metadata.get("general.alignment", 32))
        data_start = math.ceil(stream.tell() / alignment) * alignment
        matrices = []
        for name in ("encoder.conv.weight", "decoder.linear.weight"):
            shape, kind, offset = tensors[name]
            if kind != 0 or math.prod(shape) != 512 * 512:
                raise ValueError("independent codec needs two F32 512x512 matrices")
            stream.seek(data_start + offset)
            matrices.append(np.frombuffer(stream.read(512 * 512 * 4), dtype="<f4").reshape(512, 512))
    padded = Engine.pad(samples)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(padded, (HOP, 0)), 512)[::HOP]
    encoded = windows @ matrices[0].T
    decoded = encoded @ matrices[1].T
    scale = 1.0 if int(metadata.get("localvqe.version", 1)) >= 2 else 0.5
    output = np.zeros_like(padded)
    for index, frame in enumerate(decoded):
        start = index * HOP
        output[start:start + HOP] += frame[:HOP] * scale
        if start + HOP < len(output):
            output[start + HOP:start + 2 * HOP] += frame[HOP:] * scale
    return output


def run_controls(engine: Engine, original: Engine, arrays: dict, work: Path) -> dict:
    microphone, playback = arrays["mic"], arrays["playback"]
    full, raw_full, _ = engine.run(microphone, playback)
    captured_e, captured_y = engine.capture_daf(microphone, playback)
    _, raw_replayed, _ = engine.run(captured_e, captured_y, mode="mask")
    # Do not compare the replay's extra appended flush hop; both primary
    # routes must be identical over the entire original raw output.
    replay_error = float(np.max(np.abs(raw_full - raw_replayed[:len(raw_full)])))
    original_out, original_raw, _ = original.run(microphone, playback)
    original_error = float(np.max(np.abs(original_raw - raw_full)))
    n = 8193  # Deliberately not divisible by 256: exercise partial-hop flush.
    rng = np.random.default_rng(20261112)
    probe = rng.normal(0, .03, n).astype("float32")
    identity, identity_raw, _ = engine.run(probe, np.zeros_like(probe), mode="codec")
    identity_error = float(np.max(np.abs(identity - probe)))
    # Published codec weights are not a perfect identity basis. Check the
    # bridge against an independent NumPy application of those actual GGUF
    # matrices, rather than weakening a mathematically false unity gate.
    reference_raw = reference_codec(probe)
    reference_error = float(np.max(np.abs(identity_raw - reference_raw)))
    raw_corr = signal.correlate(identity_raw, probe, mode="full", method="fft")
    lag = int(signal.correlation_lags(len(identity_raw), len(probe))[int(np.argmax(np.abs(raw_corr)))])
    silent, _, _ = engine.run(np.zeros(n, "float32"), np.zeros(n, "float32"), mode="mask")
    target = arrays["clean"]
    near_mask = activity_mask(target)
    # If this particular far-only case has no near speech, use the independent
    # deterministic codec probe for the gain negative control.
    if not np.any(near_mask):
        target, near_mask = probe, np.ones(n, dtype=bool)
    muted = measure(np.zeros_like(target), target, target, near_mask, near_present=True)
    bypass = measure(probe, probe, np.zeros_like(probe), np.ones(n, bool), near_present=False)
    controls = {"daf_replay_max_abs_error": replay_error, "original_full_max_abs_error": original_error,
                "codec_vs_input_max_abs_error": identity_error,
                "codec_reference_max_abs_error": reference_error, "codec_raw_lag_samples": lag,
                "partial_hop_output_samples": len(identity), "partial_hop_input_samples": n,
                "zero_mask_output_max_abs": float(np.max(np.abs(silent))),
                "muted_near_rejected": muted["near_retention_probe"] is False,
                "bypass_far_rejected": bypass["output_attenuation_db"] == 0}
    controls["passed"] = (replay_error < 1e-6 and original_error < 1e-6 and reference_error < 2e-6
                          and lag == CODEC_DELAY and len(identity) == n and np.max(np.abs(silent)) == 0
                          and controls["muted_near_rejected"] and controls["bypass_far_rejected"])
    write_audio(work / "control-daf-e.wav", captured_e)
    write_audio(work / "control-daf-yhat.wav", captured_y)
    write_audio(work / "control-codec-probe.wav", probe)
    write_audio(work / "control-codec-output.wav", identity)
    write_json(work / "controls.json", controls)
    if not controls["passed"]:
        raise ValueError(f"mask bridge controls failed: {controls}")
    return controls


def run_case(engine: Engine, original: Engine, case, group, output: Path, *, controls=False) -> dict:
    resource_check()
    work = output / group / case["id"]
    work.mkdir(parents=True, exist_ok=False)
    arrays, scene = prepare_case(case, group, work)
    mdf, mdf_record = mdf_inputs(arrays, work)
    near_mask, far_mask = activity_mask(arrays["clean"]), activity_mask(arrays["playback"])
    selected = near_mask & far_mask if group == "double_talk" else near_mask if group == "near_single" else far_mask
    designed = np.ones(len(selected), dtype=bool)
    if group == "double_talk":
        designed.fill(False)
        for segment in case["render"]["activity_segments"]:
            if segment["label"] == "double_talk":
                designed[round(segment["start_s"] * RATE):round(segment["end_s"] * RATE)] = True
        selected &= designed
    coverage = float(np.sum(selected) / max(1, np.sum(designed)))
    valid_stimulus = np.sum(selected) >= RATE // 20 and (group != "double_talk" or coverage >= .05)
    row = {"case_id": case["id"], "category": group, "scene": scene, "mdf": mdf_record,
           "sample_rate": RATE, "samples": len(arrays["mic"]), "routes": {},
           "measurement_status": "measured" if valid_stimulus else "invalid_stimulus",
           "measurement_reason": None if valid_stimulus else "活动门限下实际同帧近端/远端覆盖不足；不计入有效双讲均值",
           "activity": {"near_s": float(np.sum(near_mask) / RATE), "far_s": float(np.sum(far_mask) / RATE),
                        "joint_s": float(np.sum(near_mask & far_mask) / RATE),
                        "measurement_s": float(np.sum(selected) / RATE),
                        "designed_measurement_s": float(np.sum(designed) / RATE),
                        "design_window_coverage": coverage}}
    if controls:
        row["controls"] = run_controls(engine, original, arrays, work)
    route_outputs = {"input_mic": (arrays["mic"], None), "mdf_residual_only": (mdf["e"], None)}
    for route, a, b, mode in (("localvqe_full", arrays["mic"], arrays["playback"], "full"),
                             ("mdf_residual_mask", mdf["e"], mdf["yhat"], "mask"),
                             ("mdf_residual_wrong_ref", mdf["e"], arrays["playback"], "mask")):
        y, raw, timing = engine.run(a, b, mode=mode)
        write_audio(work / f"{route}.raw.wav", raw)
        route_outputs[route] = y, timing
    for route, (y, timing) in route_outputs.items():
        artifact = write_audio(work / f"{route}.wav", y)
        metrics = (measure(y, arrays["mic"], arrays["clean"], selected,
                           near_present=group in {"near_single", "double_talk"})
                   if valid_stimulus else {"samples": 0, "output_rms": None,
                       "output_attenuation_db": None, "near_si_sdr_db": None,
                       "near_total_error_rms": None, "near_target_gain_db": None,
                       "near_retention_probe": None})
        if group == "path_change":
            windows = {}
            for label, lo, hi in (("before_switch", 3.0, 5.5), ("after_switch_0_1s", 6.0, 7.0),
                                  ("after_switch_2_6s", 8.0, 12.0)):
                mask = selected & (np.arange(len(y)) >= round(lo * RATE)) & (np.arange(len(y)) < round(hi * RATE))
                windows[label] = {"start_s": lo, "end_s": hi,
                    **measure(y, arrays["mic"], arrays["clean"], mask, near_present=False)}
            metrics["path_windows"] = windows
        row["routes"][route] = {"artifact": artifact, "metrics": metrics, "timing": timing,
                                  "input_domains": ["MDF residual", "quantized capture - MDF residual"] if route == "mdf_residual_mask" else None}
    write_json(work / "case.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--cases", default=5, type=int, help="cases per near/far/double/path group")
    args = parser.parse_args()
    if not 1 <= args.cases <= 20:
        parser.error("--cases must be 1..20")
    resource_check()
    if os.environ.get("LOCALVQE_ALLOW_UNHASHED"):
        raise RuntimeError("model hash bypass is forbidden")
    if digest(CLI) != CLI_SHA or digest(MODEL) != MODEL_SHA:
        raise RuntimeError("pinned binary/model identity mismatch")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a new or empty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    identity = {"cli": {"path": str(CLI), "sha256": digest(CLI)},
                "library": {"path": str(library), "sha256": digest(library)},
                "original_library": {"path": str(ORIGINAL_LIBRARY), "sha256": digest(ORIGINAL_LIBRARY)},
                "model": {"path": str(MODEL), "sha256": digest(MODEL)},
                "runner": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))}}
    build_manifest = library.parent / "manifest.json"
    if not build_manifest.is_file():
        raise ValueError("reproducible bridge build manifest missing")
    identity["bridge_build"] = {"path": str(build_manifest), "sha256": digest(build_manifest)}
    write_json(out / "identity.json", identity)
    engine, original = Engine(library), Engine(ORIGINAL_LIBRARY, original=True)
    rows = []
    try:
        for group in GROUPS:
            for index, case in enumerate(source_cases(group, args.cases)):
                row = run_case(engine, original, case, group, out, controls=index == 0)
                rows.append(row)
                print(json.dumps({"finished": len(rows), "total": len(GROUPS) * args.cases,
                                  "group": group, "case": case["id"]}), flush=True)
    finally:
        engine.close()
        original.close()
    for record in identity.values():
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("runtime identity changed during the comparison")
    result = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "identity": identity,
              "sample_rate": RATE, "duration_s": 12, "case_count": len(rows), "cases": rows,
              "alignment": {"fixed_codec_delay_samples": CODEC_DELAY, "fitted_delay_for_scoring": False,
                            "flush_output": True, "state_reset_at_path_switch": False},
              "controls_passed": all(r["controls"]["passed"] for r in rows if "controls" in r),
              "boundary": "合成工程 A/B；完整 12 秒，先运行 VCP RX 再按 FIR 重建麦克风；保留现有 VCP 500 ms MDF 原始残差。未经重新训练的外部 MDF 输入适配实验。正式 ITU-T 合规未测。",
              "formal_compliance": False}
    write_json(out / "result.json", result)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    write_json(out / "artifacts.json", {"schema": "itu-ab-artifacts-v1", "files": {
        str(path.relative_to(out)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in files}})
    print(json.dumps({"cases": len(rows), "result": str(out / "result.json"), "sha256": digest(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
