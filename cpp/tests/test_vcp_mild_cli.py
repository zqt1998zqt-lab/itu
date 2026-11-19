#!/usr/bin/env python3
"""Structural CLI acceptance for the opt-in vcp_mild profile.

Usage: python3 test_vcp_mild_cli.py <new-cli> <old-baseline-cli> <output-dir>

This checks configuration, metadata, PCM compatibility, stage extents/rates,
latency cropping and frame-level gain bounds. It makes no MOS or quality claim;
an untriggered bounded-gain path is reported as UNPROVEN for activation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import wave

import numpy as np


BOUNDED_EXPECTED = {
    "bounded_speech_gain": True,
    "bounded_gain_max_db": 3.0,
    "bounded_gain_rise_db_per_second": 3.0,
    "bounded_gain_fall_db_per_second": 6.0,
    "bounded_gain_hold_ms": 120,
}

AEC_NS_EXPECTED = {
    "ns_backend": "webrtc",
    "ns_level": 0,
    "omlsa_noise_update_scale": 1.0,
    "omlsa_speech_recovery": 0.0,
    "omlsa_release_ms": 60.0,
    "omlsa_speech_floor_db": -40.0,
    "vehicle_ns_control": False,
}

NO_AEC_NS_EXPECTED = {
    "ns_backend": "omlsa_imcra",
    "ns_level": 0,
    "omlsa_noise_update_scale": 0.5,
    "omlsa_speech_recovery": 0.7,
    "omlsa_release_ms": 120.0,
    "omlsa_speech_floor_db": -40.0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_pcm16(path: Path, rate: int, samples: np.ndarray) -> None:
    pcm = np.clip(np.rint(np.asarray(samples, dtype=np.float64)), -32768, 32767)
    pcm = pcm.astype("<i2")
    require(pcm.ndim in (1, 2), f"invalid fixture shape: {pcm.shape}")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes(order="C"))


def read_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        require(stream.getsampwidth() == 2, f"expected PCM16: {path}")
        rate = stream.getframerate()
        channels = stream.getnchannels()
        frames = stream.getnframes()
        payload = stream.readframes(frames)
    pcm = np.frombuffer(payload, dtype="<i2").reshape(frames, channels).copy()
    return rate, pcm


def wav_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    blob = path.read_bytes()
    require(len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WAVE",
            f"not a RIFF WAVE: {path}")
    end = min(len(blob), 8 + struct.unpack_from("<I", blob, 4)[0])
    chunks: list[tuple[bytes, bytes]] = []
    offset = 12
    while offset + 8 <= end:
        tag = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        require(start + size <= end, f"truncated WAV chunk: {path}")
        chunks.append((tag, blob[start:start + size]))
        offset = start + size + (size & 1)
    return chunks


def wav_data(path: Path) -> bytes:
    data = [payload for tag, payload in wav_chunks(path) if tag == b"data"]
    require(len(data) == 1, f"expected one data chunk: {path}")
    return data[0]


def read_wav_array(path: Path) -> tuple[np.ndarray, int, int, int, int]:
    chunks = wav_chunks(path)
    formats = [payload for tag, payload in chunks if tag == b"fmt "]
    require(len(formats) == 1 and len(formats[0]) >= 16,
            f"invalid fmt chunk: {path}")
    tag, channels, rate, _byte_rate, _block, bits = struct.unpack_from(
        "<HHIIHH", formats[0]
    )
    payload = wav_data(path)
    if tag == 3 and bits == 32:
        audio = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise AssertionError(f"unsupported WAV encoding {tag}/{bits}: {path}")
    require(channels > 0 and audio.size % channels == 0,
            f"channel-alignment failure: {path}")
    return audio.reshape(-1, channels), rate, channels, tag, bits


def run_cli(cli: Path, args: list[str], prefix: Path,
            expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli), *args, "--out", str(prefix)]
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=300, check=False)
    log_path = Path(str(prefix) + ".log")
    log_path.write_text(result.stdout + result.stderr)
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"CLI failed ({result.returncode}): {' '.join(command)}\n"
            f"{(result.stdout + result.stderr)[-12000:]}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            f"invalid configuration was accepted: {' '.join(command)}\n"
            f"{(result.stdout + result.stderr)[-12000:]}"
        )
    return result


def load_json(prefix: Path) -> dict:
    path = Path(str(prefix) + ".json")
    require(path.is_file(), f"missing JSON report: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"report is not an object: {path}")
    return value


def assert_number(mapping: dict, key: str, expected: float,
                  tolerance: float = 1e-7) -> None:
    require(key in mapping, f"missing metadata field {key}")
    value = mapping[key]
    require(type(value) in (int, float) and math.isfinite(value),
            f"non-finite/non-numeric {key}: {value!r}")
    require(math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance),
            f"metadata {key}={value!r}, expected {expected!r}")


def assert_mild_metadata(report: dict, processing_profile: str,
                         expected_route: str) -> None:
    require(report.get("processing_profile") == processing_profile,
            f"wrong processing_profile: {report.get('processing_profile')!r}")
    aec_enabled = report.get("aec_enabled")
    require(type(aec_enabled) is bool, f"invalid aec_enabled: {aec_enabled!r}")
    expected = {
        **BOUNDED_EXPECTED,
        **(AEC_NS_EXPECTED if aec_enabled else NO_AEC_NS_EXPECTED),
        "mild_near_only_ns": True,
        "mild_ns_route": expected_route,
    }
    for key, value in expected.items():
        if isinstance(value, bool):
            require(report.get(key) is value, f"metadata {key} != {value}")
        elif isinstance(value, str):
            require(report.get(key) == value,
                    f"metadata {key}={report.get(key)!r}, expected {value!r}")
        elif isinstance(value, (int, float)):
            assert_number(report, key, value)
        else:
            raise AssertionError(f"unsupported expected metadata type for {key}")
    require(report.get("ns_enabled") is True, "noise suppression is not enabled")
    require(report.get("agc2_enabled") is True, "AGC2 path is not enabled")
    require(type(report.get("dsp_delay_samples")) is int
            and report["dsp_delay_samples"] >= 0,
            "missing/invalid dsp_delay_samples")


def make_fixture(case: Path, rate: int) -> tuple[Path, Path, int]:
    n = 2 * rate + 37
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(0xC0DE + rate)
    noise = rng.standard_normal(n)
    mic = 0.075 * np.sin(2 * np.pi * 211.0 * t + 0.17)
    mic += 0.018 * noise
    mic[rate // 3 + 3] += 0.32
    mic[n - rate // 20] += 0.38  # keeps a real tail after latency crop
    far = 0.045 * np.sin(2 * np.pi * 431.0 * t + 0.61)
    far += 0.012 * rng.standard_normal(n)
    far = np.where(t < 0.62, far, 0.0)
    mic += 0.025 * far
    mic = np.clip(mic, -0.9, 0.9) * 32768.0
    far = np.clip(far, -0.9, 0.9) * 32768.0
    case.mkdir(parents=True, exist_ok=True)
    mic_path = case / "mic.wav"
    rx_path = case / "rx.wav"
    write_pcm16(mic_path, rate, mic)
    write_pcm16(rx_path, rate, far)
    return mic_path, rx_path, n


def default_compatibility(new_cli: Path, old_cli: Path, fixture: Path,
                          root: Path) -> dict:
    mic, rx, _ = make_fixture(fixture, 16000)
    common = ["--profile", "vcp", "--mic", str(mic), "--rx-in", str(rx)]
    new_prefix = root / "default-new"
    old_prefix = root / "default-old"
    run_cli(new_cli, common, new_prefix)
    run_cli(old_cli, common, old_prefix)
    new_data = wav_data(Path(str(new_prefix) + ".after.wav"))
    old_data = wav_data(Path(str(old_prefix) + ".after.wav"))
    require(new_data == old_data, "vcp no-flag final PCM is not byte-exact vs baseline")
    new_report = load_json(new_prefix)
    old_report = load_json(old_prefix)
    require(new_report.get("processing_profile") == "vcp", "new CLI profile is not vcp")
    require(new_report.get("bounded_speech_gain") is False,
            "vcp no-flag run unexpectedly enabled bounded gain")
    require(new_report.get("mild_near_only_ns") is False,
            "vcp no-flag run unexpectedly enabled mild near-only NS")
    require(new_report.get("mild_ns_route") == "explicit_backend",
            "vcp no-flag run did not report the explicit/default backend route")
    return {
        "pcm_sha256": sha256(new_data),
        "new_dsp_delay": new_report.get("dsp_delay_samples"),
        "old_dsp_delay": old_report.get("dsp_delay_samples"),
    }


def profile_equivalence(new_cli: Path, fixture: Path, root: Path) -> dict:
    mic, rx, _ = make_fixture(fixture, 16000)
    base = ["--mic", str(mic), "--rx-in", str(rx)]
    profile_prefix = root / "profile-vcp-mild"
    explicit_prefix = root / "profile-explicit"
    run_cli(new_cli, ["--profile", "vcp_mild", *base], profile_prefix)
    run_cli(new_cli, [
        "--profile", "vcp", *base,
        "--mild-near-only-ns",
        "--ns-backend", "omlsa_imcra",
        "--ns-level", "0",
        "--omlsa-noise-update-scale", "0.5",
        "--omlsa-speech-recovery", "0.7",
        "--omlsa-release-ms", "120",
        "--bounded-speech-gain",
    ], explicit_prefix)
    profile_data = wav_data(Path(str(profile_prefix) + ".after.wav"))
    explicit_data = wav_data(Path(str(explicit_prefix) + ".after.wav"))
    require(profile_data == explicit_data,
            "vcp_mild and explicit equivalent flags differ in final PCM bytes")
    profile_report = load_json(profile_prefix)
    explicit_report = load_json(explicit_prefix)
    assert_mild_metadata(profile_report, "vcp_mild", "established_aec_ns")
    assert_mild_metadata(explicit_report, "vcp", "established_aec_ns")
    require(explicit_report["dsp_delay_samples"] == profile_report["dsp_delay_samples"],
            "explicit profile has a different DSP delay")
    return {
        "pcm_sha256": sha256(profile_data),
        "dsp_delay": profile_report["dsp_delay_samples"],
    }


def rollback_report(new_cli: Path, fixture: Path, root: Path) -> dict:
    mic, rx, _ = make_fixture(fixture, 16000)
    prefix = root / "rollback-no-bounded"
    run_cli(new_cli, [
        "--profile", "vcp_mild", "--mic", str(mic), "--rx-in", str(rx),
        "--no-bounded-speech-gain",
    ], prefix)
    report = load_json(prefix)
    require(report.get("bounded_speech_gain") is False,
            "--no-bounded-speech-gain did not disable bounded gain")
    require(report.get("aec_enabled") is True, "rollback lost established AEC")
    require(report.get("mild_near_only_ns") is True,
            "rollback lost mild_near_only_ns")
    require(report.get("mild_ns_route") == "established_aec_ns",
            "rollback used the wrong mild NS route")
    require(report.get("ns_backend") == "webrtc",
            "AEC rollback did not select the established WebRTC NS")
    require(report.get("vehicle_ns_control") is False,
            "AEC rollback left vehicle NS control enabled")
    require(report.get("agc2_enabled") is True,
            "rollback did not retain the raw AGC2 path")
    require(report.get("agc2_speech_gain_policy") is True,
            "rollback lost the speech AGC policy")
    assert_number(report, "ns_level", 0)
    assert_number(report, "omlsa_noise_update_scale", 1.0)
    assert_number(report, "omlsa_speech_recovery", 0.0)
    assert_number(report, "omlsa_release_ms", 60.0)
    assert_number(report, "omlsa_speech_floor_db", -40.0)
    return {
        "bounded_speech_gain": report["bounded_speech_gain"],
        "mild_ns_route": report["mild_ns_route"],
        "agc2_enabled": report["agc2_enabled"],
        "agc2_adaptive_enabled": report.get("agc2_adaptive_enabled"),
        "agc2_speech_gain_policy": report["agc2_speech_gain_policy"],
    }


def no_aec_profile_equivalence(new_cli: Path, fixture: Path, root: Path) -> dict:
    mic, _rx, _ = make_fixture(fixture, 16000)
    base = ["--no-aec", "--mic", str(mic)]
    profile_prefix = root / "no-aec-profile-vcp-mild"
    explicit_prefix = root / "no-aec-profile-explicit"
    run_cli(new_cli, ["--profile", "vcp_mild", *base], profile_prefix)
    run_cli(new_cli, [
        "--profile", "vcp", *base,
        "--mild-near-only-ns",
        "--ns-backend", "omlsa_imcra",
        "--ns-level", "0",
        "--omlsa-noise-update-scale", "0.5",
        "--omlsa-speech-recovery", "0.7",
        "--omlsa-release-ms", "120",
        "--bounded-speech-gain",
    ], explicit_prefix)
    profile_data = wav_data(Path(str(profile_prefix) + ".after.wav"))
    explicit_data = wav_data(Path(str(explicit_prefix) + ".after.wav"))
    require(profile_data == explicit_data,
            "no-AEC vcp_mild and explicit OM flags differ in final PCM bytes")
    profile_report = load_json(profile_prefix)
    explicit_report = load_json(explicit_prefix)
    assert_mild_metadata(profile_report, "vcp_mild", "optimized_near_only_ns")
    assert_mild_metadata(explicit_report, "vcp", "optimized_near_only_ns")
    require(profile_report["dsp_delay_samples"] == 160,
            f"16 kHz no-AEC vcp_mild DSP delay is "
            f"{profile_report['dsp_delay_samples']}, expected 160")
    require(explicit_report["dsp_delay_samples"] == 160,
            "explicit no-AEC OM path does not report a 160-sample DSP delay")
    return {
        "pcm_sha256": sha256(profile_data),
        "dsp_delay": profile_report["dsp_delay_samples"],
        "profile_route": profile_report["mild_ns_route"],
        "explicit_route": explicit_report["mild_ns_route"],
    }


def invalid_configurations(new_cli: Path, fixture: Path, root: Path) -> list[str]:
    mic, rx, _ = make_fixture(fixture, 16000)
    common = ["--profile", "vcp_mild", "--mic", str(mic), "--rx-in", str(rx)]
    cases = {
        "bounded_nan": ["--bounded-gain-max-db", "nan"],
        "bounded_max_gt_18": ["--bounded-gain-max-db", "18.1"],
        "bounded_rise_zero": ["--bounded-gain-rise-db-per-second", "0"],
        "bounded_hold_301": ["--bounded-gain-hold-ms", "301"],
        "speech_floor_gt_0": ["--omlsa-speech-floor-db", "0.1"],
        "speech_floor_lt_minus_40": ["--omlsa-speech-floor-db", "-40.1"],
        "webrtc_with_tuned_floor": [
            "--ns-backend", "webrtc", "--omlsa-speech-floor-db", "-3",
        ],
    }
    for name, extra in cases.items():
        run_cli(new_cli, [*common, *extra], root / f"invalid-{name}",
                expect_success=False)
    return sorted(cases)


def feature_edges(new_cli: Path, fixture: Path, root: Path) -> dict:
    mic, rx, _ = make_fixture(fixture, 16000)
    base = ["--profile", "vcp_mild", "--mic", str(mic), "--rx-in", str(rx)]
    results = {}
    for name, flags, field, expected in (
        ("hold-zero", ["--bounded-gain-hold-ms", "0"], "bounded_gain_hold_ms", 0),
        ("max-zero", ["--bounded-gain-max-db", "0"], "bounded_gain_max_db", 0.0),
    ):
        prefix = root / f"edge-{name}"
        run_cli(new_cli, [*base, *flags], prefix)
        report = load_json(prefix)
        require(report.get("bounded_speech_gain") is True,
                f"{name} unexpectedly disabled bounded gain")
        assert_number(report, field, expected)
        rate, pcm = read_pcm16(Path(str(prefix) + ".after.wav"))
        require(rate == 16000 and pcm.size > 0 and np.any(pcm != 0),
                f"{name} produced empty/silent output")
        results[name] = report[field]
    return results


def stage_checks(prefix: Path, rate: int, n: int, report: dict) -> dict:
    after_path = Path(str(prefix) + ".after.wav")
    after_rate, after = read_pcm16(after_path)
    delay = report["dsp_delay_samples"]
    require(after_rate == rate, f"{rate} Hz output rate is {after_rate}")
    require(after.shape == (n + delay, 1),
            f"{rate} Hz after shape {after.shape}, expected {(n + delay, 1)}")
    after_float = after[:, 0].astype(np.float32) / np.float32(32768.0)
    require(np.isfinite(after_float).all(), f"{rate} Hz output is non-finite")
    require(np.any(after != 0), f"{rate} Hz output is silent")
    cropped = after[delay:delay + n, 0]
    require(cropped.size == n, f"{rate} Hz latency crop lost input samples")
    require(np.any(cropped != 0), f"{rate} Hz latency crop is silent")
    require(np.any(cropped[-max(1, rate // 20):] != 0),
            f"{rate} Hz latency crop lost the final signal tail")

    stages_path = Path(str(prefix) + ".stages.json")
    require(stages_path.is_file(), f"missing stage metadata: {stages_path}")
    metadata = json.loads(stages_path.read_text())
    require(metadata.get("input_sample_rate") == rate,
            f"{rate} Hz stage input_sample_rate mismatch")
    require(metadata.get("input_samples_per_channel") == n,
            f"{rate} Hz stage input length mismatch")
    rows = metadata.get("pipeline_stages")
    require(isinstance(rows, list) and rows,
            f"{rate} Hz has no pipeline stages")
    by_id = {}
    for row in rows:
        require(isinstance(row, dict), f"{rate} Hz stage row is not an object")
        stage_id = row.get("id")
        require(isinstance(stage_id, str) and stage_id and stage_id not in by_id,
                f"{rate} Hz invalid/duplicate stage id {stage_id!r}")
        for key in ("sample_rate", "channels", "samples_per_channel",
                    "fixed_delay_samples"):
            require(type(row.get(key)) is int and row[key] >= 0,
                    f"{stage_id} invalid {key}: {row.get(key)!r}")
        require(row["sample_rate"] > 0 and row["channels"] > 0,
                f"{stage_id} has zero shape")
        stage_path = Path(row.get("file", ""))
        require(stage_path.is_file(), f"missing stage WAV for {stage_id}: {stage_path}")
        audio, actual_rate, actual_channels, _tag, _bits = read_wav_array(stage_path)
        require(actual_rate == row["sample_rate"],
                f"{stage_id} WAV rate {actual_rate} != metadata {row['sample_rate']}")
        require(actual_channels == row["channels"],
                f"{stage_id} WAV channels {actual_channels} != {row['channels']}")
        expected_extent = (n * actual_rate) // rate + row["fixed_delay_samples"]
        require(audio.shape == (expected_extent, actual_channels),
                f"{stage_id} extent {audio.shape} != "
                f"{(expected_extent, actual_channels)} at {rate} Hz")
        require(np.isfinite(audio).all(), f"{stage_id} contains non-finite samples")
        by_id[stage_id] = row
    require("tx_output" in by_id, f"{rate} Hz stage metadata lacks tx_output")
    output_stage = by_id["tx_output"]
    require(output_stage["fixed_delay_samples"] == delay,
            f"{rate} Hz tx_output delay differs from report DSP delay")
    require(output_stage["samples_per_channel"] == after.shape[0],
            f"{rate} Hz tx_output extent differs from final output")
    return {"dsp_delay": delay, "stages": len(rows)}


def frame_gain_checks(prefix: Path) -> dict:
    path = Path(str(prefix) + ".frames.csv")
    require(path.is_file(), f"missing frame CSV: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        for field in ("far_active", "applied_gain_db"):
            require(field in reader.fieldnames, f"CSV lacks {field}: {path}")
        rows = list(reader)
    require(rows, f"empty frame CSV: {path}")
    previous = None
    nonfar = 0
    max_gain = -math.inf
    activation = False
    for index, row in enumerate(rows):
        far = float(row["far_active"])
        gain = float(row["applied_gain_db"])
        require(math.isfinite(gain), f"non-finite applied gain at frame {index}")
        require(-1e-6 <= gain <= BOUNDED_EXPECTED["bounded_gain_max_db"] + 1e-4,
                f"bounded gain out of range at frame {index}: {gain}")
        max_gain = max(max_gain, gain)
        if not far:
            nonfar += 1
            if previous is not None:
                delta = gain - previous
                require(delta <= BOUNDED_EXPECTED["bounded_gain_rise_db_per_second"] * 0.01 + 1e-4,
                        f"rise step exceeded at frame {index}: {delta}")
                require(-delta <= BOUNDED_EXPECTED["bounded_gain_fall_db_per_second"] * 0.01 + 1e-4,
                        f"fall step exceeded at frame {index}: {delta}")
            activation = activation or gain > 1e-4
        previous = gain
    require(nonfar > 0, "fixture produced no ordinary non-far frames")
    return {
        "frames": len(rows),
        "nonfar_frames": nonfar,
        "max_applied_gain_db": max_gain,
        "activation_observed": activation,
    }


def sample_rate_checks(new_cli: Path, root: Path) -> tuple[dict, dict]:
    rates = {}
    diagnostic_prefixes = {}
    for rate in (16000, 24000, 48000):
        case = root / f"rate-{rate}"
        mic, rx, n = make_fixture(case / "fixture", rate)
        prefix = case / "vcp-mild-diag"
        run_cli(new_cli, [
            "--profile", "vcp_mild", "--mic", str(mic), "--rx-in", str(rx),
            "--diagnostic-stages",
        ], prefix)
        report = load_json(prefix)
        require(report.get("sample_rate") == rate,
                f"report sample_rate {report.get('sample_rate')} != {rate}")
        require(report.get("processing_profile") == "vcp_mild",
                f"{rate} Hz lost vcp_mild processing_profile")
        assert_mild_metadata(report, "vcp_mild", "established_aec_ns")
        rates[rate] = stage_checks(prefix, rate, n, report)
        if rate == 16000:
            diagnostic_prefixes[rate] = prefix
    return rates, diagnostic_prefixes


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python3 test_vcp_mild_cli.py <new-cli> "
              "<old-baseline-cli> <output-dir>", file=sys.stderr)
        return 2
    new_cli = Path(sys.argv[1]).resolve()
    old_cli = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    require(new_cli.is_file(), f"new CLI not found: {new_cli}")
    require(old_cli.is_file(), f"baseline CLI not found: {old_cli}")
    root = output_dir / "vcp_mild_cli_acceptance"
    fixture_root = root / "fixtures"
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "default_compatibility": default_compatibility(
            new_cli, old_cli, fixture_root / "default-16000", root
        ),
        "profile_equivalence": profile_equivalence(
            new_cli, fixture_root / "profile-16000", root
        ),
        "no_aec_profile_equivalence": no_aec_profile_equivalence(
            new_cli, fixture_root / "no-aec-profile-16000", root
        ),
        "rollback_report": rollback_report(
            new_cli, fixture_root / "rollback-16000", root
        ),
        "invalid_rejected": invalid_configurations(
            new_cli, fixture_root / "invalid-16000", root
        ),
        "feature_edges": feature_edges(
            new_cli, fixture_root / "edge-16000", root
        ),
    }
    rates, diagnostic = sample_rate_checks(new_cli, root)
    summary["sample_rates"] = rates
    summary["frame_gain"] = frame_gain_checks(diagnostic[16000])
    summary["activation_status"] = (
        "OBSERVED" if summary["frame_gain"]["activation_observed"] else "UNPROVEN"
    )
    if summary["activation_status"] == "UNPROVEN":
        summary["activation_note"] = "fixture did not trigger AGC; H6 activation remains required"
    summary["quality_claim"] = "none; structural CLI checks only"
    summary["status"] = "PASS"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
