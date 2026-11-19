"""Real CLI integration checks for VCP TX/RX diagnostic stage exports.

Usage: python test_vcp_stage_exports.py <itu_audio_cli> <output-directory>

The test intentionally uses the public WAV/CLI contract.  It does not fit a
lag: impulse positions and fixed delays are checked directly, so a diagnostic
file cannot silently shift an upstream tap when downstream EQ/DRC is added.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import wave

import numpy as np


TX_IDS_AEC_OFF = [
    "tx_input",
    "tx_bf",
    "tx_aec",
    "tx_ns_selected",
    "tx_ns",
    "tx_eq",
    "tx_agc",
    "tx_drc",
    "tx_output",
]
TX_IDS_AEC_ON = TX_IDS_AEC_OFF[:2] + ["tx_aec_input"] + TX_IDS_AEC_OFF[2:]
RX_IDS = ["rx_input", "rx_nr", "rx_eq", "rx_drc", "rx_output"]
STAGE_FIELDS = {
    "id",
    "direction",
    "order",
    "file",
    "sample_rate",
    "channels",
    "samples_per_channel",
    "fixed_delay_samples",
    "enabled",
    "measurement_domain",
    "channel_policy",
    "capture_available",
}


@dataclass(frozen=True)
class Stage:
    meta: dict
    audio: np.ndarray
    subtype: str
    format_tag: int

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def delay(self) -> int:
        return self.meta["fixed_delay_samples"]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_pcm16(path: Path, rate: int, samples: np.ndarray) -> None:
    pcm = np.asarray(samples)
    _assert(pcm.ndim == 2 and pcm.shape[0] > 0 and pcm.shape[1] > 0,
            f"invalid PCM16 shape for {path}: {pcm.shape}")
    pcm = np.clip(np.rint(pcm), -32768, 32767).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(int(pcm.shape[1]))
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes(order="C"))


def read_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        _assert(stream.getsampwidth() == 2, f"expected PCM16: {path}")
        channels = stream.getnchannels()
        rate = stream.getframerate()
        frames = stream.getnframes()
        payload = stream.readframes(frames)
    pcm = np.frombuffer(payload, dtype="<i2").reshape(frames, channels).copy()
    return rate, pcm


def _wav_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    blob = path.read_bytes()
    _assert(len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WAVE",
            f"not a little-endian RIFF WAVE: {path}")
    end = min(len(blob), 8 + struct.unpack_from("<I", blob, 4)[0])
    chunks = []
    offset = 12
    while offset + 8 <= end:
        tag = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        _assert(start + size <= end, f"truncated WAV chunk in {path}")
        chunks.append((tag, blob[start:start + size]))
        offset = start + size + (size & 1)
    return chunks


def wav_data(path: Path) -> bytes:
    chunks = [payload for tag, payload in _wav_chunks(path) if tag == b"data"]
    _assert(len(chunks) == 1, f"expected exactly one data chunk in {path}")
    return chunks[0]


def wav_format(path: Path) -> tuple[int, int, int, int]:
    chunks = [payload for tag, payload in _wav_chunks(path) if tag == b"fmt "]
    _assert(len(chunks) == 1 and len(chunks[0]) >= 16,
            f"expected one complete fmt chunk in {path}")
    tag, channels, rate, _byte_rate, _block, bits = struct.unpack_from(
        "<HHIIHH", chunks[0]
    )
    return tag, channels, rate, bits


def read_float_wav(path: Path) -> tuple[np.ndarray, int, int, str]:
    tag, channels, rate, bits = wav_format(path)
    payload = wav_data(path)
    if tag == 3 and bits == 32:
        subtype = "FLOAT"
        audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
    elif tag == 1 and bits == 16:
        subtype = "PCM_16"
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32)
        audio /= np.float32(32768.0)
    else:
        raise AssertionError(
            f"unsupported stage WAV encoding in {path}: tag={tag}, bits={bits}"
        )
    _assert(audio.size % channels == 0,
            f"stage payload is not channel-aligned: {path}")
    audio = audio.reshape(-1, channels)
    return audio, rate, channels, subtype


def pcm16_as_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / np.float32(32768.0)


def run_cli(cli: Path, args: list[str], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli), *args, "--out", str(prefix)]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    log = Path(str(prefix) + ".log")
    log.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-12000:]
        raise AssertionError(
            f"CLI failed with exit {result.returncode}: {' '.join(command)}\n{tail}"
        )


def load_report(prefix: Path) -> dict:
    return json.loads(Path(str(prefix) + ".json").read_text())


def _resolved_stage_file(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (metadata_path.parent / path)


def load_stages(
    prefix: Path,
    input_rate: int,
    input_samples: int,
    expected_tx: list[str],
    expected_rx: list[str],
) -> tuple[dict, dict[str, Stage]]:
    metadata_path = Path(str(prefix) + ".stages.json")
    _assert(metadata_path.is_file(), f"missing stage metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    _assert(metadata.get("input_sample_rate") == input_rate,
            f"bad top-level input_sample_rate: {metadata.get('input_sample_rate')}")
    _assert(metadata.get("input_samples_per_channel") == input_samples,
            "bad top-level input_samples_per_channel: "
            f"{metadata.get('input_samples_per_channel')} != {input_samples}")
    rows = metadata.get("pipeline_stages")
    _assert(isinstance(rows, list) and rows,
            f"pipeline_stages must be a non-empty array: {metadata_path}")

    by_id: dict[str, Stage] = {}
    by_direction: dict[str, list[Stage]] = {"tx": [], "rx": []}
    for row in rows:
        _assert(isinstance(row, dict), "pipeline stage entry is not an object")
        missing = STAGE_FIELDS - set(row)
        _assert(not missing, f"stage entry missing fields {sorted(missing)}: {row}")
        stage_id = row["id"]
        direction = row["direction"]
        _assert(isinstance(stage_id, str) and stage_id,
                f"invalid stage id: {stage_id!r}")
        _assert(stage_id not in by_id, f"duplicate stage id: {stage_id}")
        _assert(direction in ("tx", "rx"),
                f"invalid direction for {stage_id}: {direction!r}")
        _assert(type(row["order"]) is int and row["order"] >= 0,
                f"invalid order for {stage_id}: {row['order']!r}")
        for key in ("sample_rate", "channels", "samples_per_channel",
                    "fixed_delay_samples"):
            _assert(type(row[key]) is int and row[key] >= 0,
                    f"invalid {key} for {stage_id}: {row[key]!r}")
        _assert(row["sample_rate"] > 0 and row["channels"] > 0,
                f"non-positive stage shape for {stage_id}: {row}")
        _assert(type(row["enabled"]) is bool,
                f"enabled must be boolean for {stage_id}")
        _assert(type(row["capture_available"]) is bool,
                f"capture_available must be boolean for {stage_id}")
        _assert(isinstance(row["measurement_domain"], str)
                and row["measurement_domain"].strip(),
                f"empty measurement_domain for {stage_id}")
        _assert(isinstance(row["channel_policy"], str)
                and row["channel_policy"].strip(),
                f"empty channel_policy for {stage_id}")
        _assert(isinstance(row["file"], str) and row["file"],
                f"empty file path for {stage_id}")

        file_path = _resolved_stage_file(metadata_path, row["file"])
        _assert(file_path.is_file(), f"stage file is missing: {file_path}")
        audio, audio_rate, audio_channels, subtype = read_float_wav(file_path)
        _assert(audio_rate == row["sample_rate"],
                f"WAV rate mismatch for {stage_id}: {audio_rate}")
        _assert(audio_channels == row["channels"],
                f"WAV channel count mismatch for {stage_id}: {audio_channels}")
        _assert(audio.shape == (row["samples_per_channel"], row["channels"]),
                f"WAV shape mismatch for {stage_id}: {audio.shape} != "
                f"{(row['samples_per_channel'], row['channels'])}")
        _assert(np.isfinite(audio).all(), f"non-finite stage samples: {stage_id}")

        scaled = input_samples * row["sample_rate"]
        _assert(scaled % input_rate == 0,
                f"integer-frame test has non-integral stage rate for {stage_id}: "
                f"{input_samples}*{row['sample_rate']}/{input_rate}")
        expected_samples = scaled // input_rate + row["fixed_delay_samples"]
        _assert(row["samples_per_channel"] == expected_samples,
                f"wrong length for {stage_id}: {row['samples_per_channel']} != "
                f"{expected_samples}")
        _assert(row["capture_available"] is True,
                f"requested stage was not captured: {stage_id}")

        format_tag, fmt_channels, fmt_rate, fmt_bits = wav_format(file_path)
        _assert((fmt_channels, fmt_rate) == (row["channels"], row["sample_rate"]),
                f"fmt mismatch for {stage_id}: {(fmt_channels, fmt_rate)}")
        if stage_id not in ("tx_input", "rx_input"):
            _assert(subtype == "FLOAT" and format_tag == 3 and fmt_bits == 32,
                    f"processing stage must be IEEE_FLOAT32: {stage_id} "
                    f"({subtype}, tag={format_tag}, bits={fmt_bits})")
        else:
            _assert((subtype, format_tag, fmt_bits) in (
                        ("PCM_16", 1, 16), ("FLOAT", 3, 32)),
                    f"input stage must be PCM16 or FLOAT32: {stage_id} "
                    f"({subtype}, tag={format_tag}, bits={fmt_bits})")

        stage = Stage(dict(row), audio, subtype, format_tag)
        by_id[stage_id] = stage
        by_direction[direction].append(stage)

    actual_tx = [stage.id for stage in sorted(by_direction["tx"], key=lambda s: s.meta["order"])]
    actual_rx = [stage.id for stage in sorted(by_direction["rx"], key=lambda s: s.meta["order"])]
    # Native 48 kHz exports the actual pre-NS PCM. The legacy 16 kHz tap stays
    # restricted to its established AEC-off mono/local-VQE envelopes.
    if input_rate == 48000:
        _assert("tx_ns_input" in actual_tx,
                "native 48 kHz TX must export tx_ns_input")
    if "tx_ns_input" in actual_tx:
        _assert(input_rate in (16000, 48000),
                "tx_ns_input must use the native 16/48 kHz core")
        boundary = by_id["tx_ns_input"]
        _assert(boundary.meta["sample_rate"] == input_rate,
                "tx_ns_input must use the input sample rate")
        _assert(boundary.meta["channels"] == 1 and boundary.meta["diagnostic"] is True,
                "tx_ns_input must be a mono diagnostic tap")
        _assert(boundary.meta["fixed_delay_samples"] <= by_id["tx_ns"].meta["fixed_delay_samples"],
                "pre-NS tap cannot occur after the NS output")
        expected_tx = list(expected_tx)
        expected_tx.insert(expected_tx.index("tx_ns_selected"), "tx_ns_input")
    _assert(actual_tx == expected_tx,
            f"TX pipeline order/coverage mismatch: {actual_tx} != {expected_tx}")
    _assert(actual_rx == expected_rx,
            f"RX pipeline order/coverage mismatch: {actual_rx} != {expected_rx}")
    for direction, expected in (("tx", expected_tx), ("rx", expected_rx)):
        orders = sorted(stage.meta["order"] for stage in by_direction[direction])
        _assert(orders == list(range(len(expected))),
                f"{direction} orders must be contiguous from zero: {orders}")
    return metadata, by_id


def stage(bundle: dict[str, Stage], stage_id: str) -> Stage:
    _assert(stage_id in bundle, f"required stage is absent: {stage_id}")
    return bundle[stage_id]


def assert_enabled(bundle: dict[str, Stage], stage_id: str, enabled: bool) -> None:
    actual = stage(bundle, stage_id).meta["enabled"]
    _assert(actual is enabled,
            f"{stage_id} enabled={actual}, expected {enabled}")


def assert_non_silent(audio: np.ndarray, label: str) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    _assert(math.isfinite(peak) and peak > 1.0e-7,
            f"diagnostic waveform is silent/non-finite: {label}")


def assert_input_matches(stage_value: Stage, pcm: np.ndarray, label: str) -> None:
    _assert(stage_value.delay == 0, f"{label} input tap must have zero delay")
    expected = pcm16_as_float(pcm)
    _assert(stage_value.audio.shape == expected.shape,
            f"{label} input shape {stage_value.audio.shape} != {expected.shape}")
    np.testing.assert_allclose(stage_value.audio, expected, rtol=0.0, atol=2.0e-7,
                               err_msg=f"{label} input waveform changed")


def assert_float_matches_pcm16(audio: np.ndarray, pcm: np.ndarray, label: str) -> None:
    _assert(audio.shape == pcm.shape,
            f"{label} shape {audio.shape} != {pcm.shape}")
    expected = pcm16_as_float(pcm)
    np.testing.assert_allclose(audio, expected, rtol=0.0, atol=1.01 / 32768.0,
                               err_msg=f"{label} differs from same-run PCM16 output")


def peak_index(audio: np.ndarray, label: str) -> int:
    mono = np.asarray(audio)[:, 0]
    assert_non_silent(mono, label)
    return int(np.argmax(np.abs(mono)))


def make_duplex_fixture(case: Path, rate: int) -> tuple[int, np.ndarray, np.ndarray]:
    n = rate + 7  # one second plus a genuinely partial 10 ms frame
    t = np.arange(n, dtype=np.float64) / rate
    mic0 = 0.16 * np.sin(2 * np.pi * 173.0 * t)
    mic1 = 0.13 * np.sin(2 * np.pi * 311.0 * t + 0.37)
    rx0 = 0.31 * np.sin(2 * np.pi * 431.0 * t + 0.11)
    rx1 = 0.27 * np.sin(2 * np.pi * 617.0 * t + 0.83)
    mic0[rate // 7 + 3] += 0.38
    mic1[rate // 3 + 5] -= 0.34
    rx0[rate // 5 + 1] += 0.44
    rx1[rate // 2 + 7] -= 0.41
    mic = np.column_stack((mic0, mic1)) * 32768.0
    rx = np.column_stack((rx0, rx1)) * 32768.0
    write_pcm16(case / "mic.wav", rate, mic)
    write_pcm16(case / "rx.wav", rate, rx)
    return n, read_pcm16(case / "mic.wav")[1], read_pcm16(case / "rx.wav")[1]


def test_duplex_stage_exports(cli: Path, root: Path) -> None:
    for rate in (16000, 24000, 48000):
        case = root / f"duplex-{rate}"
        case.mkdir(parents=True, exist_ok=True)
        n, mic, rx = make_duplex_fixture(case, rate)
        common = [
            "--profile", "vcp",
            "--mic", str(case / "mic.wav"),
            "--beam-delays", "0,0",
            "--rx-in", str(case / "rx.wav"),
        ]
        off_prefix = case / "diag-off"
        on_prefix = case / "diag-on"
        run_cli(cli, common, off_prefix)
        run_cli(cli, [*common, "--diagnostic-stages"], on_prefix)

        off_report = load_report(off_prefix)
        on_report = load_report(on_prefix)
        _assert(off_report["dsp_delay_samples"] == on_report["dsp_delay_samples"],
                f"diagnostics changed DSP delay at {rate} Hz")
        after_path_off = Path(str(off_prefix) + ".after.wav")
        after_path_on = Path(str(on_prefix) + ".after.wav")
        after_rate_off, after_off = read_pcm16(after_path_off)
        after_rate_on, after_on = read_pcm16(after_path_on)
        expected_after = (rate, n + on_report["dsp_delay_samples"], 1)
        _assert((after_rate_off, *after_off.shape) == expected_after,
                f"diagnostics-off final shape changed: {(after_rate_off, *after_off.shape)}")
        _assert((after_rate_on, *after_on.shape) == expected_after,
                f"diagnostics-on final shape changed: {(after_rate_on, *after_on.shape)}")
        _assert(wav_data(after_path_off) == wav_data(after_path_on),
                f"diagnostic taps changed final PCM16 bytes at {rate} Hz")
        np.testing.assert_array_equal(after_off, after_on,
                                      err_msg=f"diagnostic taps changed final samples at {rate} Hz")

        rx_off = {}
        rx_on = {}
        for label in ("rx-spk", "rx-ref"):
            rx_off[label] = Path(str(off_prefix) + f".{label}.wav")
            rx_on[label] = Path(str(on_prefix) + f".{label}.wav")
            rate_off, samples_off = read_pcm16(rx_off[label])
            rate_on, samples_on = read_pcm16(rx_on[label])
            _assert((rate_off, *samples_off.shape) == (rate, n, 2),
                    f"{label} off shape changed: {(rate_off, *samples_off.shape)}")
            _assert((rate_on, *samples_on.shape) == (rate, n, 2),
                    f"{label} on shape changed: {(rate_on, *samples_on.shape)}")
            _assert(wav_data(rx_off[label]) == wav_data(rx_on[label]),
                    f"diagnostics changed {label} bytes at {rate} Hz")
            np.testing.assert_array_equal(samples_off, samples_on,
                                          err_msg=f"diagnostics changed {label} at {rate} Hz")
        _assert(wav_data(rx_on["rx-spk"]) == wav_data(rx_on["rx-ref"]),
                f"same-run RX speaker/reference bytes differ at {rate} Hz")
        spk = read_pcm16(rx_on["rx-spk"])[1]
        ref = read_pcm16(rx_on["rx-ref"])[1]
        np.testing.assert_array_equal(spk, ref,
                                      err_msg=f"same-run RX speaker/reference samples differ at {rate} Hz")
        _assert(not np.array_equal(spk, rx), "active RX path was not exercised")

        _metadata, bundle = load_stages(
            on_prefix, rate, n, TX_IDS_AEC_ON, RX_IDS
        )
        if rate == 16000:
            _assert("tx_ns_input" not in bundle,
                    "16 kHz AEC-on duplex behavior must not gain tx_ns_input")

        for stage_id in ("tx_input", "tx_bf", "tx_aec_input", "tx_aec",
                         "tx_ns_selected", "tx_ns", "tx_eq", "tx_agc",
                         "tx_drc", "tx_output"):
            assert_non_silent(stage(bundle, stage_id).audio, stage_id)
        for stage_id in RX_IDS:
            assert_non_silent(stage(bundle, stage_id).audio, stage_id)

        assert_input_matches(stage(bundle, "tx_input"), mic, "TX")
        assert_input_matches(stage(bundle, "rx_input"), rx, "RX")
        _assert(stage(bundle, "tx_input").meta["channels"] == 2,
                "tx_input must preserve both microphone channels")
        _assert(stage(bundle, "tx_aec_input").meta["channels"] == 2,
                "tx_aec_input must preserve both pre-AEC capture channels")
        _assert(stage(bundle, "tx_aec").meta["channels"] == 2,
                "tx_aec must preserve the engine's interleaved capture slots")
        for stage_id in ("tx_bf", "tx_ns_selected", "tx_ns", "tx_eq",
                         "tx_agc", "tx_drc", "tx_output"):
            _assert(stage(bundle, stage_id).meta["channels"] == 1,
                    f"{stage_id} must be mono")
        for stage_id in RX_IDS:
            _assert(stage(bundle, stage_id).meta["channels"] == 2,
                    f"{stage_id} must preserve both RX channels")

        bf = stage(bundle, "tx_bf")
        _assert(bf.delay == 0, "zero-delay beamforming must not report a fixed delay")
        expected_bf = pcm16_as_float(mic).mean(axis=1)
        np.testing.assert_allclose(bf.audio[:n, 0], expected_bf,
                                   rtol=0.0, atol=2.0e-7,
                                   err_msg="tx_bf is not the pre-AEC two-mic average")
        np.testing.assert_array_equal(bf.audio[n:, 0], np.zeros(bf.audio.shape[0] - n,
                                                                dtype=np.float32),
                                      err_msg="zero-delay BF tap has a non-zero tail")

        assert_enabled(bundle, "tx_bf", True)
        assert_enabled(bundle, "tx_aec", on_report["aec_enabled"])
        assert_enabled(bundle, "tx_aec_input", True)
        if rate == 48000:
            ns_input = stage(bundle, "tx_ns_input")
            ns_output = stage(bundle, "tx_ns")
            _assert(ns_input.meta["sample_rate"] == 48000,
                    "48 kHz tx_ns_input metadata sample_rate is not 48000")
            _assert(ns_input.meta["measurement_domain"]
                    == "actual native NS input after capture conditioning",
                    "48 kHz tx_ns_input measurement_domain changed")
            _assert(ns_input.meta["channel_policy"] == "mono actual NS input",
                    "48 kHz tx_ns_input channel_policy changed")
            _assert(ns_input.delay + on_report["ns_delay_samples"] == ns_output.delay,
                    "48 kHz tx_ns_input fixed delay is inconsistent with NS stage metadata: "
                    f"{ns_input.delay} + {on_report['ns_delay_samples']} != {ns_output.delay}")
            assert_non_silent(ns_input.audio, "tx_ns_input")
            assert_enabled(bundle, "tx_ns_input", on_report["ns_enabled"])
            res_prefix = case / "diag-res"
            run_cli(
                cli,
                [*common, "--speex-residual-suppression", "--diagnostic-stages"],
                res_prefix,
            )
            res_report = load_report(res_prefix)
            _res_metadata, res_bundle = load_stages(
                res_prefix, rate, n, TX_IDS_AEC_ON, RX_IDS
            )
            res_input = stage(res_bundle, "tx_ns_input")
            res_output = stage(res_bundle, "tx_ns")
            _assert(res_input.delay == 480,
                    "48 kHz Speex RES must contribute one 480-sample pre-NS delay: "
                    f"{res_input.delay}")
            _assert(res_input.delay + res_report["ns_delay_samples"] == res_output.delay,
                    "48 kHz Speex RES tx_ns_input delay is inconsistent with NS metadata: "
                    f"{res_input.delay} + {res_report['ns_delay_samples']} != {res_output.delay}")
        assert_enabled(bundle, "tx_ns_selected", on_report["ns_enabled"])
        assert_enabled(bundle, "tx_ns", on_report["ns_enabled"])
        assert_enabled(bundle, "tx_agc", on_report["agc2_enabled"])
        assert_enabled(bundle, "tx_drc", on_report["tx_drc_enabled"])
        assert_enabled(bundle, "tx_input", True)
        assert_enabled(bundle, "tx_output", True)
        assert_enabled(bundle, "rx_input", True)
        assert_enabled(bundle, "rx_nr", on_report["rx_processing"])
        assert_enabled(bundle, "rx_eq", False)
        assert_enabled(bundle, "rx_drc", on_report["rx_processing"])
        assert_enabled(bundle, "rx_output", True)
        _assert(stage(bundle, "tx_output").delay == on_report["dsp_delay_samples"],
                "tx_output delay does not match final DSP delay")
        _assert(stage(bundle, "tx_output").meta["samples_per_channel"] == after_on.shape[0],
                "tx_output length does not match final output")
        assert_float_matches_pcm16(stage(bundle, "tx_output").audio, after_on,
                                   "tx_output")
        rx_output = stage(bundle, "rx_output")
        _assert(rx_output.delay == stage(bundle, "rx_drc").delay,
                "rx_output delay must include RX DRC and no later stage")
        assert_float_matches_pcm16(rx_output.audio[:n], spk,
                                   "rx_output/same-run speaker")
        print(f"TX/RX stage export duplex {rate} Hz, partial frame: PASS")


def make_impulse_fixture(case: Path, rate: int = 16000) -> tuple[int, np.ndarray]:
    n = rate + 7
    samples = np.zeros(n, dtype=np.float64)
    samples[1001] = 0.125
    write_pcm16(case / "impulse.wav", rate, samples[:, None] * 32768.0)
    return n, read_pcm16(case / "impulse.wav")[1]


def test_upstream_taps_and_downstream_delays(cli: Path, root: Path) -> None:
    case = root / "upstream-delay"
    case.mkdir(parents=True, exist_ok=True)
    n, impulse = make_impulse_fixture(case)
    base = [
        "--profile", "vcp",
        "--mic", str(case / "impulse.wav"),
        "--no-aec", "--no-ns", "--no-agc", "--no-limiter",
    ]
    base_prefix = case / "base"
    full_prefix = case / "eq-drc"
    run_cli(cli, [*base, "--no-tx-drc", "--diagnostic-stages"], base_prefix)
    run_cli(cli, [
        *base,
        "--eq-preamp-db", "-6",
        "--eq-phase", "linear",
        "--eq-linear-taps", "129",
        "--tx-drc",
        "--tx-drc-lookahead-ms", "5",
        "--diagnostic-stages",
    ], full_prefix)

    _base_meta, base_stages = load_stages(base_prefix, 16000, n, TX_IDS_AEC_OFF, [])
    full_meta, full_stages = load_stages(full_prefix, 16000, n, TX_IDS_AEC_OFF, [])
    _assert(full_meta["input_sample_rate"] == 16000,
            "linear-EQ/DRC fixture did not run at native 16 kHz")

    for stage_id in ("tx_input", "tx_bf", "tx_aec", "tx_ns_selected", "tx_ns"):
        left = stage(base_stages, stage_id)
        right = stage(full_stages, stage_id)
        _assert(left.delay == right.delay,
                f"downstream EQ/DRC changed {stage_id} delay: "
                f"{left.delay} != {right.delay}")
        np.testing.assert_array_equal(
            left.audio, right.audio,
            err_msg=f"downstream EQ/DRC changed upstream {stage_id} waveform",
        )

    for stage_id in ("tx_bf", "tx_aec", "tx_ns_selected", "tx_ns"):
        _assert(stage(base_stages, stage_id).meta["enabled"] is False
                and stage(full_stages, stage_id).meta["enabled"] is False,
                f"{stage_id} must remain a disabled bypass")

    assert_enabled(full_stages, "tx_aec", False)
    assert_enabled(full_stages, "tx_ns_selected", False)
    assert_enabled(full_stages, "tx_ns", False)
    assert_enabled(full_stages, "tx_eq", True)
    assert_enabled(full_stages, "tx_agc", False)
    assert_enabled(full_stages, "tx_drc", True)

    for stage_id in ("tx_bf", "tx_aec", "tx_ns_selected", "tx_ns", "tx_eq",
                     "tx_agc", "tx_drc", "tx_output"):
        _assert(stage(full_stages, stage_id).meta["sample_rate"] == 16000,
                f"{stage_id} must use native 16 kHz in this fixture")

    ns_stage = stage(full_stages, "tx_ns")
    eq_stage = stage(full_stages, "tx_eq")
    agc_stage = stage(full_stages, "tx_agc")
    drc_stage = stage(full_stages, "tx_drc")
    output_stage = stage(full_stages, "tx_output")
    _assert(eq_stage.delay - ns_stage.delay == 64,
            f"linear EQ fixed delay must be 64 samples: {eq_stage.delay - ns_stage.delay}")
    _assert(agc_stage.delay == eq_stage.delay,
            f"disabled AGC changed post-EQ delay: {agc_stage.delay} != {eq_stage.delay}")
    _assert(drc_stage.delay - agc_stage.delay == 80,
            f"DRC lookahead must add exactly 80 samples: {drc_stage.delay - agc_stage.delay}")
    _assert(output_stage.delay == drc_stage.delay,
            f"tx_output delay must equal tx_drc: {output_stage.delay} != {drc_stage.delay}")
    _assert(drc_stage.meta["samples_per_channel"] - agc_stage.meta["samples_per_channel"] == 80,
            "DRC stage length must include its 80-sample lookahead tail")
    np.testing.assert_array_equal(agc_stage.audio, eq_stage.audio,
                                  err_msg="disabled AGC must be a post-EQ bypass")
    np.testing.assert_array_equal(
        drc_stage.audio[:80], np.zeros((80, 1), dtype=np.float32),
        err_msg="DRC lookahead did not leave a zero-prefix delay",
    )

    ns_peak = peak_index(ns_stage.audio, "tx_ns")
    eq_peak = peak_index(eq_stage.audio, "tx_eq")
    agc_peak = peak_index(agc_stage.audio, "tx_agc")
    drc_peak = peak_index(drc_stage.audio, "tx_drc")
    output_peak = peak_index(output_stage.audio, "tx_output")
    _assert(eq_peak - ns_peak == 64,
            f"linear-EQ impulse moved by {eq_peak - ns_peak}, not 64 samples")
    _assert(agc_peak == eq_peak,
            f"disabled AGC moved impulse by {agc_peak - eq_peak} samples")
    _assert(drc_peak - agc_peak == 80,
            f"DRC impulse moved by {drc_peak - agc_peak}, not 80 samples")
    _assert(output_peak == drc_peak,
            f"tx_output moved DRC impulse by {output_peak - drc_peak} samples")

    after_path = Path(str(full_prefix) + ".after.wav")
    after_rate, after = read_pcm16(after_path)
    _assert(after_rate == 16000 and after.shape == (n + output_stage.delay, 1),
            f"final output shape is wrong: {(after_rate, *after.shape)}, "
            f"expected {(16000, n + output_stage.delay, 1)}")
    _assert(peak_index(after, "final PCM16 output") == output_peak,
            "final PCM16 impulse position does not match tx_output")
    assert_float_matches_pcm16(output_stage.audio, after, "tx_output/final")
    assert_non_silent(stage(full_stages, "tx_input").audio, "tx_input")
    np.testing.assert_allclose(stage(full_stages, "tx_input").audio,
                               pcm16_as_float(impulse), rtol=0.0, atol=2.0e-7,
                               err_msg="tx_input changed the raw impulse")
    print("upstream taps, linear-EQ delay 64, DRC lookahead 80: PASS")


def test_all_modules_off_are_visible_bypasses(cli: Path, root: Path) -> None:
    case = root / "all-off"
    case.mkdir(parents=True, exist_ok=True)
    n, impulse = make_impulse_fixture(case)
    prefix = case / "bypass"
    run_cli(cli, [
        "--profile", "vcp",
        "--mic", str(case / "impulse.wav"),
        "--no-bf", "--no-aec", "--no-ns", "--no-agc",
        "--no-tx-drc", "--no-limiter", "--no-low-cut",
        "--diagnostic-stages",
    ], prefix)
    _metadata, bundle = load_stages(prefix, 16000, n, TX_IDS_AEC_OFF, [])
    input_audio = pcm16_as_float(impulse)

    for stage_id in ("tx_bf", "tx_aec", "tx_ns_selected", "tx_ns", "tx_eq",
                     "tx_agc", "tx_drc"):
        bypass = stage(bundle, stage_id)
        _assert(bypass.delay == 0,
                f"all-off {stage_id} introduced a {bypass.delay}-sample delay")
        np.testing.assert_array_equal(bypass.audio, input_audio,
                                      err_msg=f"all-off {stage_id} is not an exact bypass")
    for stage_id in ("tx_bf", "tx_aec", "tx_ns_selected", "tx_ns", "tx_eq",
                     "tx_agc", "tx_drc"):
        assert_enabled(bundle, stage_id, False)
    np.testing.assert_array_equal(stage(bundle, "tx_bf").audio, input_audio,
                                  err_msg="--no-bf did not expose microphone channel zero")
    np.testing.assert_array_equal(stage(bundle, "tx_output").audio, input_audio,
                                  err_msg="all-off tx_output is not the original channel zero")
    _assert(stage(bundle, "tx_output").delay == 0,
            "all-off tx_output must have zero fixed delay")

    after_path = Path(str(prefix) + ".after.wav")
    after_rate, after = read_pcm16(after_path)
    _assert((after_rate, *after.shape) == (16000, n, 1),
            f"all-off final output shape is wrong: {(after_rate, *after.shape)}")
    _assert(wav_data(after_path) == wav_data(case / "impulse.wav"),
            "all-off final PCM16 bytes differ from the microphone input")
    print("all-off bypass, --no-bf channel zero, visible AEC bypass: PASS")


def test_rx_only_stages_and_same_run_reference(cli: Path, root: Path) -> None:
    case = root / "rx-only"
    case.mkdir(parents=True, exist_ok=True)
    rate = 16000
    n = rate + 7
    t = np.arange(n, dtype=np.float64) / rate
    rx0 = 0.42 * np.sin(2 * np.pi * 523.0 * t)
    rx1 = 0.36 * np.sin(2 * np.pi * 733.0 * t + 0.4)
    rx0[rate // 4] += 0.31
    rx1[rate // 2 + 1] -= 0.29
    rx0[-1] += 0.36  # leaves a measurable lookahead tail if RX DRC gains one
    rx = np.clip(np.column_stack((rx0, rx1)) * 32768.0, -32768, 32767)
    write_pcm16(case / "rx.wav", rate, rx)
    rx = read_pcm16(case / "rx.wav")[1]
    prefix = case / "stages"
    run_cli(cli, [
        "--profile", "vcp",
        "--rx-in", str(case / "rx.wav"),
        "--diagnostic-stages",
    ], prefix)
    report = load_report(prefix)
    _metadata, bundle = load_stages(prefix, rate, n, [], RX_IDS)
    assert_input_matches(stage(bundle, "rx_input"), rx, "RX-only")
    assert_enabled(bundle, "rx_nr", True)
    assert_enabled(bundle, "rx_eq", False)
    assert_enabled(bundle, "rx_drc", True)
    assert_enabled(bundle, "rx_output", True)
    np.testing.assert_array_equal(stage(bundle, "rx_eq").audio,
                                  stage(bundle, "rx_nr").audio,
                                  err_msg="disabled RX EQ must be an exact NR bypass")
    for stage_id in RX_IDS:
        _assert(stage(bundle, stage_id).meta["channels"] == 2,
                f"RX-only {stage_id} did not preserve both channels")

    spk_path = Path(str(prefix) + ".rx-spk.wav")
    ref_path = Path(str(prefix) + ".rx-ref.wav")
    spk_rate, spk = read_pcm16(spk_path)
    ref_rate, ref = read_pcm16(ref_path)
    _assert((spk_rate, *spk.shape) == (rate, n, 2),
            f"RX playback must end at the exact input duration: {(spk_rate, *spk.shape)}")
    _assert((ref_rate, *ref.shape) == (rate, n, 2),
            f"RX reference must end at the exact input duration: {(ref_rate, *ref.shape)}")
    _assert(wav_data(spk_path) == wav_data(ref_path),
            "RX speaker and AEC reference differ within the same run")
    np.testing.assert_array_equal(spk, ref,
                                  err_msg="RX speaker and AEC reference samples differ in the same run")
    _assert(report["rx_processing"] is True, "RX-only test did not enable RX processing")

    rx_drc = stage(bundle, "rx_drc")
    rx_output = stage(bundle, "rx_output")
    _assert(rx_output.delay == rx_drc.delay,
            f"RX output delay {rx_output.delay} != DRC delay {rx_drc.delay}")
    if rx_drc.delay:
        _assert(rx_drc.meta["samples_per_channel"] == n + rx_drc.delay,
                "RX DRC diagnostic length omits its lookahead tail")
        _assert(np.max(np.abs(rx_drc.audio[n:])) > 1.0e-7,
                "RX DRC tail is missing despite a non-zero fixed delay")
        _assert(np.max(np.abs(rx_output.audio[n:])) > 1.0e-7,
                "RX output tail is missing despite a non-zero fixed delay")
    else:
        _assert(rx_drc.meta["samples_per_channel"] == n,
                "RX DRC reports no delay but exports extra samples")
    assert_float_matches_pcm16(rx_output.audio[:n], spk,
                               "rx_output/same-run RX speaker")
    print("RX-only stages and exact-duration same-run reference: PASS "
          f"(RX DRC lookahead={rx_drc.delay}; nonzero tail checked when present)")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python test_vcp_stage_exports.py <itu_audio_cli> <output-directory>"
        )
    cli = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    _assert(cli.is_file(), f"CLI executable does not exist: {cli}")
    root.mkdir(parents=True, exist_ok=True)
    test_duplex_stage_exports(cli, root)
    test_upstream_taps_and_downstream_delays(cli, root)
    test_all_modules_off_are_visible_bypasses(cli, root)
    test_rx_only_stages_and_same_run_reference(cli, root)
    print("ALL VCP diagnostic stage export integration checks: PASS")


if __name__ == "__main__":
    main()
