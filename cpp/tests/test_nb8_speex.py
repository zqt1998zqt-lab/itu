#!/usr/bin/env python3
"""End-to-end checks for the standalone native 8 kHz Speex MDF CLI."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import wave


def write_pcm16(path: Path, samples: list[int], rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def write_float32(path: Path, samples: list[float], rate: int = 8000) -> None:
    payload = b"".join(struct.pack("<f", float(value)) for value in samples)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE"
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_pcm16(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as stream:
        if stream.getcomptype() != "NONE":
            raise AssertionError(f"unexpected WAV compression: {path}")
        return (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.readframes(stream.getnframes()),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_pair(count: int) -> tuple[list[int], list[int]]:
    reference: list[int] = []
    microphone: list[int] = []
    for index in range(count):
        far = 0.14 * math.sin(2.0 * math.pi * 220.0 * index / 8000.0)
        near = 0.055 * math.sin(2.0 * math.pi * 440.0 * index / 8000.0)
        reference.append(round(32767.0 * far))
        microphone.append(round(32767.0 * (0.58 * far + near)))
    return microphone, reference


def run(
    cli: Path,
    mic: Path,
    echo: Path,
    prefix: Path,
    *extra: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(cli),
        "--mic",
        str(mic),
        "--echo",
        str(echo),
        "--out",
        str(prefix),
        *extra,
    ]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=60, check=False
    )
    if (result.returncode == 0) != expect_success:
        raise AssertionError(
            f"unexpected exit={result.returncode}, command={command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def assert_report(
    prefix: Path,
    mic: Path,
    echo: Path,
    samples: int,
    tail_ms: int,
) -> dict:
    report_path = Path(str(prefix) + ".json")
    output_path = Path(str(prefix) + ".wav")
    report = json.loads(report_path.read_text())
    assert report["schema"] == "itu.nb8_speex.report.v1"
    assert report["schema_version"] == 1
    assert report["sample_rate"] == 8000
    assert report["channels"] == 1
    assert report["frame_samples"] == 80
    assert report["tail_ms"] == tail_ms
    assert report["tail_samples"] == tail_ms * 8
    assert report["aec_backend"] == "speex_mdf_nb8"
    assert report["aec_inference_frames"] == math.ceil(samples / 80)
    assert report["aec_padded_tail_samples"] == (-samples) % 80
    assert report["input_samples"] == samples
    assert report["mic_input_samples"] == samples
    assert report["echo_input_samples"] == samples
    assert report["output_samples"] == samples
    assert report["finite"] is True
    assert report["output_finite"] is True
    assert report["no_resample"] is True
    assert report["resampling"] == "none"
    assert report["fallback_frames"] == 0
    assert report["algorithm_delay_samples"] == 0
    assert report["evidence_scope"]
    assert report["mic_input_sha256"] == sha256(mic)
    assert report["echo_input_sha256"] == sha256(echo)
    assert report["input_sha256"] == {
        "mic": sha256(mic),
        "echo": sha256(echo),
    }
    assert report["output_sha256"] == sha256(output_path)
    assert report["input_hash_scope"] == "raw input file bytes"
    assert report["output_hash_scope"] == "PCM16 WAV file bytes"
    assert report["output_format"] == "PCM16"
    assert Path(report["mic_file"]) == mic.resolve()
    assert Path(report["echo_file"]) == echo.resolve()
    assert Path(report["output_file"]) == output_path.resolve()
    return report


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_nb8_speex.py BINARY OUTPUT_DIR")
    cli = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    if not cli.is_file():
        raise SystemExit(f"binary does not exist: {cli}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # A non-frame-multiple length exercises the explicit zero-padded final
    # inference frame while requiring output length to remain unchanged.
    count = 417
    mic_samples, echo_samples = deterministic_pair(count)
    mic_path = output_dir / "mic-pcm16.wav"
    echo_path = output_dir / "echo-pcm16.wav"
    write_pcm16(mic_path, mic_samples)
    write_pcm16(echo_path, echo_samples)

    prefix = output_dir / "speex-result"
    run(cli, mic_path, echo_path, prefix)
    rate, channels, width, payload = read_pcm16(Path(str(prefix) + ".wav"))
    assert (rate, channels, width) == (8000, 1, 2)
    assert len(payload) == count * 2
    report = assert_report(prefix, mic_path, echo_path, count, 500)
    assert report["mic_input_format"] == "PCM16"
    assert report["echo_input_format"] == "PCM16"
    assert report["input_clipped_samples"] == 0
    print("PCM16 native 8 kHz AEC/report/length: PASS")

    repeat = output_dir / "speex-result-repeat"
    run(cli, mic_path, echo_path, repeat, "--tail-ms", "500")
    assert Path(str(prefix) + ".wav").read_bytes() == Path(
        str(repeat) + ".wav"
    ).read_bytes()
    repeated_report = assert_report(repeat, mic_path, echo_path, count, 500)
    assert repeated_report["output_sha256"] == report["output_sha256"]
    print("deterministic output: PASS")

    float_count = 160
    float_mic, float_echo = deterministic_pair(float_count)
    float_mic_path = output_dir / "mic-float32.wav"
    float_echo_path = output_dir / "echo-float32.wav"
    write_float32(float_mic_path, [value / 32768.0 for value in float_mic])
    write_float32(float_echo_path, [value / 32768.0 for value in float_echo])
    float_prefix = output_dir / "speex-float32"
    run(
        cli,
        float_mic_path,
        float_echo_path,
        float_prefix,
        "--tail-ms",
        "250",
    )
    float_rate, float_channels, float_width, float_payload = read_pcm16(
        Path(str(float_prefix) + ".wav")
    )
    assert (float_rate, float_channels, float_width) == (8000, 1, 2)
    assert len(float_payload) == float_count * 2
    float_report = assert_report(
        float_prefix, float_mic_path, float_echo_path, float_count, 250
    )
    assert float_report["mic_input_format"] == "FLOAT32"
    assert float_report["echo_input_format"] == "FLOAT32"
    print("FLOAT32 input to PCM16 output without resampling: PASS")

    wrong_rate_mic = output_dir / "wrong-rate-mic.wav"
    wrong_rate_echo = output_dir / "wrong-rate-echo.wav"
    write_pcm16(wrong_rate_mic, mic_samples, rate=16000)
    write_pcm16(wrong_rate_echo, echo_samples, rate=16000)
    rejected = run(
        cli,
        wrong_rate_mic,
        wrong_rate_echo,
        output_dir / "wrong-rate",
        expect_success=False,
    )
    assert "8000 Hz" in rejected.stderr
    print("wrong sample rate rejection: PASS")

    short_echo = output_dir / "short-echo.wav"
    write_pcm16(short_echo, echo_samples[: count - 80])
    rejected = run(
        cli,
        mic_path,
        short_echo,
        output_dir / "length-mismatch",
        expect_success=False,
    )
    assert "same sample count" in rejected.stderr
    print("length mismatch rejection: PASS")

    print("all NB8 Speex MDF checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
