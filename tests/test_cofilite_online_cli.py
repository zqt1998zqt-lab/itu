"""Black-box tests for the native online CoFi-Lite pipe bridge."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import threading
import wave

import pytest
from server import ns_routes


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build/cofilite-opt-20261116/host-v3/cpp/itu_audio_cli"
FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * 4
DELAY_SAMPLES = ns_routes.NS_ROUTES["cofilite"].delay_samples
DELAY_FRAMES = DELAY_SAMPLES // FRAME_SAMPLES
assert DELAY_SAMPLES > 0 and DELAY_SAMPLES % FRAME_SAMPLES == 0


def _read_exact(fd: int, size: int, *, allow_eof: bool = False) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        try:
            block = os.read(fd, size - len(chunks))
        except InterruptedError:
            continue
        if not block:
            if not chunks and allow_eof:
                return None
            raise EOFError("unexpected EOF")
        chunks.extend(block)
    return bytes(chunks)


def _write_exact(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise EOFError("pipe write returned zero")
        offset += written


@dataclass
class DelayedProvider:
    mode: str = "normal"
    inputs: list[list[float]] = field(default_factory=list)
    outputs: list[list[float]] = field(default_factory=list)
    error: str | None = None

    def run(self, read_fd: int, write_fd: int) -> None:
        delayed = deque([[0.0] * FRAME_SAMPLES for _ in range(DELAY_FRAMES)])
        try:
            while True:
                raw = _read_exact(read_fd, FRAME_BYTES, allow_eof=True)
                if raw is None:
                    return
                frame = list(struct.unpack("<160f", raw))
                if not all(math.isfinite(value) for value in frame):
                    raise ValueError("native supplied a non-finite frame")
                self.inputs.append(frame)

                if self.mode == "early_eof":
                    return
                if self.mode == "nan" and len(self.inputs) == 2:
                    output = [0.0] * FRAME_SAMPLES
                    output[0] = math.nan
                    _write_exact(write_fd, struct.pack("<160f", *output))
                    self.outputs.append(output)
                    return

                source = delayed.popleft()
                delayed.append(frame)
                output = [value * 0.5 for value in source]
                _write_exact(write_fd, struct.pack("<160f", *output))
                self.outputs.append(output)
        except BaseException as error:  # Preserve the exact thread failure for pytest.
            self.error = f"{type(error).__name__}: {error}"
        finally:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass


def _require_cli() -> Path:
    if os.name != "posix":
        pytest.skip("online NS pipes require POSIX")
    assert CLI.is_file(), f"fixed candidate CLI is missing: {CLI}"
    return CLI


def _base_command(mic: Path, prefix: Path, input_fd: int, output_fd: int) -> list[str]:
    return [
        str(_require_cli()),
        "--mic", str(mic),
        "--no-aec",
        "--no-bf",
        "--no-low-cut",
        "--no-tx-drc",
        "--no-limiter",
        "--agc-fixed-gain", "3",
        "--eq-preamp-db", "-3",
        "--eq-band", "peak,1000,3,0.707",
        "--diagnostic-stages",
        "--ns-stream-input-fd", str(input_fd),
        "--ns-stream-output-fd", str(output_fd),
        "--ns-stream-delay-samples", str(DELAY_SAMPLES),
        "--out", str(prefix),
    ]


def _run_with_provider(
    command_builder,
    tmp_path: Path,
    *,
    mode: str = "normal",
) -> tuple[subprocess.CompletedProcess[str], DelayedProvider]:
    mic = tmp_path / "mic.wav"
    prefix = tmp_path / ("failure" if mode != "normal" else "online")
    _write_pcm16(mic, _pcm16_tone(480))

    engine_to_provider_r, engine_to_provider_w = os.pipe()
    provider_to_engine_r, provider_to_engine_w = os.pipe()
    provider = DelayedProvider(mode=mode)
    thread = threading.Thread(
        target=provider.run,
        args=(engine_to_provider_r, provider_to_engine_w),
        name="itu-ns-delay-provider",
        daemon=True,
    )
    thread.start()
    command = command_builder(mic, prefix, engine_to_provider_w, provider_to_engine_r)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(engine_to_provider_w, provider_to_engine_r),
    )
    os.close(engine_to_provider_w)
    os.close(provider_to_engine_r)
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        raise
    thread.join(timeout=5)
    assert not thread.is_alive(), "provider thread did not terminate within 5 seconds"
    assert provider.error is None, provider.error
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        provider,
    )


def _run_no_provider(
    command: list[str],
    *,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        pass_fds=pass_fds,
        check=False,
    )


def _write_pcm16(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _pcm16_tone(count: int) -> list[int]:
    return [
        int(round(0.15 * 32768.0 * math.sin(2.0 * math.pi * 440.0 * i / 16000.0)))
        for i in range(count)
    ]


def _write_float32_wav(path: Path, samples: list[float]) -> None:
    payload = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _read_float32_wav(path: Path) -> list[float]:
    blob = path.read_bytes()
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"
    end = 8 + struct.unpack_from("<I", blob, 4)[0]
    data = None
    offset = 12
    while offset + 8 <= end:
        tag = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        payload = blob[start:start + size]
        if tag == b"fmt ":
            kind, channels, rate, _byte_rate, block, bits = struct.unpack_from(
                "<HHIIHH", payload
            )
            assert (kind, channels, rate, block, bits) == (3, 1, 16000, 4, 32)
        elif tag == b"data":
            data = payload
        offset = start + size + (size & 1)
    assert data is not None and len(data) % 4 == 0
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def _stages(prefix: Path) -> dict[str, dict]:
    metadata = Path(str(prefix) + ".stages.json")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    return {stage["id"]: stage for stage in payload["pipeline_stages"]}


def _flatten(frames: list[list[float]]) -> list[float]:
    return [value for frame in frames for value in frame]


def test_online_pipe_is_single_pass_with_delayed_fifo_control(tmp_path: Path) -> None:
    completed, provider = _run_with_provider(_base_command, tmp_path)
    assert completed.returncode == 0, completed.stderr

    prefix = tmp_path / "online"
    report = json.loads(Path(str(prefix) + ".json").read_text())
    stages = _stages(prefix)
    input_frames = 480 // FRAME_SAMPLES
    pipe_frames = len(provider.outputs)

    assert pipe_frames > input_frames
    assert report["ns_backend"] == "external_stream"
    assert report["ns_external_stream_enabled"] is True
    assert report["ns_external_pcm_enabled"] is False
    assert report["ns_external_applied_frames"] == pipe_frames
    assert report["ns_external_delay_samples"] == DELAY_SAMPLES
    assert report["ns_external_fallback_frames"] == 0
    assert report["ns_stream_control_alignment"] == "delayed_fifo"
    assert report["ns_stream_control_delay_samples"] == DELAY_SAMPLES
    assert report["ns_stream_native_fallback_enabled"] is False
    assert report["neural_inferences"] == 0
    assert "not a hard real-time guarantee" in report["ns_policy"]

    provider_input = _flatten(provider.inputs[:input_frames])
    provider_output = _flatten(provider.outputs)
    ns_input = _read_float32_wav(Path(stages["tx_ns_input"]["file"]))
    ns_output = _read_float32_wav(Path(stages["tx_ns"]["file"]))
    expected_delayed = [0.0] * DELAY_SAMPLES + [
        value * 0.5 for value in provider_input
    ]
    assert ns_input == provider_input
    assert ns_output == provider_output
    assert ns_output == expected_delayed

    assert stages["tx_ns_input"]["enabled"] is True
    assert stages["tx_ns"]["enabled"] is True
    assert stages["tx_ns"]["fixed_delay_samples"] == DELAY_SAMPLES
    assert stages["tx_agc"]["enabled"] is True
    assert stages["tx_eq"]["enabled"] is True
    assert stages["tx_agc"]["fixed_delay_samples"] == DELAY_SAMPLES
    assert "worker-audio-aligned" in stages["tx_agc"]["measurement_domain"]
    assert "delayed" in stages["tx_agc"]["measurement_domain"]
    agc = _read_float32_wav(Path(stages["tx_agc"]["file"]))
    eq = _read_float32_wav(Path(stages["tx_eq"]["file"]))
    assert agc != ns_output
    assert eq != agc


@pytest.mark.parametrize("mode", ["nan", "early_eof"])
def test_provider_failure_fails_closed_without_dry_fallback(
    tmp_path: Path, mode: str
) -> None:
    completed, provider = _run_with_provider(_base_command, tmp_path, mode=mode)
    assert completed.returncode != 0
    assert "processing failed" in completed.stderr
    assert provider.inputs
    assert not (tmp_path / "failure.json").exists()
    assert not (tmp_path / "failure.after.wav").exists()


def test_invalid_stream_cli_contracts_are_rejected(tmp_path: Path) -> None:
    mic = tmp_path / "mic.wav"
    _write_pcm16(mic, _pcm16_tone(320))
    prefix = tmp_path / "rejected"

    read_fd, write_fd = os.pipe()
    try:
        duplicate = _run_no_provider(
            _base_command(mic, prefix, write_fd, write_fd),
            pass_fds=(write_fd,),
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert duplicate.returncode != 0
    assert "must differ" in duplicate.stderr

    low = _run_no_provider(_base_command(mic, prefix, 2, 3))
    assert low.returncode != 0
    assert "fd >= 3" in low.stderr

    invalid = _run_no_provider(_base_command(mic, prefix, 9875, 9876))
    assert invalid.returncode != 0
    assert "fd validation failed" in invalid.stderr

    post_ns = tmp_path / "post.float.wav"
    _write_float32_wav(post_ns, [0.0] * 320)
    engine_to_provider_r, engine_to_provider_w = os.pipe()
    provider_to_engine_r, provider_to_engine_w = os.pipe()
    try:
        mutual = _run_no_provider(
            [
                *_base_command(
                    mic, prefix, engine_to_provider_w, provider_to_engine_r
                ),
                "--post-ns-pcm", str(post_ns),
            ],
            pass_fds=(engine_to_provider_w, provider_to_engine_r),
        )
    finally:
        for fd in (
            engine_to_provider_r,
            engine_to_provider_w,
            provider_to_engine_r,
            provider_to_engine_w,
        ):
            os.close(fd)
    assert mutual.returncode != 0
    assert "mutually exclusive" in mutual.stderr
