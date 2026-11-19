"""Black-box 48 kHz/480 native external-NS callback contract."""
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


ROOT = Path(__file__).resolve().parents[2]
CLI = Path(
    os.environ.get(
        "ITU_TEST_CLI",
        ROOT / "build/itu-48k-online-20261119-v1/cpp/itu_audio_cli",
    )
)
RATE = 48_000
FRAME_SAMPLES = 480
FRAME_BYTES = FRAME_SAMPLES * 4
DELAY_SAMPLES = 2_400
DELAY_FRAMES = DELAY_SAMPLES // FRAME_SAMPLES


def _read_exact(fd: int, size: int, *, allow_eof: bool = False):
    chunks = bytearray()
    while len(chunks) < size:
        block = os.read(fd, size - len(chunks))
        if not block:
            if not chunks and allow_eof:
                return None
            raise EOFError("unexpected EOF")
        chunks.extend(block)
    return bytes(chunks)


def _write_exact(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise EOFError("pipe write returned zero")
        offset += written


@dataclass
class _Provider:
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
                frame = list(struct.unpack("<480f", raw))
                if not all(math.isfinite(value) for value in frame):
                    raise ValueError("native supplied a non-finite 48 kHz frame")
                self.inputs.append(frame)
                source = delayed.popleft()
                delayed.append(frame)
                output = [value * 0.5 for value in source]
                self.outputs.append(output)
                _write_exact(write_fd, struct.pack("<480f", *output))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass


def _write_pcm16(path: Path, samples: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        values = [
            int(round(0.1 * 32768.0 * math.sin(2 * math.pi * 440 * i / RATE)))
            for i in range(samples)
        ]
        stream.writeframes(struct.pack(f"<{samples}h", *values))


def _read_stage_float(path: Path) -> list[float]:
    import soundfile as sf

    values, rate = sf.read(path, dtype="float32")
    assert rate == RATE
    return values.tolist()


def _flatten(frames: list[list[float]]) -> list[float]:
    return [value for frame in frames for value in frame]


def test_48k_aec3_online_callback_uses_480_samples_and_1920_delay(tmp_path):
    if os.name != "posix":
        pytest.skip("online NS pipes require POSIX")
    if not CLI.is_file():
        pytest.skip(f"fixed 48 kHz CLI is missing: {CLI}")
    mic = tmp_path / "mic.wav"
    echo = tmp_path / "echo.wav"
    _write_pcm16(mic, 12 * FRAME_SAMPLES)
    _write_pcm16(echo, 12 * FRAME_SAMPLES)
    prefix = tmp_path / "online"
    engine_to_provider_r, engine_to_provider_w = os.pipe()
    provider_to_engine_r, provider_to_engine_w = os.pipe()
    provider = _Provider()
    thread = threading.Thread(
        target=provider.run,
        args=(engine_to_provider_r, provider_to_engine_w),
        daemon=True,
    )
    thread.start()
    command = [
        str(CLI),
        "--profile", "vcp",
        "--aec-backend", "webrtc",
        "--ns-backend", "webrtc",
        "--mic", str(mic),
        "--echo", str(echo),
        "--no-bf",
        "--no-low-cut",
        "--no-tx-drc",
        "--no-limiter",
        "--agc-fixed-gain", "0",
        "--ns-stream-input-fd", str(engine_to_provider_w),
        "--ns-stream-output-fd", str(provider_to_engine_r),
        "--ns-stream-delay-samples", str(DELAY_SAMPLES),
        "--diagnostic-stages",
        "--out", str(prefix),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(engine_to_provider_w, provider_to_engine_r),
    )
    os.close(engine_to_provider_w)
    os.close(provider_to_engine_r)
    stdout, stderr = process.communicate(timeout=20)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert provider.error is None, provider.error
    assert process.returncode == 0, stderr
    assert provider.inputs

    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    assert report["sample_rate"] == RATE
    assert report["frame_size"] == FRAME_SAMPLES
    assert report["ns_backend"] == "external_stream"
    assert report["ns_external_stream_enabled"] is True
    assert report["ns_external_pcm_enabled"] is False
    assert report["ns_external_delay_samples"] == DELAY_SAMPLES
    assert report["ns_external_applied_frames"] == len(provider.outputs)
    assert report["ns_external_fallback_frames"] == 0
    assert report["ns_stream_frame_samples"] == FRAME_SAMPLES
    assert report["ns_stream_chunk_samples"] == FRAME_SAMPLES
    assert report["ns_stream_sample_rate"] == RATE
    assert report["ns_stream_transport"].endswith("float32[480]")
    assert report["internal_sample_rate"] == RATE
    assert report["ns_internal_sample_rate"] == RATE
    assert report["rate_converter"] == "none"
    assert report["aec_backend_effective"] == "webrtc_aec3"

    stages = json.loads(Path(str(prefix) + ".stages.json").read_text(encoding="utf-8"))
    stage_by_id = {item["id"]: item for item in stages["pipeline_stages"]}
    tx_ns = _read_stage_float(Path(stage_by_id["tx_ns"]["file"]))
    expected = _flatten(provider.outputs)
    assert len(tx_ns) <= len(expected)
    assert tx_ns == expected[: len(tx_ns)]
    assert expected[:DELAY_SAMPLES] == [0.0] * DELAY_SAMPLES
