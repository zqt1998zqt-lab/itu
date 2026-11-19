"""Real CLI EOS contract for LocalVQE followed by an external NS drain."""
from __future__ import annotations

from collections import deque
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf

from server import ns_routes


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build/cofilite-opt-20261116/host-v3/cpp/itu_audio_cli"
LIBRARY = ROOT / "build/localvqe-host-20261010/liblocalvqe.so.0.1.0"
MODEL = ROOT / "build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf"
LIBRARY_SHA256 = (
    "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60"
)
MODEL_SHA256 = (
    "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
)
FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * 4
HOP_SAMPLES = 256
LOCALVQE_CODEC_DELAY_SAMPLES = 256
LOCALVQE_ADAPTER_DELAY_SAMPLES = 224
NS_DELAY_SAMPLES = ns_routes.NS_ROUTES["cofilite"].delay_samples
NS_DELAY_FRAMES = NS_DELAY_SAMPLES // FRAME_SAMPLES
assert NS_DELAY_SAMPLES > 0 and NS_DELAY_SAMPLES % FRAME_SAMPLES == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeContext:
    def __init__(self, library: Path, model: Path) -> None:
        self.library = ctypes.CDLL(str(library))
        self.library.localvqe_options_new.restype = ctypes.c_void_p
        self.library.localvqe_options_free.argtypes = [ctypes.c_void_p]
        self.library.localvqe_options_set_model_path.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.library.localvqe_options_set_model_path.restype = ctypes.c_int
        self.library.localvqe_options_set_threads.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.library.localvqe_options_set_threads.restype = ctypes.c_int
        self.library.localvqe_new_with_options.argtypes = [ctypes.c_void_p]
        self.library.localvqe_new_with_options.restype = ctypes.c_void_p
        self.library.localvqe_free.argtypes = [ctypes.c_void_p]
        self.library.localvqe_sample_rate.argtypes = [ctypes.c_void_p]
        self.library.localvqe_sample_rate.restype = ctypes.c_int
        self.library.localvqe_hop_length.argtypes = [ctypes.c_void_p]
        self.library.localvqe_hop_length.restype = ctypes.c_int
        self.library.localvqe_fft_size.argtypes = [ctypes.c_void_p]
        self.library.localvqe_fft_size.restype = ctypes.c_int
        self.library.localvqe_set_noise_gate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self.library.localvqe_set_noise_gate.restype = ctypes.c_int
        self.library.localvqe_process_frame_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.localvqe_process_frame_f32.restype = ctypes.c_int
        self.library.localvqe_last_error.argtypes = [ctypes.c_void_p]
        self.library.localvqe_last_error.restype = ctypes.c_char_p

        options = self.library.localvqe_options_new()
        if not options:
            raise RuntimeError("LocalVQE options allocation failed")
        try:
            if self.library.localvqe_options_set_model_path(
                options, str(model).encode()
            ):
                raise RuntimeError("LocalVQE model path rejected")
            if self.library.localvqe_options_set_threads(options, 1):
                raise RuntimeError("LocalVQE thread count rejected")
            self.handle = self.library.localvqe_new_with_options(options)
        finally:
            self.library.localvqe_options_free(options)
        if not self.handle:
            raise RuntimeError("LocalVQE context creation failed")
        try:
            contract = (
                self.library.localvqe_sample_rate(self.handle),
                self.library.localvqe_hop_length(self.handle),
                self.library.localvqe_fft_size(self.handle),
            )
            if contract != (16_000, HOP_SAMPLES, 512):
                raise RuntimeError(f"LocalVQE contract changed: {contract}")
            if self.library.localvqe_set_noise_gate(self.handle, 0, -90.0):
                raise RuntimeError("LocalVQE noise-gate control rejected")
        except Exception:
            self.close()
            raise

    def process(self, microphone: np.ndarray, reference: np.ndarray) -> np.ndarray:
        output = np.empty(HOP_SAMPLES, dtype=np.float32)
        status = self.library.localvqe_process_frame_f32(
            self.handle,
            microphone.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            reference.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            HOP_SAMPLES,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status:
            detail = self.library.localvqe_last_error(self.handle)
            raise RuntimeError(
                f"LocalVQE process_frame failed: {status} "
                f"{detail.decode(errors='replace') if detail else ''}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError("LocalVQE returned a non-finite hop")
        return output

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.library.localvqe_free(handle)
            self.handle = None

    def __enter__(self) -> "NativeContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _beam_input(
    microphone: np.ndarray, bf_delay_samples: int | None
) -> np.ndarray:
    microphone = np.asarray(microphone, dtype=np.float32)
    if bf_delay_samples is None:
        return microphone[:, 0] if microphone.ndim == 2 else microphone

    tail_samples = int(math.ceil(bf_delay_samples))
    microphone = np.concatenate(
        [
            microphone,
            np.zeros((tail_samples, microphone.shape[1]), dtype=np.float32),
        ],
        axis=0,
    )
    size = bf_delay_samples + 2
    history = np.zeros((size, microphone.shape[1]), dtype=np.float32)
    output = np.empty(len(microphone), dtype=np.float32)
    head = 0
    delays = (float(bf_delay_samples), 0.0)
    for index in range(len(microphone)):
        history[head] = microphone[index]
        aligned = []
        for channel, delay in enumerate(delays):
            integer = int(delay)
            fraction = delay - integer
            current = (head + size - integer) % size
            previous = (current + size - 1) % size
            aligned.append(
                (1.0 - fraction) * history[current, channel]
                + fraction * history[previous, channel]
            )
        output[index] = sum(aligned) / len(aligned)
        head = (head + 1) % size
    return output


def _native_eos_tail_oracle(
    microphone_path: Path, bf_delay_samples: int | None
) -> tuple[np.ndarray, np.ndarray]:
    assert LIBRARY.is_file() and MODEL.is_file()
    assert _sha256(LIBRARY) == LIBRARY_SHA256
    assert _sha256(MODEL) == MODEL_SHA256
    microphone, rate = sf.read(microphone_path, dtype="float32")
    assert rate == 16_000
    aec_input = _beam_input(microphone, bf_delay_samples)
    calls = (
        len(aec_input) + LOCALVQE_CODEC_DELAY_SAMPLES + HOP_SAMPLES - 1
    ) // HOP_SAMPLES
    padded_mic = np.zeros(calls * HOP_SAMPLES, dtype=np.float32)
    padded_ref = np.zeros(calls * HOP_SAMPLES, dtype=np.float32)
    padded_mic[: len(aec_input)] = aec_input
    direct = np.empty(calls * HOP_SAMPLES, dtype=np.float32)
    with NativeContext(LIBRARY, MODEL) as context:
        for index in range(calls):
            offset = index * HOP_SAMPLES
            direct[offset : offset + HOP_SAMPLES] = context.process(
                padded_mic[offset : offset + HOP_SAMPLES],
                padded_ref[offset : offset + HOP_SAMPLES],
            )
    expected = np.concatenate(
        [
            np.zeros(LOCALVQE_ADAPTER_DELAY_SAMPLES, dtype=np.float32),
            direct,
        ]
    )
    return direct, expected


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


class CountingDelayedProvider:
    def __init__(self) -> None:
        self.inputs: list[list[float]] = []
        self.outputs: list[list[float]] = []
        self.error: str | None = None

    def run(self, read_fd: int, write_fd: int) -> None:
        delayed = deque(
            [[0.0] * FRAME_SAMPLES for _ in range(NS_DELAY_FRAMES)]
        )
        try:
            while True:
                raw = _read_exact(read_fd, FRAME_BYTES, allow_eof=True)
                if raw is None:
                    return
                frame = list(struct.unpack("<160f", raw))
                if not all(math.isfinite(value) for value in frame):
                    raise ValueError("LocalVQE supplied a non-finite NS frame")
                self.inputs.append(frame)
                delayed.append(frame)
                source = delayed.popleft()
                output = [value * 0.5 for value in source]
                _write_exact(write_fd, struct.pack("<160f", *output))
                self.outputs.append(output)
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass


def _write_inputs(
    tmp_path: Path, samples: int, bf_delay_samples: int | None
) -> tuple[Path, Path]:
    time = np.arange(samples, dtype=np.float64) / 16_000.0
    speech = (
        0.18 * np.sin(2.0 * np.pi * 437.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 91.0 * time)
    )
    if bf_delay_samples is None:
        microphone = speech
    else:
        microphone = np.column_stack(
            [speech, np.roll(speech, bf_delay_samples)]
        )
    microphone_path = tmp_path / "microphone.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(microphone_path, microphone, 16_000, subtype="PCM_16")
    sf.write(
        reference_path,
        np.zeros(samples, dtype=np.float32),
        16_000,
        subtype="PCM_16",
    )
    return microphone_path, reference_path


def _command(
    microphone: Path,
    reference: Path,
    prefix: Path,
    input_fd: int,
    output_fd: int,
    bf_delay_samples: int | None,
) -> list[str]:
    command = [
        str(CLI),
        "--profile",
        "general",
        "--aec-backend",
        "localvqe_aec200k",
        "--mic",
        str(microphone),
        "--echo",
        str(reference),
        "--no-rx-processing",
        "--no-low-cut",
        "--no-agc",
        "--no-tx-drc",
        "--no-limiter",
        "--ns-stream-input-fd",
        str(input_fd),
        "--ns-stream-output-fd",
        str(output_fd),
        "--ns-stream-delay-samples",
        str(NS_DELAY_SAMPLES),
        "--out",
        str(prefix),
    ]
    if bf_delay_samples is None:
        command.extend(["--no-bf"])
    else:
        command.extend(["--beam-delays", f"{bf_delay_samples},0"])
    return command


def _run_case(
    tmp_path: Path, samples: int, bf_delay_samples: int | None
) -> tuple[subprocess.CompletedProcess[str], CountingDelayedProvider, Path]:
    assert CLI.is_file(), f"fixed host-v3 candidate CLI is missing: {CLI}"
    microphone, reference = _write_inputs(
        tmp_path, samples, bf_delay_samples
    )
    prefix = tmp_path / "eos"
    engine_to_provider_r, engine_to_provider_w = os.pipe()
    provider_to_engine_r, provider_to_engine_w = os.pipe()
    provider = CountingDelayedProvider()
    thread = threading.Thread(
        target=provider.run,
        args=(engine_to_provider_r, provider_to_engine_w),
        name="localvqe-eos-ns-provider",
        daemon=True,
    )
    thread.start()
    command = _command(
        microphone,
        reference,
        prefix,
        engine_to_provider_w,
        provider_to_engine_r,
        bf_delay_samples,
    )
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
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        raise
    thread.join(timeout=10)
    assert not thread.is_alive(), "NS provider did not terminate"
    assert provider.error is None, provider.error
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        provider,
        prefix,
    )


def test_localvqe_near_only_impulse_peak_is_exact_physical_480(tmp_path):
    assert CLI.is_file(), f"fixed host-v3 candidate CLI is missing: {CLI}"
    samples = 4_096
    impulse_sample = 2_048
    microphone = np.zeros(samples, dtype=np.float32)
    microphone[impulse_sample] = 0.5
    microphone_path = tmp_path / "impulse.wav"
    reference_path = tmp_path / "silence.wav"
    prefix = tmp_path / "impulse-result"
    sf.write(microphone_path, microphone, 16_000, subtype="PCM_16")
    sf.write(
        reference_path,
        np.zeros(samples, dtype=np.float32),
        16_000,
        subtype="PCM_16",
    )
    completed = subprocess.run(
        [
            str(CLI),
            "--profile",
            "general",
            "--aec-backend",
            "localvqe_aec200k",
            "--mic",
            str(microphone_path),
            "--echo",
            str(reference_path),
            "--no-rx-processing",
            "--no-bf",
            "--no-ns",
            "--no-low-cut",
            "--no-agc",
            "--no-tx-drc",
            "--no-limiter",
            "--out",
            str(prefix),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(Path(str(prefix) + ".json").read_text())
    aec, rate = sf.read(Path(str(prefix) + ".aec.wav"), dtype="float32")
    after, after_rate = sf.read(
        Path(str(prefix) + ".after.wav"), dtype="float32"
    )
    assert rate == after_rate == 16_000
    if aec.ndim == 2:
        aec = aec[:, 0]

    expected_peak = impulse_sample + 480
    assert int(np.argmax(np.abs(aec))) == expected_peak
    assert int(np.argmax(np.abs(after))) == expected_peak
    assert report["aec_neural_delay_samples"] == LOCALVQE_CODEC_DELAY_SAMPLES
    assert report["aec_hop_adapter_delay_samples_16k"] == (
        LOCALVQE_ADAPTER_DELAY_SAMPLES
    )
    assert report["aec_algorithm_delay_samples_16k"] == 480
    assert report["aec_model_codec_delay_samples_16k"] == 256
    assert report["aec_linear_frontend_delay_samples_16k"] == 0
    assert report["aec_declared_source_samples"] == samples
    assert report["aec_native_hop_samples"] == HOP_SAMPLES
    expected_hops = (
        samples + LOCALVQE_CODEC_DELAY_SAMPLES + HOP_SAMPLES - 1
    ) // HOP_SAMPLES
    assert report["aec_localvqe_inferences"] == expected_hops
    assert report["aec_native_consumed_samples"] == expected_hops * HOP_SAMPLES
    assert aec.shape == (samples + 480,)
    assert after.shape == (samples + report["dsp_delay_samples"],)


@pytest.mark.parametrize(
    ("samples", "bf_delay_samples", "expected_inferences"),
    [
        (16_000, None, 64),
        (192_000, None, 751),
        (16_001, None, 64),
        (16_000, 1, 64),
        (192_000, 1, 752),
    ],
    ids=[
        "1s-mono",
        "12s-mono",
        "1s-plus-one-sample-mono",
        "1s-bf-delay-1",
        "12s-bf-delay-1",
    ],
)
def test_localvqe_declared_eos_keeps_ns_draining_without_model_calls(
    tmp_path, samples, bf_delay_samples, expected_inferences
):
    completed, provider, prefix = _run_case(
        tmp_path, samples, bf_delay_samples
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads(Path(str(prefix) + ".json").read_text())
    with Path(str(prefix) + ".frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    inferences = [int(row["neural_inferences"]) for row in rows]

    input_callbacks = (samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    expected_hop_input = (
        samples
        + (bf_delay_samples or 0)
        + LOCALVQE_CODEC_DELAY_SAMPLES
    )
    computed_inferences = (
        expected_hop_input + 255
    ) // 256
    assert computed_inferences == expected_inferences
    assert report["aec_localvqe_inferences"] == expected_inferences
    assert report["aec_declared_source_samples"] == samples
    assert report["aec_native_hop_samples"] == HOP_SAMPLES
    assert report["aec_native_consumed_samples"] == (
        expected_inferences * HOP_SAMPLES
    )
    assert report["aec_neural_delay_samples"] == LOCALVQE_CODEC_DELAY_SAMPLES
    assert report["aec_hop_adapter_delay_samples_16k"] == (
        LOCALVQE_ADAPTER_DELAY_SAMPLES
    )
    assert report["aec_algorithm_delay_samples_16k"] == 480
    assert report["aec_model_codec_delay_samples_16k"] == (
        LOCALVQE_CODEC_DELAY_SAMPLES
    )
    assert report["aec_linear_frontend_delay_samples_16k"] == 0
    assert report["aec_native_consumed_samples"] >= expected_hop_input
    assert report["aec_native_consumed_samples"] < expected_hop_input + HOP_SAMPLES
    assert report["aec_localvqe_processed_frames"] == len(rows)
    assert report["aec_route_processed_frames"] == len(rows)
    assert report["aec_localvqe_fallback_frames"] == 0
    assert len(rows) > input_callbacks

    first_final_inference = next(
        index for index, value in enumerate(inferences)
        if value == expected_inferences
    )
    assert any(
        value < expected_inferences
        for value in inferences[:first_final_inference]
    )
    assert all(
        value == expected_inferences
        for value in inferences[first_final_inference:]
    )
    assert first_final_inference < len(rows) - 1

    assert report["ns_external_stream_enabled"] is True
    assert report["ns_external_fallback_frames"] == 0
    assert report["ns_external_applied_frames"] == len(rows)
    assert len(provider.inputs) == len(rows)
    assert len(provider.outputs) == len(rows)

    after, rate = sf.read(Path(str(prefix) + ".after.wav"), dtype="float32")
    aec, aec_rate = sf.read(
        Path(str(prefix) + ".aec.wav"), dtype="float32"
    )
    assert rate == aec_rate == 16_000
    assert after.shape == (samples + report["dsp_delay_samples"],)
    assert aec.shape[0] == samples + report["aec_output_delay_samples"]
    if aec.ndim == 2:
        aec = aec[:, 0]
    assert np.isfinite(after).all()
    assert np.isfinite(aec).all()
    assert np.max(np.abs(after[-min(320, len(after)):])) > 1e-4
    assert np.max(np.abs(aec[-min(320, len(aec)):])) > 1e-4

    direct, expected_aec = _native_eos_tail_oracle(
        tmp_path / "microphone.wav", bf_delay_samples
    )
    aec_count = samples + report["aec_output_delay_samples"]
    assert len(direct) == expected_inferences * HOP_SAMPLES
    actual_tail = aec[-min(LOCALVQE_CODEC_DELAY_SAMPLES, aec_count):]
    expected_tail = expected_aec[:aec_count][-len(actual_tail):]
    assert np.sqrt(np.mean(expected_tail * expected_tail)) > 1e-4
    assert np.sqrt(np.mean(actual_tail * actual_tail)) > 1e-4
    correlation = np.corrcoef(actual_tail, expected_tail)[0, 1]
    tail_rmse = np.sqrt(np.mean((actual_tail - expected_tail) ** 2))
    assert correlation > 0.98, (
        samples,
        bf_delay_samples,
        correlation,
        tail_rmse,
    )
    assert tail_rmse < 0.03, (
        samples,
        bf_delay_samples,
        correlation,
        tail_rmse,
    )
