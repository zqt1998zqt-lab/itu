"""Native LocalVQE oracle for the continuous 160/256 hop adapter."""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build/localvqe-host-20261010/liblocalvqe.so.0.1.0"
MODEL = ROOT / "build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf"
LINEAR_MODEL = (
    ROOT / "build/deps/localvqe-models/localvqe-v1.4-aec-2.7K-f32.gguf"
)
LIBRARY_SHA256 = "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60"
MODEL_SHA256 = "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
LINEAR_MODEL_SHA256 = (
    "d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4"
)
CALLBACK_SAMPLES = 160
HOP_SAMPLES = 256
ADAPTER_DELAY = 224
MODEL_CODEC_DELAY = 256
INPUT_BLOCK_REQUIREMENT = 256
AEC_PCM_TRIM_DELAY = MODEL_CODEC_DELAY + ADAPTER_DELAY


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
            if contract != (16000, HOP_SAMPLES, 512):
                raise RuntimeError(f"LocalVQE contract changed: {contract}")
            if self.library.localvqe_set_noise_gate(self.handle, 0, -90.0):
                raise RuntimeError("LocalVQE noise-gate control rejected")
        except Exception:
            self.close()
            raise

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        output = np.empty(HOP_SAMPLES, dtype=np.float32)
        status = self.library.localvqe_process_frame_f32(
            self.handle,
            mic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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
            raise RuntimeError("LocalVQE returned a non-finite native hop")
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


NATIVE_ADAPTER_PROBE = r"""
#include "continuous_hop_adapter.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Handle = std::uintptr_t;
using OptionsNew = Handle (*)();
using OptionsFree = void (*)(Handle);
using SetModel = int (*)(Handle, const char*);
using SetThreads = int (*)(Handle, int);
using NewWithOptions = Handle (*)(Handle);
using Free = void (*)(Handle);
using ProcessFrame = int (*)(Handle, const float*, const float*, int, float*);
using SampleRate = int (*)(Handle);
using HopLength = int (*)(Handle);
using FftSize = int (*)(Handle);
using LastError = const char* (*)(Handle);
using SetNoiseGate = int (*)(Handle, int, float);

template <class T>
T symbol(void* library, const char* name) {
  const auto value = dlsym(library, name);
  if (!value) throw std::runtime_error(std::string("missing symbol: ") + name);
  return reinterpret_cast<T>(value);
}

std::vector<float> read_float32(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open input");
  const auto bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)))
    throw std::runtime_error("invalid float32 input size");
  input.seekg(0);
  std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!input) throw std::runtime_error("cannot read float32 input");
  return values;
}

void write_float32(const char* path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot open float32 output");
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output) throw std::runtime_error("cannot write float32 output");
}

void write_metadata(const char* path, unsigned callbacks,
                    std::uint64_t inferences, unsigned pending,
                    unsigned queued, unsigned output_samples,
                    unsigned valid_output_samples,
                    std::uint64_t source_samples,
                    std::uint64_t target_hops,
                    std::uint64_t inferences_before_flush,
                    unsigned downstream_flush_callbacks,
                    unsigned native_tail_callbacks,
                    unsigned max_hops_per_callback,
                    std::uint64_t input_samples_after_tail) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open metadata output");
  output << "{\"delay_samples\":"
         << itu::ContinuousHopAdapter::kDelaySamples
         << ",\"callbacks\":" << callbacks
         << ",\"inferences\":" << inferences
         << ",\"pending\":" << pending
         << ",\"queued\":" << queued
         << ",\"output_samples\":" << output_samples
         << ",\"valid_output_samples\":" << valid_output_samples
         << ",\"source_samples\":" << source_samples
         << ",\"target_hops\":" << target_hops
         << ",\"inferences_before_flush\":" << inferences_before_flush
         << ",\"downstream_flush_callbacks\":" << downstream_flush_callbacks
         << ",\"native_tail_callbacks\":" << native_tail_callbacks
         << ",\"max_hops_per_callback\":" << max_hops_per_callback
         << ",\"input_samples_after_tail\":" << input_samples_after_tail
         << ",\"native_hop_samples\":" << itu::ContinuousHopAdapter::kModelHopSamples
         << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage: probe LIB MODEL MIC REF OUTPUT META SOURCE_SAMPLES FLUSH_CALLBACKS\n";
    return 2;
  }
  try {
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    const auto options_new = symbol<OptionsNew>(library, "localvqe_options_new");
    const auto options_free = symbol<OptionsFree>(library, "localvqe_options_free");
    const auto set_model =
        symbol<SetModel>(library, "localvqe_options_set_model_path");
    const auto set_threads =
        symbol<SetThreads>(library, "localvqe_options_set_threads");
    const auto new_with_options =
        symbol<NewWithOptions>(library, "localvqe_new_with_options");
    const auto free_context = symbol<Free>(library, "localvqe_free");
    const auto process_frame =
        symbol<ProcessFrame>(library, "localvqe_process_frame_f32");
    const auto sample_rate = symbol<SampleRate>(library, "localvqe_sample_rate");
    const auto hop_length = symbol<HopLength>(library, "localvqe_hop_length");
    const auto fft_size = symbol<FftSize>(library, "localvqe_fft_size");
    const auto last_error = symbol<LastError>(library, "localvqe_last_error");
    const auto set_noise_gate =
        symbol<SetNoiseGate>(library, "localvqe_set_noise_gate");

    auto mic = read_float32(argv[3]);
    auto ref = read_float32(argv[4]);
    if (mic.size() != ref.size() ||
        mic.size() % itu::ContinuousHopAdapter::kCallbackSamples)
      throw std::runtime_error("input sizes are not equal 160-sample callbacks");

    Handle options = options_new();
    if (!options) throw std::runtime_error("options allocation failed");
    Handle context = 0;
    try {
      if (set_model(options, argv[2])) throw std::runtime_error("model rejected");
      if (set_threads(options, 1)) throw std::runtime_error("thread count rejected");
      context = new_with_options(options);
    } catch (...) {
      options_free(options);
      throw;
    }
    options_free(options);
    if (!context) throw std::runtime_error("context creation failed");
    try {
      if (sample_rate(context) != 16000 ||
          hop_length(context) !=
              static_cast<int>(itu::ContinuousHopAdapter::kModelHopSamples) ||
          fft_size(context) != 512 ||
          set_noise_gate(context, 0, -90.f))
        throw std::runtime_error("native LocalVQE contract mismatch");

      itu::ContinuousHopAdapter adapter;
      const auto source_samples =
          static_cast<std::uint64_t>(std::stoull(argv[7]));
      const auto downstream_flush_callbacks =
          static_cast<unsigned>(std::stoul(argv[8]));
      const auto target_hops =
          source_samples
              ? (source_samples + 256 +
                 itu::ContinuousHopAdapter::kModelHopSamples - 1) /
                    itu::ContinuousHopAdapter::kModelHopSamples
              : 0u;
      const unsigned callbacks =
          static_cast<unsigned>(mic.size()) /
          itu::ContinuousHopAdapter::kCallbackSamples;
      const auto expected_hops =
          source_samples
              ? target_hops
              : (mic.size() +
                 itu::ContinuousHopAdapter::kModelHopSamples - 1) /
                    itu::ContinuousHopAdapter::kModelHopSamples;
      const unsigned valid_output_samples = static_cast<unsigned>(
          adapter.delay_samples() +
          expected_hops * itu::ContinuousHopAdapter::kModelHopSamples);
      const unsigned output_capacity =
          valid_output_samples +
          itu::ContinuousHopAdapter::kCallbackSamples - 1 +
          downstream_flush_callbacks *
              itu::ContinuousHopAdapter::kCallbackSamples;
      std::vector<float> output(output_capacity, 0.f);
      unsigned written = 0;
      auto native_hop = [&](const float* microphone, const float* reference,
                            float* native_output) {
        if (process_frame(context, microphone, reference,
                          static_cast<int>(
                              itu::ContinuousHopAdapter::kModelHopSamples),
                          native_output))
          return false;
        return std::all_of(
            native_output,
            native_output + itu::ContinuousHopAdapter::kModelHopSamples,
            [](float value) { return std::isfinite(value); });
      };
      for (unsigned callback = 0; callback < callbacks; ++callback) {
        const auto offset =
            callback * itu::ContinuousHopAdapter::kCallbackSamples;
        if (!adapter.process(mic.data() + offset, ref.data() + offset,
                             output.data() + written, native_hop))
          throw std::runtime_error("adapter callback failed");
        written += itu::ContinuousHopAdapter::kCallbackSamples;
      }
      if (source_samples) {
        if (adapter.inferences() > target_hops)
          throw std::runtime_error("source callbacks exceeded target hops");
      } else if (!adapter.finish(native_hop)) {
        throw std::runtime_error("adapter EOS finish failed");
      }
      unsigned native_tail_callbacks = 0;
      unsigned max_hops_per_callback = 0;
      std::array<float, itu::ContinuousHopAdapter::kCallbackSamples> zero{};
      if (source_samples) {
        while (adapter.inferences() < target_hops) {
          const auto before = adapter.inferences();
          if (written + zero.size() > output.size())
            throw std::runtime_error("native tail exceeded output capacity");
          if (!adapter.process(zero.data(), zero.data(),
                               output.data() + written, native_hop))
            throw std::runtime_error("native-tail zero callback failed");
          const auto advanced = adapter.inferences() - before;
          if (advanced > 1)
            throw std::runtime_error("zero callback advanced more than one hop");
          max_hops_per_callback = std::max(
              max_hops_per_callback, static_cast<unsigned>(advanced));
          written += itu::ContinuousHopAdapter::kCallbackSamples;
          ++native_tail_callbacks;
          if (native_tail_callbacks > 4)
            throw std::runtime_error("native tail exceeded callback bound");
        }
        if (!adapter.finish_after_hops(target_hops))
          throw std::runtime_error("adapter target-limited freeze failed");
      }
      const auto input_samples_after_tail = adapter.input_samples();
      while (!adapter.drained()) {
        if (written + itu::ContinuousHopAdapter::kCallbackSamples >
            output.size())
          throw std::runtime_error("adapter drain exceeded output capacity");
        if (!adapter.drain(output.data() + written))
          throw std::runtime_error("adapter drain failed");
        written += itu::ContinuousHopAdapter::kCallbackSamples;
      }
      const auto inferences_before_flush = adapter.inferences();
      std::vector<float> flush(itu::ContinuousHopAdapter::kCallbackSamples);
      for (unsigned callback = 0;
           callback < downstream_flush_callbacks; ++callback) {
        if (written + flush.size() > output.size())
          throw std::runtime_error("downstream flush exceeded output capacity");
        if (!adapter.drain(output.data() + written))
          throw std::runtime_error("downstream flush callback failed");
        if (!std::all_of(output.begin() + written,
                         output.begin() + written + flush.size(),
                         [](float value) { return value == 0.f; }))
          throw std::runtime_error("downstream flush output is not zero");
        written += itu::ContinuousHopAdapter::kCallbackSamples;
      }
      if (adapter.inferences() != inferences_before_flush)
        throw std::runtime_error("downstream flush advanced native state");
      output.resize(written);
      write_float32(argv[5], output);
      write_metadata(argv[6], callbacks, adapter.inferences(),
                     adapter.pending(), adapter.queued(), written,
                     valid_output_samples, source_samples, target_hops,
                     inferences_before_flush, downstream_flush_callbacks,
                     native_tail_callbacks, max_hops_per_callback,
                     input_samples_after_tail);
      free_context(context);
      dlclose(library);
      return 0;
    } catch (...) {
      free_context(context);
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def pinned_assets() -> tuple[Path, Path, Path]:
    if not LIBRARY.is_file() or not MODEL.is_file() or not LINEAR_MODEL.is_file():
        pytest.skip("fixed LocalVQE assets are unavailable")
    assert _sha256(LIBRARY) == LIBRARY_SHA256
    assert _sha256(MODEL) == MODEL_SHA256
    assert _sha256(LINEAR_MODEL) == LINEAR_MODEL_SHA256
    return LIBRARY, MODEL, LINEAR_MODEL


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("a C++20 compiler is unavailable")
    directory = tmp_path_factory.mktemp("localvqe-native-adapter")
    source = directory / "native_adapter_probe.cpp"
    binary = directory / "native_adapter_probe"
    source.write_text(NATIVE_ADAPTER_PROBE, encoding="utf-8")
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            f"-I{ROOT / 'cpp/src'}",
            str(source),
            "-ldl",
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    return binary


def _signal(kind: str, samples: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "impulse":
        mic = np.zeros(samples, dtype=np.float32)
        ref = np.zeros(samples, dtype=np.float32)
        mic[samples // 3] = 0.5
        return mic, ref
    if kind == "random":
        generator = np.random.default_rng(20261116)
        mic = np.ascontiguousarray(
            0.05 * generator.standard_normal(samples), dtype=np.float32
        )
        return mic, np.zeros(samples, dtype=np.float32)
    time = np.arange(samples, dtype=np.float64) / 16000.0
    mic = (
        0.07 * np.sin(2.0 * np.pi * 437.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 91.0 * time)
    )
    ref = (
        0.05 * np.sin(2.0 * np.pi * 233.0 * time)
        + 0.015 * np.sin(2.0 * np.pi * 71.0 * time)
    )
    mic[0] += 0.2
    ref[1] += 0.15
    return (
        np.ascontiguousarray(mic, dtype=np.float32),
        np.ascontiguousarray(ref, dtype=np.float32),
    )


def _direct_oracle(
    library: Path,
    model: Path,
    mic: np.ndarray,
    ref: np.ndarray,
    blocks: int | None = None,
) -> np.ndarray:
    blocks = blocks or (len(mic) + HOP_SAMPLES - 1) // HOP_SAMPLES
    padded_mic = np.zeros(blocks * HOP_SAMPLES, dtype=np.float32)
    padded_ref = np.zeros(blocks * HOP_SAMPLES, dtype=np.float32)
    padded_mic[: len(mic)] = mic
    padded_ref[: len(ref)] = ref
    output = np.empty(blocks * HOP_SAMPLES, dtype=np.float32)
    with NativeContext(library, model) as context:
        for block in range(blocks):
            offset = block * HOP_SAMPLES
            output[offset : offset + HOP_SAMPLES] = context.process(
                padded_mic[offset : offset + HOP_SAMPLES],
                padded_ref[offset : offset + HOP_SAMPLES],
            )
    return output


def _measured_waveform_lag(
    kind: str, mic: np.ndarray, output: np.ndarray
) -> tuple[int, float]:
    if kind == "impulse":
        impulse = int(np.flatnonzero(mic)[0])
        peak = int(np.argmax(np.abs(output)))
        return peak - impulse, float(abs(output[peak]))
    correlations = []
    for lag in range(0, HOP_SAMPLES * 2 + 1):
        correlations.append(
            float(np.dot(mic[: len(mic) - lag], output[lag : len(mic)]))
        )
    best = int(np.argmax(correlations))
    return best, correlations[best]


@pytest.mark.parametrize("model_name", ["full", "linear_daf"])
@pytest.mark.parametrize(
    ("kind", "samples"),
    [
        ("tone", 800),
        ("tone", 1280),
        ("impulse", 1280),
        ("random", 1280),
    ],
)
def test_native_direct_256_and_adapter_integration_agree(
    pinned_assets: tuple[Path, Path, Path],
    native_probe: Path,
    tmp_path: Path,
    model_name: str,
    kind: str,
    samples: int,
) -> None:
    library, full_model, linear_model = pinned_assets
    model = full_model if model_name == "full" else linear_model
    mic, ref = _signal(kind, samples)
    mic_path = tmp_path / "mic.f32"
    ref_path = tmp_path / "ref.f32"
    output_path = tmp_path / "adapter.f32"
    metadata_path = tmp_path / "adapter.json"
    mic.tofile(mic_path)
    ref.tofile(ref_path)

    completed = subprocess.run(
        [
            str(native_probe),
            str(library),
            str(model),
            str(mic_path),
            str(ref_path),
            str(output_path),
            str(metadata_path),
            "0",
            "0",
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    direct = _direct_oracle(library, model, mic, ref)
    inferences = (samples + HOP_SAMPLES - 1) // HOP_SAMPLES
    valid_samples = ADAPTER_DELAY + inferences * HOP_SAMPLES
    expected = np.concatenate(
        [np.zeros(ADAPTER_DELAY, dtype=np.float32), direct]
    )[:valid_samples]
    actual = np.fromfile(output_path, dtype=np.float32)

    assert metadata == {
        "delay_samples": ADAPTER_DELAY,
        "callbacks": samples // CALLBACK_SAMPLES,
        "inferences": inferences,
        "pending": 0,
        "queued": 0,
        "output_samples": (
            (valid_samples + CALLBACK_SAMPLES - 1) // CALLBACK_SAMPLES
        )
        * CALLBACK_SAMPLES,
        "valid_output_samples": valid_samples,
        "source_samples": 0,
        "target_hops": 0,
        "inferences_before_flush": inferences,
        "downstream_flush_callbacks": 0,
        "native_tail_callbacks": 0,
        "max_hops_per_callback": 0,
        "input_samples_after_tail": samples,
        "native_hop_samples": HOP_SAMPLES,
    }
    assert np.array_equal(actual[:valid_samples], expected)
    assert np.all(actual[valid_samples:] == 0.0)
    assert np.max(np.abs(actual[:valid_samples] - expected), initial=0.0) == 0.0


@pytest.mark.parametrize("model_name", ["full", "linear_daf"])
@pytest.mark.parametrize("kind", ["impulse", "random"])
def test_physical_core_split_is_measured_on_near_only(
    pinned_assets: tuple[Path, Path, Path],
    tmp_path: Path,
    model_name: str,
    kind: str,
) -> None:
    library, full_model, linear_model = pinned_assets
    model = full_model if model_name == "full" else linear_model
    samples = 1536
    mic, ref = _signal(kind, samples)
    direct = _direct_oracle(library, model, mic, ref)
    waveform_lag, score = _measured_waveform_lag(kind, mic, direct)

    # The API can emit nothing until a complete 256-sample hop has arrived.
    # That block-availability latency is real for device release, but it is
    # not a PCM sample-index shift and must not be used as a WAV trim.
    expected_waveform_lag = 0 if model_name == "linear_daf" else HOP_SAMPLES
    pcm_trim_delay = waveform_lag
    adapter_pcm_trim_delay = waveform_lag + ADAPTER_DELAY
    device_release_latency = INPUT_BLOCK_REQUIREMENT + adapter_pcm_trim_delay
    assert waveform_lag == expected_waveform_lag
    assert pcm_trim_delay == (
        0 if model_name == "linear_daf" else MODEL_CODEC_DELAY
    )

    evidence = {
        "model": model_name,
        "stimulus": kind,
        "near_only": True,
        "native_hop_samples": HOP_SAMPLES,
        "input_block_requirement": INPUT_BLOCK_REQUIREMENT,
        "daf_hop_buffer_delay": HOP_SAMPLES,
        "daf_waveform_lag": waveform_lag if model_name == "linear_daf" else 0,
        "codec_waveform_lag": waveform_lag if model_name == "full" else 0,
        "measured_waveform_lag": waveform_lag,
        "measurement_score": score,
        "pcm_trim_delay_without_adapter": pcm_trim_delay,
        "adapter_pcm_trim_delay": adapter_pcm_trim_delay,
        "device_release_latency_non_pcm": device_release_latency,
        "declared_aec_pcm_trim_delay": AEC_PCM_TRIM_DELAY,
        "adapter_delay": ADAPTER_DELAY,
    }
    (tmp_path / f"physical-core-{model_name}-{kind}.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize("source_samples", [12000, 12001, 16000, 192000, 192001])
def test_known_source_eos_preserves_tail_and_stops_native_flush(
    pinned_assets: tuple[Path, Path, Path],
    native_probe: Path,
    tmp_path: Path,
    source_samples: int,
) -> None:
    library, full_model, _ = pinned_assets
    callbacks = (source_samples + CALLBACK_SAMPLES - 1) // CALLBACK_SAMPLES
    callback_samples = callbacks * CALLBACK_SAMPLES
    mic = np.zeros(callback_samples, dtype=np.float32)
    ref = np.zeros(callback_samples, dtype=np.float32)
    generator = np.random.default_rng(20261116 + source_samples)
    mic[:source_samples] = np.asarray(
        0.05 * generator.standard_normal(source_samples), dtype=np.float32
    )
    mic[source_samples - 1] = 0.75

    target_hops = (
        source_samples + MODEL_CODEC_DELAY + HOP_SAMPLES - 1
    ) // HOP_SAMPLES
    input_samples_after_tail = callback_samples
    native_tail_callbacks = 0
    while input_samples_after_tail // HOP_SAMPLES < target_hops:
        input_samples_after_tail += CALLBACK_SAMPLES
        native_tail_callbacks += 1
    max_hops_per_callback = 1 if native_tail_callbacks else 0
    downstream_flush_callbacks = 16
    mic_path = tmp_path / "mic.f32"
    ref_path = tmp_path / "ref.f32"
    output_path = tmp_path / "adapter.f32"
    metadata_path = tmp_path / "adapter.json"
    mic.tofile(mic_path)
    ref.tofile(ref_path)

    completed = subprocess.run(
        [
            str(native_probe),
            str(library),
            str(full_model),
            str(mic_path),
            str(ref_path),
            str(output_path),
            str(metadata_path),
            str(source_samples),
            str(downstream_flush_callbacks),
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    direct = _direct_oracle(
        library, full_model, mic, ref, blocks=target_hops
    )
    valid_samples = ADAPTER_DELAY + target_hops * HOP_SAMPLES
    expected = np.concatenate(
        [np.zeros(ADAPTER_DELAY, dtype=np.float32), direct]
    )[:valid_samples]
    actual = np.fromfile(output_path, dtype=np.float32)
    callback_extent = (
        (valid_samples + CALLBACK_SAMPLES - 1) // CALLBACK_SAMPLES
    ) * CALLBACK_SAMPLES
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata == {
        "delay_samples": ADAPTER_DELAY,
        "callbacks": callbacks,
        "inferences": target_hops,
        "pending": 0,
        "queued": 0,
        "output_samples": callback_extent
        + downstream_flush_callbacks * CALLBACK_SAMPLES,
        "valid_output_samples": valid_samples,
        "source_samples": source_samples,
        "target_hops": target_hops,
        "inferences_before_flush": target_hops,
        "downstream_flush_callbacks": downstream_flush_callbacks,
        "native_tail_callbacks": native_tail_callbacks,
        "max_hops_per_callback": max_hops_per_callback,
        "input_samples_after_tail": input_samples_after_tail,
        "native_hop_samples": HOP_SAMPLES,
    }
    assert np.array_equal(actual[:valid_samples], expected)
    assert np.all(
        actual[valid_samples:] == 0.0
    ), "post-AEC downstream flush was not zero"
    assert np.any(
        np.abs(expected[ADAPTER_DELAY + source_samples - 1 :]) > 0.0
    ), "full-model core tail was truncated"

    evidence = {
        "source_samples": source_samples,
        "source_callbacks": callbacks,
        "target_hops": target_hops,
        "valid_output_samples": valid_samples,
        "output_callback_extent": callback_extent,
        "downstream_flush_callbacks": downstream_flush_callbacks,
        "native_tail_callbacks": native_tail_callbacks,
        "max_hops_per_callback": max_hops_per_callback,
        "input_samples_after_tail": input_samples_after_tail,
        "inferences_before_flush": target_hops,
        "inferences_after_flush": target_hops,
        "adapter_vs_direct_max_abs": float(
            np.max(np.abs(actual[:valid_samples] - expected), initial=0.0)
        ),
        "last_speech_input": source_samples - 1,
        "tail_energy_after_last_input_time": float(
            np.sum(expected[ADAPTER_DELAY + source_samples - 1 :] ** 2)
        ),
    }
    (tmp_path / f"eos-{source_samples}.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


def _downsample_delay_16k(sample_rate: int) -> int:
    if sample_rate == 16000:
        return 0
    library_name = ctypes.util.find_library("speexdsp")
    if not library_name:
        pytest.skip("SpeexDSP is unavailable for rate-conversion evidence")
    library = ctypes.CDLL(library_name)
    library.speex_resampler_init.restype = ctypes.c_void_p
    library.speex_resampler_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.speex_resampler_get_output_latency.argtypes = [ctypes.c_void_p]
    library.speex_resampler_get_output_latency.restype = ctypes.c_int
    library.speex_resampler_destroy.argtypes = [ctypes.c_void_p]
    error = ctypes.c_int()
    state = library.speex_resampler_init(
        1, sample_rate, 16000, 8, ctypes.byref(error)
    )
    if not state or error.value:
        raise RuntimeError("SpeexDSP rate-converter setup failed")
    try:
        return library.speex_resampler_get_output_latency(state)
    finally:
        library.speex_resampler_destroy(state)


def test_target_hop_rounding_for_16_24_48k(tmp_path: Path) -> None:
    rows = []
    for sample_rate in (16000, 24000, 48000):
        source_samples = sample_rate * 12
        downsample_delay = _downsample_delay_16k(sample_rate)
        source_core = (
            source_samples * 16000 + sample_rate - 1
        ) // sample_rate + downsample_delay
        target_hops = (
            source_core + MODEL_CODEC_DELAY + HOP_SAMPLES - 1
        ) // HOP_SAMPLES
        rows.append(
            {
                "sample_rate": sample_rate,
                "source_samples": source_samples,
                "downsample_delay_16k": downsample_delay,
                "source_core_samples": source_core,
                "target_hops": target_hops,
            }
        )

    by_rate = {row["sample_rate"]: row for row in rows}
    assert by_rate[16000]["target_hops"] == 751
    assert by_rate[24000]["target_hops"] == 752
    assert by_rate[48000]["target_hops"] == 752

    bf1_source = 192001
    bf1_target = (
        bf1_source + MODEL_CODEC_DELAY + HOP_SAMPLES - 1
    ) // HOP_SAMPLES
    assert bf1_target == 752
    evidence = {
        "rounding": "ceil at IO->16k, downsample delay, and final 256-hop boundary",
        "rows": rows,
        "bf1_16k": {
            "source_samples": bf1_source,
            "target_hops": bf1_target,
        },
    }
    (tmp_path / "target-hop-formula.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
