#pragma once

#include "itu/audio_engine.hpp"
#include "continuous_hop_adapter.hpp"
#include "rate_converter.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace itu {
namespace localvqe_detail {

inline constexpr char kBackendId[] = "localvqe_aec200k";
inline constexpr char kReportName[] = "localvqe_aec200k_cpu";
inline constexpr char kArchitecture[] = "localvqe_daf_neural_mask";
inline constexpr char kModelId[] = "localvqe_aec200k";
inline constexpr char kModelSha256[] =
    "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731";
inline constexpr char kApiSourceSha256[] =
    "76510c8250a3af232a6f241643683dad94d9b17f3c184da37870644005b77573";
inline constexpr char kLibrarySha256[] =
    "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60";
inline constexpr unsigned kSampleRate = 16000;
inline constexpr unsigned kHopSamples = 256;
inline constexpr unsigned kFftSize = 512;
inline constexpr unsigned kModelDelaySamples16k = 256;
inline constexpr unsigned kInputBlockRequirementSamples16k = 256;
inline constexpr unsigned kAdapterDelaySamples16k =
    ContinuousHopAdapter::kDelaySamples;
inline constexpr unsigned kAecDelaySamples16k =
    kModelDelaySamples16k + kAdapterDelaySamples16k;

inline std::uint64_t ceil_div(std::uint64_t numerator,
                              std::uint64_t denominator) {
  return numerator / denominator + (numerator % denominator ? 1u : 0u);
}

inline std::uint64_t source_samples_at_16k(std::uint64_t source_samples,
                                           unsigned source_rate,
                                           double downsample_delay_16k) {
  if (!source_samples || !source_rate) return 0;
  const auto whole = source_samples / source_rate;
  const auto remainder = source_samples % source_rate;
  const auto converted =
      whole * kSampleRate +
      ceil_div(remainder * kSampleRate, source_rate);
  return converted +
         static_cast<std::uint64_t>(std::ceil(downsample_delay_16k));
}

inline std::uint64_t target_native_hops(std::uint64_t source_samples,
                                        unsigned source_rate,
                                        double downsample_delay_16k) {
  const auto source_core =
      source_samples_at_16k(source_samples, source_rate, downsample_delay_16k);
  return source_core
             ? ceil_div(source_core + kModelDelaySamples16k, kHopSamples)
             : 0;
}

struct AssetPaths {
  std::filesystem::path library;
  std::filesystem::path model;
};

inline std::filesystem::path project_root() {
#ifdef ITU_PROJECT_ROOT
  return std::filesystem::path(ITU_PROJECT_ROOT);
#else
  return std::filesystem::current_path();
#endif
}

inline AssetPaths asset_paths() {
  const auto root = project_root();
  const char* library_override = std::getenv("ITU_LOCALVQE_LIBRARY");
  const char* model_override = std::getenv("ITU_LOCALVQE_MODEL");
  return {
      library_override && *library_override
          ? std::filesystem::path(library_override)
          : root / "build/localvqe-host-20261010/liblocalvqe.so.0.1.0",
      model_override && *model_override
          ? std::filesystem::path(model_override)
          : root / "build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf",
  };
}

class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

  void update(const std::uint8_t* data, std::size_t size) {
    bit_count_ += static_cast<std::uint64_t>(size) * 8u;
    while (size) {
      const auto available = buffer_.size() - buffer_size_;
      const auto take = std::min<std::size_t>(available, size);
      std::copy_n(data, take, buffer_.begin() + buffer_size_);
      buffer_size_ += take;
      data += take;
      size -= take;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
  }

  std::string finish() {
    const auto original_bits = bit_count_;
    const std::uint8_t marker = 0x80u;
    update(&marker, 1);
    const std::uint8_t zero = 0;
    while (buffer_size_ != 56) update(&zero, 1);
    std::array<std::uint8_t, 8> length{};
    for (unsigned i = 0; i < length.size(); ++i)
      length[7u - i] = static_cast<std::uint8_t>(original_bits >> (i * 8u));
    update(length.data(), length.size());
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        result.push_back(hex[(word >> shift) & 0xfu]);
      }
    }
    return result;
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32u - bits));
  }

  void transform(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::array<std::uint32_t, 64> words{};
    for (unsigned i = 0; i < 16; ++i) {
      words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24u) |
                 (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16u) |
                 (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8u) |
                 static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (unsigned i = 16; i < 64; ++i) {
      const auto s0 = rotate_right(words[i - 15], 7) ^
                      rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const auto s1 = rotate_right(words[i - 2], 17) ^
                      rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
      const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto temp1 = h + s1 + choice + constants[i] + words[i];
      const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

class ScopedStdoutSilence {
 public:
  ScopedStdoutSilence() {
    std::fflush(stdout);
    saved_ = dup(STDOUT_FILENO);
    const int null_fd = open("/dev/null", O_WRONLY);
    if (saved_ >= 0 && null_fd >= 0 && dup2(null_fd, STDOUT_FILENO) >= 0)
      active_ = true;
    if (null_fd >= 0) close(null_fd);
  }

  ~ScopedStdoutSilence() {
    if (!active_ || saved_ < 0) return;
    std::fflush(stdout);
    dup2(saved_, STDOUT_FILENO);
    close(saved_);
  }

  ScopedStdoutSilence(const ScopedStdoutSilence&) = delete;
  ScopedStdoutSilence& operator=(const ScopedStdoutSilence&) = delete;

 private:
  int saved_ = -1;
  bool active_ = false;
};

inline std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  Sha256 sha;
  std::array<std::uint8_t, 1024 * 1024> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) sha.update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!input.eof()) throw std::runtime_error("cannot read " + path.string());
  return sha.finish();
}

class Api {
 public:
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
  using Reset = void (*)(Handle);
  using SetNoiseGate = int (*)(Handle, int, float);

  explicit Api(const std::filesystem::path& path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(dlerror());
    options_new = symbol<OptionsNew>("localvqe_options_new");
    options_free = symbol<OptionsFree>("localvqe_options_free");
    set_model = symbol<SetModel>("localvqe_options_set_model_path");
    set_threads = symbol<SetThreads>("localvqe_options_set_threads");
    new_with_options = symbol<NewWithOptions>("localvqe_new_with_options");
    free = symbol<Free>("localvqe_free");
    process_frame = symbol<ProcessFrame>("localvqe_process_frame_f32");
    sample_rate = symbol<SampleRate>("localvqe_sample_rate");
    hop_length = symbol<HopLength>("localvqe_hop_length");
    fft_size = symbol<FftSize>("localvqe_fft_size");
    last_error = symbol<LastError>("localvqe_last_error");
    reset = symbol<Reset>("localvqe_reset");
    set_noise_gate = symbol<SetNoiseGate>("localvqe_set_noise_gate");
  }

  ~Api() {
    if (library_) dlclose(library_);
  }

  Api(const Api&) = delete;
  Api& operator=(const Api&) = delete;

  OptionsNew options_new = nullptr;
  OptionsFree options_free = nullptr;
  SetModel set_model = nullptr;
  SetThreads set_threads = nullptr;
  NewWithOptions new_with_options = nullptr;
  Free free = nullptr;
  ProcessFrame process_frame = nullptr;
  SampleRate sample_rate = nullptr;
  HopLength hop_length = nullptr;
  FftSize fft_size = nullptr;
  LastError last_error = nullptr;
  Reset reset = nullptr;
  SetNoiseGate set_noise_gate = nullptr;

 private:
  template <class T>
  T symbol(const char* name) {
    const auto address = dlsym(library_, name);
    if (!address)
      throw std::runtime_error(std::string("missing LocalVQE symbol: ") + name);
    return reinterpret_cast<T>(address);
  }

  void* library_ = nullptr;
};

inline std::string verify_asset(const std::filesystem::path& path,
                                const char* expected,
                                const char* label) {
  if (!std::filesystem::is_regular_file(path))
    throw std::runtime_error(std::string(label) + " is missing: " + path.string());
  const auto actual = sha256_file(path);
  if (actual != expected)
    throw std::runtime_error(std::string(label) +
                             " SHA256 mismatch: expected " + expected +
                             ", got " + actual);
  return actual;
}

inline void probe_context(Api& api, const std::filesystem::path& model) {
  auto options = api.options_new();
  if (!options) throw std::runtime_error("LocalVQE options allocation failed");
  Api::Handle context = 0;
  try {
    if (api.set_model(options, model.c_str()) != 0)
      throw std::runtime_error("LocalVQE model path rejected");
    if (api.set_threads(options, 1) != 0)
      throw std::runtime_error("LocalVQE thread count rejected");
    context = api.new_with_options(options);
  } catch (...) {
    api.options_free(options);
    throw;
  }
  api.options_free(options);
  if (!context)
    throw std::runtime_error("LocalVQE model context creation failed");
  try {
    if (api.sample_rate(context) != static_cast<int>(kSampleRate) ||
        api.hop_length(context) != static_cast<int>(kHopSamples) ||
        api.fft_size(context) != static_cast<int>(kFftSize))
      throw std::runtime_error(
          "LocalVQE streaming contract is not 16 kHz / 256 / 512");
    if (api.set_noise_gate(context, 0, -90.f) != 0)
      throw std::runtime_error("LocalVQE noise-gate control rejected");
  } catch (...) {
    api.free(context);
    throw;
  }
  api.free(context);
}

inline AecBackendCapability capability() {
  AecBackendCapability result;
  result.id = kBackendId;
  result.architecture = kArchitecture;
  result.execution_provider = "cpu";
  result.device_id = "host_cpu";
  result.model_id = kModelId;
  result.model_sha256 = kModelSha256;
  result.frontend_sha256 = kApiSourceSha256;
  result.provider_sha256 = kLibrarySha256;
  result.delay_samples = kAecDelaySamples16k;
  try {
    const auto paths = asset_paths();
    verify_asset(paths.library, kLibrarySha256, "LocalVQE library");
    verify_asset(paths.model, kModelSha256, "LocalVQE model");
    Api api(paths.library);
    ScopedStdoutSilence stdout_silence;
    probe_context(api, paths.model);
    result.available = true;
  } catch (const std::exception& error) {
    result.available = false;
    result.reason = std::string(kBackendId) + " unavailable: " + error.what();
  }
  return result;
}

}  // namespace localvqe_detail

class LocalVqeAecPipeline {
 public:
  explicit LocalVqeAecPipeline(const Config& config)
      : c_(config),
        paths_(localvqe_detail::asset_paths()),
        api_(paths_.library),
        mic16_(160),
        ref16_(160),
        aec16_(160),
        local_output_(c_.frame_size),
        post_config_(post_config(c_)) {
    using namespace localvqe_detail;
    if (c_.reference_channels != 1 ||
        (!c_.beamforming && c_.microphones != 1))
      throw std::invalid_argument(
          "localvqe_aec200k requires mono reference and mono capture or BF before AEC");
    verify_asset(paths_.library, kLibrarySha256, "LocalVQE library");
    verify_asset(paths_.model, kModelSha256, "LocalVQE model");

    ScopedStdoutSilence stdout_silence;
    auto options = api_.options_new();
    if (!options) throw std::runtime_error("LocalVQE options allocation failed");
    try {
      if (api_.set_model(options, paths_.model.c_str()) != 0)
        throw std::runtime_error("LocalVQE model path rejected");
      if (api_.set_threads(options, 1) != 0)
        throw std::runtime_error("LocalVQE thread count rejected");
      context_ = api_.new_with_options(options);
    } catch (...) {
      api_.options_free(options);
      throw;
    }
    api_.options_free(options);
    if (!context_) {
      const auto detail = api_.last_error ? api_.last_error(context_) : nullptr;
      throw std::runtime_error(
          detail && *detail ? detail : "LocalVQE model context creation failed");
    }
    if (api_.sample_rate(context_) != static_cast<int>(kSampleRate) ||
        api_.hop_length(context_) != static_cast<int>(kHopSamples) ||
        api_.fft_size(context_) != static_cast<int>(kFftSize))
      throw std::runtime_error("LocalVQE streaming contract is not 16 kHz / 256 / 512");
    if (api_.set_noise_gate(context_, 0, -90.f) != 0)
      throw std::runtime_error("LocalVQE noise-gate control rejected");

    if (c_.sample_rate != kSampleRate) {
      mic_down_ = std::make_unique<RateConverter>(c_.frame_size, 160);
      ref_down_ = std::make_unique<RateConverter>(c_.frame_size, 160);
      output_up_ = std::make_unique<RateConverter>(160, c_.frame_size);
    }
    down_delay_16k_ = mic_down_ ? mic_down_->output_delay() : 0.;
    up_delay_samples_ = output_up_ ? output_up_->output_delay() : 0.;
    const double rate_scale = static_cast<double>(c_.sample_rate) / kSampleRate;
    down_delay_samples_ =
        static_cast<unsigned>(std::lround(down_delay_16k_ * rate_scale));
    algorithm_core_delay_samples_ =
        static_cast<unsigned>(std::lround(kAecDelaySamples16k * rate_scale));
    stream_delay_samples_ =
        down_delay_samples_ + algorithm_core_delay_samples_ +
        static_cast<unsigned>(std::lround(up_delay_samples_));
    render_context_delay_.assign(stream_delay_samples_, 0.f);
    target_native_hops_ = target_native_hops(
        c_.localvqe_source_samples, c_.sample_rate, down_delay_16k_);

    // Latency probing must not consume a worker frame or advance its state.
    // For online NS, replay one silent external block through the same delay
    // contract with the callback stripped from the temporary engine.
    Config probe_config = post_config_;
    if (probe_config.ns_stream_process) {
      const unsigned stream_delay = probe_config.ns_stream_delay_samples;
      probe_config.ns_stream_process = {};
      probe_config.ns_stream_delay_samples = 0;
      probe_config.post_ns_delay_samples = stream_delay;
      probe_config.post_ns_pcm =
          std::make_shared<const std::vector<float>>(c_.frame_size, 0.f);
    }
    AudioEngine probe_engine(probe_config);
    std::vector<float> silence(c_.frame_size, 0.f), probe(c_.frame_size, 0.f);
    FrameInfo info{};
    const auto status = probe_engine.process(
        silence.data(), silence.size(), nullptr, 0, probe.data(), probe.size(),
        info);
    if (status != Status::ok)
      throw std::runtime_error("LocalVQE post-AEC chain initialization failed");
    post_delay_samples_ = info.latency_samples;
    // Construct the live callback engine only after the non-worker probe.
    // A model's internal FLOAT output can legitimately overshoot full scale.
    // Keep public capture validation at the outer engine boundary and leave
    // final limiting to the configured downstream chain.
    post_.reset(new AudioEngine(post_config_, 4096, true));
  }

  ~LocalVqeAecPipeline() {
    if (context_) api_.free(context_);
  }

  LocalVqeAecPipeline(const LocalVqeAecPipeline&) = delete;
  LocalVqeAecPipeline& operator=(const LocalVqeAecPipeline&) = delete;

  Status process(const float* mic, const float* ref, float* out,
                 FrameInfo& info, float* at, float* bt, float* lt,
                 float* it) noexcept {
    using Clock = std::chrono::steady_clock;
    const unsigned h = c_.frame_size;
    if (!mic || !ref || !out || c_.microphones != 1)
      return Status::invalid_pointer;
    try {
      float localvqe_process_ms = 0.f;
      const float* mic_frame = mic;
      const float* ref_frame = ref;
      if (mic_down_) {
        if (mic_down_->Resample(mic, h, mic16_.data(), 160) ||
            ref_down_->Resample(ref, h, ref16_.data(), 160))
          return Status::backend_error;
        mic_frame = mic16_.data();
        ref_frame = ref16_.data();
      }

      auto native_hop =
          [&](const float* microphone, const float* reference, float* output) {
            int status = 0;
            if (c_.profile_stages) {
              const auto mark = Clock::now();
              status = api_.process_frame(
                  context_, microphone, reference,
                  static_cast<int>(localvqe_detail::kHopSamples), output);
              localvqe_process_ms +=
                  std::chrono::duration<float, std::milli>(Clock::now() - mark)
                      .count();
            } else {
              status = api_.process_frame(
                  context_, microphone, reference,
                  static_cast<int>(localvqe_detail::kHopSamples), output);
            }
            if (status != 0) return false;
            return std::all_of(
                output, output + localvqe_detail::kHopSamples,
                [](float value) { return std::isfinite(value); });
          };

      if (!adapter_.finished()) {
        if (!adapter_.process(mic_frame, ref_frame, aec16_.data(), native_hop))
          return Status::backend_error;
        source_samples_seen_ += h;
        if (c_.localvqe_source_samples &&
            source_samples_seen_ >= c_.localvqe_source_samples)
          source_exhausted_ = true;
        if (source_exhausted_ && !source_finalized_) {
          if (adapter_.inferences() > target_native_hops_)
            return Status::backend_error;
          if (adapter_.inferences() == target_native_hops_) {
            if (!adapter_.finish_after_hops(target_native_hops_))
              return Status::backend_error;
            source_finalized_ = true;
          }
        }
      } else {
        source_samples_seen_ += h;
        if (!adapter_.drain(aec16_.data()))
          return Status::backend_error;
      }

      if (output_up_) {
        if (output_up_->Resample(aec16_.data(), 160, local_output_.data(), h))
          return Status::backend_error;
      } else {
        std::copy_n(aec16_.data(), h, local_output_.data());
      }

      if (at) std::copy_n(local_output_.data(), h, at);
      if (it) std::copy_n(mic, h, it);
      if (bt) std::fill_n(bt, h, 0.f);
      if (lt) std::fill_n(lt, 160, 0.f);

      double render_power = 0, aligned_render_power = 0;
      for (unsigned i = 0; i < h; ++i) {
        render_power += static_cast<double>(ref[i]) * ref[i];
        if (c_.ns_stream_process) {
          const float delayed = render_context_delay_[render_context_head_];
          render_context_delay_[render_context_head_] = ref[i];
          render_context_head_ = (render_context_head_ + 1) % render_context_delay_.size();
          aligned_render_power += static_cast<double>(delayed) * delayed;
        }
      }
      if (render_power / h > c_.activity_floor)
        render_hold_ = 100;
      else if (render_hold_)
        --render_hold_;

      if (c_.ns_stream_process) {
        if (aligned_render_power / h > c_.activity_floor)
          post_render_hold_ = 100;
        else if (post_render_hold_)
          --post_render_hold_;
      }
      PostAecContext post_context;
      // The NS input already contains the 256-sample LocalVQE codec PCM
      // shift plus the 224-sample continuous-hop framing delay. The separate
      // 256-sample input-block requirement is device availability latency and
      // is not part of the PCM trim.
      // Align its reference evidence before the NS's own 640-sample FIFO.
      post_context.far_active = (c_.ns_stream_process ? post_render_hold_ : render_hold_) > 0;
      post_context.reference_state = post_context.far_active ? 1 : 0;
      FrameInfo post_info{};
      const auto post_mark =
          c_.profile_stages ? Clock::now() : Clock::time_point{};
      const auto status = post_->process_impl(
          local_output_.data(), h, nullptr, 0, out, h, post_info,
          at, bt, nullptr, it, &post_context);
      if (status != Status::ok) return status;
      const float localvqe_post_ms =
          c_.profile_stages
              ? std::chrono::duration<float, std::milli>(Clock::now() - post_mark)
                    .count()
              : 0.f;

      info = post_info;
      info.localvqe_process_ms = localvqe_process_ms;
      info.localvqe_post_ms = localvqe_post_ms;
      info.frame_index = frames_;
      info.microphone_rms = rms(mic, h);
      info.aec_rms = rms(local_output_.data(), h);
      info.far_active = render_hold_ > 0;
      info.reference_state = info.far_active ? 1 : 0;
      info.reference_confidence = 0;
      info.reference_residual_ratio = 1;
      info.linear_residual_ratio = 1;
      info.aec_capture_channels = 1;
      info.beamforming_before_aec = false;
      info.bf_delay_samples = 0;
      info.linear_aec_available = false;
      info.linear_aec_delay_samples = 0;
      info.aec_input_delay_samples = down_delay_samples_;
      info.aec_output_delay_samples = stream_delay_samples_;
      info.latency_samples = post_info.latency_samples + stream_delay_samples_;
      info.low_cut_delay_samples = 0;
      info.neural_inferences =
          static_cast<unsigned>(adapter_.inferences());
      if (c_.diagnostic_stages) {
        auto& taps = *c_.diagnostic_stages;
        const unsigned core_extra = static_cast<unsigned>(std::lround(
            double(stream_delay_samples_) * taps.core_sample_rate / c_.sample_rate));
        const unsigned eq_extra = static_cast<unsigned>(std::lround(
            double(stream_delay_samples_) * taps.eq_sample_rate / c_.sample_rate));
        taps.core_delay_samples += core_extra;
        taps.experiment_ns_input_delay_samples += core_extra;
        taps.agc_delay_samples += core_extra;
        taps.drc_delay_samples += core_extra;
        taps.pre_eq_delay_samples += eq_extra;
        taps.post_eq_delay_samples += eq_extra;
        taps.eq_delay_samples = taps.post_eq_delay_samples;
      }
      ++frames_;
      return Status::ok;
    } catch (...) {
      return Status::backend_error;
    }
  }

  std::size_t memory_bytes() const noexcept {
    return sizeof(*this) +
           (mic16_.capacity() + ref16_.capacity() + aec16_.capacity() +
            local_output_.capacity() + render_context_delay_.capacity()) *
               sizeof(float) +
           post_->working_memory_bytes();
  }

  AecRouteDiagnostics aec_route_diagnostics() const {
    AecRouteDiagnostics result;
    result.execution_provider = "cpu";
    result.device_id = "host_cpu";
    result.model_id = localvqe_detail::kModelId;
    result.model_sha256 = localvqe_detail::kModelSha256;
    result.frontend_sha256 = localvqe_detail::kApiSourceSha256;
    result.provider_sha256 = localvqe_detail::kLibrarySha256;
    result.processed_frames = frames_;
    result.inferences = adapter_.inferences();
    result.fallback_frames = 0;
    result.neural_delay_samples = localvqe_detail::kModelDelaySamples16k;
    result.hop_adapter_delay_samples =
        localvqe_detail::kAdapterDelaySamples16k;
    return result;
  }

 private:
  static float rms(const float* input, std::size_t size) noexcept {
    double power = 0;
    for (std::size_t i = 0; i < size; ++i)
      power += static_cast<double>(input[i]) * input[i];
    return static_cast<float>(std::sqrt(power / size));
  }

  static Config post_config(Config config) {
    config.aec = false;
    config.aec_backend = AecBackend::webrtc;
    config.beamforming = false;
    config.microphones = 1;
    config.reference_channels = 1;
    config.rx_processing = false;
    config.rx_config.enabled = false;
    config.speex_native_conditioning = false;
    config.speex_post_context = false;
    config.speex_multirender_context = false;
    config.post_aec_evidence_window = false;
    config.speex_residual_suppression = false;
    config.speex_coherence_residual_suppression = false;
    config.reference_guard = false;
    config.correlation_delay = false;
    config.coherence_dtd = false;
    config.analyze_linear_noise = false;
    config.linear_speech_protection = false;
    config.multirender_nearend_policy = false;
    config.multirender_speech_protection = false;
    config.complex_pf_cpu_pcm.reset();
    config.complex_pf_residual_capture = nullptr;
    config.complex_pf_reference_capture = nullptr;
    return config;
  }

  Config c_;
  localvqe_detail::AssetPaths paths_;
  localvqe_detail::Api api_;
  localvqe_detail::Api::Handle context_ = 0;
  ContinuousHopAdapter adapter_;
  std::vector<float> mic16_, ref16_, aec16_, local_output_;
  std::unique_ptr<RateConverter> mic_down_, ref_down_, output_up_;
  Config post_config_;
  std::unique_ptr<AudioEngine> post_;
  unsigned frames_ = 0;
  std::uint64_t source_samples_seen_ = 0;
  std::uint64_t target_native_hops_ = 0;
  bool source_exhausted_ = false;
  bool source_finalized_ = false;
  unsigned render_hold_ = 0;
  std::vector<float> render_context_delay_;
  std::size_t render_context_head_ = 0;
  unsigned post_render_hold_ = 0;
  unsigned post_delay_samples_ = 0;
  unsigned down_delay_samples_ = 0;
  unsigned algorithm_core_delay_samples_ = 0;
  unsigned up_delay_samples_ = 0;
  unsigned stream_delay_samples_ = 0;
  double down_delay_16k_ = 0;
};

}  // namespace itu
