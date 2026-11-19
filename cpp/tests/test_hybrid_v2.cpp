#include "omlsa_imcra.hpp"
#include "residual_feature_bands_v2.hpp"
#include "residual_ns_control_v2.hpp"
#include "tiny_ns_residual_v2.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
std::atomic<bool> track_allocations{false};
std::atomic<std::size_t> allocation_count{0};
}

void* operator new(std::size_t size) {
  if (track_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size ? size : 1)) return pointer;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {

using itu::detail::NsSpectralChain;
using itu::detail::NsSpectralControl;
using itu::detail::NsSpectralFrame;
using itu::detail::OmlsaImcra;
using itu::detail::ResidualFeatureBandsV2;
using itu::detail::ResidualNsControlV2;
using itu::detail::TinyNsResidualV2;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callback>
void require_throw(Callback&& callback, const std::string& label) {
  bool rejected = false;
  try { callback(); } catch (const std::exception&) { rejected = true; }
  require(rejected, label + " was accepted");
}

bool same_bits(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

struct AllocationScope {
  AllocationScope() {
    allocation_count.store(0, std::memory_order_relaxed);
    track_allocations.store(true, std::memory_order_relaxed);
  }
  ~AllocationScope() { track_allocations.store(false, std::memory_order_relaxed); }
  std::size_t count() const { return allocation_count.load(std::memory_order_relaxed); }
};

struct TemporaryFiles {
  std::vector<std::filesystem::path> paths;
  ~TemporaryFiles() {
    for (const auto& path : paths) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  }
  std::filesystem::path add(const std::string& name) {
    auto path = std::filesystem::temp_directory_path() /
                ("itu_v2_" + std::to_string(::getpid()) + "_" + name);
    paths.push_back(path);
    return path;
  }
};

void put_u32(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = value >> (8 * i);
}
void put_u64(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) bytes[offset + i] = value >> (8 * i);
}
std::uint32_t float_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

struct ModelOptions {
  unsigned input = 320;
  unsigned version = 2;
  std::string magic = "ITUNSR02";
  bool trained = true;
  bool corrupt_hash = false;
  bool nonfinite = false;
  bool zero = false;
  bool constant_head = false;
  unsigned count_override = 0;
  unsigned feature_override = 0;
};

struct ModelFixture {
  std::filesystem::path path;
  std::vector<float> weights;
  unsigned input = 0;
};

ModelFixture write_model(TemporaryFiles& files, const std::string& name,
                         ModelOptions options = {}) {
  const unsigned feature_version = options.feature_override
      ? options.feature_override
      : options.input == 320 ? 2u : 1u;
  const unsigned count = options.count_override
      ? options.count_override
      : options.input == 320 ? 49664u : 45568u;
  const std::size_t file_size = 64 + count * sizeof(float) + 8;
  std::vector<unsigned char> bytes(file_size, 0);
  require(options.magic.size() == 8, "fixture magic must be eight bytes");
  std::memcpy(bytes.data(), options.magic.data(), 8);
  const std::array<std::uint32_t, 14> header = {
      options.version, 16000, 64, options.input, 64, 64, 64, count,
      options.trained ? 1u : 0u, feature_version, 512, 160, 320, 0};
  for (std::size_t i = 0; i < header.size(); ++i)
    put_u32(bytes, 8 + i * 4, header[i]);

  std::vector<float> weights(count, 0.f);
  if (!options.zero && !options.constant_head) {
    std::uint32_t state = 0x12345678u ^ options.input;
    auto next = [&] {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      return (float(state >> 8) / 8388608.f - 1.f) * .04f;
    };
    for (float& value : weights) value = next();
  }
  if (options.constant_head) {
    const std::size_t head_bias = count - 64;
    for (unsigned i = 0; i < 64; ++i) weights[head_bias + i] = .5f;
  }
  if (options.nonfinite) weights[0] = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t i = 0; i < weights.size(); ++i)
    put_u32(bytes, 64 + i * 4, float_bits(weights[i]));
  std::uint64_t hash = TinyNsResidualV2::fnv1a64(
      bytes.data(), bytes.size() - TinyNsResidualV2::checksum_size);
  if (options.corrupt_hash) ++hash;
  put_u64(bytes, bytes.size() - TinyNsResidualV2::checksum_size, hash);
  const auto path = files.add(name);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write v2 fixture");
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) throw std::runtime_error("v2 fixture write failed");
  return {path, weights, options.input};
}

float sigmoid(float value) {
  if (value >= 0.f) return 1.f / (1.f + std::exp(-value));
  const float e = std::exp(value);
  return e / (1.f + e);
}

template <unsigned InputDim>
std::array<float, 64> reference_forward(const std::vector<float>& weights,
                                        const std::array<float, InputDim>& features,
                                        std::array<float, 64>& hidden) {
  constexpr std::size_t pw = 64 * InputDim;
  constexpr std::size_t pb = pw;
  constexpr std::size_t gih = pb + 64;
  constexpr std::size_t ghh = gih + 3 * 64 * 64;
  constexpr std::size_t bih = ghh + 3 * 64 * 64;
  constexpr std::size_t bhh = bih + 3 * 64;
  constexpr std::size_t hw = bhh + 3 * 64;
  constexpr std::size_t hb = hw + 64 * 64;
  std::array<float, 64> projected{};
  for (unsigned o = 0; o < 64; ++o) {
    float sum = weights[pb + o];
    for (unsigned i = 0; i < InputDim; ++i) sum += weights[o * InputDim + i] * features[i];
    projected[o] = std::max(0.f, sum);
  }
  std::array<float, 64> r{}, z{}, n{}, next{};
  for (unsigned gate = 0; gate < 3; ++gate) for (unsigned o = 0; o < 64; ++o) {
    const unsigned row = gate * 64 + o;
    float input_sum = weights[bih + row], hidden_sum = weights[bhh + row];
    for (unsigned i = 0; i < 64; ++i) input_sum += weights[gih + row * 64 + i] * projected[i];
    for (unsigned i = 0; i < 64; ++i) hidden_sum += weights[ghh + row * 64 + i] * hidden[i];
    if (gate == 0) r[o] = sigmoid(input_sum + hidden_sum);
    else if (gate == 1) z[o] = sigmoid(input_sum + hidden_sum);
    else n[o] = std::tanh(input_sum + r[o] * hidden_sum);
  }
  for (unsigned i = 0; i < 64; ++i) next[i] = (1.f - z[i]) * n[i] + z[i] * hidden[i];
  hidden = next;
  std::array<float, 64> residual{};
  for (unsigned o = 0; o < 64; ++o) {
    float sum = weights[hb + o];
    for (unsigned i = 0; i < 64; ++i) sum += weights[hw + o * 64 + i] * next[i];
    residual[o] = 12.f * std::tanh(sum);
  }
  return residual;
}

void model_checks(TemporaryFiles& files) {
  for (unsigned input : {256u, 320u}) {
    const ModelFixture fixture = write_model(
        files, input == 320 ? "valid320.bin" : "valid256.bin",
        ModelOptions{.input = input});
    TinyNsResidualV2 model;
    model.load(fixture.path.string());
    require(model.input_dim() == input, "model input dimension differs");
    require(model.parameter_count() == (input == 320 ? 49664u : 45568u),
            "model parameter count differs");
    std::array<float, 320> features{};
    for (unsigned i = 0; i < 320; ++i) features[i] = .002f * i - .2f;
    std::array<float, 64> expected_hidden{};
    for (unsigned frame = 0; frame < 4; ++frame) {
      features[0] += .1f * frame;
      std::array<float, 64> actual{}, hidden_out{};
      require(model.process(features.data(), actual.data(), hidden_out.data()),
              "v2 model inference failed");
      std::array<float, 64> expected{};
      if (input == 256) {
        std::array<float, 256> features256{};
        std::copy_n(features.begin(), 256, features256.begin());
        expected = reference_forward<256>(fixture.weights, features256,
                                          expected_hidden);
      } else {
        expected = reference_forward<320>(fixture.weights, features,
                                          expected_hidden);
      }
      for (unsigned i = 0; i < 64; ++i) {
        require(std::abs(actual[i] - expected[i]) < 2e-5f,
                "v2 seeded fixture parity differs");
        require(std::abs(hidden_out[i] - expected_hidden[i]) < 2e-5f,
                "v2 hidden parity differs");
      }
    }
  }

  std::array<float, 320> features{};
  std::array<float, 64> residual{}, hidden{};
  for (unsigned input : {256u, 320u}) {
    const ModelFixture zero = write_model(files,
        input == 320 ? "zero320.bin" : "zero256.bin",
        ModelOptions{.input = input, .zero = true});
    TinyNsResidualV2 zero_model;
    zero_model.load(zero.path.string());
    require(zero_model.process(features.data(), residual.data(), hidden.data()),
            "zero-head model failed");
    for (float value : residual) require(same_bits(value, 0.f), "zero head was not zero");

    const ModelFixture constant = write_model(files,
        input == 320 ? "constant320.bin" : "constant256.bin",
        ModelOptions{.input = input, .constant_head = true});
    TinyNsResidualV2 constant_model;
    constant_model.load(constant.path.string());
    require(constant_model.process(features.data(), residual.data(), hidden.data()),
            "constant-head model failed");
    const float expected = 12.f * std::tanh(.5f);
    for (float value : residual)
      require(std::abs(value - expected) < 1e-6f, "constant head scaling differs");
  }

  const ModelFixture distinct256 = write_model(files, "distinct256.bin",
      ModelOptions{.input = 256});
  const ModelFixture distinct320 = write_model(files, "distinct320.bin",
      ModelOptions{.input = 320});
  TinyNsResidualV2 model256, model320;
  model256.load(distinct256.path.string());
  model320.load(distinct320.path.string());
  std::array<float, 64> residual256{}, residual320{}, hidden256{}, hidden320{};
  require(model256.process(features.data(), residual256.data(), hidden256.data()),
          "256 loss-only fixture failed");
  require(model320.process(features.data(), residual320.data(), hidden320.data()),
          "320 base-gain fixture failed");
  require(!std::equal(residual256.begin(), residual256.end(), residual320.begin(),
                      [](float a, float b) { return same_bits(a, b); }),
          "256/320 fixtures were accidentally treated as the same model");

  require_throw([&] { TinyNsResidualV2 bad; bad.load((files.add("missing")).string()); },
                "missing v2 model");
  const auto v1_magic = write_model(files, "v1magic.bin",
      ModelOptions{.input = 256, .magic = "ITUNSR01"});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(v1_magic.path.string()); },
                "v1 model magic");
  const auto v1_version = write_model(files, "v1version.bin",
      ModelOptions{.input = 256, .version = 1});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(v1_version.path.string()); },
                "v1 model version");
  const auto untrained = write_model(files, "untrained.bin",
      ModelOptions{.input = 320, .trained = false});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(untrained.path.string()); },
                "untrained v2 model");
  const auto bad_count = write_model(files, "badcount.bin",
      ModelOptions{.input = 320, .count_override = 45568});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(bad_count.path.string()); },
                "inconsistent v2 input/count");
  const auto bad_hash = write_model(files, "badhash.bin",
      ModelOptions{.input = 320, .corrupt_hash = true});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(bad_hash.path.string()); },
                "bad v2 hash");
  const auto nonfinite = write_model(files, "nonfinite.bin",
      ModelOptions{.input = 320, .nonfinite = true});
  require_throw([&] { TinyNsResidualV2 bad; bad.load(nonfinite.path.string()); },
                "non-finite v2 weight");
}

void feature_checks() {
  ResidualFeatureBandsV2 bands;
  std::array<float, 257> power{}, noise{}, gain{};
  power.fill(1.f); noise.fill(2.f); gain.fill(.25f);
  require(bands.process(power.data(), noise.data(), gain.data()),
          "v2 constant feature frame failed");
  for (unsigned band = 0; band < 64; ++band) {
    require(std::abs(bands.projected_gain()[band] - .25f) < 1e-7f,
            "base gain was not projected linearly");
    require(std::abs(bands.features()[256 + band] - 20.f * std::log10(.25f)) < 1e-5f,
            "base gain dB contract differs");
    require(same_bits(bands.features()[192 + band], 0.f),
            "first-frame delta differs");
  }
  gain.fill(.0001f);
  require(bands.process(power.data(), noise.data(), gain.data()),
          "small base gain frame failed");
  require(std::abs(bands.features()[256] - 20.f * std::log10(.0001f)) < 1e-5f,
          "base gain input was unexpectedly clamped");

  ResidualFeatureBandsV2 transactional;
  gain.fill(.5f);
  require(transactional.process(power.data(), noise.data(), gain.data()),
          "transaction baseline failed");
  gain[3] = 1.01f;
  require(!transactional.process(power.data(), noise.data(), gain.data()),
          "invalid base gain was accepted");
  gain.fill(.5f); power.fill(2.f);
  require(transactional.process(power.data(), noise.data(), gain.data()),
          "transaction continuation failed");
  require(std::abs(transactional.features()[192] - 10.f * std::log10(2.f)) < 1e-5f,
          "failed frame advanced previous-power state");
}

struct Recorder final : NsSpectralControl {
  void observe(NsSpectralFrame frame) noexcept override {
    for (unsigned bin = 0; bin < frame.bins; ++bin) {
      power[bin] = frame.power[bin]; noise[bin] = frame.noise[bin];
      xi[bin] = frame.xi[bin]; gamma[bin] = frame.gamma[bin];
      probability[bin] = frame.probability[bin]; gain[bin] = frame.gain[bin];
    }
  }
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
};

void control_checks(TemporaryFiles& files) {
  const ModelFixture fixture = write_model(files, "control320.bin",
      ModelOptions{.input = 320});
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
  power.fill(1.f); noise.fill(.5f); xi.fill(1.f); gamma.fill(2.f);
  probability.fill(.5f); gain.fill(.1f);
  const auto original_gain = gain;
  const NsSpectralFrame frame{16000, 512, 257, power.data(), noise.data(),
                              xi.data(), gamma.data(), probability.data(),
                              gain.data()};

  ResidualNsControlV2 disabled(16000, 512, fixture.path.string(), -21.f, 0.f);
  disabled.observe(frame);
  require(disabled.inferences() == 1 && disabled.applied_frames() == 0,
          "v2 mix=0 counters differ");
  for (unsigned bin = 0; bin < 257; ++bin)
    require(same_bits(gain[bin], original_gain[bin]), "v2 mix=0 changed gain");

  const ModelFixture loss_only = write_model(files, "control256.bin",
      ModelOptions{.input = 256});
  ResidualNsControlV2 loss_only_control(16000, 512, loss_only.path.string(),
                                         -21.f, 0.f);
  require(loss_only_control.input_dim() == 256 &&
              loss_only_control.parameter_count() == 45568,
          "256 loss-only controller did not dispatch by model header");
  loss_only_control.observe(frame);
  require(loss_only_control.inferences() == 1,
          "256 loss-only controller did not consume first 256 features");

  ResidualNsControlV2 control(16000, 512, fixture.path.string(), -21.f, 1.f);
  std::size_t control_allocations = 0;
  {
    AllocationScope allocations;
    control.observe(frame);
    control_allocations = allocations.count();
  }
  require(control_allocations == 0, "v2 control observe allocated on the frame path");
  for (unsigned bin = 0; bin < 257; ++bin) {
    const float expected = std::clamp(original_gain[bin] *
        std::pow(10.f, control.residual_db(bin) / 20.f),
        control.floor_gain(), 1.f);
    require(std::abs(gain[bin] - expected) < 1e-7f, "v2 gain formula differs");
  }

  TinyNsResidualV2 model;
  model.load(fixture.path.string());
  std::array<float, 320> features{};
  std::array<float, 64> residual{}, hidden{};
  std::size_t model_allocations = 0;
  bool model_ok = false;
  {
    AllocationScope allocations;
    model_ok = model.process(features.data(), residual.data(), hidden.data());
    model_allocations = allocations.count();
  }
  require(model_ok, "v2 model no-heap inference failed");
  require(model_allocations == 0, "v2 model process allocated on the frame path");

  ResidualNsControlV2 stream(16000, 512, fixture.path.string(), -21.f, 1.f);
  std::array<float, 64> before{};
  for (unsigned frame_index = 0; frame_index < 24; ++frame_index) {
    if (frame_index == 1) before = stream.hidden();
    power.fill(frame_index < 2 ? 1.f : 0.f);
    noise.fill(.1f);
    gain.fill(.8f);
    stream.observe(frame);
  }
  require(stream.inferences() == 24 && stream.fallback_frames() == 0,
          "v2 stream counters reset or fall back unexpectedly");
  bool changed = false;
  for (unsigned i = 0; i < 64; ++i) changed = changed || !same_bits(before[i], stream.hidden()[i]);
  require(changed, "silence path silently reset v2 hidden state");
}

void statistics_independence(TemporaryFiles& files) {
  const ModelFixture fixture = write_model(files, "stats320.bin",
      ModelOptions{.input = 320});
  OmlsaImcra baseline(16000, -9.f);
  OmlsaImcra disabled_engine(16000, -9.f);
  OmlsaImcra enabled_engine(16000, -9.f);
  Recorder baseline_record, disabled_record, enabled_record;
  ResidualNsControlV2 disabled(16000, 512, fixture.path.string(), -21.f, 0.f);
  ResidualNsControlV2 enabled(16000, 512, fixture.path.string(), -21.f, 1.f);
  NsSpectralChain disabled_chain(&disabled, &disabled_record);
  NsSpectralChain enabled_chain(&enabled, &enabled_record);
  std::mt19937 generator(20261115);
  std::normal_distribution<float> random(0.f, .015f);
  std::size_t enabled_gain_differences = 0;
  for (unsigned frame_index = 0; frame_index < 48; ++frame_index) {
    std::array<float, 160> input{};
    for (unsigned i = 0; i < input.size(); ++i)
      input[i] = random(generator) + .01f * std::sin(6.2831853f * 440.f *
          (frame_index * input.size() + i) / 16000.f);
    auto disabled_input = input;
    auto enabled_input = input;
    baseline.process(input.data(), &baseline_record);
    disabled_engine.process(disabled_input.data(), &disabled_chain);
    enabled_engine.process(enabled_input.data(), &enabled_chain);
    for (unsigned bin = 0; bin < 257; ++bin) {
      require(same_bits(baseline_record.power[bin], disabled_record.power[bin]),
              "mix=0 power state changed");
      require(same_bits(baseline_record.noise[bin], disabled_record.noise[bin]),
              "mix=0 noise state changed");
      require(same_bits(baseline_record.xi[bin], disabled_record.xi[bin]),
              "mix=0 xi state changed");
      require(same_bits(baseline_record.gamma[bin], disabled_record.gamma[bin]),
              "mix=0 gamma state changed");
      require(same_bits(baseline_record.probability[bin], disabled_record.probability[bin]),
              "mix=0 probability state changed");
      require(same_bits(baseline_record.gain[bin], disabled_record.gain[bin]),
              "mix=0 gain changed");
      enabled_gain_differences += !same_bits(baseline_record.gain[bin],
                                             enabled_record.gain[bin]);
      require(same_bits(baseline_record.power[bin], enabled_record.power[bin]),
              "enabled power state changed");
      require(same_bits(baseline_record.noise[bin], enabled_record.noise[bin]),
              "enabled noise state changed");
      require(same_bits(baseline_record.xi[bin], enabled_record.xi[bin]),
              "enabled xi state changed");
      require(same_bits(baseline_record.gamma[bin], enabled_record.gamma[bin]),
              "enabled gamma state changed");
      require(same_bits(baseline_record.probability[bin], enabled_record.probability[bin]),
              "enabled probability state changed");
    }
  }
  require(enabled_gain_differences > 0, "enabled v2 control did not change gain");
  std::cout << "v2_statistics_independent_frames=48 gain_bin_differences="
            << enabled_gain_differences << '\n';
}

}  // namespace

int main() {
  try {
    TemporaryFiles files;
    model_checks(files);
    feature_checks();
    control_checks(files);
    statistics_independence(files);
    std::cout << "ITUNSR02 model/feature/control checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
