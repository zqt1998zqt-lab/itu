#include "omlsa_imcra.hpp"
#include "residual_feature_bands.hpp"
#include "residual_ns_control.hpp"
#include "tiny_ns_residual.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using itu::detail::NsSpectralChain;
using itu::detail::NsSpectralControl;
using itu::detail::NsSpectralFrame;
using itu::detail::OmlsaImcra;
using itu::detail::ResidualFeatureBands;
using itu::detail::ResidualNsControl;
using itu::detail::TinyNsResidual;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callback>
void require_throw(Callback&& callback, const std::string& label) {
  bool rejected = false;
  try {
    callback();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, label + " was accepted");
}

bool same_bits(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

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
                ("itu_c64_" + std::to_string(::getpid()) + "_" + name);
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

std::uint32_t float_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

struct ModelOptions {
  std::uint32_t trained = 1;
  std::uint32_t input = 256;
  std::uint32_t rate = 16000;
  float head_bias = .5f;
  bool nonfinite = false;
  bool corrupt_hash = false;
  bool stateful = true;
};

std::filesystem::path write_model(TemporaryFiles& files,
                                  const std::string& name,
                                  ModelOptions options = {}) {
  const auto path = files.add(name);
  std::vector<unsigned char> bytes(TinyNsResidual::file_size, 0);
  std::memcpy(bytes.data(), "ITUNSR01", 8);
  const std::array<std::uint32_t, 14> metadata = {
      1, options.rate, 64, options.input, 64, 64, 64, 45568,
      options.trained, 1, 512, 160, 320, 0};
  for (std::size_t i = 0; i < metadata.size(); ++i)
    put_u32(bytes, 8 + i * 4, metadata[i]);

  std::vector<float> weights(TinyNsResidual::parameter_count, 0.f);
  constexpr std::size_t projection_weight = 0;
  constexpr std::size_t projection_bias = 64 * 256;
  constexpr std::size_t gru_weight_ih = projection_bias + 64;
  constexpr std::size_t gru_weight_hh = gru_weight_ih + 3 * 64 * 64;
  constexpr std::size_t gru_bias_ih = gru_weight_hh + 3 * 64 * 64;
  constexpr std::size_t gru_bias_hh = gru_bias_ih + 3 * 64;
  constexpr std::size_t head_weight = gru_bias_hh + 3 * 64;
  constexpr std::size_t head_bias = head_weight + 64 * 64;
  static_cast<void>(projection_weight);
  for (unsigned out = 0; out < 64; ++out) weights[projection_bias + out] = 1.f;
  if (options.stateful) {
    weights[gru_weight_ih + 64 * 64] = .2f;       // update gate, input 0
    weights[gru_weight_ih + 128 * 64] = 1.f;      // candidate, input 0
    for (unsigned i = 0; i < 64; ++i)
      weights[gru_weight_hh + (128 + i) * 64 + i] = .4f;
    for (unsigned i = 0; i < 64; ++i) weights[head_weight + i * 64 + i] = 1.f;
    for (unsigned i = 0; i < 64; ++i) weights[head_bias + i] = .1f;
  } else {
    for (unsigned i = 0; i < 64; ++i)
      weights[head_bias + i] = options.head_bias;
  }
  if (options.nonfinite)
    weights[0] = std::numeric_limits<float>::quiet_NaN();

  for (std::size_t i = 0; i < weights.size(); ++i)
    put_u32(bytes, 64 + i * 4, float_bits(weights[i]));
  std::uint64_t hash = TinyNsResidual::fnv1a64(
      bytes.data(), bytes.size() - TinyNsResidual::checksum_size);
  if (options.corrupt_hash) ++hash;
  put_u64(bytes, bytes.size() - TinyNsResidual::checksum_size, hash);

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write test model");
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) throw std::runtime_error("test model write failed");
  return path;
}

void model_and_hash_checks(TemporaryFiles& files) {
  const auto valid = write_model(files, "valid.bin");
  TinyNsResidual model;
  model.load(valid.string());
  require(model.ready(), "valid model did not load");
  require(model.metadata().count == 45568, "parameter count metadata differs");
  require(TinyNsResidual::file_size == 182344, "model byte size differs");

  std::array<float, 256> features{};
  for (unsigned i = 0; i < features.size(); ++i) features[i] = .01f * i;
  std::array<float, 64> residual{}, hidden{};
  require(model.process(features.data(), residual.data(), hidden.data()),
          "valid GRU inference failed");
  for (float value : residual) require(std::isfinite(value), "non-finite residual");
  for (float value : hidden) require(std::isfinite(value), "non-finite hidden");
  require(std::any_of(hidden.begin(), hidden.end(), [](float value) {
            return std::abs(value) > 1e-5f;
          }), "stateful test model has silent zero hidden");

  TinyNsResidual first, second;
  first.load(valid.string());
  second.load(valid.string());
  std::array<float, 64> first_residual{}, second_residual{}, first_hidden{},
      second_hidden{};
  require(first.process(features.data(), first_residual.data(), first_hidden.data()),
          "first deterministic GRU inference failed");
  require(second.process(features.data(), second_residual.data(), second_hidden.data()),
          "second deterministic GRU inference failed");
  for (unsigned i = 0; i < 64; ++i) {
    require(same_bits(first_residual[i], second_residual[i]),
            "deterministic residual differs");
    require(same_bits(first_hidden[i], second_hidden[i]),
            "deterministic hidden differs");
  }

  std::array<float, 64> repeated_residual{}, repeated_hidden{};
  require(first.process(features.data(), repeated_residual.data(),
                        repeated_hidden.data()), "continuous GRU inference failed");
  bool hidden_changed = false;
  for (unsigned i = 0; i < 64; ++i)
    hidden_changed = hidden_changed ||
                     !same_bits(repeated_hidden[i], first_hidden[i]);
  require(hidden_changed, "hidden state was silently reset");

  ModelOptions stateless;
  stateless.stateful = false;
  const auto fixed_head_path = write_model(files, "fixed-head.bin", stateless);
  TinyNsResidual fixed_head;
  fixed_head.load(fixed_head_path.string());
  std::array<float, 64> fixed_residual{}, fixed_hidden{};
  require(fixed_head.process(features.data(), fixed_residual.data(),
                             fixed_hidden.data()),
          "fixed-head GRU inference failed");
  const float expected_fixed_residual = 12.f * std::tanh(.5f);
  for (float value : fixed_residual)
    require(std::abs(value - expected_fixed_residual) < 1e-6f,
            "head output omitted the fixed 12x tanh scale");

  ModelOptions untrained;
  untrained.trained = 0;
  const auto bad_trained = write_model(files, "untrained.bin", untrained);
  require_throw([&] { TinyNsResidual bad; bad.load(bad_trained.string()); },
                "untrained model");
  ModelOptions wrong_shape;
  wrong_shape.input = 255;
  const auto bad_shape = write_model(files, "shape.bin", wrong_shape);
  require_throw([&] { TinyNsResidual bad; bad.load(bad_shape.string()); },
                "wrong-shape model");
  ModelOptions wrong_rate;
  wrong_rate.rate = 24000;
  const auto bad_rate = write_model(files, "rate.bin", wrong_rate);
  require_throw([&] { TinyNsResidual bad; bad.load(bad_rate.string()); },
                "wrong-rate model");
  ModelOptions bad_checksum;
  bad_checksum.corrupt_hash = true;
  const auto bad_hash = write_model(files, "hash.bin", bad_checksum);
  require_throw([&] { TinyNsResidual bad; bad.load(bad_hash.string()); },
                "wrong-hash model");
  ModelOptions nonfinite;
  nonfinite.nonfinite = true;
  const auto bad_value = write_model(files, "nonfinite.bin", nonfinite);
  require_throw([&] { TinyNsResidual bad; bad.load(bad_value.string()); },
                "non-finite model");
}

void feature_checks() {
  ResidualFeatureBands bands;
  require(ResidualFeatureBands::bins == 257, "feature FFT bin count differs");
  require(ResidualFeatureBands::input_size == 256, "feature width differs");
  require(bands.left_band(0) == 0, "zero Hz did not map to band 0");
  require(bands.left_band(256) == 62 && bands.right_weight(256) == 1.f,
          "Nyquist did not map to band 63");

  std::array<float, 257> power{}, noise{};
  power.fill(1.f);
  noise.fill(2.f);
  require(bands.process(power.data(), noise.data()), "constant feature frame failed");
  for (unsigned band = 0; band < 64; ++band) {
    require(std::abs(bands.projected_power()[band] - 1.f) < 1e-7f,
            "power normalization differs");
    require(std::abs(bands.projected_noise()[band] - 2.f) < 1e-7f,
            "noise normalization differs");
    require(std::abs(bands.features()[band]) < 1e-7f,
            "power dB feature differs");
    require(std::abs(bands.features()[64 + band] - 10.f * std::log10(2.f)) <
                1e-6f, "noise dB feature differs");
    require(same_bits(bands.features()[192 + band], 0.f),
            "first-frame delta was not zero");
  }
  power.fill(2.f);
  noise.fill(4.f);
  require(bands.process(power.data(), noise.data()), "second feature frame failed");
  const float expected_delta = 10.f * std::log10(2.f);
  for (unsigned band = 0; band < 64; ++band)
    require(std::abs(bands.features()[192 + band] - expected_delta) < 1e-6f,
            "current-minus-previous dB feature differs");

  std::array<float, 64> band_values{};
  for (unsigned band = 0; band < 64; ++band) band_values[band] = band;
  require(same_bits(bands.interpolate(band_values.data(), 0), 0.f),
          "interpolation low endpoint differs");
  require(same_bits(bands.interpolate(band_values.data(), 256), 63.f),
          "interpolation high endpoint differs");

  ResidualFeatureBands state_guard;
  power.fill(1.f);
  noise.fill(2.f);
  require(state_guard.process(power.data(), noise.data()),
          "state guard baseline failed");
  power[5] = -.1f;
  require(!state_guard.process(power.data(), noise.data()),
          "negative raw power bin was accepted");
  power.fill(2.f);
  noise.fill(4.f);
  require(state_guard.process(power.data(), noise.data()),
          "state guard continuation failed");
  for (unsigned band = 0; band < 64; ++band)
    require(std::abs(state_guard.features()[192 + band] - expected_delta) < 1e-6f,
            "failed feature frame advanced previous-power state");

  power.fill(0.f);
  noise.fill(0.f);
  require(bands.process(power.data(), noise.data()),
          "zero-power epsilon clamp frame failed");
  const float expected_epsilon_db =
      10.f * std::log10(ResidualFeatureBands::epsilon);
  for (unsigned band = 0; band < 64; ++band) {
    require(std::abs(bands.features()[band] - expected_epsilon_db) < 1e-5f,
            "zero power did not clamp to epsilon");
    require(std::abs(bands.features()[64 + band] - expected_epsilon_db) < 1e-5f,
            "zero noise did not clamp to epsilon");
  }

  power.fill(1.f);
  noise.fill(1.f);
  power[10] = std::numeric_limits<float>::infinity();
  require(!bands.process(power.data(), noise.data()),
          "non-finite feature input was accepted");
  require_throw([] { ResidualFeatureBands invalid(16000, 256); },
                "wrong FFT feature frontend");
}

void control_checks(TemporaryFiles& files) {
  const auto valid = write_model(files, "control.bin");
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
  power.fill(1.f);
  noise.fill(.5f);
  xi.fill(1.f);
  gamma.fill(2.f);
  probability.fill(.5f);
  gain.fill(.1f);
  const auto original_gain = gain;
  const NsSpectralFrame frame{16000, 512, 257, power.data(), noise.data(),
                              xi.data(), gamma.data(), probability.data(),
                              gain.data()};

  ResidualNsControl disabled(16000, 512, valid.string(), -9.f, 0.f);
  disabled.observe(frame);
  require(disabled.inferences() == 1, "mix=0 did not forward the model");
  require(disabled.applied_frames() == 0 && !disabled.applied(),
          "mix=0 reported an applied gain frame");
  for (unsigned bin = 0; bin < 257; ++bin)
    require(same_bits(gain[bin], original_gain[bin]),
            "mix=0 changed gain below the floor");

  gain = original_gain;
  ResidualNsControl enabled(16000, 512, valid.string(), -9.f, 1.f);
  enabled.observe(frame);
  require(enabled.inferences() == 1 && enabled.applied_frames() == 1,
          "enabled control counters differ");
  for (unsigned bin = 0; bin < 257; ++bin) {
    const float expected = std::clamp(
        original_gain[bin] * std::pow(10.f, enabled.residual_db(bin) / 20.f),
        enabled.floor_gain(), 1.f);
    require(std::abs(gain[bin] - expected) < 1e-7f,
            "residual gain formula differs");
    require(same_bits(enabled.base_gain()[bin], original_gain[bin]),
            "base gain snapshot differs");
  }

  ResidualNsControl fallback(16000, 512, {}, -9.f, 1.f);
  gain = original_gain;
  fallback.observe(frame);
  require(fallback.fallback_frames() == 1 && fallback.inferences() == 0,
          "missing model did not count fallback");
  for (unsigned bin = 0; bin < 257; ++bin)
    require(same_bits(gain[bin], original_gain[bin]),
            "fallback did not preserve base gain");
  require_throw([&] { ResidualNsControl bad(16000, 256, valid.string()); },
                "wrong-FFT residual control");
  require_throw([&] { ResidualNsControl bad(24000, 512, valid.string()); },
                "wrong-rate residual control");
  require_throw([&] { ResidualNsControl bad(16000, 512, valid.string(), -40.1f); },
                "out-of-range residual floor");

  gain = original_gain;
  gain[17] = 1.01f;
  const auto invalid_base = gain;
  ResidualNsControl invalid_gain_control(16000, 512, valid.string(), -9.f, 1.f);
  invalid_gain_control.observe(frame);
  require(invalid_gain_control.fallback_frames() == 1 &&
              invalid_gain_control.inferences() == 0,
          "base gain outside [0,1] did not fall back");
  for (unsigned bin = 0; bin < 257; ++bin)
    require(same_bits(gain[bin], invalid_base[bin]),
            "invalid base gain was modified");

  ModelOptions negative_head;
  negative_head.stateful = false;
  negative_head.head_bias = -100.f;
  const auto negative_path = write_model(files, "negative-head.bin", negative_head);
  ResidualNsControl floor_control(16000, 512, negative_path.string(), -21.f, 1.f);
  gain.fill(.1f);
  floor_control.observe(frame);
  const float expected_floor = std::pow(10.f, -21.f / 20.f);
  require(same_bits(floor_control.floor_gain(), expected_floor),
          "final floor gain conversion differs");
  for (unsigned bin = 0; bin < 257; ++bin)
    require(same_bits(gain[bin], expected_floor),
            "constant -12 dB residual did not clamp to -21 dB final floor");
}

struct Recorder final : NsSpectralControl {
  void observe(NsSpectralFrame frame) noexcept override {
    for (unsigned bin = 0; bin < frame.bins; ++bin) {
      power[bin] = frame.power[bin];
      noise[bin] = frame.noise[bin];
      xi[bin] = frame.xi[bin];
      gamma[bin] = frame.gamma[bin];
      probability[bin] = frame.probability[bin];
      gain[bin] = frame.gain[bin];
    }
    ++frames;
  }
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
  std::size_t frames = 0;
};

void statistics_independence(TemporaryFiles& files) {
  const auto model_path = write_model(files, "stats.bin");
  OmlsaImcra baseline(16000, -18.f);
  OmlsaImcra enabled(16000, -18.f);
  Recorder baseline_recorder, enabled_recorder;
  ResidualNsControl residual(16000, 512, model_path.string(), -9.f, 1.f);
  NsSpectralChain chain(&residual, &enabled_recorder);
  std::mt19937 generator(773);
  std::normal_distribution<float> random(0.f, .015f);
  std::size_t gain_differences = 0;
  for (unsigned frame_index = 0; frame_index < 80; ++frame_index) {
    std::array<float, 160> input{};
    for (unsigned i = 0; i < input.size(); ++i) {
      input[i] = random(generator) +
                 .01f * std::sin(6.2831853f * 440.f *
                                (frame_index * input.size() + i) / 16000.f);
    }
    auto enabled_input = input;
    baseline.process(input.data(), &baseline_recorder);
    enabled.process(enabled_input.data(), &chain);
    require(baseline_recorder.frames == enabled_recorder.frames,
            "recording callback count differs");
    for (unsigned bin = 0; bin < 257; ++bin) {
      require(same_bits(baseline_recorder.power[bin], enabled_recorder.power[bin]),
              "power statistic changed");
      require(same_bits(baseline_recorder.noise[bin], enabled_recorder.noise[bin]),
              "noise statistic changed");
      require(same_bits(baseline_recorder.xi[bin], enabled_recorder.xi[bin]),
              "xi statistic changed");
      require(same_bits(baseline_recorder.gamma[bin], enabled_recorder.gamma[bin]),
              "gamma statistic changed");
      require(same_bits(baseline_recorder.probability[bin],
                        enabled_recorder.probability[bin]),
              "probability statistic changed");
      gain_differences += !same_bits(baseline_recorder.gain[bin],
                                     enabled_recorder.gain[bin]);
    }
    for (float sample : enabled_input)
      require(std::isfinite(sample), "enabled OM-LSA output became non-finite");
  }
  require(gain_differences > 0, "statistics test did not exercise a gain change");
  require(residual.inferences() == 80 && residual.fallback_frames() == 0,
          "residual counters differ in OM-LSA integration");

  ResidualFeatureBands first, second;
  std::array<float, 257> power{}, noise{};
  for (unsigned frame = 0; frame < 5; ++frame) {
    for (unsigned bin = 0; bin < 257; ++bin) {
      power[bin] = .01f + frame + bin * .001f;
      noise[bin] = .005f + .5f * frame + bin * .0005f;
    }
    require(first.process(power.data(), noise.data()), "causality first stream failed");
    require(second.process(power.data(), noise.data()), "causality second stream failed");
    for (unsigned i = 0; i < first.features().size(); ++i)
      require(same_bits(first.features()[i], second.features()[i]),
              "identical prefix produced different features");
  }
  std::cout << "statistics_independent_frames=" << residual.inferences()
            << " gain_bin_differences=" << gain_differences << '\n';
}

}  // namespace

int main() {
  try {
    TemporaryFiles files;
    model_and_hash_checks(files);
    feature_checks();
    control_checks(files);
    statistics_independence(files);
    std::cout << "C64 residual model/features/control checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
