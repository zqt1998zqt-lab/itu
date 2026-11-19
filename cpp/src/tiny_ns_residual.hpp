#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace itu::detail {

// Fixed-width, allocation-free C64 residual inference.  The on-disk tensor
// order matches the deployment contract:
// projection.weight[64,256], projection.bias[64],
// gru.weight_ih_l0[192,64], gru.weight_hh_l0[192,64],
// gru.bias_ih_l0[192], gru.bias_hh_l0[192],
// head.weight[64,64], head.bias[64].
class TinyNsResidual {
 public:
  static constexpr unsigned input_size = 256;
  static constexpr unsigned projection_size = 64;
  static constexpr unsigned hidden_size = 64;
  static constexpr unsigned output_size = 64;
  static constexpr unsigned bands = 64;
  static constexpr unsigned sample_rate = 16000;
  static constexpr unsigned fft_size = 512;
  static constexpr unsigned hop_size = 160;
  static constexpr unsigned window_size = 320;
  static constexpr unsigned feature_version = 1;
  static constexpr std::size_t parameter_count = 45568;
  static constexpr std::size_t header_size = 64;
  static constexpr std::size_t checksum_size = 8;
  static constexpr std::size_t file_size =
      header_size + parameter_count * sizeof(float) + checksum_size;

  struct Metadata {
    std::uint32_t version = 0;
    std::uint32_t sample_rate = 0;
    std::uint32_t bands = 0;
    std::uint32_t input = 0;
    std::uint32_t projection = 0;
    std::uint32_t hidden = 0;
    std::uint32_t output = 0;
    std::uint32_t count = 0;
    std::uint32_t trained = 0;
    std::uint32_t feature_version = 0;
    std::uint32_t fft = 0;
    std::uint32_t hop = 0;
    std::uint32_t window = 0;
    std::uint32_t reserved = 0;
  };

  TinyNsResidual() = default;

  void load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open residual model: " + path);
    const auto end = file.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) != file_size)
      throw std::runtime_error("residual model has wrong byte size");
    std::vector<unsigned char> bytes(file_size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) throw std::runtime_error("residual model read failed");
    if (std::memcmp(bytes.data(), "ITUNSR01", 8) != 0)
      throw std::runtime_error("residual model has wrong magic");

    Metadata metadata;
    metadata.version = read_u32(bytes.data() + 8);
    metadata.sample_rate = read_u32(bytes.data() + 12);
    metadata.bands = read_u32(bytes.data() + 16);
    metadata.input = read_u32(bytes.data() + 20);
    metadata.projection = read_u32(bytes.data() + 24);
    metadata.hidden = read_u32(bytes.data() + 28);
    metadata.output = read_u32(bytes.data() + 32);
    metadata.count = read_u32(bytes.data() + 36);
    metadata.trained = read_u32(bytes.data() + 40);
    metadata.feature_version = read_u32(bytes.data() + 44);
    metadata.fft = read_u32(bytes.data() + 48);
    metadata.hop = read_u32(bytes.data() + 52);
    metadata.window = read_u32(bytes.data() + 56);
    metadata.reserved = read_u32(bytes.data() + 60);
    if (metadata.version != 1) throw std::runtime_error("residual model has wrong version");
    if (metadata.sample_rate != sample_rate)
      throw std::runtime_error("residual model requires 16000 Hz");
    if (metadata.bands != bands) throw std::runtime_error("residual model has wrong band count");
    if (metadata.input != input_size)
      throw std::runtime_error("residual model has wrong input shape");
    if (metadata.projection != projection_size)
      throw std::runtime_error("residual model has wrong projection shape");
    if (metadata.hidden != hidden_size)
      throw std::runtime_error("residual model has wrong hidden shape");
    if (metadata.output != output_size)
      throw std::runtime_error("residual model has wrong output shape");
    if (metadata.count != parameter_count)
      throw std::runtime_error("residual model has wrong parameter count");
    if (metadata.trained != 1)
      throw std::runtime_error("residual model is untrained");
    if (metadata.feature_version != feature_version)
      throw std::runtime_error("residual model has wrong feature version");
    if (metadata.fft != fft_size || metadata.hop != hop_size ||
        metadata.window != window_size || metadata.reserved != 0)
      throw std::runtime_error("residual model has wrong frontend metadata");

    const std::size_t hashed_size = file_size - checksum_size;
    const std::uint64_t actual_hash = fnv1a64(bytes.data(), hashed_size);
    const std::uint64_t stored_hash = read_u64(bytes.data() + hashed_size);
    if (actual_hash != stored_hash)
      throw std::runtime_error("residual model checksum mismatch");

    TinyNsResidual loaded;
    loaded.metadata_ = metadata;
    std::size_t cursor = header_size;
    auto read_weights = [&](float* destination, std::size_t count) {
      for (std::size_t i = 0; i < count; ++i, cursor += 4) {
        const float value = read_f32(bytes.data() + cursor);
        if (!std::isfinite(value))
          throw std::runtime_error("residual model contains a non-finite parameter");
        destination[i] = value;
      }
    };
    read_weights(loaded.projection_weight_.data(), loaded.projection_weight_.size());
    read_weights(loaded.projection_bias_.data(), loaded.projection_bias_.size());
    read_weights(loaded.gru_weight_ih_.data(), loaded.gru_weight_ih_.size());
    read_weights(loaded.gru_weight_hh_.data(), loaded.gru_weight_hh_.size());
    read_weights(loaded.gru_bias_ih_.data(), loaded.gru_bias_ih_.size());
    read_weights(loaded.gru_bias_hh_.data(), loaded.gru_bias_hh_.size());
    read_weights(loaded.head_weight_.data(), loaded.head_weight_.size());
    read_weights(loaded.head_bias_.data(), loaded.head_bias_.size());
    if (cursor != hashed_size) throw std::runtime_error("residual model tensor layout mismatch");
    loaded.ready_ = true;
    *this = loaded;
  }

  bool ready() const noexcept { return ready_; }
  const Metadata& metadata() const noexcept { return metadata_; }
  const std::array<float, hidden_size>& hidden() const noexcept { return hidden_; }

  void reset() noexcept {
    hidden_.fill(0.f);
  }

  bool process(const float* features, float* residual) noexcept {
    return process(features, residual, hidden_.data());
  }

  // Explicit hidden output is returned for parity probes.  The internal state
  // is advanced only after every output has been checked for finiteness.
  bool process(const float* features, float* residual, float* hidden_out) noexcept {
    if (!ready_ || !features || !residual || !hidden_out) return false;
    for (unsigned i = 0; i < input_size; ++i)
      if (!std::isfinite(features[i])) return false;

    std::array<float, projection_size> projected{};
    for (unsigned out = 0; out < projection_size; ++out) {
      const float* row = projection_weight_.data() + out * input_size;
      float sum = projection_bias_[out];
      for (unsigned in = 0; in < input_size; ++in) sum += row[in] * features[in];
      if (!std::isfinite(sum)) return false;
      projected[out] = std::max(0.f, sum);
    }

    std::array<float, hidden_size> reset_gate{};
    std::array<float, hidden_size> update_gate{};
    std::array<float, hidden_size> candidate{};
    std::array<float, hidden_size> next_hidden{};
    constexpr unsigned gate_stride = hidden_size;
    for (unsigned gate = 0; gate < 3; ++gate) {
      for (unsigned out = 0; out < hidden_size; ++out) {
        const unsigned row_index = gate * gate_stride + out;
        const float* input_row =
            gru_weight_ih_.data() + row_index * projection_size;
        const float* hidden_row =
            gru_weight_hh_.data() + row_index * hidden_size;
        float input_sum = gru_bias_ih_[row_index];
        float hidden_sum = gru_bias_hh_[row_index];
        for (unsigned in = 0; in < projection_size; ++in)
          input_sum += input_row[in] * projected[in];
        for (unsigned in = 0; in < hidden_size; ++in)
          hidden_sum += hidden_row[in] * hidden_[in];
        if (!std::isfinite(input_sum) || !std::isfinite(hidden_sum)) return false;
        if (gate == 0) reset_gate[out] = sigmoid(input_sum + hidden_sum);
        else if (gate == 1) update_gate[out] = sigmoid(input_sum + hidden_sum);
        else candidate[out] = std::tanh(input_sum + reset_gate[out] * hidden_sum);
      }
    }
    for (unsigned i = 0; i < hidden_size; ++i) {
      next_hidden[i] = (1.f - update_gate[i]) * candidate[i] +
                       update_gate[i] * hidden_[i];
      if (!std::isfinite(next_hidden[i])) return false;
    }

    for (unsigned out = 0; out < output_size; ++out) {
      const float* row = head_weight_.data() + out * hidden_size;
      float sum = head_bias_[out];
      for (unsigned in = 0; in < hidden_size; ++in) sum += row[in] * next_hidden[in];
      if (!std::isfinite(sum)) return false;
      residual[out] = 12.f * std::tanh(sum);
    }
    std::copy(next_hidden.begin(), next_hidden.end(), hidden_.begin());
    std::copy(next_hidden.begin(), next_hidden.end(), hidden_out);
    return true;
  }

  static std::uint64_t fnv1a64(const unsigned char* bytes, std::size_t size) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
    }
    return hash;
  }

 private:
  static std::uint32_t read_u32(const unsigned char* bytes) noexcept {
    return std::uint32_t(bytes[0]) | std::uint32_t(bytes[1]) << 8 |
           std::uint32_t(bytes[2]) << 16 | std::uint32_t(bytes[3]) << 24;
  }

  static std::uint64_t read_u64(const unsigned char* bytes) noexcept {
    return std::uint64_t(read_u32(bytes)) |
           std::uint64_t(read_u32(bytes + 4)) << 32;
  }

  static float read_f32(const unsigned char* bytes) noexcept {
    return std::bit_cast<float>(read_u32(bytes));
  }

  static float sigmoid(float value) noexcept {
    if (value >= 0.f) return 1.f / (1.f + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.f + exponential);
  }

  Metadata metadata_{};
  std::array<float, projection_size * input_size> projection_weight_{};
  std::array<float, projection_size> projection_bias_{};
  std::array<float, 3 * hidden_size * projection_size> gru_weight_ih_{};
  std::array<float, 3 * hidden_size * hidden_size> gru_weight_hh_{};
  std::array<float, 3 * hidden_size> gru_bias_ih_{};
  std::array<float, 3 * hidden_size> gru_bias_hh_{};
  std::array<float, output_size * hidden_size> head_weight_{};
  std::array<float, output_size> head_bias_{};
  std::array<float, hidden_size> hidden_{};
  bool ready_ = false;
};

static_assert(TinyNsResidual::file_size == 182344,
              "C64 residual model size changed");
static_assert(TinyNsResidual::parameter_count == 45568,
              "C64 residual parameter count changed");

}  // namespace itu::detail
