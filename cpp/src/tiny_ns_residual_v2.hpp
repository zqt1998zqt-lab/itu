#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace itu::detail {

// Independent ITUNSR02 deployment loader. It accepts the loss-only 256-wide
// fixture and the base-gain-aware 320-wide model without changing the v1
// ITUNSR01 loader. Tensor payload is allocated only in load().
class TinyNsResidualV2 {
 public:
  static constexpr unsigned projection_size = 64;
  static constexpr unsigned hidden_size = 64;
  static constexpr unsigned output_size = 64;
  static constexpr unsigned bands = 64;
  static constexpr unsigned sample_rate = 16000;
  static constexpr unsigned fft_size = 512;
  static constexpr unsigned hop_size = 160;
  static constexpr unsigned window_size = 320;
  static constexpr std::size_t header_size = 64;
  static constexpr std::size_t checksum_size = 8;

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

  static constexpr std::size_t parameter_count_for(unsigned input) noexcept {
    return input == 256 ? 45568u : input == 320 ? 49664u : 0u;
  }

  static constexpr std::size_t file_size_for(unsigned input) noexcept {
    const std::size_t count = parameter_count_for(input);
    return count ? header_size + count * sizeof(float) + checksum_size : 0u;
  }

  void load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open v2 residual model: " + path);
    const auto total = file.tellg();
    if (total < static_cast<std::streamoff>(header_size))
      throw std::runtime_error("v2 residual model is shorter than its header");
    std::array<unsigned char, header_size> header{};
    file.seekg(0);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    if (!file || std::memcmp(header.data(), "ITUNSR02", 8) != 0)
      throw std::runtime_error("v2 residual model has wrong magic");

    Metadata metadata;
    metadata.version = read_u32(header.data() + 8);
    metadata.sample_rate = read_u32(header.data() + 12);
    metadata.bands = read_u32(header.data() + 16);
    metadata.input = read_u32(header.data() + 20);
    metadata.projection = read_u32(header.data() + 24);
    metadata.hidden = read_u32(header.data() + 28);
    metadata.output = read_u32(header.data() + 32);
    metadata.count = read_u32(header.data() + 36);
    metadata.trained = read_u32(header.data() + 40);
    metadata.feature_version = read_u32(header.data() + 44);
    metadata.fft = read_u32(header.data() + 48);
    metadata.hop = read_u32(header.data() + 52);
    metadata.window = read_u32(header.data() + 56);
    metadata.reserved = read_u32(header.data() + 60);

    if (metadata.version != 2)
      throw std::runtime_error("v2 residual model has wrong version");
    if (metadata.sample_rate != sample_rate || metadata.bands != bands ||
        metadata.projection != projection_size ||
        metadata.hidden != hidden_size || metadata.output != output_size ||
        metadata.fft != fft_size || metadata.hop != hop_size ||
        metadata.window != window_size || metadata.reserved != 0)
      throw std::runtime_error("v2 residual model has wrong frontend metadata");
    if (metadata.trained != 1)
      throw std::runtime_error("v2 residual model is untrained");
    const bool input_256 = metadata.input == 256 &&
                           metadata.count == 45568 && metadata.feature_version == 1;
    const bool input_320 = metadata.input == 320 &&
                           metadata.count == 49664 && metadata.feature_version == 2;
    if (!input_256 && !input_320)
      throw std::runtime_error("v2 residual model has inconsistent input/count/feature version");
    const std::size_t expected_size = file_size_for(metadata.input);
    if (static_cast<std::uint64_t>(total) != expected_size)
      throw std::runtime_error("v2 residual model has wrong byte size");

    std::vector<unsigned char> bytes(expected_size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) throw std::runtime_error("v2 residual model read failed");
    const std::size_t hashed_size = expected_size - checksum_size;
    if (fnv1a64(bytes.data(), hashed_size) != read_u64(bytes.data() + hashed_size))
      throw std::runtime_error("v2 residual model checksum mismatch");

    std::vector<float> weights(metadata.count);
    std::size_t cursor = header_size;
    for (float& value : weights) {
      value = read_f32(bytes.data() + cursor);
      cursor += sizeof(float);
      if (!std::isfinite(value))
        throw std::runtime_error("v2 residual model contains a non-finite parameter");
    }
    if (cursor != hashed_size)
      throw std::runtime_error("v2 residual model tensor layout mismatch");

    weights_.swap(weights);
    metadata_ = metadata;
    hidden_.fill(0.f);
    ready_ = true;
  }

  bool ready() const noexcept { return ready_; }
  unsigned input_dim() const noexcept { return metadata_.input; }
  std::size_t parameter_count() const noexcept { return metadata_.count; }
  const Metadata& metadata() const noexcept { return metadata_; }
  const std::array<float, hidden_size>& hidden() const noexcept { return hidden_; }
  void reset() noexcept { hidden_.fill(0.f); }

  template <unsigned InputDim>
  bool process(const float* features, float* residual, float* hidden_out) noexcept {
    static_assert(InputDim == 256 || InputDim == 320);
    if (!ready_ || metadata_.input != InputDim || !features || !residual ||
        !hidden_out || weights_.size() != parameter_count_for(InputDim)) return false;
    for (unsigned i = 0; i < InputDim; ++i)
      if (!std::isfinite(features[i])) return false;

    constexpr std::size_t projection_weights = projection_size * InputDim;
    constexpr std::size_t projection_bias = projection_weights;
    constexpr std::size_t gru_ih = projection_bias + projection_size;
    constexpr std::size_t gru_hh = gru_ih + 3 * hidden_size * projection_size;
    constexpr std::size_t gru_bih = gru_hh + 3 * hidden_size * hidden_size;
    constexpr std::size_t gru_bhh = gru_bih + 3 * hidden_size;
    constexpr std::size_t head_weight = gru_bhh + 3 * hidden_size;
    constexpr std::size_t head_bias = head_weight + output_size * hidden_size;
    const float* weights = weights_.data();

    std::array<float, projection_size> projected{};
    for (unsigned out = 0; out < projection_size; ++out) {
      float sum = weights[projection_bias + out];
      const float* row = weights + out * InputDim;
      for (unsigned in = 0; in < InputDim; ++in) sum += row[in] * features[in];
      if (!std::isfinite(sum)) return false;
      projected[out] = std::max(0.f, sum);
    }

    std::array<float, hidden_size> reset_gate{};
    std::array<float, hidden_size> update_gate{};
    std::array<float, hidden_size> candidate{};
    std::array<float, hidden_size> next_hidden{};
    for (unsigned gate = 0; gate < 3; ++gate) {
      for (unsigned out = 0; out < hidden_size; ++out) {
        const unsigned row = gate * hidden_size + out;
        const float* input_row = weights + gru_ih + row * projection_size;
        const float* hidden_row = weights + gru_hh + row * hidden_size;
        float input_sum = weights[gru_bih + row];
        float hidden_sum = weights[gru_bhh + row];
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
      float sum = weights[head_bias + out];
      const float* row = weights + head_weight + out * hidden_size;
      for (unsigned in = 0; in < hidden_size; ++in) sum += row[in] * next_hidden[in];
      if (!std::isfinite(sum)) return false;
      residual[out] = 12.f * std::tanh(sum);
    }
    std::copy(next_hidden.begin(), next_hidden.end(), hidden_.begin());
    std::copy(next_hidden.begin(), next_hidden.end(), hidden_out);
    return true;
  }

  bool process(const float* features, float* residual, float* hidden_out) noexcept {
    if (metadata_.input == 256) return process<256>(features, residual, hidden_out);
    if (metadata_.input == 320) return process<320>(features, residual, hidden_out);
    return false;
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
  std::vector<float> weights_;
  std::array<float, hidden_size> hidden_{};
  bool ready_ = false;
};

static_assert(TinyNsResidualV2::file_size_for(256) == 182344);
static_assert(TinyNsResidualV2::file_size_for(320) == 198728);

}  // namespace itu::detail
