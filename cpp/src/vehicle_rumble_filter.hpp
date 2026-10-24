#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace itu {
// A sample-causal, native-rate rumble conditioner. No extra queue or lookahead.
// This IIR has frequency-dependent phase; callers must score speech fidelity,
// rather than treating zero buffering as a claim of zero group delay.
class VehicleRumbleFilter {
 public:
  VehicleRumbleFilter(unsigned rate, float corner_hz) : enabled_(corner_hz > 0) {
    if ((rate != 16000 && rate != 24000 && rate != 48000) ||
        !std::isfinite(corner_hz) || corner_hz < 0 || corner_hz > 120)
      throw std::invalid_argument("invalid native-rate rumble filter");
    if (!enabled_) return;
    const double w = 6.2831853071795864769 * corner_hz / rate;
    const double c = std::cos(w), a = std::sin(w) / std::sqrt(2.0);
    b0_ = (1 + c) / (2 * (1 + a)); b1_ = -2 * b0_; b2_ = b0_;
    a1_ = -2 * c / (1 + a); a2_ = (1 - a) / (1 + a);
  }
  void process(float* samples, unsigned count) noexcept {
    if (!enabled_) return;
    for (unsigned i = 0; i < count; ++i) {
      const double x = samples[i], y = b0_ * x + z1_;
      z1_ = b1_ * x - a1_ * y + z2_;
      z2_ = b2_ * x - a2_ * y;
      samples[i] = static_cast<float>(y);
    }
  }
 private:
  bool enabled_;
  double b0_=1, b1_=0, b2_=0, a1_=0, a2_=0, z1_=0, z2_=0;
};

// Explicit 20 ms linear-phase option. The common delay applies to the complete
// wet/dry output, so phase-sensitive preservation does not mix different time
// bases. All buffers and coefficients are allocated before processing starts.
class LinearRumbleFilter {
 public:
  LinearRumbleFilter(unsigned rate, float corner_hz)
      : rate_(rate), delay_(corner_hz > 0 ? rate / 50 : 0),
        size_(2 * delay_ + 1), coefficients_(delay_ + 1), history_(2 * size_) {
    if ((rate != 16000 && rate != 24000 && rate != 48000) ||
        !std::isfinite(corner_hz) || corner_hz < 0 || corner_hz > 120)
      throw std::invalid_argument("invalid linear rumble filter");
    if (!delay_) return;
    constexpr double pi = 3.14159265358979323846;
    double nyquist = 0;
    for (unsigned j = 0; j <= delay_; ++j) {
      const int m = static_cast<int>(j) - static_cast<int>(delay_);
      const double lp = m ? std::sin(2 * pi * corner_hz * m / rate) / (pi * m)
                          : 2 * corner_hz / rate;
      const double w = .54 - .46 * std::cos(2 * pi * j / (size_ - 1));
      const double h = (m == 0 ? 1.0 : 0.0) - lp * w;
      coefficients_[j] = static_cast<float>(h);
      nyquist += (j == delay_ ? 1 : 2) * h * (j % 2 ? -1 : 1);
    }
    for (float& c : coefficients_) c /= static_cast<float>(nyquist);
  }
  unsigned delay() const noexcept { return delay_; }
  void retune(float corner_hz) noexcept {
    if (!delay_ || corner_hz <= 0) return;
    constexpr double pi = 3.14159265358979323846;
    double nyquist = 0;
    for (unsigned j = 0; j <= delay_; ++j) {
      const int m = static_cast<int>(j) - static_cast<int>(delay_);
      const double lp = m ? std::sin(2 * pi * corner_hz * m / rate_) / (pi * m)
                          : 2 * corner_hz / rate_;
      const double w = .54 - .46 * std::cos(2 * pi * j / (size_ - 1));
      const double h = (m == 0 ? 1.0 : 0.0) - lp * w;
      coefficients_[j] = static_cast<float>(h);
      nyquist += (j == delay_ ? 1 : 2) * h * (j % 2 ? -1 : 1);
    }
    for (float& c : coefficients_) c /= static_cast<float>(nyquist);
  }
  float delayed_sample(float sample) noexcept {
    if (!delay_) return sample;
    history_[head_] = history_[head_ + size_] = sample;
    return history_[head_ + size_ - delay_];
  }
  float apply(float delayed) noexcept {
    if (!delay_) return delayed;
    const float* newest = history_.data() + head_ + size_;
    float y = coefficients_[delay_] * delayed;
    for (unsigned j = 0; j < delay_; ++j)
      y += coefficients_[j] * (newest[-static_cast<int>(j)] +
                              newest[-static_cast<int>(size_ - 1 - j)]);
    if (++head_ == size_) head_ = 0;
    return y;
  }
  void process(float* samples, unsigned count, const float* amount=nullptr) noexcept {
    for (unsigned i = 0; i < count; ++i) {
      const float dry=delayed_sample(samples[i]);
      const float y=apply(dry);
      samples[i] = amount ? dry+amount[i]*(y-dry) : y;
    }
  }
 private:
  unsigned rate_, delay_, size_, head_ = 0;
  std::vector<float> coefficients_, history_;
};
}
