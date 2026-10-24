#pragma once
#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace itu {
// Symmetric causal FIR EQ with an odd tap count. A constrained least-squares
// fit produces a real symmetric impulse response; tests inspect that symmetry
// and measure the fixed group delay on the real streaming path.
class LinearPhaseEq {
 public:
  static unsigned taps_for(const EqConfig& c, unsigned rate) {
    unsigned requested = c.linear_phase_taps;
    if (!requested) requested = ((rate + 49) / 50) | 1u;
    if (requested < 33 || requested > 2049 || requested % 2 == 0)
      throw std::invalid_argument(
          "linear-phase EQ requires an odd 33..2049 tap count");
    return requested;
  }
  static unsigned delay_for(const EqConfig& c, unsigned rate) {
    return (taps_for(c, rate) - 1) / 2;
  }

  LinearPhaseEq(const EqConfig& c, unsigned rate)
      : taps_(taps_for(c, rate)), delay_((taps_ - 1) / 2),
        history_(taps_), coefficients_(design(c, rate, taps_)) {}

  void process(float* pcm, unsigned count) noexcept {
    for (unsigned n = 0; n < count; ++n) {
      std::move(history_.begin() + 1, history_.end(), history_.begin());
      history_.back() = pcm[n];
      float value = 0.f;
      for (unsigned k = 0; k < taps_; ++k)
        value += coefficients_[k] * history_[taps_ - 1 - k];
      pcm[n] = value;
    }
  }
  unsigned delay() const noexcept { return delay_; }
  const std::vector<float>& coefficients() const noexcept { return coefficients_; }
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this) +
        (history_.capacity() + coefficients_.capacity()) * sizeof(float);
  }

 private:
  using C = std::complex<double>;

  static C response(const EqConfig& c, double hz, unsigned rate) {
    C result(std::pow(10., c.preamp_db / 20.), 0.);
    for (unsigned i = 0; i < c.bands; ++i) {
      const auto& b = c.band[i];
      if (b.gain_db == 0) continue;
      const double a = std::pow(10., b.gain_db / 40.);
      const double w = 2. * 3.14159265358979323846 * b.frequency_hz / rate;
      const double cs = std::cos(w), sn = std::sin(w);
      const double alpha = sn / (2. * b.q), shelf = sn / std::sqrt(2.);
      double b0, b1, b2, a0, a1, a2;
      if (b.kind == EqKind::peak) {
        b0 = 1. + alpha * a; b1 = -2. * cs; b2 = 1. - alpha * a;
        a0 = 1. + alpha / a; a1 = -2. * cs; a2 = 1. - alpha / a;
      } else if (b.kind == EqKind::low_shelf) {
        const double t = 2. * std::sqrt(a) * shelf;
        b0 = a * ((a + 1.) - (a - 1.) * cs + t);
        b1 = 2. * a * ((a - 1.) - (a + 1.) * cs);
        b2 = a * ((a + 1.) - (a - 1.) * cs - t);
        a0 = (a + 1.) + (a - 1.) * cs + t;
        a1 = -2. * ((a - 1.) + (a + 1.) * cs);
        a2 = (a + 1.) + (a - 1.) * cs - t;
      } else {
        const double t = 2. * std::sqrt(a) * shelf;
        b0 = a * ((a + 1.) + (a - 1.) * cs + t);
        b1 = -2. * a * ((a - 1.) + (a + 1.) * cs);
        b2 = a * ((a + 1.) + (a - 1.) * cs - t);
        a0 = (a + 1.) - (a - 1.) * cs + t;
        a1 = 2. * ((a - 1.) - (a + 1.) * cs);
        a2 = (a + 1.) - (a - 1.) * cs - t;
      }
      const C z = std::polar(1., -2. * 3.14159265358979323846 * hz / rate);
      result *= (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z);
    }
    return result;
  }

  static std::vector<float> design(const EqConfig& c, unsigned rate,
                                   unsigned taps) {
    const unsigned half = taps / 2, center = half;
    const unsigned bins = std::min(2048u, rate / 8);
    std::vector<double> normal(bins * (half + 1));
    std::vector<double> rhs(bins);
    for (unsigned k = 0; k < bins; ++k) {
      const double hz = .5 + (rate * .49 - .5) * k / std::max(1u, bins - 1);
      const C target = response(c, hz, rate);
      const double omega = 2. * 3.14159265358979323846 * hz / rate;
      double row[1025];
      row[0] = 1.;
      for (unsigned j = 1; j <= half; ++j)
        row[j] = 2. * std::cos(omega * j);
      for (unsigned i = 0; i <= half; ++i) {
        for (unsigned j = 0; j <= half; ++j)
          normal[i * (half + 1) + j] += row[i] * row[j];
        rhs[i] += row[i] * target.real();
      }
    }
    // Bounded coefficient energy prevents an unconstrained LS solve from
    // spending its freedom on an unstable edge response.
    for (unsigned i = 0; i <= half; ++i)
      normal[i * (half + 1) + i] += 1e-10 * double(bins);
    for (unsigned col = 0; col <= half; ++col) {
      unsigned pivot = col;
      for (unsigned row = col + 1; row <= half; ++row)
        if (std::abs(normal[row * (half + 1) + col]) >
            std::abs(normal[pivot * (half + 1) + col]))
          pivot = row;
      if (pivot != col) {
        for (unsigned j = col; j <= half; ++j)
          std::swap(normal[col * (half + 1) + j],
                    normal[pivot * (half + 1) + j]);
        std::swap(rhs[col], rhs[pivot]);
      }
      const double divisor = normal[col * (half + 1) + col];
      if (std::abs(divisor) < 1e-20) continue;
      for (unsigned row = col + 1; row <= half; ++row) {
        const double factor =
            normal[row * (half + 1) + col] / divisor;
        for (unsigned j = col; j <= half; ++j)
          normal[row * (half + 1) + j] -=
              factor * normal[col * (half + 1) + j];
        rhs[row] -= factor * rhs[col];
      }
    }
    std::vector<double> solution(half + 1);
    for (int i = int(half); i >= 0; --i) {
      double value = rhs[i];
      for (unsigned j = i + 1; j <= half; ++j)
        value -= normal[i * (half + 1) + j] * solution[j];
      solution[i] = value / normal[i * (half + 1) + i];
    }
    std::vector<float> result(taps);
    for (unsigned i = 0; i <= half; ++i)
      result[center - i] = result[center + i] = float(solution[i]);
    return result;
  }

  unsigned taps_, delay_;
  std::vector<float> history_, coefficients_;
};
}
