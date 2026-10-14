#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
namespace itu::detail {
class Fft {
 public:
  explicit Fft(unsigned n) : n_(n), reverse_(n), roots_(n / 2) {
    unsigned bits = 0;
    for (unsigned m = n; m > 1; m >>= 1) ++bits;
    for (unsigned i = 0; i < n; ++i) {
      unsigned x = i, r = 0;
      for (unsigned b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
      reverse_[i] = r;
    }
    for (unsigned i = 0; i < n / 2; ++i) {
      const float a = -6.2831853071795864769f * i / n;
      roots_[i] = {std::cos(a), std::sin(a)};
    }
  }
  void run(std::complex<float>* data, bool inverse = false) const noexcept {
    if (inverse) for (unsigned i = 0; i < n_; ++i) data[i] = std::conj(data[i]);
    for (unsigned i = 0; i < n_; ++i) if (i < reverse_[i]) std::swap(data[i], data[reverse_[i]]);
    for (unsigned len = 2; len <= n_; len <<= 1) {
      const unsigned step = n_ / len;
      for (unsigned base = 0; base < n_; base += len) for (unsigned j = 0; j < len / 2; ++j) {
        const auto a = data[base + j], b = roots_[j * step] * data[base + j + len / 2];
        data[base + j] = a + b; data[base + j + len / 2] = a - b;
      }
    }
    if (inverse) for (unsigned i = 0; i < n_; ++i) data[i] = std::conj(data[i]) / static_cast<float>(n_);
  }
  std::size_t memory() const noexcept { return reverse_.capacity()*sizeof(unsigned) + roots_.capacity()*sizeof(std::complex<float>); }
 private:
  unsigned n_;
  std::vector<unsigned> reverse_;
  std::vector<std::complex<float>> roots_;
};
inline unsigned next_power_two(unsigned n) { unsigned r = 1; while (r < n) r <<= 1; return r; }
}
