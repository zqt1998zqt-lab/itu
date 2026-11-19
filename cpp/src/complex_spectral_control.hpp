#pragma once
#include <complex>
namespace itu::detail {
// Post-gain synthesis spectrum only; statistical evidence remains immutable.
struct ComplexSpectralFrame {
  unsigned rate, fft_size, bins;
  std::complex<float>* spectrum;
  const float* gain;
};
class ComplexSpectralControl {
 public:
  virtual ~ComplexSpectralControl() = default;
  virtual void observe(ComplexSpectralFrame frame) noexcept = 0;
};
}
