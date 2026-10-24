#pragma once
#include <algorithm>
#include <cmath>

namespace itu::detail {
// Lotter/Vary, EURASIP JASP 2005, doi:10.1155/ASP.2005.1110, eq. (41).
// Joint MAP amplitude AND phase, not the Bessel approximation in eq. (33).
// p(A) is proportional to A^nu exp(-mu*A/sigma_s); Kullback-fit parameters
// from the paper are mu=1.74, nu=.126. The noisy phase is retained.
inline float super_gaussian_h1_gain(float xi,float gamma) noexcept {
  constexpr double mu=1.74,nu=.126;
  const double x=std::clamp(double(xi),1e-12,1e12);
  const double g=std::clamp(double(gamma),1e-12,1e12);
  const double u=.5-mu/(4*std::sqrt(g*x)),d=nu/(2*g);
  const double root=std::sqrt(u*u+d);
  // Rationalize the positive root when u<0; direct subtraction loses digits.
  const double gain=u>=0 ? u+root : d/(root-u);
  // Engineering constraint shared with OM-LSA: no spectral amplification.
  return float(std::clamp(gain,0.,1.));
}
enum class SpectralGainRule { omlsa,super_gaussian_map };
}
