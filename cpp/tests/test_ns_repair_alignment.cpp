#include "ns_repair_alignment.hpp"
#include <array>
#include <cmath>

int main() {
  // Compatibility mode must preserve the original three-sample control delay.
  itu::NsRepairAlignment original(3, 4, 0);
  original.begin(1, 1);
  constexpr std::array<float, 4> expected{0, 0, 0, .25f};
  for (unsigned i = 0; i < 4; ++i) {
    float mix, gain;
    original.sample(i, mix, gain);
    if (mix != expected[i] || gain != 1) return 1;
  }
  // Priming protects the first frame; subsequent transitions still traverse
  // the same control ring and do not introduce an audio queue.
  itu::NsRepairAlignment fast(3, 4, 0, true);
  fast.begin(1, 1);
  for (unsigned i = 0; i < 4; ++i) {
    float mix, gain;
    fast.sample(i, mix, gain);
    if (mix != 1 || gain != 1) return 2;
  }
  fast.begin(0, 1);
  constexpr std::array<float, 4> release{1, 1, 1, .75f};
  for (unsigned i = 0; i < 4; ++i) {
    float mix, gain;
    fast.sample(i, mix, gain);
    if (mix != release[i] || gain != 1) return 3;
  }
  itu::NsRepairAlignment smooth(336, 480, 50, true);
  for (unsigned frame = 0; frame < 100; ++frame) {
    smooth.begin(frame % 2 ? 1.f : 0.f, frame % 3 ? .7f : 1.f);
    for (unsigned i = 0; i < 480; ++i) {
      float mix, gain;
      smooth.sample(i, mix, gain);
      if (!std::isfinite(mix + gain) || mix < 0 || mix > 1 ||
          gain < .7f || gain > 1) return 4;
    }
  }
  return 0;
}
