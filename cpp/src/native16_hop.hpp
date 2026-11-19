#pragma once

#include <algorithm>
#include <cstdint>

namespace itu {
// Exact native-hop interface: every accepted call consumes 256 new samples.
// There is no 160/256 FIFO and no additional PCM shift. The model itself owns
// its measured 256-sample codec delay, including its one-hop EOS tail.
class Native16HopAdapter {
 public:
  static constexpr unsigned kDelaySamples = 0;
  static constexpr unsigned kCallbackSamples = 256;
  static constexpr unsigned kModelHopSamples = 256;

  template<class Process>
  bool process(const float* mic, const float* ref, float* out, Process&& model) noexcept {
    if (!mic || !ref || !out || finished_ || failed_) return false;
    if (!model(mic, ref, out)) { failed_ = true; return false; }
    ++inferences_;
    return true;
  }

  bool finish_after_hops(std::uint64_t expected) noexcept {
    if (failed_ || expected != inferences_) return false;
    finished_ = true;
    return true;
  }

  bool drain(float* out) noexcept {
    if (!out || !finished_ || failed_) return false;
    std::fill_n(out, kCallbackSamples, 0.f);
    return true;
  }

  bool finished() const noexcept { return finished_; }
  std::uint64_t inferences() const noexcept { return inferences_; }
  unsigned delay_samples() const noexcept { return 0; }

 private:
  std::uint64_t inferences_ = 0;
  bool finished_ = false;
  bool failed_ = false;
};
}  // namespace itu
