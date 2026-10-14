#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace itu {
struct AudioTimestamp {
  std::uint64_t frame_position=0; // Frames per channel, not interleaved samples.
  std::int64_t monotonic_ns=0; // Same timebase for playback and capture.
};
struct ClockInfo {
  bool locked=false;
  double drift_ppm=0;
  std::size_t queued_samples=0;
  std::uint64_t rejected_timestamps=0,underruns=0,overruns=0;
};
// Capture is the master clock. All calls on one processing/control thread.
// Push actual playback PCM in order, with a bounded lead over capture. This
// corrects sample rate only; AEC still estimates acoustic/buffering offset.
// After route/restart/discontinuity, reset and establish a new PCM/timestamp epoch.
class ReferenceClock {
 public:
  explicit ReferenceClock(unsigned sample_rate);
  ~ReferenceClock();
  void reset();
  bool observe_capture(AudioTimestamp timestamp) noexcept;
  bool observe_render(AudioTimestamp timestamp) noexcept;
  bool push(const float* mono,std::size_t samples) noexcept;
  // Exactly 10 ms; false means data unavailable, output/state are not advanced.
  bool pull(float* mono,std::size_t samples) noexcept;
  ClockInfo info() const noexcept;
 private:
  struct State;
  std::unique_ptr<State> state_;
};
}
