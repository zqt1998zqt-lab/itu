#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace itu {

// Convert the 160-sample callback clock to LocalVQE's native 256-sample hop
// clock without replaying samples. The adapter preloads 224 zero samples so
// every callback can emit 160 samples while the first native hop accumulates.
// After that, each native hop output is queued and drained in callback order.
//
// finish_after_hops() freezes only after callbacks have naturally advanced to
// the requested native hop count; it never adds a tail-compute burst. finish()
// is the unknown-length compatibility wrapper and advances at most one
// additional partial hop. Once finished, drain() emits the complete queued
// native tail and only then zero-fills downstream callbacks.
class ContinuousHopAdapter {
 public:
  static constexpr unsigned kCallbackSamples = 160;
  static constexpr unsigned kModelHopSamples = 256;
  static constexpr unsigned kDelaySamples =
      kModelHopSamples - std::gcd(kCallbackSamples, kModelHopSamples);
  static constexpr unsigned kInputFifoCapacity =
      kModelHopSamples + kCallbackSamples;
  static constexpr unsigned kOutputFifoCapacity =
      kDelaySamples + 4 * kModelHopSamples;

  static_assert(kDelaySamples == 224);
  static_assert(kInputFifoCapacity >=
                (kModelHopSamples - 1) + kCallbackSamples);
  static_assert(kOutputFifoCapacity >= kDelaySamples + 4 * kModelHopSamples);

  ContinuousHopAdapter() noexcept { reset(); }

  template <class Process>
  bool process(const float* mic, const float* ref, float* out,
               Process&& model) noexcept {
    if (!mic || !ref || !out || finished_ || failed_ ||
        input_size_ + kCallbackSamples > input_mic_.size())
      return false;

    std::copy_n(mic, kCallbackSamples, input_mic_.begin() + input_size_);
    std::copy_n(ref, kCallbackSamples, input_ref_.begin() + input_size_);
    input_size_ += kCallbackSamples;
    input_samples_ += kCallbackSamples;

    if (input_size_ >= kModelHopSamples) {
      if (!run_hop(model)) {
        failed_ = true;
        return false;
      }
    }

    // A 224-sample prefill covers every 160/256 phase. Reaching this branch
    // means the timeline contract is broken rather than merely starting late.
    if (output_size_ < kCallbackSamples) {
      failed_ = true;
      return false;
    }
    emit_callback(out);
    return true;
  }

  template <class Process>
  bool finish(Process&& model) noexcept {
    const auto target = inferences_ + (input_size_ ? 1u : 0u);
    if (!advance_to_hops(target, model)) {
      failed_ = true;
      return false;
    }
    return freeze();
  }

  bool finish_after_hops(std::uint64_t target_hops) noexcept {
    if (finished_) return !failed_ && inferences_ == target_hops;
    if (failed_ || target_hops != inferences_) {
      failed_ = true;
      return false;
    }
    return freeze();
  }

  bool drain(float* out) noexcept {
    if (!out || !finished_ || failed_) return false;
    std::fill_n(out, kCallbackSamples, 0.f);
    const auto count = std::min(output_size_, kCallbackSamples);
    if (count) {
      std::copy_n(output_.begin(), count, out);
      std::copy(output_.begin() + count, output_.begin() + output_size_,
                output_.begin());
      output_size_ -= count;
      emitted_samples_ += count;
    }
    return true;
  }

  void reset() noexcept {
    input_mic_.fill(0.f);
    input_ref_.fill(0.f);
    block_.fill(0.f);
    output_.fill(0.f);
    input_size_ = 0;
    output_size_ = kDelaySamples;
    input_samples_ = 0;
    emitted_samples_ = 0;
    inferences_ = 0;
    finished_ = false;
    failed_ = false;
  }

  unsigned delay_samples() const noexcept { return kDelaySamples; }
  std::uint64_t inferences() const noexcept { return inferences_; }
  std::uint64_t input_samples() const noexcept { return input_samples_; }
  unsigned pending() const noexcept { return input_size_; }
  unsigned queued() const noexcept { return output_size_; }
  bool finished() const noexcept { return finished_; }
  bool failed() const noexcept { return failed_; }
  bool drained() const noexcept {
    return finished_ && !failed_ && !output_size_;
  }

 private:
  template <class Process>
  bool advance_to_hops(std::uint64_t target_hops,
                       Process&& model) noexcept {
    if (target_hops < inferences_) return false;
    while (inferences_ < target_hops) {
      if (!run_padded_hop(model)) return false;
    }
    return true;
  }

  bool freeze() noexcept {
    if (failed_) return false;
    input_size_ = 0;
    finished_ = true;
    return true;
  }

  template <class Process>
  bool run_hop(Process&& model) noexcept {
    if (!model(input_mic_.data(), input_ref_.data(), block_.data()))
      return false;
    ++inferences_;

    const auto remaining = input_size_ - kModelHopSamples;
    if (remaining) {
      std::copy(input_mic_.begin() + kModelHopSamples,
                input_mic_.begin() + input_size_, input_mic_.begin());
      std::copy(input_ref_.begin() + kModelHopSamples,
                input_ref_.begin() + input_size_, input_ref_.begin());
    }
    input_size_ = remaining;

    if (output_size_ + kModelHopSamples > output_.size()) return false;
    std::copy(block_.begin(), block_.end(), output_.begin() + output_size_);
    output_size_ += kModelHopSamples;
    return true;
  }

  template <class Process>
  bool run_padded_hop(Process&& model) noexcept {
    if (input_size_ > kModelHopSamples) return false;
    std::fill(input_mic_.begin() + input_size_,
              input_mic_.begin() + kModelHopSamples, 0.f);
    std::fill(input_ref_.begin() + input_size_,
              input_ref_.begin() + kModelHopSamples, 0.f);
    input_size_ = kModelHopSamples;
    return run_hop(model);
  }

  void emit_callback(float* out) noexcept {
    std::copy_n(output_.begin(), kCallbackSamples, out);
    std::copy(output_.begin() + kCallbackSamples,
              output_.begin() + output_size_, output_.begin());
    output_size_ -= kCallbackSamples;
    emitted_samples_ += kCallbackSamples;
  }

  std::array<float, kInputFifoCapacity> input_mic_{};
  std::array<float, kInputFifoCapacity> input_ref_{};
  std::array<float, kModelHopSamples> block_{};
  std::array<float, kOutputFifoCapacity> output_{};
  unsigned input_size_ = 0;
  unsigned output_size_ = 0;
  std::uint64_t input_samples_ = 0;
  std::uint64_t emitted_samples_ = 0;
  std::uint64_t inferences_ = 0;
  bool finished_ = false;
  bool failed_ = false;
};

}  // namespace itu
