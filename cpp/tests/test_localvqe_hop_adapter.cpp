#include "continuous_hop_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr unsigned kCallback = 160;
constexpr unsigned kHop = 256;
constexpr unsigned kDelay = 224;
constexpr unsigned kModelCodecDelay = 256;
constexpr unsigned kMaxFrames = 1201;
constexpr unsigned kMaxSamples = kMaxFrames * kCallback;
constexpr unsigned kMaxHops =
    (kMaxSamples + kModelCodecDelay + kHop - 1) / kHop;

std::atomic<bool> track_allocations{false};
std::atomic<unsigned> callback_allocations{0};

struct Captured {
  std::array<std::array<float, kHop>, kMaxHops> mic{};
  std::array<std::array<float, kHop>, kMaxHops> ref{};
  unsigned calls = 0;
};

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <class Callback>
bool tracked(Callback&& callback) {
  callback_allocations = 0;
  track_allocations = true;
  const bool result = callback();
  track_allocations = false;
  return result;
}

enum class Signal { ramp, impulse, random };

void fill_signal(Signal kind, unsigned samples, std::array<float, kMaxSamples>& mic,
                 std::array<float, kMaxSamples>& ref) {
  std::uint32_t random = 0x12345678u;
  for (unsigned i = 0; i < samples; ++i) {
    float value = 0.f;
    switch (kind) {
      case Signal::ramp:
        value = static_cast<float>(i + 1) / 32768.f;
        break;
      case Signal::impulse:
        if (i == 0 || i == 159 || i == 160 || i == 255 || i == 256 ||
            i == 319 || i == 511 || i == 777 || i == samples - 1)
          value = (i & 1u) ? -.75f : .75f;
        break;
      case Signal::random:
        random = random * 1664525u + 1013904223u;
        value = (static_cast<float>((random >> 8u) & 0xffffu) / 32768.f) - 1.f;
        value *= .25f;
        break;
    }
    mic[i] = value;
    ref[i] = -0.5f * value;
  }
}

template <class Adapter>
bool identity_hop(Captured& captured, const float* mic, const float* ref,
                  float* out) {
  if (captured.calls >= kMaxHops) return false;
  std::copy_n(mic, kHop, captured.mic[captured.calls].begin());
  std::copy_n(ref, kHop, captured.ref[captured.calls].begin());
  std::copy_n(mic, kHop, out);
  ++captured.calls;
  return true;
}

void validate_captured(const Captured& captured,
                       const std::array<float, kMaxSamples>& mic,
                       const std::array<float, kMaxSamples>& ref,
                       unsigned real_samples) {
  const unsigned expected_calls = (real_samples + kHop - 1) / kHop;
  require(captured.calls == expected_calls, "native hop count mismatch");
  for (unsigned hop = 0; hop < captured.calls; ++hop) {
    for (unsigned i = 0; i < kHop; ++i) {
      const unsigned source = hop * kHop + i;
      const float expected_mic = source < real_samples ? mic[source] : 0.f;
      const float expected_ref = source < real_samples ? ref[source] : 0.f;
      require(captured.mic[hop][i] == expected_mic,
              "native microphone input is repeated, skipped, or reordered");
      require(captured.ref[hop][i] == expected_ref,
              "native reference input is repeated, skipped, or reordered");
    }
  }
}

bool matches_contiguous_prefix(
    const Captured& captured,
    const std::array<float, kMaxSamples>& mic,
    const std::array<float, kMaxSamples>& ref,
    unsigned real_samples) {
  const unsigned expected_calls = (real_samples + kHop - 1) / kHop;
  if (captured.calls != expected_calls) return false;
  for (unsigned source = 0; source < expected_calls * kHop; ++source) {
    const unsigned hop = source / kHop;
    const unsigned index = source % kHop;
    const float expected_mic = source < real_samples ? mic[source] : 0.f;
    const float expected_ref = source < real_samples ? ref[source] : 0.f;
    if (captured.mic[hop][index] != expected_mic ||
        captured.ref[hop][index] != expected_ref)
      return false;
  }
  return true;
}

void prefix_replay_negative_control() {
  constexpr unsigned frames = 5;
  constexpr unsigned samples = frames * kCallback;
  std::array<float, kMaxSamples> mic{}, ref{};
  fill_signal(Signal::ramp, samples, mic, ref);
  Captured replayed;
  replayed.calls = frames;
  std::array<float, 96> context_mic{}, context_ref{};
  for (unsigned frame = 0; frame < frames; ++frame) {
    std::copy(context_mic.begin(), context_mic.end(),
              replayed.mic[frame].begin());
    std::copy(context_ref.begin(), context_ref.end(),
              replayed.ref[frame].begin());
    std::copy_n(mic.begin() + frame * kCallback, kCallback,
                replayed.mic[frame].begin() + 96);
    std::copy_n(ref.begin() + frame * kCallback, kCallback,
                replayed.ref[frame].begin() + 96);
    std::copy(replayed.mic[frame].begin() + kCallback,
              replayed.mic[frame].end(), context_mic.begin());
    std::copy(replayed.ref[frame].begin() + kCallback,
              replayed.ref[frame].end(), context_ref.begin());
  }
  require(!matches_contiguous_prefix(replayed, mic, ref, samples),
          "prefix-replay negative control was not detected");
  std::cout << "prefix_replay_negative_control=detected\n";
}

void run_complete_case(Signal kind, unsigned frames) {
  const unsigned samples = frames * kCallback;
  std::array<float, kMaxSamples> mic{}, ref{};
  fill_signal(kind, samples, mic, ref);

  itu::ContinuousHopAdapter adapter;
  require(adapter.delay_samples() == kDelay, "adapter delay is not 224");
  require(adapter.pending() == 0 && adapter.queued() == kDelay,
          "adapter startup queue is not exactly the 224-sample delay");

  Captured captured;
  std::array<float, kMaxSamples + kDelay + kCallback> output{};
  unsigned written = 0;
  for (unsigned frame = 0; frame < frames; ++frame) {
    const bool ok = tracked([&] {
      return adapter.process(
          mic.data() + frame * kCallback, ref.data() + frame * kCallback,
          output.data() + written,
          [&](const float* m, const float* r, float* o) {
            return identity_hop<decltype(adapter)>(captured, m, r, o);
          });
    });
    require(ok, "adapter rejected a complete callback");
    written += kCallback;
    const unsigned expected_pending =
        ((frame + 1) * kCallback) % kHop;
    require(adapter.pending() == expected_pending,
            "pending input count does not match 160/256 phase");
  }
  require(adapter.inferences() == captured.calls,
          "adapter inference count differs from native calls");
  require(captured.calls == samples / kHop,
          "LCM input did not produce exactly floor(N/256) native calls");
  validate_captured(captured, mic, ref, samples);

  const bool finished = tracked([&] {
    return adapter.finish([&](const float* m, const float* r, float* o) {
      return identity_hop<decltype(adapter)>(captured, m, r, o);
    });
  });
  require(finished, "adapter finish failed on an LCM boundary");
  require(captured.calls == samples / kHop,
          "LCM finish fabricated a native hop");
  while (!adapter.drained()) {
    const bool drained = tracked([&] {
      return adapter.drain(output.data() + written);
    });
    require(drained, "adapter drain underflowed");
    written += kCallback;
  }
  const unsigned valid_extent =
      kDelay + captured.calls * kHop;
  const unsigned callback_extent =
      ((valid_extent + kCallback - 1) / kCallback) * kCallback;
  require(written == callback_extent,
          "complete stream callback extent is not rounded input + 224");
  require(adapter.pending() == 0 && adapter.queued() == 0,
          "complete stream left FIFO state behind");
  require(callback_allocations == 0, "adapter callback allocated memory");

  for (unsigned i = 0; i < kDelay; ++i)
    require(output[i] == 0.f, "startup delay is not zero-filled");
  for (unsigned i = 0; i < samples; ++i)
    require(output[kDelay + i] == mic[i],
            "160/256 output differs from the continuous native oracle");
  for (unsigned i = valid_extent; i < written; ++i)
    require(output[i] == 0.f, "complete stream drain tail is not zero");

  std::cout << "complete signal=" << static_cast<unsigned>(kind)
            << " samples=" << samples << " calls=" << captured.calls
            << " output=" << written << " allocations="
            << callback_allocations.load() << '\n';
}

void run_stop_and_drain(Signal kind, unsigned frames) {
  const unsigned samples = frames * kCallback;
  std::array<float, kMaxSamples> mic{}, ref{};
  fill_signal(kind, samples, mic, ref);

  itu::ContinuousHopAdapter adapter;
  Captured captured;
  std::array<float, kMaxSamples + kDelay + kCallback> output{};
  unsigned written = 0;
  for (unsigned frame = 0; frame < frames; ++frame) {
    require(adapter.process(
                mic.data() + frame * kCallback, ref.data() + frame * kCallback,
                output.data() + written,
                [&](const float* m, const float* r, float* o) {
                  return identity_hop<decltype(adapter)>(captured, m, r, o);
                }),
            "adapter rejected callback before arbitrary stop");
    written += kCallback;
  }
  require(adapter.finish([&](const float* m, const float* r, float* o) {
            return identity_hop<decltype(adapter)>(captured, m, r, o);
          }),
          "adapter zero-padding finish failed");
  require(adapter.inferences() == captured.calls &&
              captured.calls == (samples + kHop - 1) / kHop,
          "EOS inference count is not ceil(N/256)");
  validate_captured(captured, mic, ref, samples);

  while (!adapter.drained()) {
    require(adapter.drain(output.data() + written),
            "adapter zero-drain underflowed");
    written += kCallback;
  }
  const unsigned valid_extent =
      kDelay + captured.calls * kHop;
  const unsigned callback_extent =
      ((valid_extent + kCallback - 1) / kCallback) * kCallback;
  require(written == callback_extent,
          "stop/drain callback extent is not rounded input + 224");
  for (unsigned i = 0; i < kDelay; ++i)
    require(output[i] == 0.f, "startup delay changed after EOS drain");
  for (unsigned i = 0; i < captured.calls * kHop; ++i)
    require(output[kDelay + i] == (i < samples ? mic[i] : 0.f),
            "stop/drain output differs from padded continuous oracle");
  for (unsigned i = valid_extent; i < written; ++i)
    require(output[i] == 0.f, "stop/drain final callback is not zero-filled");

  std::array<float, kCallback> extra{};
  require(adapter.drain(extra.data()), "post-drain zero callback was rejected");
  require(std::all_of(extra.begin(), extra.end(),
                      [](float value) { return value == 0.f; }),
          "post-drain callback is not zero");
  require(adapter.pending() == 0 && adapter.queued() == 0,
          "drained adapter retained FIFO state");

  std::cout << "stop signal=" << static_cast<unsigned>(kind)
            << " samples=" << samples << " calls=" << captured.calls
            << " output=" << written << '\n';
}

void run_known_source_eos(unsigned source_samples) {
  constexpr unsigned downstream_flush_callbacks = 16;
  const unsigned source_callbacks = (source_samples + kCallback - 1) / kCallback;
  const unsigned callback_samples = source_callbacks * kCallback;
  const unsigned target_hops =
      (source_samples + kModelCodecDelay + kHop - 1) / kHop;
  std::array<float, kMaxSamples> mic{}, ref{};
  fill_signal(Signal::random, source_samples, mic, ref);
  mic[source_samples - 1] = .75f;
  ref[source_samples - 1] = -.25f;

  itu::ContinuousHopAdapter adapter;
  Captured captured;
  std::array<float,
             kMaxSamples + kDelay + 4 * kCallback +
                 downstream_flush_callbacks * kCallback>
      output{};
  unsigned written = 0;
  for (unsigned callback = 0; callback < source_callbacks; ++callback) {
    require(adapter.process(
                mic.data() + callback * kCallback,
                ref.data() + callback * kCallback,
                output.data() + written,
                [&](const float* m, const float* r, float* o) {
                  return identity_hop<decltype(adapter)>(captured, m, r, o);
                }),
            "known-source input callback failed");
    written += kCallback;
  }
  require(adapter.input_samples() == callback_samples,
          "known-source callback padding accounting changed");

  std::array<float, kCallback> zero{};
  unsigned native_tail_callbacks = 0;
  unsigned max_hops_per_callback = 0;
  while (adapter.inferences() < target_hops) {
    const auto before = adapter.inferences();
    require(adapter.process(
                zero.data(), zero.data(), output.data() + written,
                [&](const float* m, const float* r, float* o) {
                  return identity_hop<decltype(adapter)>(captured, m, r, o);
                }),
            "native-tail zero callback failed");
    const auto advanced = adapter.inferences() - before;
    require(advanced <= 1, "zero callback advanced more than one native hop");
    max_hops_per_callback =
        std::max(max_hops_per_callback, static_cast<unsigned>(advanced));
    written += kCallback;
    ++native_tail_callbacks;
    require(native_tail_callbacks <= 4,
            "native core tail exceeded its bounded callback count");
  }
  const auto before_freeze = adapter.inferences();
  require(adapter.finish_after_hops(target_hops),
          "target-limited EOS freeze failed");
  require(adapter.inferences() == before_freeze,
          "finish_after_hops added native work");
  require(adapter.inferences() == target_hops &&
              captured.calls == target_hops,
          "target-limited EOS did not execute exactly target hops");

  while (!adapter.drained()) {
    require(adapter.drain(output.data() + written),
            "known-source valid-queue drain failed");
    written += kCallback;
  }
  const unsigned valid_extent =
      kDelay + target_hops * kHop;
  const unsigned callback_extent =
      ((valid_extent + kCallback - 1) / kCallback) * kCallback;
  require(written == callback_extent,
          "known-source drain extent is not callback-rounded");
  require(output[kDelay + source_samples - 1] == mic[source_samples - 1],
          "last speech sample was lost during target-limited EOS");
  require(output[kDelay + source_samples - 1] == .75f,
          "last-speech sentinel changed");

  const auto inferences_before_flush = adapter.inferences();
  unsigned downstream_callbacks = 0;
  std::array<float, kCallback> flush{};
  for (unsigned callback = 0; callback < downstream_flush_callbacks; ++callback) {
    require(adapter.drain(flush.data()),
            "downstream flush callback rejected");
    require(std::all_of(flush.begin(), flush.end(),
                        [](float value) { return value == 0.f; }),
            "post-AEC downstream flush was not zero");
    ++downstream_callbacks;
    require(adapter.inferences() == inferences_before_flush,
            "downstream flush advanced the LocalVQE model");
  }
  require(downstream_callbacks == downstream_flush_callbacks,
          "downstream flush callback count changed");

  require(captured.calls == target_hops, "EOS tail added native hops");
  for (unsigned source = 0; source < target_hops * kHop; ++source) {
    const unsigned hop = source / kHop;
    const unsigned index = source % kHop;
    const float expected_mic = source < source_samples ? mic[source] : 0.f;
    const float expected_ref = source < source_samples ? ref[source] : 0.f;
    require(captured.mic[hop][index] == expected_mic,
            "known-source native microphone prefix is not contiguous");
    require(captured.ref[hop][index] == expected_ref,
            "known-source native reference prefix is not contiguous");
  }

  std::cout << "known_source=" << source_samples
            << " callbacks=" << source_callbacks
            << " target_hops=" << target_hops
            << " native_tail_callbacks=" << native_tail_callbacks
            << " max_hops_per_callback=" << max_hops_per_callback
            << " downstream_flush=" << downstream_callbacks
            << " inferences_final=" << adapter.inferences() << '\n';
}

void run_1200_callbacks() {
  itu::ContinuousHopAdapter adapter;
  std::array<float, kCallback> mic{}, ref{}, out{};
  unsigned calls = 0;
  for (unsigned frame = 0; frame < 1200; ++frame) {
    for (unsigned i = 0; i < kCallback; ++i) {
      const float value =
          static_cast<float>(frame * kCallback + i + 1) / 32768.f;
      mic[i] = value;
      ref[i] = -.5f * value;
    }
    const bool ok = tracked([&] {
      return adapter.process(mic.data(), ref.data(), out.data(),
                             [&](const float* m, const float* r, float* o) {
                               for (unsigned i = 0; i < kHop; ++i) {
                                 require(m[i] ==
                                             static_cast<float>(
                                                 calls * kHop + i + 1) /
                                                 32768.f,
                                         "1200-callback native input discontinuity");
                                 require(r[i] == -.5f * m[i],
                                         "1200-callback reference mismatch");
                                 o[i] = m[i];
                               }
                               ++calls;
                               return true;
                             });
    });
    require(ok, "1200-callback adapter processing failed");
  }
  require(calls == 750 && adapter.inferences() == 750,
          "1200 callbacks did not produce 750 native hops");
  require(adapter.pending() == 0 && adapter.queued() == kDelay,
          "1200-callback FIFO accounting is wrong");
  require(callback_allocations == 0,
          "1200-callback path allocated inside the adapter");
  std::cout << "callbacks=1200 native_calls=" << calls
            << " pending=" << adapter.pending()
            << " queued=" << adapter.queued() << '\n';
}

struct RunResult {
  Captured captured;
  std::array<float, 9 * kCallback> output{};
  unsigned written = 0;
};

void run_reset_case(itu::ContinuousHopAdapter& adapter, RunResult& result) {
  constexpr unsigned frames = 5;
  constexpr unsigned samples = frames * kCallback;
  std::array<float, kMaxSamples> mic{}, ref{};
  fill_signal(Signal::random, samples, mic, ref);
  for (unsigned frame = 0; frame < frames; ++frame) {
    require(adapter.process(
                mic.data() + frame * kCallback, ref.data() + frame * kCallback,
                result.output.data() + result.written,
                [&](const float* m, const float* r, float* o) {
                  return identity_hop<decltype(adapter)>(result.captured, m, r,
                                                         o);
                }),
            "reset isolation processing failed");
    result.written += kCallback;
  }
  require(adapter.finish([&](const float* m, const float* r, float* o) {
            return identity_hop<decltype(adapter)>(result.captured, m, r, o);
          }),
          "reset isolation finish failed");
  while (!adapter.drained()) {
    require(adapter.drain(result.output.data() + result.written),
            "reset isolation drain failed");
    result.written += kCallback;
  }
}

void reset_and_rebuild_isolation() {
  itu::ContinuousHopAdapter adapter;
  RunResult first, reset, rebuilt;
  run_reset_case(adapter, first);
  require(adapter.inferences() == first.captured.calls &&
              adapter.pending() == 0 && adapter.queued() == 0,
          "finished adapter counters are inconsistent");

  adapter.reset();
  require(adapter.inferences() == 0 && adapter.pending() == 0 &&
              adapter.queued() == kDelay && !adapter.finished() &&
              !adapter.failed(),
          "adapter reset retained stream state or counters");
  run_reset_case(adapter, reset);

  itu::ContinuousHopAdapter fresh;
  run_reset_case(fresh, rebuilt);
  require(first.captured.calls == reset.captured.calls &&
              first.captured.calls == rebuilt.captured.calls,
          "reset/rebuild changed native call count");
  require(first.output == reset.output && first.output == rebuilt.output,
          "reset/rebuild leaked buffered PCM");
  require(first.captured.mic == reset.captured.mic &&
              first.captured.mic == rebuilt.captured.mic,
          "reset/rebuild leaked native input state");
  std::cout << "reset_isolation calls=" << first.captured.calls
            << " extent=" << first.written << '\n';
}

void failure_count_contract() {
  itu::ContinuousHopAdapter adapter;
  std::array<float, kCallback> input{}, ref{}, out{};
  require(adapter.process(input.data(), ref.data(), out.data(),
                          [](const float*, const float*, float* output) {
                            std::fill_n(output, kHop, 0.f);
                            return true;
                          }),
          "startup callback did not emit its zero buffer");
  require(adapter.process(input.data(), ref.data(), out.data(),
                          [](const float*, const float*, float*) {
                            return false;
                          }) == false,
          "native failure was not propagated");
  require(adapter.inferences() == 0,
          "failed native hop incremented inference count");
  require(!adapter.process(input.data(), ref.data(), out.data(),
                           [](const float*, const float*, float* output) {
                             std::fill_n(output, kHop, 0.f);
                             return true;
                           }),
          "failed adapter was reusable without reset");
  adapter.reset();
  require(adapter.inferences() == 0 && adapter.queued() == kDelay,
          "failure reset did not restore startup state");
}

}  // namespace

void* operator new(std::size_t size) {
  if (track_allocations.load(std::memory_order_relaxed))
    callback_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  if (track_allocations.load(std::memory_order_relaxed))
    callback_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
  try {
    for (Signal kind : {Signal::ramp, Signal::impulse, Signal::random}) {
      run_complete_case(kind, 8);
      for (unsigned frames = 1; frames <= 8; ++frames)
        run_stop_and_drain(kind, frames);
    }
    run_1200_callbacks();
    for (unsigned source : {12000u, 12001u, 16000u, 192000u, 192001u})
      run_known_source_eos(source);
    reset_and_rebuild_isolation();
    failure_count_contract();
    prefix_replay_negative_control();
    std::cout << "localvqe continuous-hop adapter tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "localvqe continuous-hop adapter test failed: " << error.what()
              << '\n';
    return 1;
  }
}
