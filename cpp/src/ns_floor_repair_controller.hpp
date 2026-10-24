#pragma once
#include "fft.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace itu {
// Frame-causal, backend-independent NS protection experiment. Only observed
// capture PCM enters this observer; no scene labels, clean target or schedule.
// A continuous spectral-excess score is not an external VAD or activity gate.
class NsFloorRepairController {
 public:
  NsFloorRepairController(unsigned rate, float dry_max, float snr_threshold,
                     float noise_gain, unsigned startup_ms, float release_ms=0, float quiet_floor_db=-300, unsigned floor_guard=0)
      : rate_(rate), hop_(rate / 100), n_(detail::next_power_two(2 * hop_)),
        fft_(n_), history_(n_), spectrum_(n_), window_(n_),
        dry_max_(dry_max), snr_threshold_(snr_threshold),
        noise_gain_(noise_gain), startup_frames_((startup_ms + 9) / 10), release_ms_(release_ms), quiet_floor_db_(quiet_floor_db), floor_guard_(floor_guard) {
    for (unsigned i = 0; i < n_; ++i)
      window_[i] = .5f - .5f * std::cos(6.283185307179586f * i / (n_ - 1));
    minima_.fill(std::numeric_limits<float>::max());
  }
  void observe(const float* input) noexcept {
    std::move(history_.begin() + hop_, history_.end(), history_.begin());
    std::copy_n(input, hop_, history_.end() - hop_);
    for (unsigned i = 0; i < n_; ++i) spectrum_[i] = history_[i] * window_[i];
    fft_.run(spectrum_.data());
    std::array<float, 24> bands{};
    const unsigned first = std::max(1u, unsigned(100. * n_ / rate_));
    const unsigned last = std::min(n_ / 2, unsigned(7800. * n_ / rate_));
    double sum = 0, logs = 0;
    for (unsigned k = first; k <= last; ++k) {
      const float p = std::max(std::norm(spectrum_[k]), 1e-20f);
      bands[std::min(23u, (k - first) * 24 / (last - first + 1))] += p;
      sum += p; logs += std::log(p);
    }
    flatness_ = sum > 1e-14 ? std::exp(logs / (last-first+1)) / (sum / (last-first+1)) : 1.f;
    double previous_floor = 0;
    for (float x : floor_) previous_floor += x;
    ++frames_;
    // Initial observations and downward minima are causal. The independent
    // startup blend is bounded in time, including streams beginning in speech.
    if (frames_ <= 30) {
      if (frames_ == 1 || sum < 2.5 * std::max(previous_floor, 1e-18) || flatness_ > .35f) {
        ++floor_observations_;
        for (unsigned b = 0; b < 24; ++b)
          floor_[b] += (bands[b] - floor_[b]) / floor_observations_;
      } else {
        for (unsigned b = 0; b < 24; ++b) floor_[b] = std::min(floor_[b], bands[b]);
      }
    }
    for (unsigned b = 0; b < 24; ++b) minima_[b] = std::min(minima_[b], bands[b]);
    if (frames_ % 20 == 0) {
      blocks_[head_] = minima_; head_ = (head_ + 1) % 10; blocks_seen_ = std::min(10u, blocks_seen_ + 1);
      for (unsigned b = 0; b < 24; ++b) {
        std::array<float, 10> sorted;
        sorted.fill(std::numeric_limits<float>::max());
        for (unsigned j = 0; j < blocks_seen_; ++j) sorted[j] = blocks_[j][b];
        std::sort(sorted.begin(), sorted.end());
        const float estimate = std::max(1e-18f, 1.8f * sorted[blocks_seen_/3]);
        float alpha=.25f;
        if (floor_guard_) {
          // Upward minimum-statistics drift during sustained speech must not
          // teach weak syllables into the noise model. Downward recovery is
          // fast; a nonzero upward coefficient retains continuous tracking.
          const float speech=confidence_*std::clamp((snr_db_-6.f)/12.f,0.f,1.f);
          alpha=estimate < floor_[b] ? .5f : std::max(.01f,.25f*(1-speech)*(1-speech));
          if (floor_guard_==2 && estimate>floor_[b])
            alpha=std::max(.0025f,alpha*.5f);
        }
        floor_[b] += alpha * (estimate - floor_[b]);
      }
      minima_.fill(std::numeric_limits<float>::max());
    }
    double excess = 0, noise = 0; unsigned excited = 0;
    for (unsigned b = 0; b < 24; ++b) {
      excess += std::max(0.f, bands[b] - 1.5f * floor_[b]);
      noise += floor_[b];
      excited += bands[b] > 3.f * std::max(floor_[b], 1e-18f);
    }
    const float fraction = sum > 1e-14 ? float(excess / sum) : 0.f;
    const float broad = std::clamp((float(excited) - 1) / 5.f, 0.f, 1.f);
    const float measured = std::clamp((fraction - .25f) / .6f, 0.f, 1.f) * broad;
    confidence_ += std::clamp(measured - confidence_, -.12f, .5f);
    snr_db_ = 10.f * std::log10(std::max((sum - noise) / std::max(noise, 1e-18), 1e-6));
    const float high_snr = std::clamp((snr_db_ - (snr_threshold_ - 6.f)) / 12.f, 0.f, 1.f);
    float evidence = confidence_ * high_snr;
    if (release_ms_ > 0) {
      // Keep weak speech between nearby peaks without delaying the audio.
      protection_ = std::max(evidence, protection_ * std::exp(-10.f/release_ms_));
      evidence = protection_;
    }
    noise_floor_db_ = 10.f*std::log10(std::max(noise/(.1875*double(n_)*n_),1e-30));
    const float quiet_reliability = std::clamp((quiet_floor_db_ - noise_floor_db_)/12.f,0.f,1.f);
    const float quiet_speech = quiet_reliability * std::clamp((snr_db_-6.f)/12.f,0.f,1.f);
    // A separate fullband path is used only with a low observed noise floor.
    // It is never a source label, silence flag, or clean-target comparison.
    quiet_ = std::max(quiet_speech, quiet_*std::exp(-10.f/80.f));
    float dry_target = dry_max_ * evidence;
    if (frames_ <= startup_frames_) dry_target = std::max(dry_target, dry_max_);
    dry_ += std::clamp(dry_target - dry_, -.10f, .5f);
    gain_target_ = noise_gain_ + (1.f - noise_gain_) * (release_ms_ > 0 ? std::max(confidence_, protection_) : confidence_);
  }
  float quiet() const noexcept { return quiet_; }
  float noise_floor_db() const noexcept { return noise_floor_db_; }
  float dry() const noexcept { return dry_; }
  float confidence() const noexcept { return confidence_; }
  float snr_db() const noexcept { return snr_db_; }
  float flatness() const noexcept { return flatness_; }
  float gain() const noexcept { return gain_target_; }
 private:
  unsigned rate_, hop_, n_;
  detail::Fft fft_;
  std::vector<float> history_;
  std::vector<std::complex<float>> spectrum_;
  std::vector<float> window_;
  std::array<float, 24> floor_{}, minima_{};
  std::array<std::array<float, 24>, 10> blocks_{};
  unsigned frames_=0, floor_observations_=0, blocks_seen_=0, head_=0;
  float dry_max_, snr_threshold_, noise_gain_;
  unsigned startup_frames_;
  unsigned floor_guard_;
  float release_ms_, quiet_floor_db_, protection_=0, quiet_=0, noise_floor_db_=-300;
  float confidence_=0, dry_=0, snr_db_=0, flatness_=1, gain_target_=1;
};
}
