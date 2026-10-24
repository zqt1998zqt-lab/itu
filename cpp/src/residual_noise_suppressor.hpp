#pragma once

#include "fft.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace itu {

// Causal post-NS residual floor control. It observes only the backend output
// and current-frame evidence, never reference labels, clean targets or future
// PCM. Speech-confident frames take the unmodified backend output; only frames
// with stable noise evidence receive extra attenuation.
class ResidualNoiseSuppressor {
 public:
  ResidualNoiseSuppressor(unsigned rate, float max_extra_db, float speech_guard,
                          float gate_snr_db = -60.f,
                          float gate_width_db = 3.f)
      : rate_(rate), hop_(rate / 100),
        fft_(detail::next_power_two(2 * hop_)),
        window_(fft_size()), history_(fft_size()),
        spectrum_(fft_size()),
        max_extra_db_(max_extra_db), speech_guard_(speech_guard),
        gate_snr_db_(gate_snr_db), gate_width_db_(gate_width_db) {
    if (rate != 16000 && rate != 24000 && rate != 48000)
      throw std::invalid_argument("residual NS supports native 16/24/48 kHz");
    if (!std::isfinite(max_extra_db + speech_guard + gate_snr_db +
                       gate_width_db) || max_extra_db < 0 ||
        max_extra_db > 24 || speech_guard < 0 || speech_guard > 1)
      throw std::invalid_argument("invalid residual NS envelope");
    if (gate_snr_db != -60 && gate_width_db <= 0)
      throw std::invalid_argument("invalid residual gate width");
    band_floor_.fill(std::numeric_limits<float>::max());
    short_min_.fill(std::numeric_limits<float>::max());
    for (unsigned i = 0; i < fft_size(); ++i)
      window_[i] = .5f - .5f * std::cos(6.283185307179586f * i / (fft_size() - 1));
  }

  void process(float* frame, unsigned count,
               float controller_snr_db = 0) noexcept {
    if (!frame || count != hop_) return;
    if (!std::isfinite(controller_snr_db)) return;
    std::move(history_.begin() + hop_, history_.end(), history_.begin());
    std::copy_n(frame, hop_, history_.end() - hop_);
    for (unsigned i = 0; i < fft_size(); ++i)
      spectrum_[i] = history_[i] * window_[i];
    fft_.run(spectrum_.data());

    const unsigned first = std::max(1u, unsigned(70. * fft_size() / rate_));
    const unsigned last = std::min(fft_size() / 2,
                                   unsigned(std::min(7800u, rate_ / 2 - 1)));
    std::array<float, kBandCount> bands{};
    double total = 0;
    double log_power = 0;
    unsigned bins = 0;
    for (unsigned k = first; k <= last; ++k) {
      const float p = std::max(std::norm(spectrum_[k]), 1e-20f);
      bands[Band(k)] += p;
      total += p;
      log_power += std::log(p);
      ++bins;
    }
    const float flatness = total > 1e-15f && bins
        ? float(std::exp(log_power / bins) / (total / bins)) : 1.f;
    ++frames_;

    // A causal two-scale minimum tracker: every 100 ms records a 10-frame
    // minimum; the lower third over 1.5 s gives a robust residual floor.
    for (unsigned b = 0; b < kBandCount; ++b)
      short_min_[b] = std::min(short_min_[b], bands[b]);
    if (frames_ % 10 == 0) {
      blocks_[head_] = short_min_;
      head_ = (head_ + 1) % blocks_.size();
      blocks_seen_ = std::min<unsigned>(blocks_.size(), blocks_seen_ + 1);
      for (unsigned b = 0; b < kBandCount; ++b) {
        std::array<float, 15> sorted{};
        sorted.fill(std::numeric_limits<float>::max());
        for (unsigned j = 0; j < blocks_seen_; ++j) sorted[j] = blocks_[j][b];
        std::sort(sorted.begin(), sorted.end());
        const float sample = std::max(1e-20f, sorted[blocks_seen_ / 3]);
        if (blocks_seen_ == 1) band_floor_[b] = 1.6f * sample;
        else {
          const float alpha = sample < band_floor_[b] ? .5f : .03f;
          band_floor_[b] += alpha * (1.6f * sample - band_floor_[b]);
        }
      }
      short_min_.fill(std::numeric_limits<float>::max());
    }

    double noise = 0;
    double excess = 0;
    unsigned speech_bands = 0;
    for (unsigned b = 0; b < kBandCount; ++b) {
      const float floor = std::max(1e-20f, band_floor_[b]);
      noise += floor;
      excess += std::max(0.f, bands[b] - 2.f * floor);
      if (bands[b] > 3.f * floor) ++speech_bands;
    }
    const float snr = std::clamp(
        float(10. * std::log10(
            std::max((total - noise) / std::max(noise, 1e-20), 1e-6))),
        -30.f, 60.f);
    std::array<float, kBandCount> strongest = bands;
    std::partial_sort(strongest.begin(), strongest.begin() + 4,
                      strongest.end(), std::greater<float>());
    float peak_energy = 0;
    for (unsigned b = 0; b < 4; ++b) peak_energy += strongest[b];
    const float peak_share = peak_energy / std::max(total, 1e-20);
    const float spectral_speech = std::clamp(
        float((excess / std::max(total, 1e-20) - .18) / .55), 0.f, 1.f);
    const float broad_speech = std::clamp((float(speech_bands) - 1.f) / 4.f, 0.f, 1.f);
    const float harmonic_speech = std::clamp((peak_share - .3f) / .55f, 0.f, 1.f);
    const float voicing_speech = std::max(broad_speech, harmonic_speech) *
        std::clamp((snr - 4.f) / 8.f, 0.f, 1.f);
    const float speech = std::clamp(speech_guard_ * spectral_speech * voicing_speech +
        (1.f - speech_guard_) * std::clamp((snr - 4.f) / 12.f, 0.f, 1.f), 0.f, 1.f);
    // Only a noise-steady, flat-spectrum frame is eligible for extra
    // suppression. This avoids treating wind gusts, impulses and speech
    // onsets as steady residuals.
    const float stability = std::clamp((.7f - flatness) / .35f, 0.f, 1.f);
    float gate = 1.f;
    if (gate_snr_db_ != -60) {
      gate = std::clamp(
          (gate_snr_db_ + gate_width_db_ - controller_snr_db) /
              gate_width_db_, 0.f, 1.f);
    }
    const float noise_conf = (1.f - speech) * stability * gate;
    const float extra_db = max_extra_db_ * noise_conf;
    const float extra_gain = std::pow(10.f, -extra_db / 20.f);
    // Confirmed speech releases the safety attenuation quickly. Noise attack
    // remains bounded and causal, so isolated impulses cannot produce a deep
    // gain hole.
    const float max_release = speech > .5f ? .5f : .1f;
    extra_gain_ += std::clamp(extra_gain - extra_gain_, -.2f, max_release);
    last_speech_ = speech;
    last_extra_db_ = -20.f * std::log10(std::max(extra_gain_, 1e-6f));

    for (unsigned i = 0; i < hop_; ++i)
      frame[i] *= extra_gain_;
  }

  float speech_probability() const noexcept { return last_speech_; }
  float extra_attenuation_db() const noexcept { return last_extra_db_; }
  float floor_db() const noexcept { return last_floor_db_; }

 private:
  static constexpr unsigned kBandCount = 24;
  unsigned fft_size() const noexcept {
    return detail::next_power_two(2 * hop_);
  }
  unsigned Band(unsigned k) const noexcept {
    const unsigned first = std::max(1u, unsigned(70. * fft_size() / rate_));
    const unsigned last = std::min(fft_size() / 2,
                                   unsigned(std::min(7800u, rate_ / 2 - 1)));
    return std::min(kBandCount - 1, (k - first) * kBandCount / (last - first + 1));
  }

  unsigned rate_, hop_;
  detail::Fft fft_;
  std::vector<float> window_, history_;
  std::vector<std::complex<float>> spectrum_;
  std::array<float, kBandCount> band_floor_{};
  std::array<float, kBandCount> short_min_{};
  std::array<std::array<float, kBandCount>, 15> blocks_{};
  unsigned frames_ = 0, head_ = 0, blocks_seen_ = 0;
  float max_extra_db_ = 0, speech_guard_ = 0;
  float gate_snr_db_ = -60, gate_width_db_ = 3;
  float extra_gain_ = 1, last_speech_ = 1, last_extra_db_ = 0;
  float last_floor_db_ = -300;
};

}  // namespace itu
