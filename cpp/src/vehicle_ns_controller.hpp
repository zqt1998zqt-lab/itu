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
class VehicleNsController {
 public:
  VehicleNsController(unsigned rate, float dry_max, float snr_threshold,
                     float noise_gain, unsigned startup_ms, float release_ms=0, float quiet_floor_db=-300, unsigned floor_guard=0, unsigned repair_mode=1,
                     float weak_speech_recovery=0,float noise_step_recovery=0,
                     float transient_dry_guard=0)
      : rate_(rate), hop_(rate / 100), n_(detail::next_power_two(2 * hop_)),
        fft_(n_), history_(n_), spectrum_(n_), window_(n_),
        dry_max_(dry_max), snr_threshold_(snr_threshold),
        noise_gain_(noise_gain), startup_frames_((startup_ms + 9) / 10),
        floor_guard_(floor_guard), repair_mode_(repair_mode),
        release_ms_(release_ms), quiet_floor_db_(quiet_floor_db),
        weak_speech_recovery_(weak_speech_recovery),
        noise_step_recovery_(noise_step_recovery),
        transient_dry_guard_(transient_dry_guard) {
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
    if(noise_step_recovery_>0) {
      const float current_snr=10.f*std::log10(std::max(
          (sum-previous_floor)/std::max(previous_floor,1e-18),1e-6));
      double ratio_sum=0,ratio_logs=0,flux=0,scale=0;unsigned raised=0;
      for(unsigned b=0;b<24;++b) {
        const float old=step_smooth_[b];
        step_smooth_[b]=.85f*old+.15f*bands[b];
        const float ratio=std::clamp(step_smooth_[b]/std::max(step_anchor_[b],1e-18f),1e-3f,1000.f);
        ratio_sum+=ratio;ratio_logs+=std::log(ratio);raised+=ratio>2.f;
        flux+=std::abs(step_smooth_[b]-old);scale+=step_smooth_[b]+old;
      }
      const float shape=std::exp(ratio_logs/24)/std::max(ratio_sum/24,1e-18);
      const float broad=std::clamp((raised/24.f-.5f)/.3f,0.f,1.f);
      const float stable=std::clamp((.15f-float(flux/std::max(scale,1e-18)))/.1f,0.f,1.f);
      // A recent noise-dominant spectrum supplies the reference shape. Comparing
      // to the old long-term floor mistakes sustained speech for an unbounded
      // noise step, especially when car noise is strongly coloured.
      const float non_tonal=std::clamp((flatness_-1e-5f)/9e-5f,0.f,1.f);
      const float evidence=broad*std::clamp((shape-.8f)/.15f,0.f,1.f)*stable*non_tonal;
      if(step_cooldown_) --step_cooldown_;
      step_anchor_age_=std::min(100u,step_anchor_age_+1);
      if(!step_remaining_ && !step_cooldown_ && step_anchor_frames_>=20 && step_anchor_age_<100) {
        step_candidates_=evidence>.35f ? step_candidates_+1:0;
        if(step_candidates_>=12) {
          step_remaining_=25;step_cooldown_=150;step_anchor_frames_=0;step_candidates_=0;
        }
      } else step_candidates_=0;
      // Relearning has a finite 250 ms budget. A spectral-shape departure ends
      // it immediately; it cannot keep learning successive voiced syllables.
      step_probability_=0;
      if(step_remaining_) {
        --step_remaining_;
        if(shape<.8f) step_remaining_=0;
        else step_probability_=evidence*std::clamp((18.f-current_snr)/12.f,0.f,1.f);
      }
      if(!step_remaining_ && !step_cooldown_ && frames_>50 &&
         confidence_<.15f && current_snr<3.f && stable>.5f && non_tonal>.5f) {
        if(!step_anchor_frames_ || step_anchor_age_>=100) {
          step_anchor_=step_smooth_;step_anchor_frames_=0;
        } else for(unsigned b=0;b<24;++b)
          step_anchor_[b]+=.1f*(step_smooth_[b]-step_anchor_[b]);
        step_anchor_frames_=std::min(100u,step_anchor_frames_+1);step_anchor_age_=0;
      }
      const float amount=.04f*noise_step_recovery_*step_probability_;
      for(unsigned b=0;b<24;++b) if(bands[b]>floor_[b])
        floor_[b]+=amount*(std::min(bands[b],20*floor_[b])-floor_[b]);
    }
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
    // Low observed noise power alone is not near-end speech evidence. Apply
    // continuous evidence on the independent raw path as well as the dry path.
    double low_energy=0, low_mid_energy=0, high_energy=0, total_energy=0;
    for (unsigned k=0;k<=n_/2;++k) {
      const double e=std::norm(spectrum_[k]);
      total_energy+=e;
      if (k*rate_<80*n_) low_energy+=e;
      if (k*rate_>=80*n_ && k*rate_<300*n_) low_mid_energy+=e;
      if (k*rate_>=2000*n_) high_energy+=e;
    }
    rumble_fraction_=total_energy>1e-18 ? float(low_energy/total_energy) : 0.f;
    const float high_fraction =
        total_energy>1e-18 ? float(high_energy/total_energy) : 0.f;
    rumble_low_ratio_db_=10.f*float(std::log10(
        std::max(low_energy,1e-30)/std::max(low_mid_energy,1e-30)));
    transient_periodicity_100ms_ = 0;
    transient_periodicity_500ms_ = 0;
    transient_noise_evidence_ = 0;
    if (transient_dry_guard_ > 0) {
      transient_power_[transient_head_] = float(sum);
      transient_head_ = (transient_head_ + 1) % transient_power_.size();
      transient_seen_ = std::min(transient_seen_ + 1,
                                 unsigned(transient_power_.size()));
      if (transient_seen_ >= 51) {
        transient_periodicity_100ms_ = periodicity(10);
        transient_periodicity_500ms_ = periodicity(50);
        // Wind: repeated low-frequency gusts have a nearly tonal spectrum
        // but retain substantial 0-80 Hz energy and no speech confidence.
        const float wind = std::clamp(
            (transient_periodicity_100ms_ - .25f) / .25f, 0.f, 1.f) *
            std::clamp((.02f - flatness_) / .02f, 0.f, 1.f) *
            std::clamp((rumble_fraction_ - .08f) / .12f, 0.f, 1.f) *
            std::clamp((.05f - confidence_) / .05f, 0.f, 1.f);
        // Indicator: a repeated narrow pulse train with energy concentrated
        // above 2 kHz and no broad-band speech confidence. The 500 ms lag is
        // measured from capture history; no wall-clock schedule is consulted.
        const float indicator = std::clamp(
            (transient_periodicity_500ms_ - .55f) / .25f, 0.f, 1.f) *
            std::clamp((flatness_ - .30f) / .25f, 0.f, 1.f) *
            std::clamp((high_fraction - .65f) / .15f, 0.f, 1.f) *
            std::clamp((.05f - confidence_) / .05f, 0.f, 1.f);
        transient_noise_evidence_ = std::max(wind, indicator);
      }
      transient_guard_ = std::max(
          transient_dry_guard_ * transient_noise_evidence_,
          transient_guard_ * std::exp(-10.f/80.f));
    }
    const float spectral_support=repair_mode_>=3 ?
        std::clamp((.95f-rumble_fraction_)/.3f,0.f,1.f) : 1.f;
    const float quiet_evidence = repair_mode_ ? confidence_*confidence_*spectral_support : 1.f;
    const float quiet_speech = quiet_reliability * std::clamp((snr_db_-6.f)/12.f,0.f,1.f) * quiet_evidence;
    // A separate fullband path is used only with a low observed noise floor.
    // It is never a source label, silence flag, or clean-target comparison.
    // Optional weak-speech experiment. Preserve the backend's learned noise
    // state; only the aligned output blend changes, with no activity gate.
    const float raw_base=std::clamp((snr_db_-9.f)/12.f,0.f,1.f);
    const float raw_weak=std::clamp((snr_db_+3.f)/12.f,0.f,1.f);
    const float raw_weight=raw_base+(repair_mode_>=4 ? .09f*(raw_weak-raw_base) : 0.f);
    const float raw_speech=repair_mode_>=3 ? quiet_evidence*raw_weight : 0.f;
    quiet_ = std::max(std::max(quiet_speech,raw_speech), quiet_*std::exp(-10.f/80.f));
    if(weak_speech_recovery_>0) {
      // The high-SNR raw path cannot protect masked speech because its SNR
      // gate is closed. Add a separately bounded, continuous low-SNR path;
      // retain the existing output/control delay and independent NS states.
      const float weak=std::clamp((confidence_-.1f)/.4f,0.f,1.f);
      const float masked=std::clamp((6.f-snr_db_)/12.f,0.f,1.f);
      quiet_=std::max(quiet_,.75f*weak_speech_recovery_*weak*masked*spectral_support);
    }
    float dry_target = dry_max_ * evidence;
    if (frames_ <= startup_frames_) dry_target = std::max(dry_target, dry_max_);
    dry_ += std::clamp(dry_target - dry_, -.10f, .5f);
    gain_target_ = noise_gain_ + (1.f - noise_gain_) * (release_ms_ > 0 ? std::max(confidence_, protection_) : confidence_);
  }
  float quiet() const noexcept { return quiet_; }
  float noise_floor_db() const noexcept { return noise_floor_db_; }
  float rumble_fraction() const noexcept { return rumble_fraction_; }
  float rumble_low_ratio_db() const noexcept { return rumble_low_ratio_db_; }
  float dry() const noexcept { return dry_; }
  float confidence() const noexcept { return confidence_; }
  float snr_db() const noexcept { return snr_db_; }
  float flatness() const noexcept { return flatness_; }
  float gain() const noexcept { return gain_target_; }
  float noise_step_probability() const noexcept { return step_probability_; }
  float transient_noise_evidence() const noexcept { return transient_noise_evidence_; }
  float transient_guard() const noexcept { return transient_guard_; }
  float transient_periodicity_100ms() const noexcept { return transient_periodicity_100ms_; }
  float transient_periodicity_500ms() const noexcept { return transient_periodicity_500ms_; }
 private:
  float periodicity(unsigned lag) const noexcept {
    const unsigned count = std::min(transient_seen_, 80u);
    if (count <= lag + 8) return 0;
    double left_mean=0,right_mean=0;
    for (unsigned i=0;i<count;++i) {
      const unsigned newest = (transient_head_ + transient_power_.size() - 1 - i) %
                              transient_power_.size();
      if (i + lag < count) {
        const unsigned older = (transient_head_ + transient_power_.size() - 1 - i - lag) %
                               transient_power_.size();
        left_mean += transient_power_[newest];
        right_mean += transient_power_[older];
      }
    }
    const double pairs = double(count-lag);
    left_mean/=pairs;right_mean/=pairs;
    double covariance=0,left_var=0,right_var=0;
    for (unsigned i=0;i+lag<count;++i) {
      const unsigned newer = (transient_head_ + transient_power_.size() - 1 - i) %
                             transient_power_.size();
      const unsigned older = (transient_head_ + transient_power_.size() - 1 - i - lag) %
                             transient_power_.size();
      const double a=transient_power_[newer]-left_mean;
      const double b=transient_power_[older]-right_mean;
      covariance+=a*b;left_var+=a*a;right_var+=b*b;
    }
    return left_var>1e-24 && right_var>1e-24 ?
        float(std::clamp(covariance/std::sqrt(left_var*right_var),0.,1.)) : 0.f;
  }
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
  unsigned floor_guard_, repair_mode_;
  float release_ms_, quiet_floor_db_, protection_=0, quiet_=0, noise_floor_db_=-300;
  float confidence_=0, dry_=0, snr_db_=0, flatness_=1, gain_target_=1, rumble_fraction_=0;
  float rumble_low_ratio_db_=-60;
  float weak_speech_recovery_=0;
  float noise_step_recovery_=0,step_probability_=0;
  std::array<float,24> step_smooth_{},step_anchor_{};
  unsigned step_anchor_frames_=0,step_anchor_age_=100,step_candidates_=0;
  unsigned step_remaining_=0,step_cooldown_=0;
  float transient_dry_guard_=0,transient_guard_=0;
  float transient_noise_evidence_=0;
  float transient_periodicity_100ms_=0,transient_periodicity_500ms_=0;
  std::array<float,128> transient_power_{};
  unsigned transient_head_=0,transient_seen_=0;
};
}
