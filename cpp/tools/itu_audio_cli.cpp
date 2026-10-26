#include "itu/audio_engine.hpp"
#include "itu/reference_clock.hpp"
#include "linear_phase_eq.hpp"
#include "wav.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
namespace {
struct RecordedFrame {
  itu::FrameInfo info;
  itu::ClockInfo clock;
  double process_ms=0, release_lateness_ms=0;
  bool completion_late=false;
};
struct ClockEvent {std::uint64_t frame; double ppm;};
double power(const std::vector<float>& x,std::size_t begin,std::size_t end) {
  double p=0; for(auto i=begin;i<end;++i) p+=double(x[i])*x[i]; return p/std::max(std::size_t(1),end-begin);
}
double db(double p) {return 10*std::log10(std::max(p,1e-20));}
// Diagnostic stages retain the actual floating point amplitude, including
// low-level filter tails and peaks above PCM16 full scale.
void write_float_wav(const std::string& path,unsigned rate,const std::vector<float>& values) {
  if(values.size()>(std::numeric_limits<std::uint32_t>::max()-36)/4)
    throw std::runtime_error("diagnostic WAV exceeds RIFF size");
  std::ofstream file(path,std::ios::binary); if(!file) throw std::runtime_error("cannot write diagnostic WAV");
  auto u16=[&](std::uint16_t v){char b[2]={char(v),char(v>>8)};file.write(b,2);};
  auto u32=[&](std::uint32_t v){char b[4]={char(v),char(v>>8),char(v>>16),char(v>>24)};file.write(b,4);};
  file.write("RIFF",4);u32(36+std::uint32_t(values.size()*4));file.write("WAVEfmt ",8);
  u32(16);u16(3);u16(1);u32(rate);u32(rate*4);u16(4);u16(32);
  file.write("data",4);u32(std::uint32_t(values.size()*4));
  for(float value:values) {std::uint32_t bits;std::memcpy(&bits,&value,4);u32(bits);}
  if(!file) throw std::runtime_error("diagnostic WAV write failed");
}
std::string json_string(const std::string& value) {
  std::ostringstream s;s<<'"';
  for(unsigned char ch:value) {
    if(ch=='"' || ch=='\\') s<<'\\'<<ch;
    else if(ch<32) s<<"\\u00"<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(ch)<<std::dec;
    else s<<ch;
  }
  s<<'"';return s.str();
}
}
int main(int argc,char** argv) {
  try {
    itu::Config c; std::string mic_path,echo_path,prefix,profile="general"; double far_start=-1,far_end=-1;
    std::string rx_in_path,rx_spk_path,rx_ref_path;
    bool paced=false,explicit_channel_zero=false,simulate_clock=false,disable_clock=false,diagnostic_stages=false;
    double clock_ppm=0; std::string clock_events_path;
    unsigned requested_microphones=0,beam_delay_count=0;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];
      if(a=="--no-aec") c.aec=false;
      else if(a=="--no-bf") explicit_channel_zero=true;
      else if(a=="--no-ns") c.noise_suppression=false;
      else if(a=="--no-agc") c.automatic_gain=false;
      else if(a=="--tx-drc") c.tx_drc.enabled=true;
      else if(a=="--no-tx-drc") c.tx_drc.enabled=false;
      else if(a=="--no-limiter") c.limiter=false;
      else if(a=="--pace-10ms") paced=true;
      else if(a=="--ecnr-coordinated-control") c.ecnr_coordinated_control=true;
      else if(a=="--ns-statistical-control") c.ns_statistical_control=true;
      else if(a=="--vehicle-ns-control") c.vehicle_ns_control=true;
      else if(a=="--vehicle-ns-unified-budget") {c.vehicle_ns_control=true;c.vehicle_ns.unified_raw_budget=true;}
      else if(a=="--vehicle-ns-legacy-budget") c.vehicle_ns.unified_raw_budget=false;
      else if(a=="--speex-multirender-context") c.speex_multirender_context=true;
      else if(a=="--no-speex-multirender-context") c.speex_multirender_context=false;
      else if(a=="--vcp-joint-control") itu::apply_vcp_joint_control(c);
      else if(a=="--vehicle-adaptive-rumble") c.vehicle_ns.adaptive_rumble=true;
      else if(a=="--external-vad") c.external_vad=true;
      else if(a=="--no-external-vad") c.external_vad=false;
      else if(a=="--repeated-tone-control") c.repeated_tone_control=true;
      else if(a=="--repair-stage" && i+1<argc) {
        const auto stage=std::stoul(argv[++i]);
        if(stage!=1 && stage!=2 && stage!=3 && stage!=4)
          throw std::runtime_error("repair stage must be 1, 2, 3, or 4");
        c.repair_stage=static_cast<unsigned>(stage);
      }
    else if(a=="--ecnr-v2-control") c.ecnr_v2_control=true;
      else if(a=="--ecnr-v3-g1-control") c.ecnr_v3_g1_control=true;
      else if(a=="--ecnr-v2-double-evidence") c.ecnr_v2_double_evidence=true;
      else if(a=="--ecnr-v2-fullband-gain-calibration") c.ecnr_v2_fullband_gain_calibration=true;
      else if(a=="--diagnostic-stages") diagnostic_stages=true;
      else if(a=="--noise-transition-control") c.noise_transition_control=true;
      else if(a=="--spatial-mvdr") c.spatial_mvdr=true;
      else if(a=="--spatial-fixed-steering") c.spatial_fixed_steering=true;
      else if(a=="--spatial-continuous-covariance") c.spatial_continuous_covariance=true;
      else if(a=="--wind-control") c.wind_noise_control=true;
      else if(a=="--disable-clock-correction") disable_clock=true;
      else if(a=="--no-low-cut") c.linear_phase_low_cut=false;
      else if(a=="--native-fullband") c.native_fullband=true;
      else if(a=="--speex-speech-policy") c.speex_speech_policy=true;
      else if(a=="--transparent-nearend") c.transparent_nearend=true;
      else if(a=="--webrtc-adaptive-gain") c.adaptive_gain=true;
      else if(a=="--no-reference-guard") c.reference_guard=false;
      else if(a=="--default-nearend") c.transparent_nearend=false;
      else if(a=="--aec-fast-adaptation") c.fast_aec_adaptation=true;
      else if(a=="--very-transparent-nearend") c.very_transparent_nearend=true;
      else if(a=="--multirender-nearend-policy") c.multirender_nearend_policy=true;
      else if(a=="--multirender-speech-protection") c.multirender_speech_protection=true;
      else if(a=="--profile-stages") c.profile_stages=true;
      else if(a=="--speex-residual-suppression") c.speex_residual_suppression=true;
      else if(a=="--speex-coherence-residual") c.speex_coherence_residual_suppression=true;
      else if(a=="--no-speex-coherence-residual") c.speex_coherence_residual_suppression=false;
      else if(a=="--speex-native-conditioning") c.speex_native_conditioning=true;
      else if(a=="--speex-post-context") c.speex_post_context=true;
      else if(a=="--post-aec-evidence-window") c.post_aec_evidence_window=true;
      else if(a=="--min-stat-ns") c.min_stat_ns=true;
      else if(a=="--no-min-stat-ns") c.min_stat_ns=false;
      else if(a=="--low-level-projection-gain") c.low_level_projection_gain=true;
      else if(a=="--no-low-level-projection-gain") c.low_level_projection_gain=false;
      else if(a=="--stage4-noise-release") c.stage4_noise_release=true;
      else if(a=="--agc-fault" && i+1<argc) {
        const std::string fault=argv[++i];
        if(fault=="previous-frame") c.gain_fault=itu::GainFault::previous_frame;
        else if(fault=="inverted") c.gain_fault=itu::GainFault::inverted;
        else if(fault=="always-boost") c.gain_fault=itu::GainFault::always_boost;
        else throw std::runtime_error("unknown AGC fault");
      }
      else if(a=="--internal-delay") c.correlation_delay=false;
      else if(a=="--correlation-delay") c.correlation_delay=true;
      else if(a=="--stable-coherence") c.stable_echo_coherence=true;
      else if(a=="--linear-speech-protection") c.linear_speech_protection=true;
      else if(a=="--no-linear-speech-protection") c.linear_speech_protection=false;
      else if(a=="--no-raw-nearend-protection") c.raw_nearend_protection=false;
      else if(a=="--raw-nearend-protection") c.raw_nearend_protection=true;
      else if(a=="--speech-preserving-low-cut") c.speech_preserving_low_cut=true;
      else if(a=="--fullband-near-ns") c.fullband_near_ns=true;
      else if(a=="--hybrid-speech-ns") c.hybrid_speech_ns=true;
      else if(a=="--no-hybrid-speech-ns") c.hybrid_speech_ns=false;
      else if(a=="--default-aec-adaptation") c.fast_aec_adaptation=false;
      else if(a=="--linear-output-diagnostic") c.linear_output_diagnostic=true;
      else if(a=="--analyze-linear-noise") c.analyze_linear_noise=true;
      else if(a=="--analyze-nonlinear-noise") c.analyze_linear_noise=false;
      else if(a=="--adaptive-ns") c.adaptive_noise_policy=true;
      else if(a=="--fixed-ns-policy") c.adaptive_noise_policy=false;
      else if(a=="--no-coherence-dtd") c.coherence_dtd=false;
      else if(a=="--no-projection-safeguard") c.projection_echo_safeguard=false;
      else if(a=="--eq-before-agc") c.eq_before_agc=true;
      else if(a=="--no-eq-before-agc") c.eq_before_agc=false;
      else if(a=="--rx-processing") c.rx_processing=c.rx_config.enabled=true;
      else if(a=="--no-rx-processing") c.rx_processing=c.rx_config.enabled=false;
      else if(i+1<argc) {
        std::string v=argv[++i];
        if(a=="--profile") {
          if(i!=2) throw std::runtime_error("processing profile must precede other options");
          // One speech path only: the VCP chain. `vehicle_stream` stays as an
          // alias because the web demo's frozen command line still says it;
          // `general` remains the bare reference engine for probes. The other
          // presets are no longer selectable, so a run cannot silently measure
          // a different engine than the acceptance baseline.
          if(v=="vcp" || v=="barrot_vcp") c=itu::barrot_vcp_config();
          else if(v=="vcp_joint") c=itu::barrot_vcp_joint_config();
          else if(v=="vehicle_stream") c=itu::barrot_vcp_config();
          else if(v=="vcp_unified") c=itu::barrot_vcp_unified_config();
          else if(v=="general") c=itu::Config{};
          else throw std::runtime_error("supported profiles: vcp, vcp_joint, vcp_unified, vehicle_stream, general");
          profile=v;
        }
        else if(a=="--mic") mic_path=v; else if(a=="--echo") echo_path=v; else if(a=="--out") prefix=v;
        else if(a=="--rx-in") rx_in_path=v;
        else if(a=="--rx-spk") rx_spk_path=v;
        else if(a=="--rx-ref") rx_ref_path=v;
        else if(a=="--rx-drc-threshold") c.rx_config.drc.threshold_dbfs=std::stof(v);
        else if(a=="--rx-drc-ratio") c.rx_config.drc.ratio=std::stof(v);
        else if(a=="--rx-drc-knee") c.rx_config.drc.knee_width_db=std::stof(v);
        else if(a=="--tx-drc-threshold") c.tx_drc.threshold_dbfs=std::stof(v);
        else if(a=="--tx-drc-ratio") c.tx_drc.ratio=std::stof(v);
        else if(a=="--tx-drc-knee") c.tx_drc.knee_width_db=std::stof(v);
        else if(a=="--tx-drc-attack-ms") c.tx_drc.attack_ms=std::stof(v);
        else if(a=="--tx-drc-release-ms") c.tx_drc.release_ms=std::stof(v);
        else if(a=="--tx-drc-makeup-db") c.tx_drc.makeup_gain_db=std::stof(v);
        else if(a=="--tx-drc-ceiling-dbfs") c.tx_drc.ceiling_dbfs=std::stof(v);
        else if(a=="--tx-drc-lookahead-ms") c.tx_drc.lookahead_ms=std::stof(v);
        else if(a=="--simulate-clock-ppm") {clock_ppm=std::stod(v); simulate_clock=true;}
        else if(a=="--clock-events") {clock_events_path=v; simulate_clock=true;}
        else if(a=="--microphones") {
          if(v!="1" && v!="2" && v!="4") throw std::runtime_error("vehicle microphone selection must be 1, 2 or 4");
          requested_microphones=std::stoul(v);
        }
        else if(a=="--ns-level") c.noise_level=std::stoul(v);
        else if(a=="--aec-backend") {
          if(v=="webrtc") {
            c.aec_backend=itu::AecBackend::webrtc;
            c.speex_coherence_residual_suppression=false;
          }
          else if(v=="speex") c.aec_backend=itu::AecBackend::speex;
          else throw std::runtime_error("unknown AEC backend: "+v);
        }
        else if(a=="--speex-echo-tail-ms") c.speex_echo_tail_ms=std::stoul(v);
        else if(a=="--speex-active-echo-suppression-db") c.speex_active_echo_suppression_db=std::stoi(v);
        else if(a=="--multirender-innovation-threshold") c.multirender_innovation_threshold=std::stof(v);
        else if(a=="--ecnr-v2-nonlinear-vad-min") c.ecnr_v2_nonlinear_vad_min=std::stof(v);
        else if(a=="--ecnr-v2-coherence-margin") c.ecnr_v2_coherence_margin=std::stof(v);
        else if(a=="--ns-backend") {
          if(v=="webrtc") c.noise_backend=itu::NoiseBackend::webrtc;
          else if(v=="rnnoise") c.noise_backend=itu::NoiseBackend::rnnoise;
          else if(v=="speex") c.noise_backend=itu::NoiseBackend::speex;
          else if(v=="ulunas") c.noise_backend=itu::NoiseBackend::ulunas;
          else if(v=="omlsa_imcra") c.noise_backend=itu::NoiseBackend::omlsa_imcra;
          else if(v=="super_gaussian") c.noise_backend=itu::NoiseBackend::super_gaussian;
          else if(v=="unified") c.noise_backend=itu::NoiseBackend::unified;
          else throw std::runtime_error("unknown noise backend: "+v);
        }
        else if(a=="--speex-suppression-db") c.speex_suppression_db=std::stoi(v);
        else if(a=="--vehicle-ns-mild-strength") c.vehicle_ns.mild_strength=std::stoi(v);
        else if(a=="--vehicle-ns-repair-mode") c.vehicle_ns.repair_mode=std::stoul(v);
        else if(a=="--vehicle-rumble-hz") c.vehicle_ns.rumble_hz=std::stof(v);
        else if(a=="--vehicle-ns-policy") c.vehicle_ns.policy=v;
        else if(a=="--vehicle-ns-dry-max") c.vehicle_ns.dry_max=std::stof(v);
        else if(a=="--vehicle-ns-spectral-dry-guard") c.vehicle_ns.spectral_dry_guard=std::stof(v);
        else if(a=="--vehicle-ns-unified-learning-rate") c.vehicle_ns.unified_learning_rate=std::stof(v);
        else if(a=="--vehicle-ns-rumble-recovery") c.vehicle_ns.rumble_recovery=std::stof(v);
        else if(a=="--vehicle-ns-rumble-adaptive-hz") c.vehicle_ns.rumble_adaptive_hz=std::stof(v);
        else if(a=="--vehicle-ns-weak-speech-recovery") c.vehicle_ns.weak_speech_recovery=std::stof(v);
        else if(a=="--vehicle-ns-speech-band-recovery") c.vehicle_ns.speech_band_recovery=std::stof(v);
        else if(a=="--vehicle-ns-noise-step-recovery") c.vehicle_ns.noise_step_recovery=std::stof(v);
        else if(a=="--vehicle-ns-transient-dry-guard") c.vehicle_ns.transient_dry_guard=std::stof(v);
        else if(a=="--vehicle-ns-transient-spectral-guard-db") c.vehicle_ns.transient_spectral_guard_db=std::stof(v);
        else if(a=="--vehicle-ns-snr-threshold") c.vehicle_ns.snr_threshold_db=std::stof(v);
        else if(a=="--vehicle-ns-noise-gain") c.vehicle_ns.noise_gain=std::stof(v);
        else if(a=="--vehicle-ns-startup-ms") c.vehicle_ns.startup_ms=std::stoul(v);
        else if(a=="--vehicle-ns-slew-ms") c.vehicle_ns.slew_ms=std::stof(v);
        else if(a=="--vehicle-ns-release-ms") c.vehicle_ns.release_ms=std::stof(v);
        else if(a=="--vehicle-ns-quiet-floor-db") c.vehicle_ns.quiet_floor_db=std::stof(v);
        else if(a=="--vehicle-ns-floor-guard") c.vehicle_ns.floor_guard=std::stoul(v);
        else if(a=="--neural-model") c.neural_model_path=v;
        else if(a=="--hybrid-ns-model") c.vehicle_ns.student_model_path=v;
        else if(a=="--hybrid-ns-spp-mix") c.vehicle_ns.student_spp_mix=std::stof(v);
        else if(a=="--hybrid-ns-snr-mix") c.vehicle_ns.student_snr_mix=std::stof(v);
        else if(a=="--hybrid-ns-speech-floor-db") c.vehicle_ns.student_speech_floor_db=std::stof(v);
        else if(a=="--hybrid-speech-suppression-db") c.hybrid_speech_suppression_db=std::stoi(v);
        else if(a=="--high-snr-dry-mix") c.high_snr_dry_mix=std::stof(v);
        else if(a=="--low-cut-hz") c.low_cut_hz=std::stof(v);
        else if(a=="--eq-preamp-db") c.output_eq.preamp_db=std::stof(v);
        else if(a=="--eq-phase") {
          if(v=="minimum") c.output_eq.phase=itu::EqPhase::minimum;
          else if(v=="linear") c.output_eq.phase=itu::EqPhase::linear;
          else throw std::runtime_error("EQ phase must be minimum or linear");
        }
        else if(a=="--eq-linear-taps") c.output_eq.linear_phase_taps=std::stoul(v);
        else if(a=="--eq-band") {
          if(c.output_eq.bands==c.output_eq.band.size()) throw std::runtime_error("too many EQ bands");
          std::istringstream stream(v); std::string kind; float hz,gain,q; char comma1,comma2;
          if(!std::getline(stream,kind,',') || !(stream>>hz>>comma1>>gain>>comma2>>q) ||
              comma1!=',' || comma2!=',' || (stream>>kind))
            throw std::runtime_error("EQ band format: peak|low_shelf|high_shelf,hz,gain_db,q");
          itu::EqKind type;
          if(kind=="peak") type=itu::EqKind::peak;
          else if(kind=="low_shelf") type=itu::EqKind::low_shelf;
          else if(kind=="high_shelf") type=itu::EqKind::high_shelf;
          else throw std::runtime_error("unknown EQ kind");
          c.output_eq.band[c.output_eq.bands++]={type,hz,gain,q};
        }
        else if(a=="--fullband-min-highband-fraction") c.fullband_min_highband_fraction=std::stof(v);
        else if(a=="--aec-filter-blocks") c.aec_filter_blocks=std::stoul(v);
        else if(a=="--aec-delay-headroom") c.aec_delay_headroom_samples=std::stoul(v);
        else if(a=="--agc-initial-gain") c.initial_gain_db=std::stof(v);
        else if(a=="--agc-max-gain") c.max_gain_db=std::stof(v);
        else if(a=="--agc-rate") c.max_gain_change_db_per_second=std::stof(v);
        else if(a=="--agc-headroom") c.agc_headroom_db=std::stof(v);
        else if(a=="--agc-fixed-gain") {c.adaptive_gain=false; c.speech_gain_policy=false; c.fixed_gain_db=std::stof(v);}
        else if(a=="--stream-delay-ms") c.stream_delay_ms=std::stoul(v);
        else if(a=="--external-delay-ms") {c.correlation_delay=false; c.external_delay_estimator=true; c.stream_delay_ms=std::stoul(v);}
        else if(a=="--far-start") far_start=std::stod(v); else if(a=="--far-end") far_end=std::stod(v);
        else if(a=="--beam-delays") {
          c.beamforming=true; std::istringstream stream(v); std::string part; unsigned ch=0;
          while(std::getline(stream,part,',')) {if(ch>=8) throw std::runtime_error("too many beam delays"); c.beam_delays_samples[ch++]=std::stof(part);}
          beam_delay_count=ch;
        } else throw std::runtime_error("unknown or obsolete option: "+a);
      } else throw std::runtime_error("missing option value: "+a);
    }
    if((mic_path.empty() && rx_in_path.empty()) || prefix.empty()) throw std::runtime_error("usage: itu_audio_cli [--profile vehicle_stream|barrot_vcp|vcp_unified] [--mic before.wav] [--rx-in far.wav] [--rx-spk spk.wav] [--rx-ref ref.wav] [--microphones 1|2|4] [--echo playback.wav (1..8 synchronous channels) | --no-aec] --out prefix [...]");
    itu::Wav mic;
    if(!mic_path.empty()) {
      mic = itu::read_wav(mic_path);
      c.sample_rate = mic.sample_rate;
      c.frame_size = c.sample_rate / 100;
      c.microphones = mic.channels;
    }
    itu::Wav rx_wav;
    if(!rx_in_path.empty()) {
      if(!echo_path.empty()) throw std::runtime_error("--rx-in and --echo are mutually exclusive: use raw RX or actual playback reference");
      if(simulate_clock) throw std::runtime_error("--rx-in duplex replay uses synchronous playback; simulate drift with --echo instead");
      rx_wav = itu::read_wav(rx_in_path);
      if(rx_wav.samples.empty() || !rx_wav.channels || rx_wav.channels>8) throw std::runtime_error("RX needs 1..8 synchronous channels");
      if(mic_path.empty()) {
        c.sample_rate = rx_wav.sample_rate;
        c.frame_size = c.sample_rate / 100;
      } else if(rx_wav.sample_rate != c.sample_rate) {
        throw std::runtime_error("rx_in sample rate differs from mic sample rate");
      }
      c.reference_channels=rx_wav.channels;
      if(!mic_path.empty() && rx_wav.samples.size()/rx_wav.channels!=mic.samples.size()/mic.channels)
        throw std::runtime_error("RX and microphone streams must have the same sample count");
    }
    const unsigned h = c.frame_size;
    if(!mic_path.empty()) {
      if(requested_microphones && requested_microphones!=mic.channels)
        throw std::runtime_error("selected microphone count differs from actual WAV channels");
      if(explicit_channel_zero && c.beamforming)
        throw std::runtime_error("--no-bf cannot be combined with --beam-delays");
      if(c.microphones>1 && !c.beamforming && !explicit_channel_zero)
        throw std::runtime_error("multiple microphones require calibrated --beam-delays or explicit --no-bf channel 0 selection");
      if(c.beamforming && beam_delay_count!=mic.channels)
        throw std::runtime_error("one calibrated causal beam delay per actual microphone is required");
    }
    itu::Wav echo;
    if(!echo_path.empty()) {
      echo=itu::read_wav(echo_path);
      if(!echo.channels || echo.channels>8 || echo.sample_rate!=c.sample_rate)
        throw std::runtime_error("playback reference must have 1..8 synchronous channels, same sample rate");
      if(!mic_path.empty() && echo.samples.size()/echo.channels!=mic.samples.size()/mic.channels)
        throw std::runtime_error("playback reference must have same sample count as mic");
      c.reference_channels=echo.channels;
    }
    std::size_t n = !mic_path.empty() ? (mic.samples.size()/mic.channels) : (rx_wav.samples.size()/rx_wav.channels);
    if(!n) throw std::runtime_error("empty input");
    bool measure=far_start>=0 || far_end>=0;
    if(measure && (!c.aec || !std::isfinite(far_start) || !std::isfinite(far_end) || far_start<0 || far_end<=far_start || far_end>double(n)/c.sample_rate))
      throw std::runtime_error("ERLE estimate requires AEC and a valid explicit far-only interval");
    itu::DiagnosticStageTaps stage_taps;
    if(diagnostic_stages) c.diagnostic_stages=&stage_taps;
    itu::AudioEngine engine(c); itu::FrameInfo info;
    std::vector<float> rx_spk_samples, rx_ref_samples;
    std::vector<float> rx_chunk(h*c.reference_channels),spk_chunk(rx_chunk.size());
    std::size_t rx_processing_frames=0;
    if(!rx_in_path.empty() && mic_path.empty()) {
      const std::size_t rx_total = rx_wav.samples.size() / rx_wav.channels;
      rx_spk_samples.reserve(rx_total*c.reference_channels);
      rx_ref_samples.reserve(rx_total*c.reference_channels);
      std::vector<float> ref_chunk(rx_chunk.size());
      for(std::size_t rk = 0; rk * h < rx_total; ++rk) {
        std::size_t start = rk * h;
        std::size_t len = std::min(std::size_t(h), rx_total - start);
        std::fill(rx_chunk.begin(), rx_chunk.end(), 0.0f);
        std::copy_n(rx_wav.samples.data()+start*c.reference_channels,len*c.reference_channels,rx_chunk.data());
        if(engine.process_downlink(rx_chunk.data(),rx_chunk.size(),spk_chunk.data(),ref_chunk.data())!=itu::Status::ok)
          throw std::runtime_error("downlink processing failed");
        ++rx_processing_frames;
        rx_spk_samples.insert(rx_spk_samples.end(), spk_chunk.begin(), spk_chunk.begin() + len*c.reference_channels);
        rx_ref_samples.insert(rx_ref_samples.end(), ref_chunk.begin(), ref_chunk.begin() + len*c.reference_channels);
      }
      std::string out_spk_file = !rx_spk_path.empty() ? rx_spk_path : prefix + ".rx-spk.wav";
      std::string out_ref_file = !rx_ref_path.empty() ? rx_ref_path : prefix + ".rx-ref.wav";
      itu::write_wav(out_spk_file, c.sample_rate, c.reference_channels, rx_spk_samples);
      itu::write_wav(out_ref_file, c.sample_rate, c.reference_channels, rx_ref_samples);
    }
    if(mic_path.empty()) {
      std::ofstream report(prefix + ".json");
      report << "{\n  \"schema_version\":2,\n  \"sample_rate\":" << c.sample_rate
             << ",\n  \"mode\":\"downlink_only\""
             << ",\n  \"processing_profile\":\"" << profile << "\""
             << ",\n  \"rx_in\":\"" << rx_in_path << "\""
             << ",\n  \"rx_spk\":\"" << (!rx_spk_path.empty() ? rx_spk_path : prefix + ".rx-spk.wav") << "\""
             << ",\n  \"rx_ref\":\"" << (!rx_ref_path.empty() ? rx_ref_path : prefix + ".rx-ref.wav") << "\""
             << ",\n  \"rx_samples\":" << rx_spk_samples.size()
             << ",\n  \"rx_processing\":" << (c.rx_processing ? "true" : "false")
             << ",\n  \"reference_channels\":" << c.reference_channels
             << ",\n  \"rx_processing_frames\":" << rx_processing_frames
             << ",\n  \"standard_compliance\":false\n}\n";
      std::cout << "wrote " << prefix << ".{rx-spk.wav,rx-ref.wav,json}\n";
      return 0;
    }
    const bool has_playback=!echo.samples.empty() || !rx_in_path.empty();
    if(c.aec && !has_playback) throw std::runtime_error("AEC3 needs actual playback reference or --rx-in; otherwise use --no-aec");
    c=engine.config(); // Report effective multi-render safeguards, not requested mono heuristics.
    if((disable_clock && !simulate_clock) || !std::isfinite(clock_ppm) || std::abs(clock_ppm)>1500 ||
        (simulate_clock && echo.samples.empty())) throw std::runtime_error("simulated clock requires playback PCM and finite drift within +/-1500 ppm");
    if(simulate_clock && c.reference_channels>1) throw std::runtime_error("CLI simulated ASRC currently requires mono render; synchronous multi-render PCM is supported");
    std::unique_ptr<itu::ReferenceClock> clock;
    if(simulate_clock) clock=std::make_unique<itu::ReferenceClock>(c.sample_rate);
    const double render_rate=c.sample_rate*(1+clock_ppm*1e-6);
    std::uint64_t written=0;
    std::vector<float> render_chunk(h);
    std::vector<ClockEvent> clock_events;
    if(!clock_events_path.empty()) {
      std::ifstream file(clock_events_path); std::string line;
      if(!file) throw std::runtime_error("cannot read simulated clock events");
      while(std::getline(file,line)) {
        std::istringstream row(line); std::uint64_t frame; double ppm; char comma;
        if(!(row>>frame>>comma>>ppm) || comma!=',' || (row>>line) || !std::isfinite(ppm) || std::abs(ppm)>1500 ||
            (clock_events.empty()?frame!=0:frame<=clock_events.back().frame))
          throw std::runtime_error("clock events require ordered capture frame,ppm; first frame zero");
        clock_events.push_back({frame,ppm});
      }
      if(clock_events.empty()) throw std::runtime_error("empty clock events");
    }
    std::size_t event_index=0; std::uint64_t clock_epoch=0;
    double epoch_render_rate=render_rate;
    std::vector<float> frame(h*c.microphones),ref(h*c.reference_channels),out(h),at(frame.size()),bt(h),lt(160*c.microphones),it(frame.size());
    std::vector<float> after,aec,bf,linear,conditioned; std::vector<double> times;
    std::ofstream csv(prefix+".frames.csv"); if(!csv) throw std::runtime_error("cannot write frames CSV; create directory first");
    csv<<"frame,time_s,far_active,linear_aec_available,statistics_valid,mic_rms,aec_rms,out_rms,erl_db,erle_db,estimated_delay_ms,limited_samples,process_ms,speech_probability,applied_gain_db,reference_state,reference_residual_ratio,reference_confidence,linear_speech_probability,nearend_mix,linear_residual_ratio,nonlinear_speech_probability,echo_subtraction_gain,noise_speech_mix,noise_flatness,incoherent_ratio,echo_coherence_baseline,internal_delay_mix,lowband_restore_mix,coherence_learning_frames,fullband_ns_mix,fullband_noise_power,highband_energy_fraction,release_lateness_ms,completion_late,clock_locked,clock_estimated_ppm,clock_queued_samples,clock_capture_hz,clock_render_hz,clock_desired_ppm,clock_capture_valid,clock_render_valid,spatial_delay_samples,spatial_noise_updates,spatial_fallback_bins,wind_score,noise_control_state,noise_transition_events,startup_dry_mix,neural_inferences,native_capture_ms,native_render_ms,native_core_ms,native_output_ms,native_taps_ms,pre_ns_rms,post_ns_rms,min_stat_gain,gain_control_noise_power,gain_speech_qualified,gain_control_state,gain_gate_level,gain_input_level,gain_limit_db,gain_votes,gain_voiced_rms,gain_peak,gain_max_voiced,gain_target_db,gain_hold_db,noise_profile_observations,noise_profile_mix,v2_double_evidence_eligible,v2_near_confirmation,g1_strict_near_age,g1_startup_echo_frames,g1_startup_clear_threshold,g1_startup_clear_deferred,near_protection_hold_frames,startup_echo_confirmation,g1_echo_evidence_frames,g1_echo_clear_threshold,g1_echo_clear_deferred,g1_strong_near_confirmation,g1_near_latch_active,g1_original_confirmed_echo,g1_original_coherent_clear,ns_residual_gain,ns_residual_flatness,ns_residual_quiet_frames,ns_residual_observations,tone_frequency_hz,tone_mix,tone_concentration,tone_repeats,external_vad_active,external_vad_calls,input_excess_score,input_excess_fraction,control_evidence_probability,gain_noise_evidence,gain_noise_compensation,post_aec_context_applied,post_aec_cancellation_ratio,post_aec_echo_floor,post_aec_coherence_floor,post_aec_near_observed,post_aec_echo_clear_frames,hybrid_ns_state,hybrid_ns_inferences,hybrid_ns_fallback_frames,hybrid_ns_applied,vehicle_ns_raw_mix_budget,vehicle_ns_raw_mix_min,vehicle_ns_raw_mix_max,post_aec_reference_evidence_valid,post_aec_near_state,post_aec_multirender_frames,unified_ns_w0,unified_ns_w1,unified_ns_w2,unified_ns_w3,unified_ns_target_w0,unified_ns_target_w1,unified_ns_target_w2,unified_ns_target_w3,unified_ns_weight_entropy,unified_ns_frame_snr_db,unified_ns_flatness,unified_ns_tonality,unified_ns_minima_confidence,unified_ns_transient_evidence,unified_ns_noise_step_evidence,unified_ns_probability,unified_ns_gain,unified_ns_noise_power,unified_ns_bootstrap_seed,unified_ns_noise_updates\n";
    std::size_t frames=(n+h-1)/h,total_frames=frames,limited=0,unreliable=0; bool all_linear=c.aec;
    // Include the largest legal causal BF compensation and DSP drain before replay.
    const auto drain_capacity=std::size_t(4096/h+16);
    const auto capacity=(frames+drain_capacity)*h;
    after.reserve(capacity); bf.reserve(capacity); times.reserve(frames);
    if(!rx_in_path.empty()) {rx_spk_samples.reserve(n*c.reference_channels);rx_ref_samples.reserve(n*c.reference_channels);}
    aec.reserve(capacity*c.microphones); linear.reserve(capacity*c.microphones);
    conditioned.reserve(capacity*c.microphones);
    std::vector<RecordedFrame> recorded(frames+drain_capacity);
    const bool core_bridge=(c.sample_rate==24000 && ((c.noise_backend!=itu::NoiseBackend::omlsa_imcra && c.noise_backend!=itu::NoiseBackend::super_gaussian && c.noise_backend!=itu::NoiseBackend::unified) || (c.aec && c.aec_backend!=itu::AecBackend::speex))) || (c.sample_rate==16000 && c.noise_backend==itu::NoiseBackend::rnnoise);
    const unsigned core_h=core_bridge ? 480 : h;
    std::array<std::vector<float>,5> stages;
    if(diagnostic_stages) for(unsigned j=0;j<stages.size();++j)
      stages[j].resize((frames+drain_capacity)*((j<3 || c.eq_before_agc)?core_h:h)); // Before callback replay.
    const auto stream_start=std::chrono::steady_clock::now();
    for(std::size_t k=0;k<total_frames;++k) {
      const auto release=stream_start+std::chrono::milliseconds(k*10);
      if(paced) std::this_thread::sleep_until(release);
      const auto actual_release=std::chrono::steady_clock::now();
      std::fill(frame.begin(),frame.end(),0); std::fill(ref.begin(),ref.end(),0);
      std::size_t start=k*h,remaining=start<n ? std::min(std::size_t(h),n-start):0;
      if(remaining) {
        std::copy_n(mic.samples.data()+start*c.microphones,remaining*c.microphones,frame.data());
        if(!echo.samples.empty()) std::copy_n(echo.samples.data()+start*c.reference_channels,remaining*c.reference_channels,ref.data());
      }
      auto t=std::chrono::steady_clock::now();
      if(clock) {
        if(event_index<clock_events.size() && k==clock_events[event_index].frame) {
          clock->reset(); clock_epoch=k; written=k*h;
          epoch_render_rate=c.sample_rate*(1+clock_events[event_index++].ppm*1e-6);
        }
        if(!disable_clock && (k-clock_epoch)%50==0) {
          const auto ns=std::int64_t(1000000000LL+k*10000000LL);
          if(!clock->observe_capture({(k-clock_epoch)*h,ns}) ||
             !clock->observe_render({std::uint64_t(std::llround((k-clock_epoch)*.01*epoch_render_rate)),ns}))
            throw std::runtime_error("simulated timestamp rejected");
        }
        const auto available=clock_epoch*h+std::uint64_t(std::floor(((k-clock_epoch)*.01+.08)*epoch_render_rate));
        while(written+h<=available) {
          for(unsigned i=0;i<h;++i) render_chunk[i]=written+i<echo.samples.size() ? echo.samples[written+i] : 0;
          if(!clock->push(render_chunk.data(),h)) throw std::runtime_error("simulated playback FIFO overflow");
          written+=h;
        }
        if(!clock->pull(ref.data(),h)) throw std::runtime_error("simulated playback FIFO underrun");
      }
      if(!rx_in_path.empty() && remaining) {
        std::fill(rx_chunk.begin(),rx_chunk.end(),0.f);
        std::copy_n(rx_wav.samples.data()+start*c.reference_channels,remaining*c.reference_channels,rx_chunk.data());
        if(engine.process_downlink(rx_chunk.data(),rx_chunk.size(),spk_chunk.data(),ref.data())!=itu::Status::ok)
          throw std::runtime_error("duplex downlink processing failed");
        ++rx_processing_frames;
        // The final partial frame ends with the actual playback file, not an
        // unplayed EQ/DRC tail. DSP drain callbacks receive silent playback.
        std::fill(ref.begin()+remaining*c.reference_channels,ref.end(),0.f);
      }
      auto result=engine.process(frame.data(),frame.size(),has_playback?ref.data():nullptr,has_playback?ref.size():0,out.data(),out.size(),info,at.data(),bt.data(),lt.data(),it.data());
      const auto completed=std::chrono::steady_clock::now();
      double ms=std::chrono::duration<double,std::milli>(completed-t).count();
      if(result!=itu::Status::ok) throw std::runtime_error("processing failed: "+std::to_string(static_cast<int>(result)));
      if(!rx_in_path.empty() && remaining) {
        rx_spk_samples.insert(rx_spk_samples.end(),spk_chunk.begin(),spk_chunk.begin()+remaining*c.reference_channels);
        rx_ref_samples.insert(rx_ref_samples.end(),ref.begin(),ref.begin()+remaining*c.reference_channels);
      }
      if(k==0) total_frames=frames+(info.latency_samples+h-1)/h;
      if(diagnostic_stages) {
        const std::array<const float*,5> taps={stage_taps.nr_selected.data(),stage_taps.nr_protected.data(),
            stage_taps.post_agc.data(),stage_taps.pre_eq.data(),stage_taps.post_eq.data()};
        for(unsigned j=0;j<stages.size();++j) {
          const auto length=j<3?core_h:stage_taps.eq_frame_size;
          std::copy_n(taps[j],length,stages[j].data()+k*length);
        }
      }
      after.insert(after.end(),out.begin(),out.end()); aec.insert(aec.end(),at.begin(),at.end()); bf.insert(bf.end(),bt.begin(),bt.end());
      conditioned.insert(conditioned.end(),it.begin(),it.end());
      for(unsigned i=0;i<160;++i) for(unsigned ch=0;ch<c.microphones;++ch) linear.push_back(lt[ch*160+i]);
      if(k<frames) {limited+=info.limited_samples; times.push_back(ms); all_linear &= info.linear_aec_available; unreliable+=info.reference_state==2;}
      recorded[k]={info,clock?clock->info():itu::ClockInfo{},ms,paced?std::max(0.,std::chrono::duration<double,std::milli>(actual_release-release).count()):0.,
                   paced && completed>release+std::chrono::milliseconds(10)};
    }
    if(!rx_in_path.empty()) {
      itu::write_wav(!rx_spk_path.empty()?rx_spk_path:prefix+".rx-spk.wav",c.sample_rate,c.reference_channels,rx_spk_samples);
      itu::write_wav(!rx_ref_path.empty()?rx_ref_path:prefix+".rx-ref.wav",c.sample_rate,c.reference_channels,rx_ref_samples);
    }
    std::size_t completion_misses=0,process_misses=0;
    double release_lateness_max=0;
    // File I/O is outside the paced callback loop.
    for(std::size_t k=0;k<total_frames;++k) {
      const auto& info=recorded[k].info; const auto ms=recorded[k].process_ms;
      if(k<frames) {
        completion_misses+=recorded[k].completion_late; process_misses+=ms>10;
        release_lateness_max=std::max(release_lateness_max,recorded[k].release_lateness_ms);
      }
      csv<<k<<','<<double(k*h)/c.sample_rate<<','<<info.far_active<<','<<info.linear_aec_available<<','<<info.statistics_valid<<','<<info.microphone_rms<<','<<info.aec_rms<<','<<info.output_rms<<',';
      if(info.statistics_valid & itu::erl_valid) csv<<info.echo_return_loss_db;
      csv<<','; if(info.statistics_valid & itu::erle_valid) csv<<info.echo_return_loss_enhancement_db;
      csv<<','; if(info.statistics_valid & itu::delay_valid) csv<<info.estimated_delay_ms;
      csv<<','<<info.limited_samples<<','<<ms<<','<<info.speech_probability<<','<<info.applied_gain_db<<','<<info.reference_state<<','<<info.reference_residual_ratio<<','<<info.reference_confidence<<','<<info.linear_speech_probability<<','<<info.nearend_mix<<','<<info.linear_residual_ratio<<','<<info.nonlinear_speech_probability<<','<<info.echo_subtraction_gain<<','<<info.noise_speech_mix<<','<<info.noise_flatness<<','<<info.incoherent_ratio<<','<<info.echo_coherence_baseline<<','<<info.internal_delay_mix<<','<<info.lowband_restore_mix<<','<<info.coherence_learning_frames<<','<<info.fullband_ns_mix<<','<<info.fullband_noise_power<<','<<info.highband_energy_fraction<<','<<recorded[k].release_lateness_ms<<','<<recorded[k].completion_late<<','<<recorded[k].clock.locked<<','<<recorded[k].clock.drift_ppm<<','<<recorded[k].clock.queued_samples<<','<<recorded[k].clock.capture_rate_hz<<','<<recorded[k].clock.render_rate_hz<<','<<recorded[k].clock.desired_ppm<<','<<recorded[k].clock.capture_valid<<','<<recorded[k].clock.render_valid
         <<','<<info.spatial_delay_samples<<','<<info.spatial_noise_updates<<','<<info.spatial_fallback_bins<<','<<info.wind_score<<','<<info.noise_control_state<<','<<info.noise_transition_events<<','<<info.startup_dry_mix<<','<<info.neural_inferences
         <<','<<info.native_capture_ms<<','<<info.native_render_ms<<','<<info.native_core_ms<<','<<info.native_output_ms<<','<<info.native_taps_ms<<','<<info.pre_ns_rms<<','<<info.post_ns_rms<<','<<info.min_stat_gain<<','<<info.gain_control_noise_power<<','<<info.gain_speech_qualified<<','<<info.gain_control_state<<','<<info.gain_gate_level<<','<<info.gain_input_level<<','<<info.gain_limit_db<<','<<info.gain_votes<<','<<info.gain_voiced_rms<<','<<info.gain_peak<<','<<info.gain_max_voiced<<','<<info.gain_target_db<<','<<info.gain_hold_db<<','<<info.noise_profile_observations<<','<<info.noise_profile_mix<<','<<info.v2_double_evidence_eligible<<','<<info.v2_near_confirmation<<','<<info.g1_strict_near_age<<','<<info.g1_startup_echo_frames<<','<<info.g1_startup_clear_threshold<<','<<info.g1_startup_clear_deferred<<','<<info.near_protection_hold_frames<<','<<info.startup_echo_confirmation<<','<<info.g1_echo_evidence_frames<<','<<info.g1_echo_clear_threshold<<','<<info.g1_echo_clear_deferred<<','<<info.g1_strong_near_confirmation<<','<<info.g1_near_latch_active<<','<<info.g1_original_confirmed_echo<<','<<info.g1_original_coherent_clear<<','<<info.ns_residual_gain<<','<<info.ns_residual_flatness<<','<<info.ns_residual_quiet_frames<<','<<info.ns_residual_observations<<','<<info.tone_frequency_hz<<','<<info.tone_mix<<','<<info.tone_concentration<<','<<info.tone_repeats<<','<<info.external_vad_active<<','<<info.external_vad_calls<<','<<info.input_excess_score<<','<<info.input_excess_fraction<<','<<info.control_evidence_probability<<','<<info.gain_noise_evidence<<','<<info.gain_noise_compensation<<','<<info.post_aec_context_applied<<','<<info.post_aec_cancellation_ratio<<','<<info.post_aec_echo_floor<<','<<info.post_aec_coherence_floor<<','<<info.post_aec_near_observed<<','<<info.post_aec_echo_clear_frames<<','<<info.hybrid_ns_state<<','<<info.hybrid_ns_inferences<<','<<info.hybrid_ns_fallback_frames<<','<<info.hybrid_ns_applied<<','<<info.vehicle_ns_raw_mix_budget<<','<<info.vehicle_ns_raw_mix_min<<','<<info.vehicle_ns_raw_mix_max<<','<<info.post_aec_reference_evidence_valid<<','<<info.post_aec_near_state<<','<<info.post_aec_multirender_frames<<','
         <<info.unified_ns_weights[0]<<','<<info.unified_ns_weights[1]<<','<<info.unified_ns_weights[2]<<','<<info.unified_ns_weights[3]<<','
         <<info.unified_ns_target_weights[0]<<','<<info.unified_ns_target_weights[1]<<','<<info.unified_ns_target_weights[2]<<','<<info.unified_ns_target_weights[3]<<','
         <<info.unified_ns_weight_entropy<<','<<info.unified_ns_frame_snr_db<<','<<info.unified_ns_flatness<<','<<info.unified_ns_tonality<<','
         <<info.unified_ns_minima_confidence<<','<<info.unified_ns_transient_evidence<<','<<info.unified_ns_noise_step_evidence<<','
         <<info.unified_ns_probability<<','<<info.unified_ns_gain<<','<<info.unified_ns_noise_power<<','<<info.unified_ns_bootstrap_seed<<','
         <<info.unified_ns_noise_updates<<'\n';
    }
    unsigned aec_delay=info.aec_output_delay_samples,linear_delay=info.linear_aec_delay_samples;
    after.resize(n+info.latency_samples); aec.resize((n+aec_delay)*c.microphones);
    linear.resize((n*16000/c.sample_rate+linear_delay)*c.microphones);
    bf.resize(n+info.bf_delay_samples);
    conditioned.resize((n+info.aec_input_delay_samples)*c.microphones);
    itu::write_wav(prefix+".after.wav",c.sample_rate,1,after);
    itu::write_wav(prefix+".aec.wav",c.sample_rate,c.microphones,aec);
    if(all_linear) itu::write_wav(prefix+".linear-aec.wav",16000,c.microphones,linear);
    if(c.aec) itu::write_wav(prefix+".aec-input.wav",c.sample_rate,c.microphones,conditioned);
    itu::write_wav(prefix+".bf.wav",c.sample_rate,1,bf);
    if(diagnostic_stages) {
      const std::array<const char*,5> names={"nr-selected","nr-protected-agc-before","agc-after","eq-before","eq-after"};
      const std::array<const char*,5> domains={
        "selected NR backend plus aligned fullband branch; before shared speech protection and lowband restoration",
        "protected NR output with lowband/startup restoration; before optional pre-AGC EQ",
        "core-rate AGC output before TX DRC and limiter",
        "actual EQ input; core rate before AGC when selected, otherwise I/O rate after AGC",
        "actual EQ output; before AGC/DRC when selected, otherwise before final I/O limiter; IIR phase retained"};
      std::ofstream metadata(prefix+".stages.json");if(!metadata) throw std::runtime_error("cannot write stage metadata");
      metadata<<std::setprecision(12)<<"{\n  \"schema_version\":1,\n  \"sample_format\":\"IEEE_FLOAT32\",\n"
        <<"  \"alignment\":\"sample zero input time = (trimmed_leading_samples - fixed_delay_samples) / sample_rate; no fitted lag\",\n"
        <<"  \"stages\":[\n";
      for(unsigned j=0;j<stages.size();++j) {
        const unsigned rate=j<3?stage_taps.core_sample_rate:stage_taps.eq_sample_rate;
        const unsigned delay=j<3?stage_taps.core_delay_samples:stage_taps.eq_delay_samples;
        const std::size_t sample_count=n*rate/c.sample_rate+delay;
        if(!rate || stages[j].size()<sample_count) throw std::runtime_error("invalid diagnostic stage extent");
        stages[j].resize(sample_count);
        const std::string path=std::filesystem::absolute(prefix+"."+names[j]+".float.wav").lexically_normal().string();
        write_float_wav(path,rate,stages[j]);
        if(j) metadata<<",\n";
        metadata<<"    {\"name\":"<<json_string(names[j])<<",\"file\":"<<json_string(path)
          <<",\"sample_rate\":"<<rate<<",\"channels\":1,\"samples_per_channel\":"<<sample_count
          <<",\"fixed_delay_samples\":"<<delay<<",\"trimmed_leading_samples\":0"
          <<",\"timeline_offset_ms\":"<<(-1000.*delay/rate)<<",\"measurement_domain\":"<<json_string(domains[j])
          <<",\"nr_backend_active\":"<<(stage_taps.nr_selected_available?"true":"false")<<"}";
      }
      metadata<<"\n  ]\n}\n";if(!metadata) throw std::runtime_error("diagnostic metadata write failed");
    }
    double input_p=power(mic.samples,0,mic.samples.size()),output_p=power(after,info.latency_samples,after.size());
    double total=0; for(double t:times) total+=t; std::sort(times.begin(),times.end());
    std::ofstream report(prefix+".json"); if(!report) throw std::runtime_error("cannot write report");
    report<<std::setprecision(9)<<"{\n  \"schema_version\":2,\n  \"webrtc_revision\":\""<<itu::kWebRtcRevision<<"\",\n  \"sample_rate\":"<<c.sample_rate<<",\n  \"microphones\":"<<c.microphones<<",\n  \"rx_processing\":"<<(c.rx_processing?"true":"false")<<",\n  \"eq_before_agc\":"<<(c.eq_before_agc?"true":"false")<<",\n  \"rx_in\":"<<json_string(rx_in_path)
      <<",\n  \"reference_channels\":"<<c.reference_channels
      <<",\n  \"aec_capture_channels\":"<<info.aec_capture_channels
      <<",\n  \"beamforming_before_aec\":"<<(info.beamforming_before_aec?"true":"false")
      <<",\n  \"bf_delay_samples\":"<<info.bf_delay_samples
      <<",\n  \"tx_drc_enabled\":"<<(c.tx_drc.enabled?"true":"false")
      <<",\n  \"tx_drc_config\":{\"threshold_dbfs\":"<<c.tx_drc.threshold_dbfs
      <<",\"ratio\":"<<c.tx_drc.ratio
      <<",\"knee_width_db\":"<<c.tx_drc.knee_width_db
      <<",\"attack_ms\":"<<c.tx_drc.attack_ms
      <<",\"release_ms\":"<<c.tx_drc.release_ms
      <<",\"makeup_gain_db\":"<<c.tx_drc.makeup_gain_db
      <<",\"ceiling_dbfs\":"<<c.tx_drc.ceiling_dbfs
      <<",\"lookahead_ms\":"<<c.tx_drc.lookahead_ms<<"}"
      <<",\n  \"tx_drc_delay_samples\":"<<info.tx_drc_delay_samples
      <<",\n  \"rx_processing_frames\":"<<rx_processing_frames
      <<",\n  \"reference_source\":"<<json_string(!rx_in_path.empty()?"post_rx_processing":"provided_playback_pcm")
      <<",\n  \"multi_render_guard_policy\":\""<<(c.reference_channels>1?"aec3_native_dtd_no_mono_projection":"mono_reference_guards")<<"\""
      <<",\n  \"external_vad_enabled\":"<<(c.external_vad?"true":"false")
      <<",\n  \"noise_transition_control\":"<<(c.noise_transition_control?"true":"false")
      <<",\n  \"noise_transition_events\":"<<info.noise_transition_events
      <<",\n  \"neural_inferences\":"<<info.neural_inferences
      <<",\n  \"spatial_mvdr\":"<<(c.spatial_mvdr?"true":"false")
      <<",\n  \"spatial_steering_policy\":\""<<(c.spatial_fixed_steering?"fixed_calibrated":"observed_covariance")<<"\""
      <<",\n  \"spatial_covariance_policy\":\""<<(c.spatial_continuous_covariance?"continuous_mixture_mpdr":"noise_qualified")<<"\""
      <<",\n  \"spatial_covariance_updates\":"<<info.spatial_covariance_updates
      <<",\n  \"wind_noise_control\":"<<(c.wind_noise_control?"true":"false")
      <<",\n  \"spatial_delay_samples\":"<<info.spatial_delay_samples
      <<",\n  \"spatial_noise_updates\":"<<info.spatial_noise_updates
      <<",\n  \"spatial_fallback_bins\":"<<info.spatial_fallback_bins
      <<",\n  \"reference_unreliable_frames\":"<<unreliable
      <<",\n  \"aec_delay_policy\":\""<<(c.correlation_delay?"external_gcc_phat":c.external_delay_estimator?"external_device_buffer":"aec3_internal")<<"\""
      <<",\n  \"ns_backend\":\""<<(c.noise_backend==itu::NoiseBackend::unified?"unified":c.noise_backend==itu::NoiseBackend::super_gaussian?"super_gaussian":c.noise_backend==itu::NoiseBackend::omlsa_imcra?"omlsa_imcra":c.noise_backend==itu::NoiseBackend::ulunas?"ulunas":c.noise_backend==itu::NoiseBackend::speex?"speex":c.noise_backend==itu::NoiseBackend::rnnoise?"rnnoise":"webrtc")<<"\""
      <<",\n  \"speex_suppression_db\":"<<c.speex_suppression_db
      <<",\n  \"speex_multirender_context\":"<<(c.speex_multirender_context?"true":"false")
      <<",\n  \"vehicle_ns_control\":"<<(c.vehicle_ns_control?"true":"false")
      <<",\n  \"vehicle_ns_unified_raw_budget\":"<<(c.vehicle_ns.unified_raw_budget?"true":"false")
      <<",\n  \"vehicle_ns_unified_learning_rate\":"<<c.vehicle_ns.unified_learning_rate
      <<",\n  \"unified_ns_weights\":["<<info.unified_ns_weights[0]<<","<<info.unified_ns_weights[1]<<","<<info.unified_ns_weights[2]<<","<<info.unified_ns_weights[3]<<"]"
      <<",\n  \"unified_ns_target_weights\":["<<info.unified_ns_target_weights[0]<<","<<info.unified_ns_target_weights[1]<<","<<info.unified_ns_target_weights[2]<<","<<info.unified_ns_target_weights[3]<<"]"
      <<",\n  \"unified_ns_weight_entropy\":"<<info.unified_ns_weight_entropy
      <<",\n  \"unified_ns_frame_snr_db\":"<<info.unified_ns_frame_snr_db
      <<",\n  \"unified_ns_flatness\":"<<info.unified_ns_flatness
      <<",\n  \"unified_ns_tonality\":"<<info.unified_ns_tonality
      <<",\n  \"unified_ns_minima_confidence\":"<<info.unified_ns_minima_confidence
      <<",\n  \"unified_ns_transient_evidence\":"<<info.unified_ns_transient_evidence
      <<",\n  \"unified_ns_noise_step_evidence\":"<<info.unified_ns_noise_step_evidence
      <<",\n  \"unified_ns_probability\":"<<info.unified_ns_probability
      <<",\n  \"unified_ns_gain\":"<<info.unified_ns_gain
      <<",\n  \"unified_ns_noise_power\":"<<info.unified_ns_noise_power
      <<",\n  \"unified_ns_bootstrap_seed\":"<<info.unified_ns_bootstrap_seed
      <<",\n  \"unified_ns_noise_updates\":"<<info.unified_ns_noise_updates
      <<",\n  \"hybrid_ns_model\":"<<json_string(c.vehicle_ns.student_model_path)
      <<",\n  \"hybrid_ns_state\":"<<info.hybrid_ns_state
      <<",\n  \"hybrid_ns_inferences\":"<<info.hybrid_ns_inferences
      <<",\n  \"hybrid_ns_fallback_frames\":"<<info.hybrid_ns_fallback_frames
      <<",\n  \"hybrid_ns_spp_mix\":"<<c.vehicle_ns.student_spp_mix
      <<",\n  \"hybrid_ns_snr_mix\":"<<c.vehicle_ns.student_snr_mix
      <<",\n  \"hybrid_ns_teacher_in_runtime\":false"
      <<",\n  \"vehicle_ns_repair_mode\":"<<c.vehicle_ns.repair_mode
      <<",\n  \"vehicle_ns_mild_strength\":"<<c.vehicle_ns.mild_strength
      <<",\n  \"vehicle_rumble_hz\":"<<c.vehicle_ns.rumble_hz
      <<",\n  \"vehicle_adaptive_rumble\":"<<(c.vehicle_ns.adaptive_rumble?"true":"false")
      <<",\n  \"vehicle_ns_policy\":"<<json_string(c.vehicle_ns.policy)
      <<",\n  \"vehicle_ns_dry_max\":"<<c.vehicle_ns.dry_max
      <<",\n  \"vehicle_ns_spectral_dry_guard\":"<<c.vehicle_ns.spectral_dry_guard
      <<",\n  \"vehicle_ns_rumble_recovery\":"<<c.vehicle_ns.rumble_recovery
      <<",\n  \"vehicle_ns_rumble_adaptive_hz\":"<<c.vehicle_ns.rumble_adaptive_hz
      <<",\n  \"vehicle_ns_rumble_corner_hz\":"<<info.vehicle_ns_rumble_corner_hz
      <<",\n  \"vehicle_ns_rumble_low_ratio_db\":"<<info.vehicle_ns_rumble_low_ratio_db
      <<",\n  \"vehicle_ns_weak_speech_recovery\":"<<c.vehicle_ns.weak_speech_recovery
      <<",\n  \"vehicle_ns_speech_band_recovery\":"<<c.vehicle_ns.speech_band_recovery
      <<",\n  \"vehicle_ns_transient_dry_guard\":"<<c.vehicle_ns.transient_dry_guard
      <<",\n  \"vehicle_ns_transient_spectral_guard_db\":"<<c.vehicle_ns.transient_spectral_guard_db
      <<",\n  \"vehicle_ns_noise_step_recovery\":"<<c.vehicle_ns.noise_step_recovery
      <<",\n  \"vehicle_ns_snr_threshold\":"<<c.vehicle_ns.snr_threshold_db
      <<",\n  \"vehicle_ns_noise_gain\":"<<c.vehicle_ns.noise_gain
      <<",\n  \"vehicle_ns_startup_ms\":"<<c.vehicle_ns.startup_ms
      <<",\n  \"vehicle_ns_slew_ms\":"<<c.vehicle_ns.slew_ms
      <<",\n  \"vehicle_ns_release_ms\":"<<c.vehicle_ns.release_ms
      <<",\n  \"vehicle_ns_quiet_floor_db\":"<<c.vehicle_ns.quiet_floor_db
      <<",\n  \"vehicle_ns_floor_guard\":"<<c.vehicle_ns.floor_guard
      <<",\n  \"speex_speech_policy\":"<<(c.speex_speech_policy?"true":"false")
      <<",\n  \"hybrid_speech_ns\":"<<(c.hybrid_speech_ns?"true":"false")
      <<",\n  \"hybrid_speech_suppression_db\":"<<c.hybrid_speech_suppression_db
      <<",\n  \"high_snr_dry_mix\":"<<c.high_snr_dry_mix
      <<",\n  \"output_eq_preamp_db\":"<<c.output_eq.preamp_db
      <<",\n  \"output_eq_phase\":\""<<(c.output_eq.phase==itu::EqPhase::linear?"linear":"minimum")<<"\""
      <<",\n  \"output_eq_linear_taps\":"<<(c.output_eq.phase==itu::EqPhase::linear?
          itu::LinearPhaseEq::taps_for(c.output_eq,c.sample_rate):0)
      <<",\n  \"output_eq_delay_samples\":"<<(c.output_eq.phase==itu::EqPhase::linear?
          itu::LinearPhaseEq::delay_for(c.output_eq,c.sample_rate):0)
      <<",\n  \"output_eq_scope\":\""<<(c.eq_before_agc ?
          "uplink after protected NR, before AGC/DRC; core-rate IIR phase retained" :
          "uplink after AGC, before final I/O-rate limiter; IIR phase retained")<<"\""
      <<",\n  \"output_eq_bands\":[";
    for(unsigned i=0;i<c.output_eq.bands;++i) {
      const auto& band=c.output_eq.band[i];
      if(i) report<<',';
      report<<"{\"kind\":\""<<(band.kind==itu::EqKind::peak?"peak":band.kind==itu::EqKind::low_shelf?"low_shelf":"high_shelf")
            <<"\",\"frequency_hz\":"<<band.frequency_hz<<",\"gain_db\":"<<band.gain_db<<",\"q\":"<<band.q<<'}';
    }
    report<<']'
      <<",\n  \"processing_policy\":\""<<("shared_protection_v1")<<"\""
      <<",\n  \"stable_echo_coherence\":"<<(c.stable_echo_coherence?"true":"false")
      <<",\n  \"coherence_dtd_enabled\":"<<(c.coherence_dtd?"true":"false")
      <<",\n  \"low_cut_hz\":"<<(c.linear_phase_low_cut?c.low_cut_hz:0.f)
      <<",\n  \"raw_nearend_protection\":"<<(c.raw_nearend_protection?"true":"false")
      <<",\n  \"processing_profile\":\""<<profile<<"\""
      <<",\n  \"continuous_vehicle_policy\":"<<(c.continuous_vehicle_policy?"true":"false")
      <<",\n  \"beamforming_enabled\":"<<(c.beamforming?"true":"false")
      <<",\n  \"output_channels\":1"
      <<",\n  \"speech_preserving_low_cut\":"<<(c.speech_preserving_low_cut?"true":"false")
      <<",\n  \"fullband_near_ns\":"<<(c.fullband_near_ns?"true":"false")
      <<",\n  \"fullband_min_highband_fraction\":";
    report<<c.fullband_min_highband_fraction;
    report
      <<",\n  \"fast_aec_adaptation\":"<<(c.fast_aec_adaptation?"true":"false")
      <<",\n  \"linear_speech_protection\":"<<(c.linear_speech_protection?"true":"false")
      <<",\n  \"multirender_nearend_policy\":"<<(c.multirender_nearend_policy?"true":"false")
      <<",\n  \"multirender_speech_protection\":"<<(c.multirender_speech_protection?"true":"false")
      <<",\n  \"multirender_innovation_threshold\":"<<c.multirender_innovation_threshold
      <<",\n  \"ecnr_coordinated_control\":"<<(c.ecnr_coordinated_control?"true":"false")
      <<",\n  \"ecnr_control_version\":"<<json_string(c.ecnr_v3_g1_control?"v3_g1_experimental":c.ecnr_v2_control?"v2_selected":c.ecnr_v2_double_evidence || c.ecnr_v2_fullband_gain_calibration?"v2_diagnostic":c.ecnr_coordinated_control?"v1":"legacy")
      <<",\n  \"ecnr_v2_control\":"<<(c.ecnr_v2_control?"true":"false")
      <<",\n  \"ecnr_v3_g1_control\":"<<(c.ecnr_v3_g1_control?"true":"false")
      <<",\n  \"ecnr_v3_g1_revision\":\"v2_unified_echo_strong_095\""
      <<",\n  \"ecnr_v3_g1_scope\":\"G1 v2: original v2 eligible path and active hold only; original strict confirmation with nonlinear VAD >=0.95 refreshes latch age <=100 frames; startup or confirmed echo continuous union 40 frames gates both original clears; any original strict resets union; fallback original priority/conditions\""
      <<",\n  \"ecnr_v2_evidence_scope\":\"trusted mature mono render; delay-sum arrays only; coherence baseline < 0.15; stable valid AEC3 and acoustic delay <= 100 ms, 20 delay frames within 4 ms; other paths retain v1\""
      <<",\n  \"ecnr_v2_double_evidence\":"<<(c.ecnr_v2_double_evidence?"true":"false")
      <<",\n  \"ecnr_v2_fullband_gain_calibration\":"<<(c.ecnr_v2_fullband_gain_calibration?"true":"false")
      <<",\n  \"ecnr_v2_nonlinear_vad_min\":"<<c.ecnr_v2_nonlinear_vad_min
      <<",\n  \"ecnr_v2_coherence_margin\":"<<c.ecnr_v2_coherence_margin
      <<",\n  \"diagnostic_stages\":"<<(diagnostic_stages?"true":"false")
      <<",\n  \"stage_rms_sample_rate\":"<<(core_bridge ? 48000 : c.sample_rate)
      <<",\n  \"profile_stages\":"<<(c.profile_stages?"true":"false")
      <<",\n  \"aec_backend\":\""<<(c.aec_backend==itu::AecBackend::speex?"speex_mdf":"webrtc_aec3")<<"\""
      <<",\n  \"aec_filter_blocks\":"<<c.aec_filter_blocks
      <<",\n  \"aec_tail_ms\":"<<(c.aec_filter_blocks*4)
      <<",\n  \"speex_echo_tail_ms\":"<<c.speex_echo_tail_ms
      <<",\n  \"speex_native_conditioning\":"<<(c.speex_native_conditioning?"true":"false")
      <<",\n  \"speex_post_context\":"<<(c.speex_post_context?"true":"false")
      <<",\n  \"post_aec_evidence_window\":"<<(c.post_aec_evidence_window?"true":"false")
      <<",\n  \"low_level_projection_gain\":"<<(c.low_level_projection_gain?"true":"false")
      <<",\n  \"stage4_noise_release\":"<<(c.stage4_noise_release?"true":"false")
      <<",\n  \"agc_fault\":\""<<(c.gain_fault==itu::GainFault::previous_frame?"previous_frame":
          c.gain_fault==itu::GainFault::inverted?"inverted":
          c.gain_fault==itu::GainFault::always_boost?"always_boost":"none")<<"\""
      <<",\n  \"speex_residual_suppression\":"<<(c.speex_residual_suppression?"true":"false")
      <<",\n  \"speex_coherence_residual_suppression\":"<<(c.speex_coherence_residual_suppression?"true":"false")
      <<",\n  \"speex_active_echo_suppression_db\":"<<c.speex_active_echo_suppression_db
      <<",\n  \"analyze_linear_noise\":"<<(c.analyze_linear_noise?"true":"false")
      <<",\n  \"aec_input_delay_samples\":"<<info.aec_input_delay_samples<<",\n  \"linear_erle_uses_conditioned_input\":"<<("true")
      <<",\n  \"aec_input_conditioning\":\""<<(!c.aec ? "AEC disabled" : c.aec_backend==itu::AecBackend::speex ?
           (c.speex_native_conditioning ?
            "native MDF DC notch; external capture/render and post-chain HPF skipped" :
            c.linear_phase_low_cut ?
            "shared capture/render low-cut before Speex MDF; standalone NS post conditioning; upstream DC notch retained" :
            "native MDF DC notch; external low-cut disabled") :
           c.linear_phase_low_cut ? "shared capture/render low-cut before AEC3" : "external low-cut disabled")<<"\""
      <<",\n  \"aec_output_delay_samples\":"<<info.aec_output_delay_samples
      <<",\n  \"input_samples_per_channel\":"<<n<<",\n  \"frame_size\":"<<h<<",\n  \"repair_stage\":"<<c.repair_stage<<",\n  \"agc_preobserve_ms\":"<<0
      <<",\n  \"aec_enabled\":"<<(c.aec?"true":"false")<<",\n  \"ns_enabled\":"<<(c.noise_suppression?"true":"false")
      <<",\n  \"agc2_enabled\":"<<(c.automatic_gain?"true":"false")<<",\n  \"hpf_stages\":0,\n  \"low_cut_delay_samples\":"<<(info.low_cut_delay_samples)<<",\n  \"ns_level\":"<<c.noise_level<<",\n  \"dsp_delay_samples\":"<<info.latency_samples
      <<",\n  \"agc2_adaptive_enabled\":"<<(c.automatic_gain && c.adaptive_gain?"true":"false")<<",\n  \"agc2_speech_gain_policy\":"<<(c.automatic_gain && c.speech_gain_policy?"true":"false")<<",\n  \"linear_aec_available\":"<<(all_linear?"true":"false")<<",\n  \"linear_aec_delay_samples\":"<<linear_delay<<",\n  \"output_alignment\":\"remove leading DSP delay; BF maximum is a compensation budget\",\n  \"input_rms_dbfs\":"<<db(input_p)<<",\n  \"output_rms_dbfs\":"<<db(output_p)
      <<",\n  \"linear_aec_sample_rate\":"<<info.linear_aec_sample_rate
      <<",\n  \"overall_level_reduction_db\":"<<db(input_p)-db(output_p)<<",\n  \"limited_samples_in_input_frames\":"<<limited<<",\n  \"adaptation_frames\":null,\n  \"double_talk_frames\":null,\n  \"process_p95_ms\":"<<times[static_cast<std::size_t>(std::ceil(.95*times.size()))-1]
      <<",\n  \"process_max_ms\":"<<times.back()<<",\n  \"process_rtf\":"<<total/(frames*10.)
      <<",\n  \"paced_10ms\":"<<(paced?"true":"false")
      <<",\n  \"process_frames_over_10ms\":"<<process_misses
      <<",\n  \"completion_deadline_misses\":";
    if(paced) report<<completion_misses; else report<<"null";
    report<<",\n  \"release_lateness_max_ms\":";
    if(paced) report<<release_lateness_max; else report<<"null";
    report<<",\n  \"stream_resets\":0"
      <<",\n  \"simulated_reference_clock\":"<<(simulate_clock?"true":"false")
      <<",\n  \"clock_correction_enabled\":"<<(simulate_clock && !disable_clock?"true":"false")
      <<",\n  \"clock_injected_ppm\":"<<clock_ppm
      <<",\n  \"clock_estimated_ppm\":"<<(clock?clock->info().drift_ppm:0)
      <<",\n  \"clock_locked\":"<<(clock && clock->info().locked?"true":"false")
      <<",\n  \"clock_underruns\":"<<(clock?clock->info().underruns:0)
      <<",\n  \"clock_overruns\":"<<(clock?clock->info().overruns:0)
      <<",\n  \"clock_queued_samples\":"<<(clock?clock->info().queued_samples:0)
      <<",\n  \"simulated_clock_epoch_resets\":"<<event_index
      <<",\n  \"timing_scope\":\"C++ TX process plus RX process or clock ASRC when enabled; paced deadline includes input copy; logging and WAV I/O outside callbacks\",\n  \"wrapper_memory_bytes\":"<<engine.working_memory_bytes()
      <<",\n  \"wrapper_memory_scope\":\"wrapper-owned buffers only; opaque WebRTC/Speex states excluded\""
      <<",\n  \"apm_memory_bytes\":null,\n  \"linear_erle_far_only_estimate_db\":";
    if(measure && all_linear) {
      auto conditioned_begin=(static_cast<std::size_t>(far_start*c.sample_rate)+info.aec_input_delay_samples)*c.microphones;
      auto conditioned_end=(static_cast<std::size_t>(far_end*c.sample_rate)+info.aec_input_delay_samples)*c.microphones;
      auto linear_begin=(static_cast<std::size_t>(far_start*16000)+linear_delay)*c.microphones;
      auto linear_end=(static_cast<std::size_t>(far_end*16000)+linear_delay)*c.microphones;
      conditioned_begin=std::min(conditioned_begin,conditioned.size());
      conditioned_end=std::max(conditioned_begin,std::min(conditioned_end,conditioned.size()));
      linear_begin=std::min(linear_begin,linear.size());
      linear_end=std::max(linear_begin,std::min(linear_end,linear.size()));
      double p=power(conditioned,conditioned_begin,conditioned_end),q=power(linear,linear_begin,linear_end);
      if(p>1e-12 && q>1e-12) report<<db(p)-db(q); else report<<"null";
    } else report<<"null";
    report<<",\n  \"erle_scope\":\"caller-labelled far-only, latency-aligned "
          <<(c.aec_backend==itu::AecBackend::speex ? "raw Speex MDF error before RES/dry blend/NS/AGC2; " :
             "WebRTC linear AEC output after optional pre-AEC BF and before nonlinear suppression/NS/AGC2; ")
          <<("conditioned input path")<<"; engineering estimate only\",\n  \"aec_tap_scope\":\""
          <<(c.aec && c.aec_backend==itu::AecBackend::speex ?
             "Speex MDF/optional RES after aligned near-speech protection; separate raw MDF error is exported as linear tap" :
             c.aec ? "AEC3 output includes internal residual suppression; not a linear ERLE tap" :
             "capture path with AEC disabled; conversion delay retained")
          <<"\",\n  \"dtd_status_available\":false,\n  \"noise_estimation\":\""
          <<(!c.noise_suppression?"disabled":c.noise_backend==itu::NoiseBackend::unified?"Causal unsupervised online fusion of four shared-spectrum NS experts; no labels, future PCM, or external VAD":c.noise_backend==itu::NoiseBackend::super_gaussian?"Super-Gaussian joint MAP H1 gain with IMCRA and shared Gaussian soft evidence; no external VAD":c.noise_backend==itu::NoiseBackend::omlsa_imcra?"Causal OM-LSA with two-stage IMCRA; continuous soft speech probability; no external VAD":c.noise_backend==itu::NoiseBackend::ulunas?"UL-UNAS recurrent spectral model; 16 kHz enhancement bandwidth":c.noise_backend==itu::NoiseBackend::rnnoise?"RNNoise recurrent model; 48 kHz internal":c.noise_backend==itu::NoiseBackend::speex?"SpeexDSP online noise estimator":"WebRTC online adaptive; no user noise-only learning phase")
          <<"\",\n  \"internal_sample_rate\":"<<(core_bridge || c.noise_backend==itu::NoiseBackend::rnnoise?48000:c.sample_rate)
          <<",\n  \"aec_internal_sample_rate\":"<<(!c.aec ? 0 : c.aec_backend==itu::AecBackend::speex ? c.sample_rate : c.sample_rate==24000 ? 48000 : c.sample_rate)
          <<",\n  \"ns_internal_sample_rate\":"<<(c.noise_backend==itu::NoiseBackend::ulunas?16000:core_bridge || c.noise_backend==itu::NoiseBackend::rnnoise?48000:c.sample_rate)
          <<",\n  \"rate_converter\":\""<<(core_bridge || (c.sample_rate!=16000 && c.noise_backend==itu::NoiseBackend::ulunas)?"SpeexDSP quality 8":c.sample_rate==24000 && (c.noise_backend==itu::NoiseBackend::omlsa_imcra || c.noise_backend==itu::NoiseBackend::super_gaussian || c.noise_backend==itu::NoiseBackend::unified) && c.automatic_gain?"AGC2 may convert internally; native statistical NS PCM path":"none")<<"\""
          <<",\n  \"ns_delay_samples\":"<<info.ns_delay_samples
          <<",\n  \"post_low_cut_delay_samples\":"<<info.post_low_cut_delay_samples
          <<",\n  \"dsp_delay_ms\":"<<1000.*info.latency_samples/c.sample_rate
          <<",\n  \"standard_compliance\":false\n}\n";
    if(!report || !csv) throw std::runtime_error("report write failed");
    std::cout<<"wrote "<<prefix<<".{after.wav,aec.wav,linear-aec.wav,bf.wav,frames.csv,json}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
