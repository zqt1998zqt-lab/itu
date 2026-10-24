// Standalone mono NS driver using the same C++ component as AudioEngine. Optional
// rumble filtering reports its declared delay; no AEC, BF, AGC or EQ. Shared optional
// causal protection is explicit and disabled in the raw baseline.
// Both backends receive identical PCM16-quantized input. 24 kHz shared mode
// uses identical quality-8 SRC; native24 measures Speex without that bridge.
#include "wav.hpp"
#include "itu/vehicle_ns.hpp"
#include "rtc_base/logging.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sys/resource.h>
#include <thread>
#include <vector>

int main(int argc,char** argv) {
  try {
    if(argc<19) throw std::runtime_error("usage: ns_vehicle_probe backend strength input.wav prefix mode dry_max snr_threshold noise_gain startup_ms policy mild_strength slew_ms release_ms quiet_floor_db floor_guard repair_mode rumble_hz rumble_kind [--pace-10ms] [--spectral-dry-guard=0..1] [--residual-max-db=0..24] [--residual-speech-guard=0..1] [--residual-gate-snr-db=-60..30] [--residual-gate-width-db=0..20] [--rumble-recovery=0..1] [--rumble-adaptive-hz=0..120] [--weak-speech-recovery=0..1] [--speech-band-recovery=0..1] [--noise-step-recovery=0..1] [--transient-dry-guard=0..1] [--transient-spectral-guard-db=0..24]");
    bool paced=false;float spectral_dry_guard=0;bool guard_seen=false;
    bool residual_seen=false,speech_guard_seen=false;
    bool residual_gate_seen=false,gate_width_seen=false;
    float residual_max_db=0,residual_speech_guard=.75f;
    float residual_gate_snr_db=-60,residual_gate_width_db=3;
    float rumble_recovery=0;bool rumble_recovery_seen=false;
    float rumble_adaptive_hz=0;bool rumble_adaptive_hz_seen=false;
    float weak_speech_recovery=0;bool weak_speech_recovery_seen=false;
    float speech_band_recovery=0;bool speech_band_recovery_seen=false;
    float noise_step_recovery=0;bool noise_step_recovery_seen=false;
    float transient_dry_guard=0;bool transient_dry_guard_seen=false;
    float transient_spectral_guard_db=0;
    bool transient_spectral_guard_seen=false;
    for(int i=19;i<argc;++i) {
      const std::string option=argv[i];
      if(option=="--pace-10ms" && !paced) paced=true;
      else if(option.rfind("--spectral-dry-guard=",0)==0 && !guard_seen) {
        spectral_dry_guard=std::stof(option.substr(21));guard_seen=true;
      } else if(option.rfind("--residual-max-db=",0)==0 && !residual_seen) {
        residual_max_db=std::stof(option.substr(18));residual_seen=true;
      } else if(option.rfind("--residual-speech-guard=",0)==0 && !speech_guard_seen) {
        residual_speech_guard=std::stof(option.substr(24));speech_guard_seen=true;
      } else if(option.rfind("--residual-gate-snr-db=",0)==0 && !residual_gate_seen) {
        residual_gate_snr_db=std::stof(option.substr(23));residual_gate_seen=true;
      } else if(option.rfind("--residual-gate-width-db=",0)==0 && !gate_width_seen) {
        residual_gate_width_db=std::stof(option.substr(25));gate_width_seen=true;
      } else if(option.rfind("--rumble-recovery=",0)==0 && !rumble_recovery_seen) {
        rumble_recovery=std::stof(option.substr(18));rumble_recovery_seen=true;
      } else if(option.rfind("--rumble-adaptive-hz=",0)==0 && !rumble_adaptive_hz_seen) {
        rumble_adaptive_hz=std::stof(option.substr(21));rumble_adaptive_hz_seen=true;
      } else if(option.rfind("--weak-speech-recovery=",0)==0 && !weak_speech_recovery_seen) {
        weak_speech_recovery=std::stof(option.substr(23));weak_speech_recovery_seen=true;
      } else if(option.rfind("--speech-band-recovery=",0)==0 && !speech_band_recovery_seen) {
        speech_band_recovery=std::stof(option.substr(23));speech_band_recovery_seen=true;
      } else if(option.rfind("--noise-step-recovery=",0)==0 && !noise_step_recovery_seen) {
        noise_step_recovery=std::stof(option.substr(22));noise_step_recovery_seen=true;
      } else if(option.rfind("--transient-dry-guard=",0)==0 && !transient_dry_guard_seen) {
        transient_dry_guard=std::stof(option.substr(22));transient_dry_guard_seen=true;
      } else if(option.rfind("--transient-spectral-guard-db=",0)==0 &&
                !transient_spectral_guard_seen) {
        transient_spectral_guard_db=std::stof(option.substr(30));
        transient_spectral_guard_seen=true;
      } else throw std::runtime_error("invalid or duplicate stream option");
    }
    const std::string backend=argv[1],prefix=argv[4],mode=argv[5];
    const int strength=std::stoi(argv[2]);
    const float dry_max=std::stof(argv[6]),snr_threshold=std::stof(argv[7]),noise_gain=std::stof(argv[8]);
    const int startup_ms=std::stoi(argv[9]);
    const std::string policy=argv[10];
    const int mild_strength=std::stoi(argv[11]);
    const float slew_ms=std::stof(argv[12]);
    const float release_ms=std::stof(argv[13]), quiet_floor_db=std::stof(argv[14]);
    if(!std::isfinite(release_ms+quiet_floor_db) || release_ms<0 || release_ms>500 || quiet_floor_db> -50 || quiet_floor_db< -300)
      throw std::runtime_error("invalid repair observer configuration");
    const int floor_guard=std::stoi(argv[15]);
    if(floor_guard<0 || floor_guard>2) throw std::runtime_error("invalid floor guard");
    const unsigned repair_mode=std::stoul(argv[16]);
    if(repair_mode>4) throw std::runtime_error("invalid vehicle repair mode");
    const float rumble_hz=std::stof(argv[17]);
    if(!std::isfinite(rumble_hz) || rumble_hz<0 || rumble_hz>120) throw std::runtime_error("invalid rumble corner");
    const std::string rumble_kind=argv[18];
    if(rumble_kind!="iir" && rumble_kind!="linear" && rumble_kind!="adaptive") throw std::runtime_error("invalid rumble filter kind");
    const bool linear_rumble=rumble_kind!="iir";
    const bool direct_fullband=repair_mode>=2;
    const bool quiet_enabled=quiet_floor_db> -200 || repair_mode>=3;
    if((policy!="legacy" && policy!="aligned" && policy!="parallel" && policy!="dynamic" && policy!="steady_dynamic" && policy!="fixedblend" && policy!="parallel_start") ||
       !std::isfinite(slew_ms) || slew_ms<0 || slew_ms>500 || ((policy=="dynamic" || policy=="steady_dynamic") && backend!="speex"))
      throw std::runtime_error("invalid control policy");
    if((backend=="webrtc" && (mild_strength<0 || mild_strength>3)) ||
       (backend=="speex" && (mild_strength>0 || mild_strength< -40))) throw std::runtime_error("invalid mild strength");
    if(!std::isfinite(dry_max+snr_threshold+noise_gain) || dry_max<0 || dry_max>1 || noise_gain<.2f || noise_gain>1 || snr_threshold<0 || snr_threshold>30 || startup_ms<0 || startup_ms>500)
      throw std::runtime_error("invalid causal NS controller parameters");
    auto wav=itu::read_wav(argv[3]);
    if(wav.samples.empty() || wav.channels!=1 || (wav.sample_rate!=16000 && wav.sample_rate!=24000 && wav.sample_rate!=48000) ||
       (backend!="webrtc" && backend!="speex" && backend!="bypass" && backend!="filterbank" && backend!="omlsa_imcra" && backend!="super_gaussian") ||
       (mode!="shared" && mode!="native24") ||
       (mode=="native24" && ((backend!="speex" && backend!="omlsa_imcra" && backend!="super_gaussian") || wav.sample_rate!=24000)))
      throw std::runtime_error("invalid rate, channels, backend or mode");
    if(((backend=="webrtc" || backend=="omlsa_imcra" || backend=="super_gaussian") && (strength<0 || strength>3)) ||
       (backend=="speex" && (strength>0 || strength< -40))) throw std::runtime_error("invalid strength");
    webrtc::LogMessage::LogToDebug(webrtc::LS_NONE);
    webrtc::LogMessage::SetLogToStderr(false);
    const unsigned rate=wav.sample_rate,hop=rate/100;
    itu::VehicleNsConfig options;
    options.backend=backend=="super_gaussian" ? itu::VehicleNsBackend::super_gaussian : backend=="omlsa_imcra" ? itu::VehicleNsBackend::omlsa_imcra : backend=="webrtc" ? itu::VehicleNsBackend::webrtc : backend=="speex" ? itu::VehicleNsBackend::speex :
        backend=="filterbank" ? itu::VehicleNsBackend::filterbank : itu::VehicleNsBackend::bypass;
    options.strength=strength;options.mild_strength=mild_strength;
    options.dry_max=dry_max;options.snr_threshold_db=snr_threshold;options.noise_gain=noise_gain;
    options.startup_ms=startup_ms;options.policy=policy;options.slew_ms=slew_ms;
    options.release_ms=release_ms;options.quiet_floor_db=quiet_floor_db;options.floor_guard=floor_guard;
    options.repair_mode=repair_mode;options.rumble_hz=rumble_hz;options.linear_rumble=linear_rumble;
    options.native_24=mode=="native24";
    options.adaptive_rumble=rumble_kind=="adaptive";
    options.spectral_dry_guard=spectral_dry_guard;
    options.residual_max_db=residual_max_db;
    options.residual_speech_guard=residual_speech_guard;
    options.residual_gate_snr_db=residual_gate_snr_db;
    options.residual_gate_width_db=residual_gate_width_db;
    options.rumble_recovery=rumble_recovery;
    options.rumble_adaptive_hz=rumble_adaptive_hz;
    options.weak_speech_recovery=weak_speech_recovery;
    options.speech_band_recovery=speech_band_recovery;
    options.noise_step_recovery=noise_step_recovery;
    options.transient_dry_guard=transient_dry_guard;
    options.transient_spectral_guard_db=transient_spectral_guard_db;
    itu::VehicleNsProcessor processor(rate,options);
    const unsigned core_rate=processor.core_rate(),core_hop=core_rate/100;
    const unsigned core_delay=processor.core_delay_samples(),delay=processor.delay_samples();
    const double src_delay=processor.src_delay_samples();
    const bool policy_enabled=dry_max>0 || noise_gain<1 || weak_speech_recovery>0 ||
        speech_band_recovery>0 || noise_step_recovery>0 || transient_dry_guard>0 ||
        transient_spectral_guard_db>0;
    const std::size_t samples=wav.samples.size(),frames=(samples+hop-1)/hop;
    const std::size_t flushed_frames=(samples+delay+hop-1)/hop;
    std::vector<float> input(hop),core(core_hop),output(hop),rendered(flushed_frames*hop);
    std::vector<double> timings;
    timings.reserve(frames);
    std::ofstream csv(prefix+".csv");
    csv<<"frame,time_s,process_ms,confidence,snr_db,input_mix_target,input_gain_target,effective_suppression_db,quiet_mix_target,estimated_noise_floor_db,rumble_fraction,rumble_mix_target,rumble_mix_applied,rumble_corner_hz,rumble_low_ratio_db,steady_mix_target,omlsa_probability,omlsa_gain,omlsa_noise_power,imcra_excluded_bins,imcra_noise_updates,imcra_minima_rotations,spectral_guard_applied,transient_noise_evidence,transient_guard_applied,transient_spectral_guard_applied,transient_periodicity_100ms,transient_periodicity_500ms,residual_speech_probability,residual_attenuation_db,noise_step_probability,noise_step_updates,speech_band_frames,speech_band_floor,speech_band_gain_delta,speech_band_evidence,speech_band_mid_probability,speech_band_anchor_probability,speech_band_anchor_gain\n";
    double first_ms=0,peak=0;
    std::size_t clipped=0,deadline_misses=0;
    const auto stream_start=std::chrono::steady_clock::now();
    for(std::size_t frame=0;frame<flushed_frames;++frame) {
      if(paced && frame<frames) std::this_thread::sleep_until(stream_start+std::chrono::milliseconds(10*frame));
      for(unsigned i=0;i<hop;++i) {
        const std::size_t pos=frame*hop+i;
        const float x=pos<samples ? wav.samples[pos] : 0.f;
        input[i]=std::lround(std::clamp(x*32768.f,-32768.f,32767.f))/32768.f;
      }
      const auto start=std::chrono::steady_clock::now();
      if(!processor.process(input.data(),hop)) throw std::runtime_error("vehicle NS processing failed");
      std::copy(input.begin(),input.end(),output.begin());
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(paced && frame<frames && std::chrono::steady_clock::now()>stream_start+std::chrono::milliseconds(10*(frame+1))) ++deadline_misses;
      if(frame==0) first_ms=ms;
      const auto stats=processor.statistics();
      if(frame<frames) {timings.push_back(ms);csv<<frame<<','<<frame*.01<<','<<ms<<','<<stats.confidence<<','<<stats.snr_db<<','<<stats.dry_mix_target<<','<<stats.gain_target<<','<<stats.suppression_db<<','<<stats.quiet_mix_target<<','<<stats.noise_floor_db<<','<<stats.rumble_fraction<<','<<stats.rumble_mix_target<<','<<stats.rumble_mix_applied<<','<<stats.rumble_corner_hz<<','<<stats.rumble_low_ratio_db<<','<<stats.steady_mix_target<<','<<stats.omlsa_probability<<','<<stats.omlsa_gain<<','<<stats.omlsa_noise_power<<','<<stats.imcra_excluded_bins<<','<<stats.imcra_noise_updates<<','<<stats.imcra_minima_rotations<<','<<stats.spectral_guard_applied<<','<<stats.transient_noise_evidence<<','<<stats.transient_guard_applied<<','<<stats.transient_spectral_guard_applied<<','<<stats.transient_periodicity_100ms<<','<<stats.transient_periodicity_500ms<<','<<stats.residual_speech_probability<<','<<stats.residual_attenuation_db<<','<<stats.noise_step_probability<<','<<stats.noise_step_updates<<','<<stats.speech_band_frames<<','<<stats.speech_band_floor<<','<<stats.speech_band_gain_delta<<','<<stats.speech_band_evidence<<','<<stats.speech_band_mid_probability<<','<<stats.speech_band_anchor_probability<<','<<stats.speech_band_anchor_gain<<'\n';}
      for(unsigned i=0;i<hop;++i) {
        if(!std::isfinite(output[i])) throw std::runtime_error("nonfinite NS output");
        rendered[frame*hop+i]=output[i];
        peak=std::max(peak,double(std::abs(output[i])));
        clipped+=std::abs(output[i])>=.999f;
      }
    }
    rendered.resize(samples+delay);
    itu::write_wav(prefix+".after.wav",rate,1,rendered);
    auto sorted=timings;
    std::sort(sorted.begin(),sorted.end());
    auto percentile=[&](double p){return sorted[std::min(sorted.size()-1,std::size_t(p*(sorted.size()-1)))];};
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    const auto stats=processor.statistics();
    std::ofstream report(prefix+".json");
    report<<"{\"rumble_kind\":\""<<rumble_kind<<"\",\"rumble_delay_samples\":"<<processor.rumble_delay_samples()<<",\"rumble_hz\":"<<rumble_hz<<",\"vehicle_repair_mode\":"<<repair_mode<<",\"fullband_raw\":"<<(direct_fullband ? "true" : "false")<<",\"floor_guard\":"<<floor_guard<<",\"repair_release_ms\":"<<release_ms<<",\"quiet_floor_db\":"<<quiet_floor_db<<",\"backend\":\""<<backend<<"\",\"strength\":"<<strength
      <<",\"io_rate\":"<<rate<<",\"core_rate\":"<<core_rate
      <<",\"mode\":\""<<mode<<"\",\"delay_samples\":"<<delay
      <<",\"src_delay_samples\":"<<src_delay<<",\"core_ns_delay_samples\":"<<core_delay
      <<",\"samples\":"<<samples<<",\"frames\":"<<frames
      <<",\"first_process_ms\":"<<first_ms<<",\"p50_process_ms\":"<<percentile(.5)
      <<",\"p95_process_ms\":"<<percentile(.95)<<",\"p99_process_ms\":"<<percentile(.99)
      <<",\"max_process_ms\":"<<sorted.back()
      <<",\"mean_process_ms\":"<<std::accumulate(timings.begin(),timings.end(),0.)/timings.size()
      <<",\"over_10ms\":"<<std::count_if(timings.begin(),timings.end(),[](double t){return t>10.;})
      <<",\"peak_before_write\":"<<peak<<",\"clipped_samples_before_write\":"<<clipped
      <<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"common_pcm16_input\":true"
      <<",\"steady_suppression_db\":"<<options.steady_suppression_db
      <<",\"spectral_dry_guard\":"<<options.spectral_dry_guard
      <<",\"omlsa_probability\":"<<stats.omlsa_probability<<",\"omlsa_gain\":"<<stats.omlsa_gain
      <<",\"omlsa_bootstrap_seed\":"<<stats.omlsa_bootstrap_seed<<",\"omlsa_noise_power\":"<<stats.omlsa_noise_power
      <<",\"imcra_excluded_bins\":"<<stats.imcra_excluded_bins<<",\"imcra_noise_updates\":"<<stats.imcra_noise_updates
      <<",\"imcra_minima_rotations\":"<<stats.imcra_minima_rotations
      <<",\"residual_max_db\":"<<options.residual_max_db
      <<",\"residual_speech_guard\":"<<options.residual_speech_guard
      <<",\"residual_gate_snr_db\":"<<options.residual_gate_snr_db
      <<",\"residual_gate_width_db\":"<<options.residual_gate_width_db
      <<",\"rumble_recovery\":"<<options.rumble_recovery
      <<",\"weak_speech_recovery\":"<<options.weak_speech_recovery
      <<",\"speech_band_recovery\":"<<options.speech_band_recovery
      <<",\"noise_step_recovery\":"<<options.noise_step_recovery
      <<",\"transient_dry_guard\":"<<options.transient_dry_guard
      <<",\"transient_spectral_guard_db\":"<<options.transient_spectral_guard_db
      <<",\"noise_step_probability\":"<<stats.noise_step_probability
      <<",\"transient_noise_evidence\":"<<stats.transient_noise_evidence
      <<",\"transient_guard_applied\":"<<stats.transient_guard_applied
      <<",\"transient_spectral_guard_applied\":"<<stats.transient_spectral_guard_applied
      <<",\"transient_periodicity_100ms\":"<<stats.transient_periodicity_100ms
      <<",\"transient_periodicity_500ms\":"<<stats.transient_periodicity_500ms
      <<",\"noise_step_updates\":"<<stats.noise_step_updates
      <<",\"speech_band_frames\":"<<stats.speech_band_frames
      <<",\"speech_band_floor\":"<<stats.speech_band_floor
      <<",\"speech_band_gain_delta\":"<<stats.speech_band_gain_delta
      <<",\"speech_band_evidence\":"<<stats.speech_band_evidence
      <<",\"speech_band_mid_probability\":"<<stats.speech_band_mid_probability
      <<",\"speech_band_anchor_probability\":"<<stats.speech_band_anchor_probability
      <<",\"speech_band_anchor_gain\":"<<stats.speech_band_anchor_gain
      <<",\"residual_speech_probability\":"<<stats.residual_speech_probability
      <<",\"residual_attenuation_db\":"<<stats.residual_attenuation_db
      <<",\"paced\":"<<(paced ? "true" : "false")<<",\"paced_deadline_misses\":"<<deadline_misses
      <<",\"aec\":false,\"agc\":false,\"hpf\":"<<(rumble_hz>0 ? "true" : "false")<<",\"speech_bypass\":"<<(policy_enabled ? "true" : "false")
      <<",\"policy_enabled\":"<<(policy_enabled ? "true" : "false")<<",\"dry_max\":"<<dry_max<<",\"snr_threshold_db\":"<<snr_threshold<<",\"noise_gain\":"<<noise_gain<<",\"startup_ms\":"<<startup_ms<<",\"control_policy\":\""<<policy<<"\",\"mild_strength\":"<<mild_strength<<",\"slew_ms\":"<<slew_ms<<",\"suppression_updates\":"<<stats.suppression_updates<<",\"evidence_scope\":\"current and previous input PCM only; no labels, clean target, external VAD or oracle\"}\n";
    if(!report || !csv) throw std::runtime_error("report write failed");
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n'; return 1;
  }
}
