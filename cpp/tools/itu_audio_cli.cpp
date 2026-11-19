#include "itu/audio_engine.hpp"
#include "itu/reference_clock.hpp"
#include "linear_phase_eq.hpp"
#include "stream_ns_pipe.hpp"
#include "wav.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
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
int parse_stream_fd(const std::string& text,const char* option) {
  std::size_t used=0;
  const long long value=std::stoll(text,&used);
  if(used!=text.size() || value<3 || value>std::numeric_limits<int>::max())
    throw std::runtime_error(std::string(option)+" must be an integer fd >= 3");
  return static_cast<int>(value);
}
// Diagnostic stages retain the actual floating point amplitude, including
// low-level filter tails and peaks above PCM16 full scale.
void write_float_wav(const std::string& path,unsigned rate,const std::vector<float>& values,
                     unsigned channels=1) {
  if(!channels || channels>8 || values.size()%channels)
    throw std::runtime_error("diagnostic WAV has an invalid channel shape");
  if(values.size()>(std::numeric_limits<std::uint32_t>::max()-36)/4)
    throw std::runtime_error("diagnostic WAV exceeds RIFF size");
  std::ofstream file(path,std::ios::binary); if(!file) throw std::runtime_error("cannot write diagnostic WAV");
  auto u16=[&](std::uint16_t v){char b[2]={char(v),char(v>>8)};file.write(b,2);};
  auto u32=[&](std::uint32_t v){char b[4]={char(v),char(v>>8),char(v>>16),char(v>>24)};file.write(b,4);};
  file.write("RIFF",4);u32(36+std::uint32_t(values.size()*4));file.write("WAVEfmt ",8);
  u32(16);u16(3);u16(channels);u32(rate);u32(rate*channels*4);u16(channels*4);u16(32);
  file.write("data",4);u32(std::uint32_t(values.size()*4));
  for(float value:values) {std::uint32_t bits;std::memcpy(&bits,&value,4);u32(bits);}
  if(!file) throw std::runtime_error("diagnostic WAV write failed");
}
itu::Wav read_float32_wav(const std::string& path) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file) throw std::runtime_error("cannot open FLOAT32 WAV: "+path);
  const auto length=file.tellg();
  if(length<12 || length>512*1024*1024)
    throw std::runtime_error("FLOAT32 WAV size must be 12 bytes..512 MiB");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
  if(!file || std::memcmp(bytes.data(),"RIFF",4) || std::memcmp(bytes.data()+8,"WAVE",4))
    throw std::runtime_error("expected little-endian RIFF FLOAT32 WAVE");
  auto u16=[](const unsigned char* b){return unsigned(b[0])|unsigned(b[1])<<8;};
  auto u32=[](const unsigned char* b){return std::uint32_t(b[0])|std::uint32_t(b[1])<<8|
      std::uint32_t(b[2])<<16|std::uint32_t(b[3])<<24;};
  const std::size_t end=std::size_t(u32(bytes.data()+4))+8;
  if(end>bytes.size() || end<12) throw std::runtime_error("truncated FLOAT32 WAV RIFF");
  itu::Wav wav; unsigned format=0,bits=0,block=0; std::size_t data_at=0,data_size=0;
  bool has_fmt=false;
  for(std::size_t p=12;p+8<=end;) {
    const std::size_t size=u32(bytes.data()+p+4),start=p+8;
    if(size>end-start) throw std::runtime_error("truncated FLOAT32 WAV chunk");
    if(!std::memcmp(bytes.data()+p,"fmt ",4)) {
      if(size<16) throw std::runtime_error("short FLOAT32 WAV fmt chunk");
      format=u16(bytes.data()+start);wav.channels=u16(bytes.data()+start+2);
      wav.sample_rate=u32(bytes.data()+start+4);block=u16(bytes.data()+start+12);
      bits=u16(bytes.data()+start+14);has_fmt=true;
    } else if(!std::memcmp(bytes.data()+p,"data",4)) {
      if(data_at) throw std::runtime_error("multiple FLOAT32 WAV data chunks unsupported");
      data_at=start;data_size=size;
    }
    p=start+size+(size&1);
  }
  if(!has_fmt || !data_at || format!=3 || wav.channels!=1 ||
     bits!=32 || block!=4 || data_size<4 || data_size%4)
    throw std::runtime_error("--post-ns-pcm must be non-empty mono IEEE FLOAT32 WAV");
  wav.samples.resize(data_size/4);
  for(std::size_t i=0;i<wav.samples.size();++i) {
    const auto raw=u32(bytes.data()+data_at+i*4);
    std::memcpy(&wav.samples[i],&raw,4);
    if(!std::isfinite(wav.samples[i]))
      throw std::runtime_error("--post-ns-pcm contains non-finite FLOAT32 samples");
  }
  return wav;
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
bool eq_config_enabled(const itu::EqConfig& config) {
  if(config.preamp_db!=0) return true;
  for(unsigned i=0;i<config.bands;++i) if(config.band[i].gain_db!=0) return true;
  return false;
}
std::string absolute_output_path(const std::string& path) {
  return std::filesystem::absolute(path).lexically_normal().string();
}
void validate_vcp_c64_profile(const itu::Config& c) {
  if(c.sample_rate!=16000 || c.frame_size!=160 || c.microphones!=1 ||
     c.beamforming || c.aec)
    throw std::runtime_error("vcp_c64 requires native 16 kHz mono input with AEC off");
  if(!c.noise_suppression || c.noise_backend!=itu::NoiseBackend::omlsa_imcra ||
     c.noise_level!=0 || !c.vehicle_ns_control ||
     c.vehicle_ns.backend!=itu::VehicleNsBackend::omlsa_imcra ||
     c.vehicle_ns.strength!=0)
    throw std::runtime_error("vcp_c64 requires enabled strength-0 native OM-LSA NS");
  if(c.vehicle_ns.policy!="aligned" && c.vehicle_ns.policy!="legacy")
    throw std::runtime_error("vcp_c64 policy must be aligned or legacy");
  if(c.vehicle_ns.unified_raw_budget || c.vehicle_ns.dry_max!=0.f ||
     c.vehicle_ns.startup_ms!=0 || c.vehicle_ns.quiet_floor_db!=-300.f ||
     c.vehicle_ns.noise_gain!=1.f)
    throw std::runtime_error("vcp_c64 requires the no-raw-mix vehicle policy");
  if(c.vehicle_ns.repair_mode!=0 || c.vehicle_ns.rumble_hz!=0.f ||
     c.vehicle_ns.adaptive_rumble || c.vehicle_ns.rumble_recovery!=0.f ||
     c.vehicle_ns.rumble_adaptive_hz!=0.f ||
     c.vehicle_ns.spectral_dry_guard!=0.f)
    throw std::runtime_error("vcp_c64 requires legacy repair/rumble/spectral guards off");
  if(c.vehicle_ns.weak_speech_recovery!=0.f ||
     c.vehicle_ns.speech_band_recovery!=0.f ||
     c.vehicle_ns.noise_step_recovery!=0.f ||
     c.vehicle_ns.transient_dry_guard!=0.f ||
     c.vehicle_ns.transient_spectral_guard_db!=0.f ||
     c.vehicle_ns.residual_max_db!=0.f)
    throw std::runtime_error("vcp_c64 recovery and legacy suppression stages must be off");
  if(c.mild_near_only_ns || !c.vehicle_ns.student_model_path.empty())
    throw std::runtime_error("vcp_c64 forbids legacy NS routing and the old student model");
  const auto& tuning=c.vehicle_ns.omlsa_tuning;
  if(tuning.noise_update_scale!=1.f || tuning.speech_recovery!=0.f ||
     tuning.release_ms!=60.f || tuning.speech_floor_db!=-40.f ||
     tuning.adaptive_noise_tracking || tuning.high_snr_transparency ||
     tuning.low_band_protection)
    throw std::runtime_error("vcp_c64 requires the original unadaptive OM-LSA tuning");
}
void validate_vcp_hybrid_v2_profile(const itu::Config& c) {
  validate_vcp_c64_profile(c);
  const auto& v=c.vehicle_ns;
  if(!v.student_model_path.empty() || !v.residual_model_path.empty())
    throw std::runtime_error(
        "vcp_hybrid_v2 forbids the legacy student and residual model paths");
  if(v.student_spp_mix!=.5f || v.student_snr_mix!=.5f ||
     v.student_speech_floor_db!=-3.f || v.residual_model_mix!=1.f ||
     v.residual_model_floor_db!=-9.f)
    throw std::runtime_error(
        "vcp_hybrid_v2 forbids overriding legacy model controls");
  if(!std::isfinite(v.hybrid_v2_mix+v.hybrid_v2_floor_db+
                    v.hybrid_v2_pitch_strength) ||
     v.hybrid_v2_mix<0.f || v.hybrid_v2_mix>1.f ||
     v.hybrid_v2_floor_db< -40.f || v.hybrid_v2_floor_db>0.f ||
     v.hybrid_v2_pitch_strength<0.f || v.hybrid_v2_pitch_strength>1.f)
    throw std::runtime_error("vcp_hybrid_v2 mix/floor/pitch strength out of range");
}
void fit_stage_extent(std::vector<float>& values,unsigned channels,
                      std::size_t samples_per_channel,const std::string& id) {
  if(!channels || channels>8 || values.size()%channels)
    throw std::runtime_error(id+" has an invalid channel shape");
  const std::size_t available=values.size()/channels;
  if(available<samples_per_channel)
    throw std::runtime_error(id+" diagnostic extent is shorter than the aligned output");
  if(samples_per_channel>std::numeric_limits<std::size_t>::max()/channels)
    throw std::runtime_error(id+" diagnostic extent overflows");
  values.resize(samples_per_channel*channels);
}
struct PipelineStageExport {
  std::string id,direction,file,measurement_domain,channel_policy;
  unsigned order=0,sample_rate=0,channels=0,capture_channels=0;
  std::size_t samples_per_channel=0;
  unsigned fixed_delay_samples=0;
  bool enabled=false,capture_available=false,diagnostic=false;
  const std::vector<float>* samples=nullptr;
};
struct LegacyStageExport {
  std::string name,file,measurement_domain;
  unsigned sample_rate=0,channels=0,fixed_delay_samples=0;
  std::size_t samples_per_channel=0;
  bool nr_backend_active=false;
  const std::vector<float>* samples=nullptr;
};
void write_stage_metadata(const std::string& prefix,unsigned input_sample_rate,
                          std::size_t input_samples_per_channel,bool rx_requested,
                          const std::vector<PipelineStageExport>& pipeline_stages,
                          const std::vector<LegacyStageExport>& legacy_stages) {
  for(const auto& stage:pipeline_stages) {
    if(!stage.samples || !stage.sample_rate || !stage.channels || !stage.capture_channels ||
       stage.capture_channels>stage.channels ||
       stage.samples_per_channel>std::numeric_limits<std::size_t>::max()/stage.channels ||
       stage.samples->size()!=stage.samples_per_channel*stage.channels)
      throw std::runtime_error(stage.id+" diagnostic WAV shape mismatch");
    if(!std::filesystem::path(stage.file).is_absolute())
      throw std::runtime_error(stage.id+" diagnostic WAV path is not absolute");
    write_float_wav(stage.file,stage.sample_rate,*stage.samples,stage.channels);
  }
  for(const auto& stage:legacy_stages) {
    if(!stage.samples || !stage.sample_rate || !stage.channels ||
       stage.samples->size()!=stage.samples_per_channel*stage.channels)
      throw std::runtime_error(stage.name+" legacy diagnostic WAV shape mismatch");
    write_float_wav(stage.file,stage.sample_rate,*stage.samples,stage.channels);
  }
  std::ofstream metadata(prefix+".stages.json");
  if(!metadata) throw std::runtime_error("cannot write stage metadata");
  metadata<<std::setprecision(12)
      <<"{\n  \"schema_version\":2,\n  \"sample_format\":\"IEEE_FLOAT32\",\n"
      <<"  \"alignment\":\"sample zero input time = (trimmed_leading_samples - fixed_delay_samples) / sample_rate; no fitted lag\",\n"
      <<"  \"input_sample_rate\":"<<input_sample_rate
      <<",\n  \"input_samples_per_channel\":"<<input_samples_per_channel
      <<",\n  \"rx_capture_status\":"<<json_string(rx_requested?"captured":"not_requested")
      <<",\n  \"stages\":[";
  for(std::size_t i=0;i<legacy_stages.size();++i) {
    const auto& stage=legacy_stages[i];
    if(i) metadata<<",\n";
    metadata<<"    {\"name\":"<<json_string(stage.name)<<",\"file\":"<<json_string(stage.file)
        <<",\"sample_rate\":"<<stage.sample_rate<<",\"channels\":"<<stage.channels
        <<",\"samples_per_channel\":"<<stage.samples_per_channel
        <<",\"fixed_delay_samples\":"<<stage.fixed_delay_samples
        <<",\"trimmed_leading_samples\":0"
        <<",\"timeline_offset_ms\":"<<(-1000.*stage.fixed_delay_samples/stage.sample_rate)
        <<",\"measurement_domain\":"<<json_string(stage.measurement_domain)
        <<",\"nr_backend_active\":"<<(stage.nr_backend_active?"true":"false")<<'}';
  }
  metadata<<"\n  ],\n  \"pipeline_stages\":[";
  for(std::size_t i=0;i<pipeline_stages.size();++i) {
    const auto& stage=pipeline_stages[i];
    if(i) metadata<<",\n";
    metadata<<"    {\"id\":"<<json_string(stage.id)
        <<",\"direction\":"<<json_string(stage.direction)
        <<",\"order\":"<<stage.order
        <<",\"file\":"<<json_string(stage.file)
        <<",\"sample_rate\":"<<stage.sample_rate
        <<",\"channels\":"<<stage.channels
        <<",\"capture_channels\":"<<stage.capture_channels
        <<",\"samples_per_channel\":"<<stage.samples_per_channel
        <<",\"fixed_delay_samples\":"<<stage.fixed_delay_samples
        <<",\"enabled\":"<<(stage.enabled?"true":"false")
        <<",\"measurement_domain\":"<<json_string(stage.measurement_domain)
        <<",\"channel_policy\":"<<json_string(stage.channel_policy)
        <<",\"capture_available\":"<<(stage.capture_available?"true":"false");
    if(stage.diagnostic) metadata<<",\"diagnostic\":true";
    metadata<<'}';
  }
  metadata<<"\n  ]\n}\n";
  if(!metadata) throw std::runtime_error("diagnostic metadata write failed");
}
}
int main(int argc,char** argv) {
  try {
    itu::Config c; std::string mic_path,echo_path,prefix,profile="general"; double far_start=-1,far_end=-1;
    std::string rx_in_path,rx_spk_path,rx_ref_path,complex_pf_cpu_pcm_path,post_ns_pcm_path;
    bool paced=false,explicit_channel_zero=false,simulate_clock=false,disable_clock=false,diagnostic_stages=false;
    bool aec_capabilities=false,complex_pf_export_inputs=false;
    bool hybrid_v2_legacy_flag=false;
    std::optional<unsigned> localvqe_reference_channel;
    std::optional<int> ns_stream_input_fd,ns_stream_output_fd;
    unsigned ns_stream_delay_samples=640;
    bool ns_stream_delay_explicit=false;
    double clock_ppm=0; std::string clock_events_path;
    unsigned requested_microphones=0,beam_delay_count=0;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];
      if(profile=="vcp_hybrid_v2" &&
         (a=="--no-ns" || a=="--ns-backend" || a=="--ns-level" ||
          a=="--vehicle-rumble-hz" || a=="--mild-near-only-ns" ||
          a=="--no-mild-near-only-ns" ||
          a.rfind("--vehicle-ns-",0)==0 || a.rfind("--omlsa-",0)==0 ||
          a.rfind("--hybrid-ns-",0)==0 || a.rfind("--residual-ns-",0)==0))
        hybrid_v2_legacy_flag=true;
      if(a=="--no-aec") c.aec=false;
      else if(a=="--aec-capabilities") aec_capabilities=true;
      else if(a=="--complex-pf-export-inputs") complex_pf_export_inputs=true;
      else if(a=="--no-bf") explicit_channel_zero=true;
      else if(a=="--no-ns") c.noise_suppression=false;
      else if(a=="--no-agc") c.automatic_gain=false;
      else if(a=="--bounded-speech-gain") c.bounded_speech_gain=true;
      else if(a=="--no-bounded-speech-gain") c.bounded_speech_gain=false;
      else if(a=="--mild-near-only-ns") c.mild_near_only_ns=true;
      else if(a=="--no-mild-near-only-ns") c.mild_near_only_ns=false;
      else if(a=="--omlsa-adaptive-noise-tracking") c.vehicle_ns.omlsa_tuning.adaptive_noise_tracking=true;
      else if(a=="--no-omlsa-adaptive-noise-tracking") c.vehicle_ns.omlsa_tuning.adaptive_noise_tracking=false;
      else if(a=="--omlsa-high-snr-transparency") c.vehicle_ns.omlsa_tuning.high_snr_transparency=true;
      else if(a=="--no-omlsa-high-snr-transparency") c.vehicle_ns.omlsa_tuning.high_snr_transparency=false;
      else if(a=="--omlsa-low-band-protection") c.vehicle_ns.omlsa_tuning.low_band_protection=true;
      else if(a=="--no-omlsa-low-band-protection") c.vehicle_ns.omlsa_tuning.low_band_protection=false;
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
          else if(v=="vcp_mild") c=itu::barrot_vcp_mild_config();
          else if(v=="vcp_adaptive") c=itu::barrot_vcp_adaptive_config();
          else if(v=="vcp_c64") c=itu::barrot_vcp_c64_config();
          else if(v=="vcp_hybrid_v2") c=itu::barrot_vcp_hybrid_v2_config();
          else if(v=="vcp_joint") c=itu::barrot_vcp_joint_config();
          else if(v=="vehicle_stream") c=itu::barrot_vcp_config();
          else if(v=="vcp_unified") c=itu::barrot_vcp_unified_config();
          else if(v=="vcp_fused_aec") c=itu::barrot_vcp_fused_aec_config();
          else if(v=="general") c=itu::Config{};
          else throw std::runtime_error("supported profiles: vcp, vcp_mild, vcp_adaptive, vcp_c64, vcp_hybrid_v2, vcp_joint, vcp_unified, vcp_fused_aec, vehicle_stream, general");
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
          else if(v=="speex_fused_cpu") { c.aec_backend=itu::AecBackend::speex_fused_cpu; c.speex_post_context=true; c.speex_native_conditioning=true; }
          else if(v=="speex_fused_dsp") {
            c.aec_backend=itu::AecBackend::speex_fused_dsp;
            c.speex_post_context=c.speex_native_conditioning=true;
            c.speex_coherence_residual_suppression=c.speex_multirender_context=false;
          }
          else if(v=="computer_pf_cpu") {
            c.aec_backend=itu::AecBackend::computer_pf_cpu;
            c.speex_post_context=c.speex_native_conditioning=true;
            c.speex_coherence_residual_suppression=c.speex_multirender_context=false;
            c.coherence_dtd=false;
          }
          else if(v=="localvqe_aec200k") {
            c.aec_backend=itu::AecBackend::localvqe_aec200k;
            c.speex_post_context=c.speex_native_conditioning=false;
            c.speex_coherence_residual_suppression=c.speex_multirender_context=false;
            c.coherence_dtd=false;
          }
          else throw std::runtime_error("unknown AEC backend: "+v);
        }
        else if(a=="--complex-pf-cpu-pcm") complex_pf_cpu_pcm_path=v;
        else if(a=="--post-ns-pcm") post_ns_pcm_path=v;
        else if(a=="--post-ns-delay-samples") {
          const auto value=std::stoull(v);
          if(value>4800)
            throw std::runtime_error(
                "--post-ns-delay-samples exceeds the 100 ms contract at 48 kHz");
          c.post_ns_delay_samples=static_cast<unsigned>(value);
        }
        else if(a=="--ns-stream-input-fd")
          ns_stream_input_fd=parse_stream_fd(v,"--ns-stream-input-fd");
        else if(a=="--ns-stream-output-fd")
          ns_stream_output_fd=parse_stream_fd(v,"--ns-stream-output-fd");
        else if(a=="--ns-stream-delay-samples") {
          const auto value=std::stoull(v);
          if(value>4800)
            throw std::runtime_error(
                "--ns-stream-delay-samples exceeds the 100 ms contract at 48 kHz");
          ns_stream_delay_samples=static_cast<unsigned>(value);
          ns_stream_delay_explicit=true;
        }
        else if(a=="--aec-dsp-model-id") c.speex_fused_dsp_model_id=v;
        else if(a=="--aec-dsp-deadline-us") c.speex_fused_dsp_deadline_us=std::stoul(v);
        else if(a=="--localvqe-reference-channel") localvqe_reference_channel=std::stoul(v);
        else if(a=="--speex-echo-tail-ms") c.speex_echo_tail_ms=std::stoul(v);
        else if(a=="--aec-fused-cpu-learning-rate") c.speex_fused_cpu_learning_rate=std::stof(v);
        else if(a=="--aec-fused-cpu-shadow-taps-ms") c.speex_fused_cpu_shadow_taps_ms=std::stoul(v);
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
        else if(a=="--vehicle-ns-minima-rebuild-slew-db") c.vehicle_ns.unified_minima_rebuild_slew_db=std::stof(v);
        else if(a=="--vehicle-ns-minima-history-depth") c.vehicle_ns.unified_minima_history_depth=std::stoul(v);
        else if(a=="--vehicle-ns-conditional-minima-history-depth") c.vehicle_ns.unified_conditional_minima_history_depth=std::stoul(v);
        else if(a=="--vehicle-ns-high-snr-bypass-db") c.vehicle_ns.unified_high_snr_bypass_db=std::stof(v);
        else if(a=="--vehicle-ns-high-snr-probability") c.vehicle_ns.unified_high_snr_probability=std::stof(v);
        else if(a=="--vehicle-ns-rumble-recovery") c.vehicle_ns.rumble_recovery=std::stof(v);
        else if(a=="--vehicle-ns-rumble-adaptive-hz") c.vehicle_ns.rumble_adaptive_hz=std::stof(v);
        else if(a=="--vehicle-ns-weak-speech-recovery") c.vehicle_ns.weak_speech_recovery=std::stof(v);
        else if(a=="--vehicle-ns-speech-band-recovery") c.vehicle_ns.speech_band_recovery=std::stof(v);
        else if(a=="--omlsa-noise-update-scale") c.vehicle_ns.omlsa_tuning.noise_update_scale=std::stof(v);
        else if(a=="--omlsa-speech-recovery") c.vehicle_ns.omlsa_tuning.speech_recovery=std::stof(v);
        else if(a=="--omlsa-release-ms") c.vehicle_ns.omlsa_tuning.release_ms=std::stof(v);
        else if(a=="--omlsa-speech-floor-db") c.vehicle_ns.omlsa_tuning.speech_floor_db=std::stof(v);
        else if(a=="--bounded-gain-max-db") c.bounded_gain_max_db=std::stof(v);
        else if(a=="--bounded-gain-rise-db-per-second") c.bounded_gain_rise_db_per_second=std::stof(v);
        else if(a=="--bounded-gain-fall-db-per-second") c.bounded_gain_fall_db_per_second=std::stof(v);
        else if(a=="--bounded-gain-hold-ms") {
          const auto value=std::stoull(v);
          if(value>300) throw std::runtime_error("bounded gain hold must be 0..300 ms");
          c.bounded_gain_hold_ms=static_cast<unsigned>(value);
        }
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
        else if(a=="--hybrid-v2-model") c.vehicle_ns.hybrid_v2_model_path=v;
        else if(a=="--hybrid-v2-mix") c.vehicle_ns.hybrid_v2_mix=std::stof(v);
        else if(a=="--hybrid-v2-floor-db") c.vehicle_ns.hybrid_v2_floor_db=std::stof(v);
        else if(a=="--hybrid-v2-pitch-strength") c.vehicle_ns.hybrid_v2_pitch_strength=std::stof(v);
        else if(a=="--hybrid-ns-model") c.vehicle_ns.student_model_path=v;
        else if(a=="--residual-ns-model") c.vehicle_ns.residual_model_path=v;
        else if(a=="--residual-ns-mix") c.vehicle_ns.residual_model_mix=std::stof(v);
        else if(a=="--residual-ns-floor-db") c.vehicle_ns.residual_model_floor_db=std::stof(v);
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
    if(aec_capabilities) {
      std::cout<<"{\"schema\":\"itu-aec-capabilities-v1\",\"backends\":[";
      bool first=true;
      for(const auto backend:{itu::AecBackend::webrtc,itu::AecBackend::speex,
          itu::AecBackend::speex_fused_cpu,itu::AecBackend::speex_fused_dsp,
          itu::AecBackend::computer_pf_cpu,itu::AecBackend::localvqe_aec200k}) {
        const auto cap=itu::aec_backend_capability(backend,c.speex_fused_dsp_model_id);
        if(!first) std::cout<<',';
        first=false;
        std::cout<<"{\"id\":"<<json_string(cap.id)<<",\"available\":"<<(cap.available?"true":"false")
            <<",\"architecture\":"<<json_string(cap.architecture)<<",\"reason\":"<<json_string(cap.reason)
            <<",\"execution_provider\":"<<json_string(cap.execution_provider)
            <<",\"device_id\":"<<json_string(cap.device_id)<<",\"model_id\":"<<json_string(cap.model_id)
            <<",\"model_sha256\":"<<json_string(cap.model_sha256)
            <<",\"frontend_sha256\":"<<json_string(cap.frontend_sha256)
            <<",\"provider_sha256\":"<<json_string(cap.provider_sha256)
            <<",\"delay_samples\":"<<cap.delay_samples
            <<",\"model_codec_delay_samples_16k\":"<<(std::string(cap.id)=="localvqe_aec200k"?256:0)
            <<",\"linear_frontend_delay_samples_16k\":0"
            <<",\"input_block_requirement_samples_16k\":"<<(std::string(cap.id)=="localvqe_aec200k"?256:0)<<'}';
      }
      std::cout<<"]}\n";return 0;
    }
    const bool ns_stream_input_set=ns_stream_input_fd.has_value();
    const bool ns_stream_output_set=ns_stream_output_fd.has_value();
    if(ns_stream_input_set!=ns_stream_output_set)
      throw std::runtime_error(
          "--ns-stream-input-fd and --ns-stream-output-fd must be supplied together");
    const bool ns_stream_requested=ns_stream_input_set && ns_stream_output_set;
    if(!ns_stream_requested && ns_stream_delay_explicit)
      throw std::runtime_error(
          "--ns-stream-delay-samples requires both NS stream fds");
    if(ns_stream_requested && *ns_stream_input_fd==*ns_stream_output_fd)
      throw std::runtime_error(
          "--ns-stream-input-fd and --ns-stream-output-fd must differ");
    if(ns_stream_requested && mic_path.empty())
      throw std::runtime_error("online NS stream fds require TX microphone input");
    if(c.post_ns_delay_samples && post_ns_pcm_path.empty())
      throw std::runtime_error("--post-ns-delay-samples requires --post-ns-pcm");
    if((mic_path.empty() && rx_in_path.empty()) || prefix.empty()) throw std::runtime_error("usage: itu_audio_cli [--profile vcp|vcp_mild|vcp_adaptive|vcp_c64|vcp_hybrid_v2|vcp_unified] [--hybrid-v2-model path] [--hybrid-v2-mix 0..1] [--hybrid-v2-floor-db -40..0] [--hybrid-v2-pitch-strength 0..1] [--residual-ns-model path] [--aec-backend speex_fused_cpu|speex_fused_dsp|computer_pf_cpu|localvqe_aec200k|speex|webrtc] [--ns-stream-input-fd N --ns-stream-output-fd N] [--ns-stream-delay-samples 640] [--mic before.wav] [--rx-in far.wav | --echo actual-playback.wav] [--complex-pf-export-inputs | --complex-pf-cpu-pcm offline.wav] --out prefix; --aec-capabilities lists route availability");
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
    if(ns_stream_requested && !ns_stream_delay_explicit)
      ns_stream_delay_samples=4*h;
    if(ns_stream_requested &&
       (ns_stream_delay_samples>10*h || ns_stream_delay_samples%h))
      throw std::runtime_error(
          "--ns-stream-delay-samples must be a 10 ms frame multiple in 0..100 ms");
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
    unsigned original_reference_channels=0;
    bool localvqe_reference_selected=false;
    if(!echo_path.empty()) {
      echo=itu::read_wav(echo_path);
      if(!echo.channels || echo.channels>8 || echo.sample_rate!=c.sample_rate)
        throw std::runtime_error("playback reference must have 1..8 synchronous channels, same sample rate");
      if(!mic_path.empty() && echo.samples.size()/echo.channels!=mic.samples.size()/mic.channels)
        throw std::runtime_error("playback reference must have same sample count as mic");
      original_reference_channels=echo.channels;
      c.reference_channels=echo.channels;
    }
    if(localvqe_reference_channel) {
      if(c.aec_backend!=itu::AecBackend::localvqe_aec200k)
        throw std::runtime_error("--localvqe-reference-channel requires --aec-backend localvqe_aec200k");
      if(echo_path.empty() || echo.channels==0)
        throw std::runtime_error("--localvqe-reference-channel requires --echo playback PCM");
      if(*localvqe_reference_channel>=echo.channels)
        throw std::runtime_error("--localvqe-reference-channel is outside the playback channel range");
      const auto channel=*localvqe_reference_channel;
      std::vector<float> mono(echo.samples.size()/echo.channels);
      for(std::size_t i=0;i<mono.size();++i)
        mono[i]=echo.samples[i*echo.channels+channel];
      echo.samples=std::move(mono);
      echo.channels=1;
      c.reference_channels=1;
      localvqe_reference_selected=true;
    }
    if(profile=="vcp_hybrid_v2" && hybrid_v2_legacy_flag)
      throw std::runtime_error(
          "vcp_hybrid_v2 forbids overriding legacy NS/model flags");
    if(profile=="vcp_c64") validate_vcp_c64_profile(c);
    if(profile=="vcp_hybrid_v2") validate_vcp_hybrid_v2_profile(c);
    std::size_t n = !mic_path.empty() ? (mic.samples.size()/mic.channels) : (rx_wav.samples.size()/rx_wav.channels);
    if(!n) throw std::runtime_error("empty input");
    if(!mic_path.empty() && c.aec_backend==itu::AecBackend::localvqe_aec200k)
      c.localvqe_source_samples=n;
    bool measure=far_start>=0 || far_end>=0;
    if(measure && (!c.aec || !std::isfinite(far_start) || !std::isfinite(far_end) || far_start<0 || far_end<=far_start || far_end>double(n)/c.sample_rate))
      throw std::runtime_error("ERLE estimate requires AEC and a valid explicit far-only interval");
    itu::DiagnosticStageTaps stage_taps;
    if(diagnostic_stages) c.diagnostic_stages=&stage_taps;
    const std::size_t rx_total=!rx_in_path.empty() ?
        rx_wav.samples.size()/rx_wav.channels : 0;
    // The public engine bounds causal BF/drain delay at 4096 samples. Reserve
    // every callback-owned diagnostic destination before the first process call.
    const std::size_t drain_capacity=std::size_t(4096/h+16);
    // RX-only processing has no TX AEC. It must not initialize a neural
    // model or require HiFi solely because a TX backend was selected.
    if(mic_path.empty()) c.aec=false;
    const bool computer_pf_cpu=c.aec_backend==itu::AecBackend::computer_pf_cpu;
    const bool localvqe_aec=c.aec_backend==itu::AecBackend::localvqe_aec200k;
    const bool has_playback_input=!echo_path.empty() || !rx_in_path.empty();
    if(complex_pf_export_inputs && !complex_pf_cpu_pcm_path.empty())
      throw std::runtime_error("--complex-pf-export-inputs and --complex-pf-cpu-pcm are separate passes");
    if(complex_pf_export_inputs) {
      if(mic_path.empty() || !c.aec || c.aec_backend!=itu::AecBackend::speex ||
         c.sample_rate!=16000 || c.frame_size!=160 || c.reference_channels!=1 ||
         (!c.beamforming && c.microphones!=1) || c.speex_residual_suppression)
        throw std::runtime_error("--complex-pf-export-inputs requires mono native 16 kHz Speex MDF without RES");
      if(!has_playback_input)
        throw std::runtime_error("--complex-pf-export-inputs requires an actual playback reference");
      // Match the second-pass MDF conditioning exactly. The capture happens
      // after this conditioning, so an external low-cut here would shift the
      // offline model inputs relative to the native CPU replay.
      c.speex_native_conditioning=true;
    }
    if(!mic_path.empty() && computer_pf_cpu) {
      if(!c.aec) throw std::runtime_error("computer_pf_cpu requires enabled AEC");
      if(complex_pf_cpu_pcm_path.empty())
        throw std::runtime_error("computer_pf_cpu requires --complex-pf-cpu-pcm");
      if(!has_playback_input)
        throw std::runtime_error("computer_pf_cpu requires an actual playback reference");
    }
    if(!complex_pf_cpu_pcm_path.empty() && !computer_pf_cpu)
      throw std::runtime_error("--complex-pf-cpu-pcm requires --aec-backend computer_pf_cpu");
    std::shared_ptr<std::vector<float>> complex_pf_cpu_pcm;
    std::string complex_pf_cpu_pcm_report_path;
    if(!mic_path.empty() && computer_pf_cpu) {
      auto pcm_wav=itu::read_wav(complex_pf_cpu_pcm_path);
      if(pcm_wav.sample_rate!=16000 || pcm_wav.channels!=1 || pcm_wav.samples.empty() ||
         pcm_wav.samples.size()%c.frame_size)
        throw std::runtime_error("--complex-pf-cpu-pcm must be non-empty 16 kHz mono PCM with 160-sample blocks");
      if(std::any_of(pcm_wav.samples.begin(),pcm_wav.samples.end(),
          [](float v){return !std::isfinite(v) || std::abs(v)>1.f;}))
        throw std::runtime_error("--complex-pf-cpu-pcm samples must be finite and within [-1, 1]");
      complex_pf_cpu_pcm=std::make_shared<std::vector<float>>(std::move(pcm_wav.samples));
      c.complex_pf_cpu_pcm=complex_pf_cpu_pcm;
      complex_pf_cpu_pcm_report_path=
          std::filesystem::absolute(complex_pf_cpu_pcm_path).lexically_normal().string();
    }
    std::shared_ptr<const std::vector<float>> post_ns_pcm;
    std::string post_ns_pcm_report_path;
    if(!post_ns_pcm_path.empty()) {
      if(mic_path.empty()) throw std::runtime_error("--post-ns-pcm requires TX microphone input");
      auto pcm_wav=read_float32_wav(post_ns_pcm_path);
      if(pcm_wav.sample_rate!=c.sample_rate)
        throw std::runtime_error(
            "--post-ns-pcm sample rate must match the microphone stream");
      if(pcm_wav.samples.empty() || pcm_wav.samples.size()%c.frame_size)
        throw std::runtime_error(
            "--post-ns-pcm must contain complete 10 ms callbacks");
      if(c.post_ns_delay_samples>10*c.frame_size ||
         c.post_ns_delay_samples%c.frame_size)
        throw std::runtime_error(
            "--post-ns-delay-samples must be a 10 ms frame multiple in 0..100 ms");
      post_ns_pcm=std::make_shared<const std::vector<float>>(std::move(pcm_wav.samples));
      c.post_ns_pcm=post_ns_pcm;
      post_ns_pcm_report_path=
          std::filesystem::absolute(post_ns_pcm_path).lexically_normal().string();
    }
    std::unique_ptr<itu::StreamNsPipe> ns_stream_pipe;
    if(ns_stream_requested) {
      if(post_ns_pcm)
        throw std::runtime_error(
            "--post-ns-pcm and online NS stream fds are mutually exclusive");
      c.ns_stream_delay_samples=ns_stream_delay_samples;
      ns_stream_pipe=std::make_unique<itu::StreamNsPipe>(
          *ns_stream_input_fd,*ns_stream_output_fd,h);
      auto* pipe=ns_stream_pipe.get();
      c.ns_stream_process=
          [pipe](const float* input,float* output,unsigned samples) {
            return pipe->process(input,output,samples);
          };
    }
    std::vector<float> complex_pf_residual_capture,complex_pf_reference_capture;
    if(complex_pf_export_inputs) {
      const std::size_t capture_frames=(n+h-1)/h+std::size_t(4096/h+16);
      complex_pf_residual_capture.reserve(capture_frames*c.frame_size);
      complex_pf_reference_capture.reserve(capture_frames*c.frame_size);
      c.complex_pf_residual_capture=&complex_pf_residual_capture;
      c.complex_pf_reference_capture=&complex_pf_reference_capture;
    }
    itu::AudioEngine engine(c); itu::FrameInfo info;
    std::vector<float> rx_spk_samples, rx_ref_samples;
    std::vector<float> rx_chunk(h*c.reference_channels),spk_chunk(rx_chunk.size());
    std::size_t rx_processing_frames=0;
    std::vector<float> rx_nr_stage,rx_eq_stage,rx_drc_stage,rx_output_stage;
    const std::size_t rx_max_frames=(rx_total+h-1)/h+drain_capacity;
    if(diagnostic_stages && !rx_in_path.empty()) {
      const std::size_t capture_samples=rx_max_frames*h*c.reference_channels;
      rx_nr_stage.resize(capture_samples);
      rx_eq_stage.resize(capture_samples);
      rx_drc_stage.resize(capture_samples);
      rx_output_stage.resize(capture_samples);
    }
    std::size_t rx_tap_frames=0;
    auto capture_rx_frame=[&](const std::vector<float>& output) {
      if(!diagnostic_stages || rx_in_path.empty()) return;
      if(!stage_taps.rx_available || stage_taps.rx_sample_rate!=c.sample_rate ||
         stage_taps.rx_frame_size!=h || stage_taps.rx_channels!=c.reference_channels)
        throw std::runtime_error("downlink diagnostic tap shape changed");
      if(rx_tap_frames>=rx_max_frames)
        throw std::runtime_error("downlink diagnostic capture exceeded its preallocated drain capacity");
      const std::size_t frame_samples=std::size_t(h)*c.reference_channels;
      const std::size_t offset=rx_tap_frames*frame_samples;
      for(unsigned ch=0;ch<c.reference_channels;++ch) {
        for(unsigned i=0;i<h;++i) {
          rx_nr_stage[offset+i*c.reference_channels+ch]=stage_taps.rx_nr[ch*480+i];
          rx_eq_stage[offset+i*c.reference_channels+ch]=stage_taps.rx_eq[ch*480+i];
          rx_drc_stage[offset+i*c.reference_channels+ch]=stage_taps.rx_drc[ch*480+i];
        }
      }
      std::copy_n(output.data(),frame_samples,rx_output_stage.data()+offset);
      ++rx_tap_frames;
    };
    auto flush_rx_diagnostic_tail=[&]() {
      if(!diagnostic_stages || rx_in_path.empty()) return;
      if(!stage_taps.rx_available)
        throw std::runtime_error("downlink diagnostics were requested without an RX tap");
      const unsigned delay=stage_taps.rx_drc_delay_samples;
      const std::size_t tail_frames=(delay+h-1)/h;
      if(tail_frames>drain_capacity)
        throw std::runtime_error("downlink DRC tail exceeds the diagnostic drain capacity");
      if(!tail_frames) return;
      std::vector<float> zero(rx_chunk.size()),tail_spk(rx_chunk.size()),tail_ref(rx_chunk.size());
      for(std::size_t i=0;i<tail_frames;++i) {
        std::fill(zero.begin(),zero.end(),0.f);
        if(engine.process_downlink(zero.data(),zero.size(),tail_spk.data(),tail_ref.data())!=itu::Status::ok)
          throw std::runtime_error("downlink DRC diagnostic tail failed");
        capture_rx_frame(tail_spk);
      }
    };
    auto write_rx_outputs=[&]() {
      if(rx_in_path.empty()) return;
      itu::write_wav(!rx_spk_path.empty()?rx_spk_path:prefix+".rx-spk.wav",
          c.sample_rate,c.reference_channels,rx_spk_samples);
      itu::write_wav(!rx_ref_path.empty()?rx_ref_path:prefix+".rx-ref.wav",
          c.sample_rate,c.reference_channels,rx_ref_samples);
    };
    auto build_rx_stages=[&]() {
      std::vector<PipelineStageExport> result;
      if(rx_in_path.empty()) return result;
      const unsigned delay=stage_taps.rx_drc_delay_samples;
      const std::size_t drc_samples=rx_total+delay;
      fit_stage_extent(rx_nr_stage,c.reference_channels,rx_total,"rx_nr");
      fit_stage_extent(rx_eq_stage,c.reference_channels,rx_total,"rx_eq");
      fit_stage_extent(rx_drc_stage,c.reference_channels,drc_samples,"rx_drc");
      fit_stage_extent(rx_output_stage,c.reference_channels,drc_samples,"rx_output");
      const std::string policy="interleaved reference channels; every channel preserved; no downmix";
      auto add=[&](const char* id,unsigned order,const std::string& suffix,
                   const std::vector<float>& samples,std::size_t count,unsigned fixed_delay,
                   bool enabled,const char* domain) {
        PipelineStageExport stage;
        stage.id=id;stage.direction="rx";stage.order=order;
        stage.file=absolute_output_path(prefix+"."+suffix+".float.wav");
        stage.sample_rate=c.sample_rate;stage.channels=c.reference_channels;
        stage.capture_channels=c.reference_channels;
        stage.samples_per_channel=count;stage.fixed_delay_samples=fixed_delay;
        stage.enabled=enabled;stage.capture_available=true;
        stage.measurement_domain=domain;stage.channel_policy=policy;stage.samples=&samples;
        result.push_back(std::move(stage));
      };
      add("rx_input",0,"rx_input",rx_wav.samples,rx_total,0,true,
          "raw far-end RX input, before downlink NR/EQ/DRC");
      add("rx_nr",1,"rx_nr",rx_nr_stage,rx_total,0,
          c.rx_config.enabled && c.rx_config.enable_nr,
          "downlink NR output; zero fixed delay");
      add("rx_eq",2,"rx_eq",rx_eq_stage,rx_total,0,
          c.rx_config.enabled && eq_config_enabled(c.rx_config.eq),
          "downlink loudspeaker EQ output; zero fixed delay");
      add("rx_drc",3,"rx_drc",rx_drc_stage,drc_samples,delay,
          c.rx_config.enabled && c.rx_config.drc.enabled,
          "downlink DRC/limiter output; diagnostic lookahead tail included");
      add("rx_output",4,"rx_output",rx_output_stage,drc_samples,delay,true,
          "actual downlink speaker output and AEC reference; same DRC output");
      return result;
    };
    if(!rx_in_path.empty() && mic_path.empty()) {
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
        capture_rx_frame(spk_chunk);
        ++rx_processing_frames;
        rx_spk_samples.insert(rx_spk_samples.end(), spk_chunk.begin(), spk_chunk.begin() + len*c.reference_channels);
        rx_ref_samples.insert(rx_ref_samples.end(), ref_chunk.begin(), ref_chunk.begin() + len*c.reference_channels);
      }
      flush_rx_diagnostic_tail();
      write_rx_outputs();
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
      if(diagnostic_stages)
        write_stage_metadata(prefix,c.sample_rate,rx_total,true,build_rx_stages(),{});
      std::cout << "wrote " << prefix << ".{rx-spk.wav,rx-ref.wav,json}\n";
      return 0;
    }
    const bool has_playback=!echo.samples.empty() || !rx_in_path.empty();
    if(c.aec && !has_playback) throw std::runtime_error("AEC3 needs actual playback reference or --rx-in; otherwise use --no-aec");
    c=engine.config(); // Report effective multi-render safeguards, not requested mono heuristics.
    const bool speex_aec=c.aec_backend==itu::AecBackend::speex ||
        c.aec_backend==itu::AecBackend::speex_fused_cpu ||
        c.aec_backend==itu::AecBackend::speex_fused_dsp ||
        c.aec_backend==itu::AecBackend::computer_pf_cpu;
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
    csv<<"frame,time_s,far_active,linear_aec_available,statistics_valid,mic_rms,aec_rms,out_rms,erl_db,erle_db,estimated_delay_ms,limited_samples,process_ms,speech_probability,applied_gain_db,reference_state,reference_residual_ratio,reference_confidence,aec_fused_shadow_weight,aec_fused_primary_error_power,aec_fused_shadow_error_power,aec_fused_reference_confidence,aec_fused_double_talk,aec_fused_updates,aec_fused_delay_samples,aec_fused_path_changes,aec_fused_delay_valid,unified_eq_weight,unified_agc_fused_weight,unified_agc_gain_db,linear_speech_probability,nearend_mix,linear_residual_ratio,nonlinear_speech_probability,echo_subtraction_gain,noise_speech_mix,noise_flatness,incoherent_ratio,echo_coherence_baseline,internal_delay_mix,lowband_restore_mix,coherence_learning_frames,fullband_ns_mix,fullband_noise_power,highband_energy_fraction,release_lateness_ms,completion_late,clock_locked,clock_estimated_ppm,clock_queued_samples,clock_capture_hz,clock_render_hz,clock_desired_ppm,clock_capture_valid,clock_render_valid,spatial_delay_samples,spatial_noise_updates,spatial_fallback_bins,wind_score,noise_control_state,noise_transition_events,startup_dry_mix,neural_inferences,native_capture_ms,native_render_ms,native_core_ms,native_output_ms,native_taps_ms,pre_ns_rms,post_ns_rms,min_stat_gain,gain_control_noise_power,gain_speech_qualified,gain_control_state,gain_gate_level,gain_input_level,gain_limit_db,gain_votes,gain_voiced_rms,gain_peak,gain_max_voiced,gain_target_db,gain_hold_db,noise_profile_observations,noise_profile_mix,v2_double_evidence_eligible,v2_near_confirmation,g1_strict_near_age,g1_startup_echo_frames,g1_startup_clear_threshold,g1_startup_clear_deferred,near_protection_hold_frames,startup_echo_confirmation,g1_echo_evidence_frames,g1_echo_clear_threshold,g1_echo_clear_deferred,g1_strong_near_confirmation,g1_near_latch_active,g1_original_confirmed_echo,g1_original_coherent_clear,ns_residual_gain,ns_residual_flatness,ns_residual_quiet_frames,ns_residual_observations,tone_frequency_hz,tone_mix,tone_concentration,tone_repeats,external_vad_active,external_vad_calls,input_excess_score,input_excess_fraction,control_evidence_probability,gain_noise_evidence,gain_noise_compensation,post_aec_context_applied,post_aec_cancellation_ratio,post_aec_echo_floor,post_aec_coherence_floor,post_aec_near_observed,post_aec_echo_clear_frames,hybrid_ns_state,hybrid_ns_inferences,hybrid_ns_fallback_frames,hybrid_ns_applied,vehicle_ns_raw_mix_budget,vehicle_ns_raw_mix_min,vehicle_ns_raw_mix_max,post_aec_reference_evidence_valid,post_aec_near_state,post_aec_multirender_frames,unified_ns_w0,unified_ns_w1,unified_ns_w2,unified_ns_w3,unified_ns_target_w0,unified_ns_target_w1,unified_ns_target_w2,unified_ns_target_w3,unified_ns_weight_entropy,unified_ns_frame_snr_db,unified_ns_flatness,unified_ns_tonality,unified_ns_minima_confidence,unified_ns_transient_evidence,unified_ns_noise_step_evidence,unified_ns_probability,unified_ns_gain,unified_ns_noise_power,unified_ns_bootstrap_seed,unified_ns_noise_updates,unified_ns_minima_rotations,unified_ns_minima_band_db,unified_ns_conditional_minima_band_db,unified_ns_history_minima_band_db,unified_ns_noise_psd_band_db,unified_ns_smooth_band_db,unified_ns_conditional_band_db,unified_ns_dd_prior_band_db,unified_ns_xi_band_db,unified_ns_gamma_band_db,unified_ns_gain_band_db,unified_ns_minima_rebuild_jump_db,unified_ns_conditional_minima_rebuild_jump_db,unified_ns_high_snr_bypass,hybrid_v2_state,hybrid_v2_inferences,hybrid_v2_fallback_frames,hybrid_v2_applied,hybrid_v2_applied_frames,hybrid_v2_parameters,hybrid_v2_input_dim,hybrid_v2_raw_mix,hybrid_v2_pitch_active,hybrid_v2_pitch_active_frames,hybrid_v2_pitch_fallback_frames,hybrid_v2_pitch_lag_samples,hybrid_v2_pitch_periodicity\n";
    std::size_t frames=(n+h-1)/h,total_frames=frames,limited=0,unreliable=0; bool all_linear=c.aec;
    // Include the largest legal causal BF compensation and DSP drain before replay.
    const auto capacity=(frames+drain_capacity)*h;
    after.reserve(capacity); bf.reserve(capacity); times.reserve(frames);
    if(!rx_in_path.empty()) {rx_spk_samples.reserve(n*c.reference_channels);rx_ref_samples.reserve(n*c.reference_channels);}
    aec.reserve(capacity*c.microphones); linear.reserve(capacity*c.microphones);
    conditioned.reserve(capacity*c.microphones);
    std::vector<RecordedFrame> recorded(frames+drain_capacity);
    const bool native_aec_pipeline=speex_aec || localvqe_aec;
    const bool core_bridge=!c.ns_stream_process &&
        ((c.sample_rate==24000 && ((c.noise_backend!=itu::NoiseBackend::omlsa_imcra && c.noise_backend!=itu::NoiseBackend::super_gaussian && c.noise_backend!=itu::NoiseBackend::unified) || (c.aec && !native_aec_pipeline))) ||
         (c.sample_rate==16000 && c.noise_backend==itu::NoiseBackend::rnnoise));
    const unsigned core_h=core_bridge ? 480 : h;
    const unsigned eq_h=core_bridge && c.eq_before_agc ? core_h : h;
    std::array<std::vector<float>,5> stages;
    std::vector<float> tx_bf_stage,tx_aec_input_stage,tx_aec_stage,tx_drc_stage,tx_ns_input_stage;
    if(diagnostic_stages) {
      const std::size_t capture_frames=frames+drain_capacity;
      for(unsigned j=0;j<stages.size();++j)
        stages[j].resize(capture_frames*((j<3 || c.eq_before_agc)?core_h:h));
      tx_bf_stage.resize(capture_frames*h);
      tx_aec_stage.resize(capture_frames*h*c.microphones);
      tx_drc_stage.resize(capture_frames*core_h);
      tx_ns_input_stage.resize(capture_frames*core_h);
      if(c.aec) tx_aec_input_stage.resize(capture_frames*h*c.microphones);
    }
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
        capture_rx_frame(spk_chunk);
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
      if(k==0) {
        total_frames=frames+(info.latency_samples+h-1)/h;
        if(total_frames>frames+drain_capacity)
          throw std::runtime_error("processing drain exceeds the preallocated diagnostic capacity");
        if(post_ns_pcm && post_ns_pcm->size()<total_frames*c.frame_size)
          throw std::runtime_error("--post-ns-pcm must cover every input and drain callback");
        if(complex_pf_cpu_pcm &&
           complex_pf_cpu_pcm->size()!=total_frames*c.frame_size)
          throw std::runtime_error("--complex-pf-cpu-pcm length must exactly match all input and flush frames");
      }
      if(diagnostic_stages) {
        if(k>=frames+drain_capacity)
          throw std::runtime_error("diagnostic capture exceeded its preallocated drain capacity");
        if(stage_taps.core_frame_size!=core_h || stage_taps.eq_frame_size!=eq_h)
          throw std::runtime_error("diagnostic tap frame geometry differs from the engine topology");
        const std::array<const float*,5> taps={stage_taps.nr_selected.data(),stage_taps.nr_protected.data(),
            stage_taps.post_agc.data(),stage_taps.pre_eq.data(),stage_taps.post_eq.data()};
        for(unsigned j=0;j<stages.size();++j) {
          const auto length=j<3?core_h:stage_taps.eq_frame_size;
          std::copy_n(taps[j],length,stages[j].data()+k*length);
        }
        if(c.beamforming) std::copy_n(bt.data(),h,tx_bf_stage.data()+k*h);
        else for(unsigned i=0;i<h;++i) tx_bf_stage[k*h+i]=frame[i*c.microphones];
        std::copy_n(at.data(),at.size(),tx_aec_stage.data()+k*at.size());
        if(c.aec) std::copy_n(it.data(),it.size(),tx_aec_input_stage.data()+k*it.size());
        std::copy_n(stage_taps.post_drc.data(),core_h,tx_drc_stage.data()+k*core_h);
        std::copy_n(stage_taps.experiment_ns_input.data(),core_h,
                    tx_ns_input_stage.data()+k*core_h);
      }
      after.insert(after.end(),out.begin(),out.end()); aec.insert(aec.end(),at.begin(),at.end()); bf.insert(bf.end(),bt.begin(),bt.end());
      conditioned.insert(conditioned.end(),it.begin(),it.end());
      for(unsigned i=0;i<160;++i) for(unsigned ch=0;ch<c.microphones;++ch) linear.push_back(lt[ch*160+i]);
      if(k<frames) {limited+=info.limited_samples; times.push_back(ms); all_linear &= info.linear_aec_available; unreliable+=info.reference_state==2;}
      recorded[k]={info,clock?clock->info():itu::ClockInfo{},ms,paced?std::max(0.,std::chrono::duration<double,std::milli>(actual_release-release).count()):0.,
                   paced && completed>release+std::chrono::milliseconds(10)};
    }
    flush_rx_diagnostic_tail();
    write_rx_outputs();
    const auto aec_route=engine.aec_route_diagnostics();
    if(complex_pf_cpu_pcm &&
       (aec_route.processed_frames!=total_frames ||
        aec_route.processed_frames*c.frame_size!=complex_pf_cpu_pcm->size()))
      throw std::runtime_error("computer_pf_cpu did not consume the exact offline PCM frame sequence");
    if(complex_pf_export_inputs) {
      if(complex_pf_residual_capture.size()!=total_frames*c.frame_size ||
         complex_pf_reference_capture.size()!=total_frames*c.frame_size)
        throw std::runtime_error("Complex_PF export did not capture every input and flush frame");
      write_float_wav(prefix+".pf-residual.wav",16000,complex_pf_residual_capture);
      write_float_wav(prefix+".pf-reference.wav",16000,complex_pf_reference_capture);
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
      csv<<','<<info.limited_samples<<','<<ms<<','<<info.speech_probability<<','<<info.applied_gain_db<<','<<info.reference_state<<','<<info.reference_residual_ratio<<','<<info.reference_confidence<<','<<info.aec_fused_shadow_weight<<','<<info.aec_fused_primary_error_power<<','<<info.aec_fused_shadow_error_power<<','<<info.aec_fused_reference_confidence<<','<<info.aec_fused_double_talk<<','<<info.aec_fused_updates<<','<<info.aec_fused_delay_samples<<','<<info.aec_fused_path_changes<<','<<info.aec_fused_delay_valid<<','<<info.unified_eq_weight<<','<<info.unified_agc_fused_weight<<','<<info.unified_agc_gain_db<<','<<info.linear_speech_probability<<','<<info.nearend_mix<<','<<info.linear_residual_ratio<<','<<info.nonlinear_speech_probability<<','<<info.echo_subtraction_gain<<','<<info.noise_speech_mix<<','<<info.noise_flatness<<','<<info.incoherent_ratio<<','<<info.echo_coherence_baseline<<','<<info.internal_delay_mix<<','<<info.lowband_restore_mix<<','<<info.coherence_learning_frames<<','<<info.fullband_ns_mix<<','<<info.fullband_noise_power<<','<<info.highband_energy_fraction<<','<<recorded[k].release_lateness_ms<<','<<recorded[k].completion_late<<','<<recorded[k].clock.locked<<','<<recorded[k].clock.drift_ppm<<','<<recorded[k].clock.queued_samples<<','<<recorded[k].clock.capture_rate_hz<<','<<recorded[k].clock.render_rate_hz<<','<<recorded[k].clock.desired_ppm<<','<<recorded[k].clock.capture_valid<<','<<recorded[k].clock.render_valid
         <<','<<info.spatial_delay_samples<<','<<info.spatial_noise_updates<<','<<info.spatial_fallback_bins<<','<<info.wind_score<<','<<info.noise_control_state<<','<<info.noise_transition_events<<','<<info.startup_dry_mix<<','<<info.neural_inferences
         <<','<<info.native_capture_ms<<','<<info.native_render_ms<<','<<info.native_core_ms<<','<<info.native_output_ms<<','<<info.native_taps_ms<<','<<info.pre_ns_rms<<','<<info.post_ns_rms<<','<<info.min_stat_gain<<','<<info.gain_control_noise_power<<','<<info.gain_speech_qualified<<','<<info.gain_control_state<<','<<info.gain_gate_level<<','<<info.gain_input_level<<','<<info.gain_limit_db<<','<<info.gain_votes<<','<<info.gain_voiced_rms<<','<<info.gain_peak<<','<<info.gain_max_voiced<<','<<info.gain_target_db<<','<<info.gain_hold_db<<','<<info.noise_profile_observations<<','<<info.noise_profile_mix<<','<<info.v2_double_evidence_eligible<<','<<info.v2_near_confirmation<<','<<info.g1_strict_near_age<<','<<info.g1_startup_echo_frames<<','<<info.g1_startup_clear_threshold<<','<<info.g1_startup_clear_deferred<<','<<info.near_protection_hold_frames<<','<<info.startup_echo_confirmation<<','<<info.g1_echo_evidence_frames<<','<<info.g1_echo_clear_threshold<<','<<info.g1_echo_clear_deferred<<','<<info.g1_strong_near_confirmation<<','<<info.g1_near_latch_active<<','<<info.g1_original_confirmed_echo<<','<<info.g1_original_coherent_clear<<','<<info.ns_residual_gain<<','<<info.ns_residual_flatness<<','<<info.ns_residual_quiet_frames<<','<<info.ns_residual_observations<<','<<info.tone_frequency_hz<<','<<info.tone_mix<<','<<info.tone_concentration<<','<<info.tone_repeats<<','<<info.external_vad_active<<','<<info.external_vad_calls<<','<<info.input_excess_score<<','<<info.input_excess_fraction<<','<<info.control_evidence_probability<<','<<info.gain_noise_evidence<<','<<info.gain_noise_compensation<<','<<info.post_aec_context_applied<<','<<info.post_aec_cancellation_ratio<<','<<info.post_aec_echo_floor<<','<<info.post_aec_coherence_floor<<','<<info.post_aec_near_observed<<','<<info.post_aec_echo_clear_frames<<','<<info.hybrid_ns_state<<','<<info.hybrid_ns_inferences<<','<<info.hybrid_ns_fallback_frames<<','<<info.hybrid_ns_applied<<','<<info.vehicle_ns_raw_mix_budget<<','<<info.vehicle_ns_raw_mix_min<<','<<info.vehicle_ns_raw_mix_max<<','<<info.post_aec_reference_evidence_valid<<','<<info.post_aec_near_state<<','<<info.post_aec_multirender_frames<<','
         <<info.unified_ns_weights[0]<<','<<info.unified_ns_weights[1]<<','<<info.unified_ns_weights[2]<<','<<info.unified_ns_weights[3]<<','
         <<info.unified_ns_target_weights[0]<<','<<info.unified_ns_target_weights[1]<<','<<info.unified_ns_target_weights[2]<<','<<info.unified_ns_target_weights[3]<<','
         <<info.unified_ns_weight_entropy<<','<<info.unified_ns_frame_snr_db<<','<<info.unified_ns_flatness<<','<<info.unified_ns_tonality<<','
         <<info.unified_ns_minima_confidence<<','<<info.unified_ns_transient_evidence<<','<<info.unified_ns_noise_step_evidence<<','
         <<info.unified_ns_probability<<','<<info.unified_ns_gain<<','<<info.unified_ns_noise_power<<','<<info.unified_ns_bootstrap_seed<<','
         <<info.unified_ns_noise_updates<<','<<info.unified_ns_minima_rotations<<','<<info.unified_ns_minima_band_db<<','<<info.unified_ns_conditional_minima_band_db<<','<<info.unified_ns_history_minima_band_db<<','<<info.unified_ns_noise_psd_band_db<<','<<info.unified_ns_smooth_band_db<<','<<info.unified_ns_conditional_band_db<<','<<info.unified_ns_dd_prior_band_db<<','<<info.unified_ns_xi_band_db<<','<<info.unified_ns_gamma_band_db<<','<<info.unified_ns_gain_band_db<<','<<info.unified_ns_minima_rebuild_jump_db<<','<<info.unified_ns_conditional_minima_rebuild_jump_db<<','<<info.unified_ns_high_snr_bypass
         <<','<<info.hybrid_v2_state<<','<<info.hybrid_v2_inferences<<','<<info.hybrid_v2_fallback_frames<<','<<info.hybrid_v2_applied<<','<<info.hybrid_v2_applied_frames<<','<<info.hybrid_v2_parameters<<','<<info.hybrid_v2_input_dim<<','<<info.hybrid_v2_raw_mix<<','<<info.hybrid_v2_pitch_active<<','<<info.hybrid_v2_pitch_active_frames<<','<<info.hybrid_v2_pitch_fallback_frames<<','<<info.hybrid_v2_pitch_lag_samples<<','<<info.hybrid_v2_pitch_periodicity<<'\n';
    }
    unsigned aec_delay=info.aec_output_delay_samples,linear_delay=info.linear_aec_delay_samples;
    if(c.profile_stages) {
      std::ofstream timing(prefix+".timing.csv");
      if(!timing) throw std::runtime_error("cannot write pipeline timing CSV");
      timing<<"frame,time_s,process_ms,localvqe_process_ms,ns_stream_roundtrip_ms,localvqe_post_ms\n";
      timing<<std::setprecision(9);
      for(std::size_t frame_index=0;frame_index<total_frames;++frame_index) {
        const auto& measured=recorded[frame_index];
        timing<<frame_index<<','<<double(frame_index*h)/c.sample_rate<<','
            <<measured.process_ms<<','<<measured.info.localvqe_process_ms<<','
            <<measured.info.ns_stream_roundtrip_ms<<','
            <<measured.info.localvqe_post_ms<<'\n';
      }
      if(!timing) throw std::runtime_error("pipeline timing CSV write failed");
    }
    auto resize_checked=[&](std::vector<float>& values,std::size_t count,const char* name) {
      if(values.size()<count)
        throw std::runtime_error(std::string(name)+" diagnostic extent is shorter than the aligned output");
      values.resize(count);
    };
    resize_checked(after,n+info.latency_samples,"after");
    resize_checked(aec,(n+aec_delay)*c.microphones,"aec");
    resize_checked(linear,(n*16000/c.sample_rate+linear_delay)*c.microphones,"linear-aec");
    resize_checked(bf,n+info.bf_delay_samples,"bf");
    resize_checked(conditioned,(n+info.aec_input_delay_samples)*c.microphones,"aec-input");
    itu::write_wav(prefix+".after.wav",c.sample_rate,1,after);
    itu::write_wav(prefix+".aec.wav",c.sample_rate,c.microphones,aec);
    if(all_linear) itu::write_wav(prefix+".linear-aec.wav",16000,c.microphones,linear);
    if(c.aec) itu::write_wav(prefix+".aec-input.wav",c.sample_rate,c.microphones,conditioned);
    itu::write_wav(prefix+".bf.wav",c.sample_rate,1,bf);
    if(diagnostic_stages) {
      const std::array<const char*,5> names={"nr-selected","nr-protected-agc-before","agc-after","eq-before","eq-after"};
      const std::array<const char*,5> domains=c.ns_stream_process ?
          std::array<const char*,5>{
            "external worker output; no native NS or protection branch",
            "complete external worker output before delayed-evidence AGC/EQ/DRC",
            "core-rate AGC output using delayed FIFO control evidence",
            "actual EQ input; core rate before AGC when selected, otherwise I/O rate after AGC",
            "actual EQ output; configured EQ phase retained"} :
          std::array<const char*,5>{
            "selected NR backend plus aligned fullband branch; before shared speech protection and lowband restoration",
            "protected NR output with lowband/startup restoration; before optional pre-AGC EQ",
            "core-rate AGC output before TX DRC and limiter",
            "actual EQ input; core rate before AGC when selected, otherwise I/O rate after AGC",
            "actual EQ output; before AGC/DRC when selected, otherwise before final I/O limiter; configured EQ phase retained"};
      const unsigned core_rate=stage_taps.core_sample_rate;
      const unsigned eq_rate=stage_taps.eq_sample_rate;
      if(!core_rate || !eq_rate) throw std::runtime_error("invalid diagnostic stage sample rate");
      const std::array<unsigned,5> legacy_rates={core_rate,core_rate,core_rate,eq_rate,eq_rate};
      const std::array<unsigned,5> legacy_delays={stage_taps.core_delay_samples,
          stage_taps.core_delay_samples,stage_taps.agc_delay_samples,
          stage_taps.pre_eq_delay_samples,stage_taps.post_eq_delay_samples};
      std::array<std::size_t,5> legacy_counts{};
      std::vector<LegacyStageExport> legacy_stages;
      for(unsigned j=0;j<stages.size();++j) {
        legacy_counts[j]=n*legacy_rates[j]/c.sample_rate+legacy_delays[j];
        fit_stage_extent(stages[j],1,legacy_counts[j],names[j]);
        LegacyStageExport stage;
        stage.name=names[j];
        stage.file=absolute_output_path(prefix+"."+names[j]+".float.wav");
        stage.measurement_domain=domains[j];
        stage.sample_rate=legacy_rates[j];stage.channels=1;
        stage.fixed_delay_samples=legacy_delays[j];
        stage.samples_per_channel=legacy_counts[j];
        stage.nr_backend_active=stage_taps.nr_selected_available;
        stage.samples=&stages[j];
        legacy_stages.push_back(std::move(stage));
      }

      const unsigned outer_bf_delay=c.beamforming?info.bf_delay_samples:0;
      fit_stage_extent(tx_bf_stage,1,n+outer_bf_delay,"tx_bf");
      if(c.aec) fit_stage_extent(tx_aec_input_stage,c.microphones,n+outer_bf_delay,"tx_aec_input");
      fit_stage_extent(tx_aec_stage,c.microphones,n+aec_delay,"tx_aec");
      const std::size_t core_samples=n*core_rate/c.sample_rate+stage_taps.core_delay_samples;
      const std::size_t agc_samples=n*core_rate/c.sample_rate+stage_taps.agc_delay_samples;
      const std::size_t drc_samples=n*core_rate/c.sample_rate+stage_taps.drc_delay_samples;
      const std::size_t eq_samples=n*eq_rate/c.sample_rate+stage_taps.post_eq_delay_samples;
      fit_stage_extent(tx_drc_stage,1,drc_samples,"tx_drc");

      std::vector<PipelineStageExport> pipeline_stages;
      unsigned tx_order=0;
      auto add_tx=[&](const char* id,const std::string& suffix,const std::vector<float>& samples,
                      unsigned rate,unsigned channels,unsigned capture_channels,
                      std::size_t count,unsigned delay,bool enabled,
                      const char* domain,const char* policy,bool diagnostic=false) {
        PipelineStageExport stage;
        stage.id=id;stage.direction="tx";stage.order=tx_order++;
        stage.file=absolute_output_path(prefix+"."+suffix+".float.wav");
        stage.sample_rate=rate;stage.channels=channels;stage.capture_channels=capture_channels;
        stage.samples_per_channel=count;
        stage.fixed_delay_samples=delay;stage.enabled=enabled;stage.capture_available=true;
        stage.measurement_domain=domain;stage.channel_policy=policy;
        stage.diagnostic=diagnostic;stage.samples=&samples;
        pipeline_stages.push_back(std::move(stage));
      };
      const std::string stream_ns_input_policy=
          "mono little-endian float32["+std::to_string(h)+
          "] written to the parent-owned worker pipe";
      const char* input_policy="interleaved physical microphone channels; every channel preserved; no downmix";
      const char* bf_policy=c.beamforming?"mono beamformer output before AEC":
          "selected physical channel 0 bypass; no beamformer";
      const char* aec_policy=(c.beamforming && c.microphones>1)?
          "interleaved post-BF replicas; downstream consumes one independent mono channel; no microphone-fraction improvement implied":
          "interleaved engine capture channels; no downmix";
      const char* aec_input_policy=(c.beamforming && c.microphones>1)?
          "interleaved actual AEC input; post-BF channel slots are replicas of one beamformed channel":
          "interleaved actual AEC input; no downmix";
      const bool eq_enabled=c.unified_eq_agc || eq_config_enabled(c.output_eq);
      const unsigned aec_capture_channels=info.beamforming_before_aec?
          std::max(1u,info.aec_capture_channels):c.microphones;
      add_tx("tx_input","tx_input",mic.samples,c.sample_rate,c.microphones,c.microphones,n,0,true,
          "original microphone input before BF/AEC/NS/EQ/AGC/DRC",input_policy);
      add_tx("tx_bf","tx_bf",tx_bf_stage,c.sample_rate,1,1,n+outer_bf_delay,outer_bf_delay,
          c.beamforming,c.beamforming?"actual BF output before AEC":
          "BF bypass output: selected physical microphone channel 0; no AEC/NS substitution",bf_policy);
      if(c.aec)
        add_tx("tx_aec_input","tx_aec_input",tx_aec_input_stage,c.sample_rate,c.microphones,
            aec_capture_channels,
            n+outer_bf_delay,outer_bf_delay,true,
            "actual AEC input before cancellation; internal downsample/conditioning is not counted as upstream delay",
            aec_input_policy);
      const char* aec_domain=!c.aec?"AEC bypass output from the engine capture path":
          localvqe_aec?"LocalVQE DAF plus neural mask output before shared NS":
          "actual AEC output before shared NS";
      add_tx("tx_aec","tx_aec",tx_aec_stage,c.sample_rate,c.microphones,
          aec_capture_channels,n+aec_delay,aec_delay,
          c.aec,aec_domain,aec_policy);
      const bool native_ns_input_export=
          (c.sample_rate==16000 && core_rate==16000 &&
           ((!c.aec && !c.beamforming && c.microphones==1) || localvqe_aec)) ||
          (c.sample_rate==48000 && core_rate==48000);
      if(native_ns_input_export) {
        auto ns_input_delay=stage_taps.experiment_ns_input_delay_samples;
        if(c.sample_rate==48000) {
          if(stage_taps.core_delay_samples<info.ns_delay_samples)
            throw std::runtime_error("invalid 48 kHz NS input delay metadata");
          ns_input_delay=stage_taps.core_delay_samples-info.ns_delay_samples;
        }
        fit_stage_extent(tx_ns_input_stage,1,n+ns_input_delay,"tx_ns_input");
        add_tx("tx_ns_input","tx_ns_input",tx_ns_input_stage,core_rate,1,1,
            n+ns_input_delay,ns_input_delay,c.noise_suppression,
            c.ns_stream_process?
                "actual external stream input after capture conditioning":
                "actual native NS input after capture conditioning",
            c.ns_stream_process?
                stream_ns_input_policy.c_str():
                "mono actual NS input",true);
      }
      const char* ns_selected_domain=c.ns_stream_process?
          "external worker output; before delayed-evidence AGC, EQ, DRC and limiter":
          domains[0];
      const char* ns_protected_domain=c.ns_stream_process?
          "complete external worker NS output; no native fallback or protection":
          domains[1];
      add_tx("tx_ns_selected","tx_ns_selected",stages[0],core_rate,1,1,core_samples,
          stage_taps.core_delay_samples,c.noise_suppression,ns_selected_domain,
          c.ns_stream_process?
              "mono complete external-worker output":
              "mono selected-backend diagnostic child; not the complete protected NS output",true);
      add_tx("tx_ns","tx_ns",stages[1],core_rate,1,1,core_samples,
          stage_taps.core_delay_samples,c.noise_suppression,ns_protected_domain,
          "mono complete NS output");
      auto add_eq=[&]() {
        const char* domain=c.eq_before_agc?
            "actual EQ output before AGC/DRC; core rate and configured EQ phase retained":
            core_bridge?"outer I/O-rate EQ output after the core DRC; configured EQ phase retained":
            "actual post-AGC EQ output before TX DRC; configured EQ phase retained";
        add_tx("tx_eq","tx_eq",stages[4],eq_rate,1,1,eq_samples,stage_taps.post_eq_delay_samples,
            eq_enabled,domain,"mono TX EQ output; bypass preserves the upstream PCM");
      };
      auto add_agc=[&]() {
        const char* domain=c.ns_stream_process?
            "AGC output using worker-audio-aligned control evidence delayed by the NS stream contract":
            core_bridge && !c.eq_before_agc?
            "core-rate AGC output before core DRC; EQ runs later at I/O rate":
            "AGC output before the following EQ/DRC stages";
        add_tx("tx_agc","tx_agc",stages[2],core_rate,1,1,agc_samples,
            stage_taps.agc_delay_samples,c.automatic_gain,domain,"mono AGC output; bypass preserves upstream PCM");
      };
      auto add_drc=[&]() {
        const char* domain=core_bridge && !c.eq_before_agc?
            "core-rate TX DRC output before outer I/O-rate EQ and final limiter":
            "TX DRC output before final limiter and quiet-boundary processing";
        add_tx("tx_drc","tx_drc",tx_drc_stage,core_rate,1,1,drc_samples,
            stage_taps.drc_delay_samples,c.tx_drc.enabled,domain,
            "mono DRC output; bypass preserves upstream PCM");
      };
      if(c.eq_before_agc) {add_eq();add_agc();add_drc();}
      else if(core_bridge) {add_agc();add_drc();add_eq();}
      else {add_agc();add_eq();add_drc();}
      add_tx("tx_output","tx_output",after,c.sample_rate,1,1,n+info.latency_samples,
          info.latency_samples,true,"final TX output after DRC, limiter and quiet-boundary processing",
          "mono pre-quantization final TX float; PCM16 output is derived from the same samples");
      auto rx_stages=build_rx_stages();
      pipeline_stages.insert(pipeline_stages.end(),rx_stages.begin(),rx_stages.end());
      write_stage_metadata(prefix,c.sample_rate,n,!rx_in_path.empty(),pipeline_stages,legacy_stages);
    }
    double input_p=power(mic.samples,0,mic.samples.size()),output_p=power(after,info.latency_samples,after.size());
    double total=0; for(double t:times) total+=t; std::sort(times.begin(),times.end());
    std::ofstream report(prefix+".json"); if(!report) throw std::runtime_error("cannot write report");
    report<<std::setprecision(9)<<"{\n  \"schema_version\":2,\n  \"webrtc_revision\":\""<<itu::kWebRtcRevision<<"\",\n  \"sample_rate\":"<<c.sample_rate<<",\n  \"microphones\":"<<c.microphones<<",\n  \"rx_processing\":"<<(c.rx_processing?"true":"false")<<",\n  \"eq_before_agc\":"<<(c.eq_before_agc?"true":"false")<<",\n  \"unified_eq_agc\":"<<(c.unified_eq_agc?"true":"false")<<",\n  \"rx_in\":"<<json_string(rx_in_path)
      <<",\n  \"reference_channels\":"<<c.reference_channels
      <<",\n  \"reference_channels_original\":"<<(original_reference_channels?original_reference_channels:c.reference_channels)
      <<",\n  \"localvqe_reference_channel\":"<<(localvqe_reference_selected?static_cast<int>(*localvqe_reference_channel):-1)
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
      <<",\n  \"ns_backend\":\""<<(c.ns_stream_process?"external_stream":c.post_ns_pcm?"external_pcm":c.noise_backend==itu::NoiseBackend::unified?"unified":c.noise_backend==itu::NoiseBackend::super_gaussian?"super_gaussian":c.noise_backend==itu::NoiseBackend::omlsa_imcra?"omlsa_imcra":c.noise_backend==itu::NoiseBackend::ulunas?"ulunas":c.noise_backend==itu::NoiseBackend::speex?"speex":c.noise_backend==itu::NoiseBackend::rnnoise?"rnnoise":"webrtc")<<"\""
      <<",\n  \"ns_policy\":\""
      <<(c.ns_stream_process?
          "stateful per-frame CPU worker; not a hard real-time guarantee":
          c.post_ns_pcm?"offline replay; no runtime NS worker":"native streaming backend")
      <<"\""
      <<",\n  \"ns_stream_transport\":\""
      <<(c.ns_stream_process?
          "parent-owned POSIX pipes; little-endian float32["+std::to_string(h)+"]":"none")<<"\""
      <<",\n  \"ns_stream_frame_samples\":"<<(c.ns_stream_process?h:0)
      <<",\n  \"ns_stream_chunk_samples\":"<<(c.ns_stream_process?h:0)
      <<",\n  \"ns_stream_sample_rate\":"<<(c.ns_stream_process?c.sample_rate:0)
      <<",\n  \"ns_stream_frame_timeout_ms\":"<<(c.ns_stream_process?10000:0)
      <<",\n  \"ns_stream_control_alignment\":\""
      <<(c.ns_stream_process?"delayed_fifo":"none")<<"\""
      <<",\n  \"ns_stream_control_delay_samples\":"
      <<(c.ns_stream_process?c.ns_stream_delay_samples:0)
      <<",\n  \"ns_stream_native_fallback_enabled\":false"
      <<",\n  \"ns_external_stream_enabled\":"<<(info.ns_external_stream_enabled?"true":"false")
      <<",\n  \"ns_external_pcm_enabled\":"<<(info.ns_external_pcm_enabled?"true":"false")
      <<",\n  \"ns_external_pcm_path\":"<<json_string(post_ns_pcm_report_path)
      <<",\n  \"ns_external_applied_frames\":"<<info.ns_external_applied_frames
      <<",\n  \"ns_external_delay_samples\":"<<info.ns_external_delay_samples
      <<",\n  \"ns_external_fallback_frames\":"<<info.ns_external_fallback_frames
      <<",\n  \"speex_suppression_db\":"<<c.speex_suppression_db
      <<",\n  \"speex_multirender_context\":"<<(c.speex_multirender_context?"true":"false")
      <<",\n  \"vehicle_ns_control\":"<<(c.vehicle_ns_control?"true":"false")
      <<",\n  \"omlsa_noise_update_scale\":"<<c.vehicle_ns.omlsa_tuning.noise_update_scale
      <<",\n  \"omlsa_speech_recovery\":"<<c.vehicle_ns.omlsa_tuning.speech_recovery
      <<",\n  \"omlsa_release_ms\":"<<c.vehicle_ns.omlsa_tuning.release_ms
      <<",\n  \"omlsa_speech_floor_db\":"<<c.vehicle_ns.omlsa_tuning.speech_floor_db
      <<",\n  \"omlsa_adaptive_noise_tracking\":"<<(c.vehicle_ns.omlsa_tuning.adaptive_noise_tracking?"true":"false")
      <<",\n  \"omlsa_high_snr_transparency\":"<<(c.vehicle_ns.omlsa_tuning.high_snr_transparency?"true":"false")
      <<",\n  \"omlsa_low_band_protection\":"<<(c.vehicle_ns.omlsa_tuning.low_band_protection?"true":"false")
      <<",\n  \"bounded_speech_gain\":"<<(c.bounded_speech_gain?"true":"false")
      <<",\n  \"bounded_gain_max_db\":"<<c.bounded_gain_max_db
      <<",\n  \"bounded_gain_rise_db_per_second\":"<<c.bounded_gain_rise_db_per_second
      <<",\n  \"bounded_gain_fall_db_per_second\":"<<c.bounded_gain_fall_db_per_second
      <<",\n  \"bounded_gain_hold_ms\":"<<c.bounded_gain_hold_ms
      <<",\n  \"mild_near_only_ns\":"<<(c.mild_near_only_ns?"true":"false")
      <<",\n  \"mild_ns_route\":\""<<(c.mild_near_only_ns?
          (c.aec?"established_aec_ns":"optimized_near_only_ns"):"explicit_backend")<<"\""
      <<",\n  \"vehicle_ns_unified_raw_budget\":"<<(c.vehicle_ns.unified_raw_budget?"true":"false")
      <<",\n  \"vehicle_ns_unified_learning_rate\":"<<c.vehicle_ns.unified_learning_rate
      <<",\n  \"vehicle_ns_unified_minima_rebuild_slew_db\":"<<c.vehicle_ns.unified_minima_rebuild_slew_db
      <<",\n  \"vehicle_ns_unified_minima_history_depth\":"<<c.vehicle_ns.unified_minima_history_depth
      <<",\n  \"vehicle_ns_unified_conditional_minima_history_depth\":"<<c.vehicle_ns.unified_conditional_minima_history_depth
      <<",\n  \"vehicle_ns_unified_high_snr_bypass_db\":"<<c.vehicle_ns.unified_high_snr_bypass_db
      <<",\n  \"vehicle_ns_unified_high_snr_probability\":"<<c.vehicle_ns.unified_high_snr_probability
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
      <<",\n  \"unified_ns_minima_rotations\":"<<info.unified_ns_minima_rotations
      <<",\n  \"unified_ns_minima_band_db\":"<<info.unified_ns_minima_band_db
      <<",\n  \"unified_ns_conditional_minima_band_db\":"<<info.unified_ns_conditional_minima_band_db
      <<",\n  \"unified_ns_history_minima_band_db\":"<<info.unified_ns_history_minima_band_db
      <<",\n  \"unified_ns_noise_psd_band_db\":"<<info.unified_ns_noise_psd_band_db
      <<",\n  \"unified_ns_smooth_band_db\":"<<info.unified_ns_smooth_band_db
      <<",\n  \"unified_ns_conditional_band_db\":"<<info.unified_ns_conditional_band_db
      <<",\n  \"unified_ns_dd_prior_band_db\":"<<info.unified_ns_dd_prior_band_db
      <<",\n  \"unified_ns_xi_band_db\":"<<info.unified_ns_xi_band_db
      <<",\n  \"unified_ns_gamma_band_db\":"<<info.unified_ns_gamma_band_db
      <<",\n  \"unified_ns_gain_band_db\":"<<info.unified_ns_gain_band_db
      <<",\n  \"unified_ns_minima_rebuild_jump_db\":"<<info.unified_ns_minima_rebuild_jump_db
      <<",\n  \"unified_ns_conditional_minima_rebuild_jump_db\":"<<info.unified_ns_conditional_minima_rebuild_jump_db
      <<",\n  \"unified_ns_high_snr_bypass\":"<<(info.unified_ns_high_snr_bypass?"true":"false")
      <<",\n  \"hybrid_ns_model\":"<<json_string(c.vehicle_ns.student_model_path)
      <<",\n  \"hybrid_ns_state\":"<<info.hybrid_ns_state
      <<",\n  \"hybrid_ns_inferences\":"<<info.hybrid_ns_inferences
      <<",\n  \"hybrid_ns_fallback_frames\":"<<info.hybrid_ns_fallback_frames
      <<",\n  \"hybrid_ns_spp_mix\":"<<c.vehicle_ns.student_spp_mix
      <<",\n  \"hybrid_ns_snr_mix\":"<<c.vehicle_ns.student_snr_mix
      <<",\n  \"hybrid_ns_teacher_in_runtime\":false"
      <<",\n  \"residual_ns_model\":"<<json_string(c.vehicle_ns.residual_model_path)
      <<",\n  \"residual_ns_state\":"<<info.residual_model_state
      <<",\n  \"residual_ns_inferences\":"<<info.residual_model_inferences
      <<",\n  \"residual_ns_fallback_frames\":"<<info.residual_model_fallback_frames
      <<",\n  \"residual_ns_applied_frames\":"<<info.residual_model_applied_frames
      <<",\n  \"residual_ns_parameters\":"<<info.residual_model_parameters
      <<",\n  \"residual_ns_mix\":"<<c.vehicle_ns.residual_model_mix
      <<",\n  \"residual_ns_floor_db\":"<<c.vehicle_ns.residual_model_floor_db
      <<",\n  \"residual_ns_teacher_in_runtime\":false"
      <<",\n  \"hybrid_v2_model\":"<<json_string(c.vehicle_ns.hybrid_v2_model_path)
      <<",\n  \"hybrid_v2_state\":"<<info.hybrid_v2_state
      <<",\n  \"hybrid_v2_inferences\":"<<info.hybrid_v2_inferences
      <<",\n  \"hybrid_v2_fallback_frames\":"<<info.hybrid_v2_fallback_frames
      <<",\n  \"hybrid_v2_applied\":"<<(info.hybrid_v2_applied?"true":"false")
      <<",\n  \"hybrid_v2_applied_frames\":"<<info.hybrid_v2_applied_frames
      <<",\n  \"hybrid_v2_parameters\":"<<info.hybrid_v2_parameters
      <<",\n  \"hybrid_v2_input_dim\":"<<info.hybrid_v2_input_dim
      <<",\n  \"hybrid_v2_mix\":"<<c.vehicle_ns.hybrid_v2_mix
      <<",\n  \"hybrid_v2_floor_db\":"<<c.vehicle_ns.hybrid_v2_floor_db
      <<",\n  \"hybrid_v2_raw_mix\":"<<info.hybrid_v2_raw_mix
      <<",\n  \"hybrid_v2_pitch_strength\":"<<c.vehicle_ns.hybrid_v2_pitch_strength
      <<",\n  \"hybrid_v2_pitch_active\":"<<(info.hybrid_v2_pitch_active?"true":"false")
      <<",\n  \"hybrid_v2_pitch_active_frames\":"<<info.hybrid_v2_pitch_active_frames
      <<",\n  \"hybrid_v2_pitch_fallback_frames\":"<<info.hybrid_v2_pitch_fallback_frames
      <<",\n  \"hybrid_v2_pitch_lag_samples\":"<<info.hybrid_v2_pitch_lag_samples
      <<",\n  \"hybrid_v2_pitch_periodicity\":"<<info.hybrid_v2_pitch_periodicity
      <<",\n  \"hybrid_v2_pitch_latency_samples\":0"
      <<",\n  \"hybrid_v2_runtime_scope\":\"native_omlsa_plus_v2_gain_and_causal_pitch; not pure-NN timing\""
      <<",\n  \"hybrid_v2_teacher_in_runtime\":false"
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
      <<",\n  \"aec_backend\":"<<json_string(itu::aec_backend_report_name(c.aec_backend))
      <<",\n  \"aec_backend_requested\":"<<json_string(itu::aec_backend_id(c.aec_backend))
      <<",\n  \"aec_backend_effective\":"<<json_string(c.aec ? itu::aec_backend_report_name(c.aec_backend) : "disabled")
      <<",\n  \"aec_execution_provider\":"<<json_string(engine.aec_route_diagnostics().execution_provider)
      <<",\n  \"aec_device_id\":"<<json_string(engine.aec_route_diagnostics().device_id)
      <<",\n  \"aec_model_id\":"<<json_string(engine.aec_route_diagnostics().model_id)
      <<",\n  \"aec_model_sha256\":"<<json_string(engine.aec_route_diagnostics().model_sha256)
      <<",\n  \"aec_frontend_sha256\":"<<json_string(engine.aec_route_diagnostics().frontend_sha256)
      <<",\n  \"aec_provider_sha256\":"<<json_string(engine.aec_route_diagnostics().provider_sha256)
      <<",\n  \"aec_dsp_inferences\":"<<engine.aec_route_diagnostics().inferences
      <<",\n  \"aec_dsp_processed_frames\":"<<engine.aec_route_diagnostics().processed_frames
      <<",\n  \"aec_dsp_fallback_frames\":"<<engine.aec_route_diagnostics().fallback_frames
      <<",\n  \"aec_dsp_cycles\":"<<engine.aec_route_diagnostics().cycles
      <<",\n  \"aec_neural_delay_samples\":"<<engine.aec_route_diagnostics().neural_delay_samples
      <<",\n  \"aec_hop_adapter_delay_samples_16k\":"<<engine.aec_route_diagnostics().hop_adapter_delay_samples
      <<",\n  \"aec_algorithm_delay_samples_16k\":"
      <<engine.aec_route_diagnostics().neural_delay_samples+
          engine.aec_route_diagnostics().hop_adapter_delay_samples
      <<",\n  \"aec_model_codec_delay_samples_16k\":"<<(localvqe_aec?256:0)
      <<",\n  \"aec_linear_frontend_delay_samples_16k\":0"
      <<",\n  \"aec_input_block_requirement_samples_16k\":"<<(localvqe_aec?256:0)
      <<",\n  \"aec_localvqe_inferences\":"<<(localvqe_aec?engine.aec_route_diagnostics().inferences:0)
      <<",\n  \"aec_native_hop_samples\":"<<(localvqe_aec?256:0)
      <<",\n  \"aec_native_consumed_samples\":"<<(localvqe_aec?engine.aec_route_diagnostics().inferences*256:0)
      <<",\n  \"aec_declared_source_samples\":"<<c.localvqe_source_samples
      <<",\n  \"aec_localvqe_processed_frames\":"<<(localvqe_aec?engine.aec_route_diagnostics().processed_frames:0)
      <<",\n  \"aec_localvqe_fallback_frames\":"<<(localvqe_aec?engine.aec_route_diagnostics().fallback_frames:0)
      <<",\n  \"aec_processing_mode\":"<<json_string(localvqe_aec?"causal_streaming":computer_pf_cpu?"offline_noncausal":"streaming")
      <<",\n  \"aec_route_processed_frames\":"<<aec_route.processed_frames
      <<",\n  \"aec_route_inferences\":"<<aec_route.inferences
      <<",\n  \"aec_route_fallback_frames\":"<<aec_route.fallback_frames
      <<",\n  \"aec_route_cycles\":"<<aec_route.cycles
      <<",\n  \"aec_cpu_precomputed_pcm\":"<<json_string(complex_pf_cpu_pcm_report_path)
      <<",\n  \"aec_cpu_precomputed_samples\":"<<(complex_pf_cpu_pcm?complex_pf_cpu_pcm->size():0)
      <<",\n  \"aec_cpu_consumed_samples\":"<<(aec_route.processed_frames*c.frame_size)
      <<",\n  \"aec_cpu_offline_whole_file\":"<<(computer_pf_cpu?"true":"false")
      <<",\n  \"aec_cpu_streaming_latency_claimed\":false"
      <<",\n  \"aec_filter_blocks\":"<<c.aec_filter_blocks
      <<",\n  \"aec_tail_ms\":"<<(c.aec_filter_blocks*4)
      <<",\n  \"speex_echo_tail_ms\":"<<c.speex_echo_tail_ms
      <<",\n  \"speex_fused_cpu_learning_rate\":"<<c.speex_fused_cpu_learning_rate
      <<",\n  \"speex_fused_cpu_shadow_taps_ms\":"<<c.speex_fused_cpu_shadow_taps_ms
      <<",\n  \"aec_fused_shadow_weight\":"<<info.aec_fused_shadow_weight
      <<",\n  \"aec_fused_primary_error_power\":"<<info.aec_fused_primary_error_power
      <<",\n  \"aec_fused_shadow_error_power\":"<<info.aec_fused_shadow_error_power
      <<",\n  \"aec_fused_double_talk\":"<<(info.aec_fused_double_talk?"true":"false")
      <<",\n  \"aec_fused_updates\":"<<info.aec_fused_updates
      <<",\n  \"aec_fused_delay_samples\":"<<info.aec_fused_delay_samples
      <<",\n  \"aec_fused_path_changes\":"<<info.aec_fused_path_changes
      <<",\n  \"aec_fused_delay_valid\":"<<(info.aec_fused_delay_valid?"true":"false")
      <<",\n  \"unified_eq_weight\":"<<info.unified_eq_weight
      <<",\n  \"unified_agc_fused_weight\":"<<info.unified_agc_fused_weight
      <<",\n  \"unified_agc_gain_db\":"<<info.unified_agc_gain_db
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
      <<",\n  \"aec_input_conditioning\":\""<<(!c.aec ? "AEC disabled" : localvqe_aec ?
           "LocalVQE native microphone/reference path; no external low-cut, RES or downmix; mono reference required" :
           speex_aec ?
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
          <<(localvqe_aec ? "LocalVQE DAF plus neural mask output before the shared NS/EQ/AGC/DRC/limiter tail; no linear ERLE tap; " :
             speex_aec ? "raw Speex MDF error before RES/dry blend/NS/AGC2; " :
             "WebRTC linear AEC output after optional pre-AEC BF and before nonlinear suppression/NS/AGC2; ")
          <<("conditioned input path")<<"; engineering estimate only\",\n  \"aec_tap_scope\":\""
          <<(c.aec && computer_pf_cpu ?
             "CPU offline Complex_PF PCM replaces the MDF output before the shared post-AEC chain; raw MDF error remains the linear tap; no native shadow, coherence, RES or dry blend" :
             c.aec && localvqe_aec ?
             "LocalVQE causal DAF plus neural-mask output after sample-rate adaptation and before the shared post-AEC chain; nonlinear tap only" :
             c.aec && speex_aec ?
             "Speex MDF/optional RES after aligned near-speech protection; separate raw MDF error is exported as linear tap" :
             c.aec ? "AEC3 output includes internal residual suppression; not a linear ERLE tap" :
             "capture path with AEC disabled; conversion delay retained")
          <<"\",\n  \"dtd_status_available\":false,\n  \"noise_estimation\":\""
          <<(!c.noise_suppression?"disabled":c.ns_stream_process?"Stateful per-frame CPU worker over parent-owned pipes; no native NS or fallback; not a hard real-time guarantee":c.post_ns_pcm?"External offline post-NS FLOAT32 replay; no runtime NS backend":c.noise_backend==itu::NoiseBackend::unified?"Causal unsupervised online fusion of four shared-spectrum NS experts; no labels, future PCM, or external VAD":c.noise_backend==itu::NoiseBackend::super_gaussian?"Super-Gaussian joint MAP H1 gain with IMCRA and shared Gaussian soft evidence; no external VAD":c.noise_backend==itu::NoiseBackend::omlsa_imcra?"Causal OM-LSA with two-stage IMCRA; continuous soft speech probability; no external VAD":c.noise_backend==itu::NoiseBackend::ulunas?"UL-UNAS recurrent spectral model; 16 kHz enhancement bandwidth":c.noise_backend==itu::NoiseBackend::rnnoise?"RNNoise recurrent model; 48 kHz internal":c.noise_backend==itu::NoiseBackend::speex?"SpeexDSP online noise estimator":"WebRTC online adaptive; no user noise-only learning phase")
          <<"\",\n  \"internal_sample_rate\":"<<(c.ns_stream_process ? c.sample_rate :
              core_bridge || c.noise_backend==itu::NoiseBackend::rnnoise ? 48000 : c.sample_rate)
          <<",\n  \"aec_internal_sample_rate\":"<<(!c.aec ? 0 :
              localvqe_aec ? 16000 : speex_aec || c.ns_stream_process ? c.sample_rate :
              c.sample_rate==24000 ? 48000 : c.sample_rate)
          <<",\n  \"ns_internal_sample_rate\":"<<(c.ns_stream_process ? c.sample_rate :
              c.noise_backend==itu::NoiseBackend::ulunas ? 16000 :
              core_bridge || c.noise_backend==itu::NoiseBackend::rnnoise ? 48000 : c.sample_rate)
          <<",\n  \"rate_converter\":\""<<(c.ns_stream_process ? "none" :
              core_bridge || (c.sample_rate!=16000 && c.noise_backend==itu::NoiseBackend::ulunas) ?
              "SpeexDSP quality 8" :
              c.sample_rate==24000 && (c.noise_backend==itu::NoiseBackend::omlsa_imcra ||
                  c.noise_backend==itu::NoiseBackend::super_gaussian ||
                  c.noise_backend==itu::NoiseBackend::unified) && c.automatic_gain ?
              "AGC2 may convert internally; native statistical NS PCM path" : "none")<<"\""
          <<",\n  \"ns_delay_samples\":"<<info.ns_delay_samples
          <<",\n  \"post_low_cut_delay_samples\":"<<info.post_low_cut_delay_samples
          <<",\n  \"dsp_delay_ms\":"<<1000.*info.latency_samples/c.sample_rate
          <<",\n  \"standard_compliance\":false\n}\n";
    if(!report || !csv) throw std::runtime_error("report write failed");
    std::cout<<"wrote "<<prefix<<".{after.wav,aec.wav,linear-aec.wav,bf.wav,frames.csv,json}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
