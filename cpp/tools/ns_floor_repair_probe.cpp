// Standalone, mono NS experiment: no AEC, BF, AGC, EQ or HPF. Shared optional
// causal protection is explicit and disabled in the raw baseline.
// Both backends receive identical PCM16-quantized input. 24 kHz shared mode
// uses identical quality-8 SRC; native24 measures Speex without that bridge.
#include "wav.hpp"
#include "ns_floor_repair_controller.hpp"
#include "ns_repair_alignment.hpp"
#include "speex_ns.hpp"
#include "rate_converter.hpp"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/ns/noise_suppressor.h"
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
    if(argc!=16 && argc!=17) throw std::runtime_error("usage: ns_refinement_stream_probe backend strength input.wav prefix mode dry_max snr_threshold noise_gain startup_ms policy mild_strength slew_ms release_ms quiet_floor_db floor_guard [--pace-10ms]");
    const bool paced=argc==17 && std::string(argv[16])=="--pace-10ms";
    if(argc==17 && !paced) throw std::runtime_error("invalid pacing option");
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
    const bool quiet_enabled=quiet_floor_db> -200;
    if((policy!="legacy" && policy!="aligned" && policy!="parallel" && policy!="dynamic" && policy!="fixedblend" && policy!="parallel_start") ||
       !std::isfinite(slew_ms) || slew_ms<0 || slew_ms>500 || (policy=="dynamic" && backend!="speex"))
      throw std::runtime_error("invalid control policy");
    if((backend=="webrtc" && (mild_strength<0 || mild_strength>3)) ||
       (backend=="speex" && (mild_strength>0 || mild_strength< -40))) throw std::runtime_error("invalid mild strength");
    if(!std::isfinite(dry_max+snr_threshold+noise_gain) || dry_max<0 || dry_max>1 || noise_gain<.2f || noise_gain>1 || snr_threshold<0 || snr_threshold>30 || startup_ms<0 || startup_ms>500)
      throw std::runtime_error("invalid causal NS controller parameters");
    auto wav=itu::read_wav(argv[3]);
    if(wav.samples.empty() || wav.channels!=1 || (wav.sample_rate!=16000 && wav.sample_rate!=24000 && wav.sample_rate!=48000) ||
       (backend!="webrtc" && backend!="speex" && backend!="bypass" && backend!="filterbank") ||
       (mode!="shared" && mode!="native24") ||
       (mode=="native24" && (backend!="speex" || wav.sample_rate!=24000)))
      throw std::runtime_error("invalid rate, channels, backend or mode");
    if((backend=="webrtc" && (strength<0 || strength>3)) ||
       (backend=="speex" && (strength>0 || strength< -40))) throw std::runtime_error("invalid strength");
    webrtc::LogMessage::LogToDebug(webrtc::LS_NONE);
    webrtc::LogMessage::SetLogToStderr(false);
    const unsigned rate=wav.sample_rate,hop=rate/100;
    const unsigned core_rate=rate==24000 && mode=="shared" ? 48000 : rate;
    const unsigned core_hop=core_rate/100;
    const bool policy_enabled=dry_max>0 || noise_gain<1;
    const bool dual=policy=="parallel" || policy=="parallel_start";
    std::unique_ptr<itu::NsFloorRepairController> controller;
    if(policy_enabled) controller=std::make_unique<itu::NsFloorRepairController>(core_rate,dry_max,snr_threshold,noise_gain,startup_ms,release_ms,quiet_floor_db,floor_guard);
    std::unique_ptr<itu::RateConverter> up,down;
    if(core_rate!=rate) {
      up=std::make_unique<itu::RateConverter>(hop,core_hop);
      down=std::make_unique<itu::RateConverter>(core_hop,hop);
    }
    std::unique_ptr<itu::SpeexNs> speex;
    std::unique_ptr<webrtc::NoiseSuppressor> ns;
    std::unique_ptr<webrtc::AudioBuffer> buffer;
    if(backend=="speex") speex=std::make_unique<itu::SpeexNs>(core_rate,strength);
    if(backend=="webrtc") {
      webrtc::NsConfig config;
      config.target_level=static_cast<webrtc::NsConfig::SuppressionLevel>(strength);
      ns=std::make_unique<webrtc::NoiseSuppressor>(config,core_rate,1);
    }
    std::unique_ptr<itu::SpeexNs> mild_speex;
    std::unique_ptr<webrtc::NoiseSuppressor> mild_ns;
    std::unique_ptr<webrtc::AudioBuffer> mild_buffer;
    if(dual && policy_enabled) {
      if(backend=="speex") mild_speex=std::make_unique<itu::SpeexNs>(core_rate,mild_strength);
      else if(backend=="webrtc") {
        webrtc::NsConfig c;c.target_level=static_cast<webrtc::NsConfig::SuppressionLevel>(mild_strength);
        mild_ns=std::make_unique<webrtc::NoiseSuppressor>(c,core_rate,1);
        mild_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
      }
    }
    if(backend=="webrtc" || backend=="filterbank")
      buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    const unsigned core_delay=backend=="speex" ? core_hop :
        backend=="webrtc" ? (core_rate==48000 ? 336 : 96) :
        backend=="filterbank" && core_rate==48000 ? 48 : 0;
    const double src_delay=up ? up->output_delay()*double(rate)/core_rate+down->output_delay() : 0.;
    const unsigned delay=unsigned(std::llround(core_delay*double(rate)/core_rate+src_delay));
    std::unique_ptr<webrtc::AudioBuffer> neutral_buffer;
    if(policy_enabled && backend=="webrtc" && !dual) neutral_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    std::vector<float> neutral(core_hop),dry_history(std::max(1u,core_delay-(backend=="webrtc" && core_rate==48000 ? 48u : 0u)));
    unsigned dry_head=0;
    float previous_dry=0,previous_gain=1;
    itu::NsRepairAlignment aligned(core_delay,core_hop,slew_ms,release_ms>0);
    float dynamic_db=float(strength);
    unsigned strength_changes=0;
    // A bounded, explicitly configured cold-start crossfade. Both NS states
    // keep learning throughout; raw PCM never enters either noise model twice.
    std::unique_ptr<webrtc::AudioBuffer> startup_buffer;
    if((policy=="parallel_start" || quiet_enabled) && backend=="webrtc")
      startup_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    std::vector<float> startup_raw(core_hop),startup_history(dry_history.size());
    unsigned startup_head=0;
    itu::NsRepairAlignment startup_control(core_delay,core_hop,0,release_ms>0 || quiet_enabled);
    float startup_target=0;
    const std::size_t samples=wav.samples.size(),frames=(samples+hop-1)/hop;
    const std::size_t flushed_frames=(samples+delay+hop-1)/hop;
    std::vector<float> input(hop),core(core_hop),output(hop),rendered(flushed_frames*hop);
    std::vector<double> timings;
    timings.reserve(frames);
    std::ofstream csv(prefix+".csv");
    csv<<"frame,time_s,process_ms,confidence,snr_db,input_mix_target,input_gain_target,effective_suppression_db,quiet_mix_target,estimated_noise_floor_db\n";
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
      if(up) {
        if(up->Resample(input.data(),hop,core.data(),core_hop)) throw std::runtime_error("input SRC failed");
      } else std::copy(input.begin(),input.end(),core.begin());
      if(controller) {
        controller->observe(core.data());
        if(policy=="parallel_start" || quiet_enabled) {
          std::copy(core.begin(),core.end(),startup_raw.begin());
          if(startup_buffer) {
            const float* in[]={startup_raw.data()};float* out[]={startup_raw.data()};
            startup_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
            if(core_rate==48000) startup_buffer->SplitIntoFrequencyBands();
            if(core_rate==48000) startup_buffer->MergeFrequencyBands();
            startup_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
          }
          for(unsigned i=0;i<core_hop;++i) {
            const float delayed=startup_history[startup_head];startup_history[startup_head]=startup_raw[i];startup_raw[i]=delayed;
            startup_head=(startup_head+1)%startup_history.size();
          }
          // Immediate protection attack; smooth release after the fixed bound.
          startup_target=policy=="parallel_start" && frame*10 < unsigned(startup_ms) ? 1.f : startup_target*std::exp(-10.f/std::max(10.f,slew_ms));
          startup_control.begin(std::max(startup_target,controller->quiet()),1.f);
        }
        std::copy(core.begin(),core.end(),neutral.begin());
        if(neutral_buffer && policy!="dynamic") {
          const float* in[]={neutral.data()};float* out[]={neutral.data()};
          neutral_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
          if(core_rate==48000) neutral_buffer->SplitIntoFrequencyBands();
          if(core_rate==48000) neutral_buffer->MergeFrequencyBands();
          neutral_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
        }
        if(core_delay && !dual && policy!="dynamic") for(unsigned i=0;i<core_hop;++i) {
          const float delayed=dry_history[dry_head];dry_history[dry_head]=neutral[i];neutral[i]=delayed;
          dry_head=(dry_head+1)%dry_history.size();
        }
      }
      if(controller && policy=="dynamic") {
        const float target=float(strength)+(mild_strength-strength)*controller->dry();
        dynamic_db+=(1.f-std::exp(-10.f/std::max(10.f,slew_ms)))*(target-dynamic_db);
        const int db=int(std::lround(dynamic_db));
        speex->suppression(db);++strength_changes;
      }
      if(mild_speex) mild_speex->process(neutral.data());
      if(mild_buffer) {
        const float* in[]={neutral.data()};float* out[]={neutral.data()};
        mild_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
        if(core_rate==48000) mild_buffer->SplitIntoFrequencyBands();
        mild_ns->Analyze(*mild_buffer);mild_ns->Process(mild_buffer.get());
        if(core_rate==48000) mild_buffer->MergeFrequencyBands();
        mild_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
      }
      if(speex) speex->process(core.data());
      if(buffer) {
        const float* in[]={core.data()};float* out[]={core.data()};
        buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
        if(core_rate==48000) buffer->SplitIntoFrequencyBands();
        if(ns) {ns->Analyze(*buffer); ns->Process(buffer.get());}
        if(core_rate==48000) buffer->MergeFrequencyBands();
        buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
      }
      if(controller) {
        const float next_dry=policy=="fixedblend" ? dry_max : controller->dry(),next_gain=controller->gain();
        aligned.begin(policy=="dynamic" ? 0.f : next_dry,next_gain);
        for(unsigned i=0;i<core_hop;++i) {
          const float f=float(i+1)/core_hop;
          float mix=previous_dry+f*(next_dry-previous_dry),gain=previous_gain+f*(next_gain-previous_gain);
          if(policy!="legacy") aligned.sample(i,mix,gain);
          core[i]=gain*((1-mix)*core[i]+mix*neutral[i]);
          if(policy=="parallel_start" || quiet_enabled) {
            float start_mix,unused_gain;startup_control.sample(i,start_mix,unused_gain);
            core[i]=(1-start_mix)*core[i]+start_mix*startup_raw[i];
          }
        }
        previous_dry=next_dry;previous_gain=next_gain;
      }
      if(down) {
        if(down->Resample(core.data(),core_hop,output.data(),hop)) throw std::runtime_error("output SRC failed");
      } else std::copy(core.begin(),core.end(),output.begin());
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(paced && frame<frames && std::chrono::steady_clock::now()>stream_start+std::chrono::milliseconds(10*(frame+1))) ++deadline_misses;
      if(frame==0) first_ms=ms;
      if(frame<frames) {timings.push_back(ms);csv<<frame<<','<<frame*.01<<','<<ms<<','<<(controller ? controller->confidence() : 0)<<','<<(controller ? controller->snr_db() : 0)<<','<<previous_dry<<','<<previous_gain<<','<<dynamic_db<<','<<(controller ? controller->quiet() : 0)<<','<<(controller ? controller->noise_floor_db() : -300)<<'\n';}
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
    std::ofstream report(prefix+".json");
    report<<"{\"floor_guard\":"<<floor_guard<<",\"repair_release_ms\":"<<release_ms<<",\"quiet_floor_db\":"<<quiet_floor_db<<",\"backend\":\""<<backend<<"\",\"strength\":"<<strength
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
      <<",\"paced\":"<<(paced ? "true" : "false")<<",\"paced_deadline_misses\":"<<deadline_misses
      <<",\"aec\":false,\"agc\":false,\"hpf\":false,\"speech_bypass\":"<<(policy_enabled ? "true" : "false")
      <<",\"policy_enabled\":"<<(policy_enabled ? "true" : "false")<<",\"dry_max\":"<<dry_max<<",\"snr_threshold_db\":"<<snr_threshold<<",\"noise_gain\":"<<noise_gain<<",\"startup_ms\":"<<startup_ms<<",\"control_policy\":\""<<policy<<"\",\"mild_strength\":"<<mild_strength<<",\"slew_ms\":"<<slew_ms<<",\"suppression_updates\":"<<strength_changes<<",\"evidence_scope\":\"current and previous input PCM only; no labels, clean target, external VAD or oracle\"}\n";
    if(!report || !csv) throw std::runtime_error("report write failed");
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n'; return 1;
  }
}
