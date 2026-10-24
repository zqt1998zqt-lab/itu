// Standalone, mono NS experiment: no AEC, BF, AGC, EQ or HPF. Shared optional
// causal protection is explicit and disabled in the raw baseline.
// Both backends receive identical PCM16-quantized input. 24 kHz shared mode
// uses identical quality-8 SRC; native24 measures Speex without that bridge.
#include "wav.hpp"
#include "ns_tuning_controller.hpp"
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
    if(argc!=10 && argc!=11) throw std::runtime_error("usage: ns_tuning_probe backend strength input.wav prefix mode dry_max snr_threshold noise_gain startup_ms [--pace-10ms]");
    const bool paced=argc==11 && std::string(argv[10])=="--pace-10ms";
    if(argc==11 && !paced) throw std::runtime_error("invalid pacing option");
    const std::string backend=argv[1],prefix=argv[4],mode=argv[5];
    const int strength=std::stoi(argv[2]);
    const float dry_max=std::stof(argv[6]),snr_threshold=std::stof(argv[7]),noise_gain=std::stof(argv[8]);
    const int startup_ms=std::stoi(argv[9]);
    if(!std::isfinite(dry_max+snr_threshold+noise_gain) || dry_max<0 || dry_max>1 || noise_gain<.2f || noise_gain>1 || snr_threshold<0 || snr_threshold>30 || startup_ms<0 || startup_ms>200)
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
    std::unique_ptr<itu::NsTuningController> controller;
    if(policy_enabled) controller=std::make_unique<itu::NsTuningController>(core_rate,dry_max,snr_threshold,noise_gain,startup_ms);
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
    if(backend=="webrtc" || backend=="filterbank")
      buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    const unsigned core_delay=backend=="speex" ? core_hop :
        backend=="webrtc" ? (core_rate==48000 ? 336 : 96) :
        backend=="filterbank" && core_rate==48000 ? 48 : 0;
    const double src_delay=up ? up->output_delay()*double(rate)/core_rate+down->output_delay() : 0.;
    const unsigned delay=unsigned(std::llround(core_delay*double(rate)/core_rate+src_delay));
    std::unique_ptr<webrtc::AudioBuffer> neutral_buffer;
    if(policy_enabled && backend=="webrtc") neutral_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    std::vector<float> neutral(core_hop),dry_history(std::max(1u,core_delay-(backend=="webrtc" && core_rate==48000 ? 48u : 0u)));
    unsigned dry_head=0;
    float previous_dry=0,previous_gain=1;
    const std::size_t samples=wav.samples.size(),frames=(samples+hop-1)/hop;
    const std::size_t flushed_frames=(samples+delay+hop-1)/hop;
    std::vector<float> input(hop),core(core_hop),output(hop),rendered(flushed_frames*hop);
    std::vector<double> timings;
    timings.reserve(frames);
    std::ofstream csv(prefix+".csv");
    csv<<"frame,time_s,process_ms,confidence,snr_db,dry_mix,noise_gain\n";
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
        std::copy(core.begin(),core.end(),neutral.begin());
        if(neutral_buffer) {
          const float* in[]={neutral.data()};float* out[]={neutral.data()};
          neutral_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
          if(core_rate==48000) neutral_buffer->SplitIntoFrequencyBands();
          if(core_rate==48000) neutral_buffer->MergeFrequencyBands();
          neutral_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
        }
        if(core_delay) for(unsigned i=0;i<core_hop;++i) {
          const float delayed=dry_history[dry_head];dry_history[dry_head]=neutral[i];neutral[i]=delayed;
          dry_head=(dry_head+1)%dry_history.size();
        }
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
        const float next_dry=controller->dry(),next_gain=controller->gain();
        for(unsigned i=0;i<core_hop;++i) {
          const float f=float(i+1)/core_hop,mix=previous_dry+f*(next_dry-previous_dry),gain=previous_gain+f*(next_gain-previous_gain);
          core[i]=gain*((1-mix)*core[i]+mix*neutral[i]);
        }
        previous_dry=next_dry;previous_gain=next_gain;
      }
      if(down) {
        if(down->Resample(core.data(),core_hop,output.data(),hop)) throw std::runtime_error("output SRC failed");
      } else std::copy(core.begin(),core.end(),output.begin());
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(paced && frame<frames && std::chrono::steady_clock::now()>stream_start+std::chrono::milliseconds(10*(frame+1))) ++deadline_misses;
      if(frame==0) first_ms=ms;
      if(frame<frames) {timings.push_back(ms);csv<<frame<<','<<frame*.01<<','<<ms<<','<<(controller ? controller->confidence() : 0)<<','<<(controller ? controller->snr_db() : 0)<<','<<previous_dry<<','<<previous_gain<<'\n';}
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
    report<<"{\"backend\":\""<<backend<<"\",\"strength\":"<<strength
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
      <<",\"policy_enabled\":"<<(policy_enabled ? "true" : "false")<<",\"dry_max\":"<<dry_max<<",\"snr_threshold_db\":"<<snr_threshold<<",\"noise_gain\":"<<noise_gain<<",\"startup_ms\":"<<startup_ms<<",\"evidence_scope\":\"current and previous input PCM only; no labels, clean target, external VAD or oracle\"}\n";
    if(!report || !csv) throw std::runtime_error("report write failed");
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n'; return 1;
  }
}
