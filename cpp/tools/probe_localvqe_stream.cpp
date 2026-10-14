#include "wav.hpp"
#include "rate_converter.hpp"
#include "fixed_hop_adapter.hpp"
#include "aec_safety_mix.hpp"
#include "itu/audio_engine.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class Api {
 public:
  explicit Api(const std::string& path) {
    library_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!library_) throw std::runtime_error(dlerror());
  }
  ~Api() {if(library_) dlclose(library_);}
  template<class T> T symbol(const char* name) {
    auto address=dlsym(library_,name);
    if(!address) throw std::runtime_error(std::string("missing LocalVQE symbol: ")+name);
    return reinterpret_cast<T>(address);
  }
 private:void* library_=nullptr;
};
// Pinned GGUF v3 C ABI, from ggml/include/gguf.h. Read metadata without
// tensor allocation. The loaded SDK makes the same standalone decision;
// filenames cannot establish the codec's latency contract.
struct GgufInitParams {bool no_alloc;void** context;};
struct ModelContract {bool frontend=false,gtcrn=false;};
ModelContract model_contract(Api& api,const std::string& model) {
  auto init=api.symbol<void*(*)(const char*,GgufInitParams)>("gguf_init_from_file");
  auto free=api.symbol<void(*)(void*)>("gguf_free");
  auto find=api.symbol<std::int64_t(*)(const void*,const char*)>("gguf_find_key");
  auto type=api.symbol<int(*)(const void*,std::int64_t)>("gguf_get_kv_type");
  auto boolean=api.symbol<bool(*)(const void*,std::int64_t)>("gguf_get_val_bool");
  auto string=api.symbol<const char*(*)(const void*,std::int64_t)>("gguf_get_val_str");
  std::unique_ptr<void,decltype(free)> metadata(init(model.c_str(),{true,nullptr}),free);
  if(!metadata) throw std::runtime_error("model GGUF metadata failed");
  const auto arch=find(metadata.get(),"general.architecture");
  if(arch>=0) {
    if(type(metadata.get(),arch)!=8) throw std::runtime_error("architecture metadata must be string");
    if(std::string(string(metadata.get(),arch))=="gtcrn") return {false,true};
  }
  const auto standalone=find(metadata.get(),"localvqe.daf.standalone");
  if(standalone<0) return {};
  if(type(metadata.get(),standalone)!=7) throw std::runtime_error("standalone metadata must be bool");
  return {boolean(metadata.get(),standalone),false};
}
}

// Diagnostic only: no batch API, bulk delay prime, gate, whole-clip normalization,
// or guessed downmix. GTCRN retains its upstream causal RMS EMA, undone before
// synthesis. One model state per file, including its first-call initialization.
int main(int argc,char** argv) {
 try {
  std::string library,model,mic_path,ref_path,prefix,post_ns,post_model;bool frontend=false,safety=false;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];
    if(key=="--frontend-only") {frontend=true;continue;}
    if(key=="--aec-divergence-guard") {safety=true;continue;}
    if(i+1==argc) throw std::runtime_error("missing argument");
    const std::string value=argv[++i];
    if(key=="--library") library=value;else if(key=="--model") model=value;
    else if(key=="--mic") mic_path=value;else if(key=="--echo") ref_path=value;
    else if(key=="--out") prefix=value;else if(key=="--post-ns") post_ns=value;
    else if(key=="--post-model") post_model=value;else throw std::runtime_error("unknown argument: "+key);
  }
  if(library.empty() || model.empty() || mic_path.empty() || ref_path.empty() || prefix.empty())
    throw std::runtime_error("requires --library --model --mic --echo --out [--frontend-only]");
  auto mic=itu::read_wav(mic_path),ref=itu::read_wav(ref_path);
  if(mic.samples.empty()) throw std::runtime_error("requires nonempty microphone PCM");
  if(mic.channels!=1 || ref.channels!=1 || mic.sample_rate!=ref.sample_rate || mic.samples.size()!=ref.samples.size())
    throw std::runtime_error("requires one microphone and one synchronous reference; multirender cannot be downmixed");
  const unsigned rate=mic.sample_rate,h=rate/100;
  if(rate!=16000 && rate!=24000 && rate!=48000) throw std::runtime_error("requires 16/24/48 kHz");
  for(const auto* signal:{&mic.samples,&ref.samples}) for(float x:*signal)
    if(!std::isfinite(x) || std::abs(x)>1.f) throw std::runtime_error("invalid PCM");
  Api api(library);
  const auto contract=model_contract(api,model);
  if(frontend!=contract.frontend)
    throw std::runtime_error("--frontend-only must match GGUF standalone metadata");
  if(safety && !frontend) throw std::runtime_error("divergence guard requires subtractive AEC-only model");
  using Handle=std::uintptr_t;
  auto options=api.symbol<Handle(*)()>("localvqe_options_new")();
  auto options_free=api.symbol<void(*)(Handle)>("localvqe_options_free");
  auto set_model=api.symbol<int(*)(Handle,const char*)>("localvqe_options_set_model_path");
  auto set_threads=api.symbol<int(*)(Handle,int)>("localvqe_options_set_threads");
  auto create=api.symbol<Handle(*)(Handle)>("localvqe_new_with_options");
  if(!options) throw std::runtime_error("LocalVQE options failed");
  const bool configured=set_model(options,model.c_str())==0 && set_threads(options,1)==0;
  Handle context=configured ? create(options) : 0;options_free(options);
  auto release=api.symbol<void(*)(Handle)>("localvqe_free");
  auto deleter=[&](void* p){release(reinterpret_cast<Handle>(p));};
  std::unique_ptr<void,decltype(deleter)> guard(reinterpret_cast<void*>(context),deleter);
  if(!context) throw std::runtime_error("LocalVQE load failed");
  auto process=api.symbol<int(*)(Handle,const float*,const float*,int,float*)>("localvqe_process_frame_f32");
  auto get_rate=api.symbol<int(*)(Handle)>("localvqe_sample_rate");
  auto get_hop=api.symbol<int(*)(Handle)>("localvqe_hop_length");
  auto gate=api.symbol<int(*)(Handle,int,float)>("localvqe_set_noise_gate");
  if(get_rate(context)!=16000 || get_hop(context)!=256 || gate(context,0,-90.f)!=0)
    throw std::runtime_error("LocalVQE streaming contract mismatch");
  std::unique_ptr<itu::RateConverter> mic_down,ref_down,output_up;
  if(rate!=16000) {
    mic_down=std::make_unique<itu::RateConverter>(h,160);
    ref_down=std::make_unique<itu::RateConverter>(h,160);
    output_up=std::make_unique<itu::RateConverter>(160,h);
  }
  const unsigned model_delay=frontend ? 0 : 256;
  std::unique_ptr<itu::AudioEngine> post;
  unsigned post_delay=0;
  if(!post_ns.empty()) {
    if(!frontend) throw std::runtime_error("post NS requires the AEC-only model");
    auto config=itu::vehicle_stream_config();config.aec=false;config.noise_transition_control=true;
    if(post_ns=="webrtc") config.noise_backend=itu::NoiseBackend::webrtc;
    else if(post_ns=="rnnoise") config.noise_backend=itu::NoiseBackend::rnnoise;
    else if(post_ns=="speex") config.noise_backend=itu::NoiseBackend::speex;
    else if(post_ns=="ulunas") {config.noise_backend=itu::NoiseBackend::ulunas;config.neural_model_path=post_model;}
    else throw std::runtime_error("unknown post NS");
    post=std::make_unique<itu::AudioEngine>(config);
    std::array<float,160> silence{},output{};itu::FrameInfo info;
    if(post->process(silence.data(),160,nullptr,0,output.data(),160,info)!=itu::Status::ok)
      throw std::runtime_error("post NS latency initialization failed");
    post_delay=info.latency_samples;post->reset();
  }
  const unsigned linear_delay=256+(mic_down ? mic_down->output_delay() : 0);
  const unsigned delay=unsigned(std::lround((linear_delay+model_delay+post_delay)*double(rate)/16000+
      (output_up ? output_up->output_delay() : 0)));
  const auto n=mic.samples.size();const auto frames=(n+delay+h-1)/h;
  std::vector<float> after(frames*h),linear(frontend ? frames*160 : 0),times;times.reserve(frames);
  std::array<float,480> capture{},render{},output{};
  std::array<float,160> cm{},cr{},co{};itu::FixedHopAdapter adapter;itu::AecSafetyMix safety_mix;
  std::ofstream csv(prefix+".frames.csv");csv<<"frame,time_s,process_ms,inferences,queued,pending\n";
  std::uint64_t io_limited_samples=0;
  for(std::size_t k=0;k<frames;++k) {
    for(unsigned i=0;i<h;++i) {
      const auto index=k*h+i;capture[i]=index<n ? mic.samples[index] : 0;render[i]=index<n ? ref.samples[index] : 0;
    }
    const auto start=std::chrono::steady_clock::now();
    if(mic_down) {
      if(mic_down->Resample(capture.data(),h,cm.data(),160) || ref_down->Resample(render.data(),h,cr.data(),160))
        throw std::runtime_error("input conversion failed");
    } else {std::copy_n(capture.data(),160,cm.data());std::copy_n(render.data(),160,cr.data());}
    if(!adapter.process(cm.data(),cr.data(),co.data(),[&](const float* m,const float* r,float* o){
      if(process(context,m,r,256,o)!=0) return false;
      if(safety) return safety_mix.process(m,o,256);
      // Keep invalid inference out of the FIFO, post NS and resampler.
      for(unsigned i=0;i<256;++i) if(!std::isfinite(o[i])) return false;
      return true;})) throw std::runtime_error("streaming inference/FIFO failed");
    if(frontend) std::copy(co.begin(),co.end(),linear.begin()+k*160);
    if(post) {
      itu::FrameInfo info;
      const auto status=post->process(co.data(),160,nullptr,0,cm.data(),160,info);
      if(status!=itu::Status::ok || info.latency_samples!=post_delay) {
        float peak=0;for(float v:co) peak=std::max(peak,std::abs(v));
        throw std::runtime_error("post NS failed at frame "+std::to_string(k)+" status="+
          std::to_string(int(status))+" raw AEC peak="+std::to_string(peak)+" backend="+std::to_string(info.backend_error));
      }
      std::copy(cm.begin(),cm.end(),co.begin());
    }
    if(output_up) {
      if(output_up->Resample(co.data(),160,output.data(),h)) throw std::runtime_error("output conversion failed");
    } else std::copy_n(co.data(),160,output.data());
    if(safety && post) for(unsigned i=0;i<h;++i) {
      if(std::abs(output[i])>.98f) {output[i]=std::clamp(output[i],-.98f,.98f);++io_limited_samples;}
    }
    const float ms=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(k*h<n) times.push_back(ms);
    for(unsigned i=0;i<h;++i) {
      if(!std::isfinite(output[i])) throw std::runtime_error("nonfinite model output");
      after[k*h+i]=output[i];
    }
    csv<<k<<','<<double(k*h)/rate<<','<<ms<<','<<adapter.inferences()<<','<<adapter.queued()<<','<<adapter.pending()<<'\n';
  }
  after.resize(n+delay);itu::write_wav(prefix+".after.wav",rate,1,after);
  if(frontend) {linear.resize(n*16000/rate+linear_delay);itu::write_wav(prefix+".linear-aec.wav",16000,1,linear);}
  double sum=0;unsigned over=0;for(float ms:times) {sum+=ms;over+=ms>10;}
  std::sort(times.begin(),times.end());
  std::ofstream report(prefix+".json");report<<"{\n  \"sample_rate\":"<<rate
    <<",\n  \"aec_internal_sample_rate\":16000,\n  \"microphones\":1,\n  \"reference_channels\":1"
    <<",\n  \"dsp_delay_samples\":"<<delay<<",\n  \"model_codec_delay_samples_16k\":"<<model_delay
    <<",\n  \"hop_adapter_delay_samples_16k\":256,\n  \"noise_gate_enabled\":false"
    <<",\n  \"post_ns_delay_samples_16k\":"<<post_delay<<",\n  \"post_ns_backend\":\""<<post_ns<<"\""
    <<",\n  \"batch_delay_prime\":false,\n  \"threads\":1,\n  \"aec_backend\":\"localvqe_stream\""
    <<",\n  \"model_frontend_only\":"<<(frontend ? "true" : "false")
    <<",\n  \"model_metadata_checked\":true"
    <<",\n  \"model_arch\":\""<<(contract.gtcrn ? "gtcrn" : "localvqe")<<"\""
    <<",\n  \"causal_hop_rms_normalization\":"<<(contract.gtcrn ? "true" : "false")
    <<",\n  \"aec_divergence_guard\":"<<(safety ? "true" : "false")
    <<",\n  \"aec_protected_hops\":"<<safety_mix.protected_hops()
    <<",\n  \"model_nonfinite_output_hops\":"<<safety_mix.nonfinite_hops()
    <<",\n  \"aec_raw_peak_scope\":\"finite residual samples; nonfinite hops counted separately\""
    <<",\n  \"aec_guard_minimum_mix\":"<<safety_mix.minimum_mix()<<",\n  \"aec_raw_peak\":"<<safety_mix.raw_peak()
    <<",\n  \"post_conversion_limiter\":"<<(safety && post ? "true" : "false")
    <<",\n  \"post_conversion_limited_samples\":"<<io_limited_samples
    <<",\n  \"process_p95_ms\":"<<times[std::size_t(std::ceil(times.size()*.95))-1]
    <<",\n  \"process_max_ms\":"<<times.back()<<",\n  \"process_rtf\":"<<sum/(times.size()*10)
    <<",\n  \"process_frames_over_10ms\":"<<over
    <<",\n  \"linear_aec_available\":"<<(frontend ? "true" : "false")
    <<",\n  \"linear_aec_delay_samples\":"<<linear_delay<<",\n  \"linear_aec_sample_rate\":16000"
    <<",\n  \"input_samples_per_channel\":"<<n<<",\n  \"standard_compliance\":false\n}\n";
  if(!csv || !report) throw std::runtime_error("report failed");return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
