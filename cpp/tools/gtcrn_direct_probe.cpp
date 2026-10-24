// Isolated, native-16-kHz GTCRN comparator. Not enabled in AudioEngine.
// The model is the pinned upstream streaming graph, with all recurrent caches
// retained for one recording. No gain normalization, gate, SRC, or extra NS.
#include "wav.hpp"
#include "fft.hpp"
#include "onnxruntime_c_api.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
class Model {
 public:
  explicit Model(const std::string& path) : api_(OrtGetApiBase()->GetApi(ORT_API_VERSION)) {
    if (!api_) throw std::runtime_error("ORT C API unavailable");
    check(api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR,"itu-gtcrn-direct",&env_));
    check(api_->CreateSessionOptions(&options_));
    check(api_->SetIntraOpNumThreads(options_,1));
    check(api_->SetInterOpNumThreads(options_,1));
    check(api_->SetSessionExecutionMode(options_,ORT_SEQUENTIAL));
    check(api_->SetSessionGraphOptimizationLevel(options_,ORT_ENABLE_ALL));
    check(api_->CreateSession(env_,path.c_str(),options_,&session_));
    check(api_->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory_));
    const std::array<std::vector<int64_t>,4> shapes{{{1,257,1,2},{2,1,16,16,33},{2,3,1,1,16},{2,1,33,16}}};
    for (unsigned j=0;j<4;++j) {
      std::size_t count=1; for (auto n:shapes[j]) count*=n;
      input_data_[j].resize(count); output_data_[j].resize(count);
      check(api_->CreateTensorWithDataAsOrtValue(memory_,input_data_[j].data(),count*sizeof(float),
          shapes[j].data(),shapes[j].size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&inputs_[j]));
      check(api_->CreateTensorWithDataAsOrtValue(memory_,output_data_[j].data(),count*sizeof(float),
          shapes[j].data(),shapes[j].size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&outputs_[j]));
    }
  }
  ~Model() {
    for (auto p:inputs_) if(p) api_->ReleaseValue(p);
    for (auto p:outputs_) if(p) api_->ReleaseValue(p);
    if(memory_) api_->ReleaseMemoryInfo(memory_);
    if(session_) api_->ReleaseSession(session_);
    if(options_) api_->ReleaseSessionOptions(options_);
    if(env_) api_->ReleaseEnv(env_);
  }
  void process(const float* spectrum,float* enhanced) {
    std::copy_n(spectrum,514,input_data_[0].data());
    const char* names[]{"mix","conv_cache","tra_cache","inter_cache"};
    const char* out_names[]{"enh","conv_cache_out","tra_cache_out","inter_cache_out"};
    const OrtValue* values[]{inputs_[0],inputs_[1],inputs_[2],inputs_[3]};
    check(api_->Run(session_,nullptr,names,values,4,out_names,4,outputs_.data()));
    std::copy_n(output_data_[0].data(),514,enhanced);
    for(unsigned j=1;j<4;++j) std::swap(inputs_[j],outputs_[j]);
  }
 private:
  void check(OrtStatus* status) {
    if(status) {const std::string message=api_->GetErrorMessage(status);api_->ReleaseStatus(status);throw std::runtime_error(message);}
  }
  const OrtApi* api_;
  OrtEnv* env_=nullptr; OrtSessionOptions* options_=nullptr;
  OrtSession* session_=nullptr; OrtMemoryInfo* memory_=nullptr;
  std::array<std::vector<float>,4> input_data_,output_data_;
  std::array<OrtValue*,4> inputs_{},outputs_{};
};

class Stream {
 public:
  // A 256-sample inference block under 160-sample callbacks needs at least
  // 256-gcd(256,160)=224 samples of packetization delay. Together with WOLA's
  // 256-sample signal delay this is a fixed 480 samples (30 ms), not a fitted lag.
  static constexpr unsigned callback=160,hop=256,packet_delay=224,delay=480;
  explicit Stream(const std::string& model) : fft_(512) {
    if(!model.empty()) model_=std::make_unique<Model>(model);
    for(unsigned i=0;i<512;++i) window_[i]=std::sin(3.14159265358979323846*i/512.);
  }
  void process(const float* in,float* out) {
    std::copy_n(in,callback,pending_.data()+pending_count_);pending_count_+=callback;
    if(pending_count_>=hop) {
      for(unsigned i=0;i<hop;++i) {fft_data_[i]=past_[i]*window_[i];fft_data_[i+hop]=pending_[i]*window_[i+hop];past_[i]=pending_[i];}
      fft_.run(fft_data_.data());
      for(unsigned k=0;k<=hop;++k) {spectrum_[2*k]=fft_data_[k].real();spectrum_[2*k+1]=fft_data_[k].imag();}
      if(model_) model_->process(spectrum_.data(),enhanced_.data());
      else enhanced_=spectrum_; // Used only by --self-test; never a quality run.
      ++inferences;
      fft_data_[0]={enhanced_[0],0};fft_data_[hop]={enhanced_[2*hop],0};
      for(unsigned k=1;k<hop;++k) {fft_data_[k]={enhanced_[2*k],enhanced_[2*k+1]};fft_data_[512-k]=std::conj(fft_data_[k]);}
      fft_.run(fft_data_.data(),true);
      for(unsigned i=0;i<hop;++i) {
        const float value=tail_[i]+fft_data_[i].real()*window_[i];
        if(!std::isfinite(value)) throw std::runtime_error("nonfinite GTCRN output");
        if(queued_>=queue_.size()) throw std::runtime_error("GTCRN queue overflow");
        queue_[(head_+queued_)%queue_.size()]=value;++queued_;
        tail_[i]=fft_data_[i+hop].real()*window_[i+hop];
      }
      pending_count_-=hop;
      std::move(pending_.begin()+hop,pending_.begin()+hop+pending_count_,pending_.begin());
    }
    if(queued_<callback) throw std::runtime_error("GTCRN queue underrun");
    for(unsigned i=0;i<callback;++i) out[i]=queue_[(head_+i)%queue_.size()];
    head_=(head_+callback)%queue_.size();queued_-=callback;
  }
  unsigned inferences=0;
 private:
  std::unique_ptr<Model> model_;
  itu::detail::Fft fft_;
  std::array<float,512> pending_{},window_{};
  std::array<float,256> past_{},tail_{};
  std::array<std::complex<float>,512> fft_data_{};
  std::array<float,514> spectrum_{},enhanced_{};
  std::array<float,1024> queue_{};
  unsigned pending_count_=0,head_=0,queued_=packet_delay;
};

void self_test() {
  float maximum=0;
  for(unsigned length:{1u,17u,159u,160u,255u,256u,321u,6399u}) {
    Stream s("");std::vector<float> x(length);
    for(unsigned i=0;i<length;++i) x[i]=.15f*std::sin(.131f*i)+.07f*std::cos(.073f*i);
    x[0]=.75f;x.back()=-.8f;
    const unsigned frames=(length+Stream::delay+159)/160;
    std::vector<float> y(frames*160);std::array<float,160> in{};
    for(unsigned k=0;k<frames;++k) {
      for(unsigned j=0;j<160;++j) in[j]=k*160+j<length ? x[k*160+j]:0;
      s.process(in.data(),y.data()+k*160);
    }
    for(unsigned i=0;i<length;++i) maximum=std::max(maximum,std::abs(x[i]-y[i+Stream::delay]));
  }
  if(maximum>3e-6f) throw std::runtime_error("identity frontend/flush/delay check failed");
  std::cout<<"{\"passed\":true,\"lengths\":8,\"delay_samples\":480,\"maximum_error\":"<<maximum<<"}\n";
}
}

int main(int argc,char** argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--self-test") {self_test();return 0;}
  std::string mic,model,out,dump;
  for(int i=1;i<argc;i+=2) {
    if(i+1==argc) throw std::runtime_error("missing option value");
    const std::string k=argv[i],v=argv[i+1];
    if(k=="--mic") mic=v;else if(k=="--model") model=v;else if(k=="--out") out=v;
    else if(k=="--dump-f32") dump=v;else throw std::runtime_error("unknown option: "+k);
  }
  if(mic.empty() || model.empty() || out.empty()) throw std::runtime_error("requires --mic --model --out");
  const auto x=itu::read_wav(mic);
  if(x.sample_rate!=16000 || x.channels!=1 || x.samples.empty()) throw std::runtime_error("requires nonempty native mono 16 kHz");
  Stream stream(model);const auto n=x.samples.size(),frames=(n+159)/160,count=(n+Stream::delay+159)/160;
  std::vector<float> rendered(count*160);std::vector<double> times;times.reserve(frames);
  std::array<float,160> in{};std::size_t clipped=0;double first_inference=-1;
  std::ofstream trace(out+".csv");trace<<"frame,time_s,process_ms,inferences\n";
  for(std::size_t t=0;t<count;++t) {
    for(unsigned i=0;i<160;++i) in[i]=t*160+i<n ? std::lround(std::clamp(x.samples[t*160+i]*32768.f,-32768.f,32767.f))/32768.f:0;
    const auto before_inferences=stream.inferences;
    const auto start=std::chrono::steady_clock::now();stream.process(in.data(),rendered.data()+t*160);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(before_inferences==0 && stream.inferences>0) first_inference=ms;
    if(t<frames) {times.push_back(ms);trace<<t<<','<<t*.01<<','<<ms<<','<<stream.inferences<<'\n';}
    for(unsigned i=0;i<160;++i) clipped+=std::abs(rendered[t*160+i])>=.999f;
  }
  rendered.resize(n+Stream::delay);itu::write_wav(out+".after.wav",16000,1,rendered);
  if(!dump.empty()) {std::ofstream f(dump,std::ios::binary);f.write(reinterpret_cast<const char*>(rendered.data()),rendered.size()*sizeof(float));if(!f) throw std::runtime_error("raw output write failed");}
  const double first=times.front(),mean=std::accumulate(times.begin(),times.end(),0.)/times.size();
  std::sort(times.begin(),times.end());auto q=[&](double p){return times[std::min(times.size()-1,std::size_t(p*(times.size()-1)))];};
  std::ofstream report(out+".json");
  report<<"{\"sample_rate\":16000,\"core_rate\":16000,\"src_delay_samples\":0,\"delay_samples\":480"
    <<",\"wola_delay_samples\":256,\"packet_delay_samples\":224,\"hop_samples\":256,\"callback_samples\":160"
    <<",\"samples\":"<<n<<",\"frames\":"<<frames<<",\"inferences\":"<<stream.inferences
    <<",\"runtime\":\"onnxruntime-c-api-cpu-single-thread\",\"identity_model\":false,\"external_vad\":false"
    <<",\"ort_version\":\""<<OrtGetApiBase()->GetVersionString()<<"\",\"first_inference_process_ms\":"<<first_inference
    <<",\"first_process_ms\":"<<first<<",\"mean_process_ms\":"<<mean<<",\"p50_process_ms\":"<<q(.5)
    <<",\"p95_process_ms\":"<<q(.95)<<",\"p99_process_ms\":"<<q(.99)<<",\"max_process_ms\":"<<times.back()
    <<",\"over_10ms\":"<<std::count_if(times.begin(),times.end(),[](double t){return t>10;})
    <<",\"clipped_samples\":"<<clipped<<"}\n";
  if(!trace || !report) throw std::runtime_error("GTCRN diagnostic write failed");
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
