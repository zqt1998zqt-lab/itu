#include "neural_ns.hpp"
#include "rate_converter.hpp"
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#ifdef ITU_HAS_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace itu {
#ifdef ITU_HAS_ORT
struct NeuralNs::State {
  using C=std::complex<float>;
  unsigned rate,delay_samples,count=0,calls=0,bad_outputs=0;
  std::uint64_t samples=0;
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"itu-ulunas"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  Ort::RunOptions run;
  Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  std::array<float,514> mix{},enh{};
  std::array<float,5358> conv{},conv_out{};
  std::array<float,402> tfa{},tfa_out{};
  std::array<float,1056> inter{},inter_out{};
  std::array<Ort::Value,4> input{Ort::Value{nullptr},Ort::Value{nullptr},Ort::Value{nullptr},Ort::Value{nullptr}};
  std::array<Ort::Value,4> output{Ort::Value{nullptr},Ort::Value{nullptr},Ort::Value{nullptr},Ort::Value{nullptr}};
  std::array<C,512> spectrum{};
  std::array<C,256> roots{};
  std::array<float,512> window{},history{},overlap{},normalizer{};
  std::array<float,2048> fifo{};
  std::array<float,160> down_pcm{},enh_pcm{};
  std::unique_ptr<RateConverter> down,up;
  explicit State(unsigned r,const std::string& model):rate(r),delay_samples(512*r/16000) {
    if((r!=16000 && r!=48000) || model.empty()) throw std::invalid_argument("ULUNAS requires model and core16/48k");
    options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session=std::make_unique<Ort::Session>(env,model.c_str(),options);
    const char* names[]={"mix","conv_cache","tfa_cache","inter_cache"};
    const char* outnames[]={"enh","conv_cache_out","tfa_cache_out","inter_cache_out"};
    std::array<std::vector<int64_t>,4> shapes{{{1,257,1,2},{1,5358},{1,402},{1,1056}}};
    float* ptr[]={mix.data(),conv.data(),tfa.data(),inter.data()};
    float* optr[]={enh.data(),conv_out.data(),tfa_out.data(),inter_out.data()};
    unsigned lengths[]={514,5358,402,1056};
    if(session->GetInputCount()!=4 || session->GetOutputCount()!=4) throw std::invalid_argument("ULUNAS model arity mismatch");
    Ort::AllocatorWithDefaultOptions allocator;
    for(unsigned i=0;i<4;++i) {
      if(std::string(session->GetInputNameAllocated(i,allocator).get())!=names[i] ||
         std::string(session->GetOutputNameAllocated(i,allocator).get())!=outnames[i])
        throw std::invalid_argument("ULUNAS model names mismatch");
      for(unsigned direction=0;direction<2;++direction) {
        auto type=direction ? session->GetOutputTypeInfo(i) : session->GetInputTypeInfo(i);
        auto tensor=type.GetTensorTypeAndShapeInfo();
        if(tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || tensor.GetShape()!=shapes[i])
          throw std::invalid_argument("ULUNAS state/shape mismatch");
      }
      input[i]=Ort::Value::CreateTensor<float>(memory,ptr[i],lengths[i],shapes[i].data(),shapes[i].size());
      output[i]=Ort::Value::CreateTensor<float>(memory,optr[i],lengths[i],shapes[i].data(),shapes[i].size());
    }
    for(unsigned i=0;i<512;++i) window[i]=.5f-.5f*std::cos(2*std::numbers::pi_v<float>*i/512);
    for(unsigned i=0;i<256;++i) roots[i]=std::polar(1.f,-2*std::numbers::pi_v<float>*i/512);
    if(r==48000) {
      down=std::make_unique<RateConverter>(480,160); up=std::make_unique<RateConverter>(160,480);
      delay_samples+=unsigned(down->output_delay()*3+up->output_delay());
    }
  }
  void fft(bool inverse) noexcept {
    for(unsigned i=1,j=0;i<512;++i) {unsigned b=256; for(;j&b;b/=2) j^=b; j^=b; if(i<j) std::swap(spectrum[i],spectrum[j]);}
    for(unsigned len=2;len<=512;len*=2) for(unsigned i=0;i<512;i+=len) for(unsigned j=0;j<len/2;++j) {
      C w=roots[j*512/len]; if(inverse) w=std::conj(w);
      const C u=spectrum[i+j],v=spectrum[i+j+len/2]*w; spectrum[i+j]=u+v; spectrum[i+j+len/2]=u-v;
    }
    if(inverse) for(C& v:spectrum) v/=512.f;
  }
  void infer() {
    for(unsigned i=0;i<512;++i) spectrum[i]=history[i]*window[i];
    fft(false);
    for(unsigned b=0;b<257;++b) {mix[2*b]=spectrum[b].real(); mix[2*b+1]=spectrum[b].imag();}
    const char* names[]={"mix","conv_cache","tfa_cache","inter_cache"};
    const char* outnames[]={"enh","conv_cache_out","tfa_cache_out","inter_cache_out"};
    session->Run(run,names,input.data(),4,outnames,output.data(),4); ++calls;
    for(auto values:{std::pair<float*,unsigned>{enh.data(),514},{conv_out.data(),5358},{tfa_out.data(),402},{inter_out.data(),1056}})
      for(unsigned i=0;i<values.second;++i) if(!std::isfinite(values.first[i])) {++bad_outputs; throw std::runtime_error("ULUNAS nonfinite output");}
    conv=conv_out; tfa=tfa_out; inter=inter_out;
    for(unsigned b=0;b<257;++b) {spectrum[b]={enh[2*b],enh[2*b+1]}; if(b && b<256) spectrum[512-b]=std::conj(spectrum[b]);}
    spectrum[0]=C(spectrum[0].real(),0); spectrum[256]=C(spectrum[256].real(),0); fft(true);
    for(unsigned i=0;i<512;++i) {overlap[i]+=spectrum[i].real()*window[i]; normalizer[i]+=window[i]*window[i];}
    // Current input hop ends at samples-1. Emit this window's first half at
    // samples..samples+255, giving exactly 512 samples of causal PCM delay.
    if(calls>1) for(unsigned i=0;i<256;++i) fifo[(samples+i)%fifo.size()]=normalizer[i]>1e-9f ? overlap[i]/normalizer[i] : 0.f;
    std::move(overlap.begin()+256,overlap.end(),overlap.begin()); std::fill(overlap.begin()+256,overlap.end(),0.f);
    std::move(normalizer.begin()+256,normalizer.end(),normalizer.begin()); std::fill(normalizer.begin()+256,normalizer.end(),0.f);
    std::move(history.begin()+256,history.end(),history.begin());
  }
  void process16(const float* x,float* y) {
    for(unsigned i=0;i<160;++i) {
      history[256+count]=x[i]; ++count; ++samples;
      if(count==256) {infer(); count=0;}
      const auto position=(samples-1)%fifo.size(); y[i]=fifo[position]; fifo[position]=0;
    }
  }
};
NeuralNs::NeuralNs(unsigned r,const std::string& model):state_(std::make_unique<State>(r,model)) {}
void NeuralNs::process(float* x) {
  auto& s=*state_;
  if(s.down) {
    if(s.down->Resample(x,480,s.down_pcm.data(),160)) throw std::runtime_error("ULUNAS downsample failed");
    s.process16(s.down_pcm.data(),s.enh_pcm.data());
    if(s.up->Resample(s.enh_pcm.data(),160,x,480)) throw std::runtime_error("ULUNAS upsample failed");
  } else s.process16(x,x);
}
unsigned NeuralNs::delay() const noexcept {return state_->delay_samples;}
unsigned NeuralNs::inferences() const noexcept {return state_->calls;}
unsigned NeuralNs::nonfinite_outputs() const noexcept {return state_->bad_outputs;}
std::size_t NeuralNs::memory_bytes() const noexcept {return sizeof(State);} // ORT arena/model excluded.
#else
struct NeuralNs::State {};
NeuralNs::NeuralNs(unsigned,const std::string&) {throw std::invalid_argument("ULUNAS backend not built; set ITU_ORT_ROOT");}
void NeuralNs::process(float*) {throw std::runtime_error("ULUNAS unavailable");}
unsigned NeuralNs::delay() const noexcept {return 0;}
unsigned NeuralNs::inferences() const noexcept {return 0;}
unsigned NeuralNs::nonfinite_outputs() const noexcept {return 0;}
std::size_t NeuralNs::memory_bytes() const noexcept {return 0;}
#endif
NeuralNs::~NeuralNs()=default;
}
