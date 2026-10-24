#include "wav.hpp"
#include "hybrid_ns_control.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
void write_floats(std::ofstream& f,const float* x,unsigned n) {
  f.write(reinterpret_cast<const char*>(x),n*sizeof(float));
}
}
int main(int argc,char** argv) {
 try {
  std::string mic,clean,model,out,features,gru_input,mode="ecnr";
  unsigned rate=16000,feature_version=1;float floor_db=-18,spp_mix=.5f,snr_mix=.5f,speech_floor=-3;
  for(int i=1;i<argc;i+=2) {
    if(i+1==argc) throw std::runtime_error("missing option value");
    const std::string k=argv[i],v=argv[i+1];
    if(k=="--mic") mic=v;else if(k=="--clean") clean=v;else if(k=="--model") model=v;
    else if(k=="--out") out=v;else if(k=="--features") features=v;else if(k=="--gru-input") gru_input=v;
    else if(k=="--rate") rate=std::stoul(v);else if(k=="--floor-db") floor_db=std::stof(v);
    else if(k=="--spp-mix") spp_mix=std::stof(v);else if(k=="--snr-mix") snr_mix=std::stof(v);
    else if(k=="--speech-floor-db") speech_floor=std::stof(v);else if(k=="--mode") mode=v;
    else if(k=="--feature-version") feature_version=std::stoul(v);
    else throw std::runtime_error("unknown hybrid option: "+k);
  }
  if(out.empty() || (mode!="call" && mode!="ecnr")) throw std::runtime_error("requires --out and --mode call|ecnr");
  if(feature_version!=1 && feature_version!=2) throw std::runtime_error("unsupported feature version");
  if(!gru_input.empty()) {
    itu::detail::TinyNsCorrector net;
    if(net.load(model,rate)!=itu::detail::TinyNsCorrector::State::ready) throw std::runtime_error("invalid GRU model");
    std::ifstream f(gru_input,std::ios::binary|std::ios::ate);const auto n=f.tellg();
    const unsigned inputs=net.input_size();
    if(!f || n<=0 || n%std::streamoff(inputs*sizeof(float))) throw std::runtime_error("invalid GRU feature tensor");
    f.seekg(0);std::ofstream o(out,std::ios::binary);std::array<float,256> x{};std::array<float,64> p{},d{};
    for(std::streamoff at=0;at<n;at+=inputs*sizeof(float)) {
      f.read(reinterpret_cast<char*>(x.data()),inputs*sizeof(float));
      if(!net.process(x.data(),p.data(),d.data())) throw std::runtime_error("GRU inference failed");
      write_floats(o,p.data(),64);write_floats(o,d.data(),64);
    }
    if(!o) throw std::runtime_error("GRU output write failed");
    return 0;
  }
  auto x=itu::read_wav(mic);rate=x.sample_rate;const unsigned h=rate/100;
  if(x.channels!=1 || x.samples.empty() || (rate!=16000 && rate!=24000 && rate!=48000)) throw std::runtime_error("requires native mono 16/24/48 kHz PCM");
  itu::Wav target;
  if(!features.empty()) {
    if(clean.empty()) throw std::runtime_error("training export requires explicit --clean; runtime never reads it");
    target=itu::read_wav(clean);
    if(target.channels!=1 || target.sample_rate!=rate || target.samples.size()!=x.samples.size()) throw std::runtime_error("training clean mismatch");
  } else if(!clean.empty()) throw std::runtime_error("--clean is training-only; supply --features");
  itu::detail::OmlsaImcra ns(rate,floor_db);
  itu::detail::HybridNsControl controller(rate,ns.fft_size(),model,floor_db,spp_mix,snr_mix,speech_floor);
  const bool observing=!features.empty() || !model.empty();
  const auto samples=x.samples.size(),frames=(samples+h-1)/h;
  const auto count=(samples+ns.delay()+h-1)/h;
  std::vector<float> rendered(count*h),frame(h),past_clean(h),past_noise(h),clean_power(ns.fft_size()/2+1),noise_power(clean_power.size());
  std::vector<std::complex<float>> cs(ns.fft_size()),nspec(ns.fft_size());itu::detail::Fft fft(ns.fft_size());
  std::array<float,64> cb{},nb{};
  std::ofstream binary,trace(out+".csv");if(!features.empty()) binary.open(features,std::ios::binary);
  trace<<"frame,time_s,process_ms,mean_gain,statistical_spp,student_state,student_inferences,student_applied\n";
  std::vector<double> times;times.reserve(frames);std::size_t clipped=0;
  for(std::size_t t=0;t<count;++t) {
    for(unsigned i=0;i<h;++i) frame[i]=t*h+i<samples ? std::lround(std::clamp(x.samples[t*h+i]*32768.f,-32768.f,32767.f))/32768.f:0;
    if(binary.is_open() && t<frames) {
      std::fill(cs.begin(),cs.end(),std::complex<float>{});std::fill(nspec.begin(),nspec.end(),std::complex<float>{});
      for(unsigned i=0;i<h;++i) {
        const float c=t*h+i<samples ? target.samples[t*h+i]:0,n=frame[i]-c;
        const float a=std::sin(3.14159265358979323846f*(i+.5f)/(2*h));
        const float b=std::sin(3.14159265358979323846f*(i+h+.5f)/(2*h));
        cs[i]=past_clean[i]*a;cs[i+h]=c*b;nspec[i]=past_noise[i]*a;nspec[i+h]=n*b;
        past_clean[i]=c;past_noise[i]=n;
      }
      fft.run(cs.data());fft.run(nspec.data());
      for(unsigned k=0;k<clean_power.size();++k) {clean_power[k]=std::norm(cs[k]);noise_power[k]=std::norm(nspec[k]);}
      controller.bands.project(clean_power.data(),cb.data());controller.bands.project(noise_power.data(),nb.data());
    }
    const auto start=std::chrono::steady_clock::now();ns.process(frame.data(),observing?&controller:nullptr);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(t<frames) {
      times.push_back(ms);trace<<t<<','<<t*.01<<','<<ms<<','<<ns.mean_gain()<<','<<ns.mean_probability()<<','<<unsigned(controller.corrector.state())<<','<<controller.inferences<<','<<controller.applied<<'\n';
      if(binary.is_open()) {
        write_floats(binary,feature_version==2 ? controller.bands.enriched_features.data():controller.bands.features.data(),feature_version==2 ? 256:128);
        for(const auto* p:{controller.bands.probability.data(),controller.bands.xi.data(),controller.bands.gamma.data(),cb.data(),nb.data()}) write_floats(binary,p,64);
        if(feature_version==2) for(const auto* p:{controller.bands.power.data(),controller.bands.noise.data()}) write_floats(binary,p,64);
      }
    }
    std::copy(frame.begin(),frame.end(),rendered.begin()+t*h);
    for(float v:frame) {if(!std::isfinite(v)) throw std::runtime_error("nonfinite audio output");clipped+=std::abs(v)>=.999f;}
  }
  rendered.resize(samples+ns.delay());itu::write_wav(out+".after.wav",rate,1,rendered);
  const double first=times.front(),mean=std::accumulate(times.begin(),times.end(),0.)/times.size();std::sort(times.begin(),times.end());
  auto q=[&](double p){return times[std::min(times.size()-1,std::size_t(p*(times.size()-1)))];};
  std::ofstream report(out+".json");
  report<<"{\"sample_rate\":"<<rate<<",\"core_rate\":"<<rate<<",\"src_delay_samples\":0,\"mode\":\""<<mode
    <<"\",\"delay_samples\":"<<ns.delay()<<",\"samples\":"<<samples<<",\"frames\":"<<frames
    <<",\"student_state\":"<<unsigned(controller.corrector.state())<<",\"student_inferences\":"<<controller.inferences
    <<",\"student_hidden_size\":"<<controller.corrector.hidden_size()<<",\"student_parameters\":"<<controller.corrector.parameter_size()
    <<",\"student_contract\":"<<controller.corrector.contract()<<",\"student_inputs\":"<<controller.corrector.input_size()
    <<",\"student_fallback_frames\":"<<controller.fallback_frames<<",\"teacher_in_runtime\":false"
    <<",\"first_process_ms\":"<<first<<",\"mean_process_ms\":"<<mean<<",\"p50_process_ms\":"<<q(.5)
    <<",\"p95_process_ms\":"<<q(.95)<<",\"p99_process_ms\":"<<q(.99)<<",\"max_process_ms\":"<<times.back()
    <<",\"over_10ms\":"<<std::count_if(times.begin(),times.end(),[](double t){return t>10;})
    <<",\"clipped_samples\":"<<clipped<<",\"external_vad\":false,\"training_columns\":"<<(feature_version==2 ? 704:448)<<",\"training_rows\":"<<(binary.is_open()?frames:0)<<"}\n";
  if(!trace || !report || (binary.is_open() && !binary)) throw std::runtime_error("diagnostic write failed");
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
