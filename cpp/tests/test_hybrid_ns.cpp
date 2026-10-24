#include "hybrid_ns_control.hpp"
#include "itu/audio_engine.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <random>

namespace {
bool counting=false;
std::size_t allocations=0;
void require(bool value,const char* why) {if(!value) throw std::runtime_error(why);}
using Net=itu::detail::TinyNsCorrector;
void fixture(const std::filesystem::path& path,unsigned rate,bool bad_checksum=false,bool nan=false,unsigned width=32,unsigned contract=1) {
  const unsigned parameters=Net::parameter_count(width,contract),inputs=contract==1 ? 128:256,offset=contract==3 ? 52:0;
  std::vector<unsigned char> bytes(32+4*parameters+8);
  std::copy_n(contract==1 ? "ITUNSGR1":"ITUNSGR2",8,bytes.begin());
  auto u32=[&](unsigned at,std::uint32_t value) {for(unsigned i=0;i<4;++i) bytes[at+i]=(value>>(8*i))&255;};
  const unsigned fields[]={contract==1 ? 1u:2u,rate,64,width,contract,parameters};
  for(unsigned i=0;i<6;++i) u32(8+4*i,fields[i]);
  // A hand-computable reset-after recurrence with nonzero recurrent bias.
  const unsigned bias_hh=offset+3*width*inputs+3*width*width+3*width;
  u32(32+4*(bias_hh+2*width),std::bit_cast<std::uint32_t>(.4f));
  const unsigned head=offset+3*width*inputs+3*width*width+6*width;
  u32(32+4*head,std::bit_cast<std::uint32_t>(1.f));
  u32(32+4*(offset+2*width*inputs),std::bit_cast<std::uint32_t>(.1f));
  if(contract==3) u32(32+4,std::bit_cast<std::uint32_t>(.5f));
  if(nan) u32(32,std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN()));
  std::uint64_t hash=14695981039346656037ull;
  for(unsigned i=0;i<bytes.size()-8;++i) {hash^=bytes[i];hash*=1099511628211ull;}
  if(bad_checksum) ++hash;
  for(unsigned i=0;i<8;++i) bytes[bytes.size()-8+i]=(hash>>(8*i))&255;
  std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
  require(bool(file),"fixture write failed");
}
void recurrence(const std::filesystem::path& root,unsigned width,unsigned contract) {
  const auto file=root/"fixture.bin";fixture(file,16000,false,false,width,contract);
  Net net;require(net.load(file.string(),16000)==Net::State::ready,"valid fixture rejected");
  require(net.hidden_size()==width && net.parameter_size()==Net::parameter_count(width,contract) && net.contract()==contract,"wrong model contract loaded");
  std::array<float,256> x{};std::array<float,64> p{},d{};
  float h=0;
  for(unsigned i=0;i<10;++i) {
    counting=true;const bool ok=net.process(x.data(),p.data(),d.data());counting=false;
    require(ok,"valid inference failed");h=.5f*std::tanh(.2f)+.5f*h;
    require(std::abs(p[0]-1.f/(1.f+std::exp(-h)))<1e-7,"reset-after equation or recurrence wrong");
    require(p[1]==.5f && d[0]==0,"head partition wrong");
  }
  require(allocations==0,"GRU process allocated");
  const float before=p[0];net.reset();net.process(x.data(),p.data(),d.data());
  require(p[0]<before,"hidden reset had no effect");
  net.reset();x[0]=.3f;net.process(x.data(),p.data(),d.data());
  const float encoded=.3f+(contract==3 ? std::tanh(.15f):0.f);
  const float expected=1.f/(1.f+std::exp(-.5f*std::tanh(.1f*encoded+.2f)));
  require(std::abs(p[0]-expected)<1e-7,"frequency convolution/input binding incorrect");
  x[0]=std::numeric_limits<float>::quiet_NaN();
  require(!net.process(x.data(),p.data(),d.data()) && net.state()==Net::State::numerical_error,"nonfinite feature accepted");
  x[0]=0;require(!net.process(x.data(),p.data(),d.data()),"numerical failure did not remain explicit");
  require(net.load(file.string(),24000)==Net::State::contract_error && net.checksum()==0,"wrong native rate or stale checksum");
  fixture(file,16000,true,false,width,contract);require(net.load(file.string(),16000)==Net::State::checksum_error,"checksum negative control failed");
  fixture(file,16000,false,true,width,contract);require(net.load(file.string(),16000)==Net::State::contract_error,"NaN weight accepted");
  fixture(file,16000,false,false,48,contract);require(net.load(file.string(),16000)==Net::State::contract_error,"unsupported width accepted");
  fixture(file,16000,false,false,width,contract);std::filesystem::resize_file(file,40);
  require(net.load(file.string(),16000)==Net::State::contract_error,"truncated model accepted");
  fixture(file,16000,false,false,width==32 ? 64:32);
  require(net.load(file.string(),16000)==Net::State::ready,"width switch load failed");
  require(net.process(x.data(),p.data(),d.data()) && std::abs(p[0]-1.f/(1.f+std::exp(-.5f*std::tanh(.2f))))<1e-7,
      "width switch leaked recurrent state");
  require(net.load((root/"missing").string(),16000)==Net::State::missing,"missing model not reported");
}
void streams(const std::filesystem::path& root,unsigned width,unsigned contract) {
  for(unsigned rate:{16000u,24000u,48000u}) {
    auto file=root/(std::to_string(rate)+".bin");fixture(file,rate,false,false,width,contract);
    const unsigned h=rate/100;
    itu::detail::OmlsaImcra baseline(rate),disabled(rate),missing(rate),enabled(rate),prefix(rate);
    itu::detail::HybridNsControl zero(rate,baseline.fft_size(),file.string(),-18,0,0),
        fallback(rate,baseline.fft_size(),(root/"missing").string(),-18),
        student(rate,baseline.fft_size(),file.string(),-18),
        twin(rate,baseline.fft_size(),file.string(),-18);
    std::vector<float> a(h),b(h),c(h),d(h),e(h);
    std::mt19937 rng(rate);std::normal_distribution<float> random(0,.02f);
    double effect=0;
    for(unsigned f=0;f<150;++f) {
      for(float& v:a) v=random(rng);
      b=c=d=e=a;
      counting=true;
      baseline.process(a.data());disabled.process(b.data(),&zero);
      missing.process(c.data(),&fallback);enabled.process(d.data(),&student);prefix.process(e.data(),&twin);
      counting=false;
      require(a==b && a==c,"zero mix/missing model changed baseline PCM");
      require(d==e,"causal prefix depends on future calls / other stream");
      for(unsigned k=0;k<baseline.fft_size()/2+1;++k)
        require(baseline.estimator().noise(k)==enabled.estimator().noise(k),"neural gain contaminated statistical PSD/DD");
      for(unsigned i=0;i<h;++i) {effect+=std::abs(a[i]-d[i]);require(std::isfinite(d[i]),"nonfinite hybrid PCM");}
    }
    require(effect>.001 && student.applied && student.inferences==150 && student.corrector.hidden_size()==width && student.corrector.contract()==contract,"enabled effect check vacuous");
    require(zero.inferences==150 && !zero.applied && fallback.fallback_frames==150,"diagnostics wrong");
    require(baseline.delay()==h && enabled.delay()==h,"10 ms timebase changed");
    require(allocations==0,"audio hook process allocated");
    std::cout<<"hidden="<<width<<" native_rate="<<rate<<" delay="<<h<<" zero_mix_pcm=identical missing_pcm=identical effect_l1="<<effect<<'\n';
    itu::Config config;config.sample_rate=rate;config.frame_size=h;config.aec=false;
    config.noise_backend=itu::NoiseBackend::omlsa_imcra;config.vehicle_ns.student_model_path=file.string();
    config.automatic_gain=false;config.limiter=false;
    itu::AudioEngine engine(config);itu::FrameInfo info;std::fill(a.begin(),a.end(),.01f);
    require(engine.process(a.data(),h,nullptr,0,b.data(),h,info)==itu::Status::ok,"engine integration failed");
    require(info.hybrid_ns_state==1 && info.hybrid_ns_inferences==1,"engine student telemetry missing");
    config.noise_backend=itu::NoiseBackend::webrtc;bool rejected=false;
    try {itu::AudioEngine invalid(config);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"silent ignored student path accepted");
  }
}
}
void* operator new(std::size_t n) {if(counting) ++allocations;if(void* p=std::malloc(n?n:1)) return p;throw std::bad_alloc();}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
int main() {
  const auto root=std::filesystem::temp_directory_path()/("itu-hybrid-"+std::to_string(std::random_device{}()));
  try {std::filesystem::create_directory(root);
    for(unsigned contract:{1u,2u,3u}) for(unsigned width:{32u,64u}) {recurrence(root,width,contract);streams(root,width,contract);}
    std::filesystem::remove_all(root);std::cout<<"hybrid causal/fallback/allocation checks passed\n";return 0;}
  catch(const std::exception& e) {counting=false;std::filesystem::remove_all(root);std::cerr<<e.what()<<'\n';return 1;}
}
