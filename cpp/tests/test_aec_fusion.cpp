#include "itu/audio_engine.hpp"
#include "aec_fusion.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

// Count wrapper C++ allocations after construction. Backend mallocs are not
// observable here; this is a callback workspace regression, not a heap profiler.
static bool count_allocations=false;
static std::size_t allocations=0;
void* operator new(std::size_t size) {
  if(count_allocations) ++allocations;
  if(void* p=std::malloc(std::max(size,std::size_t(1)))) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {return ::operator new(size);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
void require(bool ok,const char* reason) {if(!ok) throw std::runtime_error(reason);}
constexpr double pi=3.141592653589793;

enum class Scene {stable,delay_jump,long_jump,gain_change,wrong_reference,silent_reference,long_tail};
struct Result {
  double attenuation=0,near_gain=0,near_sisdr=0,recovery_ms=-1,mean_ms=0,max_ms=0;
  unsigned latency=0,path_changes=0,near_frozen=0,near_frames=0;
  float max_weight=0;
  std::size_t memory=0;
};
Result run(unsigned rate,bool fused,Scene scene,unsigned seed=71) {
  auto c=itu::barrot_vcp_config();
  c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=fused ? itu::AecBackend::speex_fused : itu::AecBackend::speex;
  c.speex_coherence_residual_suppression=false;
  c.noise_suppression=c.automatic_gain=c.limiter=c.tx_drc.enabled=false;
  c.rx_processing=c.rx_config.enabled=c.linear_phase_low_cut=false;
  itu::AudioEngine engine(c);itu::FrameInfo info;
  const unsigned h=c.frame_size,frames=650,delay_a=rate*17/1000;
  const unsigned delay_b=scene==Scene::long_jump ? rate*81/1000+1 : rate*32/1000;
  std::mt19937 rng(seed);std::uniform_real_distribution<float> draw(-.12f,.12f);
  std::vector<float> ref(h),mic(h),out(h),history(rate,0.f);
  std::vector<float> capture(frames*h),clean(frames*h),output((frames+12)*h);
  Result result;result.memory=engine.working_memory_bytes();
  unsigned head=0;
  for(unsigned frame=0;frame<frames+12;++frame) {
    for(unsigned i=0;i<h;++i) {
      const bool active=frame<frames;
      const float actual=active ? draw(rng) : 0.f;
      history[head]=actual;
      const bool change=(scene==Scene::delay_jump || scene==Scene::long_jump) && frame>=260;
      const unsigned delay=change ? delay_b : delay_a;
      float echo=(scene==Scene::gain_change && frame>=260 ? -.45f : .68f)*history[(head+rate-delay)%rate];
      if(scene==Scene::long_tail) echo+=.55f*history[(head+rate-rate*110/1000)%rate];
      if(scene==Scene::wrong_reference || scene==Scene::silent_reference) echo=0;
      const float near=frame>=480 && active ? float(.035*std::sin(2*pi*997*(frame*h+i)/rate)) : 0.f;
      ref[i]=scene==Scene::silent_reference ? 0.f : scene==Scene::wrong_reference ? draw(rng) : actual;
      mic[i]=active ? echo+near : 0.f;
      if(active) {capture[frame*h+i]=mic[i];clean[frame*h+i]=near;}
      head=(head+1)%rate;
    }
    const auto start=std::chrono::steady_clock::now();
    count_allocations=frame>=32;const auto before=allocations;
    const auto status=engine.process(mic.data(),h,ref.data(),h,out.data(),h,info);
    count_allocations=false;
    require(status==itu::Status::ok,"fusion process failed");
    require(allocations==before,"AEC callback allocated C++ workspace");
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(frame<frames) {result.mean_ms+=ms;result.max_ms=std::max(result.max_ms,ms);}
    result.latency=info.latency_samples;
    result.path_changes=std::max(result.path_changes,info.aec_fused_path_changes);
    result.max_weight=std::max(result.max_weight,info.aec_fused_shadow_weight);
    if(frame>=490 && frame<600) {
      ++result.near_frames;result.near_frozen+=info.aec_fused_double_talk && info.aec_fused_updates==0;
    }
    for(unsigned i=0;i<h;++i) {
      require(std::isfinite(out[i]) && std::abs(out[i])<=1.01f,"invalid AEC output");
      output[frame*h+i]=out[i];
    }
    if(fused) require(info.aec_fused_shadow_weight>=0 && info.aec_fused_shadow_weight<=.951f,"fusion weight out of bounds");
    else require(info.aec_fused_shadow_weight==0 && info.aec_fused_updates==0,"baseline fusion activated");
  }
  result.mean_ms/=frames;
  auto power=[&](const std::vector<float>& x,unsigned first,unsigned last,unsigned offset=0) {
    double sum=0;for(unsigned i=first;i<last;++i) sum+=double(x[i+offset])*x[i+offset];
    return sum/std::max(1u,last-first);
  };
  // Identical time interval AND sample count, aligned only by declared latency.
  const double input_power=power(capture,3*rate,37*rate/10);
  const double output_power=power(output,3*rate,37*rate/10,result.latency);
  result.attenuation=10*std::log10(std::max(1e-20,input_power)/std::max(1e-20,output_power));
  double cp=0,cross=0,op=0;
  for(unsigned i=52*rate/10;i<60*rate/10;++i) {
    cp+=double(clean[i])*clean[i];cross+=double(clean[i])*output[i+result.latency];
    op+=double(output[i+result.latency])*output[i+result.latency];
  }
  result.near_gain=cross/std::max(1e-20,cp);
  result.near_sisdr=10*std::log10(std::max(1e-20,cross*cross/std::max(1e-20,cp))/
      std::max(1e-20,op-cross*cross/std::max(1e-20,cp)));
  unsigned consecutive=0;
  for(unsigned frame=260;frame<470;++frame) {
    const double a=power(capture,frame*h,(frame+1)*h);
    const double b=power(output,frame*h,(frame+1)*h,result.latency);
    if(a>1e-10 && 10*std::log10(a/std::max(1e-20,b))>=15) ++consecutive;
    else consecutive=0;
    if(consecutive>=10) {result.recovery_ms=(frame-260+1-10)*10.;break;}
  }
  return result;
}

void guards() {
  auto c=itu::barrot_vcp_fused_aec_config();
  for(unsigned invalid : {0u,1u}) {
    auto bad=c;
    if(invalid==0) {
      bad.microphones=2;
      bad.beamforming=false;
    } else {
      bad.speex_residual_suppression=true;
    }
    bool threw=false;try {itu::AudioEngine reject(bad);} catch(const std::invalid_argument&) {threw=true;}
    require(threw,"unsupported fused configuration accepted");
  }
  auto multiref=c;
  multiref.beamforming=true;
  multiref.reference_channels=2;
  try { itu::AudioEngine accepted(multiref); }
  catch(const std::exception&) { require(false,"fused BF-before-AEC multireference configuration rejected"); }
  itu::AecFusion shadow(16000,160,.12f,32);
  std::vector<float> mic(160,.1f),ref(160,.1f),primary(160,.1f);
  shadow.process(mic.data(),ref.data(),true,true);
  shadow.blend(primary.data(),true,1,3);
  require(shadow.frame().double_talk && shadow.frame().updates==0 && shadow.frame().shadow_weight==0,
          "explicit near guard did not freeze/exit shadow");
}
int main() {
  try {
    guards();
    for(unsigned rate:{16000u,24000u,48000u}) {
      for(auto scene:{Scene::stable,Scene::delay_jump,Scene::long_jump,Scene::gain_change,
                      Scene::wrong_reference,Scene::silent_reference,Scene::long_tail}) {
        const auto base=run(rate,false,scene),candidate=run(rate,true,scene);
        std::cout<<"COMPARE rate="<<rate<<" scene="<<int(scene)
          <<" baseline_db="<<base.attenuation<<" fused_db="<<candidate.attenuation
          <<" baseline_recovery_ms="<<base.recovery_ms<<" fused_recovery_ms="<<candidate.recovery_ms
          <<" baseline_near_gain="<<base.near_gain<<" fused_near_gain="<<candidate.near_gain
          <<" baseline_sisdr="<<base.near_sisdr<<" fused_sisdr="<<candidate.near_sisdr
          <<" frozen_near_frames="<<candidate.near_frozen<<"/"<<candidate.near_frames
          <<" path_changes="<<candidate.path_changes<<" mean_ms="<<candidate.mean_ms
          <<" max_ms="<<candidate.max_ms<<" latency="<<candidate.latency
          <<" memory_delta="<<candidate.memory-base.memory<<'\n';
        require(candidate.latency==base.latency,"fusion changed declared latency");
        require(candidate.near_gain>.80 && candidate.near_gain<1.25,"fusion near-end gain damaged");
        if(scene==Scene::delay_jump || scene==Scene::long_jump) {
          require(candidate.path_changes>0,"delay jump was not observed");
          require(candidate.attenuation>base.attenuation+10,"path recovery did not improve by 10 dB");
          require(candidate.recovery_ms>=0 && candidate.recovery_ms<=650,"path did not recover within 650 ms");
        }
        if(scene==Scene::stable) require(candidate.attenuation>base.attenuation-.75,"stable AEC regressed");
        if(scene==Scene::long_tail) require(candidate.attenuation>base.attenuation-1.0,"short shadow damaged long-tail cancellation");
        if(scene==Scene::silent_reference || scene==Scene::wrong_reference) {
          require(candidate.max_weight==0,"unverified reference activated shadow");
          require(std::abs(candidate.near_gain-base.near_gain)<1e-6,"negative-control PCM changed");
        }
      }
    }
    std::cout<<"[PASS] same-window multirate path recovery/near/reference/latency/allocation guards\n";
  } catch(const std::exception& e) {count_allocations=false;std::cerr<<"[FAIL] "<<e.what()<<'\n';return 1;}
}
