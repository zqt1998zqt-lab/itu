#include "itu/audio_engine.hpp"
#include "aec_fusion.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void require_throws(itu::Config c, const char* message) {
  bool threw=false; try { itu::AudioEngine engine(c); } catch (const std::invalid_argument&) { threw=true; }
  require(threw,message);
}

struct Result {
  double before=0, post=0, near_cross=0, near_power=0;
  float max_weight=0;
  unsigned updates=0;
  unsigned max_latency=0;
};

Result run(itu::AecBackend backend, bool path_change) {
  itu::Config c=itu::barrot_vcp_config();
  c.sample_rate=16000; c.frame_size=160; c.microphones=1; c.reference_channels=1;
  c.aec_backend=backend; c.speex_post_context=true; c.speex_coherence_residual_suppression=false;
  c.noise_suppression=false; c.automatic_gain=false; c.limiter=false; c.tx_drc.enabled=false;
  c.rx_processing=false; c.rx_config.enabled=false; c.linear_phase_low_cut=false;
  itu::AudioEngine engine(c); itu::FrameInfo info;
  constexpr unsigned frames=520, hop=160, rate=16000, delay_a=272, delay_b=512;
  std::mt19937 rng(71); std::uniform_real_distribution<float> draw(-.12f,.12f);
  std::vector<float> ref(hop), mic(hop), out(hop), history(rate*2,0.f);
  unsigned head=0; Result result;
  for(unsigned frame=0; frame<frames; ++frame) {
    for(unsigned i=0;i<hop;++i) {
      ref[i]=draw(rng); history[head]=ref[i];
      const unsigned delay=path_change && frame>=260 ? delay_b : delay_a;
      const float echo=.68f*history[(head+history.size()-delay)%history.size()];
      const float near=frame>=390 ? .035f*std::sin(2.f*3.141592653589793f*997.f*(frame*hop+i)/rate) : 0.f;
      mic[i]=echo+near; head=(head+1)%history.size();
    }
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok,
            "fused AEC process failed");
    double in_power=0,out_power=0;
    for(unsigned i=0;i<hop;++i) { in_power+=double(mic[i])*mic[i]; out_power+=double(out[i])*out[i]; }
    if(frame>=180 && frame<240) result.before+=in_power/hop;
    if(frame>=300 && frame<370) result.post+=out_power/hop;
    if(frame>=430 && frame<500) {
      for(unsigned i=0;i<hop;++i) {
        const float near=.035f*std::sin(2.f*3.141592653589793f*997.f*(frame*hop+i)/rate);
        result.near_cross+=double(out[i])*near; result.near_power+=double(near)*near;
      }
    }
    result.max_weight=std::max(result.max_weight,info.aec_fused_shadow_weight);
    result.updates+=info.aec_fused_updates; result.max_latency=std::max(result.max_latency,info.latency_samples);
    for(float v:out) require(std::isfinite(v) && std::abs(v)<=1.01f,"AEC output invalid");
    if(backend==itu::AecBackend::speex_fused) {
      require(info.aec_fused_shadow_weight>=.019f && info.aec_fused_shadow_weight<=.851f,"fusion weight escaped bounds");
      require(std::isfinite(info.aec_fused_primary_error_power) && std::isfinite(info.aec_fused_shadow_error_power),"fusion diagnostics invalid");
    } else {
      require(info.aec_fused_shadow_weight==0.f && info.aec_fused_updates==0,"baseline unexpectedly used fusion");
    }
  }
  return result;
}

int main() {
  try {
    itu::AecFusion guard(16000,160,.12f,32);
    std::vector<float> guard_capture(160,.1f), guard_render(160,.1f), guard_primary(160,.1f);
    guard.process(guard_capture.data(),guard_render.data(),true);
    guard.blend(guard_primary.data(),true,1,.9f,.8f,.9f,10);
    const auto guard_frame=guard.frame();
    require(guard_frame.double_talk && guard_frame.shadow_weight<=.08f,"double-talk guard allowed shadow takeover");
    guard.process(guard_capture.data(),std::vector<float>(160,0.f).data(),false);
    guard.blend(guard_primary.data(),false,0,0.f,1.f,0.f,0);
    require(guard.frame().shadow_weight<=.08f,"silent-reference guard raised fusion weight");
    auto fused=itu::barrot_vcp_fused_aec_config();
    require(fused.aec_backend==itu::AecBackend::speex_fused,"fused preset backend mismatch");
    require(fused.speex_fused_shadow_taps_ms==32,"fused preset taps mismatch");
    fused.reference_channels=2; require_throws(fused,"multi-reference fused AEC was accepted");
    fused=itu::barrot_vcp_fused_aec_config(); fused.microphones=2; require_throws(fused,"multi-microphone fused AEC was accepted");
    const auto baseline=run(itu::AecBackend::speex,true);
    const auto candidate=run(itu::AecBackend::speex_fused,true);
    const auto stable_baseline=run(itu::AecBackend::speex,false);
    const auto stable_candidate=run(itu::AecBackend::speex_fused,false);
    const double baseline_db=10*std::log10(std::max(baseline.before,1e-20)/std::max(baseline.post,1e-20));
    const double candidate_db=10*std::log10(std::max(candidate.before,1e-20)/std::max(candidate.post,1e-20));
    const double near_gain=candidate.near_cross/std::max(candidate.near_power,1e-20);
    const double stable_baseline_db=10*std::log10(std::max(stable_baseline.before,1e-20)/std::max(stable_baseline.post,1e-20));
    const double stable_candidate_db=10*std::log10(std::max(stable_candidate.before,1e-20)/std::max(stable_candidate.post,1e-20));
    std::cout<<"AEC fusion baseline_echo_attenuation_db="<<baseline_db
             <<" fused_echo_attenuation_db="<<candidate_db
             <<" stable_baseline_db="<<stable_baseline_db
             <<" stable_fused_db="<<stable_candidate_db
             <<" fused_weight_max="<<candidate.max_weight
             <<" fused_updates="<<candidate.updates
             <<" near_gain="<<near_gain
             <<" latency_samples="<<candidate.max_latency<<'\n';
    require(candidate.updates>1000,"shadow NLMS never adapted");
    require(candidate.max_weight>.05f,"fusion never left its safe floor");
    require(candidate_db>baseline_db-3.0,"fusion regressed path-change echo attenuation by more than 3 dB");
    require(stable_candidate_db>stable_baseline_db-.75,"fusion regressed stable echo attenuation by more than 0.75 dB");
    require(near_gain>.55 && near_gain<1.25,"fusion damaged near speech excessively");
    require(candidate.max_latency==baseline.max_latency,"fusion changed the declared latency");
    std::cout<<"[PASS] fused AEC construction, safety, path-change comparison, and near-end guard\n";
  } catch(const std::exception& e) { std::cerr<<"[FAIL] "<<e.what()<<'\n'; return 1; }
}
