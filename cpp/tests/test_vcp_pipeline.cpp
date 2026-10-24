#include "drc.hpp"
#include "downlink_pipeline.hpp"
#include "itu/audio_engine_c.h"
#include "spatial_frontend.hpp"
#include "rate_converter.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
float sample(unsigned& seed) {seed=1664525u*seed+1013904223u;return (int(seed>>16)-32768)/32768.f;}

void test_drc_basic() {
  std::cout << "[RUN] test_drc_basic..." << std::endl;
  itu::DrcConfig cfg;
  cfg.enabled = true;
  cfg.threshold_dbfs = -12.0f;
  cfg.ratio = 4.0f;
  cfg.ceiling_dbfs = -1.0f;
  cfg.attack_ms = 2.0f;
  cfg.release_ms = 20.0f;

  itu::DynamicRangeCompressor drc(16000, cfg);

  // 1. Low level signal (-20 dBFS -> peak ~ 0.1): should not be compressed
  std::vector<float> low_level(160);
  for (int i = 0; i < 160; ++i) {
    low_level[i] = 0.1f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
  }
  std::vector<float> low_out = low_level;
  drc.process(low_out.data(), low_out.size());

  // Check output is close to input (no compression)
  for (size_t i = 0; i < low_out.size(); ++i) {
    require(std::abs(low_out[i] - low_level[i]) < 0.05f,"DRC damaged low level signal");
  }

  // 2. High level signal (+2 dBFS peak ~ 1.25): should be compressed and clamped below ceiling (-1 dBFS = 0.891)
  std::vector<float> high_level(160 * 10);
  for (size_t i = 0; i < high_level.size(); ++i) {
    high_level[i] = 1.25f * std::sin(2.0f * 3.14159f * 1000.0f * i / 16000.0f);
  }
  std::vector<float> high_out = high_level;
  drc.process(high_out.data(), high_out.size());

  float max_val = 0.0f;
  for (float v : high_out) {
    max_val = std::max(max_val, std::abs(v));
    // Must never exceed ceiling
    require(std::abs(v) <= 0.895f,"DRC exceeded ceiling");
  }
  std::cout << "  High level input peak 1.25 -> compressed peak " << max_val << " (ceiling: 0.891)" << std::endl;
  std::cout << "[PASS] test_drc_basic" << std::endl;
}

void test_lookahead_drc() {
  std::cout << "[RUN] test_lookahead_drc..." << std::endl;
  itu::DrcConfig cfg;
  cfg.enabled = true;
  cfg.threshold_dbfs = -12.0f;
  cfg.ratio = 4.0f;
  cfg.knee_width_db = 0.0f;
  cfg.attack_ms = 2.0f;
  cfg.release_ms = 20.0f;
  cfg.ceiling_dbfs = -1.0f;
  cfg.lookahead_ms = 2.0f;

  const unsigned rate = 16000, hop = rate / 100;
  itu::DynamicRangeCompressor drc(rate, cfg);
  require(drc.delay_samples() == 32, "look-ahead delay rounded incorrectly");
  std::vector<float> frame(hop, 0.1f);
  frame[10] = 1.2f;
  std::vector<float> output = frame;
  drc.process(output.data(), output.size());
  require(output[0] == 0.0f && output[31] == 0.0f,
          "look-ahead did not delay the initial signal");
  require(output[32] != 0.0f && output[42] != 0.0f,
          "look-ahead delayed the complete signal instead of the requested window");
  float delayed_peak = 0.0f;
  for (float value : output) delayed_peak = std::max(delayed_peak, std::abs(value));
  require(delayed_peak > 1.0f,
          "look-ahead test input did not expose the delayed transient");
  unsigned flushed = 0;
  for (float& value : output) {
    float tail = 0.0f;
    if (drc.flush_sample(tail)) {
      value = tail;
      ++flushed;
    }
  }
  require(flushed == 32, "look-ahead tail length mismatch");

  cfg.lookahead_ms = 0.0f;
  itu::DynamicRangeCompressor causal(rate, cfg);
  require(causal.delay_samples() == 0, "disabled look-ahead added latency");
  std::vector<float> causal_output = frame;
  causal.process(causal_output.data(), causal_output.size());
  require(causal_output != output, "look-ahead negative control was ineffective");
  std::cout << "[PASS] test_lookahead_drc delay=32 delayed_peak="
            << delayed_peak << std::endl;
}

void test_downlink_pipeline() {
  std::cout << "[RUN] test_downlink_pipeline..." << std::endl;
  itu::DownlinkConfig cfg;
  cfg.enabled = true;
  cfg.enable_nr = true;
  cfg.drc.enabled = true;
  cfg.drc.threshold_dbfs = -6.0f;
  cfg.drc.ceiling_dbfs = -0.5f;

  itu::DownlinkPipeline pipeline(16000, 160, cfg);

  std::vector<float> in(160 * 20);
  std::vector<float> out_speaker(160 * 20);
  std::vector<float> out_ref(160 * 20);

  // Generate test signal with loud burst
  for (size_t i = 0; i < in.size(); ++i) {
    float amp = (i > 160 * 5 && i < 160 * 15) ? 1.2f : 0.2f;
    in[i] = amp * std::sin(2.0f * 3.14159f * 800.0f * i / 16000.0f);
  }

  // Process frame by frame
  for (size_t frame = 0; frame < 20; ++frame) {
    pipeline.process(&in[frame * 160], &out_speaker[frame * 160], &out_ref[frame * 160], 160);
  }

  // Verify speaker output and ref match
  for (size_t i = 0; i < out_speaker.size(); ++i) {
    require(out_speaker[i] == out_ref[i],"speaker/reference mismatch");
    require(std::abs(out_speaker[i]) <= 0.95f,"downlink clipping"); // No clipping
  }

  std::cout << "[PASS] test_downlink_pipeline" << std::endl;
}

void pre_aec_order(unsigned rate,unsigned m,bool spatial=false,
                   itu::AecBackend aec=itu::AecBackend::webrtc,
                   itu::NoiseBackend ns=itu::NoiseBackend::webrtc) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.microphones=m;c.beamforming=true;c.output_eq.preamp_db=-3;
  c.aec_backend=aec;c.noise_backend=ns;
  c.spatial_mvdr=spatial;c.spatial_fixed_steering=spatial;
  c.spatial_continuous_covariance=spatial;c.wind_noise_control=spatial;
  float maximum=0;
  for(unsigned ch=0;ch<m;++ch) maximum=std::max(maximum,c.beam_delays_samples[ch]=.5f+1.25f*(m-1-ch));
  auto mono=c;mono.microphones=1;mono.beamforming=false;mono.beam_delays_samples={};
  mono.spatial_mvdr=mono.spatial_fixed_steering=mono.spatial_continuous_covariance=mono.wind_noise_control=false;
  mono.rx_processing=mono.rx_config.enabled=false;
  itu::AudioEngine actual(c),oracle(mono),wrong(mono);
  std::unique_ptr<itu::SpatialFrontend> spatial_oracle;
  const unsigned h=c.frame_size;
  if(spatial) spatial_oracle=std::make_unique<itu::SpatialFrontend>(rate,h,m,true,true,true,true);
  std::vector<float> mic(h*m),ref(h),beam(h),out(h),expected(h),bad(h),aligned(h*m),history;
  std::vector<float> at(h*m),it(h*m),bt(h),lt(160*m),oa(h),oi(h),ol(160),channel0(h);
  itu::FrameInfo f,g,negative;unsigned seed=441;double distinction=0;
  for(unsigned k=0;k<160;++k) {
    for(unsigned i=0;i<h;++i) {
      ref[i]=.05f*sample(seed);
      for(unsigned ch=0;ch<m;++ch) {
        mic[i*m+ch]=.06f*sample(seed)+.1f*std::sin((k*h+i)*.17f+ch*.8f);
        history.push_back(mic[i*m+ch]);
      }
      channel0[i]=mic[i*m];
    }
    // Independent sample-index oracle: apply calibrated geometry to raw capture,
    // then send that physical beam into an ordinary mono cancellation engine.
    for(unsigned i=0;i<h;++i) {
      const unsigned n=k*h+i;float sum=0;
      for(unsigned ch=0;ch<m;++ch) {
        const unsigned d=unsigned(c.beam_delays_samples[ch]);
        const float a=c.beam_delays_samples[ch]-d;
        const float x=n>=d ? history[(n-d)*m+ch] : 0;
        const float y=n>d ? history[(n-d-1)*m+ch] : 0;
        aligned[i*m+ch]=(1-a)*x+a*y;sum+=aligned[i*m+ch];
      }
      beam[i]=sum/m;
    }
    if(spatial_oracle) spatial_oracle->process(aligned.data(),nullptr,nullptr,beam.data(),nullptr,nullptr,false);
    require(actual.process(mic.data(),mic.size(),ref.data(),h,out.data(),h,f,at.data(),bt.data(),lt.data(),it.data())==itu::Status::ok,"BF/TX process");
    require(oracle.process(beam.data(),h,ref.data(),h,expected.data(),h,g,oa.data(),nullptr,ol.data(),oi.data())==itu::Status::ok,"mono oracle process");
    require(wrong.process(channel0.data(),h,ref.data(),h,bad.data(),h,negative)==itu::Status::ok,"wrong geometry control");
    require(out==expected && bt==beam,"BF did not precede AEC/NR/EQ/AGC");
    require(f.aec_capture_channels==1 && f.beamforming_before_aec,"physical AEC topology metadata");
    const unsigned front=unsigned(std::ceil(maximum))+(spatial_oracle?spatial_oracle->delay():0);
    require(f.latency_samples==g.latency_samples+front,"front-end delay counted incorrectly");
    for(unsigned i=0;i<h;++i) for(unsigned ch=0;ch<m;++ch)
      require(at[i*m+ch]==oa[i] && it[i*m+ch]==oi[i],"post-BF tap replicas differ from actual AEC channel");
    for(unsigned ch=0;ch<m;++ch) for(unsigned i=0;i<160;++i)
      require(lt[ch*160+i]==ol[i],"linear AEC physical-channel contract");
    for(unsigned i=0;i<h;++i) distinction+=std::abs(out[i]-bad[i]);
  }
  require(distinction>.01,"wrong beam negative control was ineffective");
  std::cout<<"BF-before-AEC rate="<<rate<<" mics="<<m<<" spatial="<<spatial
           <<" aec="<<unsigned(aec)<<" ns="<<unsigned(ns)<<" oracle_equal=true wrong_beam_difference="<<distinction<<'\n';
}

void eq_order(unsigned rate) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec=c.noise_suppression=c.limiter=false;c.fixed_gain_db=12;c.output_eq.preamp_db=-12;
  itu::DiagnosticStageTaps taps;c.diagnostic_stages=&taps;
  itu::AudioEngine actual(c);
  auto wrong=c;wrong.eq_before_agc=false;wrong.diagnostic_stages=nullptr;itu::AudioEngine negative(wrong);
  std::vector<float> x(c.frame_size),out(c.frame_size),bad(c.frame_size);
  itu::FrameInfo f,g;double before=0,after=0,difference=0;
  for(unsigned k=0;k<100;++k) {
    for(unsigned i=0;i<c.frame_size;++i) x[i]=.65f*std::sin(2*3.141592653589793*1000*(k*c.frame_size+i)/rate);
    require(actual.process(x.data(),x.size(),nullptr,0,out.data(),out.size(),f)==itu::Status::ok,"pre-AGC EQ process");
    require(negative.process(x.data(),x.size(),nullptr,0,bad.data(),bad.size(),g)==itu::Status::ok,"post-AGC EQ control");
    require(taps.eq_sample_rate==(rate==24000?48000:rate) && taps.eq_frame_size==taps.core_frame_size,"pre-AGC EQ tap rate");
    for(unsigned i=0;i<taps.core_frame_size;++i) {
      require(taps.pre_eq[i]==taps.nr_protected[i],"EQ input is not protected NR output");
      if(k>30) {before+=double(taps.pre_eq[i])*taps.pre_eq[i];after+=double(taps.post_eq[i])*taps.post_eq[i];}
    }
    if(k>30) for(unsigned i=0;i<c.frame_size;++i) difference+=std::abs(out[i]-bad[i]);
  }
  const double gain=10*std::log10(after/before);
  require(std::abs(gain+12)<.001,"EQ skipped, repeated or placed after AGC");
  require(difference>1,"EQ/AGC order negative control did not distinguish clipping");
  std::cout<<"EQ-before-AGC rate="<<rate<<" gain_db="<<gain<<" wrong_order_difference="<<difference<<'\n';
}

void duplex(unsigned rate,unsigned m) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;c.microphones=m;
  c.beamforming=m>1;c.reference_channels=m>1?2:1;
  c.rx_config.eq.bands=1;c.rx_config.eq.band[0]={itu::EqKind::peak,1400,-3,.7f};
  auto tx_config=c;tx_config.rx_processing=tx_config.rx_config.enabled=false;
  itu::AudioEngine actual(c),tx_oracle(tx_config),rx_oracle(c),twice(c);
  const unsigned h=c.frame_size,r=c.reference_channels;
  require(actual.working_memory_bytes()>=tx_oracle.working_memory_bytes()+(2+r)*h*sizeof(float),
      "RX scratch or per-channel pipeline buffers missing from memory report");
  std::vector<float> rx(h*r),speaker(h*r),ref(h*r),expected(h*r),expected_ref(h*r),bad(h*r);
  std::vector<float> mic(h*m),out(h),control(h),played;
  itu::FrameInfo f,g;unsigned seed=731;double double_difference=0;std::vector<double> times;
  for(unsigned k=0;k<300;++k) {
    for(float& v:rx) v=.7f*sample(seed);
    const auto start=std::chrono::steady_clock::now();
    const auto rx_status=actual.process_downlink(rx.data(),rx.size(),speaker.data(),ref.data());
    const auto rx_end=std::chrono::steady_clock::now();
    require(rx_status==itu::Status::ok,"duplex RX");
    require(rx_oracle.process_downlink(rx.data(),rx.size(),expected.data(),expected_ref.data())==itu::Status::ok,"independent RX");
    require(speaker==ref && speaker==expected && ref==expected_ref,"RX state advanced by TX or reference differs from playback");
    require(twice.process_downlink(ref.data(),ref.size(),bad.data(),nullptr)==itu::Status::ok,"double RX negative control");
    for(unsigned i=0;i<ref.size();++i) double_difference+=std::abs(ref[i]-bad[i]);
    played.insert(played.end(),speaker.begin(),speaker.end());
    for(unsigned i=0;i<h;++i) {
      const unsigned n=k*h+i;
      const float echo=(n>=h*3 ? .45f*played[(n-h*3)*r] : 0)+
          (n>=h*3+7 ? .12f*played[(n-h*3-7)*r] : 0);
      const float near=k>180 ? .12f*std::sin(n*.19f) : 0;
      for(unsigned ch=0;ch<m;++ch) mic[i*m+ch]=echo+near+.002f*sample(seed);
    }
    const auto saved_ref=ref;
    if(rate==24000 && k==0 && m==1) {
      itu::RateConverter probe(h,480);std::vector<float> converted(480);
      require(probe.Resample(ref.data(),h,converted.data(),480)==0,"reference resampling probe");
      float peak=0;for(float v:converted) peak=std::max(peak,std::abs(v));
      std::cout<<"24k first-frame playback SRC peak="<<peak<<std::endl;
    }
    const auto tx_start=std::chrono::steady_clock::now();
    const auto status=actual.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),h,f);
    const auto tx_end=std::chrono::steady_clock::now();
    if(status!=itu::Status::ok) throw std::runtime_error("duplex TX rate="+std::to_string(rate)+" frame="+std::to_string(k)+" status="+std::to_string(int(status))+" backend="+std::to_string(f.backend_error));
    times.push_back(std::chrono::duration<double,std::milli>((rx_end-start)+(tx_end-tx_start)).count());
    require(tx_oracle.process(mic.data(),mic.size(),ref.data(),ref.size(),control.data(),h,g)==itu::Status::ok,"TX playback oracle");
    require(out==control && ref==saved_ref,"TX reprocessed or mutated actual playback reference");
    require(f.frame_index==k && f.latency_samples==g.latency_samples,"duplex timeline");
    for(float v:out) require(std::isfinite(v) && std::abs(v)<=c.output_ceiling,"duplex finite bounded output");
  }
  require(double_difference>1,"double RX negative control did not change playback");
  std::sort(times.begin(),times.end());
  std::cout<<"duplex rate="<<rate<<" mics="<<m<<" refs="<<r<<" frames=300 oracle_equal=true p95_ms="<<times[times.size()*95/100]<<" double_RX_difference="<<double_difference<<'\n';
}

void rx_validation_and_channels(unsigned rate) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;c.reference_channels=2;
  c.rx_config.eq.bands=1;c.rx_config.eq.band[0]={itu::EqKind::peak,900,-3,.7f};
  itu::AudioEngine actual(c),fresh(c);
  std::vector<float> x(c.frame_size*2),speaker(x.size()),ref(x.size()),expected(x.size());
  for(unsigned i=0;i<c.frame_size;++i) x[i*2]=.7f*std::sin(i*.13f);
  require(actual.process_downlink(x.data(),x.size()-1,speaker.data(),ref.data())==itu::Status::invalid_length,"RX short frame accepted");
  x.back()=NAN;require(actual.process_downlink(x.data(),x.size(),speaker.data(),ref.data())==itu::Status::non_finite,"RX invalid PCM accepted");x.back()=0;
  require(actual.process_downlink(x.data(),x.size(),speaker.data(),ref.data())==itu::Status::ok,"RX valid frame");
  require(fresh.process_downlink(x.data(),x.size(),expected.data(),nullptr)==itu::Status::ok,"RX fresh frame");
  require(speaker==expected && speaker==ref,"invalid RX advanced state");
  for(unsigned i=0;i<c.frame_size;++i) require(speaker[i*2+1]==0,"RX EQ/DRC leaked across channels");
  actual.reset();fresh.reset();
  require(actual.process_downlink(x.data(),x.size(),x.data(),x.data())==itu::Status::ok,"RX in-place");
  require(x==speaker,"RX reset/in-place changed output");
}

void c_api() {
  itu_audio_config c;itu_audio_default_config(&c);c.microphones=2;c.beamforming=1;
  char error[256];auto* engine=itu_audio_create_with_profile(&c,ITU_AUDIO_PROFILE_VCP,error,sizeof(error));
  require(engine && !error[0],"C VCP profile");
  std::vector<float> rx(160,.1f),speaker(160),ref(160),mic(320),out(160);itu_audio_frame_info f;
  require(itu_audio_aec_capture_channels(engine)==1,"C physical channel query");
  require(itu_audio_process_downlink(engine,rx.data(),160,speaker.data(),ref.data())==ITU_AUDIO_OK && speaker==ref,"C RX tap");
  require(itu_audio_process(engine,mic.data(),320,ref.data(),160,out.data(),160,&f,nullptr,nullptr,nullptr)==ITU_AUDIO_OK,"C duplex TX");
  require(itu_audio_process_downlink(engine,rx.data(),159,speaker.data(),ref.data())==ITU_AUDIO_INVALID_LENGTH,"C RX validation");
  itu_audio_destroy(engine);
}

int main() {
  try {
    test_drc_basic();test_lookahead_drc();test_downlink_pipeline();c_api();
    for(unsigned rate:{16000,24000,48000}) {
      eq_order(rate);rx_validation_and_channels(rate);
      for(unsigned m:{1,2,4}) {pre_aec_order(rate,m);duplex(rate,m);}
      for(unsigned m:{2,4}) pre_aec_order(rate,m,true);
      pre_aec_order(rate,2,false,itu::AecBackend::speex);
      pre_aec_order(rate,2,false,itu::AecBackend::webrtc,itu::NoiseBackend::speex);
#ifdef ITU_VCP_HAS_RNNOISE
      pre_aec_order(rate,2,false,itu::AecBackend::webrtc,itu::NoiseBackend::rnnoise);
#endif
    }
    std::cout << "All Barrot VCP pipeline unit tests passed successfully!" << std::endl;
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
