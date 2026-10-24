#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace itu::detail {class OmlsaImcra;class ImcraNoiseEstimator;}
void observe(const char*,const itu::detail::OmlsaImcra&,const float*);
void helper(const char*,const itu::detail::OmlsaImcra&,const float*,float,double);
void update1(unsigned,float,float,float,float,float);
void update2(unsigned,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float);
void outer(unsigned,float,float,float);
void ring(const itu::detail::ImcraNoiseEstimator&);
#define PROP_STATE(stage,s,a) observe(stage,s,a)
#define PROP_HELPER(stage,s,a,seed,flat) helper(stage,s,a,seed,flat)
#define PROP_UPDATE1(k,s,m,sm,d,i) update1(k,s,m,sm,d,i)
#define PROP_UPDATE2(k,num,den,c,cm,cs,d,g,z,pr,p,a,prev,next,pow,xi,gamma,stored) update2(k,num,den,c,cm,cs,d,g,z,pr,p,a,prev,next,pow,xi,gamma,stored)
#define PROP_OUTER(k,prior,p,h1) outer(k,prior,p,h1)
#define PROP_RING(e) ring(e)
#define private public
#include "gated_bias_sg.hpp"
#undef private
using itu::detail::OmlsaImcra;using itu::detail::ImcraNoiseEstimator;
static unsigned hop_index=0,base_calls=0,alt_calls=0,entries=0,exits=0,alt_entries=0;
static bool active=false;static std::ofstream data,index_file;static uint64_t offset=0;
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static void emit(const char*stage,const char*field,const float*x,std::size_t n){if(hop_index>=60)return;index_file<<hop_index<<','<<stage<<','<<field<<','<<offset<<','<<n<<'\n';data.write(reinterpret_cast<const char*>(x),std::streamsize(n*4));offset+=n*4;}
static void scalar(const char*s,const char*f,float v){emit(s,f,&v,1);}
static void vec(const char*s,const char*f,const std::vector<float>&v){emit(s,f,v.data(),v.size());}
static void estimator(const char*s,const ImcraNoiseEstimator&e,bool history){
 vec(s,"noise_raw",e.noise_);vec(s,"smooth",e.smooth_);vec(s,"conditional",e.conditional_);vec(s,"minimum",e.minimum_);vec(s,"conditional_minimum",e.conditional_min_);vec(s,"sub_minimum",e.sub_min_);vec(s,"conditional_sub_minimum",e.conditional_sub_min_);vec(s,"indicator",e.indicator_);
 scalar(s,"head",e.head_);scalar(s,"frames",e.frames_in_subwindow_);scalar(s,"rotations",e.rotations_);scalar(s,"initialized",e.initialized_);scalar(s,"excluded",float(e.excluded_bins_));scalar(s,"noise_updates",float(e.noise_updates_));scalar(s,"step_updates",float(e.step_updates_));
 if(history){vec(s,"history",e.history_);vec(s,"conditional_history",e.conditional_history_);}
}
void observe(const char*s,const OmlsaImcra&v,const float*a){
 if(std::strcmp(s,"after_power")==0){active=v.bias_mode_==0;if(active)++base_calls;else ++alt_calls;}
 if(!active||hop_index>=60)return;
 estimator(s,v.noise_,false);vec(s,"power",v.power_);vec(s,"gamma",v.gamma_);vec(s,"xi",v.xi_);vec(s,"gain",v.gain_);vec(s,"noise_used",v.noise_used_);vec(s,"probability",v.probability_);vec(s,"previous_h1_snr",v.previous_h1_snr_);vec(s,"zeta",v.zeta_);scalar(s,"previous_frame_zeta",v.previous_frame_zeta_);scalar(s,"peak_zeta",v.peak_zeta_);scalar(s,"bootstrap_seed",v.bootstrap_seed_);scalar(s,"process_calls",v.minima_history_process_calls_);scalar(s,"minima_enabled",v.minima_history_bootstrap_enabled_);scalar(s,"bias_mode",v.bias_mode_);
 if(std::strcmp(s,"wet")==0)emit(s,"audio",a,v.hop_);
}
void helper(const char*s,const OmlsaImcra&v,const float*a,float seed,double flat){
 if(v.bias_mode_){++alt_entries;return;}need(hop_index==1,"helper not at hop1");if(std::strcmp(s,"helper_entry")==0)++entries;else ++exits;
 observe(s,v,a);if(hop_index>=60)return;vec(s,"history",v.noise_.history_);vec(s,"conditional_history",v.noise_.conditional_history_);vec(s,"past",v.past_);vec(s,"overlap",v.overlap_);emit(s,"audio",a,v.hop_);std::vector<float>c;for(auto x:v.spectrum_){c.push_back(x.real());c.push_back(x.imag());}vec(s,"spectrum",c);scalar(s,"local_seed",seed);scalar(s,"flatness",float(flat));
}
static std::array<std::array<float,257>,5>u1;
static std::array<std::array<float,257>,17>u2;
static std::array<std::array<float,257>,3>ou;
void update1(unsigned k,float s,float m,float sm,float d,float i){if(!active||hop_index>=60)return;float vals[]={s,m,sm,d,i};for(unsigned j=0;j<5;++j)u1[j][k]=vals[j];if(k==256){const char*names[]={"smooth","minimum","sub_minimum","denominator","indicator"};for(unsigned j=0;j<5;++j)emit("update1",names[j],u1[j].data(),257);}}
void update2(unsigned k,float num,float den,float c,float cm,float cs,float d,float g,float z,float pr,float p,float a,float prev,float next,float pow,float xi,float gamma,float stored){if(!active||hop_index>=60)return;float vals[]={num,den,c,cm,cs,d,g,z,pr,p,a,prev,next,pow,xi,gamma,stored};for(unsigned j=0;j<17;++j)u2[j][k]=vals[j];if(k==256){const char*names[]={"numerator","eligible_weight","conditional","conditional_minimum","conditional_sub_minimum","denominator","g","z","prior","p","a","previous","next","power","xi","gamma","stored"};for(unsigned j=0;j<17;++j)emit("update2",names[j],u2[j].data(),257);}}
void outer(unsigned k,float prior,float p,float h1){if(!active||hop_index>=60)return;ou[0][k]=prior;ou[1][k]=p;ou[2][k]=h1;if(k==256){emit("outer","prior",ou[0].data(),257);emit("outer","p",ou[1].data(),257);emit("outer","h1",ou[2].data(),257);}}
void ring(const ImcraNoiseEstimator&e){if(active&&hop_index<60)estimator("ring",e,true);}
int main(int argc,char**argv){try{
 need(argc==5,"usage: probe input.f32 outdir on expected.f32");need(sizeof(float)==4,"binary32 required");uint32_t end=1;need(*reinterpret_cast<unsigned char*>(&end)==1,"little endian required");
 std::string d=argv[2];std::ifstream input(argv[1],std::ios::binary),expected(argv[4],std::ios::binary);std::ofstream wet(d+"/wet.f32",std::ios::binary),hops(d+"/hops.csv");data.open(d+"/stages.bin",std::ios::binary);index_file.open(d+"/stages.csv");need(input&&expected&&wet&&hops&&data&&index_file,"cannot open streams");index_file<<"hop,stage,field,offset,count\n";hops<<"hop,base_calls,alt_calls,helper_entries,helper_exits,alt_helper_entries,mix,qualification,head,frames,rotations,control_null,recovery\n";hops<<std::setprecision(9);
 itu::detail::GatedBiasSg sg(16000,-9,true,itu::detail::SpectralGainRule::super_gaussian_map);sg.enable_bias_statistics(24);bool on=std::string(argv[3])=="1";sg.set_minima_history_bootstrap(on);std::vector<float>x(160),ref(160);unsigned mix_nonzero=0;
 while(input.read(reinterpret_cast<char*>(x.data()),640)){
  need(hop_index<2402,"too many hops");sg.process(x.data(),nullptr,0);wet.write(reinterpret_cast<const char*>(x.data()),640);expected.read(reinterpret_cast<char*>(ref.data()),640);need(expected.gcount()==640&&std::memcmp(x.data(),ref.data(),640)==0,"wet byte mismatch");need(std::all_of(x.begin(),x.end(),[](float v){return std::isfinite(v);}),"nonfinite output");
  if(sg.mix_!=0)++mix_nonzero;need(!sg.alternative_->minima_history_bootstrap_enabled_&&sg.alternative_->bias_mode_==1,"alternative config changed");need(sg.baseline_.minima_history_process_calls_==hop_index+1&&sg.alternative_->minima_history_process_calls_==hop_index+1,"process completion count mismatch");hops<<hop_index<<','<<base_calls<<','<<alt_calls<<','<<entries<<','<<exits<<','<<alt_entries<<','<<sg.mix_<<','<<sg.qualification_<<','<<sg.baseline_.noise_.head_<<','<<sg.baseline_.noise_.frames_in_subwindow_<<','<<sg.baseline_.noise_.rotations_<<",1,0\n";++hop_index;
 }
 need(input.eof()&&input.gcount()==0&&hop_index==2402,"invalid input length");char extra;need(!expected.read(&extra,1),"expected has trailing data");need(base_calls==2402&&alt_calls==2402&&entries==(on?1u:0u)&&exits==entries&&alt_entries==0&&mix_nonzero==0,"dynamic counters mismatch");wet.close();data.close();index_file.close();hops.close();need(wet&&data&&index_file&&hops,"write error");
 std::cout<<"{\"status\":\"passed\",\"hops\":"<<hop_index<<",\"wet_exact\":true,\"base_calls\":"<<base_calls<<",\"alt_calls\":"<<alt_calls<<",\"helper_entries\":"<<entries<<",\"helper_exits\":"<<exits<<",\"alt_helper_entries\":"<<alt_entries<<",\"mix_nonzero_hops\":"<<mix_nonzero<<",\"trace_bytes\":"<<offset<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
