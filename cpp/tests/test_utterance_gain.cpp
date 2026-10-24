#include "utterance_gain.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

static void require(bool value,const char* message) {
  if(!value) {std::cerr<<message<<'\n';std::exit(1);}
}

int main() {
  itu::UtteranceGain gain;
  for(unsigned i=0;i<100;++i)
    require(gain.observe(.0001f,0,false,18,1e-8,0)==0,"ambient noise must not acquire gain");
  for(unsigned i=0;i<150;++i) {
    gain.observe(i==20 ? .03f : .004f,.95f,false,18,1e-8,0);
    if(i==58) require(gain.votes()>=3,"onset telemetry must expose live votes");
    if(i==59) require(std::abs(gain.target()-8)<1e-4,
                      "onset telemetry must expose the acquired target");
  }
  require(std::abs(gain.applied()-8)<1e-4,"an isolated transient must not veto weak speech");
  for(unsigned i=0;i<50;++i)
    require(std::abs(gain.observe(.0001f,0,false,18,1e-8,0)-8)<1e-4,
            "short syllable gaps must retain the utterance gain");
  for(unsigned i=0;i<350;++i) {
    const float previous=gain.applied();
    const float next=gain.observe(.0001f,0,false,18,1e-8,0);
    require(next<=previous && previous-next<=.06001f,"sustained noise must release monotonically at the configured rate");
  }
  require(gain.applied()<1e-4,"noise gain must return to unity");
  require(std::abs(gain.observe(.004f,.95f,false,18,1e-8,0)-8)<1e-4,"speech must reuse the cached utterance target");
  require(gain.observe(.004f,.95f,true,18,1e-8,0)==0,"far-end speech must disable the boost");

  itu::UtteranceGain released;
  for(unsigned i=0;i<100;++i) released.observe(.0001f,0,false,18,1e-8,0);
  for(unsigned i=0;i<150;++i) released.observe(.004f,.95f,false,18,1e-8,0);
  require(std::abs(released.applied()-8)<1e-4,"qualified weak speech must acquire gain");
  for(unsigned i=0;i<150;++i) {
    const float previous=released.applied();
    const float next=released.observe(.0001f,0,false,18,1e-8,0,6.f,true);
    require(next<=previous && previous-next<=.06001f,
            "opt-in noise-domain release must be bounded and monotonic");
  }
  require(released.applied()<1e-4,"noise-domain release must return gain to unity");

  itu::UtteranceGain normal,unknown,interrupted;
  for(unsigned i=0;i<100;++i) {
    normal.observe(.0001f,0,false,18,1e-8,0);
    interrupted.observe(.0001f,0,false,18,1e-8,0);
  }
  for(unsigned i=0;i<150;++i) {
    require(normal.observe(.07f,.95f,false,18,1e-8,0)==0,"normal-level speech must remain at unity");
    require(unknown.observe(.004f,.95f,false,18,-1,0)==0,"no ambient observation must not imply a known noise floor");
  }
  for(unsigned i=0;i<6;++i) interrupted.observe(.004f,.95f,false,18,1e-8,0);
  interrupted.observe(.004f,.95f,true,18,1e-8,0);
  for(unsigned i=0;i<150;++i)
    require(interrupted.observe(.07f,.95f,false,18,1e-8,0)==0,"far-end interruption must discard incomplete weak-speech acquisition");
  itu::UtteranceGain stale;
  for(unsigned i=0;i<100;++i) stale.observe(.0001f,0,false,18,1e-8,0);
  for(unsigned i=0;i<60;++i) stale.observe(.004f,.5f,false,18,1e-8,0);
  for(unsigned i=0;i<150;++i)
    require(stale.observe(.07f,.95f,false,18,1e-8,0)==0,
            "an expired uncertain observation must not calibrate a later normal utterance");
  std::cout<<"utterance gain controls passed\n";
}
