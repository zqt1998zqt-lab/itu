#pragma once
#include <algorithm>
#include <cmath>
namespace itu {
// Render-conditioned DTD, not a speech classifier. Baselines learn only when
// the current energy remains compatible with previously observed echo/noise.
class EnergyDtd {
 public:
  unsigned observe(bool far,bool trusted,float residual,float unexplained) noexcept {
    if(!far || !trusted || !std::isfinite(residual) || !std::isfinite(unexplained)) {
      confirmation_=clear_=hold_=0;return 0;
    }
    if(!learned_) {floor_=std::clamp(residual,1e-6f,1.f);coherence_=std::clamp(unexplained,0.f,1.f);learned_=true;}
    const bool innovation=residual>std::max(.015f,1.5f*floor_) &&
        unexplained>std::min(.97f,std::max(.03f,1.12f*coherence_+.01f));
    const bool echo=residual<std::max(.015f,1.4f*floor_) &&
        unexplained<1.3f*coherence_+.01f;
    confirmation_=innovation ? std::min(3u,confirmation_+1) : 0;
    clear_=echo ? std::min(10u,clear_+1) : 0;
    if(confirmation_>=3)hold_=200;
    else if(clear_>=10)hold_=0;
    else if(hold_)--hold_;
    // Fast downward learning, very slow upward drift. Neither double talk nor
    // its release interval can train the background estimate upward.
    if(residual<floor_)floor_+=.1f*(residual-floor_);
    else if(!innovation && !hold_)floor_+=.002f*(residual-floor_);
    if(unexplained<coherence_)coherence_+=.1f*(unexplained-coherence_);
    else if(!innovation && !hold_)coherence_+=.002f*(unexplained-coherence_);
    return hold_;
  }
 private:
  float floor_=1,coherence_=1;bool learned_=false;
  unsigned confirmation_=0,clear_=0,hold_=0;
};
}
