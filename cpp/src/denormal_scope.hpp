#pragma once
#include "rtc_base/denormal_disabler.h"
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace itu {
// The upstream x86 guard is Clang-only. Cover GCC and the wrapper DSP too,
// restoring the caller's floating-point mode on every exit from process().
class DenormalScope {
 public:
#if defined(__SSE2__)
  DenormalScope() noexcept : previous_(_mm_getcsr()) {
    _mm_setcsr(previous_ | 0x8040u);
  }
  ~DenormalScope() { _mm_setcsr(previous_); }
#else
  DenormalScope() noexcept = default;
#endif
  DenormalScope(const DenormalScope&) = delete;
  DenormalScope& operator=(const DenormalScope&) = delete;
 private:
#if defined(__SSE2__)
  unsigned previous_;
#else
  webrtc::DenormalDisabler upstream_;
#endif
};
}
