// Diagnostic wrapper: compare the same frozen CLI with and without x86 FTZ/DAZ.
#include <cstdlib>
#include <iostream>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
#define main itu_file_cli_main
#include "itu_audio_cli.cpp"
#undef main

int main(int argc, char** argv) {
#if defined(__SSE2__)
  const unsigned original = _mm_getcsr();
  const bool flush = std::getenv("ITU_PROBE_FLUSH_DENORMALS") != nullptr;
  if (flush) _mm_setcsr(original | 0x8040u);
  std::cerr << "probe_flush_denormals=" << flush << '\n';
  const int status = itu_file_cli_main(argc, argv);
  _mm_setcsr(original);
  return status;
#else
  std::cerr << "x86 diagnostic only\n";
  return 2;
#endif
}
