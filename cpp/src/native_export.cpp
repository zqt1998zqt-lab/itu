#include "itu/audio_engine_c.h"
// Anchor the C ABI object from the static archive into the shared library.
extern "C" const char* itu_audio_native_abi(void) { return itu_audio_version(); }
