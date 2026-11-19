#define ITU_GCC_DIAG_NO_ATEXIT 1
#include "gcc_diag.hpp"

#include <cstdint>

static volatile std::uint64_t burn_sink;

static void burn_cpu() {
    std::uint64_t value = 0;
    for (std::uint32_t i = 0; i < 200000; ++i) {
        value += static_cast<std::uint64_t>(i) * 3u;
    }
    burn_sink = value;
}

int main() {
    itu_gcc_diag::EventToken event = itu_gcc_diag::begin_event(16384);
    if (!event.active) return 2;
    itu_gcc_diag::set_keep(event, true);

    const itu_gcc_diag::PhaseToken ref =
        itu_gcc_diag::begin_phase();
    burn_cpu();
    itu_gcc_diag::end_phase(event, ref, itu_gcc_diag::Phase::ref_rfft);

    const itu_gcc_diag::PhaseToken mic =
        itu_gcc_diag::begin_phase();
    burn_cpu();
    itu_gcc_diag::end_phase(event, mic, itu_gcc_diag::Phase::mic_rfft);

    const itu_gcc_diag::PhaseToken ifft =
        itu_gcc_diag::begin_phase();
    burn_cpu();
    itu_gcc_diag::end_phase(event, ifft, itu_gcc_diag::Phase::ifft);

    itu_gcc_diag::end_event(event, 9.5f, 128, true);
    if (itu_gcc_diag::flush() != 1) return 3;
    if (itu_gcc_diag::flush() != 0) return 4;
    return 0;
}
