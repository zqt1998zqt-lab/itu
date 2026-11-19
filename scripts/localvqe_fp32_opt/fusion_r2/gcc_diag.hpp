#pragma once

// Fixed-capacity, callback-safe GCC-PHAT timing instrumentation.
//
// The hot path only uses clock_gettime() and lock-free GCC/Clang atomics.
// It never prints, opens files, allocates, or takes a user-space lock.
// All stderr output happens in flush(), which must be called after the
// measurement stops (or by the process-exit destructor fallback).

#include <cstdint>
#include <cstdio>
#include <ctime>

namespace itu_gcc_diag {

constexpr unsigned kCapacity = 512;

enum class Phase : unsigned {
    ref_rfft = 0,
    mic_rfft = 1,
    ifft = 2,
};

struct ClockStamp {
    std::uint64_t wall_ns;
    std::uint64_t cpu_ns;
    bool ok;
};

struct PhaseToken {
    ClockStamp start;
};

struct EventToken {
    unsigned slot;
    bool active;
    bool clock_ok;
    ClockStamp enter;
};

namespace detail {

struct Record {
    std::uint32_t seq;
    std::int64_t n_seen;
    std::int64_t frame16;
    std::int64_t block0;
    std::int64_t block1;
    std::uint64_t t_mono_ns;
    std::uint64_t wall_enter_ns;
    std::uint64_t wall_exit_ns;
    std::uint64_t wall_ns;
    std::uint64_t cpu_enter_ns;
    std::uint64_t cpu_exit_ns;
    std::uint64_t cpu_ns;
    std::uint64_t ref_wall_ns;
    std::uint64_t ref_cpu_ns;
    std::uint64_t mic_wall_ns;
    std::uint64_t mic_cpu_ns;
    std::uint64_t ifft_wall_ns;
    std::uint64_t ifft_cpu_ns;
    float conf;
    int shift;
    std::uint8_t keep;
    std::uint8_t ref_ok;
    std::uint8_t mic_ok;
    std::uint8_t ifft_ok;
    std::uint8_t locked;
    std::uint8_t clock_ok;
    std::uint8_t ready;
};

struct State {
    std::uint32_t next_seq;
    std::uint32_t dropped;
    std::uint32_t clock_errors;
    std::uint32_t emitted;
    Record records[kCapacity];
};

// Inline storage gives the DAF TU and an optional main-thread control TU one
// shared event table under C++17, so flush() can be called explicitly after
// the audio callback has stopped.
inline State g_state{};

static inline std::uint64_t timespec_ns(const struct timespec& ts) {
    if (ts.tv_sec < 0 || ts.tv_nsec < 0) return 0;
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

static inline ClockStamp clock_sample() {
    ClockStamp sample{0, 0, false};
    struct timespec wall_ts{};
    struct timespec cpu_ts{};
    const bool wall_ok = ::clock_gettime(CLOCK_MONOTONIC, &wall_ts) == 0;
    const bool cpu_ok =
        ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_ts) == 0;
    if (wall_ok) sample.wall_ns = timespec_ns(wall_ts);
    if (cpu_ok) sample.cpu_ns = timespec_ns(cpu_ts);
    sample.ok = wall_ok && cpu_ok;
    if (!sample.ok) {
        __atomic_fetch_add(&g_state.clock_errors, 1u, __ATOMIC_RELAXED);
    }
    return sample;
}

static inline bool elapsed_ns(const ClockStamp& begin, const ClockStamp& end,
                              std::uint64_t& wall_ns,
                              std::uint64_t& cpu_ns) {
    wall_ns = 0;
    cpu_ns = 0;
    if (!begin.ok || !end.ok || end.wall_ns < begin.wall_ns ||
        end.cpu_ns < begin.cpu_ns) {
        __atomic_fetch_add(&g_state.clock_errors, 1u, __ATOMIC_RELAXED);
        return false;
    }
    wall_ns = end.wall_ns - begin.wall_ns;
    cpu_ns = end.cpu_ns - begin.cpu_ns;
    return true;
}

static inline Record& record_for(const EventToken& token) {
    return g_state.records[token.slot];
}

}  // namespace detail

static inline EventToken begin_event(long long n_seen) {
    EventToken token{0, false, false, {0, 0, false}};
    const unsigned slot =
        __atomic_fetch_add(&detail::g_state.next_seq, 1u, __ATOMIC_RELAXED);
    if (slot >= kCapacity) {
        __atomic_fetch_add(&detail::g_state.dropped, 1u, __ATOMIC_RELAXED);
        return token;
    }

    detail::Record& record = detail::g_state.records[slot];
    const std::int64_t ns = static_cast<std::int64_t>(n_seen);
    const std::int64_t frame16 = ns > 0 ? (ns / 256 - 1) : -1;
    const ClockStamp enter = detail::clock_sample();

    record.seq = slot;
    record.n_seen = ns;
    record.frame16 = frame16;
    record.block0 = frame16 >= 0 ? frame16 * 2 : -1;
    record.block1 = frame16 >= 0 ? frame16 * 2 + 1 : -1;
    record.t_mono_ns = enter.wall_ns;
    record.wall_enter_ns = enter.wall_ns;
    record.cpu_enter_ns = enter.cpu_ns;
    record.conf = 0.0f;
    record.shift = 0;
    record.keep = 0;
    record.ref_ok = 0;
    record.mic_ok = 0;
    record.ifft_ok = 0;
    record.locked = 0;
    record.clock_ok = enter.ok ? 1u : 0u;
    record.ready = 0;

    token.slot = slot;
    token.active = true;
    token.clock_ok = enter.ok;
    token.enter = enter;
    return token;
}

static inline void set_keep(EventToken& token, bool keep) {
    if (!token.active) return;
    detail::record_for(token).keep = keep ? 1u : 0u;
}

static inline PhaseToken begin_phase() {
    return PhaseToken{detail::clock_sample()};
}

static inline void end_phase(EventToken& token, const PhaseToken& phase,
                             Phase kind) {
    if (!token.active) return;
    const ClockStamp end = detail::clock_sample();
    std::uint64_t wall_ns = 0;
    std::uint64_t cpu_ns = 0;
    const bool ok = detail::elapsed_ns(phase.start, end, wall_ns, cpu_ns);
    if (!ok) token.clock_ok = false;

    detail::Record& record = detail::record_for(token);
    switch (kind) {
        case Phase::ref_rfft:
            record.ref_wall_ns = wall_ns;
            record.ref_cpu_ns = cpu_ns;
            record.ref_ok = ok ? 1u : 0u;
            break;
        case Phase::mic_rfft:
            record.mic_wall_ns = wall_ns;
            record.mic_cpu_ns = cpu_ns;
            record.mic_ok = ok ? 1u : 0u;
            break;
        case Phase::ifft:
            record.ifft_wall_ns = wall_ns;
            record.ifft_cpu_ns = cpu_ns;
            record.ifft_ok = ok ? 1u : 0u;
            break;
    }
}

static inline void end_event(EventToken& token, float conf, int shift,
                             bool locked) {
    if (!token.active) return;
    const ClockStamp exit = detail::clock_sample();
    std::uint64_t wall_ns = 0;
    std::uint64_t cpu_ns = 0;
    const bool ok = detail::elapsed_ns(token.enter, exit, wall_ns, cpu_ns);

    detail::Record& record = detail::record_for(token);
    record.wall_exit_ns = exit.wall_ns;
    record.cpu_exit_ns = exit.cpu_ns;
    record.wall_ns = wall_ns;
    record.cpu_ns = cpu_ns;
    record.conf = conf;
    record.shift = shift;
    record.locked = locked ? 1u : 0u;
    record.clock_ok = (token.clock_ok && ok) ? 1u : 0u;

    // Publish only after every field is written. flush() uses acquire loads.
    __atomic_store_n(&record.ready, 1u, __ATOMIC_RELEASE);
    token.active = false;
}

// Call from the main/control thread after the measurement has stopped.
// Returns the number of event records emitted. The first call owns output;
// later calls, including the process-exit fallback, are no-ops.
static inline unsigned flush() {
    unsigned expected = 0;
    if (!__atomic_compare_exchange_n(&detail::g_state.emitted, &expected, 1u,
                                     false, __ATOMIC_ACQ_REL,
                                     __ATOMIC_ACQUIRE)) {
        return 0;
    }

    const unsigned attempted =
        __atomic_load_n(&detail::g_state.next_seq, __ATOMIC_ACQUIRE);
    const unsigned dropped =
        __atomic_load_n(&detail::g_state.dropped, __ATOMIC_ACQUIRE);
    const unsigned clock_errors =
        __atomic_load_n(&detail::g_state.clock_errors, __ATOMIC_ACQUIRE);
    const unsigned limit = attempted < kCapacity ? attempted : kCapacity;
    unsigned emitted = 0;

    std::fprintf(
        stderr,
        "ITU_GCC_DIAG summary schema=1 capacity=%u attempted=%u dropped=%u "
        "clock_errors=%u\n",
        kCapacity, attempted, dropped, clock_errors);

    for (unsigned i = 0; i < limit; ++i) {
        const detail::Record& record = detail::g_state.records[i];
        if (!__atomic_load_n(&record.ready, __ATOMIC_ACQUIRE)) continue;
        std::fprintf(
            stderr,
            "ITU_GCC_DIAG event seq=%u n_seen=%lld frame16=%lld block0=%lld "
            "block1=%lld keep=%u t_mono_ns=%llu wall_enter_ns=%llu "
            "wall_exit_ns=%llu wall_ns=%llu cpu_enter_ns=%llu "
            "cpu_exit_ns=%llu cpu_ns=%llu ref_wall_ns=%llu ref_cpu_ns=%llu "
            "ref_ok=%u mic_wall_ns=%llu mic_cpu_ns=%llu mic_ok=%u "
            "ifft_wall_ns=%llu ifft_cpu_ns=%llu ifft_ok=%u conf=%.9g "
            "shift=%d locked=%u clock_ok=%u\n",
            static_cast<unsigned>(record.seq),
            static_cast<long long>(record.n_seen),
            static_cast<long long>(record.frame16),
            static_cast<long long>(record.block0),
            static_cast<long long>(record.block1),
            static_cast<unsigned>(record.keep),
            static_cast<unsigned long long>(record.t_mono_ns),
            static_cast<unsigned long long>(record.wall_enter_ns),
            static_cast<unsigned long long>(record.wall_exit_ns),
            static_cast<unsigned long long>(record.wall_ns),
            static_cast<unsigned long long>(record.cpu_enter_ns),
            static_cast<unsigned long long>(record.cpu_exit_ns),
            static_cast<unsigned long long>(record.cpu_ns),
            static_cast<unsigned long long>(record.ref_wall_ns),
            static_cast<unsigned long long>(record.ref_cpu_ns),
            static_cast<unsigned>(record.ref_ok),
            static_cast<unsigned long long>(record.mic_wall_ns),
            static_cast<unsigned long long>(record.mic_cpu_ns),
            static_cast<unsigned>(record.mic_ok),
            static_cast<unsigned long long>(record.ifft_wall_ns),
            static_cast<unsigned long long>(record.ifft_cpu_ns),
            static_cast<unsigned>(record.ifft_ok),
            static_cast<double>(record.conf),
            record.shift,
            static_cast<unsigned>(record.locked),
            static_cast<unsigned>(record.clock_ok));
        ++emitted;
    }

    std::fprintf(
        stderr,
        "ITU_GCC_DIAG end schema=1 emitted=%u attempted=%u dropped=%u "
        "clock_errors=%u\n",
        emitted, attempted, dropped, clock_errors);
    return emitted;
}

}  // namespace itu_gcc_diag

#if !defined(ITU_GCC_DIAG_NO_ATEXIT) && \
    (defined(__GNUC__) || defined(__clang__))
__attribute__((destructor)) static void itu_gcc_diag_auto_flush() {
    itu_gcc_diag::flush();
}
#endif
