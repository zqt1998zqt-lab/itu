#ifndef ITU_AUDIO_CLOCK_AAUDIO_CLOCK_HPP
#define ITU_AUDIO_CLOCK_AAUDIO_CLOCK_HPP

#include "audio_clock_types_v2.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>
#include <utility>

#if defined(__ANDROID__)
#include <aaudio/AAudio.h>
#endif

#if defined(__ANDROID__) || defined(ITU_AUDIO_CLOCK_FAKE_ANDROID)
#include <time.h>
#include <unistd.h>
#define ITU_AUDIO_CLOCK_ANDROID_IMPL 1
#else
#define ITU_AUDIO_CLOCK_ANDROID_IMPL 0
#endif

namespace itu_audio_clock {

class Runner {
public:
    using Callback =
        std::function<bool(size_t index, int64_t release_expected_ns,
                           int64_t callback_started_ns)>;

    Runner(Config config, Callback callback)
        : config_(std::move(config)), callback_(std::move(callback)) {}

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    Runner(Runner&&) = delete;
    Runner& operator=(Runner&&) = delete;

    Report run(double max_seconds);

private:
    static constexpr int64_t kUnsetTimestamp = std::numeric_limits<int64_t>::min();
    static constexpr int64_t kPollSleepNanoseconds = 10000000;
    static constexpr int64_t kStopWaitNanoseconds = 5000000000LL;

    Report base_report() const;
    void fill_runtime_report(Report& report) const noexcept;

    void set_callback_error(RunError error) noexcept {
        int32_t expected = 0;
        callback_error_code_.compare_exchange_strong(
            expected, static_cast<int32_t>(error), std::memory_order_acq_rel);
        done_.store(true, std::memory_order_release);
    }

    Config config_;
    Callback callback_;
    std::atomic<bool> used_{false};

    std::atomic<bool> done_{false};
    std::atomic<bool> callback_returned_false_{false};
    std::atomic<bool> algorithm_completed_{false};
    std::atomic<bool> callback_exception_{false};
    std::atomic<bool> stream_error_{false};
    std::atomic<bool> main_poll_exception_{false};
    std::atomic<bool> callback_geometry_ok_{true};
    std::atomic<int32_t> callback_error_code_{0};
    std::atomic<int32_t> stream_error_code_{0};
    std::atomic<int32_t> main_error_code_{0};

    std::atomic<int64_t> callback_count_{0};
    std::atomic<int64_t> late_callbacks_{0};
    std::atomic<int64_t> geometry_errors_{0};
    std::atomic<int64_t> max_lateness_ns_{0};
    std::atomic<int64_t> first_callback_started_ns_{kUnsetTimestamp};
    std::atomic<int64_t> first_release_expected_ns_{kUnsetTimestamp};
    std::atomic<int64_t> last_callback_started_ns_{0};
    std::atomic<int64_t> last_release_expected_ns_{0};
    std::atomic<int64_t> main_poll_count_{0};
    std::atomic<int64_t> callback_thread_hook_calls_{0};
    std::atomic<int64_t> callback_started_hook_calls_{0};
    std::atomic<int64_t> callback_ended_hook_calls_{0};
    std::atomic<int32_t> callback_tid_{0};

    std::string main_poll_exception_detail_;

#if ITU_AUDIO_CLOCK_ANDROID_IMPL
    static_assert(std::atomic<int64_t>::is_always_lock_free,
                  "callback timing requires lock-free 64-bit atomics");
    static_assert(std::atomic<int32_t>::is_always_lock_free,
                  "callback timing requires lock-free 32-bit atomics");

    static bool monotonic_now_ns(int64_t* value) noexcept {
        timespec current{};
        if (::clock_gettime(CLOCK_MONOTONIC, &current) != 0) {
            return false;
        }
        *value = static_cast<int64_t>(current.tv_sec) * 1000000000LL +
                 static_cast<int64_t>(current.tv_nsec);
        return true;
    }

    static void sleep_short_ns(int64_t nanoseconds) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(nanoseconds));
    }

    static aaudio_data_callback_result_t data_callback(
        AAudioStream*, void* user_data, void* audio_data, int32_t num_frames) noexcept {
        auto* self = static_cast<Runner*>(user_data);
        if (self == nullptr) {
            return AAUDIO_CALLBACK_RESULT_STOP;
        }
        return self->on_data_callback(audio_data, num_frames);
    }

    static void error_callback(
        AAudioStream*, void* user_data, aaudio_result_t error) noexcept {
        auto* self = static_cast<Runner*>(user_data);
        if (self == nullptr) {
            return;
        }
        int32_t expected = 0;
        self->stream_error_code_.compare_exchange_strong(
            expected, error, std::memory_order_acq_rel);
        self->stream_error_.store(true, std::memory_order_release);
        self->done_.store(true, std::memory_order_release);
    }

    aaudio_data_callback_result_t on_data_callback(
        void* audio_data, int32_t num_frames) noexcept {
        const int64_t index =
            callback_count_.fetch_add(1, std::memory_order_acq_rel);
        const int32_t tid = static_cast<int32_t>(::gettid());

        try {
            const bool geometry_ok =
                audio_data != nullptr && num_frames > 0 &&
                num_frames == config_.requested_frames_per_callback &&
                config_.requested_channel_count > 0;
            if (geometry_ok) {
                const size_t sample_count =
                    static_cast<size_t>(num_frames) *
                    static_cast<size_t>(config_.requested_channel_count);
                std::memset(audio_data, 0, sample_count * sizeof(float));
            } else {
                geometry_errors_.fetch_add(1, std::memory_order_relaxed);
                callback_geometry_ok_.store(false, std::memory_order_release);
                if (audio_data != nullptr && num_frames > 0 &&
                    num_frames <= config_.requested_frames_per_callback) {
                    const size_t sample_count =
                        static_cast<size_t>(num_frames) *
                        static_cast<size_t>(config_.requested_channel_count);
                    std::memset(audio_data, 0, sample_count * sizeof(float));
                }
                set_callback_error(RunError::CallbackGeometryMismatch);
                return AAUDIO_CALLBACK_RESULT_STOP;
            }

            int64_t started_ns = 0;
            if (!monotonic_now_ns(&started_ns)) {
                set_callback_error(RunError::ClockReadFailed);
                return AAUDIO_CALLBACK_RESULT_STOP;
            }

            int64_t first_started =
                first_callback_started_ns_.load(std::memory_order_acquire);
            if (first_started == kUnsetTimestamp) {
                int64_t expected = kUnsetTimestamp;
                if (first_callback_started_ns_.compare_exchange_strong(
                        expected, started_ns, std::memory_order_acq_rel)) {
                    first_started = started_ns;
                    first_release_expected_ns_.store(
                        started_ns, std::memory_order_release);
                } else {
                    first_started = expected;
                }
            }

            const int64_t first_expected =
                first_release_expected_ns_.load(std::memory_order_acquire);
            int64_t release_expected_ns = first_expected;
            if (index > 0 && first_expected != kUnsetTimestamp) {
                const int64_t maximum_index =
                    (std::numeric_limits<int64_t>::max() - first_expected) /
                    kCallbackHopNanoseconds;
                if (index > maximum_index) {
                    set_callback_error(RunError::ClockReadFailed);
                    return AAUDIO_CALLBACK_RESULT_STOP;
                }
                release_expected_ns =
                    first_expected + index * kCallbackHopNanoseconds;
            }

            last_callback_started_ns_.store(
                started_ns, std::memory_order_release);
            last_release_expected_ns_.store(
                release_expected_ns, std::memory_order_release);
            if (started_ns > release_expected_ns) {
                const int64_t lateness = started_ns - release_expected_ns;
                late_callbacks_.fetch_add(1, std::memory_order_relaxed);
                if (lateness >
                    max_lateness_ns_.load(std::memory_order_relaxed)) {
                    max_lateness_ns_.store(
                        lateness, std::memory_order_release);
                }
            }

            const int32_t previous_tid =
                callback_tid_.load(std::memory_order_acquire);
            if (previous_tid != tid) {
                callback_tid_.store(tid, std::memory_order_release);
                if (config_.on_callback_thread) {
                    callback_thread_hook_calls_.fetch_add(
                        1, std::memory_order_relaxed);
                    config_.on_callback_thread(tid);
                }
            }

            bool started_hook_called = false;
            bool ended_hook_called = false;
            try {
                if (config_.on_callback_started) {
                    started_hook_called = true;
                    callback_started_hook_calls_.fetch_add(
                        1, std::memory_order_relaxed);
                    config_.on_callback_started();
                }

                const bool keep_running =
                    callback_(static_cast<size_t>(index),
                              release_expected_ns, started_ns);

                if (config_.on_callback_ended) {
                    ended_hook_called = true;
                    callback_ended_hook_calls_.fetch_add(
                        1, std::memory_order_relaxed);
                    config_.on_callback_ended();
                }

                if (!keep_running) {
                    callback_returned_false_.store(
                        true, std::memory_order_release);
                    algorithm_completed_.store(
                        true, std::memory_order_release);
                    done_.store(true, std::memory_order_release);
                    return AAUDIO_CALLBACK_RESULT_STOP;
                }
                return AAUDIO_CALLBACK_RESULT_CONTINUE;
            } catch (...) {
                if (started_hook_called && !ended_hook_called &&
                    config_.on_callback_ended) {
                    ended_hook_called = true;
                    callback_ended_hook_calls_.fetch_add(
                        1, std::memory_order_relaxed);
                    try {
                        config_.on_callback_ended();
                    } catch (...) {
                    }
                }
                throw;
            }
        } catch (...) {
            callback_exception_.store(true, std::memory_order_release);
            set_callback_error(RunError::CallbackException);
            return AAUDIO_CALLBACK_RESULT_STOP;
        }
    }
#endif
};

inline Report Runner::base_report() const {
    Report report;
#if ITU_AUDIO_CLOCK_ANDROID_IMPL
    report.api_available = true;
    report.full_algorithm_callback_contract = true;
#endif
    report.request_sample_rate = config_.requested_sample_rate;
    report.request_channel_count = config_.requested_channel_count;
    report.request_format = config_.requested_format;
    report.request_frames_per_callback = config_.requested_frames_per_callback;
    report.request_performance_mode =
        static_cast<int32_t>(config_.performance_preference);
    report.request_sharing_mode =
        static_cast<int32_t>(config_.sharing_preference);
    report.request_buffer_burst_multiplier =
        config_.buffer_size_burst_multiplier;
    report.expected_hop_ns = kCallbackHopNanoseconds;
    return report;
}

inline void Runner::fill_runtime_report(Report& report) const noexcept {
    report.done = report.done || done_.load(std::memory_order_acquire);
    report.completed =
        algorithm_completed_.load(std::memory_order_acquire);
    report.callback_returned_false =
        callback_returned_false_.load(std::memory_order_acquire);
    report.callback_exception =
        callback_exception_.load(std::memory_order_acquire);
    report.stream_error = stream_error_.load(std::memory_order_acquire);
    report.main_poll_exception =
        main_poll_exception_.load(std::memory_order_acquire);
    report.callback_geometry_ok =
        callback_geometry_ok_.load(std::memory_order_acquire);
    report.stream_error_code =
        stream_error_code_.load(std::memory_order_acquire);

    report.callback_count = callback_count_.load(std::memory_order_acquire);
    report.late_callbacks = late_callbacks_.load(std::memory_order_acquire);
    report.geometry_errors = geometry_errors_.load(std::memory_order_acquire);
    report.max_lateness_ns = max_lateness_ns_.load(std::memory_order_acquire);
    const int64_t first_started =
        first_callback_started_ns_.load(std::memory_order_acquire);
    const int64_t first_expected =
        first_release_expected_ns_.load(std::memory_order_acquire);
    report.callback_cadence_timebase_established =
        first_started != kUnsetTimestamp && first_expected != kUnsetTimestamp;
    report.first_callback_started_ns =
        first_started == kUnsetTimestamp ? 0 : first_started;
    report.first_release_expected_ns =
        first_expected == kUnsetTimestamp ? 0 : first_expected;
    report.last_callback_started_ns =
        last_callback_started_ns_.load(std::memory_order_acquire);
    report.last_release_expected_ns =
        last_release_expected_ns_.load(std::memory_order_acquire);
    report.main_poll_count = main_poll_count_.load(std::memory_order_acquire);
    report.callback_thread_hook_calls =
        callback_thread_hook_calls_.load(std::memory_order_acquire);
    report.callback_started_hook_calls =
        callback_started_hook_calls_.load(std::memory_order_acquire);
    report.callback_ended_hook_calls =
        callback_ended_hook_calls_.load(std::memory_order_acquire);
    report.callback_tid = callback_tid_.load(std::memory_order_acquire);
}

#if !ITU_AUDIO_CLOCK_ANDROID_IMPL

inline Report Runner::run(double) {
    Report report = base_report();
    report.api_available = false;
    report.done = true;
    report.callback_quiesced = true;
    report.error = RunError::NotAndroid;
    report.error_detail = "AAudio runner is only available on Android";
    if (used_.exchange(true, std::memory_order_acq_rel)) {
        report.error = RunError::AlreadyRun;
        report.error_detail = "Runner::run may be called only once";
    }
    return report;
}

#else

inline Report Runner::run(double max_seconds) {
    Report report = base_report();
    if (used_.exchange(true, std::memory_order_acq_rel)) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::AlreadyRun;
        report.error_detail = "Runner::run may be called only once";
        return report;
    }

    const bool valid_modes =
        (config_.sharing_preference == SharingPreference::Exclusive ||
         config_.sharing_preference == SharingPreference::Shared) &&
        (config_.performance_preference == PerformancePreference::None ||
         config_.performance_preference == PerformancePreference::PowerSaving ||
         config_.performance_preference == PerformancePreference::LowLatency);
    if (config_.requested_sample_rate != kRequestedSampleRate ||
        config_.requested_channel_count != kRequestedChannelCount ||
        config_.requested_format != kAaudioFormatPcmFloat ||
        config_.requested_frames_per_callback !=
            kRequestedFramesPerCallback ||
        config_.buffer_size_burst_multiplier <= 0 || !valid_modes ||
        !std::isfinite(max_seconds) || max_seconds <= 0.0) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::InvalidConfig;
        report.error_detail =
            "expected 48000 Hz, mono float, 768-frame callbacks, positive buffer "
            "multiplier and positive finite max_seconds";
        return report;
    }
    if (!callback_) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::CallbackMissing;
        report.error_detail = "algorithm callback is empty";
        return report;
    }

    const long double timeout_seconds =
        static_cast<long double>(max_seconds) + 10.0L;
    const long double timeout_ns_long = timeout_seconds * 1000000000.0L;
    if (timeout_ns_long >
        static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::InvalidConfig;
        report.error_detail = "max_seconds is too large";
        return report;
    }
    const int64_t timeout_ns = static_cast<int64_t>(timeout_ns_long);

    int64_t run_started_ns = 0;
    if (!monotonic_now_ns(&run_started_ns) ||
        run_started_ns >
            std::numeric_limits<int64_t>::max() - timeout_ns) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::ClockReadFailed;
        report.error_detail = "CLOCK_MONOTONIC read or deadline calculation failed";
        return report;
    }
    const int64_t deadline_ns = run_started_ns + timeout_ns;

    AAudioStreamBuilder* builder = nullptr;
    report.builder_create_result = AAudio_createStreamBuilder(&builder);
    if (report.builder_create_result != AAUDIO_OK || builder == nullptr) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::BuilderCreateFailed;
        report.error_detail = "AAudio_createStreamBuilder failed";
        fill_runtime_report(report);
        return report;
    }

    AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setSampleRate(
        builder, config_.requested_sample_rate);
    AAudioStreamBuilder_setChannelCount(
        builder, config_.requested_channel_count);
    AAudioStreamBuilder_setFormat(builder, config_.requested_format);
    AAudioStreamBuilder_setFramesPerDataCallback(
        builder, config_.requested_frames_per_callback);
    AAudioStreamBuilder_setPerformanceMode(
        builder, static_cast<aaudio_performance_mode_t>(
                     config_.performance_preference));
    AAudioStreamBuilder_setSharingMode(
        builder, static_cast<aaudio_sharing_mode_t>(
                     config_.sharing_preference));
    AAudioStreamBuilder_setDataCallback(builder, &Runner::data_callback, this);
    AAudioStreamBuilder_setErrorCallback(builder, &Runner::error_callback, this);

    AAudioStream* stream = nullptr;
    report.open_result = AAudioStreamBuilder_openStream(builder, &stream);
    report.builder_delete_result = AAudioStreamBuilder_delete(builder);
    builder = nullptr;
    if (report.open_result != AAUDIO_OK || stream == nullptr) {
        report.done = true;
        report.callback_quiesced = true;
        report.error = RunError::OpenFailed;
        report.error_detail = "AAudioStreamBuilder_openStream failed";
        fill_runtime_report(report);
        return report;
    }

    report.actual_sample_rate = AAudioStream_getSampleRate(stream);
    report.actual_channel_count = AAudioStream_getChannelCount(stream);
    report.actual_format = AAudioStream_getFormat(stream);
    report.actual_performance_mode =
        AAudioStream_getPerformanceMode(stream);
    report.actual_sharing_mode = AAudioStream_getSharingMode(stream);
    report.actual_frames_per_burst =
        AAudioStream_getFramesPerBurst(stream);
    report.actual_frames_per_callback =
        AAudioStream_getFramesPerDataCallback(stream);
    report.actual_buffer_capacity_frames =
        AAudioStream_getBufferCapacityInFrames(stream);
    report.actual_buffer_size_frames =
        AAudioStream_getBufferSizeInFrames(stream);
    report.open_state = AAudioStream_getState(stream);
    report.final_state = report.open_state;

    report.sharing_fallback =
        report.actual_sharing_mode != report.request_sharing_mode;
    report.performance_fallback =
        report.actual_performance_mode != report.request_performance_mode;
    report.requested_modes_met =
        !report.sharing_fallback && !report.performance_fallback;

    if (report.actual_frames_per_burst > 0 &&
        config_.buffer_size_burst_multiplier <=
            std::numeric_limits<int32_t>::max() /
                report.actual_frames_per_burst) {
        report.request_buffer_size_frames =
            config_.buffer_size_burst_multiplier *
            report.actual_frames_per_burst;
        report.buffer_set_result = AAudioStream_setBufferSizeInFrames(
            stream, report.request_buffer_size_frames);
        report.actual_buffer_size_frames =
            AAudioStream_getBufferSizeInFrames(stream);
        report.actual_buffer_capacity_frames =
            AAudioStream_getBufferCapacityInFrames(stream);
    }

    report.geometry_ok =
        report.actual_sample_rate == kRequestedSampleRate &&
        report.actual_channel_count == kRequestedChannelCount &&
        report.actual_format == kAaudioFormatPcmFloat &&
        report.actual_frames_per_callback ==
            kRequestedFramesPerCallback &&
        report.actual_frames_per_burst > 0 &&
        report.request_buffer_size_frames > 0;

    if (!report.geometry_ok) {
        report.final_state = AAudioStream_getState(stream);
        report.close_result = AAudioStream_close(stream);
        report.closed = true;
        report.callback_quiesced = true;
        report.done = true;
        report.error = RunError::GeometryMismatch;
        report.error_detail =
            "AAudio did not return 48000 Hz mono float with 768-frame "
            "callbacks and a positive burst";
        fill_runtime_report(report);
        int64_t ended_ns = 0;
        if (monotonic_now_ns(&ended_ns) && ended_ns >= run_started_ns) {
            report.observed_seconds =
                static_cast<double>(ended_ns - run_started_ns) / 1.0e9;
        }
        report.stop_clean = report.closed && report.close_result == AAUDIO_OK;
        return report;
    }

    report.setup_ok = true;
    report.xrun_count_before = AAudioStream_getXRunCount(stream);
    report.start_result = AAudioStream_requestStart(stream);
    if (report.start_result != AAUDIO_OK) {
        report.final_state = AAudioStream_getState(stream);
        report.close_result = AAudioStream_close(stream);
        report.closed = true;
        report.callback_quiesced = true;
        report.done = true;
        report.error = RunError::StartFailed;
        report.error_detail = "AAudioStream_requestStart failed";
        fill_runtime_report(report);
        int64_t ended_ns = 0;
        if (monotonic_now_ns(&ended_ns) && ended_ns >= run_started_ns) {
            report.observed_seconds =
                static_cast<double>(ended_ns - run_started_ns) / 1.0e9;
        }
        report.stop_clean = report.closed && report.close_result == AAUDIO_OK;
        return report;
    }
    report.stream_started = true;

    bool timed_out = false;
    bool unexpected_stream_state = false;
    while (true) {
        main_poll_count_.fetch_add(1, std::memory_order_relaxed);
        if (config_.on_main_poll) {
            try {
                config_.on_main_poll();
            } catch (const std::exception& error) {
                main_poll_exception_detail_ = error.what();
                main_poll_exception_.store(true, std::memory_order_release);
                main_error_code_.store(
                    static_cast<int32_t>(RunError::MainPollException),
                    std::memory_order_release);
                done_.store(true, std::memory_order_release);
            } catch (...) {
                main_poll_exception_detail_ = "non-standard exception";
                main_poll_exception_.store(true, std::memory_order_release);
                main_error_code_.store(
                    static_cast<int32_t>(RunError::MainPollException),
                    std::memory_order_release);
                done_.store(true, std::memory_order_release);
            }
        }

        if (done_.load(std::memory_order_acquire)) {
            break;
        }

        const int32_t state = AAudioStream_getState(stream);
        if (state == AAUDIO_STREAM_STATE_DISCONNECTED ||
            state == AAUDIO_STREAM_STATE_CLOSED ||
            state == AAUDIO_STREAM_STATE_STOPPED) {
            unexpected_stream_state = true;
            done_.store(true, std::memory_order_release);
            break;
        }

        int64_t now_ns = 0;
        if (!monotonic_now_ns(&now_ns)) {
            main_poll_exception_detail_ = "CLOCK_MONOTONIC poll failed";
            main_error_code_.store(
                static_cast<int32_t>(RunError::ClockReadFailed),
                std::memory_order_release);
            done_.store(true, std::memory_order_release);
            break;
        }
        if (now_ns >= deadline_ns) {
            timed_out = true;
            done_.store(true, std::memory_order_release);
            break;
        }
        sleep_short_ns(kPollSleepNanoseconds);
    }

    report.stop_requested = true;
    report.stop_request_result = AAudioStream_requestStop(stream);

    bool callback_quiesced = false;
    int64_t stop_wait_deadline_ns = 0;
    if (monotonic_now_ns(&stop_wait_deadline_ns)) {
        stop_wait_deadline_ns += kStopWaitNanoseconds;
    } else {
        stop_wait_deadline_ns = 0;
    }

    int32_t final_state = AAudioStream_getState(stream);
    while (true) {
        if (final_state == AAUDIO_STREAM_STATE_STOPPED ||
            final_state == AAUDIO_STREAM_STATE_CLOSED ||
            final_state == AAUDIO_STREAM_STATE_DISCONNECTED) {
            callback_quiesced = true;
            break;
        }
        int64_t now_ns = 0;
        if (stop_wait_deadline_ns == 0 ||
            !monotonic_now_ns(&now_ns) || now_ns >= stop_wait_deadline_ns) {
            break;
        }
        aaudio_stream_state_t observed =
            static_cast<aaudio_stream_state_t>(final_state);
        aaudio_stream_state_t next_state =
            static_cast<aaudio_stream_state_t>(final_state);
        report.wait_for_state_result = AAudioStream_waitForStateChange(
            stream, observed, &next_state, kPollSleepNanoseconds);
        final_state = report.wait_for_state_result == AAUDIO_OK
                          ? static_cast<int32_t>(next_state)
                          : AAudioStream_getState(stream);
    }

    report.xrun_count_after = AAudioStream_getXRunCount(stream);
    if(report.xrun_count_before >= 0 && report.xrun_count_after >= report.xrun_count_before) report.xrun_delta = report.xrun_count_after-report.xrun_count_before;
    report.final_state = final_state;
    report.callback_quiesced = callback_quiesced;
    report.forced_close_after_stop_timeout = !callback_quiesced;
    report.close_result = AAudioStream_close(stream);
    report.closed = true;
    report.stop_clean =
        report.callback_quiesced && report.close_result == AAUDIO_OK;

    const int32_t callback_error =
        callback_error_code_.load(std::memory_order_acquire);
    const int32_t main_error =
        main_error_code_.load(std::memory_order_acquire);
    if (callback_exception_.load(std::memory_order_acquire)) {
        report.error = RunError::CallbackException;
        report.error_detail =
            "callback or callback-thread hook threw; exception text was not "
            "captured in the real-time callback";
    } else if (callback_error != 0) {
        report.error = static_cast<RunError>(callback_error);
        report.error_detail = "real-time callback reported an error";
    } else if (stream_error_.load(std::memory_order_acquire)) {
        report.error = RunError::StreamError;
        report.error_detail = "AAudio error callback reported a stream error";
    } else if (main_error != 0) {
        report.error = static_cast<RunError>(main_error);
        report.error_detail = main_poll_exception_detail_.empty()
                                  ? "main polling thread reported an error"
                                  : main_poll_exception_detail_;
    } else if (timed_out) {
        report.error = RunError::Timeout;
        report.error_detail = "run exceeded max_seconds plus 10 seconds";
    } else if (unexpected_stream_state) {
        report.error = RunError::UnexpectedStreamState;
        report.error_detail =
            "stream reached a terminal state before the algorithm callback "
            "requested completion";
    } else if (!callback_quiesced) {
        report.error = RunError::StopWaitFailed;
        report.error_detail =
            "AAudio did not confirm callback quiescence before forced close";
    } else if (report.close_result != AAUDIO_OK) {
        report.error = RunError::CloseFailed;
        report.error_detail = "AAudioStream_close failed";
    } else {
        report.error = RunError::None;
        report.error_detail.clear();
    }

    report.done = true;
    report.timed_out = timed_out;
    int64_t ended_ns = 0;
    if (monotonic_now_ns(&ended_ns) && ended_ns >= run_started_ns) {
        report.observed_seconds =
            static_cast<double>(ended_ns - run_started_ns) / 1.0e9;
    }
    fill_runtime_report(report);
    report.done = true;
    return report;
}

#endif

}  // namespace itu_audio_clock

#undef ITU_AUDIO_CLOCK_ANDROID_IMPL

#endif  // ITU_AUDIO_CLOCK_AAUDIO_CLOCK_HPP
