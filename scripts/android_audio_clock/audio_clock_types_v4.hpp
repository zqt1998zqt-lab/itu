#ifndef ITU_AUDIO_CLOCK_TYPES_V4_HPP
#define ITU_AUDIO_CLOCK_TYPES_V4_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace itu_audio_clock {

inline constexpr int32_t kRequestedSampleRate = 48000;
inline constexpr int32_t kRequestedChannelCount = 1;
inline constexpr int32_t kRequestedFramesPerCallback = 768;
inline constexpr int64_t kCallbackHopNanoseconds = 16000000;
inline constexpr int32_t kAaudioFormatPcmFloat = 2;

enum class SharingPreference : int32_t {
    Exclusive = 0,
    Shared = 1,
};

enum class PerformancePreference : int32_t {
    None = 10,
    PowerSaving = 11,
    LowLatency = 12,
};

enum class RunError : int32_t {
    None = 0,
    NotAndroid = 1,
    AlreadyRun = 2,
    InvalidConfig = 3,
    CallbackMissing = 4,
    ClockReadFailed = 5,
    BuilderCreateFailed = 6,
    OpenFailed = 7,
    GeometryMismatch = 8,
    StartFailed = 9,
    CallbackGeometryMismatch = 10,
    CallbackException = 11,
    StreamError = 12,
    Timeout = 13,
    MainPollException = 14,
    UnexpectedStreamState = 15,
    StopWaitFailed = 16,
    CloseFailed = 17,
    StopRequestFailed = 18,
};

inline const char* to_string(RunError value) noexcept {
    switch (value) {
        case RunError::None: return "none";
        case RunError::NotAndroid: return "not_android";
        case RunError::AlreadyRun: return "already_run";
        case RunError::InvalidConfig: return "invalid_config";
        case RunError::CallbackMissing: return "callback_missing";
        case RunError::ClockReadFailed: return "clock_read_failed";
        case RunError::BuilderCreateFailed: return "builder_create_failed";
        case RunError::OpenFailed: return "open_failed";
        case RunError::GeometryMismatch: return "geometry_mismatch";
        case RunError::StartFailed: return "start_failed";
        case RunError::CallbackGeometryMismatch: return "callback_geometry_mismatch";
        case RunError::CallbackException: return "callback_exception";
        case RunError::StreamError: return "stream_error";
        case RunError::Timeout: return "timeout";
        case RunError::MainPollException: return "main_poll_exception";
        case RunError::UnexpectedStreamState: return "unexpected_stream_state";
        case RunError::StopWaitFailed: return "stop_wait_failed";
        case RunError::CloseFailed: return "close_failed";
        case RunError::StopRequestFailed: return "stop_request_failed";
    }
    return "unknown";
}

inline const char* sharing_mode_name(int32_t value) noexcept {
    switch (value) {
        case 0: return "exclusive";
        case 1: return "shared";
        default: return "unknown";
    }
}

inline const char* performance_mode_name(int32_t value) noexcept {
    switch (value) {
        case 10: return "none";
        case 11: return "power_saving";
        case 12: return "low_latency";
        default: return "unknown";
    }
}

inline const char* format_name(int32_t value) noexcept {
    switch (value) {
        case 2: return "pcm_float";
        default: return "unknown";
    }
}

inline const char* stream_state_name(int32_t value) noexcept {
    switch (value) {
        case 0: return "uninitialized";
        case 1: return "unknown";
        case 2: return "open";
        case 3: return "starting";
        case 4: return "started";
        case 5: return "pausing";
        case 6: return "paused";
        case 7: return "flushing";
        case 8: return "flushed";
        case 9: return "stopping";
        case 10: return "stopped";
        case 11: return "closing";
        case 12: return "closed";
        case 13: return "disconnected";
        default: return "unknown";
    }
}

struct Config {
    int32_t requested_sample_rate = kRequestedSampleRate;
    int32_t requested_channel_count = kRequestedChannelCount;
    int32_t requested_format = kAaudioFormatPcmFloat;
    int32_t requested_frames_per_callback = kRequestedFramesPerCallback;
    PerformancePreference performance_preference = PerformancePreference::LowLatency;
    SharingPreference sharing_preference = SharingPreference::Exclusive;
    int32_t buffer_size_burst_multiplier = 2;

    // Runs on the non-real-time run() polling thread. ADPF drain belongs here.
    std::function<void()> on_main_poll;

    // Runs once per observed callback-thread TID change. Store the TID atomically.
    // The hook must be real-time safe and must not allocate, block, log, or do I/O.
    std::function<void(int32_t)> on_callback_thread;

    // Reserved callback-local ADPF start/end hooks. Both must be real-time safe.
    std::function<void()> on_callback_started;
    std::function<void()> on_callback_ended;

    // Reserved for a future android_performance_hint.hpp integration. This harness
    // never dereferences or owns the pointer.
    void* performance_hint_context = nullptr;
};

struct Report {
    bool api_available = false;
    bool setup_ok = false;
    bool stream_started = false;
    bool done = false;
    bool completed = false;
    bool callback_returned_false = false;
    bool stop_requested = false;
    bool timed_out = false;
    bool stream_error = false;
    bool callback_exception = false;
    bool main_poll_exception = false;
    bool closed = false;
    bool stop_clean = false;
    bool callback_quiesced = false;
    bool forced_close_after_stop_timeout = false;

    bool geometry_ok = false;
    bool callback_geometry_ok = false;
    bool sharing_fallback = false;
    bool performance_fallback = false;
    bool requested_modes_met = false;
    bool callback_cadence_timebase_established = false;
    bool hardware_presentation_time_available = false;
    bool low_latency_guaranteed = false;
    bool silent_output = true;
    bool full_algorithm_callback_contract = false;
    bool audio_record_track_quality_validation = false;

    RunError error = RunError::None;
    std::string error_detail;

    int32_t request_sample_rate = kRequestedSampleRate;
    int32_t request_channel_count = kRequestedChannelCount;
    int32_t request_format = kAaudioFormatPcmFloat;
    int32_t request_frames_per_callback = kRequestedFramesPerCallback;
    int32_t request_performance_mode =
        static_cast<int32_t>(PerformancePreference::LowLatency);
    int32_t request_sharing_mode = static_cast<int32_t>(SharingPreference::Exclusive);
    int32_t request_buffer_burst_multiplier = 2;
    int32_t request_buffer_size_frames = 0;

    int32_t actual_sample_rate = -1;
    int32_t actual_channel_count = -1;
    int32_t actual_format = -1;
    int32_t actual_performance_mode = -1;
    int32_t actual_sharing_mode = -1;
    int32_t actual_frames_per_burst = -1;
    int32_t actual_frames_per_callback = -1;
    int32_t actual_buffer_size_frames = -1;
    int32_t actual_buffer_capacity_frames = -1;
    int32_t buffer_set_result = 0;
    int32_t xrun_count_before = -1;
    int32_t xrun_count_after = -1;
    int32_t xrun_delta = -1;

    int32_t builder_create_result = 0;
    int32_t builder_delete_result = 0;
    int32_t open_result = 0;
    int32_t start_result = 0;
    int32_t stop_request_result = 0;
    int32_t wait_for_state_result = 0;
    int32_t close_result = 0;
    int32_t stream_error_code = 0;
    int32_t open_state = -1;
    int32_t final_state = -1;

    int64_t expected_hop_ns = kCallbackHopNanoseconds;
    int64_t callback_count = 0;
    int64_t late_callbacks = 0;
    int64_t geometry_errors = 0;
    int64_t max_lateness_ns = 0;
    int64_t first_callback_started_ns = 0;
    int64_t first_release_expected_ns = 0;
    int64_t last_callback_started_ns = 0;
    int64_t last_release_expected_ns = 0;
    int64_t main_poll_count = 0;
    int64_t callback_thread_hook_calls = 0;
    int64_t callback_started_hook_calls = 0;
    int64_t callback_ended_hook_calls = 0;
    int32_t callback_tid = 0;
    double observed_seconds = 0.0;

    std::string to_json() const;
};

inline std::string json_escape(const std::string& value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped += "\\u00";
                    escaped += kHex[(ch >> 4) & 0x0f];
                    escaped += kHex[ch & 0x0f];
                } else {
                    escaped.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return escaped;
}

inline std::string Report::to_json() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << '{';
    out << "\"api_available\":" << (api_available ? "true" : "false");
    out << ",\"setup_ok\":" << (setup_ok ? "true" : "false");
    out << ",\"stream_started\":" << (stream_started ? "true" : "false");
    out << ",\"done\":" << (done ? "true" : "false");
    out << ",\"completed\":" << (completed ? "true" : "false");
    out << ",\"callback_returned_false\":"
        << (callback_returned_false ? "true" : "false");
    out << ",\"stop_requested\":" << (stop_requested ? "true" : "false");
    out << ",\"timed_out\":" << (timed_out ? "true" : "false");
    out << ",\"stream_error\":" << (stream_error ? "true" : "false");
    out << ",\"callback_exception\":" << (callback_exception ? "true" : "false");
    out << ",\"main_poll_exception\":"
        << (main_poll_exception ? "true" : "false");
    out << ",\"closed\":" << (closed ? "true" : "false");
    out << ",\"stop_clean\":" << (stop_clean ? "true" : "false");
    out << ",\"callback_quiesced\":"
        << (callback_quiesced ? "true" : "false");
    out << ",\"forced_close_after_stop_timeout\":"
        << (forced_close_after_stop_timeout ? "true" : "false");
    out << ",\"geometry_ok\":" << (geometry_ok ? "true" : "false");
    out << ",\"callback_geometry_ok\":"
        << (callback_geometry_ok ? "true" : "false");
    out << ",\"sharing_fallback\":" << (sharing_fallback ? "true" : "false");
    out << ",\"performance_fallback\":"
        << (performance_fallback ? "true" : "false");
    out << ",\"requested_modes_met\":"
        << (requested_modes_met ? "true" : "false");
    out << ",\"callback_cadence_timebase_established\":"
        << (callback_cadence_timebase_established ? "true" : "false");
    out << ",\"hardware_presentation_time_available\":"
        << (hardware_presentation_time_available ? "true" : "false");
    out << ",\"low_latency_guaranteed\":"
        << (low_latency_guaranteed ? "true" : "false");
    out << ",\"silent_output\":" << (silent_output ? "true" : "false");
    out << ",\"full_algorithm_callback_contract\":"
        << (full_algorithm_callback_contract ? "true" : "false");
    out << ",\"audio_record_track_quality_validation\":"
        << (audio_record_track_quality_validation ? "true" : "false");
    out << ",\"error_code\":" << static_cast<int32_t>(error);
    out << ",\"error\":\"" << json_escape(to_string(error)) << '"';
    out << ",\"error_detail\":\"" << json_escape(error_detail) << '"';
    out << ",\"timebase_epoch_source\":\"first_callback_started_CLOCK_MONOTONIC\"";
    out << ",\"timebase_limitation\":"
           "\"callback_cadence_not_hardware_presentation_deadline\"";
    out << ",\"quality_scope\":"
           "\"audio_callback_clock_only_silent_output_no_record_track_acceptance\"";

    out << ",\"request_sample_rate\":" << request_sample_rate;
    out << ",\"request_channel_count\":" << request_channel_count;
    out << ",\"request_format\":" << request_format;
    out << ",\"request_format_name\":\"" << format_name(request_format) << '"';
    out << ",\"request_frames_per_callback\":" << request_frames_per_callback;
    out << ",\"request_performance_mode\":" << request_performance_mode;
    out << ",\"request_performance_mode_name\":\""
        << performance_mode_name(request_performance_mode) << '"';
    out << ",\"request_sharing_mode\":" << request_sharing_mode;
    out << ",\"request_sharing_mode_name\":\""
        << sharing_mode_name(request_sharing_mode) << '"';
    out << ",\"request_buffer_burst_multiplier\":"
        << request_buffer_burst_multiplier;
    out << ",\"request_buffer_size_frames\":" << request_buffer_size_frames;

    out << ",\"actual_sample_rate\":" << actual_sample_rate;
    out << ",\"actual_channel_count\":" << actual_channel_count;
    out << ",\"actual_format\":" << actual_format;
    out << ",\"actual_format_name\":\"" << format_name(actual_format) << '"';
    out << ",\"actual_performance_mode\":" << actual_performance_mode;
    out << ",\"actual_performance_mode_name\":\""
        << performance_mode_name(actual_performance_mode) << '"';
    out << ",\"actual_sharing_mode\":" << actual_sharing_mode;
    out << ",\"actual_sharing_mode_name\":\""
        << sharing_mode_name(actual_sharing_mode) << '"';
    out << ",\"actual_frames_per_burst\":" << actual_frames_per_burst;
    out << ",\"actual_frames_per_callback\":" << actual_frames_per_callback;
    out << ",\"actual_buffer_size_frames\":" << actual_buffer_size_frames;
    out << ",\"actual_buffer_capacity_frames\":"
        << actual_buffer_capacity_frames;
    out << ",\"buffer_set_result\":" << buffer_set_result;
    out << ",\"xrun_count_before\":" << xrun_count_before;
    out << ",\"xrun_count_after\":" << xrun_count_after;
    out << ",\"xrun_delta\":" << xrun_delta;

    out << ",\"builder_create_result\":" << builder_create_result;
    out << ",\"builder_delete_result\":" << builder_delete_result;
    out << ",\"open_result\":" << open_result;
    out << ",\"start_result\":" << start_result;
    out << ",\"stop_request_result\":" << stop_request_result;
    out << ",\"wait_for_state_result\":" << wait_for_state_result;
    out << ",\"close_result\":" << close_result;
    out << ",\"stream_error_code\":" << stream_error_code;
    out << ",\"open_state\":" << open_state;
    out << ",\"open_state_name\":\"" << stream_state_name(open_state) << '"';
    out << ",\"final_state\":" << final_state;
    out << ",\"final_state_name\":\"" << stream_state_name(final_state) << '"';

    out << ",\"expected_hop_ns\":" << expected_hop_ns;
    out << ",\"callback_count\":" << callback_count;
    out << ",\"late_callbacks\":" << late_callbacks;
    out << ",\"geometry_errors\":" << geometry_errors;
    out << ",\"max_lateness_ns\":" << max_lateness_ns;
    out << ",\"first_callback_started_ns\":" << first_callback_started_ns;
    out << ",\"first_release_expected_ns\":" << first_release_expected_ns;
    out << ",\"last_callback_started_ns\":" << last_callback_started_ns;
    out << ",\"last_release_expected_ns\":" << last_release_expected_ns;
    out << ",\"main_poll_count\":" << main_poll_count;
    out << ",\"callback_thread_hook_calls\":" << callback_thread_hook_calls;
    out << ",\"callback_started_hook_calls\":"
        << callback_started_hook_calls;
    out << ",\"callback_ended_hook_calls\":" << callback_ended_hook_calls;
    out << ",\"callback_tid\":" << callback_tid;
    out << ",\"observed_seconds\":";
    if (std::isfinite(observed_seconds)) {
        out << std::setprecision(17) << observed_seconds;
    } else {
        out << "null";
    }
    out << '}';
    return out.str();
}

inline std::string to_json(const Report& report) {
    return report.to_json();
}

}  // namespace itu_audio_clock

#endif  // ITU_AUDIO_CLOCK_TYPES_V4_HPP
