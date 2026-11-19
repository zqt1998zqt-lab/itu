#ifndef ITU_AUDIO_CLOCK_ANDROID_PERFORMANCE_HINT_HPP
#define ITU_AUDIO_CLOCK_ANDROID_PERFORMANCE_HINT_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#if defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace itu_audio_clock {

// APerformanceHint is optional at runtime. Keep Android API 26 linkage
// compatible by resolving every API entry point with dlsym instead of linking
// directly against a newer platform stub.
class PerformanceHint {
public:
    using ManagerHandle = void*;
    using SessionHandle = void*;

    struct FunctionTable {
        using GetManager = ManagerHandle (*)();
        using CreateSession = SessionHandle (*)(
            ManagerHandle, const std::int32_t*, std::size_t, std::int64_t);
        using ReportActualWorkDuration = int (*)(SessionHandle, std::int64_t);
        using UpdateTargetWorkDuration = int (*)(SessionHandle, std::int64_t);
        using CloseSession = void (*)(SessionHandle);

        GetManager get_manager = nullptr;
        CreateSession create_session = nullptr;
        ReportActualWorkDuration report_actual_work_duration = nullptr;
        UpdateTargetWorkDuration update_target_work_duration = nullptr;
        CloseSession close_session = nullptr;

        bool complete() const noexcept {
            return get_manager != nullptr && create_session != nullptr &&
                   report_actual_work_duration != nullptr &&
                   update_target_work_duration != nullptr &&
                   close_session != nullptr;
        }
    };

    struct Report {
        bool requested = false;
        bool api_available = false;
        bool manager_available = false;
        bool session_created = false;
        bool session_ever_created = false;
        bool ready = false;
        bool active = false;
        bool closed = false;
        std::int32_t first_actual_tid = 0;
        std::int64_t target_duration_ns = 0;
        std::uint64_t submitted_count = 0;
        std::uint64_t reported_count = 0;
        std::uint64_t dropped_count = 0;
        std::uint64_t invalid_count = 0;
        std::uint64_t queued_count = 0;
        std::uint64_t api_error_count = 0;
        int last_api_result = 0;
        std::string error;
        std::string unsupported_reason;
    };

    static constexpr std::size_t kQueueCapacity = 256;
    static constexpr std::size_t kQueueMask = kQueueCapacity - 1;
    static constexpr std::int64_t kDefaultTargetDurationNs = 16'000'000;

    static_assert((kQueueCapacity & kQueueMask) == 0,
                  "queue capacity must be a power of two");
    static_assert(std::atomic<std::int32_t>::is_always_lock_free,
                  "callback tid handshake must be lock-free");
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "callback counters must be lock-free");
    static_assert(std::atomic<int>::is_always_lock_free,
                  "callback error reporting must be lock-free");

    explicit PerformanceHint(bool requested = false)
        : requested_(requested) {
        initialize_runtime();
    }

    // Host tests and non-Android integrations inject a complete fake table.
    explicit PerformanceHint(const FunctionTable& functions,
                             bool requested = false)
        : requested_(requested) {
        initialize_external(functions);
    }

    ~PerformanceHint() {
        close();
    }

    PerformanceHint(const PerformanceHint&) = delete;
    PerformanceHint& operator=(const PerformanceHint&) = delete;

    // Call only from a non-callback control thread, using the tid that the
    // callback already published through register_callback_tid().
    // Returns 0 on acceptance, -1 for invalid input, -2 for a changed tid,
    // or -3 when no callback tid has been handshaken yet.
    int register_thread(std::int32_t tid,
                        std::int64_t target_duration_ns) noexcept {
        if (!requested_) {
            return 0;
        }
        if (tid <= 0 || target_duration_ns <= 0) {
            set_error(ErrorCode::InvalidTarget);
            return -1;
        }

        const std::int32_t observed =
            callback_tid_.load(std::memory_order_acquire);
        if (observed == 0) {
            set_error(ErrorCode::ThreadNotHandshaken);
            return -3;
        }
        if (tid_conflict_.load(std::memory_order_acquire)) {
            set_error(ErrorCode::ThreadIdChanged);
            return -2;
        }
        if (observed != tid) {
            set_error(ErrorCode::ThreadIdChanged);
            return -2;
        }

        first_actual_tid_ = observed;
        set_target_duration_ns(target_duration_ns);
        return 0;
    }

    // Safe on the realtime callback path: one atomic compare/exchange and no
    // session, Binder, allocation, file, or logging operation.
    void register_callback_tid(std::int32_t tid) noexcept {
        if (!requested_ || closed_.load(std::memory_order_acquire) || tid <= 0) {
            return;
        }

        std::int32_t expected = 0;
        if (callback_tid_.compare_exchange_strong(
                expected, tid, std::memory_order_release,
                std::memory_order_acquire)) {
            return;
        }
        if (expected != tid) {
            tid_conflict_.store(true, std::memory_order_release);
            set_error(ErrorCode::ThreadIdChanged);
        }
    }

    // Main/control-thread configuration. A changed target is applied by the
    // next drain(); this method does not issue a Binder call itself.
    void set_target_duration_ns(std::int64_t target_duration_ns) noexcept {
        if (target_duration_ns <= 0) {
            set_error(ErrorCode::InvalidTarget);
            return;
        }
        target_duration_ns_.store(target_duration_ns,
                                  std::memory_order_release);
        target_dirty_.store(true, std::memory_order_release);
    }

    // Called from the realtime callback. Valid submissions are either
    // reported later, explicitly counted as dropped, or still pending.
    void submit_callback_duration(std::int64_t actual_duration_ns) noexcept {
        if (!requested_) {
            return;
        }
        if (closed_.load(std::memory_order_acquire)) {
            submitted_count_.fetch_add(1, std::memory_order_relaxed);
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (actual_duration_ns <= 0) {
            invalid_count_.fetch_add(1, std::memory_order_relaxed);
            set_error(ErrorCode::InvalidDuration);
            return;
        }

        submitted_count_.fetch_add(1, std::memory_order_relaxed);
        if (!api_available_.load(std::memory_order_acquire) ||
            reporting_unavailable_.load(std::memory_order_acquire)) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!enqueue(actual_duration_ns)) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            set_error(ErrorCode::QueueOverflow);
        }
    }

    // Call only from the audio run/main thread, including the existing 10 ms
    // poll. This is the sole path that may call getManager/createSession/
    // updateTargetWorkDuration/reportActualWorkDuration.
    void drain() {
        if (!requested_) {
            return;
        }
        if (closed_.load(std::memory_order_acquire)) {
            drop_all_queued();
            return;
        }
        if (!api_available_.load(std::memory_order_acquire)) {
            reporting_unavailable_.store(true, std::memory_order_release);
            drop_all_queued();
            return;
        }

        const std::int32_t tid =
            callback_tid_.load(std::memory_order_acquire);
        if (tid <= 0) {
            return;
        }
        if (first_actual_tid_ == 0) {
            first_actual_tid_ = tid;
        } else if (first_actual_tid_ != tid) {
            tid_conflict_.store(true, std::memory_order_release);
            set_error(ErrorCode::ThreadIdChanged);
        }

        if (session_ == nullptr && !session_attempted_) {
            session_attempted_ = true;
            manager_ = functions_.get_manager();
            if (manager_ == nullptr) {
                manager_available_ = false;
                reporting_unavailable_.store(true,
                                             std::memory_order_release);
                unsupported_reason_ = "manager_unavailable";
                set_error(ErrorCode::ManagerUnavailable);
                drop_all_queued();
                return;
            }
            manager_available_ = true;

            const std::int32_t thread_id = first_actual_tid_;
            const std::int64_t target =
                target_duration_ns_.load(std::memory_order_acquire);
            session_ = functions_.create_session(manager_, &thread_id, 1,
                                                 target);
            if (session_ == nullptr) {
                reporting_unavailable_.store(true,
                                             std::memory_order_release);
                unsupported_reason_ = "session_create_failed";
                set_error(ErrorCode::SessionCreateFailed);
                drop_all_queued();
                return;
            }
            session_ever_created_ = true;
            target_dirty_.store(false, std::memory_order_release);
        }

        if (session_ == nullptr) {
            reporting_unavailable_.store(true, std::memory_order_release);
            drop_all_queued();
            return;
        }

        if (target_dirty_.exchange(false, std::memory_order_acq_rel)) {
            const std::int64_t target =
                target_duration_ns_.load(std::memory_order_acquire);
            const int result =
                functions_.update_target_work_duration(session_, target);
            last_api_result_.store(result, std::memory_order_release);
            if (result != 0) {
                api_error_count_.fetch_add(1, std::memory_order_relaxed);
                set_error(ErrorCode::UpdateTargetFailed);
            }
        }

        std::int64_t actual_duration_ns = 0;
        while (pop(actual_duration_ns)) {
            const int result = functions_.report_actual_work_duration(
                session_, actual_duration_ns);
            last_api_result_.store(result, std::memory_order_release);
            if (result == 0) {
                reported_count_.fetch_add(1, std::memory_order_relaxed);
            } else {
                api_error_count_.fetch_add(1, std::memory_order_relaxed);
                dropped_count_.fetch_add(1, std::memory_order_relaxed);
                set_error(ErrorCode::ReportFailed);
            }
        }
    }

    void close() noexcept {
        if (closed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        if (session_ != nullptr && functions_.close_session != nullptr) {
            functions_.close_session(session_);
        }
        session_ = nullptr;
        manager_ = nullptr;
        drop_all_queued();

#if defined(__ANDROID__)
        if (library_ != nullptr) {
            dlclose(library_);
            library_ = nullptr;
        }
#endif
    }

    bool requested() const noexcept {
        return requested_;
    }

    bool supported() const noexcept {
        return api_available_.load(std::memory_order_acquire);
    }

    // ready only means that a session handle exists. It does not prove that
    // Android changed CPU frequency or scheduling policy.
    bool ready() const noexcept {
        return session_ != nullptr &&
               !closed_.load(std::memory_order_acquire);
    }

    // active only means that at least one report returned 0. A successful API
    // call proves acceptance, not a frequency, deadline, or hard-real-time
    // guarantee.
    bool active() const noexcept {
        return ready() &&
               reported_count_.load(std::memory_order_acquire) > 0;
    }

    std::string error() const {
        return error_name(
            static_cast<ErrorCode>(
                error_code_.load(std::memory_order_acquire)));
    }

    int last_api_result() const noexcept {
        return last_api_result_.load(std::memory_order_acquire);
    }

    std::string unsupported_reason() const {
        return unsupported_reason_;
    }

    Report snapshot() const {
        Report report;
        report.requested = requested_;
        report.api_available =
            api_available_.load(std::memory_order_acquire);
        report.manager_available = manager_available_;
        report.session_created = session_ != nullptr;
        report.session_ever_created = session_ever_created_;
        report.ready = ready();
        report.active = active();
        report.closed = closed_.load(std::memory_order_acquire);
        report.first_actual_tid = first_actual_tid_;
        report.target_duration_ns =
            target_duration_ns_.load(std::memory_order_acquire);
        report.submitted_count =
            submitted_count_.load(std::memory_order_acquire);
        report.reported_count =
            reported_count_.load(std::memory_order_acquire);
        report.dropped_count =
            dropped_count_.load(std::memory_order_acquire);
        report.invalid_count =
            invalid_count_.load(std::memory_order_acquire);
        report.queued_count = pending_count();
        report.api_error_count =
            api_error_count_.load(std::memory_order_acquire);
        report.last_api_result =
            last_api_result_.load(std::memory_order_acquire);
        report.error = error();
        report.unsupported_reason = unsupported_reason_;
        return report;
    }

    std::string to_json() const {
        const Report report = snapshot();
        std::ostringstream out;
        out << '{'
            << "\"requested\":" << json_bool(report.requested) << ','
            << "\"api_available\":" << json_bool(report.api_available) << ','
            << "\"manager_available\":"
            << json_bool(report.manager_available) << ','
            << "\"session_created\":"
            << json_bool(report.session_created) << ','
            << "\"session_ever_created\":"
            << json_bool(report.session_ever_created) << ','
            << "\"ready\":" << json_bool(report.ready) << ','
            << "\"active\":" << json_bool(report.active) << ','
            << "\"closed\":" << json_bool(report.closed) << ','
            << "\"first_actual_tid\":" << report.first_actual_tid << ','
            << "\"target_duration_ns\":"
            << report.target_duration_ns << ','
            << "\"submitted_count\":" << report.submitted_count << ','
            << "\"reported_count\":" << report.reported_count << ','
            << "\"dropped_count\":" << report.dropped_count << ','
            << "\"invalid_count\":" << report.invalid_count << ','
            << "\"queued_count\":" << report.queued_count << ','
            << "\"api_error_count\":" << report.api_error_count << ','
            << "\"last_api_result\":" << report.last_api_result << ','
            << "\"error\":\"" << json_escape(report.error) << "\","
            << "\"unsupported_reason\":\""
            << json_escape(report.unsupported_reason) << "\","
            << "\"semantics\":\""
            << "api_accepted_only;not_frequency_or_hard_realtime_proof"
            << "\"}";
        return out.str();
    }

private:
    enum class ErrorCode : int {
        None = 0,
        MissingSymbols,
        LibraryLoadFailed,
        ManagerUnavailable,
        SessionCreateFailed,
        ReportFailed,
        UpdateTargetFailed,
        InvalidDuration,
        InvalidTarget,
        QueueOverflow,
        ThreadIdChanged,
        ThreadNotHandshaken
    };

    void initialize_runtime() {
#if defined(__ANDROID__)
        FunctionTable functions;
        library_ = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            unsupported_reason_ = "library_load_failed";
            if (requested_) {
                set_error(ErrorCode::LibraryLoadFailed);
            }
            return;
        }

        functions.get_manager = reinterpret_cast<FunctionTable::GetManager>(
            dlsym(library_, "APerformanceHint_getManager"));
        functions.create_session =
            reinterpret_cast<FunctionTable::CreateSession>(
                dlsym(library_, "APerformanceHint_createSession"));
        functions.report_actual_work_duration =
            reinterpret_cast<FunctionTable::ReportActualWorkDuration>(
                dlsym(library_,
                      "APerformanceHint_reportActualWorkDuration"));
        functions.update_target_work_duration =
            reinterpret_cast<FunctionTable::UpdateTargetWorkDuration>(
                dlsym(library_,
                      "APerformanceHint_updateTargetWorkDuration"));
        functions.close_session =
            reinterpret_cast<FunctionTable::CloseSession>(
                dlsym(library_, "APerformanceHint_closeSession"));

        std::string missing = missing_symbols(functions);
        if (!missing.empty()) {
            unsupported_reason_ = "missing_symbols: " + missing;
            if (requested_) {
                set_error(ErrorCode::MissingSymbols);
            }
            return;
        }
        functions_ = functions;
        api_available_.store(true, std::memory_order_release);
#else
        unsupported_reason_ = "non_android_without_function_table";
        if (requested_) {
            set_error(ErrorCode::MissingSymbols);
        }
#endif
    }

    void initialize_external(const FunctionTable& functions) {
        functions_ = functions;
        const std::string missing = missing_symbols(functions_);
        if (!missing.empty()) {
            unsupported_reason_ = "missing_symbols: " + missing;
            if (requested_) {
                set_error(ErrorCode::MissingSymbols);
            }
            return;
        }
        api_available_.store(true, std::memory_order_release);
    }

    static std::string missing_symbols(const FunctionTable& functions) {
        std::string missing;
        auto append = [&missing](const char* name) {
            if (!missing.empty()) {
                missing += ',';
            }
            missing += name;
        };
        if (functions.get_manager == nullptr) {
            append("APerformanceHint_getManager");
        }
        if (functions.create_session == nullptr) {
            append("APerformanceHint_createSession");
        }
        if (functions.report_actual_work_duration == nullptr) {
            append("APerformanceHint_reportActualWorkDuration");
        }
        if (functions.update_target_work_duration == nullptr) {
            append("APerformanceHint_updateTargetWorkDuration");
        }
        if (functions.close_session == nullptr) {
            append("APerformanceHint_closeSession");
        }
        return missing;
    }

    bool enqueue(std::int64_t value) noexcept {
        const std::uint64_t head =
            queue_head_.load(std::memory_order_relaxed);
        const std::uint64_t tail =
            queue_tail_.load(std::memory_order_acquire);
        if (head - tail >= kQueueCapacity) {
            return false;
        }
        queue_[static_cast<std::size_t>(head) & kQueueMask] = value;
        queue_head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool pop(std::int64_t& value) noexcept {
        const std::uint64_t tail =
            queue_tail_.load(std::memory_order_relaxed);
        const std::uint64_t head =
            queue_head_.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }
        value = queue_[static_cast<std::size_t>(tail) & kQueueMask];
        queue_tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    void drop_all_queued() noexcept {
        std::int64_t ignored = 0;
        while (pop(ignored)) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::uint64_t pending_count() const noexcept {
        const std::uint64_t head =
            queue_head_.load(std::memory_order_acquire);
        const std::uint64_t tail =
            queue_tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    void set_error(ErrorCode code) noexcept {
        error_code_.store(static_cast<int>(code),
                          std::memory_order_release);
    }

    static const char* error_name(ErrorCode code) noexcept {
        switch (code) {
            case ErrorCode::None:
                return "";
            case ErrorCode::MissingSymbols:
                return "missing_symbols";
            case ErrorCode::LibraryLoadFailed:
                return "library_load_failed";
            case ErrorCode::ManagerUnavailable:
                return "manager_unavailable";
            case ErrorCode::SessionCreateFailed:
                return "session_create_failed";
            case ErrorCode::ReportFailed:
                return "report_failed";
            case ErrorCode::UpdateTargetFailed:
                return "update_target_failed";
            case ErrorCode::InvalidDuration:
                return "invalid_duration";
            case ErrorCode::InvalidTarget:
                return "invalid_target";
            case ErrorCode::QueueOverflow:
                return "queue_overflow";
            case ErrorCode::ThreadIdChanged:
                return "callback_tid_changed";
            case ErrorCode::ThreadNotHandshaken:
                return "callback_tid_not_handshaken";
        }
        return "unknown_error";
    }

    static const char* json_bool(bool value) noexcept {
        return value ? "true" : "false";
    }

    static std::string json_escape(const std::string& value) {
        std::string escaped;
        escaped.reserve(value.size());
        for (const char character : value) {
            switch (character) {
                case '"':
                    escaped += "\\\"";
                    break;
                case '\\':
                    escaped += "\\\\";
                    break;
                case '\b':
                    escaped += "\\b";
                    break;
                case '\f':
                    escaped += "\\f";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped += character;
                    break;
            }
        }
        return escaped;
    }

    const bool requested_;
    FunctionTable functions_{};
#if defined(__ANDROID__)
    void* library_ = nullptr;
#endif
    ManagerHandle manager_ = nullptr;
    SessionHandle session_ = nullptr;
    bool manager_available_ = false;
    bool session_attempted_ = false;
    bool session_ever_created_ = false;
    std::int32_t first_actual_tid_ = 0;
    std::string unsupported_reason_;

    std::atomic<bool> api_available_{false};
    std::atomic<bool> reporting_unavailable_{false};
    std::atomic<bool> closed_{false};
    std::atomic<bool> tid_conflict_{false};
    std::atomic<std::int32_t> callback_tid_{0};
    std::atomic<std::int64_t> target_duration_ns_{
        kDefaultTargetDurationNs};
    std::atomic<bool> target_dirty_{false};
    std::atomic<int> error_code_{
        static_cast<int>(ErrorCode::None)};
    std::atomic<int> last_api_result_{0};

    std::atomic<std::uint64_t> submitted_count_{0};
    std::atomic<std::uint64_t> reported_count_{0};
    std::atomic<std::uint64_t> dropped_count_{0};
    std::atomic<std::uint64_t> invalid_count_{0};
    std::atomic<std::uint64_t> api_error_count_{0};

    std::array<std::int64_t, kQueueCapacity> queue_{};
    std::atomic<std::uint64_t> queue_head_{0};
    std::atomic<std::uint64_t> queue_tail_{0};
};

}  // namespace itu_audio_clock

#endif  // ITU_AUDIO_CLOCK_ANDROID_PERFORMANCE_HINT_HPP
