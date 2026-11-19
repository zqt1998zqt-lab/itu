/*
 * Experimental LocalVQE bridge for external-AEC residuals.
 *
 * This translation unit deliberately includes the pinned upstream API
 * implementation instead of copying its graph, DAF, history-window, or OLA
 * logic. The normal API object is removed from the prebuilt link command when
 * this object is substituted, so every upstream API symbol still has exactly
 * one definition.
 */

#include <array>

#include "localvqe_api.cpp"

namespace {

enum bridge_status : int {
    k_bridge_ok = 0,
    k_bridge_null_handle = -1001,
    k_bridge_null_pointer = -1002,
    k_bridge_invalid_hop = -1003,
    k_bridge_nonfinite_input = -1004,
    k_bridge_model_contract = -1005,
    k_bridge_weight_contract = -1006,
    k_bridge_nonfinite_output = -1007,
    k_bridge_internal_exception = -1008,
};

class scoped_daf_disable {
public:
    explicit scoped_daf_disable(daf_frontend& daf) noexcept
        : daf_(daf), loaded_(daf.loaded) {
        daf_.loaded = false;
    }

    ~scoped_daf_disable() {
        daf_.loaded = loaded_;
    }

    scoped_daf_disable(const scoped_daf_disable&) = delete;
    scoped_daf_disable& operator=(const scoped_daf_disable&) = delete;

private:
    daf_frontend& daf_;
    bool loaded_;
};

void set_error_noexcept(localvqe_ctx* ctx, const char* message) noexcept {
    if (!ctx) return;
    try {
        ctx->last_error.assign(message ? message : "unknown bridge failure");
    } catch (...) {
        // The C ABI must not leak an exception from error reporting either.
    }
}

bool finite_span(const float* values, int count) noexcept {
    if (!values || count <= 0) return false;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

// The pinned allowlist has one graph model with both the embedded DAF and the
// mask backend: localvqe-v1.4-aec-200K. The model has no explicit "200K" GGUF
// key, so the enforceable identity is the complete stream contract plus the
// presence of both loaded components. Standalone DAF and GTCRN are excluded.
const char* validate_200k_mask_context(const localvqe_ctx* ctx) noexcept {
    if (!ctx) return "null LocalVQE context";
    if (ctx->is_gtcrn) return "external bridge requires the 200K DAF+mask graph, not GTCRN";
    if (ctx->daf_standalone) return "external bridge requires a mask graph, not standalone DAF";
    if (!ctx->daf.loaded) return "external bridge requires the embedded 200K DAF front-end";
    if (!ctx->stream_graph.graph) return "external bridge requires a loaded mask graph";

    const auto& hp = ctx->graph_model.hparams;
    if (hp.sample_rate != 16000 || hp.hop_length != 256 || hp.n_fft != 512) {
        return "external bridge requires 16 kHz, hop 256, FFT 512";
    }
    if (ctx->pcm_hist_mic.size() < 256 || ctx->pcm_hist_ref.size() < 256 ||
        ctx->mic_window.size() < 512 || ctx->ref_window.size() < 512 ||
        ctx->enh_window.size() < 512 || ctx->ola.size() < 512) {
        return "external bridge streaming buffers do not match the 200K graph";
    }
    return nullptr;
}

bool is_f32_512_matrix(const ggml_tensor* tensor) noexcept {
    if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_get_data(tensor) ||
        !ggml_is_contiguous(tensor) || tensor->ne[0] != 512) {
        return false;
    }
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3] == 512;
}

// The GGUF tensors retain PyTorch's contiguous [out, in] flattening, while
// ggml_mul_mat consumes each output row as a contiguous 512-float dot. Traverse
// input columns in column-major order to keep the 512x512 product explicit.
void column_major_matvec_512(const float* matrix, const float* input,
                             float* output) noexcept {
    for (int row = 0; row < 512; ++row) output[row] = 0.0f;
    for (int column = 0; column < 512; ++column) {
        const float value = input[column];
        for (int row = 0; row < 512; ++row) {
            output[row] += matrix[row * 512 + column] * value;
        }
    }
}

}  // namespace

extern "C" {

LOCALVQE_API int localvqe_external_bridge_version(void) {
    return 1;
}

LOCALVQE_API int localvqe_external_mask_frame_f32(
    localvqe_ctx_t handle, const float* e, const float* yhat, int hop,
    float* out) {
    localvqe_ctx* ctx = nullptr;
    try {
        if (!handle) return k_bridge_null_handle;
        ctx = reinterpret_cast<localvqe_ctx*>(handle);
        ctx->last_error.clear();

        const char* contract_error = validate_200k_mask_context(ctx);
        if (contract_error) {
            set_error_noexcept(ctx, contract_error);
            return k_bridge_model_contract;
        }
        if (hop != 256) {
            set_error_noexcept(ctx, "external mask hop must be exactly 256 samples");
            return k_bridge_invalid_hop;
        }
        if (!e || !yhat || !out) {
            set_error_noexcept(ctx, "external mask requires non-null e, yhat, and out");
            return k_bridge_null_pointer;
        }
        if (!finite_span(e, hop) || !finite_span(yhat, hop)) {
            set_error_noexcept(ctx, "external mask inputs must be finite");
            return k_bridge_nonfinite_input;
        }

        std::array<float, 256> frame_out{};
        {
            scoped_daf_disable without_daf(ctx->daf);
            stream_one_frame(ctx, e, yhat, frame_out.data());
        }
        if (!finite_span(frame_out.data(), hop)) {
            set_error_noexcept(ctx, "external mask graph produced non-finite output");
            return k_bridge_nonfinite_output;
        }
        std::copy(frame_out.begin(), frame_out.end(), out);
        return k_bridge_ok;
    } catch (const std::exception& error) {
        set_error_noexcept(ctx, error.what());
        return k_bridge_internal_exception;
    } catch (...) {
        set_error_noexcept(ctx, "external mask bridge caught an unknown exception");
        return k_bridge_internal_exception;
    }
}

LOCALVQE_API int localvqe_capture_daf_frame_f32(
    localvqe_ctx_t handle, const float* mic, const float* ref, int hop,
    float* e, float* yhat) {
    localvqe_ctx* ctx = nullptr;
    try {
        if (!handle) return k_bridge_null_handle;
        ctx = reinterpret_cast<localvqe_ctx*>(handle);
        ctx->last_error.clear();

        const char* contract_error = validate_200k_mask_context(ctx);
        if (contract_error) {
            set_error_noexcept(ctx, contract_error);
            return k_bridge_model_contract;
        }
        if (hop != 256) {
            set_error_noexcept(ctx, "DAF capture hop must be exactly 256 samples");
            return k_bridge_invalid_hop;
        }
        if (!mic || !ref || !e || !yhat) {
            set_error_noexcept(ctx, "DAF capture requires non-null mic, ref, e, and yhat");
            return k_bridge_null_pointer;
        }
        if (!finite_span(mic, hop) || !finite_span(ref, hop)) {
            set_error_noexcept(ctx, "DAF capture inputs must be finite");
            return k_bridge_nonfinite_input;
        }

        std::array<float, 256> captured_e{};
        std::array<float, 256> captured_yhat{};
        daf_process(ctx->daf, mic, ref, hop, captured_e.data(),
                    captured_yhat.data());
        if (!finite_span(captured_e.data(), hop) ||
            !finite_span(captured_yhat.data(), hop)) {
            set_error_noexcept(ctx, "DAF capture produced non-finite output");
            return k_bridge_nonfinite_output;
        }
        std::copy(captured_e.begin(), captured_e.end(), e);
        std::copy(captured_yhat.begin(), captured_yhat.end(), yhat);
        return k_bridge_ok;
    } catch (const std::exception& error) {
        set_error_noexcept(ctx, error.what());
        return k_bridge_internal_exception;
    } catch (...) {
        set_error_noexcept(ctx, "DAF capture bridge caught an unknown exception");
        return k_bridge_internal_exception;
    }
}

LOCALVQE_API int localvqe_external_codec_frame_f32(
    localvqe_ctx_t handle, const float* pcm, int hop, float* out) {
    localvqe_ctx* ctx = nullptr;
    try {
        if (!handle) return k_bridge_null_handle;
        ctx = reinterpret_cast<localvqe_ctx*>(handle);
        ctx->last_error.clear();

        const char* contract_error = validate_200k_mask_context(ctx);
        if (contract_error) {
            set_error_noexcept(ctx, contract_error);
            return k_bridge_model_contract;
        }
        if (hop != 256) {
            set_error_noexcept(ctx, "external codec hop must be exactly 256 samples");
            return k_bridge_invalid_hop;
        }
        if (!pcm || !out) {
            set_error_noexcept(ctx, "external codec requires non-null pcm and out");
            return k_bridge_null_pointer;
        }
        if (!finite_span(pcm, hop)) {
            set_error_noexcept(ctx, "external codec input must be finite");
            return k_bridge_nonfinite_input;
        }

        const ggml_tensor* encoder =
            ctx->graph_model.w("encoder.conv.weight");
        const ggml_tensor* decoder =
            ctx->graph_model.w("decoder.linear.weight");
        if (!is_f32_512_matrix(encoder) || !is_f32_512_matrix(decoder)) {
            set_error_noexcept(
                ctx,
                "encoder.conv.weight and decoder.linear.weight must be contiguous F32 512x512 matrices");
            return k_bridge_weight_contract;
        }

        std::array<float, 512> window{};
        std::array<float, 512> encoded{};
        std::array<float, 512> decoded{};
        std::array<float, 256> frame_out{};

        build_window(ctx->pcm_hist_mic, pcm, hop, window.data());
        column_major_matvec_512(
            static_cast<const float*>(ggml_get_data(encoder)), window.data(),
            encoded.data());
        column_major_matvec_512(
            static_cast<const float*>(ggml_get_data(decoder)), encoded.data(),
            decoded.data());

        for (int i = 0; i < 512; ++i) ctx->ola[i] += decoded[i];
        const float scale =
            (ctx->graph_model.hparams.version >= 2) ? 1.0f : 0.5f;
        for (int i = 0; i < hop; ++i) frame_out[i] = ctx->ola[i] * scale;
        if (ctx->noise_gate_enabled) {
            localvqe::apply_noise_gate(frame_out.data(), hop,
                                       ctx->noise_gate_threshold_dbfs);
        }
        if (!finite_span(frame_out.data(), hop)) {
            set_error_noexcept(ctx, "external codec produced non-finite output");
            return k_bridge_nonfinite_output;
        }

        std::copy(frame_out.begin(), frame_out.end(), out);
        std::memmove(ctx->ola.data(), ctx->ola.data() + hop,
                     (512 - hop) * sizeof(float));
        std::memset(ctx->ola.data() + (512 - hop), 0,
                    hop * sizeof(float));
        return k_bridge_ok;
    } catch (const std::exception& error) {
        set_error_noexcept(ctx, error.what());
        return k_bridge_internal_exception;
    } catch (...) {
        set_error_noexcept(ctx, "external codec bridge caught an unknown exception");
        return k_bridge_internal_exception;
    }
}

}  // extern "C"
