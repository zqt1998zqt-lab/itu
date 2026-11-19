#include "align_readout.hpp"

#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kF = 64;
constexpr int kD = 16;
constexpr int kC = 20;
constexpr int kOutputs = kF * kC;

void need(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct rng {
    std::uint32_t state;

    std::uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    float uniform(float low, float high) {
        const float unit =
            static_cast<float>(next() >> 8) /
            static_cast<float>(1u << 24);
        return low + (high - low) * unit;
    }
};

std::vector<float> make_reference_window() {
    std::vector<float> values(
        static_cast<size_t>(kF) * kD * kC);
    rng random{0x9e3779b9u};
    for (size_t index = 0; index < values.size(); ++index) {
        const float noise = random.uniform(-3.0f, 3.0f);
        const float shaped =
            0.75f * std::sin(0.011f * static_cast<float>(index)) +
            0.25f * std::cos(0.037f * static_cast<float>(index));
        values[index] = noise + shaped;
    }

    // At uniform attention, float32 accumulation loses the 0.25 terms while
    // the required rounded-product/double accumulation produces exact zero.
    const size_t c0 = 0;
    const size_t f0 = 0;
    for (int d = 0; d < kD; ++d) {
        const size_t index =
            c0 * kD * kF + static_cast<size_t>(d) * kF + f0;
        values[index] = 0.0f;
    }
    values[0 * kF + f0] = 1.0e8f;
    values[1 * kF + f0] = 1.0f;
    values[2 * kF + f0] = -1.0e8f;
    values[3 * kF + f0] = -1.0f;
    return values;
}

std::array<std::array<float, kD>, 4> make_weight_sets() {
    std::array<std::array<float, kD>, 4> sets{};

    sets[0].fill(0.0625f);

    sets[1].fill(0.0f);
    sets[1][13] = 1.0f;

    double sum = 0.0;
    for (int d = 0; d < kD; ++d) {
        sets[2][d] = static_cast<float>(d + 1);
        sum += static_cast<double>(sets[2][d]);
    }
    for (float & weight : sets[2]) {
        weight = static_cast<float>(
            static_cast<double>(weight) / sum);
    }

    sum = 0.0;
    for (int d = 0; d < kD; ++d) {
        const double value =
            std::exp(-0.37 * std::abs(static_cast<double>(d) - 7.0));
        sets[3][d] = static_cast<float>(value);
        sum += static_cast<double>(sets[3][d]);
    }
    for (float & weight : sets[3]) {
        weight = static_cast<float>(
            static_cast<double>(weight) / sum);
    }
    return sets;
}

void validate_weights(const std::array<float, kD> & weights) {
    double sum = 0.0;
    for (float weight : weights) {
        need(std::isfinite(weight) && weight >= 0.0f,
             "invalid attention weight");
        sum += static_cast<double>(weight);
    }
    need(std::fabs(sum - 1.0) < 2.0e-6,
         "attention weights are not normalized");
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
float wrong_float_accumulation(const float * ref_data,
                               const float * attn,
                               int channel,
                               int frequency) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float sum = 0.0f;
    for (int d = 0; d < kD; ++d) {
        const size_t index =
            static_cast<size_t>(channel) * kD * kF +
            static_cast<size_t>(d) * kF + frequency;
        const float product = ref_data[index] * attn[d];
        sum = sum + product;
    }
    return sum;
}

struct graph_fixture {
    explicit graph_fixture(ggml_threadpool * pool)
        : ctx(ggml_init({12 * 1024 * 1024, nullptr, false})) {
        need(ctx != nullptr, "ggml_init failed");
        ref_full = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, kF, kD, kC, 1);
        attn = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, kD, 1, 1, 1);
        need(ref_full != nullptr && attn != nullptr,
             "input tensor allocation failed");

        ggml_tensor * attn_3d =
            ggml_reshape_3d(ctx, attn, 1, kD, 1);
        ggml_tensor * weighted =
            ggml_mul(ctx, ref_full, attn_3d);
        ggml_tensor * permuted =
            ggml_permute(ctx, weighted, 1, 0, 2, 3);
        ggml_tensor * copied = ggml_cont(ctx, permuted);
        ggml_tensor * summed = ggml_sum_rows(ctx, copied);
        reference = ggml_reshape_3d(ctx, summed, kF, 1, kC);
        fused = itu_fusion_r2::align_readout_or_null(
            ctx, ref_full, attn);
        need(reference != nullptr && fused != nullptr,
             "reference or fusion graph construction failed");
        ggml_set_name(reference, "reference_mul_cont_sum_rows");

        need(fused->op == GGML_OP_CUSTOM,
             "fusion is not GGML_OP_CUSTOM");
        need(fused->type == GGML_TYPE_F32 &&
                 ggml_is_contiguous(fused) &&
                 fused->ne[0] == kF && fused->ne[1] == 1 &&
                 fused->ne[2] == kC && fused->ne[3] == 1,
             "fusion output contract mismatch");
        need(std::strcmp(fused->name, "itu_fusion_align_readout") == 0,
             "fusion name mismatch");
        need(fused->src[0] == ref_full && fused->src[1] == attn,
             "fusion source order mismatch");

        struct custom_op_params_mirror {
            ggml_custom_op_t fun;
            int n_tasks;
            void * userdata;
        };
        custom_op_params_mirror custom{};
        std::memcpy(&custom, fused->op_params, sizeof(custom));
        need(custom.fun == itu_fusion_r2::detail::align_readout_callback,
             "fusion callback mismatch");
        need(custom.n_tasks == 1 && custom.userdata == nullptr,
             "fusion task contract mismatch");

        graph = ggml_new_graph(ctx);
        need(graph != nullptr, "graph allocation failed");
        ggml_build_forward_expand(graph, reference);
        ggml_build_forward_expand(graph, fused);

        mul_nodes = 0;
        cont_nodes = 0;
        sum_rows_nodes = 0;
        custom_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL) {
                ++mul_nodes;
            } else if (node->op == GGML_OP_CONT) {
                ++cont_nodes;
            } else if (node->op == GGML_OP_SUM_ROWS) {
                ++sum_rows_nodes;
            } else if (node->op == GGML_OP_CUSTOM) {
                ++custom_nodes;
            }
        }
        need(mul_nodes == 1 && cont_nodes == 1 &&
                 sum_rows_nodes == 1 && custom_nodes == 1,
             "unexpected reference/fusion graph nodes");

        plan = ggml_graph_plan(graph, 1, pool);
        work.resize(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
    }

    ~graph_fixture() {
        ggml_free(ctx);
    }

    graph_fixture(const graph_fixture &) = delete;
    graph_fixture & operator=(const graph_fixture &) = delete;

    void set_inputs(const std::vector<float> & reference_values,
                    const std::array<float, kD> & weights) {
        need(reference_values.size() ==
                 static_cast<size_t>(ggml_nelements(ref_full)),
             "reference input extent mismatch");
        need(weights.size() ==
                 static_cast<size_t>(ggml_nelements(attn)),
             "attention input extent mismatch");
        std::memcpy(ref_full->data, reference_values.data(),
                    reference_values.size() * sizeof(float));
        std::memcpy(attn->data, weights.data(),
                    weights.size() * sizeof(float));
    }

    void run() {
        need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
             "ggml graph compute failed");
    }

    ggml_context * ctx = nullptr;
    ggml_tensor * ref_full = nullptr;
    ggml_tensor * attn = nullptr;
    ggml_tensor * reference = nullptr;
    ggml_tensor * fused = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    std::vector<std::uint8_t> work;
    int mul_nodes = 0;
    int cont_nodes = 0;
    int sum_rows_nodes = 0;
    int custom_nodes = 0;
};

void rejection_gates() {
    ggml_context * ctx =
        ggml_init({8 * 1024 * 1024, nullptr, false});
    need(ctx != nullptr, "rejection graph arena");

    ggml_tensor * ref_full =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD, kC, 1);
    ggml_tensor * attn =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kD, 1, 1, 1);
    need(itu_fusion_r2::align_readout_or_null(
             ctx, ref_full, attn) != nullptr,
         "positive rejection gate rejected");
    need(itu_fusion_r2::align_readout_or_null(
             nullptr, ref_full, attn) == nullptr,
         "null context accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, nullptr, attn) == nullptr,
         "null reference accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, ref_full, nullptr) == nullptr,
         "null attention accepted");

    ggml_tensor * bad_f =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF - 1, kD, kC, 1);
    ggml_tensor * bad_d =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD - 1, kC, 1);
    ggml_tensor * bad_c =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD, kC - 1, 1);
    ggml_tensor * bad_batch =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD, kC, 2);
    ggml_tensor * bad_weight_shape =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kD, 2, 1, 1);
    need(itu_fusion_r2::align_readout_or_null(
             ctx, bad_f, attn) == nullptr,
         "wrong frequency extent accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, bad_d, attn) == nullptr,
         "wrong delay extent accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, bad_c, attn) == nullptr,
         "wrong channel extent accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, bad_batch, attn) == nullptr,
         "wrong batch extent accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, ref_full, bad_weight_shape) == nullptr,
         "wrong attention shape accepted");

    ggml_tensor * f16_ref =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kF, kD, kC, 1);
    ggml_tensor * f16_attn =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kD, 1, 1, 1);
    need(itu_fusion_r2::align_readout_or_null(
             ctx, f16_ref, attn) == nullptr,
         "F16 reference accepted");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, ref_full, f16_attn) == nullptr,
         "F16 attention accepted");

    ggml_tensor * stride_base = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, kF * 2 * kD * kC);
    ggml_tensor * strided = ggml_view_4d(
        ctx, stride_base, kF, kD, kC, 1,
        sizeof(float) * kF * 2,
        sizeof(float) * kF * 2 * kD,
        sizeof(float) * kF * 2 * kD * kC,
        0);
    need(!ggml_is_contiguous(strided), "stride fixture is contiguous");
    need(itu_fusion_r2::align_readout_or_null(
             ctx, strided, attn) == nullptr,
         "strided reference accepted");

    ggml_tensor malformed{};
    malformed.type = GGML_TYPE_F32;
    malformed.ne[0] = kF;
    malformed.ne[1] = kD;
    malformed.ne[2] = kC;
    malformed.ne[3] = 1;
    malformed.nb[0] = sizeof(float);
    malformed.nb[1] = sizeof(float) * kF;
    malformed.nb[2] = sizeof(float) * kF * kD * 2;
    malformed.nb[3] = sizeof(float) * kF * kD * kC;
    need(!itu_fusion_r2::detail::is_align_readout_compatible(
             &malformed, attn),
         "synthetic malformed tensor unexpectedly compatible");

    ggml_free(ctx);
}

int wrong_axis_gate(const std::vector<float> & reference_values,
                    const std::array<float, kD> & weights,
                    ggml_threadpool * pool) {
    ggml_context * ctx =
        ggml_init({8 * 1024 * 1024, nullptr, false});
    need(ctx != nullptr, "wrong-axis graph arena");
    ggml_tensor * ref_full =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD, kC, 1);
    ggml_tensor * attn =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kD, 1, 1, 1);
    std::memcpy(ref_full->data, reference_values.data(),
                reference_values.size() * sizeof(float));
    std::memcpy(attn->data, weights.data(),
                weights.size() * sizeof(float));

    ggml_tensor * attn_3d =
        ggml_reshape_3d(ctx, attn, 1, kD, 1);
    ggml_tensor * weighted =
        ggml_mul(ctx, ref_full, attn_3d);
    ggml_tensor * wrong =
        ggml_sum_rows(ctx, weighted);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, wrong);
    ggml_cplan plan = ggml_graph_plan(graph, 1, pool);
    std::vector<std::uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
         "wrong-axis graph compute failed");

    need(wrong->ne[0] == 1 && wrong->ne[1] == kD &&
             wrong->ne[2] == kC && wrong->ne[3] == 1,
         "wrong-axis shape changed unexpectedly");
    need(ggml_nelements(wrong) == kD * kC,
         "wrong-axis extent unexpectedly matches readout");
    ggml_free(ctx);
    return static_cast<int>(kD * kC);
}

struct perf_fixture {
    explicit perf_fixture(bool use_fusion, ggml_threadpool * pool)
        : ctx(ggml_init({8 * 1024 * 1024, nullptr, false})) {
        need(ctx != nullptr, "performance graph arena");
        ref_full =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kF, kD, kC, 1);
        attn =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kD, 1, 1, 1);
        if (use_fusion) {
            output = itu_fusion_r2::align_readout_or_null(
                ctx, ref_full, attn);
        } else {
            ggml_tensor * attn_3d =
                ggml_reshape_3d(ctx, attn, 1, kD, 1);
            ggml_tensor * weighted =
                ggml_mul(ctx, ref_full, attn_3d);
            ggml_tensor * permuted =
                ggml_permute(ctx, weighted, 1, 0, 2, 3);
            ggml_tensor * copied = ggml_cont(ctx, permuted);
            ggml_tensor * summed = ggml_sum_rows(ctx, copied);
            output = ggml_reshape_3d(ctx, summed, kF, 1, kC);
        }
        need(output != nullptr, "performance graph construction failed");
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        plan = ggml_graph_plan(graph, 1, pool);
        work.resize(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
    }

    ~perf_fixture() {
        ggml_free(ctx);
    }

    void set(const std::vector<float> & reference_values,
             const std::array<float, kD> & weights) {
        std::memcpy(ref_full->data, reference_values.data(),
                    reference_values.size() * sizeof(float));
        std::memcpy(attn->data, weights.data(),
                    weights.size() * sizeof(float));
    }

    double sample_us(int iterations) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
                 "performance graph compute failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(stop - start).count() /
               static_cast<double>(iterations);
    }

    ggml_context * ctx = nullptr;
    ggml_tensor * ref_full = nullptr;
    ggml_tensor * attn = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    std::vector<std::uint8_t> work;
};

struct stats {
    double min_us;
    double median_us;
    double max_us;
};

stats summarize(std::vector<double> values) {
    need(!values.empty(), "empty timing sample");
    std::sort(values.begin(), values.end());
    return {values.front(), values[values.size() / 2], values.back()};
}

void performance_probe(const std::vector<float> & reference_values,
                       const std::array<float, kD> & weights,
                       ggml_threadpool * pool) {
    perf_fixture reference(false, pool);
    perf_fixture fused(true, pool);
    reference.set(reference_values, weights);
    fused.set(reference_values, weights);

    for (int i = 0; i < 500; ++i) {
        need(ggml_graph_compute(reference.graph, &reference.plan) ==
                 GGML_STATUS_SUCCESS,
             "reference warmup failed");
        need(ggml_graph_compute(fused.graph, &fused.plan) ==
                 GGML_STATUS_SUCCESS,
             "fusion warmup failed");
    }

    constexpr int kRounds = 9;
    constexpr int kIterations = 2000;
    std::vector<double> reference_samples;
    std::vector<double> fused_samples;
    reference_samples.reserve(kRounds);
    fused_samples.reserve(kRounds);
    for (int round = 0; round < kRounds; ++round) {
        if ((round & 1) == 0) {
            reference_samples.push_back(
                reference.sample_us(kIterations));
            fused_samples.push_back(fused.sample_us(kIterations));
        } else {
            fused_samples.push_back(fused.sample_us(kIterations));
            reference_samples.push_back(
                reference.sample_us(kIterations));
        }
    }

    const stats reference_stats = summarize(reference_samples);
    const stats fused_stats = summarize(fused_samples);
    std::printf(
        "PERF,local_nonconcurrent,threads=1,hops_per_sample=%d,samples=%d,"
        "reference_us_per_hop=%.6f/%.6f/%.6f,"
        "fused_us_per_hop=%.6f/%.6f/%.6f,ratio_median=%.6f\n",
        kIterations, kRounds,
        reference_stats.min_us, reference_stats.median_us,
        reference_stats.max_us,
        fused_stats.min_us, fused_stats.median_us, fused_stats.max_us,
        fused_stats.median_us / reference_stats.median_us);
}

} // namespace

int main() {
    try {
#if !defined(__aarch64__) || !defined(__ARM_NEON)
        std::fprintf(stderr, "SKIP: AArch64 NEON is required\n");
        return 77;
#else
        rejection_gates();

        ggml_threadpool_params pool_params =
            ggml_threadpool_params_default(1);
        ggml_threadpool * pool = ggml_threadpool_new(&pool_params);
        need(pool != nullptr, "single-thread pool allocation failed");

        const std::vector<float> reference_values =
            make_reference_window();
        const auto weight_sets = make_weight_sets();
        graph_fixture fixture(pool);
        size_t compared_values = 0;

        for (size_t case_index = 0;
             case_index < weight_sets.size(); ++case_index) {
            const auto & weights = weight_sets[case_index];
            validate_weights(weights);
            fixture.set_inputs(reference_values, weights);

            std::vector<float> ref_before = reference_values;
            std::array<float, kD> attn_before = weights;
            fixture.run();
            need(std::memcmp(fixture.reference->data,
                             fixture.fused->data,
                             sizeof(float) * kOutputs) == 0,
                 "fused output differs from real GGML reference");
            need(std::memcmp(fixture.ref_full->data, ref_before.data(),
                             ref_before.size() * sizeof(float)) == 0,
                 "fusion modified reference history");
            need(std::memcmp(fixture.attn->data, attn_before.data(),
                             attn_before.size() * sizeof(float)) == 0,
                 "fusion modified attention weights");
            compared_values += kOutputs;

            double weight_sum = 0.0;
            for (float weight : weights) {
                weight_sum += static_cast<double>(weight);
            }
            std::printf(
                "CASE,%zu,sum=%.9f,values=%d,bit_equal=1,"
                "reference_unchanged=1,attention_unchanged=1\n",
                case_index, weight_sum, kOutputs);
        }

        // Wrong-weight control: perturb one effective softmax weight, prove the
        // comparator rejects it, then restore and recover bitwise parity.
        fixture.set_inputs(reference_values, weight_sets[1]);
        fixture.run();
        std::vector<float> expected(
            static_cast<const float *>(fixture.reference->data),
            static_cast<const float *>(fixture.reference->data) + kOutputs);
        auto * mutable_attn = static_cast<float *>(fixture.attn->data);
        const float saved_weight = mutable_attn[13];
        mutable_attn[13] += 0.125f;
        fixture.run();
        need(std::memcmp(expected.data(), fixture.fused->data,
                         sizeof(float) * kOutputs) != 0,
             "wrong-weight control was not rejected");
        mutable_attn[13] = saved_weight;
        fixture.run();
        need(std::memcmp(expected.data(), fixture.fused->data,
                         sizeof(float) * kOutputs) == 0,
             "weight restoration did not recover parity");

        const int wrong_axis_values = wrong_axis_gate(
            reference_values, weight_sets[0], pool);

        // Wrong rounding control: the cancellation fixture must distinguish
        // FP32 accumulation from the required double accumulation.
        const auto * ref_data =
            static_cast<const float *>(fixture.ref_full->data);
        const auto * uniform =
            static_cast<const float *>(fixture.attn->data);
        fixture.set_inputs(reference_values, weight_sets[0]);
        fixture.run();
        const float expected_cancel =
            static_cast<const float *>(fixture.reference->data)[0];
        const float wrong_float =
            wrong_float_accumulation(ref_data, uniform, 0, 0);
        need(expected_cancel == 0.0f,
             "cancellation reference is not exact zero");
        need(wrong_float != expected_cancel,
             "wrong FP32 accumulation control was not rejected");

        performance_probe(reference_values, weight_sets[0], pool);

        ggml_threadpool_free(pool);
        std::printf(
            "PASS align_readout shapes=1 weight_sets=4 "
            "bit_equal_values=%zu wrong_weight_rejected=1 "
            "wrong_axis_rejected=1 wrong_axis_values=%d "
            "wrong_fp32_accumulation_rejected=1 "
            "reference=GGML_MUL_CONT_PERMUTE_SUM_ROWS "
            "local_benchmark=nonconcurrent\n",
            compared_values, wrong_axis_values);
        return 0;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 2;
    }
}
