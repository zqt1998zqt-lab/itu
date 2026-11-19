#include "bias_silu.hpp"

#include "ggml-cpu.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
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
        const float unit = static_cast<float>(next() >> 8) /
                           static_cast<float>(1u << 24);
        return low + (high - low) * unit;
    }
};

void fill_inputs(std::vector<float>& input, std::uint32_t seed) {
    rng random{seed};
    for (float& value : input) {
        value = random.uniform(-12.0f, 12.0f);
    }
    input[0] = 0.25f;

    const std::array<float, 12> edge_values{{
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        100.0f,
        -100.0f,
        88.0f,
        -88.0f,
        0.0f,
        -0.0f,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        1.0e-30f,
        -1.0e-30f,
    }};
    for (std::size_t i = 0;
         i < edge_values.size() && i + 1 < input.size(); ++i) {
        input[i + 1] = edge_values[i];
    }
}

void fill_bias(std::vector<float>& bias, std::uint32_t seed) {
    rng random{seed};
    for (float& value : bias) {
        value = random.uniform(-2.0f, 2.0f);
    }
}

void run_case(int f, int channels, std::uint32_t seed, ggml_threadpool * pool) {
    ggml_init_params params{8 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml_init failed");

    ggml_tensor * input =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, f, 1, channels, 1);
    ggml_tensor * bias =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, channels, 1);
    require(input != nullptr && bias != nullptr, "tensor allocation failed");

    std::vector<float> input_values(static_cast<std::size_t>(f) * channels);
    std::vector<float> bias_values(static_cast<std::size_t>(channels));
    fill_inputs(input_values, seed);
    fill_bias(bias_values, seed ^ 0x9e3779b9u);
    std::memcpy(input->data, input_values.data(),
                input_values.size() * sizeof(float));
    std::memcpy(bias->data, bias_values.data(),
                bias_values.size() * sizeof(float));

    ggml_tensor * reference_add = ggml_add(ctx, input, bias);
    ggml_tensor * reference = ggml_silu(ctx, reference_add);
    ggml_tensor * fused =
        itu_fusion_r1::bias_silu_or_null(ctx, input, bias);
    require(reference_add != nullptr && reference != nullptr,
            "reference graph construction failed");
    require(fused != nullptr, "compatible fusion rejected");
    require(std::strcmp(fused->name, "itu_fusion_bias_silu") == 0,
            "fusion node name changed");
    require(fused->src[0] == input && fused->src[1] == bias,
            "fusion graph sources changed");
    ggml_map_custom2_op_params custom_params{};
    std::memcpy(&custom_params, fused->op_params, sizeof(custom_params));
    require(custom_params.fun == itu_fusion_r1::detail::bias_silu_callback,
            "fusion callback changed");
    require(custom_params.n_tasks == 1 && custom_params.userdata == nullptr,
            "fusion task contract changed");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    require(graph != nullptr, "graph allocation failed");
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, fused);

    int add_nodes = 0;
    int silu_nodes = 0;
    int fused_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_ADD) {
            ++add_nodes;
        } else if (node->op == GGML_OP_UNARY &&
                   ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
            ++silu_nodes;
        } else if (node->op == GGML_OP_MAP_CUSTOM2) {
            ++fused_nodes;
        }
    }
    require(add_nodes == 1 && silu_nodes == 1 && fused_nodes == 1,
            "unexpected graph operator count");

    ggml_cplan plan = ggml_graph_plan(graph, 1, pool);
    std::vector<std::uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
            "graph compute failed");

    const std::size_t bytes = input_values.size() * sizeof(float);
    require(std::memcmp(reference->data, fused->data, bytes) == 0,
            "fused output differs from ADD plus SILU");
    std::vector<float> reference_saved(
        static_cast<const float *>(reference->data),
        static_cast<const float *>(reference->data) + input_values.size());

    static_cast<float *>(bias->data)[0] += 0.125f;
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
            "wrong-bias graph compute failed");
    require(std::memcmp(reference_saved.data(), fused->data, bytes) != 0,
            "wrong-bias negative control was not rejected");

    std::memcpy(bias->data, bias_values.data(),
                bias_values.size() * sizeof(float));
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
            "restored graph compute failed");
    require(std::memcmp(reference_saved.data(), reference->data, bytes) == 0,
            "reference restoration failed");
    require(std::memcmp(reference_saved.data(), fused->data, bytes) == 0,
            "fusion restoration failed");

    std::printf("PASS f=%d c=%d elements=%zu add=1 silu=1 custom2=1 "
                "bit_equal=1 wrong_bias_rejected=1\n",
                f, channels, input_values.size());
    ggml_free(ctx);
}

void require_rejected(ggml_context * ctx, ggml_tensor * input,
                      ggml_tensor * bias,
                      const char * message) {
    require(itu_fusion_r1::bias_silu_or_null(ctx, input, bias) == nullptr,
            message);
}

void run_unsupported_cases() {
    ggml_init_params params{4 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "unsupported-case ggml_init failed");

    ggml_tensor * valid_input =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 1, 4, 1);
    ggml_tensor * valid_bias =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 4, 1);
    require_rejected(ctx, valid_input, nullptr, "null bias was accepted");
    require(itu_fusion_r1::bias_silu_or_null(nullptr, valid_input,
                                              valid_bias) == nullptr,
            "null context was accepted");

    ggml_tensor * wrong_rank =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 1, 4, 2);
    require_rejected(ctx, wrong_rank, valid_bias,
                     "wrong input rank was accepted");

    ggml_tensor * wrong_bias_rank =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    require_rejected(ctx, valid_input, wrong_bias_rank,
                     "wrong bias rank was accepted");

    ggml_tensor * wrong_bias_channels =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1);
    require_rejected(ctx, valid_input, wrong_bias_channels,
                     "mismatched bias channels were accepted");

    ggml_tensor * f16_input =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 1, 4, 1);
    require_rejected(ctx, f16_input, valid_bias, "F16 input was accepted");

    ggml_tensor * f16_bias =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 4, 1);
    require_rejected(ctx, valid_input, f16_bias, "F16 bias was accepted");

    ggml_tensor * padded_backing =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, 4, 1);
    ggml_tensor * noncontiguous = ggml_view_4d(
        ctx, padded_backing, 8, 1, 4, 1,
        padded_backing->nb[1], padded_backing->nb[2],
        padded_backing->nb[3], 0);
    require(!ggml_is_contiguous(noncontiguous),
            "non-contiguous test view is contiguous");
    require_rejected(ctx, noncontiguous, valid_bias,
                     "non-contiguous input was accepted");

    ggml_tensor * padded_bias_backing =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 2, 4, 1);
    ggml_tensor * noncontiguous_bias = ggml_view_4d(
        ctx, padded_bias_backing, 1, 1, 4, 1,
        padded_bias_backing->nb[1], padded_bias_backing->nb[2],
        padded_bias_backing->nb[3], 0);
    require(!ggml_is_contiguous(noncontiguous_bias),
            "non-contiguous bias test view is contiguous");
    require_rejected(ctx, valid_input, noncontiguous_bias,
                     "non-contiguous bias was accepted");

    std::puts("PASS unsupported_cases=9");
    ggml_free(ctx);
}

} // namespace

int main() {
    try {
        ggml_threadpool_params defaults = ggml_threadpool_params_default(1);
        ggml_threadpool * pool = ggml_threadpool_new(&defaults);
        require(pool != nullptr, "threadpool allocation failed");

        const std::array<std::array<int, 2>, 10> shapes{{
            {{1, 1}},
            {{3, 1}},
            {{4, 1}},
            {{5, 1}},
            {{7, 3}},
            {{8, 20}},
            {{9, 2}},
            {{17, 5}},
            {{32, 20}},
            {{128, 16}},
        }};
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            run_case(shapes[i][0], shapes[i][1],
                     0x12345678u + static_cast<std::uint32_t>(i * 2654435761u),
                     pool);
        }
        run_unsupported_cases();

        ggml_threadpool_free(pool);
        std::puts("PASS bias_silu_or_null cases=10 unsupported=9");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 2;
    }
}
