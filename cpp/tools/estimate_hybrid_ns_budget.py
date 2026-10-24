#!/usr/bin/env python3
"""Analytical matrix-only budget for a proposed GRU NS student, not a benchmark."""
import argparse
import json


def estimate(input_size, hidden_size, output_size, hops_per_second):
    mac_per_hop = 3 * hidden_size * (input_size + hidden_size) + hidden_size * output_size
    parameters = (
        3 * hidden_size * input_size
        + 3 * hidden_size * hidden_size
        + 6 * hidden_size
        + hidden_size * output_size
        + output_size
    )
    return {
        "hidden_size": hidden_size,
        "matrix_mac_per_hop": mac_per_hop,
        "matrix_mmac_per_second": mac_per_hop * hops_per_second / 1_000_000,
        "parameters_with_pytorch_gru_double_bias": parameters,
        "fp32_weight_bytes": parameters * 4,
        "fp32_weight_kib": parameters * 4 / 1024,
        "per_stream_hidden_bytes": hidden_size * 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--hops-per-second", type=int, default=100)
    args = parser.parse_args()
    if min(args.input_size, args.output_size, args.hops_per_second, *args.hidden) <= 0:
        parser.error("all dimensions and hop frequency must be positive")
    print(json.dumps({
        "kind": "analytical_design_budget_not_measured_runtime",
        "input_size": args.input_size,
        "output_size": args.output_size,
        "hops_per_second": args.hops_per_second,
        "rows": [estimate(args.input_size, h, args.output_size, args.hops_per_second)
                 for h in args.hidden],
        "excluded": ["FFT/WOLA", "noise_PSD", "OMLSA_gain", "band_mapping",
                     "features", "bias_additions", "activations", "normalization",
                     "elementwise_GRU_gates", "runtime_and_memory_overheads"],
        "runtime_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
