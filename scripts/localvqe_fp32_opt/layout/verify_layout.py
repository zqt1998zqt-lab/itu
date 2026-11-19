#!/usr/bin/env python3
"""Independent semantic checks for the LocalVQE convolution-layout candidate."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(
    "/home/zhangqingtian/project/itu/build/deps/"
    "localvqe-f53063c9eb2a85f96479867d1dd911dc3bf6319b/"
    "ggml/localvqe_graph.cpp"
)
GENERATED_PATH = HERE / "localvqe_graph.cpp"
PREPARE_PATH = HERE / "prepare_layout.py"
EXPECTED_SOURCE_SHA256 = (
    "add57a11343753a0810e26d82380cc63d26e561b3cd1264efa4beef2b5900e22"
)
FUNCTION_SIGNATURE = "static struct ggml_tensor* lvq_conv_2d("
IM2COL_ANCHOR = (
    "    struct ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1,\n"
    "                                             true, GGML_TYPE_F32);\n"
)


@dataclass(frozen=True)
class TensorType:
    name: str


GGML_TYPE_F32 = TensorType("F32")


@dataclass(frozen=True)
class Tensor:
    type: TensorType
    ne: tuple[int, int, int, int]
    contiguous: bool = True


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_comments_and_strings(text: str) -> str:
    """Blank comments and literals while preserving offsets and newlines."""

    masked = list(text)
    state = "code"
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if state == "code":
            if char == "/" and following == "/":
                masked[index] = masked[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                masked[index] = masked[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                masked[index] = " "
                state = "string"
                index += 1
                continue
            if char == "'":
                masked[index] = " "
                state = "char"
                index += 1
                continue
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                masked[index] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                state = "code"
                index += 2
                continue
            if char != "\n":
                masked[index] = " "
        elif state in {"string", "char"}:
            if char == "\\":
                masked[index] = " "
                if index + 1 < len(text) and text[index + 1] != "\n":
                    masked[index + 1] = " "
                index += 2
                continue
            if (state == "string" and char == '"') or (
                state == "char" and char == "'"
            ):
                masked[index] = " "
                state = "code"
            elif char != "\n":
                masked[index] = " "
        index += 1
    return "".join(masked)


def find_matching_delimiter(
    text: str, open_index: int, opening: str, closing: str
) -> int:
    if text[open_index] != opening:
        raise ValueError(f"expected {opening!r} at offset {open_index}")

    depth = 0
    state = "code"
    index = open_index
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if state == "code":
            if char == "/" and following == "/":
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                state = "string"
                index += 1
                continue
            if char == "'":
                state = "char"
                index += 1
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return index
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
                continue
        elif state in {"string", "char"}:
            if char == "\\":
                index += 2
                continue
            if (state == "string" and char == '"') or (
                state == "char" and char == "'"
            ):
                state = "code"
        index += 1
    raise ValueError(f"unterminated {opening}{closing} pair")


def extract_function(text: str) -> tuple[int, int, int]:
    definitions = list(re.finditer(re.escape(FUNCTION_SIGNATURE), text))
    if len(definitions) != 1:
        raise AssertionError(
            f"expected one lvq_conv_2d definition, found {len(definitions)}"
        )
    start = definitions[0].start()
    opening = text.find("{", definitions[0].end())
    if opening < 0:
        raise AssertionError("lvq_conv_2d has no opening brace")
    closing = find_matching_delimiter(text, opening, "{", "}")
    return start, opening, closing


def extract_guarded_branch(body: str) -> tuple[str, str, int, int]:
    masked = mask_comments_and_strings(body)
    if_statements = list(re.finditer(r"\bif\b", masked))
    if len(if_statements) != 1:
        raise AssertionError(
            f"expected one guarded branch in lvq_conv_2d, found {len(if_statements)}"
        )

    cursor = if_statements[0].end()
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked) or masked[cursor] != "(":
        raise AssertionError("if statement is not followed by a parenthesized condition")
    condition_close = find_matching_delimiter(masked, cursor, "(", ")")
    condition = body[cursor + 1 : condition_close]

    cursor = condition_close + 1
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked) or masked[cursor] != "{":
        raise AssertionError("if statement does not have a braced fast-path body")
    block_close = find_matching_delimiter(masked, cursor, "{", "}")
    fast_block = body[cursor + 1 : block_close]
    return condition, fast_block, cursor, block_close


def evaluate_generated_condition(
    condition: str, a: Tensor, b: Tensor, im2col: Tensor
) -> bool:
    normalized = condition.replace("->", ".")
    normalized = normalized.replace("&&", " and ").replace("||", " or ")
    normalized = re.sub(r"\bGGML_TYPE_F32\b", "GGML_TYPE_F32", normalized)
    normalized = f"({normalized})"

    environment = {
        "a": a,
        "b": b,
        "im2col": im2col,
        "GGML_TYPE_F32": GGML_TYPE_F32,
        "ggml_is_contiguous": lambda tensor: tensor.contiguous,
    }
    return bool(eval(normalized, {"__builtins__": {}}, environment))


def expected_fast_path(a: Tensor, b: Tensor, im2col: Tensor) -> bool:
    positive_geometry = (
        a.ne[0] > 0
        and a.ne[1] > 0
        and a.ne[2] > 0
        and a.ne[3] > 0
        and im2col.ne[0] > 0
        and im2col.ne[1] > 0
        and im2col.ne[2] > 0
        and a.ne[0] * a.ne[1] * a.ne[2] == im2col.ne[0]
    )
    return (
        a.type == GGML_TYPE_F32
        and b.type == GGML_TYPE_F32
        and im2col.type == GGML_TYPE_F32
        and b.ne[3] == 1
        and im2col.ne[3] == 1
        and positive_geometry
        and a.contiguous
        and im2col.contiguous
    )


def check_branch_controls(condition: str) -> list[tuple[str, bool]]:
    f32 = GGML_TYPE_F32
    f16 = TensorType("F16")
    base_a = Tensor(f32, (2, 2, 3, 4))
    base_b = Tensor(f32, (4, 3, 3, 1))
    base_im2col = Tensor(f32, (12, 2, 3, 1))

    cases = [
        ("F32_batch1_geometry", base_a, base_b, base_im2col),
        ("batch2_fallback", base_a, Tensor(f32, (4, 3, 3, 2)), Tensor(f32, (12, 2, 3, 2))),
        ("weight_F16_fallback", Tensor(f16, (2, 2, 3, 4)), base_b, base_im2col),
        ("input_F16_fallback", base_a, Tensor(f16, (4, 3, 3, 1)), base_im2col),
        ("im2col_F16_fallback", base_a, base_b, Tensor(f16, (12, 2, 3, 1))),
        ("geometry_mismatch_fallback", base_a, base_b, Tensor(f32, (11, 2, 3, 1))),
        ("noncontiguous_weight_fallback", Tensor(f32, (2, 2, 3, 4), False), base_b, base_im2col),
        ("zero_channels_fallback", Tensor(f32, (2, 2, 3, 0)), base_b, Tensor(f32, (12, 2, 3, 1))),
    ]

    results: list[tuple[str, bool]] = []
    for name, a, b, im2col in cases:
        actual = evaluate_generated_condition(condition, a, b, im2col)
        expected = expected_fast_path(a, b, im2col)
        if actual != expected:
            raise AssertionError(
                f"{name}: generated guard={actual}, independent policy={expected}"
            )
        results.append((name, actual))

    if not results[0][1]:
        raise AssertionError("F32/batch-1 positive control did not select the fast path")
    for name, selected in results[1:]:
        if selected:
            raise AssertionError(f"{name} unexpectedly selected the fast path")
    return results


def conv_output_size(size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def deterministic_values(count: int, seed: int) -> list[int]:
    state = seed
    values = []
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append((state % 11) - 5)
    return values


def direct_conv2d(
    weight: list[int],
    input_values: list[int],
    weight_ne: tuple[int, int, int, int],
    input_ne: tuple[int, int, int, int],
    strides: tuple[int, int],
    pads: tuple[int, int],
    dilations: tuple[int, int],
) -> tuple[list[int], tuple[int, int, int, int]]:
    kw, kh, cin, cout = weight_ne
    iw, ih, input_channels, batch = input_ne
    if cin != input_channels or batch != 1:
        raise ValueError("direct oracle is intentionally batch-1")

    stride_w, stride_h = strides
    pad_w, pad_h = pads
    dilation_w, dilation_h = dilations
    ow = conv_output_size(iw, kw, stride_w, pad_w, dilation_w)
    oh = conv_output_size(ih, kh, stride_h, pad_h, dilation_h)
    output_ne = (ow, oh, cout, 1)
    output = [0] * (ow * oh * cout)
    kernel_size = kw * kh * cin

    for output_channel in range(cout):
        for out_y in range(oh):
            for out_x in range(ow):
                accumulator = 0
                for input_channel in range(cin):
                    for kernel_y in range(kh):
                        input_y = out_y * stride_h - pad_h + kernel_y * dilation_h
                        if input_y < 0 or input_y >= ih:
                            continue
                        for kernel_x in range(kw):
                            input_x = out_x * stride_w - pad_w + kernel_x * dilation_w
                            if input_x < 0 or input_x >= iw:
                                continue
                            kernel_index = (
                                kernel_x
                                + kw * kernel_y
                                + kw * kh * input_channel
                            )
                            weight_index = (
                                kernel_index + kernel_size * output_channel
                            )
                            input_index = (
                                input_x + iw * (input_y + ih * input_channel)
                            )
                            accumulator += (
                                weight[weight_index] * input_values[input_index]
                            )
                output_index = (
                    out_x + ow * (out_y + oh * output_channel)
                )
                output[output_index] = accumulator
    return output, output_ne


def im2col_matrix(
    input_values: list[int],
    input_ne: tuple[int, int, int, int],
    weight_ne: tuple[int, int, int, int],
    strides: tuple[int, int],
    pads: tuple[int, int],
    dilations: tuple[int, int],
) -> tuple[list[int], tuple[int, int, int, int]]:
    kw, kh, cin, _ = weight_ne
    iw, ih, input_channels, batch = input_ne
    if cin != input_channels or batch != 1:
        raise ValueError("im2col oracle is intentionally batch-1")

    stride_w, stride_h = strides
    pad_w, pad_h = pads
    dilation_w, dilation_h = dilations
    ow = conv_output_size(iw, kw, stride_w, pad_w, dilation_w)
    oh = conv_output_size(ih, kh, stride_h, pad_h, dilation_h)
    kernel_size = kw * kh * cin
    positions = ow * oh
    matrix = [0] * (kernel_size * positions)

    for out_y in range(oh):
        for out_x in range(ow):
            position = out_x + ow * out_y
            for input_channel in range(cin):
                for kernel_y in range(kh):
                    input_y = out_y * stride_h - pad_h + kernel_y * dilation_h
                    for kernel_x in range(kw):
                        input_x = (
                            out_x * stride_w - pad_w + kernel_x * dilation_w
                        )
                        kernel_index = (
                            kernel_x
                            + kw * kernel_y
                            + kw * kh * input_channel
                        )
                        matrix_index = kernel_index + kernel_size * position
                        if 0 <= input_x < iw and 0 <= input_y < ih:
                            input_index = (
                                input_x + iw * (input_y + ih * input_channel)
                            )
                            matrix[matrix_index] = input_values[input_index]
    return matrix, (kernel_size, ow, oh, 1)


def optimized_layout(
    weight: list[int],
    matrix: list[int],
    matrix_ne: tuple[int, int, int, int],
    cout: int,
) -> list[int]:
    kernel_size, ow, oh, batch = matrix_ne
    if batch != 1:
        raise ValueError("optimized layout control requires batch=1")
    positions = ow * oh
    result = [0] * (positions * cout)
    for output_channel in range(cout):
        for position in range(positions):
            accumulator = 0
            for kernel_index in range(kernel_size):
                accumulator += (
                    matrix[kernel_index + kernel_size * position]
                    * weight[kernel_index + kernel_size * output_channel]
                )
            result[position + positions * output_channel] = accumulator
    return result


def original_layout(
    weight: list[int],
    matrix: list[int],
    matrix_ne: tuple[int, int, int, int],
    cout: int,
) -> list[int]:
    kernel_size, ow, oh, batch = matrix_ne
    if batch != 1:
        raise ValueError("original layout control requires batch=1")
    positions = ow * oh
    channels_by_positions = [0] * (cout * positions)
    for position in range(positions):
        for output_channel in range(cout):
            accumulator = 0
            for kernel_index in range(kernel_size):
                accumulator += (
                    weight[kernel_index + kernel_size * output_channel]
                    * matrix[kernel_index + kernel_size * position]
                )
            channels_by_positions[output_channel + cout * position] = accumulator

    contiguous_transpose = [0] * (positions * cout)
    for position in range(positions):
        for output_channel in range(cout):
            contiguous_transpose[position + positions * output_channel] = (
                channels_by_positions[output_channel + cout * position]
            )
    return contiguous_transpose


def check_layout_mappings() -> list[str]:
    configs = [
        {
            "name": "dense_1x1_batch1",
            "weight_ne": (2, 2, 3, 4),
            "input_ne": (4, 3, 3, 1),
            "strides": (1, 1),
            "pads": (0, 0),
            "dilations": (1, 1),
        },
        {
            "name": "strided_padded_dilated_batch1",
            "weight_ne": (3, 2, 2, 5),
            "input_ne": (7, 5, 2, 1),
            "strides": (2, 1),
            "pads": (1, 0),
            "dilations": (1, 2),
        },
    ]

    summaries = []
    for seed, config in enumerate(configs, start=1):
        weight_ne = config["weight_ne"]
        input_ne = config["input_ne"]
        weight = deterministic_values(
            weight_ne[0] * weight_ne[1] * weight_ne[2] * weight_ne[3], seed
        )
        input_values = deterministic_values(
            input_ne[0] * input_ne[1] * input_ne[2] * input_ne[3], seed + 100
        )
        direct, output_ne = direct_conv2d(
            weight,
            input_values,
            weight_ne,
            input_ne,
            config["strides"],
            config["pads"],
            config["dilations"],
        )
        matrix, matrix_ne = im2col_matrix(
            input_values,
            input_ne,
            weight_ne,
            config["strides"],
            config["pads"],
            config["dilations"],
        )
        fast = optimized_layout(weight, matrix, matrix_ne, weight_ne[3])
        fallback = original_layout(weight, matrix, matrix_ne, weight_ne[3])

        if fast != direct:
            raise AssertionError(
                f"{config['name']}: optimized [positions,Cout] mapping != direct convolution"
            )
        if fallback != direct:
            raise AssertionError(
                f"{config['name']}: original transpose/cont mapping != direct convolution"
            )

        ow, oh, cout, batch = output_ne
        positions = ow * oh
        mapping = [
            position + positions * output_channel
            for output_channel in range(cout)
            for position in range(positions)
        ]
        if sorted(mapping) != list(range(len(direct))):
            raise AssertionError(
                f"{config['name']}: output mapping is not one-to-one"
            )
        summaries.append(
            f"{config['name']}: elements={len(direct)} output_ne={output_ne}"
        )
    return summaries


def main() -> int:
    for python_path in (PREPARE_PATH, Path(__file__)):
        ast.parse(python_path.read_text(encoding="utf-8"), filename=str(python_path))
    print("PASS python AST: prepare_layout.py and verify_layout.py")

    source_bytes = SOURCE_PATH.read_bytes()
    source_sha256 = sha256(source_bytes)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise AssertionError(
            f"source SHA-256 mismatch: {source_sha256} != {EXPECTED_SOURCE_SHA256}"
        )
    print(f"PASS source hash: {source_sha256}")

    source = source_bytes.decode("utf-8")
    generated_bytes = GENERATED_PATH.read_bytes()
    generated = generated_bytes.decode("utf-8")

    original_start, original_open, original_close = extract_function(source)
    generated_start, generated_open, generated_close = extract_function(generated)

    if generated[:generated_start] != source[:original_start]:
        raise AssertionError("generated file changed bytes before lvq_conv_2d")
    if generated[generated_close + 1 :] != source[original_close + 1 :]:
        raise AssertionError("generated file changed bytes after lvq_conv_2d")

    original_body = source[original_open + 1 : original_close]
    generated_body = generated[generated_open + 1 : generated_close]
    if original_body.count(IM2COL_ANCHOR) != 1:
        raise AssertionError("source im2col anchor is not unique")
    fallback_tail = original_body[
        original_body.find(IM2COL_ANCHOR) + len(IM2COL_ANCHOR) :
    ]
    if not generated_body.endswith(fallback_tail):
        raise AssertionError("original fallback statements were not preserved verbatim")

    condition, fast_block, _, _ = extract_guarded_branch(generated_body)
    fast_masked = mask_comments_and_strings(fast_block)
    fallback_masked = mask_comments_and_strings(fallback_tail)

    if "ggml_cont" in fast_masked:
        raise AssertionError("fast path still contains a CONT operation")
    if fallback_masked.count("ggml_cont(") != 2:
        raise AssertionError("fallback no longer contains exactly two CONT operations")
    if fast_masked.count("ggml_mul_mat") != 1:
        raise AssertionError("fast path must contain exactly one mul_mat")
    if fast_masked.count("ggml_reshape_2d") != 2:
        raise AssertionError("fast path must contain exactly two reshape_2d calls")
    if fast_masked.count("ggml_reshape_4d") != 1:
        raise AssertionError("fast path must contain exactly one reshape_4d call")
    if fast_masked.count("return") != 1:
        raise AssertionError("fast path must return exactly once")

    compact_fast = re.sub(r"\s+", " ", fast_masked)
    required_fragments = (
        "struct ggml_tensor* im2col_2d = ggml_reshape_2d( ctx, im2col,",
        "struct ggml_tensor* weight_2d = ggml_reshape_2d( ctx, a,",
        "ggml_mul_mat(ctx, im2col_2d, weight_2d);",
        "return ggml_reshape_4d(ctx, positions_by_channels, im2col->ne[1], "
        "im2col->ne[2], a->ne[3], 1);",
    )
    for fragment in required_fragments:
        if fragment not in compact_fast:
            raise AssertionError(
                f"fast-path operation/layout fragment missing: {fragment}"
            )
    print("PASS function structure: guarded no-CONT fast path, verbatim fallback")

    controls = check_branch_controls(condition)
    print(
        "PASS guard controls: "
        + ", ".join(f"{name}={selected}" for name, selected in controls)
    )

    layout_summaries = check_layout_mappings()
    for summary in layout_summaries:
        print(f"PASS layout one-to-one: {summary}")

    print(
        f"VERIFY_LAYOUT: PASS source_sha256={source_sha256} "
        f"generated_sha256={sha256(generated_bytes)} "
        f"generated_bytes={len(generated_bytes)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"VERIFY_LAYOUT: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
