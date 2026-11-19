#!/usr/bin/env python3
"""Create the LocalVQE FP32 batch-1 convolution-layout candidate.

The input file is hash-pinned.  This generator copies the complete source and
inserts one guarded fast path immediately after lvq_conv_2d's im2col node.  The
original fallback statements remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path


SOURCE_PATH = Path(
    "/home/zhangqingtian/project/itu/build/deps/"
    "localvqe-f53063c9eb2a85f96479867d1dd911dc3bf6319b/"
    "ggml/localvqe_graph.cpp"
)
OUTPUT_PATH = Path(__file__).resolve().with_name("localvqe_graph.cpp")
EXPECTED_SOURCE_SHA256 = (
    "add57a11343753a0810e26d82380cc63d26e561b3cd1264efa4beef2b5900e22"
)

FUNCTION_SIGNATURE = "static struct ggml_tensor* lvq_conv_2d("
IM2COL_ANCHOR = (
    "    struct ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1,\n"
    "                                             true, GGML_TYPE_F32);\n"
)

FAST_PATH = r"""
    // F32/batch-1 layout fast path: use im2col as mul_mat src0 so the
    // result is already [positions, Cout].  Flattening positions as
    // OW + OW*OH then matches [OW, OH, Cout, 1] without transpose/permute
    // materialization.  Every other case keeps the original path below.
    if (a->type == GGML_TYPE_F32 &&
        b->type == GGML_TYPE_F32 &&
        im2col->type == GGML_TYPE_F32 &&
        b->ne[3] == 1 &&
        im2col->ne[3] == 1 &&
        a->ne[0] > 0 && a->ne[1] > 0 && a->ne[2] > 0 && a->ne[3] > 0 &&
        im2col->ne[0] > 0 && im2col->ne[1] > 0 && im2col->ne[2] > 0 &&
        a->ne[0] * a->ne[1] * a->ne[2] == im2col->ne[0] &&
        ggml_is_contiguous(a) &&
        ggml_is_contiguous(im2col)) {
        const int64_t positions = im2col->ne[1] * im2col->ne[2];
        struct ggml_tensor* im2col_2d = ggml_reshape_2d(
            ctx, im2col, im2col->ne[0], positions);
        struct ggml_tensor* weight_2d = ggml_reshape_2d(
            ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]);
        struct ggml_tensor* positions_by_channels =
            ggml_mul_mat(ctx, im2col_2d, weight_2d);
        return ggml_reshape_4d(ctx, positions_by_channels,
                               im2col->ne[1], im2col->ne[2], a->ne[3], 1);
    }
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_matching_brace(text: str, open_index: int) -> int:
    if text[open_index] != "{":
        raise ValueError("internal error: expected an opening brace")

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
            if char == "{":
                depth += 1
            elif char == "}":
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

    raise ValueError("unterminated function body")


def extract_function_span(text: str) -> tuple[int, int, int]:
    matches = list(re.finditer(re.escape(FUNCTION_SIGNATURE), text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one lvq_conv_2d definition, found {len(matches)}"
        )

    signature_start = matches[0].start()
    open_brace = text.find("{", matches[0].end())
    if open_brace < 0:
        raise ValueError("lvq_conv_2d has no opening brace")
    close_brace = find_matching_brace(text, open_brace)
    return signature_start, open_brace, close_brace


def build_expected_source(source: str) -> tuple[str, dict[str, int]]:
    actual_sha256 = sha256_bytes(source.encode("utf-8"))
    if actual_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "source SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_sha256}"
        )

    _, body_open, body_close = extract_function_span(source)
    body = source[body_open + 1 : body_close]

    if body.count(IM2COL_ANCHOR) != 1:
        raise ValueError(
            "expected exactly one lvq_conv_2d im2col anchor, "
            f"found {body.count(IM2COL_ANCHOR)}"
        )
    if body.count("ggml_mul_mat(ctx,") != 1:
        raise ValueError(
            "expected exactly one original lvq_conv_2d mul_mat call, "
            f"found {body.count('ggml_mul_mat(ctx,')}"
        )
    if body.count("ggml_cont(") != 2:
        raise ValueError(
            "expected exactly two original lvq_conv_2d cont calls, "
            f"found {body.count('ggml_cont(')}"
        )
    if body.count("ggml_cont(ctx, ggml_transpose(ctx, result))") != 1:
        raise ValueError("original transpose/cont guard did not match exactly once")
    if body.count("ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2))") != 1:
        raise ValueError("original permute/cont guard did not match exactly once")
    if "positions_by_channels" in body:
        raise ValueError("source already contains the optimized layout marker")

    anchor_end = body.find(IM2COL_ANCHOR) + len(IM2COL_ANCHOR)
    rewritten_body = body[:anchor_end] + FAST_PATH + body[anchor_end:]
    rewritten = source[: body_open + 1] + rewritten_body + source[body_close:]

    if rewritten.count(FUNCTION_SIGNATURE) != 1:
        raise ValueError("rewritten source changed the lvq_conv_2d definition count")
    if rewritten.count(IM2COL_ANCHOR) != 1:
        raise ValueError("rewritten source changed the im2col anchor count")

    stats = {
        "source_bytes": len(source.encode("utf-8")),
        "source_lines": source.count("\n"),
        "function_body_bytes": len(body.encode("utf-8")),
        "inserted_bytes": len(FAST_PATH.encode("utf-8")),
        "fallback_cont_calls": body.count("ggml_cont("),
    }
    return rewritten, stats


def read_source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def write_new_file(data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(OUTPUT_PATH, flags, 0o644)
    with os.fdopen(descriptor, "wb", closefd=True) as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing generated file without writing anything",
    )
    args = parser.parse_args()

    source = read_source()
    rewritten, stats = build_expected_source(source)
    rewritten_bytes = rewritten.encode("utf-8")
    rewritten_sha256 = sha256_bytes(rewritten_bytes)

    if args.check:
        if not OUTPUT_PATH.is_file():
            raise FileNotFoundError(f"generated file is missing: {OUTPUT_PATH}")
        actual_bytes = OUTPUT_PATH.read_bytes()
        actual_sha256 = sha256_bytes(actual_bytes)
        if actual_bytes != rewritten_bytes:
            raise ValueError(
                "generated file does not match the hash-pinned transformation: "
                f"expected {rewritten_sha256}, got {actual_sha256}"
            )
        print(f"PREPARE_LAYOUT_CHECK: PASS sha256={actual_sha256}")
        print(
            "source_sha256="
            f"{EXPECTED_SOURCE_SHA256} source_lines={stats['source_lines']} "
            f"fallback_cont_calls={stats['fallback_cont_calls']}"
        )
        return 0

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite existing generated file: {OUTPUT_PATH}"
        )

    write_new_file(rewritten_bytes)
    print(f"PREPARE_LAYOUT_WRITE: PASS path={OUTPUT_PATH}")
    print(
        f"source_sha256={EXPECTED_SOURCE_SHA256} output_sha256={rewritten_sha256}"
    )
    print(
        f"inserted_bytes={stats['inserted_bytes']} "
        f"fallback_cont_calls={stats['fallback_cont_calls']}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"PREPARE_LAYOUT: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
