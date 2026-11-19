#!/usr/bin/env python3
"""Export and check the frozen fixed-state CoFi-Lite frame graph as ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path("/home/zhangqingtian/project/itu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import cofilite_stream as stream  # noqa: E402
from server.cofilite_frame import CoFiFrameCore, STATE_SHAPES, initial_frame_states  # noqa: E402
from server.cofilite_onnx import (  # noqa: E402
    ENHANCED_SHAPE,
    INPUT_NAMES,
    OUTPUT_NAMES,
    SPECTRUM_SHAPE,
    STATE_COUNT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(
    output_dir: str | Path,
    *,
    version: str = "v1",
    opset: int = 18,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / f"cofilite-frame-{version}-opset{opset}.onnx"
    identity_path = output_root / f"cofilite-frame-{version}-opset{opset}.identity.json"
    checker_path = output_root / f"cofilite-frame-{version}-opset{opset}.checker.json"
    for path in (model_path, identity_path, checker_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    model = stream.load_model("cpu", threads=1)
    core = CoFiFrameCore(model).eval()
    states = initial_frame_states()
    spectrum_real = torch.zeros(SPECTRUM_SHAPE, dtype=torch.float32)
    spectrum_imag = torch.zeros(SPECTRUM_SHAPE, dtype=torch.float32)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            core,
            (spectrum_real, spectrum_imag, *states),
            str(model_path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            opset_version=opset,
            dynamo=False,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
    export_sec = time.perf_counter() - started

    onnx.checker.check_model(str(model_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        INPUT_NAMES[0]: spectrum_real.numpy(),
        INPUT_NAMES[1]: spectrum_imag.numpy(),
        **{
            INPUT_NAMES[index + 2]: state.numpy()
            for index, state in enumerate(states)
        },
    }
    ort_outputs = session.run(list(OUTPUT_NAMES), feeds)
    if len(ort_outputs) != 1 + STATE_COUNT:
        raise RuntimeError("ORT output count changed during export check")
    if tuple(ort_outputs[0].shape) != ENHANCED_SHAPE:
        raise RuntimeError("ORT enhanced output shape changed during export check")
    if ort_outputs[0].dtype != np.float32:
        raise RuntimeError("ORT enhanced output dtype changed during export check")
    if not bool(__import__("numpy").isfinite(ort_outputs[0]).all()):
        raise FloatingPointError("ORT enhanced output is not finite")

    identity = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": version,
        "opset": opset,
        "dynamo": False,
        "do_constant_folding": False,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "weights_path": str(stream.FROZEN_CHECKPOINT_PATH),
        "weights_sha256": _sha256(stream.FROZEN_CHECKPOINT_PATH),
        "frozen_stream_source": str(ROOT / "server/cofilite_stream.py"),
        "frozen_stream_source_sha256": _sha256(ROOT / "server/cofilite_stream.py"),
        "frozen_frame_source": str(ROOT / "server/cofilite_frame.py"),
        "frozen_frame_source_sha256": _sha256(ROOT / "server/cofilite_frame.py"),
        "state_count": STATE_COUNT,
        "state_shapes": [list(shape) for shape in STATE_SHAPES],
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "enhanced_shape": list(ENHANCED_SHAPE),
        "spectrum_shape": list(SPECTRUM_SHAPE),
        "export_sec": float(export_sec),
        "onnx_checker": "pass",
        "ort_provider": session.get_providers(),
        "ort_session_check": "pass",
        "ort_output_finite": True,
    }
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checker = {
        "schema_version": 1,
        "model_sha256": identity["model_sha256"],
        "onnx_checker": "pass",
        "ort_provider": session.get_providers(),
        "ort_input_count": len(session.get_inputs()),
        "ort_output_count": len(session.get_outputs()),
        "ort_enhanced_shape": list(ort_outputs[0].shape),
        "ort_enhanced_dtype": str(ort_outputs[0].dtype),
        "ort_enhanced_finite": True,
    }
    checker_path.write_text(
        json.dumps(checker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build/cofilite-ort-20261117",
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--opset", type=int, choices=(17, 18), default=18)
    args = parser.parse_args()
    identity = export_model(
        args.output_dir,
        version=args.version,
        opset=args.opset,
    )
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
