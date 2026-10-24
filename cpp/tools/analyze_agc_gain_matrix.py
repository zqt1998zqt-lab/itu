#!/usr/bin/env python3
"""Audit actual AGC PCM gain from frozen diagnostic-stage taps.

The gain is a case-level, whole-file energy ratio between the protected-NR
input and the core-rate AGC output. Both taps already share the same causal
timeline, so no lag is fitted. A case is "nonzero" only when the absolute
energy ratio is greater than 0.01 dB; the lower threshold excludes float
round-off and resampler tails while retaining the smallest observed 8 dB
control step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[2]
INPUT_STAGE = "nr-protected-agc-before"
OUTPUT_STAGE = "agc-after"
NONZERO_DB = 0.01


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage(path: Path, expected_name: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if data.shape[1] != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError(f"invalid {expected_name} tap: {path}")
    if np.max(np.abs(data)) > 1:
        raise ValueError(f"{expected_name} tap exceeds full scale: {path}")
    return data[:, 0]


def analyze(root: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in (root / "results.jsonl").open()]
    if len({(row["original_run"], row["scene"]["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate run-scoped case")
    cases = []
    for row in rows:
        scene = row["scene"]
        run = row["original_run"]
        folder = root / run / "cases" / scene["id"] / "runs/full"
        before_path = folder / f"result.{INPUT_STAGE}.float.wav"
        after_path = folder / f"result.{OUTPUT_STAGE}.float.wav"
        before = load_stage(before_path, INPUT_STAGE)
        after = load_stage(after_path, OUTPUT_STAGE)
        if before.shape != after.shape:
            raise ValueError(f"tap length differs: {run}/{scene['id']}")
        input_power = float(np.mean(before * before))
        output_power = float(np.mean(after * after))
        if input_power <= 1e-20 or output_power <= 1e-20:
            raise ValueError(f"non-excited gain tap: {run}/{scene['id']}")
        gain_db = 10.0 * math.log10(output_power / input_power)
        cases.append(
            {
                "run": run,
                "id": scene["id"],
                "sample_rate": scene["sample_rate"],
                "near_dbfs": scene["near_dbfs"],
                "microphones": scene["mics"],
                "gain_db": gain_db,
                "status": row["acceptance"]["status"],
                "before_sha256": digest(before_path),
                "after_sha256": digest(after_path),
            }
        )
    gains = [case["gain_db"] for case in cases]
    return {
        "schema_version": 1,
        "method": {
            "scope": "whole input/output tap, case-level stable energy ratio",
            "input_stage": INPUT_STAGE,
            "output_stage": OUTPUT_STAGE,
            "delay_alignment": "declared shared diagnostic timeline; no fitted lag",
            "formula": "10*log10(mean(after^2)/mean(before^2))",
            "nonzero_db": NONZERO_DB,
            "normalization": "none",
        },
        "source": str(root),
        "cases": len(cases),
        "gain_db": {
            "median": statistics.median(gains),
            "min": min(gains),
            "max": max(gains),
            "nonzero": sum(abs(gain) > NONZERO_DB for gain in gains),
        },
        "case_results": cases,
        "script_sha256": digest(Path(__file__)),
        "standard_compliance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = analyze(root)
    output = args.output or root / "gain-analysis.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "gain_db": report["gain_db"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
