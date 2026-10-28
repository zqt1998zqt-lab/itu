#!/usr/bin/env python3
"""Verify the local Model A implementation against ETSI Annex C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from server.quality import ts103281


EXPECTED = (
    PROJECT
    / "server/quality/assets/ts103281/annex-c-model-a.csv"
)
BASE_URL = (
    "https://docbox.etsi.org/Reference/STQ/"
    "TS%20103%20281/Annex_C%20test%20vectors"
)


def expected_rows() -> dict[str, dict[str, float]]:
    with EXPECTED.open(newline="") as handle:
        return {
            row["condition"]: {
                "S-MOS": float(row["s_mos"]),
                "N-MOS": float(row["n_mos"]),
                "G-MOS": float(row["g_mos"]),
            }
            for row in csv.DictReader(handle)
        }


def download_vectors(directory: Path) -> None:
    for number in range(1, 49):
        name = f"C{number:02d}_f1s1.wav"
        destination = directory / name
        if destination.is_file():
            continue
        request = Request(
            f"{BASE_URL}/{name}",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.etsi.org/",
            },
        )
        with urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())


def verify(directory: Path, limit: int | None = None) -> dict:
    expected = expected_rows()
    names = sorted(expected)
    if limit is not None:
        names = names[:limit]
    rows = []
    for condition in names:
        path = directory / f"{condition}_f1s1.wav"
        samples, rate = sf.read(path, always_2d=True, dtype="float64")
        if samples.shape[1] != 2:
            raise ValueError(f"{path.name}: expected two channels")
        # ETSI files use channel 2 for the common reference and channel 1 for
        # the degraded signal.
        result = ts103281.evaluate_model_a(
            samples[:, 1],
            samples[:, 0],
            rate,
            include_diagnostic_calibration=True,
        )
        actual = {
            name: float(score["value"])
            for name, score in result["scores"].items()
        }
        diagnostic_actual = {
            name: float(score["value"])
            for name, score in result["diagnostic_scores"].items()
        }
        differences = {
            name: actual[name] - expected[condition][name]
            for name in actual
        }
        diagnostic_differences = {
            name: diagnostic_actual[name] - expected[condition][name]
            for name in diagnostic_actual
        }
        rows.append(
            {
                "condition": condition,
                "expected": expected[condition],
                "actual": actual,
                "difference": differences,
                "diagnostic_actual": diagnostic_actual,
                "diagnostic_difference": diagnostic_differences,
                "within_0_1": all(
                    abs(value) <= 0.1 + 1e-12
                    for value in differences.values()
                ),
            }
        )
    errors = {
        metric: np.asarray(
            [row["difference"][metric] for row in rows], dtype=np.float64
        )
        for metric in ("S-MOS", "N-MOS", "G-MOS")
    }
    diagnostic_errors = {
        metric: np.asarray(
            [row["diagnostic_difference"][metric] for row in rows],
            dtype=np.float64,
        )
        for metric in ("S-MOS", "N-MOS", "G-MOS")
    }
    conformant = all(row["within_0_1"] for row in rows)
    return {
        "standard": "ETSI TS 103 281 V1.3.2 Annex C",
        "condition_count": len(rows),
        "tolerance_mos": 0.1,
        "conformant": conformant,
        "passed_conditions": sum(row["within_0_1"] for row in rows),
        "implementation_sha256": hashlib.sha256(
            Path(ts103281.__file__).read_bytes()
        ).hexdigest(),
        "failed_reason": (
            None
            if conformant
            else "One or more official conditions exceeded the +/-0.1 MOS gate."
        ),
        "summary": {
            metric: {
                "mae": float(np.mean(np.abs(values))),
                "rmse": float(np.sqrt(np.mean(values**2))),
                "maximum_absolute_error": float(np.max(np.abs(values))),
            }
            for metric, values in errors.items()
        },
        "diagnostic_summary": {
            metric: {
                "mae": float(np.mean(np.abs(values))),
                "rmse": float(np.sqrt(np.mean(values**2))),
                "maximum_absolute_error": float(np.max(np.abs(values))),
            }
            for metric, values in diagnostic_errors.items()
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        type=Path,
        help="directory containing C01_f1s1.wav through C48_f1s1.wav",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="write the complete verification evidence JSON to this path",
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 48:
        parser.error("--limit must be in 1..48")
    if args.vectors is None:
        with tempfile.TemporaryDirectory(prefix="etsi-ts103281-") as temporary:
            directory = Path(temporary)
            download_vectors(directory)
            report = verify(directory, args.limit)
    else:
        report = verify(args.vectors, args.limit)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(
            f"{report['passed_conditions']}/{report['condition_count']} "
            f"conditions within +/-{report['tolerance_mos']} MOS"
        )
        for metric, values in report["summary"].items():
            print(
                f"{metric}: MAE={values['mae']:.4f}, "
                f"RMSE={values['rmse']:.4f}, "
                f"max={values['maximum_absolute_error']:.4f}"
            )
        print("Diagnostic calibration (not conformance evidence):")
        for metric, values in report["diagnostic_summary"].items():
            print(
                f"{metric}: MAE={values['mae']:.4f}, "
                f"RMSE={values['rmse']:.4f}, "
                f"max={values['maximum_absolute_error']:.4f}"
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0 if report["conformant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
