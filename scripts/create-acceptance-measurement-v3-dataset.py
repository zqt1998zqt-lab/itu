#!/usr/bin/env python3
"""Build measurement-aware-v3-8k-16k-48k with true 8/16/24/48 kHz rates and measurable downlink idle noise."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from scipy import signal
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(
    SCRIPTS / "create-acceptance-measurement-dataset.py",
    "itu_base_acceptance_measurement_dataset",
)

NAME = "measurement-aware-v3-8k-16k-48k"
BASE.NAME = NAME


def v3_sample_rate(conditions: dict, kind: str) -> int:
    band = conditions["mode"].split("-")[0]
    if band == "SWB":
        return 48000
    if band == "WB":
        return 24000 if kind in {"oob", "response"} else 16000
    if band == "NB":
        return 16000 if kind in {"oob", "response"} else 8000
    return 16000


BASE.sample_rate = v3_sample_rate

_orig_validate_case = BASE.validate_case


def v3_validate_case(signals: dict, case: dict, plan: dict, sidecar: dict | None = None) -> dict:
    kind = plan["kind"]
    rate = plan["rate"]
    frames = round(BASE.DURATION_S * rate)
    if kind == "idle" and case["conditions"]["direction"] == "receive":
        patched_signals = dict(signals)
        patched_signals["far_end"] = np.zeros(frames, dtype=np.float64)
        res = _orig_validate_case(patched_signals, case, plan, sidecar)
        far_rms = BASE.rms(signals["far_end"])
        far_dbfs = 20.0 * math.log10(max(far_rms, 1e-15))
        if not (-80.0 <= far_dbfs <= -70.0):
            raise ValueError(f"receive idle far_end floor out of expected -75 dBFS range: {far_dbfs:.2f} dBFS")
        res["receive_idle_far_end_dbfs"] = round(far_dbfs, 4)
        return res
    return _orig_validate_case(signals, case, plan, sidecar)


BASE.validate_case = v3_validate_case


def build_v3(output: Path, *, seed: int = 20261104, do_import: bool = True) -> dict:
    BASE.preflight(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank = BASE.SourceBank()
    manifest, plans = BASE.plan_cases()
    analyses = {}
    blobs = {}
    raw_total = 0
    for case in manifest["cases"]:
        conditions = case["conditions"]
        plan = plans[case["id"]]
        kind, rate = plan["kind"], plan["rate"]
        near, near_info = bank.speech(conditions["mode"], "reference", rate)
        far, far_info = bank.speech(conditions["mode"], "far_end", rate)
        if rate == 8000:
            sos = signal.butter(6, 3400.0, fs=rate, output="sos")
            near = signal.sosfiltfilt(sos, near)
            far = signal.sosfiltfilt(sos, far)
        noise, noise_info = bank.noise(conditions["mode"], conditions["scene"], rate, seed)
        case_seed = BASE.stable_seed(f"{conditions['mode']}:{conditions['scene']}:{kind}", seed)
        signals, details = BASE.render_signals(
            near, far, noise, rate, direction=conditions["direction"], kind=kind,
            bandwidth=conditions["mode"].split("-")[0], snr_db=BASE.SNR_DB[conditions["scene"]],
            ser_db=-5., echo_delay_ms=20. + bank.modes.index(conditions["mode"]) * 4.,
            seed=case_seed, input_level_db=float(conditions["input_level_db"]),
            volume_gain_db=BASE.VOLUME_DB[conditions["volume"]])
        if kind == "idle" and conditions["direction"] == "receive":
            signals["far_end"] = np.asarray(signals["noise"], dtype=np.float64).copy()
            details["receive_idle_far_end_floor_dbfs"] = -75.0
        v3_validate_case(signals, case, plan)
        decoded = {}
        for role in BASE.ROLES:
            blob, decoded[role] = BASE.encode(signals[role], rate)
            if len(blob) > BASE.MAX_ASSET_BYTES:
                raise ValueError("asset exceeds import bound")
            digest = BASE.sha(blob)
            relative = f"audio/{digest}.wav"
            if relative not in blobs:
                blobs[relative] = blob
                raw_total += len(blob)
                if raw_total > BASE.MAX_TOTAL_UNCOMPRESSED_BYTES - 1024 * 1024:
                    raise ValueError("unique WAV assets exceed import expansion bound")
            case["signals"][role] = relative
        checks = v3_validate_case(decoded, case, plan, details)
        checks["component_identity"] = BASE.verify_mixture(decoded, details)
        if details["global_safety_gain"] != 1.0:
            raise ValueError("source headroom insufficient; per-case scaling would change level variants")
        if "beam_delays_ms" in case and details.get("beam_delays_ms") != case["beam_delays_ms"]:
            raise ValueError("microphone geometry and CLI beam calibration disagree")
        analyses[case["id"]] = {
            **plan,
            "conditions": conditions,
            "item_ids": case["item_ids"],
            "sources": {"near": near_info, "far": far_info, "noise": noise_info},
            "signal_design": details,
            "quality": checks,
        }
    BASE.verify_levels(manifest, blobs)
    manifest_blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def write_member(name: str, blob: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, blob)
        write_member("manifest.json", manifest_blob)
        for relative, blob in sorted(blobs.items()):
            write_member(relative, blob)
    if output.stat().st_size > BASE.MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP exceeds import bound; artifact retained")
    imported = BASE.import_dataset(output.read_bytes()) if do_import else None
    report = {
        "schema_version": 1,
        "dataset_name": NAME,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standard_compliance": False,
        "scope": "measurement-aware v3 (8/16/24/48 kHz + measurable downlink idle noise); not certification",
        "baseline_dataset_id": BASE.BASELINE_ID,
        "output": str(output),
        "zip_sha256": BASE.file_sha(output),
        "zip_bytes": output.stat().st_size,
        "manifest_sha256": BASE.sha(manifest_blob),
        "case_count": len(manifest["cases"]),
        "item_count": 336,
        "duration_s": BASE.DURATION_S,
        "unique_assets": len(blobs),
        "uncompressed_bytes": raw_total + len(manifest_blob),
        "profiles": dict(Counter(plan["kind"] for plan in plans.values())),
        "speaker_ids": sorted({r["speaker"] for r in bank.used.values() if "speaker" in r}),
        "sources": list(bank.used.values()),
        "cases": analyses,
        "import_summary": imported,
    }
    report_path = output.with_suffix(".quality.json")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {key: value for key, value in report.items() if key not in {"cases", "sources"}} | {"quality_report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--no-import", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or PROJECT / "build" / f"{NAME}-{stamp}.zip").resolve()
    report = build_v3(output, seed=args.seed, do_import=not args.no_import)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
