#!/usr/bin/env python3
"""DNSMOS P.835 diagnostics for frozen five-route sustained double-talk WAVs.

No audio is repeated, normalized, resampled or sent to a service. All inference
runs locally on CPU, with cache writes disabled to preserve existing evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dt_mos_fidelity", ROOT / "scripts/compare_double_talk_fidelity.py")
assert SPEC and SPEC.loader
fidelity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fidelity)
ab = fidelity.ab
DNS_SPEC = importlib.util.spec_from_file_location("dt_dnsmos", ROOT / "cpp/tools/ecnr_perceptual.py")
assert DNS_SPEC and DNS_SPEC.loader
dns = importlib.util.module_from_spec(DNS_SPEC)
DNS_SPEC.loader.exec_module(dns)
PREVIOUS = ROOT / "build/double-talk-fidelity-20261112-x105e816"
RESULT_SHA = "5016d0b33e536ab2d9a87fe3bb10429683c304d828cc84084583163eca79ebab"
DELIVERY_SHA = "8ca8e95b02427680fb40032eb5b64e8038db2a26a60c43129a908e649149b14e"
METRICS = ("SIG", "BAK", "OVRL")
ROUTES = ("clean_near_reference", "input_mic", *fidelity.ROUTES)


def score_audio(scorer, audio: np.ndarray) -> dict:
    if audio.shape != (192000,) or not np.isfinite(audio).all():
        raise ValueError("exact original 12-second, finite mono timeline required")
    # Do not call the adapter's permissive short-input doubling path.
    if len(audio) < round(dns.INPUT_LENGTH * dns.RATE):
        raise ValueError("short speech window is unavailable; never repeat it for MOS")
    result = scorer.score_array(audio, dns.RATE, kind="double_talk", use_cache=False)
    if not result["available"] or result["short_repeat"] or result["doubling_count"] != 0:
        raise ValueError("DNSMOS did not score the original, unrepeated timeline")
    if result["num_scored_hops"] != 3 or any(not np.isfinite(result["poly"][key]) for key in METRICS):
        raise ValueError("unexpected DNSMOS hop count or nonfinite prediction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new empty evidence directory")
    args = parser.parse_args()
    ab.resource_check()
    assets = {PREVIOUS / "result.json": RESULT_SHA, PREVIOUS / "OUTPUT-FORMAT-CLARIFICATION.json": DELIVERY_SHA,
              dns.DEFAULT_ASSETS / "sig_bak_ovr.onnx": dns.MODEL_SHA, dns.DEFAULT_ASSETS / "dnsmos_local.py": dns.SCRIPT_SHA}
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError(f"frozen MOS source/model identity mismatch: {path}")
    previous = json.loads((PREVIOUS / "result.json").read_text())
    delivery = json.loads((PREVIOUS / "OUTPUT-FORMAT-CLARIFICATION.json").read_text())
    files = {(r["case_id"], r["route"]): r for r in delivery["files"]}
    selected = [r for r in previous["cases"] if r["measurement_status"] == "measured"
                and r["source_category"] in ("balanced", "near_dominant", "far_dominant")]
    if len(selected) != 3:
        raise ValueError("expected three sustained double-talk scenes")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve historical evidence; use a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    scorer = dns.DNSMOS(cache_dir=output / "disabled-cache")
    if scorer.session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("MOS inference must use the pinned CPU provider")
    controls = {}
    silent = scorer.score_array(np.zeros(192000, dtype=np.float32), 16000, kind="double_talk", use_cache=False)
    if silent["available"] or silent["poly"] is not None:
        raise ValueError("silence was assigned a valid MOS")
    rejected = scorer.score_array(np.ones(192000, dtype=np.float32) * .05, 16000, kind="far_only", use_cache=False)
    if rejected["available"]:
        raise ValueError("far-only role was accepted as near-speech MOS")
    try:
        score_audio(scorer, np.zeros(3 * 16000, dtype=np.float32))
    except ValueError:
        controls["short_window_rejected_without_repetition"] = True
    else:
        raise ValueError("three-second segment was accepted")
    controls.update(silence_rejected=True, far_only_role_rejected=True, passed=True)
    cases = []
    for row in selected:
        ab.resource_check()
        records = {}
        for route in ROUTES:
            if route == "clean_near_reference":
                path = Path(row["inputs"]["clean"]["path"])
                expected_sha = row["inputs"]["clean"]["sha256"]
            else:
                record = files[(row["case_id"], route)]
                path, expected_sha = PREVIOUS / record["path"], record["sha256"]
                if sf.info(path).subtype != "PCM_16":
                    raise ValueError("delivered processed/input audio must be actual PCM16")
            if ab.digest(path) != expected_sha:
                raise ValueError("frozen audio file changed")
            original = ab.read_mono(path)
            audio = fidelity.pcm16(original)
            if route != "clean_near_reference" and not np.array_equal(audio, original):
                raise ValueError("delivered PCM16 audio changed when decoded")
            prediction = score_audio(scorer, audio)
            records[route] = {"source": {"path": str(path), "sha256": expected_sha,
                               "source_subtype": sf.info(path).subtype, "scored_pcm_sha256": ab.pcm_digest(audio),
                               "samples": len(audio), "rate": 16000, "scored_rms_dbfs": float(10 * np.log10(np.mean(audio.astype(np.float64)**2)))},
                              "prediction": prediction}
            print(json.dumps({"case": row["case_id"], "route": route, "scores": prediction["poly"]}), flush=True)
        cases.append({"case_id": row["case_id"], "source_category": row["source_category"], "routes": records})
        ab.write_json(output / f"{row['case_id']}.json", cases[-1])
    summary = {}
    for route in ROUTES:
        values = {key: float(np.mean([r["routes"][route]["prediction"]["poly"][key] for r in cases])) for key in METRICS}
        summary[route] = {"measured_case_count": len(cases), "scores": values}
    for route in summary:
        summary[route]["delta_vs_input"] = {key: summary[route]["scores"][key] - summary["input_mic"]["scores"][key] for key in METRICS}
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError("asset changed during MOS inference")
    result = {"schema": "itu-five-route-double-talk-dnsmos-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": {"runner_sha256": ab.digest(Path(__file__)), "assets": {str(p): h for p, h in assets.items()},
                           "dnsmos_preprocessing": scorer.preprocessing, "preprocessing_sha256": scorer.preprocessing_sha256},
              "case_count": 3, "prediction_count": 21, "inference_hop_count": 63, "case_duration_s": 12,
              "cases": cases, "summary": summary, "controls": controls,
              "excluded": [{"case_id": r["case_id"], "source_category": r["source_category"],
                            "reason": "only 3 s continuous double-talk, below 9.01 s; no repetition" if r["source_category"] == "transition"
                            else "invalid stimulus; no measured simultaneous near/far activity"} for r in previous["cases"] if r not in selected],
              "aecmos": {"available": False, "reason": "no verified local AECMOS runtime/model asset; DNSMOS does not directly measure echo quality"},
              "boundary": "DNSMOS P.835 non-personalized model estimates only, not human MOS, AECMOS, TS103281 G/N/S-MOS or formal ITU compliance; three sustained synthetic DT cases, no repetition, resampling, per-route normalization or audio upload; read jointly with target gain/SI-SDR/ERLE",
              "formal_compliance": False}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
