#!/usr/bin/env python3
"""Controlled DNSMOS diagnostics: oracle echo removal, noise, and level only.

These counterfactuals explain model sensitivity. They are not new AEC candidates
or replacements for the original unnormalized benchmark scores.
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
SPEC = importlib.util.spec_from_file_location("diagnostic_mos_support", ROOT / "scripts/compare_double_talk_dnsmos.py")
assert SPEC and SPEC.loader
mos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mos)
ab, dns = mos.ab, mos.dns
SOURCE = ROOT / "build/double-talk-dnsmos-20261112-660gs9ib/result.json"
SOURCE_SHA = "0a44ab2a6ca3d46d56069179d3dc5f66945d14c872125ffea0caa774957504b9"


def rms_db(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(audio.astype(np.float64)**2)))


def set_rms(audio: np.ndarray, target_dbfs: float = -26) -> tuple[np.ndarray, float]:
    gain_db = target_dbfs - rms_db(audio)
    result = audio.astype(np.float64) * 10**(gain_db / 20)
    if np.max(np.abs(result)) >= 32767 / 32768:
        raise ValueError("controlled level test would clip; do not apply limiter")
    return mos.fidelity.pcm16(result), gain_db


def write_pcm(path: Path, audio: np.ndarray) -> dict:
    with path.open("xb") as stream:
        sf.write(stream, audio, 16000, format="WAV", subtype="PCM_16")
    decoded, rate = sf.read(path, dtype="float32")
    if rate != 16000 or not np.array_equal(audio, decoded):
        raise ValueError("diagnostic PCM delivery changed samples")
    return {"path": path.name, "sha256": ab.digest(path), "pcm_sha256": ab.pcm_digest(audio), "samples": len(audio)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ab.resource_check()
    if ab.digest(SOURCE) != SOURCE_SHA:
        raise ValueError("original MOS report identity changed")
    prior = json.loads(SOURCE.read_text())
    frozen = json.loads(mos.fidelity.paired.BASELINE.read_text())
    sources = {r["case_id"]: r for r in frozen["cases"]}
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve historical evidence; choose a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    scorer = dns.DNSMOS(cache_dir=output / "disabled-cache")
    cases = []
    replay_checks = 0
    for case in prior["cases"]:
        ab.resource_check()
        work = output / case["case_id"]
        work.mkdir(exist_ok=False)
        row = sources[case["case_id"]]
        source_folder = mos.fidelity.paired.BASELINE.parent / "double_talk" / case["case_id"]
        components = {}
        for role in ("clean", "echo", "noise"):
            path = source_folder / f"{role}.wav"
            if ab.digest(path) != row["scene"]["artifacts"][role]["sha256"]:
                raise ValueError("frozen component changed")
            components[role] = ab.read_mono(path)
        clean, echo, noise = [components[k] for k in ("clean", "echo", "noise")]
        variants = {"clean": mos.fidelity.pcm16(clean),
                    "oracle_remove_echo_keep_noise": mos.fidelity.pcm16(clean.astype(np.float64) + noise),
                    "near_plus_echo_without_noise": mos.fidelity.pcm16(clean.astype(np.float64) + echo)}
        for route in ("input_mic", "webrtc_aec3", "localvqe_full"):
            record = case["routes"][route]["source"]
            path = Path(record["path"])
            if ab.digest(path) != record["sha256"]:
                raise ValueError("frozen processed output changed")
            variants[route] = ab.read_mono(path)
        gain_changes = {}
        for route in ("clean", "oracle_remove_echo_keep_noise", "input_mic", "webrtc_aec3", "localvqe_full"):
            audio, gain_db = set_rms(variants[route])
            variants[route + "_rms_minus26_diagnostic"] = audio
            gain_changes[route + "_rms_minus26_diagnostic"] = gain_db
        records = {}
        for name, audio in variants.items():
            artifact = write_pcm(work / f"{name}.wav", audio)
            score = mos.score_audio(scorer, audio)
            records[name] = {"artifact": artifact, "rms_dbfs": rms_db(audio),
                             "gain_change_db": gain_changes.get(name, 0.0), "peak": float(np.max(np.abs(audio))),
                             "scores": score["poly"], "hops": score["hops"]}
            previous_key = "clean_near_reference" if name == "clean" else name
            if previous_key in case["routes"]:
                expected = case["routes"][previous_key]["prediction"]["poly"]
                if score["poly"] != expected:
                    raise ValueError("unchanged audio no longer reproduces original MOS")
                replay_checks += 1
        facts = {"near_noise_snr_db": rms_db(clean) - rms_db(noise),
                 "near_echo_ser_db": rms_db(clean) - rms_db(echo),
                 "clean_rms_dbfs": rms_db(clean), "noise_rms_dbfs": rms_db(noise), "echo_rms_dbfs": rms_db(echo)}
        cases.append({"case_id": case["case_id"], "category": case["source_category"], "physical": facts, "variants": records})
        ab.write_json(work / "case.json", cases[-1])
        print(json.dumps({"case": case["case_id"], "scores": {k: v["scores"]["OVRL"] for k, v in records.items()}}), flush=True)
    summary = {name: {key: float(np.mean([c["variants"][name]["scores"][key] for c in cases])) for key in mos.METRICS}
               for name in cases[0]["variants"]}
    if ab.digest(SOURCE) != SOURCE_SHA:
        raise ValueError("source changed during diagnosis")
    result = {"schema": "itu-double-talk-mos-controlled-diagnosis-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": {"source": str(SOURCE), "source_sha256": SOURCE_SHA, "runner_sha256": ab.digest(Path(__file__)),
                           "model_sha256": dns.MODEL_SHA, "preprocessing": scorer.preprocessing},
              "case_count": 3, "predictions": 33, "inference_hops": 99, "replayed_original_predictions_equal": replay_checks,
              "cases": cases, "summary": summary,
              "boundary": "diagnostic isolated mixtures and global RMS gain changes only; not alternative AEC outputs, not corrected benchmark scores; no causal claim of independent additive MOS contributions or subjective quality",
              "formal_compliance": False}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
