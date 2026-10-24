"""Read-only browsing of the verified paired NS failures and their actual audio."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from pathlib import Path
from threading import Lock
import json
import os
import tempfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
REPLAYS = PROJECT / "build/ns-omlsa-supergaussian-20261018/common-failures-20261018/frames"
CACHE = PROJECT / "build/ns-failure-audio-cache"
VARIANTS = {"omlsa_imcra_v2_balanced": "omlsa", "super_gaussian_balanced": "super_gaussian"}
router = APIRouter(prefix="/api/ns-failures", tags=["NS failure audio"])
_cache_lock = Lock()


def _within_build(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((PROJECT / "build").resolve()):
        raise ValueError("Audio evidence path is outside the build directory")
    return resolved


def _failure_metric(label: str, windows: dict) -> dict:
    window, _, metric = label.partition(":")
    frozen = windows.get(window, {}).get("frozen_metrics") or {}
    before, after = frozen.get("before", {}), frozen.get("after", {})
    value = input_value = None
    unit = ""
    requirement = "参见冻结报告"
    if metric == "attenuation":
        value, unit, requirement = frozen.get("total_attenuation_db"), "dB", "≥ 6 dB"
    elif metric == "projection_gain":
        value, input_value = after.get("projection_gain_db"), before.get("projection_gain_db")
        unit, requirement = "dB", "−3 至 +18 dB"
    elif metric == "stoi":
        value, requirement = frozen.get("stoi_improvement"), "相对输入变化 ≥ −0.02"
    elif metric in ("sisdr", "raw_sisdr"):
        value, input_value, unit = after.get("sisdr_db"), before.get("sisdr_db"), "dB"
        threshold = min(input_value, 20) - 1 if input_value is not None else None
        requirement = f"≥ {threshold:.3f} dB" if threshold is not None else "参见冻结报告"
    return {"label": label, "window": window, "metric": metric, "value": value,
            "input_value": input_value, "unit": unit, "requirement": requirement}


@lru_cache(maxsize=1)
def catalog() -> tuple[dict, dict]:
    report = json.loads((REPLAYS / "report.json").read_text())
    if report.get("state") != "completed" or not report.get("dependencies_verified"):
        raise ValueError("Failure replay is not verified and complete")
    pairs: dict[tuple, dict] = {}
    rows = [json.loads(line) for line in (REPLAYS / "results.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        role = VARIANTS[row["variant"]]
        key = (row["scope"], row["original_run"], row["scene"]["id"])
        pair = pairs.setdefault(key, {})
        if role in pair or not row["replay_hash_matches"]:
            raise ValueError("Duplicate or unverified failure evidence")
        pair[role] = row
    if not pairs or report["replays"] != len(rows) or report["exact_output_hash_matches"] != len(rows):
        raise ValueError("Failure replay count does not match the report")
    items, sources = [], {}
    order = {"steady": 0, "transient": 1, "cold": 2, "clean": 3}
    for key in sorted(pairs, key=lambda k: (order[k[0]], k[1], k[2])):
        pair = pairs[key]
        if set(pair) != {"omlsa", "super_gaussian"}:
            raise ValueError("Missing paired backend output")
        a, b = pair["omlsa"], pair["super_gaussian"]
        for field in ("scene", "input_crop_samples", "input_pcm16_sha256"):
            if a[field] != b[field]:
                raise ValueError("Paired inputs differ")
        if not a["failures"] or not b["failures"]:
            raise ValueError("Sample is not a common failure")
        scene = a["scene"]
        origin = _within_build(PROJECT / "build" / a["original_run"] / "cases" / scene["id"])
        crop, rate = a["input_crop_samples"], scene["sample_rate"]
        paths = {"input": {"path": origin / "before.wav", "offset": crop,
                           "pcm16_sha256": a["input_pcm16_sha256"]},
                 "clean": {"path": origin / "clean.wav", "offset": crop}}
        for role, row in pair.items():
            paths[role] = {"path": _within_build(PROJECT / row["folder"] / "result.after.wav"),
                           "offset": row["delay_samples"], "sha256": row["output_sha256"]}
        frames = sf.info(paths["input"]["path"]).frames - crop
        if frames <= 0:
            raise ValueError("Empty failure audio")
        for source in paths.values():
            source["path"] = _within_build(source["path"])
            info = sf.info(source["path"])
            if info.samplerate != rate or info.channels != 1 or info.frames - source["offset"] != frames:
                raise ValueError("Audio rate, channels or aligned length mismatch")
        duration = frames / rate
        window_names = sorted({label.partition(":")[0] for row in pair.values() for label in row["failures"]})
        windows = [{"id": name, "start_s": scene["windows"][name][0],
                    "end_s": scene["windows"][name][1]} for name in window_names if name in scene["windows"]]
        identity = json.dumps([key, a["input_pcm16_sha256"], a["output_sha256"], b["output_sha256"]])
        identifier = hashlib.sha256(identity.encode()).hexdigest()[:24]
        speech_failure = any(not label.endswith(":attenuation") for row in pair.values() for label in row["failures"])
        items.append({"id": identifier, "scope": key[0], "original_run": key[1], "scene_id": key[2],
                      "noise": scene["noise"], "sample_rate": rate, "snr_db": scene["snr_db"],
                      "split": scene["split"], "speaker": scene.get("near_speaker", ""),
                      "duration_s": duration, "speech_failure": speech_failure, "windows": windows,
                      "input_crop_samples": crop,
                      "backends": {role: {"variant": row["variant"], "delay_samples": row["delay_samples"],
                                            "output_sha256": row["output_sha256"],
                                            "failures": [_failure_metric(label, row["windows"]) for label in row["failures"]]}
                                   for role, row in pair.items()},
                      "audio": {role: f"/api/ns-failures/{identifier}/audio/{role}" for role in paths}})
        sources[identifier] = {"paths": paths, "rate": rate, "frames": frames,
                               "windows": {w["id"]: (w["start_s"], w["end_s"]) for w in windows}}
    actual_counts = dict(Counter(item["scope"] for item in items))
    if any(actual_counts.get(scope, 0) != count for scope, count in report["common_case_counts"].items()):
        raise ValueError("Common failure counts differ from the verified report")
    return {"total": len(items), "scope_counts": actual_counts, "items": items,
            "comparison": "OM-LSA v2 balanced / Super-Gaussian balanced", "frozen": True,
            "audio_policy": "原生采样率、原幅度；输入与评测相同的 PCM16 量化，输出只扣除声明固定延时。冷启动已应用原裁剪。"}, sources


def _load_catalog() -> tuple[dict, dict]:
    try:
        return catalog()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise HTTPException(503, "共同失败音频证据不可用，请检查冻结回放文件") from error


@router.get("")
def list_failures():
    return _load_catalog()[0]


def _prepare_audio(identifier: str, sample: dict, role: str, window: str) -> Path:
    source = sample["paths"][role]
    rate, frames = sample["rate"], sample["frames"]
    if window == "full":
        start, stop = 0, frames
    else:
        lo, hi = sample["windows"][window]
        start, stop = max(0, round(lo * rate)), min(frames, round(hi * rate))
    if stop <= start:
        raise ValueError("Empty audio window")
    # Only fixed evidence IDs, roles and named windows reach this cache. No
    # arbitrary filesystem path or user-supplied time span is accepted.
    target = CACHE / identifier / f"{role}-{window}.wav"
    with _cache_lock:
        if target.is_file():
            return target
        if "sha256" in source and hashlib.sha256(source["path"].read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Frozen output hash differs")
        if role == "input":
            values, _ = sf.read(source["path"], start=source["offset"], dtype="float64")
            pcm = np.clip(np.rint(values * 32768), -32768, 32767).astype("<i2")
            if hashlib.sha256(pcm.tobytes()).hexdigest() != source["pcm16_sha256"]:
                raise ValueError("Frozen input PCM differs")
            values = pcm[start:stop]
        else:
            values, _ = sf.read(source["path"], start=source["offset"] + start,
                                frames=stop - start, dtype="int16" if role != "clean" else "float64")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=".wav", dir=target.parent)
        os.close(fd)
        try:
            sf.write(name, values, rate, format="WAV", subtype="PCM_16")
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
    return target


@router.get("/{identifier}/audio/{role}")
def failure_audio(identifier: str, role: str, window: str = "full", download: bool = False):
    _, sources = _load_catalog()
    sample = sources.get(identifier)
    if sample is None or role not in sample["paths"]:
        raise HTTPException(404, "失败样本或音频角色不存在")
    if window != "full" and window not in sample["windows"]:
        raise HTTPException(404, "失败窗口不存在")
    try:
        path = _prepare_audio(identifier, sample, role, window)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(503, "音频证据读取或校验失败") from error
    return FileResponse(path, media_type="audio/wav", filename=f"{identifier}-{role}-{window}.wav",
                        content_disposition_type="attachment" if download else "inline")
