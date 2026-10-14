#!/usr/bin/env python3
"""Offline DNSMOS P.835 non-personalized scores; no audio upload, no gate changes.

Implementation follows Microsoft's pinned dnsmos_local.py P.835 path only.
P.808 is intentionally not evaluated. These are model estimates, not human MOS.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import scipy
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT/"build/ecnr-v2-20261011/perceptual-assets"
PINNED_COMMIT = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
MODEL_SHA = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
SCRIPT_SHA = "1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b"
ALLOWED_ROLES = frozenset(("near_only", "near_late", "double_talk"))
RATE = 16000
INPUT_LENGTH = 9.01
POLYNOMIALS = {"SIG": [-0.08397278, 1.22083953, 0.0052439],
               "BAK": [-0.13166888, 1.60915514, -0.39604546],
               "OVRL": [-0.06766283, 1.11546468, 0.04602535]}
SCOPE = "Offline non-personalized DNSMOS P.835 model estimate, not human MOS or ITU certification."


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def poly_scores(raw: np.ndarray) -> np.ndarray:
    """Official coefficient order is SIG, BAK, OVRL."""
    return np.stack([np.polyval(POLYNOMIALS[key], raw[..., i])
                     for i, key in enumerate(("SIG", "BAK", "OVRL"))], axis=-1)


def prepare_audio(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray | None, dict]:
    """The official doubling/hop formulas are preserved; empty input terminates."""
    arr = np.asarray(samples, dtype=np.float64)
    meta = {"source_sample_rate": int(sample_rate) if isinstance(sample_rate, (int, np.integer)) else None, "source_samples": int(arr.size),
            "short_repeat": False, "doubling_count": 0, "available": False}
    if arr.ndim != 1:
        return None, {**meta, "reason": "require_explicit_mono_channel"}
    if not len(arr):
        return None, {**meta, "reason": "empty_audio"}
    if not np.all(np.isfinite(arr)):
        return None, {**meta, "reason": "nonfinite_audio"}
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        return None, {**meta, "reason": "invalid_sample_rate"}
    if np.sqrt(np.mean(arr**2)) <= 1e-10 or np.std(arr) <= 1e-10:
        return None, {**meta, "reason": "silence_or_no_ac_excitation"}
    audio = (librosa.resample(arr, orig_sr=sample_rate, target_sr=RATE,
                             res_type="soxr_hq", fix=True, scale=False)
             if sample_rate != RATE else arr.copy())
    original_len = len(audio)
    n = int(INPUT_LENGTH*RATE)
    meta.update({"resampled_original_samples": original_len,
                 "original_duration_s": original_len/RATE,
                 "short_repeat": original_len < n})
    if original_len == 0:
        return None, {**meta, "reason": "empty_after_resample"}
    while len(audio) < n:
        audio = np.append(audio, audio)
        meta["doubling_count"] += 1
        if meta["doubling_count"] > 20:
            return None, {**meta, "reason": "repeat_bound_exceeded"}
    # Preserve official int(floor(seconds)-9.01)+1, including truncation toward zero.
    num_hops = int(np.floor(len(audio)/RATE)-INPUT_LENGTH)+1
    meta.update({"available": True, "effective_samples": len(audio),
                 "num_hops_official": num_hops, "input_samples_per_hop": n,
                 "hop_samples": RATE})
    return audio, meta


class DNSMOS:
    def __init__(self, assets_dir: Path = DEFAULT_ASSETS, cache_dir: Path | None = None):
        self.assets_dir = Path(assets_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.assets_dir.parent/"perceptual-cache"
        manifest_path = self.assets_dir/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("commit") != PINNED_COMMIT:
            raise ValueError("DNSMOS source commit mismatch")
        model = self.assets_dir/"sig_bak_ovr.onnx"
        official = self.assets_dir/"dnsmos_local.py"
        if digest(model.read_bytes()) != MODEL_SHA or digest(official.read_bytes()) != SCRIPT_SHA:
            raise ValueError("DNSMOS cached asset SHA mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.preprocessing = {
            "version": "ecnr-dnsmos-p835-v1",
            "official_commit": PINNED_COMMIT, "official_script_sha256": SCRIPT_SHA,
            "implementation_sha256": digest(Path(__file__).read_bytes()),
            "model_sha256": MODEL_SHA, "personalized": False, "sampling_rate": RATE,
            "input_length_s": INPUT_LENGTH, "hop_s": 1,
            "short_policy": "official repeated doubling until >=9.01s",
            "num_hops": "int(floor(effective_samples/16000)-9.01)+1",
            "resample": "librosa soxr_hq, fix=True, scale=False; no level normalization",
            "poly": POLYNOMIALS, "aggregate": "mean raw; mean polynomial per hop",
            "librosa_version": librosa.__version__, "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "soundfile_version": sf.__version__, "cpu_threads": 1,
            "provider": "CPUExecutionProvider", "p808_evaluated": False}
        self.preprocessing_sha256 = digest(json.dumps(self.preprocessing, sort_keys=True).encode())

    def score_array(self, samples: np.ndarray, sample_rate: int, *,
                    kind: str, use_cache: bool = True) -> dict[str, Any]:
        base = {"scope": SCOPE, "family": "DNSMOS P.835", "personalized": False,
                "role": kind, "model_sha256": MODEL_SHA,
                "preprocessing_sha256": self.preprocessing_sha256,
                "official_commit": PINNED_COMMIT, "p808_evaluated": False}
        if kind not in ALLOWED_ROLES:
            return {**base, "available": False, "reason": "role_excluded_from_speech_scoring",
                    "raw": None, "poly": None, "cache_hit": False}
        audio, prep = prepare_audio(samples, sample_rate)
        if audio is None:
            return {**base, **prep, "raw": None, "poly": None, "cache_hit": False}
        sample_rate = int(sample_rate)
        waveform = np.ascontiguousarray(samples, dtype="<f8")
        audio_sha = digest(str(waveform.shape).encode()+waveform.tobytes())
        key_fields = {"audio_sha256": audio_sha, "source_sample_rate": sample_rate,
                      "role": kind, "model_sha256": MODEL_SHA,
                      "preprocessing_sha256": self.preprocessing_sha256}
        cache_key = digest(json.dumps(key_fields, sort_keys=True).encode())
        path = self.cache_dir/(cache_key+".json")
        if use_cache and path.exists():
            cached = json.loads(path.read_text())
            if cached.get("cache_key") != cache_key or cached.get("cache_identity") != key_fields:
                raise ValueError("perceptual cache identity mismatch")
            return {**cached, "cache_hit": True}
        raw, starts = [], []
        n = prep["input_samples_per_hop"]
        for idx in range(prep["num_hops_official"]):
            part = audio[int(idx*RATE):int((idx+INPUT_LENGTH)*RATE)]
            if len(part) < n:
                continue
            prediction = np.asarray(self.session.run(
                None, {self.input_name: part.astype(np.float32)[np.newaxis, :]})[0][0],
                dtype=np.float64)
            if prediction.shape != (3,) or not np.all(np.isfinite(prediction)):
                return {**base, **prep, "available": False, "reason": "invalid_model_output",
                        "raw": None, "poly": None, "cache_hit": False}
            raw.append(prediction)
            starts.append(idx)
        if not raw:
            return {**base, **prep, "available": False, "reason": "no_complete_inference_hop",
                    "raw": None, "poly": None, "cache_hit": False}
        raws = np.asarray(raw)
        polys = poly_scores(raws)
        result = {**base, **prep, "audio_sha256": audio_sha,
                  "cache_key": cache_key, "cache_identity": key_fields,
                  "cache_hit": False, "num_scored_hops": len(raw),
                  "raw": dict(zip(("SIG", "BAK", "OVRL"), np.mean(raws, axis=0).tolist())),
                  "poly": dict(zip(("SIG", "BAK", "OVRL"), np.mean(polys, axis=0).tolist())),
                  "hops": [{"start_s_in_repeated_audio": start,
                            "raw": dict(zip(("SIG", "BAK", "OVRL"), r.tolist())),
                            "poly": dict(zip(("SIG", "BAK", "OVRL"), p.tolist()))}
                           for start, r, p in zip(starts, raws, polys)]}
        if use_cache:
            save_json(path, result)
        return result

    def score_file(self, audio_path: Path, *, kind: str,
                   window_s: tuple[float, float] | None = None, channel: int = 0,
                   use_cache: bool = True) -> dict:
        path = Path(audio_path)
        payload = path.read_bytes()
        try:
            data, rate = sf.read(io.BytesIO(payload), always_2d=True, dtype="float64")
        except sf.LibsndfileError as error:
            return {"scope": SCOPE, "family": "DNSMOS P.835", "role": kind,
                    "personalized": False, "available": False,
                    "reason": "empty_audio_file" if not payload else "undecodable_audio_file",
                    "raw": None, "poly": None, "cache_hit": False,
                    "model_sha256": MODEL_SHA, "preprocessing_sha256": self.preprocessing_sha256,
                    "official_commit": PINNED_COMMIT, "p808_evaluated": False,
                    "source_path": str(path.resolve()), "source_file_sha256": digest(payload),
                    "channel": channel, "decoder_error": str(error)}
        if channel < 0 or channel >= data.shape[1]:
            raise ValueError("explicit channel index out of range")
        samples = data[:, channel]
        if window_s is not None:
            a, b = window_s
            if not (0 <= a < b <= len(samples)/rate+1e-9):
                raise ValueError("window is outside audio duration")
            samples = samples[int(round(a*rate)):int(round(b*rate))]
        return {**self.score_array(samples, rate, kind=kind, use_cache=use_cache),
                "source_path": str(path.resolve()), "source_file_sha256": digest(payload),
                "channel": channel, "channel_policy": "selected channel; no downmix",
                "window_s": list(window_s) if window_s is not None else None}

    def score_windows(self, folder: Path, *, use_cache: bool = True) -> dict:
        from ecnr_window_audit import load_stream_audio
        loaded = load_stream_audio(folder, include_clean=False)
        records, excluded = [], []
        rate = loaded["sample_rate"]
        for phase in loaded["identity"]["segments"]:
            for kind, limits in phase["windows"].items():
                if kind not in ALLOWED_ROLES:
                    excluded.append({"label": phase["label"], "kind": kind,
                                     "window_s": limits, "reason": "not_a_near_speech_window"})
                    continue
                start, end = map(float, limits)
                if not 0 <= start < end <= len(loaded["before"])/rate+1e-9:
                    raise ValueError("window is outside aligned audio duration")
                a, b = int(round(start*rate)), int(round(end*rate))
                before = self.score_array(loaded["before"][a:b], rate, kind=kind, use_cache=use_cache)
                after = self.score_array(loaded["after"][a:b], rate, kind=kind, use_cache=use_cache)
                delta = None
                if before["available"] and after["available"]:
                    delta = {k: after["poly"][k]-before["poly"][k] for k in ("SIG", "BAK", "OVRL")}
                records.append({"label": phase["label"], "kind": kind, "window_s": limits,
                                "before": before, "after": after, "delta_poly": delta})
        return {"schema_version": 1, "scope": SCOPE, "folder": str(Path(folder).resolve()),
                "preprocessing": self.preprocessing, "preprocessing_sha256": self.preprocessing_sha256,
                "source_identity": loaded["source_identity"], "windows": records,
                "excluded_windows": excluded,
                "available_windows": sum(r["after"]["available"] for r in records),
                "short_repeat_windows": sum(r["after"].get("short_repeat", False) for r in records)}


def score_file(audio_path: Path, *, kind: str, assets_dir: Path = DEFAULT_ASSETS,
               cache_dir: Path | None = None, **kwargs) -> dict:
    return DNSMOS(assets_dir, cache_dir).score_file(audio_path, kind=kind, **kwargs)


def score_windows(folder: Path, *, assets_dir: Path = DEFAULT_ASSETS,
                  cache_dir: Path | None = None, **kwargs) -> dict:
    return DNSMOS(assets_dir, cache_dir).score_windows(folder, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", type=Path)
    mode.add_argument("--folder", type=Path)
    parser.add_argument("--kind", choices=sorted(ALLOWED_ROLES), default="near_only")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    args = parser.parse_args()
    scorer = DNSMOS(args.assets, args.cache)
    if args.folder:
        result = scorer.score_windows(args.folder)
    else:
        window = None
        if args.start_s is not None or args.end_s is not None:
            if args.start_s is None or args.end_s is None:
                parser.error("start-s and end-s must be provided together")
            window = (args.start_s, args.end_s)
        result = scorer.score_file(args.file, kind=args.kind, channel=args.channel, window_s=window)
    save_json(args.output, result)
    print(json.dumps({"output": str(args.output), "available": result.get("available"),
                      "available_windows": result.get("available_windows"),
                      "short_repeat_windows": result.get("short_repeat_windows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
