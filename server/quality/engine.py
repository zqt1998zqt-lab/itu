"""Paired multi-algorithm quality evidence, without averaging unrelated MOS scales."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np

from .evaluators import Evaluator
from .metrics import Audio, attenuation, physical, reference_metrics, sha

SCHEMA_VERSION = 1
SCOPE = "Diagnostic model predictions and measured PCM metrics; not human MOS, ITU certification or DSP acceptance gates."


def consensus(deltas: dict[str, float], tolerance: float) -> str:
    if len(deltas) < 2:
        return "insufficient_models"
    positive = any(v > tolerance for v in deltas.values())
    negative = any(v < -tolerance for v in deltas.values())
    if positive and negative:
        return "disagreement"
    if positive:
        return "improvement_without_model_conflict"
    if negative:
        return "degradation_without_model_conflict"
    return "within_diagnostic_deadband"


def default_windows(seconds: float) -> list[dict]:
    # Callers with known speech/noise schedules should always provide them.
    return [dict(name=f"speech-{i:03d}", kind="speech", start_s=float(a),
                 end_s=min(float(a + 10), seconds))
            for i, a in enumerate(range(0, int(np.ceil(seconds)), 10))]


def evaluate(case_id: str, before: Audio, outputs: dict[str, Audio], *, clean: Audio | None,
             windows: list[dict] | None, evaluators: list[Evaluator], mos_deadband: float = .1) -> dict:
    if not outputs or len(outputs) > 16 or any(not name or name in {"before", "clean"} for name in outputs):
        raise ValueError("require 1..16 distinctly named output algorithms")
    if not evaluators or len({s.name for s in evaluators}) != len(evaluators):
        raise ValueError("require distinct quality evaluators")
    if not np.isfinite(mos_deadband) or mos_deadband < 0:
        raise ValueError("invalid diagnostic MOS deadband")
    assets = {"before": before, **({"clean": clean} if clean is not None else {}), **outputs}
    for name, asset in assets.items():
        if asset.rate != before.rate or asset.samples.shape != before.samples.shape:
            raise ValueError(f"unaligned rate or length for {name}; declare exact offsets, do not truncate")
        if asset.samples.ndim != 1 or not len(asset.samples) or not np.isfinite(asset.samples).all():
            raise ValueError("require finite nonempty mono PCM")
    seconds = len(before.samples) / before.rate
    automatic = windows is None
    windows = default_windows(seconds) if windows is None else windows
    if not windows or len(windows) > 64:
        raise ValueError("require 1..64 declared windows")
    seen = set()
    for w in windows:
        if set(w) != {"name", "kind", "start_s", "end_s"} or w["kind"] not in {"speech", "noise"}:
            raise ValueError("invalid window contract")
        if not isinstance(w["name"], str) or not w["name"] or w["name"] in seen:
            raise ValueError("duplicate or empty window name")
        seen.add(w["name"])
        lo, hi = float(w["start_s"]), float(w["end_s"])
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= seconds + 1e-9:
            raise ValueError("window outside aligned audio duration")
        if hi - lo > 15 + 1e-9:
            raise ValueError("model windows must be <=15 seconds; split longer recordings")
        if round(lo * before.rate) == round(hi * before.rate):
            raise ValueError("empty sample window")
    records, reviews, pairwise = [], [], []
    for w in windows:
        a, b = round(float(w["start_s"]) * before.rate), round(float(w["end_s"]) * before.rate)
        sliced = {name: asset.samples[a:b] for name, asset in assets.items()}
        scored = {}
        for name, x in sliced.items():
            model_scores = {s.name: s.score(x, before.rate) if w["kind"] == "speech" else
                            dict(available=False, reason="noise_only_excluded_from_speech_models") for s in evaluators}
            reference = reference_metrics(sliced["clean"], x, before.rate) if clean is not None and w["kind"] == "speech" else None
            scored[name] = dict(physical=physical(x), perceptual=model_scores, reference=reference)
        comparisons = {}
        for name in outputs:
            delta, details = {}, {}
            for s in evaluators:
                inp, out = scored["before"]["perceptual"][s.name], scored[name]["perceptual"][s.name]
                if inp["available"] and out["available"]:
                    details[s.name] = {k: out["values"][k] - inp["values"][k] for k in out["values"]}
                    delta[s.name] = details[s.name][s.primary]
            state = consensus(delta, mos_deadband) if w["kind"] == "speech" else "not_applicable_noise_only"
            flags = []
            if state == "disagreement":
                flags.append("perceptual_model_disagreement")
            dn = details.get("dnsmos", {})
            if dn.get("BAK", 0) > mos_deadband and dn.get("SIG", 0) < -mos_deadband:
                flags.append("background_improved_but_speech_quality_degraded")
            ref_delta = {}
            qa, qb = scored[name]["reference"], scored["before"]["reference"]
            if qa and qb and qa["available"] and qb["available"]:
                ref_delta = {k: qa[k] - qb[k] for k in ("sisdr_db", "stoi", "projection_gain_db")
                             if qa.get(k) is not None and qb.get(k) is not None}
                if any(v > mos_deadband for v in delta.values()) and (
                    ref_delta.get("stoi", 0) < -.02 or ref_delta.get("sisdr_db", 0) < -1.):
                    flags.append("perceptual_improvement_conflicts_with_reference_fidelity")
            comparisons[name] = dict(primary_delta=delta, all_score_deltas=details, consensus=state,
                                     reference_delta=ref_delta, review_flags=flags,
                                     noise_reduction=attenuation(sliced["before"], sliced[name]) if w["kind"] == "noise" else None)
            if flags:
                reviews.append(dict(case_id=case_id, window=w["name"], algorithm=name, flags=flags))
        if w["kind"] == "speech":
            for left, right in combinations(outputs, 2):
                delta = {}
                for s in evaluators:
                    l, r = scored[left]["perceptual"][s.name], scored[right]["perceptual"][s.name]
                    if l["available"] and r["available"]:
                        delta[s.name] = r["values"][s.primary] - l["values"][s.primary]
                pairwise.append(dict(window=w["name"], left=left, right=right,
                                     right_minus_left=delta, consensus=consensus(delta, mos_deadband)))
        records.append(dict(**w, samples=[a, b], scores=scored, comparisons=comparisons))
    coverage = {s.name:dict(available=sum(r["scores"][name]["perceptual"][s.name]["available"]
                                        for r in records if r["kind"] == "speech" for name in assets),
                           requested=sum(r["kind"] == "speech" for r in records) * len(assets)) for s in evaluators}
    cross_checked = sum(c["consensus"] != "insufficient_models" for r in records if r["kind"] == "speech"
                        for c in r["comparisons"].values())
    requested = sum(r["kind"] == "speech" for r in records) * len(outputs)
    summaries = {}
    for name in outputs:
        summaries[name] = dict(
            perceptual_delta_median={s.name:float(np.median(values)) if values else None for s in evaluators
                for values in [[r["comparisons"][name]["primary_delta"][s.name] for r in records
                                if s.name in r["comparisons"][name]["primary_delta"]]]},
            consensus_counts=dict(Counter(r["comparisons"][name]["consensus"] for r in records)),
            review_windows=sum(bool(r["comparisons"][name]["review_flags"]) for r in records))
    report = dict(schema_version=SCHEMA_VERSION, case_id=case_id, scope=SCOPE,
                  engine_identity=dict(source_sha256={p.name:sha(p.read_bytes()) for p in
                      (Path(__file__), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("evaluators.py"))},
                      python=platform.python_version(), machine=platform.machine(), numpy=np.__version__),
                  state="completed" if requested and cross_checked == requested and all(c["available"] == c["requested"] for c in coverage.values()) else "completed_partial",
                  sample_rate=before.rate, aligned_samples=len(before.samples),
                  input_identities={name: asset.identity for name, asset in assets.items()},
                  evaluator_identities={s.name:s.identity for s in evaluators},
                  policy=dict(native_audio_only=True, level_normalization=False, external_vad=False,
                              automatic_windows_assume_speech=automatic, lag_search=False,
                              short_window_repetition=False, model_scores_averaged=False,
                              diagnostic_mos_deadband=mos_deadband, diagnostic_stoi_drop=-.02,
                              diagnostic_sisdr_drop_db=-1., thresholds_are_itu_requirements=False),
                  windows=records, pairwise_algorithm_comparisons=pairwise, review_queue=reviews,
                  coverage=coverage, cross_checked_comparisons=cross_checked,
                  requested_speech_comparisons=requested, algorithms=summaries)
    report["evidence_sha256"] = sha(json.dumps(report, sort_keys=True, allow_nan=False).encode())
    return report
