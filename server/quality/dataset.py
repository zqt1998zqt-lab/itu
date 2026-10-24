"""Quality of source material and targets, independent of enhanced audio."""
from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .metrics import physical, sha


def speech_windows(intervals, samples, rate):
    """Keep true interval boundaries; never join, extend or repeat short speech."""
    bounds = [[0, samples / rate]] if intervals is None else intervals
    result = []
    last = 0
    for lo, hi in bounds:
        a, b = round(float(lo) * rate), round(float(hi) * rate)
        if not 0 <= a < b <= samples or a < last:
            raise ValueError("invalid or overlapping speech schedule")
        for start in range(a, b, rate * 10):
            result.append((start, min(start + rate * 10, b)))
        last = b
    return result


def spectrum(x, rate):
    centered = x - x.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    weight = np.full(len(power), 2.)
    weight[0] = 1.
    if len(x) % 2 == 0:
        weight[-1] = 1.
    power *= weight
    f = np.fft.rfftfreq(len(x), 1 / rate)
    total = float(power.sum())
    return dict(ac_below_60hz_percent=float(power[(f > 0) & (f < 60)].sum() / total * 100) if total else None,
                ac_above_8khz_percent=float(power[f > 8000].sum() / total * 100) if total else None,
                dc_dbfs=float(20 * np.log10(max(abs(float(x.mean())), 1e-10))))


def inspect_asset(asset, evaluators):
    """The caller supplies frozen byte identity and the actual speech schedule."""
    path = Path(asset["path"])
    payload = path.read_bytes()
    if sha(payload) != asset["sha256"]:
        raise ValueError("dataset file SHA mismatch: " + str(path))
    data, rate = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    if rate != asset["sample_rate"] or not len(data) or not np.isfinite(data).all():
        raise ValueError("invalid dataset PCM geometry or finite samples")
    if asset.get("channels") is not None and data.shape[1] != asset["channels"]:
        raise ValueError("dataset channel metadata mismatch")
    if asset.get("samples") is not None and len(data) != asset["samples"]:
        raise ValueError("dataset sample-count metadata mismatch")
    # This canonical content identity equates +0 and -0 without changing model input.
    canonical = np.array(data, dtype="<f8", copy=True)
    canonical[canonical == 0] = 0.
    pcm_sha = sha(json.dumps(dict(rate=rate, shape=data.shape), sort_keys=True).encode() + canonical.tobytes())
    speech = asset["role"] != "noise_source"
    windows = speech_windows(asset.get("speech_intervals"), len(data), rate) if speech else []
    channels = []
    for channel in range(data.shape[1]):
        x = data[:, channel]
        stats = physical(x)
        stats.update(spectrum(x, rate))
        selected = np.concatenate([x[a:b] for a,b in windows]) if windows else np.empty(0)
        scheduled = {**physical(selected), **spectrum(selected, rate)} if selected.size else None
        flags = []
        if stats["clipping_samples"]:
            flags.append("rail_samples_review")
        if speech and scheduled:
            if scheduled["dc_dbfs"] > -50:
                flags.append("scheduled_dc_above_minus50_dbfs")
            if (scheduled["ac_below_60hz_percent"] or 0) > 5:
                flags.append("scheduled_sub60hz_above_5_percent")
            if scheduled["ac_below_60hz_percent"] is None:
                flags.append("scheduled_target_has_no_ac_excitation")
        records = []
        for start, stop in windows:
            scores = {e.name: e.score(x[start:stop], rate) for e in evaluators}
            records.append(dict(start_sample=start, stop_sample=stop, start_s=start/rate, end_s=stop/rate,
                                physical=physical(x[start:stop]), perceptual=scores))
        channels.append(dict(channel=channel, physical=stats, scheduled=scheduled,
                             review_flags=flags, windows=records,
                             perceptual_excluded_reason="noise_is_not_speech" if not speech else None))
    return dict(asset_id=asset["asset_id"], role=asset["role"], source_path=str(path), sha256=asset["sha256"],
                sample_rate=rate, samples=len(data), seconds=len(data)/rate, channels=channels,
                pcm_sha256=pcm_sha, state="completed",
                policy=dict(resampled=False, normalized=False, external_vad=False,
                            silence_repeated=False, enhanced_audio_evaluated=False))


def aggregate_assets(rows):
    counts, reasons, coverage = Counter(), Counter(), {}
    signatures = {}
    for row in rows:
        counts[(row["role"], row["sample_rate"])] += 1
        if row.get("state") != "completed":
            continue
        signatures.setdefault(row["pcm_sha256"], []).append(row["asset_id"])
        for channel in row["channels"]:
            reasons.update(channel["review_flags"])
            for window in channel["windows"]:
                for name, score in window["perceptual"].items():
                    key = f"{row['role']}:{row['sample_rate']}:{name}"
                    count = coverage.setdefault(key, dict(requested=0, available=0, unavailable_reasons=Counter()))
                    count["requested"] += 1
                    count["available"] += bool(score["available"])
                    if not score["available"]:
                        count["unavailable_reasons"][score["reason"]] += 1
    return dict(role_rate_counts={f"{role}:{rate}": n for (role,rate),n in sorted(counts.items())},
                review_flag_counts=dict(reasons), coverage=coverage,
                exact_pcm_duplicate_groups=[v for v in signatures.values() if len(v) > 1],
                unique_pcm_files=len(signatures))
