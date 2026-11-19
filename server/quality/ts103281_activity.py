"""P.56 Method B activity level and TS 103 281 activity classification.

Source and license
------------------
The Method B arithmetic below was independently translated from the official
ITU-T Software Tool Library implementation:

* https://raw.githubusercontent.com/openitu/STL/dev/src/sv56/sv-p56.c
* https://raw.githubusercontent.com/openitu/STL/dev/src/sv56/sv-p56.h
* https://raw.githubusercontent.com/openitu/STL/dev/LICENSE.md

The downloaded source is pinned at STL commit
``cbacae95828179f1238967608e1b91cd515a0261``:

* sv-p56.c SHA256 ``f56c68e3c416e6464f6c376d79d50267f6ff0c803c17253498e6d871cbf3d398``
* sv-p56.h SHA256 ``ff1f104efc794845ad24a88cccbab73f315422130097bea2e96a9ac225df09eb``

The STL source carries the ITU-T General Public License; the retained license
text is in ``build/model-a-20261105/activity-sources/LICENSE.md``.  This module
does not use or copy the separately licensed ``g160app2.c``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter


SMOOTHING_SECONDS = 0.03
HANGOVER_SECONDS = 0.20
MARGIN_DB = 15.9
INTERPOLATION_TOLERANCE_DB = 0.5
THRESHOLD_COUNT = 15
MIN_LOG_OFFSET = 1.0e-20
SILENCE_SENTINEL_DB = -100.0
FRAME_SECONDS = 0.01
SHORT_SILENCE_FRAMES = 40
STL_COMMIT = "cbacae95828179f1238967608e1b91cd515a0261"
STL_SOURCE_URL = (
    "https://raw.githubusercontent.com/openitu/STL/dev/src/sv56/sv-p56.c"
)
STL_HEADER_URL = (
    "https://raw.githubusercontent.com/openitu/STL/dev/src/sv56/sv-p56.h"
)
STL_LICENSE_URL = "https://raw.githubusercontent.com/openitu/STL/dev/LICENSE.md"
STL_SOURCE_SHA256 = (
    "f56c68e3c416e6464f6c376d79d50267f6ff0c803c17253498e6d871cbf3d398"
)
STL_HEADER_SHA256 = (
    "ff1f104efc794845ad24a88cccbab73f315422130097bea2e96a9ac225df09eb"
)
STL_LICENSE_SHA256 = (
    "b29d836c9f40f3fed28d42dc0d8a73730fe6362da60c88d948cb915c968f573e"
)
THRESHOLDS_DB = {
    "H": -1.0,
    "M": -10.0,
    "L": -16.0,
    "U": -35.0,
}


def _validated_samples(
    samples: np.ndarray,
    rate: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono signal")
    if values.size == 0:
        raise ValueError("samples must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("samples must contain only finite values")
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)):
        raise ValueError("rate must be a positive integer")
    sample_rate = int(rate)
    if sample_rate <= 0:
        raise ValueError("rate must be a positive integer")
    # The official C interface accepts float samples while its state and
    # accumulators are double precision.
    return np.asarray(values, dtype=np.float32).astype(np.float64), sample_rate


def _thresholds() -> np.ndarray:
    return 0.5 / np.exp2(np.arange(THRESHOLD_COUNT - 1, -1, -1))


def _active_counts(
    envelope: np.ndarray,
    thresholds: np.ndarray,
    hangover_samples: int,
) -> np.ndarray:
    """Count threshold hits plus the non-circular P.56 hangover window."""
    counts = np.empty(thresholds.size, dtype=np.int64)
    sample_count = envelope.size
    for index, threshold in enumerate(thresholds):
        active = envelope >= threshold
        starts = np.flatnonzero(active)
        if not starts.size:
            counts[index] = 0
            continue
        difference = np.zeros(sample_count + 1, dtype=np.int64)
        difference[starts] += 1
        stops = np.minimum(starts + hangover_samples + 1, sample_count)
        difference[stops] -= 1
        counts[index] = np.count_nonzero(
            np.cumsum(difference[:-1]) > 0
        )
    return counts


def _bin_interp(
    upcount: float,
    lowcount: float,
    upthreshold: float,
    lowthreshold: float,
    margin: float,
    tolerance: float,
) -> float:
    """Port of STL ``bin_interp`` including its tolerance relaxation."""
    tolerance = abs(tolerance)
    if abs((upcount - upthreshold) - margin) < tolerance:
        return upcount
    if abs((lowcount - lowthreshold) - margin) < tolerance:
        return lowcount
    middle_count = (upcount + lowcount) / 2.0
    middle_threshold = (upthreshold + lowthreshold) / 2.0
    iteration = 1
    difference = (middle_count - middle_threshold) - margin
    while abs(difference) > tolerance:
        iteration += 1
        if iteration > 20:
            tolerance *= 1.1
        if difference > tolerance:
            middle_count = (upcount + middle_count) / 2.0
            middle_threshold = (upthreshold + middle_threshold) / 2.0
            lowcount = middle_count
            lowthreshold = middle_threshold
        elif difference < -tolerance:
            middle_count = (middle_count + lowcount) / 2.0
            middle_threshold = (middle_threshold + lowthreshold) / 2.0
            upcount = middle_count
            upthreshold = middle_threshold
        difference = (middle_count - middle_threshold) - margin
    return middle_count


def active_speech_level(samples: np.ndarray, rate: int) -> float:
    """Return P.56 Method B active speech level in dB relative to unit 1.

    A normalized PCM sample of amplitude 1.0 is therefore 0 dB.  The official
    STL implementation returns -100 dB as a silence/no-active sentinel; this
    public API converts that sentinel, zero input, and insufficient activity
    into ``ValueError`` instead of returning a plausible numeric level.
    """
    values, sample_rate = _validated_samples(samples, rate)
    if not np.any(values):
        raise ValueError("active speech level is undefined for zero input")

    hangover_samples = int(
        math.floor(HANGOVER_SECONDS * sample_rate + 0.5)
    )
    smoothing = math.exp(-1.0 / (sample_rate * SMOOTHING_SECONDS))
    gain = 1.0 - smoothing
    envelope_1 = lfilter(
        [gain],
        [1.0, -smoothing],
        np.abs(values),
    )
    envelope = lfilter(
        [gain],
        [1.0, -smoothing],
        envelope_1,
    )
    thresholds = _thresholds()
    counts = _active_counts(
        envelope,
        thresholds,
        hangover_samples,
    )
    squared_sum = float(np.sum(values * values, dtype=np.float64))
    if counts[0] == 0:
        raise ValueError("signal contains insufficient active speech")

    activity_db = 10.0 * math.log10(
        squared_sum / counts[0] + MIN_LOG_OFFSET
    )
    threshold_db = 20.0 * math.log10(float(thresholds[0]))
    if activity_db - threshold_db < MARGIN_DB:
        raise ValueError("signal contains insufficient active speech")

    for index in range(1, THRESHOLD_COUNT):
        if counts[index] == 0:
            continue
        activity_db = 10.0 * math.log10(
            squared_sum / counts[index] + MIN_LOG_OFFSET
        )
        threshold_db = 20.0 * math.log10(
            float(thresholds[index]) + MIN_LOG_OFFSET
        )
        if activity_db - threshold_db <= MARGIN_DB:
            previous_activity_db = 10.0 * math.log10(
                squared_sum / counts[index - 1] + MIN_LOG_OFFSET
            )
            previous_threshold_db = 20.0 * math.log10(
                float(thresholds[index - 1]) + MIN_LOG_OFFSET
            )
            result = _bin_interp(
                activity_db,
                previous_activity_db,
                threshold_db,
                previous_threshold_db,
                MARGIN_DB,
                INTERPOLATION_TOLERANCE_DB,
            )
            if result <= SILENCE_SENTINEL_DB:
                raise ValueError("signal contains insufficient active speech")
            return float(result)
    raise ValueError("signal contains insufficient active speech")


def _short_silence_runs_to_pause(labels: np.ndarray) -> np.ndarray:
    result = labels.copy()
    index = 0
    while index < result.size:
        if result[index] != "S":
            index += 1
            continue
        end = index + 1
        while end < result.size and result[end] == "S":
            end += 1
        if (
            index > 0
            and end < result.size
            and end - index < SHORT_SILENCE_FRAMES
        ):
            result[index:end] = "P"
        index = end
    return result


def classify_activity(
    samples: np.ndarray,
    rate: int = 48_000,
) -> tuple[np.ndarray, dict]:
    """Classify non-overlapping 10 ms frames relative to P.56 active level.

    Levels above -1, -10, -16, and -35 dB relative to the active speech level
    are H, M, L, and U respectively.  Remaining frames are S.  Runs shorter
    than 400 ms (40 frames) are changed to P; active frames are everything
    other than S.
    """
    values, sample_rate = _validated_samples(samples, rate)
    frame_samples = int(round(sample_rate * FRAME_SECONDS))
    if frame_samples <= 0:
        raise ValueError("sampling rate cannot produce a 10 ms frame")
    frame_count = values.size // frame_samples
    if frame_count == 0:
        raise ValueError("signal is shorter than one 10 ms frame")
    used_samples = frame_count * frame_samples
    framed = values[:used_samples].reshape(frame_count, frame_samples)
    frame_power = np.mean(framed * framed, axis=1)
    relative_db = np.full(frame_count, -np.inf, dtype=np.float64)
    nonzero = frame_power > 0
    if np.any(nonzero):
        asl_db = active_speech_level(values[:used_samples], sample_rate)
        relative_db[nonzero] = (
            10.0 * np.log10(frame_power[nonzero]) - asl_db
        )
    else:
        raise ValueError("active speech level is undefined for zero input")

    if not np.any(relative_db > THRESHOLDS_DB["U"]):
        raise ValueError("signal contains insufficient 10 ms frame activity")
    labels = np.full(frame_count, "S", dtype="<U1")
    labels[relative_db > THRESHOLDS_DB["U"]] = "U"
    labels[relative_db > THRESHOLDS_DB["L"]] = "L"
    labels[relative_db > THRESHOLDS_DB["M"]] = "M"
    labels[relative_db > THRESHOLDS_DB["H"]] = "H"
    labels = _short_silence_runs_to_pause(labels)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = {
        str(label): int(count)
        for label, count in zip(unique, counts, strict=True)
    }
    active_frames = int(np.count_nonzero(labels != "S"))
    diagnostics = {
        "asl_db": float(asl_db),
        "frame_ms": FRAME_SECONDS * 1000.0,
        "frame_samples": frame_samples,
        "frame_count": frame_count,
        "used_samples": used_samples,
        "discarded_samples": int(values.size - used_samples),
        "counts": count_map,
        "active_frames": active_frames,
        "active_seconds": active_frames * FRAME_SECONDS,
        "active_mask": labels != "S",
        "ulmh_mask": np.isin(labels, ["U", "L", "M", "H"]),
        "short_silence_frames": SHORT_SILENCE_FRAMES,
        "thresholds_db_relative_to_asl": dict(THRESHOLDS_DB),
        "labels_exclude_only_s": True,
    }
    return labels, diagnostics


__all__ = [
    "active_speech_level",
    "classify_activity",
    "SILENCE_SENTINEL_DB",
    "FRAME_SECONDS",
    "SHORT_SILENCE_FRAMES",
    "THRESHOLDS_DB",
    "STL_COMMIT",
    "STL_SOURCE_SHA256",
    "STL_HEADER_SHA256",
    "STL_LICENSE_SHA256",
]
