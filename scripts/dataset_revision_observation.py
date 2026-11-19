#!/usr/bin/env python3
"""Select genuinely active near-end sources for short-overlap observations."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


MAX_SELECTION_ATTEMPTS = 128
OBSERVATION_START_S = 6.5
OBSERVATION_END_S = 7.0
FAR_ECHO_DELAY_S = 0.02
FAR_DELAYED_START_S = 6.52
FAR_DELAYED_END_S = 7.0
ACTIVITY_FRAME_S = 0.01
MIN_ACTIVE_FRAME_RMS = 1e-5
MIN_POST_BAND_RMS = 0.01
MIN_ACTIVE_FRAME_RATIO = 0.98
SELECTION_POLICY = "revision_short_overlap_real_near_activity_v1"


class ObservationSourceSelectionError(RuntimeError):
    """Raised when no crop provides a real near-end observation source."""


def _case_plan_rows(generator: Any) -> list[dict[str, Any]]:
    case_plan = getattr(generator, "case_plan", None)
    if callable(case_plan):
        rows = case_plan()
    else:
        # The current stress BASE_GEN exposes _case_plan().  Keep the requested
        # public name authoritative while supporting that real implementation.
        case_plan = getattr(generator, "_case_plan", None)
        if not callable(case_plan):
            raise TypeError("configured stress generator has no case_plan()")
        rows = case_plan()
    if not isinstance(rows, list):
        raise TypeError("generator.case_plan() must return a list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("generator.case_plan() contains a non-object row")
    return rows


def _short_overlap_group_ids(generator: Any) -> frozenset[int]:
    group_ids: set[int] = set()
    for row in _case_plan_rows(generator):
        if row.get("category") != "short_overlap":
            continue
        if "group_id" not in row:
            raise ValueError("short_overlap case-plan row has no group_id")
        group_id = row["group_id"]
        if isinstance(group_id, bool) or not isinstance(group_id, int):
            raise ValueError("short_overlap group_id must be an integer")
        if group_id <= 0:
            raise ValueError("short_overlap group_id must be positive")
        if group_id in group_ids:
            raise ValueError(f"short_overlap group_id is repeated: {group_id}")
        group_ids.add(group_id)
    if not group_ids:
        raise ValueError("case plan has no short_overlap group")
    return frozenset(group_ids)


def _window_bounds(
    frames: int,
    rate: int,
    *,
    start_s: float,
    end_s: float,
) -> tuple[int, int]:
    start = int(round(start_s * rate))
    end = int(round(end_s * rate))
    if start < 0 or end > frames or end <= start:
        raise ValueError(
            f"{start_s:g}-{end_s:g}s observation window does not fit "
            f"{frames} frames at {rate} Hz"
        )
    return start, end


def _measure_window(
    samples: np.ndarray,
    *,
    start: int,
    end: int,
    rate: int,
) -> dict[str, Any]:
    observation = samples[start:end]
    post_band_rms = float(
        np.sqrt(np.mean(np.square(observation, dtype=np.float64)))
    )

    frame_samples = int(round(ACTIVITY_FRAME_S * rate))
    if frame_samples <= 0:
        raise ValueError("activity frame rounds to zero samples")
    frame_count = len(observation) // frame_samples
    if frame_count <= 0:
        raise ValueError("observation window has no complete activity frame")
    framed = observation[: frame_count * frame_samples].reshape(
        frame_count,
        frame_samples,
    )
    frame_rms = np.sqrt(
        np.mean(np.square(framed, dtype=np.float64), axis=1)
    )
    active_frame_count = int(
        np.count_nonzero(frame_rms > MIN_ACTIVE_FRAME_RMS)
    )
    active_ratio = float(active_frame_count / frame_count)
    return {
        "start_sample": start,
        "end_sample": end,
        "post_band_rms": post_band_rms,
        "activity_frame_samples": frame_samples,
        "activity_frame_count": frame_count,
        "activity_frame_rms_threshold": MIN_ACTIVE_FRAME_RMS,
        "active_frame_count": active_frame_count,
        "active_frame_ratio": active_ratio,
        "passed": (
            post_band_rms >= MIN_POST_BAND_RMS
            and active_ratio >= MIN_ACTIVE_FRAME_RATIO
        ),
    }


def _delayed_far_source_bounds(
    frames: int,
    rate: int,
) -> tuple[int, int, int, int, int]:
    delay_samples = int(round(FAR_ECHO_DELAY_S * rate))
    delayed_start, delayed_end = _window_bounds(
        frames,
        rate,
        start_s=FAR_DELAYED_START_S,
        end_s=FAR_DELAYED_END_S,
    )
    source_start = delayed_start - delay_samples
    source_end = delayed_end - delay_samples
    if source_start < 0 or source_end <= source_start:
        raise ValueError("20 ms delayed far observation window is invalid")
    return (
        delay_samples,
        delayed_start,
        delayed_end,
        source_start,
        source_end,
    )


def _measure_candidate(
    generator: Any,
    waveform: np.ndarray,
    *,
    rate: int,
    bandwidth: str,
    role: str,
) -> dict[str, Any]:
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("near crop must be a non-empty one-dimensional array")
    if not np.isfinite(samples).all():
        raise ValueError("near crop contains non-finite samples")

    start, end = _window_bounds(
        len(samples),
        rate,
        start_s=OBSERVATION_START_S,
        end_s=OBSERVATION_END_S,
    )
    filtered, band_info = generator.STRESS.band_limit(
        samples,
        rate,
        bandwidth,
    )
    filtered = np.asarray(filtered, dtype=np.float64)
    if filtered.shape != samples.shape or not np.isfinite(filtered).all():
        raise ValueError("STRESS.band_limit returned an invalid waveform")

    base = _measure_window(
        filtered,
        start=start,
        end=end,
        rate=rate,
    )
    delayed: dict[str, Any] | None = None
    if role == "far_input":
        (
            delay_samples,
            delayed_start,
            delayed_end,
            source_start,
            source_end,
        ) = _delayed_far_source_bounds(len(filtered), rate)
        delayed_measurement = _measure_window(
            filtered,
            start=source_start,
            end=source_end,
            rate=rate,
        )
        delayed = {
            "echo_delay_s": FAR_ECHO_DELAY_S,
            "delay_samples": delay_samples,
            "delayed_window": {
                "start_s": FAR_DELAYED_START_S,
                "end_s": FAR_DELAYED_END_S,
                "start_sample": delayed_start,
                "end_sample": delayed_end,
            },
            "source_window": {
                "start_sample": source_start,
                "end_sample": source_end,
            },
            "post_band_rms": delayed_measurement["post_band_rms"],
            "activity": {
                "frame_s": ACTIVITY_FRAME_S,
                "frame_samples": delayed_measurement[
                    "activity_frame_samples"
                ],
                "frame_count": delayed_measurement["activity_frame_count"],
                "frame_rms_threshold": delayed_measurement[
                    "activity_frame_rms_threshold"
                ],
                "active_frame_count": delayed_measurement[
                    "active_frame_count"
                ],
                "active_frame_ratio": delayed_measurement[
                    "active_frame_ratio"
                ],
                "minimum_active_frame_ratio": MIN_ACTIVE_FRAME_RATIO,
            },
            "minimum_post_band_rms": MIN_POST_BAND_RMS,
            "passed": delayed_measurement["passed"],
        }

    return {
        "start_sample": start,
        "end_sample": end,
        "post_band_rms": base["post_band_rms"],
        "band_info": deepcopy(band_info),
        "activity_frame_samples": base["activity_frame_samples"],
        "activity_frame_count": base["activity_frame_count"],
        "activity_frame_rms_threshold": base[
            "activity_frame_rms_threshold"
        ],
        "active_frame_count": base["active_frame_count"],
        "active_frame_ratio": base["active_frame_ratio"],
        "delayed_far": delayed,
        "passed": (
            base["passed"]
            and (delayed is None or delayed["passed"])
        ),
    }


def _validate_non_looping_crop(info: dict[str, Any]) -> None:
    if info.get("repeated") is not False:
        raise ValueError("near observation crop does not prove repeated=False")
    if info.get("within_case_source_repetition") is not False:
        raise ValueError(
            "near observation crop does not prove "
            "within_case_source_repetition=False"
        )
    segments = info.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("near observation crop has no source segments")
    paths = [str(segment.get("path", "")) for segment in segments]
    if any(not path for path in paths):
        raise ValueError("near observation crop segment has no source path")
    if len(paths) != len(set(paths)):
        raise ValueError("near observation crop repeats a source path")


def _selection_metadata(
    *,
    role: str,
    real_group_id: int,
    crop_group_id: int,
    attempt: int,
    measurement: dict[str, Any],
    crop_info: dict[str, Any],
) -> dict[str, Any]:
    segments = crop_info["segments"]
    return {
        "policy": SELECTION_POLICY,
        "original_role": role,
        "selected_role": str(crop_info.get("role", role)),
        "original_group_id": real_group_id,
        "real_group_id": real_group_id,
        "crop_group_id": crop_group_id,
        "attempt": attempt,
        "attempt_number": attempt + 1,
        "crop_group_id_formula": "real_group_id + attempt * 1000",
        "window": {
            "start_s": OBSERVATION_START_S,
            "end_s": OBSERVATION_END_S,
            "start_sample": measurement["start_sample"],
            "end_sample": measurement["end_sample"],
        },
        "post_band_rms": measurement["post_band_rms"],
        "minimum_post_band_rms": MIN_POST_BAND_RMS,
        "band_limit": measurement["band_info"],
        "delayed_far_validation": measurement["delayed_far"],
        "activity": {
            "frame_s": ACTIVITY_FRAME_S,
            "frame_samples": measurement["activity_frame_samples"],
            "frame_count": measurement["activity_frame_count"],
            "frame_rms_threshold": measurement["activity_frame_rms_threshold"],
            "active_frame_count": measurement["active_frame_count"],
            "active_frame_ratio": measurement["active_frame_ratio"],
            "minimum_active_frame_ratio": MIN_ACTIVE_FRAME_RATIO,
        },
        "source_segment_count": len(segments),
        "source_segment_paths": [str(segment["path"]) for segment in segments],
        "source_segment_sha256": [
            str(segment.get("sha256")) for segment in segments
        ],
        "crop_recipe_sha256": crop_info.get("crop_recipe_sha256"),
        "source_untouched": True,
        "source_looped": False,
        "waveform_gain_applied": False,
        "local_gating_applied": False,
        "selection_changes_only_crop_group": True,
    }


def install_observation_source_selection(generator: Any) -> None:
    """Wrap a configured stress generator's near short-overlap crop selector.

    Candidate crops are obtained only by invoking the existing crop function
    with deterministic alternate crop group IDs.  The returned waveform is
    never copied, looped, gain-adjusted, or locally gated by this module.
    """

    if getattr(generator, "SCOPE", None) != "double_talk":
        return
    original_crop = getattr(generator, "_crop_16k", None)
    if not callable(original_crop):
        raise TypeError("configured stress generator has no _crop_16k()")

    short_overlap_groups = _short_overlap_group_ids(generator)

    def select_crop(
        bank: Any,
        records: list[dict[str, Any]],
        *,
        group_id: int,
        role: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if role not in {"near", "far_input"} or group_id not in short_overlap_groups:
            return original_crop(
                bank,
                records,
                group_id=group_id,
                role=role,
            )

        failures: list[str] = []
        for attempt in range(MAX_SELECTION_ATTEMPTS):
            crop_group_id = group_id + attempt * 1000
            try:
                waveform, raw_info = original_crop(
                    bank,
                    records,
                    group_id=crop_group_id,
                    role=role,
                )
            except (MemoryError, OSError):
                raise
            except (ValueError, ObservationSourceSelectionError) as error:
                failures.append(
                    f"crop_group={crop_group_id}: "
                    f"{type(error).__name__}: {error}"
                )
                continue

            info = deepcopy(raw_info)
            _validate_non_looping_crop(info)
            rate = int(generator.RATE)
            bandwidth = str(generator.BANDWIDTH)
            measurement = _measure_candidate(
                generator,
                waveform,
                rate=rate,
                bandwidth=bandwidth,
                role=role,
            )
            if not measurement["passed"]:
                delayed = measurement["delayed_far"]
                failures.append(
                    f"crop_group={crop_group_id}: "
                    f"post_band_rms={measurement['post_band_rms']:.12g}, "
                    f"active_frame_ratio="
                    f"{measurement['active_frame_ratio']:.12g}"
                    + (
                        ", delayed_far_post_band_rms="
                        f"{delayed['post_band_rms']:.12g}, "
                        "delayed_far_active_frame_ratio="
                        f"{delayed['activity']['active_frame_ratio']:.12g}"
                        if delayed is not None
                        else ""
                    )
                )
                continue

            info["observation_source_selection"] = _selection_metadata(
                role=role,
                real_group_id=group_id,
                crop_group_id=crop_group_id,
                attempt=attempt,
                measurement=measurement,
                crop_info=info,
            )
            return waveform, info

        tail = "; ".join(failures[-4:])
        raise ObservationSourceSelectionError(
            f"short_overlap {role} group {group_id} exhausted "
            f"{MAX_SELECTION_ATTEMPTS} distinct crop attempts without "
            f"post_band_rms>={MIN_POST_BAND_RMS} and "
            f"active_frame_ratio>={MIN_ACTIVE_FRAME_RATIO}"
            + (
                " and delayed-far observation gates"
                if role == "far_input"
                else ""
            )
            + f"; last failures: {tail}"
        )

    generator._crop_16k = select_crop


__all__ = [
    "MAX_SELECTION_ATTEMPTS",
    "ObservationSourceSelectionError",
    "install_observation_source_selection",
]
