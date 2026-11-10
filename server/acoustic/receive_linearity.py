"""Engineering receive-level linearity proxy for NB/WB receive contract rows.

This module evaluates the variation of receive level around a nominal
condition.  It deliberately does not calculate RLR: the available inputs are
uncalibrated third-octave levels, not P.79/calibrated DRP measurements.
Contract identities are resolved from the workbook-derived NV200 contract;
the default remains ``WB-wireless`` for compatibility with existing callers.

Canonical run shape (all fields are required unless supplied by ``conditions``):

    {
        "run_id": "run-nominal",
        "level_id": "nominal",                 # or offset_-5 / offset_+5
        "mode": "WB-wireless",
        "status": "completed",
        "output_normalisation": {"normalised": False},
        "clipping": False,
        "conditions": {
            "nominal_volume": "10",
            "fixture": {...},
            "ir": {...},
            "head_model": {...},
            "sample_rate_hz": 48000,
            "duration_s": 8.0,
            "active_speech_window_s": [1.0, 7.0]
        },
        "metrics": {
            "third_octave_band_levels_db": {
                "binaural": {"500": -40.0, "1000": -35.0}
            },
            "window_applied": true,
            "analysis_window_s": [1.0, 7.0]
        }
    }

``conditions`` may provide group-level values for fields omitted by individual
runs.  A run-local value is retained and must agree with the group value.
For a persisted receive report (a run with ``evaluation_id``), the supplied
band levels must explicitly attest that they were recomputed from the virtual
ear audio over ``active_speech_window_s``.  Pure numeric fixtures without an
``evaluation_id`` remain supported but are reported as uncropped provided band
levels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
from typing import Any

from .acceptance_contract import contract_items, required_conditions


SCHEMA_VERSION = 1
METRIC_NAME = "receive_level_linearity_residual_db"
TARGET_MODE = "WB-wireless"
LINEARITY_LIMIT_DB = 0.5
EVIDENCE_LEVEL = "engineering_proxy"

_LEVELS = (
    ("nominal", 0.0),
    ("offset_-5", -5.0),
    ("offset_+5", 5.0),
)
_LEVEL_BY_NAME = {name: offset for name, offset in _LEVELS}
_LEVEL_BY_OFFSET = {offset: name for name, offset in _LEVELS}
_SUCCESS_STATUSES = frozenset({
    "complete",
    "completed",
    "ok",
    "pass",
    "passed",
    "success",
    "succeeded",
})
_MISSING = object()


def _linearity_contract_items(mode: str) -> dict[str, dict[str, Any]]:
    """Return the -5/+5 receive RLR rows for one contract mode."""
    items: dict[str, dict[str, Any]] = {}
    for row in contract_items():
        if row.get("family") != "rlr_linearity":
            continue
        if row.get("direction") != "receive":
            continue
        expected = required_conditions(row)
        if expected.get("mode") != mode:
            continue
        if (
            expected.get("scene") != "baseline"
            or expected.get("volume") != "nominal"
        ):
            continue
        level_id = _LEVEL_BY_OFFSET.get(expected.get("input_level_db"))
        if level_id not in {"offset_-5", "offset_+5"}:
            continue
        if level_id in items:
            raise ValueError(
                f"contract mode {mode!r} has duplicate {level_id} RLR rows"
            )
        items[level_id] = {
            "key": row["source_cell"],
            "item_id": row["id"],
            "test_name": row["test_name"],
            "expected_delta_db": expected["input_level_db"],
            "lower_limit_db": row.get("lower_limit"),
            "upper_limit_db": row.get("upper_limit"),
        }
    return items


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 over canonical, finite JSON data."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _stable_json_value(value: Any, label: str) -> Any:
    """Copy a JSON-compatible value while normalising numeric representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a non-finite number")
        return _round(number)
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string object key")
            copied[key] = _stable_json_value(item, f"{label}.{key}")
        return {key: copied[key] for key in sorted(copied)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _stable_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{label} contains a value that is not JSON-compatible")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} must be numeric")
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"{label} must be numeric") from error
    elif isinstance(value, Real):
        number = float(value)
    else:
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _path_value(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _first_path(value: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        candidate = _path_value(value, *path)
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _run_label(run: Any, index: int) -> str:
    if isinstance(run, Mapping):
        run_id = run.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return f"run {run_id!r}"
        level = run.get("level_id", run.get("level"))
        if isinstance(level, str) and level.strip():
            return f"run level {level!r}"
    return f"runs[{index}]"


def _extract_level(run: Mapping[str, Any]) -> str:
    raw = _first_path(run, (
        ("level_id",),
        ("level",),
        ("offset_id",),
        ("condition_level",),
        ("conditions", "level_id"),
        ("conditions", "level"),
        ("conditions", "offset_id"),
        ("conditions", "offset_db"),
    ))
    if raw is _MISSING:
        raise ValueError("level_id is missing")
    if isinstance(raw, Real) and not isinstance(raw, bool):
        offset = _finite_float(raw, "level offset")
        if offset not in _LEVEL_BY_OFFSET:
            raise ValueError("level offset must be one of 0, -5, or +5 dB")
        return _LEVEL_BY_OFFSET[offset]
    if not isinstance(raw, str):
        raise ValueError("level_id must be a string or numeric offset")
    name = raw.strip()
    if name in _LEVEL_BY_NAME:
        return name
    try:
        offset = float(name)
    except ValueError as error:
        raise ValueError(
            "level_id must be nominal, offset_-5, or offset_+5"
        ) from error
    if offset not in _LEVEL_BY_OFFSET:
        raise ValueError("level offset must be one of 0, -5, or +5 dB")
    return _LEVEL_BY_OFFSET[offset]


def _extract_normalised(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("output_normalisation", "normalised"),
        ("output_normalisation", "normalized"),
        ("simulation", "output_normalisation", "normalised"),
        ("simulation", "output_normalisation", "normalized"),
        ("normalisation", "normalised"),
        ("normalization", "normalized"),
        ("normalised",),
        ("normalized",),
    ))


def _extract_clipping(run: Mapping[str, Any]) -> Any:
    for path in (
        ("clipping",),
        ("clipped",),
        ("clipping_detected",),
        ("output", "clipping"),
        ("output_levels", "clipping"),
        ("metrics", "clipping"),
        ("result", "clipping"),
    ):
        value = _path_value(run, *path)
        if value is _MISSING:
            continue
        if isinstance(value, Mapping):
            nested = value.get("detected", value.get("clipping", value.get("value")))
            if nested is not _MISSING:
                return nested
        return value
    for path in (
        ("clipping_percent",),
        ("output_levels", "clipping_percent"),
        ("metrics", "clipping_percent"),
    ):
        value = _path_value(run, *path)
        if value is not _MISSING:
            percent = _finite_float(value, "clipping_percent")
            if percent < 0.0 or percent > 100.0:
                raise ValueError("clipping_percent must be between 0 and 100")
            return percent > 0.0
    return _MISSING


def _extract_status(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("status",),
        ("run_status",),
        ("execution", "status"),
        ("result", "status"),
    ))


def _extract_mode(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("mode",),
        ("audio_mode",),
        ("conditions", "mode"),
        ("request", "audio_mode"),
    ))


def _extract_nominal_volume(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("nominal_volume",),
        ("nominal_volume_setting",),
        ("volume_setting",),
        ("conditions", "nominal_volume"),
        ("conditions", "nominal_volume_setting"),
        ("conditions", "volume_setting"),
        ("dut", "volume_setting"),
        ("request", "dut", "volume_setting"),
    ))


def _extract_fixture(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("fixture_identity",),
        ("fixture",),
        ("fixture_id",),
        ("conditions", "fixture"),
        ("conditions", "fixture_identity"),
        ("conditions", "fixture_id"),
        ("input_identity", "fixture"),
        ("request", "fixture"),
    ))


def _extract_ir(run: Mapping[str, Any]) -> Any:
    explicit = _first_path(run, (
        ("ir_identity",),
        ("ir",),
        ("conditions", "ir"),
        ("conditions", "ir_identity"),
    ))
    if explicit is not _MISSING:
        return explicit
    identity = _path_value(run, "input_identity")
    if isinstance(identity, Mapping) and any(
        key in identity
        for key in ("left_ir", "right_ir", "left_ir_processing", "right_ir_processing")
    ):
        return {
            "left_ir": identity.get("left_ir"),
            "right_ir": identity.get("right_ir"),
            "left_ir_processing": identity.get("left_ir_processing"),
            "right_ir_processing": identity.get("right_ir_processing"),
        }
    return _MISSING


def _extract_head_model(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("head_model",),
        ("head_model_identity",),
        ("conditions", "head_model"),
        ("conditions", "head_model_identity"),
        ("simulation", "head_model"),
        ("input_identity", "head_model"),
    ))


def _extract_sample_rate(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("sample_rate_hz",),
        ("sampling_rate_hz",),
        ("sample_rate",),
        ("conditions", "sample_rate_hz"),
        ("conditions", "sampling_rate_hz"),
        ("simulation", "analysis_sample_rate_hz"),
        ("input_identity", "speaker", "sample_rate_hz"),
    ))


def _extract_duration(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("duration_s",),
        ("analysis_duration_s",),
        ("conditions", "duration_s"),
        ("conditions", "analysis_duration_s"),
        ("simulation", "output_duration_s"),
        ("input_identity", "speaker", "duration_s"),
    ))


def _extract_active_window(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("active_speech_window_s",),
        ("active_speech_window",),
        ("speech_window_s",),
        ("speech_window",),
        ("conditions", "active_speech_window_s"),
        ("conditions", "active_speech_window"),
        ("conditions", "speech_window_s"),
        ("conditions", "speech_window"),
    ))


def _extract_evaluation_id(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("evaluation_id",),
        ("report", "evaluation_id"),
        ("metadata", "evaluation_id"),
    ))


def _extract_window_applied(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("window_applied",),
        ("metrics", "window_applied"),
        ("result", "metrics", "window_applied"),
        ("metadata", "window_applied"),
        ("window", "applied"),
    ))


def _extract_analysis_window(run: Mapping[str, Any]) -> Any:
    return _first_path(run, (
        ("analysis_window_s",),
        ("metrics", "analysis_window_s"),
        ("result", "metrics", "analysis_window_s"),
        ("metadata", "analysis_window_s"),
        ("window", "analysis_window_s"),
    ))


def _extract_bands(run: Mapping[str, Any], label: str) -> tuple[dict[float, float], list[str]]:
    raw = _first_path(run, (
        ("metrics", "third_octave_band_levels_db", "binaural"),
        ("third_octave_band_levels_db", "binaural"),
        ("result", "metrics", "third_octave_band_levels_db", "binaural"),
    ))
    if raw is _MISSING:
        return {}, [f"{label}: third_octave_band_levels_db.binaural is missing"]
    if not isinstance(raw, Mapping):
        return {}, [f"{label}: binaural third-octave levels must be an object"]
    if not raw:
        return {}, [f"{label}: binaural third-octave levels are empty"]

    levels: dict[float, float] = {}
    reasons: list[str] = []
    for raw_frequency, raw_level in raw.items():
        try:
            frequency = _finite_float(raw_frequency, "frequency")
        except (TypeError, ValueError):
            reasons.append(
                f"{label}: invalid third-octave frequency {raw_frequency!r}"
            )
            continue
        if frequency <= 0.0:
            reasons.append(
                f"{label}: third-octave frequency must be positive "
                f"(got {frequency:g})"
            )
            continue
        if frequency in levels:
            reasons.append(
                f"{label}: duplicate third-octave frequency {frequency:g}"
            )
            continue
        try:
            level = _finite_float(raw_level, "band level")
        except (TypeError, ValueError):
            reasons.append(
                f"{label}: band {frequency:g} Hz is missing or non-finite"
            )
            continue
        levels[frequency] = level
    return levels, reasons


def _empty_item(
    level_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "key": metadata["key"],
        "item_id": metadata["item_id"],
        "test_name": metadata["test_name"],
        "metric": METRIC_NAME,
        "unit": "dB",
        "level_id": level_id,
        "expected_delta_db": metadata["expected_delta_db"],
        "relative_level_db": None,
        "observed_delta_db": None,
        "receive_level_linearity_residual_db": None,
        "residual_db": None,
        "lower_limit_db": metadata["lower_limit_db"],
        "upper_limit_db": metadata["upper_limit_db"],
        "within_limit": None,
        "status": "invalid",
        "reasons": [],
    }


def _apply_contract_identity(
    report: dict[str, Any],
    mode: str,
    contract_by_level: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if contract_by_level is None:
        contract_by_level = _linearity_contract_items(mode)
    report["mode"] = mode
    report["standard_item_ids"] = [
        metadata["item_id"]
        for level_id in ("offset_-5", "offset_+5")
        if (metadata := contract_by_level.get(level_id)) is not None
    ]
    report["items"] = {
        metadata["key"]: _empty_item(level_id, metadata)
        for level_id in ("offset_-5", "offset_+5")
        if (metadata := contract_by_level.get(level_id)) is not None
    }
    for key in ("B4", "B5", "B6"):
        report.pop(key, None)
    for key in report["items"]:
        report[key] = None


def _base_report() -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "receive_level_linearity",
        "metric": METRIC_NAME,
        "mode": TARGET_MODE,
        "standard_item_ids": [],
        "evidence_level": EVIDENCE_LEVEL,
        "formal_rlr_linearity_available": False,
        "formal_compliance_claim_allowed": False,
        "formal_rlr_available": False,
        "status": "invalid",
        "overall_status": "invalid",
        "group_valid": False,
        "conditions": None,
        "aggregation": None,
        "window_source": "provided_band_levels",
        "window_applied": False,
        "analysis_window_s": None,
        "window_evidence": {},
        "runs": {},
        "items": {},
        "reasons": [],
        "canonical_sha256": None,
    }
    _apply_contract_identity(report, TARGET_MODE)
    return report


def _finish(
    report: dict[str, Any],
    reasons: Sequence[str],
    *,
    include_canonical_hash: bool,
) -> dict[str, Any]:
    for key in ("B4", "B5", "B6"):
        report.pop(key, None)
    for key, item in report["items"].items():
        report[key] = item
    report["reasons"] = sorted(set(reasons))
    report["overall_status"] = report["status"]
    report["canonical_sha256"] = None
    if include_canonical_hash:
        report["canonical_sha256"] = canonical_sha256({
            key: value
            for key, value in report.items()
            if key != "canonical_sha256"
        })
    return report


def _condition_source(
    explicit: Mapping[str, Any] | None,
    aliases: Sequence[str],
) -> Any:
    if explicit is None:
        return _MISSING
    for key in aliases:
        if key in explicit:
            return explicit[key]
    return _MISSING


def _validate_condition(
    key: str,
    value: Any,
) -> Any:
    if key == "mode":
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()
    if key == "nominal_volume":
        if value is None or isinstance(value, bool):
            raise ValueError("must be a non-null string or number")
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be empty")
        return _stable_json_value(value, key)
    if key == "fixture":
        if value is None:
            raise ValueError("must be present")
        return _stable_json_value(value, key)
    if key in {"ir", "head_model"}:
        return _stable_json_value(value, key)
    if key == "sample_rate_hz":
        rate = _finite_float(value, key)
        if rate <= 0.0:
            raise ValueError("must be positive")
        return _round(rate)
    if key == "duration_s":
        duration = _finite_float(value, key)
        if duration <= 0.0:
            raise ValueError("must be positive")
        return _round(duration)
    if key == "active_speech_window_s":
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ) or len(value) != 2:
            raise ValueError("must contain exactly [start_s, end_s]")
        start = _finite_float(value[0], "active speech window start")
        end = _finite_float(value[1], "active speech window end")
        if start < 0.0 or end <= start:
            raise ValueError("must satisfy 0 <= start_s < end_s")
        return [_round(start), _round(end)]
    raise AssertionError(f"unknown condition {key}")


def _resolve_conditions(
    runs_by_level: Mapping[str, Mapping[str, Any]],
    explicit: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    extractors = {
        "mode": (_extract_mode, ("mode",)),
        "nominal_volume": (
            _extract_nominal_volume,
            ("nominal_volume", "nominal_volume_setting", "volume_setting"),
        ),
        "fixture": (_extract_fixture, ("fixture", "fixture_id", "fixture_identity")),
        "ir": (_extract_ir, ("ir", "ir_identity")),
        "head_model": (
            _extract_head_model,
            ("head_model", "head_model_identity"),
        ),
        "sample_rate_hz": (
            _extract_sample_rate,
            ("sample_rate_hz", "sampling_rate_hz", "sample_rate"),
        ),
        "duration_s": (
            _extract_duration,
            ("duration_s", "analysis_duration_s"),
        ),
        "active_speech_window_s": (
            _extract_active_window,
            (
                "active_speech_window_s",
                "active_speech_window",
                "speech_window_s",
                "speech_window",
            ),
        ),
    }
    resolved: dict[str, Any] = {}
    reasons: list[str] = []
    for key, (extractor, aliases) in extractors.items():
        source = _condition_source(explicit, aliases)
        source_label = "conditions"
        if source is _MISSING:
            for level_id, _ in _LEVELS:
                candidate = extractor(runs_by_level[level_id])
                if candidate is not _MISSING:
                    source = candidate
                    source_label = f"run {level_id}"
                    break
        if source is _MISSING:
            reasons.append(f"conditions: {key} is missing")
            resolved[key] = None
            continue
        try:
            reference = _validate_condition(key, source)
        except (TypeError, ValueError) as error:
            reasons.append(f"{source_label}: {key} {error}")
            resolved[key] = None
            continue
        resolved[key] = reference

        for level_id, _ in _LEVELS:
            candidate = extractor(runs_by_level[level_id])
            if candidate is _MISSING:
                continue
            try:
                actual = _validate_condition(key, candidate)
            except (TypeError, ValueError) as error:
                reasons.append(f"run {level_id}: {key} {error}")
                continue
            if actual != reference:
                reasons.append(
                    f"run {level_id}: condition mismatch for {key}"
                )

    if resolved.get("ir") is None and resolved.get("head_model") is None:
        reasons.append(
            "conditions: at least one of ir or head_model identity is required"
        )
    return resolved, reasons


def _normalise_frequency_list(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError("frequency_hz must be a sequence")
    frequencies: list[float] = []
    seen: set[float] = set()
    for index, raw in enumerate(value):
        frequency = _finite_float(raw, f"frequency_hz[{index}]")
        if frequency <= 0.0:
            raise ValueError(f"frequency_hz[{index}] must be positive")
        if frequency in seen:
            raise ValueError(f"frequency_hz contains duplicate {frequency:g}")
        seen.add(frequency)
        frequencies.append(frequency)
    if not frequencies:
        raise ValueError("frequency_hz must not be empty")
    return frequencies


def _normalise_weights(
    raw_weights: Any,
    requested_frequencies: list[float] | None,
) -> tuple[list[float], dict[float, float], str]:
    if raw_weights is None:
        frequencies = requested_frequencies
        if not frequencies:
            raise ValueError(
                "frequency_hz is required when equal weights are used with "
                "an explicit band list"
            )
        return frequencies, {frequency: 1.0 for frequency in frequencies}, "equal"

    if isinstance(raw_weights, Mapping):
        weights: dict[float, float] = {}
        for raw_frequency, raw_weight in raw_weights.items():
            frequency = _finite_float(raw_frequency, "band weight frequency")
            weight = _finite_float(raw_weight, f"weight at {frequency:g} Hz")
            if frequency <= 0.0 or weight <= 0.0:
                raise ValueError("band frequencies and weights must be positive")
            if frequency in weights:
                raise ValueError(
                    f"band_weights contains duplicate frequency {frequency:g}"
                )
            weights[frequency] = weight
        frequencies = sorted(weights)
        if not frequencies:
            raise ValueError("band_weights must not be empty")
        if requested_frequencies is not None and set(frequencies) != set(
            requested_frequencies
        ):
            raise ValueError(
                "band_weights keys must exactly match frequency_hz"
            )
        return frequencies, weights, "explicit"

    if isinstance(raw_weights, Sequence) and not isinstance(
        raw_weights, (str, bytes, bytearray)
    ):
        if requested_frequencies is None:
            raise ValueError(
                "frequency_hz is required when band_weights is a sequence"
            )
        if len(raw_weights) != len(requested_frequencies):
            raise ValueError(
                "band_weights sequence length must match frequency_hz"
            )
        weights: dict[float, float] = {}
        for frequency, raw_weight in zip(requested_frequencies, raw_weights):
            weight = _finite_float(raw_weight, f"weight at {frequency:g} Hz")
            if weight <= 0.0:
                raise ValueError("band weights must be positive")
            weights[frequency] = weight
        return list(requested_frequencies), weights, "explicit"

    raise ValueError("band_weights must be an object or sequence")


def _aggregation_config(
    explicit: Mapping[str, Any] | None,
    run_bands: Mapping[str, Mapping[float, float]],
) -> tuple[list[float], dict[float, float], str, list[str]]:
    reasons: list[str] = []
    requested: list[float] | None = None
    if explicit is not None:
        raw_frequencies = _condition_source(
            explicit, ("frequency_hz", "bands_hz", "frequency_bands_hz")
        )
        if raw_frequencies is not _MISSING:
            try:
                requested = _normalise_frequency_list(raw_frequencies)
            except (TypeError, ValueError) as error:
                reasons.append(f"conditions: {error}")

    raw_weights = (
        _condition_source(explicit, ("band_weights", "weights"))
        if explicit is not None
        else _MISSING
    )
    if raw_weights is _MISSING:
        raw_weights = None
    try:
        if requested is not None:
            frequencies, weights, mode = _normalise_weights(raw_weights, requested)
        elif raw_weights is not None:
            frequencies, weights, mode = _normalise_weights(raw_weights, None)
        else:
            common = set.intersection(*(set(levels) for levels in run_bands.values()))
            frequencies = sorted(common)
            if not frequencies:
                raise ValueError("runs share no third-octave frequency bands")
            weights = {frequency: 1.0 for frequency in frequencies}
            mode = "equal"
    except (TypeError, ValueError) as error:
        reasons.append(f"aggregation: {error}")
        frequencies, weights, mode = [], {}, "invalid"
    return frequencies, weights, mode, reasons


def _weighted_energy_level(
    bands: Mapping[float, float],
    frequencies: Sequence[float],
    weights: Mapping[float, float],
) -> float:
    log_powers = [bands[frequency] / 10.0 for frequency in frequencies]
    maximum = max(log_powers)
    denominator = sum(weights[frequency] for frequency in frequencies)
    scaled = sum(
        weights[frequency] * (10.0 ** (bands[frequency] / 10.0 - maximum))
        for frequency in frequencies
    )
    if denominator <= 0.0 or scaled <= 0.0:
        raise ValueError("weighted band energy is not positive")
    return 10.0 * (maximum + math.log10(scaled / denominator))


def _condition_output(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _condition_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_condition_output(item) for item in value]
    return value


def _validate_window_evidence(
    runs_by_level: Mapping[str, Mapping[str, Any]],
    run_ids: Mapping[str, str],
    active_window: list[float] | None,
) -> tuple[bool, list[float] | None, dict[str, Any], list[str]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    evaluation_states: dict[str, bool] = {}
    cropped_states: dict[str, bool] = {}

    for level_id, _ in _LEVELS:
        run = runs_by_level[level_id]
        label = f"run {run_ids[level_id]!r} ({level_id})"
        raw_evaluation_id = _extract_evaluation_id(run)
        evaluation_id: str | None = None
        if raw_evaluation_id is not _MISSING and raw_evaluation_id is not None:
            if not isinstance(raw_evaluation_id, str) or not raw_evaluation_id.strip():
                reasons.append(
                    f"{label}: evaluation_id must be a non-empty string"
                )
            else:
                evaluation_id = raw_evaluation_id.strip()
        evaluation_states[level_id] = evaluation_id is not None

        raw_applied = _extract_window_applied(run)
        raw_window = _extract_analysis_window(run)
        applied: bool | None = None
        analysis_window: list[float] | None = None
        evidence_ok = True

        if raw_applied is _MISSING and raw_window is _MISSING:
            if evaluation_id is not None:
                reasons.append(
                    f"{label}: evaluation_id run must declare "
                    "window_applied=true and analysis_window_s"
                )
                evidence_ok = False
        else:
            if raw_applied is _MISSING:
                reasons.append(
                    f"{label}: window_applied is required when "
                    "analysis_window_s is declared"
                )
                evidence_ok = False
            elif not isinstance(raw_applied, bool):
                reasons.append(f"{label}: window_applied must be boolean")
                evidence_ok = False
            else:
                applied = raw_applied

            if raw_window is _MISSING:
                reasons.append(
                    f"{label}: analysis_window_s is required when "
                    "window_applied is declared"
                )
                evidence_ok = False
            else:
                try:
                    analysis_window = _validate_condition(
                        "active_speech_window_s",
                        raw_window,
                    )
                except (TypeError, ValueError) as error:
                    reasons.append(f"{label}: analysis_window_s {error}")
                    evidence_ok = False

        if analysis_window is not None:
            if active_window is None:
                reasons.append(
                    f"{label}: analysis_window_s cannot be verified because "
                    "active_speech_window_s is unavailable"
                )
                evidence_ok = False
            elif analysis_window != active_window:
                reasons.append(
                    f"{label}: analysis_window_s does not match "
                    "active_speech_window_s"
                )
                evidence_ok = False

        if evaluation_id is not None and applied is not True:
            reasons.append(
                f"{label}: evaluation_id run requires window_applied=true"
            )
            evidence_ok = False
        elif evaluation_id is None and applied is True:
            reasons.append(
                f"{label}: numeric run cannot claim window_applied=true "
                "without evaluation_id evidence"
            )
            evidence_ok = False

        evidence[level_id] = {
            "evaluation_id": evaluation_id,
            "window_applied": applied,
            "analysis_window_s": analysis_window,
        }
        cropped_states[level_id] = (
            evidence_ok
            and evaluation_id is not None
            and applied is True
            and analysis_window is not None
            and analysis_window == active_window
        )

    has_evaluation = any(evaluation_states.values())
    if has_evaluation and not all(evaluation_states.values()):
        reasons.append(
            "group mixes evaluation_id reports with numeric runs"
        )
    all_cropped = all(cropped_states.values())
    if all_cropped and not all(evaluation_states.values()):
        all_cropped = False
    return (
        all_cropped,
        active_window if all_cropped else None,
        evidence,
        reasons,
    )


def evaluate_rlr_linearity_group(
    runs: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Any] | None = None,
    *,
    include_canonical_hash: bool = False,
) -> dict[str, Any]:
    """Evaluate NB/WB receive RLR rows as an engineering linearity proxy.

    Contract identities come from the workbook-derived NV200 contract.  NB
    uses B4/B5 and WB uses B5/B6; the mode defaults to ``WB-wireless`` when it
    is present on the runs.  The group is invalid unless all three runs have
    non-normalised, non-clipped successful evidence with matching acquisition
    conditions and complete finite third-octave binaural levels.

    ``conditions["band_weights"]`` may be a frequency-to-weight object, or a
    sequence aligned with ``conditions["frequency_hz"]``.  Weights apply in
    the energy domain.  Omit weights for fixed equal-energy averaging.

    When ``include_canonical_hash`` is true, ``canonical_sha256`` is computed
    over the complete JSON result excluding that field.
    """
    report = _base_report()
    reasons: list[str] = []
    if not isinstance(include_canonical_hash, bool):
        reasons.append("include_canonical_hash must be boolean")
        include_canonical_hash = False
    if conditions is not None and not isinstance(conditions, Mapping):
        reasons.append("conditions must be an object or null")
        conditions = None
    if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Sequence):
        reasons.append("runs must be a sequence of run objects")
        return _finish(
            report,
            reasons,
            include_canonical_hash=include_canonical_hash,
        )
    if len(runs) != len(_LEVELS):
        reasons.append(
            f"group must contain exactly 3 runs; received {len(runs)}"
        )

    parsed: dict[str, Mapping[str, Any]] = {}
    run_ids: dict[str, str] = {}
    for index, run in enumerate(runs):
        label = _run_label(run, index)
        if not isinstance(run, Mapping):
            reasons.append(f"{label}: run must be an object")
            continue
        try:
            level_id = _extract_level(run)
        except (TypeError, ValueError) as error:
            reasons.append(f"{label}: {error}")
            continue
        if level_id in parsed:
            reasons.append(f"{label}: duplicate level {level_id}")
            continue
        parsed[level_id] = run
        raw_run_id = run.get("run_id")
        if raw_run_id is None:
            run_ids[level_id] = level_id
        elif isinstance(raw_run_id, str) and raw_run_id.strip():
            run_ids[level_id] = raw_run_id.strip()
        else:
            reasons.append(f"{label}: run_id must be a non-empty string")
            run_ids[level_id] = level_id

        normalised = _extract_normalised(run)
        if normalised is _MISSING:
            reasons.append(f"{label}: output_normalisation.normalised is missing")
        elif not isinstance(normalised, bool):
            reasons.append(
                f"{label}: output_normalisation.normalised must be boolean"
            )
        elif normalised:
            reasons.append(f"{label}: output normalisation was applied")

        clipping = _extract_clipping(run)
        if clipping is _MISSING:
            reasons.append(f"{label}: clipping flag is missing")
        elif not isinstance(clipping, bool):
            reasons.append(f"{label}: clipping flag must be boolean")
        elif clipping:
            reasons.append(f"{label}: clipping was detected")

        status = _extract_status(run)
        if status is not _MISSING:
            if not isinstance(status, str) or not status.strip():
                reasons.append(f"{label}: run status must be a string")
            elif status.strip().lower() not in _SUCCESS_STATUSES:
                reasons.append(
                    f"{label}: run status {status!r} is not a successful status"
                )

    for level_id, _ in _LEVELS:
        if level_id not in parsed:
            reasons.append(f"group: required level {level_id!r} is missing")

    if set(parsed) != set(_LEVEL_BY_NAME):
        return _finish(
            report,
            reasons,
            include_canonical_hash=include_canonical_hash,
        )

    resolved_conditions, condition_reasons = _resolve_conditions(parsed, conditions)
    reasons.extend(condition_reasons)
    report["conditions"] = {
        "mode": resolved_conditions["mode"],
        "nominal_volume": resolved_conditions["nominal_volume"],
        "fixture": _condition_output(resolved_conditions["fixture"]),
        "ir": _condition_output(resolved_conditions["ir"]),
        "head_model": _condition_output(resolved_conditions["head_model"]),
        "sample_rate_hz": resolved_conditions["sample_rate_hz"],
        "duration_s": resolved_conditions["duration_s"],
        "active_speech_window_s": resolved_conditions["active_speech_window_s"],
    }
    window_applied, analysis_window, window_evidence, window_reasons = (
        _validate_window_evidence(
            parsed,
            run_ids,
            resolved_conditions["active_speech_window_s"],
        )
    )
    report["window_applied"] = window_applied
    report["window_source"] = (
        "provided_cropped_band_levels"
        if window_applied
        else "provided_band_levels"
    )
    report["analysis_window_s"] = analysis_window
    report["window_evidence"] = window_evidence
    reasons.extend(window_reasons)
    mode = resolved_conditions["mode"]
    contract_by_level: dict[str, dict[str, Any]] = {}
    if mode is not None:
        try:
            contract_by_level = _linearity_contract_items(mode)
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"conditions: invalid linearity contract row: {error}")
        _apply_contract_identity(report, mode, contract_by_level)
        if not contract_by_level:
            reasons.append(
                f"conditions: no receive rlr_linearity contract rows for mode {mode!r}"
            )

    run_bands: dict[str, dict[float, float]] = {}
    for level_id, _ in _LEVELS:
        label = f"run {run_ids[level_id]!r} ({level_id})"
        bands, band_reasons = _extract_bands(parsed[level_id], label)
        reasons.extend(band_reasons)
        run_bands[level_id] = bands

    band_sets = {level_id: set(bands) for level_id, bands in run_bands.items()}
    reference_bands = band_sets["nominal"]
    for level_id, _ in _LEVELS[1:]:
        if band_sets[level_id] != reference_bands:
            missing = sorted(reference_bands - band_sets[level_id])
            extra = sorted(band_sets[level_id] - reference_bands)
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(f"{value:g}" for value in missing))
            if extra:
                detail.append("extra " + ", ".join(f"{value:g}" for value in extra))
            reasons.append(
                f"run {run_ids[level_id]!r}: third-octave frequency bands differ "
                f"from nominal ({'; '.join(detail)})"
            )

    frequencies, weights, weighting_mode, aggregation_reasons = _aggregation_config(
        conditions, run_bands
    )
    reasons.extend(aggregation_reasons)
    if frequencies:
        for level_id, _ in _LEVELS:
            missing = [
                frequency
                for frequency in frequencies
                if frequency not in run_bands[level_id]
            ]
            if missing:
                reasons.append(
                    f"run {run_ids[level_id]!r}: selected frequency band(s) "
                    f"missing: {', '.join(f'{value:g}' for value in missing)} Hz"
                )

    report["aggregation"] = {
        "method": "weighted_energy_average_of_third_octave_band_levels",
        "weighting": weighting_mode,
        "frequency_hz": frequencies,
        "weights": {
            format(_round(frequency), ".12g"): weights[frequency]
            for frequency in frequencies
        },
    }

    if reasons:
        return _finish(
            report,
            reasons,
            include_canonical_hash=include_canonical_hash,
        )

    q_by_level: dict[str, float] = {}
    for level_id, _ in _LEVELS:
        try:
            q_by_level[level_id] = _weighted_energy_level(
                run_bands[level_id], frequencies, weights
            )
            if q_by_level[level_id] <= -120.0 + 1e-9:
                reasons.append(
                    f"run {run_ids[level_id]!r}: selected bands are at the "
                    "digital measurement floor; no measurable excitation for linearity"
                )
        except (KeyError, OverflowError, ValueError) as error:
            reasons.append(f"run {run_ids[level_id]!r}: aggregation failed: {error}")
    if reasons:
        return _finish(
            report,
            reasons,
            include_canonical_hash=include_canonical_hash,
        )

    report["runs"] = {
        level_id: {
            "run_id": run_ids[level_id],
            "level_id": level_id,
            "offset_db": _LEVEL_BY_NAME[level_id],
            "relative_level_db": _round(q_by_level[level_id]),
        }
        for level_id, _ in _LEVELS
    }

    reasons.append(
        "Engineering proxy only: uncalibrated third-octave levels replace "
        "P.79/calibrated-DRP RLR; formal RLR linearity and standard compliance "
        "are unavailable."
    )
    all_within = True
    for level_id in ("offset_-5", "offset_+5"):
        metadata = contract_by_level[level_id]
        observed = q_by_level[level_id] - q_by_level["nominal"]
        residual = observed - float(metadata["expected_delta_db"])
        within = -LINEARITY_LIMIT_DB <= residual <= LINEARITY_LIMIT_DB
        all_within = all_within and within
        item = _empty_item(level_id, metadata)
        item.update({
            "relative_level_db": _round(q_by_level[level_id]),
            "observed_delta_db": _round(observed),
            "receive_level_linearity_residual_db": _round(residual),
            "residual_db": _round(residual),
            "within_limit": within,
            "status": "pass" if within else "fail",
            "reasons": (
                [] if within else [
                    f"{metadata['key']} residual {residual:.6g} dB is outside "
                    f"[-{LINEARITY_LIMIT_DB:g}, {LINEARITY_LIMIT_DB:g}] dB"
                ]
            ),
        })
        report["items"][metadata["key"]] = item
        if not within:
            reasons.append(item["reasons"][0])

    report["status"] = "pass" if all_within else "fail"
    report["group_valid"] = True
    return _finish(
        report,
        reasons,
        include_canonical_hash=include_canonical_hash,
    )


__all__ = [
    "EVIDENCE_LEVEL",
    "LINEARITY_LIMIT_DB",
    "METRIC_NAME",
    "SCHEMA_VERSION",
    "TARGET_MODE",
    "canonical_sha256",
    "evaluate_rlr_linearity_group",
]
