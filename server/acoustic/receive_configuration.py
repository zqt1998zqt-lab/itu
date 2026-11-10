"""Validation for the acoustic model used by acceptance receive cases."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import PurePosixPath
import re
from typing import Any


MEASUREMENT_POINTS = frozenset({
    "unspecified",
    "drp",
    "erp",
    "ear_entrance",
    "blocked_ear_canal",
})
IR_KINDS = frozenset({"hrir", "brir", "end_to_end", "unspecified"})

_HEAD_FIELDS = frozenset({
    "model",
    "azimuth_deg",
    "head_radius_m",
    "speed_of_sound_mps",
    "measurement_point",
    "provenance",
})
_PAIRED_IR_FIELDS = frozenset({
    "model",
    "left_ir",
    "right_ir",
    "ir_kind",
    "measurement_point",
    "provenance",
})
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:")
_DEFAULT_HEAD = {
    "model": "head",
    "azimuth_deg": 30.0,
    "head_radius_m": 0.0875,
    "speed_of_sound_mps": 343.0,
    "measurement_point": "unspecified",
    "provenance": "uncalibrated software proxy",
}


class ReceiveConfigurationError(ValueError):
    """The receive acoustic configuration is incomplete or unsafe."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReceiveConfigurationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReceiveConfigurationError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiveConfigurationError(f"{label} must be a non-empty string")
    return value.strip()


def safe_receive_ir_path(value: Any, label: str = "IR path") -> str:
    """Validate a relative WAV path without touching the filesystem."""
    if not isinstance(value, str) or not value:
        raise ReceiveConfigurationError(f"{label} must be a non-empty relative path")
    if "\x00" in value:
        raise ReceiveConfigurationError(f"{label} contains a NUL byte")
    if "\\" in value:
        raise ReceiveConfigurationError(f"{label} must not contain backslashes")
    if value.startswith("/") or _WINDOWS_ABSOLUTE.match(value):
        raise ReceiveConfigurationError(f"{label} must not be absolute")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ReceiveConfigurationError(
            f"{label} must not contain empty, current, or parent components"
        )
    if any(any(ord(character) < 32 for character in part) for part in parts):
        raise ReceiveConfigurationError(f"{label} contains a control character")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ReceiveConfigurationError(f"{label} must not be absolute")
    if path.suffix.lower() != ".wav":
        raise ReceiveConfigurationError(f"{label} must name a .wav file")
    return path.as_posix()


def _reject_unknown_fields(raw: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ReceiveConfigurationError(
            "receive_simulation contains unsupported fields: "
            + ", ".join(unknown)
        )


def _measurement_point(value: Any, *, default: str | None = None) -> str:
    candidate = default if value is None and default is not None else value
    point = _nonempty_string(candidate, "measurement_point")
    if point not in MEASUREMENT_POINTS:
        raise ReceiveConfigurationError(
            "measurement_point must be one of: "
            + ", ".join(sorted(MEASUREMENT_POINTS))
        )
    return point


def normalize_receive_configuration(value: Any) -> dict[str, Any]:
    """Return a strict, JSON-safe receive configuration.

    ``None`` selects the legacy software head proxy.  This function performs no
    file access so callers can validate a manifest before resolving assets.
    """
    if value is None:
        return dict(_DEFAULT_HEAD)
    if not isinstance(value, Mapping):
        raise ReceiveConfigurationError("receive_simulation must be an object")
    raw = dict(value)
    model = raw.get("model", "head")
    if model == "head":
        _reject_unknown_fields(raw, _HEAD_FIELDS)
        azimuth = _finite_number(
            raw.get("azimuth_deg", _DEFAULT_HEAD["azimuth_deg"]),
            "azimuth_deg",
        )
        radius = _finite_number(
            raw.get("head_radius_m", _DEFAULT_HEAD["head_radius_m"]),
            "head_radius_m",
        )
        speed = _finite_number(
            raw.get("speed_of_sound_mps", _DEFAULT_HEAD["speed_of_sound_mps"]),
            "speed_of_sound_mps",
        )
        if not -180.0 <= azimuth <= 180.0:
            raise ReceiveConfigurationError(
                "azimuth_deg must be between -180 and 180"
            )
        if not 0.04 <= radius <= 0.15:
            raise ReceiveConfigurationError(
                "head_radius_m must be between 0.04 and 0.15"
            )
        if not 300.0 <= speed <= 380.0:
            raise ReceiveConfigurationError(
                "speed_of_sound_mps must be between 300 and 380"
            )
        return {
            "model": "head",
            "azimuth_deg": azimuth,
            "head_radius_m": radius,
            "speed_of_sound_mps": speed,
            "measurement_point": _measurement_point(
                raw.get("measurement_point"),
                default=_DEFAULT_HEAD["measurement_point"],
            ),
            "provenance": _nonempty_string(
                raw.get("provenance", _DEFAULT_HEAD["provenance"]),
                "provenance",
            ),
        }
    if model != "paired_ir":
        raise ReceiveConfigurationError("model must be 'head' or 'paired_ir'")

    _reject_unknown_fields(raw, _PAIRED_IR_FIELDS)
    missing = sorted(_PAIRED_IR_FIELDS - set(raw))
    if missing:
        raise ReceiveConfigurationError(
            "paired_ir configuration is missing: " + ", ".join(missing)
        )
    ir_kind = _nonempty_string(raw["ir_kind"], "ir_kind")
    if ir_kind not in IR_KINDS:
        raise ReceiveConfigurationError(
            "ir_kind must be one of: " + ", ".join(sorted(IR_KINDS))
        )
    return {
        "model": "paired_ir",
        "left_ir": safe_receive_ir_path(raw["left_ir"], "left_ir"),
        "right_ir": safe_receive_ir_path(raw["right_ir"], "right_ir"),
        "ir_kind": ir_kind,
        "measurement_point": _measurement_point(raw["measurement_point"]),
        "provenance": _nonempty_string(raw["provenance"], "provenance"),
    }


def receive_configuration_sha256(configuration: Any, ir_sha256: Any) -> str:
    """Bind the complete acoustic configuration and its original IR assets."""
    config = normalize_receive_configuration(configuration)
    expected_roles = {"left_ir", "right_ir"} if config["model"] == "paired_ir" else set()
    if not isinstance(ir_sha256, Mapping) or set(ir_sha256) != expected_roles:
        raise ReceiveConfigurationError("IR hashes do not match the acoustic model")
    hashes = dict(ir_sha256)
    if any(
        not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
        for value in hashes.values()
    ):
        raise ReceiveConfigurationError("IR SHA256 must contain 64 lowercase hex characters")
    payload = {"configuration": config, "ir_sha256": hashes}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IR_KINDS",
    "MEASUREMENT_POINTS",
    "ReceiveConfigurationError",
    "normalize_receive_configuration",
    "receive_configuration_sha256",
    "safe_receive_ir_path",
]
