"""Fail-closed validation for generated TS 103 281 Annex C evidence.

This module validates evidence only.  A ``passed`` result means that the JSON
rows, declared counts, identity hashes, and recomputed tolerance gate agree.
It does not execute Model A and is not itself evidence of algorithm quality.

Returned status contract
------------------------
``passed``
    Full 48-condition evidence passed every structural, identity, and
    tolerance check.
``failed_tolerance_gate``
    Evidence is structurally valid and identity-bound, but recomputation does
    not reach 48/48 conditions within exactly 0.1 MOS.
``stale``
    Identity fields are missing or differ from the current implementation,
    dependency, or model hashes supplied by the caller.
``invalid_evidence``
    The document is malformed, internally inconsistent, declares a result
    different from recomputation, or attempts to bypass the strict gate.

Only ``status == "passed"`` sets ``conformant`` to true.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any


EXPECTED_CONDITIONS = frozenset(f"C{number:02d}" for number in range(1, 49))
EXPECTED_METRICS = ("S-MOS", "N-MOS", "G-MOS")
EXPECTED_TOLERANCE_MOS = 0.1
_TOLERANCE_FLOAT_SLACK = 1e-12
_PRESSURE_GAIN_DB = 15.0
_RMS_ROUNDING_TOLERANCE_DB = 0.005 + 1e-9
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_CONSISTENCY_ABS_TOLERANCE = 1e-9
_CONSISTENCY_REL_TOLERANCE = 1e-12


class _EvidenceProblem(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _historical(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_count": copy.deepcopy(evidence.get("condition_count")),
        "passed_conditions": copy.deepcopy(
            evidence.get("passed_conditions")
        ),
        "conformant": copy.deepcopy(evidence.get("conformant")),
        "tolerance_mos": copy.deepcopy(evidence.get("tolerance_mos")),
        "implementation_sha256": copy.deepcopy(
            evidence.get("implementation_sha256")
        ),
        "summary": copy.deepcopy(evidence.get("summary")),
    }


def _declared(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_count": evidence.get("condition_count"),
        "passed_conditions": evidence.get("passed_conditions"),
        "conformant": evidence.get("conformant"),
        "tolerance_mos": evidence.get("tolerance_mos"),
        "conformance_scope": evidence.get("conformance_scope"),
        "requested_limit": evidence.get("requested_limit"),
    }


def _result(
    status: str,
    *,
    evidence: Mapping[str, Any],
    reason: str | None,
    recomputed: Mapping[str, Any] | None = None,
    current_identity_matches: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "conformant": status == "passed",
        "evidence_valid": status in {"passed", "failed_tolerance_gate"},
        "failed_reason": reason,
        "declared": _declared(evidence),
        "historical": _historical(evidence),
        "recomputed": dict(recomputed or {}),
        "current_identity_matches": dict(
            current_identity_matches or {}
        ),
    }


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _valid_hash_map(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(key, str)
            and bool(key)
            and _valid_sha(digest)
            for key, digest in value.items()
        )
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def within_annex_c_tolerance(difference: Any) -> bool:
    """Apply the MOS threshold with a small floating-point arithmetic allowance."""
    return _finite_number(difference) and abs(float(difference)) <= (
        EXPECTED_TOLERANCE_MOS + _TOLERANCE_FLOAT_SLACK
    )


def _count(evidence: Mapping[str, Any], field: str, expected: int) -> None:
    value = evidence.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise _EvidenceProblem(
            "invalid_evidence",
            f"{field} must be exactly {expected}, got {value!r}",
        )


def _identity_check(
    evidence: Mapping[str, Any],
    *,
    implementation_sha256: str,
    dependency_sha256: Mapping[str, str],
    model_sha256: Mapping[str, str],
    verifier_sha256: str,
) -> dict[str, bool]:
    required_fields = (
        "implementation_sha256",
        "dependency_sha256",
        "model_sha256",
        "verifier_sha256",
        "identity_scope",
    )
    missing = [field for field in required_fields if field not in evidence]
    if missing:
        raise _EvidenceProblem(
            "stale",
            f"evidence lacks full identity fields: {missing}",
        )
    if (
        not _valid_sha(evidence.get("implementation_sha256"))
        or not _valid_hash_map(evidence.get("dependency_sha256"))
        or not _valid_hash_map(evidence.get("model_sha256"))
        or not _valid_sha(evidence.get("verifier_sha256"))
        or not isinstance(evidence.get("identity_scope"), str)
        or not evidence["identity_scope"]
    ):
        raise _EvidenceProblem(
            "invalid_evidence",
            "evidence identity fields are malformed",
        )
    if (
        not _valid_sha(implementation_sha256)
        or not _valid_hash_map(dependency_sha256)
        or not _valid_hash_map(model_sha256)
        or not _valid_sha(verifier_sha256)
    ):
        raise _EvidenceProblem(
            "invalid_evidence",
            "caller-supplied current identity is malformed",
        )
    matches = {
        "implementation": (
            evidence["implementation_sha256"] == implementation_sha256
        ),
        "dependencies": (
            dict(evidence["dependency_sha256"]) == dict(dependency_sha256)
        ),
        "models": dict(evidence["model_sha256"]) == dict(model_sha256),
        "verifier": evidence["verifier_sha256"] == verifier_sha256,
    }
    if not all(matches.values()):
        raise _EvidenceProblem(
            "stale",
            f"evidence identity differs from current identity: {matches}",
        )
    return matches


def _recompute_rows(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise _EvidenceProblem("invalid_evidence", "rows must be a list")
    if len(rows) != 48:
        raise _EvidenceProblem(
            "invalid_evidence",
            f"rows must contain exactly 48 conditions, got {len(rows)}",
        )
    conditions: set[str] = set()
    finite_scores = 0
    passed_conditions = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _EvidenceProblem(
                "invalid_evidence",
                f"row {index} must be an object",
            )
        condition = row.get("condition")
        if not isinstance(condition, str) or condition not in EXPECTED_CONDITIONS:
            raise _EvidenceProblem(
                "invalid_evidence",
                f"row {index} has invalid condition {condition!r}",
            )
        if condition in conditions:
            raise _EvidenceProblem(
                "invalid_evidence",
                f"duplicate condition {condition}",
            )
        conditions.add(condition)
        value_groups = {
            field: row.get(field)
            for field in ("expected", "actual", "difference")
        }
        for field, values in value_groups.items():
            if not isinstance(values, Mapping):
                raise _EvidenceProblem(
                    "invalid_evidence",
                    f"{condition}.{field} must be an object",
                )
            if set(values) != set(EXPECTED_METRICS):
                raise _EvidenceProblem(
                    "invalid_evidence",
                    f"{condition}.{field} must contain exactly "
                    f"{list(EXPECTED_METRICS)}",
                )
            for metric in EXPECTED_METRICS:
                if not _finite_number(values[metric]):
                    raise _EvidenceProblem(
                        "invalid_evidence",
                        f"{condition}.{field}.{metric} must be finite",
                    )
        row_within = True
        for metric in EXPECTED_METRICS:
            expected = float(value_groups["expected"][metric])
            actual = float(value_groups["actual"][metric])
            difference = float(value_groups["difference"][metric])
            computed_difference = actual - expected
            if not math.isclose(
                difference,
                computed_difference,
                rel_tol=_CONSISTENCY_REL_TOLERANCE,
                abs_tol=_CONSISTENCY_ABS_TOLERANCE,
            ):
                raise _EvidenceProblem(
                    "invalid_evidence",
                    f"{condition}.{metric} difference does not equal "
                    "actual - expected",
                )
            finite_scores += 1
            if not within_annex_c_tolerance(computed_difference):
                row_within = False
        declared_within = row.get("within_0_1")
        if not isinstance(declared_within, bool) or declared_within != row_within:
            raise _EvidenceProblem(
                "invalid_evidence",
                f"{condition}.within_0_1 is inconsistent",
            )
        passed_conditions += int(row_within)
    if conditions != EXPECTED_CONDITIONS:
        missing = sorted(EXPECTED_CONDITIONS - conditions)
        raise _EvidenceProblem(
            "invalid_evidence",
            f"rows are missing conditions: {missing}",
        )
    recomputed = {
        "condition_count": len(conditions),
        "score_count": len(rows) * len(EXPECTED_METRICS),
        "finite_score_count": finite_scores,
        "passed_conditions": passed_conditions,
    }
    return recomputed, passed_conditions == 48


def _validate_pressure_calibration(evidence: Mapping[str, Any]) -> None:
    calibration = evidence.get("pressure_calibration")
    if not isinstance(calibration, Mapping):
        raise _EvidenceProblem("invalid_evidence", "pressure_calibration must be an object")
    if not isinstance(calibration.get("method"), str) or not calibration["method"]:
        raise _EvidenceProblem("invalid_evidence", "pressure_calibration method is required")
    if calibration.get("uses_mos_targets") is not False:
        raise _EvidenceProblem("invalid_evidence", "pressure_calibration must not use MOS targets")
    if not _finite_number(calibration.get("common_gain_db")) or calibration["common_gain_db"] != _PRESSURE_GAIN_DB:
        raise _EvidenceProblem("invalid_evidence", "pressure_calibration common gain is invalid")
    _count(calibration, "verified_conditions", 48)
    errors = []
    for row in evidence["rows"]:
        details = row.get("input_calibration")
        if not isinstance(details, Mapping):
            raise _EvidenceProblem("invalid_evidence", f"{row['condition']} input_calibration must be an object")
        if (
            not _finite_number(details.get("common_gain_db"))
            or details["common_gain_db"] != _PRESSURE_GAIN_DB
            or details.get("output_unit") != "Pa"
            or details.get("applied_to_both_channels") is not True
            or details.get("uses_mos_targets") is not False
        ):
            raise _EvidenceProblem("invalid_evidence", f"{row['condition']} input_calibration is invalid")
        error = details.get("rms_error_db")
        if not _finite_number(error) or abs(float(error)) > _RMS_ROUNDING_TOLERANCE_DB:
            raise _EvidenceProblem("invalid_evidence", f"{row['condition']} calibration RMS error is invalid")
        errors.append(abs(float(error)))
    declared_max = calibration.get("max_rms_error_db")
    if (
        not _finite_number(declared_max)
        or float(declared_max) < 0
        or not math.isclose(float(declared_max), max(errors), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise _EvidenceProblem("invalid_evidence", "pressure_calibration maximum RMS error is inconsistent")


def validate_annex_c_evidence(
    evidence: dict,
    *,
    implementation_sha256: str,
    dependency_sha256: dict[str, str],
    model_sha256: dict[str, str],
    verifier_sha256: str,
) -> dict:
    """Validate Annex C evidence without reading model or source files."""
    document: Mapping[str, Any] = (
        evidence if isinstance(evidence, Mapping) else {}
    )
    if not isinstance(evidence, Mapping):
        return _result(
            "invalid_evidence",
            evidence=document,
            reason="evidence must be an object",
        )

    current_matches: dict[str, bool] = {}
    try:
        current_matches = _identity_check(
            document,
            implementation_sha256=implementation_sha256,
            dependency_sha256=dependency_sha256,
            model_sha256=model_sha256,
            verifier_sha256=verifier_sha256,
        )
        _count(document, "condition_count", 48)
        _count(document, "expected_condition_count", 48)
        _count(document, "independent_condition_count", 48)
        _count(document, "score_count", 144)
        _count(document, "finite_score_count", 144)
        if document.get("conformance_scope") != "full":
            raise _EvidenceProblem(
                "invalid_evidence",
                "conformance_scope must be 'full'",
            )
        if document.get("requested_limit") is not None:
            raise _EvidenceProblem(
                "invalid_evidence",
                "requested_limit must be null for a full run",
            )
        if document.get("original_scores_only") is not True:
            raise _EvidenceProblem(
                "invalid_evidence",
                "original_scores_only must be true",
            )
        tolerance = document.get("tolerance_mos")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or float(tolerance) != EXPECTED_TOLERANCE_MOS
        ):
            raise _EvidenceProblem(
                "invalid_evidence",
                "tolerance_mos must be exactly 0.1",
            )
        if document.get("identity_verified_during_run") is not True:
            raise _EvidenceProblem(
                "invalid_evidence",
                "identity_verified_during_run must be true",
            )

        recomputed, tolerance_conformant = _recompute_rows(document)
        _validate_pressure_calibration(document)
        declared_passed = document.get("passed_conditions")
        if (
            isinstance(declared_passed, bool)
            or not isinstance(declared_passed, int)
            or declared_passed != recomputed["passed_conditions"]
        ):
            raise _EvidenceProblem(
                "invalid_evidence",
                "declared passed_conditions disagrees with row recomputation",
            )
        declared_conformant = document.get("conformant")
        if not isinstance(declared_conformant, bool):
            raise _EvidenceProblem(
                "invalid_evidence",
                "conformant declaration must be boolean",
            )
        recomputed_conformant = bool(
            tolerance_conformant
            and recomputed["condition_count"] == 48
            and recomputed["finite_score_count"] == 144
        )
        if declared_conformant != recomputed_conformant:
            raise _EvidenceProblem(
                "invalid_evidence",
                "conformant declaration disagrees with recomputation",
            )
        if not recomputed_conformant:
            return _result(
                "failed_tolerance_gate",
                evidence=document,
                reason=(
                    f"{recomputed['passed_conditions']}/48 conditions "
                    "reproduced all three original MOS values within 0.1"
                ),
                recomputed={
                    **recomputed,
                    "conformant": False,
                },
                current_identity_matches=current_matches,
            )
        if document.get("failed_reason") is not None:
            raise _EvidenceProblem(
                "invalid_evidence",
                "passing evidence must have failed_reason=null",
            )
        return _result(
            "passed",
            evidence=document,
            reason=None,
            recomputed={
                **recomputed,
                "conformant": True,
            },
            current_identity_matches=current_matches,
        )
    except _EvidenceProblem as problem:
        return _result(
            problem.status,
            evidence=document,
            reason=problem.reason,
            current_identity_matches=current_matches,
        )


__all__ = ["validate_annex_c_evidence", "within_annex_c_tolerance"]
