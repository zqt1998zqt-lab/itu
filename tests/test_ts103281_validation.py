from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from server.quality.ts103281_validation import validate_annex_c_evidence


PROJECT = Path(__file__).resolve().parents[1]
CURRENT_IMPLEMENTATION = "a" * 64
CURRENT_VERIFIER = "e" * 64
CURRENT_DEPENDENCIES = {
    "server/quality/ts103281.py": "b" * 64,
    "server/quality/ts103281_activity.py": "c" * 64,
    "server/quality/ts103281_reference.py": "d" * 64,
}
CURRENT_MODELS = {
    "server/quality/assets/ts103281/S-MOS.model": "1" * 64,
    "server/quality/assets/ts103281/N-MOS.model": "2" * 64,
    "server/quality/assets/ts103281/G-MOS.model": "3" * 64,
}


def valid_evidence() -> dict:
    rows = []
    for number in range(1, 49):
        rows.append(
            {
                "condition": f"C{number:02d}",
                "expected": {
                    "S-MOS": 3.0,
                    "N-MOS": 3.0,
                    "G-MOS": 3.0,
                },
                "actual": {
                    "S-MOS": 3.05,
                    "N-MOS": 3.05,
                    "G-MOS": 3.05,
                },
                "difference": {
                    "S-MOS": 0.05,
                    "N-MOS": 0.05,
                    "G-MOS": 0.05,
                },
                "within_0_1": True,
                "input_calibration": {
                    "common_gain_db": 15.0,
                    "rms_error_db": 0.004,
                    "applied_to_both_channels": True,
                    "uses_mos_targets": False,
                    "output_unit": "Pa",
                },
            }
        )
    return {
        "standard": "ETSI TS 103 281 V1.3.2 Annex C",
        "condition_count": 48,
        "expected_condition_count": 48,
        "independent_condition_count": 48,
        "score_count": 144,
        "finite_score_count": 144,
        "tolerance_mos": 0.1,
        "conformance_scope": "full",
        "requested_limit": None,
        "conformant": True,
        "passed_conditions": 48,
        "implementation_sha256": CURRENT_IMPLEMENTATION,
        "dependency_sha256": copy.deepcopy(CURRENT_DEPENDENCIES),
        "model_sha256": copy.deepcopy(CURRENT_MODELS),
        "verifier_sha256": CURRENT_VERIFIER,
        "identity_scope": "synthetic full identity",
        "identity_verified_during_run": True,
        "original_scores_only": True,
        "pressure_calibration": {
            "method": (
                "Annex C Table C.1 full-file unweighted RMS calibration"
            ),
            "common_gain_db": 15.0,
            "verified_conditions": 48,
            "max_rms_error_db": 0.004,
            "uses_mos_targets": False,
        },
        "failed_reason": None,
        "summary": {
            "S-MOS": {"mae": 0.05, "rmse": 0.05, "max": 0.05},
            "N-MOS": {"mae": 0.05, "rmse": 0.05, "max": 0.05},
            "G-MOS": {"mae": 0.05, "rmse": 0.05, "max": 0.05},
        },
        "rows": rows,
    }


def validate(
    evidence: dict,
    *,
    implementation_sha256: str = CURRENT_IMPLEMENTATION,
    dependency_sha256: dict[str, str] | None = None,
    model_sha256: dict[str, str] | None = None,
    verifier_sha256: str = CURRENT_VERIFIER,
) -> dict:
    return validate_annex_c_evidence(
        evidence,
        implementation_sha256=implementation_sha256,
        dependency_sha256=(
            CURRENT_DEPENDENCIES
            if dependency_sha256 is None
            else dependency_sha256
        ),
        model_sha256=CURRENT_MODELS if model_sha256 is None else model_sha256,
        verifier_sha256=verifier_sha256,
    )


def test_complete_synthetic_evidence_passes_validator_not_algorithm():
    # This proves the validator contract only.  Synthetic rows are not Model A
    # execution evidence and must never be published as Annex C conformance.
    evidence = valid_evidence()
    original = copy.deepcopy(evidence)

    result = validate(evidence)

    assert result["status"] == "passed"
    assert result["conformant"] is True
    assert result["evidence_valid"] is True
    assert result["failed_reason"] is None
    assert result["recomputed"] == {
        "condition_count": 48,
        "score_count": 144,
        "finite_score_count": 144,
        "passed_conditions": 48,
        "conformant": True,
    }
    assert result["current_identity_matches"] == {
        "implementation": True,
        "dependencies": True,
        "models": True,
        "verifier": True,
    }
    assert result["historical"]["condition_count"] == 48
    assert result["historical"]["summary"] == evidence["summary"]
    assert result["historical"]["summary"] is not evidence["summary"]
    assert evidence == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition_count", 47),
        ("expected_condition_count", 49),
        ("independent_condition_count", 47),
        ("score_count", 143),
        ("finite_score_count", 143),
    ],
)
def test_forged_top_level_counts_fail_closed(field: str, value: int):
    evidence = valid_evidence()
    evidence[field] = value

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert field in result["failed_reason"]


def test_duplicate_condition_cannot_be_hidden_by_declared_counts():
    evidence = valid_evidence()
    evidence["rows"][-1]["condition"] = "C01"

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "duplicate condition C01" in result["failed_reason"]


def test_subset_cannot_claim_full_conformance():
    evidence = valid_evidence()
    evidence["rows"] = evidence["rows"][:24]
    evidence["condition_count"] = 24
    evidence["expected_condition_count"] = 24
    evidence["independent_condition_count"] = 24
    evidence["score_count"] = 72
    evidence["finite_score_count"] = 72
    evidence["passed_conditions"] = 24
    evidence["conformance_scope"] = "subset"

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "condition_count must be exactly 48" in result["failed_reason"]


def test_nonfinite_original_score_is_invalid_not_a_pass():
    evidence = valid_evidence()
    evidence["rows"][0]["actual"]["S-MOS"] = float("inf")
    evidence["rows"][0]["difference"]["S-MOS"] = float("inf")

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "must be finite" in result["failed_reason"]


def test_tampered_difference_is_recomputed_from_actual_and_expected():
    evidence = valid_evidence()
    evidence["rows"][0]["difference"]["N-MOS"] = 0.01

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "difference does not equal actual - expected" in result[
        "failed_reason"
    ]


def test_declared_passed_count_cannot_override_rows():
    evidence = valid_evidence()
    evidence["passed_conditions"] = 999

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "declared passed_conditions" in result["failed_reason"]


def test_declared_conformant_true_cannot_hide_a_row_failure():
    evidence = valid_evidence()
    evidence["rows"][0]["actual"]["G-MOS"] = 3.2
    evidence["rows"][0]["difference"]["G-MOS"] = 0.2
    evidence["rows"][0]["within_0_1"] = False
    evidence["passed_conditions"] = 47
    evidence["conformant"] = True

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "conformant declaration disagrees" in result["failed_reason"]


def test_consistent_failed_tolerance_is_reported_not_mislabeled_invalid():
    evidence = valid_evidence()
    evidence["rows"][0]["actual"]["G-MOS"] = 3.2
    evidence["rows"][0]["difference"]["G-MOS"] = 0.2
    evidence["rows"][0]["within_0_1"] = False
    evidence["passed_conditions"] = 47
    evidence["conformant"] = False
    evidence["failed_reason"] = "47/48"

    result = validate(evidence)

    assert result["status"] == "failed_tolerance_gate"
    assert result["conformant"] is False
    assert result["evidence_valid"] is True
    assert result["recomputed"]["passed_conditions"] == 47


def test_tolerance_above_exactly_point_one_is_invalid():
    evidence = valid_evidence()
    evidence["tolerance_mos"] = 0.1000001

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "tolerance_mos must be exactly 0.1" in result["failed_reason"]


def test_dependency_or_model_identity_change_is_stale():
    dependency = copy.deepcopy(CURRENT_DEPENDENCIES)
    dependency["server/quality/ts103281.py"] = "f" * 64
    dependency_result = validate(
        valid_evidence(),
        dependency_sha256=dependency,
    )
    model = copy.deepcopy(CURRENT_MODELS)
    model["server/quality/assets/ts103281/S-MOS.model"] = "f" * 64
    model_result = validate(valid_evidence(), model_sha256=model)

    assert dependency_result["status"] == "stale"
    assert dependency_result["conformant"] is False
    assert model_result["status"] == "stale"
    assert model_result["conformant"] is False


def test_old_evidence_without_full_identity_is_stale_and_historical():
    evidence = json.loads(
        (
            PROJECT
            / "server/quality/assets/ts103281/annex-c-validation.json"
        ).read_text(encoding="utf-8")
    )

    result = validate(evidence)

    assert result["status"] == "stale"
    assert result["conformant"] is False
    assert result["evidence_valid"] is False
    assert "lacks full identity fields" in result["failed_reason"]
    assert result["historical"]["condition_count"] == 48
    assert result["historical"]["passed_conditions"] == 0
    assert result["historical"]["conformant"] is False
    assert result["historical"]["summary"] == evidence["summary"]


def test_identity_verification_flag_is_required():
    evidence = valid_evidence()
    evidence["identity_verified_during_run"] = False

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert "identity_verified_during_run" in result["failed_reason"]


@pytest.mark.parametrize("condition", [[], {}])
def test_unhashable_condition_is_invalid_evidence(condition):
    evidence = valid_evidence()
    evidence["rows"][0]["condition"] = condition

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert result["evidence_valid"] is False


def test_verifier_identity_change_is_stale():
    changed_evidence = valid_evidence()
    changed_evidence["verifier_sha256"] = "f" * 64

    evidence_result = validate(changed_evidence)
    current_result = validate(
        valid_evidence(),
        verifier_sha256="f" * 64,
    )

    assert evidence_result["status"] == "stale"
    assert evidence_result["conformant"] is False
    assert current_result["status"] == "stale"
    assert current_result["conformant"] is False


@pytest.mark.parametrize("location", ["report", "row"])
def test_missing_pressure_calibration_is_invalid(location: str):
    evidence = valid_evidence()
    if location == "report":
        evidence.pop("pressure_calibration")
    else:
        evidence["rows"][0].pop("input_calibration")

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False
    assert result["evidence_valid"] is False


@pytest.mark.parametrize("location", ["report", "row"])
def test_wrong_pressure_calibration_gain_is_invalid(location: str):
    evidence = valid_evidence()
    calibration = (
        evidence["pressure_calibration"]
        if location == "report"
        else evidence["rows"][0]["input_calibration"]
    )
    calibration["common_gain_db"] = 14.0

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("report", float("nan")),
        ("report", 0.00400001),
        ("row", float("nan")),
        ("row", 0.00400001),
    ],
)
def test_invalid_pressure_calibration_error_is_invalid(
    location: str,
    value: float,
):
    evidence = valid_evidence()
    if location == "report":
        evidence["pressure_calibration"]["max_rms_error_db"] = value
    else:
        evidence["rows"][0]["input_calibration"]["rms_error_db"] = value

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False


@pytest.mark.parametrize("location", ["report", "row"])
def test_mos_target_pressure_calibration_is_invalid(location: str):
    evidence = valid_evidence()
    calibration = (
        evidence["pressure_calibration"]
        if location == "report"
        else evidence["rows"][0]["input_calibration"]
    )
    calibration["uses_mos_targets"] = True

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False


@pytest.mark.parametrize("output_unit", [None, "dBFS"])
def test_input_calibration_output_unit_must_be_pa(output_unit):
    evidence = valid_evidence()
    calibration = evidence["rows"][0]["input_calibration"]
    if output_unit is None:
        calibration.pop("output_unit")
    else:
        calibration["output_unit"] = output_unit

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False


def test_pressure_calibration_condition_count_must_match_rows():
    evidence = valid_evidence()
    evidence["pressure_calibration"]["verified_conditions"] = 47

    result = validate(evidence)

    assert result["status"] == "invalid_evidence"
    assert result["conformant"] is False


def test_tolerance_boundary_accepts_floating_point_roundoff():
    evidence = valid_evidence()
    actual = 3.1
    difference = actual - 3.0
    evidence["rows"][0]["actual"]["S-MOS"] = actual
    evidence["rows"][0]["difference"]["S-MOS"] = difference

    result = validate(evidence)

    assert difference > 0.1
    assert result["status"] == "passed"
    assert result["conformant"] is True


def test_tolerance_rejects_a_real_overrun():
    evidence = valid_evidence()
    actual = 3.10000001
    difference = actual - 3.0
    evidence["rows"][0]["actual"]["S-MOS"] = actual
    evidence["rows"][0]["difference"]["S-MOS"] = difference
    evidence["rows"][0]["within_0_1"] = False
    evidence["passed_conditions"] = 47
    evidence["conformant"] = False
    evidence["failed_reason"] = "47/48"

    result = validate(evidence)

    assert difference > 0.1
    assert result["status"] == "failed_tolerance_gate"
    assert result["conformant"] is False
    assert result["evidence_valid"] is True
    assert result["recomputed"]["passed_conditions"] == 47
