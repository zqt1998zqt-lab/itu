from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from server.acoustic import acoustic_stress_models, acoustic_stress_store
from server.acoustic.acoustic_stress_profiles import PROFILES


PROJECT = Path(__file__).resolve().parents[1]
BUILD = PROJECT / "build"
REPRESENTATIVE_ROOT = BUILD / "stress-vcp-representatives-d7s26mao"
FULL_ROOT = BUILD / "stress-full-execution-OsA0Iw"


def _diagnostics_expected(diagnostics: dict) -> dict:
    return {
        "expected_case_id": diagnostics["case_id"],
        "expected_tx_sha256": diagnostics["tx_sha256"],
        "expected_scope": diagnostics["scope"],
        "expected_mode": diagnostics["mode"],
        "expected_category": diagnostics["category"],
        "expected_segments": [
            {
                "label": segment["label"],
                "start_s": segment["start_s"],
                "end_s": segment["end_s"],
            }
            for segment in diagnostics["segments"]
        ],
    }


def test_immutable_stores_and_landed_v2_evidence_validate() -> None:
    if not (BUILD / "acoustic-qualification/acoustic-stress-datasets").is_dir():
        pytest.skip("acoustic stress stores are not present")
    for profile in PROFILES.values():
        dataset = acoustic_stress_store.load_stress_dataset(profile.dataset_id)
        assert dataset["case_count"] == 100
        assert dataset["observation_count"] == 100

    representative_files = [
        path
        for path in sorted(REPRESENTATIVE_ROOT.glob("*.json"))
        if path.name != "verification.json"
    ]
    assert len(representative_files) == 80
    for path in representative_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = payload["diagnostics"]
        acoustic_stress_models.validate_stress_diagnostics(
            diagnostics,
            **_diagnostics_expected(diagnostics),
        )

    full_count = 0
    for name in ("echo-batch.json", "double-talk-batch.json"):
        snapshot = json.loads((FULL_ROOT / name).read_text(encoding="utf-8"))
        acoustic_stress_store.validate_activity_provenance(
            snapshot["cases"][0]["provenance"],
            scope=snapshot["test_scope"],
            mode=snapshot["selected_mode"],
            require_source_flags=True,
        )
        for case in snapshot["cases"]:
            diagnostics = case["stress_diagnostics"]
            acoustic_stress_models.validate_stress_diagnostics(
                diagnostics,
                **_diagnostics_expected(diagnostics),
            )
            full_count += 1
    assert full_count == 200


def test_v2_activity_valid_true_cannot_be_zero_or_impossible() -> None:
    path = REPRESENTATIVE_ROOT / "double_talk-NB-bt-balanced.json"
    if not path.is_file():
        pytest.skip("representative v2 evidence is not present")
    diagnostics = json.loads(path.read_text(encoding="utf-8"))["diagnostics"]
    expected = _diagnostics_expected(diagnostics)
    acoustic_stress_models.validate_stress_diagnostics(
        diagnostics,
        **expected,
    )
    for changes in (
        {
            "near_active_samples": 0,
            "far_active_samples": 0,
            "joint_active_samples": 0,
            "near_coverage": 0.0,
            "far_coverage": 0.0,
            "joint_coverage": 0.0,
        },
        {
            "near_active_samples": 0,
            "far_active_samples": 1,
            "joint_active_samples": 100_000_000,
            "near_coverage": 1.0,
            "far_coverage": 0.0,
            "joint_coverage": 1.0,
        },
    ):
        tampered = copy.deepcopy(diagnostics)
        tampered["segments"][0]["activity"].update(changes)
        tampered["segments"][0]["activity"]["valid"] = True
        with pytest.raises(ValueError):
            acoustic_stress_models.validate_stress_diagnostics(
                tampered,
                **expected,
            )


def test_echo_path_tap_tamper_is_rejected_without_recomputing() -> None:
    assessment = next(
        (
            path
            for path in REPRESENTATIVE_ROOT.rglob("assessment.json")
            if "echo-WB-wired-delay_change-01-stress-v1" in str(path)
        ),
        None,
    )
    if assessment is None:
        pytest.skip("echo path assessment is not present")
    provenance = json.loads(assessment.read_text(encoding="utf-8"))[
        "provenance"
    ]
    acoustic_stress_store.validate_activity_provenance(
        provenance,
        scope="echo",
        mode="WB-wired",
        require_source_flags=False,
    )

    delay_tamper = copy.deepcopy(provenance)
    delay_tamper["render_metadata"]["echo_path_segments"][0]["paths"][0][
        "delay_ms"
    ] = 999.0
    with pytest.raises(ValueError):
        acoustic_stress_store.validate_activity_provenance(
            delay_tamper,
            scope="echo",
            mode="WB-wired",
            require_source_flags=False,
        )

    tap_tamper = copy.deepcopy(provenance)
    tap_tamper["render_metadata"]["echo_path_segments"][0]["paths"][0].pop(
        "tap_samples"
    )
    with pytest.raises(ValueError):
        acoustic_stress_store.validate_activity_provenance(
            tap_tamper,
            scope="echo",
            mode="WB-wired",
            require_source_flags=False,
        )


# Keep the historical diagnostics and reports above byte-for-byte unchanged,
# while making every live store lookup in this module use the current R2
# identities.
from server.acoustic.dataset_revisions_v2 import (
    MODES as CURRENT_STRESS_MODES,
    STRESS_SCOPES as CURRENT_STRESS_SCOPES,
    revision_stress_profile,
)

PROFILES = {
    (scope, mode): revision_stress_profile(scope, mode)
    for scope in CURRENT_STRESS_SCOPES
    for mode in CURRENT_STRESS_MODES
}
