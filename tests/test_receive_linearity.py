import copy
import json

import pytest

from server.acoustic.receive_linearity import evaluate_rlr_linearity_group


BANDS = {
    "500": -40.0,
    "1000": -35.0,
    "2000": -38.0,
}


def run(
    level_id: str,
    bands: dict[str, float] | None = None,
    *,
    delta_db: float | None = None,
    mode: str = "WB-wireless",
) -> dict:
    nominal = BANDS if bands is None else bands
    level_delta = {"nominal": 0.0, "offset_-5": -5.0, "offset_+5": 5.0}[level_id]
    if delta_db is not None:
        level_delta = delta_db
    return {
        "run_id": f"run-{level_id}",
        "level_id": level_id,
        "mode": mode,
        "status": "completed",
        "output_normalisation": {"normalised": False},
        "clipping": False,
        "conditions": {
            "nominal_volume": "volume-10",
            "fixture": {"fixture_id": "hats-a", "position": "driver"},
            "ir": {
                "left_sha256": "1" * 64,
                "right_sha256": "2" * 64,
            },
            "head_model": {"model": "head-a", "version": 1},
            "sample_rate_hz": 48000,
            "duration_s": 8.0,
            "active_speech_window_s": [1.0, 7.0],
        },
        "metrics": {
            "third_octave_band_levels_db": {
                "binaural": {
                    frequency: level + level_delta
                    for frequency, level in nominal.items()
                }
            }
        },
    }


def group(mode: str = "WB-wireless") -> list[dict]:
    return [
        run("nominal", mode=mode),
        run("offset_-5", mode=mode),
        run("offset_+5", mode=mode),
    ]


def report_group(
    mode: str = "WB-wireless",
    *,
    window_applied: bool = True,
    analysis_window_s: list[float] | None = None,
) -> list[dict]:
    runs = group(mode)
    window = [1.0, 7.0] if analysis_window_s is None else analysis_window_s
    for item in runs:
        item["evaluation_id"] = f"evaluation-{item['level_id']}"
        item["metrics"]["window_applied"] = window_applied
        item["metrics"]["analysis_window_s"] = window
    return runs


def test_ideal_linearity_has_near_zero_residual_and_engineering_scope() -> None:
    result = evaluate_rlr_linearity_group(group())

    assert result["status"] == "pass"
    assert result["group_valid"] is True
    assert result["items"]["B5"]["receive_level_linearity_residual_db"] == pytest.approx(0.0)
    assert result["items"]["B6"]["receive_level_linearity_residual_db"] == pytest.approx(0.0)
    assert result["items"]["B5"]["within_limit"] is True
    assert result["items"]["B6"]["within_limit"] is True
    assert result["evidence_level"] == "engineering_proxy"
    assert result["formal_rlr_linearity_available"] is False
    assert result["formal_compliance_claim_allowed"] is False
    assert result["formal_rlr_available"] is False
    assert result["window_source"] == "provided_band_levels"
    assert result["window_applied"] is False
    assert result["analysis_window_s"] is None
    assert all(
        evidence["window_applied"] is None
        for evidence in result["window_evidence"].values()
    )
    json.dumps(result, allow_nan=False)


def test_silent_group_cannot_turn_display_floor_into_linearity_residual() -> None:
    runs = group()
    for item in runs:
        item["metrics"]["third_octave_band_levels_db"]["binaural"] = {
            frequency: -120.0 for frequency in BANDS
        }
    result = evaluate_rlr_linearity_group(runs)
    assert result["group_valid"] is False
    assert "measurement floor" in " ".join(result["reasons"])
    assert all(item["receive_level_linearity_residual_db"] is None
               for item in result["items"].values())


def test_compression_or_agc_causes_residuals_to_exceed_limit() -> None:
    runs = [
        run("nominal"),
        run("offset_-5", delta_db=-3.5),
        run("offset_+5", delta_db=3.5),
    ]
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "fail"
    assert result["group_valid"] is True
    assert result["items"]["B5"]["receive_level_linearity_residual_db"] == pytest.approx(1.5)
    assert result["items"]["B6"]["receive_level_linearity_residual_db"] == pytest.approx(-1.5)
    assert result["items"]["B5"]["within_limit"] is False
    assert result["items"]["B6"]["within_limit"] is False
    assert any("outside" in reason for reason in result["reasons"])


def test_output_normalisation_rejects_the_whole_group() -> None:
    runs = group()
    runs[1]["output_normalisation"]["normalised"] = True
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert result["group_valid"] is False
    assert result["items"]["B5"]["receive_level_linearity_residual_db"] is None
    assert result["items"]["B6"]["receive_level_linearity_residual_db"] is None
    assert any("normalisation" in reason for reason in result["reasons"])


def test_clipping_rejects_the_whole_group() -> None:
    runs = group()
    runs[2]["clipping"] = True
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert result["items"]["B6"]["status"] == "invalid"
    assert any("clipping" in reason for reason in result["reasons"])


def test_missing_offset_level_is_rejected() -> None:
    result = evaluate_rlr_linearity_group(group()[:2])

    assert result["status"] == "invalid"
    assert result["group_valid"] is False
    assert any("exactly 3" in reason for reason in result["reasons"])
    assert any("offset_+5" in reason for reason in result["reasons"])


def test_mode_and_condition_mismatch_are_rejected() -> None:
    runs = group()
    runs[1]["mode"] = "NB-wireless"
    runs[2]["conditions"]["active_speech_window_s"] = [1.0, 6.0]
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert any(
        "condition mismatch for mode" in reason
        for reason in result["reasons"]
    )
    assert any(
        "condition mismatch for active_speech_window_s" in reason
        for reason in result["reasons"]
    )


def test_failed_run_status_rejects_the_group() -> None:
    runs = group()
    runs[0]["status"] = "failed"
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert any("not a successful status" in reason for reason in result["reasons"])


@pytest.mark.parametrize("bad_value", ["missing", "nan"])
def test_missing_or_nonfinite_band_is_rejected(bad_value: str) -> None:
    runs = group()
    binaural = runs[1]["metrics"]["third_octave_band_levels_db"]["binaural"]
    if bad_value == "missing":
        del binaural["1000"]
    else:
        binaural["1000"] = float("nan")
    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert any(
        "frequency bands differ" in reason or "non-finite" in reason
        for reason in result["reasons"]
    )
    json.dumps(result, allow_nan=False)


def test_explicit_energy_weights_and_canonical_hash_are_stable() -> None:
    conditions = {"band_weights": {"500": 1.0, "1000": 2.0, "2000": 0.5}}
    first = evaluate_rlr_linearity_group(
        group(),
        conditions,
        include_canonical_hash=True,
    )
    second = evaluate_rlr_linearity_group(
        list(reversed(copy.deepcopy(group()))),
        conditions,
        include_canonical_hash=True,
    )

    assert first["aggregation"]["weighting"] == "explicit"
    assert first["aggregation"]["weights"] == {
        "500": 1.0,
        "1000": 2.0,
        "2000": 0.5,
    }
    assert first["canonical_sha256"] == second["canonical_sha256"]
    assert len(first["canonical_sha256"]) == 64
    assert first["B5"]["item_id"] == "WB-wireless:B5"
    assert first["B6"]["item_id"] == "WB-wireless:B6"
    assert first["formal_compliance_claim_allowed"] is False


@pytest.mark.parametrize(
    ("mode", "minus_key", "plus_key"),
    [
        ("NB-wired", "B4", "B5"),
        ("NB-wireless", "B4", "B5"),
        ("NB-bt", "B4", "B5"),
        ("WB-wired", "B5", "B6"),
        ("WB-wireless", "B5", "B6"),
        ("WB-bt", "B5", "B6"),
    ],
)
def test_contract_identity_for_each_nb_wb_mode(
    mode: str,
    minus_key: str,
    plus_key: str,
) -> None:
    result = evaluate_rlr_linearity_group(group(mode))

    assert result["status"] == "pass"
    assert result["mode"] == mode
    assert result["standard_item_ids"] == [
        f"{mode}:{minus_key}",
        f"{mode}:{plus_key}",
    ]
    assert result["items"][minus_key]["item_id"] == f"{mode}:{minus_key}"
    assert result["items"][plus_key]["item_id"] == f"{mode}:{plus_key}"
    assert result[minus_key]["item_id"] == f"{mode}:{minus_key}"
    assert result[plus_key]["item_id"] == f"{mode}:{plus_key}"


@pytest.mark.parametrize("mode", ["SWB-wired", "SWB-wireless"])
def test_swb_has_no_rlr_linearity_contract_rows(mode: str) -> None:
    result = evaluate_rlr_linearity_group(group(mode))

    assert result["status"] == "invalid"
    assert result["mode"] == mode
    assert result["items"] == {}
    assert result["standard_item_ids"] == []
    assert "B5" not in result
    assert "B6" not in result
    assert any(
        "no receive rlr_linearity contract rows" in reason
        for reason in result["reasons"]
    )


def test_evaluation_reports_accept_matching_cropped_window_evidence() -> None:
    result = evaluate_rlr_linearity_group(report_group())

    assert result["status"] == "pass"
    assert result["window_source"] == "provided_cropped_band_levels"
    assert result["window_applied"] is True
    assert result["analysis_window_s"] == [1.0, 7.0]
    assert all(
        evidence["window_applied"] is True
        and evidence["analysis_window_s"] == [1.0, 7.0]
        for evidence in result["window_evidence"].values()
    )


def test_window_evidence_can_be_supplied_in_run_metadata() -> None:
    runs = report_group()
    for item in runs:
        item["metadata"] = {
            "window_applied": item["metrics"].pop("window_applied"),
            "analysis_window_s": item["metrics"].pop("analysis_window_s"),
        }

    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "pass"
    assert result["window_source"] == "provided_cropped_band_levels"
    assert result["window_applied"] is True
    assert result["analysis_window_s"] == [1.0, 7.0]


def test_evaluation_reports_without_window_evidence_are_rejected() -> None:
    runs = group()
    for index, item in enumerate(runs):
        item["evaluation_id"] = f"evaluation-{index}"

    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert result["window_applied"] is False
    assert result["analysis_window_s"] is None
    assert sum(
        "must declare window_applied=true and analysis_window_s" in reason
        for reason in result["reasons"]
    ) == 3


@pytest.mark.parametrize(
    ("window_applied", "analysis_window_s", "expected_reason"),
    [
        (False, [1.0, 7.0], "requires window_applied=true"),
        (True, [1.0, 6.0], "does not match active_speech_window_s"),
    ],
)
def test_evaluation_window_evidence_must_match_active_window(
    window_applied: bool,
    analysis_window_s: list[float],
    expected_reason: str,
) -> None:
    result = evaluate_rlr_linearity_group(
        report_group(
            window_applied=window_applied,
            analysis_window_s=analysis_window_s,
        )
    )

    assert result["status"] == "invalid"
    assert result["window_applied"] is False
    assert any(expected_reason in reason for reason in result["reasons"])


def test_numeric_run_cannot_claim_window_was_applied() -> None:
    runs = group()
    runs[0]["metrics"]["window_applied"] = True
    runs[0]["metrics"]["analysis_window_s"] = [1.0, 7.0]

    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert result["window_source"] == "provided_band_levels"
    assert any(
        "numeric run cannot claim window_applied=true" in reason
        for reason in result["reasons"]
    )


def test_declared_analysis_window_is_checked_on_numeric_runs() -> None:
    runs = group()
    runs[1]["metrics"]["window_applied"] = False
    runs[1]["metrics"]["analysis_window_s"] = [1.0, 6.0]

    result = evaluate_rlr_linearity_group(runs)

    assert result["status"] == "invalid"
    assert any(
        "analysis_window_s does not match active_speech_window_s" in reason
        for reason in result["reasons"]
    )
