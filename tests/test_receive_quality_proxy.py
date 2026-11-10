from __future__ import annotations

import math

import numpy as np
import pytest

from server.acoustic.receive_quality_proxy import (
    MINIMUM_SECONDS,
    NATIVE_SAMPLE_RATE,
    ReceiveQualityInputError,
    ReceiveQualityModelFailure,
    ReceiveQualityModelUnavailable,
    evaluate_binaural_dnsmos_proxy,
    evaluate_binaural_dnsmos_with_evaluator,
)


def _tone(*, amplitude: float = 0.1, seconds: float = MINIMUM_SECONDS) -> np.ndarray:
    time = np.arange(round(seconds * NATIVE_SAMPLE_RATE), dtype=np.float64)
    return amplitude * np.sin(2 * np.pi * 440 * time / NATIVE_SAMPLE_RATE)


def _row(sig: float, bak: float, ovrl: float) -> dict:
    return {
        "available": True,
        "values": {"SIG": sig, "BAK": bak, "OVRL": ovrl},
    }


def test_scores_each_ear_and_uses_conservative_minimum():
    left = _tone()
    right = _tone(amplitude=0.2)
    calls = []
    rows = iter(
        [
            _row(3.4, 4.2, 3.1),
            _row(2.8, 4.5, 2.7),
        ]
    )

    def scorer(samples, rate):
        calls.append((samples, rate))
        return next(rows)

    result = evaluate_binaural_dnsmos_proxy(
        left,
        right,
        scorer,
        {"left_file": "virtual_left.wav", "right_file": "virtual_right.wav"},
    )

    assert calls == [(left, NATIVE_SAMPLE_RATE), (right, NATIVE_SAMPLE_RATE)]
    assert result["per_ear"]["left"] == {"SIG": 3.4, "BAK": 4.2, "OVRL": 3.1}
    assert result["per_ear"]["right"] == {"SIG": 2.8, "BAK": 4.5, "OVRL": 2.7}
    assert result["proxy"] == {
        "SIG_proxy": 2.8,
        "BAK_proxy": 4.2,
        "OVRL_proxy": 2.7,
        "TMOS_proxy": 2.7,
    }
    assert result["SIG_proxy"] == 2.8
    assert result["BAK_proxy"] == 4.2
    assert result["OVRL_proxy"] == 2.7
    assert result["TMOS_proxy"] == 2.7
    assert result["contract_id"] == "B10"
    assert result["item_id"] == "WB-wireless:B10"
    assert result["metadata"]["left_file"] == "virtual_left.wav"


def test_formal_compliance_is_always_false_and_metadata_cannot_promote_it():
    result = evaluate_binaural_dnsmos_proxy(
        _tone(),
        _tone(),
        lambda samples, rate: _row(4.0, 4.0, 4.0),
        {
            "evidence_level": "formal",
            "formal_tmos_available": True,
            "formal_polqa_available": True,
            "formal_compliance_claim_allowed": True,
            "standard_compliance": True,
        },
    )

    assert result["evidence_level"] == "model_proxy"
    assert result["formal_tmos_available"] is False
    assert result["formal_polqa_available"] is False
    assert result["formal_compliance_claim_allowed"] is False
    assert result["standard_compliance"] is False
    assert "not TMOS or POLQA" in result["limitations"][0]


def test_dnsmos_is_not_exposed_as_a_formal_tmos_value():
    result = evaluate_binaural_dnsmos_proxy(
        _tone(),
        _tone(),
        lambda samples, rate: _row(4.1, 4.2, 4.3),
    )

    assert "TMOS" not in result
    assert result["TMOS_proxy"] == result["OVRL_proxy"] == 4.3
    assert result["formal_tmos_available"] is False
    assert result["formal_polqa_available"] is False
    assert result["evidence_level"] == "model_proxy"


def test_official_low_quality_scores_below_one_are_preserved():
    result = evaluate_binaural_dnsmos_proxy(
        _tone(),
        _tone(),
        lambda samples, rate: _row(1.2, 0.9, 1.0),
    )

    assert result["BAK_proxy"] == 0.9
    assert result["OVRL_proxy"] == 1.0
    assert result["standard_compliance"] is False


def test_right_ear_model_failure_rejects_the_whole_proxy():
    calls = 0

    def scorer(samples, rate):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _row(3.0, 3.5, 3.1)
        raise RuntimeError("onnx session failed")

    with pytest.raises(ReceiveQualityModelFailure, match="right DNSMOS inference failed"):
        evaluate_binaural_dnsmos_proxy(_tone(), _tone(), scorer)
    assert calls == 2


def test_unavailable_model_is_rejected():
    with pytest.raises(ReceiveQualityModelUnavailable, match="model_not_ready"):
        evaluate_binaural_dnsmos_proxy(
            _tone(),
            _tone(),
            lambda samples, rate: {
                "available": False,
                "reason": "model_not_ready",
            },
        )


@pytest.mark.parametrize(
    "bad_audio",
    [
        np.zeros(round(MINIMUM_SECONDS * NATIVE_SAMPLE_RATE)),
        np.ones(round(MINIMUM_SECONDS * NATIVE_SAMPLE_RATE)),
        np.full(round(MINIMUM_SECONDS * NATIVE_SAMPLE_RATE), np.nan),
        np.full(round(MINIMUM_SECONDS * NATIVE_SAMPLE_RATE), np.inf),
    ],
)
def test_silence_constant_and_nonfinite_audio_are_rejected(bad_audio):
    with pytest.raises(ReceiveQualityInputError):
        evaluate_binaural_dnsmos_proxy(
            bad_audio,
            _tone(),
            lambda samples, rate: _row(4.0, 4.0, 4.0),
        )


def test_unsupported_sample_rate_is_rejected_without_resampling():
    called = False

    def scorer(samples, rate):
        nonlocal called
        called = True
        return _row(4.0, 4.0, 4.0)

    with pytest.raises(ReceiveQualityInputError, match="Resampling is intentionally not performed"):
        evaluate_binaural_dnsmos_proxy(
            _tone(),
            _tone(),
            scorer,
            sample_rate=48_000,
        )
    assert called is False


def test_short_audio_is_rejected_before_model_execution():
    called = False

    def scorer(samples, rate):
        nonlocal called
        called = True
        return _row(4.0, 4.0, 4.0)

    with pytest.raises(ReceiveQualityInputError, match="too short"):
        evaluate_binaural_dnsmos_proxy(_tone(seconds=1.0), _tone(), scorer)
    assert called is False


@pytest.mark.parametrize(
    "values",
    [
        {"SIG": -0.01, "BAK": 4.0, "OVRL": 4.0},
        {"SIG": 4.0, "BAK": 5.01, "OVRL": 4.0},
        {"SIG": 4.0, "BAK": 4.0, "OVRL": math.nan},
        {"SIG": 4.0, "BAK": math.inf, "OVRL": 4.0},
    ],
)
def test_out_of_bounds_or_nonfinite_scores_are_rejected(values):
    with pytest.raises(ReceiveQualityModelFailure):
        evaluate_binaural_dnsmos_proxy(
            _tone(),
            _tone(),
            lambda samples, rate: {"available": True, "values": values},
        )


def test_missing_metric_is_rejected():
    with pytest.raises(ReceiveQualityModelFailure, match="missing BAK"):
        evaluate_binaural_dnsmos_proxy(
            _tone(),
            _tone(),
            lambda samples, rate: {
                "available": True,
                "values": {"SIG": 4.0, "OVRL": 4.0},
            },
        )


def test_evaluator_adapter_passes_native_arrays_unchanged():
    left = _tone(amplitude=0.125)
    right = _tone(amplitude=0.25)
    calls = []

    class FakeDNSMOSEvaluator:
        rate = NATIVE_SAMPLE_RATE
        minimum_seconds = MINIMUM_SECONDS

        def score(self, samples, rate):
            calls.append((samples, rate))
            return _row(3.2, 4.1, 3.0)

    result = evaluate_binaural_dnsmos_with_evaluator(
        left,
        right,
        FakeDNSMOSEvaluator(),
    )

    assert len(calls) == 2
    assert calls[0][0] is left
    assert calls[1][0] is right
    assert np.array_equal(calls[0][0], left)
    assert np.array_equal(calls[1][0], right)
    assert [rate for _, rate in calls] == [NATIVE_SAMPLE_RATE, NATIVE_SAMPLE_RATE]
    assert result["proxy"]["TMOS_proxy"] == 3.0
    assert result["standard_compliance"] is False


@pytest.mark.parametrize(
    ("mode", "contract_id"),
    [
        ("NB-wired", "B10"),
        ("NB-wireless", "B10"),
        ("NB-bt", "B10"),
        ("WB-wired", "B10"),
        ("WB-wireless", "B10"),
        ("WB-bt", "B10"),
        ("SWB-wired", "B6"),
        ("SWB-wireless", "B6"),
    ],
)
def test_contract_identity_for_all_eight_modes(
    mode: str,
    contract_id: str,
) -> None:
    result = evaluate_binaural_dnsmos_proxy(
        _tone(),
        _tone(),
        lambda samples, rate: _row(4.0, 4.0, 4.0),
        mode=mode,
    )

    assert result["mode"] == mode
    assert result["contract_id"] == contract_id
    assert result["item_id"] == f"{mode}:{contract_id}"
    assert result["standard_compliance"] is False
    assert result["formal_compliance_claim_allowed"] is False


def test_metadata_mode_selects_contract_identity() -> None:
    result = evaluate_binaural_dnsmos_proxy(
        _tone(),
        _tone(),
        lambda samples, rate: _row(4.0, 4.0, 4.0),
        {"mode": "NB-wired"},
    )

    assert result["mode"] == "NB-wired"
    assert result["contract_id"] == "B10"
    assert result["item_id"] == "NB-wired:B10"
    assert result["contract_name"] == (
        "11.05.2 One Way Sp. Qual. in RCV (TMOS) nominal DVNB"
    )


def test_evaluator_adapter_accepts_mode() -> None:
    class FakeDNSMOSEvaluator:
        rate = NATIVE_SAMPLE_RATE
        minimum_seconds = MINIMUM_SECONDS

        def score(self, samples, rate):
            return _row(3.5, 4.0, 3.25)

    result = evaluate_binaural_dnsmos_with_evaluator(
        _tone(),
        _tone(),
        FakeDNSMOSEvaluator(),
        mode="SWB-wireless",
    )

    assert result["mode"] == "SWB-wireless"
    assert result["contract_id"] == "B6"
    assert result["item_id"] == "SWB-wireless:B6"
    assert result["contract_name"] == (
        "Optional: 9.5.2b Listening Speech Qual. POLQA RCV DVSB"
    )
