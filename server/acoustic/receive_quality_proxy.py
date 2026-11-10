"""Receive-quality engineering proxy backed by binaural DNSMOS P.835.

This module never represents DNSMOS P.835 as TMOS or POLQA.  It scores the
left and right receive signals independently and reduces them with a
conservative, worst-ear engineering proxy for the receive
``listening_quality`` contract row.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Real
from typing import Any, Protocol

import numpy as np

from .acceptance_contract import contract_items, required_conditions


CONTRACT_ID = "B10"
CONTRACT_NAME = "One Way Speech Quality in RCV (TMOS)"
DEFAULT_MODE = "WB-wireless"
MODEL_NAME = "DNSMOS P.835"
NATIVE_SAMPLE_RATE = 16_000
MINIMUM_SECONDS = 9.01
DNSMOS_METRICS = ("SIG", "BAK", "OVRL")
MIN_DNSMOS_SCORE = 0.0
MAX_DNSMOS_SCORE = 5.0
SILENCE_RMS = 1e-10

Scorer = Callable[[np.ndarray, int], Mapping[str, Any]]


class ReceiveQualityProxyError(ValueError):
    """The B10 proxy request or model output is not valid."""


class ReceiveQualityInputError(ReceiveQualityProxyError):
    """An ear cannot be scored safely at its native sample rate."""


class ReceiveQualityModelUnavailable(ReceiveQualityProxyError):
    """The injected DNSMOS scorer reports that the model is unavailable."""


class ReceiveQualityModelFailure(ReceiveQualityProxyError):
    """The injected DNSMOS scorer raised or returned an invalid result."""


class DNSMOSScorer(Protocol):
    """The subset of ``DNSMOSEvaluator`` used by the adapter."""

    rate: int
    minimum_seconds: float

    def score(self, x: np.ndarray, rate: int) -> dict[str, Any]:
        """Return a DNSMOS ``available``/``values`` result."""


def _quality_contract_row(mode: str) -> dict[str, Any]:
    matches = [
        row
        for row in contract_items()
        if row.get("family") == "listening_quality"
        and row.get("direction") == "receive"
        and required_conditions(row).get("mode") == mode
    ]
    if len(matches) != 1:
        raise ReceiveQualityInputError(
            f"expected one receive listening_quality contract row for mode "
            f"{mode!r}, found {len(matches)}"
        )
    return matches[0]


def _selected_mode(
    mode: str | None,
    metadata: Mapping[str, Any] | None,
) -> str:
    if mode is None:
        metadata_mode = metadata.get("mode") if metadata is not None else None
        mode = DEFAULT_MODE if metadata_mode is None else metadata_mode
    if not isinstance(mode, str) or not mode.strip():
        raise ReceiveQualityInputError("mode must be a non-empty string")
    return mode.strip()


def _validated_ear(
    samples: np.ndarray,
    *,
    ear: str,
    sample_rate: int,
) -> np.ndarray:
    try:
        values = np.asarray(samples)
    except (TypeError, ValueError) as error:
        raise ReceiveQualityInputError(f"{ear} audio is not array-like") from error
    if values.ndim != 1:
        raise ReceiveQualityInputError(
            f"{ear} audio must be an explicit mono array, got shape {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise ReceiveQualityInputError(f"{ear} audio must contain real numeric samples")
    if values.size < round(MINIMUM_SECONDS * sample_rate):
        raise ReceiveQualityInputError(
            f"{ear} audio is too short for DNSMOS P.835: "
            f"{values.size / sample_rate:.3f} s < {MINIMUM_SECONDS:.2f} s"
        )
    try:
        numeric = values.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReceiveQualityInputError(
            f"{ear} audio cannot be represented as float64"
        ) from error
    if not np.all(np.isfinite(numeric)):
        raise ReceiveQualityInputError(f"{ear} audio contains non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(numeric))))
    if not math.isfinite(rms):
        raise ReceiveQualityInputError(f"{ear} audio has non-finite RMS")
    if rms <= SILENCE_RMS or float(np.std(numeric)) <= SILENCE_RMS:
        raise ReceiveQualityInputError(
            f"{ear} audio is silent or has no AC excitation"
        )
    return values


def _scored_metrics(
    samples: np.ndarray,
    *,
    ear: str,
    sample_rate: int,
    scorer: Scorer,
) -> dict[str, float]:
    try:
        row = scorer(samples, sample_rate)
    except Exception as error:
        raise ReceiveQualityModelFailure(
            f"{ear} DNSMOS inference failed: {error}"
        ) from error
    if not isinstance(row, Mapping):
        raise ReceiveQualityModelFailure(
            f"{ear} DNSMOS scorer returned {type(row).__name__}, expected a mapping"
        )
    if "available" in row and not bool(row["available"]):
        reason = row.get("reason", "unspecified")
        raise ReceiveQualityModelUnavailable(
            f"{ear} DNSMOS model unavailable: {reason}"
        )
    raw_values = row.get("values", row)
    if not isinstance(raw_values, Mapping):
        raise ReceiveQualityModelFailure(
            f"{ear} DNSMOS values must be a mapping"
        )

    result: dict[str, float] = {}
    for metric in DNSMOS_METRICS:
        if metric not in raw_values:
            raise ReceiveQualityModelFailure(
                f"{ear} DNSMOS output is missing {metric}"
            )
        value = raw_values[metric]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ReceiveQualityModelFailure(
                f"{ear} DNSMOS {metric} score is not a real number"
            )
        score = float(value)
        if not math.isfinite(score):
            raise ReceiveQualityModelFailure(
                f"{ear} DNSMOS {metric} score is non-finite"
            )
        if not MIN_DNSMOS_SCORE <= score <= MAX_DNSMOS_SCORE:
            raise ReceiveQualityModelFailure(
                f"{ear} DNSMOS {metric} score {score} is outside "
                f"[{MIN_DNSMOS_SCORE}, {MAX_DNSMOS_SCORE}]"
            )
        result[metric] = score
    return result


def _validated_sample_rate(sample_rate: int) -> int:
    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) != NATIVE_SAMPLE_RATE
    ):
        raise ReceiveQualityInputError(
            "DNSMOS P.835 proxy requires native 16000 Hz audio; "
            f"got {sample_rate!r}. Resampling is intentionally not performed."
        )
    return int(sample_rate)


def evaluate_binaural_dnsmos_proxy(
    left: np.ndarray,
    right: np.ndarray,
    scorer: Scorer,
    metadata: Mapping[str, Any] | None = None,
    *,
    sample_rate: int = NATIVE_SAMPLE_RATE,
    mode: str | None = None,
) -> dict[str, Any]:
    """Score both ears with an injected DNSMOS scorer and aggregate the proxy.

    ``scorer`` must accept ``(mono_samples, sample_rate)`` and return either a
    DNSMOS evaluator row with ``available``/``values`` or a mapping containing
    the three DNSMOS metrics directly.  Inputs are validated and passed through
    unchanged; this function never resamples, repeats, downmixes, or normalizes
    either ear.  ``mode`` selects the mode-specific receive listening-quality
    contract row.  When omitted, ``metadata["mode"]`` is used and
    ``WB-wireless`` is the compatibility fallback.
    """
    rate = _validated_sample_rate(sample_rate)
    if not callable(scorer):
        raise ReceiveQualityInputError("scorer must be callable")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ReceiveQualityInputError("metadata must be a mapping or None")
    selected_mode = _selected_mode(mode, metadata)
    contract = _quality_contract_row(selected_mode)

    left_samples = _validated_ear(left, ear="left", sample_rate=rate)
    right_samples = _validated_ear(right, ear="right", sample_rate=rate)
    per_ear = {
        "left": _scored_metrics(
            left_samples,
            ear="left",
            sample_rate=rate,
            scorer=scorer,
        ),
        "right": _scored_metrics(
            right_samples,
            ear="right",
            sample_rate=rate,
            scorer=scorer,
        ),
    }

    sig_proxy = min(per_ear["left"]["SIG"], per_ear["right"]["SIG"])
    bak_proxy = min(per_ear["left"]["BAK"], per_ear["right"]["BAK"])
    ovrl_proxy = min(per_ear["left"]["OVRL"], per_ear["right"]["OVRL"])
    proxy = {
        "SIG_proxy": sig_proxy,
        "BAK_proxy": bak_proxy,
        "OVRL_proxy": ovrl_proxy,
        "TMOS_proxy": ovrl_proxy,
    }
    limitation = (
        "DNSMOS P.835 is a no-reference model estimate of SIG, BAK, and OVRL. "
        "It is not TMOS or POLQA, and TMOS_proxy is an engineering-only "
        "worst-ear reduction rather than a formal speech-quality score."
    )
    return {
        "contract_id": contract["source_cell"],
        "contract_name": contract["test_name"],
        "item_id": contract["id"],
        "mode": selected_mode,
        "model": MODEL_NAME,
        "evidence_level": "model_proxy",
        "available": True,
        "sample_rate": rate,
        "minimum_seconds": MINIMUM_SECONDS,
        "per_ear": per_ear,
        "proxy": proxy,
        **proxy,
        "formal_tmos_available": False,
        "formal_polqa_available": False,
        "formal_compliance_claim_allowed": False,
        "standard_compliance": False,
        "input_policy": (
            "left and right scored independently at native rate; no resampling, "
            "repetition, downmix, or level normalization"
        ),
        "limitations": [limitation],
        "metadata": dict(metadata) if metadata is not None else {},
    }


def evaluate_binaural_dnsmos_with_evaluator(
    left: np.ndarray,
    right: np.ndarray,
    evaluator: DNSMOSScorer,
    metadata: Mapping[str, Any] | None = None,
    *,
    sample_rate: int = NATIVE_SAMPLE_RATE,
    mode: str | None = None,
) -> dict[str, Any]:
    """Adapt a ``DNSMOSEvaluator`` to a receive quality contract row."""
    evaluator_rate = getattr(evaluator, "rate", NATIVE_SAMPLE_RATE)
    evaluator_minimum = getattr(evaluator, "minimum_seconds", MINIMUM_SECONDS)
    if int(evaluator_rate) != NATIVE_SAMPLE_RATE:
        raise ReceiveQualityModelUnavailable(
            "DNSMOSEvaluator is configured for an unsupported native rate: "
            f"{evaluator_rate!r}"
        )
    if not math.isclose(
        float(evaluator_minimum),
        MINIMUM_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReceiveQualityModelUnavailable(
            "DNSMOSEvaluator minimum duration does not match the B10 proxy: "
            f"{evaluator_minimum!r}"
        )
    score = getattr(evaluator, "score", None)
    if not callable(score):
        raise ReceiveQualityModelUnavailable(
            "DNSMOSEvaluator does not expose a callable score method"
        )
    return evaluate_binaural_dnsmos_proxy(
        left,
        right,
        score,
        metadata,
        sample_rate=sample_rate,
        mode=mode,
    )
