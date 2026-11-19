"""Experimental ETSI TS 103 281 Model A implementation.

The random-forest configuration files published by ETSI are executable, but the
standard also depends on the Sottek hearing model and several speech-quality
feature definitions.  This module keeps those two layers separate:

* ``RandomForestModel`` implements the normative model-file format exactly.
* ``extract_model_a_features`` implements the documented signal path with
  explicit, inspectable approximations where the standard points to external
  papers.

The checked-in Annex C evidence currently fails the normative +/-0.1 MOS gate
for all 48 conditions.  Consequently, scores from this module are experimental
engineering diagnostics, not an implementation that may claim TS 103 281
compliance.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable
import math

import numpy as np
from scipy import signal
from scipy.stats import kurtosis

from .ts103281_activity import classify_activity
from .ts103281_hearing import auditory_spectrum
from .ts103281_modulation import modulation_features
from .ts103281_noise import NoiseReconstructionError, reconstruct_noise_log_envelope
from .ts103281_reference import optimize_reference
from .ts103281_spectral import (
    calibrate_reference, correct_reference_gain, ltass_activity_mask,
)
from .ts103281_validation import validate_annex_c_evidence


SAMPLE_RATE = 48_000
FRAME_SECONDS = 0.008
FRAME_SIZE = round(SAMPLE_RATE * FRAME_SECONDS)
FRAME_HOP = FRAME_SIZE
P0_PA = 20e-6
MODEL_DIR = Path(__file__).with_name("assets") / "ts103281"
OFFICIAL_VALIDATION = MODEL_DIR / "annex-c-validation.json"
EXPECTED_TOLERANCE_MOS = 0.1

# The published forests expect features on the scales of HEAD's original
# hearing-model implementation.  These constants provide a transparent
# diagnostic fit to the public Annex C vectors; they are deliberately not used
# for the normative/raw score and are not evidence of independent conformance.
_ANNEX_C_DIAGNOSTIC = {
    "smos": (
        1.8153161819105723,
        -59.8361374776833,
        1.7369532877617655,
        18.116477265444253,
        1.3698040081763712,
        1.1832108143041613,
        0.20335089020669828,
        0.2987177399942784,
        2.9446870886984127,
        -48.78664976502908,
        1.9583960915440362,
        -67.43408181924376,
        0.3018225404499828,
        16.145517501921226,
    ),
    "nmos": (
        3.8337133811702286,
        4.766211629733647,
        0.5597831220291615,
        8.14638560635322,
        2.4620613427047573,
    ),
    "gmos": (
        1.934271760620403,
        -2.9736909442861235,
        1.4114559394192603,
        -1.8625653939310283,
        0.35223132308959126,
        -17.01252310715582,
    ),
}


class ModelAError(ValueError):
    """The supplied audio or model asset cannot be evaluated."""


@dataclass(frozen=True)
class TreeNode:
    feature: int
    threshold: float
    left: int
    right: int
    mean: float
    standard_deviation: float | None


@dataclass(frozen=True)
class RandomForestModel:
    trees: tuple[tuple[TreeNode, ...], ...]
    feature_names: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        raw = Path(path).read_text(encoding="utf-8-sig").splitlines()
        nonempty = [line.strip() for line in raw if line.strip()]
        if len(nonempty) < 2:
            raise ModelAError(f"{path.name}: model file is empty")
        feature_line_index = next(
            (
                index
                for index in range(len(nonempty) - 1, 0, -1)
                if ";" in nonempty[index]
                or nonempty[index].startswith(
                    (
                        "Speech.",
                        "BGN.",
                        "PreProcessing.",
                        "FrequencyMetric.",
                        "N-MOS_instrumental",
                    )
                )
            ),
            None,
        )
        if feature_line_index is None:
            raise ModelAError(f"{path.name}: missing feature-name line")
        feature_names = tuple(
            part.strip() for part in nonempty[feature_line_index].split(";")
        )
        structure = nonempty[:feature_line_index]
        try:
            tree_count = int(structure[0])
        except ValueError as error:
            raise ModelAError(f"{path.name}: invalid tree count") from error
        if tree_count <= 0 or len(feature_names) == 0:
            raise ModelAError(f"{path.name}: empty forest or feature list")

        cursor = 1
        trees: list[tuple[TreeNode, ...]] = []
        for tree_index in range(tree_count):
            if cursor >= len(structure):
                raise ModelAError(f"{path.name}: missing tree {tree_index + 1}")
            try:
                node_count = int(structure[cursor])
            except ValueError as error:
                raise ModelAError(
                    f"{path.name}: invalid node count for tree {tree_index + 1}"
                ) from error
            cursor += 1
            if node_count <= 0 or cursor + 4 * node_count > len(structure):
                raise ModelAError(
                    f"{path.name}: truncated tree {tree_index + 1}"
                )
            nodes: list[TreeNode] = []
            for node_index in range(node_count):
                try:
                    feature = int(structure[cursor])
                    threshold = float(structure[cursor + 1])
                    children = structure[cursor + 2].split()
                    statistics = structure[cursor + 3].split()
                    if len(children) != 2 or not 1 <= len(statistics) <= 2:
                        raise ValueError
                    left, right = (int(value) for value in children)
                    mean = float(statistics[0])
                    deviation = (
                        float(statistics[1]) if len(statistics) == 2 else None
                    )
                except ValueError as error:
                    raise ModelAError(
                        f"{path.name}: malformed tree {tree_index + 1}, "
                        f"node {node_index + 1}"
                    ) from error
                if not 0 <= feature < len(feature_names):
                    raise ModelAError(
                        f"{path.name}: feature {feature} outside declared list"
                    )
                if left == right == 0:
                    left = right = -1
                else:
                    # The published files use zero-based node references.
                    if not 0 <= left < node_count or not 0 <= right < node_count:
                        raise ModelAError(
                            f"{path.name}: child outside tree "
                            f"{tree_index + 1}, node {node_index + 1}"
                        )
                if not all(
                    math.isfinite(value)
                    for value in (threshold, mean, deviation)
                    if value is not None
                ):
                    raise ModelAError(f"{path.name}: non-finite model value")
                nodes.append(
                    TreeNode(
                        feature,
                        threshold,
                        left,
                        right,
                        mean,
                        deviation,
                    )
                )
                cursor += 4
            trees.append(tuple(nodes))
        if cursor != len(structure):
            raise ModelAError(
                f"{path.name}: {len(structure) - cursor} unexpected model lines"
            )
        return cls(tuple(trees), feature_names)

    def predict_one_tree(
        self, features: np.ndarray, tree: tuple[TreeNode, ...]
    ) -> float:
        index = 0
        visited: set[int] = set()
        while True:
            if index < 0:
                raise ModelAError("model tree terminated without a leaf")
            if index in visited:
                raise ModelAError("model tree contains a cycle")
            visited.add(index)
            node = tree[index]
            if node.left == node.right == -1:
                return node.mean
            index = (
                node.left
                if float(features[node.feature]) < node.threshold
                else node.right
            )

    def predict(self, features: dict[str, float]) -> dict:
        if set(features) != set(self.feature_names):
            missing = sorted(set(self.feature_names) - set(features))
            extra = sorted(set(features) - set(self.feature_names))
            raise ModelAError(f"feature contract mismatch: missing={missing}, extra={extra}")
        vector = np.asarray(
            [float(features[name]) for name in self.feature_names],
            dtype=np.float64,
        )
        if not np.isfinite(vector).all():
            raise ModelAError("feature vector contains a non-finite value")
        values = np.asarray(
            [self.predict_one_tree(vector, tree) for tree in self.trees],
            dtype=np.float64,
        )
        return dict(
            value=float(np.mean(values)),
            tree_standard_deviation=float(np.std(values)),
            tree_minimum=float(np.min(values)),
            tree_maximum=float(np.max(values)),
        )


def _resample(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return np.asarray(x, dtype=np.float64)
    divisor = math.gcd(SAMPLE_RATE, rate)
    return signal.resample_poly(
        np.asarray(x, dtype=np.float64),
        SAMPLE_RATE // divisor,
        rate // divisor,
    )


def _bandpass_for_alignment(x: np.ndarray) -> np.ndarray:
    sos = signal.butter(
        6, [300.0, 3300.0], btype="bandpass", fs=SAMPLE_RATE, output="sos"
    )
    return signal.sosfiltfilt(sos, x)


def _align(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, dict]:
    length = min(len(reference), len(degraded))
    reference = reference[:length]
    degraded = degraded[:length]
    if length < round(SAMPLE_RATE * 0.25):
        raise ModelAError("audio must contain at least 0.25 seconds")
    ref_band = _bandpass_for_alignment(reference)
    deg_band = _bandpass_for_alignment(degraded)
    correlation = signal.correlate(deg_band, ref_band, mode="full", method="fft")
    envelope = np.abs(signal.hilbert(correlation))
    lags = signal.correlation_lags(length, length, mode="full")
    lag = int(lags[int(np.argmax(envelope))])
    aligned = np.zeros(length, dtype=np.float64)
    if lag >= 0:
        # Positive lag means y' is later than x'. TS 103 281 clause 6.3.2
        # compensates x' by prepending zeros and cropping its tail.
        aligned[lag:] = reference[: length - lag]
    else:
        # Negatively lagged x' is advanced by cropping its beginning.
        aligned[: length + lag] = reference[-lag:]
    peak = float(np.max(envelope))
    side = float(np.median(envelope))
    return aligned, dict(
        lag_samples=lag,
        delay_ms=1000.0 * lag / SAMPLE_RATE,
        correlation_peak=peak,
        envelope_peak_to_median=peak / max(side, 1e-30),
        alignment="300-3300 Hz Hilbert-envelope cross-correlation",
    )


def _activity(reference: np.ndarray) -> tuple[np.ndarray, dict]:
    try:
        labels, details = classify_activity(reference, SAMPLE_RATE)
    except ValueError as error:
        raise ModelAError(str(error)) from error
    spectral_frame_count = len(reference) // FRAME_SIZE
    spectral_centres = (np.arange(spectral_frame_count) + 0.5) * FRAME_SECONDS
    activity_indices = np.minimum(
        (spectral_centres / 0.010).astype(np.int64),
        len(labels) - 1,
    )
    spectral_labels = labels[activity_indices]
    active = spectral_labels != "S"
    return active, dict(
        method="ITU-T P.56 Method B / G.160 Appendix II Amendment 2",
        asl_db=details["asl_db"],
        activity_frame_ms=10.0,
        class_counts=details["counts"],
        activity_classes=labels.tolist(),
        spectral_classes=spectral_labels.tolist(),
        active_activity_frames=int(np.count_nonzero(labels != "S")),
        active_spectral_frames=int(np.count_nonzero(active)),
    )


def _band_centres() -> np.ndarray:
    # ETSI TS 103 281, 6.3.3: 33 critical bands and two intermediate
    # bands per interval, i.e. 99 overlapping analysis bands.
    z = np.arange(1.0, 100.0, 1.0) / 3.0
    frequencies = (50.0 / 0.14) * np.sinh(0.14 * z)
    return np.minimum(frequencies, 20_000.0)


def _aggregate_basic_bands(
    values: np.ndarray,
    centres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate the 99 auditory bands into 33 basic bands (clause 6.3.3).

    The standard specifies the quadratic mean of every three consecutive
    auditory bands. The third entry in each triplet is the basic-band centre;
    the preceding two are the 66 intermediate bands.
    """
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[0] != 99 or len(centres) != 99:
        raise ModelAError("expected 99 auditory bands before basic-band aggregation")
    aggregated = np.sqrt(
        np.mean(values.reshape((33, 3) + values.shape[1:]) ** 2, axis=1)
    )
    return aggregated, np.asarray(centres)[2::3]


def _auditory_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pressure spectrum after the full linear hearing front-end.

    The centre-frequency grid and original ear coefficients are still being
    checked against the cited hearing model.  The independent front-end
    records these remaining assumptions explicitly.
    """
    centres = _band_centres()
    return auditory_spectrum(x, centres, SAMPLE_RATE, FRAME_SECONDS), centres


def _nonlinear(x: np.ndarray) -> np.ndarray:
    """Compress pressure in Pa using EG 202 396-3 Annex K.1--K.4.

    Annex K defines its input as p_eff / p_0, not absolute pressure in Pa.
    Its output is normalized compressed pressure.  Keeping the normalization
    here prevents callers from applying different units to the speech and
    noise branches.
    """
    exponent = 0.2
    threshold = 5.66
    a = 0.5 * (1.0 - exponent) / (2.0 - exponent) / threshold
    b = threshold ** (1.0 - exponent) / (exponent * (2.0 - exponent))
    c = 0.5 * (1.0 - exponent) / exponent * threshold
    value = np.maximum(np.asarray(x, dtype=np.float64), 0.0) / P0_PA
    return np.where(
        value < threshold,
        value - a * value * value,
        b * np.power(np.maximum(value, threshold), exponent) - c,
    )


def _a_weight(frequencies: np.ndarray) -> np.ndarray:
    f2 = np.square(frequencies)
    numerator = (12_194.0**2) * f2**2
    denominator = (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12_194.0**2)
    )
    # IEC 61672 A-weighting is normalized to approximately 0 dB at 1 kHz.
    return (numerator / denominator) * 10 ** (2.0 / 20.0)


def _transfer_and_residual(
    reference: np.ndarray,
    degraded: np.ndarray,
    active: np.ndarray,
    activity: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    # Preserve the original 10 ms classifications for the time-domain H1;
    # expanding the 8 ms spectral mask would shift activity boundaries.
    activity_labels = np.asarray(activity["activity_classes"])
    sample_active = np.zeros(len(reference), dtype=bool)
    expanded = np.repeat(activity_labels != "S", SAMPLE_RATE // 100)
    covered = min(len(expanded), len(sample_active))
    sample_active[:covered] = expanded[:covered]
    centres = _band_centres()
    try:
        calibrated, transfer, calibration = calibrate_reference(
            reference, degraded, sample_active, centres, SAMPLE_RATE,
        )
    except ValueError as error:
        raise ModelAError(f"initial reference calibration: {error}") from error
    ref_spectrum, _ = _auditory_spectrum(calibrated)
    deg_spectrum, _ = _auditory_spectrum(degraded)
    frame_count = min(ref_spectrum.shape[1], deg_spectrum.shape[1], len(active))
    ref_spectrum = ref_spectrum[:, :frame_count]
    deg_spectrum = deg_spectrum[:, :frame_count]
    active = active[:frame_count]
    if not np.any(active):
        raise ModelAError("no active speech frames after spectral analysis")
    mask, spectral_activity = ltass_activity_mask(
        ref_spectrum, centres, calibration["scaled_reference_asl_db"],
    )
    speech, noise, separation = _separate_spectra(
        ref_spectrum, deg_spectrum, transfer, mask,
    )
    try:
        corrected_reference, gain_correction = correct_reference_gain(
            ref_spectrum, speech, active,
        )
    except ValueError as error:
        raise ModelAError(f"final reference calibration: {error}") from error
    return corrected_reference, deg_spectrum, speech, noise, dict(
        calibration=calibration, spectral_activity=spectral_activity,
        separation=separation, reference_gain=gain_correction,
    )


def _separate_spectra(
    reference: np.ndarray,
    degraded: np.ndarray,
    transfer: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Equations (8)--(10), then the amplitude split in (5)--(6).

    The noise reconstruction remains an independently implemented spectral
    deconvolution, with its numerical assumptions retained in diagnostics.
    A relative machine-precision floor makes log(0) finite for reconstruction;
    entirely zero residual bands retain zero noise estimates exactly.
    """
    filtered_reference = np.asarray(transfer)[:, None] * reference
    preliminary = np.maximum(degraded - filtered_reference, 0.0)
    nonzero_bands = np.any(preliminary > 0.0, axis=1)
    reconstructed_noise = np.zeros_like(preliminary)
    reconstruction: dict = {"method": "exact_zero_residual", "bands": []}
    floor = max(float(np.max(preliminary)) * np.finfo(float).eps, np.finfo(float).tiny)
    if np.any(nonzero_bands):
        noise_db = 20 * np.log10(np.maximum(preliminary[nonzero_bands], floor))
        try:
            reconstructed_db, reconstruction = reconstruct_noise_log_envelope(
                noise_db, reliability[nonzero_bands],
            )
        except NoiseReconstructionError as error:
            raise ModelAError(f"noise reconstruction: {error}") from error
        with np.errstate(over="raise", invalid="raise"):
            try:
                reconstructed_noise[nonzero_bands] = 10 ** (reconstructed_db / 20)
            except FloatingPointError as error:
                raise ModelAError("noise reconstruction amplitude overflow") from error
    denominator = filtered_reference**2 + reconstructed_noise**2
    wiener = np.divide(
        filtered_reference**2, denominator, out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    speech = wiener * degraded
    noise = degraded - speech
    bands = reconstruction.get("bands", [])
    return speech, noise, {
        "method": "H1/LTASS mask/spectral deconvolution/Wiener amplitude split",
        "noise_reconstruction": reconstruction,
        "nonzero_residual_bands": int(np.count_nonzero(nonzero_bands)),
        "zero_residual_bands": int(np.count_nonzero(~nonzero_bands)),
        "reconstruction_converged_bands": sum(bool(b.get("converged")) for b in bands),
        "log_zero_floor_pa": floor,
        "log_zero_floor_rule": "max residual times float64 epsilon; tiny lower bound",
        "wiener_gain_min": float(np.min(wiener)),
        "wiener_gain_max": float(np.max(wiener)),
        "amplitude_identity_max_error": float(np.max(np.abs(speech + noise - degraded))),
    }


def _spectral_difference(
    reference: np.ndarray,
    degraded: np.ndarray,
    centres: np.ndarray,
    active: np.ndarray,
    optimization_diagnostics: dict | None = None,
) -> dict[str, float]:
    optimized_reference, _, details = optimize_reference(reference, degraded)
    if optimization_diagnostics is not None:
        optimization_diagnostics.update({
            "status": details["status"],
            "complete_reference_model": details["complete_reference_model"],
            "unimplemented_steps": details["unimplemented_steps"],
            "interpolation": details["interpolation"],
        })
    ref = _nonlinear(optimized_reference)
    deg = _nonlinear(degraded)
    count = min(ref.shape[1], deg.shape[1], len(active))
    ref, deg, active = ref[:, :count], deg[:, :count], active[:count]
    difference = deg - ref
    frequency_sets = {
        "low": (centres >= 50) & (centres < 3800),
        "mid": (centres >= 3800) & (centres < 7800),
        "high": (centres >= 7800) & (centres <= 20000),
    }
    asymmetry = {"low": 0.25, "mid": 0.40, "high": 0.50}
    result: dict[str, float] = {}
    for name, mask in frequency_sets.items():
        asymmetric = np.where(
            difference[mask] > 0,
            difference[mask],
            asymmetry[name] * difference[mask],
        )
        curve = np.sqrt(np.mean(asymmetric**2, axis=0))
        if len(curve) >= 5:
            curve = signal.medfilt(curve, kernel_size=5)
        active_curve = curve[active]
        result[name] = float(
            np.sqrt(np.mean(active_curve**2)) if len(active_curve) else np.nan
        )
    # The unpartitioned path in Figure 6.11 bypasses the band-specific
    # asymmetry factors. Only the low/mid/high paths apply Table 6.3.
    overall = difference
    curve = np.sqrt(np.mean(overall**2, axis=0))
    if len(curve) >= 5:
        curve = signal.medfilt(curve, kernel_size=5)
    result["overall"] = float(
        np.sqrt(np.mean(curve[active] ** 2)) if np.any(active) else np.nan
    )
    return result


def _modulation_features(
    reference: np.ndarray,
    speech: np.ndarray,
    centres: np.ndarray,
    active: np.ndarray,
) -> dict[str, float]:
    # Clause 6.3.7.2 uses P90 over the continuous timeline, rather than
    # concatenating the selected activity frames before an FFT.
    count = min(reference.shape[1], speech.shape[1], len(active))
    try:
        return modulation_features(
            reference[:, :count], speech[:, :count], centres, FRAME_SECONDS
        )
    except ValueError as error:
        raise ModelAError(f"modulation features: {error}") from error


def extract_model_a_features(
    reference: np.ndarray,
    degraded: np.ndarray,
    rate: int,
    *,
    degraded_rate: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict]:
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    if reference.ndim != 1 or degraded.ndim != 1:
        raise ModelAError("Model A requires mono reference and degraded PCM")
    if not np.isfinite(reference).all() or not np.isfinite(degraded).all():
        raise ModelAError("audio contains non-finite samples")
    reference = _resample(reference, rate)
    degraded = _resample(
        degraded,
        rate if degraded_rate is None else degraded_rate,
    )
    length = min(len(reference), len(degraded))
    reference, degraded = reference[:length], degraded[:length]
    reference, alignment = _align(reference, degraded)
    active, activity = _activity(reference)
    ref_spectrum, deg_spectrum, speech, noise, preprocessing = _transfer_and_residual(
        reference, degraded, active, activity
    )
    centres_99 = _band_centres()
    ref_spectrum, centres = _aggregate_basic_bands(
        ref_spectrum, centres_99
    )
    deg_spectrum, _ = _aggregate_basic_bands(deg_spectrum, centres_99)
    speech, _ = _aggregate_basic_bands(speech, centres_99)
    noise, _ = _aggregate_basic_bands(noise, centres_99)
    count = min(ref_spectrum.shape[1], deg_spectrum.shape[1], len(active))
    ref_spectrum = ref_spectrum[:, :count]
    deg_spectrum = deg_spectrum[:, :count]
    speech = speech[:, :count]
    noise = noise[:, :count]
    active = active[:count]

    a_weight = _a_weight(centres)[:, None]
    noise_a = noise * a_weight
    noise_l2_frequency = np.sqrt(np.sum(_nonlinear(noise_a) ** 2, axis=0))
    noise_sum_frequency = np.sum(_nonlinear(noise_a), axis=0)
    sharpness_time = _sharpness(_nonlinear(noise), centres)

    reference_optimization: dict = {}
    differences = _spectral_difference(
        ref_spectrum, speech, centres, active, reference_optimization,
    )
    processed_differences = _spectral_difference(
        ref_spectrum, deg_spectrum, centres, active
    )
    modulation = _modulation_features(
        ref_spectrum, speech, centres, active
    )
    # The control features use U/L/M/H, whereas ordinary speech regions
    # retain short pauses P. Noise averages include the whole timeline.
    ulmh = np.isin(activity["spectral_classes"][:count], ["U", "L", "M", "H"])
    if not np.any(ulmh):
        raise ModelAError("no U/L/M/H speech frames for control parameters")
    mean_speech = np.mean(speech[:, ulmh], axis=1)
    mean_active_speech = np.mean(speech[:, active], axis=1)
    mean_noise_a = np.mean(noise_a, axis=1)
    snr_a = 20 * np.log10(max(float(np.sum(
        mean_active_speech / np.maximum(mean_noise_a, 1e-30)
    )), 1e-30))
    asl = 20 * np.log10(max(float(np.mean(np.sum(speech[:, ulmh], axis=0))), 1e-30))
    transfer_amplitude = np.divide(
        mean_speech,
        np.maximum(np.mean(ref_spectrum[:, ulmh], axis=1), 1e-30),
    )
    erb = _effective_bandwidth(transfer_amplitude, centres)

    smos_features = {
        "Speech.HMNonLinModulation_Sp_MMER_AggBlockP90_WeightA_AggFreqSum": modulation[
            "mmer"
        ],
        "Speech.HMNonLinModulation_Sp_wA_P90T_L2F": modulation[
            "weighted_difference"
        ],
        "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_Low_Mod": differences[
            "low"
        ],
        "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_Mid_Mod": differences[
            "mid"
        ],
        "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_High_Mod": differences[
            "high"
        ],
        "Speech.HearingModelNonLinDifferenceL2TMaL2F_Sp": differences[
            "overall"
        ],
        "PreProcessing.SNR-A": snr_a,
        "PreProcessing.Proc.ASL.Compensated": asl,
        "FrequencyMetric.erb": erb,
    }
    nmos_features = {
        "BGN.HearingModelNonLinNoise_A_SumF_P90T_FR": float(
            np.percentile(noise_sum_frequency, 90)
        ),
        "BGN.HearingModelNonLinNoise_A_SumF_L2T_FR": float(
            np.sqrt(np.mean(noise_sum_frequency**2))
        ),
        "BGN.HearingModelNonLinNoise_A_L2F_KurtT_FR": float(
            kurtosis(noise_l2_frequency, fisher=True, bias=False)
        ),
        "BGN.HearingModelSharpness_P95T_FR": float(
            # Preserve the published forest's feature key; Figure 6.6 and
            # clause 6.3.5.3 explicitly define its value as time P90.
            np.percentile(sharpness_time, 90)
        ),
    }
    diagnostics = dict(
        alignment=alignment,
        activity=activity,
        band_count=len(centres),
        active_frames=int(np.count_nonzero(active)),
        overall_difference=processed_differences["overall"],
        erb_hz=erb,
        snr_a_db=snr_a,
        asl_db_model_scale=asl,
        preprocessing=preprocessing,
        input_scale_assumption="PCM amplitudes used as model pressure; no physical calibration inferred",
        sharpness_definition="Eq.12 weighted first Bark moment; time P90; no unstated acum scaling",
        reference_optimization=reference_optimization,
    )
    return dict(smos=smos_features, nmos=nmos_features, gmos=processed_differences), diagnostics


def _sharpness_weight(frequencies: np.ndarray) -> np.ndarray:
    # Zwicker/Fastl sharpness weighting with fractional Bark index.
    bark = 13 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan(
        (frequencies / 7500.0) ** 2
    )
    return np.maximum(1.0, 0.066 * np.exp(0.171 * bark))


def _sharpness(compressed_noise: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Equation (12): first moment, including its fractional Bark coordinate.

    The frequency-to-Bark mapping is Zwicker's usual analytic approximation.
    The TS equation does not supply the separate 0.11 acum normalization used
    by some other sharpness standards; no such fitted scale is introduced.
    """
    bark = 13 * np.arctan(0.00076 * centres) + 3.5 * np.arctan(
        (centres / 7500.0) ** 2
    )
    numerator = np.sum(
        compressed_noise * (_sharpness_weight(centres) * bark)[:, None], axis=0,
    )
    denominator = np.sum(compressed_noise, axis=0)
    return np.divide(
        numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0,
    )


def _effective_bandwidth(transfer: np.ndarray, centres: np.ndarray) -> float:
    levels = 20 * np.log10(np.maximum(np.asarray(transfer, dtype=np.float64), 1e-30))
    if not len(levels):
        return float("nan")
    # Equations (18)-(19) add 45 dB to H and integrate using the physical
    # auditory bandwidth Delta f(j), not the distance between band centres.
    response = np.maximum(levels + 45.0, 0.0)
    if np.max(response) <= 0:
        return float("nan")
    width = np.sum(response * (50.0 + 0.14 * centres)) / np.max(response)
    return float(width)


def load_models(directory: Path = MODEL_DIR) -> dict[str, RandomForestModel]:
    directory = Path(directory)
    expected = {}
    manifest = directory / "SHA256SUMS"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest
    paths = {
        "smos": directory / "S-MOS.model",
        "nmos": directory / "N-MOS.model",
        "gmos": directory / "G-MOS.model",
    }
    for path in paths.values():
        if not path.is_file():
            raise ModelAError(f"missing Model A asset: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name in expected and digest != expected[path.name]:
            raise ModelAError(f"Model A asset checksum mismatch: {path.name}")
    return {name: RandomForestModel.load(path) for name, path in paths.items()}


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_official_validation(
    path: Path = OFFICIAL_VALIDATION,
) -> dict:
    """Load Annex C evidence and bind it to this implementation revision."""
    path = Path(path)
    if not path.is_file():
        return {
            "status": "not_run",
            "conformant": False,
            "failed_reason": "No checked-in Annex C validation evidence is available.",
        }
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_evidence",
            "conformant": False,
            "failed_reason": f"Annex C validation evidence is unreadable: {error}",
        }
    project = Path(__file__).resolve().parents[2]
    # All Model A modules and package initializers are identity-bound. This
    # conservative set also invalidates evidence when a new evaluator module
    # is added; the verifier captures the transitive import closure itself.
    dependencies = [*Path(__file__).parent.glob("ts103281*.py"),
                    project / "server/__init__.py", Path(__file__).with_name("__init__.py")]
    dependency_sha256 = {
        source.relative_to(project).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in dependencies
    }
    model_sha256 = {
        source.relative_to(project).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in (MODEL_DIR / name for name in ("S-MOS.model", "N-MOS.model", "G-MOS.model"))
    }
    implementation_sha256 = _implementation_sha256()
    verifier_sha256 = hashlib.sha256(
        (project / "scripts/verify-ts103281-model-a.py").read_bytes()
    ).hexdigest()
    checked = validate_annex_c_evidence(
        evidence, implementation_sha256=implementation_sha256,
        dependency_sha256=dependency_sha256, model_sha256=model_sha256,
        verifier_sha256=verifier_sha256,
    )
    # Bind the embedded target rows to the actual Annex C table, rather than
    # trusting a self-consistent JSON whose "expected" values were changed.
    if checked["evidence_valid"]:
        with (MODEL_DIR / "annex-c-model-a.csv").open(newline="") as handle:
            official = {row["condition"]: {metric: float(row[column]) for metric, column in
                        (("S-MOS", "s_mos"), ("N-MOS", "n_mos"), ("G-MOS", "g_mos"))}
                        for row in csv.DictReader(handle)}
        with (MODEL_DIR / "annex-c-rms.csv").open(newline="") as handle:
            official_rms = {
                row["condition"]: float(row["rms_dbpa"])
                for row in csv.DictReader(handle)
            }
        references = {
            source.relative_to(project).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in (MODEL_DIR / "annex-c-model-a.csv", MODEL_DIR / "annex-c-rms.csv")
        }
        if evidence.get("reference_assets_sha256") != references:
            checked.update(status="stale", conformant=False, evidence_valid=False, failed_reason="Annex C reference asset identity mismatch")
        elif any(row["expected"] != official.get(row["condition"]) for row in evidence["rows"]):
            checked.update(status="invalid_evidence", conformant=False, evidence_valid=False, failed_reason="Expected scores disagree with Annex C table")
        elif any(
            row["input_calibration"].get("target_degraded_rms_dbpa")
            != official_rms.get(row["condition"])
            for row in evidence["rows"]
        ):
            checked.update(status="invalid_evidence", conformant=False, evidence_valid=False, failed_reason="Calibration targets disagree with Annex C RMS table")
    status = checked["status"]
    stale = status == "stale"
    historical = checked["historical"]
    count = checked["recomputed"].get("condition_count", historical.get("condition_count"))
    passed = checked["recomputed"].get("passed_conditions", historical.get("passed_conditions"))
    return {
        "status": status,
        "standard": evidence.get("standard", "ETSI TS 103 281 V1.3.2 Annex C"),
        "condition_count": count,
        "passed_conditions": passed,
        "tolerance_mos": EXPECTED_TOLERANCE_MOS,
        "conformant": checked["conformant"],
        "implementation_sha256": implementation_sha256,
        "evidence_implementation_sha256": evidence.get("implementation_sha256"),
        "stale": stale,
        "summary": evidence.get("summary", {}),
        "diagnostic_summary": evidence.get("diagnostic_summary", {}),
        "failed_reason": checked["failed_reason"],
        "evidence_validation": checked,
    }


def _diagnostic_scores(
    features: dict[str, dict[str, float]],
    models: dict[str, RandomForestModel],
) -> dict:
    """Research-only fit; never expose it as public conformance evidence."""
    smos_raw = features["smos"]
    nmos_raw = features["nmos"]
    p = _ANNEX_C_DIAGNOSTIC["smos"]
    smos_features = dict(smos_raw)
    smos_features[
        "Speech.HMNonLinModulation_Sp_MMER_AggBlockP90_WeightA_AggFreqSum"
    ] = p[0] * smos_raw[
        "Speech.HMNonLinModulation_Sp_MMER_AggBlockP90_WeightA_AggFreqSum"
    ] + p[1]
    smos_features[
        "Speech.HMNonLinModulation_Sp_wA_P90T_L2F"
    ] = p[2] * smos_raw[
        "Speech.HMNonLinModulation_Sp_wA_P90T_L2F"
    ] + p[3]
    for index, name in enumerate(
        (
            "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_Low_Mod",
            "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_Mid_Mod",
            "Speech.HearingModelNonLinDifference_Sp_L2TMa_L2F_High_Mod",
            "Speech.HearingModelNonLinDifferenceL2TMaL2F_Sp",
        )
    ):
        smos_features[name] = p[4 + index] * smos_raw[name]
    smos_features["PreProcessing.SNR-A"] = (
        p[8] * smos_raw["PreProcessing.SNR-A"] + p[9]
    )
    smos_features["PreProcessing.Proc.ASL.Compensated"] = (
        p[10] * smos_raw["PreProcessing.Proc.ASL.Compensated"] + p[11]
    )
    smos_features["FrequencyMetric.erb"] = 10 ** (
        p[12]
        * np.log10(max(smos_raw["FrequencyMetric.erb"], 1.0))
        + p[13]
    )

    n = _ANNEX_C_DIAGNOSTIC["nmos"]
    nmos_names = models["nmos"].feature_names
    nmos_features = dict(nmos_raw)
    nmos_features[nmos_names[0]] *= 10**n[0]
    nmos_features[nmos_names[1]] *= 10**n[1]
    nmos_features[nmos_names[2]] = (
        n[2] * nmos_raw[nmos_names[2]] + n[3]
    )
    nmos_features[nmos_names[3]] *= n[4]

    smos = models["smos"].predict(smos_features)
    nmos = models["nmos"].predict(nmos_features)
    g = _ANNEX_C_DIAGNOSTIC["gmos"]
    gmos_features = {
        "N-MOS_instrumental": g[0] * nmos["value"] + g[1],
        "S-MOS_instrumental": g[2] * smos["value"] + g[3],
        "Speech.HearingModelNonLinDifferenceL2TMaL2F_Proc": (
            g[4] * features["gmos"]["overall"] + g[5]
        ),
    }
    gmos = models["gmos"].predict(gmos_features)
    return {
        "scores": {"S-MOS": smos, "N-MOS": nmos, "G-MOS": gmos},
        "calibration": {
            "id": "annex-c-diagnostic-fit-v1",
            "purpose": "engineering diagnostics only",
            "independent_validation": False,
            "conformance_evidence": False,
        },
    }


def evaluate_model_a(
    reference: np.ndarray,
    degraded: np.ndarray,
    rate: int,
    *,
    degraded_rate: int | None = None,
    models: dict[str, RandomForestModel] | None = None,
    include_diagnostic_calibration: bool = False,
) -> dict:
    models = load_models() if models is None else models
    features, diagnostics = extract_model_a_features(
        reference,
        degraded,
        rate,
        degraded_rate=degraded_rate,
    )
    smos = models["smos"].predict(features["smos"])
    nmos = models["nmos"].predict(features["nmos"])
    g_features = {
        "N-MOS_instrumental": nmos["value"],
        "S-MOS_instrumental": smos["value"],
        "Speech.HearingModelNonLinDifferenceL2TMaL2F_Proc": features["gmos"][
            "overall"
        ],
    }
    gmos = models["gmos"].predict(g_features)
    validation = load_official_validation()
    result = dict(
        standard="ETSI TS 103 281 V1.3.2",
        model="A",
        scope="SWB/FB listening-quality prediction",
        compliance_status=(
            "annex_c_verified"
            if validation.get("status") == "passed"
            else "experimental_nonconformant"
        ),
        score_use="experimental_engineering_diagnostic",
        formal_compliance_claim_allowed=False,
        official_tolerance_mos=EXPECTED_TOLERANCE_MOS,
        verification=validation,
        scores={
            "S-MOS": smos,
            "N-MOS": nmos,
            "G-MOS": gmos,
        },
        features=features,
        diagnostics=diagnostics,
        limitations=[
            "The published random-forest configuration is executed directly.",
            "The current implementation fails the official Annex C +/-0.1 MOS gate.",
            "The missing hearing-model, reference-optimisation and feature details "
            "must be replaced with a conforming reference implementation.",
            "Model A does not establish Model B compliance or NB/WB 3QUEST coverage.",
        ],
    )
    if include_diagnostic_calibration:
        diagnostic = _diagnostic_scores(features, models)
        result["diagnostic_scores"] = diagnostic["scores"]
        result["diagnostic_calibration"] = diagnostic["calibration"]
    return result
