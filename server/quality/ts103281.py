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
import hashlib
import json
from pathlib import Path
from typing import Iterable
import math

import numpy as np
from scipy import signal
from scipy.special import comb
from scipy.stats import kurtosis


SAMPLE_RATE = 48_000
FRAME_SECONDS = 0.008
FRAME_SIZE = round(SAMPLE_RATE * FRAME_SECONDS)
FRAME_HOP = FRAME_SIZE
P0_PA = 20e-6
MODEL_DIR = Path(__file__).with_name("assets") / "ts103281"
OFFICIAL_VALIDATION = MODEL_DIR / "annex-c-validation.json"
EXPECTED_TOLERANCE_MOS = 0.1
_ACTIVE_NOISE_MASK = 0.4

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


def _frame_rms(x: np.ndarray) -> np.ndarray:
    count = len(x) // FRAME_SIZE
    if count == 0:
        return np.empty(0, dtype=np.float64)
    frames = x[: count * FRAME_SIZE].reshape(count, FRAME_SIZE)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _activity(reference: np.ndarray) -> tuple[np.ndarray, dict]:
    activity_frame_size = round(SAMPLE_RATE * 0.010)
    activity_frame_count = len(reference) // activity_frame_size
    if not activity_frame_count:
        raise ModelAError("reference is too short for 10 ms activity frames")
    frames = reference[: activity_frame_count * activity_frame_size].reshape(
        activity_frame_count,
        activity_frame_size,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    levels = 20 * np.log10(np.maximum(rms / P0_PA, 1e-30))
    floor = float(np.percentile(levels, 15))
    peak = float(np.percentile(levels, 95))
    threshold = max(floor + 8.0, peak - 30.0)
    active_activity_frames = levels > threshold
    if np.count_nonzero(active_activity_frames) < 3:
        raise ModelAError("reference has insufficient active speech")
    spectral_frame_count = len(reference) // FRAME_SIZE
    spectral_centres = (np.arange(spectral_frame_count) + 0.5) * FRAME_SECONDS
    activity_indices = np.minimum(
        (spectral_centres / 0.010).astype(np.int64),
        activity_frame_count - 1,
    )
    active = active_activity_frames[activity_indices]
    return active, dict(
        floor_dbfs_pa=floor,
        peak_dbfs_pa=peak,
        threshold_dbfs_pa=threshold,
        activity_frame_ms=10.0,
        active_activity_frames=int(np.count_nonzero(active_activity_frames)),
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
    """Return the rectified 8 ms auditory-filter output on 99 bands."""
    centres = _band_centres()
    count = len(x) // FRAME_SIZE
    if count == 0:
        raise ModelAError("audio is too short for 8 ms auditory frames")
    framed_length = count * FRAME_SIZE
    bandwidth = np.sqrt(50.0**2 + (0.14 * centres) ** 2)
    order = 5
    coefficients = np.asarray([0, 1, 11, 11, 1], dtype=np.float64)
    amplitude = np.empty((len(centres), count), dtype=np.float64)
    for index, (frequency, dfz) in enumerate(zip(centres, bandwidth)):
        tau = (
            comb(2 * order - 2, order - 1)
            / (2 ** (2 * order - 1) * dfz)
        )
        decay = np.exp(-1.0 / (SAMPLE_RATE * tau))
        phase = np.exp(
            1j
            * 2
            * np.pi
            * frequency
            * np.arange(0, order + 2)
            / SAMPLE_RATE
        )
        m_a = np.arange(1, order + 1)
        denominator = np.append(
            1,
            ((-decay) ** m_a) * comb(order, m_a),
        ) * phase[: order + 1]
        m_b = np.arange(0, order)
        b_index = np.arange(1, order)
        numerator = (
            ((1 - decay) ** order)
            / np.sum(coefficients[1:] * (decay**b_index))
            * (decay**m_b)
            * coefficients
            * phase[:order]
        )
        filtered = 2 * np.real(
            signal.lfilter(numerator, denominator, x[:framed_length])
        )
        rectified = np.maximum(filtered, 0.0).reshape(count, FRAME_SIZE)
        amplitude[index] = np.sqrt(
            2.0 * np.mean(rectified * rectified, axis=1)
        )
    return amplitude, centres


def _nonlinear(x: np.ndarray) -> np.ndarray:
    # ETSI EG 202 396-3 Annex K, equation K.1/K.2-K.4.
    exponent = 0.2
    threshold = 5.66
    a = 0.5 * (1.0 - exponent) / (2.0 - exponent) / threshold
    b = (1.0 / exponent) * (1.0 / (2.0 - exponent)) * threshold ** (1.0 - 1.0 / exponent)
    c = 0.5 * (1.0 - exponent) / exponent * threshold
    value = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
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
    return 10 ** (20 * np.log10(np.maximum(numerator / denominator, 1e-30)) / 20)


def _transfer_and_residual(
    reference: np.ndarray,
    degraded: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_spectrum, centres = _auditory_spectrum(reference)
    deg_spectrum, _ = _auditory_spectrum(degraded)
    frame_count = min(ref_spectrum.shape[1], deg_spectrum.shape[1], len(active))
    ref_spectrum = ref_spectrum[:, :frame_count]
    deg_spectrum = deg_spectrum[:, :frame_count]
    active = active[:frame_count]
    if not np.any(active):
        raise ModelAError("no active speech frames after spectral analysis")
    cross = np.mean(ref_spectrum[:, active] * deg_spectrum[:, active], axis=1)
    power = np.mean(ref_spectrum[:, active] ** 2, axis=1)
    gain = np.divide(
        cross,
        power,
        out=np.ones_like(cross),
        where=power > 1e-30,
    )
    # H1 magnitude gain, clipped to avoid sign inversion dominating noise.
    gain = np.clip(gain, 0.0, None)
    speech = gain[:, None] * ref_spectrum
    residual = np.maximum(deg_spectrum - speech, 0.0)
    # The activity spectrum masks unreliable speech-dominated bins before the
    # iterative deconvolution described by Model A.
    residual[:, active] *= _ACTIVE_NOISE_MASK
    return ref_spectrum, deg_spectrum, speech, residual


def _active_rms_level(x: np.ndarray, active: np.ndarray) -> float:
    rms = _frame_rms(x)
    count = min(len(rms), len(active))
    selected = rms[:count][active[:count]]
    if not len(selected):
        return float("nan")
    # Model A's compensated ASL feature is expressed on the digital level
    # scale used by the training data, not as dB SPL re 20 uPa.
    return float(20 * np.log10(max(float(np.sqrt(np.mean(selected**2))), 1e-30)))


def _spectral_difference(
    reference: np.ndarray,
    degraded: np.ndarray,
    centres: np.ndarray,
    active: np.ndarray,
) -> dict[str, float]:
    ref = 20 * np.log10(np.maximum(_nonlinear(reference), 1e-30))
    deg = 20 * np.log10(np.maximum(_nonlinear(degraded), 1e-30))
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
    overall = np.where(difference > 0, difference, 0.5 * difference)
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
    # Approximate the 4-128 Hz modulation bank by taking the largest
    # FFT component in that band for each auditory channel.
    count = min(reference.shape[1], speech.shape[1], len(active))
    ref = reference[:, :count]
    out = speech[:, :count]
    active = active[:count]
    if np.count_nonzero(active) < 8:
        return dict(mmer=float("nan"), weighted_difference=float("nan"))
    # Interpolate over inactive frames only for the modulation transform.
    ref_active = ref[:, active]
    out_active = out[:, active]
    n_fft = 1 << int(np.ceil(np.log2(max(8, ref_active.shape[1]))))
    mod_freq = np.fft.rfftfreq(n_fft, d=FRAME_SECONDS)
    mod_mask = (mod_freq >= 4.0) & (mod_freq <= 128.0)
    if not np.any(mod_mask):
        return dict(mmer=float("nan"), weighted_difference=float("nan"))
    ref_mod = np.abs(np.fft.rfft(ref_active, n=n_fft, axis=1))[:, mod_mask]
    out_mod = np.abs(np.fft.rfft(out_active, n=n_fft, axis=1))[:, mod_mask]
    selected = np.argmax(ref_mod, axis=1)
    rows = np.arange(len(centres))
    ref_peak = ref_mod[rows, selected]
    out_peak = out_mod[rows, selected]
    ratio_db = 20 * np.log10(
        np.maximum(out_peak, 1e-30) / np.maximum(ref_peak, 1e-30)
    )
    difference = out_mod - ref_mod
    weights = np.maximum(10 ** (_a_weight(centres) / 20), 1.0)
    weighted = np.sqrt(np.mean((difference * weights[:, None]) ** 2, axis=1))
    return dict(
        mmer=float(np.mean(ratio_db)),
        weighted_difference=float(
            20 * np.log10(max(float(np.sqrt(np.mean(weighted**2))), 1e-30))
        ),
    )


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
    ref_spectrum, deg_spectrum, speech, noise = _transfer_and_residual(
        reference, degraded, active
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
    speech_a = speech * a_weight
    noise_a = noise * a_weight
    noise_l2_frequency = np.sqrt(np.mean(_nonlinear(noise_a) ** 2, axis=0))
    noise_sum_frequency = np.sum(_nonlinear(noise_a), axis=0)
    loudness_time = np.sum(_nonlinear(noise), axis=0)
    sharpness_time = np.divide(
        np.sum(_nonlinear(noise) * _sharpness_weight(centres)[:, None], axis=0),
        np.maximum(np.sum(_nonlinear(noise), axis=0), 1e-30),
    )

    differences = _spectral_difference(
        ref_spectrum, speech, centres, active
    )
    processed_differences = _spectral_difference(
        ref_spectrum, deg_spectrum, centres, active
    )
    modulation = _modulation_features(
        ref_spectrum, speech, centres, active
    )
    speech_power = float(np.mean(speech_a[:, active] ** 2))
    noise_power = float(np.mean(noise_a[:, active] ** 2))
    snr_a = 10 * np.log10(max(speech_power, 1e-30) / max(noise_power, 1e-30))
    asl = _active_rms_level(reference, active)
    transfer_power = np.divide(
        np.mean(deg_spectrum[:, active] ** 2, axis=1),
        np.maximum(np.mean(ref_spectrum[:, active] ** 2, axis=1), 1e-30),
    )
    erb = _effective_bandwidth(np.sqrt(transfer_power), centres)

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
            kurtosis(noise_l2_frequency, fisher=False, bias=False)
        ),
        "BGN.HearingModelSharpness_P95T_FR": float(
            np.percentile(sharpness_time, 95)
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
        asl_dbfs_pa=asl,
    )
    return dict(smos=smos_features, nmos=nmos_features, gmos=processed_differences), diagnostics


def _sharpness_weight(frequencies: np.ndarray) -> np.ndarray:
    # Zwicker/Fastl sharpness weighting with fractional Bark index.
    bark = 13 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan(
        (frequencies / 7500.0) ** 2
    )
    return np.maximum(1.0, 0.066 * np.exp(0.171 * bark))


def _effective_bandwidth(transfer: np.ndarray, centres: np.ndarray) -> float:
    levels = 20 * np.log10(np.maximum(np.asarray(transfer, dtype=np.float64), 1e-30))
    if not len(levels):
        return float("nan")
    ceiling = float(np.max(levels))
    # ETSI TS 103 281 equations (18)-(19): clip 45 dB below the maximum, then
    # divide the response area by its maximum.  The overlapping channels are
    # integrated on the physical frequency axis.
    response = np.maximum(levels - (ceiling - 45.0), 0.0)
    if np.max(response) <= 0:
        return float("nan")
    width = np.trapezoid(response, centres) / np.max(response)
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
    implementation_sha256 = _implementation_sha256()
    evidence_sha256 = evidence.get("implementation_sha256")
    stale = evidence_sha256 != implementation_sha256
    passed = int(evidence.get("passed_conditions", 0))
    count = int(evidence.get("condition_count", 0))
    conformant = bool(evidence.get("conformant")) and not stale
    if stale:
        status = "stale"
        failed_reason = (
            "The checked-in Annex C evidence was produced by a different "
            "implementation revision and must be regenerated."
        )
    elif conformant:
        status = "passed"
        failed_reason = None
    else:
        status = "failed_tolerance_gate"
        failed_reason = (
            f"{passed}/{count} official conditions reproduced every score "
            f"within +/-{EXPECTED_TOLERANCE_MOS} MOS."
        )
    return {
        "status": status,
        "standard": evidence.get("standard", "ETSI TS 103 281 V1.3.2 Annex C"),
        "condition_count": count,
        "passed_conditions": passed,
        "tolerance_mos": float(
            evidence.get("tolerance_mos", EXPECTED_TOLERANCE_MOS)
        ),
        "conformant": conformant,
        "implementation_sha256": implementation_sha256,
        "evidence_implementation_sha256": evidence_sha256,
        "stale": stale,
        "summary": evidence.get("summary", {}),
        "diagnostic_summary": evidence.get("diagnostic_summary", {}),
        "failed_reason": failed_reason,
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
