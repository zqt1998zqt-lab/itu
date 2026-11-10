#!/usr/bin/env python3
"""Build a measurement-aware, reproducible 336-item engineering dataset.

Uses licensed local speech/noise, distinct measurement excitations and explicit
provenance. This is software regression material, never certification evidence.
All output paths are exclusive; failed artifacts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from scipy import signal
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from acceptance_signal_design import OOB_DIAGNOSTIC_HZ, render_signals
from server.acoustic.acceptance_contract import contract_items, required_conditions
from server.acoustic.acceptance_dataset import (
    MAX_ARCHIVE_BYTES, MAX_ASSET_BYTES, MAX_TOTAL_UNCOMPRESSED_BYTES,
    import_dataset,
)
from server.acoustic.acceptance_models import (
    FREQUENCY_RESPONSE_CENTERS_HZ,
    MODE_UPPER_HZ,
)

NAME = "measurement-aware-v2-not-certification"
DURATION_S = 12.0
ROLES = ("microphone", "reference", "far_end", "noise")
SPEECH_ROOT = PROJECT / "datasets/itu-eval-v4-quality-filtered-20261010/sources"
FULLBAND_ROOT = PROJECT / "datasets/independent-fullband-20261012"
SNR_DB = {"baseline": 24., "sce1": 15., "sce2": 10., "sce3": 6., "sce4": 3.}
VOLUME_DB = {"nominal": 0., "maximum": 6., "recommended": -2.}
NOISE_PATHS = {"baseline": "noise/TCAR-ch01.wav", "sce1": "noise/TCAR-ch01.wav",
               "sce2": "noise/STRAFFIC-ch01.wav", "sce3": "noise/TCAR-ch01.wav",
               "sce4": "noise/MS-SNSD-AirConditioner_3.wav"}
BASELINE_ID = "74405324e0d24b27bf8821a27b605652"
MIN_EXCITATION_BAND_POWER_RATIO = 1e-3
MIN_OOB_DIAGNOSTIC_POWER_RATIO = 1e-2
MEASUREMENT_DB_TOLERANCE = .02


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=float) ** 2)))


def db_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return float(20.0 * math.log10(numerator / denominator))


def stable_seed(text: str, seed: int = 20261104) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:4], "little")


def profile_for(row: dict) -> str:
    family = row["family"]
    if family in {"g_mos", "n_mos", "s_mos", "listening_quality"}:
        return "speech"
    if family in {"rlr", "rlr_linearity", "frequency_response"}:
        return "response"
    if family in {"idle_noise_level", "idle_noise_peaks"}:
        return "idle"
    if family in {"cng_level", "cng_spectrum"}:
        return "cng"
    if family == "out_of_band":
        return "oob"
    return "duplex"


def sample_rate(conditions: dict, kind: str) -> int:
    band = conditions["mode"].split("-")[0]
    if band == "SWB":
        return 48000
    if band == "WB" and kind in {"oob", "response"}:
        return 24000
    return 16000


def plan_cases() -> tuple[dict, dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in contract_items():
        conditions = required_conditions(row)
        groups.setdefault((tuple(conditions.items()), profile_for(row)), []).append(row)
    manifest = {"schema_version": 1, "name": NAME, "cases": []}
    plans = {}
    for index, ((condition_pairs, kind), rows) in enumerate(groups.items(), 1):
        conditions = dict(condition_pairs)
        identifier = f"v2-{index:04d}-{kind}"
        rate = sample_rate(conditions, kind)
        start = 4. if kind in {"cng", "idle", "oob"} else 0.5
        case = {"id": identifier, "conditions": conditions,
                "item_ids": [row["id"] for row in rows], "signals": {},
                "analysis_start_s": start}
        if conditions["direction"] != "receive":
            case["beam_delays_ms"] = [1000. / rate, 0.]
        manifest["cases"].append(case)
        plans[identifier] = {"kind": kind, "rate": rate,
                             "families": sorted({row["family"] for row in rows})}
    identities = [item for case in manifest["cases"] for item in case["item_ids"]]
    if len(identities) != 336 or len(set(identities)) != 336:
        raise ValueError("336 unique contract identities are required")
    return manifest, plans


class SourceBank:
    """Hash-verified licensed sources, with recorded crops and no audio looping."""
    def __init__(self):
        self.speech_manifest = SPEECH_ROOT / "manifest.json"
        self.fullband_manifest = FULLBAND_ROOT / "dataset-manifest.json"
        sources = json.loads(self.speech_manifest.read_text())["sources"]
        self.noises = {row["path"]: row for row in sources if row.get("kind") == "noise"}
        self.speakers = defaultdict(list)
        for row in sources:
            if row.get("kind") == "clean_speech" and row.get("license") == "CC-BY-4.0":
                self.speakers[str(row["speaker"])].append(row)
        female = sorted(s for s, rows in self.speakers.items() if rows[0].get("sex") == "F")
        male = sorted(s for s, rows in self.speakers.items() if rows[0].get("sex") == "M")
        if len(female) < 6 or len(male) < 6:
            raise ValueError("need at least six female and six male licensed speech sources")
        self.pairs = [(female[i], male[i]) if i % 2 == 0 else (male[i], female[i]) for i in range(6)]
        full = json.loads(self.fullband_manifest.read_text())["files"]
        self.full_speakers = defaultdict(list)
        for row in full:
            if row.get("language") == "en" and row.get("native_sample_rate_hz") == 48000:
                self.full_speakers[str(row["speaker_id"])].append(row)
        self.full_pair = sorted(self.full_speakers)[:2]
        if len(self.full_pair) != 2:
            raise ValueError("native 48 kHz near/far speakers required for SWB")
        self.used: dict[str, dict] = {}
        self.modes = list(dict.fromkeys(row["sheet"] for row in contract_items()))

    @lru_cache(maxsize=160)
    def _read(self, path: str, expected: str) -> tuple[np.ndarray, int]:
        p = Path(path)
        actual = file_sha(p)
        if actual != expected:
            raise ValueError(f"source SHA mismatch: {p}")
        samples, rate = sf.read(p, always_2d=True)
        samples = samples[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError(f"non-finite source: {p}")
        return samples, int(rate)

    @lru_cache(maxsize=64)
    def speech(self, mode: str, role: str, rate: int) -> tuple[np.ndarray, dict]:
        index = self.modes.index(mode)
        native = mode.startswith("SWB-")
        if native:
            swb_index = [m for m in self.modes if m.startswith("SWB-")].index(mode)
            speaker = self.full_pair[(swb_index + (role == "far_end")) % 2]
            records = sorted(self.full_speakers[speaker], key=lambda r: r["native_path"])
            root = FULLBAND_ROOT
        else:
            nb_wb_modes = [m for m in self.modes if not m.startswith("SWB-")]
            pair = self.pairs[nb_wb_modes.index(mode)]
            speaker = pair[role == "far_end"]
            records = sorted(self.speakers[speaker], key=lambda r: r["path"])
            root = SPEECH_ROOT
        parts, details = [], []
        required = int(round(DURATION_S * rate))
        count = 0
        for record in records:
            rel = record["native_path"] if native else record["path"]
            expected = record["native_sha256"] if native else record["sha256"]
            original, source_rate = self._read(str(root / rel), expected)
            common = math.gcd(rate, source_rate)
            samples = signal.resample_poly(original, rate // common, source_rate // common) if rate != source_rate else original.copy()
            take = min(len(samples), required - count)
            samples = samples[:take]
            fade = min(round(.01 * rate), len(samples) // 2)
            if fade:
                taper = np.linspace(0., 1., fade)
                samples[:fade] *= taper
                samples[-fade:] *= taper[::-1]
            parts.append(samples)
            count += take
            detail = {"path": str(root / rel), "sha256": expected, "speaker": speaker,
                      "source_sample_rate_hz": source_rate, "selected_samples_at_target_rate": take,
                      "license": "CC-BY-4.0" if native else record["license"],
                      "dataset": "VoiceBank/VCTK clean testset" if native else record["dataset"],
                      "language": "en", "repeated": False}
            if not native:
                detail.update(sex=record.get("sex"), attribution=record.get("attribution"), source_page=record.get("source_page"))
            else:
                detail.update(source_page="https://datashare.ed.ac.uk/handle/10283/2791", attribution="VoiceBank corpus; Valentini-Botinhao et al.; upstream attribution retained in acquisition manifest")
            details.append(detail)
            self.used[str(root / rel)] = detail
            if count == required:
                break
        if count != required:
            raise ValueError(f"speaker {speaker} has insufficient non-repeated material")
        samples = np.concatenate(parts)
        # Business bandwidth is deliberately simulated; transport/codec is not.
        cutoff = {"NB": 3400., "WB": 7000., "SWB": 14000.}[mode.split("-")[0]]
        if cutoff < rate / 2:
            samples = signal.sosfiltfilt(signal.butter(6, cutoff, fs=rate, output="sos"), samples)
        # Condition the source once, before mixing or applying level variants.
        # Reserve enough headroom for +6 dB playback without per-case limiting.
        peak = float(np.max(np.abs(samples)))
        if peak <= 1e-8:
            raise ValueError(f"silent source speaker: {speaker}")
        source_gain = .12 / peak
        samples *= source_gain
        return samples, {"speaker": speaker, "native_fullband": native, "segments": details,
                         "band_limit_hz": cutoff, "transport_simulated": False,
                         "source_gain_linear": source_gain, "conditioned_peak": .12}

    @lru_cache(maxsize=64)
    def noise(self, mode: str, scene: str, rate: int, seed: int) -> tuple[np.ndarray, dict]:
        path = NOISE_PATHS[scene]
        row = self.noises[path]
        recorded, source_rate = self._read(str(SPEECH_ROOT / path), row["sha256"])
        rng = np.random.default_rng(stable_seed(mode + scene, seed))
        frames = round(DURATION_S * source_rate)
        if len(recorded) < frames:
            raise ValueError(f"noise source too short without looping: {path}")
        offset = int(rng.integers(0, len(recorded) - frames + 1))
        crop = recorded[offset:offset + frames].copy()
        divisor = math.gcd(source_rate, rate)
        crop = signal.resample_poly(crop, rate // divisor, source_rate // divisor) if rate != source_rate else crop
        crop = crop[:round(DURATION_S * rate)]
        crop -= np.mean(crop)
        crop /= max(rms(crop), 1e-12)
        # A controlled broadband component makes SWB noise observable without
        # falsely claiming a recorded 48 kHz vehicle scenario.
        frequencies = np.fft.rfftfreq(len(crop), 1. / rate)
        white = rng.standard_normal(len(crop))
        alpha = {"baseline": 1.4, "sce1": 1.3, "sce2": .9, "sce3": .6, "sce4": .8}[scene]
        shape = (np.maximum(frequencies, 30.) / 1000.) ** (-alpha / 2.)
        shape[frequencies < 30.] = 0.
        synthetic = np.fft.irfft(np.fft.rfft(white) * shape, n=len(crop))
        synthetic /= max(rms(synthetic), 1e-12)
        time = np.arange(len(crop)) / rate
        if scene in {"sce2", "sce3", "sce4"}:
            synthetic *= 1. + .15 * np.sin(2 * np.pi * .4 * time)
        noise = math.sqrt(.7) * crop + math.sqrt(.3) * synthetic
        noise /= rms(noise)
        source = {"path": str(SPEECH_ROOT / path), "sha256": row["sha256"],
                  "dataset": row["dataset"], "license": row["license"],
                  "attribution": row.get("attribution"), "source_page": row.get("source_page"),
                  "source_sample_rate_hz": source_rate, "start_sample": offset,
                  "duration_s": DURATION_S, "recorded_mix_power_weight": .7,
                  "synthetic_mix_power_weight": .3, "scenario": scene,
                  "actual_vehicle_speed_recording": False,
                  "scope": "controlled hybrid noise proxy; scenario labels are engineering recipes"}
        self.used[str(SPEECH_ROOT / path)] = source
        return noise, source


def _hann_power(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("spectral validation requires finite one-dimensional PCM")
    windowed = (values - float(np.mean(values))) * signal.windows.hann(len(values), sym=False)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / rate)
    total = float(np.sum(power))
    if total <= 0.0:
        raise ValueError("spectral validation requires non-silent PCM")
    return frequencies, power


def _segment_mask(metadata: dict, rate: int, frames: int, labels: set[str]) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    segments = metadata.get("segments")
    if not isinstance(segments, list):
        raise ValueError("sidecar segments are required for SNR/SER validation")
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("label") not in labels:
            continue
        start = max(0, min(frames, int(round(float(segment["start_s"]) * rate))))
        end = max(0, min(frames, int(round(float(segment["end_s"]) * rate))))
        mask[start:end] = True
    if not np.any(mask):
        raise ValueError("sidecar segments do not define the required measurement window")
    return mask


def _fir_component(
    source: np.ndarray,
    path: dict | None,
) -> np.ndarray:
    component = np.zeros(len(source), dtype=np.float64)
    if path is None:
        return component
    offsets = path.get("tap_samples")
    gains = path.get("tap_gains")
    echo_gain = path.get("echo_gain_linear")
    if not isinstance(offsets, list) or not isinstance(gains, list) or len(offsets) != len(gains):
        raise ValueError("echo FIR requires matching tap_samples and tap_gains")
    if echo_gain is None or not math.isfinite(float(echo_gain)):
        raise ValueError("echo FIR requires finite echo_gain_linear")
    for raw_offset, raw_gain in zip(offsets, gains, strict=True):
        offset, gain = int(raw_offset), float(raw_gain)
        if offset < 0 or offset != raw_offset or not math.isfinite(gain):
            raise ValueError("echo FIR taps must be finite non-negative integer samples")
        if offset < len(component):
            component[offset:] += source[:len(component) - offset] * gain
    # source is the delivered far_end PCM and already includes any common
    # safety gain. Applying global_safety_gain here would square that gain.
    return component * float(echo_gain)


def _reconstruct_noise_channel(noise0: np.ndarray, metadata: dict, channel: int) -> np.ndarray:
    if channel == 0:
        return np.asarray(noise0, dtype=np.float64)
    recipe = metadata.get("noise_spatial", {}).get("channel_1_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("dual-microphone verification requires noise[:, 1]")
    rho = recipe.get("rho")
    seed2 = recipe.get("seed2")
    normalizer = recipe.get("normalizer")
    scale = recipe.get("noise_channel_scale_linear")
    if (
        rho is None or not math.isfinite(float(rho))
        or not -1.0 < float(rho) < 1.0
        or seed2 is None or int(seed2) != seed2
        or normalizer is None or not math.isfinite(float(normalizer))
        or float(normalizer) <= 0.0
        or scale is None or float(scale) <= 0.0
    ):
        raise ValueError("dual-microphone noise recipe is incomplete")
    unit0 = np.asarray(noise0, dtype=np.float64) / float(scale)
    independent2 = np.random.default_rng(int(seed2)).normal(size=len(unit0))
    mixed = (
        float(rho) * unit0
        + math.sqrt(1.0 - float(rho) ** 2) * independent2
    )
    return mixed / float(normalizer) * float(scale)


def _validate_sidecar_measurements(
    signals: dict,
    metadata: dict,
    *,
    plan: dict,
) -> dict:
    rate = int(plan["rate"])
    kind = plan["kind"]
    frames = round(DURATION_S * rate)
    near_labels = {
        "speech": {"speech"},
        "duplex": {"near_only", "double_talk"},
        "cng": {"cng_active"},
        "response": {"response"},
        "oob": {"oob"},
    }.get(kind)
    measured_snr = None
    if metadata.get("direction") != "receive" and near_labels is not None:
        near_mask = _segment_mask(metadata, rate, frames, near_labels)
        noise = np.asarray(signals["noise"], dtype=np.float64)
        noise0 = noise[:, 0] if noise.ndim == 2 else noise
        measured_snr = db_ratio(rms(np.asarray(signals["reference"])[near_mask]), rms(noise0[near_mask]))

    measured_ser = None
    if kind == "duplex":
        far_mask = _segment_mask(metadata, rate, frames, {"far_only"})
        near_mask = _segment_mask(metadata, rate, frames, {"near_only", "double_talk"})
        paths = metadata.get("echo_paths", [])
        path0 = paths[0] if isinstance(paths, list) and paths else None
        echo = _fir_component(
            np.asarray(signals["far_end"], dtype=np.float64),
            path0,
        )
        measured_ser = db_ratio(rms(echo[far_mask]), rms(np.asarray(signals["reference"])[near_mask]))

    checked = {}
    for metric, actual in (("snr", measured_snr), ("ser", measured_ser)):
        applicable = metadata.get(f"{metric}_db_target_applicable")
        target = metadata.get(f"{metric}_db_target")
        reported = metadata.get(f"{metric}_db_measured")
        if applicable is not True:
            if target is not None:
                raise ValueError(f"{metric.upper()} target must be null when not applicable")
            if reported is not None:
                raise ValueError(f"{metric.upper()} measured value must be null when not applicable")
            checked[metric] = {"applicable": False, "target_db": None, "measured_db": None}
            continue
        if target is None or not math.isfinite(float(target)):
            raise ValueError(f"applicable {metric.upper()} target must be finite")
        if reported is None or not math.isfinite(float(reported)):
            raise ValueError(f"applicable {metric.upper()} measured value must be finite")
        effective_target = float(target)
        if metric == "ser":
            raw_effective = metadata.get("ser_db_effective_target")
            if raw_effective is None or not math.isfinite(float(raw_effective)):
                raise ValueError("applicable SER effective target must be finite")
            effective_target = float(raw_effective)
        if actual is None or not math.isfinite(actual):
            raise ValueError(f"post-PCM {metric.upper()} is not measurable")
        for label, value in (
            ("sidecar measured", float(reported)),
            ("post-PCM measured", float(actual)),
        ):
            if abs(value - effective_target) > MEASUREMENT_DB_TOLERANCE:
                raise ValueError(
                    f"{metric.upper()} {label} differs from effective target by more than "
                    f"{MEASUREMENT_DB_TOLERANCE:g} dB"
                )
        if abs(float(reported) - float(actual)) > MEASUREMENT_DB_TOLERANCE:
            raise ValueError(
                f"{metric.upper()} sidecar measured value disagrees with post-PCM measurement"
            )
        checked[metric] = {
            "applicable": True,
            "target_db": effective_target,
            "sidecar_measured_db": float(reported),
            "post_pcm_measured_db": float(actual),
            "tolerance_db": MEASUREMENT_DB_TOLERANCE,
        }
    return checked


def validate_case(signals: dict, case: dict, plan: dict, sidecar: dict | None = None) -> dict:
    rate, kind = plan["rate"], plan["kind"]
    frames = round(DURATION_S * rate)
    for role in ROLES:
        y = np.asarray(signals[role])
        if len(y) != frames or not np.isfinite(y).all() or np.max(np.abs(y)) >= .7:
            raise ValueError(f"{case['id']} invalid audio {role}")
    if np.asarray(signals["noise"]).ndim != 1:
        raise ValueError("delivered noise must be mono")
    near, far = signals["reference"], signals["far_end"]
    if case["conditions"]["direction"] != "receive":
        if np.shape(signals["microphone"]) != (frames, 2):
            raise ValueError("two microphone channels required for BF coverage")
    start = round(case["analysis_start_s"] * rate)
    window = case["conditions"]["window_ms"]
    end = start + round(window * rate / 1000) if window else frames
    result = {"checked": True, "kind": kind, "peak": max(float(np.max(np.abs(v))) for v in signals.values())}
    if kind in {"idle", "cng"}:
        if np.max(np.abs(near[start:end])) != 0 or np.max(np.abs(far[start:end])) != 0:
            raise ValueError(f"{kind} analysis window contaminated by speech")
        result["quiet_window_s"] = [start / rate, end / rate]
        result["speech_contamination_samples"] = 0
        if kind == "idle" and case["conditions"]["direction"] == "receive" and np.count_nonzero(far):
            raise ValueError("receive idle must be actual digital silence")
    if kind == "duplex":
        for label, a, b, near_active, far_active in (
            ("near_only", 2.5, 4.5, True, False), ("far_only", 5.5, 7.5, False, True),
            ("double_talk", 8.5, 10.5, True, True), ("noise_only", 11.2, 11.8, False, False)):
            for source, active in ((near, near_active), (far, far_active)):
                v = source[round(a * rate):round(b * rate)]
                if (rms(v) > 1e-6) != active:
                    raise ValueError(f"{case['id']} invalid {label} activity")
        result["explicit_duplex_segments"] = True
    if kind == "speech":
        source = far if case["conditions"]["direction"] == "receive" else near
        if rms(source) <= 1e-5 or len(source) / rate < 9.01:
            raise ValueError("MOS needs real non-repeated speech >=9.01 seconds")
        if case["conditions"]["direction"] != "receive" and np.count_nonzero(far):
            raise ValueError("single-direction MOS must not contain far-end echo")
    if kind in {"response", "oob"}:
        source = far if case["conditions"]["direction"] == "receive" else near
        band = case["conditions"]["mode"].split("-")[0]
        upper_hz = MODE_UPPER_HZ[band]
        required_centers = [center for center in FREQUENCY_RESPONSE_CENTERS_HZ if center <= upper_hz]
        highest_upper_hz = required_centers[-1] * 2 ** (1 / 6)
        if highest_upper_hz >= rate / 2:
            raise ValueError(
                f"highest response band {required_centers[-1]} Hz is not fully observable "
                f"below Nyquist {rate / 2:g} Hz"
            )
        frequencies, power = _hann_power(source, rate)
        total_power = float(np.sum(power))
        powers, fractions = {}, {}
        for center in required_centers:
            mask = (frequencies >= center / 2 ** (1 / 6)) & (frequencies <= center * 2 ** (1 / 6))
            band_power = float(np.sum(power[mask]))
            fraction = band_power / total_power
            if fraction < MIN_EXCITATION_BAND_POWER_RATIO:
                raise ValueError(f"missing excitation band {center} Hz")
            powers[str(center)] = 10 * math.log10(band_power)
            fractions[str(center)] = fraction
        result["excited_bands_dbfs"] = powers
        result["excited_band_power_fractions"] = fractions
        if kind == "oob":
            diagnostic_hz = float(OOB_DIAGNOSTIC_HZ[band])
            if diagnostic_hz >= rate / 2:
                raise ValueError(
                    f"OOB diagnostic {diagnostic_hz:g} Hz is not observable below Nyquist {rate / 2:g} Hz"
                )
            diagnostic_mask = np.abs(frequencies - diagnostic_hz) <= 2.0
            diagnostic_fraction = float(np.sum(power[diagnostic_mask])) / total_power
            if diagnostic_fraction < MIN_OOB_DIAGNOSTIC_POWER_RATIO:
                raise ValueError(
                    f"missing OOB diagnostic tone {diagnostic_hz:g} Hz "
                    f"(power fraction {diagnostic_fraction:.6g})"
                )
            result["oob_diagnostic_hz"] = diagnostic_hz
            result["oob_diagnostic_power_fraction"] = diagnostic_fraction
    if sidecar is not None:
        result["measurement_targets"] = _validate_sidecar_measurements(
            signals,
            sidecar,
            plan=plan,
        )
    return result


def encode(samples: np.ndarray, rate: int) -> tuple[bytes, np.ndarray]:
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="PCM_16")
    blob = stream.getvalue()
    decoded, actual_rate = sf.read(BytesIO(blob))
    if actual_rate != rate:
        raise ValueError("PCM rate identity mismatch")
    return blob, decoded


def verify_mixture(signals: dict, metadata: dict) -> dict:
    """Recompute each microphone channel from decoded PCM and per-channel metadata."""
    if metadata["direction"] == "receive":
        return {"applicable": False}
    microphone = np.asarray(signals["microphone"], dtype=np.float64)
    if microphone.ndim != 2 or microphone.shape[1] != 2:
        raise ValueError("two decoded microphone channels are required")
    frames = len(microphone)
    paths = metadata.get("echo_paths", [])
    if not isinstance(paths, list):
        raise ValueError("echo_paths must be a list")
    delays = metadata.get("near_arrival_delays_samples")
    if not isinstance(delays, list) or len(delays) < 2:
        raise ValueError("per-channel reference arrival delays are required")
    noise = np.asarray(signals["noise"], dtype=np.float64)
    if noise.ndim != 1 or len(noise) != frames:
        raise ValueError("delivered noise must be mono and match the microphone frame count")
    if paths and len(paths) < 2:
        raise ValueError("two echo FIR paths are required for dual-microphone verification")
    reference = np.asarray(signals["reference"], dtype=np.float64)
    far_end = np.asarray(signals["far_end"], dtype=np.float64)
    if reference.ndim != 1 or far_end.ndim != 1:
        raise ValueError("delivered reference and far_end must be mono")
    if len(reference) != frames or len(far_end) != frames:
        raise ValueError("reference and far_end must match the microphone frame count")
    residuals = []
    for channel in range(2):
        delay = int(delays[channel])
        if delay < 0 or delay != delays[channel] or delay >= frames:
            raise ValueError("reference arrival delays must be in-range non-negative samples")
        delayed_reference = np.zeros(frames, dtype=np.float64)
        delayed_reference[delay:] = reference[:frames - delay]
        echo = _fir_component(
            far_end,
            paths[channel] if paths else None,
        )
        channel_noise = _reconstruct_noise_channel(noise, metadata, channel)
        expected = delayed_reference + echo + channel_noise
        residual = float(np.max(np.abs(microphone[:, channel] - expected)))
        residuals.append(residual)
        if residual > 1e-4:
            raise ValueError(
                "delivered PCM microphone does not match reference + echo + noise "
                f"(channel {channel}, residual {residual:.6g})"
            )
    return {
        "applicable": True,
        "pcm_component_residual_max_abs": max(residuals),
        "pcm_component_residuals_max_abs": residuals,
        "tolerance": 1e-4,
        "method": "independent per-channel reference delay, FIR, and noise reconstruction from decoded delivered PCM",
    }


def preflight(output: Path) -> None:
    if output.exists() or output.with_suffix(".quality.json").exists():
        raise ValueError("refusing to overwrite existing artifacts")
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    # Dataset ZIP + import + full VCP run reserve; no automatic cleanup.
    if shutil.disk_usage(parent).free < 4 * 1024 ** 3:
        raise OSError(28, "insufficient free space; stopped without deleting files")


def build(output: Path, *, seed: int = 20261104, do_import: bool = True) -> dict:
    preflight(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank = SourceBank()
    source_code_hashes = {"generator_sha256": file_sha(Path(__file__)),
                          "signal_design_sha256": file_sha(Path(__file__).with_name("acceptance_signal_design.py"))}
    manifest, plans = plan_cases()
    analyses = {}
    blobs = {}
    raw_total = 0
    for case in manifest["cases"]:
        conditions = case["conditions"]
        plan = plans[case["id"]]
        kind, rate = plan["kind"], plan["rate"]
        near, near_info = bank.speech(conditions["mode"], "reference", rate)
        far, far_info = bank.speech(conditions["mode"], "far_end", rate)
        noise, noise_info = bank.noise(conditions["mode"], conditions["scene"], rate, seed)
        # Match waveform identity across nominal, maximum, -5, +5 variants.
        case_seed = stable_seed(f"{conditions['mode']}:{conditions['scene']}:{kind}", seed)
        signals, details = render_signals(
            near, far, noise, rate, direction=conditions["direction"], kind=kind,
            bandwidth=conditions["mode"].split("-")[0], snr_db=SNR_DB[conditions["scene"]],
            ser_db=-5., echo_delay_ms=20. + bank.modes.index(conditions["mode"]) * 4.,
            seed=case_seed, input_level_db=float(conditions["input_level_db"]),
            volume_gain_db=VOLUME_DB[conditions["volume"]])
        validate_case(signals, case, plan)
        decoded = {}
        for role in ROLES:
            blob, decoded[role] = encode(signals[role], rate)
            if len(blob) > MAX_ASSET_BYTES:
                raise ValueError("asset exceeds import bound")
            digest = sha(blob)
            relative = f"audio/{digest}.wav"
            if relative not in blobs:
                blobs[relative] = blob
                raw_total += len(blob)
                if raw_total > MAX_TOTAL_UNCOMPRESSED_BYTES - 1024 * 1024:
                    raise ValueError("unique WAV assets exceed import expansion bound")
            case["signals"][role] = relative
        checks = validate_case(decoded, case, plan, details)
        checks["component_identity"] = verify_mixture(decoded, details)
        if details["global_safety_gain"] != 1.0:
            raise ValueError("source headroom insufficient; per-case scaling would change level variants")
        if "beam_delays_ms" in case and details.get("beam_delays_ms") != case["beam_delays_ms"]:
            raise ValueError("microphone geometry and CLI beam calibration disagree")
        analyses[case["id"]] = {**plan, "conditions": conditions, "item_ids": case["item_ids"],
                                 "sources": {"near": near_info, "far": far_info, "noise": noise_info},
                                 "signal_design": details, "quality": checks}
    verify_levels(manifest, blobs)
    manifest_blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def write_member(name: str, blob: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, blob)
        write_member("manifest.json", manifest_blob)
        for relative, blob in sorted(blobs.items()):
            write_member(relative, blob)
    if output.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP exceeds import bound; artifact retained")
    if any(file_sha(Path(__file__) if key == "generator_sha256" else Path(__file__).with_name("acceptance_signal_design.py")) != digest
           for key, digest in source_code_hashes.items()):
        raise ValueError("generator code changed during execution; artifacts retained, not imported")
    imported = import_dataset(output.read_bytes()) if do_import else None
    report = {"schema_version": 1, "dataset_name": NAME, "seed": seed,
              "created_at": datetime.now(timezone.utc).isoformat(), "standard_compliance": False,
              "scope": "measurement-aware engineering regression; not certification",
              "baseline_dataset_id": BASELINE_ID, "output": str(output),
              "zip_sha256": file_sha(output), "zip_bytes": output.stat().st_size,
              "manifest_sha256": sha(manifest_blob), **source_code_hashes,
              "case_count": len(manifest["cases"]), "item_count": 336, "duration_s": DURATION_S,
              "unique_assets": len(blobs), "uncompressed_bytes": raw_total + len(manifest_blob),
              "profiles": dict(Counter(plan["kind"] for plan in plans.values())),
              "speaker_ids": sorted({r["speaker"] for r in bank.used.values() if "speaker" in r}),
              "sources": list(bank.used.values()), "cases": analyses, "import_summary": imported,
              "limitations": ["controlled hybrid noise recipes do not prove actual vehicle speed recordings",
                              "digital gains and synthetic two-microphone/echo paths are not calibrated acoustics",
                              "no codec or wireless transport simulation", "DNSMOS/DRP/Model A remain engineering proxies",
                              "receive idle silence may produce a bound/unmeasurable result; no fake floor is inserted"]}
    report_path = output.with_suffix(".quality.json")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {key: value for key, value in report.items() if key not in {"cases", "sources"}} | {"quality_report": str(report_path)}


def verify_levels(manifest: dict, blobs: dict) -> None:
    groups = defaultdict(dict)
    for case in manifest["cases"]:
        c = case["conditions"]
        if c["direction"] == "receive" and c["volume"] == "nominal" and c["window_ms"] is None:
            if any(row["family"] in {"rlr", "rlr_linearity"} for row in contract_items() if row["id"] in case["item_ids"]):
                groups[c["mode"]][c["input_level_db"]] = case
    for mode, levels in groups.items():
        if mode.startswith("SWB"):
            continue
        if set(levels) != {-5, 0, 5}:
            raise ValueError(f"incomplete matched level triplet: {mode}")
        baseline, rate = sf.read(BytesIO(blobs[levels[0]["signals"]["far_end"]]))
        for level in (-5, 5):
            y, actual_rate = sf.read(BytesIO(blobs[levels[level]["signals"]["far_end"]]))
            expected = baseline * 10 ** (level / 20.)
            if actual_rate != rate or np.max(np.abs(y - expected)) > 8e-5:
                raise ValueError(f"mismatched level waveform: {mode}/{level}")
            if abs(20 * math.log10(rms(y) / rms(baseline)) - level) > .02:
                raise ValueError(f"incorrect level gain: {mode}/{level}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--no-import", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or PROJECT / "build" / f"{NAME}-{stamp}.zip").resolve()
    report = build(output, seed=args.seed, do_import=not args.no_import)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
