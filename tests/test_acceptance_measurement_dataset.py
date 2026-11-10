from __future__ import annotations

from collections import Counter
from copy import deepcopy
from io import BytesIO
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import uuid

import numpy as np
import pytest
from scipy import signal
import soundfile as sf

from server.acoustic.acceptance_contract import contract_items, required_conditions


PROJECT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT / "scripts/create-acceptance-measurement-dataset.py"
HELPER_PATH = PROJECT / "scripts/acceptance_signal_design.py"
FAMILY_KIND = {
    "g_mos": "speech",
    "n_mos": "speech",
    "s_mos": "speech",
    "listening_quality": "speech",
    "rlr": "response",
    "rlr_linearity": "response",
    "frequency_response": "response",
    "idle_noise_level": "idle",
    "idle_noise_peaks": "idle",
    "cng_level": "cng",
    "cng_spectrum": "cng",
    "out_of_band": "oob",
    "tclw": "duplex",
    "double_talk_classification": "duplex",
    "round_trip_delay": "duplex",
    "slr": "duplex",
    "hats_snr": "duplex",
}


def _load_generator(monkeypatch, helper):
    monkeypatch.setitem(sys.modules, "acceptance_signal_design", helper)
    module_name = f"_acceptance_measurement_dataset_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator(monkeypatch):
    """Load the executable without importing the concurrently built helper."""
    helper = types.ModuleType("acceptance_signal_design")
    helper.render_signals = lambda *args, **kwargs: ({}, {})
    helper.OOB_DIAGNOSTIC_HZ = {"NB": 4_200.0, "WB": 8_500.0, "SWB": 14_500.0}
    return _load_generator(monkeypatch, helper)


@pytest.fixture
def real_generator(monkeypatch):
    helper = importlib.import_module("scripts.acceptance_signal_design")
    assert Path(helper.__file__).resolve() == HELPER_PATH
    module = _load_generator(monkeypatch, helper)
    assert module.render_signals is helper.render_signals
    return module, helper


def _first_case(
    manifest: dict,
    plans: dict,
    *,
    kind: str,
    direction: str | None = None,
    mode_prefix: str | None = None,
) -> dict:
    return next(
        case
        for case in manifest["cases"]
        if plans[case["id"]]["kind"] == kind
        and (direction is None or case["conditions"]["direction"] == direction)
        and (mode_prefix is None or case["conditions"]["mode"].startswith(mode_prefix))
    )


def _blank_signals(generator, case: dict, plan: dict) -> dict[str, np.ndarray]:
    frames = round(generator.DURATION_S * plan["rate"])
    microphone = (
        np.zeros(frames)
        if case["conditions"]["direction"] == "receive"
        else np.zeros((frames, 2))
    )
    return {
        "microphone": microphone,
        "reference": np.zeros(frames),
        "far_end": np.zeros(frames),
        "noise": np.zeros(frames),
    }


def _add_tone(
    samples: np.ndarray,
    rate: int,
    start_s: float,
    end_s: float,
    *,
    amplitude: float,
    frequency_hz: float,
) -> None:
    start = int(round(start_s * rate))
    end = int(round(end_s * rate))
    time = np.arange(end - start, dtype=np.float64) / rate
    samples[start:end] += amplitude * np.sin(
        2.0 * math.pi * frequency_hz * time
    )


def _duplex_signals(generator, case: dict, plan: dict) -> dict[str, np.ndarray]:
    signals = _blank_signals(generator, case, plan)
    rate = plan["rate"]
    _add_tone(
        signals["reference"], rate, 2.0, 5.0,
        amplitude=0.08, frequency_hz=310.0,
    )
    _add_tone(
        signals["reference"], rate, 8.0, 11.0,
        amplitude=0.07, frequency_hz=370.0,
    )
    _add_tone(
        signals["far_end"], rate, 0.5, 2.0,
        amplitude=0.06, frequency_hz=510.0,
    )
    _add_tone(
        signals["far_end"], rate, 5.0, 8.0,
        amplitude=0.06, frequency_hz=530.0,
    )
    _add_tone(
        signals["far_end"], rate, 8.0, 11.0,
        amplitude=0.05, frequency_hz=590.0,
    )
    alternating_noise = np.empty(round(generator.DURATION_S * rate))
    alternating_noise[::2] = 0.002
    alternating_noise[1::2] = -0.002
    signals["noise"][:] = alternating_noise
    return signals


def _wav_blob(samples: np.ndarray, rate: int) -> bytes:
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _fir_delivery(generator):
    rate = 16_000
    frames = 4_096
    time = np.arange(frames, dtype=np.float64) / rate
    reference = (
        0.11 * np.sin(2.0 * math.pi * 317.0 * time)
        + 0.025 * np.sin(2.0 * math.pi * 733.0 * time + 0.2)
    )
    far_end = (
        0.18 * np.sin(2.0 * math.pi * 997.0 * time)
        + 0.04 * np.sin(2.0 * math.pi * 431.0 * time)
    )
    rng = np.random.default_rng(9)
    noise_scale = 0.008
    unit_noise0 = rng.normal(size=frames)
    noise = unit_noise0 * noise_scale
    rho = 0.37
    seed2 = 123_456_789
    independent2 = np.random.default_rng(seed2).normal(size=frames)
    mixed_noise1 = (
        rho * unit_noise0
        + math.sqrt(1.0 - rho * rho) * independent2
    )
    normalizer = generator.rms(mixed_noise1)
    noise1 = mixed_noise1 / normalizer * noise_scale
    tap_sets = ([0, 8, 21, 47], [1, 13, 34, 59])
    raw_gain_sets = ([0.8, 0.35, 0.18, 0.07], [0.72, 0.29, 0.16, 0.09])
    echo_gains = (0.23, 0.17)
    echoes = []
    echo_paths = []
    for channel, (tap_samples, raw_gains, echo_gain) in enumerate(
        zip(tap_sets, raw_gain_sets, echo_gains, strict=True)
    ):
        tap_gains = [
            gain / sum(abs(value) for value in raw_gains)
            for gain in raw_gains
        ]
        echo = np.zeros(frames)
        for offset, gain in zip(tap_samples, tap_gains, strict=True):
            echo[offset:] += far_end[:frames - offset] * gain * echo_gain
        echoes.append(echo)
        echo_paths.append({
            "channel": channel,
            "tap_samples": tap_samples,
            "tap_gains": tap_gains,
            "echo_gain_linear": echo_gain,
        })
    delayed_reference = np.r_[0.0, reference[:-1]]
    signals = {
        "microphone": np.column_stack((
            reference + echoes[0] + noise,
            delayed_reference + echoes[1] + noise1,
        )),
        "reference": reference,
        "far_end": far_end,
        "noise": noise,
    }
    decoded = {}
    for role in generator.ROLES:
        _, decoded[role] = generator.encode(signals[role], rate)
    metadata = {
        "direction": "send",
        "near_arrival_delays_samples": [0, 1],
        "global_safety_gain": 1.0,
        "echo_paths": echo_paths,
        "noise_spatial": {
            "channel_1_recipe": {
                "rho": rho,
                "seed2": seed2,
                "normalizer": normalizer,
                "noise_channel_scale_linear": noise_scale,
            },
        },
    }
    return decoded, metadata, rate, time


def test_plan_cases_preserves_all_336_identities_in_180_typed_cases(generator):
    manifest, plans = generator.plan_cases()
    rows = contract_items()
    rows_by_id = {row["id"]: row for row in rows}

    assert len(rows) == 336
    assert len(rows_by_id) == 336
    assert Counter(row["sheet"] for row in rows) == {
        "NB-wired": 42,
        "NB-wireless": 42,
        "NB-bt": 42,
        "WB-wired": 42,
        "WB-wireless": 42,
        "WB-bt": 42,
        "SWB-wired": 42,
        "SWB-wireless": 42,
    }
    assert len(manifest["cases"]) == 180
    assert Counter(plan["kind"] for plan in plans.values()) == {
        "speech": 48,
        "response": 36,
        "cng": 32,
        "duplex": 26,
        "idle": 20,
        "oob": 18,
    }

    covered: list[str] = []
    item_counts_by_sheet: Counter[str] = Counter()
    condition_kind_keys: set[tuple[tuple[tuple[str, object], ...], str]] = set()
    for case in manifest["cases"]:
        plan = plans[case["id"]]
        item_ids = case["item_ids"]
        covered.extend(item_ids)
        item_counts_by_sheet[case["conditions"]["mode"]] += len(item_ids)
        condition_kind_keys.add(
            (tuple(sorted(case["conditions"].items())), plan["kind"])
        )

        expected_families = sorted(rows_by_id[item_id]["family"] for item_id in item_ids)
        expected_kinds = {FAMILY_KIND[rows_by_id[item_id]["family"]] for item_id in item_ids}
        assert plan["families"] == expected_families
        assert expected_kinds == {plan["kind"]}
        for item_id in item_ids:
            assert required_conditions(rows_by_id[item_id]) == case["conditions"]

    assert len(covered) == 336
    assert len(set(covered)) == 336
    assert set(covered) == set(rows_by_id)
    assert item_counts_by_sheet == {
        "NB-wired": 42,
        "NB-wireless": 42,
        "NB-bt": 42,
        "WB-wired": 42,
        "WB-wireless": 42,
        "WB-bt": 42,
        "SWB-wired": 42,
        "SWB-wireless": 42,
    }
    assert len(condition_kind_keys) == len(manifest["cases"])


@pytest.mark.parametrize(
    ("mode", "direction", "kind", "expected"),
    [
        ("NB-wired", "receive", "response", 16_000),
        ("WB-wired", "send", "speech", 16_000),
        ("WB-wired", "send", "response", 24_000),
        ("WB-wired", "receive", "response", 24_000),
        ("WB-wired", "send", "oob", 24_000),
        ("WB-wired", "receive", "oob", 24_000),
        ("SWB-wired", "send", "speech", 48_000),
        ("SWB-wired", "receive", "response", 48_000),
    ],
)
def test_sample_rate_planning(generator, mode, direction, kind, expected):
    assert generator.sample_rate({"mode": mode, "direction": direction}, kind) == expected


def test_every_planned_case_uses_band_and_measurement_rate(generator):
    manifest, plans = generator.plan_cases()
    for case in manifest["cases"]:
        conditions = case["conditions"]
        plan = plans[case["id"]]
        band = conditions["mode"].split("-")[0]
        if band == "NB":
            expected = 16_000
        elif band == "SWB":
            expected = 48_000
        else:
            needs_24k = plan["kind"] in {"oob", "response"}
            expected = 24_000 if needs_24k else 16_000
        assert plan["rate"] == expected


@pytest.mark.parametrize("kind", ["idle", "cng"])
@pytest.mark.parametrize("contaminated_role", ["reference", "far_end"])
def test_validate_case_rejects_tenfold_noise_speech_in_quiet_window(
    generator, kind, contaminated_role
):
    manifest, plans = generator.plan_cases()
    case = _first_case(manifest, plans, kind=kind, direction="send")
    plan = plans[case["id"]]
    signals = _blank_signals(generator, case, plan)
    rate = plan["rate"]

    noise_rms = 0.002
    alternating_noise = np.empty(round(generator.DURATION_S * rate))
    alternating_noise[::2] = noise_rms
    alternating_noise[1::2] = -noise_rms
    signals["noise"][:] = alternating_noise
    assert generator.validate_case(signals, case, plan)["checked"] is True

    start = int(round(case["analysis_start_s"] * rate))
    window_ms = case["conditions"]["window_ms"]
    end = (
        start + int(round(window_ms * rate / 1000))
        if window_ms is not None
        else len(signals[contaminated_role])
    )
    speech = np.random.default_rng(71).normal(size=end - start)
    speech *= (10.0 * noise_rms) / generator.rms(speech)
    signals[contaminated_role][start:end] = speech
    assert (
        generator.rms(signals[contaminated_role][start:end])
        / generator.rms(signals["noise"][start:end])
        == pytest.approx(10.0)
    )

    with pytest.raises(ValueError, match="analysis window contaminated by speech"):
        generator.validate_case(signals, case, plan)


@pytest.mark.parametrize(
    ("start_s", "end_s", "role", "erase"),
    [
        (2.6, 4.0, "far_end", False),
        (5.6, 7.0, "reference", False),
        (8.5, 10.5, "reference", True),
        (8.5, 10.5, "far_end", True),
        (11.3, 11.7, "reference", False),
        (11.3, 11.7, "far_end", False),
    ],
)
def test_validate_case_enforces_duplex_four_segment_activity(
    generator, start_s, end_s, role, erase
):
    manifest, plans = generator.plan_cases()
    case = _first_case(manifest, plans, kind="duplex", direction="send")
    plan = plans[case["id"]]
    signals = _duplex_signals(generator, case, plan)
    assert generator.validate_case(signals, case, plan)["explicit_duplex_segments"] is True

    rate = plan["rate"]
    start = int(round(start_s * rate))
    end = int(round(end_s * rate))
    if erase:
        signals[role][start:end] = 0.0
    else:
        _add_tone(
            signals[role], rate, start_s, end_s,
            amplitude=0.04, frequency_hz=810.0,
        )

    with pytest.raises(ValueError, match="invalid .* activity"):
        generator.validate_case(signals, case, plan)


def test_swb_response_rejects_upsampled_low_frequency_sine(generator):
    manifest, plans = generator.plan_cases()
    case = _first_case(
        manifest,
        plans,
        kind="response",
        direction="receive",
        mode_prefix="SWB",
    )
    plan = plans[case["id"]]
    signals = _blank_signals(generator, case, plan)

    source_rate = 16_000
    source_frames = round(generator.DURATION_S * source_rate)
    source_time = np.arange(source_frames, dtype=np.float64) / source_rate
    source = 0.05 * np.sin(2.0 * math.pi * 1_000.0 * source_time)
    upsampled = signal.resample_poly(source, 3, 1)
    assert len(upsampled) == len(signals["far_end"])
    signals["far_end"][:] = upsampled

    frequencies, psd = signal.welch(
        signals["far_end"], plan["rate"], nperseg=plan["rate"]
    )
    total = float(np.sum(psd[frequencies >= 125.0]))
    high_band = float(np.sum(psd[frequencies >= 4_000.0]))
    assert high_band < total * 1e-6

    with pytest.raises(ValueError, match="missing excitation band"):
        generator.validate_case(signals, case, plan)


def test_validate_case_rejects_incomplete_highest_response_band(real_generator):
    generator, helper = real_generator
    manifest, plans = generator.plan_cases()
    case = _first_case(
        manifest,
        plans,
        kind="response",
        direction="send",
        mode_prefix="NB",
    )
    plan = plans[case["id"]]
    rate = plan["rate"]
    frames = round(generator.DURATION_S * rate)
    rng = np.random.default_rng(91)
    signals, _ = helper.render_signals(
        rng.normal(size=frames),
        rng.normal(size=frames),
        rng.normal(size=frames),
        rate,
        direction="send",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=52,
    )
    frequencies = np.fft.rfftfreq(frames, d=1.0 / rate)
    highest_index = int(np.argmin(np.abs(frequencies - 4_000.0)))
    spectrum = np.fft.rfft(signals["reference"])
    spectrum[highest_index - 2:highest_index + 3] = 0.0
    signals["reference"] = np.fft.irfft(spectrum, n=frames)
    decoded = {}
    for role in generator.ROLES:
        _, decoded[role] = generator.encode(signals[role], rate)

    with pytest.raises(ValueError, match="missing excitation band 4000"):
        generator.validate_case(decoded, case, plan)


def test_validate_case_rejects_highest_band_crossing_nyquist(generator):
    manifest, plans = generator.plan_cases()
    case = _first_case(
        manifest,
        plans,
        kind="response",
        direction="send",
        mode_prefix="SWB",
    )
    plan = {**plans[case["id"]], "rate": 16_000}
    signals = _blank_signals(generator, case, plan)

    with pytest.raises(ValueError, match="not fully observable below Nyquist"):
        generator.validate_case(signals, case, plan)


@pytest.mark.parametrize(
    ("mode_prefix", "bandwidth", "rate", "diagnostic_hz"),
    [
        ("NB", "NB", 16_000, 4_200.0),
        ("WB", "WB", 24_000, 8_500.0),
        ("SWB", "SWB", 48_000, 14_500.0),
    ],
)
def test_validate_case_requires_post_pcm_oob_diagnostic_tone(
    real_generator, mode_prefix, bandwidth, rate, diagnostic_hz
):
    generator, helper = real_generator
    manifest, plans = generator.plan_cases()
    if mode_prefix == "SWB":
        template = _first_case(
            manifest,
            plans,
            kind="response",
            direction="receive",
            mode_prefix="SWB",
        )
        case = deepcopy(template)
        case["id"] = "probe-swb-oob-send"
        case["conditions"]["direction"] = "send"
        plan = {**plans[template["id"]], "kind": "oob"}
    else:
        case = _first_case(
            manifest,
            plans,
            kind="oob",
            direction="send",
            mode_prefix=mode_prefix,
        )
        plan = plans[case["id"]]
    frames = round(generator.DURATION_S * rate)
    rng = np.random.default_rng(103)
    common = (
        rng.normal(size=frames),
        rng.normal(size=frames),
        rng.normal(size=frames),
        rate,
    )
    signals, _ = helper.render_signals(
        *common,
        direction="send",
        kind="oob",
        snr_db=10.0,
        bandwidth=bandwidth,
        seed=61,
    )
    decoded = {}
    for role in generator.ROLES:
        _, decoded[role] = generator.encode(signals[role], rate)
    validation = generator.validate_case(decoded, case, plan)
    assert validation["oob_diagnostic_hz"] == diagnostic_hz
    assert validation["oob_diagnostic_power_fraction"] >= 0.01

    without_tone, _ = helper.render_signals(
        *common,
        direction="send",
        kind="response",
        snr_db=10.0,
        bandwidth=bandwidth,
        seed=61,
    )
    negative_control = {}
    for role in generator.ROLES:
        _, negative_control[role] = generator.encode(without_tone[role], rate)
    with pytest.raises(ValueError, match="missing OOB diagnostic tone"):
        generator.validate_case(negative_control, case, plan)


def _matched_level_cases(manifest: dict, plans: dict) -> dict[int, dict]:
    matches = {}
    for case in manifest["cases"]:
        conditions = case["conditions"]
        families = plans[case["id"]]["families"]
        if (
            conditions["mode"] == "NB-wired"
            and conditions["direction"] == "receive"
            and conditions["volume"] == "nominal"
            and conditions["window_ms"] is None
            and any(family in {"rlr", "rlr_linearity"} for family in families)
        ):
            matches[conditions["input_level_db"]] = case
    assert set(matches) == {-5, 0, 5}
    return matches


def test_verify_levels_accepts_matched_waveforms_and_rejects_rms_preserving_change(
    generator,
):
    manifest, plans = generator.plan_cases()
    level_cases = _matched_level_cases(manifest, plans)
    baseline = np.random.default_rng(83).normal(size=4_096)
    baseline *= 0.08 / generator.rms(baseline)
    blobs: dict[str, bytes] = {}
    for level, case in level_cases.items():
        relative = f"level-{level:+d}.wav"
        case["signals"]["far_end"] = relative
        blobs[relative] = _wav_blob(baseline * 10.0 ** (level / 20.0), 16_000)

    level_manifest = {"schema_version": 1, "cases": list(level_cases.values())}
    generator.verify_levels(level_manifest, blobs)

    plus_relative = level_cases[5]["signals"]["far_end"]
    decoded_expected, rate = sf.read(BytesIO(blobs[plus_relative]))
    corrupt = -decoded_expected
    corrupt_blob = _wav_blob(corrupt, rate)
    decoded_corrupt, _ = sf.read(BytesIO(corrupt_blob))
    assert generator.rms(decoded_corrupt) == pytest.approx(
        generator.rms(decoded_expected), rel=1e-12
    )
    assert not np.array_equal(decoded_corrupt, decoded_expected)
    blobs[plus_relative] = corrupt_blob

    with pytest.raises(ValueError, match="mismatched level waveform"):
        generator.verify_levels(level_manifest, blobs)


def test_verify_mixture_accepts_quantized_fir_delivery(generator):
    decoded, metadata, _, _ = _fir_delivery(generator)

    result = generator.verify_mixture(decoded, metadata)

    assert result["applicable"] is True
    assert result["tolerance"] == 1e-4
    assert 0.0 < result["pcm_component_residual_max_abs"] <= 1e-4
    assert result["method"] == (
        "independent per-channel reference delay, FIR, and noise reconstruction "
        "from decoded delivered PCM"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "channel0_speech",
        "channel1_speech",
        "channel1_reference_delay",
        "channel1_echo_delay",
        "channel1_noise",
    ],
)
def test_verify_mixture_rejects_microphone_or_fir_mismatch(generator, corruption):
    decoded, metadata, rate, time = _fir_delivery(generator)
    if corruption == "channel0_speech":
        microphone = decoded["microphone"].copy()
        microphone[:, 0] += 0.001 * np.sin(2.0 * math.pi * 333.0 * time)
        _, decoded["microphone"] = generator.encode(microphone, rate)
    elif corruption == "channel1_speech":
        microphone = decoded["microphone"].copy()
        microphone[:, 1] += 0.001 * np.sin(2.0 * math.pi * 457.0 * time)
        _, decoded["microphone"] = generator.encode(microphone, rate)
    elif corruption == "channel1_reference_delay":
        metadata = deepcopy(metadata)
        metadata["near_arrival_delays_samples"][1] += 1
    elif corruption == "channel1_echo_delay":
        metadata = deepcopy(metadata)
        original_path = metadata["echo_paths"][1]
        metadata = {
            **metadata,
            "echo_paths": [
                metadata["echo_paths"][0],
                {
                    **original_path,
                    "tap_samples": [offset + 1 for offset in original_path["tap_samples"]],
                },
            ],
        }
    else:
        microphone = decoded["microphone"].copy()
        microphone[:, 1] += 0.001 * np.sin(2.0 * math.pi * 619.0 * time)
        _, decoded["microphone"] = generator.encode(microphone, rate)

    with pytest.raises(
        ValueError,
        match="delivered PCM microphone does not match reference \\+ echo \\+ noise",
    ):
        generator.verify_mixture(decoded, metadata)


def test_verify_mixture_receive_does_not_require_tx_components(generator):
    result = generator.verify_mixture(
        {"microphone": np.zeros(256, dtype=np.float64)},
        {"direction": "receive"},
    )
    assert result == {"applicable": False}


def _real_duplex_delivery(real_generator):
    generator, helper = real_generator
    manifest, plans = generator.plan_cases()
    case = _first_case(manifest, plans, kind="duplex", direction="send")
    plan = plans[case["id"]]
    rate = plan["rate"]
    frames = round(generator.DURATION_S * rate)
    time = np.arange(frames, dtype=np.float64) / rate
    near = (
        0.08 * np.sin(2.0 * math.pi * 310.0 * time)
        + 0.02 * np.sin(2.0 * math.pi * 733.0 * time)
    )
    far = (
        0.06 * np.sin(2.0 * math.pi * 510.0 * time)
        + 0.015 * np.sin(2.0 * math.pi * 1_170.0 * time)
    )
    noise = np.random.default_rng(29).normal(size=frames)
    signals, metadata = helper.render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="duplex",
        snr_db=10.0,
        ser_db=-5.0,
        echo_delay_ms=20.0,
        seed=41,
        bandwidth="NB",
    )
    decoded = {}
    for role in generator.ROLES:
        _, decoded[role] = generator.encode(signals[role], rate)
    return generator, case, plan, decoded, metadata


def test_real_helper_duplex_passes_validation_and_mixture_verification(real_generator):
    generator, case, plan, decoded, metadata = _real_duplex_delivery(real_generator)

    validation = generator.validate_case(decoded, case, plan, metadata)
    mixture = generator.verify_mixture(decoded, metadata)

    assert validation["explicit_duplex_segments"] is True
    assert validation["measurement_targets"]["snr"]["target_db"] == pytest.approx(10.0)
    assert validation["measurement_targets"]["ser"]["target_db"] == pytest.approx(-5.0)
    assert metadata["global_safety_gain"] == pytest.approx(1.0)
    assert mixture["applicable"] is True
    assert mixture["pcm_component_residual_max_abs"] <= 1e-4


@pytest.mark.parametrize(
    ("field", "metric"),
    [
        ("snr_db_measured", "SNR"),
        ("ser_db_measured", "SER"),
    ],
)
def test_validate_case_rejects_invalid_sidecar_measurement(
    real_generator, field, metric
):
    generator, case, plan, decoded, metadata = _real_duplex_delivery(real_generator)
    invalid = deepcopy(metadata)
    invalid[field] += 0.021

    with pytest.raises(ValueError, match=metric):
        generator.validate_case(decoded, case, plan, invalid)


def test_validate_case_requires_null_target_when_measurement_is_not_applicable(
    real_generator,
):
    generator, helper = real_generator
    manifest, plans = generator.plan_cases()
    case = _first_case(
        manifest,
        plans,
        kind="response",
        direction="receive",
        mode_prefix="WB",
    )
    plan = plans[case["id"]]
    rate = plan["rate"]
    frames = round(generator.DURATION_S * rate)
    rng = np.random.default_rng(113)
    signals, metadata = helper.render_signals(
        rng.normal(size=frames),
        rng.normal(size=frames),
        rng.normal(size=frames),
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="WB",
        seed=71,
    )
    decoded = {}
    for role in generator.ROLES:
        _, decoded[role] = generator.encode(signals[role], rate)
    assert metadata["snr_db_target"] is None
    invalid = deepcopy(metadata)
    invalid["snr_db_target"] = 0.0

    with pytest.raises(ValueError, match="SNR target must be null"):
        generator.validate_case(decoded, case, plan, invalid)


@pytest.mark.parametrize("existing_artifact", ["zip", "quality"])
def test_preflight_refuses_to_overwrite_existing_artifacts(
    generator, tmp_path, existing_artifact
):
    output = tmp_path / "dataset.zip"
    existing = output if existing_artifact == "zip" else output.with_suffix(".quality.json")
    existing.write_bytes(b"preserve-existing-artifact")

    with pytest.raises(ValueError, match="refusing to overwrite existing artifacts"):
        generator.preflight(output)

    assert existing.read_bytes() == b"preserve-existing-artifact"
    if existing_artifact == "quality":
        assert not output.exists()


def test_preflight_stops_without_creating_or_cleaning_on_low_disk(
    generator, tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preserved = workspace / "preserved.bin"
    preserved.write_bytes(b"must-survive")
    output = workspace / "new-build" / "dataset.zip"
    monkeypatch.setattr(
        generator.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=4 * 1024 ** 3 - 1),
    )

    with pytest.raises(OSError, match="insufficient free space") as raised:
        generator.preflight(output)

    assert raised.value.errno == 28
    assert preserved.read_bytes() == b"must-survive"
    assert not output.parent.exists()
    assert not output.exists()
