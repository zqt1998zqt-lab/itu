"""Non-vacuous checks for the offline evaluation contract."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

spec = importlib.util.spec_from_file_location("eval_dataset", Path(__file__).resolve().parents[1] / "cpp/tools/run_audio_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from eval_profiles import expanded_matrix, playback_check_matrix, NOISES, SNR_DB, VEHICLE_NOISES


def test_matrix_has_independent_pairs_and_controls():
    cases = evaluation.matrix(["a", "b", "c", "d", "e", "f"])
    assert len(cases) == 129 and len({case["id"] for case in cases}) == 129
    assert sum(len(evaluation.modes(case)) for case in cases) == 561
    assert all(case["near_speaker"] != case["far_speaker"] for case in cases)
    assert {case["snr_db"] for case in cases if case["family"] == "ns"} == {-5, 5, 15}
    assert {case["echo_delay_ms"] for case in cases if case["family"] == "aec"} == {20, 80, 180}
    assert {case["ser_db"] for case in cases if case["family"] == "aec"} == {-10, 0, 10}
    assert {case["rt60_s"] for case in cases if case["family"] == "aec"} == {.15, .45}


@pytest.mark.parametrize("microphones", [2, 4])
def test_multichannel_controls_explicitly_select_channel_zero(microphones, tmp_path):
    scene = dict(family="bf", stress="none", mics=microphones,
                 beam_delays_samples=list(range(microphones)))
    for mode in evaluation.modes(scene):
        command = evaluation.invocation(scene, mode, Path("itu_audio_cli"), tmp_path)
        if mode in ("no_bf", "bypass"):
            assert "--no-bf" in command and "--beam-delays" not in command
        else:
            assert "--no-bf" not in command
            delays = command[command.index("--beam-delays") + 1].split(",")
            assert len(delays) == microphones
    mono = evaluation.invocation(dict(family="ns", stress="none", mics=1),
                                 "full", Path("itu_audio_cli"), tmp_path)
    assert "--no-bf" not in mono and "--beam-delays" not in mono


def test_sisdr_gain_invariant_and_detects_wrong_signal():
    rng = np.random.default_rng(77)
    clean = rng.normal(size=32000) * .04
    mixture = clean + rng.normal(size=len(clean)) * .01
    first = evaluation.quality(clean, mixture)
    scaled = evaluation.quality(clean, mixture * .13)
    wrong = evaluation.quality(clean, rng.normal(size=len(clean)) * .04)
    assert abs(first["sisdr_db"] - scaled["sisdr_db"]) < 1e-10
    assert wrong["sisdr_db"] < first["sisdr_db"] - 20


def test_fixed_hpf_diagnostic_and_delay_negative_control():
    t = np.arange(48000) / 16000
    clean = .05 * np.sin(2 * np.pi * 110 * t) + .05 * np.sin(2 * np.pi * 900 * t)
    sos, _, _ = evaluation.hpf_coefficients()
    filtered = signal.sosfilt(sos, clean)
    raw = evaluation.quality(clean, filtered)
    matched = evaluation.quality(filtered, filtered)
    misaligned = evaluation.quality(filtered, evaluation.shift(filtered, 96))
    assert matched["sisdr_censored"] and matched["sisdr_db"] == 120
    assert raw["sisdr_db"] < 20 and misaligned["sisdr_db"] < 20


def test_schedule_and_quantized_snr(tmp_path):
    rng = np.random.default_rng(19)
    x = evaluation.schedule(rng.normal(size=32000), 4, [(1, 3)])
    active = evaluation.mask_windows(4, [(1, 3)])
    clean = evaluation.level(x, -26, active)
    noise = evaluation.level(rng.normal(size=len(x)), -31, active)
    assert np.all(clean[~active] == 0)
    evaluation.write(tmp_path / "clean.wav", clean)
    evaluation.write(tmp_path / "noise.wav", noise)
    snr = evaluation.db(evaluation.power(evaluation.read(tmp_path / "clean.wav")[active]) / evaluation.power(evaluation.read(tmp_path / "noise.wav")[active]))
    assert abs(snr - 5) < .01


def test_expanded_inventory_is_balanced_and_fully_crossed():
    people = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    cases = expanded_matrix(people)
    ns = [c for c in cases if c["family"] == "ns"]
    assert len(ns) == 40 * len(NOISES) * len(SNR_DB)
    assert {(c["near_speaker"], c["noise"], c["snr_db"]) for c in ns} == {
        (person, noise, snr) for person in people for noise in NOISES for snr in SNR_DB}
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["near_speaker"] != c["far_speaker"] for c in cases)
    assert all(b-a == 5 for a,b in zip(SNR_DB, SNR_DB[1:]))
    for person in people:
        for noise in NOISES:
            assert len({c["seed"] for c in ns if c["near_speaker"] == person and c["noise"] == noise}) == 1
    assert len(VEHICLE_NOISES) == 6 and "car_radio" not in NOISES
    assert "car_passenger" not in NOISES
    assert all(c["noise"] != "car_passenger" for c in cases)
    assert len(cases) == 12360
    radio = [c for c in cases if c["playback_source"] == "radio"]
    assert len(radio) == 440
    assert all(c["family"] in ("aec", "combined") for c in radio)
    assert all(c["noise"] in ("none", "tcar") for c in radio)
    check = playback_check_matrix(people)
    assert len(check) == 480 and {c["near_speaker"] for c in check} == set(people)
    assert len([c for c in check if c["family"] == "ns"]) == 40


def test_generated_noise_and_vehicle_sweeps_are_reproducible():
    rng = np.random.default_rng(7)
    speakers = {str(i): rng.normal(size=64000) for i in range(8)}
    noises = {"tcar": rng.normal(size=80000), "hvac": rng.normal(size=80000)}
    for name in ("pink", "brown", "hum", "burst", "babble", *VEHICLE_NOISES):
        case = dict(noise=name, near_speaker="0", far_speaker="1", profile="expanded-v2")
        first, _, contributors = evaluation.noise_background(case, 32000, speakers, noises, np.random.default_rng(42))
        second, _, _ = evaluation.noise_background(case, 32000, speakers, noises, np.random.default_rng(42))
        assert np.array_equal(first, second) and np.isfinite(first).all() and evaluation.power(first) > 0
        assert not set(contributors) & {"0", "1"}


def test_radio_cannot_be_generated_as_background_noise():
    with pytest.raises(ValueError, match="AEC reference"):
        evaluation.noise_background(dict(noise="car_radio"), 32000, {}, {}, np.random.default_rng(42))


def test_radio_reference_drives_echo_without_entering_noise(tmp_path):
    rng = np.random.default_rng(91)
    speakers = {"near": rng.normal(size=64000), "far": rng.normal(size=64000)}
    case = dict(id="radio-test", family="aec", near_speaker="near", far_speaker="far",
                near_dbfs=-26, noise="none", snr_db=None, ser_db=0, rt60_s=.15,
                echo_delay_ms=80, stress="none", profile="expanded-v3", mics=1,
                seed=17, playback_source="radio")
    scene = evaluation.make_case(case, speakers, {}, tmp_path / "radio", {})
    folder = tmp_path / "radio"
    ref = evaluation.read(folder / "ref.wav")[:, 0]
    assert np.array_equal(ref, evaluation.read(folder / "playback.wav")[:, 0])
    assert scene["playback_reference_available"] and scene["playback_source"] == "radio"
    assert scene["noise_origin"] == "none" and scene["measured_input_snr_db"] is None
    assert np.count_nonzero(evaluation.read(folder / "noise-component.wav")) == 0
    h = np.load(folder / "echo-rir.npy")[:, 0]
    echo = evaluation.shift(signal.fftconvolve(ref, h)[:len(ref)], 1280)
    actual = evaluation.read(folder / "echo-component.wav")[:, 0]
    assert evaluation.quality(echo, actual)["sisdr_db"] > 50
    assert evaluation.quality(np.roll(echo, 160), actual)["sisdr_db"] < 5
    before = evaluation.read(folder / "before.wav")[:, 0]
    clean = evaluation.read(folder / "clean.wav")[:, 0]
    assert np.max(np.abs(before - clean - actual)) <= 2 / 32768


def test_radio_playback_cannot_run_in_ns_only_scene(tmp_path):
    with pytest.raises(ValueError, match="AEC-capable"):
        evaluation.make_case(dict(seed=1, family="ns", playback_source="radio"), {}, {}, tmp_path, {})
