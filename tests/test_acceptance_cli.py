"""Exercise native 8 kHz and legacy VCP business recording rates."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.acoustic.acceptance_batch import _run_cli
from server.acoustic import acceptance_batch
from server import alignment_runner
from server.alignment_runner import resolve_cli


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_real_cli_preserves_native_rate_and_declared_latency(tmp_path, rate):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")
    t = np.arange(rate // 2) / rate
    source = 0.03 * np.sin(2 * np.pi * 440 * t)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, source, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "case"
    destination.mkdir()
    outputs, execution = _run_cli({"conditions": {"direction": "round_trip", "mode": "NB-wired" if rate == 8000 else "WB-wired"}}, paths, destination, "vcp_unified")
    for role, output in outputs.items():
        data, output_rate = sf.read(output)
        assert output_rate == rate
        assert len(data) == output_rate // 2
        assert np.isfinite(data).all()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == execution["output_sha256"][role]
    if rate == 8000:
        assert execution["source"] == "native_rate_path"
        assert execution["algorithm_cluster_id"] == "speex_dpdfnet2_8khz"
        assert execution["requested_algorithm_cluster_id"] == "vcp_unified"
        assert execution["rate_path"]["sample_rate_hz"] == 8_000
        assert execution["rate_path"]["aec_backend"] == "speex_mdf_nb8"
        assert execution["rate_path"]["ns_backend"] == "dpdfnet2_8khz"
        assert execution["no_resample"] is True
        assert execution["fallback_frames"] == 0
        assert execution["input_adaptation"] == {}
        assert execution["pipeline_reports"]["rx"]["data"]["aec_applicability"] == (
            "not_applicable_receive_only"
        )
        assert execution["pipeline_reports"]["tx"]["data"]["stages"]["speex_mdf"][
            "backend"
        ] == "speex_mdf_nb8"
        assert execution["echo_reference"]["tx_echo_argument_sha256"] == (
            execution["output_sha256"]["rx"]
        )
        assert execution["echo_reference"]["source"] == (
            "native_nb8_rx_processed"
        )
        assert execution["vcp_rx_equivalent"] is False
        for role in ("rx", "tx"):
            alignment = execution["output_alignment"][role]
            assert alignment["declared_algorithm_latency_samples"] == 320
            assert alignment["adapter_trim_samples"] == 320
            assert alignment["batch_trim_samples"] == 0
            assert alignment["trimmed_leading_samples"] == 0
            assert alignment["remaining_delay_samples"] == 0
            assert (destination / alignment["analysis_filename"]).is_file()
            assert sf.info(destination / alignment["analysis_filename"]).frames == (
                rate // 2
            )
    else:
        assert execution["pipeline_reports"]["rx"]["data"]["rx_processing_frames"] > 0
        if rate == 24_000:
            assert execution["source"] == "alignment_batch_cli"
            assert execution["algorithm_cluster_id"] == "vcp_unified"
            assert "rate_path" not in execution
        alignment = execution["output_alignment"]["tx"]
        assert alignment["trimmed_leading_samples"] > 0
        assert (destination / alignment["raw_filename"]).is_file()
        assert sf.info(destination / alignment["raw_filename"]).frames > sf.info(
            outputs["tx"]
        ).frames


def test_real_cli_native_nb8_receive_only_preserves_8k(tmp_path):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")
    rate = 8_000
    time = np.arange(rate // 2) / rate
    far_end = 0.03 * np.sin(2 * np.pi * 440 * time)
    far_end_path = tmp_path / "far_end.wav"
    sf.write(far_end_path, far_end, rate, subtype="PCM_16")
    destination = tmp_path / "receive-case"
    destination.mkdir()

    outputs, execution = _run_cli(
        {"conditions": {"direction": "receive", "mode": "NB-wired"}},
        {"far_end": far_end_path},
        destination,
        "vcp_unified",
    )

    assert set(outputs) == {"rx"}
    data, output_rate = sf.read(outputs["rx"])
    assert output_rate == 8_000
    assert len(data) == rate // 2
    assert np.isfinite(data).all()
    assert execution["source"] == "native_rate_path"
    assert execution["algorithm_cluster_id"] == "speex_dpdfnet2_8khz"
    assert execution["no_resample"] is True
    assert execution["fallback_frames"] == 0
    assert set(execution["pipeline_reports"]) == {"rx"}
    assert execution["pipeline_reports"]["rx"]["data"]["aec_applicability"] == (
        "not_applicable_receive_only"
    )
    assert execution["echo_reference"] is None
    assert execution["rx_playback_artifact"]["processed_sha256"] == (
        execution["output_sha256"]["rx"]
    )
    assert execution["output_roles"] == ["rx"]
    assert execution["aec_applicability"] == "not_applicable_receive_only"
    assert execution["vcp_rx_equivalent"] is False
    assert execution["output_alignment"]["rx"]["analysis_filename"] == (
        "rx.rx-spk.wav"
    )


def test_real_cli_native_nb8_beamforms_stereo_without_resampling(tmp_path):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")
    rate = 8_000
    time = np.arange(rate // 2) / rate
    near = 0.03 * np.sin(2 * np.pi * 510 * time)
    far = 0.02 * np.sin(2 * np.pi * 230 * time)
    stereo_microphone = np.stack([near + far, near + 0.8 * far], axis=1)
    paths = {}
    for role, values in (
        ("microphone", stereo_microphone),
        ("far_end", far),
        ("noise", np.zeros_like(far)),
        ("reference", near),
    ):
        path = tmp_path / f"{role}.wav"
        sf.write(path, values, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "stereo-case"
    destination.mkdir()

    outputs, execution = _run_cli(
        {
            "conditions": {"direction": "round_trip", "mode": "NB-wired"},
            "beam_delays_ms": [0.125, 0.0],
        },
        paths,
        destination,
        "vcp_unified",
    )

    assert set(outputs) == {"rx", "tx"}
    for output in outputs.values():
        assert sf.info(output).samplerate == 8_000
    adaptation = execution["beam_delay_adaptation"]
    assert adaptation["method"] == "fractional delay-and-sum"
    assert adaptation["source_channels"] == 2
    assert adaptation["output_channels"] == 1
    assert adaptation["delays_samples"] == [1.0, 0.0]
    preprocessing = execution["native_preprocessing"]["beamforming"]
    assert Path(preprocessing["output_path"]).is_file()
    assert sf.info(preprocessing["output_path"]).channels == 1
    assert execution["no_resample"] is True
    assert execution["fallback_frames"] == 0


def test_native_nb8_never_calls_acceptance_resampler(tmp_path, monkeypatch):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")

    def forbidden_resample(*args, **kwargs):
        raise AssertionError("native NB8 must not call resample_poly")

    monkeypatch.setattr(
        acceptance_batch,
        "resample_poly",
        forbidden_resample,
    )
    rate = 8_000
    time = np.arange(rate // 2) / rate
    signal = 0.03 * np.sin(2 * np.pi * 440 * time)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, signal, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "case"
    destination.mkdir()

    outputs, execution = _run_cli(
        {"conditions": {"direction": "round_trip", "mode": "NB-wired"}},
        paths,
        destination,
        "vcp_unified",
    )

    assert set(outputs) == {"rx", "tx"}
    assert execution["no_resample"] is True
    assert execution["resampling"] == "none"


def test_nb_measurement_fixture_at_16k_is_not_labeled_native_nb8(tmp_path):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")
    rate = 16_000
    time = np.arange(rate // 2) / rate
    signal = 0.03 * np.sin(2 * np.pi * 440 * time)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, signal, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "case"
    destination.mkdir()

    outputs, execution = _run_cli(
        {"conditions": {"direction": "round_trip", "mode": "NB-wired"}},
        paths,
        destination,
        "vcp_unified",
    )

    assert set(outputs) == {"rx", "tx"}
    assert execution["source"] == "alignment_batch_cli"
    assert execution["rate_path_policy"] == "nb_measurement_fixture_16khz"
    assert execution["native_nb8_eligible"] is False
    assert execution["business_sample_rate_hz"] == 8_000
    assert execution["engine_sample_rate_hz"] == 16_000


def test_native_nb8_does_not_require_legacy_vcp_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: None)
    rate = 8_000
    time = np.arange(rate // 2) / rate
    signal = 0.03 * np.sin(2 * np.pi * 440 * time)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, signal, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "case"
    destination.mkdir()

    outputs, execution = _run_cli(
        {"conditions": {"direction": "round_trip", "mode": "NB-wired"}},
        paths,
        destination,
        "vcp_unified",
    )

    assert set(outputs) == {"rx", "tx"}
    assert execution["requested_cli_sha256"] is None
    assert execution["effective_cli_sha256"] == (
        execution["native_nb8_identity"]["speex_cli_sha256"]
    )


@pytest.mark.parametrize("rate", [32000, 44100])
def test_cli_does_not_silently_downsample_swb(tmp_path, rate):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(rate // 10), rate)
    with pytest.raises(ValueError, match="不受 VCP 支持"):
        _run_cli({"conditions": {"mode": "SWB-wired", "direction": "receive"}},
                 {"far_end": path}, tmp_path, "vcp_unified")
