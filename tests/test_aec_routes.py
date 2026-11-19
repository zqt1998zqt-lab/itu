from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from server import aec_routes
from server.aec_routes import (
    AEC_IDS,
    aec_backend_for,
    admit_aec_route,
    effective_aec_reference_channels,
    read_aec_capabilities,
    validate_admitted_aec_identity,
    validate_aec_route_report,
)


CLI = Path("/opt/itu/bin/itu_audio_cli")
LOCALVQE_MODEL_SHA256 = (
    "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
)
LOCALVQE_FRONTEND_SHA256 = (
    "76510c8250a3af232a6f241643683dad94d9b17f3c184da37870644005b77573"
)
LOCALVQE_PROVIDER_SHA256 = (
    "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60"
)


def _configuration(aec: str) -> dict:
    return {"selection": {"aec": aec}}


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _capabilities(*entries: dict) -> dict:
    return {"schema": "itu-aec-capabilities-v1", "backends": list(entries)}


def _dsp_capability(**overrides) -> dict:
    entry = {
        "id": "speex_fused_dsp",
        "available": True,
        "architecture": "mdf_complex_pf",
        "reason": "",
        "execution_provider": "hifi_dsp",
        "device_id": "hifi-dsp0",
        "model_id": "complex-pf-v1",
        "model_sha256": "a" * 64,
        "frontend_sha256": "b" * 64,
        "provider_sha256": "c" * 64,
        "delay_samples": 480,
    }
    entry.update(overrides)
    return entry


def _dsp_report(**overrides) -> dict:
    report = {
        "aec_enabled": True,
        "aec_backend_requested": "speex_fused_dsp",
        "aec_backend_effective": "speex_mdf_complex_pf_dsp",
        "aec_backend": "speex_mdf_complex_pf_dsp",
        "aec_execution_provider": "hifi_dsp",
        "aec_device_id": "hifi-dsp0",
        "aec_model_id": "complex-pf-v1",
        "aec_model_sha256": "a" * 64,
        "aec_frontend_sha256": "b" * 64,
        "aec_provider_sha256": "c" * 64,
        "aec_dsp_inferences": 19,
        "aec_dsp_processed_frames": 42,
        "aec_dsp_fallback_frames": 0,
        "aec_neural_delay_samples": 480,
    }
    report.update(overrides)
    return report


def _computer_pf_cpu_capability(**overrides) -> dict:
    entry = {
        "id": "computer_pf_cpu",
        "available": True,
        "architecture": "mdf_complex_pf_cpu",
        "reason": "",
        "execution_provider": "cpu",
        "device_id": "host_cpu",
        "model_id": "complex-pf-cpu-local-v1",
        "model_sha256": "1" * 64,
        "frontend_sha256": "2" * 64,
        "provider_sha256": "3" * 64,
        "delay_samples": 0,
    }
    entry.update(overrides)
    return entry


def _computer_pf_cpu_report(**overrides) -> dict:
    report = {
        "aec_enabled": True,
        "aec_backend_requested": "computer_pf_cpu",
        "aec_backend_effective": "speex_mdf_complex_pf_cpu",
        "aec_backend": "speex_mdf_complex_pf_cpu",
        "aec_execution_provider": "cpu",
        "aec_device_id": "host_cpu",
        "aec_model_id": "complex-pf-cpu-local-v1",
        "aec_model_sha256": "1" * 64,
        "aec_frontend_sha256": "2" * 64,
        "aec_provider_sha256": "3" * 64,
        "aec_cpu_inferences": 19,
        "aec_cpu_processed_frames": 42,
        "aec_cpu_fallback_frames": 0,
        "aec_neural_delay_samples": 0,
        "aec_processing_mode": "offline_noncausal",
        "model_training_status": "locally_trained_experimental",
    }
    report.update(overrides)
    return report


def _localvqe_capability(**overrides) -> dict:
    entry = {
        "id": "localvqe_aec200k",
        "available": True,
        "architecture": "localvqe_daf_neural_mask",
        "reason": "",
        "execution_provider": "cpu",
        "device_id": "host_cpu",
        "model_id": "localvqe_aec200k",
        "model_sha256": LOCALVQE_MODEL_SHA256,
        "frontend_sha256": LOCALVQE_FRONTEND_SHA256,
        "provider_sha256": LOCALVQE_PROVIDER_SHA256,
        "delay_samples": 480,
        "model_codec_delay_samples_16k": 256,
        "linear_frontend_delay_samples_16k": 0,
    }
    entry.update(overrides)
    return entry


def _localvqe_report(**overrides) -> dict:
    report = {
        "aec_enabled": True,
        "aec_backend_requested": "localvqe_aec200k",
        "aec_backend_effective": "localvqe_aec200k_cpu",
        "aec_backend": "localvqe_aec200k_cpu",
        "aec_execution_provider": "cpu",
        "aec_device_id": "host_cpu",
        "aec_model_id": "localvqe_aec200k",
        "aec_model_sha256": LOCALVQE_MODEL_SHA256,
        "aec_frontend_sha256": LOCALVQE_FRONTEND_SHA256,
        "aec_provider_sha256": LOCALVQE_PROVIDER_SHA256,
        "aec_localvqe_inferences": 19,
        "aec_localvqe_processed_frames": 42,
        "aec_localvqe_fallback_frames": 0,
        "aec_neural_delay_samples": 256,
        "aec_hop_adapter_delay_samples_16k": 224,
        "aec_algorithm_delay_samples_16k": 480,
        "aec_model_codec_delay_samples_16k": 256,
        "aec_linear_frontend_delay_samples_16k": 0,
        "aec_processing_mode": "causal_streaming",
    }
    report.update(overrides)
    return report


def _patch_cli(monkeypatch, stdout: str, *, stderr: str = "", returncode: int = 0):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(stdout, stderr, returncode)

    monkeypatch.setattr(aec_routes.subprocess, "run", fake_run)
    return calls


def test_aec_ids_and_selection_mapping():
    assert AEC_IDS == {
        "webrtc",
        "speex",
        "speex_fused_cpu",
        "speex_fused_dsp",
        "computer_pf_cpu",
        "localvqe_aec200k",
    }
    assert aec_backend_for(None) is None
    assert aec_backend_for(_configuration("off")) is None
    assert aec_backend_for(_configuration("aec3")) == "webrtc"
    assert aec_backend_for(_configuration("webrtc")) == "webrtc"
    assert aec_backend_for(_configuration("speex")) == "speex"
    assert aec_backend_for(_configuration("speex_fused_cpu")) == "speex_fused_cpu"
    assert aec_backend_for(_configuration("speex_fused_dsp")) == "speex_fused_dsp"
    assert aec_backend_for(_configuration("computer_pf_cpu")) == "computer_pf_cpu"
    assert aec_backend_for(_configuration("localvqe_aec200k")) == "localvqe_aec200k"


def test_localvqe_reference_channel_selection_is_explicit_and_mono():
    selected = {
        "selection": {"aec": "localvqe_aec200k"},
        "extra_flags": ["--localvqe-reference-channel", "0"],
    }
    assert effective_aec_reference_channels(selected, 2) == 1
    assert effective_aec_reference_channels(_configuration("localvqe_aec200k"), 2) == 2
    invalid = {
        "selection": {"aec": "localvqe_aec200k"},
        "extra_flags": ["--localvqe-reference-channel", "2"],
    }
    assert effective_aec_reference_channels(invalid, 2) == 2
    assert effective_aec_reference_channels(_configuration("speex"), 2) == 2


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"selection": None},
        {"selection": {}},
        {"selection": {"aec": None}},
        {"selection": {"aec": "unknown"}},
        {"selection": {"aec": "speex_fused"}},
    ],
)
def test_aec_backend_rejects_malformed_or_unknown_selection(configuration):
    with pytest.raises(ValueError, match="AEC|aec"):
        aec_backend_for(configuration)


@pytest.mark.parametrize(
    ("selected", "report_backend"),
    [
        ("webrtc", "webrtc_aec3"),
        ("aec3", "webrtc_aec3"),
        ("speex", "speex_mdf"),
        ("speex_fused_cpu", "speex_mdf_nlms_fused_cpu"),
    ],
)
def test_cpu_reports_require_exact_backend(selected, report_backend):
    assert validate_aec_route_report(
        {"aec_backend": report_backend}, _configuration(selected)
    ) == []


@pytest.mark.parametrize(
    ("selected", "requested", "effective", "report_backend"),
    [
        ("webrtc", "webrtc", "webrtc_aec3", "webrtc_aec3"),
        ("aec3", "webrtc", "webrtc_aec3", "webrtc_aec3"),
        ("speex", "speex", "speex_mdf", "speex_mdf"),
        (
            "speex_fused_cpu",
            "speex_fused_cpu",
            "speex_mdf_nlms_fused_cpu",
            "speex_mdf_nlms_fused_cpu",
        ),
    ],
)
def test_cpu_optional_route_fields_bind_exact_selected_route(
    selected, requested, effective, report_backend
):
    report = {
        "aec_enabled": True,
        "aec_backend_requested": requested,
        "aec_backend_effective": effective,
        "aec_backend": report_backend,
    }
    assert validate_aec_route_report(report, _configuration(selected)) == []


@pytest.mark.parametrize(
    "override",
    [
        {"aec_enabled": False},
        {"aec_enabled": 0},
        {"aec_backend_requested": "webrtc"},
        {"aec_backend_requested": "aec3"},
        {"aec_backend_effective": "disabled"},
        {"aec_backend_effective": "webrtc_aec3"},
        {"aec_backend": "disabled"},
    ],
)
def test_cpu_report_rejects_disabled_or_cross_route_fields(override):
    report = {
        "aec_enabled": True,
        "aec_backend_requested": "speex",
        "aec_backend_effective": "speex_mdf",
        "aec_backend": "speex_mdf",
        **override,
    }
    assert validate_aec_route_report(report, _configuration("speex"))


def test_historical_cpu_fused_report_name_is_cpu_only():
    cpu = _configuration("speex_fused_cpu")
    assert validate_aec_route_report(
        {"aec_backend": "speex_mdf_nlms_fused"}, cpu
    ) == []
    assert validate_aec_route_report(
        {
            "aec_enabled": True,
            "aec_backend_requested": "speex_fused_cpu",
            "aec_backend_effective": "speex_mdf_nlms_fused_cpu",
            "aec_backend": "speex_mdf_nlms_fused",
        },
        cpu,
    ) == []

    dsp_errors = validate_aec_route_report(
        {"aec_backend": "speex_mdf_nlms_fused"}, _configuration("speex_fused_dsp")
    )
    assert any("aec_backend_requested" in error for error in dsp_errors)
    assert any("aec_backend_effective" in error for error in dsp_errors)
    assert any("aec_backend" in error for error in dsp_errors)


def test_cpu_report_cannot_masquerade_as_dsp():
    errors = validate_aec_route_report(
        _dsp_report(), _configuration("speex_fused_cpu")
    )
    assert errors
    assert "aec_backend" in errors[0]


def test_valid_computer_pf_cpu_report_passes_full_execution_contract():
    assert validate_aec_route_report(
        _computer_pf_cpu_report(), _configuration("computer_pf_cpu")
    ) == []


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("aec_backend_requested", "speex_fused_dsp", "aec_backend_requested"),
        (
            "aec_backend_effective",
            "speex_mdf_complex_pf_dsp",
            "aec_backend_effective",
        ),
        ("aec_backend", "webrtc_aec3", "aec_backend"),
        ("aec_execution_provider", "hifi_dsp", "aec_execution_provider"),
        ("aec_device_id", "hifi-dsp0", "aec_device_id"),
        ("aec_model_id", "../complex-pf-cpu", "aec_model_id"),
        ("aec_model_sha256", "a" * 63, "aec_model_sha256"),
        ("aec_frontend_sha256", "g" * 64, "aec_frontend_sha256"),
        ("aec_provider_sha256", None, "aec_provider_sha256"),
        ("aec_cpu_inferences", True, "aec_cpu_inferences"),
        ("aec_cpu_inferences", 0, "aec_cpu_inferences"),
        ("aec_cpu_processed_frames", True, "aec_cpu_processed_frames"),
        ("aec_cpu_processed_frames", 0, "aec_cpu_processed_frames"),
        ("aec_cpu_fallback_frames", 1, "aec_cpu_fallback_frames"),
        ("aec_cpu_fallback_frames", False, "aec_cpu_fallback_frames"),
        ("aec_neural_delay_samples", 480, "aec_neural_delay_samples"),
        ("aec_processing_mode", "streaming_causal", "aec_processing_mode"),
        ("model_training_status", "official_pretrained", "model_training_status"),
    ],
)
def test_computer_pf_cpu_report_rejects_fake_or_incomplete_execution(
    field, value, needle
):
    errors = validate_aec_route_report(
        _computer_pf_cpu_report(**{field: value}),
        _configuration("computer_pf_cpu"),
    )
    assert any(needle in error for error in errors)


@pytest.mark.parametrize(
    "field",
    [
        "aec_execution_provider",
        "aec_device_id",
        "aec_model_id",
        "aec_model_sha256",
        "aec_frontend_sha256",
        "aec_provider_sha256",
        "aec_cpu_inferences",
        "aec_cpu_processed_frames",
        "aec_cpu_fallback_frames",
        "aec_neural_delay_samples",
        "aec_processing_mode",
        "model_training_status",
    ],
)
def test_computer_pf_cpu_report_requires_every_runtime_proof_field(field):
    report = _computer_pf_cpu_report()
    del report[field]
    errors = validate_aec_route_report(
        report, _configuration("computer_pf_cpu")
    )
    assert any(field in error for error in errors)


def test_computer_pf_cpu_report_cannot_be_a_backend_label_only():
    errors = validate_aec_route_report(
        {"aec_backend": "speex_mdf_complex_pf_cpu"},
        _configuration("computer_pf_cpu"),
    )
    for field in (
        "aec_backend_requested",
        "aec_execution_provider",
        "aec_cpu_inferences",
        "aec_cpu_processed_frames",
        "aec_processing_mode",
        "model_training_status",
    ):
        assert any(field in error for error in errors)


def test_computer_pf_cpu_and_dsp_reports_cannot_cross_routes():
    cpu_as_dsp = validate_aec_route_report(
        _computer_pf_cpu_report(), _configuration("speex_fused_dsp")
    )
    assert any("aec_backend_requested" in error for error in cpu_as_dsp)
    assert any("aec_execution_provider" in error for error in cpu_as_dsp)
    assert any("aec_dsp_inferences" in error for error in cpu_as_dsp)

    dsp_as_cpu = validate_aec_route_report(
        _dsp_report(), _configuration("computer_pf_cpu")
    )
    assert any("aec_backend_requested" in error for error in dsp_as_cpu)
    assert any("aec_execution_provider" in error for error in dsp_as_cpu)
    assert any("aec_cpu_inferences" in error for error in dsp_as_cpu)


def test_valid_dsp_report_passes_full_identity_contract():
    assert validate_aec_route_report(
        _dsp_report(), _configuration("speex_fused_dsp")
    ) == []


def test_dsp_report_rejects_aec_enabled_false():
    errors = validate_aec_route_report(
        _dsp_report(aec_enabled=False), _configuration("speex_fused_dsp")
    )
    assert any("aec_enabled" in error and "true" in error for error in errors)


def test_dsp_report_allows_absent_aec_enabled_for_legacy_report():
    report = _dsp_report()
    del report["aec_enabled"]
    assert validate_aec_route_report(
        report, _configuration("speex_fused_dsp")
    ) == []


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("aec_backend_requested", "speex_fused_cpu", "aec_backend_requested"),
        ("aec_backend_effective", "speex_mdf_nlms_fused_cpu", "aec_backend_effective"),
        ("aec_backend", "webrtc_aec3", "aec_backend"),
        ("aec_execution_provider", "cpu", "aec_execution_provider"),
        ("aec_device_id", " ", "aec_device_id"),
        ("aec_model_id", "", "aec_model_id"),
        ("aec_model_sha256", "a" * 63, "aec_model_sha256"),
        ("aec_frontend_sha256", "g" * 64, "aec_frontend_sha256"),
        ("aec_provider_sha256", None, "aec_provider_sha256"),
        ("aec_dsp_inferences", True, "aec_dsp_inferences"),
        ("aec_dsp_inferences", 0, "aec_dsp_inferences"),
        ("aec_dsp_processed_frames", True, "aec_dsp_processed_frames"),
        ("aec_dsp_processed_frames", 0, "aec_dsp_processed_frames"),
        ("aec_dsp_fallback_frames", 1, "aec_dsp_fallback_frames"),
        ("aec_dsp_fallback_frames", False, "aec_dsp_fallback_frames"),
        ("aec_neural_delay_samples", True, "aec_neural_delay_samples"),
        ("aec_neural_delay_samples", 211, "aec_neural_delay_samples"),
        ("aec_neural_delay_samples", 212, "aec_neural_delay_samples"),
        ("aec_neural_delay_samples", 481, "aec_neural_delay_samples"),
        ("aec_neural_delay_samples", 16001, "aec_neural_delay_samples"),
    ],
)
def test_dsp_report_rejects_incomplete_or_contradictory_identity(
    field, value, needle
):
    errors = validate_aec_route_report(
        _dsp_report(**{field: value}), _configuration("speex_fused_dsp")
    )
    assert any(needle in error for error in errors)


@pytest.mark.parametrize(
    "model_id",
    [
        "../complex-pf",
        "models/complex-pf",
        "/tmp/complex-pf",
        "complex.pf",
        "a" * 64,
    ],
)
def test_dsp_report_rejects_path_or_oversized_model_id(model_id):
    errors = validate_aec_route_report(
        _dsp_report(aec_model_id=model_id),
        _configuration("speex_fused_dsp"),
    )
    assert any("aec_model_id" in error for error in errors)


@pytest.mark.parametrize(
    "field",
    ["aec_dsp_processed_frames", "aec_neural_delay_samples"],
)
def test_dsp_report_rejects_missing_frame_or_delay_fields(field):
    report = _dsp_report()
    del report[field]
    errors = validate_aec_route_report(
        report, _configuration("speex_fused_dsp")
    )
    assert any(field in error for error in errors)


def test_read_capabilities_uses_exact_cli_contract(monkeypatch):
    payload = _capabilities(
        {
            "id": "webrtc",
            "available": True,
            "architecture": "webrtc_aec3",
            "reason": "",
        },
        {
            "id": "speex",
            "available": True,
            "architecture": "speex_mdf",
            "reason": "",
        },
        {
            "id": "speex_fused_cpu",
            "available": True,
            "architecture": "speex_mdf_nlms_fused_cpu",
            "reason": "",
        },
        _dsp_capability(),
        _computer_pf_cpu_capability(),
        _localvqe_capability(),
    )
    calls = _patch_cli(monkeypatch, json.dumps(payload))

    assert read_aec_capabilities(CLI) == payload
    assert {entry["id"] for entry in payload["backends"]} == AEC_IDS
    assert calls == [
        (
            ["/opt/itu/bin/itu_audio_cli", "--aec-capabilities"],
            {"capture_output": True, "text": True, "timeout": 10},
        )
    ]


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "needle"),
    [
        ("", "unknown option --aec-capabilities", 2, "unknown option"),
        ("not-json", "", 0, "JSON"),
        (json.dumps({"schema": "v2", "backends": []}), "", 0, "schema"),
        (
            json.dumps(
                {
                    "schema": "itu-aec-capabilities-v1",
                    "backends": [
                        {
                                "id": "speex_fused_dsp",
                            "architecture": "mdf_complex_pf",
                            "reason": "",
                        }
                    ],
                }
            ),
            "",
            0,
            "available",
        ),
    ],
)
def test_read_capabilities_rejects_old_cli_and_malformed_documents(
    monkeypatch, stdout, stderr, returncode, needle
):
    _patch_cli(monkeypatch, stdout, stderr=stderr, returncode=returncode)
    with pytest.raises(ValueError, match=needle):
        read_aec_capabilities(CLI)


def test_cpu_admission_does_not_query_capabilities(monkeypatch):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("CPU admission must not invoke --aec-capabilities")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    for selected in ("off", "aec3", "webrtc", "speex", "speex_fused_cpu"):
        assert admit_aec_route(CLI, _configuration(selected)) is None


def test_dsp_admission_returns_frozen_capability_identity(monkeypatch):
    entry = _dsp_capability()
    payload = _capabilities(entry)
    calls = _patch_cli(monkeypatch, json.dumps(payload))

    admitted = admit_aec_route(
        CLI,
        _configuration("speex_fused_dsp"),
        sample_rate=16000,
        reference_channels=1,
    )
    assert admitted == entry
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("capture_channels", "beamforming"),
    [(None, None), (1, None), (1, False), (1, True), (2, True)],
)
def test_computer_pf_cpu_admission_returns_frozen_capability_identity(
    monkeypatch, capture_channels, beamforming
):
    entry = _computer_pf_cpu_capability()
    calls = _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    admitted = admit_aec_route(
        CLI,
        _configuration("computer_pf_cpu"),
        sample_rate=16000,
        reference_channels=1,
        capture_channels=capture_channels,
        beamforming=beamforming,
    )
    assert admitted == entry
    assert admitted is not entry
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"sample_rate": 48000}, r"computer_pf_cpu v1.*sample_rate=16000"),
        ({"reference_channels": 2}, r"computer_pf_cpu v1.*reference_channels=1"),
        ({"sample_rate": True}, "sample_rate 必须是整数"),
        ({"reference_channels": False}, "reference_channels 必须是整数"),
        ({"capture_channels": True}, "capture_channels"),
        ({"capture_channels": 0}, "capture_channels"),
        ({"capture_channels": 9}, "capture_channels"),
        ({"capture_channels": "2"}, "capture_channels"),
        ({"beamforming": 1}, "严格布尔"),
        (
            {"capture_channels": 2, "beamforming": None},
            r"computer_pf_cpu 多声道采集.*beamforming=True",
        ),
    ],
)
def test_computer_pf_cpu_rejects_bad_input_envelope_before_cli(
    monkeypatch, kwargs, needle
):
    def unexpected_run(*args, **unused_kwargs):
        raise AssertionError(
            "invalid computer_pf_cpu input must fail before capability lookup"
        )

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match=needle):
        admit_aec_route(
            CLI,
            _configuration("computer_pf_cpu"),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("architecture", "mdf_complex_pf", "architecture"),
        ("execution_provider", "hifi_dsp", "execution_provider"),
        ("device_id", "hifi-dsp0", "device_id"),
        ("model_id", "../complex-pf-cpu", "model_id"),
        ("model_sha256", "a" * 63, "model_sha256"),
        ("frontend_sha256", "g" * 64, "frontend_sha256"),
        ("provider_sha256", None, "provider_sha256"),
        ("delay_samples", True, "delay_samples"),
        ("delay_samples", 480, "delay_samples"),
    ],
)
def test_computer_pf_cpu_admission_rejects_fake_identity(
    monkeypatch, field, value, needle
):
    entry = _computer_pf_cpu_capability(**{field: value})
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(ValueError, match=needle):
        admit_aec_route(CLI, _configuration("computer_pf_cpu"))


def test_computer_pf_cpu_unavailable_is_a_hard_failure(monkeypatch):
    entry = _computer_pf_cpu_capability(
        available=False,
        reason="locally trained Complex_PF model is not registered",
    )
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(
        ValueError,
        match="computer_pf_cpu.*locally trained Complex_PF model is not registered",
    ):
        admit_aec_route(CLI, _configuration("computer_pf_cpu"))


def test_computer_pf_cpu_missing_capability_entry_is_explicit(monkeypatch):
    _patch_cli(monkeypatch, json.dumps(_capabilities(_dsp_capability())))

    with pytest.raises(
        ValueError, match="computer_pf_cpu.*id='computer_pf_cpu'"
    ):
        admit_aec_route(CLI, _configuration("computer_pf_cpu"))


@pytest.mark.parametrize(
    ("sample_rate", "capture_channels", "beamforming"),
    [(16000, 1, False), (16000, 2, True), (24000, 1, False), (48000, 2, True)],
)
def test_localvqe_admission_returns_frozen_cpu_identity(
    monkeypatch, sample_rate, capture_channels, beamforming
):
    entry = _localvqe_capability()
    calls = _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))
    admitted = admit_aec_route(
        CLI,
        _configuration("localvqe_aec200k"),
        sample_rate=sample_rate,
        reference_channels=1,
        capture_channels=capture_channels,
        beamforming=beamforming,
    )
    assert admitted == entry
    assert admitted is not entry
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("architecture", "mdf_complex_pf_cpu", "architecture"),
        ("execution_provider", "hifi_dsp", "execution_provider"),
        ("device_id", "hifi0", "device_id"),
        ("model_id", "../localvqe", "model_id"),
        ("model_sha256", "b" * 63, "model_sha256"),
        ("frontend_sha256", "z" * 64, "frontend_sha256"),
        ("provider_sha256", None, "provider_sha256"),
        ("delay_samples", 256, "delay_samples"),
        ("delay_samples", 512, "delay_samples"),
        ("delay_samples", 640, "delay_samples"),
        ("delay_samples", 736, "delay_samples"),
        ("delay_samples", True, "delay_samples"),
        ("model_codec_delay_samples_16k", 0, "model_codec_delay_samples_16k"),
        ("model_codec_delay_samples_16k", True, "model_codec_delay_samples_16k"),
        ("linear_frontend_delay_samples_16k", 256, "linear_frontend_delay_samples_16k"),
        ("linear_frontend_delay_samples_16k", True, "linear_frontend_delay_samples_16k"),
        ("linear_frontend_delay_samples_16k", False, "linear_frontend_delay_samples_16k"),
    ],
)
def test_localvqe_admission_rejects_fake_identity(
    monkeypatch, field, value, needle
):
    entry = _localvqe_capability(**{field: value})
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))
    with pytest.raises(ValueError, match=needle):
        admit_aec_route(CLI, _configuration("localvqe_aec200k"))


@pytest.mark.parametrize(
    "backend", ["cofilite", "dpdfnet2", "hybrid_v2_pitch"]
)
@pytest.mark.parametrize("sample_rate", [24000, 48000])
def test_localvqe_external_ns_rejects_non_native_rate_before_capability(
    monkeypatch, backend, sample_rate
):
    def unexpected_capability_lookup(*args, **kwargs):
        raise AssertionError(
            "invalid LocalVQE/external-NS rate must fail before capability lookup"
        )

    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", unexpected_capability_lookup
    )
    configuration = {
        "selection": {"aec": "localvqe_aec200k", "ns": backend}
    }
    with pytest.raises(ValueError, match="外部 NS.*16 kHz"):
        admit_aec_route(
            CLI,
            configuration,
            sample_rate=sample_rate,
            reference_channels=1,
        )


@pytest.mark.parametrize(
    "backend", ["cofilite", "dpdfnet2", "hybrid_v2_pitch"]
)
def test_localvqe_external_ns_keeps_native_16k_admission(monkeypatch, backend):
    entry = _localvqe_capability()
    calls = _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))
    configuration = {
        "selection": {"aec": "localvqe_aec200k", "ns": backend}
    }
    assert admit_aec_route(
        CLI,
        configuration,
        sample_rate=16000,
        reference_channels=1,
    ) == entry
    assert len(calls) == 1


@pytest.mark.parametrize("sample_rate", [24000, 48000])
def test_localvqe_native_ns_keeps_24k_and_48k_admission(
    monkeypatch, sample_rate
):
    entry = _localvqe_capability()
    calls = _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))
    configuration = {
        "selection": {"aec": "localvqe_aec200k", "ns": "webrtc"}
    }
    assert admit_aec_route(
        CLI,
        configuration,
        sample_rate=sample_rate,
        reference_channels=1,
    ) == entry
    assert len(calls) == 1


def test_localvqe_rejects_stereo_reference_before_capability_lookup(monkeypatch):
    def unexpected_run(*args, **unused_kwargs):
        raise AssertionError(
            "invalid LocalVQE input must fail before capability lookup"
        )

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="reference_channels=1"):
        admit_aec_route(
            CLI,
            _configuration("localvqe_aec200k"),
            sample_rate=16000,
            reference_channels=2,
            capture_channels=2,
            beamforming=True,
        )


def test_localvqe_report_requires_real_cpu_streaming_identity():
    assert (
        validate_aec_route_report(
            _localvqe_report(), _configuration("localvqe_aec200k")
        )
        == []
    )
    assert validate_admitted_aec_identity(
        _localvqe_report(), _localvqe_capability()
    ) == []

    forged = _localvqe_report(
        aec_backend_effective="speex_mdf",
        aec_localvqe_fallback_frames=1,
        aec_neural_delay_samples=512,
    )
    errors = validate_aec_route_report(
        forged, _configuration("localvqe_aec200k")
    )
    assert any("aec_backend_effective" in error for error in errors)
    assert any("aec_localvqe_fallback_frames" in error for error in errors)
    assert any("aec_neural_delay_samples" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("aec_hop_adapter_delay_samples_16k", "224"),
        ("aec_algorithm_delay_samples_16k", "480"),
    ],
)
def test_localvqe_report_requires_adapter_and_algorithm_delay_fields(
    field, expected
):
    report = _localvqe_report()
    del report[field]
    errors = validate_aec_route_report(
        report, _configuration("localvqe_aec200k")
    )
    assert any(field in error and expected in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("aec_hop_adapter_delay_samples_16k", True, "224"),
        ("aec_hop_adapter_delay_samples_16k", 223, "224"),
        ("aec_algorithm_delay_samples_16k", False, "480"),
        ("aec_algorithm_delay_samples_16k", 512, "480"),
        ("aec_algorithm_delay_samples_16k", 640, "480"),
        ("aec_algorithm_delay_samples_16k", 736, "480"),
        ("aec_model_codec_delay_samples_16k", True, "256"),
        ("aec_model_codec_delay_samples_16k", False, "256"),
        ("aec_linear_frontend_delay_samples_16k", 256, "0"),
        ("aec_linear_frontend_delay_samples_16k", False, "0"),
    ],
)
def test_localvqe_report_rejects_pseudo_or_legacy_delay_identity(
    field, value, needle
):
    errors = validate_aec_route_report(
        _localvqe_report(**{field: value}),
        _configuration("localvqe_aec200k"),
    )
    assert any(field in error and needle in error for error in errors)


def test_localvqe_admitted_identity_uses_algorithm_not_neural_delay():
    assert validate_admitted_aec_identity(
        _localvqe_report(), _localvqe_capability()
    ) == []

    wrong = _localvqe_report(aec_algorithm_delay_samples_16k=512)
    errors = validate_admitted_aec_identity(wrong, _localvqe_capability())
    assert any(
        "aec_algorithm_delay_samples_16k" in error and "480" in error
        for error in errors
    )

    missing = _localvqe_report()
    del missing["aec_algorithm_delay_samples_16k"]
    errors = validate_admitted_aec_identity(missing, _localvqe_capability())
    assert any(
        error.startswith("aec_algorithm_delay_samples_16k 必须存在")
        for error in errors
    )


def test_localvqe_legacy_512_capability_and_report_cannot_claim_the_same_identity():
    capability = _localvqe_capability(delay_samples=512)
    report = _localvqe_report(aec_algorithm_delay_samples_16k=512)
    errors = validate_admitted_aec_identity(report, capability)
    assert any(
        error.startswith("aec_algorithm_delay_samples_16k 必须存在且为整数 480")
        for error in errors
    )
    assert any(
        error.startswith("capability.delay_samples 必须存在且为整数 480")
        for error in errors
    )
    assert validate_aec_route_report(
        report, _configuration("localvqe_aec200k")
    )


def test_localvqe_report_locks_original_hashes_and_zero_fallback_flow():
    report = _localvqe_report()
    assert report["aec_model_sha256"] == LOCALVQE_MODEL_SHA256
    assert report["aec_frontend_sha256"] == LOCALVQE_FRONTEND_SHA256
    assert report["aec_provider_sha256"] == LOCALVQE_PROVIDER_SHA256
    assert report["aec_localvqe_fallback_frames"] == 0

    for field in (
        "aec_model_sha256",
        "aec_frontend_sha256",
        "aec_provider_sha256",
    ):
        errors = validate_aec_route_report(
            _localvqe_report(**{field: "0" * 64}),
            _configuration("localvqe_aec200k"),
        )
        assert not errors
        errors = validate_admitted_aec_identity(
            _localvqe_report(**{field: "0" * 64}),
            _localvqe_capability(),
        )
        assert any(field in error and "完全一致" in error for error in errors)

    for value in (True, 1):
        errors = validate_aec_route_report(
            _localvqe_report(aec_localvqe_fallback_frames=value),
            _configuration("localvqe_aec200k"),
        )
        assert any("aec_localvqe_fallback_frames" in error for error in errors)


@pytest.mark.parametrize(
    ("capture_channels", "beamforming"),
    [(None, None), (1, None), (1, False), (1, True)],
)
def test_dsp_admission_accepts_mono_capture_contract(
    monkeypatch, capture_channels, beamforming
):
    entry = _dsp_capability()
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    admitted = admit_aec_route(
        CLI,
        _configuration("speex_fused_dsp"),
        capture_channels=capture_channels,
        beamforming=beamforming,
    )
    assert admitted == entry


def test_dsp_admission_accepts_multichannel_capture_with_beamforming(monkeypatch):
    entry = _dsp_capability()
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    admitted = admit_aec_route(
        CLI,
        _configuration("speex_fused_dsp"),
        capture_channels=2,
        beamforming=True,
    )
    assert admitted == entry


@pytest.mark.parametrize("beamforming", [None, False])
def test_dsp_multichannel_capture_requires_beamforming(
    monkeypatch, beamforming
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("capture contract must fail before capability lookup")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="多声道采集.*beamforming=True"):
        admit_aec_route(
            CLI,
            _configuration("speex_fused_dsp"),
            capture_channels=2,
            beamforming=beamforming,
        )


@pytest.mark.parametrize(
    "capture_channels",
    [True, False, 0, -1, 9, 2.0, "2"],
)
def test_dsp_rejects_bool_or_invalid_capture_channels_before_cli(
    monkeypatch, capture_channels
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("invalid capture channels must fail before capability lookup")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="capture_channels"):
        admit_aec_route(
            CLI,
            _configuration("speex_fused_dsp"),
            capture_channels=capture_channels,
            beamforming=True,
        )


@pytest.mark.parametrize("beamforming", [0, 1, "true", [], {}])
def test_dsp_rejects_pseudo_boolean_beamforming_before_cli(
    monkeypatch, beamforming
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("non-bool beamforming must fail before capability lookup")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="严格布尔"):
        admit_aec_route(
            CLI,
            _configuration("speex_fused_dsp"),
            capture_channels=1,
            beamforming=beamforming,
        )


@pytest.mark.parametrize(
    ("capture_channels", "beamforming"),
    [(True, 1), (9, "yes"), (2, None)],
)
def test_cpu_admission_ignores_dsp_capture_constraints(
    monkeypatch, capture_channels, beamforming
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("CPU admission must not invoke --aec-capabilities")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    assert (
        admit_aec_route(
            CLI,
            _configuration("speex_fused_cpu"),
            capture_channels=capture_channels,
            beamforming=beamforming,
        )
        is None
    )


def test_admitted_identity_none_is_a_noop():
    assert validate_admitted_aec_identity({}, None) == []


def test_admitted_identity_binds_all_report_and_capability_fields():
    assert validate_admitted_aec_identity(_dsp_report(), _dsp_capability()) == []


def test_admitted_identity_uses_zero_delay_for_computer_pf_cpu():
    assert validate_admitted_aec_identity(
        _computer_pf_cpu_report(),
        _computer_pf_cpu_capability(),
    ) == []


@pytest.mark.parametrize("delay_samples", [True, False, None, 160, 480])
def test_computer_pf_cpu_identity_requires_exact_zero_delay(delay_samples):
    report = _computer_pf_cpu_report(
        aec_neural_delay_samples=delay_samples
    )
    capability = _computer_pf_cpu_capability(delay_samples=delay_samples)
    errors = validate_admitted_aec_identity(report, capability)
    assert any("aec_neural_delay_samples" in error for error in errors)
    assert any("capability.delay_samples" in error for error in errors)


def test_dsp_identity_still_rejects_computer_pf_cpu_zero_delay():
    errors = validate_admitted_aec_identity(
        _dsp_report(aec_neural_delay_samples=0),
        _dsp_capability(delay_samples=0),
    )
    assert any("aec_neural_delay_samples" in error for error in errors)
    assert any("capability.delay_samples" in error for error in errors)


@pytest.mark.parametrize(
    ("report_field", "capability_field"),
    [
        ("aec_execution_provider", "execution_provider"),
        ("aec_device_id", "device_id"),
        ("aec_model_id", "model_id"),
        ("aec_model_sha256", "model_sha256"),
        ("aec_frontend_sha256", "frontend_sha256"),
        ("aec_provider_sha256", "provider_sha256"),
        ("aec_neural_delay_samples", "delay_samples"),
    ],
)
def test_admitted_identity_rejects_changed_capability_field(
    report_field, capability_field
):
    capability = _dsp_capability()
    capability[capability_field] = (
        "d" * 64
        if capability_field.endswith("_sha256")
        else 640
        if capability_field == "delay_samples"
        else "different"
    )
    errors = validate_admitted_aec_identity(_dsp_report(), capability)
    assert any(report_field in error and "完全一致" in error for error in errors)


@pytest.mark.parametrize(
    ("report_field", "capability_field"),
    [
        ("aec_model_sha256", "model_sha256"),
        ("aec_frontend_sha256", "frontend_sha256"),
        ("aec_provider_sha256", "provider_sha256"),
    ],
)
def test_admitted_identity_rejects_missing_report_hash(
    report_field, capability_field
):
    report = _dsp_report()
    del report[report_field]
    errors = validate_admitted_aec_identity(report, _dsp_capability())
    assert any(
        error.startswith(f"{report_field} 必须存在") for error in errors
    )
    assert not any(
        error.startswith(f"capability.{capability_field} 必须存在")
        for error in errors
    )


def test_admitted_identity_rejects_missing_capability_hash():
    capability = _dsp_capability()
    del capability["provider_sha256"]
    errors = validate_admitted_aec_identity(_dsp_report(), capability)
    assert any("capability.provider_sha256" in error for error in errors)


@pytest.mark.parametrize(
    ("report_field", "capability_field"),
    [
        ("aec_neural_delay_samples", "delay_samples"),
    ],
)
def test_admitted_identity_rejects_missing_delay_fields(
    report_field, capability_field
):
    report = _dsp_report()
    del report[report_field]
    errors = validate_admitted_aec_identity(report, _dsp_capability())
    assert any(
        error.startswith(f"{report_field} 必须存在") for error in errors
    )

    capability = _dsp_capability()
    del capability[capability_field]
    errors = validate_admitted_aec_identity(_dsp_report(), capability)
    assert any(
        error.startswith(f"capability.{capability_field} 必须存在")
        for error in errors
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "../complex-pf",
        "models/complex-pf",
        "/tmp/complex-pf",
        "complex.pf",
        "a" * 64,
    ],
)
def test_admitted_identity_rejects_path_model_id_even_when_equal(model_id):
    errors = validate_admitted_aec_identity(
        _dsp_report(aec_model_id=model_id),
        _dsp_capability(model_id=model_id),
    )
    assert any("aec_model_id" in error for error in errors)
    assert any("capability.model_id" in error for error in errors)


def test_admission_returns_deep_copy_of_capability(monkeypatch):
    entry = _dsp_capability(metadata={"revision": 1})
    payload = _capabilities(entry)
    monkeypatch.setattr(
        aec_routes,
        "read_aec_capabilities",
        lambda _cli: payload,
    )

    admitted = admit_aec_route(CLI, _configuration("speex_fused_dsp"))
    assert admitted == entry
    assert admitted is not entry
    admitted["device_id"] = "mutated"
    admitted["metadata"]["revision"] = 2
    assert entry["device_id"] == "hifi-dsp0"
    assert entry["metadata"] == {"revision": 1}


def test_dsp_unavailable_is_a_hard_failure_with_reason(monkeypatch):
    unavailable = _dsp_capability(
        available=False,
        reason="DSP SDK unavailable in this build",
    )
    _patch_cli(monkeypatch, json.dumps(_capabilities(unavailable)))

    with pytest.raises(
        ValueError, match="speex_fused_dsp.*DSP SDK unavailable in this build"
    ):
        admit_aec_route(
            CLI,
            _configuration("speex_fused_dsp"),
            sample_rate=None,
            reference_channels=None,
        )


def test_dsp_available_entry_still_requires_frozen_identity(monkeypatch):
    entry = _dsp_capability()
    del entry["model_sha256"]
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(ValueError, match="model_sha256"):
        admit_aec_route(CLI, _configuration("speex_fused_dsp"))


def test_dsp_available_entry_rejects_other_neural_ns_architecture(monkeypatch):
    entry = _dsp_capability(architecture="other_neural_ns")
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(ValueError, match="architecture.*mdf_complex_pf"):
        admit_aec_route(CLI, _configuration("speex_fused_dsp"))


@pytest.mark.parametrize(
    "delay_samples",
    [True, None, 211, 212, 481, 16001],
)
def test_dsp_available_entry_rejects_invalid_delay(
    monkeypatch, delay_samples
):
    entry = _dsp_capability(delay_samples=delay_samples)
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(ValueError, match="delay_samples"):
        admit_aec_route(CLI, _configuration("speex_fused_dsp"))


@pytest.mark.parametrize(
    "model_id",
    [
        "../complex-pf",
        "models/complex-pf",
        "/tmp/complex-pf",
        "complex.pf",
        "a" * 64,
    ],
)
def test_dsp_available_entry_rejects_path_or_oversized_model_id(
    monkeypatch, model_id
):
    entry = _dsp_capability(model_id=model_id)
    _patch_cli(monkeypatch, json.dumps(_capabilities(entry)))

    with pytest.raises(ValueError, match="model_id"):
        admit_aec_route(CLI, _configuration("speex_fused_dsp"))


@pytest.mark.parametrize(
    ("sample_rate", "reference_channels", "needle"),
    [
        (48000, 1, "sample_rate=16000"),
        (16000, 2, "reference_channels=1"),
    ],
)
def test_dsp_rejects_unsupported_v1_rate_or_reference_before_cli(
    monkeypatch, sample_rate, reference_channels, needle
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("unsupported DSP v1 input must fail before capability lookup")

    monkeypatch.setattr(aec_routes.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match=needle):
        admit_aec_route(
            CLI,
            _configuration("speex_fused_dsp"),
            sample_rate=sample_rate,
            reference_channels=reference_channels,
        )


def test_dsp_missing_capability_entry_is_explicit(monkeypatch):
    cpu = {
        "id": "webrtc",
        "available": True,
        "architecture": "x86_64",
        "reason": "",
    }
    _patch_cli(monkeypatch, json.dumps(_capabilities(cpu)))

    with pytest.raises(
        ValueError, match="speex_fused_dsp.*id='speex_fused_dsp'"
    ):
        admit_aec_route(CLI, _configuration("speex_fused_dsp"))


def test_capability_ids_must_be_canonical_backend_names(monkeypatch):
    short_id = _dsp_capability(id="dsp")
    _patch_cli(monkeypatch, json.dumps(_capabilities(short_id)))

    with pytest.raises(ValueError, match="规范 AEC 后端名.*'dsp'"):
        read_aec_capabilities(CLI)
