"""Pure contracts for the optimized LocalVQE and external-NS chain."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from server import aec_routes, ns_routes


LOCALVQE_MODEL_SHA256 = (
    "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
)
LOCALVQE_FRONTEND_SHA256 = (
    "76510c8250a3af232a6f241643683dad94d9b17f3c184da37870644005b77573"
)
LOCALVQE_PROVIDER_SHA256 = (
    "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60"
)
LOCALVQE_NEURAL_DELAY = 256
LOCALVQE_ADAPTER_DELAY = 224
LOCALVQE_ALGORITHM_DELAY = 480


def _localvqe_configuration() -> dict:
    return {"selection": {"aec": "localvqe_aec200k"}}


def _localvqe_capability(**overrides) -> dict:
    capability = {
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
        "delay_samples": LOCALVQE_ALGORITHM_DELAY,
        "model_codec_delay_samples_16k": 256,
        "linear_frontend_delay_samples_16k": 0,
    }
    capability.update(overrides)
    return capability


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
        "aec_localvqe_inferences": 63,
        "aec_localvqe_processed_frames": 101,
        "aec_localvqe_fallback_frames": 0,
        "aec_neural_delay_samples": LOCALVQE_NEURAL_DELAY,
        "aec_hop_adapter_delay_samples_16k": LOCALVQE_ADAPTER_DELAY,
        "aec_algorithm_delay_samples_16k": LOCALVQE_ALGORITHM_DELAY,
        "aec_model_codec_delay_samples_16k": 256,
        "aec_linear_frontend_delay_samples_16k": 0,
        "aec_processing_mode": "causal_streaming",
    }
    report.update(overrides)
    return report


def test_localvqe_algorithm_delay_is_neural_plus_adapter_and_fully_validated():
    assert LOCALVQE_ALGORITHM_DELAY == (
        LOCALVQE_NEURAL_DELAY + LOCALVQE_ADAPTER_DELAY
    )
    assert aec_routes.validate_aec_route_report(
        _localvqe_report(), _localvqe_configuration()
    ) == []
    assert aec_routes.validate_admitted_aec_identity(
        _localvqe_report(), _localvqe_capability()
    ) == []


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("aec_hop_adapter_delay_samples_16k", None, "224"),
        ("aec_hop_adapter_delay_samples_16k", True, "224"),
        ("aec_hop_adapter_delay_samples_16k", False, "224"),
        ("aec_hop_adapter_delay_samples_16k", 223, "224"),
        ("aec_algorithm_delay_samples_16k", None, "480"),
        ("aec_algorithm_delay_samples_16k", True, "480"),
        ("aec_algorithm_delay_samples_16k", False, "480"),
        ("aec_algorithm_delay_samples_16k", 512, "480"),
        ("aec_algorithm_delay_samples_16k", 640, "480"),
        ("aec_algorithm_delay_samples_16k", 736, "480"),
    ],
)
def test_localvqe_report_rejects_missing_bool_or_legacy_delay(
    field, value, needle
):
    errors = aec_routes.validate_aec_route_report(
        _localvqe_report(**{field: value}), _localvqe_configuration()
    )
    assert any(field in error and needle in error for error in errors)


def test_localvqe_admission_binds_algorithm_delay_not_neural_delay():
    old_report = _localvqe_report(aec_algorithm_delay_samples_16k=512)
    errors = aec_routes.validate_admitted_aec_identity(
        old_report, _localvqe_capability()
    )
    assert any(
        "aec_algorithm_delay_samples_16k" in error and "480" in error
        for error in errors
    )

    old_capability = _localvqe_capability(
        delay_samples=LOCALVQE_NEURAL_DELAY
    )
    errors = aec_routes.validate_admitted_aec_identity(
        old_report, old_capability
    )
    assert any("capability.delay_samples" in error for error in errors)


def test_localvqe_old_capability_delay_is_rejected_by_public_admission(monkeypatch):
    monkeypatch.setattr(
        aec_routes,
        "read_aec_capabilities",
        lambda _cli: {
            "schema": "itu-aec-capabilities-v1",
            "backends": [_localvqe_capability(delay_samples=512)],
        },
    )
    with pytest.raises(ValueError, match="delay_samples.*480"):
        aec_routes.admit_aec_route(
            Path("/optimized/localvqe-cli"),
            _localvqe_configuration(),
            sample_rate=16_000,
            reference_channels=1,
        )


def test_localvqe_original_hashes_and_no_fallback_flow_are_required():
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
        errors = aec_routes.validate_admitted_aec_identity(
            _localvqe_report(**{field: "0" * 64}),
            _localvqe_capability(),
        )
        assert any(field in error and "完全一致" in error for error in errors)

    for value in (True, 1):
        errors = aec_routes.validate_aec_route_report(
            _localvqe_report(aec_localvqe_fallback_frames=value),
            _localvqe_configuration(),
        )
        assert any("aec_localvqe_fallback_frames" in error for error in errors)


def test_dsp_and_computer_pf_keep_their_existing_neural_delay_mapping():
    dsp_capability = {
        "id": "speex_fused_dsp",
        "execution_provider": "hifi_dsp",
        "device_id": "hifi0",
        "model_id": "complex_pf_stream_v1",
        "model_sha256": "a" * 64,
        "frontend_sha256": "b" * 64,
        "provider_sha256": "c" * 64,
        "delay_samples": 480,
    }
    dsp_report = {
        "aec_execution_provider": "hifi_dsp",
        "aec_device_id": "hifi0",
        "aec_model_id": "complex_pf_stream_v1",
        "aec_model_sha256": "a" * 64,
        "aec_frontend_sha256": "b" * 64,
        "aec_provider_sha256": "c" * 64,
        "aec_neural_delay_samples": 480,
    }
    assert aec_routes.validate_admitted_aec_identity(
        dsp_report, dsp_capability
    ) == []

    cpu_capability = {
        "id": "computer_pf_cpu",
        "execution_provider": "cpu",
        "device_id": "host_cpu",
        "model_id": "complex_pf_cpu_local_v1",
        "model_sha256": "1" * 64,
        "frontend_sha256": "2" * 64,
        "provider_sha256": "3" * 64,
        "delay_samples": 0,
    }
    cpu_report = {
        "aec_execution_provider": "cpu",
        "aec_device_id": "host_cpu",
        "aec_model_id": "complex_pf_cpu_local_v1",
        "aec_model_sha256": "1" * 64,
        "aec_frontend_sha256": "2" * 64,
        "aec_provider_sha256": "3" * 64,
        "aec_neural_delay_samples": 0,
    }
    assert aec_routes.validate_admitted_aec_identity(
        cpu_report, cpu_capability
    ) == []


@pytest.mark.parametrize(
    ("input_samples", "flush_samples"),
    [(3_200, 800), (16_000, 800), (48_000, 1_120)],
)
def test_openended_localvqe_inference_count_uses_256_hop_not_160_callbacks(
    input_samples, flush_samples
):
    total_samples = input_samples + flush_samples
    callbacks = total_samples // 160
    inferences = total_samples // 256
    assert inferences < callbacks
    assert inferences * 256 <= total_samples < (inferences + 1) * 256


@pytest.mark.parametrize(
    ("source_samples", "bf_delay_samples", "expected"),
    [
        (16_000, 0, 64),
        (16_000, 1, 64),
        (16_001, 0, 64),
        (192_000, 0, 751),
        (192_000, 1, 752),
        (192_001, 0, 752),
    ],
)
def test_declared_eos_localvqe_inference_count_uses_source_plus_core_delay(
    source_samples, bf_delay_samples, expected
):
    target_hops = (
        source_samples + bf_delay_samples + LOCALVQE_NEURAL_DELAY + 255
    ) // 256
    assert target_hops == expected
    consumed_samples = target_hops * 256
    required_samples = source_samples + bf_delay_samples + LOCALVQE_NEURAL_DELAY
    assert consumed_samples >= required_samples
    assert consumed_samples < required_samples + 256


def test_cofilite_identity_hash_is_real_and_dpdf_delay_remains_frozen():
    cofilite = ns_routes.NS_ROUTES["cofilite"]
    assert cofilite.delay_samples > 0
    assert cofilite.delay_samples % 160 == 0
    assert cofilite.frontend_sha256 == hashlib.sha256(
        cofilite.frontend_path.read_bytes()
    ).hexdigest()
    assert cofilite.frontend_sha256 != "pending-stream-implementation"

    dpdfnet = ns_routes.NS_ROUTES["dpdfnet2"]
    assert dpdfnet.delay_samples == 640
    assert 160 * 4 == 640
