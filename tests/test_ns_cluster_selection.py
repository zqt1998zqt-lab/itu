"""Contract tests for routing LocalVQE and pinned external NS clusters."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from server import aec_routes, alignment_runner, ns_routes, synthesize
from server.acoustic import acceptance_batch


@pytest.mark.parametrize(
    "cluster_id",
    ["localvqe_aec200k", "localvqe_cofilite", "localvqe_dpdfnet2", "localvqe_hybrid_v2_pitch"],
)
def test_localvqe_default_eq_reaches_actual_execution_flags(cluster_id):
    preset, flags, configuration, digest = alignment_runner.resolve_algorithm_cluster(cluster_id)
    assert preset["selection"]["eq"] == "three_band"
    assert configuration["selection"]["eq"] == "three_band"
    assert flags[flags.index("--eq-preamp-db") + 1] == "-2"
    assert [flags[index + 1] for index, value in enumerate(flags) if value == "--eq-band"] == [
        "low_shelf,180,-2,0.707", "peak,1800,2,1", "high_shelf,4500,-1,0.707",
    ]
    assert digest == hashlib.sha256(synthesize.CHAIN_CATALOG.read_bytes()).hexdigest()


OPTIMIZED_CLI = Path("build/cofilite-opt-20261116/host-v3/cpp/itu_audio_cli")
LEGACY_CLI = Path("build/ns-clusters-20261115/host-root/cpp/itu_audio_cli")
HR_48K_CLI = Path("build/itu-48k-online-20261119-v1/cpp/itu_audio_cli")
HR_48K_NS = "dpdfnet2_48k_hr"
LOCALVQE_EXTERNAL_NS_IDS = (
    "cofilite",
    "dpdfnet2",
    "hybrid_v2_pitch",
)


def _configuration(ns: str, *, aec: str = "localvqe_aec200k") -> dict:
    return {
        "profile": "vcp",
        "selection": {
            "bf": "off",
            "aec": aec,
            "ns": ns,
            "eq": "three_band",
            "agc": "adaptive",
            "drc": "speech",
            "limiter": "off",
        },
    }


def _candidate(
    root: Path,
    backend: str = "dpdfnet2",
    *,
    aec: str = "localvqe_aec200k",
    executable: bool = True,
) -> Path:
    if backend == HR_48K_NS:
        relative = HR_48K_CLI
    else:
        relative = (
            OPTIMIZED_CLI
            if backend == "cofilite" or aec == "localvqe_aec200k"
            else LEGACY_CLI
        )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if executable else 0o644)
    return path


def _default_cli(root: Path) -> Path:
    path = root / "legacy-cli"
    path.write_bytes(b"legacy")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("backend", LOCALVQE_EXTERNAL_NS_IDS)
def test_localvqe_aec_uses_optimized_binary_for_every_external_ns(
    monkeypatch, tmp_path, backend
):
    selected = _candidate(tmp_path, backend)
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    assert ns_routes.resolve_ns_cli(default, _configuration(backend)) == selected


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cofilite", OPTIMIZED_CLI),
        ("dpdfnet2", LEGACY_CLI),
        (HR_48K_NS, HR_48K_CLI),
        ("hybrid_v2_pitch", LEGACY_CLI),
    ],
)
def test_standalone_external_ns_selects_its_own_fixed_binary(
    monkeypatch, tmp_path, backend, expected
):
    aec = "webrtc" if backend == HR_48K_NS else "off"
    configuration = _configuration(backend, aec=aec)
    selected = _candidate(
        tmp_path, backend, aec=configuration["selection"]["aec"]
    )
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    assert ns_routes.resolve_ns_cli(default, configuration) == selected
    assert selected == tmp_path / expected


@pytest.mark.parametrize("sample_rate", [16_000, 48_000])
@pytest.mark.parametrize("aec", ["localvqe_aec200k", "speex", "off"])
def test_48k_hr_rejects_non_aec3_before_capability_lookup(
    monkeypatch, sample_rate, aec
):
    def unexpected_capability_lookup(*args, **kwargs):
        raise AssertionError(
            "invalid dpdfnet2_48k_hr AEC selection must fail before capability lookup"
        )

    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", unexpected_capability_lookup
    )
    configuration = _configuration(HR_48K_NS, aec=aec)

    with pytest.raises(ValueError, match="必须使用 WebRTC AEC3"):
        aec_routes.admit_aec_route(
            Path("/opt/itu/bin/itu_audio_cli"),
            configuration,
            sample_rate=sample_rate,
            reference_channels=1,
        )


def test_48k_hr_parses_webrtc_as_the_admitted_aec3_configuration(monkeypatch):
    def unexpected_capability_lookup(*args, **kwargs):
        raise AssertionError(
            "WebRTC AEC3 must not trigger a neural capability lookup"
        )

    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", unexpected_capability_lookup
    )
    configuration = _configuration(HR_48K_NS, aec="webrtc")

    assert aec_routes.aec_backend_for(configuration) == "webrtc"
    assert aec_routes.admit_aec_route(
        Path("/opt/itu/bin/itu_audio_cli"),
        configuration,
        sample_rate=48_000,
        reference_channels=1,
    ) is None


def test_localvqe_with_native_ns_still_uses_optimized_binary(monkeypatch, tmp_path):
    configuration = _configuration("webrtc")
    selected = _candidate(tmp_path, "webrtc")
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    assert ns_routes.resolve_ns_cli(default, configuration) == selected
    assert selected == tmp_path / OPTIMIZED_CLI


def test_cofilite_standalone_never_reuses_the_legacy_host_root_binary(
    monkeypatch, tmp_path
):
    _candidate(tmp_path, "dpdfnet2", aec="off")
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="外部 NS 引擎"):
        ns_routes.resolve_ns_cli(
            default, _configuration("cofilite", aec="off")
        )


@pytest.mark.parametrize("backend", LOCALVQE_EXTERNAL_NS_IDS)
def test_localvqe_never_falls_back_to_legacy_external_binary(
    monkeypatch, tmp_path, backend
):
    stale_paths = [
        tmp_path / LEGACY_CLI,
        tmp_path / "build/cofilite-stream-20261116-v1/host-v2/cpp/itu_audio_cli",
    ]
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="外部 NS 引擎"):
        ns_routes.resolve_ns_cli(default, _configuration(backend))


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        {},
        {"selection": {}},
        {"selection": {"ns": "webrtc", "aec": "off"}},
        {"selection": {"ns": "not_a_pinned_model", "aec": "speex_fused_cpu"}},
    ],
)
def test_legacy_routes_keep_the_incoming_default(monkeypatch, tmp_path, configuration):
    _candidate(tmp_path, aec="off")
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    assert ns_routes.resolve_ns_cli(default, configuration) == default
    assert ns_routes.resolve_ns_cli(None, configuration) is None


@pytest.mark.parametrize("backend", sorted(ns_routes.EXTERNAL_NS_IDS))
def test_external_routes_fail_closed_when_selected_binary_is_missing(
    monkeypatch, tmp_path, backend
):
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="外部 NS 引擎不存在或不可执行"):
        ns_routes.resolve_ns_cli(default, _configuration(backend))


@pytest.mark.parametrize("backend", sorted(ns_routes.EXTERNAL_NS_IDS))
def test_external_route_fails_closed_when_binary_is_not_executable(
    monkeypatch, tmp_path, backend,
):
    _candidate(tmp_path, backend, executable=False)
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="外部 NS 引擎不存在或不可执行"):
        ns_routes.resolve_ns_cli(default, _configuration(backend))


def test_invalid_ns_selection_is_not_silently_treated_as_legacy(monkeypatch, tmp_path):
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="NS selection.ns 必须是字符串"):
        ns_routes.resolve_ns_cli(default, {"selection": {"ns": 7}})


class _FakeAudio:
    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.rate = 16_000
        self.original_channels = 1
        self.sha256 = "0" * 64
        self.byte_count = int(data.nbytes)


def test_synthesize_passes_one_selected_binary_to_admit_and_execution(
    monkeypatch, tmp_path
):
    configuration = _configuration("cofilite")
    selected = _candidate(
        tmp_path, "cofilite", aec=configuration["selection"]["aec"]
    )
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)
    monkeypatch.setattr(synthesize, "ROOT", tmp_path)
    monkeypatch.setattr(synthesize, "cli_path", lambda: default)
    monkeypatch.setattr(
        synthesize, "_chain_flags", lambda profile, chain: ([], configuration, "catalog")
    )

    microphone = np.ones((1600, 1), dtype=np.float32) * 0.1
    reference = np.ones((1600, 1), dtype=np.float32) * 0.2
    monkeypatch.setattr(
        synthesize,
        "decode",
        lambda role, name, blob, channel=None: _FakeAudio(
            f"{role}.wav", microphone if role == "before" else reference
        ),
    )
    monkeypatch.setattr(synthesize, "_resample", lambda data, source, target: data)
    monkeypatch.setattr(synthesize, "power", lambda data: 1.0)
    monkeypatch.setattr(synthesize, "_levels", lambda data: {})
    monkeypatch.setattr(synthesize, "_band_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(synthesize, "_quality_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(synthesize, "_chart", lambda *args, **kwargs: {})
    monkeypatch.setattr(synthesize, "_joint_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(synthesize, "ns_route_identity", lambda chain: {"backend": "cofilite"})
    monkeypatch.setattr(synthesize, "validate_aec_route_report", lambda *args: [])
    monkeypatch.setattr(synthesize, "validate_admitted_aec_identity", lambda *args: [])
    monkeypatch.setattr(synthesize, "validate_ns_route_report", lambda *args: [])

    admitted = {}
    executed = {}

    def fake_admit(binary, chain, **kwargs):
        admitted["binary"] = binary
        admitted["chain"] = chain
        return {"id": "localvqe_aec200k"}

    def fake_execute(command, **kwargs):
        executed["command"] = list(command)
        prefix = Path(command[command.index("--out") + 1])
        sf.write(
            str(prefix) + ".after.wav",
            np.ones(1600, dtype=np.float32) * 0.1,
            16_000,
            subtype="PCM_16",
        )
        Path(str(prefix) + ".json").write_text(
            json.dumps({"dsp_delay_samples": 0, "aec_enabled": True}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(synthesize, "admit_aec_route", fake_admit)
    monkeypatch.setattr(synthesize, "execute_ns_command", fake_execute)

    result = synthesize.synthesize(
        "mic.wav", b"", "ref.wav", b"", rate=16_000, profile="vcp",
        chain=configuration,
    )

    assert admitted["binary"] == selected
    assert admitted["chain"] == configuration
    assert executed["command"][0] == str(selected)
    assert Path(executed["command"][0]) == admitted["binary"]
    assert result["binary"] == str(selected)


def test_acceptance_snapshot_uses_the_selected_binary(monkeypatch, tmp_path):
    configuration = _configuration("dpdfnet2")
    selected = _candidate(
        tmp_path, "dpdfnet2", aec=configuration["selection"]["aec"]
    )
    default = _default_cli(tmp_path)
    monkeypatch.setattr(ns_routes, "ROOT", tmp_path)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster: ({"profile": "vcp", "id": cluster}, [], configuration, "catalog"),
    )
    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: default)
    monkeypatch.setattr(acceptance_batch, "aec_route_requires_capability", lambda chain: True)
    monkeypatch.setattr(acceptance_batch, "ns_route_identity", lambda chain: {"backend": "dpdfnet2"})
    monkeypatch.setattr(acceptance_batch, "required_conditions", lambda row: row["conditions"])
    monkeypatch.setattr(
        acceptance_batch,
        "normalize_receive_configuration",
        lambda value: {"model": "none"},
    )
    monkeypatch.setattr(acceptance_batch, "receive_configuration_sha256", lambda *args: "digest")
    monkeypatch.setattr(
        acceptance_batch,
        "get_dataset",
        lambda dataset_id: {
            "name": "fixture",
            "manifest_sha256": "m" * 64,
            "receive_simulation": {},
            "manifest": [],
        },
    )
    row = {"id": "item-1", "sheet": "WB", "conditions": {"direction": "transmit"}}
    case = {"id": "case-1", "item_ids": ["item-1"], "conditions": {"direction": "transmit"}}
    monkeypatch.setattr(
        acceptance_batch, "select_batch_scope", lambda *args, **kwargs: ([row], [case])
    )
    monkeypatch.setattr(acceptance_batch, "_LOCK", nullcontext())
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    saved = {}
    monkeypatch.setattr(
        acceptance_batch, "_save", lambda snapshot: saved.setdefault("snapshot", snapshot)
    )
    monkeypatch.setattr(
        acceptance_batch,
        "_EXECUTOR",
        SimpleNamespace(submit=lambda *args, **kwargs: None),
    )

    admitted = {}
    monkeypatch.setattr(
        acceptance_batch,
        "admit_aec_route",
        lambda binary, chain, **kwargs: admitted.setdefault(
            "binary", binary
        ) and {"id": "localvqe_aec200k"},
    )

    snapshot = acceptance_batch.start_batch("fixture", "cluster", test_scope="full")

    assert admitted["binary"] == selected
    assert snapshot["cli_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert saved["snapshot"] is snapshot

