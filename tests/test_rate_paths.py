from __future__ import annotations

import pytest

from server.acoustic import acceptance_batch
from server.rate_paths import (
    LEGACY_RATE_PATHS,
    PRIMARY_RATE_PATHS,
    PRIMARY_RATE_HZ,
    public_rate_paths,
    require_primary_path,
    resolve_rate_path,
)


def test_primary_paths_are_exactly_8_16_48_and_use_distinct_clusters() -> None:
    assert tuple(PRIMARY_RATE_PATHS) == PRIMARY_RATE_HZ == (8_000, 16_000, 48_000)
    clusters = [path.cluster_id for path in PRIMARY_RATE_PATHS.values()]
    assert len(set(clusters)) == 3
    assert clusters == [
        "speex_dpdfnet2_8khz",
        "localvqe_cofilite_native16",
        "aec3_serial",
    ]


def test_primary_paths_keep_engine_analysis_and_output_rates_native() -> None:
    for rate, path in PRIMARY_RATE_PATHS.items():
        assert path.rate_policy == "native"
        assert path.engine_sample_rate_hz == rate
        assert path.analysis_sample_rate_hz == rate
        assert path.output_sample_rate_hz == rate


def test_24khz_is_compatibility_only() -> None:
    path = resolve_rate_path(24_000)
    assert path is LEGACY_RATE_PATHS[24_000]
    assert path.status == "compatibility"
    with pytest.raises(ValueError, match="兼容路径"):
        require_primary_path(24_000)


@pytest.mark.parametrize("rate", [True, "16000", 0, 32_000, 44_100])
def test_invalid_rate_types_and_values_are_rejected(rate: object) -> None:
    with pytest.raises(ValueError):
        resolve_rate_path(rate)  # type: ignore[arg-type]


def test_public_rate_paths_include_three_primary_and_one_legacy() -> None:
    rows = public_rate_paths()
    assert [row["sample_rate_hz"] for row in rows] == [8_000, 16_000, 48_000, 24_000]
    assert sum(row["status"] == "maintained" for row in rows) == 3
    assert sum(row["status"] == "compatibility" for row in rows) == 1


def test_native_nb8_requested_and_effective_identity_are_separate() -> None:
    identity = acceptance_batch.native_nb8_identity()
    rate_path = PRIMARY_RATE_PATHS[8_000].as_dict()
    snapshot = {
        "algorithm_cluster_id": "vcp_unified",
        "configuration": {"profile": "vcp_unified"},
        "catalog_sha256": "a" * 64,
        "cli_sha256": "b" * 64,
        "native_nb8_identity": identity,
    }
    execution = {
        "source": "native_rate_path",
        "rate_path": rate_path,
        "algorithm_cluster_id": "speex_dpdfnet2_8khz",
        "configuration": {
            "profile": "native_rate_path",
            "selection": {
                "aec": "speex_mdf_nb8",
                "ns": "dpdfnet2_8khz",
            },
        },
        "requested_algorithm_cluster_id": "vcp_unified",
        "requested_configuration": {"profile": "vcp_unified"},
        "requested_catalog_sha256": "a" * 64,
        "requested_cli_sha256": "b" * 64,
        "native_nb8_identity": identity,
        "effective_cli_sha256": identity["speex_cli_sha256"],
        "aec_route_capability": None,
        "no_resample": True,
        "fallback_frames": 0,
    }

    acceptance_batch.validate_execution_identity(snapshot, execution)
    execution["requested_algorithm_cluster_id"] = "wrong_cluster"
    with pytest.raises(ValueError, match="请求算法簇"):
        acceptance_batch.validate_execution_identity(snapshot, execution)


def test_nb8_case_eligibility_requires_every_signal_to_be_native_8k() -> None:
    dataset = {
        "assets": {
            "mic.wav": {"sample_rate": 8_000},
            "far.wav": {"sample_rate": 8_000},
        }
    }
    case = {
        "conditions": {"mode": "NB-wired"},
        "signals": {
            "microphone": "mic.wav",
            "far_end": "far.wav",
        },
    }
    assert acceptance_batch._case_is_native_nb8(case, dataset) is True
    dataset["assets"]["far.wav"]["sample_rate"] = 16_000
    assert acceptance_batch._case_is_native_nb8(case, dataset) is False


def test_rate_specific_cluster_validation_and_direct_nb8_resolution() -> None:
    from server.alignment_runner import resolve_algorithm_cluster
    from server.rate_paths import (
        NB8_CLUSTER_ID,
        selectable_clusters_for_rate,
        validate_cluster_rate_compatibility,
    )

    assert selectable_clusters_for_rate(8_000) == (NB8_CLUSTER_ID,)
    assert NB8_CLUSTER_ID not in selectable_clusters_for_rate(16_000)
    assert "vcp_aec3_dpdfnet2_48k_hr" not in selectable_clusters_for_rate(16_000)
    assert "localvqe_cofilite_native16" in selectable_clusters_for_rate(16_000)
    assert "vcp_aec3_dpdfnet2_48k_hr" in selectable_clusters_for_rate(48_000)
    assert "aec3_serial" in selectable_clusters_for_rate(48_000)
    for wb_only in (
        "localvqe_cofilite_native16",
        "localvqe_cofilite",
        "localvqe_dpdfnet2",
        "localvqe_hybrid_v2_pitch",
        "speex_fused_dsp",
        "computer_pf_cpu",
    ):
        assert wb_only not in selectable_clusters_for_rate(48_000)
        with pytest.raises(ValueError, match="16 kHz"):
            validate_cluster_rate_compatibility(wb_only, 48_000)

    preset, flags, configuration, digest = resolve_algorithm_cluster(NB8_CLUSTER_ID)
    assert preset["id"] == NB8_CLUSTER_ID
    assert preset["profile"] == "vcp"
    assert flags == []
    assert configuration["profile"] == "native_rate_path"
    assert len(digest) == 64

    validate_cluster_rate_compatibility(NB8_CLUSTER_ID, 8_000, configuration=configuration)
    with pytest.raises(ValueError, match="8 kHz"):
        validate_cluster_rate_compatibility(NB8_CLUSTER_ID, 16_000, configuration=configuration)
    with pytest.raises(ValueError, match="8 kHz"):
        validate_cluster_rate_compatibility(NB8_CLUSTER_ID, 48_000, configuration=configuration)

    validate_cluster_rate_compatibility("vcp_aec3_dpdfnet2_48k_hr", 48_000)
    with pytest.raises(ValueError, match="48 kHz"):
        validate_cluster_rate_compatibility("vcp_aec3_dpdfnet2_48k_hr", 16_000)
    with pytest.raises(ValueError, match="48 kHz"):
        validate_cluster_rate_compatibility("vcp_aec3_dpdfnet2_48k_hr", 8_000)


def test_direct_nb8_cluster_passes_real_cli_and_identity_validation(tmp_path) -> None:
    import numpy as np
    import soundfile as sf
    from server import nb8_route
    from server.alignment_runner import resolve_algorithm_cluster, resolve_cli

    if not nb8_route.DEFAULT_SPEEX_CLI.is_file():
        pytest.skip("native 8 kHz Speex CLI is not built")
    rate = 8_000
    t = np.arange(rate // 2) / rate
    source = 0.03 * np.sin(2 * np.pi * 440 * t)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, source, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "nb8-direct"
    destination.mkdir()
    outputs, execution = acceptance_batch._run_cli(
        {"conditions": {"direction": "round_trip", "mode": "NB-wired"}},
        paths,
        destination,
        "speex_dpdfnet2_8khz",
    )
    assert set(outputs) == {"rx", "tx"}
    assert execution["algorithm_cluster_id"] == "speex_dpdfnet2_8khz"
    assert execution["requested_algorithm_cluster_id"] == "speex_dpdfnet2_8khz"
    preset, _, configuration, catalog_sha = resolve_algorithm_cluster("speex_dpdfnet2_8khz")
    default_cli = resolve_cli()
    snapshot = {
        "algorithm_cluster_id": preset["id"],
        "configuration": configuration,
        "catalog_sha256": catalog_sha,
        "cli_sha256": acceptance_batch._sha(default_cli) if default_cli else acceptance_batch.native_nb8_identity()["speex_cli_sha256"],
        "requested_vcp_cli_sha256": acceptance_batch._sha(default_cli) if default_cli else None,
        "native_nb8_identity": acceptance_batch.native_nb8_identity(),
    }
    acceptance_batch.validate_execution_identity(
        snapshot,
        execution,
        require_algorithm_configuration=True,
    )


def test_start_batch_rejects_mismatched_rate_clusters() -> None:
    from server.acoustic import acoustic_stress_batch, scene_condition_batch
    from server.acoustic.acoustic_stress_profiles import stress_profile
    from server.acoustic.scene_mode_conversation_profiles import profile_for_scene_and_mode

    wb_scene = profile_for_scene_and_mode("sce1", "WB-wired")
    swb_scene = profile_for_scene_and_mode("sce1", "SWB-wired")
    nb_scene = profile_for_scene_and_mode("sce1", "NB-wired")

    with pytest.raises(ValueError, match="8 kHz"):
        scene_condition_batch.start_batch(
            wb_scene.dataset_id, "speex_dpdfnet2_8khz", mode="WB-wired", scene="sce1"
        )
    with pytest.raises(ValueError, match="48 kHz"):
        scene_condition_batch.start_batch(
            wb_scene.dataset_id, "vcp_aec3_dpdfnet2_48k_hr", mode="WB-wired", scene="sce1"
        )
    with pytest.raises(ValueError, match="16 kHz"):
        scene_condition_batch.start_batch(
            swb_scene.dataset_id, "localvqe_cofilite_native16", mode="SWB-wired", scene="sce1"
        )
    with pytest.raises(ValueError, match="48 kHz"):
        scene_condition_batch.start_batch(
            nb_scene.dataset_id, "vcp_aec3_dpdfnet2_48k_hr", mode="NB-wired", scene="sce1"
        )

    swb_echo = stress_profile("echo", "SWB-wired")
    wb_echo = stress_profile("echo", "WB-wired")
    with pytest.raises(ValueError, match="16 kHz"):
        acoustic_stress_batch.start_batch(
            swb_echo.dataset_id, "localvqe_cofilite", mode="SWB-wired", scene="baseline"
        )
    with pytest.raises(ValueError, match="48 kHz"):
        acoustic_stress_batch.start_batch(
            wb_echo.dataset_id, "vcp_aec3_dpdfnet2_48k_hr", mode="WB-wired", scene="baseline"
        )
    with pytest.raises(ValueError, match="8 kHz"):
        acoustic_stress_batch.start_batch(
            wb_echo.dataset_id, "speex_dpdfnet2_8khz", mode="WB-wired", scene="baseline"
        )
