"""Canonical sample-rate paths for the maintained VCP operating modes.

The project deliberately keeps three primary paths:

* NB: native 8 kHz Speex MDF followed by native 8 kHz DPDFNet-2.
* WB: native 16 kHz LocalVQE 203K followed by native 16 kHz CoFi-Lite.
* SWB: native 48 kHz WebRTC AEC3/NS chain.

24 kHz remains available as a legacy compatibility rate.  It is intentionally
kept out of the primary mapping so new callers cannot mistake it for one of the
three maintained paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRIMARY_RATE_HZ = (8_000, 16_000, 48_000)
LEGACY_RATE_HZ = (24_000,)
NB8_CLUSTER_ID = "speex_dpdfnet2_8khz"

MODE_SAMPLE_RATE_HZ: dict[str, int] = {
    "NB": 8_000,
    "WB": 16_000,
    "SWB": 48_000,
}

NB8_CLUSTER_PRESET: dict[str, Any] = {
    "id": NB8_CLUSTER_ID,
    "label": "VCP · SpeexDSP MDF 8 kHz + DPDFNet-2 8 kHz",
    "description": (
        "原生 8 kHz 维护路径：8 kHz SpeexDSP MDF 线性回声消除 + "
        "DPDFNet-2 8 kHz 神经降噪；全程保持 8 kHz PCM，无重采样与静默回退。"
    ),
    "profile": "vcp",
    "selection": {
        "bf": "das",
        "aec": "speex_mdf_nb8",
        "ns": "dpdfnet2_8khz",
        "eq": "off",
        "agc": "off",
        "drc": "off",
        "limiter": "off",
    },
}

NB8_CLUSTER_CONFIGURATION: dict[str, Any] = {
    "profile": "native_rate_path",
    "selection": {
        "aec": "speex_mdf_nb8",
        "ns": "dpdfnet2_8khz",
    },
}

RATE_16K_ONLY_NS_BACKENDS = frozenset(
    {
        "cofilite",
        "cofilite_native16",
        "dpdfnet2",
        "hybrid_v2_pitch",
    }
)
RATE_16K_ONLY_AEC_BACKENDS = frozenset(
    {
        "speex_fused_dsp",
        "computer_pf_cpu",
    }
)
RATE_48K_ONLY_NS_BACKENDS = frozenset({"dpdfnet2_48k_hr"})

SELECTABLE_CLUSTERS_BY_RATE: dict[int, tuple[str, ...]] = {
    8_000: (NB8_CLUSTER_ID,),
    16_000: (
        "localvqe_cofilite_native16",
        "localvqe_cofilite",
        "localvqe_dpdfnet2",
        "localvqe_hybrid_v2_pitch",
        "localvqe_aec200k",
        "speex_fused_dsp",
        "computer_pf_cpu",
        "vcp_unified",
        "vcp_unified_eq_fused",
        "vcp_default",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "aec3_serial",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
    ),
    48_000: (
        "vcp_aec3_dpdfnet2_48k_hr",
        "aec3_serial",
        "vcp_unified",
        "vcp_unified_eq_fused",
        "vcp_default",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
    ),
    24_000: (
        "vcp_default",
        "vcp_unified",
        "vcp_unified_eq_fused",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "aec3_serial",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
    ),
}


@dataclass(frozen=True)
class RatePath:
    sample_rate_hz: int
    bandwidth: str
    cluster_id: str
    engine_sample_rate_hz: int
    analysis_sample_rate_hz: int
    output_sample_rate_hz: int
    aec_backend: str
    ns_backend: str
    rate_policy: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_RATE_PATHS: dict[int, RatePath] = {
    8_000: RatePath(
        sample_rate_hz=8_000,
        bandwidth="NB",
        cluster_id="speex_dpdfnet2_8khz",
        engine_sample_rate_hz=8_000,
        analysis_sample_rate_hz=8_000,
        output_sample_rate_hz=8_000,
        aec_backend="speex_mdf_nb8",
        ns_backend="dpdfnet2_8khz",
        rate_policy="native",
        status="maintained",
    ),
    16_000: RatePath(
        sample_rate_hz=16_000,
        bandwidth="WB",
        cluster_id="localvqe_cofilite_native16",
        engine_sample_rate_hz=16_000,
        analysis_sample_rate_hz=16_000,
        output_sample_rate_hz=16_000,
        aec_backend="localvqe_aec200k",
        ns_backend="cofilite_native16",
        rate_policy="native",
        status="maintained",
    ),
    48_000: RatePath(
        sample_rate_hz=48_000,
        bandwidth="SWB",
        cluster_id="aec3_serial",
        engine_sample_rate_hz=48_000,
        analysis_sample_rate_hz=48_000,
        output_sample_rate_hz=48_000,
        aec_backend="webrtc_aec3",
        ns_backend="webrtc",
        rate_policy="native",
        status="maintained",
    ),
}


LEGACY_RATE_PATHS: dict[int, RatePath] = {
    24_000: RatePath(
        sample_rate_hz=24_000,
        bandwidth="LEGACY",
        cluster_id="vcp_default",
        engine_sample_rate_hz=24_000,
        analysis_sample_rate_hz=24_000,
        output_sample_rate_hz=24_000,
        aec_backend="speex_mdf",
        ns_backend="webrtc",
        rate_policy="legacy_backend_dependent",
        status="compatibility",
    ),
}


def resolve_rate_path(sample_rate_hz: int) -> RatePath:
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int):
        raise ValueError("sample_rate_hz 必须是整数")
    path = PRIMARY_RATE_PATHS.get(sample_rate_hz) or LEGACY_RATE_PATHS.get(sample_rate_hz)
    if path is None:
        supported = sorted((*PRIMARY_RATE_PATHS, *LEGACY_RATE_PATHS))
        raise ValueError(
            f"不支持的采样率路径 {sample_rate_hz} Hz；可用值为 {supported}"
        )
    return path


def public_rate_paths(*, include_legacy: bool = True) -> list[dict[str, Any]]:
    rows = [path.as_dict() for path in PRIMARY_RATE_PATHS.values()]
    if include_legacy:
        rows.extend(path.as_dict() for path in LEGACY_RATE_PATHS.values())
    return rows


def require_primary_path(sample_rate_hz: int) -> RatePath:
    path = resolve_rate_path(sample_rate_hz)
    if sample_rate_hz not in PRIMARY_RATE_PATHS:
        raise ValueError(f"{sample_rate_hz} Hz 是兼容路径，不属于 8/16/48 kHz 主维护路径")
    return path


def sample_rate_for_mode(mode: str) -> int:
    if not isinstance(mode, str) or "-" not in mode:
        raise ValueError(f"不支持的业务模式 {mode!r}")
    bandwidth = mode.split("-", 1)[0]
    rate = MODE_SAMPLE_RATE_HZ.get(bandwidth)
    if rate is None:
        raise ValueError(f"不支持的业务模式 {mode!r}")
    return rate


def selectable_clusters_for_rate(sample_rate_hz: int) -> tuple[str, ...]:
    resolve_rate_path(sample_rate_hz)
    return SELECTABLE_CLUSTERS_BY_RATE[sample_rate_hz]


def selectable_clusters_by_rate(*, include_legacy: bool = True) -> dict[str, list[str]]:
    rates = PRIMARY_RATE_HZ + (LEGACY_RATE_HZ if include_legacy else ())
    return {str(rate): list(SELECTABLE_CLUSTERS_BY_RATE[rate]) for rate in rates}


def supported_sample_rates_for_cluster(
    cluster_id: str,
    configuration: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    selection = (
        configuration.get("selection")
        if isinstance(configuration, dict) and isinstance(configuration.get("selection"), dict)
        else {}
    )
    aec = selection.get("aec")
    ns = selection.get("ns")
    if cluster_id == NB8_CLUSTER_ID or ns == "dpdfnet2_8khz" or aec == "speex_mdf_nb8":
        return (8_000,)
    if cluster_id == "vcp_aec3_dpdfnet2_48k_hr" or ns in RATE_48K_ONLY_NS_BACKENDS:
        return (48_000,)
    if (
        cluster_id
        in {
            "localvqe_cofilite_native16",
            "localvqe_cofilite",
            "localvqe_dpdfnet2",
            "localvqe_hybrid_v2_pitch",
            "speex_fused_dsp",
            "computer_pf_cpu",
        }
        or ns in RATE_16K_ONLY_NS_BACKENDS
        or aec in RATE_16K_ONLY_AEC_BACKENDS
    ):
        return (16_000,)
    return (16_000, 24_000, 48_000)


def validate_cluster_rate_compatibility(
    cluster_id: str,
    sample_rate_hz: int,
    *,
    configuration: dict[str, Any] | None = None,
    allow_nb8_fallback: bool = True,
) -> None:
    resolve_rate_path(sample_rate_hz)
    supported = supported_sample_rates_for_cluster(cluster_id, configuration)
    if sample_rate_hz in supported:
        return
    if (
        allow_nb8_fallback
        and sample_rate_hz == 8_000
        and supported == (16_000, 24_000, 48_000)
    ):
        return
    labels = "/".join(f"{rate // 1000} kHz" for rate in supported)
    raise ValueError(
        f"算法簇 {cluster_id} 仅支持 {labels}，不能用于 {sample_rate_hz // 1000} kHz（{sample_rate_hz} Hz）"
    )


assert tuple(PRIMARY_RATE_PATHS) == PRIMARY_RATE_HZ
assert tuple(LEGACY_RATE_PATHS) == LEGACY_RATE_HZ
assert not set(PRIMARY_RATE_PATHS).intersection(LEGACY_RATE_PATHS)
assert all(
    path.engine_sample_rate_hz == rate
    and path.analysis_sample_rate_hz == rate
    and path.output_sample_rate_hz == rate
    for rate, path in PRIMARY_RATE_PATHS.items()
)
