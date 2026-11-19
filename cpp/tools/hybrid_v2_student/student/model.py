"""Causal C64 residual student for dpdf-frame-v1/v2 contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


BANDS: Final = 64
HIDDEN: Final = 64
HEAD_UNITS: Final = 64
FFT_SIZE: Final = 512
FFT_BINS: Final = 257
HOP: Final = 160
WINDOW: Final = 320
SAMPLE_RATE: Final = 16000
FRAME_RATE: Final = 100.0
RESIDUAL_LIMIT_DB: Final = 12.0
INPUT_DIMS: Final = (256, 320)
FEATURE_V1: Final = "dpdf-frame-v1-db"
FEATURE_V2: Final = "dpdf-frame-v2-basegain-db"
FEATURE_CHANNELS_V1: Final = (
    "logpower_db",
    "imcra_noise_db",
    "log_posterior_snr_db",
    "delta_logpower_db",
)
FEATURE_CHANNELS_V2: Final = FEATURE_CHANNELS_V1 + ("basegain_db",)
RAW_FEATURE_SCALING: Final = "none"


def feature_contract(input_dim: int) -> tuple[str, int, int]:
    if input_dim == 256:
        return FEATURE_V1, 1, 45568
    if input_dim == 320:
        return FEATURE_V2, 2, 49664
    raise ValueError(f"input_dim must be 256 or 320, got {input_dim}")


def mac_per_frame(input_dim: int) -> int:
    feature_contract(input_dim)
    return input_dim * 64 + 3 * (64 * HIDDEN + HIDDEN * HIDDEN) + HIDDEN * HEAD_UNITS


@dataclass(frozen=True)
class StudentStats:
    input_dim: int
    hidden_size: int
    parameters: int
    weights_fp32_bytes: int
    state_fp32_bytes_per_item: int
    macs_per_frame: int
    macs_per_second_at_100fps: int
    mmacs_per_second_at_100fps: float
    feature_contract: str
    feature_version: int


@dataclass(frozen=True)
class CausalFeatureState:
    previous_log_power: Tensor | None = None

    @classmethod
    def initial(cls) -> "CausalFeatureState":
        return cls(previous_log_power=None)


def projected_band_matrix(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    centers = torch.empty(BANDS, device=device, dtype=dtype)
    for band in range(BANDS):
        if band < 24:
            centers[band] = 1000.0 * band / 23.0
        elif band < 40:
            centers[band] = 1000.0 + 3000.0 * (band - 23) / 16.0
        else:
            centers[band] = 4000.0 + 4000.0 * (band - 39) / 24.0
    hz = torch.arange(FFT_BINS, device=device, dtype=dtype) * (SAMPLE_RATE / FFT_SIZE)
    left = (torch.searchsorted(centers, hz) - 1).clamp(0, BANDS - 2)
    weight = ((hz - centers[left]) / (centers[left + 1] - centers[left])).clamp(0.0, 1.0)
    matrix = F.one_hot(left, BANDS).to(dtype) * (1.0 - weight)[:, None]
    matrix = matrix + F.one_hot(left + 1, BANDS).to(dtype) * weight[:, None]
    return matrix / matrix.sum(dim=0).clamp_min(1e-12)[None, :]


def project_fft_to_bands(values: Tensor) -> Tensor:
    if values.shape[-1] != FFT_BINS or not values.is_floating_point():
        raise ValueError("FFT input must be floating [...,257]")
    if not bool(torch.isfinite(values).all().item()) or bool(torch.any(values < 0.0).item()):
        raise ValueError("FFT input must be finite and non-negative")
    return values @ projected_band_matrix(device=values.device, dtype=values.dtype)


def _validate_bands(name: str, value: Tensor, *, upper: float | None = None) -> None:
    if value.ndim != 3 or value.shape[-1] != BANDS or not value.is_floating_point():
        raise ValueError(f"{name} must be floating [B,T,64]")
    if not bool(torch.isfinite(value).all().item()) or bool(torch.any(value < 0.0).item()):
        raise ValueError(f"{name} must be finite and non-negative")
    if upper is not None and bool(torch.any(value > upper).item()):
        raise ValueError(f"{name} must be <= {upper}")


def compose_causal_frame_features(
    power: Tensor,
    noise: Tensor,
    basegain: Tensor | None = None,
    previous_log_power: Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Return 256-D v1 features or 320-D v1 + projected base-gain dB."""

    _validate_bands("power", power)
    _validate_bands("noise", noise)
    if (power.shape != noise.shape or power.device != noise.device or
            power.dtype != noise.dtype):
        raise ValueError("power and noise must share shape, device and dtype")
    if power.shape[1] <= 0 or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("power/noise shapes or epsilon are invalid")
    if previous_log_power is not None:
        expected = (power.shape[0], BANDS)
        if tuple(previous_log_power.shape) != expected:
            raise ValueError(f"previous_log_power must have shape {expected}")
        if previous_log_power.device != power.device or previous_log_power.dtype != power.dtype:
            raise ValueError("previous_log_power must share power device and dtype")
        if not bool(torch.isfinite(previous_log_power).all().item()):
            raise ValueError("previous_log_power must be finite")
    logpower = 10.0 * torch.log10(power.clamp_min(eps))
    lognoise = 10.0 * torch.log10(noise.clamp_min(eps))
    delta = torch.zeros_like(logpower)
    if previous_log_power is not None:
        delta[:, 0, :] = logpower[:, 0, :] - previous_log_power
    if power.shape[1] > 1:
        delta[:, 1:, :] = logpower[:, 1:, :] - logpower[:, :-1, :]
    v1 = torch.stack((logpower, lognoise, logpower - lognoise, delta), -2).flatten(-2)
    if basegain is None:
        return v1
    _validate_bands("basegain", basegain, upper=1.0)
    if (basegain.shape != power.shape or basegain.device != power.device or
            basegain.dtype != power.dtype):
        raise ValueError("basegain must match power shape, device and dtype")
    basegain_db = 20.0 * torch.log10(basegain.clamp_min(eps))
    return torch.cat((v1, basegain_db), dim=-1)


def compose_causal_frame_features_with_state(
    power: Tensor,
    noise: Tensor,
    basegain: Tensor | None,
    state: CausalFeatureState,
) -> tuple[Tensor, CausalFeatureState]:
    if not isinstance(state, CausalFeatureState):
        raise TypeError("state must be CausalFeatureState")
    features = compose_causal_frame_features(
        power, noise, basegain, state.previous_log_power
    )
    last = 10.0 * torch.log10(power[:, -1, :].clamp_min(1e-8))
    return features, CausalFeatureState(last)


class ResidualStudent(nn.Module):
    def __init__(self, input_dim: int = 320, hidden: int = 64) -> None:
        super().__init__()
        if input_dim not in INPUT_DIMS or hidden != HIDDEN:
            raise ValueError("supported model is input_dim=256/320, hidden=64")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.feature_contract, self.feature_version, _ = feature_contract(input_dim)
        self.projection = nn.Linear(input_dim, 64)
        self.activation = nn.ReLU()
        self.gru = nn.GRU(64, HIDDEN, 1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(HIDDEN, HEAD_UNITS)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        p = next(self.parameters())
        return torch.zeros(batch_size, HIDDEN, device=p.device if device is None else device,
                           dtype=p.dtype if dtype is None else dtype)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or tuple(hidden.shape) != (x.shape[0], HIDDEN):
            raise ValueError("expected x[B,T,input_dim] and hidden[B,64]")
        y, h = self.gru(self.activation(self.projection(x)), hidden[None])
        return RESIDUAL_LIMIT_DB * torch.tanh(self.head(y)), h[0]

    def step(self, frame: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if frame.ndim != 3 or frame.shape[1] != 1:
            raise ValueError("frame must be [B,1,input_dim]")
        return self.forward(frame, hidden)

    def stats_dict(self) -> dict[str, int | float | str]:
        count = sum(p.numel() for p in self.parameters())
        macs = mac_per_frame(self.input_dim)
        return asdict(StudentStats(self.input_dim, HIDDEN, count, count * 4, HIDDEN * 4,
                                   macs, macs * 100, macs * 100 / 1e6,
                                   self.feature_contract, self.feature_version))


__all__ = (
    "BANDS", "CausalFeatureState", "FEATURE_V1", "FEATURE_V2", "FFT_BINS",
    "HIDDEN", "INPUT_DIMS", "RAW_FEATURE_SCALING", "ResidualStudent",
    "compose_causal_frame_features", "compose_causal_frame_features_with_state",
    "feature_contract", "mac_per_frame", "project_fft_to_bands", "projected_band_matrix",
)
