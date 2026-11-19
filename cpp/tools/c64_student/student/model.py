"""Causal single-head C64 residual-noise student.

The model consumes already-computed raw dB features and applies its trainable
input projection without any external normalisation. It has one 64-band output:
a bounded amplitude-gain residual in dB. No speech posterior, SNR head, teacher,
or runtime-quality input exists in this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


NUM_BANDS: Final = 64
FEATURES_PER_BAND: Final = 4
FEATURE_DIM: Final = NUM_BANDS * FEATURES_PER_BAND
HIDDEN_SIZE: Final = 64
HEAD_UNITS: Final = 64
SAMPLE_RATE_HZ: Final = 16000
FFT_SIZE: Final = 512
HOP_SAMPLES: Final = 160
WINDOW_SAMPLES: Final = 320
NUM_FFT_BINS: Final = FFT_SIZE // 2 + 1
FRAME_RATE_HZ: Final = 100.0
RESIDUAL_LIMIT_DB: Final = 12.0
FEATURE_CONTRACT_VERSION: Final = "dpdf-frame-v1-db"
FEATURE_CONTRACT_ID: Final = 1
RAW_FEATURE_SCALING: Final = "none"
PARAMETER_COUNT: Final = 45568
MAC_PER_FRAME: Final = 45056
MAC_PER_SECOND_AT_100FPS: Final = 4505600
WEIGHTS_FP32_BYTES: Final = PARAMETER_COUNT * 4
STATE_FP32_BYTES_PER_ITEM: Final = HIDDEN_SIZE * 4
FEATURE_COMPATIBILITY_NOTE: Final = (
    "dpdf-frame-v1-db uses raw dB feature scales and projected bands; it is not "
    "compatible with existing TinyNsCorrector weights or feature scaling."
)
FEATURE_CHANNELS: Final[tuple[str, ...]] = (
    "logpower_db",
    "imcra_noise_db",
    "log_posterior_snr_db",
    "delta_logpower_db",
)


@dataclass(frozen=True)
class StudentStats:
    hidden_size: int
    parameters: int
    weights_fp32_bytes: int
    state_fp32_bytes_per_item: int
    macs_per_frame: int
    macs_per_second_at_100fps: int
    mmacs_per_second_at_100fps: float


@dataclass(frozen=True)
class CausalFeatureState:
    """Explicit previous-frame history for projected-band feature construction."""

    previous_log_power: Tensor | None = None

    @classmethod
    def initial(cls) -> "CausalFeatureState":
        return cls(previous_log_power=None)


def _validate_spectral_power(name: str, value: Tensor) -> None:
    if value.ndim != 3 or value.shape[-1] != NUM_BANDS:
        raise ValueError(f"{name} must have shape [B,T,{NUM_BANDS}]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if bool(torch.any(value < 0.0).item()):
        raise ValueError(f"{name} must be non-negative projected-band power")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_previous_log_power(previous: Tensor, batch_size: int) -> None:
    if previous.ndim != 2 or tuple(previous.shape) != (batch_size, NUM_BANDS):
        raise ValueError(
            f"previous_log_power must have shape [{batch_size},{NUM_BANDS}]"
        )
    if not previous.is_floating_point():
        raise TypeError("previous_log_power must be floating point")
    if not bool(torch.isfinite(previous).all().item()):
        raise ValueError("previous_log_power must contain only finite values")


def projected_band_centers_hz(
    sample_rate: int = SAMPLE_RATE_HZ,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the 64 Hz-domain band centers used by the native feature map."""

    if sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(f"only {SAMPLE_RATE_HZ} Hz is supported")
    bands = torch.arange(NUM_BANDS, device=device, dtype=dtype)
    nyquist = 0.5 * sample_rate
    low = 1000.0 * bands / 23.0
    mid = 1000.0 + 3000.0 * (bands - 23.0) / 16.0
    high = 4000.0 + (nyquist - 4000.0) * (bands - 39.0) / 24.0
    return torch.where(bands < 24.0, low, torch.where(bands < 40.0, mid, high))


def projected_band_matrix(
    sample_rate: int = SAMPLE_RATE_HZ,
    nfft: int = FFT_SIZE,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the FFT512-bin-to-64-projected-band linear map ``[257,64]``."""

    if sample_rate != SAMPLE_RATE_HZ or nfft != FFT_SIZE:
        raise ValueError("the deployment contract is 16 kHz with FFT512")
    centers = projected_band_centers_hz(sample_rate, device=device, dtype=dtype)
    hz = torch.arange(NUM_FFT_BINS, device=device, dtype=dtype) * (
        sample_rate / float(nfft)
    )
    left = torch.searchsorted(centers, hz, right=False) - 1
    left = left.clamp(min=0, max=NUM_BANDS - 2)
    width = centers[left + 1] - centers[left]
    weight = ((hz - centers[left]) / width).clamp(min=0.0, max=1.0)
    left_basis = F.one_hot(left, NUM_BANDS).to(dtype=dtype)
    right_basis = F.one_hot(left + 1, NUM_BANDS).to(dtype=dtype)
    matrix = left_basis * (1.0 - weight).unsqueeze(1)
    matrix = matrix + right_basis * weight.unsqueeze(1)
    normalization = matrix.sum(dim=0).clamp_min(1e-12)
    return matrix / normalization.unsqueeze(0)


def project_fft_power_to_bands(
    fft_power: Tensor,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
    nfft: int = FFT_SIZE,
) -> Tensor:
    """Project non-negative FFT512 power/noise bins to 64 projected bands."""

    if fft_power.shape[-1] != NUM_FFT_BINS:
        raise ValueError(f"fft_power last dimension must be {NUM_FFT_BINS}")
    if not fft_power.is_floating_point():
        raise TypeError("fft_power must be floating point")
    if bool(torch.any(fft_power < 0.0).item()) or not bool(
        torch.isfinite(fft_power).all().item()
    ):
        raise ValueError("fft_power must be finite and non-negative")
    matrix = projected_band_matrix(
        sample_rate,
        nfft,
        device=fft_power.device,
        dtype=fft_power.dtype,
    )
    return fft_power @ matrix


def compose_causal_frame_features(
    power: Tensor,
    imcra_noise: Tensor,
    previous_log_power: Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Build four causal 64-band groups in channel-major order.

    The trailing 64 axis is projected bands, not raw FFT bins. Group order is
    ``logpower_db, imcra_noise_db, logpostSNR_db, delta_logpower_db``. The delta
    uses only the previous frame; the first frame is zero only when no explicit
    previous log-power state is supplied.
    """

    _validate_spectral_power("power", power)
    _validate_spectral_power("imcra_noise", imcra_noise)
    if power.shape != imcra_noise.shape:
        raise ValueError("power and imcra_noise must have identical shapes")
    if power.shape[1] <= 0:
        raise ValueError("at least one frame is required")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if previous_log_power is not None:
        _validate_previous_log_power(previous_log_power, power.shape[0])
        if previous_log_power.device != power.device:
            raise ValueError("previous_log_power must be on the power device")
        if previous_log_power.dtype != power.dtype:
            raise TypeError("previous_log_power must have the power dtype")

    logpower = 10.0 * torch.log10(power.clamp_min(eps))
    log_noise = 10.0 * torch.log10(imcra_noise.clamp_min(eps))
    logpost_snr = logpower - log_noise
    delta = torch.zeros_like(logpower)
    if previous_log_power is not None:
        delta[:, 0, :] = logpower[:, 0, :] - previous_log_power
    if logpower.shape[1] > 1:
        delta[:, 1:, :] = logpower[:, 1:, :] - logpower[:, :-1, :]
    return torch.stack(
        (logpower, log_noise, logpost_snr, delta), dim=-2
    ).flatten(start_dim=-2)


def compose_causal_frame_features_with_state(
    power: Tensor,
    imcra_noise: Tensor,
    state: CausalFeatureState,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, CausalFeatureState]:
    """Chunk-safe feature construction with explicit, non-resetting history."""

    if not isinstance(state, CausalFeatureState):
        raise TypeError("state must be CausalFeatureState")
    features = compose_causal_frame_features(
        power,
        imcra_noise,
        previous_log_power=state.previous_log_power,
        eps=eps,
    )
    last_log_power = 10.0 * torch.log10(power[:, -1, :].clamp_min(eps))
    return features, CausalFeatureState(previous_log_power=last_log_power)


class ResidualStudent(nn.Module):
    """256 -> 64 ReLU -> GRU64 -> 64 bounded amplitude residual dB."""

    def __init__(self, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__()
        if hidden != HIDDEN_SIZE:
            raise ValueError(f"this deployment contract requires hidden={HIDDEN_SIZE}")
        self.hidden = int(hidden)
        self.projection = nn.Linear(FEATURE_DIM, 64)
        self.activation = nn.ReLU()
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=self.hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(self.hidden, HEAD_UNITS)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return explicit zero GRU state ``[batch,64]``; silence never resets it."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``residual_db[B,T,64], hidden_out[B,64]``."""

        if x.ndim != 3 or x.shape[-1] != FEATURE_DIM:
            raise ValueError(f"x must have shape [B,T,{FEATURE_DIM}]")
        batch_size = x.shape[0]
        if hidden.ndim != 2 or tuple(hidden.shape) != (batch_size, self.hidden):
            raise ValueError(
                f"hidden must have shape [{batch_size},{self.hidden}]"
            )
        if x.device != hidden.device or x.dtype != hidden.dtype:
            raise ValueError("x and hidden must share device and dtype")
        projected = self.activation(self.projection(x))
        sequence, hidden_out = self.gru(projected, hidden.unsqueeze(0))
        residual_db = RESIDUAL_LIMIT_DB * torch.tanh(self.head(sequence))
        return residual_db, hidden_out.squeeze(0)

    def step(self, frame: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Run one frame with explicit state; output time axis remains singleton."""

        if frame.ndim != 3 or frame.shape[1] != 1:
            raise ValueError("frame must have shape [B,1,256]")
        return self.forward(frame, hidden)

    def architecture_stats(self) -> StudentStats:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        input_macs = FEATURE_DIM * 64
        gru_macs = 3 * (64 * self.hidden + self.hidden * self.hidden)
        head_macs = self.hidden * HEAD_UNITS
        macs_per_frame = input_macs + gru_macs + head_macs
        return StudentStats(
            hidden_size=self.hidden,
            parameters=parameters,
            weights_fp32_bytes=parameters * 4,
            state_fp32_bytes_per_item=self.hidden * 4,
            macs_per_frame=macs_per_frame,
            macs_per_second_at_100fps=macs_per_frame * 100,
            mmacs_per_second_at_100fps=macs_per_frame * 100 / 1_000_000.0,
        )

    def stats_dict(self) -> dict[str, int | float]:
        return asdict(self.architecture_stats())


__all__: Sequence[str] = (
    "CausalFeatureState",
    "FEATURES_PER_BAND",
    "FEATURE_CHANNELS",
    "FEATURE_COMPATIBILITY_NOTE",
    "FEATURE_CONTRACT_ID",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_DIM",
    "FFT_SIZE",
    "FRAME_RATE_HZ",
    "HEAD_UNITS",
    "HIDDEN_SIZE",
    "HOP_SAMPLES",
    "MAC_PER_FRAME",
    "MAC_PER_SECOND_AT_100FPS",
    "NUM_BANDS",
    "NUM_FFT_BINS",
    "PARAMETER_COUNT",
    "RAW_FEATURE_SCALING",
    "RESIDUAL_LIMIT_DB",
    "ResidualStudent",
    "SAMPLE_RATE_HZ",
    "STATE_FP32_BYTES_PER_ITEM",
    "StudentStats",
    "WEIGHTS_FP32_BYTES",
    "WINDOW_SAMPLES",
    "compose_causal_frame_features",
    "compose_causal_frame_features_with_state",
    "project_fft_power_to_bands",
    "projected_band_centers_hz",
    "projected_band_matrix",
)
