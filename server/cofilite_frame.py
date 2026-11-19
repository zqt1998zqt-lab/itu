"""Fixed-shape frame execution and bounded buffers for CoFi-Lite streaming."""

from __future__ import annotations

import threading
import time
import weakref
from typing import Any

import torch
from torch import Tensor


TIME_K3_BLOCKS = (
    "en_erb1",
    "en_lfb1",
    "de_erb1",
    "de_lfb1",
)
TRA_BLOCKS = (
    "en_erb1",
    "en_erb2",
    "en_erb3",
    "en_lfb1",
    "de_erb3",
    "de_erb2",
    "de_erb1",
    "de_lfb1",
)
INTER_RNN_BLOCKS = (
    "inter_rnn1",
    "inter_rnn2",
    "inter_rnn3",
    "inter_rnn4",
)

CONV_STATE_SHAPES = (
    (1, 6, 2, 129),
    (1, 6, 2, 65),
    (1, 6, 2, 65),
    (1, 6, 2, 33),
)
TRA_STATE_SHAPES = (
    (1, 1, 12),
    (1, 1, 12),
    (1, 1, 12),
    (1, 1, 12),
    (1, 1, 12),
    (1, 1, 12),
    (1, 1, 2),
    (1, 1, 2),
)
INTER_STATE_SHAPES = (
    (1, 17, 6),
    (1, 33, 6),
    (1, 17, 6),
    (1, 33, 6),
)
CROSS_STATE_SHAPES = (
    (1, 1, 38),
    (1, 1, 38),
)
STATE_SHAPES = (
    *CONV_STATE_SHAPES,
    *TRA_STATE_SHAPES,
    *INTER_STATE_SHAPES,
    *CROSS_STATE_SHAPES,
)

_EXECUTOR_CACHE: weakref.WeakKeyDictionary[
    torch.nn.Module, tuple[torch.jit.ScriptModule, dict[str, Any]]
] = weakref.WeakKeyDictionary()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def initial_frame_states() -> tuple[Tensor, ...]:
    """Return one zero state for every recurrent/convolution history tensor."""
    return tuple(
        torch.zeros(shape, dtype=torch.float32, device="cpu") for shape in STATE_SHAPES
    )


class CoFiFrameCore(torch.nn.Module):
    """Pure one-frame CoFi-Lite body with explicit fixed-shape state tensors."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _apply_tra(name: str, x: Tensor, hidden: Tensor, tra: torch.nn.Module) -> tuple[Tensor, Tensor]:
        zt = torch.mean(x.pow(2), dim=-1).transpose(1, 2)
        at, next_hidden = tra.att_gru(zt, hidden)
        at = tra.att_fc(at).transpose(1, 2)
        at = tra.att_act(at)
        return x * at[..., None], next_hidden

    def _apply_mobile_block(
        self, name: str, x: Tensor, history: Tensor, tra_hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        block = getattr(self.model, name)
        x = block.pwconv1(x)
        if name in TIME_K3_BLOCKS:
            sequence = torch.cat((history, x), dim=2)
            x = block.dwconv(sequence)[..., -1:, :]
            next_history = sequence[..., -2:, :]
        else:
            x = block.dwconv(x)
            next_history = history
        x = block.pwconv2(x)
        x, next_tra_hidden = self._apply_tra(name, x, tra_hidden, block.tra)
        return x, next_history, next_tra_hidden

    @staticmethod
    def _apply_inter_rnn(
        name: str, x: Tensor, hidden: Tensor, module: torch.nn.Module
    ) -> tuple[Tensor, Tensor]:
        _, channels, _, frequencies = x.shape
        sequence = x.permute(0, 3, 2, 1).reshape(frequencies, 1, channels)
        output, next_hidden = module.inter_rnn(sequence, hidden)
        output = module.inter_fc(output)
        output = output.reshape(1, frequencies, 1, channels).permute(0, 2, 1, 3)
        output = module.inter_ln(output)
        return x + output.permute(0, 3, 1, 2), next_hidden

    def _apply_cross_path(
        self, x1: Tensor, x2: Tensor, hidden1: Tensor, hidden2: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        module = self.model.spgrnn
        x1_residual = x1
        x2_residual = x2
        channels1 = 6
        channels2 = 6
        frequencies1 = 17
        frequencies2 = 33

        x1_flat = x1.permute(0, 2, 1, 3).flatten(2)
        x2_flat = x2.permute(0, 2, 1, 3).flatten(2)
        x = torch.cat((x1_flat, x2_flat), dim=-1)
        x = module.fc(x)
        x = module.ln(x)
        x = module.act(x)
        x1_part, x2_part = torch.chunk(x, chunks=2, dim=-1)

        x1_part, next_hidden1 = module.rnn.rnn1(x1_part, hidden1)
        x2_part, next_hidden2 = module.rnn.rnn2(x2_part, hidden2)
        x = torch.cat((x1_part, x2_part), dim=-1)
        x = module.fc_out(x)
        x1_out, x2_out = torch.split(
            x, (channels1 * frequencies1, channels2 * frequencies2), dim=-1
        )
        x1_out = x1_out.reshape(1, 1, channels1, frequencies1).permute(0, 2, 1, 3)
        x2_out = x2_out.reshape(1, 1, channels2, frequencies2).permute(0, 2, 1, 3)
        return (
            x1_out + x1_residual,
            x2_out + x2_residual,
            next_hidden1,
            next_hidden2,
        )

    def forward(
        self,
        spectrum_real: Tensor,
        spectrum_imag: Tensor,
        conv_en_erb1: Tensor,
        conv_en_lfb1: Tensor,
        conv_de_erb1: Tensor,
        conv_de_lfb1: Tensor,
        tra_en_erb1: Tensor,
        tra_en_erb2: Tensor,
        tra_en_erb3: Tensor,
        tra_en_lfb1: Tensor,
        tra_de_erb3: Tensor,
        tra_de_erb2: Tensor,
        tra_de_erb1: Tensor,
        tra_de_lfb1: Tensor,
        inter_rnn1: Tensor,
        inter_rnn2: Tensor,
        inter_rnn3: Tensor,
        inter_rnn4: Tensor,
        cross1: Tensor,
        cross2: Tensor,
    ) -> tuple[Tensor, ...]:
        x = torch.stack((spectrum_real, spectrum_imag), dim=1).unsqueeze(2)
        x_ref = x.permute(0, 3, 2, 1)

        x_amp = torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + 1e-12).unsqueeze(1)
        x_ric = x.div(x_amp.pow(0.7))
        x_log = torch.log10(x_amp)
        x = torch.cat((x_log, x_ric), dim=1)

        x_erb = self.model.ERB.erb(x_amp)
        x_erb = torch.log10(x_erb)
        x_erb = self.model.sfe(x_erb)

        en_x1, conv_en_erb1, tra_en_erb1 = self._apply_mobile_block(
            "en_erb1", x_erb, conv_en_erb1, tra_en_erb1
        )
        en_x2, _, tra_en_erb2 = self._apply_mobile_block(
            "en_erb2", en_x1, conv_en_erb1, tra_en_erb2
        )
        en_x3, _, tra_en_erb3 = self._apply_mobile_block(
            "en_erb3", en_x2, conv_en_erb1, tra_en_erb3
        )
        emb_erb, inter_rnn1 = self._apply_inter_rnn(
            "inter_rnn1", en_x3, inter_rnn1, self.model.inter_rnn1
        )

        x_f = self.model.sfe(x[..., :65])
        enf_x1, conv_en_lfb1, tra_en_lfb1 = self._apply_mobile_block(
            "en_lfb1", x_f, conv_en_lfb1, tra_en_lfb1
        )
        emb_lfb, inter_rnn2 = self._apply_inter_rnn(
            "inter_rnn2", enf_x1, inter_rnn2, self.model.inter_rnn2
        )

        emb1, emb2, cross1, cross2 = self._apply_cross_path(
            emb_erb, emb_lfb, cross1, cross2
        )

        emb_erb, inter_rnn3 = self._apply_inter_rnn(
            "inter_rnn3", emb1, inter_rnn3, self.model.inter_rnn3
        )
        de_x3, conv_de_erb3, tra_de_erb3 = self._apply_mobile_block(
            "de_erb3", emb_erb + en_x3, conv_de_erb1, tra_de_erb3
        )
        de_x2, _, tra_de_erb2 = self._apply_mobile_block(
            "de_erb2", de_x3 + en_x2, conv_de_erb1, tra_de_erb2
        )
        de_x1, conv_de_erb1, tra_de_erb1 = self._apply_mobile_block(
            "de_erb1", de_x2 + en_x1, conv_de_erb1, tra_de_erb1
        )
        de_x1 = self.model.out_act(de_x1)

        emb_lfb, inter_rnn4 = self._apply_inter_rnn(
            "inter_rnn4", emb2, inter_rnn4, self.model.inter_rnn4
        )
        def_x1, conv_de_lfb1, tra_de_lfb1 = self._apply_mobile_block(
            "de_lfb1", emb_lfb + enf_x1, conv_de_lfb1, tra_de_lfb1
        )
        def_x1 = self.model.out_act(def_x1)

        est_erb = self.model.ERB.ierb(de_x1).permute(0, 3, 2, 1).contiguous()
        enhanced = x_ref * est_erb
        mask = def_x1.permute(0, 3, 2, 1)
        enhanced = torch.cat((enhanced[:, :65] * mask, enhanced[:, 65:]), dim=1)
        return (
            enhanced,
            conv_en_erb1,
            conv_en_lfb1,
            conv_de_erb1,
            conv_de_lfb1,
            tra_en_erb1,
            tra_en_erb2,
            tra_en_erb3,
            tra_en_lfb1,
            tra_de_erb3,
            tra_de_erb2,
            tra_de_erb1,
            tra_de_lfb1,
            inter_rnn1,
            inter_rnn2,
            inter_rnn3,
            inter_rnn4,
            cross1,
            cross2,
        )


class InputRingBuffer:
    """Fixed 512-sample input ring with a reusable contiguous gather scratch."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("input ring capacity must be positive")
        self.data = torch.zeros(capacity, dtype=torch.float32)
        self.scratch = torch.empty(capacity, dtype=torch.float32)
        self.capacity = int(capacity)
        self.write_index = 0
        self.valid_samples = 0

    def reset(self) -> None:
        self.data.zero_()
        self.write_index = 0
        self.valid_samples = 0

    def append(self, samples: Tensor) -> None:
        count = int(samples.numel())
        if count > self.capacity:
            raise ValueError(
                f"input append exceeds ring capacity: {count}>{self.capacity}"
            )
        first = min(count, self.capacity - self.write_index)
        if first:
            self.data[self.write_index : self.write_index + first].copy_(samples[:first])
        remaining = count - first
        if remaining:
            self.data[:remaining].copy_(samples[first:])
        self.write_index = (self.write_index + count) % self.capacity
        self.valid_samples = min(self.capacity, self.valid_samples + count)

    def view(self, count: int) -> Tensor:
        if count < 1 or count > self.valid_samples or count > self.capacity:
            raise RuntimeError(
                f"input ring has {self.valid_samples} valid samples, cannot view {count}"
            )
        start = (self.write_index - count) % self.capacity
        if start + count <= self.capacity:
            return self.data[start : start + count]
        first = self.capacity - start
        self.scratch[:first].copy_(self.data[start:])
        self.scratch[first:count].copy_(self.data[: count - first])
        return self.scratch[:count]


class DelayRingBuffer:
    """Fixed PCM ring that stores the WOLA queue and initial output delay."""

    def __init__(self, delay_samples: int, capacity: int):
        if delay_samples < 1 or capacity < delay_samples:
            raise ValueError("delay ring capacity must be at least the delay")
        self.data = torch.zeros(capacity, dtype=torch.float32)
        self.capacity = int(capacity)
        self.delay_samples = int(delay_samples)
        self.read_index = 0
        self.write_index = delay_samples
        self.available = delay_samples

    def reset(self) -> None:
        self.data.zero_()
        self.read_index = 0
        self.write_index = self.delay_samples
        self.available = self.delay_samples

    def append(self, samples: Tensor) -> None:
        count = int(samples.numel())
        if count > self.capacity - self.available:
            raise RuntimeError(
                "delay ring overflow: "
                f"available={self.available}, append={count}, capacity={self.capacity}"
            )
        first = min(count, self.capacity - self.write_index)
        if first:
            self.data[self.write_index : self.write_index + first].copy_(samples[:first])
        remaining = count - first
        if remaining:
            self.data[:remaining].copy_(samples[first:])
        self.write_index = (self.write_index + count) % self.capacity
        self.available += count

    def pop_into(self, count: int, destination: Tensor) -> None:
        if count < 1 or int(destination.numel()) != count:
            raise ValueError("delay pop count must match a positive destination size")
        if self.available < count:
            raise RuntimeError(
                f"stream output queue underrun: need {count}, have {self.available}"
            )
        first = min(count, self.capacity - self.read_index)
        destination[:first].copy_(self.data[self.read_index : self.read_index + first])
        remaining = count - first
        if remaining:
            destination[first:].copy_(self.data[:remaining])
        self.read_index = (self.read_index + count) % self.capacity
        self.available -= count


def get_frame_executor(
    model: torch.nn.Module, *, jit: bool
) -> tuple[CoFiFrameCore | torch.jit.ScriptModule, dict[str, Any]]:
    """Return a cached eager or once-traced/frozen fixed-state frame executor."""
    if not jit:
        return CoFiFrameCore(model), {
            "mode": "eager",
            "trace_sec": 0.0,
            "freeze": False,
        }

    with _EXECUTOR_CACHE_LOCK:
        cached = _EXECUTOR_CACHE.get(model)
        if cached is not None:
            executor, metadata = cached
            return executor, dict(metadata, cache_hit=True)

        was_training = model.training
        core = CoFiFrameCore(model)
        states = initial_frame_states()
        dummy_real = torch.zeros(1, 257, dtype=torch.float32)
        dummy_imag = torch.zeros(1, 257, dtype=torch.float32)
        started = time.perf_counter()
        try:
            core.eval()
            with torch.inference_mode():
                traced = torch.jit.trace(
                    core,
                    (dummy_real, dummy_imag, *states),
                    strict=False,
                    check_trace=False,
                )
                traced = torch.jit.freeze(traced.eval())
        finally:
            model.train(was_training)
        trace_sec = time.perf_counter() - started
        metadata = {
            "mode": "jit_freeze",
            "trace_sec": float(trace_sec),
            "freeze": True,
            "cache_hit": False,
        }
        _EXECUTOR_CACHE[model] = (traced, metadata)
        return traced, dict(metadata)
