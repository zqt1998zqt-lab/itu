"""Bounded-buffer and minimum-delay tests for the CoFi-Lite frame core."""

from __future__ import annotations

import pytest
import torch

from server.cofilite_frame import DelayRingBuffer, InputRingBuffer


FFT_SIZE = 512
HOP_SAMPLES = 256
CALLBACK_SAMPLES = 160
DEFAULT_DELAY = 480
DEFAULT_CAPACITY = DEFAULT_DELAY + HOP_SAMPLES


def _advance(
    ring: DelayRingBuffer,
    *,
    total_input: int,
    next_frame_index: int,
    count: int,
) -> tuple[int, int]:
    remaining = count
    while remaining:
        frame_ready_at = (
            FFT_SIZE // 2 + 1
            if next_frame_index == 0
            else (next_frame_index + 1) * HOP_SAMPLES
        )
        needed = frame_ready_at - total_input
        if needed > 0:
            take = min(needed, remaining)
            total_input += take
            remaining -= take
        if total_input == frame_ready_at:
            if next_frame_index > 0:
                ring.append(torch.zeros(HOP_SAMPLES, dtype=torch.float32))
            next_frame_index += 1
    return total_input, next_frame_index


def _consume(
    ring: DelayRingBuffer,
    *,
    total_input: int,
    next_frame_index: int,
    count: int,
) -> tuple[int, int]:
    offset = 0
    while offset < count:
        piece = min(CALLBACK_SAMPLES, count - offset)
        total_input, next_frame_index = _advance(
            ring,
            total_input=total_input,
            next_frame_index=next_frame_index,
            count=piece,
        )
        ring.pop_into(piece, torch.empty(piece, dtype=torch.float32))
        offset += piece
    return total_input, next_frame_index


def test_input_ring_wrap_gather_has_stable_bounded_storage():
    ring = InputRingBuffer(FFT_SIZE)
    values = torch.arange(700, dtype=torch.float32)
    ring.append(values[:160])
    ring.append(values[160:320])
    ring.append(values[320:480])
    ring.append(values[480:640])

    gathered = ring.view(FFT_SIZE)
    assert gathered.numel() == FFT_SIZE
    assert torch.equal(gathered, values[128:640])
    assert gathered.data_ptr() == ring.scratch.data_ptr()

    ring.append(values[640:700])
    gathered = ring.view(FFT_SIZE)
    assert torch.equal(gathered, values[188:700])
    assert gathered.data_ptr() == ring.scratch.data_ptr()


def test_delay_480_is_minimum_safe_160_multiple_across_lcm_phases():
    ring = DelayRingBuffer(DEFAULT_DELAY, DEFAULT_CAPACITY)
    total_input = 0
    next_frame_index = 0
    for _ in range(80):
        total_input, next_frame_index = _consume(
            ring,
            total_input=total_input,
            next_frame_index=next_frame_index,
            count=CALLBACK_SAMPLES,
        )

    assert total_input == 12_800
    assert next_frame_index == 50

    unsafe = DelayRingBuffer(320, 320 + HOP_SAMPLES)
    total_input = 0
    next_frame_index = 0
    with pytest.raises(RuntimeError, match="underrun"):
        for _ in range(80):
            total_input, next_frame_index = _consume(
                unsafe,
                total_input=total_input,
                next_frame_index=next_frame_index,
                count=CALLBACK_SAMPLES,
            )


def test_delay_480_flush_covers_every_eos_remainder():
    for sample_count in range(257, 1537):
        ring = DelayRingBuffer(DEFAULT_DELAY, DEFAULT_CAPACITY)
        full_samples = (sample_count // CALLBACK_SAMPLES) * CALLBACK_SAMPLES
        total_input, next_frame_index = _consume(
            ring,
            total_input=0,
            next_frame_index=0,
            count=full_samples,
        )
        tail_samples = sample_count - full_samples
        if tail_samples:
            total_input, next_frame_index = _advance(
                ring,
                total_input=total_input,
                next_frame_index=next_frame_index,
                count=tail_samples,
            )
        assert total_input == sample_count

        valid_model_samples = next_frame_index * HOP_SAMPLES
        remainder = sample_count - valid_model_samples
        ring.append(torch.zeros(HOP_SAMPLES, dtype=torch.float32))
        if remainder:
            ring.append(torch.zeros(remainder, dtype=torch.float32))

        assert ring.available == DEFAULT_DELAY + tail_samples
        if tail_samples:
            ring.pop_into(
                tail_samples, torch.empty(tail_samples, dtype=torch.float32)
            )
        output = torch.empty(DEFAULT_DELAY, dtype=torch.float32)
        ring.pop_into(DEFAULT_DELAY, output)
        assert ring.available == 0
