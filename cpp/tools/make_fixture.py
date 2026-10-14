#!/usr/bin/env python3
"""Deterministic synthetic implementation fixture, never a device recording."""
import json
import math
import random
import struct
import sys
import wave
from pathlib import Path

directory = Path(sys.argv[1] if len(sys.argv) > 1 else "build/fixture")
directory.mkdir(parents=True, exist_ok=True)
rate = 16000
count = 8 * rate
rng = random.Random(20261008)
far = [rng.uniform(-0.2, 0.2) if 1 <= n/rate < 4 or 5 <= n/rate < 8 else 0.0 for n in range(count)]
near = [0.12 * math.sin(2*math.pi*317*n/rate) + 0.08 * math.sin(2*math.pi*913*n/rate)
        if 4 <= n/rate < 7 else 0.0 for n in range(count)]
mic = []
for n in range(count):
    for channel in range(2):
        delay = channel*3
        speech = near[n-delay] if n >= delay else 0
        echo = 0.6*far[n-7-channel*4] if n >= 7+channel*4 else 0
        echo += 0.12*far[n-29-channel*4] if n >= 29+channel*4 else 0
        mic.append(speech+echo+rng.uniform(-0.003, 0.003))

def write(name, channels, samples):
    with wave.open(str(directory/name), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(struct.pack("<h", round(max(-1, min(0.999969, v))*32768)) for v in samples))

write("before.wav", 2, mic)
write("echo.wav", 1, far)
write("reference.wav", 1, near)
(directory/"source.json").write_text(json.dumps({
    "input_kind": "deterministic synthetic fixture, not a microphone recording",
    "seed": 20261008, "sample_rate": rate, "microphones": 2,
    "beam_compensation_samples": [3, 0],
    "noise_only_seconds": [0, 1], "far_only_seconds": [[1, 4], [7, 8]],
    "near_only_seconds": [4, 5], "double_talk_seconds": [5, 7]
}, indent=2)+"\n")
print(directory)
