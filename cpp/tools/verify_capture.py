#!/usr/bin/env python3
"""Recompute observable real-capture levels without claiming reference-based quality."""
import hashlib
import json
import math
import struct
import sys
import wave
from pathlib import Path

directory = Path(sys.argv[1])
report = json.loads((directory / "capture.json").read_text())
assert report["input_kind"] == "device microphone PCM16"
assert report["frames"] > 0 and report["sample_rate"] == 16000

def read(name):
    with wave.open(str(directory / name)) as wav:
        assert wav.getframerate() == report["sample_rate"]
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        count = wav.getnframes()
        raw = wav.readframes(count)
        assert len(raw) == count * 2
        return [v[0] / 32768 for v in struct.iter_unpack("<h", raw)]

def levels(samples):
    assert samples
    power = math.fsum(v*v for v in samples) / len(samples)
    peak = max(abs(v) for v in samples)
    return {
        "samples": len(samples), "rms_dbfs": 10*math.log10(max(power, 1e-20)),
        "peak_dbfs": 20*math.log10(max(peak, 1e-10)),
        "clipping_percent_abs_ge_0_999": sum(abs(v) >= 0.999 for v in samples)*100/len(samples),
        "dc_offset": math.fsum(samples)/len(samples)
    }

before = read("capture.before.wav")
after = read("capture.after.wav")
delay = report["dsp_delay_samples"]
assert len(before) == report["frames"] * report["frame_size"]
assert len(after) == len(before)+delay
if report.get("schema_version",1) == 1:
    assert all(v == 0 for v in after[:delay])
aligned = after[delay:]
stats_before, stats_after = levels(before), levels(aligned)
result = {
    "schema_version": 1, "input_kind": report["input_kind"],
    "source_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                      for name in ["capture.before.wav", "capture.after.wav", "capture.json"]},
    "dsp_delay_samples": delay, "dsp_delay_ms": delay*1000/report["sample_rate"],
    "alignment": "remove leading dsp_delay_samples from after", "before": stats_before,
    "after_aligned": stats_after,
    "overall_level_increase_db": stats_after["rms_dbfs"]-stats_before["rms_dbfs"],
    "first_0_5_seconds": {"before": levels(before[:8000]), "after_aligned": levels(aligned[:8000])},
    "remaining_0_5_to_end": {"before": levels(before[8000:]), "after_aligned": levels(aligned[8000:])},
    "quality_claims": {"snr_improvement": None, "erle": None, "mos": None},
    "notes": ["WebRTC NS uses online noise estimation; the first 0.5 seconds are a descriptive window, not a certified noise segment.",
              "AGC changes total level; total level increase is not SNR or quality improvement.",
              "This capture exercises NS/AGC/limiter only; no hardware AEC/BF/DTD validation."]
}
destination = directory/"capture.verified.json"
destination.write_text(json.dumps(result, indent=2)+"\n")
print(destination)
print(json.dumps({"level_increase_db": result["overall_level_increase_db"],
                  "before_dbfs": stats_before["rms_dbfs"], "after_dbfs": stats_after["rms_dbfs"]}))
