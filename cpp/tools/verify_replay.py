#!/usr/bin/env python3
"""Compare device and host PCM16 replay outputs from the same real input."""
import hashlib
import json
import math
import struct
import sys
import wave
from pathlib import Path

directory = Path(sys.argv[1])
source = Path(sys.argv[2])

def read(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return [v[0] for v in struct.iter_unpack("<h", w.readframes(w.getnframes()))]

before = read(source)
host = read(directory/"host.after.wav")
device = read(directory/"android.after.wav")
host_report = json.loads((directory/"host.json").read_text())
device_report = json.loads((directory/"android.json").read_text())
assert host_report["webrtc_revision"] == device_report["webrtc_revision"]
assert host_report["dsp_delay_samples"] == device_report["dsp_delay_samples"] == 96
assert len(host) == len(device) == len(before)+96
maximum = max(abs(a-b) for a,b in zip(host,device))
assert maximum <= 2, maximum

def dbfs(x):
    return 10*math.log10(max(math.fsum((v/32768)**2 for v in x)/len(x),1e-20))

result = {
    "webrtc_revision": device_report["webrtc_revision"],
    "input_kind": "replay of previously user-recorded device microphone PCM16",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "output_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                      for name in ["host.after.wav","android.after.wav"]},
    "input_samples": len(before), "output_samples": len(device),
    "delay_samples": 96, "max_difference_pcm16_lsb": maximum,
    "before_rms_dbfs": dbfs(before), "android_after_aligned_rms_dbfs": dbfs(device[96:]),
    "android_process_p95_ms": device_report["process_p95_ms"],
    "android_process_max_ms": device_report["process_max_ms"],
    "modules": "WebRTC NS + AGC2 + application limiter; AEC/BF disabled",
    "snr_improvement": None, "erle": None, "mos": None,
    "standard_compliance": False
}
(directory/"parity.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
