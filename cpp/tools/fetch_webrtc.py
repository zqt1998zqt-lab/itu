#!/usr/bin/env python3
"""Fetch exact official WebRTC and Chromium audio dependency snapshots."""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

WEBRTC = "52022620d7f1f97eed6bcfbfcb2744fb26acc86f"
CHROMIUM = "8a83e63525b495aa7f35648a3b4b676279deb864"
root = Path(sys.argv[1] if len(sys.argv) > 1 else "build/webrtc-src").resolve()
root.mkdir(parents=True, exist_ok=True)
sources = [("webrtc", f"https://webrtc.googlesource.com/src/+archive/{WEBRTC}.tar.gz", root)]
for name in ["abseil-cpp", "pffft", "rnnoise"]:
    sources.append((name, f"https://chromium.googlesource.com/chromium/src/third_party/+archive/{CHROMIUM}/{name}.tar.gz", root/"third_party"/name))
record = {"webrtc_commit": WEBRTC, "chromium_third_party_commit": CHROMIUM, "archives": {}}
if (root/".git").exists():
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != WEBRTC:
        raise SystemExit(f"WebRTC checkout is {head}; expected {WEBRTC}; use a separate source directory")
for name, url, dest in sources:
    cache = root.parent/f"{name}-{WEBRTC if name=='webrtc' else CHROMIUM}.tar.gz"
    if not cache.exists():
        print(f"Downloading {name} {url}", flush=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        cache.write_bytes(data)
    else:
        data = cache.read_bytes()
    dest.mkdir(parents=True, exist_ok=True)
    # Archive paths are still checked despite using official immutable endpoints.
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(dest, filter="data")
    record["archives"][name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
(root/"ITU_SOURCE_LOCK.json").write_text(json.dumps(record, indent=2)+"\n")
print(root)
