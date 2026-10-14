#!/usr/bin/env python3
"""Fetch the pinned official SpeexDSP release used for rate conversion."""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz"
SHA256 = "8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("build/deps"))
    parser.add_argument("--archive", type=Path, help="Use an existing official archive")
    args = parser.parse_args()
    if args.archive:
        data = args.archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=180) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("SpeexDSP archive SHA-256 mismatch; nothing extracted")
    args.destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(args.destination, filter="data")
    root = args.destination / "speexdsp-1.2.1"
    (root / "ITU_SOURCE_LOCK.json").write_text(
        json.dumps({"version": "1.2.1", "url": URL, "sha256": SHA256}, indent=2) + "\n"
    )
    print(root.resolve())


if __name__ == "__main__":
    main()
