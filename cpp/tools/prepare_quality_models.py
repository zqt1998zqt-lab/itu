#!/usr/bin/env python3
"""Explicit, checksum-verified download of SCOREQ natural/no-reference assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[2]
COMMIT = "0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e"
SIZE = 378049307
RELEASE_SHA256 = "d493ba8acfe37d0aa0acceb85a41f22405fc4b9a5101359a6363d919307300b4"
ZENODO_MD5 = "00f512e40db4e447df54d73150efbebe"
DEFAULT_ASSETS = PROJECT / "build/quality-models/scoreq-natural-nr"
SOURCES = [
    ("https://github.com/alessandroragano/scoreq/releases/download/v1.0.0/adapt_nr_telephone.onnx", "sha256", RELEASE_SHA256),
    ("https://zenodo.org/records/15739280/files/adapt_nr_telephone.onnx", "md5", ZENODO_MD5),
]


def download(url: str, target: Path, expected_size: int | None = None) -> None:
    with urlopen(Request(url, headers={"User-Agent": "ITU-quality-evaluation/1.0"}), timeout=60) as response:
        with target.open("wb") as out:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > (expected_size if expected_size is not None else 2_000_000):
                    raise ValueError("download exceeds expected asset size")
                out.write(chunk)
    if expected_size is not None and target.stat().st_size != expected_size:
        raise ValueError("downloaded model size mismatch")


def prepare(assets: Path) -> dict:
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / "adapt_nr_telephone.onnx"
    if target.exists():
        raw = target.read_bytes()
        if len(raw) != SIZE or hashlib.md5(raw).hexdigest() != ZENODO_MD5:
            raise ValueError("existing model differs from official checksum; preserved")
        origin = "existing verified local model"
    else:
        if shutil.disk_usage(assets).free < 4 * 1024 ** 3 + SIZE:
            raise ValueError("insufficient disk reserve for model download")
        errors = []
        for url, algorithm, expected in SOURCES:
            try:
                with tempfile.NamedTemporaryFile(dir=assets, suffix=".download", delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    print("downloading", url, flush=True)
                    download(url, temporary, SIZE)
                    if hashlib.new(algorithm, temporary.read_bytes()).hexdigest() != expected:
                        raise ValueError("official model checksum mismatch")
                    temporary.replace(target)
                    origin = url
                    break
                finally:
                    temporary.unlink(missing_ok=True)
            except (OSError, ValueError) as error:
                errors.append(f"{url}: {error}")
                print(errors[-1], flush=True)
        else:
            raise RuntimeError("all official sources failed: " + "; ".join(errors))
    source_dir = assets / "source"
    source_dir.mkdir(exist_ok=True)
    snapshots = {}
    for name, relative in (("README.md", "README.md"), ("scoreq.py", "src/scoreq/scoreq.py")):
        path = source_dir / name
        url = f"https://raw.githubusercontent.com/alessandroragano/scoreq/{COMMIT}/{relative}"
        download(url, path)
        snapshots[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    raw = target.read_bytes()
    document = dict(model="scoreq-natural-nr", sample_rate=16000, filename=target.name,
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    md5=hashlib.md5(raw).hexdigest(), official_zenodo_md5=ZENODO_MD5,
                    github_release_sha256=RELEASE_SHA256, source_url=origin, source_commit=COMMIT,
                    weights_license="CC-BY-4.0 (Zenodo record 15739280)", code_license="MIT",
                    attribution="Alessandro Ragano, Jan Skoglund, Andrew Hines; SCOREQ, NeurIPS 2024",
                    license_source="https://zenodo.org/records/15739280", source_snapshots=snapshots)
    from server.quality.metrics import sha
    assert document["md5"] == ZENODO_MD5 and sha(raw) == document["sha256"]
    (assets / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--download", action="store_true", help="explicitly allow official model download")
    args = parser.parse_args()
    if not args.download:
        parser.error("pass --download to acquire model assets; inference itself never downloads")
    sys.path.insert(0, str(PROJECT))
    print(json.dumps(prepare(args.assets), indent=2))


if __name__ == "__main__":
    main()
