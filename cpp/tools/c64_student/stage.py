#!/usr/bin/env python3
"""Copy versioned C64 training sources into a fresh run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_NATIVE_PROBE_SHA256 = "4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a"
SOURCE_FILES = (
    "prepare_data.py", "prepare_features.py", "fetch_training_noise.py",
    "train_student.py", "training-contract.md", "README.md", "stage.py",
    "student/model.py", "student/export.py", "student/verify.py",
    "student/__init__.py", "student/README.md",
)
RUN_DIRECTORIES = ("data", "features", "train", "logs", "native", "reports", "verification", "delivery")
PROBE_NOTE = f"""# Native probe required

No `native/residual_ns_probe` was staged. Use the verified binary from delivery, or rebuild
with the original current `g++ -std=c++20 -O2` configuration. Its SHA-256 must be:

`{EXPECTED_NATIVE_PROBE_SHA256}`

Training refuses any other binary. Never weaken or edit the hard-coded hash lock.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise SystemExit(f"error: output is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise SystemExit(f"error: refusing non-empty output directory: {output_dir}")
    missing = [name for name in SOURCE_FILES if not (source_root / name).is_file()]
    if missing:
        raise SystemExit(f"error: missing versioned source files: {missing}")
    probe = args.probe.expanduser().resolve() if args.probe else None
    probe_sha = None
    if probe is not None:
        if not probe.is_file():
            raise SystemExit(f"error: probe is not a file: {probe}")
        probe_sha = sha256_file(probe)
        if probe_sha != EXPECTED_NATIVE_PROBE_SHA256:
            raise SystemExit(f"error: native probe SHA256 mismatch: got {probe_sha}, expected {EXPECTED_NATIVE_PROBE_SHA256}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, destination)
    for name in RUN_DIRECTORIES:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    if probe is not None:
        destination = output_dir / "native/residual_ns_probe"
        shutil.copy2(probe, destination)
        destination.chmod(0o755)
        (output_dir / "native/probe.json").write_text(
            json.dumps({"source": str(probe), "sha256": probe_sha}, indent=2) + "\n", encoding="utf-8")
    else:
        (output_dir / "native/README.md").write_text(PROBE_NOTE, encoding="utf-8")
    print(json.dumps({"status": "staged", "source_root": str(source_root), "output_dir": str(output_dir),
                      "source_file_count": len(SOURCE_FILES), "probe_sha256": probe_sha,
                      "downloaded": False, "trained": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
