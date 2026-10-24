#!/usr/bin/env python3
"""Build and retain an explicit source-to-binary identity for each DSP revision."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=PROJECT / "build/repair-sequential-20261017/build-tree")
    args = parser.parse_args()
    stage = args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "identity.json").exists():
        raise FileExistsError(stage / "identity.json")
    names = subprocess.check_output(["rg", "--files", "cpp"], cwd=PROJECT, text=True).splitlines()
    paths = [PROJECT / n for n in names if Path(n).suffix in (".hpp", ".h", ".cpp", ".cc", ".c", ".cmake")
             or Path(n).name == "CMakeLists.txt"] + [PROJECT / "CMakeLists.txt"]
    before = {str(p.relative_to(PROJECT)): digest(p) for p in paths}
    for p in paths:
        target = stage / "source" / p.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    (stage / "source-before.json").write_text(json.dumps(before, indent=2) + "\n")
    with (stage / "build-verified.log").open("x") as log:
        subprocess.run(["cmake", "--build", str(args.build), "-j", "8"], cwd=PROJECT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    changed = [name for name, expected in before.items() if digest(PROJECT / name) != expected]
    if changed:
        raise ValueError(f"source changed during build: {changed}")
    artifacts = {}
    for name in ("itu_audio_cli", "itu_audio_tests", "itu_vcp_pipeline_tests", "itu_ecnr_v2_tests"):
        output = stage / name
        shutil.copy2(args.build / "cpp" / name, output)
        artifacts[name] = digest(output)
    identity = dict(source_sha256=before, source_verified_after_build=True, artifact_sha256=artifacts,
                    build_directory=str(args.build.resolve()), script_sha256=digest(Path(__file__)))
    (stage / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(dict(source_files=len(paths), source_verified=True, artifacts=artifacts)), flush=True)


if __name__ == "__main__":
    main()
