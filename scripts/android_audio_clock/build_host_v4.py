"""Compile the v4 callback CLI against the already-built v5 host DSP objects.

This avoids rebuilding the very large unchanged audio_engine/D chain while
still exercising the hardened AAudio lifecycle source on the host mock path.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path("/home/zhangqingtian/project/itu")
HERE = Path(__file__).resolve().parent
HENGINE = ROOT / "build/native16-20261117/compile-gz6ek6n1"
ORT_HOST = ROOT / "build/deps/onnxruntime-linux-aarch64-1.23.2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resources() -> None:
    mem = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if shutil.disk_usage(ROOT).free < 100 * 1024**3 or mem < 5 * 1024**3:
        raise RuntimeError("Resource stop line; no cleanup allowed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-manifest", type=Path, required=True)
    parser.add_argument("--host-manifest", type=Path, required=True)
    args = parser.parse_args()
    resources()

    android_manifest = json.loads(args.android_manifest.read_text())
    host_manifest = json.loads(args.host_manifest.read_text())
    android_build = Path(android_manifest["build_directory"])
    generated = android_build / "generated/itu_audio_cli.cpp"
    source = generated.read_text()
    if '#include "aaudio_clock_v4.hpp"' not in source:
        raise RuntimeError("generated CLI does not include v4 runtime")
    if '#include "aaudio_clock_v3.hpp"' in source:
        raise RuntimeError("generated CLI still includes v3 runtime")

    host = host_manifest["targets"]["host"]
    host_build = Path(host["binaries"]["callback"]).parent
    engine_object = host_build / "engine.o"
    host_library = Path(host["D_library"])
    overlay = host_build / "overlay"
    if not engine_object.is_file() or not host_library.is_file():
        raise RuntimeError("v5 host engine/D objects missing")
    if sha(host_library) != host["D_library_sha256"]:
        raise RuntimeError("v5 host D library hash changed")

    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime(
        "%Y%m%dT%H%M%S%f"
    )
    out = ROOT / "build/android-audio-clock-20261117" / f"host-v4-{stamp}"
    out.mkdir(parents=True)
    commands = []

    def run(label: str, argv: list[str], cwd: Path) -> None:
        resources()
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        record = {
            "label": label,
            "argv": argv,
            "cwd": str(cwd),
            "exitcode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        (out / f"{label}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        commands.append(record)
        if result.returncode:
            raise RuntimeError(f"{label} failed; evidence {out}")

    recipe = json.loads((HENGINE / "manifest.json").read_text())
    engine_template = recipe["commands"][0]
    compile_argv = list(engine_template)
    clock_object = out / "clock-cli-v4.o"
    compile_argv[compile_argv.index("-c") + 1] = str(generated)
    compile_argv[compile_argv.index("-o") + 1] = str(clock_object)
    compile_argv[1:1] = [
        "-I" + str(overlay),
        "-I" + str(HERE),
        "-I" + str(ROOT / "build/android-native16-20261117/root/overlay"),
        "-I" + str(ORT_HOST / "include"),
        "-DITU_AUDIO_CLOCK_TESTING=1",
    ]
    run("compile-host-clock-v4", compile_argv, ROOT)

    binary = out / "itu_audio_callback_v4"
    link_argv = list(recipe["commands"][-1])
    link_argv[link_argv.index("-o") + 1] = str(binary)
    for index, token in enumerate(link_argv):
        if token == str(HENGINE / "audio_engine.o"):
            link_argv[index] = str(engine_object)
        elif token == str(HENGINE / "itu_audio_cli.o"):
            link_argv[index] = str(clock_object)
    link_argv += [
        str(ORT_HOST / "lib/libonnxruntime.so"),
        "-Wl,-rpath," + str(ORT_HOST / "lib"),
        "-pthread",
    ]
    run("link-host-clock-v4", link_argv, ROOT)

    manifest = {
        "schema": "itu-host-v4-clock-build-v1",
        "created_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "build_directory": str(out),
        "generated_source": {
            "path": str(generated),
            "sha256": sha(generated),
        },
        "host_reference": {
            "path": host["binaries"]["reference"],
            "sha256": host["binaries_sha256"]["reference"],
        },
        "host_callback_v4": {
            "path": str(binary),
            "sha256": sha(binary),
        },
        "host_engine_object": {
            "path": str(engine_object),
            "sha256": sha(engine_object),
        },
        "host_D_library": {
            "path": str(host_library),
            "sha256": sha(host_library),
        },
        "source_files": {
            str(path): sha(path)
            for path in [
                HERE / "aaudio_clock_v4.hpp",
                HERE / "audio_clock_types_v4.hpp",
                HERE / "prepare_cli_v6.py",
                Path(__file__),
            ]
        },
        "commands": commands,
        "scope": (
            "host mock/normal parity only; no Android timing, hardware "
            "callback, or end-to-end audio claim"
        ),
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(out / "MANIFEST.json"),
                "callback": str(binary),
                "callback_sha256": manifest["host_callback_v4"]["sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
