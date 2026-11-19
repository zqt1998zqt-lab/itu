#!/usr/bin/env python3
"""Offline host/NDK verification for the Android performance hint helper."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEADER = HERE.parent / "android_performance_hint.hpp"
TEST = HERE / "test_performance_hint.cpp"
OUTPUT_ROOT = ROOT / "build/android-audio-clock-20261117/adpf-tests"

HOST_CXX = Path("/usr/bin/clang++-17")
ANDROID_CXX = Path("/usr/bin/clang++-17")
NDK_ROOT = (
    ROOT
    / "build/android-native16-20261117/root/ndk/android-ndk-r27c"
)
ANDROID_SYSROOT = (
    NDK_ROOT / "toolchains/llvm/prebuilt/linux-x86_64/sysroot"
)
ANDROID_RESOURCE_DIR = (
    ROOT
    / "build/android-native16-20261117/root/clang17-android-resource"
)

GIB = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_snapshot() -> dict[str, int]:
    mem_available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    return {
        "disk_available_bytes": shutil.disk_usage(ROOT).free,
        "mem_available_bytes": mem_available,
    }


def enforce_resource_guard(snapshot: dict[str, int]) -> None:
    if snapshot["disk_available_bytes"] < 100 * GIB:
        raise RuntimeError("disk below 100 GiB; stop without cleanup")
    if snapshot["mem_available_bytes"] < 5 * GIB:
        raise RuntimeError("MemAvailable below 5 GiB; stop without cleanup")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    initial_resources = resource_snapshot()
    enforce_resource_guard(initial_resources)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"run-{stamp}"
    output.mkdir(exist_ok=False)

    events: list[dict[str, object]] = []
    result_path = output / "RESULT.json"

    def run(
        label: str,
        argv: list[str],
        *,
        expected_return: int = 0,
        required_lines: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        event = {
            "label": label,
            "argv": argv,
            "exitcode": process.returncode,
            "expected_return": expected_return,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        events.append(event)
        write_text_exclusive(output / f"{label}.stdout.txt", process.stdout)
        write_text_exclusive(output / f"{label}.stderr.txt", process.stderr)
        write_text_exclusive(
            output / f"{label}.command.json",
            json.dumps(event, ensure_ascii=False, indent=2) + "\n",
        )
        if process.returncode != expected_return:
            raise RuntimeError(
                f"{label}: exit {process.returncode}, expected "
                f"{expected_return}"
            )
        for line in required_lines:
            if line not in process.stdout:
                raise RuntimeError(f"{label}: missing required line {line!r}")
        return process

    try:
        host_binary = output / "test-performance-hint-host"
        run(
            "compile-host",
            [
                str(HOST_CXX),
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-pthread",
                "-I",
                str(HEADER.parent),
                str(TEST),
                "-ldl",
                "-o",
                str(host_binary),
            ],
        )
        run(
            "host-positive",
            [str(host_binary)],
            required_lines=(
                "PASS off requested=0 getManager=0 createSession=0",
                "PASS missing_symbol explicit=1 dropped=1",
                "PASS null_manager error=manager_unavailable dropped=1",
                "PASS create_fail error=session_create_failed dropped=1",
                "PASS forward tid=4242 reports=3 exact=1 "
                "session_ever_created_after_close=1",
                "PASS report_error last_api_result=-32 active=0",
                "PASS queue_overflow dropped=3 delivered=256",
                "PASS concurrent_spsc submitted=5000 conserved=1",
                "PASS tid_change first=1001 createSession=1 explicit_error=1",
                "PASS main_tid_rejected=1 actual_tid=700",
                "NEGATIVE wrong_tid expected_failure detected=1",
                "NEGATIVE wrong_duration expected_failure detected=1",
                "RESULT PASS scenarios=13 fake_tid_and_duration_checked=1 "
                "negative_controls=2",
            ),
        )

        negative_binary = output / "test-performance-hint-negative"
        run(
            "compile-injected-negative",
            [
                str(HOST_CXX),
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-DITU_ADPF_TEST_INJECT_WRONG_EXPECTATION=1",
                "-pthread",
                "-I",
                str(HEADER.parent),
                str(TEST),
                "-ldl",
                "-o",
                str(negative_binary),
            ],
        )
        run(
            "host-injected-negative",
            [str(negative_binary)],
            expected_return=1,
            required_lines=(
                "FAIL reportActualWorkDuration durations exact",
                "RESULT FAIL",
            ),
        )

        android_flags = [
            str(ANDROID_CXX),
            "--target=aarch64-linux-android26",
            f"--sysroot={ANDROID_SYSROOT}",
            f"-resource-dir={ANDROID_RESOURCE_DIR}",
            "-DANDROID",
            "-stdlib=libc++",
            "-rtlib=compiler-rt",
            "-std=gnu++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(HEADER.parent),
            "-fsyntax-only",
        ]
        run(
            "android-header-syntax",
            [*android_flags, "-x", "c++", str(HEADER)],
        )
        run(
            "android-test-syntax",
            [*android_flags, str(TEST)],
        )

        final_resources = resource_snapshot()
        enforce_resource_guard(final_resources)
        result = {
            "status": "passed",
            "completion_marker": "ITU_ADPF_OFFLINE_VERIFICATION_PASS",
            "real_phone_tested": False,
            "adb_used": False,
            "android_install_or_settings_changed": False,
            "python_ast_only": True,
            "py_compile_used": False,
            "host_fake_cases": [
                "off",
                "missing_symbol",
                "null_manager",
                "create_failure",
                "report_error",
                "queue_overflow",
                "callback_tid_change",
                "main_tid_rejected",
                "invalid_duration",
                "close_pending",
            ],
            "fake_received_tid_and_duration_verified": True,
            "injected_negative_exitcode": 1,
            "android_syntax": {
                "compiler": str(ANDROID_CXX),
                "target": "aarch64-linux-android26",
                "ndk": str(NDK_ROOT),
                "header_exitcode": 0,
                "test_exitcode": 0,
            },
            "files_sha256": {
                str(HEADER.relative_to(ROOT)): sha256(HEADER),
                str(TEST.relative_to(ROOT)): sha256(TEST),
                str(Path(__file__).resolve().relative_to(ROOT)): sha256(
                    Path(__file__).resolve()
                ),
            },
            "initial_resources": initial_resources,
            "final_resources": final_resources,
            "events": events,
        }
        write_text_exclusive(
            result_path,
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "completion_marker": result["completion_marker"],
                    "evidence": str(result_path),
                    "real_phone_tested": False,
                    "events": len(events),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "error": str(error),
            "real_phone_tested": False,
            "events": events,
            "initial_resources": initial_resources,
        }
        if not result_path.exists():
            write_text_exclusive(
                result_path,
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "evidence": str(result_path),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
