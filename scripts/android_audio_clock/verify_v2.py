"""Offline acceptance for the hardened AAudio v5 runtime and host DSP parity."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path("/home/zhangqingtian/project/itu")
HERE = Path(__file__).resolve().parent


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-manifest", type=Path, required=True)
    parser.add_argument("--host-manifest", type=Path, required=True)
    parser.add_argument("--host-v4-manifest", type=Path, required=True)
    args = parser.parse_args()
    android_manifest = json.loads(args.android_manifest.read_text())
    host_manifest = json.loads(args.host_manifest.read_text())
    host_v4_manifest = json.loads(args.host_v4_manifest.read_text())

    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime(
        "%Y%m%dT%H%M%S%f"
    )
    out = (
        ROOT
        / "build/android-audio-clock-20261117"
        / f"validation-v5-{stamp}"
    )
    out.mkdir(parents=True)
    commands = []

    def run(label: str, argv: list[str], env=None, success: bool = True):
        result = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        record = {
            "label": label,
            "argv": argv,
            "exitcode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        (out / f"{label}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        commands.append(record)
        require(
            (result.returncode == 0) == success,
            f"{label} unexpected exit; evidence {out}",
        )
        return result

    for path in HERE.rglob("*.py"):
        ast.parse(path.read_text())

    android = android_manifest["targets"]["android"]
    generated = Path(android_manifest["build_directory"]) / "generated/itu_audio_cli.cpp"
    source = generated.read_text()
    require('#include "aaudio_clock_v5.hpp"' in source, "v5 runtime missing")
    require('#include "aaudio_clock_v4.hpp"' not in source, "v4 runtime mixed")
    require('#include "aaudio_clock_v3.hpp"' not in source, "v3 runtime mixed")
    require(
        android["D_archive_sha256"]
        == "a08ca4868fa3a19b6b527d7769e1c427766fd617202ee5353688161bc8b893ba",
        "vcp-D archive hash changed",
    )
    require(sha(Path(android["D_archive"])) == android["D_archive_sha256"], "D archive drift")
    require(sha(Path(android["D_library"])) == android["D_library_sha256"], "D library drift")

    fake = out / "aaudio-contract-v5"
    run(
        "compile-fake-aaudio-v5",
        [
            "/usr/bin/g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-DITU_AUDIO_CLOCK_FAKE_ANDROID",
            str(HERE / "tests/aaudio_clock_contract_v5.cpp"),
            "-o",
            str(fake),
        ],
    )
    fake_run = run("fake-aaudio-v5", [str(fake)])
    require("RESULT PASS cases=11" in fake_run.stdout, "v5 fake scenarios failed")
    for scenario in [
        "buffer_target_1536",
        "buffer_capacity_below_callback",
        "actual_buffer_below_callback",
        "stop_wait_timeout_no_close",
        "start_failed_close_success",
    ]:
        require(
            f"SCENARIO {scenario} PASS" in fake_run.stdout,
            f"missing scenario {scenario}",
        )

    run(
        "adpf-component-verification",
        [
            sys.executable,
            "-B",
            str(HERE / "tests/verify_performance_hint.py"),
        ],
    )
    elf = run(
        "android-elf",
        [
            "/usr/bin/llvm-readelf-17",
            "-h",
            "-l",
            "-d",
            android["binary"],
        ],
    )
    require(
        "AArch64" in elf.stdout
        and "/system/bin/linker64" in elf.stdout
        and "libaaudio.so" in elf.stdout
        and "libandroid.so" in elf.stdout,
        "Android AAudio linkage invalid",
    )

    host = host_manifest["targets"]["host"]
    host_v4 = host_v4_manifest["host_callback_v4"]
    require(sha(Path(host_v4["path"])) == host_v4["sha256"], "host v4 drift")
    require(
        sha(Path(host["D_library"])) == host["D_library_sha256"],
        "host D drift",
    )
    scenes = json.loads(
        (ROOT / "build/localvqe-opt-20261117/INPUTS.json").read_text()
    )["cases"]
    env = dict(
        os.environ,
        ITU_LOCALVQE_LIBRARY=host["D_library"],
        ITU_LOCALVQE_MODEL=host_manifest["model_paths"]["localvqe"],
    )
    cases = []
    for scene in scenes:
        name = scene["condition"]
        prefixes = {}
        reports = {}
        for variant, binary, extras in [
            ("classic", host["binaries"]["reference"], []),
            ("v4-callback", host_v4["path"], []),
            ("v5-mock", host_v4["path"], ["--audio-clock-test"]),
        ]:
            directory = out / f"{name}-{variant}"
            directory.mkdir()
            prefix = directory / "out"
            prefixes[variant] = prefix
            argv = [
                binary,
                "--profile",
                "vcp",
                "--no-rx-processing",
                "--aec-backend",
                "localvqe_aec200k",
                "--ns-backend",
                "webrtc",
                "--eq-preamp-db",
                "-2",
                "--eq-band",
                "low_shelf,180,-2,0.707",
                "--eq-band",
                "peak,1800,2,1",
                "--eq-band",
                "high_shelf,4500,-1,0.707",
                "--tx-drc",
                "--localvqe-reference-channel",
                "0",
                "--beam-delays",
                "1.0,0.0",
                "--native16-cofi-model",
                host_manifest["model_paths"]["cofi"],
                "--native16-cofi-window",
                host_manifest["model_paths"]["window"],
                "--mic",
                scene["files"]["mic"]["path"],
                "--echo",
                scene["files"]["echo"]["path"],
                "--profile-stages",
                "--out",
                str(prefix),
                *extras,
            ]
            run(f"{name}-{variant}", argv, env)
            reports[variant] = json.loads(
                Path(str(prefix) + ".json").read_text()
            )
            require(
                reports[variant]["aec_provider_sha256"]
                == host["D_library_sha256"],
                f"{name}/{variant} provider drift",
            )
            require(
                reports[variant]["aec_model_sha256"]
                == host_manifest["model_sha256"]["localvqe"],
                f"{name}/{variant} model drift",
            )
            require(
                reports[variant]["aec_localvqe_inferences"] == 752
                and reports[variant]["ns_inference_calls"] == 753
                and reports[variant]["aec_localvqe_fallback_frames"] == 0
                and reports[variant]["ns_external_fallback_frames"] == 0,
                f"{name}/{variant} inference/EOS mismatch",
            )
            require(
                reports[variant]["dsp_delay_samples"] == 513
                and reports[variant]["input_samples_per_channel"] == 192000,
                f"{name}/{variant} timeline mismatch",
            )
        wav_hashes = {}
        for suffix in ["after.wav", "aec.wav", "aec-input.wav", "bf.wav"]:
            payloads = [
                Path(str(prefixes[variant]) + "." + suffix).read_bytes()
                for variant in prefixes
            ]
            require(
                all(payload == payloads[0] for payload in payloads),
                f"{name}/{suffix} host PCM mismatch",
            )
            wav_hashes[suffix] = hashlib.sha256(payloads[0]).hexdigest()
        for variant in ["v4-callback", "v5-mock"]:
            sidecar = json.loads(
                Path(str(prefixes[variant]) + ".audio-clock.json").read_text()
            )
            rows = list(
                csv.DictReader(
                    Path(
                        str(prefixes[variant]) + ".audio-clock.frames.csv"
                    ).open()
                )
            )
            require(
                sidecar["callbacks"] == 753
                and sidecar["clock_errors"] == 0
                and len(rows) == 753
                and all(row["clocks_valid"] == "1" for row in rows),
                f"{name}/{variant} callback trace invalid",
            )
            require(
                sidecar["mock_clock"] == (variant == "v5-mock")
                and not sidecar["hardware_audio_clock_executed"]
                and not sidecar["microphone_capture_verified"]
                and not sidecar["processed_audio_playback_verified"],
                f"{name}/{variant} false hardware claim",
            )
        cases.append(
            {
                "scene": name,
                "four_wavs_bit_exact_across_three_paths": True,
                "wav_sha256": wav_hashes,
                "AEC_inferences": 752,
                "CoFi_inferences": 753,
                "fallback": 0,
                "delay_samples": 513,
            }
        )

    scene = scenes[0]
    unsupported_dir = out / "unsupported-host"
    unsupported_dir.mkdir()
    unsupported_prefix = unsupported_dir / "out"
    unsupported_argv = [
        host_v4["path"],
        "--profile",
        "vcp",
        "--no-rx-processing",
        "--aec-backend",
        "localvqe_aec200k",
        "--ns-backend",
        "webrtc",
        "--native16-cofi-model",
        host_manifest["model_paths"]["cofi"],
        "--native16-cofi-window",
        host_manifest["model_paths"]["window"],
        "--mic",
        scene["files"]["mic"]["path"],
        "--echo",
        scene["files"]["echo"]["path"],
        "--beam-delays",
        "1.0,0.0",
        "--out",
        str(unsupported_prefix),
        "--aaudio-clock",
    ]
    run("host-aaudio-unavailable", unsupported_argv, env, False)
    failed = json.loads(
        Path(str(unsupported_prefix) + ".audio-clock.json").read_text()
    )
    require(
        not failed["hardware_audio_clock_executed"]
        and not failed["audio_stream"]["completed"]
        and not Path(str(unsupported_prefix) + ".json").exists(),
        "host AAudio failure became false success",
    )

    result = {
        "status": "passed_offline",
        "android_manifest": str(args.android_manifest),
        "host_manifest": str(args.host_manifest),
        "host_v4_manifest": str(args.host_v4_manifest),
        "fake_aaudio_v5_scenarios": 11,
        "adpf_fake_scenarios": 13,
        "host_cases": cases,
        "four_wav_comparisons": 5 * 4 * 2,
        "android_binary_linked": True,
        "android_v5_buffer_target": 1536,
        "android_d_archive_sha256": android["D_archive_sha256"],
        "android_d_library_sha256": android["D_library_sha256"],
        "phone_connected_or_executed": False,
        "actual_device_audio_configuration": "measured separately in device trials",
        "latency_or_scheduling_improvement": "not established by this offline check",
        "end_to_end_capture_playback": "not_verified",
        "hardware_performance_hint": "not_measured_offline",
        "commands": commands,
    }
    (out / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "passed_offline",
                "result": str(out / "RESULT.json"),
                "fake_scenarios": 11,
                "host_cases": len(cases),
                "wav_comparisons": result["four_wav_comparisons"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
