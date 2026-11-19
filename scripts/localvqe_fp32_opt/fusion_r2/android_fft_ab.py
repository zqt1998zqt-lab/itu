"""Run an exact-state check and same-binary GCC FFT A/B (60 seconds per arm)."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from build_r2 import BUILD, R1, ROOT, guard, need, new_dir, read, save, sha
from verify_r2 import INVENTORY, MIC, MIC_SHA, NODES, REF, REF_SHA, manifest, qualify

R1_ACCEPTANCE = R1 / "device-campaign.8HBRVI/FINAL-ACCEPTANCE.json"
R1_ACCEPTANCE_SHA = "6d75a0763504bf26895af1cd8d92b54b1382a556af393d9b2835650ec2ecdc72"
R1_PLAN_SHA = "c5d2eb9c09ecf0e04aeb9828719bd8b52a4d063a0c929b19a8197eec0b7715cd"
NO_RESOURCE = re.compile(r"no space left on device|ENOSPC|out of memory|std::bad_alloc", re.I)
LOCK = BUILD / "android-audio-clock-20261117/phone-exclusive.lock"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    build = manifest(args.manifest, "android")
    need(sha(R1_ACCEPTANCE) == R1_ACCEPTANCE_SHA, "Frozen AAudio acceptance drift")
    baseline = read(R1_ACCEPTANCE)
    source_plan = Path(baseline["runs"][0]["directory"]) / "PLAN.json"
    need(sha(source_plan) == R1_PLAN_SHA, "Frozen 60-second plan drift")
    frozen = read(source_plan)
    need(build["gcc_profile"] and build.get("gcc_fft") == "table", "Expected GCC FFT table build")
    need(any(item["sha256"] == "9f632971f60a6db07e8ca2447a51a7a7b4e1da9cc90ce2b2728f92b3d8f6fc05"
             and Path(item["path"]).name == "gcc_fft_table.hpp" and sha(Path(item["path"])) == item["sha256"]
             for item in build["sources"]), "Final FFT header provenance mismatch")
    modes = ("original", "table")
    out = new_dir("device-fft-ab")
    remote = "/data/local/tmp/itu-lvq-r2-" + out.name
    phone = ["adb", "-s", frozen["serial"]]
    receipt_path = out / "ADB"
    receipt_path.mkdir()
    serial = 0
    save(out / "PLAN.json", {"manifest": str(args.manifest.resolve()),
                             "manifest_sha256": sha(args.manifest), "modes": modes,
                             "source_plan": str(source_plan), "remote": remote,
                             "seconds": 60, "phone": phone[-1], "gcc_profile": True,
                             "graph_mode": "norm", "only_timing_variable": "ITU_LVQ_GCC_FFT"})
    print("EVIDENCE", out, flush=True)

    def adb(label: str, *extra: str, timeout: int = 45, code: int = 0):
        nonlocal serial
        guard(out)
        serial += 1
        argv = [*phone, *extra]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace", timeout=timeout)
        raw = result.stdout + result.stderr
        need(not NO_RESOURCE.search(raw), "Resource failure from device; retain files")
        guard(out)
        save(receipt_path / f"{serial:03}-{label}.json",
             {"argv": argv, "exitcode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        need(result.returncode == code, f"Device step {label} failed; evidence {out}")
        return result

    def shell(label: str, command: str, **kw) -> str:
        return adb(label, "shell", "-T", command, **kw).stdout

    def phone_ready(*, idle: bool) -> None:
        data = shell("resources", "getprop ro.soc.model\n"
                     "df -k /data/local/tmp\nawk '/^MemAvailable:/ {print}' /proc/meminfo")
        need(data.splitlines()[0] == "SM8650", "Unexpected phone")
        usable = next(int(l.split()[3]) for l in data.splitlines() if l.startswith("/dev/"))
        free_mem = next(int(l.split()[1]) for l in data.splitlines() if l.startswith("MemAvailable:"))
        # The user's Android memory exception permits the existing 1 GiB admission floor.
        need(usable >= 104857600 and free_mem >= 1048576, "Phone resource stop line")
        if idle:
            tasks = shell("processes", "ps -A -o NAME,ARGS")
            need(not re.search(r"^(?:itu_audio|state-probe|simpleperf)|^sh\s+.*itu-lvq.*runner\.sh",
                               tasks, flags=re.M), "Phone already has an audio test")

    def push(path: str | Path, target: str, digest: str) -> None:
        need(sha(Path(path)) == digest, f"Local payload drift: {path}")
        adb("push", "push", str(path), target)
        need(shell("sha-pushed", shlex.join(["sha256sum", target])).split() == [digest, target],
             f"Upload identity drift: {target}")

    def hit_gate(log: str, fft: str, expected: int) -> list[tuple[int, int]]:
        hits = [(int(on), int(n)) for on, n in re.findall(
            r"^ITU_GCC_FFT_TABLE enabled=([01]) hits=(\d+)$", log, re.M)]
        need(bool(hits) and all(on == int(fft == "table") for on, _ in hits)
             and sum(n for _, n in hits) == expected
             and sum(n != 0 for _, n in hits) == int(expected != 0),
             f"Incorrect FFT mode or execution count: {fft}: {hits}")
        return hits

    def state_probe() -> None:
        files = (("state-probe", build["state_runner"]["path"], build["state_runner"]["sha256"]),
                 ("original.so", build["reference_diagnostic"]["path"], build["reference_diagnostic"]["sha256"]),
                 ("fft-table.so", build["table_diagnostic"]["path"], build["table_diagnostic"]["sha256"]),
                 ("mic.wav", MIC, MIC_SHA), ("ref.wav", REF, REF_SHA))
        for name, path, digest in files:
            push(path, remote + "/" + name, digest)
        shell("chmod-state", shlex.join(["chmod", "700", remote + "/state-probe"]))
        d = out / "state"
        d.mkdir()
        mode = "norm"
        for label, policy, code in (("table", "exact", 0), ("negative", "negative", 3)):
            phone_ready(idle=True)
            cmd = [frozen["argv"][0], frozen["argv"][2], "ITU_LVQ_GCC_FFT=table", remote + "/state-probe",
                   remote + "/original.so", remote + "/fft-table.so",
                   frozen["argv"][4].split("=", 1)[1], remote + "/mic.wav", remote + "/ref.wav", mode, policy]
            p = adb("state-" + label, "shell", "-T", shlex.join(cmd), timeout=90, code=code)
            wanted = ("STATE_MISMATCH mode=norm frame=12 first_state_byte=128 pcm_exact=1"
                      if code else f"STATE_PASS mode={mode} frames=750 tensors=77 bytes_per_frame=1353104 pcm_samples=192000")
            need(wanted in p.stdout + p.stderr, f"Bad state result: {label}")
            actual = INVENTORY.findall(p.stderr)
            need(len(actual) == 2 and all(g[0] == m and tuple(map(int, g[1:])) == (*NODES[m], 1)
                     for g, m in zip(actual, ("norm", mode))), "State probe wrong graph")
            hits = hit_gate(p.stderr, "table", 0 if code else 18)
            save(d / (label + ".json"), {"result": wanted, "exitcode": code, "graphs": actual, "fft_hits": hits})
            print(wanted, flush=True)

    def run_trial(i: int, fft: str) -> dict:
        mode = "norm"
        phone_ready(idle=True)
        trial = out / f"{i}-{fft}"
        trial.mkdir()
        remote_trial = f"{remote}/{i}-{fft}"
        argv = [part.replace(frozen["remote"] + "/out", remote_trial + "/out")
                    .replace(frozen["remote"], remote) for part in frozen["argv"]]
        argv[1] = "ITU_LVQ_FUSION=" + mode
        argv.insert(2, "ITU_LVQ_GCC_FFT=" + fft)
        # The old root is a prefix of the command's binary/SO tokens only.
        need("taskset" not in argv and "--performance-hint" not in argv and
             "--audio-clock-shared" in argv and "--aaudio-clock" in argv, "Timing configuration drift")
        temp = "dumpsys battery | awk '/^[[:space:]]*temperature:/ {print $2}'"
        before = shell("battery-before", temp).strip()
        save(trial / "COMMAND.json", {"argv": argv, "battery_before_deci_c": before})
        script = trial / "runner.sh"
        save(script, f"#!/system/bin/sh\nset -C\ncd {shlex.quote(remote_trial)} || exit 1\n"
                     "date +%s%3N > started_ms\n"
                     f"{shlex.join(argv)} > stdout.log 2> stderr.log\n"
                     "status=$?\ndate +%s%3N > finished_ms\nprintf '%s\\n' \"$status\" > exitcode\n")
        shell("trial-mkdir", shlex.join(["mkdir", remote_trial]))
        push(script, remote_trial + "/runner.sh", sha(script))
        shell("launch", f"set -C\nnohup setsid sh {shlex.quote(remote_trial + '/runner.sh')} "
                        f"> {shlex.quote(remote_trial + '/supervisor.log')} 2>&1 < /dev/null &")
        print(f"START {i} {fft} 60s", flush=True)
        start = time.monotonic()
        while time.monotonic() - start < 105:
            status = shell("poll", f"if test -f {shlex.quote(remote_trial + '/exitcode')}; "
                                   f"then cat {shlex.quote(remote_trial + '/exitcode')}; fi").strip()
            if status.isdigit():
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"No terminal marker; do not start another trial ({remote_trial})")
        phone_ready(idle=True)
        collected = trial / "collected"
        adb("pull", "pull", remote_trial, str(collected), timeout=100)
        need(int((collected / "exitcode").read_text()) == int(status) == 0,
             f"Runtime rejected by Android ({trial})")
        elapsed_ms = int((collected / "finished_ms").read_text()) - int((collected / "started_ms").read_text())
        need(58_000 < elapsed_ms < 80_000, f"Incorrect trial duration ({trial})")
        after = shell("battery-after", temp).strip()
        log = (collected / "stderr.log").read_text()
        need(not NO_RESOURCE.search(log), "Resource failure from DSP")
        hits = hit_gate(log, fft, 18 if fft == "table" else 0)
        result = qualify(collected / "out", build, mode, log, 3750, phone=True)
        result.update({"gcc_fft": fft, "fft_hits": hits})
        need(result["wav_sha256"] == {"." + name: value for name, value in baseline["quality"]["wav_sha256"].items()},
             f"Output differs from frozen R1 byte baseline ({trial})")
        result.update({"trial": i, "directory": str(trial), "device_ms": elapsed_ms,
                       "battery_deci_c": [before, after], "clock_sha256": sha(collected / "out.audio-clock.frames.csv")})
        guard(out)
        save(trial / "RESULT.json", result)
        print("DONE", i, fft, "steady_p99_ms", result["windows"]["steady"]["wall"]["p99_ms"],
              "over16", result["windows"]["all"]["over16"], "xrun", result["xrun"], flush=True)
        return result

    with LOCK.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phone_ready(idle=True)
        assets = frozen["expected_existing_phone_assets"]
        digests = shell("frozen-assets", shlex.join(["sha256sum", *assets]))
        need({path: digest for digest, path in (line.split() for line in digests.splitlines())} == assets,
             "Phone inputs or model assets drifted")
        shell("bundle-mkdir", shlex.join(["mkdir", remote]))
        push(build["runtime"]["path"], remote + "/liblocalvqe.so.0.1.0", build["runtime"]["sha256"])
        push(build["binary"]["path"], remote + "/itu_audio_clock", build["binary"]["sha256"])
        shell("chmod-cli", shlex.join(["chmod", "700", remote + "/itu_audio_clock"]))
        state_probe()
        result = [run_trial(i, mode) for i, mode in enumerate(modes, 1)]
        phone_ready(idle=True)
    need(len({r["clock_sha256"] for r in result}) == len(result), "Trace reuse detected")
    guard(out)
    save(out / "RESULT.json", {"manifest": str(args.manifest.resolve()), "gcc_profile": build["gcc_profile"],
                               "trials": result, "expected_device_assets": assets,
                               "state_bit_exact": True, "state_negative_rejected": True})
    print("PASS", out / "RESULT.json", flush=True)


if __name__ == "__main__":
    main()
