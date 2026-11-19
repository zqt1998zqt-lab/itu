#!/usr/bin/env python3
"""Build the experimental LocalVQE external-mask bridge without touching prebuilt artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "f53063c9eb2a85f96479867d1dd911dc3bf6319b"
DEFAULT_SOURCE = (
    ROOT / "build" / "deps" / f"localvqe-{SOURCE_REVISION}" / "ggml"
)
DEFAULT_PREBUILT = ROOT / "build" / "localvqe-host-20261010"
BRIDGE_SOURCE = ROOT / "cpp" / "tools" / "localvqe_external_mask.cpp"
MIN_FREE_DISK = 100 * 1024**3
MIN_MEM_AVAILABLE = 5 * 1024**3
OUTPUT_NAME = "liblocalvqe_external_mask.so"
REQUIRED_SYMBOLS = (
    "localvqe_external_bridge_version",
    "localvqe_external_mask_frame_f32",
    "localvqe_capture_daf_frame_f32",
    "localvqe_external_codec_frame_f32",
)


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            return int(fields[1]) * 1024
    raise BuildError("cannot parse MemAvailable from /proc/meminfo")


def require_new_empty_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise BuildError(f"--output must already exist and be empty: {resolved}")
    if not resolved.is_dir():
        raise BuildError(f"--output is not a directory: {resolved}")
    if any(resolved.iterdir()):
        raise BuildError(f"--output is not empty; refusing to overwrite: {resolved}")
    return resolved


def reject_unhashed_bypass(flags_text: str) -> None:
    if "LOCALVQE_ALLOW_UNHASHED" in os.environ:
        raise BuildError(
            "LOCALVQE_ALLOW_UNHASHED is set; refusing any unhashed-model build"
        )
    if "LOCALVQE_ALLOW_UNHASHED" in flags_text:
        raise BuildError(
            "flags.make attempts to define LOCALVQE_ALLOW_UNHASHED; refusing build"
        )


def parse_flags_make(path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    compiler: str | None = None
    values: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        compiler_match = re.match(r"^# compile CXX with (.+)$", line)
        if compiler_match:
            compiler_tokens = shlex.split(compiler_match.group(1))
            if len(compiler_tokens) != 1:
                raise BuildError("flags.make CXX compiler line is not a single path")
            compiler = compiler_tokens[0]
            continue
        match = re.match(r"^(CXX_DEFINES|CXX_INCLUDES|CXX_FLAGS) = (.*)$", line)
        if match:
            values[match.group(1)] = shlex.split(match.group(2))
    missing = [
        key for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS") if key not in values
    ]
    if compiler is None or missing:
        raise BuildError(f"flags.make is missing fields: {missing or ['CXX compiler']}")
    return (
        compiler,
        values["CXX_DEFINES"],
        values["CXX_INCLUDES"],
        values["CXX_FLAGS"],
    )


def resolve_prebuilt_token(token: str, prebuilt: Path) -> str:
    if token.startswith("-"):
        return token
    candidate = Path(token)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (prebuilt / candidate).resolve()
    if not resolved.exists():
        raise BuildError(f"prebuilt link token does not exist: {token} -> {resolved}")
    return str(resolved)


def transform_link_command(
    tokens: list[str], prebuilt: Path, bridge_object: Path, output_so: Path
) -> tuple[list[str], Path, Path]:
    if not tokens:
        raise BuildError("link.txt is empty")
    compiler = tokens[0]
    try:
        output_index = tokens.index("-o")
    except ValueError as error:
        raise BuildError("link.txt has no -o target") from error
    if output_index + 1 >= len(tokens):
        raise BuildError("link.txt has a dangling -o")
    original_output = (prebuilt / tokens[output_index + 1]).resolve()

    api_indexes = [
        index
        for index, token in enumerate(tokens)
        if not token.startswith("-")
        and Path(token).name == "localvqe_api.cpp.o"
    ]
    if len(api_indexes) != 1:
        raise BuildError(
            f"link.txt must contain exactly one localvqe_api.cpp.o, found {len(api_indexes)}"
        )
    api_index = api_indexes[0]
    original_api_object = resolve_prebuilt_token(tokens[api_index], prebuilt)

    transformed: list[str] = [compiler]
    soname_seen = False
    for index, token in enumerate(tokens[1:], start=1):
        if index == output_index + 1:
            transformed.append(str(output_so))
            continue
        if index == api_index:
            transformed.append(str(bridge_object))
            continue
        if token.startswith("-Wl,-soname,"):
            transformed.append(f"-Wl,-soname,{OUTPUT_NAME}")
            soname_seen = True
            continue
        transformed.append(resolve_prebuilt_token(token, prebuilt))

    if not soname_seen:
        raise BuildError("link.txt has no -Wl,-soname entry")
    return transformed, Path(original_api_object), original_output


def run_logged(command: list[str], cwd: Path, env: dict[str, str]) -> dict:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "command": command,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    if process.returncode != 0:
        raise BuildError(
            f"command failed ({process.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prebuilt", type=Path, default=DEFAULT_PREBUILT)
    args = parser.parse_args()

    output = require_new_empty_output(args.output)
    source = args.source.expanduser().resolve()
    prebuilt = args.prebuilt.expanduser().resolve()
    if not source.is_dir():
        raise BuildError(f"--source is not a directory: {source}")
    if not prebuilt.is_dir():
        raise BuildError(f"--prebuilt is not a directory: {prebuilt}")

    flags_path = prebuilt / "CMakeFiles" / "localvqe_shared.dir" / "flags.make"
    link_path = prebuilt / "CMakeFiles" / "localvqe_shared.dir" / "link.txt"
    for required in (flags_path, link_path, BRIDGE_SOURCE):
        if not required.is_file():
            raise BuildError(f"required input is missing: {required}")

    flags_text = flags_path.read_text(encoding="utf-8")
    reject_unhashed_bypass(flags_text)
    disk_free = int(shutil.disk_usage(output).free)
    mem_available = mem_available_bytes()
    resource_snapshot = {
        "output_disk_free_bytes": disk_free,
        "output_disk_free_gib": disk_free / 1024**3,
        "mem_available_bytes": mem_available,
        "mem_available_gib": mem_available / 1024**3,
        "minimum_disk_free_bytes": MIN_FREE_DISK,
        "minimum_mem_available_bytes": MIN_MEM_AVAILABLE,
    }
    if disk_free < MIN_FREE_DISK:
        raise BuildError(
            f"disk free {disk_free} is below the required {MIN_FREE_DISK}; stopping without writes"
        )
    if mem_available < MIN_MEM_AVAILABLE:
        raise BuildError(
            f"MemAvailable {mem_available} is below the required {MIN_MEM_AVAILABLE}; stopping without writes"
        )

    compiler, defines, includes, flags = parse_flags_make(flags_path)
    if not Path(compiler).is_file():
        raise BuildError(f"compiler from flags.make does not exist: {compiler}")
    object_path = output / "localvqe_external_mask.cpp.o"
    output_so = output / OUTPUT_NAME
    manifest_path = output / "manifest.json"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if "LOCALVQE_ALLOW_UNHASHED" in env:
        raise BuildError("LOCALVQE_ALLOW_UNHASHED is set; refusing build")

    source_files = {
        "bridge": BRIDGE_SOURCE,
        "upstream_api_cpp": source / "localvqe_api.cpp",
        "upstream_api_header": source / "localvqe_api.h",
        "localvqe_graph_header": source / "localvqe_graph.h",
        "localvqe_graph_source": source / "localvqe_graph.cpp",
        "daf_frontend_header": source / "daf_frontend.h",
        "daf_frontend_source": source / "daf_frontend.cpp",
        "localvqe_model_header": source / "localvqe_model.h",
        "noise_gate_header": source / "noise_gate.h",
        "ggml_header": source / "vendor" / "ggml" / "include" / "ggml.h",
    }
    for path in source_files.values():
        if not path.is_file():
            raise BuildError(f"manifest source is missing: {path}")

    link_tokens = shlex.split(link_path.read_text(encoding="utf-8"))
    link_command, original_api_object, original_output = transform_link_command(
        link_tokens, prebuilt, object_path, output_so
    )
    archive_paths = [
        Path(token)
        for token in link_command
        if not token.startswith("-") and Path(token).suffix == ".a"
    ]
    if not archive_paths or any(not path.is_absolute() for path in archive_paths):
        raise BuildError("all archives in the transformed link command must be absolute")

    source_hashes_before = {
        name: sha256(path) for name, path in source_files.items()
    }
    prebuilt_hashes_before = {
        "flags_make": sha256(flags_path),
        "link_txt": sha256(link_path),
        "replaced_api_object": sha256(original_api_object),
        "original_prebuilt_so": sha256(original_output),
    }
    archive_hashes_before = {str(path): sha256(path) for path in archive_paths}

    compile_command = [
        compiler,
        *defines,
        *includes,
        *flags,
        "-c",
        str(BRIDGE_SOURCE),
        "-o",
        str(object_path),
    ]
    compile_record = run_logged(compile_command, output, env)

    link_record = run_logged(link_command, output, env)
    if not output_so.is_file():
        raise BuildError(f"link completed without producing {output_so}")

    nm_command = ["nm", "-D", "--defined-only", str(output_so)]
    nm_record = run_logged(nm_command, output, env)
    nm_output = nm_record["stdout"]
    missing_symbols = [
        symbol for symbol in REQUIRED_SYMBOLS if not re.search(rf"\b{symbol}\b", nm_output)
    ]
    if missing_symbols:
        raise BuildError(f"shared library is missing exported symbols: {missing_symbols}")

    readelf_command = ["readelf", "-d", str(output_so)]
    readelf_record = run_logged(readelf_command, output, env)
    expected_soname = f"Library soname: [{OUTPUT_NAME}]"
    if expected_soname not in readelf_record["stdout"]:
        raise BuildError(f"unexpected soname; expected {expected_soname}")

    source_hashes_after = {
        name: sha256(path) for name, path in source_files.items()
    }
    prebuilt_hashes_after = {
        "flags_make": sha256(flags_path),
        "link_txt": sha256(link_path),
        "replaced_api_object": sha256(original_api_object),
        "original_prebuilt_so": sha256(original_output),
    }
    archive_hashes_after = {str(path): sha256(path) for path in archive_paths}
    if source_hashes_after != source_hashes_before:
        raise BuildError("source or header SHA changed during the build")
    if prebuilt_hashes_after != prebuilt_hashes_before:
        raise BuildError("prebuilt metadata or artifact SHA changed during the build")
    if archive_hashes_after != archive_hashes_before:
        raise BuildError("archive SHA changed during the build")

    archive_records = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": archive_hashes_after[str(path)],
        }
        for path in archive_paths
    ]
    artifact = {
        "path": str(output_so),
        "bytes": output_so.stat().st_size,
        "sha256": sha256(output_so),
        "soname": OUTPUT_NAME,
    }
    manifest = {
        "schema_version": 1,
        "bridge_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": SOURCE_REVISION,
        "paths": {
            "output": str(output),
            "source": str(source),
            "prebuilt": str(prebuilt),
            "flags_make": str(flags_path),
            "link_txt": str(link_path),
            "original_api_object": str(original_api_object),
            "original_prebuilt_so": str(original_output),
        },
        "resources": resource_snapshot,
        "source_sha256": {
            name: {"path": str(path), "sha256": source_hashes_after[name]}
            for name, path in source_files.items()
        },
        "prebuilt_metadata_sha256": prebuilt_hashes_after,
        "compile": compile_record,
        "link": link_record,
        "archives": archive_records,
        "nm": {
            **nm_record,
            "required_symbols": list(REQUIRED_SYMBOLS),
        },
        "readelf": readelf_record,
        "artifact": artifact,
        "safety": {
            "output_was_new_and_empty": True,
            "prebuilt_artifacts_overwritten": False,
            "temporary_files_deleted": False,
            "localvqe_allow_unhashed_rejected": True,
        },
    }
    manifest_blob = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(manifest_blob)
    print(
        json.dumps(
            {
                "artifact": artifact,
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256(manifest_path),
                },
                "exported_symbols": list(REQUIRED_SYMBOLS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"build-localvqe-external-mask: {error}", file=sys.stderr)
        raise SystemExit(2)
