#!/home/zhangqingtian/.venv/bin/python
"""Create or verify the fixed OM-LSA tuning evaluation fixture.

Every destination is created with exclusive-create semantics. An existing file
is accepted only when its bytes are identical; this script never deletes,
moves, renames, or overwrites an existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
sys.dont_write_bytecode = True

import omlsa_tuning_fixture as fixture


MINIMUM_FREE_BYTES = 100 * 1024**3
MINIMUM_MEMAVAILABLE_BYTES = 5 * 1024**3


class PreparationError(RuntimeError):
    pass


def _memavailable_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise PreparationError("MemAvailable is missing from /proc/meminfo")


def assert_resource_floor() -> dict[str, int]:
    free = shutil.disk_usage(PROJECT_ROOT).free
    available = _memavailable_bytes()
    if free < MINIMUM_FREE_BYTES or available < MINIMUM_MEMAVAILABLE_BYTES:
        raise PreparationError(
            "resource safety floor reached; preserve all files and stop writes: "
            f"disk_free={free}, mem_available={available}"
        )
    return {"disk_free": free, "mem_available": available}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_source_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise PreparationError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        raise PreparationError(f"{label} is missing: {path}")
    if not path.is_file():
        raise PreparationError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _identity(path: Path, label: str) -> dict[str, Any]:
    source = _assert_source_file(path, label)
    return {
        "source_path": str(path.absolute()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_upstream_module() -> Any:
    path = PROJECT_ROOT / fixture.UPSTREAM_SCRIPT_RELATIVE
    module_name = "itu_omlsa_tuning_upstream_for_fixture_prepare"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot construct import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_upstream_protocol() -> Any:
    upstream = _load_upstream_module()
    fixture.assert_upstream_protocol(
        upstream.DEV_REFERENCE_SHA256,
        upstream.HOLDOUT_REFERENCES,
        upstream.EXPECTED_MODEL_SHA256,
    )
    upstream_path = PROJECT_ROOT / fixture.UPSTREAM_SCRIPT_RELATIVE
    upstream_identity = _identity(upstream_path, "upstream evaluation script")
    if upstream_identity["sha256"] != fixture.UPSTREAM_SCRIPT_SHA256:
        raise PreparationError("upstream evaluation script SHA-256 changed")
    if upstream_identity["bytes"] != fixture.UPSTREAM_SCRIPT_BYTES:
        raise PreparationError("upstream evaluation script byte count changed")
    return upstream


def _decode_json_file(path: Path, label: str) -> Any:
    source = _assert_source_file(path, label)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot parse {label} {path}: {error}") from error


def _dev_manifest_origin() -> dict[str, Any]:
    source_path = fixture.DEV_MANIFEST_SOURCE_PATH
    frozen_metadata = dict(fixture.KNOWN_CASE_PROTOCOLS[0])
    record: dict[str, Any] = {
        "source_path": str(source_path),
        "present_at_preparation": source_path.is_file(),
        "sha256": None,
        "bytes": None,
        "policy": (
            "The /tmp manifest is optional historical provenance. The fixed dev "
            "protocol below is authoritative and the current evaluator never reads /tmp."
        ),
        "frozen_metadata": frozen_metadata,
    }
    if source_path.is_symlink():
        raise PreparationError(f"historical dev manifest must not be a symlink: {source_path}")
    if not source_path.exists():
        return record

    identity = _identity(source_path, "historical dev manifest")
    record.update(
        {
            "present_at_preparation": True,
            "sha256": identity["sha256"],
            "bytes": identity["bytes"],
        }
    )
    manifest = _decode_json_file(source_path, "historical dev manifest")
    if not isinstance(manifest, Mapping):
        raise PreparationError("historical dev manifest must be a JSON object")
    declared_sha = manifest.get("clean_reference_sha256")
    if declared_sha is not None and declared_sha != fixture.DEV_REFERENCE_SHA256:
        raise PreparationError(
            "historical dev manifest clean_reference_sha256 disagrees with fixed source"
        )
    if manifest.get("noise_seed") not in (None, frozen_metadata["noise_seed"]):
        raise PreparationError("historical dev manifest noise_seed changed")
    if manifest.get("target_snr_db") not in (None, frozen_metadata["target_snr_db"]):
        raise PreparationError("historical dev manifest target_snr_db changed")
    if manifest.get("rate") not in (None, fixture.RATE):
        raise PreparationError("historical dev manifest rate changed")

    declared_path = manifest.get("clean_reference_path")
    if declared_path is not None:
        declared = Path(str(declared_path))
        if not declared.is_absolute():
            raise PreparationError("historical dev manifest reference path must be absolute")
        canonical = fixture.DATASET_AUDIO_DIR / f"{fixture.DEV_REFERENCE_SHA256}.wav"
        if os.path.normpath(str(declared)) != os.path.normpath(str(canonical)):
            raise PreparationError(
                "historical dev manifest reference path disagrees with the fixed source"
            )
        declared_identity = _identity(declared, "historical dev manifest reference")
        if declared_identity["sha256"] != fixture.DEV_REFERENCE_SHA256:
            raise PreparationError(
                "historical dev manifest reference WAV SHA-256 changed"
            )
    record.update(
        {
            "declared_reference_path": declared_path,
            "declared_noise_seed": manifest.get("noise_seed"),
            "declared_target_snr_db": manifest.get("target_snr_db"),
            "declared_rate": manifest.get("rate"),
            "verification": "metadata_and_reference_sha_verified",
        }
    )
    return record


def _artifact_record(
    artifact_id: str,
    kind: str,
    role: str,
    source_case_id: str,
    source_path: Path,
    relative_path: str,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    identity = _identity(source_path, artifact_id)
    if identity["sha256"] != expected_sha256:
        raise PreparationError(
            f"{artifact_id} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {identity['sha256']}"
        )
    if identity["bytes"] != expected_bytes:
        raise PreparationError(
            f"{artifact_id} byte count mismatch: expected {expected_bytes}, "
            f"got {identity['bytes']}"
        )
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "role": role,
        "source_case_id": source_case_id,
        "source_path": str(source_path.absolute()),
        "relative_path": relative_path,
        "sha256": expected_sha256,
        "bytes": expected_bytes,
    }


def _build_manifest() -> tuple[dict[str, Any], list[tuple[Path, str, int]]]:
    _verify_upstream_protocol()
    upstream_path = PROJECT_ROOT / fixture.UPSTREAM_SCRIPT_RELATIVE
    artifacts: list[dict[str, Any]] = []
    copies: list[tuple[Path, str, int]] = []

    reference_specs = [
        (
            f"reference:{fixture.DEV_REFERENCE_SHA256}",
            "dev",
            "sce1-wb-wired-near_single-01-conversation-v2",
            fixture.DEV_REFERENCE_SHA256,
        ),
        *[
            (
                f"reference:{record[0]}",
                "holdout",
                f"sce1-wb-wired-near_single-{suffix}-conversation-v2",
                record[0],
            )
            for suffix, record in sorted(fixture.HOLDOUT_REFERENCES.items())
        ],
    ]
    for artifact_id, role, source_case_id, reference_sha in reference_specs:
        source_path = fixture.DATASET_AUDIO_DIR / f"{reference_sha}.wav"
        relative_path = f"references/{reference_sha}.wav"
        record = _artifact_record(
            artifact_id,
            "reference",
            role,
            source_case_id,
            source_path,
            relative_path,
            reference_sha,
            fixture.REFERENCE_BYTES,
        )
        artifacts.append(record)
        copies.append((source_path, relative_path, fixture.REFERENCE_BYTES))

    for suffix, (reference_sha, _, _) in sorted(fixture.HOLDOUT_REFERENCES.items()):
        source_case_id = f"sce1-wb-wired-near_single-{suffix}-conversation-v2"
        source_path = (
            fixture.HOLDOUT_BATCH_DIR
            / "cases"
            / source_case_id
            / "assessment.json"
        )
        relative_path = f"assessments/{source_case_id}/assessment.json"
        record = _artifact_record(
            f"assessment:{suffix}",
            "assessment",
            "holdout",
            source_case_id,
            source_path,
            relative_path,
            fixture.ASSESSMENT_SHA256[suffix],
            fixture.ASSESSMENT_BYTES[suffix],
        )
        assessment = _decode_json_file(source_path, f"assessment {suffix}")
        if not isinstance(assessment, Mapping):
            raise PreparationError(f"assessment {suffix} must be a JSON object")
        if assessment.get("case_id") != source_case_id:
            raise PreparationError(f"assessment {suffix} case_id changed")
        canonical = fixture.DATASET_AUDIO_DIR / f"{reference_sha}.wav"
        record["reference_binding"] = fixture.assessment_reference_binding(
            assessment,
            reference_sha,
            canonical,
        )
        artifacts.append(record)
        copies.append((source_path, relative_path, fixture.ASSESSMENT_BYTES[suffix]))

    manifest = {
        "schema": fixture.SCHEMA,
        "fixture_id": fixture.FIXTURE_ID,
        "upstream_script": {
            "path": fixture.UPSTREAM_SCRIPT_RELATIVE,
            "source_path": str(upstream_path.absolute()),
            "sha256": fixture.UPSTREAM_SCRIPT_SHA256,
            "bytes": fixture.UPSTREAM_SCRIPT_BYTES,
        },
        "source_store": {
            "dataset_id": fixture.DATASET_ID,
            "dataset_audio_dir": str(fixture.DATASET_AUDIO_DIR.absolute()),
            "holdout_batch_id": fixture.HOLDOUT_BATCH_ID,
            "holdout_batch_dir": str(fixture.HOLDOUT_BATCH_DIR.absolute()),
        },
        "dev_manifest_origin": _dev_manifest_origin(),
        "protocol": {
            "rate": fixture.RATE,
            "duration_seconds": fixture.DURATION_SECONDS,
            "samples": fixture.SAMPLES,
            "hop": fixture.HOP,
            "frame": fixture.FRAME,
            "dev_reference_sha256": fixture.DEV_REFERENCE_SHA256,
            "holdout_references": {
                suffix: [record[0], list(record[1]), list(record[2])]
                for suffix, record in sorted(fixture.HOLDOUT_REFERENCES.items())
            },
            "cases": [dict(case) for case in fixture.KNOWN_CASE_PROTOCOLS],
        },
        "artifacts": artifacts,
    }
    return manifest, copies


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _assert_destination_parent(path: Path, fixture_root: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise PreparationError(f"fixture destination parent must not be a symlink: {parent}")
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(fixture_root.resolve(strict=True))
    except ValueError as error:
        raise PreparationError(
            f"fixture destination escapes fixture root: {path}"
        ) from error


def _write_exclusive_or_verify(
    destination: Path,
    data: bytes,
    fixture_root: Path,
) -> str:
    if destination.is_symlink():
        raise PreparationError(f"fixture destination must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise PreparationError(f"fixture destination must be a regular file: {destination}")
        existing = destination.read_bytes()
        if existing != data:
            raise PreparationError(
                f"existing fixture file differs and will not be overwritten: {destination}"
            )
        return "verified"

    _assert_destination_parent(destination, fixture_root)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial file is evidence of the failed write and is intentionally retained.
        raise
    if destination.read_bytes() != data:
        raise PreparationError(f"new fixture file verification failed: {destination}")
    return "created"


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            first_chunk = first.read(1024 * 1024)
            second_chunk = second.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def _copy_exclusive_or_verify(
    source: Path,
    destination: Path,
    expected_bytes: int,
    fixture_root: Path,
) -> str:
    source_resolved = _assert_source_file(source, f"fixture source {source}")
    if destination.is_symlink():
        raise PreparationError(f"fixture destination must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise PreparationError(f"fixture destination must be a regular file: {destination}")
        if destination.stat().st_size != expected_bytes:
            raise PreparationError(
                f"existing fixture file byte count differs: {destination}"
            )
        if not _same_bytes(source_resolved, destination):
            raise PreparationError(
                f"existing fixture file differs and will not be overwritten: {destination}"
            )
        return "verified"

    _assert_destination_parent(destination, fixture_root)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with source_resolved.open("rb") as source_stream, os.fdopen(
            descriptor, "wb"
        ) as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
    except BaseException:
        # A partial file is evidence of the failed write and is intentionally retained.
        raise
    if destination.stat().st_size != expected_bytes:
        raise PreparationError(f"copied fixture byte count changed: {destination}")
    if not _same_bytes(source_resolved, destination):
        raise PreparationError(f"copied fixture bytes changed: {destination}")
    return "created"


def _verified_result(
    fixture_root: Path,
    lock: fixture.FixtureLock,
    resource: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "status": "verified",
        "fixture_root": str(fixture_root),
        "manifest": {
            "path": str(lock.manifest_path),
            "status": "verified",
            "sha256": lock.manifest_sha256,
            "bytes": lock.manifest_bytes,
        },
        "artifacts": [
            {
                "relative_path": artifact.relative_path,
                "status": "verified",
                "bytes": artifact.bytes,
            }
            for artifact in lock.artifacts
        ],
        "resource": dict(resource),
    }


def prepare_fixture(
    fixture_root: Path = fixture.FIXTURE_ROOT,
    check_only: bool = False,
) -> dict[str, Any]:
    resource = assert_resource_floor()
    fixture_root = Path(fixture_root).expanduser().absolute()
    manifest_path = fixture_root / "manifest.json"

    if manifest_path.is_symlink():
        raise PreparationError(f"fixture manifest must not be a symlink: {manifest_path}")
    if manifest_path.exists():
        try:
            existing_lock = fixture.freeze_source_lock(manifest_path)
        except fixture.FixtureError:
            if check_only:
                raise
        else:
            return _verified_result(fixture_root, existing_lock, resource)

    if check_only:
        raise PreparationError(f"fixture manifest is missing or invalid: {manifest_path}")

    if fixture_root.is_symlink():
        raise PreparationError(f"fixture root must not be a symlink: {fixture_root}")
    fixture_root.mkdir(parents=True, exist_ok=True)
    if not fixture_root.is_dir():
        raise PreparationError(f"fixture root must be a directory: {fixture_root}")

    manifest, copies = _build_manifest()
    manifest_data = _manifest_bytes(manifest)
    statuses: list[dict[str, Any]] = []
    for source, relative_path, expected_bytes in copies:
        destination = fixture_root / Path(relative_path)
        status = _copy_exclusive_or_verify(
            source,
            destination,
            expected_bytes,
            fixture_root,
        )
        statuses.append(
            {
                "relative_path": relative_path,
                "status": status,
                "bytes": expected_bytes,
            }
        )
    manifest_status = _write_exclusive_or_verify(
        manifest_path,
        manifest_data,
        fixture_root,
    )
    lock = fixture.freeze_source_lock(manifest_path)
    return {
        "status": "prepared_or_verified",
        "fixture_root": str(fixture_root),
        "manifest": {
            "path": str(manifest_path),
            "status": manifest_status,
            "sha256": lock.manifest_sha256,
            "bytes": lock.manifest_bytes,
        },
        "artifacts": statuses,
        "resource": resource,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=fixture.FIXTURE_ROOT,
        help="fixture destination root (default: benchmarks/fixtures/omlsa-tuning-v1)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the existing fixture without creating any file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_fixture(args.fixture_root, check_only=args.check_only)
    except Exception as error:
        print(
            f"prepare-omlsa-tuning-fixture: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
