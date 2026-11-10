"""Validated storage for batch acoustic acceptance datasets."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
import errno
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import uuid
import zipfile

import numpy as np
import soundfile as sf

from . import service
from .acceptance_contract import contract_items, required_conditions
from .acceptance_visualization import precompute_visualizations
from .receive_configuration import (
    ReceiveConfigurationError,
    normalize_receive_configuration,
    safe_receive_ir_path,
)
from .receive_simulation import IR_SAMPLE_RATES


STORE_NAME = "acceptance-datasets"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ASSET_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
SIGNAL_NAMES = ("microphone", "reference", "far_end", "noise")
IR_ROLES = ("left_ir", "right_ir")
MAX_IR_DURATION_S = 10.0
CONDITION_KEYS = (
    "mode",
    "direction",
    "scene",
    "volume",
    "window_ms",
    "input_level_db",
)
_DATASET_ID = re.compile(r"^[0-9a-f]{32}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:")
_LINEAR_WAV_SUBTYPES = frozenset({
    "PCM_S8",
    "PCM_U8",
    "PCM_16",
    "PCM_24",
    "PCM_32",
    "FLOAT",
    "DOUBLE",
})


class AcceptanceDatasetError(ValueError):
    """An acceptance dataset archive or stored dataset failed validation."""


class _DatasetReimportRequired(AcceptanceDatasetError):
    """A stored dataset predates the assets index integrity binding."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _store() -> Path:
    return Path(service.BUILD) / STORE_NAME


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_dataset_id(dataset_id: str) -> bool:
    return isinstance(dataset_id, str) and bool(_DATASET_ID.fullmatch(dataset_id))


def _safe_relative_path(value: Any, label: str, *, wav: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceDatasetError(f"{label} must be a non-empty relative path")
    if "\x00" in value:
        raise AcceptanceDatasetError(f"{label} contains a NUL byte")
    if "\\" in value:
        raise AcceptanceDatasetError(f"{label} must not contain backslashes")
    if value.startswith("/") or _WINDOWS_ABSOLUTE.match(value):
        raise AcceptanceDatasetError(f"{label} must not be absolute")

    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise AcceptanceDatasetError(
            f"{label} must not contain empty, current, or parent components"
        )
    if any(any(ord(character) < 32 for character in part) for part in parts):
        raise AcceptanceDatasetError(f"{label} contains a control character")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise AcceptanceDatasetError(f"{label} must not be absolute")
    if wav and path.suffix.lower() != ".wav":
        raise AcceptanceDatasetError(f"{label} must name a .wav file")
    return path.as_posix()


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceDatasetError(f"{label} must be a non-empty string")
    return value


def _validate_case_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CASE_ID.fullmatch(value) is None:
        raise AcceptanceDatasetError(
            f"{label} must match ^[A-Za-z0-9_-]{{1,128}}$"
        )
    return value


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceDatasetError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AcceptanceDatasetError(f"{label} must be finite")
    if nonnegative and numeric < 0:
        raise AcceptanceDatasetError(f"{label} must be non-negative")
    return numeric


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceDatasetError(f"manifest contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AcceptanceDatasetError(f"manifest contains non-finite number {value}")


def _load_manifest_json(blob: bytes) -> dict[str, Any]:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcceptanceDatasetError("manifest.json must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except AcceptanceDatasetError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AcceptanceDatasetError(f"manifest.json is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise AcceptanceDatasetError("manifest.json must contain a JSON object")
    return value


def _validate_conditions(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(CONDITION_KEYS):
        raise AcceptanceDatasetError(
            f"{label} must contain exactly {', '.join(CONDITION_KEYS)}"
        )
    for key in ("mode", "direction", "scene", "volume"):
        _string(value[key], f"{label}.{key}")
    window = value["window_ms"]
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, int) or window not in {
            50,
            100,
        }:
            raise AcceptanceDatasetError(
                f"{label}.window_ms must be null, 50, or 100"
            )
    level = value["input_level_db"]
    if isinstance(level, bool) or not isinstance(level, int) or level not in {
        -5,
        0,
        5,
    }:
        raise AcceptanceDatasetError(
            f"{label}.input_level_db must be -5, 0, or 5"
        )
    return {key: value[key] for key in CONDITION_KEYS}


def _validate_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict], dict[str, str]]:
    allowed_top_level = {"schema_version", "name", "cases", "receive_simulation"}
    if not set(manifest).issubset(allowed_top_level):
        raise AcceptanceDatasetError(
            "manifest contains unsupported top-level fields"
        )
    if not {"schema_version", "name", "cases"}.issubset(manifest):
        raise AcceptanceDatasetError(
            "manifest must contain schema_version, name, and cases"
        )
    if (
        isinstance(manifest["schema_version"], bool)
        or manifest["schema_version"] != 1
    ):
        raise AcceptanceDatasetError("manifest.schema_version must be 1")
    _string(manifest["name"], "manifest.name")
    if "receive_simulation" in manifest:
        try:
            manifest["receive_simulation"] = normalize_receive_configuration(
                manifest["receive_simulation"]
            )
        except ReceiveConfigurationError as error:
            raise AcceptanceDatasetError(
                f"manifest.receive_simulation: {error}"
            ) from error
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise AcceptanceDatasetError("manifest.cases must be a non-empty array")
    receive_simulation = normalize_receive_configuration(
        manifest.get("receive_simulation")
    )

    rows = contract_items()
    rows_by_id = {row["id"]: row for row in rows}
    if len(rows) != 336 or len(rows_by_id) != 336:
        raise AcceptanceDatasetError(
            "acceptance contract must contain exactly 336 unique items"
        )

    case_ids: set[str] = set()
    seen_items: set[str] = set()
    duplicate_items: set[str] = set()
    unknown_items: set[str] = set()
    referenced_assets: set[str] = set()

    for index, case in enumerate(cases):
        label = f"manifest.cases[{index}]"
        if not isinstance(case, dict):
            raise AcceptanceDatasetError(f"{label} must be an object")
        allowed = {"id", "conditions", "item_ids", "signals"}
        if not set(case).issubset(allowed | {"beam_delays_ms", "analysis_start_s"}):
            raise AcceptanceDatasetError(f"{label} contains unsupported fields")
        if not {"id", "conditions", "item_ids", "signals"}.issubset(case):
            raise AcceptanceDatasetError(
                f"{label} must contain id, conditions, item_ids, and signals"
            )

        case_id = _validate_case_id(case["id"], f"{label}.id")
        if case_id in case_ids:
            raise AcceptanceDatasetError(f"duplicate case id {case_id!r}")
        case_ids.add(case_id)

        conditions = _validate_conditions(case["conditions"], f"{label}.conditions")
        item_ids = case["item_ids"]
        if not isinstance(item_ids, list) or not item_ids:
            raise AcceptanceDatasetError(f"{label}.item_ids must be a non-empty array")
        for item_index, item_id in enumerate(item_ids):
            item_label = f"{label}.item_ids[{item_index}]"
            item_id = _string(item_id, item_label)
            if item_id not in rows_by_id:
                unknown_items.add(item_id)
                continue
            if item_id in seen_items:
                duplicate_items.add(item_id)
            else:
                seen_items.add(item_id)
            expected = required_conditions(rows_by_id[item_id])
            if conditions != expected:
                raise AcceptanceDatasetError(
                    f"{item_label} conditions do not match contract item {item_id}"
                )

        signals = case["signals"]
        if not isinstance(signals, dict) or set(signals) != set(SIGNAL_NAMES):
            raise AcceptanceDatasetError(
                f"{label}.signals must contain exactly {', '.join(SIGNAL_NAMES)}"
            )
        for signal_name in SIGNAL_NAMES:
            referenced_assets.add(
                _safe_relative_path(
                    signals[signal_name],
                    f"{label}.signals.{signal_name}",
                    wav=True,
                )
            )
        if receive_simulation["model"] == "paired_ir":
            for role in IR_ROLES:
                referenced_assets.add(
                    safe_receive_ir_path(
                        receive_simulation[role],
                        f"manifest.receive_simulation.{role}",
                    )
                )

        if "beam_delays_ms" in case:
            delays = case["beam_delays_ms"]
            if not isinstance(delays, list):
                raise AcceptanceDatasetError(f"{label}.beam_delays_ms must be an array")
            for delay_index, delay in enumerate(delays):
                _number(
                    delay,
                    f"{label}.beam_delays_ms[{delay_index}]",
                )
        if "analysis_start_s" in case:
            _number(
                case["analysis_start_s"],
                f"{label}.analysis_start_s",
                nonnegative=True,
            )

    if duplicate_items:
        raise AcceptanceDatasetError(
            "manifest item_ids contain duplicates: "
            + ", ".join(sorted(duplicate_items))
        )
    if unknown_items:
        raise AcceptanceDatasetError(
            "manifest item_ids contain unknown ids: "
            + ", ".join(sorted(unknown_items))
        )
    missing = set(rows_by_id) - seen_items
    if missing:
        raise AcceptanceDatasetError(
            "manifest is missing contract item_ids: " + ", ".join(sorted(missing))
        )
    return rows_by_id, {path: path for path in referenced_assets}


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def _validate_zip_members(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, zipfile.ZipInfo], zipfile.ZipInfo]:
    if len(archive.infolist()) > MAX_ARCHIVE_MEMBERS:
        raise AcceptanceDatasetError("ZIP contains too many members")

    files: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    total_bytes = 0
    for info in archive.infolist():
        if _is_symlink_member(info):
            raise AcceptanceDatasetError(
                f"ZIP symlink entries are not allowed: {info.filename!r}"
            )
        if info.flag_bits & 0x1:
            raise AcceptanceDatasetError(
                f"encrypted ZIP entries are not allowed: {info.filename!r}"
            )
        if info.create_system == 3:
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise AcceptanceDatasetError(
                    f"special ZIP entry is not allowed: {info.filename!r}"
                )

        directory = info.is_dir()
        raw_name = info.filename[:-1] if directory else info.filename
        normalized = _safe_relative_path(raw_name, "ZIP member")
        if normalized in seen:
            raise AcceptanceDatasetError(
                f"duplicate ZIP member name: {normalized!r}"
            )
        seen.add(normalized)
        if directory:
            continue
        if info.file_size < 0 or info.file_size > MAX_ASSET_BYTES:
            raise AcceptanceDatasetError(
                f"ZIP member exceeds 32 MiB: {normalized!r}"
            )
        total_bytes += info.file_size
        if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise AcceptanceDatasetError(
                "ZIP uncompressed payload exceeds 512 MiB"
            )
        files[normalized] = info

    if "manifest.json" not in files:
        raise AcceptanceDatasetError("ZIP must contain manifest.json at its root")
    return files, files["manifest.json"]


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    destination: Path | None = None,
) -> bytes:
    digest = hashlib.sha256()
    output = BytesIO() if destination is None else None
    count = 0
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive.open(info, "r") as source:
            target = destination.open("xb") if destination is not None else None
            try:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > MAX_ASSET_BYTES:
                        raise AcceptanceDatasetError(
                            f"ZIP member exceeds 32 MiB: {info.filename!r}"
                        )
                    digest.update(chunk)
                    if output is not None:
                        output.write(chunk)
                    elif target is not None:
                        target.write(chunk)
            finally:
                if target is not None:
                    target.close()
    except AcceptanceDatasetError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise
        raise AcceptanceDatasetError(
            f"cannot read ZIP member {info.filename!r}: {error}"
        ) from error
    if count != info.file_size:
        raise AcceptanceDatasetError(
            f"ZIP member size changed while reading: {info.filename!r}"
        )
    return output.getvalue() if output is not None else b""


def _wav_metadata(
    path: Path,
    relative: str,
    roles: set[str],
) -> dict[str, Any]:
    try:
        with sf.SoundFile(path) as audio:
            if audio.format != "WAV":
                raise AcceptanceDatasetError(f"{relative} must use the WAV container")
            channels = int(audio.channels)
            sample_rate = int(audio.samplerate)
            frames = int(audio.frames)
            subtype = audio.subtype
            is_ir = bool(roles.intersection(IR_ROLES))
            if "microphone" in roles:
                if not 1 <= channels <= 8:
                    raise AcceptanceDatasetError(
                        f"{relative} microphone must have 1..8 channels"
                    )
            if roles - {"microphone"} and channels != 1:
                raise AcceptanceDatasetError(
                    f"{relative} reference/far_end/noise must be mono"
                )
            if sample_rate <= 0 or frames <= 0:
                raise AcceptanceDatasetError(
                    f"{relative} must have a positive sample rate and duration"
                )
            if is_ir:
                if subtype not in _LINEAR_WAV_SUBTYPES:
                    raise AcceptanceDatasetError(
                        f"{relative} IR must use uncompressed linear PCM or "
                        "IEEE-float WAV samples"
                    )
                if sample_rate not in IR_SAMPLE_RATES:
                    raise AcceptanceDatasetError(
                        f"{relative} IR sample rate must be a supported common "
                        "rate between 8000 and 192000 Hz"
                    )
                if frames / sample_rate > MAX_IR_DURATION_S:
                    raise AcceptanceDatasetError(
                        f"{relative} IR must not exceed 10 seconds"
                    )
            has_nonzero = False
            while True:
                block = audio.read(65_536, dtype="float64", always_2d=True)
                if not len(block):
                    break
                if not np.isfinite(block).all():
                    raise AcceptanceDatasetError(
                        f"{relative} contains non-finite samples"
                    )
                if is_ir and np.any(block != 0.0):
                    has_nonzero = True
            if is_ir and not has_nonzero:
                raise AcceptanceDatasetError(
                    f"{relative} IR must contain nonzero samples"
                )
    except AcceptanceDatasetError:
        raise
    except (OSError, RuntimeError) as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise
        raise AcceptanceDatasetError(f"cannot decode WAV {relative}: {error}") from error
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "sample_rate": sample_rate,
        "duration_s": round(frames / sample_rate, 9),
        "frames": frames,
        "channels": channels,
        "subtype": subtype,
        "size_bytes": path.stat().st_size,
    }


def _validate_audio(
    assets_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    references: dict[str, set[str]] = {}
    for case in manifest["cases"]:
        for role in SIGNAL_NAMES:
            relative = case["signals"][role]
            references.setdefault(relative, set()).add(role)
    receive_simulation = normalize_receive_configuration(
        manifest.get("receive_simulation")
    )
    if receive_simulation["model"] == "paired_ir":
        for role in IR_ROLES:
            relative = receive_simulation[role]
            references.setdefault(relative, set()).add(role)

    metadata: dict[str, dict[str, Any]] = {}
    for relative, roles in sorted(references.items()):
        path = assets_root / relative
        if not path.is_file() or path.is_symlink():
            raise AcceptanceDatasetError(f"missing regular WAV asset: {relative}")
        metadata[relative] = _wav_metadata(path, relative, roles)

    for index, case in enumerate(manifest["cases"]):
        signals = case["signals"]
        microphone = metadata[signals["microphone"]]
        reference = metadata[signals["reference"]]
        far_end = metadata[signals["far_end"]]
        noise = metadata[signals["noise"]]
        timeline = (microphone, reference, far_end)
        rates = {row["sample_rate"] for row in (*timeline, noise)}
        if len(rates) != 1:
            raise AcceptanceDatasetError(
                f"manifest.cases[{index}] signal sample rates do not match"
            )
        if len({row["frames"] for row in timeline}) != 1:
            raise AcceptanceDatasetError(
                f"manifest.cases[{index}] microphone/reference/far_end timelines differ"
            )
        start = case.get("analysis_start_s", 0)
        if float(start) >= microphone["duration_s"]:
            raise AcceptanceDatasetError(
                f"manifest.cases[{index}].analysis_start_s is outside the timeline"
            )
    return metadata


def _referenced_assets(manifest: dict[str, Any]) -> set[str]:
    referenced = {
        signal
        for case in manifest["cases"]
        for signal in case["signals"].values()
    }
    receive_simulation = normalize_receive_configuration(
        manifest.get("receive_simulation")
    )
    if receive_simulation["model"] == "paired_ir":
        referenced.update(receive_simulation[role] for role in IR_ROLES)
    return referenced


def _extract_archive(
    archive_bytes: bytes,
    temporary: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except (OSError, zipfile.BadZipFile) as error:
        raise AcceptanceDatasetError(f"invalid ZIP archive: {error}") from error

    with archive:
        files, manifest_info = _validate_zip_members(archive)
        manifest_blob = _read_member(archive, manifest_info)
        manifest = _load_manifest_json(manifest_blob)
        _validate_manifest(manifest)

        referenced = _referenced_assets(manifest)
        asset_names = set(files) - {"manifest.json"}
        extra = asset_names - referenced
        if extra:
            raise AcceptanceDatasetError(
                "ZIP contains unreferenced assets: " + ", ".join(sorted(extra))
            )
        missing = referenced - asset_names
        if missing:
            raise AcceptanceDatasetError(
                "ZIP is missing referenced assets: " + ", ".join(sorted(missing))
            )

        assets_root = temporary / "assets"
        assets_root.mkdir()
        actual_total = 0
        for relative in sorted(referenced):
            destination = assets_root / relative
            _read_member(
                archive,
                files[relative],
                destination=destination,
            )
            actual_total += destination.stat().st_size
            if actual_total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise AcceptanceDatasetError(
                    "ZIP uncompressed payload exceeds 512 MiB"
                )
    metadata = _validate_audio(assets_root, manifest)
    return manifest, metadata


def _summary(
    *,
    dataset_id: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    assets_sha256: str,
    created_at: str,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_items = {row["id"] for row in contract_items()}
    covered_items = {
        item_id for case in manifest["cases"] for item_id in case["item_ids"]
    }
    missing = sorted(expected_items - covered_items)
    unknown = sorted(covered_items - expected_items)
    duplicate_count = sum(
        len(case["item_ids"]) for case in manifest["cases"]
    ) - len(covered_items)
    complete = not missing and not unknown and duplicate_count == 0
    receive_simulation = normalize_receive_configuration(
        manifest.get("receive_simulation")
    )
    return {
        "dataset_id": dataset_id,
        "name": manifest["name"],
        "receive_simulation": receive_simulation,
        "case_count": len(manifest["cases"]),
        "item_count": len(covered_items),
        "manifest_sha256": manifest_sha256,
        "assets_sha256": assets_sha256,
        "created_at": created_at,
        "asset_count": len(assets),
        "total_bytes": sum(row["size_bytes"] for row in assets.values()),
        "integrity": {
            "complete": complete,
            "expected_item_count": len(expected_items),
            "covered_item_count": len(covered_items),
            "missing_item_ids": missing,
            "unknown_item_ids": unknown,
            "duplicate_item_count": duplicate_count,
            "all_assets_hashed": all(
                re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is not None
                for row in assets.values()
            ),
        },
    }


def _asset_index(assets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = ("sha256", "sample_rate", "frames", "channels", "duration_s")
    return {
        relative: {key: row[key] for key in keys}
        for relative, row in sorted(assets.items())
    }


def template_manifest() -> dict:
    """Return the canonical 336-item import template grouped by conditions."""
    grouped: OrderedDict[tuple[tuple[str, Any], ...], dict[str, Any]] = OrderedDict()
    for row in contract_items():
        conditions = required_conditions(row)
        key = tuple(conditions.items())
        group = grouped.setdefault(
            key,
            {"conditions": conditions, "item_ids": []},
        )
        group["item_ids"].append(row["id"])

    cases = []
    for index, group in enumerate(grouped.values(), start=1):
        case_id = f"case-{index:04d}"
        cases.append(
            {
                "id": case_id,
                "conditions": group["conditions"],
                "item_ids": group["item_ids"],
                "signals": {
                    role: f"audio/{case_id}/{role}.wav" for role in SIGNAL_NAMES
                },
            }
        )
    return {"schema_version": 1, "name": "NV200 acceptance dataset", "cases": cases}


def import_dataset(archive: bytes) -> dict:
    """Validate and atomically import one acceptance dataset ZIP archive."""
    if not isinstance(archive, (bytes, bytearray, memoryview)):
        raise AcceptanceDatasetError("archive must be bytes")
    archive_bytes = bytes(archive)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise AcceptanceDatasetError("ZIP archive exceeds 256 MiB")

    dataset_id = uuid.uuid4().hex
    created_at = _now_iso()
    store = _store()
    store.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".import-{dataset_id}-", dir=store)
    )
    try:
        manifest, assets = _extract_archive(archive_bytes, temporary)
        precompute_visualizations(temporary / "assets", assets)
        manifest_blob = _canonical_json_bytes(manifest)
        manifest_sha256 = _sha256_bytes(manifest_blob)
        asset_index = _asset_index(assets)
        asset_index_blob = _canonical_json_bytes(asset_index)
        assets_sha256 = _sha256_bytes(asset_index_blob)
        summary = _summary(
            dataset_id=dataset_id,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            assets_sha256=assets_sha256,
            created_at=created_at,
            assets=assets,
        )
        (temporary / "manifest.json").write_bytes(manifest_blob)
        (temporary / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "assets.json").write_bytes(asset_index_blob)

        destination = store / dataset_id
        if destination.exists() or destination.is_symlink():
            raise AcceptanceDatasetError("generated dataset id already exists")
        os.replace(temporary, destination)
    except Exception:
        # Preserve every failed staging directory for forensic recovery.
        raise

    return {
        "dataset_id": dataset_id,
        "name": manifest["name"],
        "case_count": len(manifest["cases"]),
        "item_count": summary["item_count"],
        "manifest_sha256": manifest_sha256,
        "created_at": created_at,
    }


def _dataset_directory(dataset_id: str) -> Path | None:
    if not _safe_dataset_id(dataset_id):
        return None
    directory = _store() / dataset_id
    if directory.is_symlink() or not directory.is_dir():
        return None
    return directory


def _load_summary(
    directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
] | None:
    summary_path = directory / "summary.json"
    manifest_path = directory / "manifest.json"
    assets_path = directory / "assets.json"
    if (
        summary_path.is_symlink()
        or manifest_path.is_symlink()
        or assets_path.is_symlink()
        or not summary_path.is_file()
        or not manifest_path.is_file()
        or not assets_path.is_file()
    ):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        manifest_blob = manifest_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict) or not isinstance(assets, dict):
        return None
    if summary.get("dataset_id") != directory.name:
        return None
    if "assets_sha256" not in summary:
        raise _DatasetReimportRequired(
            "dataset summary lacks assets_sha256; re-import the dataset"
        )
    assets_sha256 = summary["assets_sha256"]
    if (
        not isinstance(assets_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", assets_sha256) is None
    ):
        return None
    try:
        canonical_assets_sha256 = _sha256_bytes(_canonical_json_bytes(assets))
    except (TypeError, ValueError, UnicodeError):
        return None
    if assets_sha256 != canonical_assets_sha256:
        return None
    if summary.get("manifest_sha256") != _sha256_bytes(manifest_blob):
        return None
    try:
        manifest = _load_manifest_json(manifest_blob)
        _validate_manifest(manifest)
    except AcceptanceDatasetError:
        return None
    receive_simulation = normalize_receive_configuration(
        manifest.get("receive_simulation")
    )
    if "receive_simulation" not in summary:
        if "receive_simulation" in manifest:
            return None
        summary["receive_simulation"] = receive_simulation
    elif summary["receive_simulation"] != receive_simulation:
        return None
    integrity = summary.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("complete") is not True:
        return None
    if summary.get("asset_count") != len(assets):
        return None
    if (
        summary.get("name") != manifest["name"]
        or summary.get("case_count") != len(manifest["cases"])
        or summary.get("item_count") != 336
    ):
        return None
    referenced_assets = _referenced_assets(manifest)
    if set(assets) != referenced_assets:
        return None
    for relative, row in assets.items():
        if not isinstance(relative, str) or not isinstance(row, dict):
            return None
        try:
            normalized = _safe_relative_path(
                relative,
                "assets.json asset",
                wav=True,
            )
        except AcceptanceDatasetError:
            return None
        if normalized != relative:
            return None
        if set(row) != {
            "sha256",
            "sample_rate",
            "frames",
            "channels",
            "duration_s",
        }:
            return None
        digest = row["sha256"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
        ):
            return None
        for key in ("sample_rate", "frames", "channels"):
            value = row[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                return None
        duration = row["duration_s"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            return None
    return summary, assets, manifest


def get_dataset(dataset_id: str) -> dict | None:
    """Return a stored dataset summary, manifest, and asset metadata."""
    directory = _dataset_directory(dataset_id)
    if directory is None:
        return None
    loaded = _load_summary(directory)
    if loaded is None:
        return None
    summary, assets, manifest = loaded
    return {**summary, "manifest": manifest, "assets": assets}


def list_datasets() -> list:
    """List complete stored dataset summaries, newest first."""
    store = _store()
    if not store.is_dir() or store.is_symlink():
        return []
    summaries = []
    for directory in store.iterdir():
        if not _safe_dataset_id(directory.name):
            continue
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            loaded = _load_summary(directory)
        except _DatasetReimportRequired:
            continue
        if loaded is not None:
            summaries.append(loaded[0])
    summaries.sort(
        key=lambda row: (row.get("created_at", ""), row.get("dataset_id", "")),
        reverse=True,
    )
    return summaries


def dataset_asset(
    dataset_id: str,
    relative: str,
    *,
    assets: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Resolve one declared dataset asset without escaping its read-only store."""
    directory = _dataset_directory(dataset_id)
    if directory is None:
        raise AcceptanceDatasetError("unknown dataset")
    if assets is None:
        loaded = _load_summary(directory)
        if loaded is None:
            raise AcceptanceDatasetError("dataset integrity check failed")
        _, assets, _ = loaded
    normalized = _safe_relative_path(relative, "dataset asset")
    if normalized not in assets:
        raise AcceptanceDatasetError("asset is not declared by the dataset manifest")

    root = directory / "assets"
    if root.is_symlink() or not root.is_dir():
        raise AcceptanceDatasetError("dataset asset directory is invalid")
    candidate = root / normalized
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise AcceptanceDatasetError("dataset asset path contains a symlink")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AcceptanceDatasetError("dataset asset does not exist") from error
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise AcceptanceDatasetError("dataset asset escapes its store")
    return resolved


__all__ = [
    "AcceptanceDatasetError",
    "dataset_asset",
    "get_dataset",
    "import_dataset",
    "list_datasets",
    "template_manifest",
]
