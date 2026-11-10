"""Read-only store contract for the idle-page engineering preview samples.

This sidecar is deliberately outside the 336-item acceptance dataset contract.
It never creates batches, never reuses old TX reports, and never claims formal
ITU-T compliance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from threading import Lock
from typing import Any

from . import service


SCHEMA_VERSION = "scene-condition-preview-v1"
DATASET_ID = "edea6f9a51df87d0a4aefb2303dd8e1f"
DATASET_NAME = "background-noise-sce1-wb-wired-20-condition-preview-v1"
DATASET_KIND = "background_noise_idle_preview"
TEST_SCOPE = "background_noise"
MODE = "WB-wired"
SCENE = "sce1"
CONTRACT_ITEM_COUNT = 5
PREVIEW_CONDITION_GROUP_COUNT = 20
STANDARD_COMPLIANCE = False
FORMAL_TEST_STATUS = "not_measured"
ROLES = ("microphone", "reference", "far_end", "noise")
CONTRACT_ITEM_IDS = frozenset(
    {
        "WB-wired:B24",
        "WB-wired:B25",
        "WB-wired:B26",
        "WB-wired:B27",
        "WB-wired:B28",
    }
)
CONDITIONS = {
    "mode": MODE,
    "direction": "send",
    "scene": SCENE,
    "volume": "nominal",
    "window_ms": None,
    "input_level_db": 0,
}

_ID = re.compile(r"^[a-f0-9]{32}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_name",
        "dataset_kind",
        "test_scope",
        "mode",
        "scene",
        "contract_item_count",
        "preview_condition_group_count",
        "standard_compliance",
        "formal_test_status",
        "cases",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "conditions",
        "families",
        "contract_item_ids",
        "signals",
        "signals_sha256",
        "provenance",
    }
)
_FAMILIES = frozenset(
    {"g_mos", "n_mos", "s_mos", "cng_level", "cng_spectrum"}
)
_LOAD_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_LOAD_CACHE_LOCK = Lock()


class SceneConditionDatasetError(ValueError):
    """A scene-condition sidecar is missing, invalid, or unsafe to serve."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SceneConditionDatasetError(
                f"scene-condition JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SceneConditionDatasetError(
        f"scene-condition JSON contains non-finite number {value}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SceneConditionDatasetError(
            "scene-condition asset path must be a non-empty relative string"
        )
    if "\x00" in value or "\\" in value or value.startswith("/"):
        raise SceneConditionDatasetError(
            "scene-condition asset path contains an invalid character"
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SceneConditionDatasetError(
            "scene-condition asset path contains an unsafe component"
        )
    if any(any(ord(character) < 32 for character in part) for part in parts):
        raise SceneConditionDatasetError(
            "scene-condition asset path contains a control character"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or path.suffix.lower() != ".wav":
        raise SceneConditionDatasetError(
            "scene-condition asset path must name a relative WAV file"
        )
    return path.as_posix()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SceneConditionDatasetError(
            f"cannot inspect scene-condition file {path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise SceneConditionDatasetError(
            f"scene-condition file is not regular: {path}"
        )
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except SceneConditionDatasetError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SceneConditionDatasetError(
            f"cannot read scene-condition JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SceneConditionDatasetError(
            f"scene-condition JSON must be an object: {path}"
        )
    return value, payload


def dataset_root(dataset_id: str = DATASET_ID) -> Path:
    if not isinstance(dataset_id, str) or _ID.fullmatch(dataset_id) is None:
        raise SceneConditionDatasetError("scene-condition dataset id is invalid")
    return Path(service.BUILD) / "acoustic-scene-condition-datasets" / dataset_id


def asset_path(root: Path, relative: str) -> Path:
    normalized = _safe_relative_path(relative)
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / normalized).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise SceneConditionDatasetError(
            f"scene-condition asset escapes its store: {relative}"
        )
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise SceneConditionDatasetError(
            f"cannot inspect scene-condition asset {relative}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise SceneConditionDatasetError(
            f"scene-condition asset is not a regular file: {relative}"
        )
    return candidate


def _store_signature(root: Path) -> tuple[Any, ...]:
    signature: list[Any] = []
    for relative in ("manifest.json", "assets.json"):
        metadata = (root / relative).lstat()
        signature.extend(
            (relative, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        )
    for path in sorted(root.glob("assets/**/*.wav")):
        metadata = path.lstat()
        signature.extend(
            (
                path.relative_to(root).as_posix(),
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        )
    return tuple(signature)


def _load_scene_condition_dataset_uncached(
    dataset_id: str,
) -> dict[str, Any]:
    if dataset_id != DATASET_ID:
        raise SceneConditionDatasetError("unknown scene-condition dataset")
    root = dataset_root(dataset_id)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise SceneConditionDatasetError(
            f"scene-condition dataset store is missing: {root}"
        ) from error
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise SceneConditionDatasetError(
            "scene-condition dataset store is not a regular directory"
        )

    manifest, manifest_bytes = _read_json(resolved_root / "manifest.json")
    assets, assets_bytes = _read_json(resolved_root / "assets.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise SceneConditionDatasetError(
            "scene-condition manifest fields differ from the v1 contract"
        )
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "test_scope": TEST_SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "preview_condition_group_count": PREVIEW_CONDITION_GROUP_COUNT,
        "standard_compliance": STANDARD_COMPLIANCE,
        "formal_test_status": FORMAL_TEST_STATUS,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise SceneConditionDatasetError(
                f"scene-condition manifest field {key!r} is invalid"
            )

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) < PREVIEW_CONDITION_GROUP_COUNT:
        raise SceneConditionDatasetError(
            "scene-condition manifest must contain at least 20 preview cases"
        )
    if len(cases) != manifest["preview_condition_group_count"]:
        raise SceneConditionDatasetError(
            "scene-condition preview count does not match its cases"
        )
    if not isinstance(assets, dict):
        raise SceneConditionDatasetError(
            "scene-condition assets must be an object"
        )

    seen_case_ids: set[str] = set()
    covered_items: set[str] = set()
    referenced_assets: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != _CASE_FIELDS:
            raise SceneConditionDatasetError(
                "scene-condition case fields differ from the v1 contract"
            )
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise SceneConditionDatasetError(
                "scene-condition case ids must be unique non-empty strings"
            )
        seen_case_ids.add(case_id)
        if case.get("conditions") != CONDITIONS:
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has unexpected conditions"
            )
        families = case.get("families")
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(value, str) or value not in _FAMILIES for value in families)
            or len(set(families)) != len(families)
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has invalid families"
            )
        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or not item_ids
            or any(value not in CONTRACT_ITEM_IDS for value in item_ids)
            or len(set(item_ids)) != len(item_ids)
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has invalid contract item ids"
            )
        covered_items.update(item_ids)
        signals = case.get("signals")
        hashes = case.get("signals_sha256")
        if (
            not isinstance(signals, dict)
            or set(signals) != set(ROLES)
            or not isinstance(hashes, dict)
            or set(hashes) != set(ROLES)
            or not isinstance(case.get("provenance"), dict)
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has incomplete signals or provenance"
            )
        for role in ROLES:
            relative = signals[role]
            expected_sha = hashes[role]
            if (
                not isinstance(expected_sha, str)
                or _SHA256.fullmatch(expected_sha) is None
            ):
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} has an invalid {role} hash"
                )
            normalized = _safe_relative_path(relative)
            referenced_assets.add(normalized)
            metadata = assets.get(normalized)
            if not isinstance(metadata, dict):
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} is missing"
                )
            if metadata.get("sha256") != expected_sha:
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} hash differs"
                )
            path = asset_path(resolved_root, normalized)
            try:
                size_bytes = int(metadata.get("size_bytes"))
            except (TypeError, ValueError) as error:
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} size is invalid"
                ) from error
            if size_bytes != path.stat().st_size:
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} size differs"
                )
            if _sha256_file(path) != expected_sha:
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} content differs"
                )
    if covered_items != CONTRACT_ITEM_IDS:
        raise SceneConditionDatasetError(
            "scene-condition cases do not cover the five Sce1 contract items"
        )
    if set(assets) != referenced_assets:
        raise SceneConditionDatasetError(
            "scene-condition assets contain unreferenced or missing paths"
        )

    return {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "root": resolved_root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_assets_sha256": hashlib.sha256(assets_bytes).hexdigest(),
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "preview_condition_group_count": len(cases),
        "referenced_assets": referenced_assets,
    }


def load_scene_condition_dataset(dataset_id: str = DATASET_ID) -> dict[str, Any]:
    if dataset_id != DATASET_ID:
        raise SceneConditionDatasetError("unknown scene-condition dataset")
    root = dataset_root(dataset_id)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise SceneConditionDatasetError(
            f"scene-condition dataset store is missing: {root}"
        ) from error
    try:
        signature = _store_signature(resolved_root)
    except OSError as error:
        raise SceneConditionDatasetError(
            f"cannot inspect scene-condition dataset store: {error}"
        ) from error
    cache_key = str(resolved_root)
    with _LOAD_CACHE_LOCK:
        cached = _LOAD_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
    value = _load_scene_condition_dataset_uncached(dataset_id)
    with _LOAD_CACHE_LOCK:
        _LOAD_CACHE[cache_key] = (signature, dict(value))
    return value


__all__ = [
    "CONDITIONS",
    "CONTRACT_ITEM_COUNT",
    "CONTRACT_ITEM_IDS",
    "DATASET_ID",
    "DATASET_KIND",
    "DATASET_NAME",
    "FORMAL_TEST_STATUS",
    "MODE",
    "PREVIEW_CONDITION_GROUP_COUNT",
    "ROLES",
    "SCENE",
    "SCHEMA_VERSION",
    "STANDARD_COMPLIANCE",
    "TEST_SCOPE",
    "SceneConditionDatasetError",
    "asset_path",
    "dataset_root",
    "load_scene_condition_dataset",
]
