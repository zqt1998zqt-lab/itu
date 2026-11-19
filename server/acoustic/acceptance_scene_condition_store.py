"""Validated read-only store for all four scene-condition datasets."""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from . import service
from . import acceptance_scene_conditions as legacy
from .acceptance_scope import background_noise_scene_items
from .conversation_diagnostics import (
    EXPECTED_RENDER_KIND,
    validate_activity_metadata,
)
from .scene_mode_conversation_profiles import (
    profile_for_dataset,
    profile_counts,
    profile_mode,
    profile_sample_rate_hz,
)


SceneConditionDatasetError = legacy.SceneConditionDatasetError
asset_path = legacy.asset_path
dataset_root = legacy.dataset_root

_LOAD_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_LOAD_CACHE_LOCK = Lock()


def _schema_version(profile: Any) -> str:
    return str(getattr(profile, "schema_version", legacy.SCHEMA_VERSION))


def _counts(profile: Any) -> tuple[int, int, int, int]:
    return profile_counts(profile)


def _conversation_counts(profile: Any) -> dict[str, int] | None:
    value = getattr(profile, "conversation_counts", None)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SceneConditionDatasetError(
            "scene-condition profile conversation counts are invalid"
        )
    return {str(key): int(count) for key, count in value.items()}


def _expected_manifest_identity(profile: Any) -> dict[str, Any]:
    case_count, _, _, _ = _counts(profile)
    return {
        "schema_version": _schema_version(profile),
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "test_scope": legacy.TEST_SCOPE,
        "mode": profile_mode(profile),
        "scene": profile.scene,
        "contract_item_count": profile.contract_item_count,
        "preview_condition_group_count": case_count,
        "standard_compliance": legacy.STANDARD_COMPLIANCE,
        "formal_test_status": legacy.FORMAL_TEST_STATUS,
    }


def _family_by_item(profile: Any) -> dict[str, str]:
    rows = background_noise_scene_items(profile_mode(profile), profile.scene)
    return {row["id"]: row["family"] for row in rows}


def _manifest_fields(profile: Any) -> set[str]:
    fields = set(legacy._MANIFEST_FIELDS)
    if _schema_version(profile) == "scene-condition-preview-v2":
        fields.update(
            {
                "conversation_counts",
                "case_count",
                "speech_case_count",
                "cng_case_count",
                "observation_count",
            }
        )
    return fields


def _validate_optional_count_fields(
    manifest: dict[str, Any],
    profile: Any,
) -> None:
    case_count, speech_count, cng_count, observation_count = _counts(profile)
    expected = {
        "case_count": case_count,
        "speech_case_count": speech_count,
        "cng_case_count": cng_count,
        "observation_count": observation_count,
    }
    for field, value in expected.items():
        if field in manifest and manifest[field] != value:
            raise SceneConditionDatasetError(
                f"scene-condition manifest field {field!r} is invalid"
            )
    expected_conversations = _conversation_counts(profile)
    if expected_conversations is not None and "conversation_counts" in manifest:
        if manifest["conversation_counts"] != expected_conversations:
            raise SceneConditionDatasetError(
                "scene-condition manifest conversation_counts is invalid"
            )


def _load_scene_condition_dataset_uncached(
    dataset_id: str,
) -> dict[str, Any]:
    profile = profile_for_dataset(dataset_id)
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

    manifest, manifest_bytes = legacy._read_json(
        resolved_root / "manifest.json"
    )
    assets, assets_bytes = legacy._read_json(resolved_root / "assets.json")
    allowed_manifest_fields = _manifest_fields(profile)
    base_manifest_fields = set(legacy._MANIFEST_FIELDS)
    if (
        not base_manifest_fields <= set(manifest)
        or not set(manifest) <= allowed_manifest_fields
    ):
        raise SceneConditionDatasetError(
            "scene-condition manifest fields differ from the profile contract"
        )
    for key, expected in _expected_manifest_identity(profile).items():
        if manifest.get(key) != expected:
            raise SceneConditionDatasetError(
                f"scene-condition manifest field {key!r} is invalid"
            )
    _validate_optional_count_fields(manifest, profile)

    case_count, speech_target, cng_target, observation_target = _counts(profile)
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != case_count
    ):
        raise SceneConditionDatasetError(
            f"scene-condition manifest must contain exactly {case_count} cases"
        )
    if not isinstance(assets, dict):
        raise SceneConditionDatasetError(
            "scene-condition assets must be an object"
        )

    expected_conditions = profile.conditions
    contract_item_ids = frozenset(profile.contract_item_ids)
    family_by_item = _family_by_item(profile)
    speech_item_ids = frozenset(
        item_id
        for item_id, family in family_by_item.items()
        if family in {"g_mos", "n_mos", "s_mos"}
    )
    cng_item_ids = frozenset(
        item_id
        for item_id, family in family_by_item.items()
        if family in {"cng_level", "cng_spectrum"}
    )

    seen_case_ids: set[str] = set()
    covered_items: set[str] = set()
    referenced_assets: set[str] = set()
    speech_case_count = 0
    cng_case_count = 0
    conversation_counts: dict[str, int] = {}
    expected_conversation_counts = _conversation_counts(profile)
    schema_version = _schema_version(profile)
    for case in cases:
        if not isinstance(case, dict) or set(case) != set(legacy._CASE_FIELDS):
            raise SceneConditionDatasetError(
                "scene-condition case fields differ from the profile contract"
            )
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_case_ids
        ):
            raise SceneConditionDatasetError(
                "scene-condition case ids must be unique non-empty strings"
            )
        seen_case_ids.add(case_id)
        if case.get("conditions") != expected_conditions:
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has unexpected conditions"
            )

        provenance = case.get("provenance")
        if not isinstance(provenance, dict):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has invalid provenance"
            )
        render_metadata = provenance.get("render_metadata")
        render_kind = (
            render_metadata.get("kind")
            if isinstance(render_metadata, dict)
            else None
        )
        conversation_mode = None
        if schema_version == "scene-condition-preview-v2":
            try:
                activity = validate_activity_metadata(
                    provenance,
                    expected_duration_s=float(
                        getattr(profile, "duration_s", 12.0)
                    ),
                    expected_sample_rate_hz=profile_sample_rate_hz(profile),
                )
            except ValueError as error:
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} activity metadata is invalid: {error}"
                ) from error
            conversation_mode = activity["conversation_mode"]
            conversation_counts[conversation_mode] = (
                conversation_counts.get(conversation_mode, 0) + 1
            )
            if render_kind != EXPECTED_RENDER_KIND[conversation_mode]:
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} kind/mode mismatch"
                )
            if (
                "signal_kind" in provenance
                and provenance["signal_kind"] != render_kind
            ):
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} signal_kind mismatch"
                )
            if (
                "scene" in provenance
                and provenance["scene"] != profile.scene
            ):
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} provenance scene mismatch"
                )
            if (
                isinstance(render_metadata, dict)
                and "direction" in render_metadata
                and render_metadata["direction"]
                != expected_conditions["direction"]
            ):
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} direction mismatch"
                )
        elif render_kind not in {"speech", "cng"}:
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} lacks a speech/CNG kind"
            )

        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or not item_ids
            or any(
                not isinstance(value, str)
                or value not in contract_item_ids
                for value in item_ids
            )
            or len(set(item_ids)) != len(item_ids)
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has invalid contract item ids"
            )
        item_id_set = set(item_ids)
        if item_id_set == set(speech_item_ids):
            speech_case_count += 1
            expected_kind = "speech"
        elif item_id_set == set(cng_item_ids):
            cng_case_count += 1
            expected_kind = "cng"
        else:
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} mixes speech and CNG rows"
            )
        if render_kind != expected_kind:
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} rows do not match render kind"
            )
        covered_items.update(item_ids)

        families = case.get("families")
        expected_families = {family_by_item[item_id] for item_id in item_ids}
        if (
            not isinstance(families, list)
            or not families
            or any(
                not isinstance(value, str)
                or value not in legacy._FAMILIES
                for value in families
            )
            or len(set(families)) != len(families)
            or set(families) != expected_families
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has invalid families"
            )

        signals = case.get("signals")
        hashes = case.get("signals_sha256")
        if (
            not isinstance(signals, dict)
            or set(signals) != set(legacy.ROLES)
            or not isinstance(hashes, dict)
            or set(hashes) != set(legacy.ROLES)
        ):
            raise SceneConditionDatasetError(
                f"scene-condition case {case_id!r} has incomplete signals or provenance"
            )
        for role in legacy.ROLES:
            relative = signals[role]
            expected_sha = hashes[role]
            if (
                not isinstance(expected_sha, str)
                or legacy._SHA256.fullmatch(expected_sha) is None
            ):
                raise SceneConditionDatasetError(
                    f"scene-condition case {case_id!r} has an invalid {role} hash"
                )
            normalized = legacy._safe_relative_path(relative)
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
            if legacy._sha256_file(path) != expected_sha:
                raise SceneConditionDatasetError(
                    f"scene-condition asset {normalized!r} content differs"
                )

    if covered_items != set(contract_item_ids):
        raise SceneConditionDatasetError(
            f"scene-condition cases do not cover the five {profile.scene} rows"
        )
    if speech_case_count != speech_target or cng_case_count != cng_target:
        raise SceneConditionDatasetError(
            "scene-condition cases must contain the profile speech/CNG split"
        )
    if expected_conversation_counts is not None:
        if conversation_counts != expected_conversation_counts:
            raise SceneConditionDatasetError(
                "scene-condition cases must contain the profile conversation split"
            )
    observation_count = speech_case_count * len(speech_item_ids) + (
        cng_case_count * len(cng_item_ids)
    )
    if observation_count != observation_target:
        raise SceneConditionDatasetError(
            "scene-condition observation count differs from the profile"
        )
    if set(assets) != referenced_assets:
        raise SceneConditionDatasetError(
            "scene-condition assets contain unreferenced or missing paths"
        )

    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": profile.dataset_kind,
        "schema_version": schema_version,
        "mode": profile_mode(profile),
        "scene": profile.scene,
        "root": resolved_root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_assets_sha256": hashlib.sha256(assets_bytes).hexdigest(),
        "contract_item_count": profile.contract_item_count,
        "contract_item_ids": profile.contract_item_ids,
        "preview_condition_group_count": len(cases),
        "preview_count": len(cases),
        "case_count": len(cases),
        "speech_case_count": speech_case_count,
        "cng_case_count": cng_case_count,
        "observation_count": observation_count,
        "conversation_counts": conversation_counts,
        "referenced_assets": referenced_assets,
    }


def load_scene_condition_dataset(dataset_id: str) -> dict[str, Any]:
    profile = profile_for_dataset(dataset_id)
    root = dataset_root(profile.dataset_id)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise SceneConditionDatasetError(
            f"scene-condition dataset store is missing: {root}"
        ) from error
    try:
        signature = legacy._store_signature(resolved_root)
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
    "SceneConditionDatasetError",
    "asset_path",
    "dataset_root",
    "load_scene_condition_dataset",
]
