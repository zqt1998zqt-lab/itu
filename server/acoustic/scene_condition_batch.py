"""Append-only VCP batches for profile-driven scene-condition datasets.

This module deliberately lives beside, rather than inside, the legacy
``acceptance_batch`` engine.  It reuses the legacy CLI runner and case metric
calculator, but gives every scene-condition case its own observation identity
so five contract rows can be measured across 20/50 or 100/280 profiles
without collapsing them.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any
import uuid

from fastapi import HTTPException

from . import acceptance_batch, service
from server.aec_routes import admit_aec_route, aec_route_requires_capability
from .acceptance_models import calculate_case_items
from .acceptance_scope import background_noise_scene_items
from .acceptance_scene_conditions import (
    CONDITIONS,
    CONTRACT_ITEM_COUNT,
    MODE,
    ROLES,
    SCENE,
    TEST_SCOPE,
    asset_path,
)
from .conversation_diagnostics import (
    EXPECTED_RENDER_KIND,
    diagnose_conversation_case,
    validate_activity_metadata,
    validate_diagnostics,
)
from .scene_mode_conversation_profiles import (
    profile_for_dataset,
    profile_for_scene,
    profile_for_scene_and_mode,
    profile_counts,
    profile_mode,
    profile_sample_rate_hz,
)
from .acoustic_stress_profiles import (
    is_stress_dataset,
    stress_profile_for_dataset,
)


SCHEMA = "scene-condition-batch-v1"
BATCH_ROOT_NAME = "scene-condition-batches"
FIXTURE_MARKER_NAME = "test-fixture.json"
FIXTURE_MARKER_SCHEMA = "scene-condition-test-fixture-v1"
FIXTURE_MARKER_REASON = "synthetic test fixture"
FIXTURE_MARKER_FIELDS = frozenset(
    {"schema", "batch_id", "reason", "snapshot_sha256"}
)
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3
# Compatibility exports for callers of the original v1 profile. Runtime
# validation always derives counts from ``profile_for_dataset`` instead.
EXPECTED_CASE_COUNT = 20
EXPECTED_SPEECH_CASE_COUNT = 10
EXPECTED_CNG_CASE_COUNT = 10
EXPECTED_OBSERVATION_COUNT = 50


def _profile_schema_version(profile: Any) -> str:
    return str(
        getattr(profile, "schema_version", "scene-condition-preview-v1")
    )


def _profile_counts(profile: Any) -> tuple[int, int, int, int]:
    return profile_counts(profile)


def _profile_conversation_counts(profile: Any) -> dict[str, int]:
    value = getattr(profile, "conversation_counts", None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("场景条件 profile conversation_counts 无效")
    return {str(key): int(count) for key, count in value.items()}


def _profile_contract_rows(profile: Any) -> dict[str, dict[str, Any]]:
    mode = profile_mode(profile)
    rows = background_noise_scene_items(mode, profile.scene)
    contract_by_id = {row["id"]: row for row in rows}
    if len(contract_by_id) != CONTRACT_ITEM_COUNT:
        raise ValueError(
            f"{profile.scene}/{mode} 合同项不是唯一的 5 项"
        )
    return contract_by_id


def _item_ids_by_kind(
    contract_by_id: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    speech = {
        item_id
        for item_id, row in contract_by_id.items()
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    }
    cng = {
        item_id
        for item_id, row in contract_by_id.items()
        if row["family"] in {"cng_level", "cng_spectrum"}
    }
    if len(speech) != 3 or len(cng) != 2:
        raise ValueError("场景条件合同项必须为 3 个语音项和 2 个 CNG 项")
    return speech, cng

SUMMARY_FIELDS = (
    "pass",
    "fail",
    "review_required",
    "not_measured",
    "invalid",
    "total",
    "computed",
)
PROGRESS_FIELDS = (
    "completed_cases",
    "total_cases",
    "completed_items",
    "total_items",
)
TX_VISUAL_FIELDS = (
    "source_sha256",
    "duration_s",
    "sample_rate",
    "channels",
    "peak",
    "wave_scale",
    "wave_min_i16",
    "wave_max_i16",
    "mel_i16",
    "mel_ceiling_db",
)
EVIDENCE_BINDING_FIELDS = (
    *tuple(CONDITIONS),
    "case_id",
    "report_id",
    "batch_id",
    "dataset_id",
    "input_sha256",
    "output_sha256",
)
CASE_REQUIRED_FIELDS = (
    "case_id",
    "status",
    "report_id",
    "conditions",
    "contract_item_ids",
    "signals",
    "signals_sha256",
    "input_sha256",
    "provenance",
    "tx_visual",
)
CASE_COMPLETED_FIELDS = (
    *CASE_REQUIRED_FIELDS,
    "output_sha256",
    "execution",
    "assessment_sha256",
)
ITEM_CORE_FIELDS = (
    "contract_id",
    "item_id",
    "observation_id",
    "case_id",
    "mode",
    "status",
    "value",
    "metric",
    "reason",
    "evidence_binding",
)
SNAPSHOT_REQUIRED_FIELDS = (
    "schema_version",
    "batch_id",
    "dataset_id",
    "dataset_name",
    "dataset_manifest_sha256",
    "dataset_assets_sha256",
    "algorithm_cluster_id",
    "configuration",
    "catalog_sha256",
    "cli_sha256",
    "status",
    "created_at",
    "updated_at",
    "selected_mode",
    "selected_scene",
    "test_scope",
    "standard_compliance",
    "formal_test_status",
    "contract_item_count",
    "observation_count",
    "progress",
    "summary",
    "items",
    "cases",
)
SNAPSHOT_COMPAT_FIELDS = ("schema", "name")

_BATCH_ID = re.compile(r"^[a-f0-9]{32}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SNAPSHOT_NAME = re.compile(r"^(\d{8})\.json$")
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed"})
_TERMINAL_CASE_STATUSES = frozenset({"completed", "failed"})
_ITEM_STATUSES = frozenset(
    {
        "pending",
        "running",
        "review_required",
        "not_measured",
        "invalid",
    }
)
SINGLE_CASE_EXECUTION_SCOPE = "single_case"
SINGLE_CASE_SELECTION_FIELDS = (
    "execution_scope",
    "selected_case_id",
    "dataset_case_count",
    "dataset_observation_count",
)
CASE_WORKERS_ENV = "ITU_ACCEPTANCE_CASE_WORKERS"
DEFAULT_CASE_WORKERS = 2
MAX_CASE_WORKERS = 4


class ResourceSafetyStop(RuntimeError):
    """The mandatory disk or memory floor was reached."""


class ArtifactIntegrityError(ValueError):
    """A frozen TX artifact path or digest no longer matches its snapshot."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _existing_ancestor(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise OSError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) < 2:
                    break
                return int(fields[1]) * 1024
    raise OSError("MemAvailable is missing from /proc/meminfo")


def _resource_preflight(path: Path | None = None) -> dict[str, int]:
    """Fail closed before every write and before each real case."""
    target = Path(path) if path is not None else Path(service.BUILD)
    try:
        usage = shutil.disk_usage(_existing_ancestor(target))
        snapshot = {
            "disk_free_bytes": int(usage.free),
            "mem_available_bytes": _mem_available_bytes(),
        }
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise ResourceSafetyStop(
                "ENOSPC while checking the scene-condition batch filesystem"
            ) from error
        raise
    if snapshot["disk_free_bytes"] < MIN_FREE_BYTES:
        raise ResourceSafetyStop(
            "disk free is below the 100 GiB scene-condition batch floor: "
            f"{snapshot['disk_free_bytes']}"
        )
    if snapshot["mem_available_bytes"] < MIN_MEM_AVAILABLE_BYTES:
        raise ResourceSafetyStop(
            "MemAvailable is below the 5 GiB scene-condition batch floor: "
            f"{snapshot['mem_available_bytes']}"
        )
    return snapshot


def _raise_if_enospc(error: BaseException) -> None:
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        raise ResourceSafetyStop(
            "ENOSPC while writing scene-condition batch evidence"
        ) from error


def _is_enospc_error(error: BaseException) -> bool:
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return True
    message = str(error).lower()
    return (
        "no space left on device" in message
        or "enospc" in message
    )


def _batch_root() -> Path:
    return Path(service.BUILD) / BATCH_ROOT_NAME


def _batch_directory(identifier: str) -> Path:
    if not isinstance(identifier, str) or _BATCH_ID.fullmatch(identifier) is None:
        raise ValueError("非法场景条件批次 ID")
    return _batch_root() / identifier


def _snapshot_directory(identifier: str) -> Path:
    return _batch_directory(identifier) / "snapshots"


def _case_directory(identifier: str, case_id: str) -> Path:
    if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
        raise ValueError("非法场景条件 case ID")
    cases_root = _batch_directory(identifier) / "cases"
    destination = cases_root / case_id
    if destination.parent != cases_root:
        raise ValueError("场景条件 case 目录越界")
    return destination


def _summary(items: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        status: sum(row.get("status") == status for row in items)
        for status in SUMMARY_FIELDS[:5]
    }
    result.update(
        total=len(items),
        computed=sum(
            _finite_number(row.get("value"))
            and row.get("status")
            not in {"pending", "running", "invalid", "not_measured"}
            for row in items
        ),
    )
    return result


def _progress(snapshot: dict[str, Any]) -> dict[str, int]:
    cases = snapshot.get("cases", [])
    items = snapshot.get("items", [])
    values = {
        "completed_cases": sum(
            case.get("status") in _TERMINAL_CASE_STATUSES for case in cases
        ),
        "total_cases": len(cases),
        "completed_items": sum(
            row.get("status") not in {"pending", "running"} for row in items
        ),
        "total_items": len(items),
    }
    return {field: values[field] for field in PROGRESS_FIELDS}


def summarize_items(items: list[dict[str, Any]]) -> dict[str, int]:
    """Pure summary entry point used by API tests and persistence."""
    return _summary(items)


def progress_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    """Pure progress entry point used by API tests and persistence."""
    return _progress(snapshot)


def _snapshot_versions(identifier: str) -> list[int]:
    batch_directory = _batch_directory(identifier)
    try:
        batch_metadata = batch_directory.lstat()
    except FileNotFoundError:
        return []
    if (
        not stat.S_ISDIR(batch_metadata.st_mode)
        or batch_directory.is_symlink()
    ):
        raise ValueError("场景条件批次目录不是普通目录")
    directory = _snapshot_directory(identifier)
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
        raise ValueError("场景条件快照目录不是普通目录")
    versions: list[int] = []
    for path in directory.iterdir():
        match = _SNAPSHOT_NAME.fullmatch(path.name)
        if match is None:
            continue
        versions.append(int(match.group(1)))
    return sorted(versions)


def _read_snapshot(identifier: str, version: int) -> dict[str, Any]:
    path = _snapshot_directory(identifier) / f"{version:08d}.json"
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"场景条件快照不可读取: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError("场景条件快照不是普通文件")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"场景条件快照不可解析: {path}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA
        or value.get("batch_id") != identifier
    ):
        raise ValueError("场景条件快照身份无效")
    return value


def _latest_snapshot_bytes(identifier: str) -> bytes:
    versions = _snapshot_versions(identifier)
    if not versions:
        raise ValueError(
            f"fixture marker has no snapshot for batch {identifier}"
        )
    path = _snapshot_directory(identifier) / f"{versions[-1]:08d}.json"
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(
            f"fixture marker latest snapshot is unreadable: {path}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError("fixture marker latest snapshot is not a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"fixture marker latest snapshot is unreadable: {path}"
        ) from error


def _strict_marker_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate fixture marker key: {key}")
        result[key] = value
    return result


def _reject_marker_constant(value: str) -> None:
    raise ValueError(f"non-finite fixture marker value: {value}")


def _is_fixture_batch(directory: Path) -> bool:
    """Return true only for a fully bound, explicit synthetic fixture."""
    marker_path = directory / FIXTURE_MARKER_NAME
    try:
        metadata = marker_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError(
            f"fixture marker is unreadable: {marker_path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or marker_path.is_symlink():
        raise ValueError("fixture marker must be a regular file")
    try:
        marker = json.loads(
            marker_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_marker_object,
            parse_constant=_reject_marker_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(
            f"fixture marker is not valid JSON: {marker_path}: {error}"
        ) from error
    if (
        not isinstance(marker, dict)
        or set(marker) != set(FIXTURE_MARKER_FIELDS)
        or marker.get("schema") != FIXTURE_MARKER_SCHEMA
        or marker.get("batch_id") != directory.name
        or marker.get("reason") != FIXTURE_MARKER_REASON
        or not isinstance(marker.get("snapshot_sha256"), str)
        or re.fullmatch(r"[a-f0-9]{64}", marker["snapshot_sha256"]) is None
    ):
        raise ValueError("fixture marker schema or identity is invalid")
    actual_sha256 = hashlib.sha256(
        _latest_snapshot_bytes(directory.name)
    ).hexdigest()
    if marker["snapshot_sha256"] != actual_sha256:
        raise ValueError(
            "fixture marker snapshot_sha256 does not match latest snapshot"
        )
    return True


def _latest_snapshot(identifier: str) -> dict[str, Any] | None:
    versions = _snapshot_versions(identifier)
    if not versions:
        return None
    return _read_snapshot(identifier, versions[-1])


def _stale_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return a restart-safe failed view without rewriting the last version."""
    result = deepcopy(snapshot)
    message = "服务重启中断了该场景条件批次，请重新执行；已完成证据仍保留"
    result["status"] = "failed"
    result["error"] = message
    result.pop("current_case", None)
    for case in result.get("cases", []):
        if case.get("status") not in _TERMINAL_CASE_STATUSES:
            case["status"] = "failed"
            case["error"] = message
    for row in result.get("items", []):
        if row.get("status") in {"pending", "running"}:
            row.update(
                status="invalid",
                value=None,
                reason=message,
            )
    result["progress"] = _progress(result)
    result["summary"] = _summary(result.get("items", []))
    return result


def get_batch(identifier: str) -> dict[str, Any] | None:
    """Read the highest append-only snapshot, or a non-persisted stale view."""
    with acceptance_batch._LOCK:
        snapshot = _latest_snapshot(identifier)
        if snapshot is None:
            return None
        validate_snapshot(snapshot)
        if (
            snapshot.get("status") not in _TERMINAL_BATCH_STATUSES
            and identifier not in acceptance_batch._ACTIVE
        ):
            return validate_snapshot(_stale_snapshot(snapshot))
        return snapshot


def _save(snapshot: dict[str, Any]) -> None:
    """Create exactly one new numbered snapshot; never replace an old one."""
    identifier = snapshot.get("batch_id")
    if not isinstance(identifier, str) or _BATCH_ID.fullmatch(identifier) is None:
        raise ValueError("场景条件批次 ID 无效")
    with acceptance_batch._LOCK:
        _resource_preflight(_batch_directory(identifier))
        snapshot["summary"] = _summary(snapshot["items"])
        snapshot["progress"] = _progress(snapshot)
        snapshot["updated_at"] = _now()
        payload = _json_bytes(snapshot)
        versions = _snapshot_versions(identifier)
        version = versions[-1] + 1 if versions else 1
        path = _snapshot_directory(identifier) / f"{version:08d}.json"
        try:
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            _raise_if_enospc(error)
            raise


def _write_once_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        _raise_if_enospc(error)
        raise
    return hashlib.sha256(payload).hexdigest()


def _validated_dataset(
    dataset_id: str,
    requested_scene: str | None = None,
    requested_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if is_stress_dataset(dataset_id):
        from .acoustic_stress_batch import validated_dataset

        return validated_dataset(dataset_id, requested_scene)
    # Import lazily to keep the API -> backend module graph acyclic.
    from .acceptance_scene_condition_api import _load_dataset

    try:
        dataset = _load_dataset(dataset_id)
    except HTTPException:
        raise
    profile = profile_for_dataset(dataset_id)
    schema_version = _profile_schema_version(profile)
    case_target, speech_target, cng_target, observation_target = (
        _profile_counts(profile)
    )
    if (
        dataset.get("dataset_id") != dataset_id
        or dataset.get("dataset_name") != profile.dataset_name
        or dataset.get("dataset_kind") != profile.dataset_kind
        or dataset.get("mode") != profile_mode(profile)
        or dataset.get("scene") != profile.scene
        or dataset.get("schema_version", schema_version) != schema_version
        or requested_scene is not None
        and requested_scene != profile.scene
        or requested_mode is not None
        and requested_mode != profile_mode(profile)
    ):
        raise ValueError("场景条件测试集与请求模式或场景不一致")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != case_target:
        raise ValueError(
            f"场景条件测试集必须恰好包含 {case_target} 个 case"
        )
    if (
        dataset.get("contract_item_count") != CONTRACT_ITEM_COUNT
        or CONTRACT_ITEM_COUNT != 5
    ):
        raise ValueError("场景条件测试集合同项数量不是 5")
    if dataset.get("preview_condition_group_count", case_target) != case_target:
        raise ValueError("场景条件测试集 preview count 与 profile 不一致")
    if dataset.get("observation_count", observation_target) != observation_target:
        raise ValueError("场景条件测试集 observation count 与 profile 不一致")

    contract_by_id = _profile_contract_rows(profile)
    speech_item_ids, cng_item_ids = _item_ids_by_kind(contract_by_id)
    expected_conversation_counts = _profile_conversation_counts(profile)
    if schema_version == "scene-condition-preview-v2":
        if dataset.get("conversation_counts") != expected_conversation_counts:
            raise ValueError(
                "场景条件测试集 conversation_counts 与 profile 不一致"
            )

    kinds = {"speech": 0, "cng": 0}
    conversation_counts: dict[str, int] = {}
    seen_case_ids: set[str] = set()
    covered_items: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("场景条件 case 必须是对象")
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            or case_id in seen_case_ids
        ):
            raise ValueError("场景条件 case ID 无效或重复")
        seen_case_ids.add(case_id)
        if case.get("conditions") != profile.conditions:
            raise ValueError(
                f"{case_id} 的条件组与 {profile.scene} profile 不一致"
            )
        if set(case.get("signals", {})) != set(ROLES):
            raise ValueError(f"{case_id} 必须恰好包含四路输入角色")
        provenance = case.get("provenance")
        render_metadata = (
            provenance.get("render_metadata")
            if isinstance(provenance, dict)
            else None
        )
        kind = (
            render_metadata.get("kind")
            if isinstance(render_metadata, dict)
            else None
        )
        if kind not in kinds:
            raise ValueError(f"{case_id} 缺少有效的 speech/CNG 类型")
        kinds[kind] += 1
        conversation_mode = None
        if schema_version == "scene-condition-preview-v2":
            if not isinstance(provenance, dict):
                raise ValueError(f"{case_id} provenance 无效")
            try:
                activity = validate_activity_metadata(
                    provenance,
                    expected_duration_s=float(
                        getattr(profile, "duration_s", 12.0)
                    ),
                    expected_sample_rate_hz=profile_sample_rate_hz(profile),
                )
            except ValueError as error:
                raise ValueError(
                    f"{case_id} 活动元数据无效: {error}"
                ) from error
            conversation_mode = activity["conversation_mode"]
            conversation_counts[conversation_mode] = (
                conversation_counts.get(conversation_mode, 0) + 1
            )
            if kind != EXPECTED_RENDER_KIND[conversation_mode]:
                raise ValueError(
                    f"{case_id} conversation_mode 与 render kind 不一致"
                )
            if (
                "signal_kind" in provenance
                and provenance["signal_kind"] != kind
            ):
                raise ValueError(
                    f"{case_id} provenance.signal_kind 与 kind 不一致"
                )
            if (
                "scene" in provenance
                and provenance["scene"] != profile.scene
            ):
                raise ValueError(
                    f"{case_id} provenance.scene 与 profile 不一致"
                )
            if (
                "direction" in render_metadata
                and render_metadata["direction"]
                != profile.conditions["direction"]
            ):
                raise ValueError(
                    f"{case_id} render direction 与 profile 不一致"
                )
            if (
                conversation_mode in {"near_single", "double_talk"}
                and activity["analysis_start_s"] != 0.0
            ):
                raise ValueError(
                    f"{case_id} 完整 12 秒 TX 必须从 analysis_start_s=0 开始"
                )
        delays = (
            render_metadata.get("beam_delays_ms")
            if isinstance(render_metadata, dict)
            else None
        )
        if (
            not isinstance(delays, list)
            or not delays
            or any(not _finite_number(value) or value < 0 for value in delays)
        ):
            raise ValueError(f"{case_id} 的 beam_delays_ms 无效")
        analysis_start_s = provenance.get("analysis_start_s")
        if not _finite_number(analysis_start_s) or analysis_start_s < 0:
            raise ValueError(f"{case_id} 的 analysis_start_s 无效")
        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or len(set(item_ids)) != len(item_ids)
            or any(
                not isinstance(item_id, str)
                or item_id not in contract_by_id
                for item_id in item_ids
            )
        ):
            raise ValueError(f"{case_id} 合同项身份无效")
        expected_items = (
            speech_item_ids if kind == "speech" else cng_item_ids
        )
        if set(item_ids) != expected_items:
            raise ValueError(
                f"{case_id} 的 {kind} 合同项映射不符合 "
                f"{profile.scene} 合同"
            )
        covered_items.update(item_ids)
        expected_families = {
            contract_by_id[item_id]["family"]
            for item_id in item_ids
        }
        families = case.get("families")
        if (
            not isinstance(families, list)
            or len(set(families)) != len(families)
            or set(families) != expected_families
        ):
            raise ValueError(f"{case_id} 的 families 与合同项不一致")

    if (
        kinds["speech"] != speech_target
        or kinds["cng"] != cng_target
    ):
        raise ValueError(
            "场景条件测试集 speech/CNG 分组与 profile 不一致"
        )
    if (
        schema_version == "scene-condition-preview-v2"
        and conversation_counts != expected_conversation_counts
    ):
        raise ValueError(
            "场景条件测试集 conversation_mode 分组与 profile 不一致"
        )
    if covered_items != set(contract_by_id):
        raise ValueError("场景条件 case 未完整覆盖五个合同项")
    observations = (
        kinds["speech"] * len(speech_item_ids)
        + kinds["cng"] * len(cng_item_ids)
    )
    if observations != observation_target:
        raise ValueError(
            "场景条件测试集 observation count 与 profile 不一致"
        )
    return dataset, contract_by_id


def _execution_case(case: dict[str, Any]) -> dict[str, Any]:
    provenance = case["provenance"]
    render_metadata = provenance["render_metadata"]
    dynamic_metadata_present = (
        isinstance(render_metadata, dict)
        and any(
            key in render_metadata
            for key in (
                "near_arrival_delays_samples",
                "echo_paths",
                "noise_spatial",
            )
        )
    )
    dynamic_microphone_from_post_rx = (
        case["conditions"].get("direction") != "receive"
        and isinstance(render_metadata, dict)
        and isinstance(
            render_metadata.get("near_arrival_delays_samples"), list
        )
        and isinstance(render_metadata.get("noise_spatial"), dict)
    )
    if (
        case["conditions"].get("direction") != "receive"
        and (
            provenance.get("conversation_mode") is not None
            or dynamic_metadata_present
        )
        and not dynamic_microphone_from_post_rx
    ):
        raise ValueError(
            f"{case['case_id']} 缺少运行时 post-RX 麦克风合成元数据"
        )
    return {
        "id": case["case_id"],
        "conditions": deepcopy(case["conditions"]),
        "item_ids": list(case["contract_item_ids"]),
        "beam_delays_ms": list(render_metadata["beam_delays_ms"]),
        "analysis_start_s": provenance["analysis_start_s"],
        "conversation_mode": provenance.get("conversation_mode"),
        "dynamic_microphone_from_post_rx": dynamic_microphone_from_post_rx,
        "provenance": deepcopy(provenance),
    }


def _initial_items(
    dataset: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    identifier: str,
) -> list[dict[str, Any]]:
    profile = profile_for_dataset(dataset["dataset_id"])
    _, _, _, expected_observations = _profile_counts(profile)
    items: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        case_id = case["case_id"]
        input_sha256 = deepcopy(case["signals_sha256"])
        for item_id in case["contract_item_ids"]:
            contract = contract_by_id.get(item_id)
            if contract is None:
                raise ValueError(f"未知合同项 {item_id}")
            items.append(
                {
                    **deepcopy(contract),
                    "contract_id": item_id,
                    "item_id": item_id,
                    "observation_id": f"{case_id}:{item_id}",
                    "case_id": case_id,
                    "mode": dataset["mode"],
                    "status": "pending",
                    "value": None,
                    "metric": None,
                    "reason": "等待对应场景执行",
                    "evidence_binding": {
                        **deepcopy(case["conditions"]),
                        "case_id": case_id,
                        "report_id": None,
                        "batch_id": identifier,
                        "dataset_id": dataset["dataset_id"],
                        "input_sha256": input_sha256,
                        "output_sha256": {},
                    },
                }
            )
    if len(items) != expected_observations:
        raise ValueError(
            f"场景条件批次没有形成 {expected_observations} 个 observation"
        )
    observation_ids = {row["observation_id"] for row in items}
    if len(observation_ids) != expected_observations:
        raise ValueError("场景条件 observation_id 不唯一")
    return items


def _initial_cases(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "status": "pending",
            "report_id": None,
            "conditions": deepcopy(case["conditions"]),
            "contract_item_ids": list(case["contract_item_ids"]),
            "signals": deepcopy(case["signals"]),
            "signals_sha256": deepcopy(case["signals_sha256"]),
            "input_sha256": deepcopy(case["signals_sha256"]),
            "provenance": deepcopy(case["provenance"]),
            "tx_visual": None,
            "rx_visual": None,
        }
        for case in dataset["cases"]
    ]


def _validated_case_id(case_id: Any) -> str:
    if (
        not isinstance(case_id, str)
        or not case_id
        or _CASE_ID.fullmatch(case_id) is None
    ):
        raise ValueError("case_id 必须是非空且合法的场景条件 case ID")
    return case_id


def _selected_source_case(
    dataset: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    selected_case_id = _validated_case_id(case_id)
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise ValueError("场景条件测试集 cases 无效")
    matches = [
        case
        for case in cases
        if isinstance(case, dict) and case.get("case_id") == selected_case_id
    ]
    if not matches:
        raise ValueError(
            "case_id 不属于当前测试集的场景、模式或范围: "
            f"{selected_case_id}"
        )
    if len(matches) != 1:
        raise ValueError(f"场景条件测试集 case_id 重复: {selected_case_id}")
    return matches[0]


def _single_case_selected_id(
    snapshot: dict[str, Any],
) -> str | None:
    """Return the selected id for a single-case snapshot, or None for full."""
    present = [
        field for field in SINGLE_CASE_SELECTION_FIELDS if field in snapshot
    ]
    if not present:
        return None
    if len(present) != len(SINGLE_CASE_SELECTION_FIELDS):
        raise ValueError("单案例快照必须同时包含全部四个选择字段")
    if snapshot.get("execution_scope") != SINGLE_CASE_EXECUTION_SCOPE:
        raise ValueError("单案例快照 execution_scope 无效")
    return _validated_case_id(snapshot.get("selected_case_id"))


def _single_conversation_counts(
    counts: dict[str, int],
    selected_mode: str,
) -> dict[str, int]:
    if selected_mode not in counts:
        raise ValueError("所选 case 的 conversation_mode 不在 profile 中")
    return {
        str(mode): int(mode == selected_mode)
        for mode in counts
    }


def _single_category_counts(
    counts: dict[str, int],
    selected_category: str,
) -> dict[str, int]:
    if selected_category not in counts:
        raise ValueError("所选 case 的 workload_category 不在 profile 中")
    return {
        str(category): int(category == selected_category)
        for category in counts
    }


def _filter_snapshot_to_single_case(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Rebuild one selected case from a full snapshot and source dataset."""
    selected_case_id = _validated_case_id(case_id)
    source_case = _selected_source_case(dataset, selected_case_id)
    source_cases = dataset["cases"]
    full_case_count = snapshot.get("case_count")
    if (
        not isinstance(full_case_count, int)
        or isinstance(full_case_count, bool)
        or full_case_count != len(source_cases)
    ):
        raise ValueError("完整快照 case 数量与源数据集不一致")
    source_observation_count = dataset.get(
        "observation_count",
        snapshot["observation_count"],
    )
    if (
        not isinstance(source_observation_count, int)
        or isinstance(source_observation_count, bool)
        or source_observation_count != snapshot["observation_count"]
    ):
        raise ValueError("完整快照 observation 数量与源数据集不一致")

    frozen_cases = [
        case
        for case in snapshot.get("cases", [])
        if isinstance(case, dict)
        and case.get("case_id") == selected_case_id
    ]
    if len(frozen_cases) != 1:
        raise ValueError("完整快照未唯一包含所选 case")
    selected_items = [
        row
        for row in snapshot.get("items", [])
        if isinstance(row, dict)
        and row.get("case_id") == selected_case_id
    ]
    contract_item_ids = source_case.get("contract_item_ids")
    if (
        not isinstance(contract_item_ids, list)
        or len(contract_item_ids) != len(set(contract_item_ids))
        or len(selected_items) != len(contract_item_ids)
        or {row.get("item_id") for row in selected_items}
        != set(contract_item_ids)
    ):
        raise ValueError("完整快照中的所选 case 合同 observation 不完整")

    result = deepcopy(snapshot)
    result["cases"] = [deepcopy(frozen_cases[0])]
    result["items"] = deepcopy(selected_items)
    result["case_count"] = 1
    result["observation_count"] = len(selected_items)
    result["preview_condition_group_count"] = full_case_count
    result["execution_scope"] = SINGLE_CASE_EXECUTION_SCOPE
    result["selected_case_id"] = selected_case_id
    result["dataset_case_count"] = full_case_count
    result["dataset_observation_count"] = source_observation_count

    if is_stress_dataset(snapshot.get("dataset_id")):
        category_counts = snapshot.get("category_counts")
        if not isinstance(category_counts, dict):
            raise ValueError("stress 完整快照 category_counts 无效")
        provenance = source_case.get("provenance")
        category = (
            provenance.get("workload_category")
            if isinstance(provenance, dict)
            else None
        )
        result["category_counts"] = _single_category_counts(
            category_counts,
            str(category),
        )
    else:
        provenance = source_case.get("provenance")
        render_metadata = (
            provenance.get("render_metadata")
            if isinstance(provenance, dict)
            else None
        )
        kind = (
            render_metadata.get("kind")
            if isinstance(render_metadata, dict)
            else None
        )
        if kind not in {"speech", "cng"}:
            raise ValueError("所选 case 缺少有效的 speech/CNG 类型")
        result["speech_case_count"] = int(kind == "speech")
        result["cng_case_count"] = int(kind == "cng")
        schema_version = _profile_schema_version(
            profile_for_dataset(snapshot["dataset_id"])
        )
        if schema_version == "scene-condition-preview-v2":
            conversation_counts = snapshot.get("conversation_counts")
            if not isinstance(conversation_counts, dict):
                raise ValueError("v2 完整快照 conversation_counts 无效")
            conversation_mode = (
                provenance.get("conversation_mode")
                if isinstance(provenance, dict)
                else None
            )
            result["conversation_counts"] = _single_conversation_counts(
                conversation_counts,
                str(conversation_mode),
            )
        else:
            result["conversation_counts"] = {}

    result["summary"] = _summary(result["items"])
    result["progress"] = _progress(result)
    return result


def _build_snapshot(
    dataset: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    identifier: str,
    algorithm_cluster_id: str,
    configuration: dict[str, Any],
    catalog_sha256: str,
    cli_sha256: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    if is_stress_dataset(dataset["dataset_id"]):
        from .acoustic_stress_batch import build_snapshot

        return build_snapshot(
            dataset,
            contract_by_id,
            identifier,
            algorithm_cluster_id,
            configuration,
            catalog_sha256,
            cli_sha256,
            case_id=case_id,
        )
    timestamp = _now()
    profile = profile_for_dataset(dataset["dataset_id"])
    case_count, speech_count, cng_count, observation_count = (
        _profile_counts(profile)
    )
    schema_version = _profile_schema_version(profile)
    conversation_counts = _profile_conversation_counts(profile)
    items = _initial_items(dataset, contract_by_id, identifier)
    cases = _initial_cases(dataset)
    snapshot = {
        "schema": SCHEMA,
        "schema_version": SCHEMA,
        "batch_id": identifier,
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "name": dataset["dataset_name"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_assets_sha256": dataset["dataset_assets_sha256"],
        "algorithm_cluster_id": algorithm_cluster_id,
        "configuration": deepcopy(configuration),
        "catalog_sha256": catalog_sha256,
        "cli_sha256": cli_sha256,
        "status": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "selected_mode": dataset["mode"],
        "selected_scene": dataset["scene"],
        "test_scope": TEST_SCOPE,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "dataset_schema_version": schema_version,
        "case_count": case_count,
        "speech_case_count": speech_count,
        "cng_case_count": cng_count,
        "preview_condition_group_count": case_count,
        "observation_count": observation_count,
        "conversation_counts": conversation_counts,
        "progress": {
            "completed_cases": 0,
            "total_cases": case_count,
            "completed_items": 0,
            "total_items": observation_count,
        },
        "summary": _summary(items),
        "items": items,
        "cases": cases,
    }
    if profile_mode(profile).startswith("NB-"):
        from server.rate_paths import require_primary_path

        snapshot["native_nb8_identity"] = (
            acceptance_batch.native_nb8_identity()
        )
        snapshot["rate_paths"] = {
            "8000": require_primary_path(8_000).as_dict()
        }
    if case_id is not None:
        return _filter_snapshot_to_single_case(snapshot, dataset, case_id)
    return snapshot


def _create_batch_directories(identifier: str) -> None:
    root = _batch_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("场景条件批次根目录不是普通目录")
        directory = _batch_directory(identifier)
        directory.mkdir(parents=False, exist_ok=False)
        (directory / "snapshots").mkdir(exist_ok=False)
        (directory / "cases").mkdir(exist_ok=False)
    except OSError as error:
        _raise_if_enospc(error)
        raise


def _validate_completed_v2_diagnostics(
    case_record: dict[str, Any],
) -> None:
    """Bind persisted diagnostics to the exact case, TX, and activity window."""
    provenance = case_record.get("provenance")
    diagnostics = case_record.get("conversation_diagnostics")
    output_sha256 = case_record.get("output_sha256")
    if (
        not isinstance(provenance, dict)
        or not isinstance(diagnostics, dict)
        or not isinstance(output_sha256, dict)
        or not isinstance(output_sha256.get("tx"), str)
    ):
        raise ValueError(
            "v2 completed case provenance/diagnostics/output binding is missing"
        )
    activity = validate_activity_metadata(provenance)
    validate_diagnostics(
        diagnostics,
        expected_mode=activity["conversation_mode"],
        expected_case_id=case_record["case_id"],
        expected_tx_sha256=output_sha256["tx"],
        expected_activity=activity,
    )


def start_batch(
    dataset_id: str,
    algorithm_cluster_id: str,
    *,
    mode: str = MODE,
    scene: str = SCENE,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Validate, reserve the shared executor, then persist a queued batch."""
    if is_stress_dataset(dataset_id):
        from .acoustic_stress_batch import start_batch as start_stress_batch

        return start_stress_batch(
            dataset_id,
            algorithm_cluster_id,
            mode=mode,
            scene=scene,
            case_id=case_id,
        )
    try:
        target_profile = profile_for_dataset(dataset_id)
    except ValueError:
        target_profile = None
    if target_profile is not None and _profile_schema_version(target_profile) != "scene-condition-preview-v2":
        if mode != MODE:
            raise ValueError(f"场景条件批次仅支持 {MODE}")
        try:
            profile = profile_for_scene(scene)
        except ValueError as error:
            raise ValueError(f"场景条件批次不支持场景 {scene}") from error
    else:
        try:
            profile = profile_for_scene_and_mode(scene, mode)
        except ValueError as error:
            raise ValueError(f"场景条件批次不支持场景/模式 {scene}/{mode}") from error

    _resource_preflight()
    from server.alignment_runner import (
        resolve_algorithm_cluster,
        resolve_cli,
    )

    preset, _, configuration, catalog_sha256 = resolve_algorithm_cluster(
        algorithm_cluster_id
    )
    algorithm_profile = preset.get("profile")
    if (
        not isinstance(algorithm_profile, str)
        or not algorithm_profile.startswith("vcp")
    ):
        raise ValueError("场景条件批次只允许 VCP 算法簇")
    from server.rate_paths import (
        NB8_CLUSTER_ID,
        validate_cluster_rate_compatibility,
    )

    target_rate = profile_sample_rate_hz(profile)
    validate_cluster_rate_compatibility(
        algorithm_cluster_id,
        target_rate,
        configuration=configuration,
    )
    from server.ns_routes import ns_route_identity, resolve_ns_cli

    is_native_nb8_cluster = algorithm_cluster_id == NB8_CLUSTER_ID
    cli = (
        resolve_cli()
        if is_native_nb8_cluster
        else resolve_ns_cli(resolve_cli(), configuration)
    )
    if cli is None and not (is_native_nb8_cluster and profile_mode(profile).startswith("NB-")):
        raise ValueError("未找到可执行的 VCP CLI")
    selected_ns_identity = None if is_native_nb8_cluster else ns_route_identity(configuration)
    admitted_aec = None
    if (
        not is_native_nb8_cluster
        and not profile_mode(profile).startswith("NB-")
        and isinstance(configuration.get("selection"), dict)
        and aec_route_requires_capability(configuration)
    ):
        admitted_aec = admit_aec_route(
            cli,
            configuration,
            sample_rate=(48000 if target_rate == 48000 else 16000),
            reference_channels=1,
        )
    cli_sha256 = (
        _sha256_file(cli)
        if cli is not None
        else acceptance_batch.native_nb8_identity()["speex_cli_sha256"]
    )

    dataset, contract_by_id = _validated_dataset(
        dataset_id,
        requested_scene=profile.scene,
        requested_mode=profile_mode(profile),
    )
    if case_id is not None:
        _selected_source_case(dataset, case_id)
    identifier = uuid.uuid4().hex
    snapshot = _build_snapshot(
        dataset,
        contract_by_id,
        identifier,
        algorithm_cluster_id,
        configuration,
        catalog_sha256,
        cli_sha256,
        case_id=case_id,
    )
    if admitted_aec is not None:
        snapshot["aec_route_capability"] = admitted_aec
    if selected_ns_identity is not None:
        snapshot["ns_route_identity"] = selected_ns_identity

    with acceptance_batch._LOCK:
        if acceptance_batch._ACTIVE or acceptance_batch._LEGACY_ACTIVE:
            raise ValueError("已有测试任务正在执行，请等待完成后再启动")
        _resource_preflight()
        acceptance_batch._ACTIVE.add(identifier)
        try:
            _create_batch_directories(identifier)
            _save(snapshot)
            acceptance_batch._EXECUTOR.submit(run_batch, identifier)
        except Exception:
            acceptance_batch._ACTIVE.discard(identifier)
            raise
    return deepcopy(snapshot)


def _validate_snapshot_contract(snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("场景条件快照必须是对象")
    if is_stress_dataset(snapshot.get("dataset_id")):
        from .acoustic_stress_batch import validate_snapshot as validate_stress_snapshot

        validate_stress_snapshot(snapshot)
        return
    missing_snapshot_fields = [
        field for field in SNAPSHOT_REQUIRED_FIELDS if field not in snapshot
    ]
    if missing_snapshot_fields:
        raise ValueError(
            "场景条件快照缺少字段: " + ", ".join(missing_snapshot_fields)
        )
    if snapshot.get("schema_version") != SCHEMA:
        raise ValueError("场景条件快照 schema_version 无效")
    if (
        "schema" in snapshot
        and snapshot["schema"] != SCHEMA
    ):
        raise ValueError("场景条件快照 schema 兼容字段无效")
    if (
        "name" in snapshot
        and snapshot["name"] != snapshot["dataset_name"]
    ):
        raise ValueError("场景条件快照 name 兼容字段无效")
    if (
        not isinstance(snapshot.get("batch_id"), str)
        or _BATCH_ID.fullmatch(snapshot["batch_id"]) is None
        or not isinstance(snapshot.get("dataset_id"), str)
        or _BATCH_ID.fullmatch(snapshot["dataset_id"]) is None
    ):
        raise ValueError("场景条件批次或测试集 ID 无效")
    try:
        profile = profile_for_dataset(snapshot["dataset_id"])
    except ValueError as error:
        raise ValueError("场景条件快照绑定了未注册的数据集") from error
    schema_version = _profile_schema_version(profile)
    case_target, speech_target, cng_target, observation_target = (
        _profile_counts(profile)
    )
    selected_case_id = _single_case_selected_id(snapshot)
    single_case = selected_case_id is not None
    expected_conversation_counts = _profile_conversation_counts(profile)
    if (
        snapshot.get("dataset_name") != profile.dataset_name
        or snapshot.get("selected_scene") != profile.scene
        or snapshot.get("dataset_schema_version", schema_version)
        != schema_version
    ):
        raise ValueError("场景条件快照数据集名称、schema 或场景身份无效")
    for field in (
        "dataset_name",
        "algorithm_cluster_id",
        "created_at",
        "updated_at",
    ):
        if not isinstance(snapshot.get(field), str) or not snapshot[field]:
            raise ValueError(f"场景条件快照字段无效: {field}")
    for field in (
        "dataset_manifest_sha256",
        "dataset_assets_sha256",
        "catalog_sha256",
        "cli_sha256",
    ):
        if (
            not isinstance(snapshot.get(field), str)
            or re.fullmatch(r"[a-f0-9]{64}", snapshot[field]) is None
        ):
            raise ValueError(f"场景条件快照哈希无效: {field}")
    if not isinstance(snapshot.get("configuration"), dict):
        raise ValueError("场景条件快照 configuration 无效")
    if snapshot.get("status") not in {
        "queued",
        "running",
        "completed",
        "failed",
    }:
        raise ValueError("场景条件批次状态无效")
    cases = snapshot.get("cases")
    items = snapshot.get("items")
    if not isinstance(cases, list) or not isinstance(items, list):
        raise ValueError("场景条件快照 cases/items 必须是数组")
    if any(not isinstance(case, dict) for case in cases) or any(
        not isinstance(row, dict) for row in items
    ):
        raise ValueError("场景条件快照 cases/items 元素必须是对象")
    if snapshot.get("contract_item_count") != CONTRACT_ITEM_COUNT:
        raise ValueError("场景条件快照合同项总数与 profile 不一致")
    if single_case:
        if (
            not isinstance(snapshot.get("case_count"), int)
            or isinstance(snapshot.get("case_count"), bool)
            or snapshot.get("case_count") != 1
            or len(cases) != 1
            or snapshot.get("preview_condition_group_count") != case_target
            or not isinstance(snapshot.get("dataset_case_count"), int)
            or isinstance(snapshot.get("dataset_case_count"), bool)
            or snapshot.get("dataset_case_count") != case_target
            or not isinstance(snapshot.get("dataset_observation_count"), int)
            or isinstance(snapshot.get("dataset_observation_count"), bool)
            or snapshot.get("dataset_observation_count") != observation_target
            or not isinstance(snapshot.get("observation_count"), int)
            or isinstance(snapshot.get("observation_count"), bool)
            or snapshot["observation_count"] <= 0
            or len(items) != snapshot["observation_count"]
            or not isinstance(snapshot.get("speech_case_count"), int)
            or isinstance(snapshot.get("speech_case_count"), bool)
            or not isinstance(snapshot.get("cng_case_count"), int)
            or isinstance(snapshot.get("cng_case_count"), bool)
            or not isinstance(snapshot.get("conversation_counts"), dict)
        ):
            raise ValueError(
                "单案例快照选择字段、case/observation 数量或 profile 总量无效"
            )
    elif (
        snapshot.get("observation_count") != observation_target
        or len(cases) != case_target
        or len(items) != observation_target
        or snapshot.get("case_count", case_target) != case_target
        or snapshot.get("preview_condition_group_count", case_target)
        != case_target
        or snapshot.get("speech_case_count", speech_target) != speech_target
        or snapshot.get("cng_case_count", cng_target) != cng_target
    ):
        raise ValueError(
            "场景条件快照 case/observation 数量与 profile 不一致"
        )
    if schema_version == "scene-condition-preview-v2":
        if not single_case and (
            snapshot.get("conversation_counts") != expected_conversation_counts
        ):
            raise ValueError(
                "场景条件快照 conversation_counts 与 profile 不一致"
            )
    elif not single_case and (
        "conversation_counts" in snapshot
        and snapshot["conversation_counts"]
        not in ({}, expected_conversation_counts)
    ):
        raise ValueError("旧版场景条件快照 conversation_counts 无效")
    if (
        snapshot.get("selected_mode") != profile_mode(profile)
        or snapshot.get("selected_scene") != profile.scene
        or snapshot.get("test_scope") != TEST_SCOPE
        or snapshot.get("standard_compliance") is not False
        or snapshot.get("formal_test_status") != "not_measured"
    ):
        raise ValueError("场景条件快照执行范围或合规边界无效")
    item_keys = {
        (row.get("case_id"), row.get("item_id"))
        for row in items
    }
    if len(item_keys) != len(items):
        raise ValueError("场景条件快照 observation 不唯一")
    case_ids = [case.get("case_id") for case in cases]
    expected_batch_case_count = 1 if single_case else case_target
    if (
        any(
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            for case_id in case_ids
        )
        or len(set(case_ids)) != expected_batch_case_count
    ):
        raise ValueError("场景条件快照 case ID 无效或重复")
    case_by_id = {case["case_id"]: case for case in cases}
    contract_by_id = _profile_contract_rows(profile)
    speech_item_ids, cng_item_ids = _item_ids_by_kind(contract_by_id)
    covered_items: set[str] = set()
    kinds = {"speech": 0, "cng": 0}
    conversation_counts: dict[str, int] = {}
    for case in cases:
        missing_case_fields = [
            field for field in CASE_REQUIRED_FIELDS if field not in case
        ]
        if missing_case_fields:
            raise ValueError(
                "场景条件 case 缺少字段: " + ", ".join(missing_case_fields)
            )
        if case.get("status") not in {
            "pending",
            "running",
            "completed",
            "failed",
        }:
            raise ValueError("场景条件 case 状态无效")
        input_sha256 = case.get("input_sha256")
        signals_sha256 = case.get("signals_sha256")
        if (
            not isinstance(input_sha256, dict)
            or set(input_sha256) != set(ROLES)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-f0-9]{64}", value) is None
                for value in input_sha256.values()
            )
        ):
            raise ValueError("场景条件 case 必须绑定四路输入哈希")
        if (
            not isinstance(signals_sha256, dict)
            or set(signals_sha256) != set(ROLES)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-f0-9]{64}", value) is None
                for value in signals_sha256.values()
            )
            or input_sha256 != signals_sha256
        ):
            raise ValueError(
                "场景条件 case.input_sha256 必须完全等于 "
                "case.signals_sha256 的真实 manifest 四路哈希"
            )
        if case.get("conditions") != profile.conditions:
            raise ValueError(
                "场景条件 case conditions 与 dataset profile 不一致"
            )
        provenance = case.get("provenance")
        render_metadata = (
            provenance.get("render_metadata")
            if isinstance(provenance, dict)
            else None
        )
        kind = (
            render_metadata.get("kind")
            if isinstance(render_metadata, dict)
            else None
        )
        if kind not in kinds:
            raise ValueError("场景条件 case 缺少 speech/CNG 类型")
        kinds[kind] += 1
        conversation_mode = None
        if schema_version == "scene-condition-preview-v2":
            if not isinstance(provenance, dict):
                raise ValueError("场景条件 case provenance 无效")
            try:
                activity = validate_activity_metadata(
                    provenance,
                    expected_duration_s=float(
                        getattr(profile, "duration_s", 12.0)
                    ),
                    expected_sample_rate_hz=profile_sample_rate_hz(profile),
                )
            except ValueError as error:
                raise ValueError(
                    f"场景条件 case 活动元数据无效: {error}"
                ) from error
            conversation_mode = activity["conversation_mode"]
            conversation_counts[conversation_mode] = (
                conversation_counts.get(conversation_mode, 0) + 1
            )
            if kind != EXPECTED_RENDER_KIND[conversation_mode]:
                raise ValueError(
                    "场景条件 case conversation_mode 与 kind 不一致"
                )
            if (
                "signal_kind" in provenance
                and provenance["signal_kind"] != kind
            ):
                raise ValueError(
                    "场景条件 case provenance.signal_kind 与 kind 不一致"
                )
            if (
                "scene" in provenance
                and provenance["scene"] != profile.scene
            ):
                raise ValueError(
                    "场景条件 case provenance.scene 与 profile 不一致"
                )
            if (
                "direction" in render_metadata
                and render_metadata["direction"]
                != profile.conditions["direction"]
            ):
                raise ValueError(
                    "场景条件 case render direction 与 profile 不一致"
                )
            if (
                conversation_mode in {"near_single", "double_talk"}
                and activity["analysis_start_s"] != 0.0
            ):
                raise ValueError(
                    "near_single/double_talk 必须使用完整 12 秒 TX"
                )
        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or len(set(item_ids)) != len(item_ids)
            or any(
                not isinstance(item_id, str)
                or item_id not in contract_by_id
                for item_id in item_ids
            )
        ):
            raise ValueError("场景条件 case 合同项身份无效")
        expected_items = (
            speech_item_ids if kind == "speech" else cng_item_ids
        )
        if set(item_ids) != expected_items:
            raise ValueError(
                "场景条件 case 三/二行合同项与 kind 不一致"
            )
        covered_items.update(item_ids)
        report_id = case.get("report_id")
        if report_id is not None and (
            not isinstance(report_id, str)
            or re.fullmatch(r"[a-f0-9]{32}", report_id) is None
        ):
            raise ValueError("场景条件 case report_id 无效")
        if case.get("status") == "completed":
            missing_completed_fields = [
                field
                for field in CASE_COMPLETED_FIELDS
                if field not in case
            ]
            if missing_completed_fields:
                raise ValueError(
                    "完成 case 缺少字段: "
                    + ", ".join(missing_completed_fields)
                )
            output_sha256 = case.get("output_sha256")
            if (
                not isinstance(output_sha256, dict)
                or set(output_sha256) not in ({"tx"}, {"rx", "tx"})
                or not isinstance(output_sha256.get("tx"), str)
                or re.fullmatch(r"[a-f0-9]{64}", output_sha256["tx"]) is None
                or (
                    "rx" in output_sha256
                    and (
                        not isinstance(output_sha256.get("rx"), str)
                        or re.fullmatch(
                            r"[a-f0-9]{64}",
                            output_sha256["rx"],
                        )
                        is None
                    )
                )
                or not isinstance(case.get("assessment_sha256"), str)
                or re.fullmatch(
                    r"[a-f0-9]{64}",
                    case["assessment_sha256"],
                )
                is None
                or not isinstance(case.get("execution"), dict)
                or not isinstance(case.get("tx_visual"), dict)
            ):
                raise ValueError("完成 case 缺少 TX 输出或可视化证据")
            if set(case["tx_visual"]) != set(TX_VISUAL_FIELDS):
                raise ValueError("完成 case 的 tx_visual 字段不完整")
            if "rx" in output_sha256:
                rx_visual = case.get("rx_visual")
                if (
                    not isinstance(rx_visual, dict)
                    or set(rx_visual) != set(TX_VISUAL_FIELDS)
                    or rx_visual.get("source_sha256") != output_sha256["rx"]
                ):
                    raise ValueError(
                        "完成 case 的 RX 处理后可视化证据不完整"
                    )
            execution = case["execution"]
            acceptance_batch.validate_execution_identity(
                snapshot,
                execution,
                require_algorithm_configuration=True,
            )
            if (
                execution.get("output_sha256") != output_sha256
            ):
                raise ValueError("完成 case 的执行身份与批次冻结配置不一致")
            if (
                case["tx_visual"].get("source_sha256")
                != output_sha256["tx"]
            ):
                raise ValueError("完成 case 的可视化源哈希与 TX 不一致")
            echo_reference = execution.get("echo_reference")
            if {"rx", "tx"}.issubset(output_sha256):
                if (
                    not isinstance(echo_reference, dict)
                    or echo_reference.get("source")
                    not in {
                        "vcp_rx_processed",
                        "native_nb8_rx_processed",
                    }
                    or echo_reference.get("processed_output") != "rx"
                    or echo_reference.get("processed_sha256")
                    != output_sha256["rx"]
                    or echo_reference.get("tx_echo_argument_sha256")
                    != output_sha256["rx"]
                ):
                    raise ValueError(
                        "完成 case 未证明 TX 使用 RX 处理后回声参考"
                    )
            elif echo_reference is not None:
                raise ValueError(
                    "旧版 TX-only case 不得声明 RX 处理后回声参考"
                )
            dynamic_microphone = execution.get("dynamic_microphone")
            if dynamic_microphone is not None:
                dynamic_visual = case.get("dynamic_microphone_visual")
                effective_inputs = case.get("effective_input_sha256")
                if (
                    not isinstance(dynamic_microphone, dict)
                    or case.get("dynamic_microphone")
                    != dynamic_microphone
                    or not isinstance(dynamic_visual, dict)
                    or set(dynamic_visual) != set(TX_VISUAL_FIELDS)
                    or dynamic_visual.get("source_sha256")
                    != dynamic_microphone.get("sha256")
                    or not isinstance(effective_inputs, dict)
                    or set(effective_inputs)
                    != {"microphone", "reference", "noise", "far_end"}
                    or any(
                        not isinstance(value, str)
                        or re.fullmatch(r"[a-f0-9]{64}", value) is None
                        for value in effective_inputs.values()
                    )
                    or execution.get("effective_input_sha256")
                    != effective_inputs
                    or effective_inputs.get("far_end")
                    != output_sha256.get("rx")
                ):
                    raise ValueError(
                        "完成 case 的运行时动态麦克风证据不完整"
                    )
            elif (
                case.get("dynamic_microphone") is not None
                or case.get("dynamic_microphone_visual") is not None
                or case.get("effective_input_sha256") is not None
            ):
                raise ValueError(
                    "case 声明了动态麦克风但 execution 未证明其来源"
                )
            if schema_version == "scene-condition-preview-v2":
                try:
                    _validate_completed_v2_diagnostics(case)
                except ValueError as error:
                    raise ValueError(
                        "v2 完成 case conversation_diagnostics 绑定无效: "
                        f"{error}"
                    ) from error
    for row in snapshot["items"]:
        missing_item_fields = [
            field for field in ITEM_CORE_FIELDS if field not in row
        ]
        if missing_item_fields:
            raise ValueError(
                "场景条件 item 缺少字段: " + ", ".join(missing_item_fields)
            )
        if row.get("status") not in _ITEM_STATUSES:
            raise ValueError("场景条件 item 状态无效")
        case_id = row.get("case_id")
        item_id = row.get("item_id")
        if case_id not in case_by_id:
            raise ValueError("场景条件 item 引用了未知 case")
        if item_id not in case_by_id[case_id]["contract_item_ids"]:
            raise ValueError("场景条件 item 不属于对应 case")
        contract = contract_by_id.get(item_id)
        if contract is None:
            raise ValueError("场景条件 item 不属于当前 dataset profile")
        for field in (
            "id",
            "family",
            "source_cell",
            "direction",
        ):
            if row.get(field) != contract.get(field):
                raise ValueError(
                    f"场景条件 item {field} 不能跨场景或合同行"
                )
        if row.get("observation_id") != f"{case_id}:{item_id}":
            raise ValueError("场景条件 observation_id 不符合 case:item 契约")
        if row.get("mode") != snapshot["selected_mode"]:
            raise ValueError("场景条件 item mode 无效")
        if row.get("value") is not None and not _finite_number(row["value"]):
            raise ValueError("场景条件 item value 不是有限数")
        if (
            row.get("value") is not None
            and row.get("status") != "review_required"
        ):
            raise ValueError("有限 item 不得越过 review_required")
        binding = row.get("evidence_binding")
        case_conditions = case_by_id[case_id]["conditions"]
        if (
            not isinstance(binding, dict)
            or set(binding) != set(EVIDENCE_BINDING_FIELDS)
            or binding.get("case_id") != case_id
            or binding.get("batch_id") != snapshot["batch_id"]
            or binding.get("dataset_id") != snapshot["dataset_id"]
            or binding.get("input_sha256")
            != case_by_id[case_id]["input_sha256"]
            or any(
                binding.get(field) != value
                for field, value in case_conditions.items()
            )
        ):
            raise ValueError("场景条件 item evidence_binding 不完整")
        if row.get("status") == "review_required" and (
            not isinstance(binding.get("output_sha256"), dict)
            or binding.get("output_sha256")
            != case_by_id[case_id].get("output_sha256")
        ):
            raise ValueError(
                "review_required item 必须绑定该 case 的完整输出证据"
            )
    if single_case:
        selected_case = cases[0]
        expected_observation_count = len(
            selected_case["contract_item_ids"]
        )
        if (
            selected_case["case_id"] != selected_case_id
            or snapshot["observation_count"] != expected_observation_count
            or len(items) != expected_observation_count
            or kinds["speech"] + kinds["cng"] != 1
            or snapshot["speech_case_count"] != kinds["speech"]
            or snapshot["cng_case_count"] != kinds["cng"]
            or covered_items != set(selected_case["contract_item_ids"])
        ):
            raise ValueError(
                "单案例快照所选 case、合同 observation 或 speech/CNG 计数不一致"
            )
        if schema_version == "scene-condition-preview-v2":
            selected_mode = selected_case["provenance"]["conversation_mode"]
            expected_single_conversations = _single_conversation_counts(
                expected_conversation_counts,
                selected_mode,
            )
            if snapshot["conversation_counts"] != expected_single_conversations:
                raise ValueError(
                    "单案例快照 conversation_counts 与所选 case 不一致"
                )
        elif snapshot["conversation_counts"] != {}:
            raise ValueError("旧版单案例快照 conversation_counts 必须为空")
    else:
        if (
            kinds["speech"] != speech_target
            or kinds["cng"] != cng_target
        ):
            raise ValueError("场景条件快照 speech/CNG 分组与 profile 不一致")
        if (
            schema_version == "scene-condition-preview-v2"
            and conversation_counts != expected_conversation_counts
        ):
            raise ValueError(
                "场景条件快照 conversation_mode 分组与 profile 不一致"
            )
        if covered_items != set(contract_by_id):
            raise ValueError("场景条件快照未覆盖五个合同项")
    if snapshot.get("summary") != summarize_items(snapshot["items"]):
        raise ValueError("场景条件 summary 与 items 不一致")
    if snapshot.get("progress") != progress_counts(snapshot):
        raise ValueError("场景条件 progress 与 cases/items 不一致")
    if set(snapshot["summary"]) != set(SUMMARY_FIELDS):
        raise ValueError("场景条件 summary 字段不完整")
    if set(snapshot["progress"]) != set(PROGRESS_FIELDS):
        raise ValueError("场景条件 progress 字段不完整")
    if snapshot["status"] in _TERMINAL_BATCH_STATUSES and any(
        case.get("status") not in _TERMINAL_CASE_STATUSES
        for case in cases
    ):
        raise ValueError("终态批次仍包含未完成 case")
    if snapshot["status"] in _TERMINAL_BATCH_STATUSES and any(
        row.get("status") in {"pending", "running"}
        for row in snapshot["items"]
    ):
        raise ValueError("终态批次仍包含 pending/running observation")
    if snapshot["status"] == "completed" and any(
        case.get("status") == "failed" for case in cases
    ):
        raise ValueError("completed 批次不得包含 failed case")
    failed_case_ids = {
        case["case_id"]
        for case in cases
        if case.get("status") == "failed"
    }
    if any(
        row.get("case_id") in failed_case_ids
        and row.get("status") != "invalid"
        for row in snapshot["items"]
    ):
        raise ValueError("failed case 的所有 observation 必须为 invalid")


def validate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Public pure validation entry point; returns the same snapshot."""
    _validate_snapshot_contract(snapshot)
    return snapshot


def _verify_dataset_frozen(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
) -> None:
    if is_stress_dataset(snapshot.get("dataset_id")):
        from .acoustic_stress_batch import verify_dataset_frozen

        verify_dataset_frozen(snapshot, dataset)
        return
    profile = profile_for_dataset(snapshot["dataset_id"])
    schema_version = _profile_schema_version(profile)
    case_count, speech_count, cng_count, observation_count = (
        _profile_counts(profile)
    )
    selected_case_id = _single_case_selected_id(snapshot)
    single_case = selected_case_id is not None
    for field in (
        "dataset_id",
        "dataset_manifest_sha256",
        "dataset_assets_sha256",
    ):
        actual = dataset.get(field)
        expected = snapshot.get(field)
        if actual != expected:
            raise ValueError(f"批次开始后测试集身份发生变化: {field}")
    if (
        dataset.get("mode") != snapshot.get("selected_mode")
        or dataset.get("scene") != snapshot.get("selected_scene")
        or dataset.get("dataset_name") != profile.dataset_name
        or dataset.get("dataset_kind") != profile.dataset_kind
        or dataset.get("schema_version", schema_version) != schema_version
        or len(dataset.get("cases", [])) != case_count
        or dataset.get("preview_condition_group_count", case_count)
        != case_count
        or dataset.get("observation_count", observation_count)
        != observation_count
    ):
        raise ValueError(
            "批次开始后测试集模式、场景、schema 或 profile 计数发生变化"
        )
    if single_case:
        if (
            snapshot.get("dataset_schema_version", schema_version)
            != schema_version
            or snapshot.get("dataset_case_count") != case_count
            or snapshot.get("dataset_observation_count") != observation_count
            or snapshot.get("preview_condition_group_count") != case_count
            or snapshot.get("case_count") != 1
        ):
            raise ValueError("单案例快照的源数据集总量发生变化")
    elif (
        snapshot.get("dataset_schema_version", schema_version) != schema_version
        or snapshot.get("case_count", case_count) != case_count
        or snapshot.get("speech_case_count", speech_count) != speech_count
        or snapshot.get("cng_case_count", cng_count) != cng_count
        or snapshot.get("observation_count") != observation_count
    ):
        raise ValueError("批次快照的 profile 计数发生变化")
    if schema_version == "scene-condition-preview-v2":
        expected_conversations = _profile_conversation_counts(profile)
        if dataset.get("conversation_counts") != expected_conversations:
            raise ValueError(
                "批次开始后 conversation_counts 发生变化"
            )
        if single_case:
            selected_case = _selected_source_case(
                dataset,
                selected_case_id,
            )
            selected_mode = selected_case["provenance"]["conversation_mode"]
            if snapshot.get("conversation_counts") != (
                _single_conversation_counts(
                    expected_conversations,
                    selected_mode,
                )
            ):
                raise ValueError(
                    "单案例快照 conversation_counts 发生变化"
                )
        elif snapshot.get("conversation_counts") != expected_conversations:
            raise ValueError(
                "批次开始后 conversation_counts 发生变化"
            )
    expected_cases = {
        case["case_id"]: case for case in dataset["cases"]
    }
    if single_case:
        selected_source_case = _selected_source_case(
            dataset,
            selected_case_id,
        )
        expected_cases = {
            selected_case_id: selected_source_case,
        }
    snapshot_cases = {
        case["case_id"]: case for case in snapshot["cases"]
    }
    if set(expected_cases) != set(snapshot_cases):
        raise ValueError("批次开始后场景条件 case 集合发生变化")
    for case_id, case in expected_cases.items():
        frozen_case = snapshot_cases[case_id]
        if frozen_case.get("signals_sha256") != case["signals_sha256"]:
            raise ValueError(f"批次开始后 {case_id} 输入身份发生变化")
        for field in (
            "conditions",
            "contract_item_ids",
            "signals",
        ):
            if frozen_case.get(field) != case.get(field):
                raise ValueError(
                    f"批次开始后 {case_id} {field} 发生变化"
                )
        frozen_provenance = frozen_case.get("provenance")
        current_provenance = case.get("provenance")
        if not isinstance(frozen_provenance, dict) or not isinstance(
            current_provenance,
            dict,
        ):
            raise ValueError(f"批次开始后 {case_id} provenance 无效")
        for field in ("analysis_start_s", "render_metadata"):
            if frozen_provenance.get(field) != current_provenance.get(field):
                raise ValueError(
                    f"批次开始后 {case_id} provenance.{field} 发生变化"
                )


def _hash_case_inputs(
    dataset: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for role in ROLES:
        path = asset_path(dataset["root"], case["signals"][role])
        actual[role] = _sha256_file(path)
        if actual[role] != case["signals_sha256"][role]:
            raise ValueError(f"{case['case_id']} 输入 {role} 文件身份已改变")
    return actual


def _assert_execution_frozen(
    snapshot: dict[str, Any],
    execution: dict[str, Any],
    input_sha256: dict[str, str],
) -> None:
    acceptance_batch.validate_execution_identity(
        snapshot,
        execution,
        require_algorithm_configuration=True,
    )
    if execution.get("input_sha256") != input_sha256:
        raise ValueError("执行器报告的输入哈希与实际四路输入不一致")
    outputs = execution.get("output_sha256")
    if (
        not isinstance(outputs, dict)
        or set(outputs) not in ({"tx"}, {"rx", "tx"})
    ):
        raise ValueError("执行器必须报告 TX 及可选的 RX 处理后输出")
    if {"rx", "tx"}.issubset(outputs):
        echo_reference = execution.get("echo_reference")
        if (
            not isinstance(echo_reference, dict)
            or echo_reference.get("source")
            not in {
                "vcp_rx_processed",
                "native_nb8_rx_processed",
            }
            or echo_reference.get("processed_output") != "rx"
            or echo_reference.get("processed_sha256") != outputs["rx"]
            or echo_reference.get("tx_echo_argument_sha256")
            != outputs["rx"]
        ):
            raise ValueError("执行器未证明 TX 使用 RX 处理后回声参考")


def _actual_output_hashes(
    outputs: dict[str, Path],
    execution: dict[str, Any],
) -> dict[str, str]:
    actual = {role: _sha256_file(path) for role, path in outputs.items()}
    if actual != execution.get("output_sha256"):
        raise ValueError("执行期间 VCP 输出身份发生变化")
    return actual


def _normalized_item(
    existing: dict[str, Any],
    calculated: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    case: dict[str, Any],
    report_id: str,
    input_sha256: dict[str, str],
    output_sha256: dict[str, str],
) -> dict[str, Any]:
    row = {**existing, **deepcopy(calculated)}
    row.update(
        contract_id=existing["contract_id"],
        item_id=existing["item_id"],
        observation_id=existing["observation_id"],
        case_id=case["case_id"],
        mode=snapshot["selected_mode"],
        report_id=report_id,
        standard_compliance=False,
        evidence_binding={
            **deepcopy(case["conditions"]),
            "case_id": case["case_id"],
            "report_id": report_id,
            "batch_id": snapshot["batch_id"],
            "dataset_id": snapshot["dataset_id"],
            "input_sha256": deepcopy(input_sha256),
            "output_sha256": deepcopy(output_sha256),
        },
    )
    value = row.get("value")
    status = row.get("status")
    metric = row.get("metric")
    reason = row.get("reason")
    if value is not None and not _finite_number(value):
        row.update(
            value=None,
            metric=None,
            status="invalid",
            reason="计算值不是有限数",
            evidence_grade="invalid",
        )
    elif value is not None:
        if not isinstance(metric, str) or not metric:
            row.update(
                value=None,
                metric=None,
                status="invalid",
                reason="有限计算值缺少 metric 身份",
                evidence_grade="invalid",
            )
        else:
            row.update(
                status="review_required",
                evidence_grade="engineering_proxy",
            )
    elif status == "invalid":
        row.update(status="invalid", evidence_grade="invalid")
    else:
        row.update(
            status="not_measured",
            metric=None,
            evidence_grade="not_measured",
        )
    if not isinstance(row.get("reason"), str) or not row["reason"]:
        row["reason"] = "计算模型未提供原因"
    return row


def _compact_visual(visual: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in TX_VISUAL_FIELDS if field not in visual]
    if missing:
        raise ValueError(
            "TX visualization 缺少字段: " + ", ".join(missing)
        )
    return {field: deepcopy(visual[field]) for field in TX_VISUAL_FIELDS}


def _verified_assessment(
    snapshot: dict[str, Any],
    case_record: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    if is_stress_dataset(snapshot.get("dataset_id")):
        from .acoustic_stress_batch import validate_assessment

        return validate_assessment(snapshot, case_record, destination)
    assessment_path = destination / "assessment.json"
    if assessment_path.is_symlink():
        raise ArtifactIntegrityError("assessment.json 不能是符号链接")
    if not assessment_path.is_file():
        raise ArtifactIntegrityError("assessment.json 不存在")
    expected_sha256 = case_record.get("assessment_sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
    ):
        raise ArtifactIntegrityError("case 缺少 assessment_sha256")
    try:
        payload = assessment_path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        assessment = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"assessment.json 不可读取或不可解析: {error}"
        ) from error
    if actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError("assessment.json SHA-256 校验失败")
    if (
        not isinstance(assessment, dict)
        or assessment.get("schema_version") != SCHEMA
        or assessment.get("case_id") != case_record.get("case_id")
        or assessment.get("report_id") != case_record.get("report_id")
        or assessment.get("batch_id") != snapshot.get("batch_id")
        or assessment.get("dataset_id") != snapshot.get("dataset_id")
        or assessment.get("conditions") != case_record.get("conditions")
        or case_record.get("input_sha256")
        != case_record.get("signals_sha256")
        or assessment.get("input_sha256")
        != case_record.get("input_sha256")
        or assessment.get("output_sha256") != case_record.get("output_sha256")
        or assessment.get("rx_visual") != case_record.get("rx_visual")
    ):
        raise ArtifactIntegrityError(
            "assessment.json 的 case/report/batch/dataset/input/output "
            "绑定不一致"
        )
    profile = profile_for_dataset(snapshot["dataset_id"])
    schema_version = _profile_schema_version(profile)
    if "provenance" in assessment and (
        assessment.get("provenance") != case_record.get("provenance")
    ):
        raise ArtifactIntegrityError(
            "assessment.json 的 provenance 与 case 不一致"
        )
    for field in (
        "dynamic_microphone",
        "dynamic_microphone_visual",
        "effective_input_sha256",
    ):
        if assessment.get(field) != case_record.get(field):
            raise ArtifactIntegrityError(
                f"assessment.json 的 {field} 与 case 不一致"
            )
    if schema_version == "scene-condition-preview-v2":
        try:
            _validate_completed_v2_diagnostics(case_record)
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                "assessment.json 的 v2 诊断身份绑定无效: "
                f"{error}"
            ) from error
        diagnostics = case_record.get("conversation_diagnostics")
        if (
            assessment.get("provenance") != case_record.get("provenance")
            or not isinstance(diagnostics, dict)
            or assessment.get("conversation_diagnostics") != diagnostics
        ):
            raise ArtifactIntegrityError(
                "assessment.json 的 v2 provenance/诊断绑定不一致"
            )
    return assessment


def _case_output_path(
    snapshot: dict[str, Any],
    case_record: dict[str, Any],
    kind: str = "tx",
) -> Path:
    if kind not in {"tx", "rx"}:
        raise ValueError("未知 VCP 输出方向")
    case_id = case_record["case_id"]
    batch_directory = _batch_directory(snapshot["batch_id"])
    cases_root = batch_directory / "cases"
    destination = _case_directory(snapshot["batch_id"], case_id)
    for path, label in (
        (batch_directory, "批次"),
        (cases_root, "case 根"),
        (destination, "case"),
    ):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FileNotFoundError(f"{label}目录不存在") from error
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"{label}目录不是普通目录")
    _verified_assessment(snapshot, case_record, destination)
    execution = case_record.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("条件组没有完成的 execution")
    alignment = execution.get("output_alignment", {}).get(kind, {})
    filename = alignment.get(
        "analysis_filename",
        "tx.after.wav" if kind == "tx" else "rx.rx-spk.wav",
    )
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise ArtifactIntegrityError(f"{kind.upper()} 输出文件名无效")
    path = destination / filename
    if path.is_symlink():
        raise ArtifactIntegrityError(f"{kind.upper()} 输出不能是符号链接")
    if not path.is_file():
        raise FileNotFoundError(f"{kind.upper()} 输出不存在或不是普通文件")
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ArtifactIntegrityError(f"{kind.upper()} 输出越出条件组目录")
    expected = execution.get("output_sha256", {}).get(kind)
    if not isinstance(expected, str) or _sha256_file(path) != expected:
        raise ArtifactIntegrityError(f"{kind.upper()} 输出哈希校验失败")
    return path


def output_path(
    identifier: str,
    case_id: str,
    kind: str = "tx",
) -> tuple[Path, str]:
    """Return a verified VCP output path and its frozen SHA-256."""
    if kind not in {"tx", "rx"}:
        raise ValueError("未知 VCP 输出方向")
    snapshot = get_batch(identifier)
    if snapshot is None:
        raise FileNotFoundError("场景条件批次不存在")
    case_record = next(
        (
            row
            for row in snapshot.get("cases", [])
            if row.get("case_id") == case_id
        ),
        None,
    )
    if case_record is None or case_record.get("status") != "completed":
        raise FileNotFoundError(
            f"条件组 {kind.upper()} 输出不存在或未执行成功"
        )
    path = _case_output_path(snapshot, case_record, kind)
    return path, case_record["execution"]["output_sha256"][kind]


def dynamic_microphone_path(
    identifier: str,
    case_id: str,
) -> tuple[Path, str]:
    """Return the verified runtime microphone consumed by the TX invocation."""
    snapshot = get_batch(identifier)
    if snapshot is None:
        raise FileNotFoundError("场景条件批次不存在")
    case_record = next(
        (
            row
            for row in snapshot.get("cases", [])
            if row.get("case_id") == case_id
        ),
        None,
    )
    if case_record is None or case_record.get("status") != "completed":
        raise FileNotFoundError("条件组动态麦克风不存在或未执行成功")
    destination = _case_directory(identifier, case_id)
    _verified_assessment(snapshot, case_record, destination)
    execution = case_record.get("execution")
    dynamic = (
        execution.get("dynamic_microphone")
        if isinstance(execution, dict)
        else None
    )
    if not isinstance(dynamic, dict):
        raise FileNotFoundError("该条件组没有运行时动态麦克风")
    raw_path = dynamic.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ArtifactIntegrityError("动态麦克风文件名无效")
    filename = Path(raw_path).name
    if not filename:
        raise ArtifactIntegrityError("动态麦克风文件名无效")
    path = destination / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("动态麦克风文件不存在")
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ArtifactIntegrityError("动态麦克风越出条件组目录")
    expected = dynamic.get("sha256")
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected) is None
        or _sha256_file(path) != expected
    ):
        raise ArtifactIntegrityError("动态麦克风哈希校验失败")
    return path, expected


def _case_worker_count(case_count: int) -> int:
    """Return the bounded number of condition groups executed concurrently."""
    if type(case_count) is not int or case_count < 0:
        raise ValueError("case_count must be a non-negative integer")
    raw = os.environ.get(CASE_WORKERS_ENV, str(DEFAULT_CASE_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{CASE_WORKERS_ENV} must be an integer between 1 and "
            f"{MAX_CASE_WORKERS}"
        ) from error
    if not 1 <= workers <= MAX_CASE_WORKERS:
        raise ValueError(
            f"{CASE_WORKERS_ENV} must be between 1 and {MAX_CASE_WORKERS}"
        )
    return min(workers, case_count) if case_count else 0


def _case_work_snapshot(
    snapshot: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Create an isolated snapshot for one case worker.

    Completed cases and unrelated observation rows are excluded so parallel
    workers never read the mutable parent snapshot or copy prior visual data.
    """
    cases = [
        case for case in snapshot.get("cases", [])
        if case.get("case_id") == case_id
    ]
    if len(cases) != 1:
        raise ValueError(f"条件组 {case_id} 在批次中不唯一")
    items = [
        deepcopy(row) for row in snapshot.get("items", [])
        if row.get("case_id") == case_id
    ]
    expected_item_ids = set(cases[0].get("contract_item_ids", []))
    if {row.get("item_id") for row in items} != expected_item_ids:
        raise ValueError(f"条件组 {case_id} 的观测行不完整")
    work = {
        key: deepcopy(value)
        for key, value in snapshot.items()
        if key not in {"cases", "items", "current_cases"}
    }
    work["cases"] = [deepcopy(cases[0])]
    work["items"] = items
    work["current_case"] = case_id
    return work


def _run_case_snapshot(
    work: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one isolated case snapshot without writing a batch snapshot."""
    case_record = work["cases"][0]
    case_id = case_record["case_id"]
    item_by_key = {
        (row["case_id"], row["item_id"]): row
        for row in work["items"]
    }
    _run_one_case(
        work,
        dataset,
        case_record,
        item_by_key,
        persist=False,
    )
    return (
        deepcopy(case_record),
        deepcopy(work["items"]),
    )


def _merge_case_result(
    snapshot: dict[str, Any],
    case_record: dict[str, Any],
    item_rows: list[dict[str, Any]],
) -> None:
    """Merge one isolated worker result into the parent in stable order."""
    case_id = case_record.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError("条件组结果缺少 case_id")
    with acceptance_batch._LOCK:
        case_indexes = [
            index for index, row in enumerate(snapshot.get("cases", []))
            if row.get("case_id") == case_id
        ]
        if len(case_indexes) != 1:
            raise ValueError(f"条件组 {case_id} 在父批次中不唯一")
        snapshot["cases"][case_indexes[0]] = deepcopy(case_record)

        row_indexes = {
            (row.get("case_id"), row.get("item_id")): index
            for index, row in enumerate(snapshot.get("items", []))
        }
        expected_item_ids = set(case_record.get("contract_item_ids", []))
        if {row.get("item_id") for row in item_rows} != expected_item_ids:
            raise ValueError(f"条件组 {case_id} 的并行结果行不完整")
        for row in item_rows:
            key = (case_id, row.get("item_id"))
            index = row_indexes.get(key)
            if index is None:
                raise ValueError(f"条件组 {case_id} 的并行结果行不属于父批次")
            snapshot["items"][index] = deepcopy(row)



def _adapt_nb_outputs_to_8k(
    outputs: dict[str, Path],
    execution: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """Restore 16 kHz engine outputs to 8 kHz for NB-mode scene-condition cases."""
    from scipy import signal as sp_signal
    import soundfile as sf

    for role in ("rx", "tx"):
        out_path = outputs.get(role)
        if out_path is None or not out_path.is_file():
            continue
        info = sf.info(out_path)
        if info.samplerate == 16_000:
            samples, _ = sf.read(out_path, dtype="float64", always_2d=True)
            adapted = sp_signal.resample_poly(samples, 1, 2, axis=0)
            subtype = "FLOAT" if out_path.name.endswith(".analysis.wav") else "PCM_16"
            sf.write(out_path, adapted, 8_000, subtype=subtype)
            new_sha = _sha256_file(out_path)
            if isinstance(execution.get("output_sha256"), dict):
                execution["output_sha256"][role] = new_sha
            alignment = execution.get("output_alignment", {}).get(role)
            if isinstance(alignment, dict) and "analysis_sha256" in alignment:
                alignment["analysis_sha256"] = new_sha
    rx_sha = execution.get("output_sha256", {}).get("rx")
    if isinstance(rx_sha, str) and isinstance(execution.get("echo_reference"), dict):
        execution["echo_reference"]["processed_sha256"] = rx_sha
        execution["echo_reference"]["tx_echo_argument_sha256"] = rx_sha

    dm = execution.get("dynamic_microphone")
    if isinstance(dm, dict):
        mic_path = Path(dm["path"])
        if mic_path.is_file() and sf.info(mic_path).samplerate == 16_000:
            mic_samples, _ = sf.read(mic_path, dtype="float64", always_2d=True)
            mic_adapted = sp_signal.resample_poly(mic_samples, 1, 2, axis=0)
            sf.write(mic_path, mic_adapted, 8_000, subtype="PCM_16")
            dm["sha256"] = _sha256_file(mic_path)
            dm["sample_rate"] = 8_000
            dm["frames"] = int(len(mic_adapted))
        for role_name, path_key, sha_key in (
            ("reference", "reference_path", "reference_sha256"),
            ("noise", "noise_path", "noise_sha256"),
        ):
            p = Path(dm[path_key])
            if float(dm.get("capture_gain", 1.0)) == 1.0:
                orig_p = paths[role_name]
                dm[path_key] = str(orig_p)
                dm[sha_key] = _sha256_file(orig_p)
            elif p.is_file() and sf.info(p).samplerate == 16_000:
                s, _ = sf.read(p, dtype="float64", always_2d=True)
                s_adapted = sp_signal.resample_poly(s, 1, 2, axis=0)
                sf.write(p, s_adapted, 8_000, subtype="PCM_16")
                dm[sha_key] = _sha256_file(p)
        if isinstance(rx_sha, str):
            dm["playback_sha256"] = rx_sha
        execution["effective_input_sha256"] = {
            "microphone": dm["sha256"],
            "reference": dm["reference_sha256"],
            "noise": dm["noise_sha256"],
            "far_end": rx_sha,
        }


def _run_one_case(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
    case_record: dict[str, Any],
    item_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    persist: bool = True,
) -> None:
    case_id = case_record["case_id"]
    dataset_case = next(
        case for case in dataset["cases"] if case["case_id"] == case_id
    )
    stress = is_stress_dataset(snapshot["dataset_id"])
    if stress:
        profile = stress_profile_for_dataset(snapshot["dataset_id"])
        schema_version = profile.schema_version
        conversation_mode = None
    else:
        profile = profile_for_dataset(snapshot["dataset_id"])
        schema_version = _profile_schema_version(profile)
        conversation_mode = (
            dataset_case.get("provenance", {}).get("conversation_mode")
            if schema_version == "scene-condition-preview-v2"
            else None
        )
    destination = _case_directory(snapshot["batch_id"], case_id)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        _raise_if_enospc(error)
        raise

    report_id = uuid.uuid4().hex
    item_ids = list(case_record["contract_item_ids"])
    case_record["status"] = "running"
    case_record["report_id"] = report_id
    for item_id in item_ids:
        item_by_key[(case_id, item_id)].update(
            status="running",
            value=None,
            metric=None,
            reason="对应场景正在执行",
            report_id=report_id,
        )
        item_by_key[(case_id, item_id)]["evidence_binding"]["report_id"] = (
            report_id
        )
    snapshot["current_case"] = case_id
    if persist:
        _save(snapshot)

    try:
        input_before = _hash_case_inputs(dataset, dataset_case)
        paths = {
            role: asset_path(dataset["root"], dataset_case["signals"][role])
            for role in ROLES
        }
        if stress:
            from .acoustic_stress_batch import execution_case as stress_execution_case

            execution_case = stress_execution_case(dataset_case, paths)
        else:
            execution_case = _execution_case(dataset_case)
        outputs, execution = acceptance_batch._run_cli(
            execution_case,
            paths,
            destination,
            snapshot["algorithm_cluster_id"],
        )
        if not acceptance_batch.is_native_rate_path_execution(execution) and (
            (stress and profile.mode.startswith("NB-"))
            or (
                not stress
                and profile_sample_rate_hz(profile) == 8_000
            )
        ):
            _adapt_nb_outputs_to_8k(outputs, execution, paths)
        effective_paths = dict(paths)
        dynamic_microphone = execution.get("dynamic_microphone")
        if isinstance(dynamic_microphone, dict):
            effective_paths["microphone"] = Path(
                dynamic_microphone["path"]
            )
            effective_paths["reference"] = Path(
                dynamic_microphone["reference_path"]
            )
            effective_paths["noise"] = Path(
                dynamic_microphone["noise_path"]
            )
            rx_path = outputs.get("rx")
            if rx_path is not None:
                effective_paths["far_end"] = rx_path
        _assert_execution_frozen(snapshot, execution, input_before)
        input_after = _hash_case_inputs(dataset, dataset_case)
        if input_after != input_before:
            raise ValueError("VCP 执行前后四路输入身份发生变化")
        output_sha256 = _actual_output_hashes(outputs, execution)
        execution.update(
            actual_input_sha256=deepcopy(input_after),
            actual_output_sha256=deepcopy(output_sha256),
            batch_id=snapshot["batch_id"],
            dataset_id=snapshot["dataset_id"],
        )

        conversation_diagnostics = None
        stress_diagnostics = None
        if stress:
            from .acoustic_stress_models import (
                calculate_stress_case as calculate_stress_rows,
                validate_activity_segments,
                validate_stress_diagnostics,
            )

            tx_path = outputs.get("tx")
            if tx_path is None:
                raise ValueError("stress execution lacks a real VCP TX")
            calculated_rows, stress_diagnostics = calculate_stress_rows(
                execution_case,
                effective_paths,
                outputs,
                execution,
                report_id=report_id,
                profile=profile,
            )
            activity = validate_activity_segments(
                dataset_case["provenance"],
                scope=profile.scope,
                mode=profile.mode,
            )
            validate_stress_diagnostics(
                stress_diagnostics,
                expected_case_id=case_id,
                expected_tx_sha256=output_sha256["tx"],
                expected_scope=profile.scope,
                expected_mode=profile.mode,
                expected_category=dataset_case["provenance"][
                    "workload_category"
                ],
                expected_segments=activity,
            )
        elif schema_version == "scene-condition-preview-v2":
            tx_path = outputs.get("tx")
            if tx_path is None:
                raise ValueError("v2 场景条件执行缺少真实 VCP TX")
            conversation_diagnostics = diagnose_conversation_case(
                dataset_case,
                effective_paths,
                tx_path,
                tx_sha256=output_sha256["tx"],
            )
            activity = validate_activity_metadata(
                dataset_case["provenance"]
            )
            validate_diagnostics(
                conversation_diagnostics,
                expected_mode=str(conversation_mode),
                expected_case_id=case_id,
                expected_tx_sha256=output_sha256["tx"],
                expected_activity=activity,
            )

        if not stress:
            calculation_case = execution_case
            no_mos_mode = conversation_mode in {"far_single", "transition"}
            if no_mos_mode:
                # Empty item_ids deliberately bypasses DNSMOS while still
                # using the same calculator to validate case/output identities.
                calculation_case = {**execution_case, "item_ids": []}
            calculated_rows = calculate_case_items(
                calculation_case,
                effective_paths,
                outputs,
                execution,
                report_id=report_id,
            )
            if no_mos_mode:
                if calculated_rows:
                    raise ValueError(
                        "far_single/transition 禁止产生 DNSMOS/GNS 数值行"
                    )
                if conversation_mode == "far_single":
                    no_mos_reason = (
                        "未计算 DNSMOS/G-MOS/N-MOS/S-MOS：far_single "
                        "没有持续近端语音，不满足 12 秒 P.835 近端窗口；"
                        "真实 VCP TX 与工程诊断已保留"
                    )
                else:
                    no_mos_reason = (
                        "未计算 DNSMOS/G-MOS/N-MOS/S-MOS：transition "
                        "由 far_only/near_only/double_talk/noise_only "
                        "混合短窗口组成，没有连续 12 秒近端语音；"
                        "不拼接短段冒充完整 P.835 窗口，真实 VCP TX "
                        "与逐段工程诊断已保留"
                    )
                calculated_rows = [
                    {
                        "item_id": item_id,
                        "value": None,
                        "status": "not_measured",
                        "metric": None,
                        "reason": no_mos_reason,
                    }
                    for item_id in item_ids
                ]
        if (
            len(calculated_rows) != len(item_ids)
            or {row.get("item_id") for row in calculated_rows} != set(item_ids)
        ):
            raise ValueError("计算模型返回的合同条目身份不一致")
        if _hash_case_inputs(dataset, dataset_case) != input_after:
            raise ValueError("指标计算期间四路输入身份发生变化")
        output_sha256 = _actual_output_hashes(outputs, execution)

        tx_path = outputs.get("tx")
        if tx_path is None or _sha256_file(tx_path) != output_sha256["tx"]:
            raise ValueError("TX 可视化前输出哈希校验失败")
        _resource_preflight(destination)
        from .acceptance_scene_condition_api import _visual_track

        try:
            rx_visual = None
            dynamic_microphone_visual = None
            rx_path = outputs.get("rx")
            if rx_path is not None:
                if _sha256_file(rx_path) != output_sha256["rx"]:
                    raise ValueError(
                        "RX 处理后可视化前输出哈希校验失败"
                    )
                rx_visual = _compact_visual(
                    _visual_track(rx_path, output_sha256["rx"])
                )
            tx_visual = _compact_visual(
                _visual_track(tx_path, output_sha256["tx"])
            )
            if isinstance(dynamic_microphone, dict):
                dynamic_microphone_visual = _compact_visual(
                    _visual_track(
                        Path(dynamic_microphone["path"]),
                        dynamic_microphone["sha256"],
                    )
                )
        except OSError as error:
            _raise_if_enospc(error)
            raise
        if _sha256_file(tx_path) != output_sha256["tx"]:
            raise ValueError("TX 可视化期间输出身份发生变化")

        normalized_rows = []
        for calculated in calculated_rows:
            item_id = calculated["item_id"]
            normalized = _normalized_item(
                item_by_key[(case_id, item_id)],
                calculated,
                snapshot=snapshot,
                case=dataset_case,
                report_id=report_id,
                input_sha256=input_after,
                output_sha256=output_sha256,
            )
            normalized_rows.append(normalized)
            item_by_key[(case_id, item_id)].update(normalized)

        assessment = {
            "schema": SCHEMA,
            "schema_version": SCHEMA,
            "batch_id": snapshot["batch_id"],
            "dataset_id": snapshot["dataset_id"],
            "dataset_name": snapshot["dataset_name"],
            "case_id": case_id,
            "report_id": report_id,
            "conditions": deepcopy(dataset_case["conditions"]),
            "input_sha256": deepcopy(input_after),
            "output_sha256": deepcopy(output_sha256),
            "execution": deepcopy(execution),
            "items": deepcopy(normalized_rows),
            "tx_visual": deepcopy(tx_visual),
            "standard_compliance": False,
            "formal_test_status": "not_measured",
        }
        if isinstance(dynamic_microphone, dict):
            assessment["dynamic_microphone"] = deepcopy(dynamic_microphone)
            assessment["effective_input_sha256"] = deepcopy(
                execution.get("effective_input_sha256", {})
            )
        if dynamic_microphone_visual is not None:
            assessment["dynamic_microphone_visual"] = deepcopy(
                dynamic_microphone_visual
            )
        if rx_visual is not None:
            assessment["rx_visual"] = deepcopy(rx_visual)
        if stress:
            assessment["provenance"] = deepcopy(dataset_case["provenance"])
            assessment["stress_diagnostics"] = deepcopy(stress_diagnostics)
        elif schema_version == "scene-condition-preview-v2":
            assessment["provenance"] = deepcopy(dataset_case["provenance"])
            assessment["conversation_diagnostics"] = deepcopy(
                conversation_diagnostics
            )
        _resource_preflight(destination)
        assessment_sha256 = _write_once_json(
            destination / "assessment.json",
            assessment,
        )
        case_update = {
            "status": "completed",
            "report_id": report_id,
            "input_sha256": deepcopy(input_after),
            "output_sha256": deepcopy(output_sha256),
            "execution": deepcopy(execution),
            "tx_visual": deepcopy(tx_visual),
            "assessment_sha256": assessment_sha256,
        }
        if isinstance(dynamic_microphone, dict):
            case_update["dynamic_microphone"] = deepcopy(
                dynamic_microphone
            )
            case_update["effective_input_sha256"] = deepcopy(
                execution.get("effective_input_sha256", {})
            )
        if dynamic_microphone_visual is not None:
            case_update["dynamic_microphone_visual"] = deepcopy(
                dynamic_microphone_visual
            )
        if rx_visual is not None:
            case_update["rx_visual"] = deepcopy(rx_visual)
        case_record.update(case_update)
        if stress:
            case_record["stress_diagnostics"] = deepcopy(stress_diagnostics)
        elif schema_version == "scene-condition-preview-v2":
            case_record["conversation_diagnostics"] = deepcopy(
                conversation_diagnostics
            )
    except ResourceSafetyStop:
        raise
    except Exception as error:
        if _is_enospc_error(error):
            raise ResourceSafetyStop(
                "ENOSPC reported by scene-condition case execution"
            ) from error
        message = f"执行失败: {type(error).__name__}: {error}"
        for item_id in item_ids:
            item_by_key[(case_id, item_id)].update(
                status="invalid",
                value=None,
                metric=None,
                reason=message,
                report_id=report_id,
                evidence_grade="invalid",
            )
            item_by_key[(case_id, item_id)]["evidence_binding"].update(
                report_id=report_id,
                input_sha256=deepcopy(case_record["signals_sha256"]),
                output_sha256={},
            )
        case_record.update(
            status="failed",
            report_id=report_id,
            error=message,
        )
    snapshot["progress"]["completed_cases"] += 1
    snapshot["progress"]["completed_items"] += len(item_ids)
    snapshot["progress"] = _progress(snapshot)
    if persist:
        _save(snapshot)


def _run_pending_cases(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
    item_by_key: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Execute pending cases serially or with bounded case-level parallelism."""
    pending = list(snapshot["cases"])
    for case_record in pending:
        if case_record.get("status") != "pending":
            raise ValueError("场景条件批次不是全新 pending 状态")
    workers = _case_worker_count(len(pending))
    if workers <= 1:
        for case_record in pending:
            _resource_preflight(_batch_directory(snapshot["batch_id"]))
            _run_one_case(
                snapshot,
                dataset,
                case_record,
                item_by_key,
            )
        return

    active: dict[Any, str] = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="scene-condition-case",
    ) as executor:
        try:
            while pending or active:
                while pending and len(active) < workers:
                    case_record = pending.pop(0)
                    case_id = case_record["case_id"]
                    work = _case_work_snapshot(snapshot, case_id)
                    future = executor.submit(
                        _run_case_snapshot,
                        work,
                        dataset,
                    )
                    active[future] = case_id
                    with acceptance_batch._LOCK:
                        parent = next(
                            row for row in snapshot["cases"]
                            if row.get("case_id") == case_id
                        )
                        parent["status"] = "running"
                        parent.pop("error", None)

                active_ids = sorted(active.values())
                snapshot["current_cases"] = active_ids
                snapshot["current_case"] = active_ids[0] if active_ids else None
                _save(snapshot)

                done, _ = wait(
                    tuple(active),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    active.pop(future)
                    case_record, item_rows = future.result()
                    _merge_case_result(
                        snapshot,
                        case_record,
                        item_rows,
                    )

                active_ids = sorted(active.values())
                snapshot["current_cases"] = active_ids
                snapshot["current_case"] = active_ids[0] if active_ids else None
                _save(snapshot)
        except BaseException:
            for future in active:
                future.cancel()
            raise


def run_batch(identifier: str) -> dict[str, Any] | None:
    """Execute the queued batch through the legacy single-worker executor."""
    with acceptance_batch._LOCK:
        other_active = acceptance_batch._ACTIVE - {identifier}
        if other_active or acceptance_batch._LEGACY_ACTIVE:
            acceptance_batch._ACTIVE.discard(identifier)
            raise ValueError("已有测试任务正在执行，禁止并发场景条件批次")
        acceptance_batch._ACTIVE.add(identifier)
    snapshot: dict[str, Any] | None = None
    try:
        snapshot = get_batch(identifier)
        if snapshot is None:
            raise ValueError("场景条件批次不存在")
        _validate_snapshot_contract(snapshot)
        if snapshot.get("status") != "queued":
            raise ValueError("场景条件批次不是 queued 状态")
        dataset, _ = _validated_dataset(
            snapshot["dataset_id"],
            requested_scene=snapshot["selected_scene"],
        )
        _verify_dataset_frozen(snapshot, dataset)
        item_by_key = {
            (row["case_id"], row["item_id"]): row
            for row in snapshot["items"]
        }
        case_count = snapshot["case_count"]
        observation_count = snapshot["observation_count"]
        if len(item_by_key) != observation_count:
            raise ValueError("场景条件 item 索引不完整")

        snapshot["status"] = "running"
        snapshot.pop("error", None)
        _save(snapshot)
        _run_pending_cases(snapshot, dataset, item_by_key)

        if (
            len(snapshot["cases"]) != case_count
            or len(snapshot["items"]) != observation_count
            or any(
                case["status"] not in _TERMINAL_CASE_STATUSES
                for case in snapshot["cases"]
            )
            or any(
                row["status"] in {"pending", "running"}
                for row in snapshot["items"]
            )
        ):
            raise ValueError("场景条件批次未完整收口")
        failed_cases = sum(
            case["status"] == "failed" for case in snapshot["cases"]
        )
        snapshot["status"] = "failed" if failed_cases else "completed"
        snapshot.pop("current_case", None)
        snapshot.pop("current_cases", None)
        if failed_cases:
            snapshot["error"] = (
                f"{failed_cases} 个条件组执行失败；"
                "已完成证据与逐项失败原因均已保留"
            )
        _save(snapshot)
        return deepcopy(snapshot)
    except ResourceSafetyStop:
        # The floor is now closed: do not write another byte.  Once the active
        # identifier is released, get_batch() returns a derived failed view.
        if snapshot is not None:
            snapshot["status"] = "failed"
            snapshot["error"] = (
                "资源安全线触发，批次已中断；最后可读快照已保留"
            )
            snapshot.pop("current_case", None)
            snapshot.pop("current_cases", None)
            for case in snapshot.get("cases", []):
                if case.get("status") not in _TERMINAL_CASE_STATUSES:
                    case["status"] = "failed"
                    case["error"] = snapshot["error"]
            for row in snapshot.get("items", []):
                if row.get("status") in {"pending", "running"}:
                    row.update(
                        status="invalid",
                        value=None,
                        reason=snapshot["error"],
                        evidence_grade="invalid",
                    )
        return snapshot
    except Exception as error:
        if snapshot is None:
            raise
        message = f"{type(error).__name__}: {error}"
        snapshot["status"] = "failed"
        snapshot["error"] = message
        snapshot.pop("current_case", None)
        snapshot.pop("current_cases", None)
        for case in snapshot["cases"]:
            if case.get("status") not in _TERMINAL_CASE_STATUSES:
                case.update(status="failed", error=message)
        for row in snapshot["items"]:
            if row.get("status") in {"pending", "running"}:
                row.update(
                    status="invalid",
                    value=None,
                    reason=message,
                    evidence_grade="invalid",
                )
        try:
            _save(snapshot)
        except (ResourceSafetyStop, OSError, TypeError, ValueError):
            pass
        return snapshot
    finally:
        with acceptance_batch._LOCK:
            acceptance_batch._ACTIVE.discard(identifier)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def list_batches(
    limit: int = 20,
    dataset_id: str | None = None,
    algorithm_cluster_id: str | None = None,
    latest_per_case: bool = False,
) -> list[dict[str, Any]]:
    """List snapshots, optionally selecting the newest coverage per case."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 20
    ):
        raise ValueError("limit 必须为 1 到 20 的整数")
    if not isinstance(latest_per_case, bool):
        raise ValueError("latest_per_case 必须为布尔值")
    if dataset_id is not None and (
        not isinstance(dataset_id, str)
        or _BATCH_ID.fullmatch(dataset_id) is None
    ):
        raise ValueError("dataset_id 必须为 32 位小写十六进制")
    if algorithm_cluster_id is not None and (
        not isinstance(algorithm_cluster_id, str)
        or not algorithm_cluster_id
    ):
        raise ValueError("algorithm_cluster_id 无效")
    if latest_per_case and (
        dataset_id is None or algorithm_cluster_id is None
    ):
        raise ValueError(
            "latest_per_case=true 时必须同时指定 "
            "dataset_id 和 algorithm_cluster_id"
        )

    expected_case_ids: set[str] | None = None
    if latest_per_case:
        dataset, _ = _validated_dataset(dataset_id)
        expected_case_ids = {
            case["case_id"]
            for case in dataset["cases"]
        }
        if is_stress_dataset(dataset_id):
            profile_case_count = stress_profile_for_dataset(
                dataset_id
            ).case_count
        else:
            profile_case_count = _profile_counts(
                profile_for_dataset(dataset_id)
            )[0]
        if len(expected_case_ids) != profile_case_count:
            raise ValueError(
                "latest_per_case 的源数据集 case 数与 profile 不一致"
            )

    root = _batch_root()
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise OSError("场景条件批次历史目录不是普通目录")

    rows: list[tuple[datetime, str, dict[str, Any]]] = []
    for directory in root.iterdir():
        if (
            _BATCH_ID.fullmatch(directory.name) is None
            or directory.is_symlink()
        ):
            continue
        if _is_fixture_batch(directory):
            continue
        try:
            snapshot = get_batch(directory.name)
        except (OSError, ValueError):
            continue
        if snapshot is None:
            continue
        if dataset_id is not None and snapshot.get("dataset_id") != dataset_id:
            continue
        if (
            algorithm_cluster_id is not None
            and snapshot.get("algorithm_cluster_id") != algorithm_cluster_id
        ):
            continue
        created_at = _timestamp(snapshot.get("created_at"))
        if created_at is None:
            continue
        rows.append((created_at, directory.name, snapshot))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    if latest_per_case:
        assert expected_case_ids is not None
        selected: list[dict[str, Any]] = []
        covered: set[str] = set()
        for _, _, snapshot in rows:
            case_ids = {
                case["case_id"]
                for case in snapshot.get("cases", [])
            }
            if not case_ids or not case_ids.issubset(expected_case_ids):
                continue
            new_case_ids = case_ids - covered
            if not new_case_ids:
                continue
            selected.append(snapshot)
            covered.update(new_case_ids)
            if covered == expected_case_ids:
                break
        return selected
    return [snapshot for _, _, snapshot in rows[:limit]]


__all__ = [
    "ArtifactIntegrityError",
    "BATCH_ROOT_NAME",
    "CASE_COMPLETED_FIELDS",
    "CASE_REQUIRED_FIELDS",
    "EVIDENCE_BINDING_FIELDS",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_OBSERVATION_COUNT",
    "FIXTURE_MARKER_FIELDS",
    "FIXTURE_MARKER_NAME",
    "FIXTURE_MARKER_REASON",
    "FIXTURE_MARKER_SCHEMA",
    "ITEM_CORE_FIELDS",
    "PROGRESS_FIELDS",
    "ResourceSafetyStop",
    "SCHEMA",
    "SINGLE_CASE_EXECUTION_SCOPE",
    "SINGLE_CASE_SELECTION_FIELDS",
    "SNAPSHOT_COMPAT_FIELDS",
    "SNAPSHOT_REQUIRED_FIELDS",
    "SUMMARY_FIELDS",
    "TX_VISUAL_FIELDS",
    "get_batch",
    "list_batches",
    "output_path",
    "progress_counts",
    "run_batch",
    "start_batch",
    "summarize_items",
    "validate_snapshot",
]
