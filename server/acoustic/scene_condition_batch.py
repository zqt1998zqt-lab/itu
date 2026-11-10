"""Append-only VCP batches for the 20-case scene-condition dataset.

This module deliberately lives beside, rather than inside, the legacy
``acceptance_batch`` engine.  It reuses the legacy CLI runner and case metric
calculator, but gives every scene-condition case its own observation identity
so five contract rows can be measured twenty times without collapsing them.
"""
from __future__ import annotations

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


SCHEMA = "scene-condition-batch-v1"
BATCH_ROOT_NAME = "scene-condition-batches"
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3
EXPECTED_CASE_COUNT = 20
EXPECTED_SPEECH_CASE_COUNT = 10
EXPECTED_CNG_CASE_COUNT = 10
EXPECTED_OBSERVATION_COUNT = 50

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


def _validated_dataset(dataset_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    # Import lazily to keep the API -> backend module graph acyclic.
    from .acceptance_scene_condition_api import _load_dataset

    try:
        dataset = _load_dataset(dataset_id)
    except HTTPException:
        raise
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("场景条件测试集必须恰好包含 20 个 case")
    if (
        dataset.get("contract_item_count") != CONTRACT_ITEM_COUNT
        or CONTRACT_ITEM_COUNT != 5
    ):
        raise ValueError("场景条件测试集合同项数量不是 5")

    kinds = {"speech": 0, "cng": 0}
    seen_case_ids: set[str] = set()
    covered_items: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            or case_id in seen_case_ids
        ):
            raise ValueError("场景条件 case ID 无效或重复")
        seen_case_ids.add(case_id)
        if set(case.get("signals", {})) != set(ROLES):
            raise ValueError(f"{case_id} 必须恰好包含四路输入角色")
        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or len(item_ids) not in {2, 3}
            or len(set(item_ids)) != len(item_ids)
        ):
            raise ValueError(f"{case_id} 合同项身份无效")
        covered_items.update(item_ids)
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

    if (
        kinds["speech"] != EXPECTED_SPEECH_CASE_COUNT
        or kinds["cng"] != EXPECTED_CNG_CASE_COUNT
    ):
        raise ValueError("场景条件测试集必须为 10 个 speech 和 10 个 CNG")
    contract_rows = background_noise_scene_items(MODE, SCENE)
    contract_by_id = {row["id"]: row for row in contract_rows}
    if len(contract_by_id) != CONTRACT_ITEM_COUNT:
        raise ValueError("Sce1/WB-wired 合同项不是唯一的 5 项")
    if covered_items != set(contract_by_id):
        raise ValueError("场景条件 case 未完整覆盖五个合同项")
    speech_item_ids = {
        row["id"]
        for row in contract_rows
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    }
    cng_item_ids = {
        row["id"]
        for row in contract_rows
        if row["family"] in {"cng_level", "cng_spectrum"}
    }
    for case in cases:
        kind = case["provenance"]["render_metadata"]["kind"]
        expected_items = (
            speech_item_ids if kind == "speech" else cng_item_ids
        )
        if set(case["contract_item_ids"]) != expected_items:
            raise ValueError(
                f"{case['case_id']} 的 {kind} 合同项映射不符合 B24-B28"
            )
    return dataset, contract_by_id


def _execution_case(case: dict[str, Any]) -> dict[str, Any]:
    provenance = case["provenance"]
    render_metadata = provenance["render_metadata"]
    return {
        "id": case["case_id"],
        "conditions": deepcopy(case["conditions"]),
        "item_ids": list(case["contract_item_ids"]),
        "beam_delays_ms": list(render_metadata["beam_delays_ms"]),
        "analysis_start_s": provenance["analysis_start_s"],
    }


def _initial_items(
    dataset: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    identifier: str,
) -> list[dict[str, Any]]:
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
                    "mode": MODE,
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
    if len(items) != EXPECTED_OBSERVATION_COUNT:
        raise ValueError("场景条件批次没有形成 50 个 observation")
    observation_ids = {row["observation_id"] for row in items}
    if len(observation_ids) != EXPECTED_OBSERVATION_COUNT:
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
        }
        for case in dataset["cases"]
    ]


def _build_snapshot(
    dataset: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    identifier: str,
    algorithm_cluster_id: str,
    configuration: dict[str, Any],
    catalog_sha256: str,
    cli_sha256: str,
) -> dict[str, Any]:
    timestamp = _now()
    items = _initial_items(dataset, contract_by_id, identifier)
    cases = _initial_cases(dataset)
    return {
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
        "selected_mode": MODE,
        "selected_scene": SCENE,
        "test_scope": TEST_SCOPE,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "progress": {
            "completed_cases": 0,
            "total_cases": EXPECTED_CASE_COUNT,
            "completed_items": 0,
            "total_items": EXPECTED_OBSERVATION_COUNT,
        },
        "summary": _summary(items),
        "items": items,
        "cases": cases,
    }


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


def start_batch(
    dataset_id: str,
    algorithm_cluster_id: str,
    *,
    mode: str = MODE,
    scene: str = SCENE,
) -> dict[str, Any]:
    """Validate, reserve the shared executor, then persist a queued batch."""
    if mode != MODE:
        raise ValueError(f"场景条件批次仅支持 {MODE}")
    if scene != SCENE:
        raise ValueError(f"场景条件批次仅支持 {SCENE}")

    _resource_preflight()
    from server.alignment_runner import (
        resolve_algorithm_cluster,
        resolve_cli,
    )

    cli = resolve_cli()
    if cli is None:
        raise ValueError("未找到可执行的 VCP CLI")
    preset, _, configuration, catalog_sha256 = resolve_algorithm_cluster(
        algorithm_cluster_id
    )
    profile = preset.get("profile")
    if not isinstance(profile, str) or not profile.startswith("vcp"):
        raise ValueError("场景条件批次只允许 VCP 算法簇")
    cli_sha256 = _sha256_file(cli)

    dataset, contract_by_id = _validated_dataset(dataset_id)
    identifier = uuid.uuid4().hex
    snapshot = _build_snapshot(
        dataset,
        contract_by_id,
        identifier,
        algorithm_cluster_id,
        configuration,
        catalog_sha256,
        cli_sha256,
    )

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
    if (
        snapshot.get("contract_item_count") != CONTRACT_ITEM_COUNT
        or snapshot.get("observation_count") != EXPECTED_OBSERVATION_COUNT
        or len(snapshot.get("cases", [])) != EXPECTED_CASE_COUNT
        or len(snapshot.get("items", [])) != EXPECTED_OBSERVATION_COUNT
    ):
        raise ValueError("场景条件快照 20 case/50 observation 合同无效")
    if (
        snapshot.get("selected_mode") != MODE
        or snapshot.get("selected_scene") != SCENE
        or snapshot.get("test_scope") != TEST_SCOPE
        or snapshot.get("standard_compliance") is not False
        or snapshot.get("formal_test_status") != "not_measured"
    ):
        raise ValueError("场景条件快照执行范围或合规边界无效")
    item_keys = {
        (row.get("case_id"), row.get("item_id"))
        for row in snapshot["items"]
    }
    if len(item_keys) != EXPECTED_OBSERVATION_COUNT:
        raise ValueError("场景条件快照 observation 不唯一")
    cases = snapshot["cases"]
    case_ids = [case.get("case_id") for case in cases]
    if (
        any(
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            for case_id in case_ids
        )
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
    ):
        raise ValueError("场景条件快照 case ID 无效或重复")
    case_by_id = {case["case_id"]: case for case in cases}
    covered_items: set[str] = set()
    kinds = {"speech": 0, "cng": 0}
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
        if not isinstance(case.get("conditions"), dict):
            raise ValueError("场景条件 case 缺少完整 conditions")
        item_ids = case.get("contract_item_ids")
        if (
            not isinstance(item_ids, list)
            or len(item_ids) not in {2, 3}
            or len(set(item_ids)) != len(item_ids)
        ):
            raise ValueError("场景条件 case 合同项身份无效")
        covered_items.update(item_ids)
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
                or set(output_sha256) != {"tx"}
                or not isinstance(output_sha256.get("tx"), str)
                or re.fullmatch(r"[a-f0-9]{64}", output_sha256["tx"]) is None
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
            execution = case["execution"]
            if (
                execution.get("algorithm_cluster_id")
                != snapshot["algorithm_cluster_id"]
                or execution.get("configuration")
                != snapshot["configuration"]
                or execution.get("catalog_sha256")
                != snapshot["catalog_sha256"]
                or execution.get("cli_sha256") != snapshot["cli_sha256"]
                or execution.get("output_sha256") != output_sha256
            ):
                raise ValueError("完成 case 的执行身份与批次冻结配置不一致")
            if (
                case["tx_visual"].get("source_sha256")
                != output_sha256["tx"]
            ):
                raise ValueError("完成 case 的可视化源哈希与 TX 不一致")
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
        if row.get("observation_id") != f"{case_id}:{item_id}":
            raise ValueError("场景条件 observation_id 不符合 case:item 契约")
        if row.get("mode") != MODE:
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
            or set(binding["output_sha256"]) != {"tx"}
        ):
            raise ValueError("review_required item 必须绑定 TX 输出")
    if kinds != {
        "speech": EXPECTED_SPEECH_CASE_COUNT,
        "cng": EXPECTED_CNG_CASE_COUNT,
    }:
        raise ValueError("场景条件快照不是 10 speech/10 CNG")
    contract_rows = background_noise_scene_items(MODE, SCENE)
    if covered_items != {row["id"] for row in contract_rows}:
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
    for field in (
        "dataset_id",
        "dataset_manifest_sha256",
        "dataset_assets_sha256",
    ):
        actual = dataset.get(field)
        expected = snapshot.get(field)
        if actual != expected:
            raise ValueError(f"批次开始后测试集身份发生变化: {field}")
    expected_cases = {
        case["case_id"]: case for case in dataset["cases"]
    }
    snapshot_cases = {
        case["case_id"]: case for case in snapshot["cases"]
    }
    if set(expected_cases) != set(snapshot_cases):
        raise ValueError("批次开始后场景条件 case 集合发生变化")
    for case_id, case in expected_cases.items():
        if snapshot_cases[case_id].get("signals_sha256") != case["signals_sha256"]:
            raise ValueError(f"批次开始后 {case_id} 输入身份发生变化")


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
    if execution.get("algorithm_cluster_id") != snapshot["algorithm_cluster_id"]:
        raise ValueError("执行期间算法簇发生变化")
    if execution.get("configuration") != snapshot["configuration"]:
        raise ValueError("执行期间算法簇 configuration 发生变化")
    if execution.get("catalog_sha256") != snapshot["catalog_sha256"]:
        raise ValueError("执行期间算法目录 catalog_sha256 发生变化")
    if execution.get("cli_sha256") != snapshot["cli_sha256"]:
        raise ValueError("执行期间 VCP CLI 发生变化")
    if execution.get("input_sha256") != input_sha256:
        raise ValueError("执行器报告的输入哈希与实际四路输入不一致")
    outputs = execution.get("output_sha256")
    if not isinstance(outputs, dict) or set(outputs) != {"tx"}:
        raise ValueError("执行器必须且只能报告一个 TX 输出")


def _actual_output_hashes(
    outputs: dict[str, Path],
    execution: dict[str, Any],
) -> dict[str, str]:
    actual = {role: _sha256_file(path) for role, path in outputs.items()}
    if actual != execution.get("output_sha256"):
        raise ValueError("执行期间 VCP TX 输出身份发生变化")
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
        mode=MODE,
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
        or case_record.get("input_sha256")
        != case_record.get("signals_sha256")
        or assessment.get("input_sha256")
        != case_record.get("input_sha256")
        or assessment.get("output_sha256") != case_record.get("output_sha256")
    ):
        raise ArtifactIntegrityError(
            "assessment.json 的 case/report/batch/dataset/input/output "
            "绑定不一致"
        )
    return assessment


def _case_output_path(
    snapshot: dict[str, Any],
    case_record: dict[str, Any],
) -> Path:
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
    alignment = execution.get("output_alignment", {}).get("tx", {})
    filename = alignment.get(
        "analysis_filename",
        "tx.after.wav",
    )
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise ArtifactIntegrityError("TX 输出文件名无效")
    path = destination / filename
    if path.is_symlink():
        raise ArtifactIntegrityError("TX 输出不能是符号链接")
    if not path.is_file():
        raise FileNotFoundError("TX 输出不存在或不是普通文件")
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ArtifactIntegrityError("TX 输出越出条件组目录")
    expected = execution.get("output_sha256", {}).get("tx")
    if not isinstance(expected, str) or _sha256_file(path) != expected:
        raise ArtifactIntegrityError("TX 输出哈希校验失败")
    return path


def output_path(
    identifier: str,
    case_id: str,
) -> tuple[Path, str]:
    """Return a verified TX path and its frozen SHA-256."""
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
        raise FileNotFoundError("条件组 TX 输出不存在或未执行成功")
    path = _case_output_path(snapshot, case_record)
    return path, case_record["execution"]["output_sha256"]["tx"]


def _run_one_case(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
    case_record: dict[str, Any],
    item_by_key: dict[tuple[str, str], dict[str, Any]],
) -> None:
    case_id = case_record["case_id"]
    dataset_case = next(
        case for case in dataset["cases"] if case["case_id"] == case_id
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
    _save(snapshot)

    try:
        input_before = _hash_case_inputs(dataset, dataset_case)
        paths = {
            role: asset_path(dataset["root"], dataset_case["signals"][role])
            for role in ROLES
        }
        execution_case = _execution_case(dataset_case)
        outputs, execution = acceptance_batch._run_cli(
            execution_case,
            paths,
            destination,
            snapshot["algorithm_cluster_id"],
        )
        _assert_execution_frozen(snapshot, execution, input_before)
        input_after = _hash_case_inputs(dataset, dataset_case)
        if input_after != input_before:
            raise ValueError("VCP 执行前后四路输入身份发生变化")
        output_sha256 = _actual_output_hashes(outputs, execution)
        execution.update(
            actual_input_sha256=deepcopy(input_after),
            actual_output_sha256=deepcopy(output_sha256),
        )

        calculated_rows = calculate_case_items(
            execution_case,
            paths,
            outputs,
            execution,
            report_id=report_id,
        )
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
            tx_visual = _compact_visual(
                _visual_track(tx_path, output_sha256["tx"])
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
        _resource_preflight(destination)
        assessment_sha256 = _write_once_json(
            destination / "assessment.json",
            assessment,
        )
        case_record.update(
            status="completed",
            report_id=report_id,
            input_sha256=deepcopy(input_after),
            output_sha256=deepcopy(output_sha256),
            execution=deepcopy(execution),
            tx_visual=deepcopy(tx_visual),
            assessment_sha256=assessment_sha256,
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
    _save(snapshot)


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
        dataset, _ = _validated_dataset(snapshot["dataset_id"])
        _verify_dataset_frozen(snapshot, dataset)
        item_by_key = {
            (row["case_id"], row["item_id"]): row
            for row in snapshot["items"]
        }
        if len(item_by_key) != EXPECTED_OBSERVATION_COUNT:
            raise ValueError("场景条件 item 索引不完整")

        snapshot["status"] = "running"
        snapshot.pop("error", None)
        _save(snapshot)
        for case_record in snapshot["cases"]:
            if case_record.get("status") != "pending":
                raise ValueError("场景条件批次不是全新 pending 状态")
            _resource_preflight(
                _batch_directory(snapshot["batch_id"])
            )
            _run_one_case(
                snapshot,
                dataset,
                case_record,
                item_by_key,
            )

        if (
            len(snapshot["cases"]) != EXPECTED_CASE_COUNT
            or len(snapshot["items"]) != EXPECTED_OBSERVATION_COUNT
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
) -> list[dict[str, Any]]:
    """List complete new-store snapshots, newest first."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 20
    ):
        raise ValueError("limit 必须为 1 到 20 的整数")
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
    return [snapshot for _, _, snapshot in rows[:limit]]


__all__ = [
    "ArtifactIntegrityError",
    "BATCH_ROOT_NAME",
    "CASE_COMPLETED_FIELDS",
    "CASE_REQUIRED_FIELDS",
    "EVIDENCE_BINDING_FIELDS",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_OBSERVATION_COUNT",
    "ITEM_CORE_FIELDS",
    "PROGRESS_FIELDS",
    "ResourceSafetyStop",
    "SCHEMA",
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
