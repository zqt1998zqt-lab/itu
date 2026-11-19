"""Explicit retirement gate for superseded acoustic dataset identities."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import parse_qsl

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import dataset_revisions_v2 as _active_revision
from .scene_condition_profiles import profile_for_scene as _legacy_profile_for_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETIREMENT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "fixtures"
    / "retired-acoustic-metadata-v1"
    / "retirement.json"
)

RETIREMENT_SCHEMA = "itu-dataset-retirement-v1"
ACTIVE_REVISION = _active_revision.DATASET_REVISION
MAX_JSON_BODY_BYTES = 64 * 1024
ACCEPTANCE_PREFIX = "/api/alignment/acceptance"

ACTIVE_R2_DATASET_IDS = frozenset(
    profile.dataset_id
    for profile in (
        *_active_revision.REVISION_SCENE_PROFILES.values(),
        *_active_revision.REVISION_STRESS_PROFILES.values(),
    )
)

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "active_revision",
        "retired_dataset_ids",
        "replacement_dataset_ids",
    }
)
_OPTIONAL_FIELDS = frozenset({"metadata_archive_manifest_sha256"})
_CREATE_BATCH_SUFFIXES = frozenset(
    {
        "/scene-condition-batches",
        "/batches",
        "/basic-uplink/batches",
        "/basic-downlink/batches",
        "/echo/batches",
        "/double-talk/batches",
        "/background-noise/batches",
    }
)


class DatasetRetirementError(ValueError):
    """The retirement receipt is malformed and must fail closed."""


@dataclass(frozen=True)
class RetirementReceipt:
    """Validated immutable retirement policy loaded once at app installation."""

    active_revision: str
    retired_dataset_ids: frozenset[str]
    replacement_dataset_ids: Mapping[str, str]
    metadata_archive_manifest_sha256: str | None = None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _validate_dataset_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise DatasetRetirementError(f"{label} must be a 32-character dataset ID")
    return value


def _validate_receipt(value: dict[str, Any]) -> RetirementReceipt:
    fields = set(value)
    missing = sorted(_REQUIRED_FIELDS - fields)
    extra = sorted(fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if missing or extra:
        raise DatasetRetirementError(
            "retirement receipt fields differ "
            f"(missing={missing}, extra={extra})"
        )
    if value.get("schema") != RETIREMENT_SCHEMA:
        raise DatasetRetirementError("retirement receipt schema differs")
    if value.get("active_revision") != ACTIVE_REVISION:
        raise DatasetRetirementError(
            "retirement receipt active_revision differs"
        )

    retired_value = value.get("retired_dataset_ids")
    if not isinstance(retired_value, list) or not retired_value:
        raise DatasetRetirementError(
            "retired_dataset_ids must be a non-empty list"
        )
    retired: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(retired_value):
        dataset_id = _validate_dataset_id(
            item,
            f"retired_dataset_ids[{index}]",
        )
        if dataset_id in seen:
            raise DatasetRetirementError(
                f"retired_dataset_ids contains a duplicate: {dataset_id}"
            )
        seen.add(dataset_id)
        retired.append(dataset_id)

    overlap = seen & ACTIVE_R2_DATASET_IDS
    if overlap:
        raise DatasetRetirementError(
            "active R2 dataset IDs cannot be retired: "
            f"{sorted(overlap)}"
        )

    replacements_value = value.get("replacement_dataset_ids")
    if not isinstance(replacements_value, dict):
        raise DatasetRetirementError(
            "replacement_dataset_ids must be a JSON object"
        )
    replacements: dict[str, str] = {}
    for old_id, replacement in replacements_value.items():
        old_id = _validate_dataset_id(
            old_id,
            "replacement_dataset_ids key",
        )
        replacement = _validate_dataset_id(
            replacement,
            f"replacement_dataset_ids[{old_id}]",
        )
        if replacement in seen:
            raise DatasetRetirementError(
                f"replacement for {old_id} is itself retired"
            )
        replacements[old_id] = replacement
    if set(replacements) != seen:
        missing_replacements = sorted(seen - set(replacements))
        extra_replacements = sorted(set(replacements) - seen)
        raise DatasetRetirementError(
            "replacement_dataset_ids must cover exactly the retired IDs "
            f"(missing={missing_replacements}, "
            f"extra={extra_replacements})"
        )

    metadata_sha256 = value.get("metadata_archive_manifest_sha256")
    if (
        metadata_sha256 is not None
        and (
            not isinstance(metadata_sha256, str)
            or _SHA256_PATTERN.fullmatch(metadata_sha256) is None
        )
    ):
        raise DatasetRetirementError(
            "metadata_archive_manifest_sha256 must be a lowercase SHA-256"
        )

    return RetirementReceipt(
        active_revision=ACTIVE_REVISION,
        retired_dataset_ids=frozenset(seen),
        replacement_dataset_ids=MappingProxyType(replacements),
        metadata_archive_manifest_sha256=metadata_sha256,
    )


def load_retirement_receipt(
    path: Path | None = None,
) -> RetirementReceipt | None:
    """Load and validate the retirement policy; a missing file is inactive."""
    receipt_path = Path(RETIREMENT_PATH if path is None else path)
    try:
        metadata = receipt_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise DatasetRetirementError(
            f"retirement receipt path cannot be inspected: {receipt_path}: "
            f"{error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or receipt_path.is_symlink():
        raise DatasetRetirementError(
            f"retirement receipt is not a regular file: {receipt_path}"
        )
    try:
        text = receipt_path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise DatasetRetirementError(
            f"retirement receipt cannot be read: {receipt_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DatasetRetirementError(
            "retirement receipt must be a JSON object"
        )
    return _validate_receipt(value)


class _ReplayReceive:
    def __init__(self, messages: list[Message], receive: Receive) -> None:
        self._messages = messages
        self._receive = receive

    async def __call__(self) -> Message:
        if self._messages:
            return self._messages.pop(0)
        return await self._receive()


def _path_without_trailing_slash(path: str) -> str:
    return path.rstrip("/") or "/"


def _is_acceptance_path(path: str) -> bool:
    return path == ACCEPTANCE_PREFIX or path.startswith(
        f"{ACCEPTANCE_PREFIX}/"
    )


def _query_values(scope: Scope, key: str) -> list[str]:
    query = scope.get("query_string", b"").decode("latin-1")
    return [
        value
        for name, value in parse_qsl(query, keep_blank_values=True)
        if name == key
    ]


def _retired_path_dataset(
    path: str,
    retired_dataset_ids: frozenset[str],
) -> str | None:
    return next(
        (
            segment
            for segment in path.split("/")
            if segment in retired_dataset_ids
        ),
        None,
    )


def _retired_query_dataset(
    scope: Scope,
    retired_dataset_ids: frozenset[str],
) -> str | None:
    return next(
        (
            value
            for value in _query_values(scope, "dataset_id")
            if value in retired_dataset_ids
        ),
        None,
    )


def _is_batch_history_read(method: str, path: str) -> bool:
    if method != "GET":
        return False
    relative = path.removeprefix(ACCEPTANCE_PREFIX)
    return (
        relative == "/batches"
        or relative.startswith("/batches/")
        or relative == "/scene-condition-batches"
        or relative.startswith("/scene-condition-batches/")
    )


def _is_create_batch_path(path: str) -> bool:
    relative = path.removeprefix(ACCEPTANCE_PREFIX)
    if relative in _CREATE_BATCH_SUFFIXES:
        return True
    components = relative.strip("/").split("/")
    return (
        len(components) >= 3
        and components[-3] == "scenes"
        and bool(components[-2])
        and components[-1] == "batches"
    )


def _is_json_content_type(headers: Headers) -> bool:
    media_type = (
        headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    return media_type == "application/json" or media_type.endswith("+json")


def _declares_oversized_body(headers: Headers) -> bool:
    value = headers.get("content-length")
    if value is None:
        return False
    try:
        return int(value) > MAX_JSON_BODY_BYTES
    except ValueError:
        return False


async def _buffer_json_body(
    receive: Receive,
) -> tuple[bytes | None, Receive]:
    messages: list[Message] = []
    body = bytearray()
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            return None, _ReplayReceive(messages, receive)
        chunk = message.get("body", b"")
        if not isinstance(chunk, (bytes, bytearray)):
            return None, _ReplayReceive(messages, receive)
        body.extend(chunk)
        if len(body) > MAX_JSON_BODY_BYTES:
            return None, _ReplayReceive(messages, receive)
        if not message.get("more_body", False):
            return bytes(body), _ReplayReceive(messages, receive)


def _legacy_retirement(
    scope: Scope,
    receipt: RetirementReceipt,
) -> tuple[str, str] | None:
    if _path_without_trailing_slash(scope.get("path", "")) != (
        f"{ACCEPTANCE_PREFIX}/scene-condition-datasets"
    ):
        return None
    versions = _query_values(scope, "version")
    modes = _query_values(scope, "mode")
    scenes = _query_values(scope, "scene")
    if versions != ["legacy"] or len(modes) != 1 or len(scenes) != 1:
        return None
    try:
        legacy_id = _legacy_profile_for_scene(scenes[0]).dataset_id
        replacement = _active_revision.revision_scene_profile(
            scenes[0],
            modes[0],
        ).dataset_id
    except (KeyError, TypeError, ValueError):
        return None
    if replacement not in ACTIVE_R2_DATASET_IDS:
        return None
    return legacy_id, replacement


async def _send_retirement(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    dataset_id: str,
    replacement_dataset_id: str,
    detail: str,
) -> None:
    response = JSONResponse(
        status_code=410,
        content={
            "detail": detail,
            "dataset_id": dataset_id,
            "replacement_dataset_id": replacement_dataset_id,
        },
        headers={"Cache-Control": "no-store"},
    )
    await response(scope, receive, send)


class RetiredDatasetMiddleware:
    """Reject new use of retired identities while preserving batch history."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        receipt: RetirementReceipt,
    ) -> None:
        self.app = app
        self.receipt = receipt

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        path = _path_without_trailing_slash(scope.get("path", ""))
        if not _is_acceptance_path(path):
            await self.app(scope, receive, send)
            return

        if method == "GET":
            legacy = _legacy_retirement(scope, self.receipt)
            if legacy is not None:
                dataset_id, replacement = legacy
                await _send_retirement(
                    scope,
                    receive,
                    send,
                    dataset_id=dataset_id,
                    replacement_dataset_id=replacement,
                    detail="旧版场景条件测试集已退役",
                )
                return

        if _is_batch_history_read(method, path):
            await self.app(scope, receive, send)
            return

        retired_path_id = _retired_path_dataset(
            path,
            self.receipt.retired_dataset_ids,
        )
        retired_query_id = _retired_query_dataset(
            scope,
            self.receipt.retired_dataset_ids,
        )
        retired_id = retired_path_id or retired_query_id
        if retired_id is not None:
            await _send_retirement(
                scope,
                receive,
                send,
                dataset_id=retired_id,
                replacement_dataset_id=(
                    self.receipt.replacement_dataset_ids[retired_id]
                ),
                detail="该声学数据集已退役，不能再启动新的测试执行",
            )
            return

        if method != "POST" or not _is_create_batch_path(path):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if (
            not _is_json_content_type(headers)
            or headers.get("content-encoding", "identity").lower()
            not in {"identity", ""}
            or _declares_oversized_body(headers)
        ):
            await self.app(scope, receive, send)
            return

        body, downstream_receive = await _buffer_json_body(receive)
        if body is not None:
            try:
                payload = json.loads(
                    body,
                    parse_constant=_reject_constant,
                )
            except (UnicodeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                dataset_id = payload.get("dataset_id")
                if (
                    isinstance(dataset_id, str)
                    and dataset_id in self.receipt.retired_dataset_ids
                ):
                    await _send_retirement(
                        scope,
                        receive,
                        send,
                        dataset_id=dataset_id,
                        replacement_dataset_id=(
                            self.receipt.replacement_dataset_ids[dataset_id]
                        ),
                        detail=(
                            "该声学数据集已退役，不能再启动新的测试执行"
                        ),
                    )
                    return
        await self.app(scope, downstream_receive, send)


def install_retired_datasets(
    app: Starlette,
    *,
    path: Path | None = None,
) -> RetirementReceipt | None:
    """Install the retirement middleware after loading its receipt once."""
    receipt = load_retirement_receipt(path)
    if receipt is None:
        return None
    app.add_middleware(RetiredDatasetMiddleware, receipt=receipt)
    return receipt


__all__ = [
    "ACTIVE_R2_DATASET_IDS",
    "ACTIVE_REVISION",
    "ACCEPTANCE_PREFIX",
    "DatasetRetirementError",
    "MAX_JSON_BODY_BYTES",
    "RETIREMENT_PATH",
    "RETIREMENT_SCHEMA",
    "RetiredDatasetMiddleware",
    "RetirementReceipt",
    "install_retired_datasets",
    "load_retirement_receipt",
]
