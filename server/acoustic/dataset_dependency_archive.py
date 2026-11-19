"""Independent, hash-locked metadata for reading retired dataset histories.

This archive carries no execution permission and never substitutes new audio
for an old result. Original generator source and input stores are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = PROJECT / "benchmarks/fixtures/retired-acoustic-metadata-v1"
SCHEMA = "itu-retired-dataset-metadata-v1"
_ID = re.compile(r"[a-f0-9]{32}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class DatasetArchiveError(ValueError):
    pass


class RetiredDatasetExecutionError(DatasetArchiveError):
    def __init__(self, dataset_id: str, message: str):
        super().__init__(message)
        self.dataset_id = dataset_id


def encode_metadata(value: Any) -> Any:
    if isinstance(value, Path):
        return {"__itu_path__": str(value)}
    if isinstance(value, set):
        return {"__itu_set__": sorted(encode_metadata(item) for item in value)}
    if isinstance(value, tuple):
        return {"__itu_tuple__": [encode_metadata(item) for item in value]}
    if isinstance(value, dict):
        return {key: encode_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_metadata(item) for item in value]
    return value


def decode_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__itu_path__"}:
            return Path(value["__itu_path__"])
        if set(value) == {"__itu_set__"}:
            return set(decode_metadata(item) for item in value["__itu_set__"])
        if set(value) == {"__itu_tuple__"}:
            return tuple(decode_metadata(item) for item in value["__itu_tuple__"])
        return {key: decode_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_metadata(item) for item in value]
    return value


def _read_json(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise DatasetArchiveError(f"archive metadata is not a regular file: {path}")
    payload = path.read_bytes()
    try:
        result = json.loads(payload)
    except ValueError as error:
        raise DatasetArchiveError(f"invalid archive JSON: {path}") from error
    if not isinstance(result, dict):
        raise DatasetArchiveError(f"archive JSON is not an object: {path}")
    return result, payload


class DatasetMetadataArchive:
    def __init__(self, root: Path = ARCHIVE_ROOT):
        self.root = Path(root)
        if self.root.is_symlink():
            raise DatasetArchiveError("archive root cannot be a symlink")
        receipt, _ = _read_json(self.root / "retirement.json")
        if receipt.get("schema") != "itu-dataset-retirement-v1":
            raise DatasetArchiveError("invalid retirement schema")
        expected = receipt.get("metadata_archive_manifest_sha256")
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise DatasetArchiveError("retirement is missing archive manifest identity")
        manifest, raw = _read_json(self.root / "manifest.json")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise DatasetArchiveError("archive manifest SHA mismatch")
        if manifest.get("schema") != SCHEMA:
            raise DatasetArchiveError("invalid metadata archive schema")
        entries = manifest.get("datasets")
        retired = receipt.get("retired_dataset_ids")
        if not isinstance(entries, dict) or not isinstance(retired, list) or set(entries) != set(retired):
            raise DatasetArchiveError("archive and retired dataset identities differ")
        if len(retired) != len(set(retired)) or any(not isinstance(item, str) or not _ID.fullmatch(item) for item in retired):
            raise DatasetArchiveError("retired dataset IDs are invalid")
        self.entries = entries
        self._cache: dict[str, dict] = {}

    def contains(self, dataset_id: object) -> bool:
        return isinstance(dataset_id, str) and dataset_id in self.entries

    def load(self, dataset_id: str) -> dict:
        if not self.contains(dataset_id):
            raise DatasetArchiveError(f"dataset not archived: {dataset_id}")
        entry = self.entries[dataset_id]
        relative = entry.get("path")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts or relative != f"datasets/{dataset_id}.json"):
            raise DatasetArchiveError("unsafe archive payload path")
        path = self.root / relative
        if (self.root / "datasets").is_symlink() or path.is_symlink():
            raise DatasetArchiveError("archive payload cannot use symlinks")
        result, raw = _read_json(path)
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256") or len(raw) != entry.get("bytes"):
            raise DatasetArchiveError("archive dataset SHA/size mismatch")
        if result.get("dataset_id") != dataset_id or result.get("kind") != entry.get("kind"):
            raise DatasetArchiveError("archive dataset identity mismatch")
        decoded = decode_metadata(result)
        dataset = decoded.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("dataset_id") != dataset_id:
            raise DatasetArchiveError("archived dataset payload identity mismatch")
        # Return independent mutable dictionaries to existing validators.
        return deepcopy(decoded)


def install_dataset_archive_readers(root: Path = ARCHIVE_ROOT) -> dict:
    """Allow historical validation/listing from metadata; forbid new runs."""
    if not (Path(root) / "retirement.json").exists():
        return {"installed": False}
    archive = DatasetMetadataArchive(root)
    from . import acceptance_batch, acceptance_dataset, acceptance_api
    from . import scene_condition_batch, acoustic_stress_batch

    if getattr(acceptance_batch, "_retired_metadata_archive_installed", False):
        return {"installed": True, "datasets": len(archive.entries)}
    original_get = acceptance_batch.get_dataset
    original_validated = scene_condition_batch._validated_dataset
    validated_signature = inspect.signature(original_validated)

    @wraps(original_get)
    def get_for_history(dataset_id):
        if archive.contains(dataset_id):
            record = archive.load(dataset_id)
            if record["kind"] != "contract":
                return None
            return record["dataset"]
        return original_get(dataset_id)

    @wraps(original_validated)
    def validated_for_history(dataset_id, *args, **kwargs):
        arguments = validated_signature.bind(dataset_id, *args, **kwargs)
        arguments.apply_defaults()
        if archive.contains(dataset_id):
            record = archive.load(dataset_id)
            if record["kind"] not in {"scene", "stress"}:
                raise DatasetArchiveError("expected archived scene/stress metadata")
            dataset = record["dataset"]
            if arguments.arguments.get("requested_scene") not in (None, dataset["scene"]):
                raise DatasetArchiveError("archived scene selection differs")
            if arguments.arguments.get("requested_mode") not in (None, dataset["mode"]):
                raise DatasetArchiveError("archived mode selection differs")
            return dataset, record["contracts"]
        return original_validated(dataset_id, *args, **kwargs)

    def guarded_start(original):
        @wraps(original)
        def start(dataset_id, *args, **kwargs):
            if archive.contains(dataset_id):
                raise RetiredDatasetExecutionError(dataset_id, "旧测试集已退役；历史报告仍可读，请使用当前 R2 测试集新建测试")
            return original(dataset_id, *args, **kwargs)
        return start

    def guarded_run(original, get_snapshot):
        @wraps(original)
        def run(identifier, *args, **kwargs):
            snapshot = get_snapshot(identifier)
            if snapshot and archive.contains(snapshot.get("dataset_id")):
                raise RetiredDatasetExecutionError(snapshot["dataset_id"], "退役测试集仅供历史查看，不允许重新执行")
            return original(identifier, *args, **kwargs)
        return run

    original_list = acceptance_dataset.list_datasets

    def current_datasets():
        store = acceptance_dataset._store()
        if not store.is_dir() or store.is_symlink():
            return []
        rows = []
        for directory in store.iterdir():
            # Filter the identifier before any stat or metadata access to the
            # retired store. A post-filter would still depend on old JSON.
            if archive.contains(directory.name) or not acceptance_dataset._safe_dataset_id(directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                loaded = acceptance_dataset._load_summary(directory)
            except acceptance_dataset._DatasetReimportRequired:
                continue
            if loaded is not None:
                rows.append(loaded[0])
        return sorted(rows, key=lambda row: (row.get("created_at", ""), row.get("dataset_id", "")), reverse=True)

    acceptance_batch.get_dataset = get_for_history
    scene_condition_batch._validated_dataset = validated_for_history
    for module in (acceptance_batch, scene_condition_batch, acoustic_stress_batch):
        module.start_batch = guarded_start(module.start_batch)
    # Router modules import callable values; update their aliases as well.
    from . import scene_condition_batch_api
    scene_condition_batch_api.start_batch = scene_condition_batch.start_batch
    acceptance_api.start_batch = acceptance_batch.start_batch
    acceptance_dataset.list_datasets = current_datasets
    acceptance_api.list_datasets = current_datasets
    acceptance_batch.run_batch = guarded_run(acceptance_batch.run_batch, acceptance_batch.get_batch)
    scene_condition_batch.run_batch = guarded_run(scene_condition_batch.run_batch, scene_condition_batch.get_batch)
    acceptance_batch._retired_metadata_archive_installed = True
    return {"installed": True, "datasets": len(archive.entries)}


def install_retired_execution_exception_handler(app, root: Path = ARCHIVE_ROOT):
    """Preserve HTTP 410 even when a large JSON body bypasses middleware inspection."""
    if not (Path(root) / "retirement.json").is_file():
        return
    from fastapi.responses import JSONResponse
    from fastapi.exception_handlers import http_exception_handler
    from starlette.exceptions import HTTPException
    receipt, _ = _read_json(Path(root) / "retirement.json")
    replacements = receipt["replacement_dataset_ids"]
    previous = app.exception_handlers.get(HTTPException, http_exception_handler)

    async def handler(request, error):
        cause = error.__cause__
        if isinstance(cause, RetiredDatasetExecutionError) and cause.dataset_id in replacements:
            return JSONResponse(status_code=410, content={
                "detail": str(cause), "dataset_id": cause.dataset_id,
                "replacement_dataset_id": replacements[cause.dataset_id],
            })
        return await previous(request, error)

    app.add_exception_handler(HTTPException, handler)
