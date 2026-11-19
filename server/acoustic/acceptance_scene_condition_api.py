"""Read-only API for the background-noise 20-condition engineering preview."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from . import service
from .acceptance_scene_conditions import (
    CONTRACT_ITEM_COUNT,
    FORMAL_TEST_STATUS,
    MODE,
    PREVIEW_CONDITION_GROUP_COUNT,
    ROLES,
    SCHEMA_VERSION,
    STANDARD_COMPLIANCE,
    TEST_SCOPE,
)
from .acceptance_scene_condition_store import (
    SceneConditionDatasetError,
    asset_path,
    load_scene_condition_dataset,
)
from .scene_mode_conversation_profiles import (
    SceneConditionScene,
    profile_for_dataset,
    profile_for_scene,
    profile_for_scene_and_mode,
)
from .scene_condition_profiles import profile_for_scene as legacy_profile_for_scene
from .acceptance_visualization import (
    MEL_BANDS,
    MEL_DB_OFFSET_DB,
    MEL_DB_STEP_DB,
    MEL_DYNAMIC_RANGE_DB,
    MEL_FFT_SIZE,
    MEL_FRAMES,
    MEL_HOP_MS,
    SCHEMA as VISUAL_SCHEMA,
    WAVEFORM_BINS,
    VisualizationCacheError,
    load_or_build_visualization,
)


router = APIRouter(
    prefix="/api/alignment/acceptance",
    tags=["alignment-acceptance"],
)

SceneConditionScope = Literal["background_noise"]
SceneConditionMode = Literal["NB-wired", "NB-wireless", "NB-bt", "WB-wired", "WB-wireless", "WB-bt", "SWB-wired", "SWB-wireless"]

_VISUAL_CACHE: dict[str, dict[str, Any]] = {}
_VISUAL_CACHE_LOCK = Lock()
_DATASET_ID_PATTERN = r"^[a-f0-9]{32}$"


def _reject_unexpected_query_parameters(
    request: Request,
    allowed: frozenset[str],
) -> None:
    for key in request.query_params:
        if key not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"unexpected query parameter: {key}",
            )
        if len(request.query_params.getlist(key)) != 1:
            raise HTTPException(
                status_code=422,
                detail=f"query parameter must occur once: {key}",
            )


def _load_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        profile = profile_for_dataset(dataset_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="场景条件测试集不存在")
    _validate_runtime_contract_binding(profile.dataset_id)
    try:
        return load_scene_condition_dataset(dataset_id)
    except SceneConditionDatasetError as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件测试集不可用: {error}",
        ) from error


def _validate_runtime_contract_binding(dataset_id: str) -> None:
    profile = profile_for_dataset(dataset_id)
    root = Path(service.BUILD) / "acoustic-scene-condition-datasets" / dataset_id
    if getattr(profile, "schema_version", "") == "scene-condition-preview-v2":
        _validate_conversation_binding(root, dataset_id)
        return
    binding_path = root / "runtime-contract-binding.json"
    contract_path = (
        Path(__file__).resolve().parent / "acceptance_scene_conditions.py"
    )
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        runtime_contract_sha256 = hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest()
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件运行时契约绑定不可用: {error}",
        ) from error
    difference = binding.get("difference") if isinstance(binding, dict) else None
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != "scene-condition-runtime-binding-v1"
        or binding.get("dataset_id") != dataset_id
        or binding.get("status") != "reconciled"
        or binding.get("runtime_contract_sha256")
        != runtime_contract_sha256
        or not isinstance(difference, dict)
        or difference.get("scope") not in {
            "module_export_only", "generation_contract_equals_runtime"
        }
    ):
        raise HTTPException(
            status_code=503,
            detail="场景条件运行时契约绑定已过期",
        )
    artifact_sha256 = binding.get("artifact_sha256")
    required_artifacts = {
        "manifest.json",
        "assets.json",
        "summary.json",
        "quality.json",
    }
    if (
        not isinstance(artifact_sha256, dict)
        or set(artifact_sha256) != required_artifacts
    ):
        raise HTTPException(
            status_code=503,
            detail="场景条件运行时契约绑定缺少产物哈希",
        )
    artifacts = {}
    for name, expected in artifact_sha256.items():
        if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
            raise HTTPException(
                status_code=503,
                detail="场景条件运行时契约绑定产物哈希无效",
            )
        try:
            payload = (root / name).read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            artifacts[name] = json.loads(payload)
        except (OSError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail=f"场景条件产物不可读取: {error}",
            ) from error
        if actual != expected:
            raise HTTPException(
                status_code=503,
                detail="场景条件运行时契约绑定产物哈希不匹配",
            )
    generation_sha = binding.get("generation_contract_sha256")
    try:
        recorded_shas = {
            artifacts[name]["source_hashes"]["scene_contract_sha256"]
            for name in ("summary.json", "quality.json")
        }
        if generation_sha not in recorded_shas or len(recorded_shas) != 1:
            raise ValueError("generation contract does not match recorded provenance")
        if difference["scope"] == "generation_contract_equals_runtime":
            reconstructed_sha = runtime_contract_sha256
        else:
            source = contract_path.read_bytes()
            added_export = b'    "DATASET_KIND",\n'
            if source.count(added_export) != 1:
                raise ValueError("reconciled export is no longer unique")
            reconstructed_sha = hashlib.sha256(
                source.replace(added_export, b"", 1)
            ).hexdigest()
        if reconstructed_sha != generation_sha:
            raise ValueError("generation contract cannot be reconstructed")
        for source_key, source_path in (
            (
                "scene_profiles_sha256",
                Path(__file__).resolve().parent
                / "scene_condition_profiles.py",
            ),
            (
                "acceptance_scope_sha256",
                Path(__file__).resolve().parent / "acceptance_scope.py",
            ),
        ):
            recorded = {
                artifacts[name]
                .get("source_hashes", {})
                .get(source_key)
                for name in ("summary.json", "quality.json")
            }
            if recorded == {None} and profile.scene == "sce1":
                continue
            current = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if len(recorded) != 1 or next(iter(recorded)) != current:
                raise ValueError(
                    f"generation source does not match: {source_key}"
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件来源契约校验失败: {error}",
        ) from error


def _validate_conversation_binding(root: Path, dataset_id: str) -> None:
    """V2 binds the frozen inputs and generating sources without rewriting V1."""
    project = Path(__file__).resolve().parents[2]
    required_sources = {
        "server/acoustic/scene_conversation_profiles.py",
        "scripts/acceptance_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }
    try:
        binding = json.loads((root / "runtime-contract-binding.json").read_text())
        if (not isinstance(binding, dict)
            or binding.get("schema_version") != "scene-condition-runtime-binding-v2"
            or binding.get("dataset_id") != dataset_id
            or binding.get("status") != "reconciled"):
            raise ValueError("conversation dataset binding identity differs")
        artifacts = binding.get("artifact_sha256")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "manifest.json", "assets.json", "summary.json", "quality.json",
        }:
            raise ValueError("conversation dataset artifact hashes incomplete")
        for name, digest in artifacts.items():
            if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise ValueError("conversation artifact digest invalid")
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"conversation artifact changed: {name}")
        sources = binding.get("source_sha256")
        if not isinstance(sources, dict) or not required_sources.issubset(sources):
            raise ValueError("conversation generating sources incomplete")
        for name, digest in sources.items():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("conversation source path invalid")
            source = (project / name).resolve(strict=True)
            if not source.is_relative_to(project) or not isinstance(digest, str):
                raise ValueError("conversation source escapes project")
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"conversation source changed: {name}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise HTTPException(503, f"通话场景测试集来源契约校验失败: {error}") from error


def _visual_track(path, source_sha256: str) -> dict[str, Any]:
    with _VISUAL_CACHE_LOCK:
        cached = _VISUAL_CACHE.get(source_sha256)
        if cached is None:
            visual = load_or_build_visualization(
                path,
                source_sha256,
                verify_source=False,
            )
            cached = {
                "source_sha256": visual["source_sha256"],
                "duration_s": visual["duration_s"],
                "sample_rate": visual["sample_rate"],
                "channels": visual["channels"],
                "peak": visual["peak"],
                "wave_scale": visual["wave_scale"],
                "wave_min_i16": visual["wave_min_i16"],
                "wave_max_i16": visual["wave_max_i16"],
                "mel_i16": visual["mel_i16"],
                "mel_ceiling_db": visual["mel_ceiling_db"],
            }
            _VISUAL_CACHE[source_sha256] = cached
        return dict(cached)


def _preview_visuals(
    dataset: dict[str, Any], *, waveform_only: bool = False,
) -> dict[str, Any]:
    visual_cases: dict[str, dict[str, dict[str, Any]]] = {}
    for case in dataset["cases"]:
        tracks: dict[str, dict[str, Any]] = {}
        for role in sorted(ROLES):
            relative = case["signals"][role]
            source_sha256 = case["signals_sha256"][role]
            path = asset_path(dataset["root"], relative)
            track = _visual_track(path, source_sha256)
            if waveform_only:
                track = {
                    key: value for key, value in track.items()
                    if key not in {"mel_i16", "mel_ceiling_db"}
                }
            tracks[role] = track
        visual_cases[case["case_id"]] = tracks
    return {
        **({"detail": "waveform"} if waveform_only else {}),
        "schema": VISUAL_SCHEMA,
        "waveform_bins": WAVEFORM_BINS,
        "mel_bands": MEL_BANDS,
        "mel_frames": MEL_FRAMES,
        "mel_fft_size": MEL_FFT_SIZE,
        "mel_hop_ms": MEL_HOP_MS,
        "mel_dynamic_range_db": MEL_DYNAMIC_RANGE_DB,
        "mel_db_offset_db": MEL_DB_OFFSET_DB,
        "mel_db_step_db": MEL_DB_STEP_DB,
        "cases": visual_cases,
    }


def _content_disposition(case_id: str, role: str) -> str:
    filename = f"{case_id}-{role}.wav"
    ascii_filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    encoded_filename = quote(filename, safe="")
    return (
        f'attachment; filename="{ascii_filename}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )


@router.get("/scene-condition-datasets")
def scene_condition_datasets(
    request: Request,
    scope: SceneConditionScope,
    mode: SceneConditionMode,
    scene: SceneConditionScene,
    version: Literal["current", "legacy"] = "current",
) -> dict[str, Any]:
    _reject_unexpected_query_parameters(
        request,
        frozenset({"scope", "mode", "scene", "version"}),
    )
    if version == "legacy":
        if mode != MODE:
            raise HTTPException(
                status_code=422,
                detail="20 组历史预览测试集仅支持 WB-wired",
            )
        profile = legacy_profile_for_scene(scene)
    else:
        profile = profile_for_scene_and_mode(scene, mode)
    dataset = _load_dataset(profile.dataset_id)
    if dataset["mode"] != mode or dataset["scene"] != scene:
        raise HTTPException(
            status_code=422,
            detail="场景、模式与场景条件测试集身份不一致",
        )
    return {
        "test_scope": TEST_SCOPE,
        "mode": dataset["mode"],
        "scene": dataset["scene"],
        "contract_item_count": dataset["contract_item_count"],
        "preview_condition_group_count": dataset["preview_condition_group_count"],
        "observation_count": dataset.get("observation_count", 50),
        "conversation_counts": dataset.get("conversation_counts", {}),
        "standard_compliance": STANDARD_COMPLIANCE,
        "formal_test_status": FORMAL_TEST_STATUS,
        "datasets": [
            {
                "dataset_id": dataset["dataset_id"],
                "dataset_name": dataset["dataset_name"],
                "test_scope": TEST_SCOPE,
                "mode": dataset["mode"],
                "scene": dataset["scene"],
                "contract_item_count": dataset["contract_item_count"],
                "preview_condition_group_count": (
                    dataset["preview_condition_group_count"]
                ),
                "observation_count": dataset.get("observation_count", 50),
                "conversation_counts": dataset.get("conversation_counts", {}),
                "standard_compliance": STANDARD_COMPLIANCE,
                "formal_test_status": FORMAL_TEST_STATUS,
            }
        ]
    }


@router.get("/scene-condition-preview")
def scene_condition_preview(
    request: Request,
    dataset_id: Annotated[
        str,
        Query(
            min_length=32,
            max_length=32,
            pattern=_DATASET_ID_PATTERN,
        ),
    ],
    mode: SceneConditionMode,
    scene: SceneConditionScene,
    visual_detail: Literal["full", "waveform"] = "full",
):
    _reject_unexpected_query_parameters(
        request,
        frozenset({"dataset_id", "mode", "scene", "visual_detail"}),
    )
    dataset = _load_dataset(dataset_id)
    if dataset["mode"] != mode or dataset["scene"] != scene:
        raise HTTPException(
            status_code=422,
            detail="场景、模式与场景条件测试集身份不一致",
        )
    cases = []
    for case in dataset["cases"]:
        roles = sorted(ROLES)
        cases.append(
            {
                "case_id": case["case_id"],
                "conditions": dict(case["conditions"]),
                "families": sorted(case["families"]),
                "signals": roles,
                "signals_sha256": {
                    role: case["signals_sha256"][role] for role in roles
                },
                "contract_item_ids": sorted(case["contract_item_ids"]),
                "provenance": copy.deepcopy(case["provenance"]),
            }
        )
    try:
        visuals = (
            _preview_visuals(dataset, waveform_only=True)
            if visual_detail == "waveform" else _preview_visuals(dataset)
        )
    except (OSError, SceneConditionDatasetError, VisualizationCacheError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件可视化不可用: {error}",
        ) from error
    payload = {
        **({"visual_detail": "waveform"} if visual_detail == "waveform" else {}),
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_assets_sha256": dataset["dataset_assets_sha256"],
        "test_scope": TEST_SCOPE,
        "mode": dataset["mode"],
        "scene": dataset["scene"],
        "cases": cases,
        "visuals": visuals,
        "preview_condition_group_count": dataset["preview_condition_group_count"],
        "observation_count": dataset.get("observation_count", 50),
        "conversation_counts": dataset.get("conversation_counts", {}),
        "contract_item_count": dataset["contract_item_count"],
        "standard_compliance": STANDARD_COMPLIANCE,
        "formal_test_status": FORMAL_TEST_STATUS,
        "dataset_kind": dataset["dataset_kind"],
        "schema_version": dataset["manifest"]["schema_version"],
    }
    etag = (
        f'"{dataset["dataset_manifest_sha256"]}-'
        f'{dataset["dataset_assets_sha256"]}"'
    )
    if visual_detail == "waveform":
        etag = f'{etag[:-1]}-waveform"'
    # The browser can reuse the immutable representation after revalidating its
    # identity; the full and waveform-only variants have distinct validators.
    validators = request.headers.get("if-none-match", "").split(",")
    if any(value.strip().removeprefix("W/") in {etag, "*"} for value in validators):
        return Response(status_code=304, headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag,
        })
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag,
        },
    )


@router.get(
    "/scene-condition-datasets/{dataset_id}/cases/{case_id}/inputs/{role}"
)
def scene_condition_case_input(
    dataset_id: str,
    case_id: str,
    role: str,
) -> Response:
    dataset = _load_dataset(dataset_id)
    case = next(
        (
            item
            for item in dataset["cases"]
            if item["case_id"] == case_id
        ),
        None,
    )
    if case is None:
        raise HTTPException(status_code=404, detail="场景条件组不存在")
    relative = case["signals"].get(role)
    if not isinstance(relative, str):
        raise HTTPException(status_code=404, detail="场景条件组没有该输入音频")
    expected_sha256 = case["signals_sha256"][role]
    try:
        path = asset_path(dataset["root"], relative)
        payload = path.read_bytes()
    except (OSError, SceneConditionDatasetError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件输入音频不可用: {error}",
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise HTTPException(
            status_code=409,
            detail="场景条件输入音频身份校验失败",
        )
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "Content-Disposition": _content_disposition(case_id, role),
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{expected_sha256}"',
        },
    )
