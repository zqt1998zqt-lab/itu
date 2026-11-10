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
    DATASET_ID,
    DATASET_KIND,
    DATASET_NAME,
    FORMAL_TEST_STATUS,
    MODE,
    PREVIEW_CONDITION_GROUP_COUNT,
    ROLES,
    SCENE,
    SCHEMA_VERSION,
    STANDARD_COMPLIANCE,
    TEST_SCOPE,
    SceneConditionDatasetError,
    asset_path,
    load_scene_condition_dataset,
)
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
SceneConditionMode = Literal["WB-wired"]
SceneConditionScene = Literal["sce1"]

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
    if dataset_id != DATASET_ID:
        raise HTTPException(status_code=404, detail="场景条件测试集不存在")
    _validate_runtime_contract_binding()
    try:
        return load_scene_condition_dataset(dataset_id)
    except SceneConditionDatasetError as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件测试集不可用: {error}",
        ) from error


def _validate_runtime_contract_binding() -> None:
    root = Path(service.BUILD) / "acoustic-scene-condition-datasets" / DATASET_ID
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
        or binding.get("dataset_id") != DATASET_ID
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
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件来源契约校验失败: {error}",
        ) from error


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


def _preview_visuals(dataset: dict[str, Any]) -> dict[str, Any]:
    visual_cases: dict[str, dict[str, dict[str, Any]]] = {}
    for case in dataset["cases"]:
        tracks: dict[str, dict[str, Any]] = {}
        for role in sorted(ROLES):
            relative = case["signals"][role]
            source_sha256 = case["signals_sha256"][role]
            path = asset_path(dataset["root"], relative)
            tracks[role] = _visual_track(path, source_sha256)
        visual_cases[case["case_id"]] = tracks
    return {
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
) -> dict[str, Any]:
    _reject_unexpected_query_parameters(
        request,
        frozenset({"scope", "mode", "scene"}),
    )
    _load_dataset(DATASET_ID)
    return {
        "test_scope": TEST_SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "preview_condition_group_count": PREVIEW_CONDITION_GROUP_COUNT,
        "standard_compliance": STANDARD_COMPLIANCE,
        "formal_test_status": FORMAL_TEST_STATUS,
        "datasets": [
            {
                "dataset_id": DATASET_ID,
                "dataset_name": DATASET_NAME,
                "test_scope": TEST_SCOPE,
                "mode": MODE,
                "scene": SCENE,
                "contract_item_count": CONTRACT_ITEM_COUNT,
                "preview_condition_group_count": (
                    PREVIEW_CONDITION_GROUP_COUNT
                ),
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
):
    _reject_unexpected_query_parameters(
        request,
        frozenset({"dataset_id", "mode", "scene"}),
    )
    dataset = _load_dataset(dataset_id)
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
        visuals = _preview_visuals(dataset)
    except (OSError, SceneConditionDatasetError, VisualizationCacheError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"场景条件可视化不可用: {error}",
        ) from error
    payload = {
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_assets_sha256": dataset["dataset_assets_sha256"],
        "test_scope": TEST_SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "cases": cases,
        "visuals": visuals,
        "preview_condition_group_count": PREVIEW_CONDITION_GROUP_COUNT,
        "contract_item_count": CONTRACT_ITEM_COUNT,
        "standard_compliance": STANDARD_COMPLIANCE,
        "formal_test_status": FORMAL_TEST_STATUS,
        "dataset_kind": DATASET_KIND,
        "schema_version": SCHEMA_VERSION,
    }
    etag = (
        f'"{dataset["dataset_manifest_sha256"]}-'
        f'{dataset["dataset_assets_sha256"]}"'
    )
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
