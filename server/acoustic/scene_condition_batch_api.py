"""Read/write API for the 20-case scene-condition VCP batches."""
from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from .acoustic_stress_profiles import is_stress_dataset, stress_profile_for_dataset
from .scene_condition_batch import (
    ArtifactIntegrityError,
    ResourceSafetyStop,
    dynamic_microphone_path,
    get_batch,
    list_batches,
    output_path,
    start_batch,
)


router = APIRouter(
    prefix="/api/alignment/acceptance/scene-condition-batches",
    tags=["alignment-acceptance"],
)

_DATASET_ID_PATTERN = r"^[a-f0-9]{32}$"


class SceneConditionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    algorithm_cluster_id: str
    mode: Literal["NB-wired", "NB-wireless", "NB-bt", "WB-wired", "WB-wireless", "WB-bt", "SWB-wired", "SWB-wireless"]
    scene: Literal["baseline", "sce1", "sce2", "sce3", "sce4"]
    test_scope: Literal["background_noise", "echo", "double_talk"] | None = None
    case_id: str | None = None


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


def _raise_api_error(error: Exception) -> None:
    if isinstance(error, ArtifactIntegrityError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, ResourceSafetyStop):
        raise HTTPException(status_code=507, detail=str(error)) from error
    if isinstance(error, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, OSError):
        raise HTTPException(status_code=503, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise error


@router.post("", status_code=202)
def create_scene_condition_batch(
    request: SceneConditionBatchRequest,
) -> dict[str, Any]:
    try:
        expected_scope = stress_profile_for_dataset(request.dataset_id).scope if is_stress_dataset(request.dataset_id) else "background_noise"
        if request.test_scope is not None and request.test_scope != expected_scope:
            raise ValueError("批次测试范围与数据集不一致")
        return start_batch(
            request.dataset_id,
            request.algorithm_cluster_id,
            mode=request.mode,
            scene=request.scene,
            case_id=request.case_id,
        )
    except Exception as error:
        _raise_api_error(error)
        raise


@router.get("")
def scene_condition_batches(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=20)] = 20,
    dataset_id: Annotated[
        str | None,
        Query(
            min_length=32,
            max_length=32,
            pattern=_DATASET_ID_PATTERN,
        ),
    ] = None,
    algorithm_cluster_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128),
    ] = None,
    latest_per_case: Annotated[bool, Query()] = False,
) -> list[dict[str, Any]]:
    _reject_unexpected_query_parameters(
        request,
        frozenset(
            {
                "limit",
                "dataset_id",
                "algorithm_cluster_id",
                "latest_per_case",
            }
        ),
    )
    try:
        return list_batches(
            limit=limit,
            dataset_id=dataset_id,
            algorithm_cluster_id=algorithm_cluster_id,
            latest_per_case=latest_per_case,
        )
    except Exception as error:
        _raise_api_error(error)
        raise


@router.get("/{identifier}")
def scene_condition_batch(identifier: str) -> dict[str, Any]:
    try:
        snapshot = get_batch(identifier)
    except Exception as error:
        _raise_api_error(error)
        raise
    if snapshot is None:
        raise HTTPException(status_code=404, detail="场景条件批次不存在")
    return snapshot


def _scene_condition_output(
    identifier: str,
    case_id: str,
    kind: str,
) -> Response:
    try:
        path, expected_sha256 = output_path(identifier, case_id, kind)
        payload = path.read_bytes()
    except Exception as error:
        _raise_api_error(error)
        raise
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=409,
            detail=f"场景条件 {kind.upper()} 输出哈希校验失败",
        )
    filename = f"{case_id}-{kind}.wav"
    encoded_filename = quote(filename, safe="")
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{encoded_filename}"
            ),
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{actual_sha256}"',
        },
    )


@router.get("/{identifier}/cases/{case_id}/outputs/tx")
def scene_condition_batch_tx(
    identifier: str,
    case_id: str,
) -> Response:
    return _scene_condition_output(identifier, case_id, "tx")


@router.get("/{identifier}/cases/{case_id}/outputs/rx")
def scene_condition_batch_rx(
    identifier: str,
    case_id: str,
) -> Response:
    return _scene_condition_output(identifier, case_id, "rx")


@router.get("/{identifier}/cases/{case_id}/ns-quality-metrics")
def scene_condition_batch_ns_quality_metrics(
    identifier: str,
    case_id: str,
) -> dict[str, Any]:
    """Calculate supplemental metrics from hash-verified historical artifacts."""
    from .ns_quality_metrics import case_ns_metrics

    try:
        return case_ns_metrics(identifier, case_id)
    except Exception as error:
        _raise_api_error(error)
        raise


@router.get(
    "/{identifier}/cases/{case_id}/inputs/dynamic-microphone"
)
def scene_condition_batch_dynamic_microphone(
    identifier: str,
    case_id: str,
) -> Response:
    try:
        path, expected_sha256 = dynamic_microphone_path(
            identifier, case_id
        )
        payload = path.read_bytes()
    except Exception as error:
        _raise_api_error(error)
        raise
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=409,
            detail="运行时动态麦克风哈希校验失败",
        )
    filename = f"{case_id}-dynamic-microphone.wav"
    encoded_filename = quote(filename, safe="")
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{encoded_filename}"
            ),
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{actual_sha256}"',
        },
    )


__all__ = ["router"]
