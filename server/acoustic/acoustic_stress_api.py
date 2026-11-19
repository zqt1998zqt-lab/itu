"""Read-only, scope-aware dataset and waveform API for echo/double-talk."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .acoustic_stress_profiles import CATEGORIES, stress_profile, stress_profile_for_dataset
from .acceptance_scene_condition_api import _preview_visuals, _reject_unexpected_query_parameters
from .acceptance_scene_conditions import asset_path

router = APIRouter(prefix="/api/alignment/acceptance/stress-datasets", tags=["alignment-acceptance"])
StressScope = Literal["echo", "double_talk"]


def load_stress(dataset_id: str) -> dict:
    from .acoustic_stress_store import load_stress_dataset
    try:
        stress_profile_for_dataset(dataset_id)
    except ValueError as error:
        raise HTTPException(404, "回声/双讲测试集不存在") from error
    try:
        return load_stress_dataset(dataset_id)
    except (ValueError, OSError) as error:
        raise HTTPException(503, f"回声/双讲测试集不可用: {error}") from error


def identity(dataset: dict) -> dict:
    profile = stress_profile_for_dataset(dataset["dataset_id"])
    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "test_scope": profile.scope,
        "mode": profile.mode,
        "scene": "baseline",
        "contract_item_count": profile.contract_item_count,
        "preview_condition_group_count": profile.case_count,
        "observation_count": profile.observation_count,
        "category_counts": profile.category_counts,
        "category_labels": CATEGORIES[profile.scope],
        "standard_compliance": False,
        "formal_test_status": "not_measured",
    }


@router.get("")
def datasets(request: Request, scope: StressScope, mode: str) -> dict:
    _reject_unexpected_query_parameters(request, frozenset({"scope", "mode"}))
    try:
        profile = stress_profile(scope, mode)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    dataset = load_stress(profile.dataset_id)
    view = identity(dataset)
    return {**view, "contracts": profile.contract_rows, "datasets": [view]}


@router.get("/{dataset_id}/preview")
def preview(request: Request, dataset_id: str):
    _reject_unexpected_query_parameters(request, frozenset())
    dataset = load_stress(dataset_id)
    profile = stress_profile_for_dataset(dataset_id)
    try:
        visuals = _preview_visuals(dataset)
    except (ValueError, OSError) as error:
        raise HTTPException(503, f"回声/双讲波形不可用: {error}") from error
    cases = [{
        "case_id": case["case_id"],
        "conditions": deepcopy(case["conditions"]),
        "families": list(case["families"]),
        "signals": sorted(case["signals"]),
        "signals_sha256": dict(case["signals_sha256"]),
        "contract_item_ids": list(case["contract_item_ids"]),
        "provenance": deepcopy(case["provenance"]),
    } for case in dataset["cases"]]
    return JSONResponse({
        **identity(dataset),
        "schema_version": profile.schema_version,
        "dataset_kind": profile.dataset_kind,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_assets_sha256": dataset["dataset_assets_sha256"],
        "contracts": profile.contract_rows,
        "cases": cases,
        "visuals": visuals,
    }, headers={"Cache-Control": "no-cache",
                "ETag": f'"{dataset["dataset_manifest_sha256"]}-{dataset["dataset_assets_sha256"]}"'})


@router.get("/{dataset_id}/cases/{case_id}/inputs/{role}")
def input_audio(dataset_id: str, case_id: str, role: str):
    dataset = load_stress(dataset_id)
    case = next((case for case in dataset["cases"] if case["case_id"] == case_id), None)
    if case is None or role not in {"microphone", "reference", "far_end", "noise"}:
        raise HTTPException(404, "该测试集没有指定条件组或输入音频")
    try:
        payload = asset_path(dataset["root"], case["signals"][role]).read_bytes()
    except (OSError, ValueError) as error:
        raise HTTPException(503, f"输入音频不可用: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != case["signals_sha256"][role]:
        raise HTTPException(409, "输入音频哈希校验失败")
    filename = quote(f"{case_id}-{role}.wav", safe="")
    return Response(payload, media_type="audio/wav", headers={
        "Cache-Control": "public,max-age=31536000,immutable",
        "ETag": f'"{digest}"',
        "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
    })
