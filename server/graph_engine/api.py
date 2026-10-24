"""FastAPI routes for interactive audio figure generation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import secrets
from threading import Lock
import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from server.audio import AudioError, MAX_BYTES, decode
from server.graph_engine.engine import FIGURES, GraphOptions, analyze_graphics
from server.graph_engine.evaluation_cases import (
    EvaluationCaseNotFound,
    build_evaluation_case_graphics,
)


router = APIRouter(prefix="/api/audio-graphics", tags=["audio visualization"])
_RESULTS: dict[str, tuple[float, dict, dict[str, bytes]]] = {}
_LOCK = Lock()
_FIGURE_GENERATION_LOCK = Lock()
_TTL_SECONDS = 1800
_KEEP = 32


def _store(report: dict, figures: dict[str, bytes]) -> str:
    identifier = secrets.token_urlsafe(16)
    with _LOCK:
        now = time.monotonic()
        for key in [key for key, (created, _, _) in _RESULTS.items() if now - created > _TTL_SECONDS]:
            _RESULTS.pop(key, None)
        _RESULTS[identifier] = (now, report, figures)
        while len(_RESULTS) > _KEEP:
            _RESULTS.pop(next(iter(_RESULTS)))
    return identifier


def _publish_result(report: dict, figures: dict[str, bytes]) -> dict:
    identifier = _store(report, figures)
    result = dict(report)
    result["id"] = identifier
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["figures"] = [
        {
            "id": figure_id,
            "title": FIGURES[figure_id].title,
            "category": FIGURES[figure_id].category,
            "description": FIGURES[figure_id].description,
            "detail": FIGURES[figure_id].detail,
            "url": f"/api/audio-graphics/{identifier}/figures/{figure_id}.png",
            "download_name": f"audio-{figure_id}.png",
        }
        for figure_id in figures
    ]
    with _LOCK:
        created, stored_report, stored_figures = _RESULTS[identifier]
        _RESULTS[identifier] = (created, {**stored_report, **result}, stored_figures)
    return result


@router.post("")
async def create_graphics(
    after: Annotated[UploadFile, File()],
    before: Annotated[UploadFile | None, File()] = None,
    options: Annotated[str, Form()] = "{}",
):
    uploads = {key: value for key, value in {"before": before, "after": after}.items() if value is not None}
    try:
        config = GraphOptions.model_validate(json.loads(options or "{}"))
        blobs = {}
        for role, upload in uploads.items():
            payload = await upload.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise AudioError(f"{role}：文件超过 32 MB")
            blobs[role] = (upload.filename or "audio.wav", payload)

        def calculate():
            decoded_before = decode(
                "before", *blobs["before"], config.before_channel
            ) if "before" in blobs else None
            decoded_after = decode(
                "after", *blobs["after"], config.after_channel
            )
            # Matplotlib is not thread-safe, including its mathtext parser.
            with _FIGURE_GENERATION_LOCK:
                report, figures = analyze_graphics(decoded_before, decoded_after, config)
            return _publish_result(report, figures)

        return await run_in_threadpool(calculate)
    except (AudioError, ValidationError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        for upload in uploads.values():
            await upload.close()


@router.post("/evaluation-cases/{asset_id}")
async def create_evaluation_case_graphics(asset_id: str):
    def calculate():
        with _FIGURE_GENERATION_LOCK:
            return build_evaluation_case_graphics(asset_id)

    try:
        report, figures = await run_in_threadpool(calculate)
        return _publish_result(report, figures)
    except EvaluationCaseNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (AudioError, ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/{result_id}/figures/{figure_id}.png")
def figure_png(result_id: str, figure_id: str):
    with _LOCK:
        item = _RESULTS.get(result_id)
        if item and time.monotonic() - item[0] > _TTL_SECONDS:
            _RESULTS.pop(result_id, None)
            item = None
    if item is None:
        raise HTTPException(404, "图形结果已过期，请重新分析")
    payload = item[2].get(figure_id)
    if payload is None:
        raise HTTPException(404, "图形不存在")
    return Response(
        payload,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="audio-{figure_id}.png"'},
    )
