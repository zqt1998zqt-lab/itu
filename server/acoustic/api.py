"""HTTP surface for acoustic plans, verified evidence and persisted execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .models import (
    Artifact, EvidenceRecord, ExecutionRequest, MeasurementCompileRequest,
    RunCreate,
)
from .five_signal import (
    MAX_BYTES as FIVE_SIGNAL_MAX_BYTES,
    FiveSignalError,
    evaluate_five_signal,
    get_evaluation,
    list_evaluations,
)
from .service import (
    CATALOG, EXECUTIONS, EvidenceError, classify_evidence,
    compile_measurement_evidence, create_run,
    evidence_with_namespace, execute_run, get_execution, get_run,
    list_executions, list_runs, save_evidence, validate_run_request,
    write_measurement_bundle,
)


router = APIRouter(prefix="/api/acoustic", tags=["acoustic-qualification"])


@router.get("/catalog")
def catalog() -> dict:
    return json.loads(CATALOG.read_text())


@router.get("/evidence")
def evidence_status() -> dict:
    from . import service as current
    status = current.verify_evidence_manifest()
    return {**status, "failed": len(status["failed"])}


@router.get("/evidence/{path:path}")
def evidence_download(path: str) -> FileResponse:
    from . import service as current
    candidate = current.EVIDENCE / path
    try:
        metadata = candidate.with_suffix(candidate.suffix + ".json")
        record = EvidenceRecord.model_validate_json(metadata.read_text())
        verified = current._evidence_path(Artifact(
            path=path, sha256=record.artifact.sha256,
            kind=record.artifact.kind,
        ))
    except (EvidenceError, OSError) as error:
        raise HTTPException(404, "unknown or changed evidence artifact") from error
    return FileResponse(verified, filename=candidate.name)


@router.post("/evidence", status_code=201)
async def upload_evidence(
    file: Annotated[UploadFile, File()],
    kind: Annotated[str, Form()] = "unclassified",
    namespace: Annotated[str, Form()] = "",
) -> dict:
    try:
        data = await file.read(64 * 1024 * 1024 + 1)
        record = save_evidence(
            file.filename or "evidence.bin",
            file.content_type or "application/octet-stream", data,
        )
        if namespace:
            record = evidence_with_namespace(record, namespace)
        if kind != "unclassified":
            record = classify_evidence(record.artifact, kind)
        return record.model_dump(mode="json")
    except EvidenceError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        await file.close()


@router.post("/evidence/classify")
def classify(artifact: Artifact, kind: str = Body(embed=True)) -> dict:
    try:
        return classify_evidence(artifact, kind).model_dump(mode="json")
    except (EvidenceError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@router.post("/measurements/compile", status_code=201)
def compile_measurement(request: MeasurementCompileRequest) -> dict:
    try:
        row, traces = compile_measurement_evidence(
            item_id=request.item_id,
            unit=request.unit,
            kind=request.kind,
            reference=Path(request.reference_path),
            capture=Path(request.capture_path),
            max_delay_ms=request.max_delay_ms,
            start_s=request.start_s,
            end_s=request.end_s,
            mask=request.mask,
            delay_samples=request.delay_samples,
        )
        bundle, evidence = write_measurement_bundle([row])
        return {
            "measurement": row,
            "traces": traces,
            "bundle": evidence.model_dump(mode="json"),
            "bundle_sha256": bundle.backend.identity_sha256,
            "standard_compliance": False,
        }
    except (EvidenceError, OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@router.post("/five-signal/evaluate", status_code=201)
async def five_signal_evaluate(
    mode: Annotated[str, Form()],
    near_end: Annotated[UploadFile, File()],
    noise: Annotated[UploadFile, File()],
    microphone: Annotated[UploadFile, File()],
    far_end: Annotated[UploadFile, File()],
    reference: Annotated[UploadFile, File()],
    channel: Annotated[int, Form()] = 0,
    max_delay_ms: Annotated[float, Form()] = 2000,
) -> dict:
    uploads = {
        "near_end": near_end,
        "noise": noise,
        "microphone": microphone,
        "far_end": far_end,
        "reference": reference,
    }
    blobs: dict[str, tuple[str, bytes]] = {}
    try:
        for role, upload in uploads.items():
            data = await upload.read(FIVE_SIGNAL_MAX_BYTES + 1)
            if len(data) > FIVE_SIGNAL_MAX_BYTES:
                raise FiveSignalError(f"{role}：文件超过 32 MB")
            blobs[role] = (upload.filename or f"{role}.wav", data)
        return await run_in_threadpool(
            evaluate_five_signal,
            mode=mode,
            uploads=blobs,
            channel=channel,
            max_delay_ms=max_delay_ms,
        )
    except (FiveSignalError, OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        for upload in uploads.values():
            await upload.close()


@router.get("/five-signal/evaluations")
def five_signal_evaluations(limit: int = 20) -> list[dict]:
    return list_evaluations(limit)


@router.get("/five-signal/evaluations/{evaluation_id}")
def five_signal_evaluation(evaluation_id: str) -> dict:
    result = get_evaluation(evaluation_id)
    if result is None:
        raise HTTPException(404, "unknown five-signal evaluation")
    return result


@router.get("/runs")
def runs() -> list[dict]:
    return [run.model_dump(mode="json") for run in list_runs()]


@router.post("/runs", status_code=201)
def plan_run(request: RunCreate) -> dict:
    try:
        validate_run_request(request)
        return create_run(request).model_dump(mode="json")
    except EvidenceError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/runs/{run_id}")
def run(run_id: str) -> dict:
    result = get_run(run_id)
    if result is None:
        raise HTTPException(404, "unknown acoustic qualification run")
    return result.model_dump(mode="json")


@router.post("/runs/{run_id}/execute", status_code=201)
def execute(run_id: str, request: ExecutionRequest) -> dict:
    result = get_run(run_id)
    if result is None:
        raise HTTPException(404, "unknown acoustic qualification run")
    if not request.confirm_execute:
        raise HTTPException(422, "execution requires confirm_execute=true")
    try:
        return execute_run(result).model_dump(mode="json")
    except (EvidenceError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/executions/{execution_id}")
def execution(execution_id: str) -> dict:
    result = get_execution(execution_id)
    if result is None:
        raise HTTPException(404, "unknown acoustic execution")
    return result.model_dump(mode="json")


@router.get("/runs/{run_id}/executions")
def executions(run_id: str) -> list[dict]:
    return [row.model_dump(mode="json") for row in list_executions(run_id)]


@router.get("/executions/{execution_id}/download")
def download_execution(execution_id: str) -> FileResponse:
    row = get_execution(execution_id)
    if row is None or row.result is None:
        raise HTTPException(404, "unknown acoustic execution")
    path = EXECUTIONS / execution_id / "results.json"
    if not path.is_file():
        raise HTTPException(404, "execution result is unavailable")
    return FileResponse(path, media_type="application/json",
                        filename=f"acoustic-{execution_id}.json")
