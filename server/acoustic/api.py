"""HTTP surface for acoustic plans, verified evidence and persisted execution."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
import soundfile as sf
from starlette.concurrency import run_in_threadpool

from server.quality.evaluators import PROJECT as QUALITY_PROJECT, create_evaluators

from .models import (
    Artifact, EvidenceRecord, ExecutionRequest, MeasurementCompileRequest,
    OutcomeAnalysisRequest, RunCreate,
)
from .five_signal import (
    MAX_BYTES as FIVE_SIGNAL_MAX_BYTES,
    ROLE_LABELS,
    FiveSignalError,
    evaluate_five_signal,
    get_evaluation,
    list_evaluations,
)
from .receive_simulation import (
    MAX_BYTES as SYNTHETIC_RECEIVE_MAX_BYTES,
    SyntheticReceiveError,
    _third_octave_levels,
    evaluate_synthetic_receive,
    get_synthetic_receive_evaluation,
    get_synthetic_receive_file,
    list_synthetic_receive_evaluations,
)
from .receive_linearity import (
    canonical_sha256,
    evaluate_rlr_linearity_group,
)
from .receive_quality_proxy import (
    ReceiveQualityProxyError,
    evaluate_binaural_dnsmos_with_evaluator,
)
from .service import (
    CATALOG, EXECUTIONS, EvidenceError, classify_evidence,
    compile_measurement_evidence, create_run,
    evidence_with_namespace, execute_run, get_execution, get_run,
    list_executions, list_runs, save_evidence, validate_run_request,
    write_measurement_bundle,
)
from .outcome_codex import (
    OutcomeCodexError,
    OutcomeCodexFailure,
    OutcomeCodexUnavailable,
    analyze_outcome,
)


router = APIRouter(prefix="/api/acoustic", tags=["acoustic-qualification"])


class RlrLinearityEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_ids: dict[str, str]
    conditions: dict[str, Any]


@lru_cache(maxsize=1)
def _dnsmos_evaluator():
    return create_evaluators(
        ["dnsmos"],
        QUALITY_PROJECT / "build/quality-evaluations/cache",
    )[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linearity_window(conditions: dict[str, Any]) -> tuple[float, float]:
    raw = conditions.get("active_speech_window_s")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise HTTPException(
            422,
            "conditions.active_speech_window_s must contain exactly "
            "[start_s, end_s]",
        )

    values: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise HTTPException(
                422,
                f"active speech window [{index}] must be numeric",
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                422,
                f"active speech window [{index}] must be numeric",
            ) from error
        if not math.isfinite(number):
            raise HTTPException(
                422,
                f"active speech window [{index}] must be finite",
            )
        values.append(number)

    start, end = values
    if start < 0.0 or end <= start:
        raise HTTPException(
            422,
            "active_speech_window_s must satisfy 0 <= start_s < end_s",
        )
    return start, end


def _linearity_run(
    report: dict[str, Any],
    level_id: str,
    analysis_window: tuple[float, float],
) -> dict[str, Any]:
    evaluation_id = report.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise HTTPException(409, f"{level_id}: receive report has no evaluation_id")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HTTPException(409, f"{level_id}: receive report has no artifacts")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for ear, filename in (
        ("left", "virtual_left.wav"),
        ("right", "virtual_right.wav"),
    ):
        artifact = artifacts.get(ear)
        path = get_synthetic_receive_file(evaluation_id, filename)
        if path is None or not isinstance(artifact, dict):
            raise HTTPException(
                404,
                f"missing synthetic receive {ear} artifact for {level_id}",
            )
        expected = artifact.get("sha256")
        try:
            actual = _sha256_file(path)
        except OSError as error:
            raise HTTPException(
                409,
                f"synthetic receive {ear} artifact unavailable for {level_id}",
            ) from error
        if not isinstance(expected, str) or actual != expected:
            raise HTTPException(
                409,
                f"synthetic receive {ear} artifact hash mismatch for {level_id}",
            )
        paths[ear] = path
        hashes[ear] = actual

    try:
        left, left_rate = sf.read(
            paths["left"],
            dtype="float64",
            always_2d=False,
        )
        right, right_rate = sf.read(
            paths["right"],
            dtype="float64",
            always_2d=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{level_id}: cannot read synthetic receive artifacts: {error}"
        ) from error

    if int(left_rate) != int(right_rate):
        raise ValueError(
            f"{level_id}: left and right receive artifacts use different "
            "sample rates"
        )
    if getattr(left, "ndim", 0) != 1 or getattr(right, "ndim", 0) != 1:
        raise ValueError(f"{level_id}: synthetic receive artifacts must be mono")
    if len(left) != len(right):
        raise ValueError(
            f"{level_id}: left and right receive artifacts have different "
            "frame counts"
        )

    rate = int(left_rate)
    start_s, end_s = analysis_window
    duration_s = len(left) / rate
    if end_s > duration_s + 1e-9:
        raise ValueError(
            f"{level_id}: active_speech_window_s ends after the receive audio"
        )
    start_sample = max(0, int(round(start_s * rate)))
    end_sample = min(len(left), int(round(end_s * rate)))
    if start_sample >= end_sample:
        raise ValueError(
            f"{level_id}: active_speech_window_s selects no receive samples"
        )

    cropped_bands = _third_octave_levels(
        left[start_sample:end_sample],
        right[start_sample:end_sample],
        rate,
    )
    simulation = report.get("simulation")
    normalisation = (
        simulation.get("output_normalisation")
        if isinstance(simulation, dict)
        else report.get("output_normalisation")
    )
    clipping = report.get("clipping")
    if clipping is None and isinstance(normalisation, dict):
        # The simulator rejects clipped speaker input and normalises any
        # over-range binaural output. A non-normalised output therefore gives
        # a deterministic, auditable no-clipping flag for this engineering run.
        clipping = False
    return {
        **report,
        "run_id": report.get("evaluation_id"),
        "level_id": level_id,
        "clipping": clipping,
        "metrics": {
            **(
                report["metrics"]
                if isinstance(report.get("metrics"), dict)
                else {}
            ),
            "third_octave_band_levels_db": cropped_bands,
            "third_octave_band_levels_dbfs": cropped_bands,
            "window_applied": True,
            "analysis_window_s": [start_s, end_s],
            "analysis_sample_bounds": {
                "left": [start_sample, end_sample],
                "right": [start_sample, end_sample],
            },
            "analysis_artifact_sha256": hashes,
        },
    }


def _verified_receive_report(evaluation_id: str) -> dict[str, Any] | None:
    report = get_synthetic_receive_evaluation(evaluation_id)
    if report is None:
        return None
    expected = report.get("result_sha256")
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"result", "result_sha256"}
    }
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise HTTPException(
            409,
            f"synthetic receive report hash mismatch: {evaluation_id}",
        )
    return report


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
                raise FiveSignalError(f"{ROLE_LABELS[role]}：文件超过 32 MB")
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
def five_signal_evaluations(limit: int = 20, algorithm_cluster_id: str | None = None) -> list[dict]:
    return list_evaluations(limit, algorithm_cluster_id)


@router.get("/five-signal/evaluations/{evaluation_id}")
def five_signal_evaluation(evaluation_id: str) -> dict:
    result = get_evaluation(evaluation_id)
    if result is None:
        raise HTTPException(404, "unknown five-signal evaluation")
    return result


@router.post("/receive-simulation/evaluate", status_code=201)
async def receive_simulation_evaluate(
    mode: Annotated[str, Form()],
    speaker_output: Annotated[UploadFile, File()],
    left_ir: Annotated[UploadFile | None, File()] = None,
    right_ir: Annotated[UploadFile | None, File()] = None,
    azimuth_deg: Annotated[float, Form()] = 30,
    head_radius_m: Annotated[float, Form()] = 0.0875,
    speed_of_sound_mps: Annotated[float, Form()] = 343,
    upper_hz: Annotated[float | None, Form()] = None,
    noise_start_s: Annotated[float | None, Form()] = None,
    noise_end_s: Annotated[float | None, Form()] = None,
) -> dict:
    uploads = [speaker_output, left_ir, right_ir]
    try:
        if bool(left_ir) != bool(right_ir):
            raise SyntheticReceiveError("left_ir 与 right_ir 必须同时提供")

        async def payload(upload: UploadFile) -> tuple[str, bytes]:
            data = await upload.read(SYNTHETIC_RECEIVE_MAX_BYTES + 1)
            if len(data) > SYNTHETIC_RECEIVE_MAX_BYTES:
                raise SyntheticReceiveError(
                    f"{upload.filename or 'audio'}：文件超过 32 MB"
                )
            return upload.filename or "audio.wav", data

        speaker_name, speaker_bytes = await payload(speaker_output)
        left_name = left_bytes = right_name = right_bytes = None
        if left_ir and right_ir:
            left_name, left_bytes = await payload(left_ir)
            right_name, right_bytes = await payload(right_ir)
        return await run_in_threadpool(
            evaluate_synthetic_receive,
            mode=mode,
            speaker_name=speaker_name,
            speaker_bytes=speaker_bytes,
            left_ir_name=left_name,
            left_ir_bytes=left_bytes,
            right_ir_name=right_name,
            right_ir_bytes=right_bytes,
            azimuth_deg=azimuth_deg,
            head_radius_m=head_radius_m,
            speed_of_sound_mps=speed_of_sound_mps,
            upper_hz=upper_hz,
            noise_start_s=noise_start_s,
            noise_end_s=noise_end_s,
        )
    except (SyntheticReceiveError, OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        for upload in uploads:
            if upload is not None:
                await upload.close()


@router.get("/receive-simulation/evaluations")
def receive_simulation_evaluations(limit: int = 20) -> list[dict]:
    return list_synthetic_receive_evaluations(limit)


@router.get("/receive-simulation/evaluations/{evaluation_id}")
def receive_simulation_evaluation(evaluation_id: str) -> dict:
    result = get_synthetic_receive_evaluation(evaluation_id)
    if result is None:
        raise HTTPException(404, "unknown synthetic receive evaluation")
    return result


@router.get("/receive-simulation/evaluations/{evaluation_id}/files/{filename}")
def receive_simulation_file(evaluation_id: str, filename: str) -> FileResponse:
    path = get_synthetic_receive_file(evaluation_id, filename)
    if path is None:
        raise HTTPException(404, "unknown synthetic receive artifact")
    return FileResponse(path, filename=path.name)


@router.post("/receive-simulation/rlr-linearity/evaluate")
def receive_rlr_linearity_evaluate(
    request: RlrLinearityEvaluationRequest,
) -> dict:
    required_levels = ("nominal", "offset_-5", "offset_+5")
    if set(request.evaluation_ids) != set(required_levels):
        raise HTTPException(
            422,
            "evaluation_ids must contain exactly nominal, offset_-5, and offset_+5",
        )
    identifiers = [request.evaluation_ids[level] for level in required_levels]
    if len(set(identifiers)) != len(identifiers):
        raise HTTPException(422, "each linearity level must use a distinct evaluation")

    verified_reports: list[tuple[str, dict[str, Any]]] = []
    for level_id in required_levels:
        evaluation_id = request.evaluation_ids[level_id]
        report = _verified_receive_report(evaluation_id)
        if report is None:
            raise HTTPException(
                404,
                f"unknown synthetic receive evaluation for {level_id}: {evaluation_id}",
            )
        verified_reports.append((level_id, report))

    try:
        analysis_window = _linearity_window(request.conditions)
        reports = [
            _linearity_run(report, level_id, analysis_window)
            for level_id, report in verified_reports
        ]
        return evaluate_rlr_linearity_group(
            reports,
            request.conditions,
            include_canonical_hash=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@router.post(
    "/receive-simulation/evaluations/{evaluation_id}/quality-proxy",
)
def receive_simulation_quality_proxy(evaluation_id: str) -> dict:
    report = _verified_receive_report(evaluation_id)
    if report is None:
        raise HTTPException(404, "unknown synthetic receive evaluation")

    paths: dict[str, Path] = {}
    for ear, filename in (
        ("left", "virtual_left.wav"),
        ("right", "virtual_right.wav"),
    ):
        path = get_synthetic_receive_file(evaluation_id, filename)
        artifact = report.get("artifacts", {}).get(ear, {})
        if path is None or not isinstance(artifact, dict):
            raise HTTPException(404, f"missing synthetic receive {ear} artifact")
        if _sha256_file(path) != artifact.get("sha256"):
            raise HTTPException(409, f"synthetic receive {ear} artifact hash mismatch")
        paths[ear] = path

    try:
        left, left_rate = sf.read(paths["left"], dtype="float64", always_2d=False)
        right, right_rate = sf.read(paths["right"], dtype="float64", always_2d=False)
        if int(left_rate) != int(right_rate):
            raise ValueError("left and right receive artifacts use different sample rates")
        if getattr(left, "ndim", 0) != 1 or getattr(right, "ndim", 0) != 1:
            raise ValueError("synthetic receive artifacts must be mono")
        result = evaluate_binaural_dnsmos_with_evaluator(
            left,
            right,
            _dnsmos_evaluator(),
            {
                "evaluation_id": evaluation_id,
                "mode": report.get("mode"),
                "left_sha256": report["artifacts"]["left"].get("sha256"),
                "right_sha256": report["artifacts"]["right"].get("sha256"),
            },
            sample_rate=int(left_rate),
        )
        return {
            "evaluation_id": evaluation_id,
            "mode": report.get("mode"),
            **result,
        }
    except (OSError, ReceiveQualityProxyError, TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@router.post("/five-signal/outcomes/{outcome}/analyze")
async def five_signal_outcome_analyze(
    outcome: str,
    request: OutcomeAnalysisRequest,
) -> dict:
    try:
        return await run_in_threadpool(
            analyze_outcome,
            outcome,
            request.evaluation_ids,
            refresh=request.refresh,
        )
    except OutcomeCodexUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except OutcomeCodexFailure as error:
        raise HTTPException(502, str(error)) from error
    except OutcomeCodexError as error:
        raise HTTPException(422, str(error)) from error


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
