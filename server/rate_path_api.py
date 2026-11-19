"""Rate-path discovery and the native 8 kHz processing endpoint."""
from __future__ import annotations

from pathlib import Path
import secrets
import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from server.audio import MAX_BYTES
from server import nb8_route
from server.rate_paths import public_rate_paths, selectable_clusters_by_rate


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "build/nb8-api-runs"
router = APIRouter(prefix="/api", tags=["rate-paths"])


@router.get("/rate-paths")
def get_rate_paths(include_legacy: bool = True) -> dict[str, Any]:
    return {
        "primary_sample_rates": [8_000, 16_000, 48_000],
        "legacy_sample_rates": [24_000],
        "paths": public_rate_paths(include_legacy=include_legacy),
        "clusters_by_sample_rate": selectable_clusters_by_rate(
            include_legacy=include_legacy
        ),
    }


def _read_upload(path: Path, upload: UploadFile, *, role: str) -> None:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{role}文件超过 32 MB")
    if not data:
        raise HTTPException(422, f"{role}文件为空")
    path.write_bytes(data)


def _safe_run(run_id: str) -> Path:
    if not run_id or any(
        character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        for character in run_id
    ):
        raise HTTPException(404, "运行记录不存在")
    run = RUNS / run_id
    if not run.is_dir():
        raise HTTPException(404, "运行记录不存在")
    return run


@router.post("/nb8/process")
async def process_nb8(
    microphone: Annotated[UploadFile, File()],
    reference: Annotated[UploadFile, File()],
    tail_ms: Annotated[int, Form()] = 500,
) -> dict[str, Any]:
    if not 10 <= tail_ms <= 10_000:
        raise HTTPException(422, "tail_ms 必须是 10..10000 的整数")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + secrets.token_hex(6)
    run = RUNS / run_id
    run.mkdir(parents=True, exist_ok=False)
    microphone_path = run / "microphone.wav"
    reference_path = run / "reference.wav"
    try:
        _read_upload(microphone_path, microphone, role="麦克风混合输入")
        _read_upload(reference_path, reference, role="远端播放信号 / 回声参考")
        result = await run_in_threadpool(
            nb8_route.run,
            microphone_path,
            reference_path,
            run / "result",
            tail_ms=tail_ms,
        )
    except nb8_route.NB8RouteError as error:
        raise HTTPException(422, str(error)) from error
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(503, str(error)) from error
    finally:
        await microphone.close()
        await reference.close()
    return {
        "run_id": run_id,
        "result": result,
        "output_url": f"/api/nb8/runs/{run_id}/output.wav",
        "report_url": f"/api/nb8/runs/{run_id}/report.json",
    }


@router.get("/nb8/runs/{run_id}/output.wav")
def get_nb8_output(run_id: str) -> FileResponse:
    path = _safe_run(run_id) / "result.after.wav"
    if not path.is_file():
        raise HTTPException(404, "输出不存在")
    return FileResponse(path, media_type="audio/wav", filename="nb8-processed.wav")


@router.get("/nb8/runs/{run_id}/report.json")
def get_nb8_report(run_id: str) -> FileResponse:
    path = _safe_run(run_id) / "result.json"
    if not path.is_file():
        raise HTTPException(404, "报告不存在")
    return FileResponse(path, media_type="application/json", filename="nb8-report.json")
