"""Dataset and asynchronous execution endpoints for the alignment batch UI."""
from typing import Annotated
import hashlib
import json
import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from .acceptance_batch import get_batch, list_batch_summaries, start_batch
from .acceptance_dataset import (
    AcceptanceDatasetError,
    dataset_asset,
    get_dataset,
    import_dataset,
    list_datasets,
    template_manifest,
)
from .acceptance_scope import (
    BACKGROUND_NOISE_SCENES,
    BackgroundNoiseScene,
    basic_uplink_items,
    scope_items,
    select_batch_scope,
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
from .receive_configuration import normalize_receive_configuration
from . import service

router = APIRouter(prefix="/api/alignment/acceptance", tags=["alignment-acceptance"])


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    algorithm_cluster_id: str


class ScopedBatchRequest(BatchRequest):
    mode: str


class BackgroundNoiseBatchRequest(ScopedBatchRequest):
    scene: BackgroundNoiseScene


class BasicUplinkRequest(ScopedBatchRequest):
    pass


def _catalog(
    test_scope: str,
    mode: str,
    scene: BackgroundNoiseScene | None = None,
) -> dict:
    try:
        items = scope_items(test_scope, mode, scene)
        return {
            "test_scope": test_scope,
            "mode": mode,
            "scene": scene,
            "item_count": len(items),
            "items": items,
        }
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def _preview_visuals(
    dataset_id: str,
    dataset: dict,
    cases: list[dict],
) -> dict:
    by_sha256: dict[str, dict] = {}
    visual_cases: dict[str, dict] = {}
    for case in cases:
        tracks: dict[str, dict] = {}
        for role, relative in case["signals"].items():
            metadata = dataset["assets"][relative]
            source_sha256 = metadata["sha256"]
            visual = by_sha256.get(source_sha256)
            if visual is None:
                path = dataset_asset(
                    dataset_id,
                    relative,
                    assets=dataset["assets"],
                )
                visual = load_or_build_visualization(path, source_sha256)
                by_sha256[source_sha256] = visual
            tracks[role] = {
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
        visual_cases[case["id"]] = tracks
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


def _create_scoped_batch(request: ScopedBatchRequest, test_scope: str) -> dict:
    try:
        return start_batch(
            request.dataset_id,
            request.algorithm_cluster_id,
            test_scope=test_scope,
            mode=request.mode,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error


def _create_background_scene_batch(request: BackgroundNoiseBatchRequest) -> dict:
    try:
        return start_batch(
            request.dataset_id,
            request.algorithm_cluster_id,
            test_scope="background_noise",
            mode=request.mode,
            scene=request.scene,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/basic-uplink/catalog")
def basic_uplink_catalog(mode: str):
    return _catalog("basic_uplink", mode)


@router.get("/basic-uplink/batches")
def basic_uplink_batches(limit: Annotated[int, Query(ge=1, le=100)] = 20):
    try:
        return list_batch_summaries(limit, test_scope="basic_uplink")
    except OSError as error:
        raise HTTPException(503, "基础上行测试历史目录不可访问") from error


@router.post("/basic-uplink/batches", status_code=202)
def create_basic_uplink_batch(request: BasicUplinkRequest):
    return _create_scoped_batch(request, "basic_uplink")


@router.get("/basic-downlink/catalog")
def basic_downlink_catalog(mode: str):
    return _catalog("basic_downlink", mode)


@router.get("/basic-downlink/batches")
def basic_downlink_batches(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    try:
        return list_batch_summaries(limit, test_scope="basic_downlink")
    except OSError as error:
        raise HTTPException(503, "基础下行测试历史目录不可访问") from error


@router.post("/basic-downlink/batches", status_code=202)
def create_basic_downlink_batch(request: ScopedBatchRequest):
    return _create_scoped_batch(request, "basic_downlink")


@router.get("/echo/catalog")
def echo_catalog(mode: str):
    return _catalog("echo", mode)


@router.get("/echo/batches")
def echo_batches(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    try:
        return list_batch_summaries(limit, test_scope="echo")
    except OSError as error:
        raise HTTPException(503, "回声测试历史目录不可访问") from error


@router.post("/echo/batches", status_code=202)
def create_echo_batch(request: ScopedBatchRequest):
    return _create_scoped_batch(request, "echo")


@router.get("/double-talk/catalog")
def double_talk_catalog(mode: str):
    return _catalog("double_talk", mode)


@router.get("/double-talk/batches")
def double_talk_batches(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    try:
        return list_batch_summaries(limit, test_scope="double_talk")
    except OSError as error:
        raise HTTPException(503, "双讲测试历史目录不可访问") from error


@router.post("/double-talk/batches", status_code=202)
def create_double_talk_batch(request: ScopedBatchRequest):
    return _create_scoped_batch(request, "double_talk")


@router.get("/background-noise/catalog")
def background_noise_catalog(mode: str):
    return _catalog("background_noise", mode)


@router.get("/background-noise/batches")
def background_noise_batches(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    try:
        return list_batch_summaries(limit, test_scope="background_noise")
    except OSError as error:
        raise HTTPException(503, "背景噪声测试历史目录不可访问") from error


@router.post("/background-noise/batches", status_code=202)
def create_background_noise_batch(request: ScopedBatchRequest):
    return _create_scoped_batch(request, "background_noise")


@router.get("/background-noise/scenes/{scene}/catalog")
def background_noise_scene_catalog(mode: str, scene: str):
    if scene not in BACKGROUND_NOISE_SCENES:
        raise HTTPException(422, "未知背景噪声场景")
    try:
        scope_items("background_noise", mode, scene)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return _catalog("background_noise", mode, scene)


@router.get("/background-noise/scenes/{scene}/preview")
def background_noise_scene_preview(
    scene: str,
    dataset_id: str,
    mode: str,
):
    if scene not in BACKGROUND_NOISE_SCENES:
        raise HTTPException(422, "未知背景噪声场景")
    try:
        dataset = get_dataset(dataset_id)
    except AcceptanceDatasetError as error:
        raise HTTPException(409, str(error)) from error
    if dataset is None:
        raise HTTPException(404, "测试集不存在")
    try:
        rows, cases = select_batch_scope(
            dataset["manifest"],
            test_scope="background_noise",
            mode=mode,
            scene=scene,
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    rows_by_id = {row["id"]: row for row in rows}
    payload = {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "dataset_assets_sha256": dataset["assets_sha256"],
        "test_scope": "background_noise",
        "mode": mode,
        "scene": scene,
        "cases": [
            {
                "case_id": case["id"],
                "conditions": case["conditions"],
                "families": sorted(
                    {
                        rows_by_id[item_id]["family"]
                        for item_id in case["item_ids"]
                    }
                ),
                "signals": sorted(case["signals"]),
                "signals_sha256": {
                    role: dataset["assets"][relative]["sha256"]
                    for role, relative in case["signals"].items()
                },
            }
            for case in cases
        ],
    }
    try:
        payload["visuals"] = _preview_visuals(dataset_id, dataset, cases)
    except AcceptanceDatasetError as error:
        raise HTTPException(409, str(error)) from error
    except VisualizationCacheError:
        # A missing or corrupt derived visualization must not hide the
        # condition metadata; the client retains its explicit WAV fallback.
        pass
    etag = (
        f'"{dataset["manifest_sha256"]}-{dataset["assets_sha256"]}-'
        f'{mode}-{scene}-{VISUAL_SCHEMA}"'
    )
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag,
        },
    )


@router.get("/background-noise/scenes/{scene}/batches")
def background_noise_scene_batches(
    scene: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    if scene not in BACKGROUND_NOISE_SCENES:
        raise HTTPException(422, "未知背景噪声场景")
    try:
        return list_batch_summaries(
            limit,
            test_scope="background_noise",
            scene=scene,
        )
    except OSError as error:
        raise HTTPException(503, "背景噪声场景历史目录不可访问") from error


@router.post("/background-noise/scenes/{scene}/batches", status_code=202)
def create_background_noise_scene_batch(
    scene: str,
    request: ScopedBatchRequest,
):
    if scene not in BACKGROUND_NOISE_SCENES:
        raise HTTPException(422, "未知背景噪声场景")
    return _create_background_scene_batch(
        BackgroundNoiseBatchRequest(
            dataset_id=request.dataset_id,
            algorithm_cluster_id=request.algorithm_cluster_id,
            mode=request.mode,
            scene=scene,
        )
    )


@router.get("/template")
def template():
    manifest = {**template_manifest(), "receive_simulation": normalize_receive_configuration(None)}
    return JSONResponse(manifest, headers={"Content-Disposition": 'attachment; filename="manifest.json"'})


@router.get("/datasets")
def datasets(
    scope: str | None = None,
    mode: str | None = None,
    scene: BackgroundNoiseScene | None = None,
):
    rows = list_datasets()
    if scope is None and mode is None and scene is None:
        return rows
    if scope != "background_noise" or mode is None or scene is None:
        raise HTTPException(
            422,
            "场景测试集筛选必须同时提供 scope=background_noise、mode 和 scene",
        )
    if scene not in BACKGROUND_NOISE_SCENES:
        raise HTTPException(422, "未知背景噪声场景")
    candidates: list[tuple[dict, dict, list[dict]]] = []
    for summary in rows:
        dataset = get_dataset(summary["dataset_id"])
        if dataset is None:
            continue
        try:
            items, cases = select_batch_scope(
                dataset["manifest"],
                test_scope="background_noise",
                mode=mode,
                scene=scene,
            )
        except ValueError:
            continue
        candidates.append((summary, {row["id"]: row for row in items}, cases))
    if not candidates:
        raise HTTPException(422, "没有覆盖当前场景的已校验测试集")
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[dict, dict[str, dict], list[dict]]] = []
    for summary, items, cases in candidates:
        key = (
            str(summary.get("manifest_sha256", "")),
            str(summary.get("assets_sha256", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((summary, items, cases))

    def rank(entry: tuple[dict, dict[str, dict], list[dict]]) -> tuple:
        summary, items, cases = entry
        signals = {
            path
            for case in cases
            for path in case.get("signals", {}).values()
        }
        return (
            1 if "measurement-aware" in str(summary.get("name", "")) else 0,
            len(cases),
            len(signals),
            -len(items),
            str(summary.get("created_at", "")),
            str(summary.get("dataset_id", "")),
        )

    summary, items, cases = max(unique, key=rank)
    signals = {
        path
        for case in cases
        for path in case.get("signals", {}).values()
    }
    return {
        "test_scope": "background_noise",
        "mode": mode,
        "scene": scene,
        "scene_item_count": len(items),
        "scene_case_count": len(cases),
        "scene_signal_count": len(signals),
        "total_dataset_count": len(rows),
        "deduplicated_dataset_count": len(unique),
        "datasets": [summary],
    }


@router.get("/batches")
def batches(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    algorithm_cluster_id: str | None = None,
):
    try:
        return list_batch_summaries(limit, algorithm_cluster_id)
    except OSError as error:
        raise HTTPException(503, "批次历史目录不可访问") from error


@router.post("/datasets", status_code=201)
async def upload_dataset(archive: Annotated[UploadFile, File()]):
    try:
        contents = await archive.read(256 * 1024 * 1024 + 1)
        if len(contents) > 256 * 1024 * 1024:
            raise ValueError("测试集 ZIP 超过 256 MiB")
        return await run_in_threadpool(import_dataset, contents)
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        await archive.close()


@router.get("/datasets/{dataset_id}/cases/{case_id}/inputs/{role}")
def dataset_case_input(dataset_id: str, case_id: str, role: str):
    try:
        dataset = get_dataset(dataset_id)
    except AcceptanceDatasetError as error:
        raise HTTPException(409, str(error)) from error
    if dataset is None:
        raise HTTPException(404, "测试集不存在")
    case = next(
        (
            row
            for row in dataset["manifest"]["cases"]
            if row["id"] == case_id
        ),
        None,
    )
    if case is None:
        raise HTTPException(404, "条件组不存在")
    relative = case.get("signals", {}).get(role)
    if not isinstance(relative, str):
        raise HTTPException(404, "条件组没有该输入音频")
    try:
        path = dataset_asset(
            dataset_id,
            relative,
            assets=dataset["assets"],
        )
    except AcceptanceDatasetError as error:
        raise HTTPException(409, str(error)) from error
    expected = dataset["assets"].get(relative, {}).get("sha256")
    if not isinstance(expected, str):
        raise HTTPException(409, "测试集输入音频身份校验失败")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise HTTPException(409, "测试集输入音频身份校验失败")
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{case_id}-{role}.wav"'
            ),
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{expected}"',
        },
    )


@router.post("/batches", status_code=202)
def create_batch(request: BatchRequest):
    try:
        return start_batch(request.dataset_id, request.algorithm_cluster_id)
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/batches/{batch_id}")
def batch(batch_id: str):
    try:
        result = get_batch(batch_id)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if result is None:
        raise HTTPException(404, "批次不存在")
    return result


@router.get("/batches/{batch_id}/report")
def report(batch_id: str):
    return JSONResponse(batch(batch_id), headers={"Content-Disposition": f'attachment; filename="acceptance-{batch_id}.json"'})


def _case_artifact(batch_id: str, case_id: str, kind: str) -> tuple[Path, bytes]:
    snapshot = batch(batch_id)
    record = next((row for row in snapshot["cases"] if row["case_id"] == case_id), None)
    if record is None or record["status"] != "completed" or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case_id):
        raise HTTPException(404, "条件组产物不存在或未执行成功")
    root = (service.BUILD / "acceptance-batches" / batch_id / "cases" / case_id).resolve()
    base = (service.BUILD / "acceptance-batches" / batch_id).resolve()
    if not root.is_relative_to(base):
        raise HTTPException(404, "非法产物路径")
    assessment = root / "assessment.json"
    if not assessment.is_file() or assessment.is_symlink():
        raise HTTPException(409, "条件组计算报告身份校验失败")
    assessment_payload = assessment.read_bytes()
    if hashlib.sha256(assessment_payload).hexdigest() != record.get("assessment_sha256"):
        raise HTTPException(409, "条件组计算报告身份校验失败")
    if kind == "assessment":
        return assessment, assessment_payload
    evidence = json.loads(assessment_payload)
    expected = record.get("execution", {}).get("output_sha256", {}).get(kind)
    if expected is None:
        raise HTTPException(404, "此条件组没有该方向的输出")
    alignment = record.get("execution", {}).get("output_alignment", {}).get(kind, {})
    filename = alignment.get("analysis_filename", "tx.after.wav" if kind == "tx" else "rx.rx-spk.wav")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise HTTPException(409, "非法输出文件身份")
    output = root / filename
    if (
        not output.is_file()
        or output.is_symlink()
        or not output.resolve().is_relative_to(root)
    ):
        raise HTTPException(409, "VCP 输出身份校验失败")
    output_payload = output.read_bytes()
    if hashlib.sha256(output_payload).hexdigest() != expected:
        raise HTTPException(409, "VCP 输出身份校验失败")
    if evidence.get("execution", {}).get("output_sha256", {}).get(kind) != expected:
        raise HTTPException(409, "输出与计算报告身份不一致")
    return output, output_payload


@router.get("/batches/{batch_id}/cases/{case_id}/assessment")
def case_assessment(batch_id: str, case_id: str):
    _, payload = _case_artifact(batch_id, case_id, "assessment")
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{case_id}-assessment.json"'
            )
        },
    )


@router.get("/batches/{batch_id}/cases/{case_id}/outputs/{kind}")
def case_output(batch_id: str, case_id: str, kind: str):
    if kind not in {"tx", "rx"}:
        raise HTTPException(404, "未知输出方向")
    _, payload = _case_artifact(batch_id, case_id, kind)
    etag = hashlib.sha256(payload).hexdigest()
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{case_id}-{kind}.wav"'
            ),
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{etag}"',
        },
    )
