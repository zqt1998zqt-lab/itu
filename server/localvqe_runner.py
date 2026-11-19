"""Independent LocalVQE CPU inference endpoints and pinned-asset helpers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import numpy as np
import soundfile as sf
from server.localvqe_ab_results import ROUTES as AB_ROUTES, summarize_ab

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "build/deps/localvqe-models"
LIBRARY = ROOT / "build/localvqe-host-20261010/liblocalvqe.so.0.1.0"
RUNS = ROOT / "build/localvqe-runs"
MAX_BYTES = 32 * 1024 * 1024
MAX_SECONDS = 30
SAMPLE_RATE = 16_000
MIN_SAMPLES = 512

router = APIRouter(prefix="/api/localvqe", tags=["localvqe"])
AB_RESULT = ROOT / "build/mdf-localvqe-ab-verified-20261112-r3-v0JjLL/result.json"
AB_RESULT_SHA256 = "d4a2189c143c7e7a25ede1eb27e625ec41ab2955fb47917038971d5577b1e3d9"

MODELS = {
    "aec200k": {
        "file": "localvqe-v1.4-aec-200K-f32.gguf",
        "sha256": "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731",
        "label": "v1.4-AEC 203K · 线性 DAF + 神经 mask",
        "architecture": "gcc_phat_pbfdkf_neural_mask",
    },
    "aec2k": {
        "file": "localvqe-v1.4-aec-2.7K-f32.gguf",
        "sha256": "d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4",
        "label": "v1.4-AEC 2.7K · 仅线性 DAF",
        "architecture": "gcc_phat_pbfdkf",
    },
    "joint": {
        "file": "localvqe-v1.2-1.3M-f32.gguf",
        "sha256": "4856ecf5f522b23fb2bc5caeac81f323c0ef1c4c156a9c7d40a6adbe092ba9ce",
        "label": "v1.2 1.3M · AEC + NS + 去混响",
        "architecture": "joint_neural_aec",
    },
    "compact": {
        "file": "localvqe-pi-v1-49k-f32.gguf",
        "sha256": "0e0c82a8e9703e818b64dedd0fc306394cf5bbb59fcec1ccca82099d352d0c26",
        "label": "compact 49K · DAF + GTCRN",
        "architecture": "daf_gtcrn",
    },
}


@dataclass(frozen=True)
class ModelAsset:
    identifier: str
    path: Path
    sha256: str
    label: str
    architecture: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(identifier: str) -> ModelAsset:
    model = MODELS.get(identifier)
    if model is None:
        raise HTTPException(422, f"未知 LocalVQE 模型：{identifier}")
    path = ASSETS / model["file"]
    if not path.is_file():
        raise HTTPException(503, f"LocalVQE 权重缺失：{model['file']}")
    actual = sha256_file(path)
    if actual != model["sha256"]:
        raise HTTPException(409, f"LocalVQE 权重 SHA256 不匹配：{model['file']}")
    return ModelAsset(identifier, path, actual, model["label"], model["architecture"])


class _Library:
    """Single shared handle; LocalVQE contexts are not thread-safe."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.handle: ctypes.CDLL | None = None

    def get(self) -> ctypes.CDLL:
        with self.lock:
            if self.handle is not None:
                return self.handle
            if not LIBRARY.is_file():
                raise RuntimeError("LocalVQE 共享库缺失")
            lib = ctypes.CDLL(str(LIBRARY))
            handle = ctypes.c_uint64
            float_ptr = ctypes.POINTER(ctypes.c_float)
            lib.localvqe_options_new.restype = handle
            lib.localvqe_options_free.argtypes = [handle]
            lib.localvqe_options_set_model_path.argtypes = [handle, ctypes.c_char_p]
            lib.localvqe_options_set_model_path.restype = ctypes.c_int
            lib.localvqe_options_set_threads.argtypes = [handle, ctypes.c_int]
            lib.localvqe_options_set_threads.restype = ctypes.c_int
            lib.localvqe_new_with_options.argtypes = [handle]
            lib.localvqe_new_with_options.restype = handle
            lib.localvqe_free.argtypes = [handle]
            lib.localvqe_process_f32.argtypes = [handle, float_ptr, float_ptr, ctypes.c_int, float_ptr]
            lib.localvqe_process_f32.restype = ctypes.c_int
            lib.localvqe_sample_rate.argtypes = [handle]
            lib.localvqe_sample_rate.restype = ctypes.c_int
            lib.localvqe_hop_length.argtypes = [handle]
            lib.localvqe_hop_length.restype = ctypes.c_int
            lib.localvqe_fft_size.argtypes = [handle]
            lib.localvqe_fft_size.restype = ctypes.c_int
            lib.localvqe_last_error.argtypes = [handle]
            lib.localvqe_last_error.restype = ctypes.c_char_p
            self.handle = lib
            return lib


_LIBRARY = _Library()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localvqe-cpu")


def _decode(name: str, blob: bytes) -> tuple[np.ndarray, int, int]:
    if len(blob) > MAX_BYTES:
        raise ValueError(f"{name} 超过 32 MB")
    import io
    try:
        audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError(f"{name} 不是有效音频：{error}") from error
    if rate != SAMPLE_RATE:
        raise ValueError(f"{name} 必须是 16 kHz，实际为 {rate} Hz")
    if audio.shape[1] != 1:
        raise ValueError(f"{name} 必须是单声道，实际为 {audio.shape[1]} 声道")
    if len(audio) < MIN_SAMPLES:
        raise ValueError(f"{name} 至少需要 {MIN_SAMPLES} 个样本")
    if len(audio) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError(f"{name} 最长支持 {MAX_SECONDS} 秒")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError(f"{name} 含有非有限样本或超过 ±1.0")
    return np.ascontiguousarray(audio[:, 0]), rate, len(audio)


def _inference(asset: ModelAsset, microphone: np.ndarray, playback: np.ndarray, threads: int) -> tuple[np.ndarray, dict]:
    lib = _LIBRARY.get()
    handle = ctypes.c_uint64
    float_ptr = ctypes.POINTER(ctypes.c_float)
    options = lib.localvqe_options_new()
    if not options:
        raise RuntimeError("LocalVQE options 创建失败")
    try:
        if lib.localvqe_options_set_model_path(options, str(asset.path).encode()) != 0:
            raise RuntimeError("LocalVQE 模型路径设置失败")
        if lib.localvqe_options_set_threads(options, threads) != 0:
            raise RuntimeError("LocalVQE 线程设置失败")
        started = time.perf_counter()
        context = lib.localvqe_new_with_options(options)
        load_ms = (time.perf_counter() - started) * 1000
    finally:
        lib.localvqe_options_free(options)
    if not context:
        raise RuntimeError("LocalVQE 权重加载失败或 SHA256 allowlist 拒绝")
    try:
        rate = lib.localvqe_sample_rate(context)
        hop = lib.localvqe_hop_length(context)
        fft = lib.localvqe_fft_size(context)
        if (rate, hop, fft) != (SAMPLE_RATE, 256, 512):
            raise RuntimeError(f"LocalVQE 流契约不匹配：rate={rate} hop={hop} fft={fft}")
        count = len(microphone)
        output = np.zeros(count, dtype="float32")
        started = time.perf_counter()
        result = lib.localvqe_process_f32(
            context,
            microphone.ctypes.data_as(float_ptr),
            playback.ctypes.data_as(float_ptr),
            count,
            output.ctypes.data_as(float_ptr),
        )
        process_ms = (time.perf_counter() - started) * 1000
        if result != 0:
            detail = lib.localvqe_last_error(context)
            raise RuntimeError(detail.decode("utf-8", "replace") if detail else f"LocalVQE 推理退出码 {result}")
        if not np.isfinite(output).all():
            raise RuntimeError("LocalVQE 输出包含非有限样本")
        metrics = {
            "sample_rate": rate,
            "hop_samples": hop,
            "fft_size": fft,
            "samples": count,
            "load_ms": round(load_ms, 3),
            "process_ms": round(process_ms, 3),
            "rtf": round(process_ms / 1000 / (count / SAMPLE_RATE), 6),
            "input_rms": round(float(np.sqrt(np.mean(microphone.astype(np.float64) ** 2))), 8),
            "output_rms": round(float(np.sqrt(np.mean(output.astype(np.float64) ** 2))), 8),
            "delta_rms": round(float(np.sqrt(np.mean((output.astype(np.float64) - microphone.astype(np.float64)) ** 2))), 8),
            "max_abs": round(float(np.max(np.abs(output))), 8),
            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        }
        return output, metrics
    finally:
        lib.localvqe_free(context)


def _model_status() -> list[dict]:
    rows = []
    for identifier, model in MODELS.items():
        path = ASSETS / model["file"]
        available = path.is_file() and sha256_file(path) == model["sha256"] if path.is_file() else False
        rows.append({
            "id": identifier,
            "label": model["label"],
            "architecture": model["architecture"],
            "file": model["file"],
            "sha256": model["sha256"],
            "available": available,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    return rows


@router.get("/models")
def models() -> dict:
    return {
        "engine": "LocalVQE CPU / GGML",
        "library": str(LIBRARY),
        "library_available": LIBRARY.is_file(),
        "sample_rate": SAMPLE_RATE,
        "scope": "独立 LocalVQE 完整 AEC 链路；输入麦克风混合输入与远端播放信号 / 回声参考；不接入现有 VCP MDF 后级",
        "standard_compliance": False,
        "models": _model_status(),
    }


@router.get("/mdf-mask-ab")
def mdf_mask_ab() -> dict:
    """Serve only a completed, independently checked full-duration release."""
    payload, summary = _load_ab_release()
    return {**summary, "result_sha256": AB_RESULT_SHA256,
            "report_url": "/api/localvqe/mdf-mask-ab/report", "result_path": str(AB_RESULT)}


def _load_ab_release() -> tuple[dict, dict]:
    if not AB_RESULT_SHA256 or not AB_RESULT.is_file():
        raise HTTPException(503, "完整 12 秒 A/B 结果尚未完成验收")
    if sha256_file(AB_RESULT) != AB_RESULT_SHA256:
        raise HTTPException(409, "A/B 报告哈希与已验收版本不符")
    try:
        payload = json.loads(AB_RESULT.read_text())
        summary = summarize_ab(payload)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(503, "A/B 报告未通过场景、时延或负控校验") from error
    return payload, summary


@router.get("/mdf-mask-ab/report")
def mdf_mask_ab_report():
    payload, _ = _load_ab_release()
    return Response(json.dumps(payload, ensure_ascii=False, allow_nan=False), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="mdf-localvqe-mask-ab.json"'})


@router.get("/mdf-mask-ab/cases/{case_id}/{route}.wav")
def mdf_mask_ab_audio(case_id: str, route: str):
    payload, _ = _load_ab_release()
    case = next((row for row in payload["cases"] if row["case_id"] == case_id), None)
    if case is None or route not in AB_ROUTES:
        raise HTTPException(404, "指定 A/B 条件组或路线不存在")
    record = case["routes"][route]["artifact"]
    path = AB_RESULT.parent / case["category"] / case["case_id"] / record["path"]
    if not path.is_file():
        raise HTTPException(404, "A/B 输出音频不可用")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise HTTPException(409, "A/B 输出音频哈希与报告不符")
    return Response(content, media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{case_id}-{route}.wav"'})


@router.post("/process")
async def process(
    microphone: UploadFile = File(...),
    playback: UploadFile = File(...),
    model_id: str = Form("aec200k"),
    threads: int = Form(4),
):
    if threads < 1 or threads > 8:
        raise HTTPException(422, "threads 必须是 1..8")
    asset = _asset(model_id)
    microphone_bytes = await microphone.read(MAX_BYTES + 1)
    playback_bytes = await playback.read(MAX_BYTES + 1)
    try:
        mic, _, mic_count = _decode("麦克风混合输入", microphone_bytes)
        ref, _, ref_count = _decode("远端播放信号 / 回声参考", playback_bytes)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    count = min(mic_count, ref_count)
    mic, ref = mic[:count], ref[:count]
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        # The C contexts hold mutable recurrent state; serialize them behind one worker.
        output, metrics = await loop.run_in_executor(_EXECUTOR, _inference, asset, mic, ref, threads)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
    directory = RUNS / run_id
    directory.mkdir(parents=True, exist_ok=False)
    output_path = directory / "output.wav"
    sf.write(output_path, output, SAMPLE_RATE, subtype="FLOAT")
    record = {
        "id": run_id,
        "model": {"id": asset.identifier, "label": asset.label, "architecture": asset.architecture,
                  "file": asset.path.name, "sha256": asset.sha256},
        "inputs": {"microphone": microphone.filename, "playback": playback.filename,
                   "microphone_sha256": hashlib.sha256(microphone_bytes).hexdigest(),
                   "playback_sha256": hashlib.sha256(playback_bytes).hexdigest(),
                   "trimmed_samples": abs(mic_count - ref_count)},
        "metrics": metrics,
        "output": {"path": str(output_path), "download_url": f"/api/localvqe/runs/{run_id}/output.wav"},
        "boundary": "工程实验；仅证明本权重在本机 CPU 上执行，未证明 ITU-T 合规或优于现有 VCP 路线",
        "formal_compliance": False,
    }
    (directory / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


@router.get("/runs/{run_id}/output.wav")
def download(run_id: str):
    if not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise HTTPException(422, "运行 ID 非法")
    path = RUNS / run_id / "output.wav"
    if not path.is_file():
        raise HTTPException(404, "输出音频不存在")
    return Response(path.read_bytes(), media_type="audio/wav",
                    headers={"Content-Disposition": f'attachment; filename="localvqe-{run_id}.wav"'})
