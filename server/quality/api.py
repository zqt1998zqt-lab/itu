"""Upload API for offline quality comparison. No client-controlled server paths."""
from __future__ import annotations

import json
from pathlib import Path
import re
from threading import Lock
from typing import Annotated
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError
from starlette.concurrency import run_in_threadpool

from .engine import evaluate
from .evaluators import DEFAULT_SCOREQ_ASSETS, PROJECT, create_evaluators
from .metrics import load_audio
from .ts103281 import (
    MODEL_DIR,
    ModelAError,
    evaluate_model_a,
    load_official_validation,
)

router = APIRouter(prefix="/api/quality", tags=["Multi-model audio quality"])
RUNS = PROJECT / "build/quality-evaluations/runs"
CACHE = PROJECT / "build/quality-evaluations/cache"
JOB_LOCK = Lock()
_MODELS = {}
MAX_BYTES = 32 * 1024 ** 2


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str
    start_s: float
    end_s: float


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluators: list[str] = Field(default_factory=lambda:["dnsmos", "scoreq"])
    output_names: list[str] | None = None
    windows: list[Window] | None = None
    channels: dict[str, StrictInt] = Field(default_factory=dict)
    offsets: dict[str, StrictInt] = Field(default_factory=dict)
    pcm16_before: bool = False
    diagnostic_mos_deadband: float = .1


def _evaluators(names: tuple[str, ...]):
    if names in _MODELS:
        return _MODELS[names]
    items = create_evaluators(list(names), CACHE)
    if all(s.identity.get("available", True) for s in items):
        _MODELS[names] = items
    return items


@router.get("/engines")
def engines():
    return dict(evaluators=[
        dict(id="dnsmos", native_sample_rates=[16000], minimum_seconds=9.01,
             assets_present=(PROJECT / "build/ecnr-v2-20261011/perceptual-assets/sig_bak_ovr.onnx").is_file(),
             outputs=["SIG", "BAK", "OVRL"]),
        dict(id="scoreq", native_sample_rates=[16000], minimum_seconds=1.,
             assets_present=(DEFAULT_SCOREQ_ASSETS / "manifest.json").is_file(), outputs=["MOS"]),
        dict(id="ts103281-model-a", native_sample_rates=[48000], minimum_seconds=.25,
             assets_present=all((MODEL_DIR / name).is_file() for name in
                                ("S-MOS.model", "N-MOS.model", "G-MOS.model")),
             outputs=["S-MOS", "N-MOS", "G-MOS"],
             input_resampling=True, requires_reference=True,
             compliance_status="experimental_nonconformant",
             verification=load_official_validation())],
        assets_verified="on model initialization", inference_downloads=False,
        input_resampling=False, external_vad=False,
        scope="Offline diagnostic predictions; not ITU certification")


@router.post("/ts103281/model-a")
async def evaluate_ts103281_model_a(
    reference: Annotated[UploadFile, File()],
    degraded: Annotated[UploadFile, File()],
    reference_channel: Annotated[int, Form()] = 0,
    degraded_channel: Annotated[int, Form()] = 0,
):
    uploads = (reference, degraded)
    try:
        blobs = {}
        total = 0
        for name, upload in (("reference", reference), ("degraded", degraded)):
            blob = await upload.read(MAX_BYTES + 1)
            total += len(blob)
            if len(blob) > MAX_BYTES or total > 64 * 1024 ** 2:
                raise ValueError("upload limit: 32 MiB/file, 64 MiB/request")
            blobs[name] = blob

        def calculate():
            if not JOB_LOCK.acquire(blocking=False):
                raise HTTPException(429, "quality evaluator is busy; retry after the current job")
            try:
                ref = load_audio(blobs["reference"], channel=reference_channel)
                deg = load_audio(blobs["degraded"], channel=degraded_channel)
                if len(ref.samples) / ref.rate > 120 or len(deg.samples) / deg.rate > 120:
                    raise ValueError("maximum Model A duration is 120 seconds")
                identifier = uuid.uuid4().hex
                result = evaluate_model_a(
                    ref.samples,
                    deg.samples,
                    ref.rate,
                    degraded_rate=deg.rate,
                )
                result.update(
                    evaluation_id=identifier,
                    inputs={
                        "reference": ref.identity,
                        "degraded": deg.identity,
                    },
                )
                from cpp.tools.ecnr_perceptual import save_json
                save_json(RUNS / (identifier + ".json"), result)
                return dict(
                    id=identifier,
                    report_url=f"/api/quality/runs/{identifier}",
                    report=result,
                )
            finally:
                JOB_LOCK.release()

        return await run_in_threadpool(calculate)
    except ModelAError as error:
        raise HTTPException(422, str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        for upload in uploads:
            await upload.close()


@router.post("/evaluate")
async def evaluate_uploads(before: Annotated[UploadFile, File()],
                           outputs: Annotated[list[UploadFile], File()],
                           clean: Annotated[UploadFile | None, File()] = None,
                           options: Annotated[str, Form()] = "{}"):
    uploads = [before, *outputs, *([clean] if clean is not None else [])]
    try:
        config = Options.model_validate(json.loads(options))
        if not 1 <= len(outputs) <= 8:
            raise ValueError("upload 1..8 output algorithms")
        names = config.output_names if config.output_names is not None else [Path(u.filename or "output").stem for u in outputs]
        if len(names) != len(outputs) or len(set(names)) != len(names) or any(not n or n in {"before", "clean"} for n in names):
            raise ValueError("output_names must be distinct and match uploaded outputs")
        mapping = {"before":before, **dict(zip(names, outputs))}
        if clean is not None:
            mapping["clean"] = clean
        if (set(config.channels) | set(config.offsets)) - set(mapping):
            raise ValueError("channel/offset key does not identify an uploaded role")
        blobs, total = {}, 0
        for name, upload in mapping.items():
            blob = await upload.read(MAX_BYTES + 1)
            total += len(blob)
            if len(blob) > MAX_BYTES or total > 128 * 1024 ** 2:
                raise ValueError("upload limit: 32 MiB/file, 128 MiB/request")
            blobs[name] = blob

        def calculate():
            if not JOB_LOCK.acquire(blocking=False):
                raise HTTPException(429, "quality evaluator is busy; retry after the current job")
            try:
                audios = {name:load_audio(blob, channel=config.channels.get(name),
                                         offset_samples=config.offsets.get(name, 0),
                                         pcm16=name == "before" and config.pcm16_before) for name,blob in blobs.items()}
                if any(len(a.samples) / a.rate > 120 for a in audios.values()):
                    raise ValueError("maximum aligned audio length is 120 seconds")
                identifier = uuid.uuid4().hex
                report = evaluate(identifier, audios["before"], {n:audios[n] for n in names},
                                  clean=audios.get("clean"), windows=[w.model_dump() for w in config.windows] if config.windows is not None else None,
                                  evaluators=_evaluators(tuple(config.evaluators)), mos_deadband=config.diagnostic_mos_deadband)
                from cpp.tools.ecnr_perceptual import save_json
                save_json(RUNS / (identifier + ".json"), report)
                return dict(id=identifier, report_url=f"/api/quality/runs/{identifier}", report=report)
            finally:
                JOB_LOCK.release()
        return await run_in_threadpool(calculate)
    except (ValueError, ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(422, str(error)) from error
    except OSError as error:
        raise HTTPException(503, "quality model or evidence storage unavailable") from error
    finally:
        for upload in uploads:
            await upload.close()


@router.get("/runs/{identifier}")
def result(identifier: str):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise HTTPException(404, "unknown quality evaluation")
    path = RUNS / (identifier + ".json")
    if not path.is_file():
        raise HTTPException(404, "unknown quality evaluation")
    return json.loads(path.read_text())
