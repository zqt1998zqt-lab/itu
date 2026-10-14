from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from server.audio import AudioError, MAX_BYTES, Options, ROLES, analyze, decode, demo_wav

app = FastAPI(title="ITU Audio Lab", version="2.0.0")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "itu-engineering-dsp/2.0", "formats": ["WAV", "FLAC", "MP3", "OGG", "M4A", "AAC", "AIFF"], "max_seconds": 120, "max_bytes": MAX_BYTES}


@app.get("/api/demo/{role}")
def demo(role: str):
    if role not in ROLES:
        raise HTTPException(404, "音频角色不存在")
    return Response(demo_wav(role), media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="demo-{role}.wav"'})


@app.post("/api/analyze")
async def analyze_uploads(
    after: Annotated[UploadFile, File()],
    options: Annotated[str, Form()] = "{}",
    reference: Annotated[UploadFile | None, File()] = None,
    before: Annotated[UploadFile | None, File()] = None,
    echo: Annotated[UploadFile | None, File()] = None,
):
    uploads = {k: v for k, v in {"reference": reference, "before": before, "after": after, "echo": echo}.items() if v is not None}
    try:
        config = Options.model_validate(json.loads(options))
        blobs = {}
        for role, upload in uploads.items():
            data = await upload.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise AudioError(f"{ROLES[role]}：文件超过 32 MB")
            blobs[role] = (upload.filename or "audio.wav", data)

        def calculate():
            audios = {role: decode(role, name, blob, config.channels.get(role, 0)) for role, (name, blob) in blobs.items()}
            result = analyze(audios, config)
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            return result

        return await run_in_threadpool(calculate)
    except (AudioError, ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        for upload in uploads.values():
            await upload.close()
