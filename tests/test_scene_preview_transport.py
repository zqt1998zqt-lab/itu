from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server.acoustic import acceptance_scene_condition_api as api
from server.acoustic.scene_conversation_profiles import profile_for_scene
from server.app import app


PREVIEW_PATH = "/api/alignment/acceptance/scene-condition-preview"
PROFILE = profile_for_scene("sce1")
DATASET_ID = PROFILE.dataset_id
MODE = "WB-wired"
SCENE = "sce1"
ROLES = tuple(api.ROLES)
WAVE_MIN = base64.b64encode(b"\x00\x00" * api.WAVEFORM_BINS).decode("ascii")
WAVE_MAX = base64.b64encode(b"\x01\x00" * api.WAVEFORM_BINS).decode("ascii")
MEL = base64.b64encode(
    b"\x00\x00" * api.MEL_BANDS * api.MEL_FRAMES
).decode("ascii")


class RawASGICapture:
    """Capture bytes emitted after GZipMiddleware without HTTP auto-decoding."""

    def __init__(self, asgi_app: ASGIApp) -> None:
        self.asgi_app = asgi_app
        self.bodies: list[bytes] = []

    def clear(self) -> None:
        self.bodies.clear()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.asgi_app(scope, receive, send)
            return

        body = bytearray()

        async def capture(message: Message) -> None:
            if message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
            await send(message)

        await self.asgi_app(scope, receive, capture)
        self.bodies.append(bytes(body))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_visual_track(path: Path, source_sha256: str) -> dict[str, Any]:
    role = Path(path).stem
    return {
        "source_sha256": source_sha256,
        "duration_s": 12,
        "sample_rate": 16_000,
        "channels": 2 if role == "microphone" else 1,
        "peak": 0.25,
        "wave_scale": 1,
        "wave_min_i16": WAVE_MIN,
        "wave_max_i16": WAVE_MAX,
        "mel_i16": MEL,
        "mel_ceiling_db": -5.5,
    }


def _synthetic_dataset(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "scene-condition-dataset"
    root.mkdir()
    cases: list[dict[str, Any]] = []
    for index, (case_id, family_ids, conversation_mode) in enumerate(
        (
            (
                "synthetic-near-01",
                PROFILE.contract_item_ids[:3],
                "near_single",
            ),
            (
                "synthetic-cng-01",
                PROFILE.contract_item_ids[3:],
                "noise_only",
            ),
        ),
        start=1,
    ):
        hashes: dict[str, str] = {}
        signals: dict[str, str] = {}
        for role in ROLES:
            relative = f"case-{index}/{role}.wav"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            signals[role] = relative
            hashes[role] = _sha(f"{case_id}:{role}")
        cases.append(
            {
                "case_id": case_id,
                "conditions": {
                    "mode": MODE,
                    "direction": "send",
                    "scene": SCENE,
                    "volume": "nominal",
                    "window_ms": None,
                    "input_level_db": 0,
                },
                "families": (
                    ["g_mos", "n_mos", "s_mos"]
                    if conversation_mode != "noise_only"
                    else ["cng_level", "cng_spectrum"]
                ),
                "signals": signals,
                "signals_sha256": hashes,
                "contract_item_ids": list(family_ids),
                "provenance": {
                    "engineering_preview": True,
                    "actual_vehicle_recording": False,
                    "conversation_mode": conversation_mode,
                },
            }
        )
    return {
        "root": root,
        "dataset_id": DATASET_ID,
        "dataset_name": PROFILE.dataset_name,
        "dataset_manifest_sha256": _sha("manifest"),
        "dataset_assets_sha256": _sha("assets"),
        "test_scope": "background_noise",
        "mode": MODE,
        "scene": SCENE,
        "cases": cases,
        "preview_condition_group_count": len(cases),
        "observation_count": 7,
        "conversation_counts": dict(PROFILE.conversation_counts or {}),
        "contract_item_count": 5,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "dataset_kind": PROFILE.dataset_kind,
        "manifest": {"schema_version": PROFILE.schema_version},
    }


@pytest.fixture
def transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[TestClient, RawASGICapture]:
    dataset = _synthetic_dataset(tmp_path)
    monkeypatch.setattr(api, "_load_dataset", lambda _dataset_id: dataset)
    monkeypatch.setattr(api, "_visual_track", _fake_visual_track)
    capture = RawASGICapture(app)
    with TestClient(capture) as client:
        yield client, capture


def _params(visual_detail: str | None = None) -> dict[str, str]:
    value = {
        "dataset_id": DATASET_ID,
        "mode": MODE,
        "scene": SCENE,
    }
    if visual_detail is not None:
        value["visual_detail"] = visual_detail
    return value


def test_scene_preview_full_default_and_waveform_keep_case_metadata(
    transport: tuple[TestClient, RawASGICapture],
) -> None:
    client, _capture = transport
    full_response = client.get(PREVIEW_PATH, params=_params())
    light_response = client.get(
        PREVIEW_PATH,
        params=_params("waveform"),
    )
    assert full_response.status_code == 200, full_response.text
    assert light_response.status_code == 200, light_response.text
    full = full_response.json()
    light = light_response.json()

    assert "visual_detail" not in full
    assert "detail" not in full["visuals"]
    assert light["visual_detail"] == "waveform"
    assert light["visuals"]["detail"] == "waveform"
    assert set(light) == set(full) | {"visual_detail"}
    assert set(light["visuals"]) == set(full["visuals"]) | {"detail"}
    for key in full:
        if key != "visuals":
            assert light[key] == full[key], key
    for key in full["visuals"]:
        if key != "cases":
            assert light["visuals"][key] == full["visuals"][key], key
    assert light["cases"] == full["cases"]

    for case in full["cases"]:
        case_id = case["case_id"]
        assert set(full["visuals"]["cases"][case_id]) == set(ROLES)
        assert set(light["visuals"]["cases"][case_id]) == set(ROLES)
        for role in ROLES:
            full_track = full["visuals"]["cases"][case_id][role]
            light_track = light["visuals"]["cases"][case_id][role]
            assert "mel_i16" in full_track
            assert "mel_ceiling_db" in full_track
            assert "mel_i16" not in light_track
            assert "mel_ceiling_db" not in light_track
            assert light_track == {
                key: value
                for key, value in full_track.items()
                if key not in {"mel_i16", "mel_ceiling_db"}
            }


def test_scene_preview_etags_are_variant_specific_and_revalidate(
    transport: tuple[TestClient, RawASGICapture],
) -> None:
    client, _capture = transport
    full = client.get(PREVIEW_PATH, params=_params())
    light = client.get(PREVIEW_PATH, params=_params("waveform"))
    assert full.status_code == 200, full.text
    assert light.status_code == 200, light.text
    full_etag = full.headers["etag"]
    light_etag = light.headers["etag"]
    assert full_etag != light_etag

    for params, etag in (
        (_params(), full_etag),
        (_params("waveform"), light_etag),
    ):
        cached = client.get(
            PREVIEW_PATH,
            params=params,
            headers={"If-None-Match": etag},
        )
        assert cached.status_code == 304
        assert cached.content == b""
        assert cached.headers["etag"] == etag

    for params, etag, expected in (
        (_params("waveform"), full_etag, light_etag),
        (_params(), light_etag, full_etag),
    ):
        crossed = client.get(
            PREVIEW_PATH,
            params=params,
            headers={"If-None-Match": etag},
        )
        assert crossed.status_code == 200, crossed.text
        assert crossed.headers["etag"] == expected


def test_scene_preview_rejects_invalid_unknown_and_duplicate_variants(
    transport: tuple[TestClient, RawASGICapture],
) -> None:
    client, _capture = transport
    invalid = client.get(
        PREVIEW_PATH,
        params=_params("spectrogram"),
    )
    assert invalid.status_code == 422

    unknown = client.get(
        PREVIEW_PATH,
        params={**_params("waveform"), "unexpected": "1"},
    )
    assert unknown.status_code == 422

    duplicate = client.get(
        PREVIEW_PATH,
        params=[
            ("dataset_id", DATASET_ID),
            ("mode", MODE),
            ("scene", SCENE),
            ("visual_detail", "waveform"),
            ("visual_detail", "full"),
        ],
    )
    assert duplicate.status_code == 422


@pytest.mark.parametrize("visual_detail", [None, "waveform"])
def test_scene_preview_gzip_matches_identity_and_sets_transport_headers(
    transport: tuple[TestClient, RawASGICapture],
    visual_detail: str | None,
) -> None:
    client, capture = transport
    params = _params(visual_detail)

    capture.clear()
    identity = client.get(
        PREVIEW_PATH,
        params=params,
        headers={"Accept-Encoding": "identity"},
    )
    identity_raw = capture.bodies[-1]
    assert identity.status_code == 200, identity.text
    assert identity.headers.get("content-encoding") is None
    assert int(identity.headers["content-length"]) == len(identity_raw)

    capture.clear()
    compressed = client.get(
        PREVIEW_PATH,
        params=params,
        headers={"Accept-Encoding": "gzip"},
    )
    compressed_raw = capture.bodies[-1]
    assert compressed.status_code == 200, compressed.text
    assert compressed.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in compressed.headers["vary"].lower()
    assert int(compressed.headers["content-length"]) == len(compressed_raw)
    assert compressed.headers["etag"] == identity.headers["etag"]
    assert compressed.headers["cache-control"] == identity.headers["cache-control"]
    assert gzip.decompress(compressed_raw) == identity_raw
    assert compressed.content == identity.content
    assert compressed.json() == identity.json()


def test_demo_wav_is_not_gzipped(
    transport: tuple[TestClient, RawASGICapture],
) -> None:
    client, capture = transport
    capture.clear()
    response = client.get(
        "/api/demo/before",
        headers={"Accept-Encoding": "gzip"},
    )
    raw = capture.bodies[-1]
    assert response.status_code == 200
    assert response.headers["content-type"].split(";", 1)[0] == "audio/wav"
    assert response.headers.get("content-encoding") is None
    assert int(response.headers["content-length"]) == len(raw)
    assert response.content == raw
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
