from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from server.acoustic import retired_datasets as retirement


OLD_ID = "0123456789abcdef0123456789abcdef"
ACTIVE_ID = sorted(retirement.ACTIVE_R2_DATASET_IDS)[0]
V4_CONTRACT_ID = "fa11d83bbeb6480c98a9eb3bae35045f"
CREATE_PATHS = (
    "/api/alignment/acceptance/scene-condition-batches",
    "/api/alignment/acceptance/batches",
    "/api/alignment/acceptance/basic-uplink/batches",
    "/api/alignment/acceptance/basic-downlink/batches",
    "/api/alignment/acceptance/echo/batches",
    "/api/alignment/acceptance/double-talk/batches",
    "/api/alignment/acceptance/background-noise/batches",
    "/api/alignment/acceptance/scenes/sce1/batches",
    "/api/alignment/acceptance/background-noise/scenes/sce1/batches",
)


def _receipt(
    *,
    retired_ids: list[str] | None = None,
    replacements: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": retirement.RETIREMENT_SCHEMA,
        "active_revision": retirement.ACTIVE_REVISION,
        "retired_dataset_ids": [OLD_ID] if retired_ids is None else retired_ids,
        "replacement_dataset_ids": (
            {OLD_ID: ACTIVE_ID} if replacements is None else replacements
        ),
        "metadata_archive_manifest_sha256": "a" * 64,
    }


def _write_receipt(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _build_app(monkeypatch: pytest.MonkeyPatch, path: Path) -> FastAPI:
    app = FastAPI()

    async def echo_body(request: Request) -> dict[str, Any]:
        body = await request.body()
        return {
            "length": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "body": body.decode("utf-8") if body.startswith(b"{") else None,
            "content_type": request.headers.get("content-type"),
        }

    for create_path in CREATE_PATHS:
        app.add_api_route(
            create_path,
            echo_body,
            methods=["POST"],
            name=f"echo-{create_path}",
        )

    @app.get("/api/alignment/acceptance/batches")
    def historical_batches(dataset_id: str | None = None) -> dict[str, Any]:
        return {"history": True, "dataset_id": dataset_id}

    @app.get("/api/alignment/acceptance/scene-condition-batches")
    def scene_condition_history(
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        return {"history": True, "dataset_id": dataset_id}

    @app.get("/api/alignment/acceptance/batches/{batch_id}/report")
    def historical_report(batch_id: str) -> dict[str, str]:
        return {"batch_id": batch_id}

    @app.get(
        "/api/alignment/acceptance/scene-condition-batches/{batch_id}"
        "/cases/{case_id}/outputs/{kind}"
    )
    def historical_output(
        batch_id: str,
        case_id: str,
        kind: str,
    ) -> dict[str, str]:
        return {"batch_id": batch_id, "case_id": case_id, "kind": kind}

    @app.get("/api/alignment/acceptance/scene-condition-datasets")
    def legacy_dataset() -> dict[str, bool]:
        return {"should_not_run": True}

    monkeypatch.setattr(retirement, "RETIREMENT_PATH", path)
    assert retirement.install_retired_datasets(app) is not None
    return app


def test_missing_receipt_is_inactive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = FastAPI()

    @app.post("/api/alignment/acceptance/batches")
    async def create(request: Request) -> dict[str, str]:
        return {"dataset_id": (await request.json())["dataset_id"]}

    monkeypatch.setattr(
        retirement,
        "RETIREMENT_PATH",
        tmp_path / "missing-retirement.json",
    )
    assert retirement.install_retired_datasets(app) is None
    response = TestClient(app).post(
        "/api/alignment/acceptance/batches",
        json={"dataset_id": OLD_ID},
    )
    assert response.status_code == 200
    assert response.json() == {"dataset_id": OLD_ID}


@pytest.mark.parametrize(
    "url",
    (
        f"/api/alignment/acceptance/datasets/{OLD_ID}/cases",
        f"/api/alignment/acceptance/scene-condition-datasets/{OLD_ID}"
        "/cases/case-1/inputs/microphone",
        "/api/alignment/acceptance/scene-condition-preview"
        f"?dataset_id={OLD_ID}&mode=WB-wired&scene=sce1",
    ),
)
def test_retired_path_or_query_returns_explicit_410(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    url: str,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    response = TestClient(app).get(url)
    assert response.status_code == 410
    assert response.json() == {
        "detail": "该声学数据集已退役，不能再启动新的测试执行",
        "dataset_id": OLD_ID,
        "replacement_dataset_id": ACTIVE_ID,
    }


@pytest.mark.parametrize("path", CREATE_PATHS)
def test_retired_create_batch_body_returns_410(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    response = TestClient(app).post(path, json={"dataset_id": OLD_ID})
    assert response.status_code == 410
    assert response.json()["dataset_id"] == OLD_ID
    assert response.json()["replacement_dataset_id"] == ACTIVE_ID


def test_active_create_body_is_replayed_without_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    body = json.dumps(
        {
            "dataset_id": ACTIVE_ID,
            "algorithm_cluster_id": "cluster-a",
        }
    )
    response = TestClient(app).post(
        "/api/alignment/acceptance/batches",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["length"] == len(body.encode("utf-8"))
    assert payload["sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert json.loads(payload["body"])["dataset_id"] == ACTIVE_ID


def test_historical_lists_and_batch_reads_remain_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    client = TestClient(app)
    for path in (
        "/api/alignment/acceptance/batches",
        "/api/alignment/acceptance/scene-condition-batches",
    ):
        response = client.get(path, params={"dataset_id": OLD_ID})
        assert response.status_code == 200, response.text
        assert response.json() == {
            "history": True,
            "dataset_id": OLD_ID,
        }

    report = client.get(
        f"/api/alignment/acceptance/batches/{OLD_ID}/report"
    )
    assert report.status_code == 200
    assert report.json() == {"batch_id": OLD_ID}

    output = client.get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{OLD_ID}/cases/case-1/outputs/tx"
    )
    assert output.status_code == 200
    assert output.json()["batch_id"] == OLD_ID


def test_legacy_scene_query_returns_current_mode_scene_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mode = "NB-bt"
    scene = "sce2"
    replacement = retirement._active_revision.revision_scene_profile(
        scene,
        mode,
    ).dataset_id
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    response = TestClient(app).get(
        "/api/alignment/acceptance/scene-condition-datasets",
        params={
            "scope": "background_noise",
            "mode": mode,
            "scene": scene,
            "version": "legacy",
        },
    )
    assert response.status_code == 410
    payload = response.json()
    assert payload["dataset_id"] != replacement
    assert payload["replacement_dataset_id"] == replacement


def test_oversized_chunked_json_is_replayed_to_real_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    chunks = (
        b'{"dataset_id":"' + ACTIVE_ID.encode("ascii") + b'","padding":"',
        b"x" * (retirement.MAX_JSON_BODY_BYTES + 1024),
        b'"}',
    )
    body = b"".join(chunks)

    def stream():
        yield from chunks

    response = TestClient(app).post(
        "/api/alignment/acceptance/batches",
        content=stream(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["length"] == len(body)
    assert payload["sha256"] == hashlib.sha256(body).hexdigest()


def test_multipart_create_body_is_not_consumed_by_middleware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_app(
        monkeypatch,
        _write_receipt(tmp_path / "retirement.json", _receipt()),
    )
    content = b"not-audio-but-opaque-multipart-payload"
    response = TestClient(app).post(
        "/api/alignment/acceptance/batches",
        files={"upload": ("sample.wav", content, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["length"] > len(content)
    assert payload["content_type"].startswith("multipart/form-data; boundary=")


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda value: value["retired_dataset_ids"].append(OLD_ID),
            "duplicate",
        ),
        (
            lambda value: value.update(
                retired_dataset_ids=[ACTIVE_ID],
                replacement_dataset_ids={ACTIVE_ID: ACTIVE_ID},
            ),
            "active R2",
        ),
        (
            lambda value: value.update(replacement_dataset_ids={}),
            "cover exactly",
        ),
        (
            lambda value: value.update(
                metadata_archive_manifest_sha256="not-a-sha"
            ),
            "SHA-256",
        ),
        (
            lambda value: value.update(schema="unknown"),
            "schema differs",
        ),
    ),
)
def test_malformed_receipt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    value = _receipt()
    mutate(value)
    path = _write_receipt(tmp_path / "retirement.json", value)
    monkeypatch.setattr(retirement, "RETIREMENT_PATH", path)
    with pytest.raises(retirement.DatasetRetirementError, match=message):
        retirement.install_retired_datasets(FastAPI())


def test_current_contract_replacement_outside_revision_profiles_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_receipt(
        tmp_path / "retirement.json",
        _receipt(replacements={OLD_ID: V4_CONTRACT_ID}),
    )
    monkeypatch.setattr(retirement, "RETIREMENT_PATH", path)
    receipt = retirement.load_retirement_receipt()
    assert receipt is not None
    assert receipt.replacement_dataset_ids[OLD_ID] == V4_CONTRACT_ID


def test_retirement_path_must_be_a_regular_non_symlink_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _write_receipt(tmp_path / "target.json", _receipt())
    symlink = tmp_path / "retirement.json"
    symlink.symlink_to(target)
    monkeypatch.setattr(retirement, "RETIREMENT_PATH", symlink)
    with pytest.raises(retirement.DatasetRetirementError, match="regular file"):
        retirement.install_retired_datasets(FastAPI())
