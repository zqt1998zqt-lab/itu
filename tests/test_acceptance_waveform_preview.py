from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server import alignment_runner
from server.acoustic import acceptance_batch, service
from server.acoustic.acceptance_dataset import (
    get_dataset,
    import_dataset,
    template_manifest,
)
from server.acoustic.acceptance_scope import scope_items, select_batch_scope
from server.app import app


SIGNALS = ("microphone", "reference", "far_end", "noise")
SCENE1_CELLS = {"B24", "B25", "B26", "B27", "B28"}
SCENE1_FAMILIES = {"cng_level", "cng_spectrum", "g_mos", "n_mos", "s_mos"}
SCENE1_CASE_ID = "case-0023"
UNKNOWN_DATASET_ID = "f" * 32


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


def _wav(values: np.ndarray, rate: int = 16_000) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _complete_archive() -> bytes:
    manifest = template_manifest()
    shared = {
        "audio/shared/microphone.wav": _wav(
            np.zeros((16_000, 2), dtype=np.float64)
        ),
        "audio/shared/reference.wav": _wav(np.zeros(16_000)),
        "audio/shared/far_end.wav": _wav(np.zeros(16_000)),
        "audio/shared/noise.wav": _wav(np.zeros(16_000)),
    }
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav" for role in SIGNALS
        }
        case["analysis_start_s"] = 0

    stream = BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for name, payload in shared.items():
            archive.writestr(name, payload)
    return stream.getvalue()


@pytest.fixture
def waveform_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)

    fake_cli = tmp_path / "fake-vcp"
    fake_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: fake_cli)

    imported = import_dataset(_complete_archive())
    assert imported["case_count"] == 106
    assert imported["item_count"] == 336
    return imported, TestClient(app), executor


def test_sce1_preview_exposes_b24_to_b28_speech_and_cng_case(
    waveform_environment,
):
    imported, client, _ = waveform_environment
    manifest = template_manifest()
    rows, cases = select_batch_scope(
        manifest,
        test_scope="background_noise",
        mode="WB-wired",
        scene="sce1",
    )
    assert {row["source_cell"] for row in rows} == SCENE1_CELLS
    assert {row["family"] for row in rows} == SCENE1_FAMILIES
    assert [case["id"] for case in cases] == [SCENE1_CASE_ID]

    response = client.get(
        "/api/alignment/acceptance/background-noise/scenes/sce1/preview",
        params={
            "dataset_id": imported["dataset_id"],
            "mode": "WB-wired",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset_id"] == imported["dataset_id"]
    assert payload["test_scope"] == "background_noise"
    assert payload["mode"] == "WB-wired"
    assert payload["scene"] == "sce1"
    dataset = get_dataset(imported["dataset_id"])
    assert dataset is not None
    stored_case = next(
        row
        for row in dataset["manifest"]["cases"]
        if row["id"] == SCENE1_CASE_ID
    )
    assert (
        payload["dataset_manifest_sha256"]
        == dataset["manifest_sha256"]
    )
    assert payload["dataset_assets_sha256"] == dataset["assets_sha256"]
    assert payload["cases"] == [
        {
            "case_id": SCENE1_CASE_ID,
            "conditions": cases[0]["conditions"],
            "families": sorted(SCENE1_FAMILIES),
            "signals": sorted(SIGNALS),
            "signals_sha256": {
                role: dataset["assets"][stored_case["signals"][role]][
                    "sha256"
                ]
                for role in SIGNALS
            },
        }
    ]
    assert set(payload["cases"][0]["families"]) == SCENE1_FAMILIES
    assert set(payload["cases"][0]["signals"]) == set(SIGNALS)
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.headers["etag"].startswith('"')
    visuals = payload["visuals"]
    assert visuals["schema"] == "scoped-acoustic-visual-v1"
    assert visuals["waveform_bins"] == 720
    assert visuals["mel_bands"] == 40
    assert visuals["mel_frames"] == 240
    assert visuals["mel_fft_size"] == 512
    assert visuals["mel_hop_ms"] == 10
    assert visuals["mel_dynamic_range_db"] == 80
    assert set(visuals["cases"]) == {SCENE1_CASE_ID}
    for role in SIGNALS:
        track = visuals["cases"][SCENE1_CASE_ID][role]
        assert track["source_sha256"] == payload["cases"][0]["signals_sha256"][role]
        assert len(base64.b64decode(track["wave_min_i16"])) == 720 * 2
        assert len(base64.b64decode(track["wave_max_i16"])) == 720 * 2
        assert len(base64.b64decode(track["mel_i16"])) == 40 * 240 * 2


@pytest.mark.parametrize("role", SIGNALS)
def test_declared_input_roles_stream_wav_audio(waveform_environment, role):
    imported, client, _ = waveform_environment
    response = client.get(
        "/api/alignment/acceptance/datasets/"
        f"{imported['dataset_id']}/cases/{SCENE1_CASE_ID}/inputs/{role}"
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";", 1)[0] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.headers["etag"].startswith('"')
    assert (
        f"{SCENE1_CASE_ID}-{role}.wav"
        in response.headers["content-disposition"]
    )


def test_unknown_waveform_identifiers_fail_closed(waveform_environment):
    imported, client, _ = waveform_environment

    unknown_scene = client.get(
        "/api/alignment/acceptance/background-noise/scenes/sce5/preview",
        params={
            "dataset_id": imported["dataset_id"],
            "mode": "WB-wired",
        },
    )
    assert unknown_scene.status_code == 422
    assert unknown_scene.json()["detail"] == "未知背景噪声场景"

    unknown_preview_dataset = client.get(
        "/api/alignment/acceptance/background-noise/scenes/sce1/preview",
        params={
            "dataset_id": UNKNOWN_DATASET_ID,
            "mode": "WB-wired",
        },
    )
    assert unknown_preview_dataset.status_code == 404
    assert unknown_preview_dataset.json()["detail"] == "测试集不存在"

    unknown_input_dataset = client.get(
        "/api/alignment/acceptance/datasets/"
        f"{UNKNOWN_DATASET_ID}/cases/{SCENE1_CASE_ID}/inputs/microphone"
    )
    assert unknown_input_dataset.status_code == 404
    assert unknown_input_dataset.json()["detail"] == "测试集不存在"

    unknown_case = client.get(
        "/api/alignment/acceptance/datasets/"
        f"{imported['dataset_id']}/cases/not-a-case/inputs/microphone"
    )
    assert unknown_case.status_code == 404
    assert unknown_case.json()["detail"] == "条件组不存在"

    unknown_role = client.get(
        "/api/alignment/acceptance/datasets/"
        f"{imported['dataset_id']}/cases/{SCENE1_CASE_ID}/inputs/not-a-role"
    )
    assert unknown_role.status_code == 404
    assert unknown_role.json()["detail"] == "条件组没有该输入音频"


def test_waveform_input_rehashes_declared_asset(waveform_environment):
    imported, client, _ = waveform_environment
    dataset = service.BUILD / "acceptance-datasets" / imported["dataset_id"]
    assets = json.loads(
        (dataset / "assets.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (dataset / "manifest.json").read_text(encoding="utf-8")
    )
    case = next(
        row for row in manifest["cases"] if row["id"] == SCENE1_CASE_ID
    )
    relative = case["signals"]["microphone"]
    assert relative in assets
    audio_path = dataset / "assets" / relative
    payload = bytearray(audio_path.read_bytes())
    payload[-1] ^= 0xFF
    audio_path.write_bytes(payload)

    response = client.get(
        "/api/alignment/acceptance/datasets/"
        f"{imported['dataset_id']}/cases/{SCENE1_CASE_ID}/inputs/microphone"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "测试集输入音频身份校验失败"


def test_incomplete_batch_never_exposes_tx_output(waveform_environment):
    imported, client, executor = waveform_environment
    created = client.post(
        "/api/alignment/acceptance/background-noise/scenes/sce1/batches",
        json={
            "dataset_id": imported["dataset_id"],
            "algorithm_cluster_id": "vcp_default",
            "mode": "WB-wired",
        },
    )
    assert created.status_code == 202, created.text
    queued = created.json()
    assert queued["status"] == "queued"
    assert len(executor.submissions) == 1

    batch_id = queued["batch_id"]
    case_directory = (
        Path(service.BUILD)
        / "acceptance-batches"
        / batch_id
        / "cases"
        / SCENE1_CASE_ID
    )
    case_directory.mkdir(parents=True)
    output = case_directory / "tx.after.wav"
    output.write_bytes(_wav(np.zeros(160)))
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    execution = {"output_sha256": {"tx": output_sha256}}
    assessment = case_directory / "assessment.json"
    assessment.write_text(
        json.dumps(
            {"execution": execution, "items": []},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    report_path = (
        Path(service.BUILD)
        / "acceptance-batches"
        / batch_id
        / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "running"
    report["cases"] = [
        {
            "case_id": SCENE1_CASE_ID,
            "status": "running",
            "execution": execution,
            "assessment_sha256": hashlib.sha256(
                assessment.read_bytes()
            ).hexdigest(),
        }
    ]
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    response = client.get(
        "/api/alignment/acceptance/batches/"
        f"{batch_id}/cases/{SCENE1_CASE_ID}/outputs/tx"
    )

    assert output.is_file()
    assert assessment.is_file()
    assert response.status_code == 404
    assert response.json()["detail"] == "条件组产物不存在或未执行成功"

    report["status"] = "completed"
    report["cases"][0]["status"] = "completed"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    completed = client.get(
        "/api/alignment/acceptance/batches/"
        f"{batch_id}/cases/{SCENE1_CASE_ID}/outputs/tx"
    )
    assert completed.status_code == 200, completed.text
    assert completed.content == output.read_bytes()
    assert completed.headers["cache-control"] == (
        "public, max-age=31536000, immutable"
    )
    assert completed.headers["etag"].startswith('"')
