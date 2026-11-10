from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest
from fastapi.testclient import TestClient

from server.acoustic import service
from server.acoustic.acceptance_scene_conditions import (
    DATASET_ID,
    DATASET_NAME,
    MODE,
    SCENE,
)
from server.app import app


DATASET_PREFIX = "/api/alignment/acceptance/scene-condition-datasets"
PREVIEW_PATH = "/api/alignment/acceptance/scene-condition-preview"
ROLES = {"microphone", "reference", "far_end", "noise"}
CONTRACT_ITEM_IDS = {
    f"{MODE}:B{cell}" for cell in range(24, 29)
}
SHA256 = re.compile(r"^[a-f0-9]{64}$")
REAL_BUILD = (
    Path(__file__).resolve().parents[1]
    / "build"
    / "acoustic-qualification"
)
SIDECAR_MANIFEST = (
    REAL_BUILD
    / "acoustic-scene-condition-datasets"
    / DATASET_ID
    / "manifest.json"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    if Path(service.BUILD).resolve() != REAL_BUILD.resolve():
        pytest.skip("service.BUILD was monkeypatched away from the real sidecar")
    if not SIDECAR_MANIFEST.is_file():
        pytest.skip("the 20-condition engineering-preview sidecar is not generated")
    return TestClient(app)


@pytest.fixture(scope="module")
def preview(client: TestClient) -> dict:
    response = client.get(
        PREVIEW_PATH,
        params={
            "dataset_id": DATASET_ID,
            "mode": MODE,
            "scene": SCENE,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def _signals(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value)
    assert isinstance(value, list)
    return set(value)


def _datasets(value: object) -> list[dict]:
    if isinstance(value, list):
        datasets = value
    else:
        assert isinstance(value, dict)
        datasets = value.get("datasets")
    assert isinstance(datasets, list)
    assert all(isinstance(dataset, dict) for dataset in datasets)
    return datasets


def test_scene_condition_dataset_list_exposes_exactly_one_preview(
    client: TestClient,
):
    response = client.get(
        DATASET_PREFIX,
        params={
            "scope": "background_noise",
            "mode": MODE,
            "scene": SCENE,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["test_scope"] == "background_noise"
    assert payload["mode"] == MODE
    assert payload["scene"] == SCENE
    assert payload["contract_item_count"] == 5
    assert payload["preview_condition_group_count"] == 20
    assert payload["standard_compliance"] is False
    assert payload["formal_test_status"] == "not_measured"
    datasets = _datasets(payload)
    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset["dataset_id"] == DATASET_ID
    assert dataset.get("dataset_name", dataset.get("name")) == DATASET_NAME
    assert dataset["preview_condition_group_count"] == 20
    assert dataset["standard_compliance"] is False


def test_scene_condition_preview_matches_the_twenty_group_contract(
    preview: dict,
):
    assert preview["dataset_id"] == DATASET_ID
    assert preview["dataset_name"] == DATASET_NAME
    assert preview["dataset_kind"] == "background_noise_idle_preview"
    assert preview["schema_version"] == "scene-condition-preview-v1"
    assert preview["test_scope"] == "background_noise"
    assert preview["mode"] == MODE
    assert preview["scene"] == SCENE
    assert preview["contract_item_count"] == 5
    assert preview["preview_condition_group_count"] == 20
    assert preview["standard_compliance"] is False
    assert preview["formal_test_status"] == "not_measured"

    cases = preview["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 20
    case_ids = [case["case_id"] for case in cases]
    assert len(set(case_ids)) == 20
    covered_items: set[str] = set()
    for case in cases:
        assert set(case) == {
            "case_id",
            "conditions",
            "families",
            "signals",
            "signals_sha256",
            "contract_item_ids",
            "provenance",
        }
        assert isinstance(case["case_id"], str) and case["case_id"]
        assert isinstance(case["conditions"], dict)
        assert case["conditions"]["mode"] == MODE
        assert case["conditions"]["scene"] == SCENE
        assert isinstance(case["families"], list) and case["families"]
        assert _signals(case["signals"]) == ROLES
        assert isinstance(case["contract_item_ids"], list)
        assert case["contract_item_ids"]
        covered_items.update(case["contract_item_ids"])
        assert isinstance(case["provenance"], dict)
        hashes = case["signals_sha256"]
        assert isinstance(hashes, dict)
        assert set(hashes) == ROLES
        assert all(
            isinstance(value, str) and SHA256.fullmatch(value)
            for value in hashes.values()
        )
    assert covered_items == CONTRACT_ITEM_IDS

    visuals = preview["visuals"]
    assert visuals["schema"] == "scoped-acoustic-visual-v1"
    assert set(visuals["cases"]) == set(case_ids)
    for case in cases:
        tracks = visuals["cases"][case["case_id"]]
        assert isinstance(tracks, dict)
        assert set(tracks) == ROLES
        for role, track in tracks.items():
            assert track["source_sha256"] == case["signals_sha256"][role]


def test_scene_condition_input_asset_streams_the_declared_wav(
    client: TestClient,
    preview: dict,
):
    case = preview["cases"][0]
    case_id = case["case_id"]
    role = "microphone"
    response = client.get(
        f"{DATASET_PREFIX}/{DATASET_ID}/cases/{case_id}/inputs/{role}"
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";", 1)[0] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert hashlib.sha256(response.content).hexdigest() == (
        case["signals_sha256"][role]
    )


def test_scene_condition_preview_fails_closed_for_bad_identifiers(
    client: TestClient,
    preview: dict,
):
    case_id = preview["cases"][0]["case_id"]
    unknown_case = client.get(
        f"{DATASET_PREFIX}/{DATASET_ID}/cases/not-a-case/inputs/microphone"
    )
    assert unknown_case.status_code == 404

    unknown_role = client.get(
        f"{DATASET_PREFIX}/{DATASET_ID}/cases/{case_id}/inputs/not-a-role"
    )
    assert unknown_role.status_code == 404

    invalid_query = client.get(
        DATASET_PREFIX,
        params={"scope": "background_noise", "scene": SCENE},
    )
    assert invalid_query.status_code == 422


def test_legacy_acceptance_dataset_routes_remain_compatible(
    client: TestClient,
):
    unfiltered = client.get("/api/alignment/acceptance/datasets")
    assert unfiltered.status_code == 200, unfiltered.text
    assert isinstance(unfiltered.json(), list)
    assert len(unfiltered.json()) == 6

    scene_view = client.get(
        "/api/alignment/acceptance/datasets",
        params={
            "scope": "background_noise",
            "mode": MODE,
            "scene": SCENE,
        },
    )
    assert scene_view.status_code == 200, scene_view.text
    scene_payload = scene_view.json()
    assert scene_payload["test_scope"] == "background_noise"
    assert scene_payload["mode"] == MODE
    assert scene_payload["scene"] == SCENE
    assert scene_payload["scene_item_count"] == 5
    assert scene_payload["scene_case_count"] == 2
    assert scene_payload["scene_signal_count"] == 7
    assert len(scene_payload["datasets"]) == 1

    scene_dataset_id = scene_payload["datasets"][0]["dataset_id"]
    legacy_preview = client.get(
        f"/api/alignment/acceptance/background-noise/scenes/{SCENE}/preview",
        params={
            "dataset_id": scene_dataset_id,
            "mode": MODE,
            "visual_schema": "scoped-acoustic-visual-v1",
        },
    )
    assert legacy_preview.status_code == 200, legacy_preview.text
    legacy_cases = legacy_preview.json()["cases"]
    assert len(legacy_cases) == 2
    assert [case["case_id"] for case in legacy_cases] == [
        "v2-0039-speech",
        "v2-0040-cng",
    ]


def test_scene_condition_runtime_contract_binding_is_reconciled():
    binding_path = (
        SIDECAR_MANIFEST.parent / "runtime-contract-binding.json"
    )
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "server"
        / "acoustic"
        / "acceptance_scene_conditions.py"
    )
    assert binding_path.is_file()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["status"] == "reconciled"
    assert binding["dataset_id"] == DATASET_ID
    assert binding["generation_contract_sha256"] == (
        "7cc511140215a4b082fb60f98ddf5cbf276743b603890c0ef133389c4c831fc2"
    )
    assert binding["runtime_contract_sha256"] == hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    assert binding["runtime_contract_sha256"] == (
        "029488f8075e009fea46248da0b6c40a75806ca4b0c00558ab2a9fc59c75b86e"
    )
    assert binding["difference"]["scope"] == "module_export_only"
    assert binding["reconstruction_check"][
        "matches_generation_contract_sha256"
    ] is True
    assert set(binding["artifact_sha256"]) == {
        "manifest.json",
        "assets.json",
        "summary.json",
        "quality.json",
    }
    for name, expected_sha in binding["artifact_sha256"].items():
        actual_sha = hashlib.sha256(
            (binding_path.parent / name).read_bytes()
        ).hexdigest()
        assert actual_sha == expected_sha


@pytest.mark.parametrize(
    "changes",
    [
        {"difference": []},
        {"runtime_contract_sha256": "0" * 64},
        {"generation_contract_sha256": "0" * 64},
        {"artifact_sha256": {}},
    ],
)
def test_scene_condition_binding_errors_fail_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, changes: dict,
):
    binding_path = SIDECAR_MANIFEST.parent / "runtime-contract-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding.update(changes)
    original_read = Path.read_text

    def read_binding(path: Path, *args, **kwargs):
        if path == binding_path:
            return json.dumps(binding)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_binding)
    for endpoint, params in (
        (DATASET_PREFIX, {"scope": "background_noise", "mode": MODE, "scene": SCENE}),
        (PREVIEW_PATH, {"dataset_id": DATASET_ID, "mode": MODE, "scene": SCENE}),
    ):
        response = client.get(endpoint, params=params)
        assert response.status_code == 503, response.text
        assert isinstance(response.json()["detail"], str)


def test_scene_condition_missing_binding_fails_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
):
    binding_path = SIDECAR_MANIFEST.parent / "runtime-contract-binding.json"
    original_read = Path.read_text

    def read_binding(path: Path, *args, **kwargs):
        if path == binding_path:
            raise FileNotFoundError("binding unavailable")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_binding)
    response = client.get(
        PREVIEW_PATH, params={"dataset_id": DATASET_ID, "mode": MODE, "scene": SCENE},
    )
    assert response.status_code == 503


def test_scene_condition_missing_visuals_never_returns_wav_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
):
    from server.acoustic import acceptance_scene_condition_api as api

    def unavailable_visuals(_dataset):
        raise api.VisualizationCacheError("visual cache unavailable")

    monkeypatch.setattr(api, "_preview_visuals", unavailable_visuals)
    response = client.get(
        PREVIEW_PATH, params={"dataset_id": DATASET_ID, "mode": MODE, "scene": SCENE},
    )
    assert response.status_code == 503
    assert "cases" not in response.json()
