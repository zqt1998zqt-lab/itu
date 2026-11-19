from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from server.app import app
from server.acoustic import acoustic_stress_api, service
from server.acoustic.acoustic_stress_profiles import PROFILES, stress_profile
from server.acoustic.acceptance_contract import required_conditions

PREFIX = "/api/alignment/acceptance/stress-datasets"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize("scope,mode", tuple(PROFILES))
def test_scope_dataset_and_preview_have_100_distinct_workloads(client, scope, mode):
    profile = stress_profile(scope, mode)
    root = Path(service.BUILD) / "acoustic-stress-datasets" / profile.dataset_id
    if not (root / "manifest.json").is_file():
        pytest.skip("generated scope dataset required")
    response = client.get(PREFIX, params={"scope": scope, "mode": mode})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["test_scope"] == scope
    assert data["mode"] == mode
    assert data["preview_condition_group_count"] == 100
    assert data["observation_count"] == 100
    assert data["contract_item_count"] == len(profile.contract_rows)
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["dataset_id"] == profile.dataset_id
    assert data["category_counts"] == profile.category_counts
    response = client.get(f"{PREFIX}/{profile.dataset_id}/preview")
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["schema_version"] == profile.schema_version
    assert preview["dataset_kind"] == profile.dataset_kind
    assert len(preview["cases"]) == 100
    assert len({case["case_id"] for case in preview["cases"]}) == 100
    assert Counter(case["provenance"]["workload_category"] for case in preview["cases"]) == profile.category_counts
    assert len(preview["visuals"]["cases"]) == 100
    contracts = {row["id"]: row for row in profile.contract_rows}
    for case in preview["cases"]:
        assert len(case["contract_item_ids"]) == 1
        item = contracts[case["contract_item_ids"][0]]
        assert case["conditions"] == required_conditions(item)
        assert case["families"] == [item["family"]]
        assert set(case["signals"]) == {"reference", "far_end", "microphone", "noise"}
        tracks = preview["visuals"]["cases"][case["case_id"]]
        for role in case["signals"]:
            assert tracks[role]["source_sha256"] == case["signals_sha256"][role]
            assert tracks[role]["sample_rate"] == (48_000 if mode.startswith("SWB-") else 8_000 if mode.startswith("NB-") else 16_000)
    case = preview["cases"][0]
    response = client.get(f"{PREFIX}/{profile.dataset_id}/cases/{case['case_id']}/inputs/microphone")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert hashlib.sha256(response.content).hexdigest() == case["signals_sha256"]["microphone"]


def test_invalid_scope_mode_unknown_dataset_and_duplicate_query_fail_closed(client):
    assert client.get(PREFIX, params={"scope": "background_noise", "mode": "WB-wired"}).status_code == 422
    assert client.get(PREFIX, params={"scope": "echo", "mode": "bad"}).status_code == 422
    assert client.get(PREFIX + "/" + "0" * 32 + "/preview").status_code == 404
    assert client.get(PREFIX + "?scope=echo&scope=double_talk&mode=WB-wired").status_code == 422
    assert client.get(PREFIX + "?scope=echo&mode=WB-wired&scene=sce1").status_code == 422


def test_scope_mismatch_never_starts_a_batch(client, monkeypatch):
    from server.acoustic import scene_condition_batch_api
    calls = []
    monkeypatch.setattr(scene_condition_batch_api, "start_batch", lambda *args, **kwargs: calls.append((args, kwargs)))
    profile = stress_profile("echo", "WB-wired")
    response = client.post("/api/alignment/acceptance/scene-condition-batches", json={
        "dataset_id": profile.dataset_id, "algorithm_cluster_id": "vcp_unified",
        "mode": "WB-wired", "scene": "baseline", "test_scope": "double_talk",
    })
    assert response.status_code == 422
    assert calls == []


def test_cross_dataset_case_cannot_be_served(client):
    profile = stress_profile("echo", "WB-wired")
    root = Path(service.BUILD) / "acoustic-stress-datasets" / profile.dataset_id
    if not (root / "manifest.json").is_file():
        pytest.skip("generated scope dataset required")
    response = client.get(f"{PREFIX}/{profile.dataset_id}/cases/double_talk-WB-wired-balanced-01-stress-v1/inputs/microphone")
    assert response.status_code == 404
    response = client.get(f"{PREFIX}/{profile.dataset_id}/cases/echo-WB-wired-steady_echo-01-stress-v1/inputs/not-a-role")
    assert response.status_code == 404
