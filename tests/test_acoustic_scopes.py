from __future__ import annotations

import copy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
import soundfile as sf

from server.acoustic import acceptance_batch, acceptance_linearity, service
from server.acoustic import acceptance_scope
from server.acoustic.acceptance_contract import (
    contract_items,
    required_conditions,
)
from server.acoustic.acceptance_dataset import import_dataset, template_manifest
from server.acoustic.acceptance_scope import (
    BACKGROUND_NOISE_SCENE_CELLS,
    BACKGROUND_NOISE_SCENES,
    BACKGROUND_NOISE_FAMILIES,
    BASIC_UPLINK_FAMILIES,
    basic_downlink_items,
    basic_uplink_items,
    background_noise_items,
    double_talk_items,
    echo_items,
    scope_items,
    select_batch_scope,
)
from server.app import app


MODES = (
    "NB-wired",
    "NB-wireless",
    "NB-bt",
    "WB-wired",
    "WB-wireless",
    "WB-bt",
    "SWB-wired",
    "SWB-wireless",
)
SUB_SCOPES = (
    "basic_downlink",
    "echo",
    "double_talk",
    "background_noise",
)
API_SLUGS = {
    "basic_downlink": "basic-downlink",
    "echo": "echo",
    "double_talk": "double-talk",
    "background_noise": "background-noise",
}
SIGNALS = ("microphone", "reference", "far_end", "noise")


def _expected_count(test_scope: str, mode: str) -> int:
    if test_scope == "basic_uplink":
        return 8
    if test_scope == "basic_downlink":
        return 8 if mode.startswith("SWB-") else 9
    if test_scope == "echo":
        return 3 if mode.startswith("SWB-") else 2
    if test_scope == "double_talk":
        return 2
    if test_scope == "background_noise":
        return 20
    raise AssertionError(test_scope)


def _matches_scope(test_scope: str, row: dict) -> bool:
    if test_scope == "basic_uplink":
        return (
            row["direction"] == "send"
            and row["family"] in BASIC_UPLINK_FAMILIES
        )
    if test_scope == "basic_downlink":
        return row["direction"] == "receive"
    if test_scope == "echo":
        return row["family"] == "tclw"
    if test_scope == "double_talk":
        return row["family"] == "double_talk_classification"
    if test_scope == "background_noise":
        return row["family"] in BACKGROUND_NOISE_FAMILIES
    raise AssertionError(test_scope)


def _wav(values: np.ndarray, rate: int) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _complete_archive(rate: int = 16_000) -> bytes:
    manifest = template_manifest()
    payloads = {
        role: _wav(np.zeros(rate * 3, dtype=np.float64), rate)
        for role in SIGNALS
    }
    shared = {
        f"audio/shared/{role}.wav": payload
        for role, payload in payloads.items()
    }
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav"
            for role in SIGNALS
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


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    return executor


@pytest.mark.parametrize("test_scope", SUB_SCOPES)
@pytest.mark.parametrize("mode", MODES)
def test_scope_catalog_and_manifest_filter_are_contract_driven(
    test_scope,
    mode,
):
    manifest = template_manifest()
    original = copy.deepcopy(manifest)
    rows = scope_items(test_scope, mode)
    selected_rows, cases = select_batch_scope(
        manifest,
        test_scope=test_scope,
        mode=mode,
    )

    expected_count = _expected_count(test_scope, mode)
    assert rows == selected_rows
    assert len(rows) == expected_count
    assert len({row["id"] for row in rows}) == expected_count
    assert {row["sheet"] for row in rows} == {mode}
    assert all(_matches_scope(test_scope, row) for row in rows)
    assert manifest == original

    by_id = {row["id"]: row for row in rows}
    flattened = [
        item_id
        for case in cases
        for item_id in case["item_ids"]
    ]
    assert len(flattened) == len(set(flattened)) == expected_count
    assert set(flattened) == set(by_id)
    assert all(
        required_conditions(by_id[item_id]) == case["conditions"]
        for case in cases
        for item_id in case["item_ids"]
    )
    assert all(
        set(case["item_ids"]) <= set(by_id)
        for case in cases
    )

    cases[0]["item_ids"].append("mutated-copy")
    assert manifest == original


def test_scope_wrappers_preserve_basic_and_full_compatibility():
    manifest = template_manifest()
    original = copy.deepcopy(manifest)

    assert basic_uplink_items("WB-wired") == scope_items(
        "basic_uplink",
        "WB-wired",
    )
    assert basic_downlink_items("WB-wired") == scope_items(
        "basic_downlink",
        "WB-wired",
    )
    assert echo_items("WB-wired") == scope_items("echo", "WB-wired")
    assert double_talk_items("WB-wired") == scope_items(
        "double_talk",
        "WB-wired",
    )
    assert background_noise_items("WB-wired") == scope_items(
        "background_noise",
        "WB-wired",
    )

    full_rows, full_cases = select_batch_scope(manifest)
    assert len(full_rows) == 336
    assert len(full_cases) == 106
    assert sum(len(case["item_ids"]) for case in full_cases) == 336
    assert full_cases == original["cases"]
    assert manifest == original


@pytest.mark.parametrize("scene", sorted(BACKGROUND_NOISE_SCENES))
@pytest.mark.parametrize("mode", ("NB-wired", "WB-wireless", "SWB-wired"))
def test_background_noise_scene_selects_exactly_five_contract_items(
    scene,
    mode,
):
    rows = scope_items("background_noise", mode, scene)
    selected_rows, cases = select_batch_scope(
        template_manifest(),
        test_scope="background_noise",
        mode=mode,
        scene=scene,
    )

    assert rows == selected_rows
    assert len(rows) == 5
    assert {row["source_cell"] for row in rows} == set(
        BACKGROUND_NOISE_SCENE_CELLS[scene]
    )
    assert all(required_conditions(row)["scene"] == scene for row in rows)
    flattened = [
        item_id
        for case in cases
        for item_id in case["item_ids"]
    ]
    assert len(flattened) == len(set(flattened)) == 5
    assert set(flattened) == {row["id"] for row in rows}
    assert all(
        case["conditions"]["scene"] == scene
        for case in cases
    )


def test_background_noise_scene_is_rejected_outside_background_scope():
    with pytest.raises(ValueError, match="只有背景噪声"):
        scope_items("echo", "WB-wired", "sce1")
    with pytest.raises(ValueError, match="未知背景噪声场景"):
        scope_items("background_noise", "WB-wired", "sce5")
    with pytest.raises(ValueError, match="完整批次"):
        select_batch_scope(template_manifest(), scene="sce1")


@pytest.mark.parametrize("test_scope", SUB_SCOPES)
def test_scope_selection_rejects_invalid_mode_scope_missing_duplicate_and_conditions(
    test_scope,
):
    manifest = template_manifest()
    with pytest.raises(ValueError, match="未知测试范围"):
        select_batch_scope(manifest, test_scope="partial")
    with pytest.raises(ValueError, match="业务模式"):
        select_batch_scope(
            manifest,
            test_scope=test_scope,
        )
    with pytest.raises(ValueError, match="业务模式"):
        select_batch_scope(
            manifest,
            test_scope=test_scope,
            mode="WB-invalid",
        )
    with pytest.raises(ValueError, match="不能指定单一模式"):
        select_batch_scope(manifest, mode="WB-wired")

    selected = scope_items(test_scope, "WB-wired")
    target_id = selected[0]["id"]

    missing = copy.deepcopy(manifest)
    for case in missing["cases"]:
        case["item_ids"] = [
            item_id
            for item_id in case["item_ids"]
            if item_id != target_id
        ]
    with pytest.raises(ValueError, match="未唯一覆盖"):
        select_batch_scope(
            missing,
            test_scope=test_scope,
            mode="WB-wired",
        )

    duplicate = copy.deepcopy(manifest)
    duplicate_case = next(
        case
        for case in duplicate["cases"]
        if target_id in case["item_ids"]
    )
    duplicate_case["item_ids"].append(target_id)
    with pytest.raises(ValueError, match="未唯一覆盖"):
        select_batch_scope(
            duplicate,
            test_scope=test_scope,
            mode="WB-wired",
        )

    mismatched = copy.deepcopy(manifest)
    mismatched_case = next(
        case
        for case in mismatched["cases"]
        if target_id in case["item_ids"]
    )
    mismatched_case["conditions"]["window_ms"] = (
        100
        if mismatched_case["conditions"]["window_ms"] is None
        else None
    )
    with pytest.raises(ValueError, match="采集工况不一致"):
        select_batch_scope(
            mismatched,
            test_scope=test_scope,
            mode="WB-wired",
        )


def test_scope_contract_count_and_identity_validation_do_not_use_source_cell(
    monkeypatch,
):
    rows = contract_items()
    target = next(
        row
        for row in rows
        if row["sheet"] == "WB-wired"
        and row["direction"] == "receive"
    )
    missing = [
        row
        for row in rows
        if row["id"] != target["id"]
    ]
    monkeypatch.setattr(acceptance_scope, "contract_items", lambda: missing)
    with pytest.raises(ValueError, match="9 个唯一测项"):
        scope_items("basic_downlink", "WB-wired")

    duplicate = [*rows, copy.deepcopy(target)]
    monkeypatch.setattr(acceptance_scope, "contract_items", lambda: duplicate)
    with pytest.raises(ValueError, match="9 个唯一测项"):
        scope_items("basic_downlink", "WB-wired")

    renamed = copy.deepcopy(rows)
    for row in renamed:
        row["source_cell"] = "not-used-by-selection"
    monkeypatch.setattr(acceptance_scope, "contract_items", lambda: renamed)
    selected = scope_items("basic_downlink", "WB-wired")
    assert {row["id"] for row in selected} == {
        row["id"]
        for row in rows
        if row["sheet"] == "WB-wired"
        and row["direction"] == "receive"
    }


def test_basic_downlink_keeps_three_input_levels_and_swb_skips_linearity(
    tmp_path,
):
    _, wb_cases = select_batch_scope(
        template_manifest(),
        test_scope="basic_downlink",
        mode="WB-wired",
    )
    levels = {
        case["conditions"]["input_level_db"]
        for case in wb_cases
    }
    assert {-5, 0, 5} <= levels

    rows = basic_downlink_items("SWB-wired")
    assert not any(row["family"] == "rlr_linearity" for row in rows)
    snapshot = {
        "items": [
            {
                "item_id": row["id"],
                "mode": row["sheet"],
                "family": row["family"],
                "status": "review_required",
            }
            for row in rows
        ],
    }
    destination = tmp_path / "swb-linearity"
    acceptance_linearity.finish_linearity(
        snapshot,
        {"manifest": template_manifest()},
        destination,
    )
    assert not destination.exists()


@pytest.mark.parametrize(
    ("test_scope", "mode"),
    (
        ("basic_downlink", "SWB-wired"),
        ("echo", "WB-wired"),
        ("double_talk", "WB-wired"),
        ("background_noise", "SWB-wireless"),
    ),
)
def test_scope_api_catalog_create_and_history_are_isolated(
    isolated_store,
    test_scope,
    mode,
):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)
    slug = API_SLUGS[test_scope]
    expected_count = _expected_count(test_scope, mode)

    catalog = client.get(
        f"/api/alignment/acceptance/{slug}/catalog",
        params={"mode": mode},
    )
    assert catalog.status_code == 200
    assert catalog.json()["test_scope"] == test_scope
    assert catalog.json()["mode"] == mode
    assert catalog.json()["item_count"] == expected_count
    assert len(catalog.json()["items"]) == expected_count
    assert client.get(
        f"/api/alignment/acceptance/{slug}/catalog",
        params={"mode": "WB-invalid"},
    ).status_code == 422

    request = {
        "dataset_id": imported["dataset_id"],
        "algorithm_cluster_id": "vcp_default",
        "mode": mode,
    }
    created = client.post(
        f"/api/alignment/acceptance/{slug}/batches",
        json=request,
    )
    assert created.status_code == 202, created.text
    queued = created.json()
    assert queued["test_scope"] == test_scope
    assert queued["selected_mode"] == mode
    assert queued["progress"]["total_items"] == expected_count
    assert len(queued["items"]) == expected_count
    assert queued["batch_id"] in acceptance_batch._ACTIVE

    history = client.get(
        f"/api/alignment/acceptance/{slug}/batches"
    )
    assert history.status_code == 200
    assert [row["batch_id"] for row in history.json()] == [
        queued["batch_id"]
    ]
    assert history.json()[0]["test_scope"] == test_scope
    assert history.json()[0]["selected_mode"] == mode
    assert history.json()[0]["summary"]["total"] == expected_count
    assert client.get("/api/alignment/acceptance/batches").json() == []

    assert client.post(
        f"/api/alignment/acceptance/{slug}/batches",
        json={**request, "test_scope": test_scope},
    ).status_code == 422
    missing_mode = dict(request)
    missing_mode.pop("mode")
    assert client.post(
        f"/api/alignment/acceptance/{slug}/batches",
        json=missing_mode,
    ).status_code == 422
    assert len(isolated_store.submissions) == 1


def test_scope_api_history_excludes_other_scopes_and_dynamic_counts(
    isolated_store,
):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)
    created = []
    selected_modes = {
        "basic_downlink": "SWB-wired",
        "echo": "NB-wired",
        "double_talk": "WB-wireless",
        "background_noise": "SWB-wireless",
    }
    for test_scope in SUB_SCOPES:
        mode = selected_modes[test_scope]
        response = client.post(
            f"/api/alignment/acceptance/{API_SLUGS[test_scope]}/batches",
            json={
                "dataset_id": imported["dataset_id"],
                "algorithm_cluster_id": "vcp_default",
                "mode": mode,
            },
        )
        assert response.status_code == 202, response.text
        created.append((test_scope, mode, response.json()))
        acceptance_batch._ACTIVE.clear()

    for test_scope, mode, queued in created:
        history = client.get(
            f"/api/alignment/acceptance/{API_SLUGS[test_scope]}/batches"
        ).json()
        assert [(row["batch_id"], row["selected_mode"]) for row in history] == [
            (queued["batch_id"], mode)
        ]
        assert history[0]["summary"]["total"] == _expected_count(
            test_scope,
            mode,
        )
    assert client.get("/api/alignment/acceptance/batches").json() == []
    assert len(isolated_store.submissions) == len(SUB_SCOPES)


@pytest.mark.parametrize("scene", sorted(BACKGROUND_NOISE_SCENES))
def test_background_noise_scene_api_catalog_create_and_history_are_isolated(
    isolated_store,
    scene,
):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)
    slug = f"/api/alignment/acceptance/background-noise/scenes/{scene}"

    catalog = client.get(f"{slug}/catalog", params={"mode": "WB-wired"})
    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["test_scope"] == "background_noise"
    assert catalog.json()["scene"] == scene
    assert catalog.json()["item_count"] == 5
    assert {
        row["source_cell"]
        for row in catalog.json()["items"]
    } == set(BACKGROUND_NOISE_SCENE_CELLS[scene])

    request = {
        "dataset_id": imported["dataset_id"],
        "algorithm_cluster_id": "vcp_default",
        "mode": "WB-wired",
    }
    created = client.post(f"{slug}/batches", json=request)
    assert created.status_code == 202, created.text
    queued = created.json()
    assert queued["selected_scene"] == scene
    assert queued["progress"]["total_items"] == 5
    assert len(queued["items"]) == 5
    assert all(
        item["evidence_binding"]["scene"] == scene
        for item in queued["items"]
    )

    history = client.get(f"{slug}/batches")
    assert history.status_code == 200
    assert [row["batch_id"] for row in history.json()] == [
        queued["batch_id"]
    ]
    assert history.json()[0]["selected_scene"] == scene
    assert history.json()[0]["summary"]["total"] == 5
    assert client.get(
        "/api/alignment/acceptance/background-noise/scenes/sce5/catalog",
        params={"mode": "WB-wired"},
    ).status_code == 422
    assert len(isolated_store.submissions) == 1


def test_background_noise_scene_histories_do_not_cross_scenes(
    isolated_store,
):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)
    created = {}
    for scene in sorted(BACKGROUND_NOISE_SCENES):
        response = client.post(
            f"/api/alignment/acceptance/background-noise/scenes/{scene}/batches",
            json={
                "dataset_id": imported["dataset_id"],
                "algorithm_cluster_id": "vcp_default",
                "mode": "WB-wired",
            },
        )
        assert response.status_code == 202, response.text
        created[scene] = response.json()["batch_id"]
        acceptance_batch._ACTIVE.clear()

    for scene, batch_id in created.items():
        history = client.get(
            f"/api/alignment/acceptance/background-noise/scenes/{scene}/batches"
        )
        assert history.status_code == 200
        assert [row["batch_id"] for row in history.json()] == [batch_id]
        assert history.json()[0]["selected_scene"] == scene
        assert history.json()[0]["summary"]["total"] == 5

    parent = client.post(
        "/api/alignment/acceptance/background-noise/batches",
        json={
            "dataset_id": imported["dataset_id"],
            "algorithm_cluster_id": "vcp_default",
            "mode": "WB-wired",
        },
    )
    assert parent.status_code == 202, parent.text
    parent_batch_id = parent.json()["batch_id"]
    assert parent.json()["selected_scene"] is None
    assert parent.json()["progress"]["total_items"] == 20

    parent_history = client.get(
        "/api/alignment/acceptance/background-noise/batches"
    )
    assert parent_history.status_code == 200
    assert [row["batch_id"] for row in parent_history.json()] == [
        parent_batch_id
    ]
    assert parent_history.json()[0]["selected_scene"] is None
    assert parent_history.json()[0]["summary"]["total"] == 20
    assert len(isolated_store.submissions) == len(BACKGROUND_NOISE_SCENES) + 1


@pytest.mark.parametrize("test_scope", SUB_SCOPES)
def test_non_vcp_is_rejected_for_every_subscope(
    isolated_store,
    test_scope,
):
    imported = import_dataset(_complete_archive())
    with pytest.raises(ValueError, match="VCP"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "array_control",
            test_scope=test_scope,
            mode="WB-wired",
        )
    assert isolated_store.submissions == []
    assert not acceptance_batch._ACTIVE


def test_unknown_scope_is_rejected_before_batch_creation(isolated_store):
    imported = import_dataset(_complete_archive())
    with pytest.raises(ValueError, match="未知测试范围"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
            test_scope="partial",
            mode="WB-wired",
        )
    assert isolated_store.submissions == []
    assert not acceptance_batch._ACTIVE


def test_all_scopes_share_active_and_legacy_locks(
    isolated_store,
    monkeypatch,
):
    imported = import_dataset(_complete_archive())
    queued = acceptance_batch.start_batch(
        imported["dataset_id"],
        "vcp_default",
        test_scope="basic_downlink",
        mode="WB-wired",
    )
    for test_scope, mode in (
        ("echo", "WB-wired"),
        ("double_talk", "WB-wired"),
        ("background_noise", "WB-wired"),
    ):
        with pytest.raises(ValueError, match="已有|正在执行"):
            acceptance_batch.start_batch(
                imported["dataset_id"],
                "vcp_default",
                test_scope=test_scope,
                mode=mode,
            )
    with pytest.raises(ValueError, match="已有|正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
        )

    acceptance_batch._ACTIVE.clear()
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", True)
    with pytest.raises(ValueError, match="正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
            test_scope="background_noise",
            mode="NB-wired",
        )
    assert queued["batch_id"] not in acceptance_batch._ACTIVE
    assert len(isolated_store.submissions) == 1


@pytest.mark.parametrize(
    ("test_scope", "mode", "scene"),
    (
        ("basic_downlink", "WB-wired", None),
        ("echo", "WB-wired", None),
        ("double_talk", "WB-wired", None),
        ("background_noise", "WB-wired", None),
        ("background_noise", "WB-wired", "sce1"),
    ),
)
def test_controlled_scope_batches_keep_direction_and_linearity_gate(
    isolated_store,
    monkeypatch,
    test_scope,
    mode,
    scene,
):
    imported = import_dataset(_complete_archive())
    queued = acceptance_batch.start_batch(
        imported["dataset_id"],
        "vcp_default",
        test_scope=test_scope,
        mode=mode,
        scene=scene,
    )
    finish_calls = []
    monkeypatch.setattr(
        acceptance_linearity,
        "finish_linearity",
        lambda snapshot, dataset, destination: finish_calls.append(
            (snapshot, dataset, destination)
        ),
    )
    directions = []

    def controlled_run_cli(case, paths, destination, cluster):
        assert cluster == "vcp_default"
        direction = case["conditions"]["direction"]
        directions.append(direction)
        output_kind = "rx" if direction == "receive" else "tx"
        output = destination / f"{output_kind}.controlled.wav"
        sf.write(
            output,
            np.zeros(160, dtype=np.float64),
            16_000,
            subtype="PCM_16",
        )
        execution = {
            "source": "controlled-scope-test",
            "cli_sha256": queued["cli_sha256"],
            "catalog_sha256": queued["catalog_sha256"],
            "input_sha256": {
                role: acceptance_batch._sha(path)
                for role, path in paths.items()
            },
            "output_sha256": {
                output_kind: acceptance_batch._sha(output),
            },
        }
        return {output_kind: output}, execution

    def controlled_calculate(
        case,
        input_paths,
        outputs,
        execution,
        *,
        report_id,
    ):
        assert set(outputs) == {
            "rx" if case["conditions"]["direction"] == "receive" else "tx"
        }
        return [
            {
                "item_id": item_id,
                "value": 1.0,
                "status": "review_required",
                "reason": f"controlled {report_id}",
            }
            for item_id in case["item_ids"]
        ]

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    expected_direction = (
        "receive" if test_scope == "basic_downlink" else "send"
    )
    assert result["status"] == "completed", result.get("error")
    expected_count = 5 if scene is not None else _expected_count(
        test_scope,
        mode,
    )
    assert result["progress"]["completed_items"] == expected_count
    assert result["mode_reports"] == []
    assert directions
    assert set(directions) == {expected_direction}
    assert len(finish_calls) == (
        1 if test_scope == "basic_downlink" else 0
    )
    if test_scope == "basic_downlink":
        assert finish_calls[0][0]["batch_id"] == queued["batch_id"]
        assert finish_calls[0][2].name == queued["batch_id"]


def test_full_mode_report_gate_is_unchanged_and_subscope_cannot_pollute(
    isolated_store,
):
    def snapshot(batch_id: str, test_scope: str, count: int) -> dict:
        return {
            "batch_id": batch_id,
            "test_scope": test_scope,
            "items": [
                {
                    "item_id": f"item-{index}",
                    "mode": "WB-wired",
                    "status": "review_required",
                    "value": 1.0,
                    "evidence_binding": {"case_id": "case-1"},
                }
                for index in range(count)
            ],
            "cases": [{"case_id": "case-1", "status": "completed"}],
        }

    full = snapshot("a" * 32, "full", 42)
    acceptance_batch._save(full)
    assert len(full["mode_reports"]) == 1
    assert full["mode_reports"][0]["summary"]["total"] == 42

    subscope = snapshot("b" * 32, "background_noise", 42)
    acceptance_batch._save(subscope)
    assert subscope["mode_reports"] == []


def test_cli_command_direction_uses_original_case_conditions(
    tmp_path,
    monkeypatch,
):
    from server import alignment_runner

    cli = tmp_path / "fake-vcp"
    cli.write_bytes(b"fake executable identity")
    cli.chmod(0o755)
    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster: (
            {"profile": "vcp"},
            [],
            {"profile": "vcp"},
            "c" * 64,
        ),
    )

    def validate_report(path, output, profile):
        assert profile == "vcp"
        return json.loads(path.read_text()), []

    monkeypatch.setattr(
        alignment_runner,
        "validate_pipeline_artifacts",
        validate_report,
    )
    frames = 160
    rate = 16_000
    paths = {}
    for role in SIGNALS:
        path = tmp_path / f"{role}.wav"
        sf.write(
            path,
            np.zeros(frames, dtype=np.float64),
            rate,
            subtype="PCM_16",
        )
        paths[role] = path
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        output_prefix = Path(command[command.index("--out") + 1])
        if "--rx-processing" in command:
            source = Path(command[command.index("--rx-in") + 1])
            samples, source_rate = sf.read(source, always_2d=True)
            sf.write(
                Path(f"{output_prefix}.rx-spk.wav"),
                samples,
                source_rate,
                subtype="PCM_16",
            )
            report = {
                "processing_profile": "vcp",
                "rx_processing": True,
                "rx_processing_frames": 1,
                "dsp_delay_samples": 0,
            }
        else:
            source = Path(command[command.index("--mic") + 1])
            samples, source_rate = sf.read(source, always_2d=True)
            sf.write(
                Path(f"{output_prefix}.after.wav"),
                samples,
                source_rate,
                subtype="PCM_16",
            )
            report = {"dsp_delay_samples": 0}
        Path(f"{output_prefix}.json").write_text(json.dumps(report))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(acceptance_batch.subprocess, "run", fake_run)
    receive_destination = tmp_path / "receive"
    receive_destination.mkdir()
    receive_outputs, receive_execution = acceptance_batch._run_cli(
        {"conditions": {"mode": "WB-wired", "direction": "receive"}},
        paths,
        receive_destination,
        "vcp_default",
    )
    assert set(receive_outputs) == {"rx"}
    assert len(receive_execution["commands"]) == 1
    assert "--rx-processing" in receive_execution["commands"][0]

    send_destination = tmp_path / "send"
    send_destination.mkdir()
    send_outputs, send_execution = acceptance_batch._run_cli(
        {"conditions": {"mode": "WB-wired", "direction": "send"}},
        paths,
        send_destination,
        "vcp_default",
    )
    assert set(send_outputs) == {"tx"}
    assert len(send_execution["commands"]) == 1
    assert "--mic" in send_execution["commands"][0]
    assert "--rx-processing" not in send_execution["commands"][0]
    assert len(commands) == 2
