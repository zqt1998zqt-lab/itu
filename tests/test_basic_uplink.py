from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy import signal
import soundfile as sf
from fastapi.testclient import TestClient

from server.acoustic import acceptance_batch, acceptance_models, service
from server.acoustic.acceptance_dataset import (
    get_dataset,
    import_dataset,
    template_manifest,
)
from server.acoustic.acceptance_scope import (
    BASIC_UPLINK_FAMILIES,
    basic_uplink_items,
    select_batch_scope,
)
from server.alignment_runner import resolve_cli
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
SIGNALS = ("microphone", "reference", "far_end", "noise")


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


def _wav(values: np.ndarray, rate: int) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _zero_signals(rate: int) -> dict[str, bytes]:
    frames = rate * 3
    return {
        "microphone": _wav(np.zeros(frames), rate),
        "reference": _wav(np.zeros(frames), rate),
        "far_end": _wav(np.zeros(frames), rate),
        "noise": _wav(np.zeros(frames), rate),
    }


def _real_signals(rate: int) -> dict[str, bytes]:
    """Build deterministic three-second, speech-gated excitation for real CLI."""
    frames = rate * 3
    rng = np.random.default_rng(20261104 + rate)
    sos = signal.butter(
        4,
        (100.0, min(rate * 0.45, 0.9 * rate / 2)),
        btype="bandpass",
        fs=rate,
        output="sos",
    )

    reference = signal.sosfiltfilt(sos, rng.standard_normal(frames))
    reference /= max(float(np.max(np.abs(reference))), 1e-12)
    block = max(1, round(rate * 0.25))
    gate = (
        ((np.arange(frames) // block) % 4 != 3)
        .astype(np.float64)
    )
    smooth = max(1, round(rate * 0.01))
    gate = signal.convolve(
        gate,
        np.ones(smooth, dtype=np.float64) / smooth,
        mode="same",
    )
    reference *= 0.05 * gate

    noise = signal.sosfiltfilt(sos, rng.standard_normal(frames))
    noise /= max(float(np.max(np.abs(noise))), 1e-12)
    noise *= 0.0005

    microphone = np.clip(reference + 0.2 * noise, -0.9, 0.9)
    far_end = np.zeros(frames)
    return {
        "microphone": _wav(microphone, rate),
        "reference": _wav(reference, rate),
        "far_end": _wav(far_end, rate),
        "noise": _wav(noise, rate),
    }


def _complete_archive(rate: int = 16_000, *, real: bool = False) -> bytes:
    manifest = template_manifest()
    payloads = _real_signals(rate) if real else _zero_signals(rate)
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
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
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
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    return executor


def _controlled_outputs(
    case: dict,
    destination: Path,
    *,
    cli_sha256: str,
    catalog_sha256: str,
    paths: dict[str, Path],
) -> tuple[dict[str, Path], dict]:
    outputs = {"tx": destination / "tx.controlled.wav"}
    sf.write(outputs["tx"], np.zeros(160), 16_000, subtype="PCM_16")
    execution = {
        "source": "controlled-executor",
        "cli_sha256": cli_sha256,
        "catalog_sha256": catalog_sha256,
        "input_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in paths.items()
        },
        "output_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in outputs.items()
        },
    }
    return outputs, execution


def _controlled_rows(
    case: dict,
    execution: dict,
    *,
    report_id: str,
) -> list[dict]:
    return [
        {
            "item_id": item_id,
            "value": 1.0,
            "status": "review_required",
            "reason": "controlled engineering result",
            "standard_compliance": False,
            "evidence_binding": {
                **case["conditions"],
                "case_id": case["id"],
                "report_id": report_id,
                "batch_id": execution["batch_id"],
                "dataset_id": execution["dataset_id"],
                "input_sha256": execution["input_sha256"],
                "output_sha256": execution["output_sha256"],
            },
        }
        for item_id in case["item_ids"]
    ]


def _cli_inputs(
    tmp_path: Path,
    rate: int,
    *,
    channels: int = 2,
) -> dict[str, Path]:
    frames = rate // 2
    time = np.arange(frames, dtype=np.float64) / rate
    source = 0.03 * np.sin(2 * np.pi * 440 * time)
    microphone = (
        source
        if channels == 1
        else np.column_stack([source, 0.8 * source])
    )
    paths = {}
    for role, samples in (
        ("microphone", microphone),
        ("reference", source),
        ("far_end", source),
        ("noise", source),
    ):
        path = tmp_path / f"{role}.wav"
        sf.write(path, samples, rate, subtype="PCM_16")
        paths[role] = path
    return paths


def _command_beam_delays(command: list[str]) -> list[float]:
    index = command.index("--beam-delays")
    return [float(value) for value in command[index + 1].split(",")]


@pytest.mark.parametrize("mode", MODES)
def test_basic_uplink_catalog_is_exactly_eight_send_rows(mode):
    rows = basic_uplink_items(mode)

    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8
    assert {row["sheet"] for row in rows} == {mode}
    assert {row["direction"] for row in rows} == {"send"}
    assert {row["family"] for row in rows} <= BASIC_UPLINK_FAMILIES


def test_basic_uplink_catalog_rejects_invalid_mode():
    for mode in (None, "", "WB-invalid", "NB", "WB-wired-extra"):
        with pytest.raises(ValueError, match="业务模式"):
            basic_uplink_items(mode)


@pytest.mark.parametrize("mode", MODES)
def test_select_batch_scope_filters_without_mutating_manifest(mode):
    manifest = template_manifest()
    original = copy.deepcopy(manifest)
    rows, cases = select_batch_scope(
        manifest,
        test_scope="basic_uplink",
        mode=mode,
    )

    selected_ids = {row["id"] for row in rows}
    flattened = [
        item_id
        for case in cases
        for item_id in case["item_ids"]
    ]
    expected_case_ids = {
        case["id"]
        for case in original["cases"]
        if any(item_id in selected_ids for item_id in case["item_ids"])
    }
    assert len(rows) == 8
    assert len(cases) == len(expected_case_ids)
    assert {case["id"] for case in cases} == expected_case_ids
    assert len(flattened) == len(set(flattened)) == 8
    assert set(flattened) == selected_ids
    assert manifest == original

    cases_by_id = {case["id"]: case for case in original["cases"]}
    for case in cases:
        source = cases_by_id[case["id"]]
        assert set(case["item_ids"]) <= selected_ids
        assert {
            key: value
            for key, value in case.items()
            if key != "item_ids"
        } == {
            key: value
            for key, value in source.items()
            if key != "item_ids"
        }
    assert any(
        any(item_id not in selected_ids for item_id in source["item_ids"])
        for source in original["cases"]
        if source["id"] in expected_case_ids
    )

    full_rows, full_cases = select_batch_scope(manifest)
    assert len(full_rows) == 336
    assert len(full_cases) == 106
    assert sum(len(case["item_ids"]) for case in full_cases) == 336
    assert full_cases == original["cases"]
    assert manifest == original


def test_select_batch_scope_rejects_invalid_scope_mode_and_coverage():
    manifest = template_manifest()
    original = copy.deepcopy(manifest)

    with pytest.raises(ValueError, match="未知测试范围"):
        select_batch_scope(manifest, test_scope="partial")
    with pytest.raises(ValueError, match="不能指定单一模式"):
        select_batch_scope(manifest, mode="WB-wired")
    with pytest.raises(ValueError, match="业务模式"):
        select_batch_scope(manifest, test_scope="basic_uplink")
    with pytest.raises(ValueError, match="业务模式"):
        select_batch_scope(
            manifest,
            test_scope="basic_uplink",
            mode="not-a-mode",
        )

    rows, _ = select_batch_scope(
        manifest,
        test_scope="basic_uplink",
        mode="WB-wired",
    )
    missing_id = rows[0]["id"]
    for case in manifest["cases"]:
        case["item_ids"] = [
            item_id
            for item_id in case["item_ids"]
            if item_id != missing_id
        ]
    with pytest.raises(ValueError, match="未唯一覆盖"):
        select_batch_scope(
            manifest,
            test_scope="basic_uplink",
            mode="WB-wired",
        )
    assert manifest != original

    duplicate = copy.deepcopy(original)
    duplicate_id = basic_uplink_items("WB-wired")[0]["id"]
    duplicate_case = next(
        case
        for case in duplicate["cases"]
        if duplicate_id in case["item_ids"]
    )
    duplicate_case["item_ids"].append(duplicate_id)
    with pytest.raises(ValueError, match="未唯一覆盖"):
        select_batch_scope(
            duplicate,
            test_scope="basic_uplink",
            mode="WB-wired",
        )

    mismatched = copy.deepcopy(original)
    selected_case = next(
        case
        for case in mismatched["cases"]
        if "WB-wired:B12" in case["item_ids"]
    )
    selected_case["conditions"]["window_ms"] = 100
    with pytest.raises(ValueError, match="采集工况不一致"):
        select_batch_scope(
            mismatched,
            test_scope="basic_uplink",
            mode="WB-wired",
        )


def test_basic_batch_runs_only_two_cases_and_eight_items(
    isolated_store,
    monkeypatch,
):
    imported = import_dataset(_complete_archive())
    before = copy.deepcopy(get_dataset(imported["dataset_id"])["manifest"])
    queued = acceptance_batch.start_batch(
        imported["dataset_id"],
        "vcp_default",
        test_scope="basic_uplink",
        mode="WB-wired",
    )

    assert queued["test_scope"] == "basic_uplink"
    assert queued["selected_mode"] == "WB-wired"
    assert queued["progress"] == {
        "completed_cases": 0,
        "total_cases": 2,
        "completed_items": 0,
        "total_items": 8,
    }
    assert len(queued["items"]) == 8
    assert {row["mode"] for row in queued["items"]} == {"WB-wired"}
    assert queued["batch_id"] in acceptance_batch._ACTIVE
    assert len(isolated_store.submissions) == 1
    assert acceptance_batch.list_batch_summaries() == []
    basic_history = acceptance_batch.list_batch_summaries(
        test_scope="basic_uplink"
    )
    assert [row["batch_id"] for row in basic_history] == [queued["batch_id"]]
    assert basic_history[0]["selected_mode"] == "WB-wired"
    assert get_dataset(imported["dataset_id"])["manifest"] == before

    _, selected_cases = select_batch_scope(
        before,
        test_scope="basic_uplink",
        mode="WB-wired",
    )
    expected_items = {
        case["id"]: list(case["item_ids"])
        for case in selected_cases
    }
    run_calls = []
    calculate_calls = []

    def controlled_run_cli(case, paths, destination, cluster):
        assert cluster == "vcp_default"
        assert set(paths) == set(SIGNALS)
        outputs, execution = _controlled_outputs(
            case,
            destination,
            cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"],
            paths=paths,
        )
        run_calls.append((case["id"], set(paths)))
        return outputs, execution

    def controlled_calculate(
        case,
        input_paths,
        outputs,
        execution,
        *,
        report_id,
    ):
        calculate_calls.append(
            {
                "case_id": case["id"],
                "item_ids": list(case["item_ids"]),
                "input_paths": dict(input_paths),
                "outputs": dict(outputs),
                "report_id": report_id,
            }
        )
        return _controlled_rows(
            case,
            execution,
            report_id=report_id,
        )

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert result["status"] == "completed"
    assert result["progress"] == {
        "completed_cases": 2,
        "total_cases": 2,
        "completed_items": 8,
        "total_items": 8,
    }
    assert len(result["items"]) == 8
    assert {row["item_id"] for row in result["items"]} == {
        row["id"] for row in basic_uplink_items("WB-wired")
    }
    assert len(result["cases"]) == 2
    assert result["mode_reports"] == []
    assert len(run_calls) == len(calculate_calls) == 2
    assert {
        call["case_id"]: call["item_ids"]
        for call in calculate_calls
    } == expected_items
    assert all(call["report_id"] for call in calculate_calls)
    assert queued["batch_id"] not in acceptance_batch._ACTIVE

    assert acceptance_batch.list_batch_summaries() == []
    basic_history = acceptance_batch.list_batch_summaries(
        test_scope="basic_uplink"
    )
    assert basic_history[0]["status"] == "completed"
    assert basic_history[0]["summary"]["total"] == 8
    assert basic_history[0]["selected_mode"] == "WB-wired"


def test_basic_api_catalog_post_report_and_history_are_isolated(isolated_store):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)

    catalog = client.get(
        "/api/alignment/acceptance/basic-uplink/catalog",
        params={"mode": "WB-wired"},
    )
    assert catalog.status_code == 200
    catalog_body = catalog.json()
    assert catalog_body["test_scope"] == "basic_uplink"
    assert catalog_body["mode"] == "WB-wired"
    assert catalog_body["item_count"] == 8
    assert len(catalog_body["items"]) == 8
    assert {
        row["id"] for row in catalog_body["items"]
    } == {
        row["id"] for row in basic_uplink_items("WB-wired")
    }

    invalid_catalog = client.get(
        "/api/alignment/acceptance/basic-uplink/catalog",
        params={"mode": "WB-invalid"},
    )
    assert invalid_catalog.status_code == 422

    request = {
        "dataset_id": imported["dataset_id"],
        "algorithm_cluster_id": "vcp_default",
        "mode": "WB-wired",
    }
    created = client.post(
        "/api/alignment/acceptance/basic-uplink/batches",
        json=request,
    )
    assert created.status_code == 202, created.text
    queued = created.json()
    assert queued["test_scope"] == "basic_uplink"
    assert queued["selected_mode"] == "WB-wired"
    assert queued["progress"]["total_cases"] == 2
    assert queued["progress"]["total_items"] == 8
    assert len(isolated_store.submissions) == 1

    history = client.get(
        "/api/alignment/acceptance/basic-uplink/batches"
    )
    assert history.status_code == 200
    assert [row["batch_id"] for row in history.json()] == [queued["batch_id"]]
    assert history.json()[0]["test_scope"] == "basic_uplink"
    assert history.json()[0]["selected_mode"] == "WB-wired"

    full_history = client.get("/api/alignment/acceptance/batches")
    assert full_history.status_code == 200
    assert full_history.json() == []

    report = client.get(
        f"/api/alignment/acceptance/batches/{queued['batch_id']}/report"
    )
    assert report.status_code == 200
    assert report.json()["test_scope"] == "basic_uplink"
    assert report.json()["selected_mode"] == "WB-wired"
    assert len(report.json()["items"]) == 8
    assert report.headers["content-disposition"] == (
        f'attachment; filename="acceptance-{queued["batch_id"]}.json"'
    )

    assert client.post(
        "/api/alignment/acceptance/basic-uplink/batches",
        json={
            **request,
            "test_scope": "basic_uplink",
        },
    ).status_code == 422
    missing_mode = dict(request)
    missing_mode.pop("mode")
    assert client.post(
        "/api/alignment/acceptance/basic-uplink/batches",
        json=missing_mode,
    ).status_code == 422
    invalid_mode = client.post(
        "/api/alignment/acceptance/basic-uplink/batches",
        json={**request, "mode": "WB-invalid"},
    )
    assert invalid_mode.status_code == 422


def test_basic_and_full_share_active_and_legacy_gates(
    isolated_store,
    monkeypatch,
):
    imported = import_dataset(_complete_archive())
    queued = acceptance_batch.start_batch(
        imported["dataset_id"],
        "vcp_default",
        test_scope="basic_uplink",
        mode="WB-wired",
    )

    with pytest.raises(ValueError, match="已有|正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
        )
    with pytest.raises(ValueError, match="已有|正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
            test_scope="basic_uplink",
            mode="NB-wired",
        )
    assert queued["batch_id"] in acceptance_batch._ACTIVE
    assert len(isolated_store.submissions) == 1

    acceptance_batch._ACTIVE.clear()
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", True)
    with pytest.raises(ValueError, match="正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
        )
    with pytest.raises(ValueError, match="正在执行"):
        acceptance_batch.start_batch(
            imported["dataset_id"],
            "vcp_default",
            test_scope="basic_uplink",
            mode="WB-wired",
        )
    assert len(isolated_store.submissions) == 1


def test_array_control_is_rejected_for_basic_but_allowed_for_full(
    isolated_store,
):
    imported = import_dataset(_complete_archive())
    client = TestClient(app)
    request = {
        "dataset_id": imported["dataset_id"],
        "algorithm_cluster_id": "array_control",
    }

    rejected = client.post(
        "/api/alignment/acceptance/basic-uplink/batches",
        json={**request, "mode": "WB-wired"},
    )
    assert rejected.status_code == 422
    assert "VCP" in rejected.json()["detail"]
    assert isolated_store.submissions == []
    assert not acceptance_batch._ACTIVE
    batches = service.BUILD / "acceptance-batches"
    assert list(batches.glob("*/report.json")) == []

    full = client.post(
        "/api/alignment/acceptance/batches",
        json=request,
    )
    assert full.status_code == 202, full.text
    queued = full.json()
    assert queued["algorithm_cluster_id"] == "array_control"
    assert queued["test_scope"] == "full"
    assert queued["progress"]["total_cases"] == 106
    assert queued["progress"]["total_items"] == 336
    assert len(isolated_store.submissions) == 1
    assert len(list(batches.glob("*/report.json"))) == 1


@pytest.mark.parametrize(
    ("rate", "mode", "engine_rate", "expected_samples"),
    (
        (8_000, "NB-wired", 16_000, [2.0, 8.0]),
        (48_000, "SWB-wired", 48_000, [6.0, 24.0]),
    ),
)
def test_beam_delays_are_converted_at_engine_sample_rate(
    tmp_path,
    monkeypatch,
    rate,
    mode,
    engine_rate,
    expected_samples,
):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")

    paths = _cli_inputs(tmp_path, rate, channels=2)
    destination = tmp_path / "case"
    destination.mkdir()
    case = {
        "conditions": {
            "mode": mode,
            "direction": "send",
        },
        "beam_delays_ms": [0.125, 0.5],
    }
    captured_commands = []
    real_run = acceptance_batch.subprocess.run

    def recording_run(command, *args, **kwargs):
        captured_commands.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(
        acceptance_batch.subprocess,
        "run",
        recording_run,
    )
    outputs, execution = acceptance_batch._run_cli(
        case,
        paths,
        destination,
        "vcp_default",
    )

    assert len(captured_commands) == 1
    assert _command_beam_delays(captured_commands[0]) == pytest.approx(
        expected_samples
    )
    assert _command_beam_delays(execution["commands"][0]) == pytest.approx(
        expected_samples
    )
    adaptation = execution["beam_delay_adaptation"]
    assert adaptation["source_unit"] == "ms"
    assert adaptation["engine_rate"] == engine_rate
    assert adaptation["delays_ms"] == [0.125, 0.5]
    assert adaptation["delays_samples"] == pytest.approx(expected_samples)
    assert execution["pipeline_reports"]["tx"]["data"][
        "dsp_delay_samples"
    ] >= 0
    assert execution["output_sha256"]["tx"] == hashlib.sha256(
        outputs["tx"].read_bytes()
    ).hexdigest()
    if rate == 8_000:
        assert execution["input_adaptation"]["microphone"][
            "engine_rate"
        ] == engine_rate


@pytest.mark.parametrize(
    "delays",
    (
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [True, 0.0],
        [-1.0, 0.0],
        [0.0],
        "0.0,0.0",
    ),
)
def test_multi_mic_invalid_beam_delays_are_rejected_before_cli(
    tmp_path,
    monkeypatch,
    delays,
):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")

    paths = _cli_inputs(tmp_path, 16_000, channels=2)
    destination = tmp_path / "case"
    destination.mkdir()
    calls = []

    def must_not_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid beam delays must be rejected before CLI")

    monkeypatch.setattr(acceptance_batch.subprocess, "run", must_not_run)
    with pytest.raises(ValueError, match="beam_delays_ms"):
        acceptance_batch._run_cli(
            {
                "conditions": {
                    "mode": "WB-wired",
                    "direction": "send",
                },
                "beam_delays_ms": delays,
            },
            paths,
            destination,
            "vcp_default",
        )
    assert calls == []


@pytest.mark.parametrize(
    ("mode", "rate"),
    (("NB-wired", 8_000), ("WB-wired", 16_000)),
)
def test_real_vcp_and_real_calculation_cover_eight_items(
    isolated_store,
    tmp_path,
    monkeypatch,
    mode,
    rate,
):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")

    def unavailable_external_dnsmos(samples, sample_rate, *, label):
        return {
            "available": False,
            "reason": (
                "targeted basic-uplink regression intentionally excludes "
                "the external DNSMOS model"
            ),
        }

    monkeypatch.setattr(
        acceptance_models,
        "_score_dnsmos",
        unavailable_external_dnsmos,
    )
    imported = import_dataset(_complete_archive(rate, real=True))
    dataset = get_dataset(imported["dataset_id"])
    assert dataset is not None
    rows, cases = select_batch_scope(
        dataset["manifest"],
        test_scope="basic_uplink",
        mode=mode,
    )
    assert len(rows) == 8
    assert len(cases) == 2

    calculated_rows = []
    for index, case in enumerate(cases, start=1):
        destination = tmp_path / "real-vcp" / case["id"]
        destination.mkdir(parents=True)
        paths = {
            role: acceptance_batch.dataset_asset(
                imported["dataset_id"],
                relative,
            )
            for role, relative in case["signals"].items()
        }
        outputs, execution = acceptance_batch._run_cli(
            case,
            paths,
            destination,
            "vcp_default",
        )
        assert set(outputs) == {"tx"}
        assert outputs["tx"].is_file()
        assert len(execution["cli_sha256"]) == 64
        assert len(execution["catalog_sha256"]) == 64
        assert execution["output_sha256"]["tx"] == hashlib.sha256(
            outputs["tx"].read_bytes()
        ).hexdigest()
        assert set(execution["input_sha256"]) == set(SIGNALS)
        assert execution["beam_delay_adaptation"] is None

        execution.update(
            batch_id="b" * 32,
            dataset_id=imported["dataset_id"],
        )
        case_rows = acceptance_models.calculate_case_items(
            case,
            paths,
            outputs,
            execution,
            report_id=f"{index:032x}",
        )
        assert [row["item_id"] for row in case_rows] == case["item_ids"]
        assert all(row["standard_compliance"] is False for row in case_rows)
        assert all(row["status"] != "pass" for row in case_rows)
        assert all(
            row["evidence_binding"]["input_sha256"]
            == execution["input_sha256"]
            for row in case_rows
        )
        assert all(
            row["evidence_binding"]["output_sha256"]["tx"]
            == execution["output_sha256"]["tx"]
            for row in case_rows
        )
        calculated_rows.extend(case_rows)

    assert len(calculated_rows) == 8
    assert {
        row["item_id"] for row in calculated_rows
    } == {
        row["id"] for row in basic_uplink_items(mode)
    }
    assert any(
        isinstance(row.get("value"), (int, float))
        and not isinstance(row.get("value"), bool)
        for row in calculated_rows
    )
