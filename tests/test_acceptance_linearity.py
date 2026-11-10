from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import acceptance_linearity, receive_simulation
from server.acoustic.acceptance_contract import contract_items
from server.acoustic.acceptance_dataset import template_manifest


RATE = 16_000
DURATION_S = 0.5
ANALYSIS_START_S = 0.1
LINEARITY_MODES = (
    "NB-wired",
    "NB-wireless",
    "NB-bt",
    "WB-wired",
    "WB-wireless",
    "WB-bt",
)
LEVELS = (0, -5, 5)
LEVEL_IDS = {
    0: "nominal",
    -5: "offset_-5",
    5: "offset_+5",
}


def _speaker_wav(input_level_db: int) -> bytes:
    samples = np.zeros(round(DURATION_S * RATE), dtype=np.float64)
    start = round(ANALYSIS_START_S * RATE)
    time = np.arange(len(samples) - start, dtype=np.float64) / RATE
    amplitude = 0.1 * 10 ** (input_level_db / 20)
    samples[start:] = amplitude * np.sin(2 * np.pi * 440 * time)
    stream = BytesIO()
    sf.write(stream, samples, RATE, format="WAV", subtype="PCM_24")
    return stream.getvalue()


def _build_batch_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    modes: tuple[str, ...] = LINEARITY_MODES,
    v2_decoys: bool = False,
) -> dict:
    store = tmp_path / "synthetic-receive"
    monkeypatch.setattr(receive_simulation, "STORE", store)

    manifest = template_manifest()
    selected_cases: list[dict] = []
    reports_by_case: dict[str, dict] = {}
    input_sha256: dict[str, str] = {}
    case_records: list[dict] = []
    linearity_case_ids_by_mode: dict[str, dict[int, str]] = {}
    destination = tmp_path / "batch"
    contract_by_id = {
        row["id"]: row for row in contract_items()
    }

    for mode in modes:
        for level in LEVELS:
            matches = [
                case
                for case in manifest["cases"]
                if case["conditions"]["mode"] == mode
                and case["conditions"]["direction"] == "receive"
                and case["conditions"]["scene"] == "baseline"
                and case["conditions"]["volume"] == "nominal"
                and case["conditions"]["window_ms"] is None
                and case["conditions"]["input_level_db"] == level
            ]
            assert len(matches) == 1
            case = copy.deepcopy(matches[0])
            case["analysis_start_s"] = ANALYSIS_START_S
            decoy_cases: list[dict] = []
            if v2_decoys and level == 0:
                nominal_items = [
                    contract_by_id[item_id]
                    for item_id in case["item_ids"]
                ]
                rlr_item_ids = [
                    row["id"]
                    for row in nominal_items
                    if row["family"] == "rlr"
                ]
                assert len(rlr_item_ids) == 1
                case["item_ids"] = rlr_item_ids
                for row in nominal_items:
                    family = row["family"]
                    if family == "rlr":
                        continue
                    decoy = copy.deepcopy(case)
                    decoy["id"] = f"v2-decoy-{mode}-{family}"
                    decoy["item_ids"] = [row["id"]]
                    decoy_cases.append(decoy)
            selected_cases.append(case)
            selected_cases.extend(decoy_cases)
            linearity_case_ids_by_mode.setdefault(mode, {})[level] = case["id"]

            speaker = _speaker_wav(level)
            input_sha256[case["id"]] = hashlib.sha256(speaker).hexdigest()
            for decoy in decoy_cases:
                input_sha256[decoy["id"]] = input_sha256[case["id"]]
            report = receive_simulation.evaluate_synthetic_receive(
                mode,
                f"{case['id']}.wav",
                speaker,
            )
            reports_by_case[case["id"]] = report
            assessment_report_id = hashlib.sha256(
                f"{case['id']}:assessment-report".encode()
            ).hexdigest()[:32]
            assert assessment_report_id != report["evaluation_id"]

            execution = {
                "source": "controlled-acceptance-fixture",
                "batch_id": "batch-linearity-test",
                "dataset_id": "dataset-linearity-test",
                "cli_sha256": "c" * 64,
                "catalog_sha256": "d" * 64,
                "input_sha256": {
                    "far_end": input_sha256[case["id"]],
                },
                "output_sha256": {
                    "rx": hashlib.sha256(
                        f"{case['id']}:rx".encode()
                    ).hexdigest(),
                },
            }
            case_directory = destination / "cases" / case["id"]
            case_directory.mkdir(parents=True)
            assessment_path = case_directory / "assessment.json"
            assessment_path.write_text(
                json.dumps({
                    "report_id": assessment_report_id,
                    "case_id": case["id"],
                    "execution": execution,
                    "items": [
                        {
                            "item_id": case["item_ids"][0],
                            "model_details": {
                                "receive_report_id": report["evaluation_id"],
                            },
                        }
                    ],
                }),
                encoding="utf-8",
            )
            case_records.append({
                "case_id": case["id"],
                "report_id": assessment_report_id,
                "conditions": copy.deepcopy(case["conditions"]),
                "status": "completed",
                "assessment_sha256": hashlib.sha256(
                    assessment_path.read_bytes()
                ).hexdigest(),
                "execution": execution,
            })

    case_by_item = {
        item_id: case
        for case in selected_cases
        for item_id in case["item_ids"]
    }
    snapshot_items = []
    for row in contract_items():
        case = case_by_item.get(row["id"])
        report = reports_by_case.get(case["id"]) if case else None
        model_details = (
            {"receive_report_id": report["evaluation_id"]}
            if report is not None
            else {}
        )
        snapshot_items.append(
            {
                **row,
                "item_id": row["id"],
                "mode": row["sheet"],
                "value": 999.0,
                "status": "review_required",
                "evidence_grade": "engineering_proxy",
                "model_details": model_details,
                "evidence_binding": {
                    "case_id": case["id"] if case else None,
                    "input_sha256": (
                        {"far_end": input_sha256[case["id"]]}
                        if case
                        else {}
                    ),
                },
            }
        )

    return {
        "snapshot": {
            "items": snapshot_items,
            "cases": case_records,
        },
        "dataset": {"manifest": {"cases": selected_cases}},
        "destination": destination,
        "reports_by_case": reports_by_case,
        "selected_cases": selected_cases,
        "input_sha256": input_sha256,
        "case_records_by_case": {
            record["case_id"]: record for record in case_records
        },
        "linearity_case_ids_by_mode": linearity_case_ids_by_mode,
    }


def _linearity_rows(mode: str | None = None) -> list[dict]:
    return [
        row
        for row in contract_items()
        if row["family"] == "rlr_linearity"
        and (mode is None or row["sheet"] == mode)
    ]


def test_finish_linearity_uses_real_cropped_receive_outputs_for_all_nb_wb_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(tmp_path, monkeypatch)
    captured: dict[str, dict] = {}
    original = acceptance_linearity.evaluate_rlr_linearity_group

    def capture(runs, conditions=None, *, include_canonical_hash=False):
        captured[conditions["mode"]] = {
            "runs": copy.deepcopy(runs),
            "conditions": copy.deepcopy(conditions),
        }
        return original(
            runs,
            conditions,
            include_canonical_hash=include_canonical_hash,
        )

    monkeypatch.setattr(
        acceptance_linearity,
        "evaluate_rlr_linearity_group",
        capture,
    )
    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    assert set(captured) == set(LINEARITY_MODES)
    snapshot_by_id = {
        item["item_id"]: item for item in fixture["snapshot"]["items"]
    }
    expected_start_sample = round(ANALYSIS_START_S * RATE)
    expected_end_sample = round(DURATION_S * RATE)
    expected_window = [
        expected_start_sample / RATE,
        expected_end_sample / RATE,
    ]

    updated_items = 0
    for mode, captured_mode in captured.items():
        runs = captured_mode["runs"]
        conditions = captured_mode["conditions"]
        assert [run["level_id"] for run in runs] == [
            LEVEL_IDS[level] for level in LEVELS
        ]
        assert conditions["mode"] == mode
        assert conditions["nominal_volume"] == "nominal"
        assert conditions["fixture"]["fixture_id"] == "software-only-virtual-drp"
        assert conditions["fixture"]["fixture_type"] == "SIMULATED_CABIN"
        assert conditions["fixture"]["calibration"] == "none"
        assert conditions["active_speech_window_s"] == expected_window

        for run in runs:
            case_id = run["metadata"]["case_id"]
            report = fixture["reports_by_case"][case_id]
            case_record = fixture["case_records_by_case"][case_id]
            execution = case_record["execution"]
            execution_sha256 = acceptance_linearity.canonical_sha256(execution)
            left_path = receive_simulation.get_synthetic_receive_file(
                report["evaluation_id"],
                "virtual_left.wav",
            )
            right_path = receive_simulation.get_synthetic_receive_file(
                report["evaluation_id"],
                "virtual_right.wav",
            )
            assert left_path is not None and right_path is not None
            left, rate = sf.read(left_path, dtype="float64")
            right, right_rate = sf.read(right_path, dtype="float64")
            assert rate == right_rate
            expected_bands = receive_simulation._third_octave_levels(
                left[expected_start_sample:expected_end_sample],
                right[expected_start_sample:expected_end_sample],
                rate,
            )

            assert run["conditions"]["nominal_volume"] == "nominal"
            assert run["conditions"]["fixture"] == conditions["fixture"]
            assert run["conditions"]["head_model"] == report["simulation"]["head_model"]
            assert run["conditions"]["sample_rate_hz"] == RATE
            assert run["conditions"]["active_speech_window_s"] == expected_window
            assert case_record["report_id"] != report["evaluation_id"]
            assert (
                run["metadata"]["assessment_report_id"]
                == case_record["report_id"]
            )
            assert run["metadata"]["evaluation_id"] == report["evaluation_id"]
            assert run["metadata"]["window_applied"] is True
            assert run["metadata"]["analysis_window_s"] == expected_window
            assert run["metadata"]["window_source"] == "cropped_virtual_ear_wav"
            assert run["metadata"]["analysis_sample_bounds"] == {
                "left": [expected_start_sample, expected_end_sample],
                "right": [expected_start_sample, expected_end_sample],
            }
            assert run["metadata"]["artifact_sha256"] == {
                "left": report["artifacts"]["left"]["sha256"],
                "right": report["artifacts"]["right"]["sha256"],
            }
            assert run["metadata"]["input_sha256"] == {
                "far_end": fixture["input_sha256"][case_id]
            }
            assert (
                run["metadata"]["assessment_sha256"]
                == case_record["assessment_sha256"]
            )
            assert run["metadata"]["execution_sha256"] == execution_sha256
            assert run["metadata"]["execution_binding"] == {
                "batch_id": execution["batch_id"],
                "dataset_id": execution["dataset_id"],
                "cli_sha256": execution["cli_sha256"],
                "catalog_sha256": execution["catalog_sha256"],
                "input_sha256": execution["input_sha256"],
                "output_sha256": execution["output_sha256"],
            }
            assert run["metadata"]["report_result_sha256"] == report["result_sha256"]
            assert run["metrics"]["third_octave_band_levels_db"] == expected_bands
            assert run["metrics"]["third_octave_band_levels_dbfs"] == expected_bands
            assert (
                run["metrics"]["third_octave_band_levels_db"]
                != report["metrics"]["third_octave_band_levels_db"]
            )
            assert run["metrics"]["window_applied"] is True
            assert run["metrics"]["analysis_window_s"] == expected_window

        result_path = fixture["destination"] / f"{mode}-linearity.json"
        assert result_path.is_file()
        result = json.loads(result_path.read_text())
        mode_rows = _linearity_rows(mode)
        expected_cells = {row["source_cell"] for row in mode_rows}
        assert set(result["items"]) == expected_cells
        assert result["group_valid"] is True
        assert result["window_applied"] is True
        assert result["window_source"] == "provided_cropped_band_levels"
        assert result["analysis_window_s"] == expected_window
        assert set(result["window_evidence"]) == {
            LEVEL_IDS[level] for level in LEVELS
        }

        for row in mode_rows:
            item = snapshot_by_id[row["id"]]
            residual = result["items"][row["source_cell"]]["residual_db"]
            assert math.isfinite(residual)
            assert abs(residual) < 0.01
            assert item["value"] == residual
            assert item["status"] == "review_required"
            assert item["model_details"]["linearity"]["group_valid"] is True
            assert len(item["model_details"]["source_runs"]) == 3
            assert len(item["evidence_binding"]["source_runs"]) == 3
            for source in item["model_details"]["source_runs"]:
                source_record = fixture["case_records_by_case"][
                    source["case_id"]
                ]
                source_report = fixture["reports_by_case"][source["case_id"]]
                assert source_record["report_id"] != source_report["evaluation_id"]
                assert source["assessment_report_id"] == source_record["report_id"]
                assert source["report_id"] == source_report["evaluation_id"]
                assert (
                    source["assessment_sha256"]
                    == source_record["assessment_sha256"]
                )
                assert source["execution_sha256"] == (
                    acceptance_linearity.canonical_sha256(
                        source_record["execution"]
                    )
                )
                assert source["execution_binding"]["batch_id"] == (
                    source_record["execution"]["batch_id"]
                )
                assert source["execution_binding"]["input_sha256"] == (
                    source_record["execution"]["input_sha256"]
                )
                assert source["execution_binding"]["output_sha256"] == (
                    source_record["execution"]["output_sha256"]
                )
            updated_items += 1

    assert len(_linearity_rows()) == 12
    assert updated_items == 12
    assert set(LINEARITY_MODES) | {"SWB-wired", "SWB-wireless"} == {
        row["sheet"] for row in contract_items()
    }
    assert _linearity_rows("SWB-wired") == []
    assert _linearity_rows("SWB-wireless") == []
    assert not (fixture["destination"] / "SWB-wired-linearity.json").exists()
    assert not (fixture["destination"] / "SWB-wireless-linearity.json").exists()


def test_finish_linearity_v2_selects_contract_bound_rlr_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(
        tmp_path,
        monkeypatch,
        modes=("WB-wireless",),
        v2_decoys=True,
    )
    captured: dict[str, list[dict]] = {}
    original = acceptance_linearity.evaluate_rlr_linearity_group

    def capture(runs, conditions=None, *, include_canonical_hash=False):
        captured["runs"] = copy.deepcopy(runs)
        return original(
            runs,
            conditions,
            include_canonical_hash=include_canonical_hash,
        )

    monkeypatch.setattr(
        acceptance_linearity,
        "evaluate_rlr_linearity_group",
        capture,
    )
    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    expected_case_ids = fixture["linearity_case_ids_by_mode"]["WB-wireless"]
    assert [
        run["metadata"]["case_id"] for run in captured["runs"]
    ] == [
        expected_case_ids[level] for level in LEVELS
    ]
    decoy_cases = [
        case
        for case in fixture["dataset"]["manifest"]["cases"]
        if case["id"].startswith("v2-decoy-")
    ]
    assert len(decoy_cases) == 3
    assert {
        next(
            row["family"]
            for row in contract_items()
            if row["id"] == case["item_ids"][0]
        )
        for case in decoy_cases
    } == {"frequency_response", "listening_quality", "out_of_band"}
    assert not {
        case["id"] for case in decoy_cases
    }.intersection(
        run["metadata"]["case_id"] for run in captured["runs"]
    )
    result_path = fixture["destination"] / "WB-wireless-linearity.json"
    assert result_path.is_file()
    assert json.loads(result_path.read_text())["status"] == "pass"


def test_finish_linearity_rejects_ambiguous_nominal_rlr_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(
        tmp_path,
        monkeypatch,
        modes=("WB-wireless",),
    )
    nominal_case_id = fixture["linearity_case_ids_by_mode"][
        "WB-wireless"
    ][0]
    nominal_case = next(
        case
        for case in fixture["dataset"]["manifest"]["cases"]
        if case["id"] == nominal_case_id
    )
    duplicate = copy.deepcopy(nominal_case)
    duplicate["id"] = "duplicate-nominal-rlr"
    fixture["dataset"]["manifest"]["cases"].append(duplicate)

    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    targets = [
        item
        for item in fixture["snapshot"]["items"]
        if item["mode"] == "WB-wireless"
        and item["family"] == "rlr_linearity"
    ]
    assert len(targets) == 2
    assert all(item["value"] is None for item in targets)
    assert all(item["status"] == "not_measured" for item in targets)
    assert all("唯一的 nominal 接收运行" in item["reason"] for item in targets)
    assert not (
        fixture["destination"] / "WB-wireless-linearity.json"
    ).exists()


def test_linearity_case_binding_requires_exact_family_mode_and_level() -> None:
    contract_by_id = {
        row["id"]: row for row in contract_items()
    }
    nominal = copy.deepcopy(
        next(
            case
            for case in template_manifest()["cases"]
            if case["conditions"]["mode"] == "WB-wired"
            and case["conditions"]["direction"] == "receive"
            and case["conditions"]["scene"] == "baseline"
            and case["conditions"]["volume"] == "nominal"
            and case["conditions"]["window_ms"] is None
            and case["conditions"]["input_level_db"] == 0
        )
    )

    assert acceptance_linearity._case_has_linearity_contract(
        nominal,
        contract_by_id,
        mode="WB-wired",
        level=0,
        family="rlr",
    )
    missing = {**nominal, "item_ids": []}
    assert not acceptance_linearity._case_has_linearity_contract(
        missing,
        contract_by_id,
        mode="WB-wired",
        level=0,
        family="rlr",
    )
    wrong_mode = {**nominal, "item_ids": ["NB-wired:B3"]}
    assert not acceptance_linearity._case_has_linearity_contract(
        wrong_mode,
        contract_by_id,
        mode="WB-wired",
        level=0,
        family="rlr",
    )
    wrong_level = {**nominal, "item_ids": ["WB-wired:B5"]}
    assert not acceptance_linearity._case_has_linearity_contract(
        wrong_level,
        contract_by_id,
        mode="WB-wired",
        level=0,
        family="rlr",
    )
    frequency_only = {**nominal, "item_ids": ["WB-wired:B7"]}
    assert not acceptance_linearity._case_has_linearity_contract(
        frequency_only,
        contract_by_id,
        mode="WB-wired",
        level=0,
        family="rlr",
    )


def test_finish_linearity_rejects_tampered_virtual_ear_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(
        tmp_path,
        monkeypatch,
        modes=("WB-wireless",),
    )
    nominal_case = next(
        case
        for case in fixture["selected_cases"]
        if case["conditions"]["input_level_db"] == 0
    )
    nominal_report = fixture["reports_by_case"][nominal_case["id"]]
    left_path = receive_simulation.get_synthetic_receive_file(
        nominal_report["evaluation_id"],
        "virtual_left.wav",
    )
    assert left_path is not None
    left, rate = sf.read(left_path, dtype="float64")
    sf.write(left_path, left * 0.5, rate, format="WAV", subtype="PCM_24")

    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    targets = [
        item
        for item in fixture["snapshot"]["items"]
        if item["mode"] == "WB-wireless"
        and item["family"] == "rlr_linearity"
    ]
    assert len(targets) == 2
    assert all(item["value"] is None for item in targets)
    assert all(item["status"] == "not_measured" for item in targets)
    assert all(item["evidence_grade"] == "not_measured" for item in targets)
    assert all("left 接收产物身份已改变" in item["reason"] for item in targets)
    assert not (
        fixture["destination"] / "WB-wireless-linearity.json"
    ).exists()


def test_finish_linearity_rejects_tampered_receive_report_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(
        tmp_path,
        monkeypatch,
        modes=("WB-wireless",),
    )
    nominal_case = next(
        case
        for case in fixture["selected_cases"]
        if case["conditions"]["input_level_db"] == 0
    )
    nominal_report = fixture["reports_by_case"][nominal_case["id"]]
    report_path = receive_simulation.get_synthetic_receive_file(
        nominal_report["evaluation_id"],
        "report.json",
    )
    assert report_path is not None
    report = json.loads(report_path.read_text())
    report["scope"] = "tampered after the original result hash"
    report_path.write_text(json.dumps(report))

    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    targets = [
        item
        for item in fixture["snapshot"]["items"]
        if item["mode"] == "WB-wireless"
        and item["family"] == "rlr_linearity"
    ]
    assert len(targets) == 2
    assert all(item["value"] is None for item in targets)
    assert all(item["status"] == "not_measured" for item in targets)
    assert all(
        "接收报告内容身份已改变" in item["reason"] for item in targets
    )
    assert not (
        fixture["destination"] / "WB-wireless-linearity.json"
    ).exists()


def test_finish_linearity_rejects_tampered_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_batch_fixture(
        tmp_path,
        monkeypatch,
        modes=("WB-wireless",),
    )
    nominal_case = next(
        case
        for case in fixture["selected_cases"]
        if case["conditions"]["input_level_db"] == 0
    )
    assessment_path = (
        fixture["destination"]
        / "cases"
        / nominal_case["id"]
        / "assessment.json"
    )
    assessment = json.loads(assessment_path.read_text())
    assessment["tampered"] = True
    assessment_path.write_text(json.dumps(assessment))

    acceptance_linearity.finish_linearity(
        fixture["snapshot"],
        fixture["dataset"],
        fixture["destination"],
    )

    targets = [
        item
        for item in fixture["snapshot"]["items"]
        if item["mode"] == "WB-wireless"
        and item["family"] == "rlr_linearity"
    ]
    assert len(targets) == 2
    assert all(item["value"] is None for item in targets)
    assert all(item["status"] == "not_measured" for item in targets)
    assert all("assessment.json 身份已改变" in item["reason"] for item in targets)
    assert not (
        fixture["destination"] / "WB-wireless-linearity.json"
    ).exists()
