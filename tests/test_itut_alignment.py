import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from import_itut_contract import read_contract
from run_itut_suite import collect
from build_itut_ledger import build_ledger
from build_vehicle_array_dataset import source_folder, spatial_case
from measure_itut_pcm import transfer_response, noise_windows
from publish_itut_ledger import publish
from tune_calibrated_eq import fit_response, response_db
import numpy as np


def workbook(path):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="WB-wired" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}"><si><t>11.02.1 Cal.: Round Trip Delay DVWB</t></si><si><t>Calculated Value [ms]</t></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="G1"><v>45862</v></c></row><row r="2"><c r="B2" t="s"><v>0</v></c><c r="C2" t="s"><v>1</v></c><c r="D2"><v>275</v></c><c r="E2"><v>170</v></c><c r="G2"><v>664.3</v></c></row></sheetData></worksheet>')


def test_workbook_values_stay_historical_and_unmeasured(tmp_path):
    path = tmp_path / "source.xlsx"
    workbook(path)
    report = read_contract(path)
    row = report["items"][0]
    assert row["id"] == "WB-wired:B2" and row["upper_limit"] == 275
    assert row["historical_measurements"][0]["value"] == 664.3
    assert row["historical_measurements"][0]["source_date"] == "2025-07-24"
    assert row["current_status"] == "not_measured" and report["standard_compliance"] is False


@pytest.mark.parametrize("process_exit,passed,expected", [(0, False, False), (1, True, False), (0, True, True)])
def test_suite_exit_zero_never_masks_failed_acceptance(tmp_path, process_exit, passed, expected):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(all_available_passed=passed, required_measurements_complete=True, phases=10, passed_phases=10 if passed else 9)))
    report = collect([dict(name="vehicle", kind="vehicle", report=str(path))], [process_exit])
    assert report["all_engineering_passed"] is expected
    assert report["standard_compliance"] is False and report["formal_test_status"] == "not_measured"


def test_suite_missing_report_is_failed_not_excluded(tmp_path):
    result = collect([dict(name="missing", kind="vehicle", report=str(tmp_path / "missing.json"))], [0])
    assert result["all_engineering_passed"] is False
    assert len(result["jobs"]) == 1 and result["jobs"][0]["status"] == "execution_error"
    assert collect([], [])["all_engineering_passed"] is False


@pytest.mark.parametrize("content", ['broken', '{}', '{"all_available_passed":"true"}', '{"all_available_passed":true}'])
def test_suite_corrupt_report_is_failed_not_excluded(tmp_path, content):
    path = tmp_path / "report.json"
    path.write_text(content)
    result = collect([dict(name="corrupt", kind="vehicle", report=str(path))], [0])
    assert not result["all_engineering_passed"] and result["jobs"][0]["status"] == "invalid_report"


def test_ledger_engineering_pass_never_passes_physical_test():
    contract = dict(source_sha256="source", items=[dict(id="WB:B2", sheet="WB", source_cell="B2",
                    test_name="delay", family="round_trip_delay", direction="round_trip", bandwidth="WB",
                    transport="wired", unit="ms", lower_limit=0, upper_limit=275, limit_source="customer")])
    alignment = dict(required_evidence={"round_trip_delay": ["physical calibrated loopback"]},
                     engineering_proxies={"round_trip_delay": "fixed DSP delay"}, standards={},
                     online_adaptation={}, forbidden_online_inputs=["clean"], formal_completion_rule="physical evidence")
    summary = dict(jobs=[dict(name="vehicle", acceptance_passed=True)], completed_jobs=1, planned_jobs=1)
    ledger = build_ledger(contract, alignment, summary)
    assert ledger["engineering"]["all_passed"]
    assert ledger["items"][0]["status"] == "not_measured" and not ledger["standard_compliance"]
    summary["planned_jobs"] = 2
    assert not build_ledger(contract, alignment, summary)["engineering"]["all_passed"]


def test_publication_rejects_changed_report_and_formal_claim(tmp_path):
    import hashlib
    suite = tmp_path / "suite"
    suite.mkdir()
    result = suite / "measured.json"
    result.write_text('{"passed":false}')
    ledger = dict(standard_compliance=False, formal_test_status="not_measured",
                  engineering=dict(results=[dict(name="test", report=str(result),
                                  report_sha256=hashlib.sha256(result.read_bytes()).hexdigest())]))
    (suite / "itut-ledger.json").write_text(json.dumps(ledger))
    for name in ("itut-ledger.md", "summary.json", "plan.json"):
        (suite / name).write_text('{}')
    publish(suite, tmp_path / "valid")
    result.write_text('{"passed":true}')
    with pytest.raises(ValueError, match="changed"):
        publish(suite, tmp_path / "wrong")
    ledger["standard_compliance"] = True
    (suite / "itut-ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="simulation"):
        publish(suite, tmp_path / "claim")


def test_array_catalog_resolves_input_then_clears_old_pointers():
    scene = dict(id="catalog-id", source_dataset="/frozen", source_case_id="original-id",
                 mics=1, files={"clean.wav": {"sha256": "frozen"}})
    assert source_folder(Path("/new-catalog"), scene) == Path("/frozen/cases/original-id")
    case = spatial_case(scene, 4)
    assert case["mics"] == 4 and case["derived_from_files"] == scene["files"]
    assert "source_dataset" not in case and "source_case_id" not in case
    assert source_folder(Path("/new-array"), case) == Path("/new-array/cases/catalog-id-m4")


def test_digital_response_and_noise_reject_invalid_measurement():
    rate = 16000
    x = np.random.default_rng(99).normal(0, .05, rate * 3)
    mask = dict(measurement_id="test", mask_source="synthetic engineering tolerance; not ITU", normalized_at_hz=1000,
                frequency_hz=np.geomspace(80, 6000, 20).tolist(), lower_db=[-.1] * 20, upper_db=[.1] * 20)
    y = np.r_[np.zeros(32), x * .5]
    report = transfer_response(x, y, rate, mask, 32)
    assert report["shape_mask_passed"] and not report["standard_compliance"]
    assert abs(report["absolute_gain_at_normalization_db"] + 6.0206) < .0001
    with pytest.raises(ValueError, match="excitation"):
        transfer_response(np.zeros_like(x), y, rate, mask)
    unrelated = np.random.default_rng(100).normal(0, .05, len(x))
    with pytest.raises(ValueError, match="coherence"):
        transfer_response(x, unrelated, rate, mask)
    assert noise_windows(np.full(rate, .1), rate, 50)["max_rms_dbfs"] == pytest.approx(-20)
    assert noise_windows(np.zeros(rate), rate, 100)["max_rms_dbfs"] is None


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_eq_fit_removes_measured_resonance_and_rejects_missing_calibration(rate):
    frequencies = np.unique(np.r_[np.geomspace(80, 6000, 100), 1000])
    measured = response_db(frequencies, [250], [6], rate)
    measured -= response_db([1000], [250], [6], rate)[0]
    record = dict(sample_rate=rate, scope="digital_diagnostic", measurement_id="synthetic-resonance",
                  mask_source="synthetic test tolerance; not ITU", normalized_at_hz=1000,
                  frequency_hz=frequencies.tolist(), measured_db=measured.tolist(),
                  lower_db=np.full(len(frequencies), -.25).tolist(), upper_db=np.full(len(frequencies), .25).tolist())
    result = fit_response(record)
    assert result["accepted_candidate"] and result["shape_mask_passed"]
    assert result["before_shape_margin_db"] < -4 and result["after_shape_margin_db"] > .20
    assert not result["standard_compliance"] and not result["online_adaptive_eq"]
    record["scope"] = "acoustic_calibrated"
    with pytest.raises(ValueError, match="calibration"):
        fit_response(record)
    record["calibration_id"] = "synthetic-test-id"
    record["frequency_hz"][3] = record["frequency_hz"][2]
    with pytest.raises(ValueError, match="ordered"):
        fit_response(record)
