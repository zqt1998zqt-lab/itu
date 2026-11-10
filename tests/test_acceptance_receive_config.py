from __future__ import annotations

import copy
import errno
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import (
    acceptance_batch,
    acceptance_linearity,
    receive_simulation,
    service,
)
from server.acoustic.acceptance_contract import contract_items
from server.acoustic.acceptance_dataset import (
    AcceptanceDatasetError,
    dataset_asset,
    get_dataset,
    import_dataset,
    template_manifest,
)
from server.acoustic.receive_configuration import (
    ReceiveConfigurationError,
    normalize_receive_configuration,
    receive_configuration_sha256,
)


SIGNALS = ("microphone", "reference", "far_end", "noise")
LEVELS = (0, -5, 5)
LEVEL_IDS = {0: "nominal", -5: "offset_-5", 5: "offset_+5"}


def test_enospc_is_propagated_without_cleanup_or_error_downgrade(tmp_path, monkeypatch):
    from server.acoustic.acceptance_dataset import _read_member
    class FullDestination:
        def write(self, data):
            raise OSError(errno.ENOSPC, "disk full")
        def close(self):
            pass
    destination = tmp_path / "asset.wav"
    original_open = Path.open
    def open_destination(path, *args, **kwargs):
        if path == destination:
            return FullDestination()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_destination)
    with zipfile.ZipFile(BytesIO(_zip([("asset.wav", b"source")])) ) as archive:
        with pytest.raises(OSError) as raised:
            _read_member(archive, archive.getinfo("asset.wav"), destination=destination)
    assert raised.value.errno == errno.ENOSPC


def test_download_template_exposes_acoustic_configuration():
    from server.acoustic.acceptance_api import template
    manifest = json.loads(template().body)
    assert manifest["receive_simulation"]["model"] == "head"
    assert sum(len(case["item_ids"]) for case in manifest["cases"]) == 336


def _wav(
    values: np.ndarray,
    rate: int = 16_000,
    subtype: str = "PCM_16",
) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype=subtype)
    return stream.getvalue()


def _zip(files: list[tuple[str, bytes]]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files:
            archive.writestr(name, payload)
    return stream.getvalue()


def _shared_signals() -> dict[str, bytes]:
    frames = 1_600
    time = np.arange(frames) / 16_000
    mono = 0.05 * np.sin(2 * np.pi * 440 * time)
    return {
        "audio/shared/microphone.wav": _wav(
            np.column_stack((mono, mono * 0.5))
        ),
        "audio/shared/reference.wav": _wav(mono),
        "audio/shared/far_end.wav": _wav(mono * 0.25),
        "audio/shared/noise.wav": _wav(np.zeros(2_400)),
    }


def _manifest(
    receive_configuration: dict | None = None,
) -> dict:
    manifest = template_manifest()
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav" for role in SIGNALS
        }
        case["analysis_start_s"] = 0
    if receive_configuration is not None:
        manifest["receive_simulation"] = copy.deepcopy(receive_configuration)
    return manifest


def _paired_ir_configuration() -> dict:
    return {
        "model": "paired_ir",
        "left_ir": "irs/left.wav",
        "right_ir": "irs/right.wav",
        "ir_kind": "hrir",
        "measurement_point": "drp",
        "provenance": "controlled paired-IR fixture",
    }


def _valid_ir_files() -> dict[str, bytes]:
    left = np.zeros(320)
    left[0] = 0.5
    left[3] = 0.1
    right = np.zeros(320)
    right[0] = 0.4
    right[5] = 0.08
    return {
        "irs/left.wav": _wav(left, subtype="FLOAT"),
        "irs/right.wav": _wav(right, subtype="FLOAT"),
    }


def _archive(
    manifest: dict,
    *,
    signals: dict[str, bytes] | None = None,
    ir_files: dict[str, bytes] | None = None,
) -> bytes:
    files = [
        (
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        ),
        *(_shared_signals() if signals is None else signals).items(),
    ]
    configuration = manifest.get("receive_simulation")
    if (
        isinstance(configuration, dict)
        and configuration.get("model") == "paired_ir"
    ):
        files.extend(
            (_valid_ir_files() if ir_files is None else ir_files).items()
        )
    return _zip(files)


class _CapturingExecutor:
    def __init__(self) -> None:
        self.submissions = []

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return None


def _controlled_outputs(
    case: dict,
    destination: Path,
    paths: dict[str, Path],
    *,
    cli_sha256: str,
    catalog_sha256: str,
) -> tuple[dict[str, Path], dict]:
    direction = case["conditions"]["direction"]
    kinds = (
        ("rx",)
        if direction == "receive"
        else ("tx", "rx")
        if direction == "round_trip"
        else ("tx",)
    )
    outputs = {
        kind: destination / f"{kind}.controlled.wav" for kind in kinds
    }
    for path in outputs.values():
        sf.write(path, np.zeros(160), 16_000, subtype="PCM_16")
    return outputs, {
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


def _controlled_rows(case: dict, report_id: str, execution: dict) -> list[dict]:
    return [
        {
            "item_id": item_id,
            "value": None,
            "status": "not_measured",
            "reason": "controlled receive-configuration test",
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


def test_receive_configuration_defaults_and_strict_head_contract():
    assert normalize_receive_configuration(None) == {
        "model": "head",
        "azimuth_deg": 30.0,
        "head_radius_m": 0.0875,
        "speed_of_sound_mps": 343.0,
        "measurement_point": "unspecified",
        "provenance": "uncalibrated software proxy",
    }
    assert normalize_receive_configuration({
        "model": "head",
        "azimuth_deg": 42,
        "measurement_point": "ear_entrance",
        "provenance": "fixture",
    })["azimuth_deg"] == 42.0


def test_receive_configuration_digest_canonicalizes_negative_zero_azimuth():
    negative_zero = normalize_receive_configuration({
        "model": "head",
        "azimuth_deg": -0.0,
    })
    positive_zero = normalize_receive_configuration({
        "model": "head",
        "azimuth_deg": 0.0,
    })

    assert negative_zero == positive_zero
    assert negative_zero["azimuth_deg"] == 0.0
    assert math.copysign(1.0, negative_zero["azimuth_deg"]) == 1.0
    assert receive_configuration_sha256(negative_zero, {}) == (
        receive_configuration_sha256(positive_zero, {})
    )


@pytest.mark.parametrize(
    "configuration",
    [
        {"model": "head", "azimuth_deg": True},
        {"model": "head", "azimuth_deg": 181},
        {"model": "head", "head_radius_m": 0.03},
        {"model": "head", "speed_of_sound_mps": 299},
        {"model": "head", "formal_hats_evidence": False},
        {"model": "head", "provenance": " "},
        {"model": "paired_ir", "left_ir": "irs/left.wav"},
        {
            "model": "paired_ir",
            "left_ir": "../left.wav",
            "right_ir": "irs/right.wav",
            "ir_kind": "hrir",
            "measurement_point": "drp",
            "provenance": "fixture",
        },
    ],
)
def test_receive_configuration_rejects_invalid_or_pseudo_formal_values(
    configuration,
):
    with pytest.raises(ReceiveConfigurationError):
        normalize_receive_configuration(configuration)


def test_legacy_dataset_import_exposes_default_configuration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    imported = import_dataset(_archive(_manifest()))
    stored = get_dataset(imported["dataset_id"])
    assert stored is not None
    assert "receive_simulation" not in stored["manifest"]
    assert stored["receive_simulation"] == normalize_receive_configuration(None)
    assert set(stored["assets"]) == set(_shared_signals())


def test_paired_ir_is_indexed_decoded_and_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    configuration = _paired_ir_configuration()
    imported = import_dataset(_archive(_manifest(configuration)))
    stored = get_dataset(imported["dataset_id"])
    assert stored is not None
    assert stored["receive_simulation"] == configuration
    assert set(stored["assets"]) == {
        *_shared_signals(),
        *_valid_ir_files(),
    }
    for relative, payload in _valid_ir_files().items():
        path = dataset_asset(imported["dataset_id"], relative)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            stored["assets"][relative]["sha256"]
        )
        assert stored["assets"][relative]["channels"] == 1
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            hashlib.sha256(payload).hexdigest()
        )


def test_ir_sample_rate_may_differ_from_signal_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    left = np.zeros(1_103)
    left[0] = 0.5
    right = np.zeros(1_103)
    right[0] = 0.4
    imported = import_dataset(
        _archive(
            _manifest(_paired_ir_configuration()),
            ir_files={
                "irs/left.wav": _wav(left, rate=11_025, subtype="FLOAT"),
                "irs/right.wav": _wav(right, rate=11_025, subtype="FLOAT"),
            },
        )
    )
    stored = get_dataset(imported["dataset_id"])
    assert stored is not None
    assert stored["assets"]["irs/left.wav"]["sample_rate"] == 11_025
    assert stored["assets"]["audio/shared/far_end.wav"]["sample_rate"] == 16_000


@pytest.mark.parametrize(
    ("configuration", "ir_files", "message"),
    [
        (_paired_ir_configuration(), {}, "missing referenced assets"),
        (
            _paired_ir_configuration(),
            {
                "irs/left.wav": b"not a wav",
                "irs/right.wav": _valid_ir_files()["irs/right.wav"],
            },
            "readable|decode",
        ),
        (
            _paired_ir_configuration(),
            {
                "irs/left.wav": _wav(np.zeros(320)),
                "irs/right.wav": _valid_ir_files()["irs/right.wav"],
            },
            "nonzero",
        ),
        (
            _paired_ir_configuration(),
            {
                "irs/left.wav": _wav(
                    np.asarray([0.5]),
                    rate=4_000,
                    subtype="FLOAT",
                ),
                "irs/right.wav": _valid_ir_files()["irs/right.wav"],
            },
            "sample rate",
        ),
        (
            _paired_ir_configuration(),
            {
                "irs/left.wav": _valid_ir_files()["irs/left.wav"],
                "irs/right.wav": _wav(
                    np.asarray([0.4]),
                    rate=192_001,
                    subtype="FLOAT",
                ),
            },
            "sample rate",
        ),
    ],
)
def test_dataset_rejects_missing_damaged_or_silent_ir(
    tmp_path,
    monkeypatch,
    configuration,
    ir_files,
    message,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    with pytest.raises(AcceptanceDatasetError, match=message):
        import_dataset(
            _archive(
                _manifest(configuration),
                ir_files=ir_files,
            )
        )


def test_dataset_rejects_unpaired_escaped_or_pseudo_formal_configuration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    unpaired = _paired_ir_configuration()
    unpaired.pop("right_ir")
    with pytest.raises(AcceptanceDatasetError, match="missing"):
        import_dataset(_archive(_manifest(unpaired)))

    escaped = _paired_ir_configuration()
    escaped["left_ir"] = "../left.wav"
    with pytest.raises(AcceptanceDatasetError, match="parent|relative"):
        import_dataset(_archive(_manifest(escaped)))

    pseudo_formal = {
        "model": "head",
        "standard_compliance": True,
    }
    with pytest.raises(AcceptanceDatasetError, match="unsupported fields"):
        import_dataset(_archive(_manifest(pseudo_formal)))


def test_batch_passes_resolved_configuration_and_ir_hashes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(
        acceptance_linearity,
        "finish_linearity",
        lambda snapshot, dataset, destination: None,
    )
    configuration = _paired_ir_configuration()
    imported = import_dataset(_archive(_manifest(configuration)))
    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    queued = acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")

    expected_ir_hashes = {
        role: get_dataset(imported["dataset_id"])["assets"][configuration[role]][
            "sha256"
        ]
        for role in ("left_ir", "right_ir")
    }
    assert queued["receive_simulation"] == configuration
    assert queued["receive_ir_sha256"] == expected_ir_hashes
    digest = receive_configuration_sha256(configuration, expected_ir_hashes)
    assert queued["receive_configuration_sha256"] == digest
    assert all(
        item["evidence_binding"]["receive_configuration_sha256"] == digest
        and item["evidence_binding"]["receive_ir_sha256"] == expected_ir_hashes
        for item in queued["items"]
    )
    assert queued["standard_compliance"] is False
    assert queued["formal_test_status"] == "not_measured"

    calculate_calls = []

    def controlled_run_cli(case, paths, destination, cluster):
        assert cluster == "vcp_default"
        assert set(paths) == set(SIGNALS)
        return _controlled_outputs(
            case,
            destination,
            paths,
            cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"],
        )

    def controlled_calculate(
        case,
        input_paths,
        outputs,
        execution,
        *,
        report_id,
        receive_config,
        receive_ir_paths,
    ):
        assert receive_config == configuration
        assert set(receive_ir_paths) == {"left_ir", "right_ir"}
        assert {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in receive_ir_paths.items()
        } == expected_ir_hashes
        assert execution["receive_simulation"] == configuration
        assert execution["receive_ir_sha256"] == expected_ir_hashes
        assert execution["standard_compliance"] is False
        calculate_calls.append(case["id"])
        return _controlled_rows(case, report_id, execution)

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)
    monkeypatch.setattr(
        acceptance_batch,
        "calculate_case_items",
        controlled_calculate,
    )
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert result["status"] == "completed"
    assert len(calculate_calls) == len(get_dataset(imported["dataset_id"])["manifest"]["cases"])
    assert result["receive_simulation"] == configuration
    assert result["receive_ir_sha256"] == expected_ir_hashes
    assert result["standard_compliance"] is False
    assert result["formal_test_status"] == "not_measured"
    first_assessment = next(
        (service.BUILD / "acceptance-batches" / queued["batch_id"] / "cases").glob(
            "*/assessment.json"
        )
    )
    assessment = json.loads(first_assessment.read_text())
    assert assessment["execution"]["receive_simulation"] == configuration
    assert assessment["execution"]["receive_ir_sha256"] == expected_ir_hashes


@pytest.mark.parametrize("change_configuration", [False, True])
def test_failed_batch_preserves_queued_acoustic_binding(tmp_path, monkeypatch, change_configuration):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", _CapturingExecutor())
    imported = import_dataset(_archive(_manifest(_paired_ir_configuration())))
    queued = acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    digest = queued["receive_configuration_sha256"]
    calls = []

    def fail_cli(*args):
        calls.append(args[0]["id"])
        raise ValueError("controlled executor failure")

    monkeypatch.setattr(acceptance_batch, "_run_cli", fail_cli)
    if change_configuration:
        dataset = get_dataset(imported["dataset_id"])
        dataset["receive_simulation"]["provenance"] = "changed after queueing"
        monkeypatch.setattr(acceptance_batch, "get_dataset", lambda identifier: dataset)
    result = acceptance_batch.run_batch(queued["batch_id"])
    assert result["status"] == "failed"
    assert result["receive_configuration_sha256"] == digest
    assert all(item["status"] == "invalid" for item in result["items"])
    assert all(
        item["evidence_binding"]["receive_configuration_sha256"] == digest
        and item["evidence_binding"]["receive_ir_sha256"] == queued["receive_ir_sha256"]
        for item in result["items"]
    )
    if change_configuration:
        assert not calls
        assert "配置身份发生变化" in result["error"]
    else:
        assert calls
        assert all(item["evidence_binding"]["report_id"] for item in result["items"])


@pytest.mark.parametrize("missing_field", ["batch", "item"])
def test_run_batch_rejects_missing_persisted_receive_binding_before_cli(
    tmp_path,
    monkeypatch,
    missing_field,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", _CapturingExecutor())
    imported = import_dataset(_archive(_manifest(_paired_ir_configuration())))
    queued = acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    report_path = (
        service.BUILD
        / "acceptance-batches"
        / queued["batch_id"]
        / "report.json"
    )
    persisted = json.loads(report_path.read_text())
    if missing_field == "batch":
        persisted.pop("receive_configuration_sha256")
    else:
        persisted["items"][0]["evidence_binding"].pop("receive_ir_sha256")
    report_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2))

    cli_calls = []

    def must_not_run(*args):
        cli_calls.append(args)
        raise AssertionError("missing persisted receive binding must fail before CLI")

    monkeypatch.setattr(acceptance_batch, "_run_cli", must_not_run)
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert cli_calls == []
    assert result["status"] == "failed"
    assert "接收声学配置" in result["error"]
    assert all(item["status"] == "invalid" for item in result["items"])
    saved = json.loads(report_path.read_text())
    if missing_field == "batch":
        assert "receive_configuration_sha256" not in saved
    else:
        assert "receive_ir_sha256" not in saved["items"][0]["evidence_binding"]


@pytest.mark.parametrize("mutation", ["trim_case", "change_analysis_start"])
def test_run_batch_rejects_manifest_drift_after_queueing_before_cli(
    tmp_path,
    monkeypatch,
    mutation,
):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", _CapturingExecutor())
    imported = import_dataset(_archive(_manifest()))
    queued = acceptance_batch.start_batch(imported["dataset_id"], "vcp_default")
    dataset = acceptance_batch.get_dataset(imported["dataset_id"])
    assert dataset is not None
    old_manifest_sha256 = dataset["manifest_sha256"]
    if mutation == "trim_case":
        dataset["manifest"]["cases"] = dataset["manifest"]["cases"][:-1]
    else:
        dataset["manifest"]["cases"][0]["analysis_start_s"] = 0.25
    assert dataset["manifest_sha256"] == old_manifest_sha256
    monkeypatch.setattr(
        acceptance_batch,
        "get_dataset",
        lambda identifier: dataset,
    )

    cli_calls = []

    def must_not_run(*args):
        cli_calls.append(args)
        raise AssertionError("manifest drift must fail before CLI")

    monkeypatch.setattr(acceptance_batch, "_run_cli", must_not_run)
    result = acceptance_batch.run_batch(queued["batch_id"])

    assert cli_calls == []
    assert result["status"] == "failed"
    assert "清单身份" in result["error"]
    assert all(item["status"] == "invalid" for item in result["items"])


def _speaker_wav(level_db: int) -> bytes:
    rate = 16_000
    frames = 3_200
    time = np.arange(frames) / rate
    amplitude = 0.02 * 10 ** (level_db / 20)
    return _wav(
        amplitude * np.sin(2 * np.pi * 440 * time),
        rate=rate,
        subtype="FLOAT",
    )


def _linearity_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    different_ir: bool,
) -> tuple[dict, dict, Path]:
    store = tmp_path / "synthetic-receive"
    monkeypatch.setattr(receive_simulation, "STORE", store)
    manifest = template_manifest()
    contract_by_id = {row["id"]: row for row in contract_items()}
    mode = "WB-wireless"
    selected = []
    for level in LEVELS:
        family = "rlr" if level == 0 else "rlr_linearity"
        matches = [
            case
            for case in manifest["cases"]
            if case["conditions"]["direction"] == "receive"
            and case["conditions"]["scene"] == "baseline"
            and case["conditions"]["volume"] == "nominal"
            and case["conditions"]["window_ms"] is None
            if acceptance_linearity._case_has_linearity_contract(
                case,
                contract_by_id,
                mode=mode,
                level=level,
                family=family,
            )
        ]
        assert len(matches) == 1
        case = copy.deepcopy(matches[0])
        case["analysis_start_s"] = 0
        selected.append(case)

    left_ir = np.zeros(32)
    left_ir[0] = 0.5
    left_ir[3] = 0.1
    right_ir = np.zeros(32)
    right_ir[0] = 0.45
    right_ir[5] = 0.08
    alternate_right_ir = np.zeros(32)
    alternate_right_ir[0] = 0.3
    alternate_right_ir[7] = 0.04
    left_ir_bytes = _wav(left_ir, subtype="FLOAT")
    right_ir_bytes = _wav(right_ir, subtype="FLOAT")
    alternate_right_ir_bytes = _wav(
        alternate_right_ir,
        subtype="FLOAT",
    )
    speaker_bytes_by_level = {
        level: _speaker_wav(level) for level in LEVELS
    }
    destination = tmp_path / "batch"
    snapshot = {
        "cases": [],
        "items": [],
    }
    reports = {}
    for case in selected:
        level = case["conditions"]["input_level_db"]
        selected_right_ir_bytes = (
            alternate_right_ir_bytes
            if different_ir and level == 5
            else right_ir_bytes
        )
        speaker_bytes = speaker_bytes_by_level[level]
        report = receive_simulation.evaluate_synthetic_receive(
            mode,
            f"{case['id']}.wav",
            speaker_bytes,
            left_ir_name="left.wav",
            left_ir_bytes=left_ir_bytes,
            right_ir_name="right.wav",
            right_ir_bytes=selected_right_ir_bytes,
        )
        reports[case["id"]] = report
        execution = {
            "source": "controlled-paired-ir-fixture",
            "batch_id": "batch-paired-ir",
            "dataset_id": "dataset-paired-ir",
            "cli_sha256": "c" * 64,
            "catalog_sha256": "d" * 64,
            "input_sha256": {
                "far_end": hashlib.sha256(speaker_bytes).hexdigest()
            },
            "output_sha256": {
                "rx": hashlib.sha256(f"{case['id']}:rx".encode()).hexdigest()
            },
        }
        report_id = hashlib.sha256(
            f"{case['id']}:assessment".encode()
        ).hexdigest()[:32]
        assessment = {
            "report_id": report_id,
            "case_id": case["id"],
            "execution": execution,
            "items": [{
                "item_id": case["item_ids"][0],
                "model_details": {
                    "receive_report_id": report["evaluation_id"],
                },
            }],
        }
        case_directory = destination / "cases" / case["id"]
        case_directory.mkdir(parents=True)
        assessment_path = case_directory / "assessment.json"
        assessment_path.write_text(json.dumps(assessment))
        snapshot["cases"].append({
            "case_id": case["id"],
            "report_id": report_id,
            "conditions": case["conditions"],
            "status": "completed",
            "assessment_sha256": hashlib.sha256(
                assessment_path.read_bytes()
            ).hexdigest(),
            "execution": execution,
        })
        snapshot["items"].append({
            "item_id": f"evidence-{case['id']}",
            "mode": mode,
            "family": "evidence",
            "status": "not_measured",
            "value": None,
            "evidence_binding": {
                "case_id": case["id"],
                "input_sha256": execution["input_sha256"],
            },
        })
    for row in contract_items():
        if row["sheet"] == mode and row["family"] == "rlr_linearity":
            snapshot["items"].append({
                "item_id": row["id"],
                "mode": mode,
                "family": "rlr_linearity",
                "source_cell": row["source_cell"],
                "status": "pending",
                "value": None,
                "evidence_binding": {},
            })
    dataset = {"manifest": {"cases": selected}}
    return snapshot, dataset, destination


def test_linearity_accepts_same_paired_ir_across_three_levels(
    tmp_path,
    monkeypatch,
):
    snapshot, dataset, destination = _linearity_fixture(
        tmp_path,
        monkeypatch,
        different_ir=False,
    )
    acceptance_linearity.finish_linearity(snapshot, dataset, destination)

    result_path = destination / "WB-wireless-linearity.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["group_valid"] is True
    assert result["conditions"]["head_model"] is None
    assert result["conditions"]["ir"]["evidence_kind"] == "synthetic_ir"
    assert result["formal_hats_evidence"] is False
    assert result["standard_compliance"] is False
    for item in snapshot["items"]:
        if item["family"] == "rlr_linearity":
            assert item["status"] == "review_required"
            assert math.isfinite(item["value"])


def test_linearity_rejects_different_paired_ir_across_levels(
    tmp_path,
    monkeypatch,
):
    snapshot, dataset, destination = _linearity_fixture(
        tmp_path,
        monkeypatch,
        different_ir=True,
    )
    acceptance_linearity.finish_linearity(snapshot, dataset, destination)

    assert not (destination / "WB-wireless-linearity.json").exists()
    targets = [
        item
        for item in snapshot["items"]
        if item["family"] == "rlr_linearity"
    ]
    assert targets
    assert all(item["value"] is None for item in targets)
    assert all(item["status"] == "not_measured" for item in targets)
    assert all(
        "声学模型身份不一致" in item["reason"] for item in targets
    )


def test_linearity_keeps_legacy_evidence_unbound_with_default_dataset_summary(
    tmp_path,
    monkeypatch,
):
    snapshot, dataset, destination = _linearity_fixture(
        tmp_path,
        monkeypatch,
        different_ir=False,
    )
    dataset["receive_simulation"] = normalize_receive_configuration(None)
    assert "receive_simulation" not in dataset["manifest"]
    assert dataset["receive_simulation"]["model"] == "head"

    cases_by_id = {
        case["id"]: case for case in dataset["manifest"]["cases"]
    }
    selected = []
    for record in snapshot["cases"]:
        assessment = json.loads(
            (
                destination
                / "cases"
                / record["case_id"]
                / "assessment.json"
            ).read_text()
        )
        receive_report_id = (
            assessment["items"][0]["model_details"]["receive_report_id"]
        )
        selected.append({
            "case": cases_by_id[record["case_id"]],
            "execution": record["execution"],
            "report": receive_simulation.get_synthetic_receive_evaluation(
                receive_report_id
            ),
        })

    assert acceptance_linearity._verified_configuration_binding(
        snapshot,
        dataset,
        selected,
    ) is None
    acceptance_linearity.finish_linearity(snapshot, dataset, destination)

    result = json.loads(
        (destination / "WB-wireless-linearity.json").read_text()
    )
    assert result["group_valid"] is True
    assert result["conditions"]["ir"]["evidence_kind"] == "synthetic_ir"
    assert result["configuration_binding_status"] == "legacy_unbound"
    assert result["receive_configuration_sha256"] is None
    for item in snapshot["items"]:
        if item["family"] == "rlr_linearity":
            assert all(
                "receive_configuration_sha256"
                not in run["execution_binding"]
                for run in item["model_details"]["source_runs"]
            )


def _bind_linearity_configuration(snapshot, dataset, destination):
    configuration = _paired_ir_configuration()
    for record in snapshot["cases"]:
        path = destination / "cases" / record["case_id"] / "assessment.json"
        assessment = json.loads(path.read_text())
        receive_id = assessment["items"][0]["model_details"]["receive_report_id"]
        report = receive_simulation.get_synthetic_receive_evaluation(receive_id)
        hashes = {role: report["input_identity"][role]["sha256"] for role in ("left_ir", "right_ir")}
        digest = receive_configuration_sha256(configuration, hashes)
        fields = {"receive_simulation": configuration, "receive_ir_sha256": hashes,
                  "receive_configuration_sha256": digest}
        record["execution"].update(copy.deepcopy(fields))
        snapshot.update(copy.deepcopy(fields))
        assessment["execution"] = record["execution"]
        path.write_text(json.dumps(assessment))
        record["assessment_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in snapshot["items"]:
            if item["evidence_binding"].get("case_id") == record["case_id"]:
                item["evidence_binding"].update(receive_configuration_sha256=digest, receive_ir_sha256=hashes)
    dataset["manifest"]["receive_simulation"] = copy.deepcopy(configuration)
    return digest


def test_linearity_preserves_verified_configuration_digest(tmp_path, monkeypatch):
    snapshot, dataset, destination = _linearity_fixture(tmp_path, monkeypatch, different_ir=False)
    digest = _bind_linearity_configuration(snapshot, dataset, destination)
    acceptance_linearity.finish_linearity(snapshot, dataset, destination)
    result = json.loads((destination / "WB-wireless-linearity.json").read_text())
    assert result["group_valid"] is True
    assert result["configuration_binding_status"] == "verified"
    assert result["receive_configuration_sha256"] == digest
    for item in snapshot["items"]:
        if item["family"] == "rlr_linearity":
            assert all(run["execution_binding"]["receive_configuration_sha256"] == digest
                       for run in item["model_details"]["source_runs"])


@pytest.mark.parametrize("change", ["provenance", "measurement_point", "ir_kind", "missing", "stale_hash"])
def test_linearity_rejects_config_drift_with_identical_ir(tmp_path, monkeypatch, change):
    snapshot, dataset, destination = _linearity_fixture(tmp_path, monkeypatch, different_ir=False)
    _bind_linearity_configuration(snapshot, dataset, destination)
    record = snapshot["cases"][-1]
    execution = record["execution"]
    if change == "missing":
        execution.pop("receive_configuration_sha256")
    elif change == "stale_hash":
        execution["receive_configuration_sha256"] = "0" * 64
    else:
        execution["receive_simulation"][change] = {
            "provenance": "different source", "measurement_point": "erp", "ir_kind": "brir",
        }[change]
        execution["receive_configuration_sha256"] = receive_configuration_sha256(
            execution["receive_simulation"], execution["receive_ir_sha256"],
        )
    # Keep assessment identity valid so rejection tests semantic binding drift.
    path = destination / "cases" / record["case_id"] / "assessment.json"
    assessment = json.loads(path.read_text())
    assessment["execution"] = execution
    path.write_text(json.dumps(assessment))
    record["assessment_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    acceptance_linearity.finish_linearity(snapshot, dataset, destination)
    assert not (destination / "WB-wireless-linearity.json").exists()
    targets = [item for item in snapshot["items"] if item["family"] == "rlr_linearity"]
    assert targets and all(item["status"] == "not_measured" and item["value"] is None for item in targets)
    assert all("配置" in item["reason"] for item in targets), [item["reason"] for item in targets]
