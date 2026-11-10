from __future__ import annotations

import copy
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
    acceptance_models,
    receive_simulation,
    service,
)
from server.acoustic.acceptance_contract import contract_items
from server.acoustic.acceptance_dataset import (
    get_dataset,
    import_dataset,
    template_manifest,
)
from server.acoustic.receive_configuration import (
    receive_configuration_sha256,
)


RATE = 24_000
FRAMES = RATE * 4
SIGNAL_NAMES = ("microphone", "reference", "far_end", "noise")


def _wav(values: np.ndarray, *, rate: int = RATE) -> bytes:
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def _shared_signals() -> dict[str, bytes]:
    rng = np.random.default_rng(20261104)
    far_end = 0.02 * rng.standard_normal(FRAMES)
    return {
        "audio/shared/microphone.wav": _wav(0.5 * far_end),
        "audio/shared/reference.wav": _wav(far_end),
        "audio/shared/far_end.wav": _wav(far_end),
        "audio/shared/noise.wav": _wav(
            0.0001 * rng.standard_normal(FRAMES)
        ),
    }


def _paired_ir_configuration() -> dict:
    return {
        "model": "paired_ir",
        "left_ir": "irs/left.wav",
        "right_ir": "irs/right.wav",
        "ir_kind": "hrir",
        "measurement_point": "drp",
        "provenance": "controlled paired-IR integration fixture",
    }


def _scaled_ir_files(gain: float) -> dict[str, bytes]:
    left = np.zeros(32)
    left[0] = 0.5 * gain
    left[3] = 0.1 * gain
    right = np.zeros(32)
    right[0] = 0.4 * gain
    right[5] = 0.08 * gain
    return {
        "irs/left.wav": _wav(left),
        "irs/right.wav": _wav(right),
    }


def _manifest(configuration: dict) -> dict:
    manifest = template_manifest()
    for case in manifest["cases"]:
        case["signals"] = {
            role: f"audio/shared/{role}.wav" for role in SIGNAL_NAMES
        }
        case["analysis_start_s"] = 0
    manifest["receive_simulation"] = copy.deepcopy(configuration)
    return manifest


def _archive(
    manifest: dict,
    *,
    signals: dict[str, bytes],
    ir_gain: float,
) -> bytes:
    files = {
        "manifest.json": json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        **signals,
        **_scaled_ir_files(ir_gain),
    }
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
    outputs = {}
    if direction in {"send", "round_trip"}:
        samples, rate = sf.read(paths["microphone"], always_2d=True)
        output = destination / "tx.controlled.wav"
        sf.write(output, np.asarray(samples) * 0.75, rate, subtype="PCM_24")
        outputs["tx"] = output
    if direction in {"receive", "round_trip"}:
        samples, rate = sf.read(paths["far_end"], always_2d=True)
        output = destination / "rx.controlled.wav"
        sf.write(output, np.asarray(samples) * 0.75, rate, subtype="PCM_24")
        outputs["rx"] = output
    return outputs, {
        "source": "controlled-external-vcp-executor",
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


def _run_until_first_receive_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_id: str,
    target_case_id: str,
) -> tuple[dict, dict, dict, list[str]]:
    stored = get_dataset(dataset_id)
    assert stored is not None

    executor = _CapturingExecutor()
    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", executor)
    queued = acceptance_batch.start_batch(dataset_id, "vcp_default")
    assert len(executor.submissions) == 1

    completed_calls = []
    target_returned = False

    def controlled_run_cli(case, paths, destination, cluster):
        nonlocal target_returned
        if target_returned:
            raise KeyboardInterrupt(
                "controlled stop after the first real receive case"
            )
        assert cluster == "vcp_default"
        assert set(paths) == set(SIGNAL_NAMES)
        result = _controlled_outputs(
            case,
            destination,
            paths,
            cli_sha256=queued["cli_sha256"],
            catalog_sha256=queued["catalog_sha256"],
        )
        completed_calls.append(case["id"])
        if case["id"] == target_case_id:
            target_returned = True
        return result

    monkeypatch.setattr(acceptance_batch, "_run_cli", controlled_run_cli)

    with pytest.raises(KeyboardInterrupt, match="controlled stop"):
        acceptance_batch.run_batch(queued["batch_id"])
    assert target_returned
    assert queued["batch_id"] not in acceptance_batch._ACTIVE

    # get_batch intentionally converts a nonterminal orphan into failed. Keep
    # the just-finished worker identity live only for this read, then release
    # it in finally so the persisted running snapshot can be inspected.
    try:
        acceptance_batch._ACTIVE.add(queued["batch_id"])
        snapshot = acceptance_batch.get_batch(queued["batch_id"])
    finally:
        acceptance_batch._ACTIVE.discard(queued["batch_id"])
    assert snapshot is not None
    return snapshot, stored, queued, completed_calls


def _read_ear_pcm(
    evaluation_id: str,
    artifact: dict,
) -> tuple[np.ndarray, np.ndarray]:
    ears = []
    for role in ("left", "right"):
        path = receive_simulation.get_synthetic_receive_file(
            evaluation_id,
            f"virtual_{role}.wav",
        )
        assert path is not None
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            artifact[role]["sha256"]
        )
        samples, rate = sf.read(path, dtype="float64", always_2d=False)
        assert rate == RATE
        assert samples.ndim == 1
        assert np.isfinite(samples).all()
        ears.append(samples)
    assert len(ears[0]) == len(ears[1])
    return ears[0], ears[1]


def _assert_receive_evidence(
    snapshot: dict,
    stored: dict,
    queued: dict,
    *,
    case_id: str,
    item_id: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    configuration = stored["receive_simulation"]
    expected_ir_hashes = {
        role: stored["assets"][configuration[role]]["sha256"]
        for role in ("left_ir", "right_ir")
    }
    expected_configuration_digest = receive_configuration_sha256(
        configuration,
        expected_ir_hashes,
    )
    assert queued["receive_simulation"] == configuration
    assert queued["receive_ir_sha256"] == expected_ir_hashes
    assert queued["receive_configuration_sha256"] == (
        expected_configuration_digest
    )
    assert len(queued["items"]) == 336
    assert all(
        item["evidence_binding"]["receive_configuration_sha256"]
        == expected_configuration_digest
        and item["evidence_binding"]["receive_ir_sha256"]
        == expected_ir_hashes
        for item in queued["items"]
    )
    assert queued["progress"]["total_cases"] == 106
    assert queued["progress"]["total_items"] == 336

    assert snapshot["status"] == "running"
    assert snapshot["receive_simulation"] == configuration
    assert snapshot["receive_ir_sha256"] == expected_ir_hashes
    assert snapshot["progress"]["total_cases"] == 106
    assert snapshot["progress"]["total_items"] == 336
    assert 0 < snapshot["progress"]["completed_cases"] < 106
    assert 0 < snapshot["progress"]["completed_items"] < 336
    assert snapshot["standard_compliance"] is False
    assert snapshot["formal_test_status"] == "not_measured"
    assert "finished_at" not in snapshot
    assert any(
        case["case_id"] == case_id and case["status"] == "completed"
        for case in snapshot["cases"]
    )

    row = next(item for item in snapshot["items"] if item["item_id"] == item_id)
    assert row["status"] == "review_required"
    assert row["metric"] == "receive_h1_frequency_response_span_db"
    assert math.isfinite(row["value"])
    assert row["standard_compliance"] is False

    binding = row["evidence_binding"]
    assert binding["receive_ir_sha256"] == expected_ir_hashes
    configuration_digest = binding["receive_configuration_sha256"]
    assert isinstance(configuration_digest, str)
    assert len(configuration_digest) == 64
    int(configuration_digest, 16)
    assert configuration_digest == expected_configuration_digest
    assert snapshot["receive_configuration_sha256"] == (
        expected_configuration_digest
    )

    assessment_path = (
        service.BUILD
        / "acceptance-batches"
        / snapshot["batch_id"]
        / "cases"
        / case_id
        / "assessment.json"
    )
    assessment = json.loads(assessment_path.read_text())
    execution = assessment["execution"]
    assert execution["source"] == "controlled-external-vcp-executor"
    assert execution["receive_simulation"] == configuration
    assert execution["receive_ir_sha256"] == expected_ir_hashes
    assert execution["receive_configuration_sha256"] == configuration_digest
    assert execution["synthetic_proxy"] is True
    assert execution["standard_compliance"] is False
    assessment_item = next(
        item for item in assessment["items"] if item["item_id"] == item_id
    )
    assert assessment_item["evidence_binding"] == binding

    diagnostics = row["diagnostics"]["synthetic_receive"]
    assert diagnostics["evidence_kind"] == "synthetic_ir"
    assert diagnostics["acoustic_configuration"] == configuration
    assert diagnostics["simulation"]["head_model"] is None
    assert diagnostics["simulation"]["output_normalisation"]["normalised"] is False
    transfer = row["model_details"]["receive_transfer_response"]
    assert transfer["available"] is True, transfer
    assert math.isfinite(transfer["span_db"])
    assert transfer["reference_sha256"] == binding["input_sha256"]["far_end"]

    evaluation_id = row["model_details"]["receive_report_id"]
    report = receive_simulation.get_synthetic_receive_evaluation(evaluation_id)
    assert report is not None
    assert report["evidence_kind"] == "synthetic_ir"
    assert report["formal_hats_evidence"] is False
    assert report["standard_compliance"] is False
    for role in ("left_ir", "right_ir"):
        assert report["input_identity"][role]["sha256"] == (
            expected_ir_hashes[role]
        )
    left, right = _read_ear_pcm(evaluation_id, report["artifacts"])
    return left, right, report


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def test_zip_to_real_batch_receive_calculator_binds_ir_and_scales_ears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(
        receive_simulation,
        "STORE",
        tmp_path / "synthetic-receive",
    )
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)

    def forbidden_external_quality_model():
        raise AssertionError(
            "frequency-response receive evidence must not invoke an external model"
        )

    monkeypatch.setattr(
        acceptance_models,
        "_dnsmos_evaluator",
        forbidden_external_quality_model,
    )

    rows = contract_items()
    manifest = template_manifest()
    selected_case = next(
        case
        for case in manifest["cases"]
        if case["conditions"]["direction"] == "receive"
    )
    target = next(
        row
        for row in rows
        if row["id"] in selected_case["item_ids"]
        and row["family"] == "frequency_response"
    )
    assert len(manifest["cases"]) == 106
    assert sum(len(case["item_ids"]) for case in manifest["cases"]) == 336

    configuration = _paired_ir_configuration()
    signals = _shared_signals()
    unity_dataset_id = import_dataset(
        _archive(
            _manifest(configuration),
            signals=signals,
            ir_gain=1.0,
        )
    )["dataset_id"]
    half_dataset_id = import_dataset(
        _archive(
            _manifest(configuration),
            signals=signals,
            ir_gain=0.5,
        )
    )["dataset_id"]

    unity_snapshot, unity_stored, unity_queued, unity_calls = (
        _run_until_first_receive_case(
            monkeypatch,
            dataset_id=unity_dataset_id,
            target_case_id=selected_case["id"],
        )
    )
    half_snapshot, half_stored, half_queued, half_calls = (
        _run_until_first_receive_case(
            monkeypatch,
            dataset_id=half_dataset_id,
            target_case_id=selected_case["id"],
        )
    )
    expected_calls = [
        case["id"]
        for case in manifest["cases"][: manifest["cases"].index(selected_case) + 1]
    ]
    assert unity_calls == expected_calls
    assert half_calls == expected_calls

    unity_left, unity_right, unity_report = _assert_receive_evidence(
        unity_snapshot,
        unity_stored,
        unity_queued,
        case_id=selected_case["id"],
        item_id=target["id"],
    )
    half_left, half_right, half_report = _assert_receive_evidence(
        half_snapshot,
        half_stored,
        half_queued,
        case_id=selected_case["id"],
        item_id=target["id"],
    )
    assert unity_report["input_identity"]["left_ir"]["sha256"] != (
        half_report["input_identity"]["left_ir"]["sha256"]
    )
    assert unity_report["input_identity"]["right_ir"]["sha256"] != (
        half_report["input_identity"]["right_ir"]["sha256"]
    )

    left_change_db = 20 * math.log10(_rms(half_left) / _rms(unity_left))
    right_change_db = 20 * math.log10(_rms(half_right) / _rms(unity_right))
    assert left_change_db == pytest.approx(-6.0206, abs=0.002)
    assert right_change_db == pytest.approx(-6.0206, abs=0.002)
