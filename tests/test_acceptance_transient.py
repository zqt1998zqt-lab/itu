from __future__ import annotations

import hashlib
from io import BytesIO
import json

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import acceptance_models, five_signal, service
from server.acoustic.acceptance_contract import required_conditions


def _wav(samples: np.ndarray, rate: int) -> bytes:
    buffer = BytesIO()
    sf.write(
        buffer,
        np.asarray(samples, dtype=np.float32),
        rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def _signals(
    rate: int = 16_000,
    seconds: float = 4.0,
) -> dict[str, tuple[str, bytes]]:
    rng = np.random.default_rng(20261104)
    frames = round(rate * seconds)
    time = np.arange(frames) / rate
    near = np.zeros(frames)
    far = np.zeros(frames)
    near_active = (time >= 0.25) & (time < 1.4)
    double_talk_active = (time >= 1.55) & (time < 2.5)
    far_active = (time >= 2.65) & (time < 3.6)
    near[near_active | double_talk_active] = (
        0.18
        * np.sin(2 * np.pi * 220 * time[near_active | double_talk_active])
        * (
            0.65
            + 0.35
            * np.sin(2 * np.pi * 3.3 * time[near_active | double_talk_active])
        )
    )
    far[double_talk_active | far_active] = (
        0.12
        * np.sin(2 * np.pi * 310 * time[double_talk_active | far_active])
        * (
            0.7
            + 0.3
            * np.cos(2 * np.pi * 2.1 * time[double_talk_active | far_active])
        )
    )
    noise = rng.normal(0, 0.003, frames)
    echo = np.r_[np.zeros(160), far[:-160]] * 0.35
    microphone = near + echo + noise
    near_end = near + echo * 0.05 + noise * 0.2
    return {
        "near_end": ("near-end.wav", _wav(near_end, rate)),
        "noise": ("noise.wav", _wav(noise, rate)),
        "microphone": ("microphone.wav", _wav(microphone, rate)),
        "far_end": ("far-end.wav", _wav(far, rate)),
        "reference": ("reference.wav", _wav(near, rate)),
    }


def test_default_evaluation_still_persists_evidence_and_report(
    tmp_path,
    monkeypatch,
) -> None:
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")

    report = five_signal.evaluate_five_signal(
        mode="NB-wired",
        uploads=_signals(),
    )

    assert "persistence" not in report
    assert set(report["evidence"]) == set(five_signal.ROLE_ORDER)
    assert report["result"]["path"].endswith("result.payload.json")
    result_path = build / report["result"]["path"]
    assert result_path.is_file()
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        report["result"]["sha256"]
    )
    assert json.loads(result_path.read_text())["evaluation_id"] == (
        report["evaluation_id"]
    )
    for record in report["evidence"].values():
        evidence_path = build / "evidence" / record["artifact"]["path"]
        assert evidence_path.is_file()
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == (
            record["artifact"]["sha256"]
        )


def test_transient_evaluation_skips_all_persistence(
    tmp_path,
    monkeypatch,
) -> None:
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")

    def reject_persistence(*args, **kwargs):
        pytest.fail("persist=False must not invoke persistence helpers")

    monkeypatch.setattr(five_signal, "_persist_inputs", reject_persistence)
    monkeypatch.setattr(five_signal, "_persist_report", reject_persistence)

    report = five_signal.evaluate_five_signal(
        mode="NB-wired",
        uploads=_signals(),
        persist=False,
    )

    assert report["persistence"]["mode"] == "transient"
    assert report["persistence"]["persisted"] is False
    assert "evidence" not in report
    assert "result" not in report
    assert report["summary"]["total"] == 42
    assert len(report["items"]) == 42
    assert report["analysis"]["values"]
    assert not (build / "evidence").exists()
    assert not (build / "five-signal-evaluations").exists()


def test_calculate_case_items_uses_transient_five_signal_and_returns_rows(
    tmp_path,
    monkeypatch,
) -> None:
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    uploads = _signals()
    input_paths = {}
    for role in ("reference", "noise", "microphone", "far_end"):
        path = tmp_path / "inputs" / uploads[role][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploads[role][1])
        input_paths[role] = path
    output_path = tmp_path / "outputs" / "tx.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(uploads["near_end"][1])
    outputs = {"tx": output_path}
    execution = {
        "batch_id": "batch-transient",
        "dataset_id": "dataset-transient",
        "output_sha256": {
            "tx": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
    }
    original = acceptance_models.evaluate_five_signal
    calls = []

    def evaluate_transient(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        evaluate_transient,
    )
    item_ids = ["NB-wired:B19", "NB-wired:B21", "NB-wired:B23"]
    case = {
        "id": "transient-case",
        "conditions": required_conditions(
            acceptance_models._contract_lookup()[item_ids[0]]
        ),
        "item_ids": item_ids,
    }

    rows = acceptance_models.calculate_case_items(
        case,
        input_paths,
        outputs,
        execution,
        report_id="transient-report",
    )

    assert len(calls) == 1
    assert calls[0]["persist"] is False
    assert [row["item_id"] for row in rows] == item_ids
    assert all(row["standard_compliance"] is False for row in rows)
    assert all(
        row["evidence_binding"]["output_sha256"]["tx"]
        == execution["output_sha256"]["tx"]
        for row in rows
    )
    assert output_path.is_file()
    assert not (build / "evidence").exists()
    assert not (build / "five-signal-evaluations").exists()
