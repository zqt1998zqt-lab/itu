from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import outcome_codex, service


MODE = "WB-wired"
EVALUATION_ID = "a" * 32
THREAD_ID = "01a18053-0ba2-7c63-ab3a-10b9f5e0f20a"


def _report(
    *,
    counts: dict[str, int] | None = None,
    evaluation_id: str = EVALUATION_ID,
    result_sha: str = "b" * 64,
) -> dict:
    counts = counts or {
        "pass": 35,
        "fail": 2,
        "review_required": 3,
        "not_measured": 1,
        "invalid": 1,
    }
    statuses = [
        status
        for status in (
            "pass",
            "fail",
            "review_required",
            "not_measured",
            "invalid",
        )
        for _ in range(counts[status])
    ]
    assert len(statuses) == 42
    items = []
    for index, status in enumerate(statuses):
        items.append(
            {
                "item_id": f"{MODE}:row-{index}",
                "source_cell": f"A{index + 1}",
                "family": "sisdr",
                "direction": "send",
                "test_name": f"工程测项 {index}",
                "unit": "dB",
                "lower_limit": 0,
                "upper_limit": None,
                "metric": "output_sisdr_db",
                "value": -3.0 if status == "fail" else 12.0,
                "status": status,
                "reason": {
                    "fail": "低于工程门限",
                    "review_required": "需要人工确认",
                    "not_measured": "缺少正式方法包",
                    "invalid": "计算值不是有限数",
                    "pass": "",
                }[status],
                "evidence_grade": (
                    "not_measured" if status == "not_measured" else "engineering_proxy"
                ),
                "standard_compliance": False,
            }
        )
    return {
        "evaluation_id": evaluation_id,
        "created_at": "2026-10-28T00:00:00+00:00",
        "mode": MODE,
        "summary": {
            "total": 42,
            "measured": 42 - counts["not_measured"] - counts["invalid"],
            **counts,
        },
        "analysis": {
            "alignments": {
                "microphone": {"delay_ms": 10.0, "correlation": 0.98},
                "near_end": {"delay_ms": 11.0, "correlation": 0.97},
            },
            "segments": {"double_talk_seconds": 1.2},
            "values": {"output_sisdr_db": 8.4, "output_stoi": 0.91},
        },
        "items": items,
        "limitations": ["工程代理，不是正式 ITU-T 合规。"],
        "result": {"path": f"five-signal-evaluations/{evaluation_id}/result.json", "sha256": result_sha},
    }


def _run(payload: dict | None = None, *, thread_id: str = THREAD_ID):
    return outcome_codex.CodexThreadRun(
        thread_id=thread_id,
        thread_created=True,
        payload=payload
        or {
            "summary": "失败集中在输出 SI-SDR，先排查 AEC/NS 链路。",
            "root_causes": [
                {
                    "cause": "输出残留回声压低了 SI-SDR。",
                    "evidence_refs": [f"{MODE}:row-35"],
                    "confidence": "medium",
                }
            ],
            "actions": [
                {
                    "priority": "P0",
                    "action": "检查 AEC 收敛和参考时延。",
                    "validation": "重跑同一五路输入，确认失败项消失。",
                    "evidence_refs": [f"{MODE}:row-35"],
                }
            ],
            "limitations": ["未直接复算原始音频。"],
        },
        model="gpt-6-luna",
        stderr_tail="",
    )


def test_empty_outcome_never_calls_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    report = _report(
        counts={
            "pass": 37,
            "fail": 0,
            "review_required": 3,
            "not_measured": 1,
            "invalid": 1,
        }
    )
    monkeypatch.setattr(outcome_codex, "get_evaluation", lambda _: report)
    monkeypatch.setattr(
        outcome_codex,
        "_run_codex_thread",
        lambda *args, **kwargs: pytest.fail("empty outcome must not start a thread"),
    )
    result = outcome_codex.analyze_outcome("fail", [EVALUATION_ID])
    assert result["status"] == "empty"
    assert result["item_count"] == 0
    assert result["root_causes"] == []
    assert "未调用 Codex 线程" in result["summary"]


def test_fail_agent_uses_persisted_items_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(outcome_codex, "get_evaluation", lambda _: _report())
    captured = {}

    def fake_run(outcome, prompt, thread_id):
        captured["outcome"] = outcome
        captured["prompt"] = prompt
        captured["thread_id"] = thread_id
        return _run()

    monkeypatch.setattr(outcome_codex, "_run_codex_thread", fake_run)
    first = outcome_codex.analyze_outcome("fail", [EVALUATION_ID])
    assert first["status"] == "ready"
    assert first["item_count"] == 2
    assert first["thread_id"] == THREAD_ID
    assert captured["outcome"] == "fail"
    assert captured["thread_id"] is None
    assert f"{MODE}:row-35" in captured["prompt"]
    assert "row-0" not in captured["prompt"]
    store = json.loads(
        (service.BUILD / outcome_codex.THREAD_STORE_NAME).read_text()
    )
    assert store["fail"]["thread_id"] == THREAD_ID

    monkeypatch.setattr(
        outcome_codex,
        "_run_codex_thread",
        lambda *args, **kwargs: pytest.fail("cached analysis must not start a thread"),
    )
    cached = outcome_codex.analyze_outcome("fail", [EVALUATION_ID])
    assert cached["cached"] is True
    assert cached["summary"] == first["summary"]


def test_second_analysis_resumes_the_same_saved_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    reports = [
        _report(result_sha="b" * 64),
        _report(result_sha="c" * 64),
    ]
    monkeypatch.setattr(
        outcome_codex,
        "get_evaluation",
        lambda evaluation_id: reports[0],
    )
    calls = []

    def fake_run(outcome, prompt, thread_id):
        calls.append(thread_id)
        return _run()

    monkeypatch.setattr(outcome_codex, "_run_codex_thread", fake_run)
    outcome_codex.analyze_outcome("fail", [EVALUATION_ID])
    monkeypatch.setattr(
        outcome_codex,
        "get_evaluation",
        lambda evaluation_id: reports[1],
    )
    outcome_codex.analyze_outcome("fail", [EVALUATION_ID], refresh=True)
    assert calls == [None, THREAD_ID]


def test_unknown_evidence_reference_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(outcome_codex, "get_evaluation", lambda _: _report())
    payload = _run().payload
    payload["root_causes"][0]["evidence_refs"] = ["invented:row"]
    monkeypatch.setattr(
        outcome_codex,
        "_run_codex_thread",
        lambda *args, **kwargs: _run(payload),
    )
    with pytest.raises(outcome_codex.OutcomeCodexFailure, match="不存在的证据"):
        outcome_codex.analyze_outcome("fail", [EVALUATION_ID])


def test_contradictory_report_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    report = _report()
    report["summary"]["fail"] = 3
    monkeypatch.setattr(outcome_codex, "get_evaluation", lambda _: report)
    with pytest.raises(outcome_codex.OutcomeCodexError, match="汇总计数自相矛盾"):
        outcome_codex.analyze_outcome("fail", [EVALUATION_ID])


def test_http_endpoint_routes_outcome_to_codex_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path / "build")
    monkeypatch.setattr(outcome_codex, "get_evaluation", lambda _: _report())
    monkeypatch.setattr(
        outcome_codex,
        "_run_codex_thread",
        lambda *args, **kwargs: _run(),
    )
    response = TestClient(app).post(
        "/api/acoustic/five-signal/outcomes/fail/analyze",
        json={"evaluation_ids": [EVALUATION_ID], "refresh": False},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "ready"
    assert result["thread_id"] == THREAD_ID
    assert result["item_count"] == 2
