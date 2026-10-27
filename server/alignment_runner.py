"""Auditable one-click engineering self-check for the ITU audio stack."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Literal
import xml.etree.ElementTree as ET

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/alignment", tags=["alignment-runner"])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXEC = "/home/zhangqingtian/.venv/bin/python"

CLI_CANDIDATES = [
    PROJECT_ROOT / "build/vcp-unified-final-20261027/cpp/itu_audio_cli",
    PROJECT_ROOT / "build/enhancement-chain-matrix-20261027-vcp-unified-final/artifacts/itu_audio_cli",
]

DEFAULT_MIC = (
    PROJECT_ROOT
    / "datasets/itu-vehicle-fields-float-20261010/cases/field-p0-m2-cold-snr0/before.wav"
)
DEFAULT_REF = (
    PROJECT_ROOT
    / "datasets/itu-vehicle-fields-float-20261010/cases/field-p0-m2-cold-snr0/ref.wav"
)

PIPELINE_SPECS: dict[str, dict[str, str]] = {
    "vcp_unified": {
        "cli_profile": "vcp_unified",
        "name": "VCP Unified 统一车载语音处理流水线",
        "topology": "TX IN (2ch) → BF → Fused AEC → Unified NS → Fused EQ/AGC → DRC → TX OUT",
    },
    "vcp_joint": {
        "cli_profile": "vcp_joint",
        "name": "VCP Joint 联合降噪流水线",
        "topology": "TX IN (2ch) → BF → AEC → Super-Gaussian NS → AGC/DRC → TX OUT",
    },
    "vcp_default": {
        # The public API keeps its existing id, while the pinned CLI uses `vcp`.
        "cli_profile": "vcp",
        "name": "VCP 传统标准基线",
        "topology": "TX IN (2ch) → BF → Speex AEC → WebRTC NS → AGC/DRC → TX OUT",
    },
}

TEST_FILE_GROUPS = {
    "methods": "tests/test_itu_method_metrics.py",
    "acoustic": "tests/test_acoustic_qualification.py",
    "contract": "tests/test_itut_alignment.py",
}

SCOPE_TEST_FILES = {
    "full": [
        TEST_FILE_GROUPS["contract"],
        TEST_FILE_GROUPS["methods"],
        TEST_FILE_GROUPS["acoustic"],
    ],
    "methods": [TEST_FILE_GROUPS["methods"]],
    "contract": [TEST_FILE_GROUPS["contract"]],
}


class AlignmentRunRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pipeline: Literal["vcp_unified", "vcp_joint", "vcp_default"] = "vcp_unified"
    scope: Literal["full", "methods", "contract"] = "full"
    strict: bool = True


def resolve_cli() -> Path | None:
    for candidate in CLI_CANDIDATES:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_spec(pipeline: str) -> dict[str, str]:
    return PIPELINE_SPECS[pipeline]


def _junit_file(classname: str, file_attr: str | None) -> str:
    if file_attr and file_attr.endswith(".py"):
        return file_attr.lstrip("./")
    path = classname.replace(".", "/")
    return path if path.endswith(".py") else f"{path}.py"


def parse_junit(path: Path) -> dict[str, Any]:
    """Parse pytest JUnit output without inferring counts from human text."""
    summary: dict[str, Any] = {
        "cases": [],
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "parse_error": None,
    }
    if not path.is_file():
        summary["parse_error"] = "pytest 未生成 JUnit 结果文件"
        return summary
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        summary["parse_error"] = f"JUnit 结果不可解析: {error}"
        return summary

    for case in root.iter("testcase"):
        failure = case.find("failure")
        error_node = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            status = "failed"
            summary["failed_count"] += 1
        elif error_node is not None:
            status = "error"
            summary["error_count"] += 1
        elif skipped is not None:
            status = "skipped"
            summary["skipped_count"] += 1
        else:
            status = "passed"
            summary["passed_count"] += 1
        try:
            duration_s = float(case.attrib.get("time", "0"))
        except ValueError:
            duration_s = 0.0
        summary["cases"].append(
            {
                "name": case.attrib.get("name", ""),
                "classname": case.attrib.get("classname", ""),
                "file": _junit_file(
                    case.attrib.get("classname", ""), case.attrib.get("file")
                ),
                "status": status,
                "duration_ms": round(duration_s * 1000, 3),
            }
        )
    return summary


def _group_cases(summary: dict[str, Any], file_path: str) -> list[dict[str, Any]]:
    expected = file_path.lstrip("./")
    return [case for case in summary["cases"] if case["file"].lstrip("./") == expected]


def group_status(
    summary: dict[str, Any], file_path: str, *, strict: bool
) -> Literal["passed", "failed", "skipped"]:
    cases = _group_cases(summary, file_path)
    if not cases:
        return "failed"
    if any(case["status"] in {"failed", "error"} for case in cases):
        return "failed"
    if any(case["status"] == "skipped" for case in cases):
        return "failed" if strict else "skipped"
    return "passed"


def group_duration_ms(summary: dict[str, Any], file_path: str) -> int:
    return round(sum(case["duration_ms"] for case in _group_cases(summary, file_path)))


def compute_pytest_ok(
    returncode: int,
    summary: dict[str, Any],
    selected_files: list[str],
    *,
    strict: bool,
) -> bool:
    if returncode != 0 or summary["parse_error"] or not summary["cases"]:
        return False
    return all(
        group_status(summary, file_path, strict=strict) in {"passed", "skipped"}
        for file_path in selected_files
    )


def compute_overall_success(
    *, pytest_ok: bool, pipeline_ok: bool, strict: bool, skipped_count: int
) -> bool:
    """Pipeline failures can never be masked by unrelated passing unit tests."""
    return bool(
        pytest_ok
        and pipeline_ok
        and (not strict or skipped_count == 0)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_pipeline_artifacts(
    report_path: Path,
    output_path: Path,
    expected_profile: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not report_path.is_file():
        return None, [f"CLI 未生成报告: {report_path.name}"]
    try:
        data = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"CLI 报告不可解析: {error}"]
    if not isinstance(data, dict):
        return None, ["CLI 报告根节点不是对象"]

    actual_profile = data.get("processing_profile")
    if actual_profile != expected_profile:
        errors.append(
            f"Profile 不一致: 期望 {expected_profile}, 实际 {actual_profile!r}"
        )
    if not _finite_number(data.get("process_rtf")) or data["process_rtf"] < 0:
        errors.append("process_rtf 缺失或非法")
    if not _finite_number(data.get("dsp_delay_ms")) or data["dsp_delay_ms"] < 0:
        errors.append("dsp_delay_ms 缺失或非法")
    if not isinstance(data.get("aec_backend"), str) or not data["aec_backend"]:
        errors.append("aec_backend 缺失")
    if not isinstance(data.get("ns_backend"), str) or not data["ns_backend"]:
        errors.append("ns_backend 缺失")
    if not isinstance(data.get("tx_drc_enabled"), bool):
        errors.append("tx_drc_enabled 缺失或非法")

    if expected_profile == "vcp_unified":
        weights = data.get("unified_ns_weights")
        if (
            not isinstance(weights, list)
            or len(weights) != 4
            or not all(_finite_number(weight) and weight >= 0 for weight in weights)
            or sum(weights) <= 0
        ):
            errors.append("Unified NS 权重缺失或非法")

    if not output_path.is_file():
        errors.append(f"CLI 未生成输出音频: {output_path.name}")
    else:
        header = output_path.read_bytes()[:12]
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            errors.append(f"输出音频不是有效 WAV: {output_path.name}")
    return data, errors


def _stage(
    identifier: int,
    name: str,
    detail: str,
    status: Literal["passed", "failed", "skipped"],
    duration_ms: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "name": name,
        "detail": detail,
        "status": status,
        "duration_ms": duration_ms,
    }


def _pipeline_payload(
    request: AlignmentRunRequest,
    *,
    status: Literal["passed", "failed"],
    data: dict[str, Any] | None,
    returncode: int | None,
) -> dict[str, Any]:
    spec = _pipeline_spec(request.pipeline)
    payload = data or {}
    return {
        "id": request.pipeline,
        "cli_profile": spec["cli_profile"],
        "name": spec["name"],
        "topology": spec["topology"],
        "profile": payload.get("processing_profile"),
        "dsp_delay_ms": payload.get("dsp_delay_ms"),
        "rtf": payload.get("process_rtf"),
        "aec_backend": payload.get("aec_backend"),
        "ns_backend": payload.get("ns_backend"),
        "weights": payload.get("unified_ns_weights", []),
        "tx_drc": payload.get("tx_drc_enabled"),
        "tx_drc_config": payload.get("tx_drc_config"),
        "standard_compliance": False,
        "status": status,
        "returncode": returncode,
    }


def run_alignment_suite(request: AlignmentRunRequest) -> dict[str, Any]:
    started = time.monotonic()
    spec = _pipeline_spec(request.pipeline)
    selected_files = SCOPE_TEST_FILES[request.scope]
    pipeline_required = request.scope == "full"
    failure_reasons: list[str] = []
    stages: list[dict[str, Any]] = []
    log: list[str] = [
        "=" * 72,
        "ITU Audio engineering self-check",
        f"scope={request.scope} pipeline={request.pipeline} strict={request.strict}",
        "formal standard compliance: NOT MEASURED",
        "=" * 72,
    ]

    with tempfile.TemporaryDirectory(prefix="itu-alignment-") as temp_dir:
        temp = Path(temp_dir)
        output_prefix = temp / "result"

        # Stage 1: validate exactly what this scope claims to execute.
        stage_started = time.monotonic()
        cli = resolve_cli()
        required_inputs = [DEFAULT_MIC, DEFAULT_REF] if pipeline_required else []
        test_paths = [PROJECT_ROOT / file_path for file_path in selected_files]
        missing = [str(path) for path in [*required_inputs, *test_paths] if not path.is_file()]
        if pipeline_required and cli is None:
            missing.append("itu_audio_cli")
        stage1_ok = not missing
        stage1_detail = (
            f"{cli.name if cli else 'CLI 未找到'} · "
            f"{len(required_inputs)} 个音频输入 · {len(test_paths)} 个测试文件"
        )
        if missing:
            stage1_detail += f" · 缺失: {', '.join(missing)}"
            failure_reasons.append(f"执行前置检查失败: {', '.join(missing)}")
        stages.append(
            _stage(
                1,
                "执行范围、CLI 与测试文件校验",
                stage1_detail,
                "passed" if stage1_ok else "failed",
                round((time.monotonic() - stage_started) * 1000),
            )
        )
        log.extend(
            [
                "",
                "[1/4] 执行范围与输入校验",
                f"pipeline_required={pipeline_required}",
                f"cli={cli or 'MISSING'}",
                f"test_files={', '.join(selected_files)}",
            ]
        )
        if cli:
            log.append(f"cli_sha256={sha256_file(cli)}")
        for path in required_inputs:
            if path.is_file():
                log.append(f"input={path} sha256={sha256_file(path)}")

        # Stage 2: run the pinned CLI only when the selected scope claims audio processing.
        pipeline_data: dict[str, Any] | None = None
        pipeline_payload: dict[str, Any] | None = None
        pipeline_ok = not pipeline_required
        stage2_started = time.monotonic()
        if not pipeline_required:
            stages.append(
                _stage(
                    2,
                    "VCP 真实音频处理",
                    "当前范围只运行 pytest，不调用音频 CLI",
                    "skipped",
                    0,
                )
            )
            log.append("[2/4] VCP 音频处理: SKIPPED（当前 scope 不运行 CLI）")
        elif not stage1_ok or cli is None:
            stages.append(
                _stage(
                    2,
                    "VCP 真实音频处理",
                    "前置检查失败，未启动 CLI",
                    "failed",
                    round((time.monotonic() - stage2_started) * 1000),
                )
            )
            pipeline_payload = _pipeline_payload(
                request, status="failed", data=None, returncode=None
            )
            log.append("[2/4] VCP 音频处理: FAILED（前置检查失败）")
        else:
            cli_cmd = [
                str(cli),
                "--profile",
                spec["cli_profile"],
                "--mic",
                str(DEFAULT_MIC),
                "--echo",
                str(DEFAULT_REF),
                "--beam-delays",
                "1.440314751,0.000000000",
                "--out",
                str(output_prefix),
            ]
            log.extend(
                [
                    "",
                    "[2/4] VCP 真实音频处理",
                    f"command={' '.join(cli_cmd)}",
                ]
            )
            cli_returncode: int | None = None
            try:
                cli_proc = subprocess.run(
                    cli_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                cli_returncode = cli_proc.returncode
                if cli_proc.stdout.strip():
                    log.append(f"cli_stdout:\n{cli_proc.stdout.strip()}")
                if cli_proc.stderr.strip():
                    log.append(f"cli_stderr:\n{cli_proc.stderr.strip()}")
                pipeline_data, pipeline_errors = validate_pipeline_artifacts(
                    Path(f"{output_prefix}.json"),
                    Path(f"{output_prefix}.after.wav"),
                    spec["cli_profile"],
                )
                if cli_proc.returncode != 0:
                    pipeline_errors.insert(0, f"CLI 退出码 {cli_proc.returncode}")
                pipeline_ok = not pipeline_errors
                if pipeline_errors:
                    failure_reasons.extend(pipeline_errors)
                    stage2_detail = "；".join(pipeline_errors)
                else:
                    stage2_detail = (
                        f"{pipeline_data['processing_profile']} · "
                        f"RTF {pipeline_data['process_rtf']:.4f} · "
                        f"延迟 {pipeline_data['dsp_delay_ms']:.3f} ms · "
                        f"输出 WAV 校验通过"
                    )
                pipeline_payload = _pipeline_payload(
                    request,
                    status="passed" if pipeline_ok else "failed",
                    data=pipeline_data,
                    returncode=cli_returncode,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                failure_reasons.append(f"CLI 执行异常: {error}")
                pipeline_ok = False
                stage2_detail = str(error)
                pipeline_payload = _pipeline_payload(
                    request, status="failed", data=None, returncode=cli_returncode
                )
                log.append(f"cli_error={error}")
            stages.append(
                _stage(
                    2,
                    "VCP 真实音频处理",
                    stage2_detail,
                    "passed" if pipeline_ok else "failed",
                    round((time.monotonic() - stage2_started) * 1000),
                )
            )

        # Stage 3-5: run pytest once and map real JUnit results into named gates.
        pytest_started = time.monotonic()
        junit_path = temp / "pytest.xml"
        pytest_cmd = [
            PYTHON_EXEC,
            "-m",
            "pytest",
            *selected_files,
            "-v",
            "--tb=short",
            f"--junitxml={junit_path}",
        ]
        pytest_stdout = ""
        pytest_stderr = ""
        pytest_returncode = 1
        try:
            pytest_proc = subprocess.run(
                pytest_cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": "."},
                timeout=300,
            )
            pytest_returncode = pytest_proc.returncode
            pytest_stdout = pytest_proc.stdout or ""
            pytest_stderr = pytest_proc.stderr or ""
        except (OSError, subprocess.TimeoutExpired) as error:
            failure_reasons.append(f"pytest 执行异常: {error}")
            pytest_stderr = str(error)

        summary = parse_junit(junit_path)
        if summary["parse_error"]:
            failure_reasons.append(summary["parse_error"])
        if pytest_returncode != 0:
            failure_reasons.append(f"pytest 退出码 {pytest_returncode}")
        if summary["failed_count"]:
            failure_reasons.append(f"{summary['failed_count']} 个 pytest 用例失败")
        if summary["error_count"]:
            failure_reasons.append(f"{summary['error_count']} 个 pytest 用例错误")
        if request.strict and summary["skipped_count"]:
            failure_reasons.append(
                f"严格模式不接受跳过项（{summary['skipped_count']} 个）"
            )

        group_definitions = [
            (
                3,
                "方法指标实现自检",
                TEST_FILE_GROUPS["methods"],
                "双讲、CNG、连续性等指标函数的正负例",
            ),
            (
                4,
                "声学证据与门禁实现自检",
                TEST_FILE_GROUPS["acoustic"],
                "证据完整性、篡改阻断与仿真不得升格",
            ),
            (
                5,
                "契约台账与防遮掩实现自检",
                TEST_FILE_GROUPS["contract"],
                "台账、发布门禁与失败不得被退出码 0 遮掩",
            ),
        ]
        for identifier, name, file_path, description in group_definitions:
            if file_path not in selected_files:
                stages.append(
                    _stage(
                        identifier,
                        name,
                        "当前范围未选择此测试组",
                        "skipped",
                        0,
                    )
                )
                continue
            status = group_status(summary, file_path, strict=request.strict)
            cases = _group_cases(summary, file_path)
            failed = sum(case["status"] in {"failed", "error"} for case in cases)
            skipped = sum(case["status"] == "skipped" for case in cases)
            detail = f"{description} · {len(cases)} 项"
            if failed:
                detail += f" · {failed} 项失败"
            if skipped:
                detail += f" · {skipped} 项跳过"
            stages.append(
                _stage(
                    identifier,
                    name,
                    detail,
                    status,
                    group_duration_ms(summary, file_path),
                )
            )

        pytest_ok = compute_pytest_ok(
            pytest_returncode,
            summary,
            selected_files,
            strict=request.strict,
        )
        success = compute_overall_success(
            pytest_ok=pytest_ok,
            pipeline_ok=pipeline_ok,
            strict=request.strict,
            skipped_count=summary["skipped_count"],
        )

        log.extend(
            [
                "",
                "[3/4] pytest 执行",
                f"command={' '.join(pytest_cmd)}",
                pytest_stdout.strip(),
            ]
        )
        if pytest_stderr.strip():
            log.append(f"pytest_stderr:\n{pytest_stderr.strip()}")
        log.extend(
            [
                "",
                "[4/4] 结果汇总",
                (
                    f"passed={summary['passed_count']} failed={summary['failed_count']} "
                    f"errors={summary['error_count']} skipped={summary['skipped_count']}"
                ),
                f"pipeline_ok={pipeline_ok} pytest_ok={pytest_ok}",
                "formal_standard_compliance=NOT_MEASURED",
                f"engineering_self_check={'PASS' if success else 'FAIL'}",
            ]
        )
        if failure_reasons:
            log.append("failure_reasons=" + " | ".join(failure_reasons))

        total_count = (
            summary["passed_count"]
            + summary["failed_count"]
            + summary["error_count"]
            + summary["skipped_count"]
        )
        duration_s = round(time.monotonic() - started, 3)
        return {
            "success": success,
            "engineering_success": success,
            "standard_compliance": False,
            "formal_test_status": "not_measured",
            "returncode": pytest_returncode,
            "duration_s": duration_s,
            "total_count": total_count,
            "passed_count": summary["passed_count"],
            "failed_count": summary["failed_count"],
            "error_count": summary["error_count"],
            "skipped_count": summary["skipped_count"],
            "pass_rate": (
                round(summary["passed_count"] / total_count * 100, 1)
                if total_count
                else 0.0
            ),
            "failure_reasons": list(dict.fromkeys(failure_reasons)),
            "stages": stages,
            "log": "\n".join(log),
            "scope": request.scope,
            "strict": request.strict,
            "test_groups": {
                group: {
                    "file": file_path,
                    "selected": file_path in selected_files,
                    "status": (
                        group_status(summary, file_path, strict=request.strict)
                        if file_path in selected_files
                        else "skipped"
                    ),
                    "count": len(_group_cases(summary, file_path)),
                }
                for group, file_path in TEST_FILE_GROUPS.items()
            },
            "pipeline": pipeline_payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@router.get("/status")
def get_runner_status() -> dict[str, Any]:
    cli = resolve_cli()
    return {
        "status": "ready" if cli is not None else "degraded",
        "engine": "ITU-T P.1100/P.1110/P.1120 engineering self-check",
        "formal_compliance": False,
        "active_pipeline": {
            "id": "vcp_unified",
            "name": PIPELINE_SPECS["vcp_unified"]["name"],
            "desc": PIPELINE_SPECS["vcp_unified"]["topology"],
            "cli_available": cli is not None,
            "cli_path": str(cli) if cli else None,
        },
        "supported_pipelines": [
            {
                "id": pipeline,
                "cli_profile": spec["cli_profile"],
                "label": f"{spec['name']} (CLI profile: {spec['cli_profile']})",
                "recommended": pipeline == "vcp_unified",
            }
            for pipeline, spec in PIPELINE_SPECS.items()
        ],
        "supported_scopes": [
            {
                "id": "full",
                "label": "工程全检（真实 CLI 单场景 + 三组 pytest；不执行 8 模式或 336 项正式实测）",
                "recommended": True,
            },
            {
                "id": "methods",
                "label": "方法指标自检（仅 pytest，不运行音频 CLI）",
                "recommended": False,
            },
            {
                "id": "contract",
                "label": "契约与防遮掩自检（仅 pytest，不运行音频 CLI）",
                "recommended": False,
            },
        ],
        "default_scope": "full",
    }


@router.post("/run")
async def execute_alignment_run(
    request: AlignmentRunRequest = Body(default=AlignmentRunRequest()),
) -> dict[str, Any]:
    return await run_in_threadpool(run_alignment_suite, request)
