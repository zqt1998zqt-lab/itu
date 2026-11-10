"""Persisted, serial VCP execution of condition-bound NV200 dataset cases."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import threading
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import service
from .acceptance_contract import required_conditions
from .acceptance_dataset import dataset_asset, get_dataset
from .acceptance_models import calculate_case_items
from .acceptance_scope import (
    BACKGROUND_NOISE_SCENES,
    BackgroundNoiseScene,
    TEST_SCOPES,
    select_batch_scope,
)
from .receive_configuration import normalize_receive_configuration, receive_configuration_sha256

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nv200-acceptance")
_LOCK = threading.RLock()
_ACTIVE: set[str] = set()
_LEGACY_ACTIVE = False
_TERMINAL = {"completed", "failed", "cancelled"}
_BATCH_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}
_PROGRESS_COUNTERS = (
    "completed_cases",
    "total_cases",
    "completed_items",
    "total_items",
)
_SUMMARY_COUNTERS = (
    "total",
    "computed",
    "pass",
    "fail",
    "review_required",
    "not_measured",
    "invalid",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory(identifier: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("非法批次 ID")
    return service.BUILD / "acceptance-batches" / identifier


def _summary(items: list[dict]) -> dict:
    result = {status: sum(row["status"] == status for row in items)
              for status in ("pass", "fail", "review_required", "not_measured", "invalid")}
    result.update(total=len(items), computed=sum(
        isinstance(row.get("value"), (int, float)) and not isinstance(row.get("value"), bool)
        and math.isfinite(row["value"]) and row["status"] not in {"pending", "running", "invalid", "not_measured"}
        for row in items))
    return result


def _save(snapshot: dict) -> None:
    with _LOCK:
        snapshot["summary"] = _summary(snapshot["items"])
        mode_reports = []
        completed_cases = {case["case_id"] for case in snapshot["cases"] if case["status"] == "completed"}
        if snapshot.get("test_scope", "full") == "full":
            for mode in dict.fromkeys(row["mode"] for row in snapshot["items"]):
                rows = [row for row in snapshot["items"] if row["mode"] == mode]
                if len(rows) != 42 or any(row["status"] in {"pending", "running"} for row in rows):
                    continue
                if any(row["evidence_binding"]["case_id"] not in completed_cases for row in rows):
                    continue
                summary = _summary(rows)
                mode_reports.append({"mode": mode, "evaluation_id": f"batch-{snapshot['batch_id']}-{mode}",
                                     "summary": {"total": 42, "measured": summary["computed"],
                                                 **{status: summary[status] for status in ("pass", "fail", "review_required", "not_measured", "invalid")}}})
        snapshot["mode_reports"] = mode_reports
        snapshot["updated_at"] = _now()
        path = _directory(snapshot["batch_id"]) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def get_batch(identifier: str) -> dict | None:
    path = _directory(identifier) / "report.json"
    with _LOCK:
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        if result["status"] not in _TERMINAL and identifier not in _ACTIVE:
            result["status"] = "failed"
            result["error"] = "服务重启中断了该批次，请重新执行；已完成证据仍保留"
            for row in result["items"]:
                if row["status"] in {"pending", "running"}:
                    row.update(status="invalid", reason=result["error"], value=None)
            _save(result)
        return result


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _counters(value: object, fields: tuple[str, ...]) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result = {}
    for field in fields:
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        result[field] = count
    return result


def list_batch_summaries(
    limit: int = 20,
    algorithm_cluster_id: str | None = None,
    *,
    test_scope: str = "full",
    scene: BackgroundNoiseScene | None = None,
) -> list[dict]:
    """List lightweight persisted batch summaries without rewriting reports."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("limit 必须为 1 到 100 的整数")
    if test_scope not in TEST_SCOPES:
        raise ValueError("未知测试范围")
    if scene is not None:
        if test_scope != "background_noise":
            raise ValueError("只有背景噪声测试可以按场景筛选")
        if scene not in BACKGROUND_NOISE_SCENES:
            raise ValueError("未知背景噪声场景")
    root = service.BUILD / "acceptance-batches"
    try:
        directories = list(root.iterdir())
    except FileNotFoundError:
        return []

    rows: list[tuple[datetime, str, dict]] = []
    for directory in directories:
        if not re.fullmatch(r"[a-f0-9]{32}", directory.name):
            continue
        try:
            directory_stat = directory.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(directory_stat.st_mode):
            continue

        report_path = directory / "report.json"
        try:
            report_stat = report_path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(report_stat.st_mode):
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (UnicodeError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        if report.get("test_scope", "full") != test_scope:
            continue
        if test_scope == "background_noise":
            if scene is None and report.get("selected_scene") is not None:
                continue
            if scene is not None and report.get("selected_scene") != scene:
                continue

        required_strings = (
            "batch_id",
            "dataset_id",
            "dataset_name",
            "algorithm_cluster_id",
            "status",
        )
        if any(
            not isinstance(report.get(field), str) or not report[field]
            for field in required_strings
        ):
            continue
        if report["batch_id"] != directory.name:
            continue
        if not re.fullmatch(r"[a-f0-9]{32}", report["dataset_id"]):
            continue
        if report["status"] not in _BATCH_STATUSES:
            continue
        progress = _counters(report.get("progress"), _PROGRESS_COUNTERS)
        summary_counts = _counters(report.get("summary"), _SUMMARY_COUNTERS)
        if progress is None or summary_counts is None:
            continue
        if (
            progress["completed_cases"] > progress["total_cases"]
            or progress["completed_items"] > progress["total_items"]
            or progress["total_items"] != summary_counts["total"]
            or summary_counts["computed"] > summary_counts["total"]
            or any(
                summary_counts[field] > summary_counts["total"]
                for field in _SUMMARY_COUNTERS[2:]
            )
        ):
            continue
        created_at = _timestamp(report.get("created_at"))
        if created_at is None:
            continue
        if (
            algorithm_cluster_id is not None
            and report["algorithm_cluster_id"] != algorithm_cluster_id
        ):
            continue

        summary = {
            field: report[field]
            for field in required_strings
        }
        summary.update(
            status=report["status"],
            progress=progress,
            summary=summary_counts,
            created_at=report["created_at"],
        )
        if test_scope != "full":
            summary.update(
                test_scope=test_scope,
                selected_mode=report.get("selected_mode"),
                selected_scene=report.get("selected_scene"),
            )
        invalid_optional = False
        for field in ("finished_at", "updated_at"):
            value = report.get(field)
            if value is None:
                continue
            if _timestamp(value) is None:
                invalid_optional = True
                break
            summary[field] = value
        if invalid_optional:
            continue
        error = report.get("error")
        if error is not None:
            if not isinstance(error, str):
                continue
            summary["error"] = error
        rows.append((created_at, directory.name, summary))

    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [summary for _, _, summary in rows[:limit]]


def _run_cli(case: dict, paths: dict[str, Path], destination: Path, cluster: str) -> tuple[dict, dict]:
    # Import lazily: alignment_runner also imports the legacy five-signal module.
    from server.alignment_runner import resolve_cli, resolve_algorithm_cluster, validate_pipeline_artifacts
    cli = resolve_cli()
    if cli is None:
        raise ValueError("未找到可执行的 VCP CLI")
    preset, flags, configuration, catalog_sha = resolve_algorithm_cluster(cluster)
    profile = preset["profile"]
    cli_hash = _sha(cli)
    outputs: dict[str, Path] = {}
    commands = []
    reports = {}
    prepared_paths = {}
    preprocessing = {}
    beam_delay_adaptation = None
    for role, path in paths.items():
        info = sf.info(path)
        if info.samplerate in {16000, 24000, 48000}:
            prepared_paths[role] = path
            continue
        mode = case["conditions"].get("mode", "")
        if info.samplerate != 8000 or not mode.startswith("NB-"):
            raise ValueError(f"{mode} 输入 {role} 的 {info.samplerate} Hz 不受 VCP 支持；仅 NB 8 kHz 可显式适配到 16 kHz")
        # NB recordings remain NB evidence. The engine's supported PCM rate is
        # an explicit adapter, not a change to the declared business mode.
        samples, source_rate = sf.read(path, always_2d=True)
        divisor = math.gcd(source_rate, 16000)
        adapted = resample_poly(samples, 16000 // divisor, source_rate // divisor, axis=0)
        prepared = destination / f"prepared-{role}.wav"
        sf.write(prepared, adapted, 16000, subtype="PCM_16")
        prepared_paths[role] = prepared
        preprocessing[role] = {"method": "scipy.signal.resample_poly", "source_rate": source_rate,
                               "engine_rate": 16000, "source_sha256": _sha(path), "prepared_sha256": _sha(prepared)}
    output_processing = {}
    direction = case["conditions"]["direction"]
    for kind in (["rx"] if direction == "receive" else ["tx", "rx"] if direction == "round_trip" else ["tx"]):
        prefix = destination / kind
        command = [str(cli), "--profile", profile, *flags]
        if kind == "rx":
            command += ["--rx-in", str(prepared_paths["far_end"]), "--rx-processing"]
            output = Path(f"{prefix}.rx-spk.wav")
        else:
            command += ["--mic", str(prepared_paths["microphone"]), "--echo", str(prepared_paths["far_end"])]
            microphone_info = sf.info(prepared_paths["microphone"])
            channels = microphone_info.channels
            delays = case.get("beam_delays_ms")
            if channels == 1:
                command += ["--no-bf"]
            elif isinstance(delays, list) and len(delays) == channels and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
                for value in delays
            ):
                # The manifest records physical time in ms; the CLI expects
                # samples at the actual engine rate, including NB adaptation.
                delay_samples = [float(value) * microphone_info.samplerate / 1000.0 for value in delays]
                if not all(math.isfinite(value) for value in delay_samples):
                    raise ValueError("beam_delays_ms 转换后的样本延时不是有限数")
                command += ["--beam-delays", ",".join(str(value) for value in delay_samples)]
                beam_delay_adaptation = {"source_unit": "ms", "engine_rate": microphone_info.samplerate,
                                         "delays_ms": list(delays), "delays_samples": delay_samples}
            else:
                raise ValueError("多麦克输入必须提供每通道的 beam_delays_ms 校准值")
            output = Path(f"{prefix}.after.wav")
        command += ["--out", str(prefix)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (destination / f"{kind}.log").write_text(process.stdout + "\n" + process.stderr)
        if process.returncode:
            raise ValueError(f"VCP {kind} 退出码 {process.returncode}: {process.stderr[-1500:]}")
        report_path = Path(f"{prefix}.json")
        if kind == "tx":
            report, errors = validate_pipeline_artifacts(report_path, output, profile)
            if errors:
                raise ValueError("; ".join(errors))
        else:
            report = json.loads(report_path.read_text())
            if report.get("processing_profile") != profile or report.get("rx_processing") is not True or report.get("rx_processing_frames", 0) <= 0:
                raise ValueError("下行报告未证实 VCP RX 实际执行")
        data, rate = sf.read(output, always_2d=True)
        source = sf.info(prepared_paths["far_end"] if kind == "rx" else prepared_paths["microphone"])
        delay = report.get("dsp_delay_samples", 0) if kind == "tx" else 0
        if not isinstance(delay, int) or isinstance(delay, bool) or not 0 <= delay <= rate:
            raise ValueError("VCP 报告固定延时非法")
        if not data.size or not np.isfinite(data).all() or data.shape[1] != 1 or rate != source.samplerate or len(data) != source.frames + delay:
            raise ValueError("VCP 输出 WAV 的采样率、样本数、通道或数值非法")
        if delay:
            raw_output = output
            output = destination / f"{kind}.analysis.wav"
            sf.write(output, data[delay:delay + source.frames], rate, subtype="FLOAT")
            output_processing[kind] = {"method": "declared_fixed_delay_trim", "trimmed_leading_samples": delay,
                                       "raw_filename": raw_output.name, "raw_sha256": _sha(raw_output),
                                       "analysis_filename": output.name, "analysis_sha256": _sha(output)}
        outputs[kind] = output
        commands.append(command)
        reports[kind] = {"path": str(report_path), "sha256": _sha(report_path), "data": report}
    if _sha(cli) != cli_hash:
        raise ValueError("执行期间 VCP CLI 被替换，本次证据无效")
    return outputs, {
        "source": "alignment_batch_cli", "algorithm_cluster_id": cluster,
        "configuration": configuration, "catalog_sha256": catalog_sha,
        "cli_sha256": cli_hash, "commands": commands, "pipeline_reports": reports,
        "input_adaptation": preprocessing, "output_alignment": output_processing,
        "beam_delay_adaptation": beam_delay_adaptation,
        "input_sha256": {role: _sha(path) for role, path in paths.items()},
        "output_sha256": {role: _sha(path) for role, path in outputs.items()},
    }


def run_batch(identifier: str) -> dict:
    with _LOCK:
        _ACTIVE.add(identifier)
    snapshot = None
    try:
        snapshot = get_batch(identifier)
        if snapshot is None:
            raise ValueError("批次不存在")
        dataset = get_dataset(snapshot["dataset_id"])
        if dataset is None:
            raise ValueError("测试集不存在")
        manifest_digest = hashlib.sha256(json.dumps(
            dataset["manifest"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if manifest_digest != snapshot.get("dataset_manifest_sha256"):
            raise ValueError("批次开始后测试集清单身份发生变化")
        _, execution_cases = select_batch_scope(
            dataset["manifest"], test_scope=snapshot.get("test_scope", "full"),
            mode=snapshot.get("selected_mode"),
            scene=snapshot.get("selected_scene"),
        )
        expected_cases = {case["id"] for case in execution_cases}
        expected_items = [item_id for case in execution_cases for item_id in case["item_ids"]]
        frozen_items = {row["item_id"]: row["evidence_binding"]["case_id"] for row in snapshot["items"]}
        if (
            len(expected_cases) != snapshot["progress"]["total_cases"]
            or len(expected_items) != snapshot["progress"]["total_items"]
            or len(expected_items) != len(set(expected_items))
            or len(frozen_items) != len(snapshot["items"])
            or len(frozen_items) != len(expected_items)
            or any(frozen_items.get(item_id) != case["id"]
                   for case in execution_cases for item_id in case["item_ids"])
        ):
            raise ValueError("批次条件组或测项身份不完整")
        receive_config = normalize_receive_configuration(
            dataset.get("receive_simulation")
        )
        receive_ir_sha256: dict[str, str] = {}
        if receive_config["model"] == "paired_ir":
            for role in ("left_ir", "right_ir"):
                relative = receive_config[role]
                asset = dataset["assets"].get(relative)
                if not isinstance(asset, dict):
                    raise ValueError(f"测试集缺少 {role} IR 资产")
                receive_ir_sha256[role] = asset["sha256"]
        acoustic_digest = receive_configuration_sha256(receive_config, receive_ir_sha256)
        acoustic_fields = {
            "receive_simulation": receive_config,
            "receive_ir_sha256": receive_ir_sha256,
            "receive_configuration_sha256": acoustic_digest,
        }
        for key, value in acoustic_fields.items():
            if key not in snapshot or snapshot[key] != value:
                raise ValueError("批次开始后接收声学配置身份发生变化")
        for row in snapshot["items"]:
            binding = row["evidence_binding"]
            for key in ("receive_ir_sha256", "receive_configuration_sha256"):
                if key not in binding or binding[key] != acoustic_fields[key]:
                    raise ValueError("测项接收声学配置身份不一致")
                binding[key] = acoustic_fields[key]
        snapshot.update(acoustic_fields)
        snapshot["status"] = "running"
        _save(snapshot)
        by_id = {row["item_id"]: index for index, row in enumerate(snapshot["items"])}
        for case in execution_cases:
            for item_id in case["item_ids"]:
                snapshot["items"][by_id[item_id]]["status"] = "running"
            snapshot["current_case"] = case["id"]
            _save(snapshot)
            report_id = uuid.uuid4().hex
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case["id"]):
                raise ValueError("非法条件组 ID")
            cases_root = _directory(identifier) / "cases"
            if cases_root.is_symlink() or _directory(identifier).is_symlink():
                raise ValueError("批次目录不能为符号链接")
            destination = cases_root / case["id"]
            if destination.is_symlink() or not destination.resolve().is_relative_to(cases_root.resolve()):
                raise ValueError("条件组目录越界")
            destination.mkdir(parents=True, exist_ok=False)
            try:
                paths = {role: dataset_asset(snapshot["dataset_id"], name) for role, name in case["signals"].items()}
                for role, path in paths.items():
                    if _sha(path) != dataset["assets"][case["signals"][role]]["sha256"]:
                        raise ValueError(f"数据集 {role} 文件身份已改变")
                receive_ir_paths = {
                    role: dataset_asset(snapshot["dataset_id"], receive_config[role])
                    for role in ("left_ir", "right_ir")
                } if receive_config["model"] == "paired_ir" else {}
                for role, path in receive_ir_paths.items():
                    if _sha(path) != receive_ir_sha256[role]:
                        raise ValueError(f"数据集 {role} IR 文件身份已改变")
                outputs, execution = _run_cli(case, paths, destination, snapshot["algorithm_cluster_id"])
                execution.update(
                    batch_id=identifier,
                    dataset_id=snapshot["dataset_id"],
                    receive_simulation=receive_config,
                    receive_ir_sha256=receive_ir_sha256,
                    receive_configuration_sha256=acoustic_digest,
                    synthetic_proxy=True,
                    standard_compliance=False,
                    formal_hats_evidence=False,
                )
                if snapshot.get("cli_sha256") and execution["cli_sha256"] != snapshot["cli_sha256"]:
                    raise ValueError("批次开始后 VCP 二进制发生变化")
                if execution.get("catalog_sha256") != snapshot.get("catalog_sha256"):
                    raise ValueError("批次开始后算法簇配置发生变化")
                for role, digest in execution["input_sha256"].items():
                    if digest != dataset["assets"][case["signals"][role]]["sha256"]:
                        raise ValueError("执行期间数据集输入身份发生变化")
                calculate_kwargs = {"report_id": report_id}
                if "receive_simulation" in dataset["manifest"]:
                    calculate_kwargs.update(
                        receive_config=receive_config,
                        receive_ir_paths=receive_ir_paths,
                    )
                rows = calculate_case_items(
                    case,
                    paths,
                    outputs,
                    execution,
                    **calculate_kwargs,
                )
                if any(_sha(path) != execution["input_sha256"].get(role) for role, path in paths.items()):
                    raise ValueError("计算期间输入文件身份发生变化")
                if any(_sha(path) != execution["output_sha256"].get(role) for role, path in outputs.items()):
                    raise ValueError("计算期间 VCP 输出身份发生变化")
                if any(_sha(path) != receive_ir_sha256[role] for role, path in receive_ir_paths.items()):
                    raise ValueError("计算期间 IR 文件身份发生变化")
                if {row["item_id"] for row in rows} != set(case["item_ids"]) or len(rows) != len(case["item_ids"]):
                    raise ValueError("计算模型返回的合同条目身份不一致")
                for row in rows:
                    row["mode"] = case["conditions"]["mode"]
                    row["standard_compliance"] = False
                    row["evidence_binding"] = {
                        **case["conditions"], "case_id": case["id"], "report_id": report_id,
                        "batch_id": identifier, "dataset_id": snapshot["dataset_id"],
                        "input_sha256": execution["input_sha256"], "output_sha256": execution["output_sha256"],
                        "receive_configuration_sha256": acoustic_digest,
                        "receive_ir_sha256": receive_ir_sha256,
                    }
                    value = row.get("value")
                    if value is None:
                        state = "invalid" if row.get("status") == "invalid" else "not_measured"
                        row.update(status=state, evidence_grade="bound" if row.get("bound") and state == "not_measured" else state)
                    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                        row.update(value=None, status="invalid", reason="计算值不是有限数", evidence_grade="invalid")
                    else:
                        row["status"] = "review_required"
                    snapshot["items"][by_id[row["item_id"]]].update(row)
                case_report = {"report_id": report_id, "case_id": case["id"], "execution": execution, "items": rows}
                (destination / "assessment.json").write_text(json.dumps(case_report, ensure_ascii=False, indent=2, allow_nan=False))
                snapshot["cases"].append({"case_id": case["id"], "report_id": report_id,
                                          "conditions": case["conditions"], "status": "completed",
                                          "assessment_sha256": _sha(destination / "assessment.json"), "execution": execution})
            except Exception as error:
                if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                    raise
                for item_id in case["item_ids"]:
                    row = snapshot["items"][by_id[item_id]]
                    row.update(status="invalid", value=None, reason=f"执行失败: {type(error).__name__}: {error}", report_id=report_id)
                    row["evidence_binding"]["report_id"] = report_id
                snapshot["cases"].append({"case_id": case["id"], "report_id": report_id,
                                          "conditions": case["conditions"], "status": "failed", "error": str(error)})
            snapshot["progress"]["completed_cases"] += 1
            snapshot["progress"]["completed_items"] += len(case["item_ids"])
            _save(snapshot)
        if (
            snapshot["progress"]["completed_cases"] != snapshot["progress"]["total_cases"]
            or snapshot["progress"]["completed_items"] != snapshot["progress"]["total_items"]
            or len(snapshot["cases"]) != len(expected_cases)
            or {record["case_id"] for record in snapshot["cases"]} != expected_cases
            or any(row["status"] in {"pending", "running"} for row in snapshot["items"])
        ):
            raise ValueError("批次未完整执行，不能标记为完成")
        if snapshot.get("test_scope", "full") in {"full", "basic_downlink"}:
            from .acceptance_linearity import finish_linearity
            finish_linearity(snapshot, dataset, _directory(identifier))
        failed_cases = sum(case["status"] == "failed" for case in snapshot["cases"])
        snapshot["status"] = "failed" if failed_cases else "completed"
        if failed_cases:
            snapshot["error"] = f"{failed_cases} 个条件组执行失败；已完成证据与逐项失败原因均已保留"
        snapshot["finished_at"] = _now()
        _save(snapshot)
    except Exception as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise
        if snapshot is None:
            raise
        snapshot["status"] = "failed"
        snapshot["error"] = f"{type(error).__name__}: {error}"
        for row in snapshot["items"]:
            if row["status"] in {"pending", "running"}:
                row.update(status="invalid", reason=snapshot["error"], value=None)
        _save(snapshot)
    finally:
        with _LOCK:
            _ACTIVE.discard(identifier)
    return snapshot


def start_batch(
    dataset_id: str,
    algorithm_cluster_id: str,
    *,
    test_scope: str = "full",
    mode: str | None = None,
    scene: BackgroundNoiseScene | None = None,
) -> dict:
    from server.alignment_runner import resolve_algorithm_cluster, resolve_cli
    if test_scope not in TEST_SCOPES:
        raise ValueError("未知测试范围")
    preset, _, configuration, catalog_sha = resolve_algorithm_cluster(algorithm_cluster_id)
    if test_scope != "full" and not preset["profile"].startswith("vcp"):
        raise ValueError("子范围测试仅支持 VCP 算法簇")
    cli = resolve_cli()
    if cli is None:
        raise ValueError("未找到可执行的 VCP CLI")
    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise ValueError("测试集不存在")
    selected_rows, cases = select_batch_scope(
        dataset["manifest"],
        test_scope=test_scope,
        mode=mode,
        scene=scene,
    )
    receive_config = normalize_receive_configuration(
        dataset.get("receive_simulation")
    )
    receive_ir_sha256: dict[str, str] = {}
    if receive_config["model"] == "paired_ir":
        for role in ("left_ir", "right_ir"):
            relative = receive_config[role]
            asset = dataset["assets"].get(relative)
            if not isinstance(asset, dict):
                raise ValueError(f"测试集缺少 {role} IR 资产")
            receive_ir_sha256[role] = asset["sha256"]
    acoustic_digest = receive_configuration_sha256(receive_config, receive_ir_sha256)
    identifier = uuid.uuid4().hex
    case_for_item = {item_id: case for case in cases for item_id in case["item_ids"]}
    items = [{**row, "item_id": row["id"], "mode": row["sheet"], "status": "pending", "value": None,
              "reason": "等待对应场景执行", "standard_compliance": False,
              "evidence_binding": {**required_conditions(row), "case_id": case_for_item[row["id"]]["id"],
                                   "batch_id": identifier, "dataset_id": dataset_id, "report_id": None,
                                   "receive_configuration_sha256": acoustic_digest,
                                   "receive_ir_sha256": receive_ir_sha256}}
             for row in selected_rows]
    snapshot = {"schema_version": 1, "batch_id": identifier, "dataset_id": dataset_id,
                "dataset_name": dataset["name"], "dataset_manifest_sha256": dataset["manifest_sha256"],
                "algorithm_cluster_id": algorithm_cluster_id, "status": "queued", "created_at": _now(),
                "test_scope": test_scope, "selected_mode": mode,
                "selected_scene": scene,
                "configuration": configuration, "catalog_sha256": catalog_sha, "cli_sha256": _sha(cli), "cases": [],
                "receive_simulation": receive_config, "receive_ir_sha256": receive_ir_sha256,
                "receive_configuration_sha256": acoustic_digest,
                "scope": "condition-bound VCP engineering assessment; formal ITU compliance not measured",
                "standard_compliance": False, "formal_test_status": "not_measured",
                "progress": {"completed_cases": 0, "total_cases": len(cases), "completed_items": 0, "total_items": len(items)},
                "items": items}
    with _LOCK:
        if _ACTIVE or _LEGACY_ACTIVE:
            raise ValueError("已有测试任务正在执行，请等待完成后再启动")
        _ACTIVE.add(identifier)
        try:
            _save(snapshot)
            _EXECUTOR.submit(run_batch, identifier)
        except Exception:
            _ACTIVE.discard(identifier)
            raise
    return snapshot
