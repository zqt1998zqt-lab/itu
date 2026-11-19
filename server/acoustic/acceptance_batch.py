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
from .dynamic_microphone import (
    synthesize_post_rx_microphone,
    synthesize_scoped_post_rx_microphone,
)
from .receive_configuration import normalize_receive_configuration, receive_configuration_sha256
from server.aec_routes import (
    admit_aec_route, aec_route_requires_capability, effective_aec_reference_channels,
    validate_aec_route_report, validate_admitted_aec_identity,
)
from server.ns_routes import execute_ns_command, ns_route_identity, resolve_ns_cli, validate_ns_route_report

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
_SINGLE_CASE_FIELDS = (
    "execution_scope", "selected_case_id", "dataset_case_count",
    "dataset_observation_count",
)


def _select_one_case(rows: list[dict], cases: list[dict], case_id: str | None):
    if case_id is None:
        return rows, cases
    if not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case_id) is None:
        raise ValueError("非法单组测试 case_id")
    matches = [case for case in cases if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("所选条件组不属于当前测试集、范围或模式")
    item_ids = set(matches[0]["item_ids"])
    selected = [row for row in rows if row["id"] in item_ids]
    if not selected or len(selected) != len(item_ids):
        raise ValueError("所选条件组的合同测项不完整")
    return selected, matches


def _case_is_native_nb8(case: dict, dataset: dict) -> bool:
    mode = case.get("conditions", {}).get("mode")
    if not isinstance(mode, str) or not mode.startswith("NB-"):
        return False
    rates = set()
    for relative in case.get("signals", {}).values():
        asset = dataset.get("assets", {}).get(relative)
        if not isinstance(asset, dict):
            return False
        rate = asset.get("sample_rate")
        if not isinstance(rate, int) or isinstance(rate, bool):
            return False
        rates.add(rate)
    return bool(rates) and rates == {8_000}


def _selected_execution(snapshot: dict, rows: list[dict], cases: list[dict]):
    present = [field for field in _SINGLE_CASE_FIELDS if field in snapshot]
    if not present:
        return rows, cases
    if (
        len(present) != len(_SINGLE_CASE_FIELDS)
        or snapshot.get("execution_scope") != "single_case"
        or not isinstance(snapshot.get("selected_case_id"), str)
        or type(snapshot.get("dataset_case_count")) is not int
        or snapshot["dataset_case_count"] != len(cases)
        or type(snapshot.get("dataset_observation_count")) is not int
        or snapshot["dataset_observation_count"] != len(rows)
        or snapshot.get("preview_condition_group_count") != len(cases)
    ):
        raise ValueError("单组执行选择与源测试范围不一致")
    selected_rows, selected_cases = _select_one_case(rows, cases, snapshot["selected_case_id"])
    if snapshot.get("case_count") != 1 or snapshot.get("observation_count") != len(selected_rows):
        raise ValueError("单组执行数量不一致")
    return selected_rows, selected_cases


def _validate_single_snapshot(snapshot: dict) -> None:
    if not any(field in snapshot for field in _SINGLE_CASE_FIELDS):
        return
    dataset = get_dataset(snapshot["dataset_id"])
    if dataset is None or dataset["manifest_sha256"] != snapshot.get("dataset_manifest_sha256"):
        raise ValueError("单组测试的源数据集身份不一致")
    rows, cases = select_batch_scope(
        dataset["manifest"], test_scope=snapshot.get("test_scope", "full"),
        mode=snapshot.get("selected_mode"), scene=snapshot.get("selected_scene"),
    )
    rows, cases = _selected_execution(snapshot, rows, cases)
    expected_case = cases[0]
    items = snapshot.get("items", [])
    if (
        len(items) != len(rows)
        or {row.get("item_id") for row in items} != {row["id"] for row in rows}
        or snapshot.get("progress", {}).get("total_cases") != 1
        or snapshot.get("progress", {}).get("total_items") != len(rows)
    ):
        raise ValueError("单组测试结果数量或合同行身份不一致")
    for item in items:
        binding = item.get("evidence_binding", {})
        if (
            binding.get("case_id") != expected_case["id"]
            or binding.get("dataset_id") != snapshot["dataset_id"]
            or binding.get("batch_id") != snapshot["batch_id"]
            or any(binding.get(field) != value for field, value in expected_case["conditions"].items())
        ):
            raise ValueError("单组测试证据绑定不一致")
    records = snapshot.get("cases", [])
    if (
        len(records) > 1
        or any(record.get("case_id") != expected_case["id"] for record in records)
        or snapshot.get("summary") != _summary(items)
    ):
        raise ValueError("单组测试记录或汇总不一致")
    if snapshot.get("status") == "completed" and (
        len(records) != 1 or records[0].get("status") != "completed"
        or any(item.get("status") in {"pending", "running"} for item in items)
        or snapshot["progress"].get("completed_cases") != 1
        or snapshot["progress"].get("completed_items") != len(rows)
    ):
        raise ValueError("单组测试未完成，不能标记为完成")
    if snapshot.get("status") == "completed":
        record = records[0]
        execution = record.get("execution")
        report_id = record.get("report_id")
        assessment_hash = record.get("assessment_sha256")
        if (
            not isinstance(report_id, str) or re.fullmatch(r"[a-f0-9]{32}", report_id) is None
            or not isinstance(assessment_hash, str) or re.fullmatch(r"[a-f0-9]{64}", assessment_hash) is None
            or not isinstance(execution, dict)
            or not isinstance(execution.get("output_sha256"), dict) or not execution["output_sha256"]
            or not isinstance(execution.get("output_alignment"), dict)
            or record.get("conditions") != expected_case["conditions"]
        ):
            raise ValueError("单组测试缺少完整执行或报告证据")
        source_hashes = {role: dataset["assets"][relative]["sha256"]
                         for role, relative in expected_case["signals"].items()}
        validate_execution_identity(snapshot, execution)
        if (
            execution.get("input_sha256") != source_hashes
            or any(item["evidence_binding"].get("report_id") != report_id
                   or item["evidence_binding"].get("input_sha256") != source_hashes
                   or item["evidence_binding"].get("output_sha256") != execution["output_sha256"]
                   for item in items)
        ):
            raise ValueError("单组测试输入、执行或测项报告身份不一致")
        destination = _directory(snapshot["batch_id"]) / "cases" / expected_case["id"]
        assessment_path = destination / "assessment.json"
        if assessment_path.is_symlink() or not assessment_path.is_file() or _sha(assessment_path) != assessment_hash:
            raise ValueError("单组测试计算报告哈希不一致")
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        if (
            assessment.get("case_id") != expected_case["id"]
            or assessment.get("report_id") != report_id
            or assessment.get("execution") != execution
            or not isinstance(assessment.get("items"), list)
            or len(assessment["items"]) != len(items)
            or any(not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                   for row in assessment["items"])
            or {row["item_id"]: row for row in assessment["items"]}
            != {row["item_id"]: row for row in items}
        ):
            raise ValueError("单组计算报告的条件组或报告身份不一致")
        for kind, digest in execution["output_sha256"].items():
            filename = execution["output_alignment"].get(kind, {}).get(
                "analysis_filename", "tx.after.wav" if kind == "tx" else "rx.rx-spk.wav",
            )
            if (
                kind not in {"tx", "rx"} or not isinstance(digest, str)
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None
                or not isinstance(filename, str) or Path(filename).name != filename
            ):
                raise ValueError("单组测试输出文件身份无效")
            output = destination / filename
            if output.is_symlink() or not output.is_file() or _sha(output) != digest:
                raise ValueError("单组测试输出音频哈希不一致")


def _report_path(identifier: str) -> Path:
    directory = _directory(identifier)
    snapshots = directory / "snapshots"
    if snapshots.is_symlink() or directory.is_symlink():
        raise ValueError("批次目录不能为符号链接")
    if snapshots.is_dir():
        candidates = sorted(path for path in snapshots.iterdir()
                            if re.fullmatch(r"[0-9]{8}\.json", path.name))
        if candidates:
            for latest in reversed(candidates):
                if latest.is_symlink() or not latest.is_file():
                    continue
                try:
                    candidate = json.loads(latest.read_text(encoding="utf-8"))
                    if not isinstance(candidate, dict) or candidate.get("execution_scope") != "single_case":
                        continue
                    _validate_single_snapshot(candidate)
                except (ValueError, OSError, TypeError, KeyError):
                    continue
                return latest
            raise ValueError("单组批次没有可恢复的完整快照")
    return directory / "report.json"


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
        if snapshot.get("test_scope", "full") == "full" and snapshot.get("execution_scope") != "single_case":
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
        if snapshot.get("execution_scope") == "single_case":
            from .scene_condition_batch import _resource_preflight
            _resource_preflight(service.BUILD)
            directory = _directory(snapshot["batch_id"])
            if directory.is_symlink():
                raise ValueError("单组批次目录不能为符号链接")
            snapshots = directory / "snapshots"
            snapshots.mkdir(parents=True, exist_ok=True)
            if snapshots.is_symlink():
                raise ValueError("单组快照目录不能为符号链接")
            sequence = max((int(path.stem) for path in snapshots.iterdir()
                            if re.fullmatch(r"[0-9]{8}\.json", path.name)), default=0) + 1
            with (snapshots / f"{sequence:08d}.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            return
        path = _directory(snapshot["batch_id"]) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def get_batch(identifier: str) -> dict | None:
    path = _report_path(identifier)
    with _LOCK:
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        _validate_single_snapshot(result)
        result = _derived_single_state(result)
        if result["status"] not in _TERMINAL and identifier not in _ACTIVE:
            result["status"] = "failed"
            result["error"] = "服务重启中断了该批次，请重新执行；已完成证据仍保留"
            for row in result["items"]:
                if row["status"] in {"pending", "running"}:
                    row.update(status="invalid", reason=result["error"], value=None)
            if result.get("execution_scope") != "single_case":
                _save(result)
        return result


def _derived_single_state(snapshot: dict) -> dict:
    if (snapshot.get("execution_scope") == "single_case"
            and snapshot.get("status") not in _TERMINAL
            and snapshot.get("batch_id") not in _ACTIVE):
        snapshot["status"] = "failed"
        snapshot["error"] = "服务重启中断了该单组测试；已有证据已保留，请重新执行"
        for item in snapshot.get("items", []):
            if item.get("status") in {"pending", "running"}:
                item.update(status="invalid", value=None, reason=snapshot["error"])
        snapshot["summary"] = _summary(snapshot["items"])
    return snapshot


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

        try:
            report_path = _report_path(directory.name)
        except (ValueError, OSError):
            continue
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
        try:
            _validate_single_snapshot(report)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        report = _derived_single_state(report)
        if report.get("test_scope", "full") != test_scope:
            continue
        if test_scope == "full" and report.get("execution_scope") == "single_case":
            # The 336-item console consumes complete batches. An individual
            # case remains available by its own identifier, never as 336 rows.
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
        if any(field in report for field in _SINGLE_CASE_FIELDS):
            if (
                any(field not in report for field in _SINGLE_CASE_FIELDS)
                or report.get("execution_scope") != "single_case"
                or progress["total_cases"] != 1
                or not isinstance(report.get("selected_case_id"), str)
            ):
                continue
            for field in (*_SINGLE_CASE_FIELDS, "case_count", "observation_count", "preview_condition_group_count"):
                summary[field] = report.get(field)
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


_TX_ONLY_RX_VALUE_FLAGS = frozenset({"--localvqe-reference-channel"})


def _rx_command_flags(flags: list[str]) -> list[str]:
    """Strip TX-only AEC options before constructing the RX-only command."""
    command_flags: list[str] = []
    skip_value = False
    for flag in flags:
        if skip_value:
            skip_value = False
            continue
        if flag in {"--no-rx-processing", "--rx-processing"}:
            continue
        if flag in _TX_ONLY_RX_VALUE_FLAGS:
            skip_value = True
            continue
        command_flags.append(flag)
    if skip_value:
        raise ValueError("RX command flags contain a TX-only option without a value")
    return command_flags


def _write_runtime_microphone(
    case: dict,
    prepared_paths: dict[str, Path],
    post_rx_playback: Path,
    destination: Path,
) -> tuple[dict[str, Path], dict]:
    """Derive the actual TX microphone after RX has produced playback PCM."""
    reference_info = sf.info(prepared_paths["reference"])
    noise_info = sf.info(prepared_paths["noise"])
    playback_info = sf.info(post_rx_playback)
    if reference_info.channels != 1 or noise_info.channels != 1:
        raise ValueError("运行时混音要求 reference/noise 为单通道")
    if not (
        reference_info.samplerate
        == noise_info.samplerate
        == playback_info.samplerate
    ):
        raise ValueError("运行时混音的三路源必须使用同一采样率")
    if not (
        reference_info.frames == noise_info.frames == playback_info.frames
    ):
        raise ValueError("运行时混音的三路源必须等长")
    reference, _ = sf.read(
        prepared_paths["reference"], dtype="float64", always_2d=True
    )
    noise, _ = sf.read(
        prepared_paths["noise"], dtype="float64", always_2d=True
    )
    playback, _ = sf.read(
        post_rx_playback, dtype="float64", always_2d=True
    )
    render_metadata = (
        case.get("provenance", {}).get("render_metadata")
        if isinstance(case.get("provenance"), dict)
        else None
    )
    if isinstance(render_metadata, dict):
        mix = synthesize_post_rx_microphone(
            reference[:, 0],
            noise[:, 0],
            playback[:, 0],
            render_metadata,
            sample_rate=int(reference_info.samplerate),
        )
        digest_payload: dict = {
            "echo_paths": render_metadata.get("echo_paths", []),
            "near_arrival_delays_samples": render_metadata.get(
                "near_arrival_delays_samples"
            ),
            "noise_spatial": render_metadata.get("noise_spatial"),
        }
        if render_metadata.get("echo_path_segments"):
            digest_payload["echo_path_segments"] = render_metadata[
                "echo_path_segments"
            ]
            digest_payload["echo_gain_linear"] = render_metadata.get(
                "echo_gain_linear"
            )
    elif case.get("dynamic_microphone_allow_scoped_default") is True:
        mic_channels = int(sf.info(prepared_paths["microphone"]).channels)
        mix = synthesize_scoped_post_rx_microphone(
            reference[:, 0],
            noise[:, 0],
            playback[:, 0],
            sample_rate=int(reference_info.samplerate),
            channels=mic_channels,
            beam_delays_ms=case.get("beam_delays_ms"),
            case_id=str(case.get("id", "")),
        )
        digest_payload = {
            "mode": "scoped_runtime_default",
            "channels": mic_channels,
            "sample_rate": int(reference_info.samplerate),
            "beam_delays_ms": case.get("beam_delays_ms"),
            "case_id": str(case.get("id", "")),
            "echo_gain_linear": 0.25,
        }
    else:
        raise ValueError("运行时混音缺少 render_metadata")

    microphone_path = destination / "tx.dynamic-microphone.wav"
    sf.write(
        microphone_path,
        mix.microphone,
        reference_info.samplerate,
        subtype="PCM_16",
    )
    dynamic_paths = {
        "microphone": microphone_path,
        "reference": prepared_paths["reference"],
        "noise": prepared_paths["noise"],
    }
    if mix.capture_gain < 1.0:
        reference_path = destination / "tx.dynamic-reference.wav"
        noise_path = destination / "tx.dynamic-noise.wav"
        sf.write(
            reference_path,
            mix.reference,
            reference_info.samplerate,
            subtype="PCM_16",
        )
        sf.write(
            noise_path,
            mix.noise,
            reference_info.samplerate,
            subtype="PCM_16",
        )
        dynamic_paths["reference"] = reference_path
        dynamic_paths["noise"] = noise_path

    echo_paths_sha256 = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source": "post_rx_runtime_mix",
        "formula": mix.formula,
        "path": str(microphone_path),
        "sha256": _sha(microphone_path),
        "sample_rate": int(reference_info.samplerate),
        "channels": mix.channels,
        "frames": int(reference_info.frames),
        "capture_gain": mix.capture_gain,
        "peak_before_gain": mix.peak_before_gain,
        "reference_path": str(dynamic_paths["reference"]),
        "reference_sha256": _sha(dynamic_paths["reference"]),
        "noise_path": str(dynamic_paths["noise"]),
        "noise_sha256": _sha(dynamic_paths["noise"]),
        "playback_path": str(post_rx_playback),
        "playback_sha256": _sha(post_rx_playback),
        "render_metadata_sha256": echo_paths_sha256,
    }
    return dynamic_paths, metadata


def is_native_rate_path_execution(execution: dict) -> bool:
    """Return whether execution used a native per-rate algorithm path."""
    rate_path = execution.get("rate_path")
    return (
        isinstance(rate_path, dict)
        and execution.get("source") == "native_rate_path"
        and rate_path.get("rate_policy") == "native"
        and rate_path.get("engine_sample_rate_hz")
        == rate_path.get("analysis_sample_rate_hz")
        == rate_path.get("output_sample_rate_hz")
        and execution.get("no_resample") is True
        and execution.get("fallback_frames") == 0
    )


def is_native_nb8_execution(execution: dict) -> bool:
    rate_path = execution.get("rate_path")
    return (
        is_native_rate_path_execution(execution)
        and isinstance(rate_path, dict)
        and rate_path.get("sample_rate_hz") == 8_000
        and rate_path.get("cluster_id") == "speex_dpdfnet2_8khz"
        and execution.get("algorithm_cluster_id") == "speex_dpdfnet2_8khz"
    )


def native_nb8_identity() -> dict:
    from server import dpdfnet8, nb8_route

    cli = nb8_route._resolve_speex_cli(None)
    model = Path(dpdfnet8.DEFAULT_MODEL_PATH).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"DPDFNet2 8 kHz 模型不存在：{model}")
    model_sha256 = _sha(model)
    if model_sha256 != dpdfnet8.MODEL_SHA256:
        raise ValueError("DPDFNet2 8 kHz 模型 SHA-256 与冻结值不一致")
    return {
        "cluster_id": nb8_route.CLUSTER_ID,
        "speex_cli_path": str(cli),
        "speex_cli_sha256": _sha(cli),
        "dpdfnet_model_path": str(model),
        "dpdfnet_model_sha256": model_sha256,
        "sample_rate_hz": 8_000,
        "frame_samples": nb8_route.FRAME_SAMPLES,
    }


def validate_execution_identity(
    snapshot: dict,
    execution: dict,
    *,
    require_algorithm_configuration: bool = False,
) -> None:
    """Validate requested and effective execution identities.

    A mixed-rate batch records the operator's requested cluster once.  Native
    rate paths then record that request plus the effective per-rate cluster and
    frozen binary/model identities.  Legacy execution keeps the original
    one-cluster equality contract.
    """
    if not isinstance(execution, dict):
        raise ValueError("执行报告不是对象")
    if is_native_rate_path_execution(execution):
        rate_path = execution["rate_path"]
        if execution.get("requested_algorithm_cluster_id") != snapshot.get(
            "algorithm_cluster_id"
        ):
            raise ValueError("原生采样率路径的请求算法簇与批次不一致")
        if execution.get("requested_configuration") != snapshot.get(
            "configuration"
        ):
            raise ValueError("原生采样率路径的请求配置与批次不一致")
        if execution.get("requested_catalog_sha256") != snapshot.get(
            "catalog_sha256"
        ):
            raise ValueError("原生采样率路径的请求目录身份与批次不一致")
        expected_requested_cli = snapshot.get(
            "requested_vcp_cli_sha256",
            snapshot.get("cli_sha256"),
        )
        if execution.get("requested_cli_sha256") != expected_requested_cli:
            raise ValueError("原生采样率路径的请求 CLI 身份与批次不一致")
        if execution.get("algorithm_cluster_id") != rate_path.get("cluster_id"):
            raise ValueError("原生采样率路径的实际算法簇与 rate_path 不一致")
        if execution.get("aec_route_capability") is not None:
            raise ValueError("原生采样率路径不得伪造通用 AEC 能力准入")
        if "native_nb8_identity" not in snapshot:
            raise ValueError("原生 8 kHz 执行缺少批次冻结身份")
        if execution.get("native_nb8_identity") != snapshot.get(
            "native_nb8_identity"
        ):
            raise ValueError("原生 8 kHz AEC/NS 身份与批次冻结值不一致")
        identity = snapshot["native_nb8_identity"]
        if execution.get("effective_cli_sha256") != identity.get(
            "speex_cli_sha256"
        ):
            raise ValueError("原生 8 kHz Speex CLI 身份与批次冻结值不一致")
        return

    if require_algorithm_configuration:
        if execution.get("algorithm_cluster_id") != snapshot.get(
            "algorithm_cluster_id"
        ):
            raise ValueError("执行期间算法簇发生变化")
        if execution.get("configuration") != snapshot.get("configuration"):
            raise ValueError("执行期间算法簇 configuration 发生变化")
    if snapshot.get("aec_route_capability") != execution.get(
        "aec_route_capability"
    ):
        raise ValueError("执行期间 AEC DSP 模型或设备准入身份发生变化")
    if snapshot.get("ns_route_identity") != execution.get("ns_route_identity"):
        raise ValueError("执行期间 NS 模型身份发生变化")
    if execution.get("catalog_sha256") != snapshot.get("catalog_sha256"):
        raise ValueError("执行期间算法目录 catalog_sha256 发生变化")
    if execution.get("cli_sha256") != snapshot.get("cli_sha256"):
        raise ValueError("执行期间 VCP CLI 发生变化")


def _native_nb8_beamform_microphone(
    case: dict,
    microphone_path: Path,
    destination: Path,
) -> tuple[Path, dict | None, dict | None]:
    info = sf.info(microphone_path)
    if info.channels == 1:
        return microphone_path, None, None
    delays = case.get("beam_delays_ms")
    if not (
        isinstance(delays, list)
        and len(delays) == info.channels
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in delays
        )
    ):
        raise ValueError(
            "原生 8 kHz 多麦克风输入必须提供每通道 beam_delays_ms"
        )
    source, source_rate = sf.read(
        microphone_path,
        dtype="float64",
        always_2d=True,
    )
    delay_samples = [
        float(value) * source_rate / 1000.0 for value in delays
    ]
    if max(delay_samples) > source_rate:
        raise ValueError("原生 8 kHz beam_delays_ms 超过 1 秒")
    aligned = np.zeros_like(source, dtype=np.float64)
    for channel, delay in enumerate(delay_samples):
        whole = int(delay)
        fraction = delay - whole
        if whole:
            aligned[whole:, channel] = source[:-whole, channel]
        else:
            aligned[:, channel] = source[:, channel]
        if fraction and whole + 1 < len(source):
            aligned[whole + 1 :, channel] += (
                fraction * source[: -whole - 1, channel]
            )
    beamformed = np.mean(aligned, axis=1)
    output = destination / "tx.native8k-beamformed.wav"
    if output.exists():
        raise ValueError(f"拒绝覆盖既有 8 kHz 波束形成证据：{output}")
    sf.write(output, beamformed, source_rate, subtype="PCM_16")
    adaptation = {
        "source_unit": "ms",
        "engine_rate": source_rate,
        "delays_ms": list(delays),
        "delays_samples": delay_samples,
        "method": "fractional delay-and-sum",
        "source_channels": info.channels,
        "output_channels": 1,
    }
    preprocessing = {
        **adaptation,
        "source_path": str(microphone_path),
        "source_sha256": _sha(microphone_path),
        "output_path": str(output),
        "output_sha256": _sha(output),
    }
    return output, adaptation, preprocessing


def _native_nb8_alignment(report: dict, output: Path) -> dict:
    dpdfnet_info = report["stages"]["dpdfnet2_8khz"]["info"]
    return {
        "method": "native_8khz_aligned_output",
        "declared_algorithm_latency_samples": int(
            report["algorithm_delay_samples"]
        ),
        "adapter_trim_samples": int(dpdfnet_info.get("trim_samples", 0)),
        "batch_trim_samples": 0,
        "trimmed_leading_samples": 0,
        "padded_samples": int(dpdfnet_info.get("padded_samples", 0)),
        "remaining_delay_samples": int(
            dpdfnet_info.get("remaining_delay_samples", 0)
        ),
        "raw_filename": output.name,
        "raw_sha256": _sha(output),
        "analysis_filename": output.name,
        "analysis_sha256": _sha(output),
    }


def _run_native_nb8(
    case: dict,
    paths: dict[str, Path],
    destination: Path,
    *,
    requested_cluster: str,
    requested_configuration: dict,
    requested_catalog_sha256: str,
    requested_cli_sha256: str | None,
) -> tuple[dict[str, Path], dict]:
    from server import dpdfnet8, nb8_route
    from server.rate_paths import require_primary_path

    mode = case["conditions"].get("mode", "")
    if not isinstance(mode, str) or not mode.startswith("NB-"):
        raise ValueError("原生 8 kHz 路径只接受 NB-* 业务模式")
    for role, path in paths.items():
        info = sf.info(path)
        if info.samplerate != 8_000:
            raise ValueError(
                f"原生 8 kHz 路径要求 {role} 为 8 kHz，"
                f"实际 {info.samplerate} Hz"
            )
        if role != "microphone" and info.channels != 1:
            raise ValueError(
                f"原生 8 kHz 路径要求 {role} 为 mono，"
                f"实际 {info.channels} 声道"
            )

    direction = case["conditions"].get("direction")
    if direction not in {"receive", "send", "round_trip"}:
        raise ValueError(f"未知测试方向: {direction!r}")
    rate_path = require_primary_path(8_000)
    runtime = dpdfnet8.load_runtime()
    outputs: dict[str, Path] = {}
    reports: dict[str, dict] = {}
    output_alignment: dict[str, dict] = {}
    commands: list[list[str]] = []
    dynamic_microphone: dict | None = None
    native_preprocessing: dict[str, dict] = {}
    beam_delay_adaptation: dict | None = None

    rx_result = nb8_route.run_receive(
        paths["far_end"],
        destination / "rx",
        runtime=runtime,
    )
    rx_output = Path(rx_result["output_path"]).resolve()
    rx_report_path = Path(rx_result["report_path"]).resolve()
    rx_report = json.loads(rx_report_path.read_text(encoding="utf-8"))
    outputs["rx"] = rx_output
    reports["rx"] = {
        "path": str(rx_report_path),
        "sha256": _sha(rx_report_path),
        "data": rx_report,
    }
    output_alignment["rx"] = _native_nb8_alignment(rx_report, rx_output)

    tx_output: Path | None = None
    if direction != "receive":
        microphone_source = paths["microphone"]
        if case.get("dynamic_microphone_from_post_rx") is True:
            dynamic_paths, dynamic_microphone = _write_runtime_microphone(
                case,
                paths,
                rx_output,
                destination,
            )
            microphone_source = dynamic_paths["microphone"]
        (
            microphone_path,
            beam_delay_adaptation,
            beamforming,
        ) = _native_nb8_beamform_microphone(
            case,
            microphone_source,
            destination,
        )
        if beamforming is not None:
            native_preprocessing["beamforming"] = beamforming
        tx_result = nb8_route.run(
            microphone_path,
            rx_output,
            destination / "tx",
            runtime=runtime,
        )
        tx_output = Path(tx_result["output_path"]).resolve()
        tx_report_path = Path(tx_result["report_path"]).resolve()
        tx_report = json.loads(tx_report_path.read_text(encoding="utf-8"))
        outputs["tx"] = tx_output
        reports["tx"] = {
            "path": str(tx_report_path),
            "sha256": _sha(tx_report_path),
            "data": tx_report,
        }
        output_alignment["tx"] = _native_nb8_alignment(
            tx_report,
            tx_output,
        )
        commands.extend(tx_report.get("commands", []))

    input_sha256 = {role: _sha(path) for role, path in paths.items()}
    output_sha256 = {role: _sha(path) for role, path in outputs.items()}
    rx_playback_artifact = {
        "source": "native_nb8_rx_processed",
        "input_role": "far_end",
        "input_sha256": input_sha256["far_end"],
        "processed_output": "rx",
        "processed_sha256": output_sha256["rx"],
        "rate_path_cluster_id": rate_path.cluster_id,
        "sample_rate_hz": 8_000,
        "channels": 1,
        "stage_chain": ["dpdfnet2_8khz"],
        "vcp_rx_equivalent": False,
    }
    echo_reference = None
    if tx_output is not None:
        echo_reference = {
            **rx_playback_artifact,
            "source": "native_nb8_rx_processed",
            "tx_echo_argument_sha256": output_sha256["rx"],
            "tx_consumed_output_role": "rx",
        }
    execution = {
        "source": "native_rate_path",
        "rate_path": rate_path.as_dict(),
        "algorithm_cluster_id": rate_path.cluster_id,
        "configuration": {
            "profile": "native_rate_path",
            "selection": {
                "aec": rate_path.aec_backend,
                "ns": rate_path.ns_backend,
            },
        },
        "requested_algorithm_cluster_id": requested_cluster,
        "requested_configuration": requested_configuration,
        "requested_catalog_sha256": requested_catalog_sha256,
        "requested_cli_sha256": requested_cli_sha256,
        "catalog_sha256": requested_catalog_sha256,
        "cli_sha256": _sha(nb8_route._resolve_speex_cli(None)),
        "effective_cli_sha256": _sha(nb8_route._resolve_speex_cli(None)),
        "native_nb8_identity": native_nb8_identity(),
        "commands": commands,
        "pipeline_reports": reports,
        "echo_reference": echo_reference,
        "rx_playback_artifact": rx_playback_artifact,
        "direction": direction,
        "output_roles": sorted(outputs),
        "aec_applicability": (
            "not_applicable_receive_only"
            if direction == "receive"
            else "speex_mdf_tx_uses_native_nb8_rx_reference"
        ),
        "rx_stage_chain": ["dpdfnet2_8khz"],
        "tx_stage_chain": (
            ["speex_mdf_nb8", "dpdfnet2_8khz"]
            if tx_output is not None
            else []
        ),
        "vcp_rx_equivalent": False,
        "aec_route_capability": None,
        "ns_route_identity": {
            "backend": rate_path.ns_backend,
            "model_sha256": dpdfnet8.MODEL_SHA256,
            "sample_rate_hz": 8_000,
        },
        "ns_execution": (
            reports.get("tx", reports.get("rx", {}))
            .get("data", {})
            .get("stages", {})
            .get("dpdfnet2_8khz")
        ),
        "input_adaptation": {},
        "native_preprocessing": native_preprocessing,
        "output_alignment": output_alignment,
        "beam_delay_adaptation": beam_delay_adaptation,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "no_resample": True,
        "resampling": "none",
        "fallback_frames": 0,
    }
    if dynamic_microphone is not None:
        execution["dynamic_microphone"] = dynamic_microphone
        effective_microphone_sha = (
            native_preprocessing.get("beamforming", {}).get("output_sha256")
            or dynamic_microphone["sha256"]
        )
        execution["effective_input_sha256"] = {
            "microphone": effective_microphone_sha,
            "reference": dynamic_microphone["reference_sha256"],
            "noise": dynamic_microphone["noise_sha256"],
            "far_end": rx_playback_artifact["processed_sha256"],
        }
    return outputs, execution


def _run_cli(case: dict, paths: dict[str, Path], destination: Path, cluster: str) -> tuple[dict, dict]:
    # Import lazily: alignment_runner also imports the legacy five-signal module.
    from server.alignment_runner import resolve_cli, resolve_algorithm_cluster, validate_pipeline_artifacts
    from server.rate_paths import NB8_CLUSTER_ID

    preset, flags, configuration, catalog_sha = resolve_algorithm_cluster(cluster)
    mode = case["conditions"].get("mode", "")
    rates = {sf.info(path).samplerate for path in paths.values()}
    if isinstance(mode, str) and mode.startswith("NB-") and rates == {8_000}:
        default_cli = resolve_cli()
        try:
            requested_cli = (
                default_cli
                if cluster == NB8_CLUSTER_ID
                else (
                    resolve_ns_cli(default_cli, configuration)
                    if default_cli is not None
                    else None
                )
            )
        except ValueError:
            requested_cli = None
        return _run_native_nb8(
            case,
            paths,
            destination,
            requested_cluster=cluster,
            requested_configuration=configuration,
            requested_catalog_sha256=catalog_sha,
            requested_cli_sha256=(
                _sha(requested_cli) if requested_cli is not None else None
            ),
        )
    if cluster == NB8_CLUSTER_ID:
        raise ValueError("算法簇 speex_dpdfnet2_8khz 仅支持原生 8 kHz（NB-*）输入")
    cli = resolve_ns_cli(resolve_cli(), configuration)
    if cli is None:
        raise ValueError("未找到可执行的 VCP CLI")
    profile = preset["profile"]
    cli_hash = _sha(cli)
    outputs: dict[str, Path] = {}
    commands = []
    reports = {}
    prepared_paths = {}
    preprocessing = {}
    beam_delay_adaptation = None
    dynamic_microphone: dict | None = None
    for role, path in paths.items():
        info = sf.info(path)
        if info.samplerate in {16000, 24000, 48000}:
            prepared_paths[role] = path
            continue
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
    admitted_aec = None
    tx_beam_delays: list[float] | None = None
    tx_beam_delay_adaptation = None
    if direction != "receive":
        microphone_info = sf.info(prepared_paths["microphone"])
        channels = microphone_info.channels
        delays = case.get("beam_delays_ms")
        if channels > 1:
            if not (
                isinstance(delays, list)
                and len(delays) == channels
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                    for value in delays
                )
            ):
                raise ValueError(
                    "多麦克输入必须提供每通道的 beam_delays_ms 校准值"
                )
            delay_samples = [
                float(value) * microphone_info.samplerate / 1000.0
                for value in delays
            ]
            if not all(math.isfinite(value) for value in delay_samples):
                raise ValueError(
                    "beam_delays_ms 转换后的样本延时不是有限数"
                )
            tx_beam_delays = delay_samples
            tx_beam_delay_adaptation = {
                "source_unit": "ms",
                "engine_rate": microphone_info.samplerate,
                "delays_ms": list(delays),
                "delays_samples": delay_samples,
            }
        if (
            isinstance(configuration.get("selection"), dict)
            and aec_route_requires_capability(configuration)
        ):
            playback_info = sf.info(prepared_paths["far_end"])
            if microphone_info.samplerate != playback_info.samplerate:
                raise ValueError("AEC 路由的麦克风混合输入与远端播放信号 / 回声参考采样率必须一致")
            admitted_aec = admit_aec_route(
                cli, configuration,
                sample_rate=microphone_info.samplerate,
                reference_channels=effective_aec_reference_channels(
                    configuration, playback_info.channels
                ),
                capture_channels=channels,
                beamforming=(
                    tx_beam_delays is not None
                    and configuration["selection"].get("bf") != "off"
                ),
            )
    if direction == "receive":
        execution_order = ["rx"]
    elif direction == "round_trip":
        execution_order = ["rx", "tx"]
    else:
        # The RX-processed playback PCM is the only echo reference accepted by
        # TX. This makes the downlink chain explicit instead of feeding TX the
        # pre-RX dataset input.
        execution_order = ["rx", "tx"]
    processed_echo_path: Path | None = None
    echo_reference: dict | None = None
    for kind in execution_order:
        prefix = destination / kind
        command_flags = flags
        if kind == "rx":
            # The cluster flags describe the TX invocation and explicitly
            # disable RX there. RX must not inherit that opt-out; it gets the
            # positive processing switch below. TX-only options that require
            # playback PCM must not leak into the RX-only invocation.
            command_flags = _rx_command_flags(flags)
        command = [str(cli), "--profile", profile, *command_flags]
        if kind == "rx":
            command += ["--rx-in", str(prepared_paths["far_end"]), "--rx-processing"]
            output = Path(f"{prefix}.rx-spk.wav")
        else:
            if processed_echo_path is None:
                raise ValueError("TX 必须使用已完成的 VCP RX 处理后回声参考")
            command += [
                "--mic",
                str(prepared_paths["microphone"]),
                "--echo",
                str(processed_echo_path),
            ]
            microphone_info = sf.info(prepared_paths["microphone"])
            channels = microphone_info.channels
            if channels == 1:
                command += ["--no-bf"]
            else:
                if tx_beam_delays is None:
                    raise ValueError(
                        "多麦克输入缺少已校验的 beam_delays_ms"
                    )
                command += [
                    "--beam-delays",
                    ",".join(str(value) for value in tx_beam_delays),
                ]
                beam_delay_adaptation = tx_beam_delay_adaptation
            output = Path(f"{prefix}.after.wav")
            if dynamic_microphone is not None:
                command_microphone = Path(
                    command[command.index("--mic") + 1]
                ).resolve()
                command_echo = Path(
                    command[command.index("--echo") + 1]
                ).resolve()
                if (
                    command_microphone
                    != Path(dynamic_microphone["path"]).resolve()
                    or command_echo != processed_echo_path.resolve()
                ):
                    raise ValueError(
                        "TX 命令未绑定运行时动态麦克风或 RX 处理后回声参考"
                    )
        command += ["--out", str(prefix)]
        process = execute_ns_command(
            command, configuration=configuration,
            capture_output=True, text=True, timeout=300,
        )
        (destination / f"{kind}.log").write_text(process.stdout + "\n" + process.stderr)
        if process.returncode:
            raise ValueError(f"VCP {kind} 退出码 {process.returncode}: {process.stderr[-1500:]}")
        report_path = Path(f"{prefix}.json")
        if kind == "tx":
            report, errors = validate_pipeline_artifacts(report_path, output, profile)
            if "selection" in configuration:
                errors += validate_aec_route_report(report, configuration)
                errors += validate_admitted_aec_identity(report, admitted_aec)
                errors += validate_ns_route_report(report, configuration)
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
        if kind == "rx":
            processed_echo_path = output
            echo_reference = {
                "source": "vcp_rx_processed",
                "input_role": "far_end",
                "input_sha256": _sha(paths["far_end"]),
                "processed_output": "rx",
                "processed_sha256": _sha(output),
                "tx_echo_argument_sha256": _sha(output),
            }
            if case.get("dynamic_microphone_from_post_rx") is True:
                dynamic_paths, dynamic_microphone = _write_runtime_microphone(
                    case,
                    prepared_paths,
                    output,
                    destination,
                )
                prepared_paths.update(dynamic_paths)
        outputs[kind] = output
        commands.append(command)
        reports[kind] = {"path": str(report_path), "sha256": _sha(report_path), "data": report}
    if _sha(cli) != cli_hash:
        raise ValueError("执行期间 VCP CLI 被替换，本次证据无效")
    rx_playback_artifact = None
    if echo_reference is not None:
        rx_info = sf.info(outputs["rx"])
        rx_playback_artifact = {
            "source": "vcp_rx_processed",
            "input_role": "far_end",
            "input_sha256": echo_reference["input_sha256"],
            "processed_output": "rx",
            "processed_sha256": echo_reference["processed_sha256"],
            "sample_rate_hz": rx_info.samplerate,
            "channels": rx_info.channels,
        }
    if direction == "receive":
        echo_reference = None
    execution = {
        "source": "alignment_batch_cli", "algorithm_cluster_id": cluster,
        "configuration": configuration, "catalog_sha256": catalog_sha,
        "cli_sha256": cli_hash, "commands": commands, "pipeline_reports": reports,
        "echo_reference": echo_reference,
        "rx_playback_artifact": rx_playback_artifact,
        "direction": direction,
        "output_roles": sorted(outputs),
        "aec_route_capability": admitted_aec,
        "ns_route_identity": ns_route_identity(configuration),
        "ns_execution": reports.get("tx", {}).get("data", {}).get("ns_execution"),
        "input_adaptation": preprocessing, "output_alignment": output_processing,
        "beam_delay_adaptation": beam_delay_adaptation,
        "input_sha256": {role: _sha(path) for role, path in paths.items()},
        "output_sha256": {role: _sha(path) for role, path in outputs.items()},
    }
    if isinstance(mode, str) and mode.startswith("NB-"):
        execution.update(
            business_sample_rate_hz=8_000,
            engine_sample_rate_hz=16_000,
            rate_path_policy="nb_measurement_fixture_16khz",
            native_nb8_eligible=False,
            rate_path_reason=(
                "NB response/OOB 或混合测量夹具需要高于 4 kHz 的可观测带宽；"
                "该 case 不是原生 8 kHz 主路径证据"
            ),
        )
    if dynamic_microphone is not None:
        execution["dynamic_microphone"] = dynamic_microphone
        rx_artifact = rx_playback_artifact
        if not isinstance(rx_artifact, dict):
            raise ValueError("动态麦克风执行缺少 RX 播放产物")
        execution["effective_input_sha256"] = {
            "microphone": dynamic_microphone["sha256"],
            "reference": dynamic_microphone["reference_sha256"],
            "noise": dynamic_microphone["noise_sha256"],
            "far_end": rx_artifact["processed_sha256"],
        }
    return outputs, execution


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
        execution_rows, execution_cases = select_batch_scope(
            dataset["manifest"], test_scope=snapshot.get("test_scope", "full"),
            mode=snapshot.get("selected_mode"),
            scene=snapshot.get("selected_scene"),
        )
        _, execution_cases = _selected_execution(snapshot, execution_rows, execution_cases)
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
            if snapshot.get("execution_scope") == "single_case":
                from .scene_condition_batch import _resource_preflight
                _resource_preflight(_directory(identifier))
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
                run_case = case
                if (
                    snapshot.get("test_scope") == "background_noise"
                    and case["conditions"].get("direction") != "receive"
                ):
                    run_case = {
                        **case,
                        "dynamic_microphone_from_post_rx": True,
                        "dynamic_microphone_allow_scoped_default": True,
                    }
                outputs, execution = _run_cli(run_case, paths, destination, snapshot["algorithm_cluster_id"])
                validate_execution_identity(snapshot, execution)
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
                for role, digest in execution["input_sha256"].items():
                    if digest != dataset["assets"][case["signals"][role]]["sha256"]:
                        raise ValueError("执行期间数据集输入身份发生变化")
                effective_paths = dict(paths)
                dynamic_microphone = execution.get("dynamic_microphone")
                dynamic_microphone_visual = None
                if isinstance(dynamic_microphone, dict):
                    effective_paths["microphone"] = Path(dynamic_microphone["path"])
                    effective_paths["reference"] = Path(dynamic_microphone["reference_path"])
                    effective_paths["noise"] = Path(dynamic_microphone["noise_path"])
                    rx_path = outputs.get("rx")
                    if rx_path is not None:
                        effective_paths["far_end"] = rx_path
                calculate_kwargs = {"report_id": report_id}
                if "receive_simulation" in dataset["manifest"]:
                    calculate_kwargs.update(
                        receive_config=receive_config,
                        receive_ir_paths=receive_ir_paths,
                    )
                rows = calculate_case_items(
                    run_case,
                    effective_paths,
                    outputs,
                    execution,
                    **calculate_kwargs,
                )
                if any(_sha(path) != execution["input_sha256"].get(role) for role, path in paths.items()):
                    raise ValueError("计算期间输入文件身份发生变化")
                if isinstance(dynamic_microphone, dict):
                    if _sha(Path(dynamic_microphone["path"])) != dynamic_microphone["sha256"]:
                        raise ValueError("计算期间运行时动态麦克风身份发生变化")
                    from .acceptance_scene_condition_api import _visual_track
                    from .scene_condition_batch import _compact_visual

                    dynamic_microphone_visual = _compact_visual(
                        _visual_track(
                            Path(dynamic_microphone["path"]),
                            dynamic_microphone["sha256"],
                        )
                    )
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
                if isinstance(dynamic_microphone, dict):
                    case_report["dynamic_microphone"] = dynamic_microphone
                    case_report["effective_input_sha256"] = execution.get("effective_input_sha256", {})
                if dynamic_microphone_visual is not None:
                    case_report["dynamic_microphone_visual"] = dynamic_microphone_visual
                (destination / "assessment.json").write_text(json.dumps(case_report, ensure_ascii=False, indent=2, allow_nan=False))
                case_entry = {
                    "case_id": case["id"],
                    "report_id": report_id,
                    "conditions": case["conditions"],
                    "status": "completed",
                    "assessment_sha256": _sha(destination / "assessment.json"),
                    "execution": execution,
                }
                if isinstance(dynamic_microphone, dict):
                    case_entry["dynamic_microphone"] = dynamic_microphone
                    case_entry["effective_input_sha256"] = execution.get("effective_input_sha256", {})
                if dynamic_microphone_visual is not None:
                    case_entry["dynamic_microphone_visual"] = dynamic_microphone_visual
                snapshot["cases"].append(case_entry)
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
        from .scene_condition_batch import ResourceSafetyStop
        if isinstance(error, ResourceSafetyStop):
            raise
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
    case_id: str | None = None,
) -> dict:
    from server.alignment_runner import resolve_algorithm_cluster, resolve_cli
    if test_scope not in TEST_SCOPES:
        raise ValueError("未知测试范围")
    from server.rate_paths import (
        NB8_CLUSTER_ID,
        sample_rate_for_mode,
        validate_cluster_rate_compatibility,
    )

    preset, _, configuration, catalog_sha = resolve_algorithm_cluster(algorithm_cluster_id)
    if test_scope != "full" and not preset["profile"].startswith("vcp"):
        raise ValueError("子范围测试仅支持 VCP 算法簇")
    is_native_nb8_cluster = algorithm_cluster_id == NB8_CLUSTER_ID
    if mode is not None:
        validate_cluster_rate_compatibility(
            algorithm_cluster_id,
            sample_rate_for_mode(mode),
            configuration=configuration,
        )
    elif is_native_nb8_cluster:
        raise ValueError("算法簇 speex_dpdfnet2_8khz 仅支持 8 kHz（NB-*），不能用于全模式 336 项批次")
    selected_ns_identity = None if is_native_nb8_cluster else ns_route_identity(configuration)
    cli = (
        resolve_cli()
        if is_native_nb8_cluster
        else resolve_ns_cli(resolve_cli(), configuration)
    )
    native_only_candidate = (
        test_scope != "full"
        and isinstance(mode, str)
        and mode.startswith("NB-")
    )
    if cli is None and not native_only_candidate:
        raise ValueError("未找到可执行的 VCP CLI")
    requested_vcp_cli_sha256 = _sha(cli) if cli is not None else None
    snapshot_cli_sha256 = (
        requested_vcp_cli_sha256
        if requested_vcp_cli_sha256 is not None
        else native_nb8_identity()["speex_cli_sha256"]
    )
    admitted_aec = None
    if (
        not is_native_nb8_cluster
        and not (isinstance(mode, str) and mode.startswith("NB-"))
        and test_scope != "basic_downlink"
        and not (test_scope == "full" and case_id is not None)
        and aec_route_requires_capability(configuration)
    ):
        admitted_aec = admit_aec_route(
            cli,
            configuration,
            sample_rate=(sample_rate_for_mode(mode) if mode is not None else None),
        )
    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise ValueError("测试集不存在")
    selected_rows, cases = select_batch_scope(
        dataset["manifest"],
        test_scope=test_scope,
        mode=mode,
        scene=scene,
    )
    dataset_case_count, dataset_observation_count = len(cases), len(selected_rows)
    selected_rows, cases = _select_one_case(selected_rows, cases, case_id)
    case_modes = {
        case.get("conditions", {}).get("mode")
        for case in cases
        if isinstance(case.get("conditions", {}).get("mode"), str)
        and "-" in case.get("conditions", {}).get("mode", "")
    }
    for case_mode in sorted(case_modes):
        validate_cluster_rate_compatibility(
            algorithm_cluster_id,
            sample_rate_for_mode(case_mode),
            configuration=configuration,
        )
    all_native_nb8 = bool(cases) and all(
        _case_is_native_nb8(case, dataset)
        for case in cases
    )
    if (
        not is_native_nb8_cluster
        and not all_native_nb8
        and admitted_aec is None
        and test_scope != "basic_downlink"
        and any(
            case["conditions"]["direction"] != "receive"
            for case in cases
        )
        and aec_route_requires_capability(configuration)
    ):
        admitted_aec = admit_aec_route(cli, configuration)
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
                "configuration": configuration, "catalog_sha256": catalog_sha,
                "cli_sha256": snapshot_cli_sha256,
                "requested_vcp_cli_sha256": requested_vcp_cli_sha256,
                "cases": [],
                "receive_simulation": receive_config, "receive_ir_sha256": receive_ir_sha256,
                "receive_configuration_sha256": acoustic_digest,
                "scope": "condition-bound VCP engineering assessment; formal ITU compliance not measured",
                "standard_compliance": False, "formal_test_status": "not_measured",
                "progress": {"completed_cases": 0, "total_cases": len(cases), "completed_items": 0, "total_items": len(items)},
                "items": items}
    if admitted_aec is not None:
        snapshot["aec_route_capability"] = admitted_aec
    if selected_ns_identity is not None:
        snapshot["ns_route_identity"] = selected_ns_identity
    if any(_case_is_native_nb8(case, dataset) for case in cases):
        from server.rate_paths import require_primary_path

        snapshot["native_nb8_identity"] = native_nb8_identity()
        snapshot["rate_paths"] = {
            "8000": require_primary_path(8_000).as_dict()
        }
    if case_id is not None:
        snapshot.update(
            execution_scope="single_case", selected_case_id=case_id,
            dataset_case_count=dataset_case_count,
            dataset_observation_count=dataset_observation_count,
            preview_condition_group_count=dataset_case_count,
            case_count=1, observation_count=len(items),
        )
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
