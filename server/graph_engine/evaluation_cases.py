"""Generate graphics from published frozen pipeline evaluation cases."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from server.audio import Audio, AudioError, decode
from server.graph_engine.engine import GraphOptions, analyze_graphics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "public/evaluation/pipeline-cases/manifest.json"


class EvaluationCaseNotFound(LookupError):
    """The requested frozen case is not part of the published manifest."""


@dataclass(frozen=True)
class EvaluationCaseBundle:
    case: dict[str, Any]
    before: Audio
    after: Audio
    options: GraphOptions
    manifest_path: Path

    @property
    def delay_ms(self) -> float:
        return self.case["delay_samples"] * 1000 / self.case["sample_rate"]

    @property
    def duration_seconds(self) -> float:
        return len(self.before.data) / self.before.rate


def _manifest_cases(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AudioError(f"冻结案例清单不可读取：{error}") from error
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list) or not assets:
        raise AudioError("冻结案例清单缺少 assets")
    return [item for item in assets if isinstance(item, dict)]


def _safe_file(project_root: Path, relative: str, role: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise AudioError(f"{role} 文件路径必须是项目内相对路径")
    try:
        path = (project_root / relative).resolve(strict=True)
    except OSError as error:
        raise AudioError(f"{role} 文件不存在：{relative}") from error
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise AudioError(f"{role} 文件越过项目目录：{relative}") from error
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise AudioError(f"{role} 必须是项目内 WAV 文件：{relative}")
    return path


def _window_values(
    audio: Audio,
    *,
    crop_samples: int,
    window: tuple[float, float],
    delay_samples: int,
    role: str,
) -> np.ndarray:
    data = audio.data[crop_samples:]
    start = round(window[0] * audio.rate)
    end = round(window[1] * audio.rate)
    source_start = start + delay_samples
    source_end = end + delay_samples
    if start < 0 or end <= start or source_end > len(data):
        raise AudioError(f"{role} 案例窗口或固定延迟超出音频范围")
    values = np.asarray(data[source_start:source_end], dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise AudioError(f"{role} 案例窗口为空或包含非有限样本")
    if float(np.max(np.abs(values))) < 1e-7:
        raise AudioError(f"{role} 案例窗口接近数字静音")
    return values


def load_evaluation_case(
    asset_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> EvaluationCaseBundle:
    if not re.fullmatch(r"[a-z0-9_-]{1,120}", asset_id):
        raise EvaluationCaseNotFound("案例 ID 格式无效")
    case = next(
        (item for item in _manifest_cases(manifest_path) if item.get("asset_id") == asset_id),
        None,
    )
    if case is None:
        raise EvaluationCaseNotFound("冻结评测案例不存在")

    try:
        sample_rate = int(case["sample_rate"])
        before_channel = int(case["before_channel"])
        after_channel = int(case["after_channel"])
        crop_samples = int(case["before_crop_samples"])
        delay_samples = int(case["delay_samples"])
        window = (float(case["window_seconds"][0]), float(case["window_seconds"][1]))
        before_label = str(case["before_label"])
        after_label = str(case["after_label"])
        before_path = _safe_file(project_root, str(case["before"]), "处理前")
        after_path = _safe_file(project_root, str(case["after"]), "处理后")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AudioError("冻结评测案例元数据不完整") from error

    if crop_samples < 0 or delay_samples < 0:
        raise AudioError("冻结评测案例的裁剪或延迟值无效")
    if window[1] * sample_rate + delay_samples > 120 * sample_rate:
        raise AudioError("冻结评测案例窗口超过 120 秒")

    before = decode("before", before_path.name, before_path.read_bytes(), before_channel)
    after = decode("after", after_path.name, after_path.read_bytes(), after_channel)
    if before.rate != sample_rate or after.rate != sample_rate:
        raise AudioError("冻结评测案例的实际采样率与清单不一致")
    if before.sha256 != case.get("before_sha256") or after.sha256 != case.get("after_sha256"):
        raise AudioError("冻结评测案例音频与清单 SHA-256 不一致")

    before = replace(
        before,
        data=_window_values(
            before,
            crop_samples=crop_samples,
            window=window,
            delay_samples=0,
            role="处理前",
        ),
    )
    after = replace(
        after,
        data=_window_values(
            after,
            crop_samples=crop_samples,
            window=window,
            delay_samples=delay_samples,
            role="处理后",
        ),
    )
    if len(before.data) != len(after.data):
        raise AudioError("固定延迟裁剪后，案例输入和输出长度不一致")

    options = GraphOptions(
        sample_rate=sample_rate,
        before_channel=before_channel,
        after_channel=after_channel,
        align=False,
        time_offset_seconds=window[0],
        before_label=before_label,
        after_label=after_label,
    )
    return EvaluationCaseBundle(case, before, after, options, manifest_path)


def build_evaluation_case_graphics(
    asset_id: str,
) -> tuple[dict, dict[str, bytes]]:
    bundle = load_evaluation_case(asset_id)
    report, figures = analyze_graphics(bundle.before, bundle.after, bundle.options)
    case = bundle.case
    report["scope"] = "frozen evaluation-case visualization and engineering diagnostics"
    report["alignment"] = {
        "requested": False,
        "applied": True,
        "delay_ms": bundle.delay_ms,
        "correlation": None,
        "reliable": True,
        "note": "按冻结评测清单声明的固定因果延迟裁剪；未执行跨相关或内容匹配。",
        "overlap_seconds": bundle.duration_seconds,
        "search_ms": None,
    }
    report["notices"] = [
        note
        for note in report["notices"]
        if "用户关闭自动对齐" not in note
    ]
    report["notices"].extend(
        [
            (
                f"冻结案例窗 {case['window_seconds'][0]:g}–"
                f"{case['window_seconds'][1]:g} s；输出扣除 "
                f"{case['delay_samples']} samples 固定延迟。"
            ),
            "PASS / FAIL 仍由整链工程门限判定；图形指标不重新解释门限。",
        ]
    )
    report["figure_scope"]["fixed_delay_samples"] = case["delay_samples"]
    report["evaluation_case"] = {
        "asset_id": case["asset_id"],
        "case_id": case["case_id"],
        "detail": case["detail"],
        "window_seconds": case["window_seconds"],
        "delay_samples": case["delay_samples"],
        "before_sha256": bundle.before.sha256,
        "after_sha256": bundle.after.sha256,
        "manifest": str(bundle.manifest_path.relative_to(PROJECT_ROOT)),
    }
    return report, figures
