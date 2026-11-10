"""Join independently processed receive level cases, using actual cropped PCM."""
from __future__ import annotations

from collections.abc import Mapping
import json
import hashlib
import errno
import math
from pathlib import Path
import re

import soundfile as sf

from .acceptance_contract import contract_items, required_conditions
from .receive_linearity import canonical_sha256, evaluate_rlr_linearity_group
from .receive_configuration import normalize_receive_configuration, receive_configuration_sha256
from .receive_simulation import _third_octave_levels, get_synthetic_receive_evaluation, get_synthetic_receive_file


_LEVELS = ((0, "nominal"), (-5, "offset_-5"), (5, "offset_+5"))
_SHA256 = re.compile(r"[a-f0-9]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_identity(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "fixture_id": "software-only-virtual-drp",
        "fixture_type": "SIMULATED_CABIN",
        "calibration": "none",
        "synthetic_proxy": report.get("synthetic_proxy") is True,
        "formal_hats_evidence": False,
        "standard_compliance": False,
    }


def _acoustic_model_identity(
    report: Mapping[str, object],
    fixture: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    evidence_kind = report.get("evidence_kind")
    simulation = report.get("simulation")
    input_identity = report.get("input_identity")
    if report.get("synthetic_proxy") is not True:
        raise ValueError("接收报告必须明确标记为 software synthetic proxy")
    if evidence_kind == "synthetic_head":
        head_model = (
            simulation.get("head_model")
            if isinstance(simulation, Mapping)
            else None
        )
        if not isinstance(head_model, Mapping) or not head_model:
            raise ValueError("synthetic_head 接收报告缺少真实 head metadata")
        normalised_head = dict(head_model)
        return ({
            "evidence_kind": "synthetic_head",
            "head_model": normalised_head,
            "fixture": dict(fixture),
        }, normalised_head)
    if evidence_kind == "synthetic_ir":
        if not isinstance(input_identity, Mapping):
            raise ValueError("synthetic_ir 接收报告缺少 IR input identity")
        left_ir = input_identity.get("left_ir")
        right_ir = input_identity.get("right_ir")
        left_processing = input_identity.get("left_ir_processing")
        right_processing = input_identity.get("right_ir_processing")
        if not isinstance(left_ir, Mapping) or not isinstance(right_ir, Mapping):
            raise ValueError("synthetic_ir 接收报告缺少左右 IR 哈希")
        left_sha256 = left_ir.get("sha256")
        right_sha256 = right_ir.get("sha256")
        if (
            not isinstance(left_sha256, str)
            or _SHA256.fullmatch(left_sha256) is None
            or not isinstance(right_sha256, str)
            or _SHA256.fullmatch(right_sha256) is None
        ):
            raise ValueError("synthetic_ir 左右 IR SHA256 非法")
        if (
            not isinstance(left_processing, Mapping)
            or not isinstance(right_processing, Mapping)
        ):
            raise ValueError("synthetic_ir 接收报告缺少 IR processing identity")
        return ({
            "evidence_kind": "synthetic_ir",
            "left_ir_sha256": left_sha256,
            "right_ir_sha256": right_sha256,
            "left_ir_processing": dict(left_processing),
            "right_ir_processing": dict(right_processing),
            "fixture": dict(fixture),
        }, None)
    raise ValueError(f"不支持的接收证据类型: {evidence_kind!r}")


def _analysis_start(case: Mapping[str, object]) -> float:
    raw = case.get("analysis_start_s", 0.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("analysis_start_s 必须是有限非负数")
    start = float(raw)
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("analysis_start_s 必须是有限非负数")
    return start


def _case_has_linearity_contract(
    case: Mapping[str, object],
    contract_by_id: Mapping[str, Mapping[str, object]],
    *,
    mode: str,
    level: int,
    family: str,
) -> bool:
    raw_item_ids = case.get("item_ids")
    conditions = case.get("conditions")
    if not isinstance(raw_item_ids, list) or not isinstance(conditions, Mapping):
        return False
    for item_id in raw_item_ids:
        if not isinstance(item_id, str):
            continue
        row = contract_by_id.get(item_id)
        if not isinstance(row, Mapping):
            continue
        if (
            row.get("family") != family
            or row.get("sheet") != mode
            or row.get("direction") != "receive"
        ):
            continue
        expected = required_conditions(dict(row))
        if (
            expected.get("mode") != mode
            or expected.get("input_level_db") != level
            or expected != dict(conditions)
        ):
            continue
        return True
    return False


def _verified_receive_report(report_id: str) -> dict:
    report = get_synthetic_receive_evaluation(report_id)
    if not isinstance(report, dict):
        raise ValueError("缺少可验证的接收左右耳输出")
    expected = report.get("result_sha256")
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"result", "result_sha256"}
    }
    if (
        not isinstance(expected, str)
        or canonical_sha256(payload) != expected
    ):
        raise ValueError(f"{report_id} 接收报告内容身份已改变")
    result = report.get("result")
    if not isinstance(result, Mapping) or result.get("sha256") != expected:
        raise ValueError(f"{report_id} 接收报告 result 身份不一致")
    return report


def _verified_case_assessment(
    snapshot: Mapping[str, object],
    case: Mapping[str, object],
    assessment_path: Path,
) -> tuple[dict, str, str, str]:
    records = [
        record
        for record in snapshot.get("cases", [])
        if isinstance(record, Mapping) and record.get("case_id") == case["id"]
    ]
    if len(records) != 1:
        raise ValueError(f"{case['id']} 需要唯一的批次 case 记录")
    record = records[0]
    if record.get("status") != "completed":
        raise ValueError(f"{case['id']} 批次 case 未完成")
    if record.get("conditions") != case["conditions"]:
        raise ValueError(f"{case['id']} 批次 case 条件不一致")
    assessment_report_id = record.get("report_id")
    if not isinstance(assessment_report_id, str) or not assessment_report_id:
        raise ValueError(f"{case['id']} 批次 case 缺少 report id")
    expected_sha256 = record.get("assessment_sha256")
    if not isinstance(expected_sha256, str):
        raise ValueError(f"{case['id']} 批次 case 缺少 assessment hash")
    actual_sha256 = _sha256_file(assessment_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{case['id']} assessment.json 身份已改变")
    evidence = json.loads(assessment_path.read_text())
    if evidence.get("case_id") != case["id"]:
        raise ValueError(f"{case['id']} assessment case id 不一致")
    if evidence.get("report_id") != assessment_report_id:
        raise ValueError(f"{case['id']} assessment report id 不一致")
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{case['id']} 批次 case 缺少原始 execution")
    if evidence.get("execution") != execution:
        raise ValueError(f"{case['id']} assessment execution 身份不一致")
    item_report_ids = {
        item.get("model_details", {}).get("receive_report_id")
        for item in evidence.get("items", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("model_details"), Mapping)
        and item["model_details"].get("receive_report_id")
    }
    if (
        len(item_report_ids) != 1
        or not isinstance(next(iter(item_report_ids)), str)
        or not next(iter(item_report_ids))
    ):
        raise ValueError(f"{case['id']} assessment 接收报告绑定不一致")
    receive_report_id = next(iter(item_report_ids))
    return (
        record,
        assessment_report_id,
        receive_report_id,
        actual_sha256,
    )


def _verified_configuration_binding(snapshot: dict, dataset: dict, selected: list[dict]) -> str | None:
    """Keep old evidence readable, but require complete bindings for new runs."""
    fields = ("receive_simulation", "receive_ir_sha256", "receive_configuration_sha256")
    containers = [snapshot, *(item["execution"] for item in selected)]
    declared = dataset.get("receive_simulation", dataset["manifest"].get("receive_simulation"))
    if "receive_simulation" not in dataset["manifest"] and not any(
        key in value for value in containers for key in fields
    ):
        return None  # Explicitly legacy evidence; no configuration digest is invented.
    if any(any(key not in value for key in fields) for value in containers):
        raise ValueError("三档接收声学配置绑定不完整")
    expected_config = normalize_receive_configuration(declared)
    expected_hashes = snapshot["receive_ir_sha256"]
    digest = receive_configuration_sha256(expected_config, expected_hashes)
    for container in containers:
        config = normalize_receive_configuration(container["receive_simulation"])
        if (
            config != expected_config
            or container["receive_ir_sha256"] != expected_hashes
            or container["receive_configuration_sha256"] != digest
            or receive_configuration_sha256(config, container["receive_ir_sha256"]) != digest
        ):
            raise ValueError("三档接收声学配置身份不一致")
    for item in selected:
        identity = item["report"]["input_identity"]
        if expected_config["model"] == "paired_ir":
            if item["report"].get("evidence_kind") != "synthetic_ir":
                raise ValueError("配置与接收报告声学模型不一致")
            for role in ("left_ir", "right_ir"):
                if identity[role]["sha256"] != expected_hashes[role]:
                    raise ValueError("配置与接收报告 IR 身份不一致")
                assets = dataset.get("assets")
                if isinstance(assets, Mapping):
                    if assets.get(expected_config[role], {}).get("sha256") != expected_hashes[role]:
                        raise ValueError("配置与数据集 IR 身份不一致")
        else:
            if item["report"].get("evidence_kind") != "synthetic_head" or any(
                not math.isclose(identity[key], expected_config[key], rel_tol=0, abs_tol=1e-9)
                for key in ("azimuth_deg", "head_radius_m", "speed_of_sound_mps")
            ):
                raise ValueError("配置与接收报告 head 参数不一致")
        bindings = [
            row.get("evidence_binding", {}) for row in snapshot["items"]
            if row.get("evidence_binding", {}).get("case_id") == item["case"]["id"]
        ]
        if not bindings or any(
            value.get("receive_configuration_sha256") != digest
            or value.get("receive_ir_sha256") != expected_hashes for value in bindings
        ):
            raise ValueError("测项接收声学配置绑定不一致")
    return digest


def finish_linearity(snapshot: dict, dataset: dict, destination: Path) -> None:
    by_id = {item["item_id"]: item for item in snapshot["items"]}
    cases = dataset["manifest"]["cases"]
    contract_by_id = {
        str(row["id"]): row for row in contract_items()
    }
    for mode in dict.fromkeys(item["mode"] for item in snapshot["items"]):
        targets = [item for item in snapshot["items"] if item["mode"] == mode and item["family"] == "rlr_linearity"]
        if not targets:
            continue
        candidates = [case for case in cases if case["conditions"]["mode"] == mode
                      and case["conditions"]["direction"] == "receive"
                      and case["conditions"]["scene"] == "baseline"
                      and case["conditions"]["volume"] == "nominal"
                      and case["conditions"]["window_ms"] is None]
        runs = []
        sources = []
        try:
            selected = []
            for level, level_id in _LEVELS:
                family = "rlr" if level == 0 else "rlr_linearity"
                matches = [
                    case
                    for case in candidates
                    if _case_has_linearity_contract(
                        case,
                        contract_by_id,
                        mode=mode,
                        level=level,
                        family=family,
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(f"{mode} 需要唯一的 {level_id} 接收运行")
                case = matches[0]
                case_file = destination / "cases" / case["id"] / "assessment.json"
                (
                    case_record,
                    assessment_report_id,
                    receive_report_id,
                    assessment_sha256,
                ) = (
                    _verified_case_assessment(snapshot, case, case_file)
                )
                execution = case_record["execution"]
                execution_sha256 = canonical_sha256(execution)
                report = _verified_receive_report(receive_report_id)
                if report.get("mode") != mode:
                    raise ValueError(f"{level_id} 接收报告 mode 与合同不一致")
                fixture = _fixture_identity(report)
                acoustic_identity, head_model = _acoustic_model_identity(
                    report,
                    fixture,
                )
                artifacts = report.get("artifacts")
                if not isinstance(artifacts, Mapping):
                    raise ValueError(f"{level_id} 接收报告缺少 artifact 身份")
                left_path = get_synthetic_receive_file(
                    receive_report_id,
                    "virtual_left.wav",
                )
                right_path = get_synthetic_receive_file(
                    receive_report_id,
                    "virtual_right.wav",
                )
                artifact_hashes = {}
                for role, path in (("left", left_path), ("right", right_path)):
                    artifact = artifacts.get(role)
                    if (
                        path is None
                        or not isinstance(artifact, Mapping)
                        or _sha256_file(path) != artifact.get("sha256")
                    ):
                        raise ValueError(f"{role} 接收产物身份已改变")
                    artifact_hashes[role] = str(artifact["sha256"])
                left, rate = sf.read(left_path, dtype="float64", always_2d=False)
                right, right_rate = sf.read(right_path, dtype="float64", always_2d=False)
                if int(rate) != int(right_rate):
                    raise ValueError("左右耳采样率不一致")
                if getattr(left, "ndim", 0) != 1 or getattr(right, "ndim", 0) != 1:
                    raise ValueError("接收左右耳产物必须是单声道")
                if len(left) != len(right):
                    raise ValueError("接收左右耳产物帧数不一致")
                selected.append({
                    "case": case,
                    "level_id": level_id,
                    "report": report,
                    "assessment_report_id": assessment_report_id,
                    "receive_report_id": receive_report_id,
                    "fixture": fixture,
                    "head_model": head_model,
                    "acoustic_identity": acoustic_identity,
                    "left": left,
                    "right": right,
                    "rate": int(rate),
                    "artifact_hashes": artifact_hashes,
                    "assessment_sha256": assessment_sha256,
                    "execution_sha256": execution_sha256,
                    "execution": execution,
                })
            rates = {item["rate"] for item in selected}
            if len(rates) != 1:
                raise ValueError("三档接收输出采样率不一致")
            rate = rates.pop()
            acoustic_identities = [item["acoustic_identity"] for item in selected]
            if any(
                identity != acoustic_identities[0]
                for identity in acoustic_identities[1:]
            ):
                raise ValueError("三档接收输出声学模型身份不一致")
            configuration_digest = _verified_configuration_binding(snapshot, dataset, selected)
            volumes = {item["case"]["conditions"]["volume"] for item in selected}
            if len(volumes) != 1:
                raise ValueError("三档接收输出 volume 不一致")
            volume = volumes.pop()
            start = max(_analysis_start(item["case"]) for item in selected)
            duration = min(len(item["left"]) / rate for item in selected)
            if not 0 <= start < duration:
                raise ValueError("线性残差分析窗不在有效音频内")
            lo, hi = round(start * rate), round(duration * rate)
            if lo >= hi:
                raise ValueError("线性残差分析窗没有有效样本")
            analysis_window = [lo / rate, hi / rate]
            fixture = selected[0]["fixture"]
            for item in selected:
                case = item["case"]
                level_id = item["level_id"]
                report = item["report"]
                assessment_report_id = item["assessment_report_id"]
                receive_report_id = item["receive_report_id"]
                left = item["left"]
                right = item["right"]
                head_model = item["head_model"]
                acoustic_identity = item["acoustic_identity"]
                ir_identity = (
                    acoustic_identity
                    if acoustic_identity["evidence_kind"] == "synthetic_ir"
                    else None
                )
                cropped_bands = _third_octave_levels(left[lo:hi], right[lo:hi], rate)
                case_binding = next(
                    (
                        candidate["evidence_binding"]
                        for candidate in by_id.values()
                        if candidate.get("evidence_binding", {}).get("case_id") == case["id"]
                    ),
                    None,
                )
                if not isinstance(case_binding, Mapping):
                    raise ValueError(f"{case['id']} 缺少证据绑定")
                if case_binding.get("input_sha256") != item["execution"].get(
                    "input_sha256"
                ):
                    raise ValueError(f"{case['id']} 输入执行绑定不一致")
                run_conditions = {
                    "nominal_volume": volume,
                    "fixture": fixture,
                    "ir": ir_identity,
                    "head_model": head_model,
                    "acoustic_model_identity": acoustic_identity,
                    "sample_rate_hz": rate,
                    "duration_s": round(duration, 12),
                    "active_speech_window_s": analysis_window,
                }
                metadata = {
                    "evaluation_id": receive_report_id,
                    "assessment_report_id": assessment_report_id,
                    "case_id": case["id"],
                    "case_conditions": dict(case["conditions"]),
                    "window_applied": True,
                    "analysis_window_s": analysis_window,
                    "window_source": "cropped_virtual_ear_wav",
                    "analysis_sample_bounds": {
                        "left": [lo, hi],
                        "right": [lo, hi],
                    },
                    "artifact_sha256": dict(item["artifact_hashes"]),
                    "artifact_frame_counts": {
                        "left": len(left),
                        "right": len(right),
                    },
                    "acoustic_model_identity": acoustic_identity,
                    "fixture_identity": fixture,
                    "input_sha256": dict(case_binding["input_sha256"]),
                    "assessment_sha256": item["assessment_sha256"],
                    "execution_sha256": item["execution_sha256"],
                    "execution_binding": {
                        **({
                            "receive_configuration_sha256": configuration_digest,
                            "receive_ir_sha256": dict(item["execution"]["receive_ir_sha256"]),
                            "configuration_binding_status": "verified",
                        } if configuration_digest is not None else {}),
                        "batch_id": item["execution"].get("batch_id"),
                        "dataset_id": item["execution"].get("dataset_id"),
                        "cli_sha256": item["execution"].get("cli_sha256"),
                        "catalog_sha256": item["execution"].get("catalog_sha256"),
                        "input_sha256": dict(
                            item["execution"].get("input_sha256", {})
                        ),
                        "output_sha256": dict(
                            item["execution"].get("output_sha256", {})
                        ),
                    },
                    "report_result_sha256": report.get("result_sha256"),
                }
                run = {
                    **report,
                    "synthetic_proxy": True,
                    "standard_compliance": False,
                    "formal_hats_evidence": False,
                    "run_id": report["evaluation_id"],
                    "level_id": level_id,
                    "clipping": bool(max(abs(left).max(), abs(right).max()) >= 1),
                    "conditions": run_conditions,
                    "metadata": metadata,
                    "analysis_window_s": analysis_window,
                    "window_applied": True,
                    "metrics": {
                        **report["metrics"],
                        "third_octave_band_levels_db": cropped_bands,
                        "third_octave_band_levels_dbfs": cropped_bands,
                        "analysis_window_s": analysis_window,
                        "window_applied": True,
                    },
                }
                runs.append(run)
                sources.append({
                    "case_id": case["id"],
                    "report_id": receive_report_id,
                    "assessment_report_id": assessment_report_id,
                    "input_sha256": dict(case_binding["input_sha256"]),
                    "artifact_sha256": dict(item["artifact_hashes"]),
                    "assessment_sha256": item["assessment_sha256"],
                    "execution_sha256": item["execution_sha256"],
                    "execution_binding": dict(metadata["execution_binding"]),
                    "analysis_window_s": analysis_window,
                    "analysis_sample_bounds": [lo, hi],
                    "window_source": "cropped_virtual_ear_wav",
                })
            result = evaluate_rlr_linearity_group(runs, {
                "mode": mode,
                "nominal_volume": volume,
                "fixture": fixture,
                "ir": (
                    selected[0]["acoustic_identity"]
                    if selected[0]["acoustic_identity"]["evidence_kind"] == "synthetic_ir"
                    else None
                ),
                "head_model": selected[0]["head_model"],
                "acoustic_model_identity": selected[0]["acoustic_identity"],
                "sample_rate_hz": rate,
                "duration_s": round(duration, 12),
                "active_speech_window_s": analysis_window,
            }, include_canonical_hash=True)
            if not result.get("group_valid"):
                raise ValueError("; ".join(result.get("reasons", ["三档条件不一致"])))
            result.update(
                synthetic_proxy=True,
                standard_compliance=False,
                formal_hats_evidence=False,
                receive_configuration_sha256=configuration_digest,
                configuration_binding_status="verified" if configuration_digest else "legacy_unbound",
            )
            result["canonical_sha256"] = canonical_sha256({
                key: value
                for key, value in result.items()
                if key != "canonical_sha256"
            })
            (destination / f"{mode}-linearity.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for item in targets:
                value = result["items"][item["source_cell"]]["residual_db"]
                item.update(value=value, metric="receive_level_linearity_residual_db", status="review_required",
                            evidence_grade="engineering_proxy", reason="独立三档 VCP 接收输出的电平线性残差；未作 P.79/DRP 标定")
                model_details = item.get("model_details")
                if not isinstance(model_details, dict):
                    model_details = {}
                model_details.update({"linearity": result, "source_runs": sources})
                item["model_details"] = model_details
                item["evidence_binding"]["source_runs"] = sources
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                raise
            for item in targets:
                if item["status"] != "invalid":
                    item.update(value=None, status="not_measured", evidence_grade="not_measured",
                                reason=f"三档线性模型不可用: {error}")
