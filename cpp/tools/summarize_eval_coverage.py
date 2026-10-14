#!/usr/bin/env python3
"""Stratify audited scene verdicts without changing any acceptance thresholds."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def vehicle(scene):
    return scene["noise"] == "tcar" or scene["noise"].startswith("car_") or scene.get("playback_source") == "radio"


def summarize(manifest, acceptance):
    scenes = manifest["scenes"]
    details = acceptance["details"]
    ids = [s["id"] for s in scenes]
    checked_ids = [s["id"] for s in details]
    if not ids or len(set(ids)) != len(ids) or len(set(checked_ids)) != len(checked_ids) or set(ids) != set(checked_ids):
        raise ValueError("coverage requires the complete unique audited scene inventory")
    checked = {s["id"]: s for s in details}

    def group(items):
        counts = Counter(g["status"] for s in items for g in checked[s["id"]]["gates"])
        failures = Counter(g["segment"] + "." + g["metric"] for s in items
                           for g in checked[s["id"]]["gates"] if g["status"] == "fail")
        return dict(scenes=len(items), passed_scenes=sum(checked[s["id"]]["passed"] for s in items),
                    failed_scenes=sum(not checked[s["id"]]["passed"] for s in items),
                    counts={k: counts[k] for k in ("pass", "fail", "unavailable")},
                    failed_metrics=dict(failures.most_common()))

    def dimension(key):
        values = sorted({s[key] for s in scenes if s[key] is not None})
        if any(s[key] is None for s in scenes):
            values.append(None)
        return [dict(value=value, **group([s for s in scenes if s[key] == value])) for value in values]

    total = group(scenes)
    if acceptance["scenes"] != total["scenes"] or acceptance["passed_scenes"] != total["passed_scenes"] or any(
        acceptance["counts"].get(k, 0) != v for k, v in total["counts"].items()
    ):
        raise ValueError("coverage totals disagree with audited acceptance")
    ns = [s for s in scenes if s["family"] == "ns"]
    vehicle_scenes = [s for s in scenes if vehicle(s)]
    sweeps = defaultdict(list)
    for scene in ns:
        sweeps[(scene["near_speaker"], scene["noise"])].append(scene)
    ranges = [max(s.get("effective_near_dbfs", s.get("near_dbfs", 0)) for s in sweep)
              - min(s.get("effective_near_dbfs", s.get("near_dbfs", 0)) for s in sweep)
              for sweep in sweeps.values()]
    snr_errors = [abs(s["measured_input_snr_db"] - s["snr_db"]) for s in ns
                  if s.get("measured_input_snr_db") is not None]
    diagnostics = dict(ns_sweeps=len(sweeps), peak_scaled_ns_scenes=sum(s.get("common_gain", 1) < 1 for s in ns),
                       ns_sweeps_with_level_range_over_1db=sum(value > 1 for value in ranges),
                       max_ns_sweep_level_range_db=max(ranges, default=0),
                       max_ns_measured_snr_error_db=max(snr_errors, default=None),
                       no_noise_scenes_with_legacy_snr=sum(s["noise"] == "none" and s.get("measured_input_snr_db") is not None for s in scenes))
    return dict(schema_version=1, profile=manifest.get("profile", "baseline-v1"),
                total=total, vehicle=group(vehicle_scenes),
                input_diagnostics=diagnostics,
                vehicle_noise_types=sorted({s["noise"] for s in vehicle_scenes if s["noise"] != "none"}),
                by_playback=[dict(value=value, **group([s for s in scenes if
                    (s.get("playback_source", "far_speech") if s["family"] not in ("ns", "agc") else "none") == value]))
                    for value in sorted({s.get("playback_source", "far_speech") if s["family"] not in ("ns", "agc") else "none" for s in scenes})],
                vehicle_families={family: group([s for s in vehicle_scenes if s["family"] == family])
                                  for family in sorted({s["family"] for s in vehicle_scenes})},
                by_family=dimension("family"), by_noise=dimension("noise"),
                by_snr_db=dimension("snr_db"), by_near_speaker=dimension("near_speaker"),
                ns_noise_snr=[dict(noise=noise, snr_db=snr,
                                  **group([s for s in ns if s["noise"] == noise and s["snr_db"] == snr]))
                              for noise, snr in sorted({(s["noise"], s["snr_db"]) for s in ns})])


def write_report(root, report):
    (root / "coverage-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    total, car = report["total"], report["vehicle"]
    lines = ["# 分组评测结果", "",
             f"全部场景：{total['passed_scenes']} / {total['scenes']} 通过；失败指标 {total['counts']['fail']}。",
             f"车载场景：{car['passed_scenes']} / {car['scenes']} 通过；失败指标 {car['counts']['fail']}。", "",
             "采用原冻结工程阈值；失败场景和不可用指标均保留，分组统计不改变验收结论。",
             "车载组合以 DEMAND TCAR 实录为底噪叠加可控干扰，不代表实测车速、车型或空调档位。", ""]
    diagnostic = report["input_diagnostics"]
    lines += ["## 输入与测量边界", "",
              f"NS 扫描组 {diagnostic['ns_sweeps']}；使用整体峰值保护缩放的 NS 场景 {diagnostic['peak_scaled_ns_scenes']}。",
              f"同一扫描组内近端电平极差超过 1 dB 的组数 {diagnostic['ns_sweeps_with_level_range_over_1db']}；最大极差 {diagnostic['max_ns_sweep_level_range_db']:.3f} dB。",
              f"NS 实测 SNR 与标称 SNR 的最大偏差：{diagnostic['max_ns_measured_snr_error_db']} dB。",
              "同一扫描使用同一原始语音、噪声片段和随机种子；防削波会同时缩放各分量，绝对近端电平不是控制变量。",
              f"无噪声场景中 {diagnostic['no_noise_scenes_with_legacy_snr']} 条冻结元数据保留了旧生成器的非适用 SNR 数值。网站将其显示为不适用，冻结证据不改写。",
              "当前 C++ 链路关闭了 WebRTC IIR 高通，固定线性相位低切的延时按报告补偿。高通匹配阶段为 0 的指标与原始指标相同，不是独立诊断证据。", ""]
    for title, key in (("技术维度", "by_family"), ("背景噪声条件", "by_noise"), ("AEC 播放来源", "by_playback"),
                       ("SNR / dB", "by_snr_db"), ("近端说话人", "by_near_speaker")):
        lines += [f"## {title}", "", "| 条件 | 通过 / 全部场景 | 失败指标 | 不可用指标 |",
                  "| --- | --- | --- | --- |"]
        for row in report[key]:
            value = "不适用" if row["value"] is None else row["value"]
            lines.append(f"| {value} | {row['passed_scenes']} / {row['scenes']} | {row['counts']['fail']} | {row['counts']['unavailable']} |")
        lines.append("")
    lines += ["## NS 噪声与 SNR 交叉覆盖", "", "| 噪声 | SNR / dB | 通过 / 全部场景 | 失败指标 |",
              "| --- | --- | --- | --- |"]
    for row in report["ns_noise_snr"]:
        lines.append(f"| {row['noise']} | {row['snr_db']} | {row['passed_scenes']} / {row['scenes']} | {row['counts']['fail']} |")
    (root / "COVERAGE_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    acceptance = json.loads((root / "acceptance.json").read_text())
    if acceptance["scope"] != "final full matrix" or not acceptance["controls_verified"]:
        raise ValueError("stratification requires an audited full matrix with verified controls")
    for name in ("manifest.json", "results.json"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != acceptance["evidence_sha256"][name]:
            raise ValueError(f"audited evidence changed: {name}")
    manifest = json.loads((root / "manifest.json").read_text())
    report = summarize(manifest, acceptance)
    report["acceptance_sha256"] = hashlib.sha256((root / "acceptance.json").read_bytes()).hexdigest()
    write_report(root, report)
    print(json.dumps({k: report[k] for k in ("total", "vehicle")}, indent=2))


if __name__ == "__main__":
    main()
