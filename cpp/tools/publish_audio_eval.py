#!/usr/bin/env python3
"""Publish a specific, audited evaluation batch and its paired audio."""
import argparse
import json
import hashlib
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATIVES = {"004-ns-p0", "012-aec-p0", "025-aec-p0", "030-combined-p0",
                   "036-stress-p0", "039-agc-p0", "041-bf-p0", "054-ns-p1"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative_ids(scenes, details, sharded):
    if not sharded:
        return REPRESENTATIVES | {scene["id"] for scene in details if not scene["passed"]}
    selected = set()
    for family in ("ns", "aec", "combined", "stress", "agc", "bf"):
        candidate = next((scene for scene in scenes if scene["family"] == family), None)
        if candidate:
            selected.add(candidate["id"])
    vehicle_noises = ("tcar", "car_hvac", "car_window_wind", "car_rain", "car_horn", "car_indicator", "car_engine_ramp", "babble")
    for noise in vehicle_noises:
        candidate = next((scene for scene in scenes if scene["noise"] == noise and scene["snr_db"] == 5), None)
        if candidate:
            selected.add(candidate["id"])
    for family in ("aec", "combined"):
        candidate = next((scene for scene in scenes if scene.get("playback_source") == "radio" and scene["family"] == family), None)
        if candidate:
            selected.add(candidate["id"])
    speakers = sorted({scene["near_speaker"] for scene in scenes})
    snrs = sorted({scene["snr_db"] for scene in scenes if scene["family"] == "ns" and scene["snr_db"] is not None})
    for index, speaker in enumerate(speakers):
        candidates = [scene for scene in scenes if scene["near_speaker"] == speaker]
        candidate = next((scene for scene in candidates if scene["family"] == "ns"
                          and scene["noise"] == vehicle_noises[index % len(vehicle_noises)]
                          and snrs and scene["snr_db"] == snrs[index % len(snrs)]), candidates[0])
        selected.add(candidate["id"])
    budget = len(selected) + 8
    for scene in details:
        if len(selected) >= budget:
            break
        if not scene["passed"]:
            selected.add(scene["id"])
    return selected


def publish(dataset, public, sharded=False):
    manifest = json.loads((dataset / "manifest.json").read_text())
    results = json.loads((dataset / "results.json").read_text())
    acceptance = json.loads((dataset / "acceptance.json").read_text())
    stratified = None
    if (dataset / "coverage-summary.json").exists():
        stratified = json.loads((dataset / "coverage-summary.json").read_text())
        if stratified["acceptance_sha256"] != digest(dataset / "acceptance.json"):
            raise ValueError("coverage summary belongs to another acceptance report")
    if acceptance["scope"] != "final full matrix" or not acceptance["controls_verified"]:
        raise ValueError("publication requires an audited full control matrix")
    if acceptance["dataset_cli_sha256"] != digest(dataset / "artifacts/itu_audio_cli"):
        raise ValueError("published acceptance belongs to another binary")
    if acceptance["rules_sha256"] != digest(ROOT / "cpp/evaluation/acceptance-v1.json"):
        raise ValueError("published thresholds differ from frozen rules")
    evidence = {"manifest.json": dataset / "manifest.json", "results.json": dataset / "results.json",
                "checker": ROOT / "cpp/tools/check_audio_acceptance.py",
                "evaluator": ROOT / "cpp/tools/run_audio_eval.py",
                "measurement_notes": ROOT / "cpp/evaluation/MEASUREMENT_NOTES.md"}
    if acceptance.get("evidence_sha256") != {name: digest(path) for name, path in evidence.items()}:
        raise ValueError("evaluation evidence changed after acceptance; rerun the checker")
    ids = {scene["id"] for scene in manifest["scenes"]}
    if ids != {scene["id"] for scene in acceptance["details"]}:
        raise ValueError("acceptance and audio inventory differ")
    checked = {scene["id"]: scene for scene in acceptance["details"]}
    by_id = defaultdict(list)
    for row in results:
        by_id[row["id"]].append(row)
    selected = representative_ids(manifest["scenes"], acceptance["details"], sharded)
    public.mkdir(parents=True, exist_ok=True)
    catalog = []
    for scene in manifest["scenes"]:
        entry = {key: value for key, value in scene.items() if key != "files"}
        if scene["noise"] == "none":
            entry["noise_origin"] = "none"
            entry["measured_input_snr_db"] = None
        entry["runs"] = [{"mode": row["mode"], "segments": row["segments"], "engine": row["engine"],
                          "hpf_stages": row["hpf_reference_diagnostic"]["stages"]}
                         for row in by_id[scene["id"]]]
        entry["audio_available"] = scene["id"] in selected
        if entry["audio_available"]:
            source = dataset / "cases" / scene["id"]
            full = next(row for row in by_id[scene["id"]] if row["mode"] == "full")
            if digest(source / "runs/full/result.after.wav") != full["output_sha256"]:
                raise ValueError("playable processed audio changed after evaluation")
            for name, record in scene["files"].items():
                if digest(source / name) != record["sha256"]:
                    raise ValueError("playable input changed after evaluation")
            target = public / "audio" / scene["id"]
            target.mkdir(parents=True, exist_ok=True)
            names = ["clean.wav", "ref.wav", "before.wav", "scene.json", "website-options.json"]
            for name in names:
                shutil.copyfile(source / name, target / name)
            shutil.copyfile(source / "runs/full/result.after.wav", target / "denoise.wav")
            entries = {name: target / name for name in names + ["denoise.wav"]}
            entries.update({name: source / name for name in ("playback.wav", "clean-beam.wav")})
            entries.update({"licenses/" + name: dataset / "sources" / name
                            for name in ("CC-BY-4.0.txt", "CC-BY-SA-3.0.txt")})
            if (dataset / "sources/CC0-1.0.txt").exists():
                entries["licenses/CC0-1.0.txt"] = dataset / "sources/CC0-1.0.txt"
            entries["ATTRIBUTION.md"] = dataset / "ATTRIBUTION.md"
            if (dataset / "artifacts/source-verification.json").exists():
                entries["artifacts/source-verification.json"] = dataset / "artifacts/source-verification.json"
            entries.update({"processing/" + name: source / "runs/full" / name
                            for name in ("result.json", "metrics.json", "invocation.json")})
            entries["acceptance-v1.json"] = ROOT / "cpp/evaluation/acceptance-v1.json"
            entries["MEASUREMENT_NOTES.md"] = ROOT / "cpp/evaluation/MEASUREMENT_NOTES.md"
            checksums = "\n".join(f"{digest(path)}  {name}" for name, path in entries.items()) + "\n"
            (target / "SHA256SUMS").write_text(checksums)
            with zipfile.ZipFile(target / "paired-audio.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name, path in entries.items():
                    archive.write(path, name)
                archive.write(target / "SHA256SUMS", "SHA256SUMS")
        if sharded:
            target = public / "scene-data" / f"{scene['id']}.json"
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(dict(scene=entry, acceptance=checked[scene["id"]]), separators=(",", ":"), allow_nan=False) + "\n")
            entry = {key: entry[key] for key in ("id", "family", "near_speaker", "far_speaker", "noise", "snr_db", "ser_db", "rt60_s", "echo_delay_ms", "mics", "near_dbfs", "stress", "seconds", "audio_available", "windows", "near_sex", "far_sex", "language", "noise_origin", "playback_source", "playback_reference_available", "playback_origin", "vehicle_condition") if key in entry}
            entry["runs"] = []
        catalog.append(entry)
    summary = json.loads((dataset / "summary.json").read_text())
    catalog_acceptance = acceptance
    if sharded:
        catalog_acceptance = dict(acceptance, details=[dict(id=scene["id"], passed=scene["passed"], gates=[])
                                                       for scene in acceptance["details"]])
    (public / "catalog.json").write_text(json.dumps({"scenes": catalog, "summary": summary,
        "environment": manifest["environment"], "acceptance": catalog_acceptance,
        "coverage": manifest.get("coverage"), "stratified": stratified, "sharded": sharded,
        "batch": {"name": dataset.name, "manifest_sha256": digest(dataset / "manifest.json"),
                  "evaluation_scope": manifest.get("evaluation_scope", "full expanded matrix"),
                  "profile": manifest.get("profile", "baseline-v1"),
                  "results_sha256": digest(dataset / "results.json"),
                  "acceptance_sha256": digest(dataset / "acceptance.json"),
                  "playable_scenes": len(selected & ids)}}, separators=(",", ":"), allow_nan=False) + "\n")
    for name in ("summary.json", "metrics.csv", "REPORT.md", "ATTRIBUTION.md", "acceptance.json", "acceptance.csv"):
        shutil.copyfile(dataset / name, public / name)
    if stratified is not None:
        for name in ("coverage-summary.json", "COVERAGE_REPORT.md"):
            shutil.copyfile(dataset / name, public / name)
    if (dataset / "artifacts/source-verification.json").exists():
        (public / "artifacts").mkdir(exist_ok=True)
        shutil.copyfile(dataset / "artifacts/source-verification.json", public / "artifacts/source-verification.json")
    shutil.copyfile(ROOT / "cpp/evaluation/acceptance-v1.json", public / "acceptance-v1.json")
    shutil.copyfile(ROOT / "cpp/evaluation/MEASUREMENT_NOTES.md", public / "MEASUREMENT_NOTES.md")
    shutil.copyfile(ROOT / "cpp/docs/QUALITY_REPAIR_REPORT.md", public / "QUALITY_REPAIR_REPORT.md")
    passed = acceptance["all_available_passed"]
    conclusion = ["# 仿真工程验收", "", f"批次：{dataset.name}",
                  f"范围：{manifest.get('evaluation_scope', 'full batch inventory')}",
                  f"结论：{'全部可用指标达标' if passed else '未全部达标'}。",
                  f"场景通过：{acceptance['passed_scenes']} / {acceptance['scenes']}。",
                  f"失败项：{acceptance['counts'].get('fail', 0)}；不可用值：{acceptance['counts'].get('unavailable', 0)}。",
                  "", "阈值是冻结的工程规则，不是 ITU 通用合格线。Android 实机声学验收尚未完成。",
                  "DTD 分类准确率、MOS/POLQA 不可用，未计为通过。", "",
                  "数字静音的精确衰减保持不可用；PCM16 保守下界独立校验同一抑制门槛。",
                  "计算定义与适用边界见 MEASUREMENT_NOTES.md。", "",
                  f"CLI SHA256：{acceptance['dataset_cli_sha256']}",
                  f"规则 SHA256：{acceptance['rules_sha256']}", "", "## 未通过项", "",
                  "| 场景 | 窗口 | 指标 | 实测 | 运算 | 阈值 |", "| --- | --- | --- | --- | --- | --- |"]
    for scene in acceptance["details"]:
        for gate in scene["gates"]:
            if gate["status"] == "fail" or (gate["status"] == "unavailable" and not gate["optional"]):
                conclusion.append(f"| {scene['id']} | {gate['segment']} | {gate['metric']} | {gate['value']} | {gate['operator']} | {gate['threshold']} |")
    (public / "评测结论.md").write_text("\n".join(conclusion) + "\n")
    print(f"published {len(catalog)} scenes, {len(selected & ids)} playable pairs; all_available_passed={passed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=ROOT / "public/evaluation")
    parser.add_argument("--sharded", action="store_true", help="load individual scene metrics on demand")
    args = parser.parse_args()
    publish(args.dataset.resolve(), args.public.resolve(), args.sharded)


if __name__ == "__main__":
    main()
