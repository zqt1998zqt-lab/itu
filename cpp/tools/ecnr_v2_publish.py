#!/usr/bin/env python3
"""Verify a complete ECNR experiment and copy six representative original stage streams."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import soundfile as sf
from ecnr_campaign_calibration import sha
from ecnr_v2_evidence import verify_sequence
from run_ecnr_v2_matrix import PILOT, compare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URL_PREFIX = "/evaluation/ecnr-v2/"
EXPECTED_SEQUENCES = 30
EXPECTED_PHASES = 270
EXPECTED_BASELINE = 184
DURATION_S = 235.0
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

def sequence_name(configuration):
    rate, microphones, variant, pair = configuration
    return f"r{rate}-m{microphones}-{variant}-pair-{pair:02}"

def relative_stage_url(value):
    if not isinstance(value, str) or not value.startswith(URL_PREFIX):
        raise ValueError("stage must use the established local evaluation URL")
    url = urlsplit(value)
    if url.scheme or url.netloc or url.query or url.fragment or "%" in value or "\\" in value:
        raise ValueError("encoded/external/ambiguous stage URL refused")
    relative = PurePosixPath(value[len(URL_PREFIX):])
    parts = relative.parts
    if len(parts) != 4 or parts[0] != "audio" or parts[2] not in ("baseline", "candidate") or any(
            not SAFE_NAME.fullmatch(part) or part in (".", "..") for part in parts):
        raise ValueError("stage URL path is outside its sequence/side")
    if not parts[3].endswith(".wav"):
        raise ValueError("only declared stage WAV files may be published")
    return relative

def validate_target(destination):
    expected = PROJECT_ROOT / "public/evaluation/ecnr-v2"
    destination = Path(destination)
    if destination.absolute() != expected.absolute():
        raise ValueError("publication target must be the existing ECNR static entry")
    for node in (destination, *destination.parents):
        if node == PROJECT_ROOT.parent:
            break
        if node.is_symlink():
            raise ValueError("publication destination or parent is a symlink")
    if destination.resolve() != expected.absolute():
        raise ValueError("publication target resolution differs")
    return destination

def checked_reports(item):
    report_data = {}
    build_root = (PROJECT_ROOT / "build").resolve()
    for side in ("baseline", "candidate"):
        path = Path(item[side + "_report"]).resolve()
        if not path.is_relative_to(build_root) or path.name != "stream-evaluation.json":
            raise ValueError("source report is outside the frozen build tree")
        data = json.loads(path.read_text())
        for name in {**data["identity"]["files"], **data["identity"]["output_files"]}:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or not (path.parent / name).resolve().is_relative_to(path.parent):
                raise ValueError("source inventory contains an external path")
        report_data[side] = data
    verify_sequence(item)
    return report_data

def validate_complete(report):
    if report.get("schema_version") != 1 or report.get("state") != "completed" or not report.get("complete") or not report.get("identity_verified"):
        raise ValueError("only complete and identity-verified experiments can be imported")
    q = report.get("quality", {})
    if q.get("total_phases") != EXPECTED_PHASES or q.get("baseline_pass") != EXPECTED_BASELINE:
        raise ValueError("full 270-phase pairing against the 184-phase baseline required")
    indexed = report.get("identity_sequences", [])
    if len(indexed) != EXPECTED_SEQUENCES or len({item["id"] for item in indexed}) != EXPECTED_SEQUENCES:
        raise ValueError("all thirty unique identity sequences required")
    if report.get("perceptual", {}).get("status") != "completed":
        raise ValueError("complete paired perceptual measurements required")
    for item in indexed:
        if not isinstance(item["id"], str) or not SAFE_NAME.fullmatch(item["id"]):
            raise ValueError("invalid identity sequence id")
    configurations = report["identity"].get("configurations", [])
    if len(configurations) != EXPECTED_SEQUENCES or {sequence_name(c) for c in configurations} != {item["id"] for item in indexed}:
        raise ValueError("frozen configuration and sequence inventory differ")
    previews = {item["id"]: item for item in report.get("sequences", [])}
    if len(previews) != EXPECTED_SEQUENCES or set(previews) != {item["id"] for item in indexed}:
        raise ValueError("complete private stage inventory required before representative import")
    wanted = [sequence_name(c) for c in PILOT]
    if not set(wanted).issubset(previews):
        raise ValueError("six fixed development streams are absent")
    return indexed, previews, wanted

def matrix_binding(web, report):
    path = web.parent / "report.json"
    private = json.loads(path.read_text())
    if not private.get("complete") or not private.get("identity_verified") or private.get("state") != "completed":
        raise ValueError("current private matrix is incomplete")
    for key in ("identity", "quality", "pairs", "calibration", "perceptual", "audits", "coverage", "runtime"):
        if private.get(key) != report.get(key):
            raise ValueError("web evidence is stale relative to private matrix: " + key)
    if private.get("sequences") != report.get("identity_sequences"):
        raise ValueError("web identity index differs from current private matrix")
    return path, sha(path)

def validate_quality(report, checked):
    pairs = []
    failures = []
    expected = {(p["sequence_id"], p["label"]): p for p in report.get("pairs", [])}
    if len(expected) != EXPECTED_PHASES or len(report.get("pairs", [])) != EXPECTED_PHASES:
        raise ValueError("full unique paired phase inventory required")
    for sid, data in checked.items():
        matched = compare(data["baseline"], data["candidate"])
        if len(matched) != 9:
            raise ValueError("each complete stream must contain nine frozen phases")
        for pair in matched:
            key = (sid, pair["label"])
            old = expected.get(key)
            if old is None or any(old.get(name) != pair[name] for name in (
                    "baseline_pass", "candidate_pass", "new_failures", "repaired_gates", "candidate_failures")):
                raise ValueError("paired phase summary differs from frozen source gates")
            pairs.append(pair)
            failures.extend(dict(sequence_id=sid, label=pair["label"], **gate)
                            for gate in pair["candidate_failures"])
    sort_failure = lambda f: (f["sequence_id"], f["label"], f["segment"], f["metric"])
    if sorted(report.get("failures", []), key=sort_failure) != sorted(failures, key=sort_failure):
        raise ValueError("all original candidate failures must remain visible")
    actual = dict(baseline_pass=sum(p["baseline_pass"] for p in pairs),
        candidate_pass=sum(p["candidate_pass"] for p in pairs), total_phases=len(pairs),
        phase_wins=sum(not p["baseline_pass"] and p["candidate_pass"] for p in pairs),
        phase_regressions=sum(p["baseline_pass"] and not p["candidate_pass"] for p in pairs),
        repaired_gates=sum(len(p["repaired_gates"]) for p in pairs),
        new_failed_gates=sum(len(p["new_failures"]) for p in pairs))
    if report["quality"] != actual:
        raise ValueError("published quality counts do not match all original frozen gates")

def stage_source(web, sequence_id, stage, source_data, item):
    relative = relative_stage_url(stage["file"])
    if relative.parts[1] != sequence_id:
        raise ValueError("stage URL belongs to another sequence")
    side = relative.parts[2]
    if not stage["name"].startswith(side + "/"):
        raise ValueError("stage display side and source side differ")
    folder = Path(item[side + "_report"]).resolve().parent
    source = (web / Path(str(relative))).resolve()
    if not source.is_file() or not source.is_relative_to(folder):
        raise ValueError("stage link does not resolve inside its verified stream")
    inventory = {**source_data[side]["identity"]["files"], **source_data[side]["identity"]["output_files"]}
    filename = str(source.relative_to(folder))
    expected = inventory.get(filename)
    if not expected or stage.get("sha256") != expected or sha(source) != expected:
        raise ValueError("stage content does not match the verified source inventory")
    info = sf.info(source)
    if stage["sample_rate"] != info.samplerate or stage["channels"] != info.channels:
        raise ValueError("stage must retain original full stream rate/channels/duration")
    if not math.isclose(stage["duration_s"], info.duration, abs_tol=1 / info.samplerate):
        raise ValueError("declared stage duration differs")
    delay = stage["fixed_delay_samples"]
    trimmed = stage["trimmed_leading_samples"]
    if not isinstance(delay, int) or delay < 0 or delay >= info.frames or trimmed != 0:
        raise ValueError("original full stage delay/trim metadata required")
    if not math.isclose(info.duration - delay / info.samplerate, DURATION_S, abs_tol=1 / info.samplerate):
        raise ValueError("stage must retain all input frames plus declared fixed delay")
    offset = (trimmed - delay) / info.samplerate * 1000
    if not math.isclose(stage["timeline_offset_ms"], offset, abs_tol=1e-7):
        raise ValueError("stage input timeline metadata disagrees")
    return source, relative, expected, info.frames

def ensure_space(destination, required, reserve_bytes):
    nearest = destination
    while not nearest.exists():
        nearest = nearest.parent
    if shutil.disk_usage(nearest).free < reserve_bytes + required:
        raise RuntimeError("disk reserve reached; current public report and all history are preserved")

def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def add_baseline_values(report, checked):
    result = []
    for failure in report.get("failures", []):
        item = copy.deepcopy(failure)
        baseline = checked[item["sequence_id"]]["baseline"]
        matches = [gate for phase in baseline["details"] if phase["label"] == item["label"]
                   for gate in phase["gates"] if gate["segment"] == item["segment"] and gate["metric"] == item["metric"]]
        if len(matches) != 1:
            raise ValueError("failure baseline display value is ambiguous")
        item["baseline_value"] = matches[0]["value"]
        result.append(item)
    report["failures"] = result

def publish_static(web, destination=None, reserve_bytes=3 * 1024**3):
    web = Path(web).resolve()
    if not web.is_relative_to((PROJECT_ROOT / "build").resolve()):
        raise ValueError("input web report must be inside the frozen build tree")
    destination = validate_target(destination or PROJECT_ROOT / "public/evaluation/ecnr-v2")
    if reserve_bytes < 3 * 1024**3:
        raise ValueError("at least 3 GiB must remain available")
    source_report = web / "report.json"
    original = source_report.read_bytes()
    report = json.loads(original)
    indexed, previews, wanted = validate_complete(report)
    matrix_report, matrix_hash = matrix_binding(web, report)
    source_hash = hashlib.sha256(original).hexdigest()
    release_id = "report-" + source_hash[:20] + "-" + sha(__file__)[:8]
    checked = {}
    for number, item in enumerate(indexed, 1):
        data = checked_reports(item)
        if data["baseline"]["identity"]["cli_sha256"] != report["identity"]["baseline_cli_sha256"] or data["candidate"]["identity"]["cli_sha256"] != report["identity"]["cli_sha256"]:
            raise ValueError("source stream binary differs from campaign identity")
        checked[item["id"]] = data
        if number % 5 == 0:
            print(f"verified source streams {number}/{EXPECTED_SEQUENCES}", flush=True)
    validate_quality(report, checked)
    result = copy.deepcopy(report)
    add_baseline_values(result, checked)
    index = {item["id"]: item for item in indexed}
    selected = []
    copies = {}
    for sid in wanted:
        sequence = copy.deepcopy(previews[sid])
        if not sequence.get("stages") or not {"baseline/final-output", "candidate/final-output"}.issubset({stage["name"] for stage in sequence["stages"]}):
            raise ValueError("both complete final outputs are required in every representative stream")
        for stage in sequence["stages"]:
            source, relative, expected, frames = stage_source(web, sid, stage, checked[sid], index[sid])
            target_relative = Path("audio") / (expected + ".wav")
            target = destination / target_relative
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent.is_relative_to(destination)):
                raise ValueError("published asset or parent is a symlink")
            if target.exists() and (not target.is_file() or sha(target) != expected):
                raise ValueError("existing version asset differs; preserve it and stop")
            copies[str(target_relative)] = dict(source=source, target=target, sha256=expected,
                bytes=source.stat().st_size, already_present=target.is_file())
            stage["file"] = URL_PREFIX + target_relative.as_posix()
        selected.append(sequence)
    remaining = sum(row["bytes"] for row in copies.values() if not row["already_present"])
    ensure_space(destination, remaining, reserve_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(copies.values(), 1):
        if row["already_present"]:
            continue
        ensure_space(destination, remaining, reserve_bytes)
        target = row["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
        shutil.copy2(row["source"], temporary)
        if sha(temporary) != row["sha256"]:
            raise ValueError("copied original stage bytes failed hash verification")
        temporary.replace(target)
        remaining -= row["bytes"]
        if number % 10 == 0:
            print(f"copied original stage files {number}/{len(copies)}", flush=True)
    for row in copies.values():
        if not row["target"].is_file() or row["target"].is_symlink() or sha(row["target"]) != row["sha256"]:
            raise ValueError("publication asset verification failed")
    if sha(source_report) != source_hash or sha(matrix_report) != matrix_hash:
        raise ValueError("input web report changed during publication")
    result["sequences"] = selected
    result["recommendation"] = dict(eligible=False, reason="当前仍为实验结果；20,298 个独立场景配对回归与独立实际节拍复测尚未全部验收，继续保留当前配置。")
    result["perceptual"]["scope"] = "DNSMOS P.835 模型预测；评价固定时间映射的近端语音窗口，不使用人工 MOS 或未经验证的绝对合格线。"
    result["publication"] = dict(complete=True, identity_verified=True, source_report_sha256=source_hash,
        release_id=release_id, tool_sha256=sha(__file__), matrix_report_sha256=matrix_hash, verified_sequences=EXPECTED_SEQUENCES,
        quality_phases=EXPECTED_PHASES, audio_sequences=len(selected), logical_stage_files=sum(len(x["stages"]) for x in selected), audio_scope="六条完整235秒代表连续流的原字节阶段音频；全部30流270阶段指标、失败与诊断保留。",
        original_audio_bytes=sum(row["bytes"] for row in copies.values()), original_audio_files=len(copies),
        reserve_bytes=reserve_bytes, assets=[dict(file=key,sha256=row["sha256"],bytes=row["bytes"]) for key,row in copies.items()])
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    current = destination / "report.json"
    if current.is_symlink():
        raise ValueError("current report must not be a symlink")
    if current.exists():
        previous = current.read_bytes()
        archived = destination / "history" / ("previous-" + hashlib.sha256(previous).hexdigest()[:20] + ".json")
        if archived.exists() and archived.read_bytes() != previous:
            raise ValueError("historical report archive collision")
        if not archived.exists():
            atomic_bytes(archived, previous)
    versioned = destination / "history" / (release_id + ".json")
    if versioned.exists() and versioned.read_bytes() != data:
        raise ValueError("version report already exists with different bytes")
    if not versioned.exists():
        atomic_bytes(versioned, data)
    # The current index becomes visible only after all original assets have verified.
    atomic_bytes(current, data)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=3)
    args = parser.parse_args()
    try:
        result = publish_static(args.web, reserve_bytes=args.reserve_gib * 1024**3)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps(dict(state="blocked", complete=False, reason=str(error), current_report_preserved=True)), flush=True)
        return 2
    print(json.dumps(result["publication"], ensure_ascii=False), flush=True)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
