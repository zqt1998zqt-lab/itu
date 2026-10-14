import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf

TOOLS = Path(__file__).resolve().parents[1] / "cpp/tools"
sys.path.insert(0, str(TOOLS))
import ecnr_v2_publish as pub
from ecnr_campaign_calibration import save, sha
from run_ecnr_v2_matrix import compare

PHASES = ["cold_start_speech", "engine_change", "window_wind", "radio_and_road",
          "long_radio_path", "reference_clock_drift", "reference_mismatch", "path_change", "radio_restart"]

@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pub, "DURATION_S", .01)
    matrix = tmp_path / "build/matrix"
    web = matrix / "web"
    web.mkdir(parents=True)
    configurations = [(rate, mics, variant, pair)
        for rate in (16000, 24000, 48000)
        for mics, variants in ((1, ("control",)), (2, ("control", "spatial")), (4, ("control", "spatial")))
        for variant in variants for pair in (0, 2)]
    indexed, sequences, pairs = [], [], []
    baseline_sha, candidate_sha = "b" * 64, "c" * 64
    global_phase = 0
    for configuration in configurations:
        rate, mics, variant, pair = configuration
        sid = pub.sequence_name(configuration)
        data_by_side, identity_item, stages = {}, {"id": sid}, []
        for side in ("baseline", "candidate"):
            folder = tmp_path / "build" / side / sid
            folder.mkdir(parents=True)
            values = np.zeros((round(rate * .01), mics), dtype=np.float32)
            sf.write(folder / "before.wav", values, rate, subtype="PCM_16")
            sf.write(folder / "stream.after.wav", np.zeros(round(rate * .01), dtype=np.float32), rate, subtype="PCM_16")
            identity = dict(files={"before.wav": sha(folder / "before.wav")},
                output_files={"stream.after.wav": sha(folder / "stream.after.wav")},
                source_manifest_sha256="a" * 64, sample_rate=rate, microphones=mics, engine_resets=0,
                target_alignment={}, simulated_clock=True, processing_profile="vehicle_stream", beam_delays_samples=[0] * mics,
                cli_sha256=baseline_sha if side == "baseline" else candidate_sha)
            details = []
            for offset, phase in enumerate(PHASES):
                passed = global_phase + offset < (184 if side == "baseline" else 193)
                details.append(dict(label=phase, gates=[dict(segment="near_only", metric="raw_sisdr_db",
                    value=2.0 if passed else 0.0, operator=">=", threshold=1.0,
                    status="pass" if passed else "fail", reason=None, optional=False)]))
            data = dict(identity=identity, details=details)
            source_report = folder / "stream-evaluation.json"
            save(source_report, data)
            identity_item[side + "_report"] = str(source_report)
            identity_item[side + "_report_sha256"] = sha(source_report)
            data_by_side[side] = data
            link = web / "audio" / sid / side / "stream.after.wav"
            link.parent.mkdir(parents=True)
            link.symlink_to(folder / "stream.after.wav")
            stages.append(dict(name=side + "/final-output",
                file="/evaluation/ecnr-v2/" + str(link.relative_to(web)),
                sha256=sha(folder / "stream.after.wav"), sample_rate=rate, channels=1,
                fixed_delay_samples=0, trimmed_leading_samples=0, timeline_offset_ms=0,
                duration_s=.01, measurement_domain="fullband",
                waveform=dict(min=[0., 0.], max=[0., 0.], channel=0)))
        matched = compare(data_by_side["baseline"], data_by_side["candidate"])
        for item in matched:
            item.update(sequence_id=sid, rate=rate, microphones=mics, variant=variant, pair=pair)
        pairs.extend(matched)
        indexed.append(identity_item)
        sequences.append(dict(id=sid, stages=stages))
        global_phase += 9
    quality = dict(baseline_pass=184, candidate_pass=193, total_phases=270, phase_wins=9,
        phase_regressions=0, repaired_gates=9, new_failed_gates=0)
    report = dict(schema_version=1, state="completed", complete=True, identity_verified=True,
        identity=dict(configurations=[list(c) for c in configurations], baseline_cli_sha256=baseline_sha, cli_sha256=candidate_sha),
        baseline=dict(id="ecnr-v1", quality_passed=184, total_phases=270),
        quality=quality, identity_sequences=indexed, sequences=sequences, pairs=pairs,
        failures=[dict(sequence_id=p["sequence_id"], label=p["label"], **g) for p in pairs for g in p["candidate_failures"]],
        configurations=[], calibration=dict(status="passed", summary="independent controls"),
        audits=dict(nearzero_windows=[], post_playback_tails=[]),
        coverage=dict(completed_sequences=30, expected_sequences=30),
        perceptual=dict(status="completed", scores=[], scope="test fixture"),
        recommendation=dict(eligible=True, reason="source flag must not promote"))
    save(web / "report.json", report)
    private = copy.deepcopy(report)
    private["sequences"] = private.pop("identity_sequences")
    save(matrix / "report.json", private)
    return SimpleNamespace(root=tmp_path, matrix=matrix, web=web, report=report,
        destination=tmp_path / "public/evaluation/ecnr-v2")

def write_web(campaign, report):
    save(campaign.web / "report.json", report)

@pytest.mark.parametrize("field,value", [("complete", False), ("identity_verified", False), ("state", "running")])
def test_partial_or_unidentified_report_preserves_current(campaign, field, value):
    campaign.destination.mkdir(parents=True)
    current = campaign.destination / "report.json"
    current.write_bytes(b'{"historical":true}\n')
    candidate = copy.deepcopy(campaign.report)
    candidate[field] = value
    write_web(campaign, candidate)
    with pytest.raises(ValueError, match="complete"):
        pub.publish_static(campaign.web)
    assert current.read_bytes() == b'{"historical":true}\n'
    assert not (campaign.destination / "audio").exists()

def test_missing_full_denominator_is_rejected(campaign):
    candidate = copy.deepcopy(campaign.report)
    candidate["quality"]["total_phases"] = 269
    write_web(campaign, candidate)
    with pytest.raises(ValueError, match="270-phase"):
        pub.publish_static(campaign.web)
    assert not campaign.destination.exists()

def test_stale_web_report_is_not_bound_to_new_private_matrix(campaign):
    private = json.loads((campaign.matrix / "report.json").read_text())
    private["identity"]["cli_sha256"] = "d" * 64
    save(campaign.matrix / "report.json", private)
    with pytest.raises(ValueError, match="stale"):
        pub.publish_static(campaign.web)

def test_tampered_input_fails_complete_identity_verification(campaign):
    item = campaign.report["identity_sequences"][-1]
    folder = Path(item["candidate_report"]).parent
    (folder / "before.wav").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="audio/log changed"):
        pub.publish_static(campaign.web)
    assert not campaign.destination.exists()

@pytest.mark.parametrize("url", [
    "/evaluation/ecnr-v2/audio/../candidate/x.wav",
    "/evaluation/ecnr-v2/audio/x/candidate/%2e%2e.wav",
    "https://example.com/audio/x/candidate/x.wav",
    "/evaluation/ecnr-v2/audio/x/candidate/x.wav?source=/etc/passwd",
    "/evaluation/ecnr-v2/audio/x/other/x.wav",
    "/evaluation/ecnr-v2/audio/x/candidate/x.json",
])
def test_external_or_ambiguous_source_path_is_rejected(url):
    with pytest.raises(ValueError):
        pub.relative_stage_url(url)

def test_stage_link_outside_verified_sequence_is_rejected(campaign):
    sid = pub.sequence_name(pub.PILOT[0])
    link = campaign.web / "audio" / sid / "candidate/stream.after.wav"
    link.unlink()
    unrelated = campaign.root / "unrelated.wav"
    sf.write(unrelated, np.zeros(160), 16000)
    link.symlink_to(unrelated)
    with pytest.raises(ValueError, match="inside its verified stream"):
        pub.publish_static(campaign.web)

def test_source_stage_metadata_identity_is_enforced(campaign):
    candidate = copy.deepcopy(campaign.report)
    sid = pub.sequence_name(pub.PILOT[0])
    sequence = next(s for s in candidate["sequences"] if s["id"] == sid)
    sequence["stages"][0]["timeline_offset_ms"] = 4
    write_web(campaign, candidate)
    with pytest.raises(ValueError, match="timeline"):
        pub.publish_static(campaign.web)

def test_disk_reserve_blocks_before_publication(campaign, monkeypatch):
    monkeypatch.setattr(pub.shutil, "disk_usage", lambda _: SimpleNamespace(free=3 * 1024**3 - 1))
    with pytest.raises(RuntimeError, match="disk reserve"):
        pub.publish_static(campaign.web)
    assert not campaign.destination.exists()

def test_corrupt_copy_does_not_publish_report(campaign, monkeypatch):
    def corrupt(source, target):
        Path(target).write_bytes(b"copy corruption")
    monkeypatch.setattr(pub.shutil, "copy2", corrupt)
    with pytest.raises(ValueError, match="hash verification"):
        pub.publish_static(campaign.web)
    assert not (campaign.destination / "report.json").exists()
    assert list(campaign.destination.rglob("*.partial-*"))

def test_target_parent_symlink_is_rejected(campaign):
    public = campaign.root / "public"
    external = campaign.root / "external"
    external.mkdir()
    public.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pub.publish_static(campaign.web)
    assert not list(external.iterdir())

def test_completed_publish_preserves_all_metrics_and_original_bytes(campaign):
    campaign.destination.mkdir(parents=True)
    old = b'{"old_experiment":"preserved"}\n'
    (campaign.destination / "report.json").write_bytes(old)
    original_web = (campaign.web / "report.json").read_bytes()
    result = pub.publish_static(campaign.web)
    assert result["quality"] == campaign.report["quality"]
    assert len(result["pairs"]) == 270
    assert len(result["identity_sequences"]) == 30
    assert len(result["sequences"]) == 6
    assert [s["id"] for s in result["sequences"]] == [pub.sequence_name(c) for c in pub.PILOT]
    assert result["recommendation"]["eligible"] is False
    assert len(result["failures"]) == len(campaign.report["failures"])
    assert all("baseline_value" in failure for failure in result["failures"])
    assert (campaign.web / "report.json").read_bytes() == original_web
    assert any(p.read_bytes() == old for p in (campaign.destination / "history").glob("previous-*.json"))
    for asset in result["publication"]["assets"]:
        path = campaign.destination / asset["file"]
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == asset["sha256"]
    assert result["publication"]["logical_stage_files"] == 12
    assert result["publication"]["original_audio_files"] < 12
    # A retry reuses identical immutable bytes and retains the complete current report.
    before = (campaign.destination / "report.json").read_bytes()
    again = pub.publish_static(campaign.web)
    assert again["quality"] == result["quality"]
    assert (campaign.destination / "report.json").read_bytes() == before


def test_missing_failure_is_refused(campaign):
    web, destination = campaign.web, campaign.destination
    r = json.loads((web / "report.json").read_text())
    r["failures"] = r["failures"][1:]
    (web / "report.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="failures must remain visible"):
        pub.publish_static(web, destination)
    assert not (destination / "report.json").exists()


def test_original_stage_includes_declared_delay(campaign):
    r = json.loads((campaign.web / "report.json").read_text())
    sequence = r["sequences"][0]
    item = r["identity_sequences"][0]
    data = pub.checked_reports(item)
    stage = copy.deepcopy(sequence["stages"][1])
    relative = pub.relative_stage_url(stage["file"])
    original = (campaign.web / str(relative)).resolve()
    sf.write(original, np.zeros((320, 1)), 16000, subtype="PCM_16")
    digest = sha(original)
    stage.update(sample_rate=16000, channels=1, duration_s=.02,
                 fixed_delay_samples=160, timeline_offset_ms=-10, sha256=digest)
    folder = Path(item["candidate_report"]).parent
    data["candidate"]["identity"]["output_files"][str(original.relative_to(folder))] = digest
    source, _, actual_digest, frames = pub.stage_source(campaign.web, sequence["id"], stage, data, item)
    assert source == original and frames == 320 and actual_digest == digest
    stage["fixed_delay_samples"] = 80
    stage["timeline_offset_ms"] = -5
    with pytest.raises(ValueError, match="all input frames"):
        pub.stage_source(campaign.web, sequence["id"], stage, data, item)
