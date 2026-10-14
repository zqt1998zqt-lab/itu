import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cleanup_audio", ROOT / "cpp/tools/clean_intermediate_audio.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def test_cleanup_preserves_inputs_public_targets_and_external_hardlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    case = tmp_path / "datasets/itu-eval-v2-fixed/cases/example"
    run = case / "runs/full"
    run.mkdir(parents=True)
    for name in ("before.wav", "clean.wav", "ref.wav"):
        (case / name).write_bytes(b"input")
    for name in ("result.after.wav", "result.bf.wav", "result.aec.wav"):
        (run / name).write_bytes(b"output")
    (run / "metrics.json").write_text("{}")
    (run / "result.frames.csv").write_text("frame\n")
    os.link(run / "result.after.wav", case / "denoise.wav")
    os.link(run / "result.aec.wav", tmp_path / "external.wav")
    public = tmp_path / "public"
    public.mkdir()
    (public / "listen.wav").symlink_to(run / "result.bf.wav")
    monkeypatch.setattr(cleanup, "candidate_roots", lambda: iter([case]))
    protected = {str(public / "listen.wav"): cleanup.signature(public / "listen.wav")}
    plan = cleanup.build_plan(protected)
    assert {row["path"] for row in plan["files"]} == {
        "datasets/itu-eval-v2-fixed/cases/example/denoise.wav",
        "datasets/itu-eval-v2-fixed/cases/example/runs/full/result.after.wav",
    }
    assert plan["skipped_protected_links"] == 1
    assert plan["skipped_external_hardlinks"] == 1
    assert plan["estimated_allocated_bytes"] == (run / "result.after.wav").stat().st_blocks * 512


def test_cleanup_does_not_follow_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    case = tmp_path / "candidate"
    case.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.after.wav").write_bytes(b"protected")
    (case / "linked").symlink_to(outside, target_is_directory=True)
    (case / "copy.after.wav").symlink_to(outside / "result.after.wav")
    monkeypatch.setattr(cleanup, "candidate_roots", lambda: iter([case]))
    assert cleanup.build_plan({})["files"] == []


def test_cleanup_roots_exclude_frozen_baselines_and_current_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    for name in ("radio-optimization", "quality-repair", "data-algorithm-v4"):
        (tmp_path / "build" / name).mkdir(parents=True)
    for name in cleanup.KEEP:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    keep = tmp_path / "build/data-algorithm-v4/curated-hybrid-48000-final"
    keep.mkdir()
    probe = tmp_path / "build/radio-optimization/old-probe"
    probe.mkdir()
    current = tmp_path / "build/repair-directions-expanded-probe-v2-20261010"
    current.mkdir()
    full = tmp_path / "build/quality-filtered-full-20261010"
    full.mkdir()
    roots = set(cleanup.candidate_roots())
    assert probe in roots
    assert keep not in roots
    assert current not in roots
    assert full not in roots
    assert not roots.intersection(tmp_path / name for name in cleanup.KEEP)
