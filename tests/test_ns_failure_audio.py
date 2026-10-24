"""Audio audition must preserve the scored PCM, crop and declared time base."""
from io import BytesIO
import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server.app import app
from server import ns_failures as ns


@pytest.fixture(params=[16000, 24000, 48000])
def evidence(tmp_path, monkeypatch, request):
    rate = request.param
    project = tmp_path / "project"
    origin = project / "build/origin/cases/example"
    origin.mkdir(parents=True)
    crop, delay = rate // 100, 3 * rate // 100
    time = np.arange(rate) / rate
    clean = 0.2 * np.sin(2 * np.pi * 337 * time)
    before = clean + 0.07 * np.cos(2 * np.pi * 113 * time)
    sf.write(origin / "before.wav", before, rate, subtype="FLOAT")
    sf.write(origin / "clean.wav", clean, rate, subtype="FLOAT")
    before = sf.read(origin / "before.wav")[0]
    canonical = np.clip(np.rint(before[crop:] * 32768), -32768, 32767).astype("<i2")
    replays = project / "build/replays"
    replays.mkdir()
    rows, expected = [], {"input": canonical}
    for i, (variant, role) in enumerate(ns.VARIANTS.items()):
        folder = project / "build" / role
        folder.mkdir()
        output = np.rint(canonical.astype(float) * (0.6 + 0.1 * i)).astype("int16")
        sf.write(folder / "result.after.wav", np.r_[np.zeros(delay, dtype="int16"), output], rate, subtype="PCM_16")
        expected[role] = output
        metrics = {"before": {"projection_gain_db": 0., "sisdr_db": 9.},
                   "after": {"projection_gain_db": -4., "sisdr_db": 7.}, "stoi_improvement": -0.03}
        rows.append({"variant": variant, "scope": "cold", "original_run": "origin",
                     "scene": {"id": "example", "sample_rate": rate, "noise": "test-noise",
                               "snr_db": 5, "split": "test", "windows": {"near_only": [0.1, 0.2]}},
                     "input_crop_samples": crop, "input_pcm16_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
                     "delay_samples": delay, "output_sha256": hashlib.sha256((folder / "result.after.wav").read_bytes()).hexdigest(),
                     "folder": str(folder.relative_to(project)), "replay_hash_matches": True,
                     "failures": ["near_only:projection_gain", "near_only:raw_sisdr", "near_only:stoi"],
                     "windows": {"near_only": {"frozen_metrics": metrics}}})
    (replays / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (replays / "report.json").write_text(json.dumps({"state": "completed", "dependencies_verified": True,
        "replays": 2, "exact_output_hash_matches": 2, "common_case_counts": {"cold": 1, "steady": 0}}))
    monkeypatch.setattr(ns, "PROJECT", project)
    monkeypatch.setattr(ns, "REPLAYS", replays)
    monkeypatch.setattr(ns, "CACHE", tmp_path / "cache")
    ns.catalog.cache_clear()
    yield TestClient(app), rate, expected, rows
    ns.catalog.cache_clear()


def test_original_pcm_crop_declared_delay_and_named_window(evidence):
    client, rate, expected, _ = evidence
    response = client.get("/api/ns-failures")
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["total"] == 1 and catalog["scope_counts"] == {"cold": 1}
    sample = catalog["items"][0]
    assert sample["duration_s"] == pytest.approx(0.99)
    for role, values in expected.items():
        response = client.get(sample["audio"][role])
        actual, ar = sf.read(BytesIO(response.content), dtype="int16")
        assert ar == rate
        np.testing.assert_array_equal(actual, values)
        response = client.get(sample["audio"][role], params={"window": "near_only"})
        actual, ar = sf.read(BytesIO(response.content), dtype="int16")
        assert ar == rate
        np.testing.assert_array_equal(actual, values[round(.1 * rate):round(.2 * rate)])
    # A broken delay/crop implementation must be detected by this fixture.
    assert np.any(expected["input"][:rate // 100] != 0)
    assert np.any(expected["omlsa"][:3 * rate // 100] != 0)
    failures = sample["backends"]["omlsa"]["failures"]
    assert [f["value"] for f in failures] == [-4., 7., -.03]
    assert failures[1]["requirement"] == "≥ 8.000 dB"


def test_media_range_download_and_invalid_identifiers(evidence):
    client, _, _, _ = evidence
    sample = client.get("/api/ns-failures").json()["items"][0]
    url = sample["audio"]["omlsa"]
    full = client.get(url)
    part = client.get(url, headers={"Range": "bytes=0-43"})
    assert part.status_code == 206 and part.content == full.content[:44]
    assert part.headers["content-range"].startswith("bytes 0-43/")
    assert "attachment" in client.get(url, params={"download": True}).headers["content-disposition"]
    for route in ["/api/ns-failures/unknown/audio/input", url.replace("/omlsa", "/invalid"), url + "?window=../../secret"]:
        assert client.get(route).status_code == 404
    assert client.get(url, headers={"Range": "bytes=9999999-"}).status_code == 416


def test_unverified_or_changed_evidence_is_not_served(evidence):
    client, _, _, rows = evidence
    sample = client.get("/api/ns-failures").json()["items"][0]
    (ns.PROJECT / rows[0]["folder"] / "result.after.wav").write_bytes(b"changed output")
    assert client.get(sample["audio"]["omlsa"]).status_code == 503
    report = ns.REPLAYS / "report.json"
    data = json.loads(report.read_text())
    data["dependencies_verified"] = False
    report.write_text(json.dumps(data))
    ns.catalog.cache_clear()
    assert client.get("/api/ns-failures").status_code == 503
