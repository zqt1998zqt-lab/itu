"""Test evidence contracts and failure detection independently of model quality."""
from io import BytesIO
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server.app import app
from server.quality import api
from server.quality.engine import consensus, evaluate
from server.quality.evaluators import Evaluator, ScoreQEvaluator
from server.quality.metrics import load_audio, reference_metrics


def wav(x, rate=16000):
    b = BytesIO()
    sf.write(b, x, rate, format="WAV", subtype="FLOAT")
    return b.getvalue()


def signal(seconds=10, rate=16000):
    t = np.arange(round(seconds * rate)) / rate
    return .1 * np.sin(2 * np.pi * (230 * t + 35 * t ** 2))


class Fake(Evaluator):
    primary = "MOS"
    minimum_seconds = 9.01

    def __init__(self, cache, name, sign=1):
        super().__init__(cache)
        self.name, self.sign, self.calls = name, sign, 0
        self.identity = dict(name=name, sign=sign, fixture=True)

    def predict(self, x):
        self.calls += 1
        return dict(MOS=3 + self.sign * 10 * float(np.sqrt(np.mean(x ** 2)))), {}


def scorers(tmp_path, signs=(1, -1)):
    return [Fake(tmp_path / "cache", name, sign) for name,sign in zip(("a", "b"), signs)]


def test_paired_model_and_algorithm_disagreement_is_detected(tmp_path):
    x = signal()
    s = scorers(tmp_path)
    result = evaluate("case", load_audio(wav(x)), {"soft":load_audio(wav(x * .5)), "same":load_audio(wav(x))},
                      clean=load_audio(wav(x)), windows=None, evaluators=s)
    assert result["state"] == "completed"
    assert result["cross_checked_comparisons"] == 2
    assert result["windows"][0]["comparisons"]["soft"]["consensus"] == "disagreement"
    assert result["pairwise_algorithm_comparisons"][0]["consensus"] == "disagreement"
    assert result["windows"][0]["comparisons"]["same"]["consensus"] == "within_diagnostic_deadband"
    assert len(result["review_queue"]) == 1
    assert not result["policy"]["model_scores_averaged"]


def test_noise_excluded_from_models_and_silent_input_not_claimed(tmp_path):
    a = load_audio(wav(signal(2)))
    s = scorers(tmp_path)
    window = [dict(name="noise", kind="noise", start_s=0, end_s=2)]
    r = evaluate("noise", a, {"ns":load_audio(wav(a.samples * .1))}, clean=None, windows=window, evaluators=s)
    assert [i.calls for i in s] == [0, 0]
    c = r["windows"][0]["comparisons"]["ns"]
    assert c["noise_reduction"]["attenuation_db"] == pytest.approx(20, abs=1e-5)
    z = load_audio(wav(np.zeros(32000)))
    r = evaluate("zero", z, {"ns":z}, clean=None, windows=window, evaluators=s)
    assert not r["windows"][0]["comparisons"]["ns"]["noise_reduction"]["available"]


@pytest.mark.parametrize("rate", [24000, 48000])
def test_unsupported_native_rates_are_partial_without_resampling(tmp_path, rate):
    a = load_audio(wav(signal(rate=rate), rate))
    r = evaluate("native", a, {"ns":a}, clean=a, windows=None, evaluators=scorers(tmp_path))
    assert r["state"] == "completed_partial" and r["cross_checked_comparisons"] == 0
    assert r["windows"][0]["scores"]["ns"]["perceptual"]["a"]["reason"] == "unsupported_native_sample_rate"
    assert not a.identity["resampled"]


def test_exact_alignment_and_wrong_timebase_negative_control(tmp_path):
    x = signal()
    delay = 480
    before = load_audio(wav(x))
    after = load_audio(wav(np.r_[np.zeros(delay), x]), offset_samples=delay)
    r = evaluate("aligned", before, {"ns":after}, clean=before, windows=None, evaluators=scorers(tmp_path))
    assert r["windows"][0]["scores"]["ns"]["reference"]["sisdr_censored"]
    wrong = load_audio(wav(np.r_[np.zeros(delay), x[:-delay]]))
    assert reference_metrics(before.samples, wrong.samples, 16000)["sisdr_db"] < 0
    with pytest.raises(ValueError, match="unaligned"):
        evaluate("bad", before, {"ns":load_audio(wav(np.r_[np.zeros(delay),x]))}, clean=None,
                 windows=None, evaluators=scorers(tmp_path))


def test_cache_binds_waveform_model_and_rejects_corruption(tmp_path):
    s = Fake(tmp_path, "model")
    x = signal()
    a = s.score(x, 16000)
    assert not a["cache_hit"] and s.calls == 1
    assert s.score(x, 16000)["cache_hit"] and s.calls == 1
    assert not s.score(x * .5, 16000)["cache_hit"] and s.calls == 2
    path = next(p for p in tmp_path.glob("*.json") if json.loads(p.read_text())["cache_identity"] == a["cache_identity"])
    row = json.loads(path.read_text()); row["cache_identity"]["native_rate"] = 48000
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="identity"):
        s.score(x, 16000)


def test_short_audio_is_not_repeated_and_one_model_is_not_cross_validation(tmp_path):
    a = load_audio(wav(signal(2)))
    s = scorers(tmp_path)
    r = evaluate("short", a, {"ns":a}, clean=None, windows=None, evaluators=s)
    assert r["state"] == "completed_partial" and not any(i.calls for i in s)
    assert consensus({"one":1}, .1) == "insufficient_models"
    assert consensus({"a":.05,"b":-.05}, .1) == "within_diagnostic_deadband"


def test_cache_detects_changed_finite_scores(tmp_path):
    s = Fake(tmp_path, "model")
    x = signal()
    s.score(x, 16000)
    path = next(tmp_path.glob("*.json"))
    row = json.loads(path.read_text()); row["values"]["MOS"] += 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="checksum"):
        s.score(x, 16000)


@pytest.mark.parametrize("kwargs", [dict(offset_samples=-1),dict(offset_samples=2.5),dict(channel=-1)])
def test_bad_channel_and_offset_rejected(kwargs):
    with pytest.raises(ValueError):
        load_audio(wav(signal()), **kwargs)


def test_multichannel_and_nonfinite_need_explicit_handling():
    x = signal()
    with pytest.raises(ValueError, match="explicit channel"):
        load_audio(wav(np.stack([x,x], axis=1)))
    np.testing.assert_array_equal(load_audio(wav(np.stack([x,x],axis=1)),channel=1).samples, load_audio(wav(x)).samples)
    with pytest.raises(ValueError, match="nonfinite"):
        load_audio(wav(np.array([0.,np.nan])))


def test_pcm16_matches_cpp_half_away_from_zero():
    values = np.array([.5, 1.5, -.5, -1.5, 32768., -32769.]) / 32768
    actual = load_audio(wav(values), pcm16=True).samples * 32768
    np.testing.assert_array_equal(actual, [1, 2, -1, -2, 32767, -32768])


def test_upload_api_persists_report_and_rejects_busy_or_misaligned(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(api, "_evaluators", lambda names:scorers(tmp_path))
    client = TestClient(app)
    x = signal()
    def files(output=x):
        return [("before",("input.wav",wav(x))), ("clean",("clean.wav",wav(x))),
                ("outputs",("algorithm.wav",wav(output)))]
    r = client.post("/api/quality/evaluate", files=files())
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["report"]["state"] == "completed"
    assert client.get(result["report_url"]).json()["evidence_sha256"] == result["report"]["evidence_sha256"]
    assert client.post("/api/quality/evaluate", files=files(x[:-1])).status_code == 422
    with api.JOB_LOCK:
        assert client.post("/api/quality/evaluate", files=files()).status_code == 429
    assert client.get("/api/quality/runs/invalid").status_code == 404


def test_engine_rejects_window_typos_and_reversed_limits(tmp_path):
    a=load_audio(wav(signal()))
    for window in [dict(name="bad",kind="far",start_s=0,end_s=10),
                   dict(name="bad",kind="speech",start_s=10,end_s=0)]:
        with pytest.raises(ValueError):
            evaluate("bad",a,{"ns":a},clean=None,windows=[window],evaluators=scorers(tmp_path))
