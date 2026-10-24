"""Optional asset-backed tests: real speech, pinned models, and a bad-audio control."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.quality.evaluators import DEFAULT_SCOREQ_ASSETS, ScoreQEvaluator, create_evaluators

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "build/ns-all-family-comparison-20261019/clean-reference-audit/references.jsonl"


@pytest.mark.skipif(not (DEFAULT_SCOREQ_ASSETS / "manifest.json").is_file() or not REFERENCES.is_file(),
                    reason="requires prepared real model assets and audited speech")
def test_real_models_detect_fixed_seed_zero_db_white_noise(tmp_path):
    r = json.loads(REFERENCES.read_text().splitlines()[0])
    if not Path(r["path"]).is_file():
        pytest.skip("audited source speech not present")
    x, rate = sf.read(r["path"], frames=160000)
    n = np.random.default_rng(20261019).normal(size=len(x))
    n *= np.sqrt(np.mean(x*x) / np.mean(n*n))
    models = create_evaluators(["dnsmos","scoreq"], tmp_path / "cache")
    for s in models:
        clean, noisy = s.score(x,rate), s.score(x+n,rate)
        assert clean["available"] and noisy["available"]
        assert clean["values"][s.primary] > noisy["values"][s.primary] + .1
        assert s.score(x,rate)["cache_hit"]


def test_wrong_scoreq_asset_is_rejected_before_inference(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(model="scoreq-natural-nr",sample_rate=16000,
                                                           sha256="bad",source_commit="bad")))
    (tmp_path / "adapt_nr_telephone.onnx").write_bytes(b"wrong graph")
    with pytest.raises(ValueError, match="SHA"):
        ScoreQEvaluator(tmp_path / "cache", tmp_path)
