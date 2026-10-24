"""Pinned local model adapters. Inference never downloads or resamples audio."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np

from .metrics import sha

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SCOREQ_ASSETS = PROJECT / "build/quality-models/scoreq-natural-nr"
SCOREQ_SHA256 = "d493ba8acfe37d0aa0acceb85a41f22405fc4b9a5101359a6363d919307300b4"
SCOREQ_COMMIT = "0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e"
_INFERENCE_LOCK = threading.Lock()


class Evaluator:
    name = ""
    primary = ""
    minimum_seconds = 0.
    rate = 16000

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def score(self, x: np.ndarray, rate: int) -> dict:
        if rate != self.rate:
            return dict(available=False, reason="unsupported_native_sample_rate", expected_rate=self.rate,
                        actual_rate=rate, resampled=False)
        if len(x) < round(self.minimum_seconds * rate):
            return dict(available=False, reason="window_too_short", minimum_seconds=self.minimum_seconds,
                        actual_seconds=len(x) / rate, repeated=False)
        if np.std(x) <= 1e-10 or np.sqrt(np.mean(x ** 2)) <= 1e-10:
            return dict(available=False, reason="silence_or_no_ac_excitation")
        fields = dict(model=self.identity, native_rate=rate,
                      waveform_sha256=sha(np.asarray(x, dtype="<f8").tobytes()))
        key = sha(json.dumps(fields, sort_keys=True).encode())
        path = self.cache_dir / (key + ".json")
        # Serialize model sessions as well as same-key atomic cache writes.
        with _INFERENCE_LOCK:
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("cache_identity") != fields:
                    raise ValueError("quality cache identity mismatch")
                values = row.get("values", {})
                if not values or not all(np.isfinite(v) for v in values.values()):
                    raise ValueError("invalid cached quality scores")
                if row.get("score_sha256") != sha(json.dumps(dict(values=values, detail=row["detail"], identity=fields), sort_keys=True).encode()):
                    raise ValueError("cached quality score checksum mismatch")
                return {**row, "cache_hit": True}
            start = time.monotonic()
            values, detail = self.predict(x)
            if not values or not all(np.isfinite(v) for v in values.values()):
                raise ValueError("invalid quality model output")
            row = dict(available=True, values=values, detail=detail, cache_hit=False,
                       cache_identity=fields, inference_seconds=time.monotonic() - start,
                       resampled=False, level_normalized=False, repeated=False)
            row["score_sha256"] = sha(json.dumps(dict(values=values, detail=detail, identity=fields), sort_keys=True).encode())
            from cpp.tools.ecnr_perceptual import save_json
            save_json(path, row)
            return row


class DNSMOSEvaluator(Evaluator):
    name = "dnsmos"
    primary = "OVRL"
    minimum_seconds = 9.01

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        from cpp.tools.ecnr_perceptual import DNSMOS
        self.model = DNSMOS(cache_dir=cache_dir / "official-dnsmos")
        self.identity = dict(name="DNSMOS P.835", **self.model.preprocessing,
                             primary=self.primary, engine_short_policy="unavailable; no repetition",
                             adapter_sha256=sha(Path(__file__).read_bytes()))

    def predict(self, x: np.ndarray) -> tuple[dict, dict]:
        row = self.model.score_array(x, self.rate, kind="near_only")
        if not row["available"]:
            raise ValueError("DNSMOS inference unavailable: " + row["reason"])
        return row["poly"], dict(hops=row["num_scored_hops"], official_commit=row["official_commit"])


class ScoreQEvaluator(Evaluator):
    name = "scoreq"
    primary = "MOS"
    minimum_seconds = 1.

    def __init__(self, cache_dir: Path, assets: Path = DEFAULT_SCOREQ_ASSETS):
        super().__init__(cache_dir)
        import onnxruntime as ort
        assets = Path(assets)
        manifest = json.loads((assets / "manifest.json").read_text())
        if manifest.get("model") != "scoreq-natural-nr" or manifest.get("sample_rate") != 16000:
            raise ValueError("SCOREQ asset manifest contract mismatch")
        model_path = assets / "adapt_nr_telephone.onnx"
        if manifest.get("source_commit") != SCOREQ_COMMIT or manifest.get("sha256") != SCOREQ_SHA256 or sha(model_path.read_bytes()) != SCOREQ_SHA256:
            raise ValueError("SCOREQ model SHA mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 2:
            raise ValueError("unexpected SCOREQ ONNX input contract")
        self.input_name = inputs[0].name
        self.identity = dict(name="SCOREQ natural/no-reference", model_sha256=manifest["sha256"],
                             source_commit=manifest["source_commit"], primary=self.primary,
                             sampling_rate=16000, preprocessing="float32 mono; right zero pad to multiple of 320",
                             level_normalization=False, provider="CPUExecutionProvider", cpu_threads=2,
                             onnxruntime_version=ort.__version__, numpy_version=np.__version__,
                             implementation_sha256=sha(Path(__file__).read_bytes()))

    def predict(self, x: np.ndarray) -> tuple[dict, dict]:
        extra = (-len(x)) % 320
        data = np.pad(x.astype(np.float32), (0, extra))[None, :]
        output = np.asarray(self.session.run(None, {self.input_name: data})[0])
        if output.size != 1:
            raise ValueError("unexpected SCOREQ output shape")
        return dict(MOS=float(output.item())), dict(padding_samples=extra, input_samples=len(x),
                                                  score_clamped=False)


class MissingEvaluator(Evaluator):
    def __init__(self, name: str, error: str):
        self.name, self.primary, self.error = name, "OVRL" if name == "dnsmos" else "MOS", error
        self.identity = dict(name=name, available=False, reason=error)

    def score(self, x: np.ndarray, rate: int) -> dict:
        return dict(available=False, reason="model_not_ready", detail=self.error)


def create_evaluators(names: list[str], cache_dir: Path, scoreq_assets: Path = DEFAULT_SCOREQ_ASSETS) -> list[Evaluator]:
    if not names or len(names) != len(set(names)) or set(names) - {"dnsmos", "scoreq"}:
        raise ValueError("evaluators must be distinct entries from dnsmos, scoreq")
    result = []
    for name in names:
        try:
            item = DNSMOSEvaluator(cache_dir) if name == "dnsmos" else ScoreQEvaluator(cache_dir, scoreq_assets)
        except (OSError, ImportError) as error:
            item = MissingEvaluator(name, str(error))
        result.append(item)
    return result
