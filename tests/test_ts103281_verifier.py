from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.quality import ts103281


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/verify-ts103281-model-a.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_ts103281_model_a",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

CONDITIONS = [f"C{number:02d}" for number in range(1, 49)]


def write_wav(path: Path, *, frames: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((frames, 2), dtype=np.float64)
    samples[:, 0] = 0.1
    samples[:, 1] = 0.05
    sf.write(path, samples, 48_000, format="WAV", subtype="FLOAT")


def write_vectors(directory: Path, conditions: list[str]) -> None:
    rms_rows = verifier.expected_rms_rows()
    for condition in conditions:
        path = directory / f"{condition}_f1s1.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        level = 10 ** ((rms_rows[condition] - 15) / 20)
        samples = np.tile([level, .5 * level], (32, 1))
        sf.write(path, samples, 48_000, format="WAV", subtype="FLOAT")


def exact_evaluator(
    expected: dict[str, dict[str, float]],
    conditions: list[str],
):
    calls: list[tuple[int, int, int]] = []
    values = iter([expected[condition] for condition in conditions])

    def evaluate(reference, degraded, rate):
        calls.append((len(reference), len(degraded), int(rate)))
        scores = next(values)
        return {
            "scores": {
                metric: {"value": value}
                for metric, value in scores.items()
            },
            "features": {
                "smos": {"PreProcessing.SNR-A": 12.5},
                "nmos": {"BGN.HearingModelSharpness_P95T_FR": 1.2},
                "gmos": {"overall": 0.25},
            },
            "diagnostics": {
                "alignment": {"lag_samples": 0},
                "band_count": 33,
            },
            # Diagnostic scores are deliberately wrong. Conformance must use
            # only the original model scores above.
            "diagnostic_scores": {
                "S-MOS": {"value": 1.0},
                "N-MOS": {"value": 1.0},
                "G-MOS": {"value": 1.0},
            },
        }

    return evaluate, calls


def write_reference_marked_vectors(
    directory: Path,
    conditions: list[str],
) -> None:
    rms_rows = verifier.expected_rms_rows()
    for condition in conditions:
        path = directory / f"{condition}_f1s1.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        level = 10 ** ((rms_rows[condition] - 15) / 20)
        marker = float(int(condition[1:])) / 1_000
        samples = np.tile([level, marker], (32, 1))
        sf.write(path, samples, 48_000, format="WAV", subtype="FLOAT")


def deterministic_model_a_from_reference(
    expected: dict[str, dict[str, float]],
    conditions: list[str],
):
    """Return deterministic scores based on each condition's reference marker."""
    pressure_gain = 10 ** (verifier.ANNEX_C_PRESSURE_GAIN_DB / 20)
    markers = [
        (float(int(condition[1:])) / 1_000 * pressure_gain, condition)
        for condition in conditions
    ]
    calls: list[str] = []

    def evaluate(reference, degraded, rate):
        sample = np.asarray(reference)
        observed_marker = float(sample.flat[0])
        condition = min(
            markers,
            key=lambda item: abs(item[0] - observed_marker),
        )[1]
        calls.append(condition)
        scores = expected[condition]
        return {
            "scores": {
                metric: {"value": value}
                for metric, value in scores.items()
            },
            "features": {
                "smos": {"PreProcessing.SNR-A": 12.5},
                "nmos": {"BGN.HearingModelSharpness_P95T_FR": 1.2},
                "gmos": {"overall": 0.25},
            },
            "diagnostics": {
                "alignment": {"lag_samples": 0},
                "band_count": 33,
            },
        }

    return evaluate, calls


def comparable_report(report: dict) -> dict:
    """Remove intentionally variable timing fields before comparing reports."""
    result = copy.deepcopy(report)
    result.pop("generated_at", None)
    result.pop("duration_seconds", None)
    result["environment"].pop("workers", None)
    for row in result["rows"]:
        row.pop("duration_seconds", None)
        row.pop("duration_includes_queue", None)
        row["progress"].pop("elapsed_seconds", None)
    return result


def test_full_verification_requires_all_48_conditions_and_144_original_scores(
    tmp_path: Path,
):
    expected = verifier.expected_rows()
    write_vectors(tmp_path, CONDITIONS)
    evaluate, calls = exact_evaluator(expected, CONDITIONS)
    progress: list[dict] = []

    report = verifier.verify(
        tmp_path,
        include_features=True,
        evaluator=evaluate,
        on_progress=progress.append,
    )

    assert report["conformant"] is True
    assert report["conformance_scope"] == "full"
    assert report["condition_count"] == 48
    assert report["independent_condition_count"] == 48
    assert report["score_count"] == 144
    assert report["finite_score_count"] == 144
    assert report["passed_conditions"] == 48
    assert report["failed_reason"] is None
    assert len(calls) == 48
    assert all(reference == degraded == 32 and rate == 48_000 for reference, degraded, rate in calls)
    assert len(progress) == 48
    assert [entry["completed"] for entry in progress] == list(range(1, 49))
    assert all(row["features"] is not None for row in report["rows"])
    assert all(row["diagnostics"] is not None for row in report["rows"])
    assert all(row["duration_seconds"] >= 0 for row in report["rows"])
    assert all(row["progress"]["total"] == 48 for row in report["rows"])
    assert all(row["vector"]["sha256"] for row in report["rows"])
    assert (tmp_path / "vector-manifest.json").is_file()
    assert report["identity_verified_during_run"] is True
    assert report["pressure_calibration"]["verified_conditions"] == 48
    assert report["pressure_calibration"]["common_gain_db"] == 15.0
    assert report["pressure_calibration"]["uses_mos_targets"] is False
    start_identity = verifier.implementation_identity()
    assert report["implementation_sha256"] == (
        start_identity["implementation_sha256"]
    )
    assert report["dependency_sha256"] == (
        start_identity["dependency_sha256"]
    )
    assert report["model_sha256"] == start_identity["model_sha256"]
    assert report["verifier_sha256"] == start_identity["verifier_sha256"]
    json.dumps(report, allow_nan=False)


def test_pressure_calibration_uses_one_gain_and_preserves_channel_ratio():
    samples = np.tile([.01, .03], (32, 1))
    before = samples.copy()
    calibrated, details = verifier.calibrate_vector_pressure(samples, -25.0)
    np.testing.assert_allclose(calibrated, samples * 10 ** (15 / 20))
    np.testing.assert_allclose(calibrated[:, 1] / calibrated[:, 0], 3)
    np.testing.assert_array_equal(samples, before)
    assert details["rms_error_db"] == pytest.approx(0, abs=1e-12)


def test_wrong_pressure_scale_is_rejected_before_scoring():
    with pytest.raises(verifier.VerificationError, match="pressure calibration mismatch"):
        verifier.calibrate_vector_pressure(np.full((32, 2), .01), -24.0)


def test_subset_can_be_measured_but_can_never_claim_conformance(tmp_path: Path):
    expected = verifier.expected_rows()
    write_vectors(tmp_path, CONDITIONS)
    evaluate, calls = exact_evaluator(expected, CONDITIONS[:2])

    report = verifier.verify(tmp_path, limit=2, evaluator=evaluate)

    assert report["conformant"] is False
    assert report["conformance_scope"] == "subset"
    assert report["requested_limit"] == 2
    assert report["condition_count"] == 2
    assert report["score_count"] == 6
    assert report["finite_score_count"] == 6
    assert report["passed_conditions"] == 2
    assert "cannot claim Annex C conformance" in report["failed_reason"]
    assert len(calls) == 2
    assert all("features" not in row for row in report["rows"])


def test_nonfinite_original_score_blocks_conformance_and_remains_json_safe(
    tmp_path: Path,
):
    expected = verifier.expected_rows()
    write_vectors(tmp_path, CONDITIONS)
    evaluate, _ = exact_evaluator(expected, CONDITIONS)

    # Use a small wrapper with an explicit first-call flag so the negative
    # control is deterministic.
    first = True

    def evaluator(reference, degraded, rate):
        nonlocal first
        result = evaluate(reference, degraded, rate)
        if first:
            result["scores"]["N-MOS"]["value"] = float("nan")
            first = False
        return result

    report = verifier.verify(tmp_path, evaluator=evaluator)

    assert report["conformant"] is False
    assert report["score_count"] == 144
    assert report["finite_score_count"] == 143
    assert report["rows"][0]["actual"]["N-MOS"] is None
    assert report["rows"][0]["difference"]["N-MOS"] is None
    assert "Only 143/144" in report["failed_reason"]
    json.dumps(report, allow_nan=False)


def test_existing_truncated_vector_fails_without_download_or_overwrite(
    tmp_path: Path,
):
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros((64, 2), dtype=np.float64),
        48_000,
        format="WAV",
        subtype="FLOAT",
    )
    truncated = vector_dir / "C01_f1s1.wav"
    truncated.write_bytes(buffer.getvalue()[:-96])

    def forbidden_opener(*args, **kwargs):
        raise AssertionError("existing bad vector must not be overwritten")

    with pytest.raises(
        verifier.VerificationError,
        match="invalid or truncated WAV",
    ):
        verifier.ensure_vectors(
            vector_dir,
            ["C01"],
            opener=forbidden_opener,
        )
    assert truncated.stat().st_size == len(buffer.getvalue()) - 96


def test_missing_vector_is_downloaded_to_persistent_path_with_hash(
    tmp_path: Path,
):
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros((64, 2), dtype=np.float64),
        48_000,
        format="WAV",
        subtype="FLOAT",
    )
    payload = buffer.getvalue()
    requested: list[str] = []

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def opener(request, timeout):
        requested.append(request.full_url)
        assert timeout == 60
        return Response(payload)

    records = verifier.ensure_vectors(
        tmp_path / "vectors",
        ["C01"],
        opener=opener,
    )
    repeated = verifier.ensure_vectors(
        tmp_path / "vectors",
        ["C01"],
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid persisted vector must be reused")
        ),
    )

    path = tmp_path / "vectors/C01_f1s1.wav"
    assert path.read_bytes() == payload
    assert records["C01"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert repeated == records
    assert records["C01"]["channels"] == 2
    assert records["C01"]["frames"] == 64
    assert requested == [f"{verifier.BASE_URL}/C01_f1s1.wav"]
    manifest = json.loads(
        (tmp_path / "vectors/vector-manifest.json").read_text()
    )
    assert manifest["condition_count"] == 1
    assert manifest["vectors"]["C01"]["sha256"] == records["C01"]["sha256"]


def test_expected_csv_must_be_exactly_c01_through_c48(tmp_path: Path):
    path = tmp_path / "annex.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "s_mos", "n_mos", "g_mos"],
        )
        writer.writeheader()
        for condition in CONDITIONS[:-1]:
            writer.writerow(
                {
                    "condition": condition,
                    "s_mos": 3.0,
                    "n_mos": 3.0,
                    "g_mos": 3.0,
                }
            )

    with pytest.raises(
        verifier.VerificationError,
        match="exactly C01..C48",
    ):
        verifier.expected_rows(path)


def test_implementation_identity_records_models_and_ts103281_dependencies():
    identity = verifier.implementation_identity()
    implementation = Path(ts103281.__file__).resolve()

    assert identity["implementation_sha256"] == hashlib.sha256(
        implementation.read_bytes()
    ).hexdigest()
    assert set(identity["model_sha256"]) == {
        "server/quality/assets/ts103281/S-MOS.model",
        "server/quality/assets/ts103281/N-MOS.model",
        "server/quality/assets/ts103281/G-MOS.model",
    }
    assert identity["dependency_sha256"]["server/quality/ts103281.py"] == (
        identity["implementation_sha256"]
    )
    assert {
        "server/__init__.py",
        "server/quality/__init__.py",
        "server/quality/ts103281_activity.py",
        "server/quality/ts103281_hearing.py",
        "server/quality/ts103281_modulation.py",
        "server/quality/ts103281_reference.py",
    } <= set(identity["dependency_sha256"])
    assert identity["verifier_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()
    assert "transitive local imports" in identity["identity_scope"]


def test_identity_rehashes_real_dependency_changes(tmp_path: Path):
    root = tmp_path / "project"
    quality = root / "server/quality"
    quality.mkdir(parents=True)
    (root / "server/__init__.py").write_text("", encoding="utf-8")
    (quality / "__init__.py").write_text("", encoding="utf-8")
    main = quality / "ts103281.py"
    dependency = quality / "local_dependency.py"
    verifier_path = root / "scripts/verify.py"
    verifier_path.parent.mkdir()
    verifier_path.write_text("# verifier\n", encoding="utf-8")
    main.write_text(
        "from .local_dependency import VALUE\n",
        encoding="utf-8",
    )
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    model_dir = root / "models"
    model_dir.mkdir()
    for name in ("S-MOS.model", "N-MOS.model", "G-MOS.model"):
        (model_dir / name).write_text(name, encoding="utf-8")

    first = verifier.implementation_identity(
        project_root=root,
        main_path=main,
        model_dir=model_dir,
        verifier_path=verifier_path,
    )
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    second = verifier.implementation_identity(
        project_root=root,
        main_path=main,
        model_dir=model_dir,
        verifier_path=verifier_path,
    )

    key = "server/quality/local_dependency.py"
    assert first["dependency_sha256"][key] != second["dependency_sha256"][key]
    assert first["implementation_sha256"] == second["implementation_sha256"]
    assert verifier._identity_differences(first, second) == [
        f"dependency_sha256.{key}"
    ]


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dependency_sha256", "server/quality/ts103281.py"),
        ("dependency_sha256", "server/quality/ts103281_activity.py"),
        ("model_sha256", "server/quality/assets/ts103281/S-MOS.model"),
        ("verifier_sha256", None),
    ],
)
def test_runtime_identity_change_raises_instead_of_conformant(
    tmp_path: Path,
    monkeypatch,
    section: str,
    key: str | None,
):
    expected = verifier.expected_rows()
    write_vectors(tmp_path, CONDITIONS)
    evaluate, calls = exact_evaluator(expected, CONDITIONS)
    start = verifier.implementation_identity()
    changed = copy.deepcopy(start)
    if key is None:
        changed[section] = "0" * 64
    else:
        changed[section][key] = "0" * 64
    identity_calls = 0

    def changing_identity():
        nonlocal identity_calls
        identity_calls += 1
        return start if identity_calls == 1 else changed

    monkeypatch.setattr(verifier, "implementation_identity", changing_identity)

    with pytest.raises(
        verifier.VerificationError,
        match="identity changed during verification",
    ):
        verifier.verify(tmp_path, evaluator=evaluate)

    assert identity_calls == 2
    assert len(calls) == 48


def test_default_vectors_are_persistent_and_evidence_writes_are_exclusive(
    tmp_path: Path,
):
    assert verifier.DEFAULT_VECTOR_DIR == (
        PROJECT / "build/model-a-20261105/baseline/vectors"
    )
    assert "TemporaryDirectory" not in SCRIPT.read_text(encoding="utf-8")

    evidence = tmp_path / "evidence.json"
    verifier._write_json_exclusive(evidence, {"first": True})
    with pytest.raises(FileExistsError):
        verifier._write_json_exclusive(evidence, {"second": True})
    assert json.loads(evidence.read_text()) == {"first": True}


@pytest.mark.parametrize("workers", [0, 17])
def test_verify_rejects_workers_outside_supported_range(
    tmp_path: Path,
    workers: int,
):
    write_vectors(tmp_path, CONDITIONS)

    with pytest.raises(verifier.VerificationError, match="workers"):
        verifier.verify(tmp_path, limit=2, workers=workers)


def test_parallel_verification_rejects_custom_evaluator(
    tmp_path: Path,
):
    write_vectors(tmp_path, CONDITIONS)

    def custom_evaluator(reference, degraded, rate):
        raise AssertionError("parallel runs must not use an injected evaluator")

    with pytest.raises(verifier.VerificationError, match="evaluator"):
        verifier.verify(
            tmp_path,
            limit=2,
            evaluator=custom_evaluator,
            workers=2,
        )


def test_parallel_verification_preserves_order_scores_and_progress(
    tmp_path: Path,
    monkeypatch,
):
    expected = verifier.expected_rows()
    write_reference_marked_vectors(tmp_path, CONDITIONS)
    evaluate, calls = deterministic_model_a_from_reference(expected, CONDITIONS)
    monkeypatch.setattr(ts103281, "evaluate_model_a", evaluate)
    # ProcessPoolExecutor is intentionally replaced with a thread executor:
    # this keeps the test hermetic while still exercising the real verify path.
    monkeypatch.setattr(verifier, "ProcessPoolExecutor", ThreadPoolExecutor)
    sequential_progress: list[dict] = []
    parallel_progress: list[dict] = []

    sequential = verifier.verify(
        tmp_path,
        include_features=True,
        workers=1,
        on_progress=sequential_progress.append,
    )
    parallel = verifier.verify(
        tmp_path,
        include_features=True,
        workers=16,
        on_progress=parallel_progress.append,
    )

    assert sequential["conformant"] is True
    assert parallel["conformant"] is True
    assert [row["condition"] for row in parallel["rows"]] == CONDITIONS
    assert [entry["completed"] for entry in sequential_progress] == list(
        range(1, 49)
    )
    assert [entry["completed"] for entry in parallel_progress] == list(
        range(1, 49)
    )
    assert len(calls) == 96
    assert all(
        row["actual"] == row["expected"]
        for row in parallel["rows"]
    )
    assert comparable_report(sequential) == comparable_report(parallel)


def test_cli_workers_is_forwarded_to_verify(monkeypatch):
    observed: dict = {}

    def fake_verify(directory, limit=None, **kwargs):
        observed["directory"] = directory
        observed["limit"] = limit
        observed.update(kwargs)
        return {"conformant": True}

    monkeypatch.setattr(verifier, "verify", fake_verify)
    monkeypatch.setattr(
        verifier.sys,
        "argv",
        [
            "verify-ts103281-model-a.py",
            "--vectors",
            "/tmp/parallel-test-vectors",
            "--limit",
            "2",
            "--workers",
            "16",
            "--json",
        ],
    )

    assert verifier.main() == 0
    assert observed["directory"] == Path("/tmp/parallel-test-vectors")
    assert observed["limit"] == 2
    assert observed["workers"] == 16
    assert callable(observed["on_progress"])
