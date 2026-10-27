import json

from server.alignment_runner import (
    _pipeline_spec,
    compute_overall_success,
    compute_pytest_ok,
    get_runner_status,
    group_status,
    parse_junit,
    validate_pipeline_artifacts,
)


def test_default_profile_maps_to_pinned_cli_name() -> None:
    assert _pipeline_spec("vcp_default")["cli_profile"] == "vcp"
    assert _pipeline_spec("vcp_joint")["cli_profile"] == "vcp_joint"
    assert _pipeline_spec("vcp_unified")["cli_profile"] == "vcp_unified"


def test_pipeline_failure_cannot_be_masked_by_passing_tests() -> None:
    assert not compute_overall_success(
        pytest_ok=True,
        pipeline_ok=False,
        strict=True,
        skipped_count=0,
    )
    assert compute_overall_success(
        pytest_ok=True,
        pipeline_ok=True,
        strict=True,
        skipped_count=0,
    )


def test_strict_mode_rejects_skipped_tests(tmp_path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """
        <testsuite>
          <testcase classname="tests.test_example" name="passed" time="0.1" />
          <testcase classname="tests.test_example" name="skipped" time="0.2">
            <skipped />
          </testcase>
        </testsuite>
        """
    )
    summary = parse_junit(junit)
    assert summary["passed_count"] == 1
    assert summary["skipped_count"] == 1
    assert group_status(summary, "tests/test_example.py", strict=False) == "skipped"
    assert group_status(summary, "tests/test_example.py", strict=True) == "failed"
    assert not compute_pytest_ok(
        0, summary, ["tests/test_example.py"], strict=True
    )
    assert compute_pytest_ok(
        0, summary, ["tests/test_example.py"], strict=False
    )


def test_junit_parser_counts_failures_errors_and_files(tmp_path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """
        <testsuite>
          <testcase classname="tests.test_one" name="pass" time="0.1" />
          <testcase classname="tests.test_one" name="failure" time="0.2">
            <failure message="bad" />
          </testcase>
          <testcase classname="tests.test_two" name="error" time="0.3">
            <error message="broken" />
          </testcase>
          <testcase classname="tests.test_two" name="skip" time="0.4">
            <skipped />
          </testcase>
        </testsuite>
        """
    )
    summary = parse_junit(junit)
    assert summary["passed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["error_count"] == 1
    assert summary["skipped_count"] == 1
    assert {case["file"] for case in summary["cases"]} == {
        "tests/test_one.py",
        "tests/test_two.py",
    }


def test_pipeline_artifacts_reject_wrong_profile_and_bad_wav(tmp_path) -> None:
    report = tmp_path / "result.json"
    wav = tmp_path / "result.after.wav"
    report.write_text(
        json.dumps(
            {
                "processing_profile": "vcp",
                "process_rtf": 0.01,
                "dsp_delay_ms": 6.125,
                "aec_backend": "speex_mdf",
                "ns_backend": "webrtc",
                "tx_drc_enabled": True,
                "unified_ns_weights": [0, 0, 0, 0],
            }
        )
    )
    wav.write_bytes(b"not-a-wave")
    data, errors = validate_pipeline_artifacts(report, wav, "vcp_unified")
    assert data is not None
    assert any("Profile 不一致" in error for error in errors)
    assert any("Unified NS 权重" in error for error in errors)
    assert any("不是有效 WAV" in error for error in errors)


def test_status_never_claims_formal_standard_compliance() -> None:
    status = get_runner_status()
    assert status["formal_compliance"] is False
    assert "engineering self-check" in status["engine"]
