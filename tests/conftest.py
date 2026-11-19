"""Collect the active API contract while keeping the retired V1 suite explicit."""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RETIRED_V1_SUITE = Path(__file__).with_name("test_acceptance_scene_condition_api.py")
RETIREMENT = PROJECT / "benchmarks/fixtures/retired-acoustic-metadata-v1/retirement.json"


def pytest_addoption(parser):
    parser.addoption(
        "--include-retired-v1", action="store_true", default=False,
        help="Collect the archived 20-case HTTP-200 suite; requires its historical API environment.",
    )


def pytest_ignore_collect(collection_path, config):
    # Only the incompatible V1 preview suite is isolated. New R2, retirement,
    # source-integrity and historical-result schema tests remain collected.
    return (Path(collection_path) == RETIRED_V1_SUITE
            and RETIREMENT.is_file()
            and not config.getoption("--include-retired-v1"))


def pytest_report_header(config):
    if RETIREMENT.is_file() and not config.getoption("--include-retired-v1"):
        return ("Archived V1 preview HTTP-200 suite excluded after retirement; "
                "R2 and HTTP-410 contracts remain active. Use --include-retired-v1 only in a historical environment.")


# These two files test the engine with synthetic 20/100-case stores in tmp_path.
# Keep the production retirement policy in dedicated API tests; unwrap it only
# within the existing synthetic unit-test harness, never in the application.
import pytest


@pytest.fixture(autouse=True)
def isolate_synthetic_store_engine_tests(request, monkeypatch):
    if request.path.name not in {"test_scene_condition_batch.py", "test_conversation_batches.py"}:
        return
    from server.acoustic import acceptance_batch, scene_condition_batch
    from server.acoustic import acoustic_stress_batch, scene_condition_batch_api, acceptance_api
    targets = (
        (acceptance_batch, "get_dataset"),
        (acceptance_batch, "start_batch"),
        (acceptance_batch, "run_batch"),
        (scene_condition_batch, "_validated_dataset"),
        (scene_condition_batch, "start_batch"),
        (scene_condition_batch, "run_batch"),
        (acoustic_stress_batch, "start_batch"),
        (scene_condition_batch_api, "start_batch"),
        (acceptance_api, "start_batch"),
    )
    for module, name in targets:
        function = getattr(module, name)
        if hasattr(function, "__wrapped__"):
            monkeypatch.setattr(module, name, function.__wrapped__)


def pytest_collection_modifyitems(config, items):
    # Explicit file arguments bypass pytest_ignore_collect. Keep the same
    # opt-in rule there and report deselection instead of silently skipping.
    if not RETIREMENT.is_file() or config.getoption("--include-retired-v1"):
        return
    archived = [item for item in items if Path(item.path) == RETIRED_V1_SUITE]
    if archived:
        items[:] = [item for item in items if Path(item.path) != RETIRED_V1_SUITE]
        config.hook.pytest_deselected(items=archived)
