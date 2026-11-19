"""Stress-workload hooks for the shared scene-condition batch engine."""
from __future__ import annotations
from server.aec_routes import admit_aec_route, aec_route_requires_capability

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import uuid

import soundfile as sf

from . import acceptance_batch
from .acceptance_contract import required_conditions
from .acceptance_scene_conditions import ROLES, _safe_relative_path
from .acoustic_stress_models import (
    validate_activity_segments,
    validate_stress_diagnostics,
)
from .acoustic_stress_profiles import (
    CATEGORIES,
    StressProfile,
    stress_profile_for_dataset,
)
from .acoustic_stress_store import (
    EXPECTED_CASE_COUNT,
    EXPECTED_OBSERVATION_COUNT,
    SCENE,
    load_stress_dataset,
)


SCHEMA = "scene-condition-batch-v1"
_BATCH_ID = re.compile(r"^[a-f0-9]{32}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ITEM_STATUSES = {
    "pending",
    "running",
    "review_required",
    "not_measured",
    "invalid",
}
_TERMINAL_CASE_STATUSES = {"completed", "failed"}
_TERMINAL_BATCH_STATUSES = {"completed", "failed"}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_output_alignment(execution: dict[str, Any]) -> None:
    alignment = execution.get("output_alignment", {})
    if not isinstance(alignment, dict):
        raise ValueError("stress execution output_alignment is invalid")
    tx_alignment = alignment.get("tx")
    if tx_alignment is None:
        return
    if not isinstance(tx_alignment, dict):
        raise ValueError("stress execution TX alignment is invalid")
    filename = tx_alignment.get("analysis_filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError("stress execution TX alignment filename is invalid")


def _engine():
    # Lazy import keeps the shared engine -> stress hook -> engine cycle safe.
    from . import scene_condition_batch

    return scene_condition_batch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_by_id(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for row in dataset["contracts"]
    }


def _case_contract(
    case: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item_ids = case.get("contract_item_ids")
    if (
        not isinstance(item_ids, list)
        or len(item_ids) != 1
        or not isinstance(item_ids[0], str)
        or item_ids[0] not in contracts
    ):
        raise ValueError("stress case must contain exactly one contract row")
    contract = contracts[item_ids[0]]
    if case.get("conditions") != required_conditions(contract):
        raise ValueError(
            "stress case conditions differ from required_conditions"
        )
    return contract


def validated_dataset(
    dataset_id: str,
    requested_scene: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and re-check the stress store before any snapshot is written."""
    profile = stress_profile_for_dataset(dataset_id)
    dataset = load_stress_dataset(dataset_id)
    if requested_scene not in (None, SCENE):
        raise ValueError("stress workload only supports the baseline scene")
    if (
        dataset.get("dataset_id") != dataset_id
        or dataset.get("dataset_name") != profile.dataset_name
        or dataset.get("dataset_kind") != "acoustic_stress_preview"
        or dataset.get("schema_version") != profile.schema_version
        or dataset.get("mode") != profile.mode
        or dataset.get("test_scope") != profile.scope
        or dataset.get("scene") != SCENE
        or dataset.get("case_count") != EXPECTED_CASE_COUNT
        or dataset.get("observation_count") != EXPECTED_OBSERVATION_COUNT
        or dataset.get("category_counts") != profile.category_counts
    ):
        raise ValueError("stress dataset identity or profile counts differ")
    contracts = _contract_by_id(dataset)
    if set(contracts) != set(profile.contract_item_ids):
        raise ValueError("stress dataset contract rows differ from profile")
    seen_categories: dict[str, set[int]] = {
        category: set() for category in CATEGORIES[profile.scope]
    }
    for case in dataset["cases"]:
        contract = _case_contract(case, contracts)
        provenance = case.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("stress case provenance is invalid")
        category = provenance.get("workload_category")
        index = provenance.get("category_index")
        if (
            not isinstance(category, str)
            or category not in seen_categories
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 1 <= index <= 20
        ):
            raise ValueError("stress case category/index is invalid")
        seen_categories[category].add(index)
        validate_activity_segments(
            provenance,
            scope=profile.scope,
            mode=profile.mode,
        )
        if contract["id"] not in case["contract_item_ids"]:
            raise ValueError("stress case contract row is not bound")
    if any(
        indices != set(range(1, 21))
        for indices in seen_categories.values()
    ):
        raise ValueError("stress category index coverage is incomplete")
    return dataset, contracts


def execution_case(
    case: dict[str, Any],
    input_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    provenance = case.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("stress case provenance is invalid")
    delays = provenance.get("beam_delays_ms")
    if not isinstance(delays, list) or not delays:
        if input_paths is not None:
            info = sf.info(input_paths["microphone"])
            delays = [0.0] * int(info.channels)
        else:
            delays = [0.0]
    render_metadata = provenance.get("render_metadata")
    has_echo_spec = isinstance(render_metadata, dict) and (
        (
            isinstance(render_metadata.get("echo_path_segments"), list)
            and len(render_metadata["echo_path_segments"]) > 0
        )
        or isinstance(render_metadata.get("echo_paths"), list)
    )
    dynamic_microphone_from_post_rx = (
        case["conditions"].get("direction") != "receive"
        and isinstance(render_metadata, dict)
        and isinstance(render_metadata.get("near_arrival_delays_samples"), list)
        and isinstance(render_metadata.get("noise_spatial"), dict)
        and has_echo_spec
    )
    return {
        "id": case["case_id"],
        "conditions": deepcopy(case["conditions"]),
        "item_ids": list(case["contract_item_ids"]),
        "beam_delays_ms": list(delays),
        "analysis_start_s": float(provenance.get("analysis_start_s", 0.0)),
        "dynamic_microphone_from_post_rx": dynamic_microphone_from_post_rx,
        "provenance": deepcopy(provenance),
    }


def _initial_items(
    dataset: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    identifier: str,
) -> list[dict[str, Any]]:
    items = []
    for case in dataset["cases"]:
        contract = _case_contract(case, contracts)
        item_id = case["contract_item_ids"][0]
        items.append(
            {
                **deepcopy(contract),
                "contract_id": item_id,
                "item_id": item_id,
                "observation_id": f"{case['case_id']}:{item_id}",
                "case_id": case["case_id"],
                "mode": dataset["mode"],
                "status": "pending",
                "value": None,
                "metric": None,
                "reason": "等待 stress workload 执行",
                "evidence_binding": {
                    **deepcopy(case["conditions"]),
                    "case_id": case["case_id"],
                    "report_id": None,
                    "batch_id": identifier,
                    "dataset_id": dataset["dataset_id"],
                    "input_sha256": deepcopy(case["signals_sha256"]),
                    "output_sha256": {},
                },
            }
        )
    if len(items) != EXPECTED_OBSERVATION_COUNT:
        raise ValueError("stress snapshot must contain 100 observations")
    return items


def _initial_cases(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "status": "pending",
            "report_id": None,
            "conditions": deepcopy(case["conditions"]),
            "contract_item_ids": list(case["contract_item_ids"]),
            "signals": deepcopy(case["signals"]),
            "signals_sha256": deepcopy(case["signals_sha256"]),
            "input_sha256": deepcopy(case["signals_sha256"]),
            "provenance": deepcopy(case["provenance"]),
            "tx_visual": None,
            "rx_visual": None,
        }
        for case in dataset["cases"]
    ]


def build_snapshot(
    dataset: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    identifier: str,
    algorithm_cluster_id: str,
    configuration: dict[str, Any],
    catalog_sha256: str,
    cli_sha256: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    engine = _engine()
    profile = stress_profile_for_dataset(dataset["dataset_id"])
    items = _initial_items(dataset, contracts, identifier)
    cases = _initial_cases(dataset)
    timestamp = engine._now()
    snapshot = {
        "schema": SCHEMA,
        "schema_version": SCHEMA,
        "batch_id": identifier,
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "name": dataset["dataset_name"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_assets_sha256": dataset["dataset_assets_sha256"],
        "algorithm_cluster_id": algorithm_cluster_id,
        "configuration": deepcopy(configuration),
        "catalog_sha256": catalog_sha256,
        "cli_sha256": cli_sha256,
        "status": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "selected_mode": dataset["mode"],
        "selected_scene": SCENE,
        "test_scope": profile.scope,
        "stress_scope": profile.scope,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "contract_item_count": profile.contract_item_count,
        "dataset_schema_version": profile.schema_version,
        "case_count": EXPECTED_CASE_COUNT,
        "preview_condition_group_count": EXPECTED_CASE_COUNT,
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "category_counts": deepcopy(profile.category_counts),
        "progress": {
            "completed_cases": 0,
            "total_cases": EXPECTED_CASE_COUNT,
            "completed_items": 0,
            "total_items": EXPECTED_OBSERVATION_COUNT,
        },
        "summary": engine._summary(items),
        "items": items,
        "cases": cases,
    }
    if profile.mode.startswith("NB-"):
        from server.rate_paths import require_primary_path

        snapshot["native_nb8_identity"] = (
            acceptance_batch.native_nb8_identity()
        )
        snapshot["rate_paths"] = {
            "8000": require_primary_path(8_000).as_dict()
        }
    if case_id is not None:
        return engine._filter_snapshot_to_single_case(
            snapshot,
            dataset,
            case_id,
        )
    return snapshot


def _validate_case_diagnostics(
    case: dict[str, Any],
    *,
    profile: StressProfile,
) -> None:
    diagnostics = case.get("stress_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("completed stress case lacks stress_diagnostics")
    provenance = case.get("provenance")
    output_sha256 = case.get("output_sha256")
    if (
        not isinstance(provenance, dict)
        or not isinstance(output_sha256, dict)
        or set(output_sha256) not in ({"tx"}, {"rx", "tx"})
        or not _is_sha256(output_sha256.get("tx"))
        or (
            "rx" in output_sha256
            and not _is_sha256(output_sha256.get("rx"))
        )
    ):
        raise ValueError("stress case provenance/output is missing")
    if "rx" in output_sha256:
        echo_reference = case.get("execution", {}).get("echo_reference")
        if (
            not isinstance(echo_reference, dict)
            or echo_reference.get("source")
            not in {
                "vcp_rx_processed",
                "native_nb8_rx_processed",
            }
            or echo_reference.get("processed_output") != "rx"
            or echo_reference.get("processed_sha256")
            != output_sha256["rx"]
            or echo_reference.get("tx_echo_argument_sha256")
            != output_sha256["rx"]
        ):
            raise ValueError(
                "stress TX does not prove RX-processed echo reference"
            )
    segments = validate_activity_segments(
        provenance,
        scope=profile.scope,
        mode=profile.mode,
    )
    validate_stress_diagnostics(
        diagnostics,
        expected_case_id=case["case_id"],
        expected_tx_sha256=output_sha256["tx"],
        expected_scope=profile.scope,
        expected_mode=profile.mode,
        expected_category=provenance["workload_category"],
        expected_segments=segments,
    )


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    engine = _engine()
    if not isinstance(snapshot, dict):
        raise ValueError("stress snapshot must be an object")
    missing_snapshot_fields = [
        field
        for field in engine.SNAPSHOT_REQUIRED_FIELDS
        if field not in snapshot
    ]
    if missing_snapshot_fields:
        raise ValueError(
            "stress snapshot missing fields: "
            + ", ".join(missing_snapshot_fields)
        )
    if (
        snapshot.get("schema_version") != SCHEMA
        or snapshot.get("schema") != SCHEMA
        or not isinstance(snapshot.get("batch_id"), str)
        or _BATCH_ID.fullmatch(snapshot["batch_id"]) is None
        or not isinstance(snapshot.get("dataset_id"), str)
        or _BATCH_ID.fullmatch(snapshot["dataset_id"]) is None
    ):
        raise ValueError("stress snapshot schema or id is invalid")
    if "name" in snapshot and snapshot["name"] != snapshot.get("dataset_name"):
        raise ValueError("stress snapshot compatibility name is invalid")
    profile = stress_profile_for_dataset(snapshot["dataset_id"])
    selected_case_id = engine._single_case_selected_id(snapshot)
    single_case = selected_case_id is not None
    for field in (
        "dataset_manifest_sha256",
        "dataset_assets_sha256",
        "catalog_sha256",
        "cli_sha256",
    ):
        if not _is_sha256(snapshot.get(field)):
            raise ValueError(f"stress snapshot hash is invalid: {field}")
    for field in (
        "dataset_name",
        "algorithm_cluster_id",
        "created_at",
        "updated_at",
    ):
        if not isinstance(snapshot.get(field), str) or not snapshot[field]:
            raise ValueError(f"stress snapshot field is invalid: {field}")
    if not isinstance(snapshot.get("configuration"), dict):
        raise ValueError("stress snapshot configuration is invalid")
    if snapshot.get("status") not in {
        "queued",
        "running",
        "completed",
        "failed",
    }:
        raise ValueError("stress snapshot status is invalid")
    if (
        snapshot.get("dataset_name") != profile.dataset_name
        or snapshot.get("selected_mode") != profile.mode
        or snapshot.get("selected_scene") != SCENE
        or snapshot.get("test_scope") != profile.scope
        or snapshot.get("stress_scope") != profile.scope
        or snapshot.get("dataset_schema_version") != profile.schema_version
        or snapshot.get("contract_item_count") != profile.contract_item_count
        or snapshot.get("preview_condition_group_count") != EXPECTED_CASE_COUNT
        or not isinstance(snapshot.get("category_counts"), dict)
        or snapshot.get("standard_compliance") is not False
        or snapshot.get("formal_test_status") != "not_measured"
    ):
        raise ValueError("stress snapshot profile identity differs")
    cases = snapshot.get("cases")
    items = snapshot.get("items")
    expected_case_count = 1 if single_case else EXPECTED_CASE_COUNT
    expected_observation_count = (
        1 if single_case else EXPECTED_OBSERVATION_COUNT
    )
    if (
        not isinstance(cases, list)
        or len(cases) != expected_case_count
        or not isinstance(items, list)
        or len(items) != expected_observation_count
        or not isinstance(snapshot.get("case_count"), int)
        or isinstance(snapshot.get("case_count"), bool)
        or snapshot.get("case_count") != expected_case_count
        or not isinstance(snapshot.get("observation_count"), int)
        or isinstance(snapshot.get("observation_count"), bool)
        or snapshot.get("observation_count") != expected_observation_count
    ):
        raise ValueError("stress snapshot counts are invalid")
    if any(not isinstance(case, dict) for case in cases) or any(
        not isinstance(row, dict) for row in items
    ):
        raise ValueError("stress snapshot cases/items entries are invalid")
    if single_case:
        if (
            not isinstance(snapshot.get("dataset_case_count"), int)
            or isinstance(snapshot.get("dataset_case_count"), bool)
            or snapshot.get("dataset_case_count") != EXPECTED_CASE_COUNT
            or not isinstance(
                snapshot.get("dataset_observation_count"),
                int,
            )
            or isinstance(
                snapshot.get("dataset_observation_count"),
                bool,
            )
            or snapshot.get("dataset_observation_count")
            != EXPECTED_OBSERVATION_COUNT
        ):
            raise ValueError("stress single-case source counts are invalid")
    elif snapshot.get("category_counts") != profile.category_counts:
        raise ValueError("stress snapshot profile identity differs")
    summary = snapshot.get("summary")
    progress = snapshot.get("progress")
    if not isinstance(summary, dict) or summary != engine._summary(items):
        raise ValueError("stress snapshot summary differs")
    if not isinstance(progress, dict) or progress != engine._progress(snapshot):
        raise ValueError("stress snapshot progress differs")
    if set(summary) != set(engine.SUMMARY_FIELDS):
        raise ValueError("stress summary fields differ")
    if set(progress) != set(engine.PROGRESS_FIELDS):
        raise ValueError("stress progress fields differ")

    contracts = _contract_by_id({"contracts": profile.contract_rows})
    case_by_id: dict[str, dict[str, Any]] = {}
    category_indices = {
        category: set() for category in CATEGORIES[profile.scope]
    }
    actual_category_counts = {
        category: 0 for category in CATEGORIES[profile.scope]
    }
    item_keys = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("stress case must be an object")
        missing = [field for field in engine.CASE_REQUIRED_FIELDS if field not in case]
        if missing:
            raise ValueError("stress case missing fields: " + ", ".join(missing))
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not _CASE_ID.fullmatch(case_id)
            or case_id in case_by_id
        ):
            raise ValueError("stress case id is invalid or duplicate")
        case_by_id[case_id] = case
        contract = _case_contract(case, contracts)
        if case.get("status") not in {
            "pending",
            "running",
            "completed",
            "failed",
        }:
            raise ValueError("stress case status is invalid")
        report_id = case.get("report_id")
        if report_id is not None and not (
            isinstance(report_id, str)
            and re.fullmatch(r"[a-f0-9]{32}", report_id) is not None
        ):
            raise ValueError("stress case report_id is invalid")
        if case.get("status") == "completed" and report_id is None:
            raise ValueError("completed stress case lacks report_id")
        signals = case.get("signals")
        if not isinstance(signals, dict) or set(signals) != set(ROLES):
            raise ValueError("stress case signal roles are invalid")
        for relative in signals.values():
            _safe_relative_path(relative)
        signals_sha = case.get("signals_sha256")
        input_sha = case.get("input_sha256")
        if (
            not isinstance(signals_sha, dict)
            or set(signals_sha) != set(ROLES)
            or any(not _is_sha256(value) for value in signals_sha.values())
            or input_sha != signals_sha
        ):
            raise ValueError("stress case input hashes differ")
        provenance = case.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("stress case provenance is invalid")
        category = provenance.get("workload_category")
        index = provenance.get("category_index")
        if (
            not isinstance(category, str)
            or category not in category_indices
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 1 <= index <= 20
        ):
            raise ValueError("stress case category/index is invalid")
        category_indices[category].add(index)
        actual_category_counts[category] += 1
        validate_activity_segments(
            provenance,
            scope=profile.scope,
            mode=profile.mode,
        )
        if case.get("status") == "completed":
            output = case.get("output_sha256")
            if (
                not isinstance(output, dict)
                or set(output) not in ({"tx"}, {"rx", "tx"})
                or not _is_sha256(output.get("tx"))
                or (
                    "rx" in output
                    and not _is_sha256(output.get("rx"))
                )
                or not _is_sha256(case.get("assessment_sha256"))
                or not isinstance(case.get("execution"), dict)
                or not isinstance(case.get("tx_visual"), dict)
                or set(case["tx_visual"]) != set(engine.TX_VISUAL_FIELDS)
                or case["tx_visual"].get("source_sha256") != output["tx"]
            ):
                raise ValueError("completed stress case evidence is invalid")
            if "rx" in output:
                rx_visual = case.get("rx_visual")
                if (
                    not isinstance(rx_visual, dict)
                    or set(rx_visual) != set(engine.TX_VISUAL_FIELDS)
                    or rx_visual.get("source_sha256") != output["rx"]
                ):
                    raise ValueError(
                        "completed stress case RX visual evidence is invalid"
                    )
            acceptance_batch.validate_execution_identity(
                snapshot,
                case["execution"],
                require_algorithm_configuration=True,
            )
            if (
                case["execution"].get("output_sha256") != output
                or case["execution"].get("input_sha256") != input_sha
            ):
                raise ValueError("stress case execution identity differs")
            dynamic_microphone = case["execution"].get("dynamic_microphone")
            if dynamic_microphone is not None:
                dynamic_visual = case.get("dynamic_microphone_visual")
                effective_inputs = case.get("effective_input_sha256")
                if (
                    not isinstance(dynamic_microphone, dict)
                    or case.get("dynamic_microphone") != dynamic_microphone
                    or not isinstance(dynamic_visual, dict)
                    or set(dynamic_visual) != set(engine.TX_VISUAL_FIELDS)
                    or dynamic_visual.get("source_sha256")
                    != dynamic_microphone.get("sha256")
                    or not isinstance(effective_inputs, dict)
                    or set(effective_inputs)
                    != {"microphone", "reference", "noise", "far_end"}
                    or any(
                        not _is_sha256(value)
                        for value in effective_inputs.values()
                    )
                    or case["execution"].get("effective_input_sha256")
                    != effective_inputs
                    or effective_inputs.get("far_end") != output.get("rx")
                ):
                    raise ValueError(
                        "completed stress case runtime dynamic microphone evidence is invalid"
                    )
            elif (
                case.get("dynamic_microphone") is not None
                or case.get("dynamic_microphone_visual") is not None
                or case.get("effective_input_sha256") is not None
            ):
                raise ValueError(
                    "stress case declares dynamic microphone without execution proof"
                )
            _validate_output_alignment(case["execution"])
            _validate_case_diagnostics(case, profile=profile)
        elif case.get("status") == "failed":
            pass
        case_item_ids = case.get("contract_item_ids")
        if not isinstance(case_item_ids, list) or case_item_ids != [contract["id"]]:
            raise ValueError("stress case contract identity differs")
    if single_case and set(case_by_id) != {selected_case_id}:
        raise ValueError("stress single-case selected_case_id differs")

    for row in items:
        if not isinstance(row, dict):
            raise ValueError("stress item must be an object")
        missing = [field for field in engine.ITEM_CORE_FIELDS if field not in row]
        if missing:
            raise ValueError("stress item missing fields: " + ", ".join(missing))
        if row.get("status") not in _ITEM_STATUSES:
            raise ValueError("stress item status is invalid")
        case_id = row.get("case_id")
        if case_id not in case_by_id:
            raise ValueError("stress item references unknown case")
        case = case_by_id[case_id]
        contract = _case_contract(case, contracts)
        if row.get("item_id") != contract["id"]:
            raise ValueError("stress item contract identity differs")
        for field in ("id", "family", "source_cell", "direction"):
            if row.get(field) != contract.get(field):
                raise ValueError("stress item crosses contract row")
        if row.get("observation_id") != f"{case_id}:{row['item_id']}":
            raise ValueError("stress observation id differs")
        if row.get("mode") != profile.mode:
            raise ValueError("stress item mode differs")
        value = row.get("value")
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or row.get("status") != "review_required"
        ):
            raise ValueError("stress item value is invalid")
        if row.get("status") == "review_required" and value is None:
            raise ValueError("review_required stress item needs a finite value")
        if value is not None and (
            not isinstance(row.get("metric"), str) or not row["metric"]
        ):
            raise ValueError("finite stress item needs a metric identity")
        if not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError("stress item reason is invalid")
        binding = row.get("evidence_binding")
        if (
            not isinstance(binding, dict)
            or set(binding) != set(engine.EVIDENCE_BINDING_FIELDS)
            or binding.get("case_id") != case_id
            or binding.get("report_id") != case.get("report_id")
            or binding.get("batch_id") != snapshot["batch_id"]
            or binding.get("dataset_id") != snapshot["dataset_id"]
            or binding.get("input_sha256") != case.get("input_sha256")
            or any(
                binding.get(field) != value
                for field, value in case["conditions"].items()
            )
        ):
            raise ValueError("stress item evidence binding differs")
        if row.get("status") == "review_required" and (
            binding.get("output_sha256") != case.get("output_sha256")
        ):
            raise ValueError("reviewable stress item lacks bound TX output")
        item_keys.add((case_id, row["item_id"]))
    if len(item_keys) != expected_observation_count:
        raise ValueError("stress item keys are not unique")
    if snapshot.get("category_counts") != actual_category_counts:
        raise ValueError("stress category_counts differ from selected cases")
    if single_case:
        if sum(actual_category_counts.values()) != 1:
            raise ValueError("stress single-case category count differs")
    elif any(
        indices != set(range(1, 21))
        for indices in category_indices.values()
    ):
        raise ValueError("stress category coverage differs")
    if snapshot.get("status") in _TERMINAL_BATCH_STATUSES and any(
        case.get("status") not in _TERMINAL_CASE_STATUSES
        for case in cases
    ):
        raise ValueError("terminal stress batch has unfinished cases")
    if snapshot.get("status") in _TERMINAL_BATCH_STATUSES and any(
        row.get("status") in {"pending", "running"} for row in items
    ):
        raise ValueError("terminal stress batch has unfinished observations")
    if snapshot.get("status") == "completed" and any(
        case.get("status") == "failed" for case in cases
    ):
        raise ValueError("completed stress batch contains a failed case")
    failed_case_ids = {
        case["case_id"]
        for case in cases
        if case.get("status") == "failed"
    }
    if any(
        row.get("case_id") in failed_case_ids
        and row.get("status") != "invalid"
        for row in items
    ):
        raise ValueError("failed stress case observations must be invalid")


def verify_dataset_frozen(
    snapshot: dict[str, Any],
    dataset: dict[str, Any],
) -> None:
    profile = stress_profile_for_dataset(snapshot["dataset_id"])
    selected_case_id = _engine()._single_case_selected_id(snapshot)
    single_case = selected_case_id is not None
    if (
        dataset.get("dataset_id") != snapshot.get("dataset_id")
        or dataset.get("dataset_name") != snapshot.get("dataset_name")
        or dataset.get("dataset_manifest_sha256")
        != snapshot.get("dataset_manifest_sha256")
        or dataset.get("dataset_assets_sha256")
        != snapshot.get("dataset_assets_sha256")
        or dataset.get("mode") != profile.mode
        or dataset.get("test_scope") != profile.scope
        or dataset.get("scene") != SCENE
        or dataset.get("case_count") != EXPECTED_CASE_COUNT
        or dataset.get("observation_count") != EXPECTED_OBSERVATION_COUNT
    ):
        raise ValueError("stress dataset identity changed after batch start")
    expected_snapshot_case_count = (
        1 if single_case else EXPECTED_CASE_COUNT
    )
    expected_snapshot_observation_count = (
        1 if single_case else EXPECTED_OBSERVATION_COUNT
    )
    if (
        snapshot.get("case_count") != expected_snapshot_case_count
        or snapshot.get("observation_count")
        != expected_snapshot_observation_count
        or snapshot.get("preview_condition_group_count")
        != EXPECTED_CASE_COUNT
    ):
        raise ValueError("stress snapshot counts changed after batch start")
    if single_case:
        selected_source_case = _engine()._selected_source_case(
            dataset,
            selected_case_id,
        )
        selected_category = selected_source_case["provenance"][
            "workload_category"
        ]
        if snapshot.get("category_counts") != (
            _engine()._single_category_counts(
                profile.category_counts,
                selected_category,
            )
        ):
            raise ValueError("stress category_counts changed after batch start")
    elif snapshot.get("category_counts") != profile.category_counts:
        raise ValueError("stress category_counts changed after batch start")
    expected = {case["case_id"]: case for case in dataset["cases"]}
    if single_case:
        selected_source_case = _engine()._selected_source_case(
            dataset,
            selected_case_id,
        )
        expected = {
            selected_case_id: selected_source_case,
        }
    frozen = {case["case_id"]: case for case in snapshot["cases"]}
    if set(expected) != set(frozen):
        raise ValueError("stress case set changed after batch start")
    for case_id, case in expected.items():
        if (
            frozen[case_id].get("signals_sha256") != case.get("signals_sha256")
            or frozen[case_id].get("signals") != case.get("signals")
            or frozen[case_id].get("conditions") != case.get("conditions")
            or frozen[case_id].get("contract_item_ids")
            != case.get("contract_item_ids")
            or frozen[case_id].get("provenance") != case.get("provenance")
        ):
            raise ValueError(f"stress case {case_id} changed after batch start")


def validate_assessment(
    snapshot: dict[str, Any],
    case_record: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    engine = _engine()
    assessment_path = Path(destination) / "assessment.json"
    if assessment_path.is_symlink() or not assessment_path.is_file():
        raise engine.ArtifactIntegrityError(
            "stress assessment.json is missing or unsafe"
        )
    expected_sha = case_record.get("assessment_sha256")
    if (
        not isinstance(expected_sha, str)
        or not re.fullmatch(r"[a-f0-9]{64}", expected_sha)
    ):
        raise engine.ArtifactIntegrityError(
            "stress case lacks assessment_sha256"
        )
    payload = assessment_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise engine.ArtifactIntegrityError(
            "stress assessment SHA-256 differs"
        )
    try:
        assessment = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise engine.ArtifactIntegrityError(
            f"stress assessment JSON is invalid: {error}"
        ) from error
    if (
        not isinstance(assessment, dict)
        or assessment.get("schema_version") != SCHEMA
        or assessment.get("case_id") != case_record.get("case_id")
        or assessment.get("report_id") != case_record.get("report_id")
        or assessment.get("batch_id") != snapshot.get("batch_id")
        or assessment.get("dataset_id") != snapshot.get("dataset_id")
        or assessment.get("conditions") != case_record.get("conditions")
        or assessment.get("input_sha256") != case_record.get("input_sha256")
        or assessment.get("output_sha256") != case_record.get("output_sha256")
        or assessment.get("rx_visual") != case_record.get("rx_visual")
        or assessment.get("provenance") != case_record.get("provenance")
        or assessment.get("stress_diagnostics")
        != case_record.get("stress_diagnostics")
        or assessment.get("dynamic_microphone")
        != case_record.get("dynamic_microphone")
        or assessment.get("dynamic_microphone_visual")
        != case_record.get("dynamic_microphone_visual")
        or assessment.get("effective_input_sha256")
        != case_record.get("effective_input_sha256")
    ):
        raise engine.ArtifactIntegrityError(
            "stress assessment identity differs"
        )
    profile = stress_profile_for_dataset(snapshot["dataset_id"])
    _validate_case_diagnostics(case_record, profile=profile)
    return assessment


def start_batch(
    dataset_id: str,
    algorithm_cluster_id: str,
    *,
    mode: str,
    scene: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    engine = _engine()
    profile = stress_profile_for_dataset(dataset_id)
    if mode != profile.mode or scene != SCENE:
        raise ValueError("stress batch mode or baseline scene differs")
    engine._resource_preflight()
    from server.alignment_runner import resolve_algorithm_cluster, resolve_cli

    preset, _, configuration, catalog_sha256 = resolve_algorithm_cluster(
        algorithm_cluster_id
    )
    algorithm_profile = preset.get("profile")
    if (
        not isinstance(algorithm_profile, str)
        or not algorithm_profile.startswith("vcp")
    ):
        raise ValueError("stress batch only permits a VCP algorithm cluster")
    from server.rate_paths import (
        NB8_CLUSTER_ID,
        sample_rate_for_mode,
        validate_cluster_rate_compatibility,
    )

    target_rate = sample_rate_for_mode(mode)
    validate_cluster_rate_compatibility(
        algorithm_cluster_id,
        target_rate,
        configuration=configuration,
    )
    from server.ns_routes import ns_route_identity, resolve_ns_cli

    is_native_nb8_cluster = algorithm_cluster_id == NB8_CLUSTER_ID
    cli = (
        resolve_cli()
        if is_native_nb8_cluster
        else resolve_ns_cli(resolve_cli(), configuration)
    )
    if cli is None and not (is_native_nb8_cluster and mode.startswith("NB-")):
        raise ValueError("未找到可执行的 VCP CLI")
    selected_ns_identity = None if is_native_nb8_cluster else ns_route_identity(configuration)
    admitted_aec = None
    if (
        not is_native_nb8_cluster
        and not mode.startswith("NB-")
        and isinstance(configuration.get("selection"), dict)
        and aec_route_requires_capability(configuration)
    ):
        admitted_aec = admit_aec_route(
            cli,
            configuration,
            sample_rate=(48000 if target_rate == 48000 else 16000),
            reference_channels=1,
        )
    cli_sha256 = (
        engine._sha256_file(cli)
        if cli is not None
        else acceptance_batch.native_nb8_identity()["speex_cli_sha256"]
    )
    dataset, contracts = validated_dataset(dataset_id, requested_scene=scene)
    if case_id is not None:
        engine._selected_source_case(dataset, case_id)
    identifier = uuid.uuid4().hex
    snapshot = build_snapshot(
        dataset,
        contracts,
        identifier,
        algorithm_cluster_id,
        configuration,
        catalog_sha256,
        cli_sha256,
        case_id=case_id,
    )
    if admitted_aec is not None:
        snapshot["aec_route_capability"] = admitted_aec
    if selected_ns_identity is not None:
        snapshot["ns_route_identity"] = selected_ns_identity
    with acceptance_batch._LOCK:
        if acceptance_batch._ACTIVE or acceptance_batch._LEGACY_ACTIVE:
            raise ValueError("已有测试任务正在执行，请等待完成后再启动")
        engine._resource_preflight()
        acceptance_batch._ACTIVE.add(identifier)
        try:
            engine._create_batch_directories(identifier)
            engine._save(snapshot)
            acceptance_batch._EXECUTOR.submit(engine.run_batch, identifier)
        except Exception:
            acceptance_batch._ACTIVE.discard(identifier)
            raise
    return deepcopy(snapshot)


__all__ = [
    "SCHEMA",
    "build_snapshot",
    "execution_case",
    "start_batch",
    "validate_assessment",
    "validate_snapshot",
    "validated_dataset",
    "verify_dataset_frozen",
]
