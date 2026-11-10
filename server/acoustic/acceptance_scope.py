"""Select explicit contract subsets without changing the imported dataset."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from .acceptance_contract import contract_items, required_conditions


BASIC_UPLINK_FAMILIES = frozenset({
    "slr", "frequency_response", "listening_quality", "idle_noise_level",
    "idle_noise_peaks", "out_of_band", "hats_snr",
})
BACKGROUND_NOISE_FAMILIES = frozenset({
    "g_mos", "n_mos", "s_mos", "cng_level", "cng_spectrum",
})
ACCEPTANCE_MODES = frozenset({
    "NB-wired", "NB-wireless", "NB-bt",
    "WB-wired", "WB-wireless", "WB-bt",
    "SWB-wired", "SWB-wireless",
})
SUB_SCOPES = frozenset({
    "basic_uplink", "basic_downlink", "echo", "double_talk",
    "background_noise",
})
TEST_SCOPES = frozenset({"full", *SUB_SCOPES})
BackgroundNoiseScene = Literal["sce1", "sce2", "sce3", "sce4"]
BACKGROUND_NOISE_SCENES = frozenset({"sce1", "sce2", "sce3", "sce4"})
BACKGROUND_NOISE_SCENE_CELLS = {
    "sce1": frozenset({"B24", "B25", "B26", "B27", "B28"}),
    "sce2": frozenset({"B29", "B30", "B31", "B32", "B33"}),
    "sce3": frozenset({"B34", "B35", "B36", "B37", "B38"}),
    "sce4": frozenset({"B39", "B40", "B41", "B42", "B43"}),
}


def _expected_item_count(
    test_scope: str,
    mode: str,
    scene: BackgroundNoiseScene | None = None,
) -> int:
    if test_scope == "basic_uplink":
        return 8
    if test_scope == "basic_downlink":
        return 8 if mode.startswith("SWB-") else 9
    if test_scope == "echo":
        return 3 if mode.startswith("SWB-") else 2
    if test_scope == "double_talk":
        return 2
    if test_scope == "background_noise":
        return 5 if scene is not None else 20
    raise ValueError("未知测试范围")


def _validate_scene(
    test_scope: str,
    scene: BackgroundNoiseScene | None,
) -> None:
    if scene is None:
        return
    if test_scope != "background_noise":
        raise ValueError("只有背景噪声测试可以指定场景")
    if scene not in BACKGROUND_NOISE_SCENES:
        raise ValueError("未知背景噪声场景")


def scope_items(
    test_scope: str,
    mode: str,
    scene: BackgroundNoiseScene | None = None,
) -> list[dict[str, Any]]:
    """Return the validated contract subset for one executable test scope."""
    if test_scope not in SUB_SCOPES:
        raise ValueError("未知测试范围")
    if not isinstance(mode, str) or mode not in ACCEPTANCE_MODES:
        raise ValueError("请选择有效的业务模式")
    _validate_scene(test_scope, scene)
    rows = contract_items()
    if test_scope == "basic_uplink":
        selected = [
            row for row in rows
            if row["sheet"] == mode
            and row["direction"] == "send"
            and row["family"] in BASIC_UPLINK_FAMILIES
        ]
    elif test_scope == "basic_downlink":
        selected = [
            row for row in rows
            if row["sheet"] == mode and row["direction"] == "receive"
        ]
    elif test_scope == "echo":
        selected = [
            row for row in rows
            if row["sheet"] == mode and row["family"] == "tclw"
        ]
    elif test_scope == "double_talk":
        selected = [
            row for row in rows
            if row["sheet"] == mode
            and row["family"] == "double_talk_classification"
        ]
    else:
        selected = [
            row for row in rows
            if row["sheet"] == mode
            and row["family"] in BACKGROUND_NOISE_FAMILIES
        ]
        if scene is not None:
            cells = BACKGROUND_NOISE_SCENE_CELLS[scene]
            selected = [
                row for row in selected
                if row["source_cell"] in cells
            ]
    expected_count = _expected_item_count(test_scope, mode, scene)
    if (
        len(selected) != expected_count
        or len({row["id"] for row in selected}) != expected_count
    ):
        raise ValueError(
            f"{test_scope}/{mode} 合同必须包含 {expected_count} 个唯一测项"
        )
    return selected


def basic_uplink_items(mode: str) -> list[dict[str, Any]]:
    return scope_items("basic_uplink", mode)


def basic_downlink_items(mode: str) -> list[dict[str, Any]]:
    return scope_items("basic_downlink", mode)


def echo_items(mode: str) -> list[dict[str, Any]]:
    return scope_items("echo", mode)


def double_talk_items(mode: str) -> list[dict[str, Any]]:
    return scope_items("double_talk", mode)


def background_noise_items(mode: str) -> list[dict[str, Any]]:
    return scope_items("background_noise", mode)


def background_noise_scene_items(
    mode: str,
    scene: BackgroundNoiseScene,
) -> list[dict[str, Any]]:
    return scope_items("background_noise", mode, scene)


def select_batch_scope(
    manifest: dict,
    *,
    test_scope: str = "full",
    mode: str | None = None,
    scene: BackgroundNoiseScene | None = None,
) -> tuple[list[dict], list[dict]]:
    """Keep acquisition conditions; calculate only selected contract rows."""
    if test_scope not in TEST_SCOPES:
        raise ValueError("未知测试范围")
    if test_scope == "full":
        if mode is not None or scene is not None:
            raise ValueError("完整批次不能指定单一模式或场景")
        return contract_items(), deepcopy(manifest["cases"])
    rows = scope_items(test_scope, mode, scene)
    by_id = {row["id"]: row for row in rows}
    cases = []
    seen = []
    for original in manifest["cases"]:
        selected = [item_id for item_id in original["item_ids"] if item_id in by_id]
        if not selected:
            continue
        if any(required_conditions(by_id[item_id]) != original["conditions"] for item_id in selected):
            raise ValueError("基础上行测项与采集工况不一致")
        case = deepcopy(original)
        case["item_ids"] = selected
        cases.append(case)
        seen.extend(selected)
    if (
        len(seen) != len(rows)
        or len(set(seen)) != len(rows)
        or set(seen) != set(by_id)
    ):
        raise ValueError(
            f"测试集未唯一覆盖所选模式的 {len(rows)} 项{test_scope}测试"
        )
    return rows, cases
