"""Stable contract identities and acquisition conditions for batch assessment."""
from __future__ import annotations

import json
import re
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "cpp/evaluation/nv200-test-contract.json"


def contract_items() -> list[dict]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["items"]


def required_conditions(row: dict) -> dict:
    name = row["test_name"].lower()
    scene = re.search(r"sce\s*([1-4])", name)
    window = re.search(r"(50|100)\s*ms", name)
    volume = "maximum" if re.search(r"\b(maximum|max)\b", name) else "nominal"
    if "recommended" in name:
        volume = "recommended"
    level = -5 if "-5db" in name.replace(" ", "") else 5 if "+5db" in name.replace(" ", "") else 0
    return {
        "mode": row["sheet"],
        "direction": row["direction"],
        "scene": f"sce{scene.group(1)}" if scene else "baseline",
        "volume": volume,
        "window_ms": int(window.group(1)) if window else None,
        "input_level_db": level,
    }


def matching_conditions(row: dict, conditions: dict) -> bool:
    expected = required_conditions(row)
    return all(conditions.get(key) == value for key, value in expected.items())
