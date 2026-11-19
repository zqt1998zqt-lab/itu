"""Dedicated, immutable workload identities for echo and double-talk tests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .acceptance_scope import ACCEPTANCE_MODES, scope_items
from .acceptance_contract import required_conditions

SCOPES = ("echo", "double_talk")
CATEGORIES = {
    "echo": {
        "steady_echo": "稳定远端单讲",
        "convergence": "启动与收敛",
        "delay_change": "回声时延变化",
        "path_change": "回声路径变化",
        "noisy_echo": "噪声中的回声",
    },
    "double_talk": {
        "balanced": "均衡持续双讲",
        "near_dominant": "近端较强双讲",
        "far_dominant": "远端较强双讲",
        "transition": "单讲 / 双讲切换",
        "short_overlap": "短时重叠双讲",
    },
}


@dataclass(frozen=True)
class StressProfile:
    scope: str
    mode: str
    scene: str = "baseline"
    case_count: int = 100
    observation_count: int = 100
    duration_s: float = 12.0
    schema_version: str = "acoustic-stress-preview-v1"
    dataset_kind: str = "acoustic_stress_preview"

    @property
    def dataset_id(self) -> str:
        return hashlib.sha256(
            f"acoustic-stress-v1:{self.scope}:{self.mode}".encode()
        ).hexdigest()[:32]

    @property
    def dataset_name(self) -> str:
        return f"{self.scope}-{self.mode}-100-workloads-v1"

    @property
    def contract_rows(self) -> list[dict]:
        return scope_items(self.scope, self.mode)

    @property
    def contract_item_ids(self) -> tuple[str, ...]:
        return tuple(row["id"] for row in self.contract_rows)

    @property
    def contract_item_count(self) -> int:
        return len(self.contract_item_ids)

    @property
    def category_counts(self) -> dict[str, int]:
        return {category: 20 for category in CATEGORIES[self.scope]}

    def conditions_for_item(self, item_id: str) -> dict:
        row = next((row for row in self.contract_rows if row["id"] == item_id), None)
        if row is None:
            raise ValueError("workload contract row does not belong to scope/mode")
        return required_conditions(row)


PROFILES = {
    (scope, mode): StressProfile(scope, mode)
    for scope in SCOPES for mode in sorted(ACCEPTANCE_MODES)
}
BY_DATASET_ID = {profile.dataset_id: profile for profile in PROFILES.values()}


def is_stress_dataset(dataset_id: object) -> bool:
    return isinstance(dataset_id, str) and dataset_id in BY_DATASET_ID


def stress_profile(scope: str, mode: str) -> StressProfile:
    try:
        return PROFILES[(scope, mode)]
    except (KeyError, TypeError) as error:
        raise ValueError("请选择有效的回声/双讲范围及业务模式") from error


def stress_profile_for_dataset(dataset_id: str) -> StressProfile:
    try:
        return BY_DATASET_ID[dataset_id]
    except (KeyError, TypeError) as error:
        raise ValueError("未知回声/双讲测试集") from error
