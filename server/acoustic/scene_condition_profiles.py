"""Canonical scene-condition dataset identities and contract rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .acceptance_scope import background_noise_scene_items


SceneConditionScene = Literal["sce1", "sce2", "sce3", "sce4"]


@dataclass(frozen=True)
class SceneConditionProfile:
    scene: SceneConditionScene
    dataset_id: str
    dataset_name: str
    dataset_kind: str
    contract_item_ids: tuple[str, ...]
    snr_db: float
    contract_item_count: int = 5

    @property
    def conditions(self) -> dict[str, object]:
        return {
            "mode": "WB-wired",
            "direction": "send",
            "scene": self.scene,
            "volume": "nominal",
            "window_ms": None,
            "input_level_db": 0,
        }


def _profile(
    scene: SceneConditionScene,
    dataset_id: str,
    dataset_kind: str,
    snr_db: float,
) -> SceneConditionProfile:
    rows = background_noise_scene_items("WB-wired", scene)
    item_ids = tuple(row["id"] for row in rows)
    if len(item_ids) != 5 or len(set(item_ids)) != 5:
        raise ValueError(f"{scene} must expose exactly five WB-wired rows")
    return SceneConditionProfile(
        scene=scene,
        dataset_id=dataset_id,
        dataset_name=(
            f"background-noise-{scene}-wb-wired-20-condition-preview-v1"
        ),
        dataset_kind=dataset_kind,
        contract_item_ids=item_ids,
        snr_db=snr_db,
    )


SCENE_CONDITION_PROFILES = {
    "sce1": _profile(
        "sce1",
        "edea6f9a51df87d0a4aefb2303dd8e1f",
        "background_noise_idle_preview",
        15.0,
    ),
    "sce2": _profile(
        "sce2",
        "d71bf7153f51f098fdead44059f65389",
        "background_noise_scene_preview",
        10.0,
    ),
    "sce3": _profile(
        "sce3",
        "73034293a0a2738f7b367e11676f16bf",
        "background_noise_scene_preview",
        6.0,
    ),
    "sce4": _profile(
        "sce4",
        "8783aac64dfaf4746e4798ec7c636e15",
        "background_noise_scene_preview",
        3.0,
    ),
}

PROFILE_BY_SCENE = SCENE_CONDITION_PROFILES
PROFILE_BY_DATASET_ID = {
    profile.dataset_id: profile
    for profile in SCENE_CONDITION_PROFILES.values()
}


def profile_for_scene(scene: str) -> SceneConditionProfile:
    try:
        return PROFILE_BY_SCENE[scene]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unknown scene-condition scene: {scene}") from error


def profile_for_dataset(dataset_id: str) -> SceneConditionProfile:
    try:
        return PROFILE_BY_DATASET_ID[dataset_id]
    except KeyError as error:
        raise ValueError(
            f"unknown scene-condition dataset: {dataset_id}"
        ) from error


__all__ = [
    "PROFILE_BY_DATASET_ID",
    "PROFILE_BY_SCENE",
    "SCENE_CONDITION_PROFILES",
    "SceneConditionProfile",
    "SceneConditionScene",
    "profile_for_dataset",
    "profile_for_scene",
]
