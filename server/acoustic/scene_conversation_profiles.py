"""Versioned conversation coverage alongside immutable twenty-case datasets."""
from __future__ import annotations

from dataclasses import dataclass

from .scene_condition_profiles import (
    SCENE_CONDITION_PROFILES as LEGACY_PROFILES,
    SceneConditionProfile,
    SceneConditionScene,
)

CONVERSATION_MODES = (
    "near_single", "far_single", "double_talk", "transition", "noise_only",
)


@dataclass(frozen=True)
class ConversationProfile:
    scene: SceneConditionScene
    dataset_id: str
    dataset_name: str
    contract_item_ids: tuple[str, ...]
    snr_db: float
    dataset_kind: str = "background_noise_conversation_preview"
    schema_version: str = "scene-condition-preview-v2"
    contract_item_count: int = 5
    case_count: int = 100
    speech_case_count: int = 80
    cng_case_count: int = 20
    observation_count: int = 280
    duration_s: float = 12.0

    @property
    def conditions(self) -> dict[str, object]:
        return {**LEGACY_PROFILES[self.scene].conditions}

    @property
    def conversation_counts(self) -> dict[str, int]:
        return {mode: 20 for mode in CONVERSATION_MODES}


_IDENTITIES = {
    "sce1": "cf29475e343052eaffa73dfebad1b0b0",
    "sce2": "1f5451c6dbb3af6245dd7e6fb3d7338f",
    "sce3": "093445541b9d5273419d844b18d209c8",
    "sce4": "f7f69b9900f87bea7546365b49d23af3",
}
CONVERSATION_PROFILES = {
    scene: ConversationProfile(
        scene=legacy.scene,
        dataset_id=_IDENTITIES[scene],
        dataset_name=f"background-noise-{scene}-wb-wired-100-conversation-v2",
        contract_item_ids=legacy.contract_item_ids,
        snr_db=legacy.snr_db,
    )
    for scene, legacy in LEGACY_PROFILES.items()
}
PROFILE_BY_DATASET_ID = {
    profile.dataset_id: profile
    for profile in (*LEGACY_PROFILES.values(), *CONVERSATION_PROFILES.values())
}


def profile_for_scene(scene: str) -> ConversationProfile:
    try:
        return CONVERSATION_PROFILES[scene]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown scene-condition scene: {scene}") from error


def profile_for_dataset(dataset_id: str) -> ConversationProfile | SceneConditionProfile:
    try:
        return PROFILE_BY_DATASET_ID[dataset_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown scene-condition dataset: {dataset_id}") from error


def profile_counts(profile) -> tuple[int, int, int, int]:
    return (
        getattr(profile, "case_count", 20),
        getattr(profile, "speech_case_count", 10),
        getattr(profile, "cng_case_count", 10),
        getattr(profile, "observation_count", 50),
    )
