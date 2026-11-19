"""Multi-mode conversation dataset profiles for all 8 business modes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from .acceptance_scope import ACCEPTANCE_MODES, background_noise_scene_items
from .scene_condition_profiles import (
    SCENE_CONDITION_PROFILES as LEGACY_PROFILES,
    SceneConditionProfile,
    SceneConditionScene,
)
from .scene_conversation_profiles import (
    CONVERSATION_MODES,
    CONVERSATION_PROFILES,
    ConversationProfile,
    profile_counts,
)

MODE_SAMPLE_RATE_HZ = {
    "NB": 8_000,
    "WB": 16_000,
    "SWB": 48_000,
}


def sample_rate_for_mode(mode: str) -> int:
    bandwidth = str(mode).split("-", 1)[0]
    return MODE_SAMPLE_RATE_HZ.get(bandwidth, 16_000)


@dataclass(frozen=True)
class ModeConversationProfile:
    scene: SceneConditionScene
    mode: str
    dataset_id: str
    dataset_name: str
    contract_item_ids: tuple[str, ...]
    snr_db: float
    sample_rate_hz: int = 16_000
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
        return {
            "mode": self.mode,
            "direction": "send",
            "scene": self.scene,
            "volume": "nominal",
            "window_ms": None,
            "input_level_db": 0,
        }

    @property
    def conversation_counts(self) -> dict[str, int]:
        return {mode: 20 for mode in CONVERSATION_MODES}


def _dataset_id_for(scene: str, mode: str) -> str:
    if mode == "WB-wired":
        return CONVERSATION_PROFILES[scene].dataset_id
    return hashlib.sha256(
        f"background-noise-conversation-v2:{scene}:{mode}".encode("utf-8")
    ).hexdigest()[:32]


def _build_mode_profile(
    scene: SceneConditionScene,
    mode: str,
) -> ConversationProfile | ModeConversationProfile:
    if mode == "WB-wired":
        return CONVERSATION_PROFILES[scene]
    legacy = LEGACY_PROFILES[scene]
    rows = background_noise_scene_items(mode, scene)
    item_ids = tuple(row["id"] for row in rows)
    if len(item_ids) != 5 or len(set(item_ids)) != 5:
        raise ValueError(f"{scene}/{mode} must expose exactly five contract rows")
    return ModeConversationProfile(
        scene=scene,
        mode=mode,
        dataset_id=_dataset_id_for(scene, mode),
        dataset_name=f"background-noise-{scene}-{mode.lower()}-100-conversation-v2",
        contract_item_ids=item_ids,
        snr_db=legacy.snr_db,
        sample_rate_hz=sample_rate_for_mode(mode),
    )


MODE_CONVERSATION_PROFILES: dict[tuple[str, str], ConversationProfile | ModeConversationProfile] = {
    (scene, mode): _build_mode_profile(scene, mode)  # type: ignore[arg-type]
    for scene in LEGACY_PROFILES
    for mode in sorted(ACCEPTANCE_MODES)
}

PROFILE_BY_DATASET_ID: dict[str, ConversationProfile | ModeConversationProfile | SceneConditionProfile] = {
    profile.dataset_id: profile
    for profile in (
        *LEGACY_PROFILES.values(),
        *CONVERSATION_PROFILES.values(),
        *MODE_CONVERSATION_PROFILES.values(),
    )
}


def profile_mode(profile: Any) -> str:
    mode = getattr(profile, "mode", None)
    if isinstance(mode, str) and mode:
        return mode
    conditions = getattr(profile, "conditions", None)
    if isinstance(conditions, dict) and isinstance(conditions.get("mode"), str):
        return str(conditions["mode"])
    return "WB-wired"


def profile_sample_rate_hz(profile: Any) -> int:
    rate = getattr(profile, "sample_rate_hz", None)
    if isinstance(rate, int) and rate > 0:
        return rate
    return sample_rate_for_mode(profile_mode(profile))


def profile_for_scene(scene: str, mode: str = "WB-wired") -> ConversationProfile | ModeConversationProfile:
    return profile_for_scene_and_mode(scene, mode)


def profile_for_scene_and_mode(
    scene: str,
    mode: str = "WB-wired",
) -> ConversationProfile | ModeConversationProfile:
    try:
        return MODE_CONVERSATION_PROFILES[(scene, mode)]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown scene-condition scene/mode: {scene}/{mode}") from error


def profile_for_dataset(
    dataset_id: str,
) -> ConversationProfile | ModeConversationProfile | SceneConditionProfile:
    try:
        return PROFILE_BY_DATASET_ID[dataset_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown scene-condition dataset: {dataset_id}") from error


__all__ = [
    "CONVERSATION_MODES",
    "CONVERSATION_PROFILES",
    "ConversationProfile",
    "MODE_CONVERSATION_PROFILES",
    "MODE_SAMPLE_RATE_HZ",
    "ModeConversationProfile",
    "PROFILE_BY_DATASET_ID",
    "SceneConditionProfile",
    "SceneConditionScene",
    "profile_counts",
    "profile_for_dataset",
    "profile_for_scene",
    "profile_for_scene_and_mode",
    "profile_mode",
    "profile_sample_rate_hz",
    "sample_rate_for_mode",
]
