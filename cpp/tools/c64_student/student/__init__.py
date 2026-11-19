"""Distilled C64 residual-noise student package."""

from .export import export_checkpoint
from .model import (
    CausalFeatureState,
    FEATURE_CONTRACT_VERSION,
    FEATURE_DIM,
    HIDDEN_SIZE,
    ResidualStudent,
    compose_causal_frame_features,
    compose_causal_frame_features_with_state,
    project_fft_power_to_bands,
)

__all__ = (
    "CausalFeatureState",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_DIM",
    "HIDDEN_SIZE",
    "ResidualStudent",
    "compose_causal_frame_features",
    "compose_causal_frame_features_with_state",
    "export_checkpoint",
    "project_fft_power_to_bands",
)
