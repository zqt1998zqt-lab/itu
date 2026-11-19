#!/usr/bin/env python3
"""Run the existing conversation verifier explicitly against the activated R2."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True


def configured_verifier():
    from server.acoustic.dataset_revisions_v2 import (
        SCENES, install_from_activation, revision_scene_profile,
    )
    if not install_from_activation()["activated"]:
        raise RuntimeError("R2 activation is required; do not fall back to historical audio")
    name = "itu_current_conversation_verifier"
    source = PROJECT / "scripts/verify-conversation-vcp.py"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the conversation verification implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.CONVERSATION_PROFILES = {
        scene: revision_scene_profile(scene, "WB-wired") for scene in SCENES
    }
    return module


if __name__ == "__main__":
    raise SystemExit(configured_verifier().main())
