"""R2 identities reuse the audited registry with an isolated module namespace.

R1 source bytes and generated evidence remain intact. R2 adds an observation
source-selection gate so short double talk survives delivery as PCM16.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_name = "server.acoustic._dataset_revision_r2_implementation"
_path = Path(__file__).with_name("dataset_revisions.py")
if _name in sys.modules:
    _implementation = sys.modules[_name]
else:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("cannot load the revision registry implementation")
    _implementation = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _implementation
    _spec.loader.exec_module(_implementation)
    _implementation.DATASET_REVISION = "measurement-20261117-r2"
    _implementation.ACTIVATION_PATH = (
        _implementation.PROJECT_ROOT / "build/dataset-revision-20261117/ACTIVATION-r2.json"
    )
    _implementation.REVISION_SCENE_PROFILES = {
        (scene, mode): _implementation._revision_scene_profile(scene, mode)
        for scene in _implementation.SCENES for mode in _implementation.MODES
    }
    _implementation.REVISION_STRESS_PROFILES = {
        (scope, mode): _implementation.RevisionStressProfile(scope, mode)
        for scope in _implementation.STRESS_SCOPES for mode in _implementation.MODES
    }
    _implementation._REVISION_IDS = {
        profile.dataset_id for profile in (
            *_implementation.REVISION_SCENE_PROFILES.values(),
            *_implementation.REVISION_STRESS_PROFILES.values(),
        )
    }
    _extra_sources = frozenset({
        "server/acoustic/dataset_revisions_v2.py",
        "scripts/create-revised-acoustic-datasets-v2.py",
        "scripts/dataset_revision_observation.py",
    })
    _implementation._SCENE_REQUIRED_SOURCES |= _extra_sources
    _implementation._STRESS_REQUIRED_SOURCES |= _extra_sources

for _export in _implementation.__all__:
    globals()[_export] = getattr(_implementation, _export)
__all__ = list(_implementation.__all__)
