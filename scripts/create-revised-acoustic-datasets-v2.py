#!/usr/bin/env python3
"""Build R2 datasets, adding an activity gate for the short overlap source."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT, PROJECT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
sys.dont_write_bytecode = True


def _base_entry():
    from server.acoustic import dataset_revisions as first
    from server.acoustic import dataset_revisions_v2 as current
    names = ("DATASET_REVISION", "install_revision_profiles", "revision_scene_profile", "revision_stress_profile")
    prior = {name: getattr(first, name) for name in names}
    # Only this isolated generator process sees these aliases. No source file
    # or existing dataset is modified, and the public website uses R2's hook.
    first.DATASET_REVISION = current.DATASET_REVISION
    first.install_revision_profiles = current.install_revision_profiles
    first.revision_scene_profile = current.revision_scene_profile
    first.revision_stress_profile = current.revision_stress_profile
    path = PROJECT / "scripts/create-revised-acoustic-datasets.py"
    spec = importlib.util.spec_from_file_location("itu_revision_r2_entry_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R1 generation helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, first, prior


def configure(kind, condition, mode):
    base, first, prior = _base_entry()
    try:
        generator = base.configure_scene(condition, mode) if kind == "scene" else base.configure_stress(condition, mode)
    finally:
        for name, value in prior.items():
            setattr(first, name, value)
    if kind == "stress":
        from dataset_revision_observation import install_observation_source_selection
        install_observation_source_selection(generator)
        original_render = generator.STRESS.render_stress_signals

        def guarded_render(near, far, noise, rate, **kwargs):
            signals, design = original_render(near, far, noise, rate, **kwargs)
            if kwargs["category"] == "short_overlap":
                import numpy as np
                window = slice(round(6.5 * rate), round(7.0 * rate))
                delivered_near = generator.CONVERSATION.encode(signals["reference"], rate)[1]
                delivered_noise = generator.CONVERSATION.encode(signals["noise"], rate)[1]
                near_rms = float(np.sqrt(np.mean(delivered_near[window] ** 2)))
                noise_rms = float(np.sqrt(np.mean(delivered_noise[window] ** 2)))
                if near_rms <= 0 or noise_rms <= 0:
                    raise ValueError("short overlap has no resolvable PCM16 near/noise")
                measured = 20 * math.log10(near_rms / noise_rms)
                error = abs(measured - float(design["noise_snr_db_target"]))
                if error >= 0.1:
                    raise ValueError(f"delivered short-overlap PCM16 SNR error {error:.9f} dB exceeds 0.1")
                design["noise_snr_db_measured_post_pcm_overlap"] = measured
                design["noise_snr_db_error_post_pcm_overlap"] = error
            return signals, design

        generator.STRESS.render_stress_signals = guarded_render
    base._bind_sources(generator, [
        "server/acoustic/dataset_revisions_v2.py",
        "scripts/create-revised-acoustic-datasets-v2.py",
        "scripts/dataset_revision_observation.py",
    ])
    return generator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("scene", "stress"), required=True)
    parser.add_argument("--scene", choices=("sce1", "sce2", "sce3", "sce4"))
    parser.add_argument("--scope", choices=("echo", "double_talk"))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    condition = args.scene if args.kind == "scene" else args.scope
    if condition is None or (args.kind == "scene" and args.scope) or (args.kind == "stress" and args.scene):
        parser.error("select exactly one condition matching the dataset kind")
    generator = configure(args.kind, condition, args.mode)
    generator._assert_resource_floor()
    if args.validate_only:
        factory = generator.SourceBank if args.kind == "scene" else generator.CONVERSATION.SourceBank
        cases, blobs, _ = generator._build_cases(factory())
        result = {"status": "VALIDATED_IN_MEMORY", "case_count": len(cases), "unique_wavs": len(blobs)}
    else:
        result = generator.build(verify_store=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
