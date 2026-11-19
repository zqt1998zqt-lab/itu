#!/usr/bin/env python3
"""Build an immutable revision of one conversation/stress dataset.

Legacy generators and stores remain untouched.  This entry point binds revised
sources, signal recipes and identities before invoking their existing checks.
It never overwrites a dataset and does not activate it in the website.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (PROJECT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
sys.dont_write_bytecode = True


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_sources(generator, extra: list[str]) -> None:
    for relative in (
        "scripts/create-revised-acoustic-datasets.py",
        "scripts/dataset_revision_sources.py",
        "scripts/dataset_revision_signal_design.py",
        "server/acoustic/dataset_revisions.py",
        *extra,
    ):
        generator.SOURCE_PATHS[relative] = PROJECT / relative
    generator.IMPORTED_SOURCE_HASHES = {
        relative: _hash(path)
        for relative, path in generator.SOURCE_PATHS.items()
    }


def _bank_factory(original, mode: str):
    from dataset_revision_sources import RevisionSourceBank
    from dataset_revision_signal_design import highway_noise

    class NoiseBank:
        def __init__(self):
            self.bank = RevisionSourceBank(original(), mode)

        def __getattr__(self, key):
            return getattr(self.bank, key)

        def noise(self, business_mode, scene, rate, seed):
            if scene == "sce3":
                return highway_noise(self.bank, business_mode, rate, seed)
            return self.bank.noise(business_mode, scene, rate, seed)

    return NoiseBank


def _crop_for(generator, original):
    from dataset_revision_sources import revision_speech_crop

    def crop(bank, records, *, group_id, role):
        if generator.RATE != 48_000:
            return original(bank, records, group_id=group_id, role=role)
        return revision_speech_crop(
            bank, records, rate=generator.RATE,
            dataset_id=generator.DATASET_ID, group_id=group_id, role=role,
            base_seed=generator.BASE_SEED,
            duration_s=generator.DURATION_S,
        )

    return crop


def _add_case_revision(cases, revision: str):
    for case in cases:
        provenance = case["provenance"]
        design = provenance["render_metadata"]
        provenance["dataset_revision"] = revision
        provenance["near_to_echo_ser_db"] = design.get(
            "near_to_echo_ser_db_measured"
        )
        provenance["echo_to_near_esr_db"] = design.get(
            "echo_to_near_esr_db_measured"
        )
        provenance["transport"] = deepcopy(design["transport"])


def configure_scene(scene: str, mode: str):
    from server.acoustic.dataset_revisions import (
        DATASET_REVISION, install_revision_profiles, revision_scene_profile,
    )
    from dataset_revision_signal_design import add_observed_ser, add_ser_aliases, apply_transport

    install_revision_profiles(activate=False)
    generator = _load(
        "create-background-noise-mode-conversation-dataset.py",
        "itu_revised_scene_generator",
    )
    generator.profile_for_scene_and_mode = revision_scene_profile
    generator.configure_scene(scene, mode)
    generator.SourceBank = _bank_factory(generator.SourceBank, mode)
    generator._speech_crop = _crop_for(generator, generator._speech_crop)
    # Legacy ser_db is Echo/Near (ESR); include both signs, without changing it.
    generator.SER_DB_LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
    original_render = generator.render_signals

    def render(near, far, noise, rate, **kwargs):
        playback, transport = apply_transport(far, rate, mode, kwargs["seed"])
        signals, design = original_render(near, playback, noise, rate, **kwargs)
        design = add_ser_aliases(design)
        design = add_observed_ser(signals, design, rate)
        design["dataset_revision"] = DATASET_REVISION
        design["transport"] = transport
        return signals, design

    generator.render_signals = render
    original_build_cases = generator._build_cases

    def build_cases(bank):
        cases, blobs, counts = original_build_cases(bank)
        _add_case_revision(cases, DATASET_REVISION)
        return cases, blobs, counts

    generator._build_cases = build_cases
    _bind_sources(generator, [])
    return generator


def configure_stress(scope: str, mode: str):
    from server.acoustic.dataset_revisions import (
        DATASET_REVISION, install_revision_profiles, revision_stress_profile,
    )
    from dataset_revision_signal_design import (
        add_observed_ser, add_ser_aliases, apply_transport, calibrate_short_overlap,
    )

    install_revision_profiles(activate=False)
    wrapper = _load(
        "create-acoustic-stress-mode-datasets.py",
        "itu_revised_stress_wrapper",
    )
    generator = wrapper.BASE_GEN
    generator.stress_profile = revision_stress_profile
    wrapper.configure_mode_dataset(scope, mode)
    generator.CONVERSATION.SourceBank = _bank_factory(
        generator.CONVERSATION.SourceBank, mode,
    )
    generator._crop_16k = _crop_for(generator, generator._crop_16k)
    original_render = generator.STRESS.render_stress_signals

    def render(near, far, noise, rate, **kwargs):
        playback, transport = apply_transport(far, rate, mode, kwargs["seed"])
        signals, design = original_render(near, playback, noise, rate, **kwargs)
        if kwargs["category"] == "short_overlap":
            signals, design = calibrate_short_overlap(
                signals, design, rate, float(design["noise_snr_db_target"]),
            )
        design = add_ser_aliases(design)
        design = add_observed_ser(signals, design, rate)
        design["dataset_revision"] = DATASET_REVISION
        design["transport"] = transport
        return signals, design

    generator.STRESS.render_stress_signals = render
    original_build_cases = generator._build_cases

    def build_cases(bank):
        cases, blobs, counts = original_build_cases(bank)
        _add_case_revision(cases, DATASET_REVISION)
        return cases, blobs, counts

    generator._build_cases = build_cases
    _bind_sources(generator, ["scripts/create-acoustic-stress-mode-datasets.py"])
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("scene", "stress"), required=True)
    parser.add_argument("--scene", choices=("sce1", "sce2", "sce3", "sce4"))
    parser.add_argument("--scope", choices=("echo", "double_talk"))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.kind == "scene":
        if not args.scene or args.scope:
            parser.error("scene generation requires --scene and excludes --scope")
        generator = configure_scene(args.scene, args.mode)
        bank_factory = generator.SourceBank
    else:
        if not args.scope or args.scene:
            parser.error("stress generation requires --scope and excludes --scene")
        generator = configure_stress(args.scope, args.mode)
        bank_factory = generator.CONVERSATION.SourceBank
    generator._assert_resource_floor()
    if args.validate_only:
        cases, blobs, counts = generator._build_cases(bank_factory())
        result: dict[str, Any] = {
            "status": "VALIDATED_IN_MEMORY", "dataset_id": generator.DATASET_ID,
            "case_count": len(cases), "unique_wavs": len(blobs),
            "bytes": sum(map(len, blobs.values())),
            "source_inventory": counts.get("source_inventory"),
        }
    else:
        result = generator.build(verify_store=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
