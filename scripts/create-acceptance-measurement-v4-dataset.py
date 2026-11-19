#!/usr/bin/env python3
"""Add a versioned 336-item engineering dataset with explicit link/SER recipes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import shutil
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT, PROJECT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
sys.dont_write_bytecode = True


def configure():
    from dataset_revision_signal_design import add_observed_ser, add_ser_aliases, apply_transport, highway_noise

    path = PROJECT / "scripts/create-acceptance-measurement-v3-dataset.py"
    spec = importlib.util.spec_from_file_location("itu_measurement_v4_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load measurement v3 generator")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.NAME = "measurement-aware-v4-native-swb-link-ser"
    generator.BASE.NAME = generator.NAME
    original_preflight = generator.BASE.preflight

    def preflight(output):
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))) * 1024
        if shutil.disk_usage(PROJECT).free < 100 * 1024**3 or available < 5 * 1024**3:
            raise RuntimeError("resource safety floor reached; preserve all artifacts")
        original_preflight(output)

    generator.BASE.preflight = preflight
    original_bank = generator.BASE.SourceBank

    class Bank(original_bank):
        def noise(self, mode, scene, rate, seed):
            if scene == "sce3":
                return highway_noise(self, mode, rate, seed)
            return super().noise(mode, scene, rate, seed)

    generator.BASE.SourceBank = Bank
    original_render = generator.BASE.render_signals

    def render(near, far, noise, rate, **kwargs):
        mode = current_mode[0]
        processed, transport = apply_transport(far, rate, mode, kwargs["seed"])
        signals, metadata = original_render(near, processed, noise, rate, **kwargs)
        metadata = add_ser_aliases(metadata)
        if signals["microphone"].ndim == 2:
            metadata = add_observed_ser(signals, metadata, rate)
        else:
            metadata.update({
                "near_to_echo_overlap_ser_db_measured": None,
                "echo_to_near_overlap_esr_db_measured": None,
                "ser_observation_window_s": None,
                "ser_observation_window_definition": "not applicable to receive-only stimulus",
            })
        metadata["transport"] = transport
        metadata["dataset_revision"] = "measurement-20261117-r1"
        return signals, metadata

    # build_v3 asks for both source roles before rendering every case.
    current_mode = [None]
    original_speech = Bank.speech

    def speech(self, mode, role, rate):
        current_mode[0] = mode
        return original_speech(self, mode, role, rate)

    Bank.speech = speech
    generator.BASE.render_signals = render
    # Reuse the checked builder without hand-copying its implementation. Its
    # report contains an otherwise hard-coded historical V3 scope string.
    builder_source = inspect.getsource(generator.build_v3)
    old_scope = "measurement-aware v3 (8/16/24/48 kHz + measurable downlink idle noise); not certification"
    if builder_source.count(old_scope) != 1:
        raise RuntimeError("measurement builder scope changed; review before reuse")
    builder_source = builder_source.replace(
        old_scope,
        "measurement-aware v4: native SWB speech, explicit link proxies, independent sce3 noise, SER provenance; not certification",
    )
    exec(compile(builder_source, str(Path(__file__).resolve()), "exec"), generator.__dict__)
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-import", action="store_true")
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args(argv)
    generator = configure()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or PROJECT / "build/dataset-revision-20261117" / f"contract-v4-{stamp}.zip"
    result = generator.build_v3(output.resolve(), seed=args.seed, do_import=not args.no_import)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
