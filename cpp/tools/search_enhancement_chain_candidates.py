#!/usr/bin/env python3
"""Screen single-stage replacements in the frozen 72-case enhancement chain."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import run_enhancement_chain_matrix as matrix


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CLI_SHA256 = "747c0bc0b48d6fe7a9edadc9453dd02489465888c6d1fbbead92970556836e6a"
CHANGES = [
    ("base_drc_off", "VCP + no DRC", {"drc": "off"}),
    ("base_drc_gentle", "VCP + gentle DRC", {"drc": "gentle"}),
    ("base_drc_aggressive", "VCP + aggressive DRC", {"drc": "aggressive"}),
    ("base_drc_fast", "VCP + fast DRC", {"drc": "fast"}),
    ("base_limiter_off", "VCP + no limiter", {"limiter": "off"}),
    ("base_eq_three", "VCP + three-band EQ", {"eq": "three_band"}),
    ("base_eq_low", "VCP + low-shelf EQ", {"eq": "low_shelf"}),
    ("base_eq_parametric", "VCP + parametric EQ", {"eq": "parametric"}),
    ("base_agc_stage4", "VCP + Stage4 AGC", {"agc": "stage4"}),
    ("base_agc_release", "VCP + Stage4 noise release", {"agc": "stage4_release"}),
    ("base_agc2", "VCP + AGC2", {"agc": "agc2"}),
    ("base_fixed6", "VCP + fixed +6 dB", {"agc": "fixed6"}),
    ("base_ns_speex", "VCP + Speex NS", {"ns": "speex"}),
    ("base_ns_rnnoise", "VCP + RNNoise", {"ns": "rnnoise"}),
    ("base_ns_omlsa", "VCP + OM-LSA", {"ns": "omlsa"}),
    ("base_ns_super", "VCP + Super-Gaussian", {"ns": "super_gaussian"}),
    ("base_bf_mvdr", "VCP + MVDR BF", {"bf": "mvdr"}),
    ("base_aec3", "VCP + AEC3", {"aec": "aec3"}),
]
EQ_AFTER_SUPER_CHANGES = [
    ("super_eq_off", "Super-Gaussian + EQ bypass", {"ns": "super_gaussian"}),
    (
        "super_eq_parametric",
        "Super-Gaussian + parametric EQ",
        {"ns": "super_gaussian", "eq": "parametric"},
    ),
    (
        "super_eq_three",
        "Super-Gaussian + three-band EQ",
        {"ns": "super_gaussian", "eq": "three_band"},
    ),
    (
        "super_eq_low",
        "Super-Gaussian + low-shelf EQ",
        {"ns": "super_gaussian", "eq": "low_shelf"},
    ),
]
SEARCH_MODES = {
    "single-stage": CHANGES,
    "eq-after-super": EQ_AFTER_SUPER_CHANGES,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_config(mode: str):
    config = json.loads(matrix.CONFIG.read_text())
    base = next(preset for preset in config["presets"] if preset["id"] == "vcp_default")
    control = next(preset for preset in config["presets"] if preset["id"] == "array_control")
    selected = dict(base["selection"])
    candidates = [control, base]
    for candidate_id, label, changes in SEARCH_MODES[mode]:
        selection = dict(selected)
        selection.update(changes)
        candidates.append(
            {
                "id": candidate_id,
                "label": label,
                "description": label,
                "profile": "vcp",
                "selection": selection,
            }
        )
    config["presets"] = candidates
    return config


def ranking_key(item):
    far = item["mean_far_delta_db"]
    return (
        -item["pass"],
        -item["mean_sisdr_delta_db"],
        -item["mean_stoi_delta"],
        -item["mean_noise_delta_db"],
        -(far if far is not None else -1e9),
        item["clipped_cases"],
        item["limited_samples"],
        item["mean_rtf"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--cli",
        type=Path,
        default=ROOT / "build/host/cpp/itu_audio_cli",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=tuple(SEARCH_MODES),
        default="single-stage",
        help="candidate family to replay",
    )
    parser.add_argument("--expected-cli-sha256", default=EXPECTED_CLI_SHA256)
    args = parser.parse_args()

    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing search root: {root}")
    if not 1 <= args.jobs <= 8:
        raise ValueError("jobs must be between 1 and 8")

    cli = args.cli.resolve()
    cli_sha256 = digest(cli)
    if cli_sha256 != args.expected_cli_sha256:
        raise ValueError(
            f"CLI hash changed: expected {args.expected_cli_sha256}, got {cli_sha256}"
        )

    manifest = json.loads(matrix.MANIFEST.read_text())
    scenes = [scene for scene in manifest["scenes"] if scene["mics"] > 1]
    if len(scenes) != 72 or len({scene["id"] for scene in scenes}) != 72:
        raise ValueError("expected 72 unique paired 2/4-microphone scenes")
    for scene in scenes:
        source = matrix.DATASET / "cases" / scene["id"]
        for name, record in scene["files"].items():
            if digest(source / name) != record["sha256"]:
                raise ValueError(f"frozen input hash changed: {scene['id']}/{name}")

    config = search_config(args.mode)
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    for source in [
        cli,
        matrix.CONFIG,
        Path(matrix.__file__).resolve(),
        Path(__file__).resolve(),
        matrix.MANIFEST,
    ]:
        shutil.copy2(source, artifacts / source.name)

    started = time.monotonic()
    rows, controls = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [
            pool.submit(matrix.process_scene, (root, scene, config, cli))
            for scene in scenes
        ]
        for count, task in enumerate(as_completed(tasks), 1):
            records, control = task.result()
            rows.extend(records)
            controls.append(control)
            print(
                f"scenes {count}/{len(scenes)} · "
                f"CLI runs {len(rows)}/{len(scenes) * len(config['presets'])}",
                flush=True,
            )

    scene_order = {scene["id"]: index for index, scene in enumerate(scenes)}
    preset_order = {
        preset["id"]: index for index, preset in enumerate(config["presets"])
    }
    rows.sort(key=lambda row: (scene_order[row["scene_id"]], preset_order[row["variant"]]))
    controls.sort(key=lambda row: scene_order[row["scene_id"]])

    by_scene = {scene["id"]: scene for scene in scenes}
    variants = []
    for preset in config["presets"]:
        selected = [row for row in rows if row["variant"] == preset["id"]]
        if {row["scene_id"] for row in selected} != set(by_scene):
            raise ValueError(f"incomplete candidate: {preset['id']}")
        item = matrix.summary(selected, preset, by_scene)
        item["flags"] = matrix.flags_for(config, preset)
        variants.append(item)
    variants.sort(key=ranking_key)

    results_path = root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_scope": (
            "one-stage replacement of the current vcp_default"
            if args.mode == "single-stage"
            else "EQ variants applied after the Super-Gaussian NS replacement"
        ),
        "global_optimum_claim": False,
        "cases": len(scenes),
        "candidate_count": len(variants),
        "cli_runs": len(rows),
        "gates": matrix.GATES,
        "ranking_key": [
            "pass descending",
            "mean_sisdr_delta_db descending",
            "mean_stoi_delta descending",
            "mean_noise_delta_db descending",
            "mean_far_delta_db descending",
            "clipped_cases ascending",
            "limited_samples ascending",
            "mean_rtf ascending",
        ],
        "negative_control": {
            "kind": "muted output, scored with the identical gates",
            "cases": len(controls),
            "rejected": sum(control["rejected"] for control in controls),
        },
        "cli_sha256": cli_sha256,
        "config_sha256": digest(matrix.CONFIG),
        "manifest_sha256": digest(matrix.MANIFEST),
        "results_sha256": digest(results_path),
        "host": {
            "architecture": platform.machine(),
            "workers": args.jobs,
            "executor": "thread pool around subprocess CLI runs",
        },
        "wall_seconds": time.monotonic() - started,
        "variants": variants,
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (root / "negative-controls.json").write_text(
        json.dumps(controls, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(root / "report.json"),
                "cases": len(scenes),
                "cli_runs": len(rows),
                "negative_controls_rejected": report["negative_control"]["rejected"],
                "best": variants[0]["id"],
                "best_pass": variants[0]["pass"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
