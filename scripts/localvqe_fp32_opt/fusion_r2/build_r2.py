"""Build an isolated R2 experiment by replaying pinned R1 compiler receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
R1_SOURCE = HERE.parent / "fusion_r1"
ROOT = HERE.parents[2]
BUILD = ROOT / "build"
OUT = BUILD / "localvqe-fusion-r2-20261118"
R1 = BUILD / "localvqe-fusion-20261118"
R1_BUILDS = {
    "host": ("host-20261118T044223465060Z", "467a2bb7151e0fcb2d26405cb61e2acb2c43ac2a505fd137b7aad497cb5b129f"),
    "android": ("android-20261118T044337136700Z", "5472d25d5121ea71fb411eb7fd0aaf5166b964ea36979721abb9714b8c9a531a"),
}
DAF = BUILD / "deps/localvqe-f53063c9eb2a85f96479867d1dd911dc3bf6319b/ggml/daf_frontend.cpp"
DAF_SHA = "465bab6446341be83adeba58e7ba53b16bf2400e1fe83b2832071403da426bea"
STATE_SHA = "68710574470d7fb5fdb885f9abdbf6f26a218cf2aaa021f1782d4866e318b653"
BASE_MODES = ("norm", "norm_align")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entry(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, content: str | dict) -> None:
    with path.open("x") as stream:
        stream.write(content if isinstance(content, str) else json.dumps(content, indent=2) + "\n")


def need(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def guard(path: Path = ROOT) -> None:
    available = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    need(shutil.disk_usage(path).free >= 100 * 1024**3 and available >= 5 * 1024**3,
         "Resource stop line: retain all existing files")


def new_dir(label: str) -> Path:
    guard()
    result = OUT / f"{label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    result.mkdir(parents=True)
    return result


def once(source: str, old: str, new: str) -> str:
    need(source.count(old) == 1, f"Missing or ambiguous anchor: {old!r}")
    return source.replace(old, new, 1)


def frozen_r1(platform: str) -> tuple[Path, dict]:
    name, digest = R1_BUILDS[platform]
    path = R1 / name / "MANIFEST.json"
    need(sha(path) == digest, f"R1 manifest drift: {path}")
    m = read(path)
    return path.parent, m


def graph(source: str, header: Path | None) -> str:
    source = once(source, 'return mode && *mode ? mode : "baseline";',
                  'return mode && *mode ? mode : "norm";')
    if header:
        source = f'#include "{header.name}"\n' + source
        source = once(source, '    if (itu_fusion_on("norm")) {',
                      '    if (itu_fusion_on("norm") || itu_fusion_on("norm_align")) {')
        anchor = '    // Weighted sum (batched): aligned[f, 0, c]'
        source = once(source, anchor, '''    if (itu_fusion_on("norm_align")) {
        if (auto* aligned = itu_fusion_r2::align_readout_or_null(ctx, ref_full, attn))
            return aligned;
    }

''' + anchor)
    start = source.index("    int n_norm = 0, n_tile")
    end = source.index("\n    // Allocate\n    sg.win_buf", start)
    source = source[:start] + r'''    int norm = 0, align = 0, sum_rows = 0, col = 0, mat = 0, silu = 0;
    for (int i = 0; i < ggml_graph_n_nodes(sg.graph); ++i) {
        const auto* node = ggml_graph_node(sg.graph, i);
        if (node->op == GGML_OP_MAP_CUSTOM3 && std::strcmp(node->name, "itu_fusion_norm") == 0) ++norm;
        if (node->op == GGML_OP_CUSTOM && std::strcmp(node->name, "itu_fusion_align_readout") == 0) ++align;
        if (node->op == GGML_OP_SUM_ROWS) ++sum_rows;
        if (node->op == GGML_OP_IM2COL) ++col;
        if (node->op == GGML_OP_MUL_MAT) ++mat;
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) ++silu;
    }
    const bool use_align = itu_fusion_on("norm_align");
    const bool valid = (itu_fusion_on("norm") || (ITU_ALLOW_ALIGN && use_align)) &&
        norm == 29 && align == int(use_align) && sum_rows == 4 - int(use_align) &&
        col == 32 && mat == 37 && silu == 23;
    std::fprintf(stderr, "itu_fusion_r2 mode=%s norm=%d align=%d sum_rows=%d im2col=%d mul_mat=%d silu=%d accepted=%d\n",
        itu_fusion_mode(), norm, align, sum_rows, col, mat, silu, int(valid));
    if (!valid) return false;
''' + source[end:]
    return source.replace("ITU_ALLOW_ALIGN", "true" if header else "false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=tuple(R1_BUILDS))
    parser.add_argument("--gcc-diagnostic", action="store_true")
    parser.add_argument("--align-header", default="align_readout.hpp")
    args = parser.parse_args()
    android = args.target == "android"
    frozen, r1 = frozen_r1(args.target)
    header = None if args.gcc_diagnostic else HERE / args.align_header
    if header:
        need(header.parent == HERE and header.is_file(), f"Invalid Align header: {header}")
    for record in [r1["runtime"], r1["diagnostic"], r1["binary"],
                   *r1["generated_graphs"], *r1["headers"].values()]:
        need(sha(Path(record["path"])) == record["sha256"], f"Frozen input drift: {record['path']}")
    records = dict(r1["commands"])
    out = new_dir(f"{args.target}-{'gcc' if args.gcc_diagnostic else 'align'}")
    print("BUILD", out, flush=True)
    commands = []

    def run(role: str, label: str, mapping: dict[str, Path | str], prefix: tuple[str, ...] = ()) -> None:
        guard(out)
        old = records[label]
        cmd = old["argv"]
        need(all(cmd.count(k) == 1 for k in mapping), f"Ambiguous compiler/linker receipt: {label}")
        argv = [str(mapping.get(token, token)) for token in cmd]
        argv[1:1] = prefix
        with (out / (role + ".log")).open("xb") as log:
            p = subprocess.run(argv, cwd=old["cwd"], env={**os.environ, "TMPDIR": str(out)},
                               stdout=log, stderr=subprocess.STDOUT, timeout=180)
        item = {"name": role, "argv": argv, "cwd": old["cwd"], "exitcode": p.returncode}
        save(out / (role + ".json"), item)
        commands.append(item)
        need(p.returncode == 0, f"{role} failed; see {out / (role + '.log')}")

    def location(label: str, flag: str) -> str:
        cmd = records[label]["argv"]
        need(cmd.count(flag) == 1, f"Invalid command receipt {label}: {flag}")
        return cmd[cmd.index(flag) + 1]

    extra_sources: list[Path] = []
    if args.gcc_diagnostic:
        import gcc_diag_overlay
        need(sha(DAF) == DAF_SHA, "Frozen DAF drift")
        gcc_source = out / "daf_frontend.cpp"
        save(gcc_source, gcc_diag_overlay.instrument(DAF.read_text()))
        gcc_obj = out / "daf_frontend.o"
        # DAF was compiled independently with contraction off, unlike the main graph.
        toolchain = (BUILD / "android-native16-20261117/root/localvqe-build-v4" if android
                     else BUILD / "localvqe-host-20261010")
        matches = [r for r in read(toolchain / "compile_commands.json")
                   if Path(r["file"]).name == "daf_frontend.cpp"]
        need(len(matches) == 1, "Missing frozen DAF recipe")
        cmd = shlex.split(matches[0]["command"])
        need("-ffp-contract=off" in cmd, "Frozen DAF needs contraction off")
        records["daf"] = {"argv": cmd, "cwd": matches[0]["directory"]}
        run("compile-daf", "daf", {location("daf", "-c"): gcc_source, location("daf", "-o"): gcc_obj},
            ("-I" + str(HERE), "-pipe"))
        extra_sources.extend([DAF, gcc_source, HERE / "gcc_diag.hpp", Path(gcc_diag_overlay.__file__)])
    else:
        gcc_obj = None

    def make_library(kind: str) -> Path:
        folder = out / kind
        folder.mkdir()
        old_folder = frozen / kind
        source = folder / "localvqe_graph.cpp"
        save(source, graph((old_folder / source.name).read_text(), header))
        obj = folder / "layout.o"
        run(kind + "-graph", kind + "-graph",
            {str(old_folder / source.name): source, str(old_folder / obj.name): obj},
            ("-I" + str(HERE), "-pipe"))
        archive = folder / "graph.a"
        mapping = {str(old_folder / archive.name): archive, str(old_folder / obj.name): obj}
        if gcc_obj is not None:
            map_key, = [p for p in records[kind + "-archive"]["argv"] if p.endswith("/daf_frontend.cpp.o")]
            mapping[map_key] = gcc_obj
        run(kind + "-archive", kind + "-archive", mapping)
        run(kind + "-index", kind + "-index", {str(old_folder / archive.name): archive})
        lib = folder / "liblocalvqe.so.0.1.0"
        run(kind + "-link-library", kind + "-link-library",
            {str(old_folder / archive.name): archive, str(old_folder / lib.name): lib})
        return lib

    runtime = make_library("runtime")
    diagnostic = runner = None
    if header:
        diagnostic = make_library("diagnostic")
        source = R1_SOURCE / "state_acceptance.cpp"
        need(sha(source) == STATE_SHA, "Frozen state comparer drift")
        generated = out / source.name
        save(generated, once(source.read_text(), 'Pipeline base(base_library, argv[3], "baseline");',
                             'Pipeline base(base_library, argv[3], "norm");'))
        obj = out / "state-probe.o"
        runner = out / "state-probe"
        c = "compile-state-acceptance"
        l = "link-state-acceptance"
        run(c, c, {str(source): generated, location(c, "-o"): obj}, ("-I" + str(R1_SOURCE), "-pipe"))
        run(l, l, {location(c, "-o"): obj, location(l, "-o"): runner})
        extra_sources.extend([source, generated, header])

    overlay = out / "overlay"
    overlay.mkdir()
    original_overlay = frozen / "overlay"
    source = original_overlay / "localvqe_aec.hpp"
    save(overlay / source.name, once(source.read_text(), r1["runtime"]["sha256"], sha(runtime)))
    engine_source = overlay / "audio_engine.cpp"
    save(engine_source, (original_overlay / engine_source.name).read_text())
    engine = out / "engine.o"
    cli = out / "clock-cli.o"
    run("compile-engine", "compile-engine",
        {"-I" + str(original_overlay): "-I" + str(overlay),
         location("compile-engine", "-c"): engine_source, location("compile-engine", "-o"): engine}, ("-pipe",))
    run("compile-clock-cli", "compile-clock-cli",
        {"-I" + str(original_overlay): "-I" + str(overlay), location("compile-clock-cli", "-o"): cli}, ("-pipe",))
    unbound = location("compile-engine", "-o")
    if android:
        archive = out / "vcp-fusion.a"
        old_archive = str(frozen / archive.name)
        run("archive-vcp", "archive-vcp", {old_archive: archive, unbound: engine})
        run("index-vcp", "index-vcp", {old_archive: archive})
        unbound, engine = old_archive, archive
    binary = out / "itu_audio_clock"
    run("link-vcp", "link-vcp", {unbound: engine, location("compile-clock-cli", "-o"): cli,
                                 location("link-vcp", "-o"): binary})
    manifest = {"schema": "itu-lvq-fusion-r2-build", "platform": args.target, "gcc_profile": bool(gcc_obj),
                "modes": ["norm"] if gcc_obj else list(BASE_MODES), "binary": entry(binary),
                "runtime": entry(runtime), "diagnostic": entry(diagnostic) if diagnostic else None,
                "state_runner": entry(runner) if runner else None,
                "frozen_recipe": entry(frozen / "MANIFEST.json"),
                "sources": [entry(p) for p in [Path(__file__), *extra_sources, *sorted(out.glob("*/localvqe_graph.cpp"))]],
                "model_paths": r1["model_paths"], "model_sha256": r1["model_sha256"], "commands": commands}
    guard(out)
    save(out / "MANIFEST.json", manifest)
    print("MANIFEST", out / "MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
