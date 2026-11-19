"""Build one opt-in LocalVQE/VCP experiment using frozen D compile receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "build"
OUT = BUILD / "localvqe-fusion-20261118"
ANDROID_D = BUILD / "localvqe-opt-20261117/build-20261117T173046192276"
ANDROID_DIAG = BUILD / "localvqe-opt-20261117/build-20261117T172431731298"
ANDROID_CLOCK = BUILD / "android-audio-clock-20261117/build-20261118T095731589893"
HOST_CLOCK = BUILD / "android-audio-clock-20261117/build-20261118T090351405495"
FROZEN_ANDROID = BUILD / "android-native16-20261117/root"
GGML_PRIVATE = BUILD / "deps/localvqe-f53063c9eb2a85f96479867d1dd911dc3bf6319b/ggml/vendor/ggml/src"
BASE_GRAPH = HERE.parent / "layout/localvqe_graph.cpp"
PINNED_GRAPH = HERE.parent / "validation-root/layout-lifetime-graph.cpp"
PROBE_API = HERE.parent / "validation-root/probe_api.cpp"

PLAIN_HASH = "d65b7556e8cdf993c7d31e39be0721c8cee35d91cffbc7d6383bffcb7cd222df"
PINNED_HASH = "fb0caf8345f25a68714d07ac11f51b564b9dbfd5a800feab41613ba80d30a684"
PROBE_API_HASH = "2976b7d01a2dbc77892f74edc80b8c1e126b1dbd05359daf650ae5d0abfe3ff4"
ANDROID_LIB_HASH = "ff4d9d2afb68e8a46c966d45d5eeb340fb75e9a525c154cb080fda03507c6ee3"
HOST_LIB_HASH = "c5a80276667c9ccd184dd78b5f59cd2b8211d0b96ee1a320d9f924de6ac96b44"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def guard(path: Path = ROOT) -> None:
    mem = next(int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    if shutil.disk_usage(path).free < 100 * 1024**3 or mem < 5 * 1024**3:
        raise RuntimeError("Resource stop line; preserve existing files")


def save(path: Path, text: str) -> None:
    with path.open("x") as handle:
        handle.write(text)


def once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError(f"Expected one frozen anchor: {before!r}")
    return text.replace(before, after, 1)


def graph_source(path: Path, headers: dict[str, Path]) -> str:
    hooks = ''.join(f'#include "{p.name}"\n' for p in headers.values())
    hooks += r'''
#include <cstdlib>
#include <cstring>

namespace {
const char * itu_fusion_mode() {
    const char * mode = std::getenv("ITU_LVQ_FUSION");
    return mode && *mode ? mode : "baseline";
}
bool itu_fusion_on(const char * mode) {
    return std::strcmp(itu_fusion_mode(), mode) == 0;
}
}
'''
    text = hooks + path.read_text()
    if "norm" in headers:
        old = "    // Permute (F, T, C) -> (F, C, T) so all (F, C) values for a given T"
        text = once(text, old, '''    if (itu_fusion_on("norm")) {
        if (auto* fused = itu_fusion_r1::norm_affine_or_null(ctx, x, gamma, beta, eps))
            return fused;
    }
''' + old)
    start = '    int sF\n) {\n    int64_t F = x->ne[0], C_in = x->ne[2];'
    text = once(text, start, '''    int sF,
    bool* silu_fused = nullptr,
    bool tile_dec1 = false
) {
    int64_t F = x->ne[0], C_in = x->ne[2];''')
    before = '''    struct ggml_tensor* conv = lvq_conv_2d(ctx, weight, x4d,
                                             sF, 1, 0, 0, 1, 1);
    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    conv = ggml_add(ctx, conv, b);'''
    after = '    struct ggml_tensor* conv = nullptr;\n'
    if "tiled" in headers:
        after += '''    if (tile_dec1 && sF == 1 && itu_fusion_on("tiled"))
        conv = itu_fusion_r1::tiled_dec1_or_null(ctx, weight, x4d);
'''
    else:
        after += '    (void)tile_dec1;\n'
    after += '''    if (!conv) conv = lvq_conv_2d(ctx, weight, x4d, sF, 1, 0, 0, 1, 1);
    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
'''
    if "epilogue" in headers:
        after += '''    if (silu_fused && itu_fusion_on("epilogue")) {
        if (auto* fused = itu_fusion_r1::bias_silu_or_null(ctx, conv, b)) {
            *silu_fused = true;
            return fused;
        }
    }
'''
    else:
        after += '    (void)silu_fused;\n'
    text = once(text, before, after + '    conv = ggml_add(ctx, conv, b);')
    text = once(text, '''    struct ggml_tensor* y = build_causal_conv_s(ctx, sg, xn, conv_w, conv_b, 2);
    y = build_act(ctx, y, act);''', '''    bool fused_main = false;
    struct ggml_tensor* y = build_causal_conv_s(ctx, sg, xn, conv_w, conv_b, 2,
        act == lvq_activation::SILU ? &fused_main : nullptr);
    if (!fused_main) y = build_act(ctx, y, act);''')
    for shape in ("res->ne[0], res->ne[1], res->ne[2]", "F, 1, C"):
        text = once(text, f'''    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, yn, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, {shape});
    res = ggml_add(ctx, build_act(ctx, res, act), y);''', f'''    bool fused_res = false;
    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, yn, res_w, res_b, 1,
        act == lvq_activation::SILU ? &fused_res : nullptr);
    res = ggml_reshape_3d(ctx, res, {shape});
    res = ggml_add(ctx, fused_res ? res : build_act(ctx, res, act), y);''')
    text = once(text, '''    struct ggml_tensor* deconv = build_causal_conv_s(ctx, sg, res_n, deconv_w, deconv_b, 1);''',
                '''    struct ggml_tensor* deconv = build_causal_conv_s(ctx, sg, res_n, deconv_w, deconv_b, 1,
        nullptr, is_last);''')
    text = once(text, "    // Allocate\n    sg.win_buf", r'''    int n_norm = 0, n_tile = 0, n_epi = 0, n_col = 0, n_mm = 0, n_silu = 0;
    for (int i = 0; i < ggml_graph_n_nodes(sg.graph); ++i) {
        const auto* op = ggml_graph_node(sg.graph, i);
        const char* name = ggml_get_name(op);
        if (std::strcmp(name, "itu_fusion_norm") == 0 && op->op == GGML_OP_MAP_CUSTOM3) ++n_norm;
        if (std::strcmp(name, "itu_fusion_tiled_dec1") == 0 && op->op == GGML_OP_CUSTOM) ++n_tile;
        if (std::strcmp(name, "itu_fusion_bias_silu") == 0 && op->op == GGML_OP_MAP_CUSTOM2) ++n_epi;
        if (op->op == GGML_OP_IM2COL) ++n_col;
        if (op->op == GGML_OP_MUL_MAT) ++n_mm;
        if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU) ++n_silu;
    }
    const char* mode = itu_fusion_mode();
    const bool known = itu_fusion_on("baseline") || itu_fusion_on("norm") ||
        itu_fusion_on("tiled") || itu_fusion_on("epilogue");
    const bool good = known &&
        n_norm == (itu_fusion_on("norm") ? 29 : 0) &&
        n_tile == (itu_fusion_on("tiled") ? 1 : 0) &&
        n_epi == (itu_fusion_on("epilogue") ? 19 : 0);
    std::fprintf(stderr, "itu_fusion graph mode=%s norm=%d tiled=%d epilogue=%d im2col=%d mul_mat=%d silu=%d accepted=%d\n",
        mode, n_norm, n_tile, n_epi, n_col, n_mm, n_silu, int(good));
    if (!good) return false;

    // Allocate
    sg.win_buf''')
    if "tiled" in headers and headers["tiled"].name != "tiled_conv.hpp":
        text = once(text, '    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }',
                    '    if (sg.ctx) {\n        itu_fusion_r1::release_tiled_dec1(sg.ctx);\n'
                    '        ggml_free(sg.ctx); sg.ctx = nullptr;\n    }')
    return once(text, '    ggml_backend_graph_compute(m.backend, sg.graph);',
                '    GGML_ASSERT(ggml_backend_graph_compute(m.backend, sg.graph) == GGML_STATUS_SUCCESS);')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "android"), required=True)
    parser.add_argument("--norm", default="norm_affine.hpp")
    parser.add_argument("--tiled", default="tiled_conv_v2.hpp")
    parser.add_argument("--epilogue", default="bias_silu.hpp")
    args = parser.parse_args()
    headers = {k: HERE / getattr(args, k) for k in ("norm", "tiled", "epilogue")
               if getattr(args, k) != "none"}
    for file in headers.values():
        if file.parent != HERE or not file.is_file():
            raise RuntimeError(f"Missing or out-of-scope header: {file}")
    for file, digest in ((BASE_GRAPH, PLAIN_HASH), (PINNED_GRAPH, PINNED_HASH),
                         (PROBE_API, PROBE_API_HASH)):
        if sha(file) != digest:
            raise RuntimeError(f"Frozen source changed: {file}")
    android = args.target == "android"
    guard()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / f"{args.target}-{stamp}"
    directory.mkdir(parents=True)
    print(f"BUILD {directory}", flush=True)
    parts = [ANDROID_D, ANDROID_DIAG, ANDROID_CLOCK] if android else [HOST_CLOCK]
    recipes = {}
    for part in parts:
        for recipe in json.loads((part / "MANIFEST.json").read_text())["commands"]:
            recipes.setdefault(recipe["label"], recipe)
    receipts = {}

    def run(name: str, argv: list[str], cwd: Path) -> None:
        guard(directory)
        # Stage outputs never reuse a frozen path. A failure keeps its log.
        env = dict(os.environ, TMPDIR=str(directory))
        with (directory / (name + ".log")).open("xb") as log:
            child = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        item = {"argv": argv, "cwd": str(cwd), "exitcode": child.returncode}
        receipts[name] = item
        if child.returncode:
            raise RuntimeError(f"{name} failed; {directory / (name + '.log')}")
        save(directory / (name + ".json"), json.dumps(item, indent=2) + "\n")
        print(f"OK {name}", flush=True)

    def use(label: str, name: str, change: dict[str, Path | str],
            source: Path | None = None, output: Path | None = None,
            extra: tuple[str, ...] = ()) -> None:
        rec = recipes[label]
        if any(rec["argv"].count(token) != 1 for token in change):
            raise RuntimeError(f"Missing or ambiguous recipe argument in {label}: {change}")
        argv = [str(change.get(a, a)) for a in rec["argv"]]
        if source is not None:
            argv[argv.index("-c") + 1] = str(source)
        if output is not None:
            argv[argv.index("-o") + 1] = str(output)
        argv[1:1] = extra
        run(name, argv, Path(rec["cwd"]))

    if android:
        frozen_library = ANDROID_D / "D/liblocalvqe.so.0.1.0"
        frozen_provider = ANDROID_D / "D/overlay/localvqe_aec.hpp"
        layout, link_library, compile_provider = (
            "compile-layout", "link-library-D", "compile-provider-D")
        frozen_graph_archive = ANDROID_D / "graph-D.a"
        frozen_vcp = ANDROID_D / "vcp-D.a"
    else:
        frozen_library = HOST_CLOCK / "host/liblocalvqe.so.0.1.0"
        frozen_provider = HOST_CLOCK / "host/overlay/localvqe_aec.hpp"
        layout, link_library, compile_provider = (
            "host-layout", "host-link-localvqe", "host-engine")
        frozen_graph_archive = HOST_CLOCK / "host-d-graph.a"
        frozen_vcp = None
    old_hash = ANDROID_LIB_HASH if android else HOST_LIB_HASH
    if sha(frozen_library) != old_hash:
        raise RuntimeError("Frozen runtime library changed")
    frozen_link = recipes[link_library]["argv"]
    api = next(v for v in frozen_link if v.endswith("/localvqe_api.cpp.o"))
    archive_recipe = "archive-graph-D" if android else "archive-host-d-graph"
    prev_layout = recipes[layout]["argv"][recipes[layout]["argv"].index("-o") + 1]
    includes = (f"-I{HERE}", f"-I{GGML_PRIVATE}", f"-I{GGML_PRIVATE / 'ggml-cpu'}")

    def library(diag: bool) -> Path:
        name = "diagnostic" if diag else "runtime"
        target = directory / name
        target.mkdir()
        source = target / "localvqe_graph.cpp"
        save(source, graph_source(PINNED_GRAPH if diag else BASE_GRAPH, headers))
        obj = target / "layout.o"
        use(layout, name + "-graph", {}, source, obj, includes)
        archive = target / "graph.a"
        use(archive_recipe, name + "-archive", {str(frozen_graph_archive): archive,
                                                prev_layout: obj})
        ranlib = "/usr/bin/llvm-ranlib-17" if android else "/usr/bin/ranlib"
        run(name + "-index", [ranlib, str(archive)], directory)
        mapping = {str(frozen_graph_archive): archive}
        if diag:
            api_object = target / "probe-api.o"
            if android:
                use("compile-probe-api", name + "-api", {}, PROBE_API, api_object)
            else:
                use(layout, name + "-api", {}, PROBE_API, api_object,
                    ("-DLOCALVQE_BUILD", "-I" + str(PROBE_API.parent)))
            mapping[api] = api_object
        lib = target / "liblocalvqe.so.0.1.0"
        use(link_library, name + "-link-library", mapping, output=lib)
        return lib

    def state_runner() -> Path:
        source = HERE / "state_acceptance.cpp"
        binary = directory / "state-probe"
        if android:
            obj = directory / "state-probe.o"
            use("compile-state-runner", "compile-state-acceptance", {}, source, obj)
            command = recipes["link-state-runner"]["argv"]
            old_obj = next(v for v in command if v.endswith("/probe-runner.o"))
            use("link-state-runner", "link-state-acceptance", {old_obj: obj}, output=binary)
        else:
            obj = directory / "state-probe.o"
            run("compile-state-acceptance", ["/usr/bin/clang++-17", "-std=c++17",
                "-O2", "-c", str(source), "-o", str(obj)], directory)
            run("link-state-acceptance", ["/usr/bin/clang++-17", str(obj),
                "-ldl", "-o", str(binary)], directory)
        return binary

    with ThreadPoolExecutor(max_workers=3) as pool:
        current = pool.submit(library, False)
        diagnostic = pool.submit(library, True)
        state = pool.submit(state_runner)
        lib = current.result()
        provider = directory / "overlay"
        provider.mkdir()
        save(provider / "localvqe_aec.hpp",
             once(frozen_provider.read_text(), old_hash, sha(lib)))
        provider_source = provider / "audio_engine.cpp"
        save(provider_source, (FROZEN_ANDROID / "overlay/audio_engine.cpp").read_text())
        engine = directory / "engine.o"
        use(compile_provider, "compile-engine", {"-I" + str(frozen_provider.parent):
                                                "-I" + str(provider)}, provider_source, engine)
        label = "compile-android-clock-cli" if android else "host-clock-cli"
        cli = directory / "clock-cli.o"
        use(label, "compile-clock-cli", {"-I" + str(frozen_provider.parent):
                                        "-I" + str(provider)}, output=cli)
        binary = directory / "itu_audio_clock"
        linker = "link-android-clock-cli" if android else "host-link-callback"
        argv = recipes[linker]["argv"]
        cli_before = next(v for v in argv if v.endswith("/clock_cli.o") or
                          v.endswith("/clock-cli.o"))
        substitutions = {cli_before: cli}
        if android:
            vcp = directory / "vcp-fusion.a"
            use("archive-vcp-D", "archive-vcp", {str(frozen_vcp): vcp,
                str(ANDROID_D / "D/audio_engine.cpp.o"): engine})
            run("index-vcp", ["/usr/bin/llvm-ranlib-17", str(vcp)], directory)
            substitutions[str(frozen_vcp)] = vcp
        else:
            substitutions[str(HOST_CLOCK / "host/engine.o")] = engine
        use(linker, "link-vcp", substitutions, output=binary)
        diaglib, runner = diagnostic.result(), state.result()

    model_manifest = json.loads((ANDROID_CLOCK / "MANIFEST.json").read_text())
    entry = lambda p: {"path": str(p), "sha256": sha(p)}
    info = {"schema": "itu-lvq-fusion-r1-build", "platform": args.target,
            "base_graph": entry(BASE_GRAPH), "diagnostic_base_graph": entry(PINNED_GRAPH),
            "processor_flags_preserved": True, "frozen_runtime": entry(frozen_library),
            "enabled_candidates": list(headers), "headers": {k: entry(v) for k,v in headers.items()},
            "generated_graphs": [entry(p) for p in sorted(directory.glob("*/localvqe_graph.cpp"))],
            "frozen_recipes": [entry(part / "MANIFEST.json") for part in parts],
            "runtime": entry(lib), "diagnostic": entry(diaglib), "state_runner": entry(runner),
            "binary": entry(binary), "model_paths": model_manifest["model_paths"],
            "model_sha256": model_manifest["model_sha256"], "commands": receipts}
    save(directory / "MANIFEST.json", json.dumps(info, indent=2) + "\n")
    print(f"MANIFEST {directory / 'MANIFEST.json'}", flush=True)


if __name__ == "__main__":
    main()
