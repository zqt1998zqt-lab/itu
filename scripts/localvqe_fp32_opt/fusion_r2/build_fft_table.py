"""Replay the pinned Android GCC profile build with only its large FFT replaced."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from build_r2 import DAF, DAF_SHA, HERE, OUT, entry, guard, need, new_dir, once, read, save, sha

PROFILE = OUT / "android-gcc-20261118T075527984917Z/MANIFEST.json"
PROFILE_SHA = "a8077b3d20339db9ad114e57fa47125cae10662e363f955f1b81cadd5ddf0ba6"
ALIGN = OUT / "android-align-20261118T074310032967Z/MANIFEST.json"
ALIGN_SHA = "aeb2e59e473074321b797cc274132d4cc0f4dbb1092db37e3e7898d528f43d39"


def main() -> None:
    need(sha(PROFILE) == PROFILE_SHA and sha(ALIGN) == ALIGN_SHA and sha(DAF) == DAF_SHA,
         "Frozen R2 evidence changed")
    old, state = read(PROFILE), read(ALIGN)
    for obj in [old["binary"], old["runtime"], state["diagnostic"], state["state_runner"],
                *old["sources"]]:
        need(sha(Path(obj["path"])) == obj["sha256"], f"Frozen source drift: {obj['path']}")
    header = HERE / "gcc_fft_table.hpp"
    need(header.is_file(), "FFT plan header is not ready")
    out = new_dir("android-fft-table")
    print("BUILD", out, flush=True)
    previous = PROFILE.parent
    commands = []
    records = {item["name"]: item for item in old["commands"]}
    records.update({item["name"]: item for item in state["commands"] if item["name"].startswith("diagnostic-")})

    def replay(role: str, mapping: dict[str, str | Path], *, flags: tuple[str, ...] = ()) -> None:
        guard(out)
        source = records[role]
        need(all(source["argv"].count(k) == 1 for k in mapping), f"Bad replay mapping: {role}")
        argv = [str(mapping.get(k, k)) for k in source["argv"]]
        argv[1:1] = flags
        with (out / (role + ".log")).open("xb") as log:
            p = subprocess.run(argv, cwd=source["cwd"], env={**os.environ, "TMPDIR": str(out)},
                               stdout=log, stderr=subprocess.STDOUT, timeout=180)
        receipt = {"name": role, "argv": argv, "cwd": source["cwd"], "exitcode": p.returncode}
        guard(out)
        save(out / (role + ".json"), receipt)
        need(p.returncode == 0, f"FFT build failed: {out / (role + '.log')}")
        commands.append(receipt)

    generated = previous / "daf_frontend.cpp"
    text = once(generated.read_text(), '#include "gcc_diag.hpp"\n', r'''#include "gcc_diag.hpp"
#include "gcc_fft_table.hpp"
#include <atomic>

namespace {
std::atomic<unsigned> itu_fft_table_hits{0};
__attribute__((destructor)) void report_itu_fft_table_hits() {
    std::fprintf(stderr, "ITU_GCC_FFT_TABLE enabled=%d hits=%u\n",
                 int(itu_gcc_fft_table::enabled()), itu_fft_table_hits.load(std::memory_order_relaxed));
}
}
''')
    text = once(text, '    const size_t n = re.size();\n', '''    const size_t n = re.size();
    if (n == 32768 && itu_gcc_fft_table::try_fft(re, im, inv)) {
        itu_fft_table_hits.fetch_add(1, std::memory_order_relaxed);
        return;
    }
''')
    new_daf = out / generated.name
    save(new_daf, text)
    daf_obj = out / "daf_frontend.o"
    need("-ffp-contract=off" in records["compile-daf"]["argv"], "Changed DAF FP32 rules")
    replay("compile-daf", {str(generated): new_daf, str(previous / daf_obj.name): daf_obj},
           flags=("-DITU_GCC_FFT_TABLE_DEFAULT=1",))

    def make_so(kind: str, old_so: Path) -> Path:
        folder = out / kind
        folder.mkdir()
        source_archive = old_so.parent / "graph.a"
        new_archive = folder / source_archive.name
        new_so = folder / old_so.name
        archive_role = kind + "-archive"
        any_daf, = [arg for arg in records[archive_role]["argv"] if arg.endswith(("/daf_frontend.o", "/daf_frontend.cpp.o"))]
        replay(archive_role, {str(source_archive): new_archive, any_daf: daf_obj})
        replay(kind + "-index", {str(source_archive): new_archive})
        replace = {str(old_so): new_so, str(source_archive): new_archive}
        if kind == "diagnostic":
            replace["-Wl,-soname,liblocalvqe.so.0"] = "-Wl,-soname,liblocalvqe_r2_fft_diag.so"
        replay(kind + "-link-library", replace)
        return new_so

    runtime = make_so("runtime", Path(old["runtime"]["path"]))
    diagnostic = make_so("diagnostic", Path(state["diagnostic"]["path"]))
    overlay = out / "overlay"
    overlay.mkdir()
    original_overlay = previous / "overlay"
    source = original_overlay / "localvqe_aec.hpp"
    save(overlay / source.name, once(source.read_text(), old["runtime"]["sha256"], sha(runtime)))
    engine_cpp = overlay / "audio_engine.cpp"
    save(engine_cpp, (original_overlay / engine_cpp.name).read_text())
    old_include = "-I" + str(original_overlay)
    new_include = "-I" + str(overlay)
    engine = out / "engine.o"
    clock = out / "clock-cli.o"
    archive = out / "vcp-fusion.a"
    binary = out / "itu_audio_clock"
    substitute = lambda p: str(previous / p.name)
    replay("compile-engine", {old_include: new_include, substitute(engine): engine,
                              str(original_overlay / engine_cpp.name): engine_cpp})
    replay("compile-clock-cli", {old_include: new_include, substitute(clock): clock})
    replay("archive-vcp", {substitute(archive): archive, substitute(engine): engine})
    replay("index-vcp", {substitute(archive): archive})
    replay("link-vcp", {substitute(archive): archive, substitute(clock): clock, substitute(binary): binary})
    result = {"schema": old["schema"], "platform": "android", "gcc_profile": True,
              "modes": ["norm"], "gcc_fft": "table", "binary": entry(binary), "runtime": entry(runtime),
              "diagnostic": None, "state_runner": state["state_runner"],
              "table_diagnostic": entry(diagnostic), "reference_diagnostic": state["diagnostic"],
              "original_gcc_manifest": entry(PROFILE), "model_paths": old["model_paths"],
              "model_sha256": old["model_sha256"], "commands": commands,
              "sources": [entry(p) for p in (Path(__file__), header, generated, new_daf)]}
    save(out / "MANIFEST.json", result)
    print("MANIFEST", out / "MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
