"""Draw the SM8650 R2 callback distribution from frozen trial evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import fmean


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "build/localvqe-fusion-r2-20261118"
ALIGN = ARTIFACTS / "device-align-20261118T075454162560Z/RESULT.json"
GCC = ARTIFACTS / "final-evidence-20261118T081349763012Z/RESULT.json"
ALIGN_SHA = "43ee3b50def82a416a90db15cb0f2b1f400a096aaac8c2fb83eff707775462ac"
GCC_SHA = "eba2109b24d6bc22b86706d30766cc05c782c852b83f6c3436369e8512fb6e7a"
INPUT, DRAIN, STARTUP, BUDGET_NS = 3750, 3, 250, 16_000_000


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def trial_rows(trial):
    path = Path(trial["directory"]) / "collected"
    clock_csv = path / "out.audio-clock.frames.csv"
    timing_csv = path / "out.timing.csv"
    check(sha(clock_csv) == trial["clock_sha256"], f"Clock evidence drift: {clock_csv}")
    clock, timing = csv_rows(clock_csv), csv_rows(timing_csv)
    check(len(clock) == len(timing) == INPUT + DRAIN, f"Missing callback rows: {path}")
    rows = []
    for i, (c, t) in enumerate(zip(clock, timing, strict=True)):
        ns = int(c["completed_ns"]) - int(c["callback_start_ns"])
        aec, ns_ms, post = (float(t[key]) for key in
                            ("localvqe_process_ms", "ns_stream_roundtrip_ms", "localvqe_post_ms"))
        check(int(c["frame"]) == int(t["frame"]) == i and c["clocks_valid"] == "1"
              and ns > 0 and 0 <= ns_ms <= post and 0 <= aec <= ns / 1e6 - post,
              f"Bad timing row {i}: {path}")
        if i < INPUT:
            rows.append((ns, aec, ns_ms))
    check(sum(r[0] > BUDGET_NS for r in rows) == trial["windows"]["all"]["over16"]
          and sum(r[0] > BUDGET_NS for r in rows[STARTUP:]) == trial["windows"]["steady"]["over16"],
          f"Deadline counts disagree: {path}")
    return rows, {"trial": trial["trial"], "mode": trial["mode"], "clock_sha256": sha(clock_csv),
                  "stages_sha256": sha(timing_csv), "directory": str(path)}


def stages(rows):
    total = fmean(r[0] for r in rows) / 1e6
    aec, cofi = (fmean(r[k] for r in rows) for k in (1, 2))
    return {"frames": len(rows), "total_ms": total, "aec_ms": aec,
            "cofi_ms": cofi, "other_ms": total - aec - cofi}


def extract():
    check(sha(ALIGN) == ALIGN_SHA and sha(GCC) == GCC_SHA, "Frozen acceptance evidence changed")
    trials = json.loads(ALIGN.read_text())["trials"]
    check([t["mode"] for t in trials] == ["norm", "norm_align", "norm_align", "norm"],
          "Expected four ABBA trials")
    series = {"norm": [], "norm_align": []}
    steady = []
    sources = []
    for t in trials:
        rows, receipt = trial_rows(t)
        series[t["mode"]].extend(r[0] for r in rows)
        if t["mode"] == "norm":
            steady.extend(rows[STARTUP:])
        sources.append(receipt)
    norm_splits = [stages(steady), stages([r for r in steady if r[0] > BUDGET_NS])]
    orig_gcc = json.loads(GCC.read_text())["original_gcc"]
    gcc = []
    for e in orig_gcc["events"]:
        check((e["n_seen"] - 1) // 256 == e["frame"]
              and 0 < e["fft_wall_ms"] <= e["gcc_wall_ms"] <= e["callback_ms"],
              "Invalid nested GCC wall times")
        gcc.append({"frame": e["frame"], "total_ms": e["callback_ms"], "fft_ms": e["fft_wall_ms"],
                    "other_gcc_ms": e["gcc_wall_ms"] - e["fft_wall_ms"],
                    "other_callback_ms": e["callback_ms"] - e["gcc_wall_ms"]})
    check([e["frame"] for e in gcc] == [63, 95, 126, 157, 188, 220], "Unexpected GCC event frames")
    groups = {mode: {"frames": len(ns), "over16": sum(x > BUDGET_NS for x in ns),
                     "p99_ms": sorted(ns)[math.ceil(.99 * len(ns)) - 1] / 1e6}
              for mode, ns in series.items()}
    evidence = {"scope": "60 s frozen Android AAudio file replays, input callbacks only",
                "metric": "callback_started_to_DSP_completed", "budget_ms": 16, "groups": groups,
                "norm_steady_stage_mean_wall_ms": dict(zip(("all", "over16"), norm_splits)),
                "original_gcc_diagnostic_events": gcc, "raw_ABBA_trials": sources,
                "summaries": [{"path": str(p), "sha256": sha(p)} for p in (ALIGN, GCC)]}
    return series, norm_splits, gcc, evidence


def plot(out, series, splits, gcc, evidence):
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib-cache")
    (out / "matplotlib-cache").mkdir()
    import matplotlib as mpl
    import numpy as np
    from contextlib import nullcontext
    from unittest.mock import patch

    mpl.use("Agg")
    # Only this process owns the new cache, so no temporary font lock is needed.
    with patch.object(mpl.cbook, "_lock_path", lambda *args, **kwargs: nullcontext()):
        import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator, StrMethodFormatter

    ink, muted, grid = "#162839", "#42576B", "#DFE7ED"
    teal, amber, budget = "#106883", "#AF4A20", "#A43446"
    gold, slate, rest, violet = "#E7AF50", "#BCCAD6", "#D7E2EA", "#8067A8"
    mpl.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 12,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.unicode_minus": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "path"})
    fig = plt.figure(figsize=(11.8, 8.5), dpi=170, facecolor="white")
    layout = fig.add_gridspec(2, 2, left=.105, right=.965, top=.805, bottom=.145,
                              height_ratios=[1.32, 1], hspace=.66, wspace=.30)
    dist = fig.add_subplot(layout[0, :])
    breakdown = fig.add_subplot(layout[1, 0])
    startup = fig.add_subplot(layout[1, 1])
    fig.text(.105, .953, "LocalVQE 真机耗时分布", fontsize=25, weight="bold", va="top")
    fig.text(.105, .899, "SM8650 / Android 16  ·  AAudio 双讲回放  ·  每轮 60 秒  ·  不绑核",
             fontsize=13, color=muted, va="top")

    for ax in (dist, breakdown, startup):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(color=grid, length=4, pad=5, labelsize=11)
        ax.set_axisbelow(True)

    dist.set_title("A   整链逐帧耗时分布", loc="left", fontsize=15, weight="bold", pad=14)
    right = math.ceil(max(max(v) for v in series.values()) / 8e6) * 8
    dist.axvspan(16, right, color="#FCF3F1", zorder=-2)
    dist.axvline(16, color=budget, linestyle=(0, (3, 3)), lw=1.4, zorder=2)
    dist.text(16.55, 59, "16 ms 预算", fontsize=11, color=budget, va="center")
    for mode, name, color, line, marker in (
            ("norm", "norm", teal, "-", "o"), ("norm_align", "norm + Align", amber, "--", "s")):
        values, counts = np.unique(series[mode], return_counts=True)
        n = sum(counts)
        pct = 100 * (n - counts.cumsum()) / n
        late = evidence["groups"][mode]["over16"]
        dist.stairs(np.r_[100, pct[:-1]], np.r_[0, values / 1e6], baseline=None,
                    label=f"{name}    超时 {late:,} / {n:,}（{100 * late/n:.2f}%）",
                    color=color, linestyle=line, linewidth=2.0)
        dist.scatter(16, 100 * late/n, color=color, marker=marker, s=32, edgecolors="white", zorder=5)
    dist.set(xlim=(0, right), ylim=(.01, 125), yscale="log",
             xlabel="从回调开始到 DSP 完成（ms）", ylabel="超过该耗时的输入帧（%）")
    dist.xaxis.set_major_locator(MultipleLocator(8))
    dist.yaxis.set_major_locator(FixedLocator([.01, .1, 1, 10, 100]))
    dist.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    dist.yaxis.set_minor_locator(NullLocator())
    dist.grid(axis="y", color=grid, lw=.8)
    dist.legend(loc="upper right", frameon=False, fontsize=11, handlelength=3, borderaxespad=.8)

    def keys(ax, labels, colors, columns):
        ax.legend([Patch(facecolor=c) for c in colors], labels, loc="upper left",
                  bbox_to_anchor=(0, -.29), borderaxespad=0, frameon=False, ncol=columns,
                  handletextpad=.45, columnspacing=.85, handlelength=1.0, fontsize=10)

    breakdown.set_title("B   norm 稳态耗时构成", loc="left", fontsize=15, weight="bold", pad=16)
    for i, row in enumerate(splits):
        left = 0
        for j, (k, color) in enumerate(zip(("aec_ms", "cofi_ms", "other_ms"), (teal, gold, slate))):
            value = row[k]
            breakdown.barh(i, value, left=left, color=color, height=.53)
            if j < 2:
                breakdown.text(left + value / 2, i, f"{value:.2f}", ha="center", va="center",
                               color="white" if j == 0 else ink, fontsize=10)
            left += value
        breakdown.text(left + .35, i, f"{left:.2f}", va="center", weight="bold", fontsize=11)
        tag = ("全部" if i == 0 else "超时") + f"\n{row['frames']:,} 帧"
        breakdown.text(0, i - .36, tag.replace("\n", " · "), va="bottom", fontsize=10, color=muted)
    breakdown.set(xlim=(0, math.ceil((splits[-1]["total_ms"]+1.8)/4)*4),
                  ylim=(1.5, -.68), yticks=[], xlabel="平均墙钟耗时（ms/帧）")
    breakdown.xaxis.set_major_locator(MultipleLocator(4))
    breakdown.grid(axis="x", color=grid, lw=.8)
    keys(breakdown, ["AEC（含 DAF、卷积）", "CoFi", "其他"], [teal, gold, slate], 3)

    startup.set_title("C   启动六次 GCC 更新", loc="left", fontsize=15, weight="bold", pad=16)
    x = np.arange(len(gcc))
    bottom = np.zeros(len(gcc))
    for key, color in (("fft_ms", gold), ("other_gcc_ms", violet), ("other_callback_ms", rest)):
        y = np.array([e[key] for e in gcc])
        startup.bar(x, y, bottom=bottom, color=color, width=.63)
        bottom += y
    startup.axhline(16, color=budget, linestyle=(0, (3, 3)), lw=1.4, zorder=3)
    for at, top in zip(x, bottom, strict=True):
        startup.text(at, top + .65, f"{top:.1f}", ha="center", va="bottom", fontsize=9, color=ink)
    startup.set(ylim=(0, math.ceil((max(bottom)+2)/8)*8), xlabel="真实输入帧号（零基）", ylabel="耗时（ms）")
    startup.set_xticks(x, [str(e["frame"]) for e in gcc])
    startup.yaxis.set_major_locator(MultipleLocator(8))
    startup.grid(axis="y", color=grid, lw=.8)
    keys(startup, ["三次 FFT", "GCC 其他", "本帧其余"], [gold, violet, rest], 3)
    fig.text(.105, .048, "A、B 使用真机正反序对照；B 仅统计后 56 秒。C 为独立的原 FFT 诊断轮。图中排除尾部排空帧。",
             fontsize=10, color=muted)
    for fmt in ("png", "svg"):
        with (out / f"localvqe-latency-distribution.{fmt}").open("xb") as stream:
            fig.savefig(stream, format=fmt, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    mem = next(int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines()
               if x.startswith("MemAvailable:"))
    check(shutil.disk_usage(out.parent).free >= 100 * 1024**3 and mem >= 5 * 1024**3,
          "Resource stop line")
    series, splits, gcc, evidence = extract()
    out.mkdir()
    plot(out, series, splits, gcc, evidence)
    with (out / "METRICS.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(out)


if __name__ == "__main__":
    main()
