#!/usr/bin/env python3
"""Reproduce Figure 11: decode-phase accuracy across the accumulation design space.

Reads the two sweeps' result JSONs, writes figure11.csv (tidy data), then renders
figure11.png in the style of the paper's Figure 11:

  (a) FP8 CoFDA  — GSM8K / HumanEval accuracy vs F (fractional bits), chunk 32
  (b) NVFP4 GDFS — the same two benchmarks over the (G, F) grid

    python3 figure11.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# Liberation Sans is not in the image; matplotlib falls back to DejaVu Sans,
# which is fine — silence the per-font "not found" warnings.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
FP8_RESULTS = HERE / "fp8_cofda" / "results"
NVFP4_RESULTS = HERE / "nvfp4_gdfs" / "results"
CSV_OUT = HERE / "figure11.csv"
PNG_OUT = HERE / "figure11.png"

# --- Style (paper's; Liberation Sans if present, else DejaVu Sans) ---
plt.style.use("default")
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
# Match the paper's math font only if Liberation Sans is actually installed;
# otherwise leave matplotlib's default so "$F$" / "$G$" still render cleanly.
if "Liberation Sans" in {f.name for f in mpl.font_manager.fontManager.ttflist}:
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Liberation Sans"
    mpl.rcParams["mathtext.it"] = "Liberation Sans:italic"
    mpl.rcParams["mathtext.bf"] = "Liberation Sans:bold"
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["font.size"] = 7
mpl.rcParams["pdf.fonttype"] = 42

# --- Sweep layout / palette (paper) ---
F_COFDA = [7, 9, 10, 11, 12, 13, 17, 21, 25]   # (a) FP8 CoFDA fused-sum bits
G_GDFS = [3, 4, 5, 6]                           # (b) NVFP4 GDFS group bits
F_GDFS = [10, 13, 25, 35]                       # (b) NVFP4 GDFS fused-sum bits
BENCHES = ["GSM8K", "HumanEval"]
BENCH_COLOR = {"GSM8K": "#1A237E", "HumanEval": "#26A69A"}  # navy / teal

# our task name -> (metric key in the JSON, the paper's benchmark label)
TASK = {
    "gsm8k_cot": ("exact_match", "GSM8K"),
    "humaneval_instruct": ("pass@1", "HumanEval"),
}

Y_LIM, Y_TICKS = (0, 80), [0, 20, 40, 60, 80]


def load_data() -> pd.DataFrame:
    """Read both sweeps into one tidy frame: algorithm, benchmark, G, F, accuracy."""
    rows = []
    for p in sorted(FP8_RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        f = int(d["sweep_params"]["f_bits"])
        for task, (metric, bench) in TASK.items():
            rows.append({"algorithm": "cofda", "benchmark": bench, "G": np.nan,
                         "F": f, "accuracy": float(d["tasks"][task]["metrics"][metric])})
    for p in sorted(NVFP4_RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        g = int(d["sweep_params"]["g_bits"])
        f = int(d["sweep_params"]["f_bits"])
        for task, (metric, bench) in TASK.items():
            rows.append({"algorithm": "gdfs", "benchmark": bench, "G": g,
                         "F": f, "accuracy": float(d["tasks"][task]["metrics"][metric])})
    return pd.DataFrame(rows)


def cofda_series(df, bench):
    """(a): F -> accuracy(%) for one benchmark."""
    sub = df[(df.algorithm == "cofda") & (df.benchmark == bench)]
    out = []
    for f in F_COFDA:
        q = sub[sub.F == f]
        out.append(float(q["accuracy"].values[0]) * 100 if len(q) else np.nan)
    return out


def gdfs_map(df, bench):
    """(b): (G, F) -> accuracy(%) for one benchmark."""
    sub = df[(df.algorithm == "gdfs") & (df.benchmark == bench)]
    m = {}
    for g in G_GDFS:
        for f in F_GDFS:
            q = sub[(sub.G == g) & (sub.F == f)]
            m[(g, f)] = float(q["accuracy"].values[0]) * 100 if len(q) else np.nan
    return m


def draw_panel_a(ax, cofda):
    """FP8 CoFDA — paired bars over F (GSM8K | HumanEval)."""
    xa = np.arange(len(F_COFDA))
    bw = 0.38
    for bi, bench in enumerate(BENCHES):
        ax.bar(xa + (bi - 0.5) * bw, cofda[bench], bw, color=BENCH_COLOR[bench],
               edgecolor="white", lw=0.3, zorder=3, label=bench)

    # F=7 catastrophic collapse: highlight band + rotated value labels
    ax.axvspan(-0.5, 0.5, color="#F4F4F4", zorder=0)
    ax.text(0 - 0.5 * bw, 2.0, f"{cofda['GSM8K'][0]:.1f}", fontsize=5.0,
            color=BENCH_COLOR["GSM8K"], ha="center", va="bottom",
            fontweight="bold", rotation=90)
    ax.text(0 + 0.5 * bw, 2.0, f"{cofda['HumanEval'][0]:.1f}", fontsize=5.0,
            color=BENCH_COLOR["HumanEval"], ha="center", va="bottom",
            fontweight="bold", rotation=90)

    ax.set_xticks(xa)
    ax.set_xticklabels([str(f) for f in F_COFDA], fontsize=5)
    ax.set_xlabel("Fractional bits ($F$)", fontsize=6, labelpad=1.5)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.tick_params(axis="x", length=2)
    ax.tick_params(axis="y", labelsize=5, length=2)
    ax.set_ylabel("Accuracy (%)", fontsize=6, labelpad=1.5)
    ax.set_title("(a) FP8 CoFDA", fontsize=7, fontweight="bold", pad=2)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=3.8, framealpha=0.80, edgecolor="#CCC",
              handlelength=1.0, handletextpad=0.4, borderpad=0.25, labelspacing=0.25)


def draw_panel_b(ax, gdfs):
    """NVFP4 GDFS — nested bars: G group -> F slot -> (GSM8K | HumanEval)."""
    n_G, n_F = len(G_GDFS), len(F_GDFS)
    group_w = 0.84
    slot_w = group_w / n_F
    bw = slot_w / 2 * 0.86

    slot_centers, slot_flabels = [], []
    for gi, g in enumerate(G_GDFS):
        for fi, f in enumerate(F_GDFS):
            sc = gi + (fi - (n_F - 1) / 2) * slot_w
            slot_centers.append(sc)
            slot_flabels.append(str(f))
            for bi, bench in enumerate(BENCHES):
                ax.bar(sc + (bi - 0.5) * bw, gdfs[bench][(g, f)], bw,
                       color=BENCH_COLOR[bench], edgecolor="white", lw=0.25, zorder=3)

    for i in range(1, n_G):
        ax.axvline(i - 0.5, color="#CCCCCC", linestyle="--", lw=0.7, zorder=1)

    ax.set_xticks(slot_centers)
    ax.set_xticklabels(slot_flabels, fontsize=3.8)
    ax.tick_params(axis="x", length=1.5, pad=1)
    ax.set_xlim(-0.5, n_G - 0.5)
    ax.set_xlabel("Fractional bits ($F$)", fontsize=6, labelpad=1.5)

    for gi, g in enumerate(G_GDFS):
        ax.text(gi, 77.0, f"$G$={g}", ha="center", va="top", fontsize=6,
                fontweight="bold", color="#333333", zorder=5)

    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False)
    ax.set_title("(b) NVFP4 GDFS", fontsize=7, fontweight="bold", pad=2)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)


def _short(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def main():
    bar = "═" * 68
    print(f"\n{bar}")
    print("  Figure 11  —  decode-phase accuracy across the design space")
    print(bar)
    print("  Reproduces the paper's Figure 11 from the emulated sweep results:")
    print("    (a) FP8 CoFDA   — GSM8K / HumanEval vs F (chunk 32)")
    print("    (b) NVFP4 GDFS  — the same two benchmarks over the (G, F) grid")
    print(bar)

    df = load_data()
    n_fp8 = len(list(FP8_RESULTS.glob("*.json")))
    n_nvfp4 = len(list(NVFP4_RESULTS.glob("*.json")))
    print(f"  ✓ read    {n_fp8} + {n_nvfp4} runs     fp8_cofda/ + nvfp4_gdfs/ results")

    df.to_csv(CSV_OUT, index=False)
    print(f"  ✓ wrote   {CSV_OUT.name:<13} tidy data ({len(df)} rows)")

    COFDA = {b: cofda_series(df, b) for b in BENCHES}
    GDFS = {b: gdfs_map(df, b) for b in BENCHES}

    fig = plt.figure(figsize=(3.33, 1.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.05)
    draw_panel_a(fig.add_subplot(gs[0, 0]), COFDA)
    draw_panel_b(fig.add_subplot(gs[0, 1]), GDFS)
    for ax in fig.axes:
        ax.xaxis.set_label_coords(0.5, -0.165)
    fig.subplots_adjust(left=0.100, right=0.985, bottom=0.20, top=0.90)
    fig.savefig(PNG_OUT, dpi=300)
    print(f"  ✓ wrote   {PNG_OUT.name:<13} the plot")

    print(bar)
    loc = _short(HERE)
    print(f"  outputs in  {HERE.name if loc == '.' else loc}/")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
