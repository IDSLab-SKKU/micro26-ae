#!/usr/bin/env python3
"""Reproduce Figure 6(a): WikiText-2 perplexity across the FP8 CoFDA design space.

Reads the sweep results in results/, writes figure6a.csv (the tidy data), then
renders figure6a.png in the style of the paper's Figure 6(a) FP8 panel:
C-fused (cofda) vs C-decoupled (cofda_decoupled), at chunk sizes 16 and 32, over
the fractional-bit sweep, on a piecewise y-axis with iso-accuracy / transition /
catastrophic zones.

    python3 figure6a.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter

# Liberation Sans is not in the image; matplotlib falls back to DejaVu Sans,
# which is fine — silence the per-font "not found" warnings.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_OUT = HERE / "figure6a.csv"
PNG_OUT = HERE / "figure6a.png"

# --- Style (paper's; Liberation Sans if present, else DejaVu Sans) ---
plt.style.use("default")
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
# Match the paper's math font only if Liberation Sans is actually installed;
# otherwise leave matplotlib's default so "$F$" still renders cleanly.
if "Liberation Sans" in {f.name for f in mpl.font_manager.fontManager.ttflist}:
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Liberation Sans"
    mpl.rcParams["mathtext.it"] = "Liberation Sans:italic"
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 6.5
mpl.rcParams["ytick.labelsize"] = 6.5
mpl.rcParams["pdf.fonttype"] = 42

# --- Colors / markers (paper) ---
CLR_FUSED = "#E65100"    # deep orange — C-fused
CLR_DECOUP = "#1976D2"   # material blue — C-decoupled
CS_MARKERS = {16: "o", 32: "^"}
ZONE_COLORS = {"iso": "#C8E6C9", "trans": "#FFF9C4", "cata": "#FFCDD2"}

# --- FP8 sweep layout (paper) ---
F_AXIS = [3, 5, 7, 9, 11, 13, 17, 21, 25]
F_INDEX = {f: i for i, f in enumerate(F_AXIS)}
CS_VALUES = [16, 32]
CFUSED_MIN_F = 7            # C-fused below F=7 is catastrophic (off-chart)
PPL_YRANGE = (8.7, 14.0)   # (y_lo, y_top)
PPL_TICKS = [8.75, 8.8, 9.5, 12]

# config algorithm name -> the paper's variant name
VARIANT = {"cofda": "C-fused", "cofda_decoupled": "C-decoupled"}


def load_data() -> pd.DataFrame:
    """Read every results/<combo>.json into a tidy frame."""
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.stem.endswith("_samples"):
            continue
        d = json.loads(p.read_text())
        sp = d.get("sweep_params", {})
        ppl = d["tasks"]["wikitext"]["metrics"]["word_perplexity"]
        rows.append({
            "variant": VARIANT.get(sp["algorithm"], sp["algorithm"]),
            "C": int(sp["chunk_size"]),
            "F": int(sp["f_bits"]),
            "word_perplexity": float(ppl),
        })
    df = pd.DataFrame(rows).sort_values(["variant", "C", "F"]).reset_index(drop=True)
    df["x_idx"] = df["F"].map(F_INDEX)
    return df


def make_piecewise(y_lo, iso_upper, trans_upper, y_top):
    """Piecewise-linear data->display map: 60% iso, 25% transition, 15% catastrophic."""
    def forward(y):
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        m1 = y <= iso_upper
        out[m1] = 0.6 * (y[m1] - y_lo) / (iso_upper - y_lo)
        m2 = (y > iso_upper) & (y <= trans_upper)
        out[m2] = 0.6 + 0.25 * (y[m2] - iso_upper) / (trans_upper - iso_upper)
        m3 = y > trans_upper
        out[m3] = 0.85 + 0.15 * (y[m3] - trans_upper) / (y_top - trans_upper)
        return out

    def inverse(v):
        v = np.asarray(v, dtype=float)
        out = np.empty_like(v)
        m1 = v <= 0.6
        out[m1] = y_lo + (v[m1] / 0.6) * (iso_upper - y_lo)
        m2 = (v > 0.6) & (v <= 0.85)
        out[m2] = iso_upper + ((v[m2] - 0.6) / 0.25) * (trans_upper - iso_upper)
        m3 = v > 0.85
        out[m3] = trans_upper + ((v[m3] - 0.85) / 0.15) * (y_top - trans_upper)
        return out

    return forward, inverse


def draw_axis_break(ax, y_pos, dx=0.025, size=0.006):
    """Little double-slash break mark on the y-axis at a zone boundary."""
    y_d2a = ax.transData + ax.transAxes.inverted()
    y_frac = y_d2a.transform((0, y_pos))[1]
    for d in (-0.01, 0.01):
        ax.plot([-dx, dx], [y_frac + d - size, y_frac + d + size],
                transform=ax.transAxes, color="white", lw=1.5,
                clip_on=False, zorder=10, solid_capstyle="butt")
        ax.plot([-dx, dx], [y_frac + d - size, y_frac + d + size],
                transform=ax.transAxes, color="black", lw=0.4,
                clip_on=False, zorder=11, solid_capstyle="butt")


def draw_panel(ax, df):
    bl_ppl = df[(df.variant == "C-fused") & (df.C == 32)
                & (df.F == 25)]["word_perplexity"].iloc[0]
    iso_upper, trans_upper = bl_ppl * 1.01, bl_ppl * 1.50
    y_lo, y_top = PPL_YRANGE
    y_top_edge = y_top * 0.998

    fwd, inv = make_piecewise(y_lo, iso_upper, trans_upper, y_top)
    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(y_lo, y_top)

    for key, lo, hi in [("iso", y_lo, iso_upper),
                        ("trans", iso_upper, trans_upper),
                        ("cata", trans_upper, y_top)]:
        ax.axhspan(lo, hi, color=ZONE_COLORS[key], alpha=0.4, zorder=0)
    ax.axhline(bl_ppl, color="#888888", linestyle="--", linewidth=0.8, zorder=2)
    draw_axis_break(ax, trans_upper)

    for variant, clr in [("C-fused", CLR_FUSED), ("C-decoupled", CLR_DECOUP)]:
        for cs in CS_VALUES:
            sub = df[(df.variant == variant) & (df.C == cs)].sort_values("F")
            if variant == "C-fused":
                sub = sub[sub.F >= CFUSED_MIN_F]
            if sub.empty:
                continue
            off = sub["word_perplexity"] > y_top
            ppl_line = sub["word_perplexity"].clip(upper=y_top_edge)
            xs = sub["x_idx"].values
            if len(xs) >= 2:
                ax.plot(xs, ppl_line.values, color=clr, lw=1.2, alpha=0.7, zorder=3)
            inr = sub[~off]
            if not inr.empty:
                ax.scatter(inr["x_idx"], inr["word_perplexity"], c=clr,
                           marker=CS_MARKERS[cs], s=35, edgecolors="white",
                           linewidths=0.3, zorder=5)
            if variant != "C-fused":
                continue
            # C-fused off-chart (F=7): clipped marker at the top edge + wave break
            for _, orow in sub[off].iterrows():
                x_off = orow["x_idx"]
                ax.scatter(x_off, y_top_edge, marker=CS_MARKERS[cs], s=30, c=clr,
                           edgecolors="white", linewidths=0.3, zorder=5)
                i = F_AXIS.index(int(orow["F"]))
                if i + 1 >= len(F_AXIS):
                    continue
                nxt = sub[sub.F == F_AXIS[i + 1]]
                if nxt.empty:
                    continue
                x_next = int(nxt.iloc[0]["x_idx"])
                y_next = min(float(nxt.iloc[0]["word_perplexity"]), y_top_edge)
                bx = (x_off + x_next) / 2 + 0.08
                by = (y_top_edge + y_next) / 2
                dyd = (trans_upper - iso_upper) * 0.08
                for xo in (-0.08, 0.08):
                    ax.plot([bx - 0.10 + xo, bx + 0.10 + xo], [by - dyd, by + dyd],
                            color="white", lw=1.5, solid_capstyle="round", zorder=6)
                    ax.plot([bx - 0.10 + xo, bx + 0.10 + xo], [by - dyd, by + dyd],
                            color=clr, lw=0.9, solid_capstyle="round", zorder=7)

    f7 = df[(df.variant == "C-fused") & (df.C == 32) & (df.F == 7)]
    if not f7.empty:
        ax.annotate(f"{f7['word_perplexity'].iloc[0]:.0f}", xy=(F_INDEX[7], y_top_edge),
                    xytext=(8, -1), textcoords="offset points", fontsize=6,
                    color="#C62828", fontweight="bold", ha="left", va="top", zorder=7)

    ax.set_xlim(-0.5, len(F_AXIS) - 0.5)
    ax.set_xticks(range(len(F_AXIS)))
    ax.set_xticklabels([str(f) for f in F_AXIS])
    ax.set_xlabel("Fractional bits ($F$)")
    ax.yaxis.set_major_locator(FixedLocator(PPL_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("WikiText-2 Perplexity")
    ax.grid(True, which="major", axis="y", alpha=0.2, zorder=0)

    zone_handles = [
        Patch(facecolor="#FFCDD2", edgecolor="#C62828", linewidth=0.5, label="Catastrophic"),
        Patch(facecolor="#FFF9C4", edgecolor="#F57F17", linewidth=0.5, label="Transition"),
        Patch(facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=0.5, label="Iso-accuracy"),
    ]
    ax.legend(handles=zone_handles, loc="upper right", fontsize=6.0, frameon=True,
              fancybox=False, edgecolor="#CCCCCC", handlelength=1.0,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.2)
    ax.set_title("FP8-E4M3 — Accuracy", fontsize=8, fontweight="bold", pad=4)


def _short(p: Path) -> str:
    """Path relative to the current directory when possible, else absolute."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def main():
    bar = "═" * 64
    print(f"\n{bar}")
    print("  Figure 6(a)  —  FP8 CoFDA design space")
    print(bar)
    print("  Reproduces the paper's Figure 6(a): WikiText-2 perplexity vs F,")
    print("  C-fused (cofda) vs C-decoupled (cofda_decoupled), at CS 16 / 32,")
    print("  from the emulated sweep results.")
    print(bar)

    df = load_data()
    print(f"  ✓ read    {len(df):>2} runs        from  {_short(RESULTS)}/")

    df[["variant", "C", "F", "word_perplexity"]].to_csv(CSV_OUT, index=False)
    print(f"  ✓ wrote   {CSV_OUT.name:<13} tidy data ({len(df)} rows)")

    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    draw_panel(ax, df)

    marker_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_FUSED,
               markeredgecolor="white", markersize=5, label="C-fused CS=16"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=CLR_FUSED,
               markeredgecolor="white", markersize=5, label="C-fused CS=32"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_DECOUP,
               markeredgecolor="white", markersize=5, label="C-decoup CS=16"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=CLR_DECOUP,
               markeredgecolor="white", markersize=5, label="C-decoup CS=32"),
    ]
    fig.legend(handles=marker_legend, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC",
               fontsize=6, columnspacing=0.8, handletextpad=0.3, borderpad=0.3)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.135, right=0.97)
    fig.savefig(PNG_OUT, dpi=200)
    print(f"  ✓ wrote   {PNG_OUT.name:<13} the plot")
    print(bar)
    loc = _short(HERE)
    print(f"  outputs in  {HERE.name if loc == '.' else loc}/")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
