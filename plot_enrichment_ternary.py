#!/usr/bin/env python3
"""
plot_enrichment_ternary.py

Plots enrichment values on a ternary (triangular) composition diagram for
POPC / POPS / CHOL. Produces 6 figures: for each species pair (POPC-POPS,
POPC-CHOL, POPS-CHOL), one triangle colored by the first species' enrichment
and one by the second species' enrichment.

Color scheme (scientific): enrichment > 1 → blue gradient (darker = more
enrichment); enrichment < 1 → red gradient (darker = more depletion); 1 = neutral.

Requires enrichment_means_all.csv from collect_enrichment_means.py (run after
analyze_enrichment.py on all compositions).

Usage:
  python plot_enrichment_ternary.py [root_dir] [--outdir DIR]
  root_dir defaults to Production_out.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm


def parse_composition(folder_name: str):
    """Return (popc_frac, pops_frac, chol_frac) with sum=1, or None."""
    m = re.search(r"POPC_(\d+)_POPS_(\d+)_CHOL_(\d+)", folder_name, re.IGNORECASE)
    if not m:
        return None
    popc, pops, chol = map(int, m.groups())
    s = popc + pops + chol
    return (popc / s, pops / s, chol / s)


def barycentric_to_xy(popc: float, pops: float, chol: float):
    """
    Ternary: CHOL at top, POPS bottom-left, POPC bottom-right.
    Vertices: POPS (0,0), POPC (1,0), CHOL (0.5, 1). Return (x, y).
    """
    x = popc + 0.5 * chol
    y = chol
    return x, y


# Which column in enrichment_means_all.csv for each of the 6 plots
TERNARY_PLOTS = [
    ("POPC / POPS", "POPC", "mean_fePOPC_POPSview"),
    ("POPC / POPS", "POPS", "mean_fePOPS_POPSview"),
    ("POPC / CHOL", "POPC", "mean_fePOPC_POPCview"),
    ("POPC / CHOL", "CHOL", "mean_feCHOL_POPCview"),
    ("POPS / CHOL", "POPS", "mean_fePOPS_POPSview"),
    ("POPS / CHOL", "CHOL", "mean_feCHOL_POPSview"),
]

def draw_ternary_axes(ax, label_offset=0.12, tick_fontsize=7):
    """
    Draw full equilateral triangle: vertices POPS (0,0), POPC (1,0), CHOL (0.5, 1).
    Labels: POPC bottom center, CHOL right edge, POPS left edge. Percentage ticks along each edge.
    POPS left axis: 100% at bottom (towards POPC corner), 0% at top (CHOL vertex).
    """
    x_bl, y_bl = 0.0, 0.0
    x_br, y_br = 1.0, 0.0
    x_t, y_t = 0.5, 1.0
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.18
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)

    # Triangle outline
    ax.plot([x_bl, x_br], [y_bl, y_br], "k-", linewidth=1.2)
    ax.plot([x_bl, x_t], [y_bl, y_t], "k-", linewidth=1.2)
    ax.plot([x_br, x_t], [y_br, y_t], "k-", linewidth=1.2)

    # Full grid: constant-CHOL horizontal tiers
    for i in range(1, 10):
        f = i / 10.0
        x_l = 0.5 * f
        x_r = 1.0 - 0.5 * f
        ax.plot([x_l, x_r], [f, f], "k-", linewidth=0.35, alpha=0.35)
    # Diagonals (constant POPC / POPS)
    for k in range(1, 10):
        kf = k / 10.0
        ax.plot([kf, 0.5 + 0.5 * kf], [0, 1 - kf], "k-", linewidth=0.35, alpha=0.35)
        ax.plot([1 - kf, 0.5 * (1 - kf)], [0, 1 - kf], "k-", linewidth=0.35, alpha=0.35)

    # Axis labels: POPC closer to bottom axis; POPS and CHOL at edge midpoints with small outward offset
    fontsize = 11
    # POPC: bottom center, slightly closer to axis
    ax.text(0.5, y_bl - 0.12, "POPC", ha="center", va="top", fontsize=fontsize)
    # Left edge (POPS): midpoint (0.25, 0.5), place just outside to the left
    ax.text(0.25 - 0.1, 0.5, "POPS", ha="right", va="center", fontsize=fontsize, rotation=60)
    # Right edge (CHOL): midpoint (0.75, 0.5), place clearly outside to the right (was inside before)
    ax.text(0.75 + 0.1, 0.5, "CHOL", ha="left", va="center", fontsize=fontsize, rotation=-60)

    # Percentage ticks: bottom POPC 0–100; left POPS 100 at bottom → 0 at top; right CHOL 0–100
    for pct in range(0, 101, 10):
        f = pct / 100.0
        # Bottom: POPC 0 at left to 100 at right
        x_b = x_bl + f * (x_br - x_bl)
        ax.text(x_b, y_bl - 0.06, str(pct), ha="center", va="top", fontsize=tick_fontsize, rotation=-30)
        # Left edge (POPS): reverse — 100% POPS at bottom (f=0), 0% at top (f=1)
        x_l = x_bl + f * (x_t - x_bl)
        y_l = y_bl + f * (y_t - y_bl)
        ax.text(x_l - 0.02, y_l+0.05, str(100 - pct), ha="right", va="center", fontsize=tick_fontsize, rotation=30)
        # Right edge: CHOL 0 at bottom to 100 at top
        x_r = x_br + f * (x_t - x_br)
        y_r = y_br + f * (y_t - y_br)
        ax.text(x_r +0.06, y_r, str(pct), ha="left", va="center", fontsize=tick_fontsize, rotation=0)



def plot_one_ternary(ax, df: pd.DataFrame, value_col: str, title: str, pair_label: str, species_label: str):
    """
    Plot composition points on full ternary triangle. Points with data: filled circles
    colored by value_col. Points with no data (NaN/missing): empty white circles (outline only).
    """
    xy_filled = []
    vals = []
    xy_empty = []
    for _, row in df.iterrows():
        comp = row["composition"]
        parsed = parse_composition(comp)
        if parsed is None:
            continue
        popc, pops, chol = parsed
        x, y = barycentric_to_xy(popc, pops, chol)
        v = row.get(value_col)
        if pd.notna(v):
            try:
                vals.append(float(v))
                xy_filled.append((x, y))
            except (ValueError, TypeError):
                xy_empty.append((x, y))
        else:
            xy_empty.append((x, y))

    draw_ternary_axes(ax)
    # Same size as filled points so they align on the grid
    point_size = 450
    edge_kw = dict(edgecolors="k", linewidths=0.8)
    # Draw empty circles first (behind), then dashed line across each to mean "no data"
    if xy_empty:
        xy_e = np.array(xy_empty)
        ax.scatter(
            xy_e[:, 0], xy_e[:, 1],
            s=point_size, facecolors="white", **edge_kw
        )
        # Solid cross (X) inside each empty circle to indicate no data
        half = 0.032  # half-length in data coords so lines sit inside circle
        for x, y in xy_e:
            ax.plot(
                [x - half, x + half], [y - half, y + half],
                "k-", linewidth=0.8, solid_capstyle="butt"
            )
            ax.plot(
                [x - half, x + half], [y + half, y - half],
                "k-", linewidth=0.8, solid_capstyle="butt"
            )
    if xy_filled:
        xy = np.array(xy_filled)
        vals = np.array(vals)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        vcenter = 1.0
        if vmin >= 1 and vmax > 1:
            vmin = min(vmin, 0.99)
        if vmax <= 1 and vmin < 1:
            vmax = max(vmax, 1.01)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=vals, cmap="RdBu_r", norm=norm,
            s=point_size, **edge_kw
        )
        plt.colorbar(sc, ax=ax, label="Enrichment (mean)", shrink=0.7, aspect=25)
    else:
        ax.set_title(f"{pair_label} ({species_label})\nEnrichment (no data)", fontsize=11)
        return
    ax.set_title(f"{pair_label} ({species_label})\nEnrichment", fontsize=11)


def main(root_dir: str, outdir: str = None):
    if outdir is None:
        outdir = os.path.join(root_dir, "analysis_out")
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(root_dir, "enrichment_means_all.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Run collect_enrichment_means.py first.\n"
            f"  python collect_enrichment_means.py {root_dir}"
        )
    df = pd.read_csv(csv_path)
    if "composition" not in df.columns:
        raise RuntimeError(f"enrichment_means_all.csv must have 'composition' column. Columns: {list(df.columns)}")

    for pair_label, species_label, value_col in TERNARY_PLOTS:
        if value_col not in df.columns:
            print(f"Skipping {pair_label} ({species_label}): column {value_col} not in CSV.")
            continue
        fig, ax = plt.subplots(figsize=(6, 5.5))
        title = f"{pair_label} – {species_label} enrichment"
        plot_one_ternary(ax, df, value_col, title, pair_label, species_label)
        safe_name = f"enrichment_ternary_{pair_label.replace(' ', '_').replace('/', '_')}_{species_label}.png"
        safe_name = re.sub(r"[^\w\-_.]", "_", safe_name)
        outpath = os.path.join(outdir, safe_name)
        fig.tight_layout()
        fig.savefig(outpath, dpi=1500, bbox_inches="tight")
        plt.close(fig)
        print("Wrote:", outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot enrichment on ternary composition diagrams.")
    parser.add_argument("root_dir", nargs="?", default="Production_out", help="Root with enrichment_means_all.csv")
    parser.add_argument("--outdir", default=None, help="Output directory for PNGs (default: root_dir/analysis_out)")
    args = parser.parse_args()
    root_dir = os.path.abspath(args.root_dir)
    outdir = os.path.abspath(args.outdir) if args.outdir else None
    main(root_dir, outdir=outdir)
