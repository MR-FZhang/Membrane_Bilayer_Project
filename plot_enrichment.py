#!/usr/bin/env python3
"""
plot_enrichment.py

Plots enrichment time series (with lines of best fit) from analyze_enrichment.py.

Modes:
  1. Batch (directory or glob): Loop over all composition folders and produce one
     figure per composition (all three pairs that have data, or 1–2 panels if some
     species are missing). Example: python plot_enrichment.py Production_out
     or: python plot_enrichment.py Production_out/POPC_*
  2. Single folder: One figure for that composition (all pairs that have data).
  3. Two folders (comparison): Two panels side-by-side comparing two compositions.
  Panels can show 1 or 2 species when a composition has only one of the pair (e.g. POPS-only).

POPS / POPC / CHOL view (what the CSV files mean):
  - "POPS view": Enrichment *with respect to POPS lipids* — in the neighbourhood of
    each POPS, how enriched is POPS, POPC, or CHOL vs bulk?
  - "POPC view": Same idea *with respect to POPC lipids*.
  - "CHOL view": Same idea *with respect to cholesterol* (enrichment around CHOL).
  So we have one CSV per reference species (POPS, POPC, CHOL) to capture enrichment
  from each perspective.

Usage:
  Batch (all compositions under a directory):
    python plot_enrichment.py Production_out
  Batch (shell glob):
    python plot_enrichment.py Production_out/POPC_*
  Single composition:
    python plot_enrichment.py <folder> [--pair POPC_POPS|POPC_CHOL|POPS_CHOL]
  Comparison (two compositions):
    python plot_enrichment.py <folder_low> <folder_high> [--pair ...] [--label-low ...] [--label-high ...]
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _is_composition_dir(name: str) -> bool:
    """True if folder name looks like POPC_XX_POPS_YY_CHOL_ZZ."""
    return re.search(r"POPC_\d+_POPS_\d+_CHOL_\d+", name, re.IGNORECASE) is not None


def _composition_subdirs(root: str):
    """Yield paths to composition subdirectories under root."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path) and _is_composition_dir(name):
            yield path

# Pair configuration: (timeseries_file, col1, col2, label1, label2)
# POPC_POPS = WRT POPS (enrichment around POPS). POPC_POPS_POPCview = WRT POPC (around POPC).
PAIR_CONFIG = {
    "POPC_POPS": ("enrichment_pops_timeseries.csv", "fePOPS", "fePOPC", "POPS", "POPC"),
    "POPC_POPS_POPCview": ("enrichment_timeseries_POPC_view.csv", "fePOPS", "fePOPC", "POPS", "POPC"),
    "POPC_CHOL": ("enrichment_timeseries_POPC_view.csv", "fePOPC", "feCHOL", "POPC", "CHOL"),
    "POPS_CHOL": ("enrichment_timeseries_POPS_view.csv", "fePOPS", "feCHOL", "POPS", "CHOL"),
}
# Order of panels for single-folder figure: POPC/POPS WRT POPS, then WRT POPC, then other pairs
PANEL_ORDER = ["POPC_POPS", "POPC_POPS_POPCview", "POPC_CHOL", "POPS_CHOL"]
# Human-readable panel titles (pair_key -> subtitle)
PANEL_TITLES = {
    "POPC_POPS": "POPC / POPS (WRT POPS)",
    "POPC_POPS_POPCview": "POPC / POPS (WRT POPC)",
    "POPC_CHOL": "POPC / CHOL",
    "POPS_CHOL": "POPS / CHOL",
}

COLORS = {"POPS": "#c45a3b", "POPC": "#1f77b4", "CHOL": "#2ca02c"}


def load_timeseries_for_pair(folder: str, pair: str) -> pd.DataFrame:
    """Load the timeseries CSV appropriate for the given pair. Needs at least one of the pair's columns."""
    if pair not in PAIR_CONFIG:
        raise ValueError(f"Unknown pair '{pair}'. Use one of: {list(PAIR_CONFIG.keys())}")
    filename, c1, c2, _, _ = PAIR_CONFIG[pair]
    path = os.path.join(folder, "analysis_out", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run analyze_enrichment.py first for this folder.")
    df = pd.read_csv(path)
    if "time_us" not in df.columns:
        raise RuntimeError(f"'time_us' column missing in {path}. Columns: {list(df.columns)}")
    if c1 not in df.columns and c2 not in df.columns:
        raise RuntimeError(f"Pair {pair} needs at least one of [{c1}, {c2}] in {path}. Columns: {list(df.columns)}")
    return df


def pick_cols(df: pd.DataFrame, pair: str):
    """Return (col1, col2, label1, label2) for the pair; use None for missing column."""
    _, c1, c2, l1, l2 = PAIR_CONFIG[pair]
    col1 = c1 if c1 in df.columns else None
    col2 = c2 if c2 in df.columns else None
    return col1, col2, l1, l2


def pick_cols_available(df: pd.DataFrame, pair: str):
    """Return list of (col, label) for columns that exist for this pair (1 or 2 items)."""
    col1, col2, l1, l2 = pick_cols(df, pair)
    out = []
    if col1 is not None:
        out.append((col1, l1))
    if col2 is not None:
        out.append((col2, l2))
    return out


def best_fit_and_mean(t: np.ndarray, y: np.ndarray):
    """Linear fit y = a + b*t. Return (a, b, mean_y)."""
    mean_y = float(np.nanmean(y))
    if len(t) < 2 or np.all(np.isnan(y)):
        return np.nan, np.nan, mean_y
    mask = ~(np.isnan(t) | np.isnan(y))
    if np.sum(mask) < 2:
        return np.nan, np.nan, mean_y
    t_ = t[mask]
    y_ = y[mask]
    A = np.column_stack([np.ones_like(t_), t_])
    coeffs, _, _, _ = np.linalg.lstsq(A, y_, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    return a, b, mean_y


def _draw_one_panel(ax, df, pair, title, save_means, folder, composition):
    """Draw one panel: time series + best-fit (average) lines for 1 or 2 species in this pair."""
    series = pick_cols_available(df, pair)
    if not series:
        return
    t = df["time_us"].values
    means_saved = {}
    for col, label in series:
        color = COLORS.get(label, "#333333")
        _, _, mean_y = best_fit_and_mean(t, df[col].values)
        means_saved[label] = mean_y
        ax.plot(df["time_us"], df[col], label=label, color=color, linewidth=1.2, alpha=0.65)
        # Average over time: horizontal line at mean value
        if not np.isnan(mean_y):
            ax.axhline(mean_y, linestyle="--", color=color, linewidth=1.6, dashes=(6, 3), label=f"{label} (avg)")
    ax.axhline(1.0, linestyle="--", linewidth=1.5, color="black")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (µs)", fontsize=10)
    ylab = " / ".join(l for _, l in series)
    ax.set_ylabel(f"Enrichment ({ylab})", fontsize=10)
    ax.set_ylim(0.6, 1.4)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    if save_means and folder and means_saved:
        means_path = os.path.join(folder, "analysis_out", "enrichment_means.csv")
        labels = [l for _, l in series]
        row_dict = {"composition": composition, "pair": pair}
        if len(labels) >= 1:
            row_dict["species1"] = labels[0]
            row_dict["mean_species1"] = means_saved[labels[0]]
        if len(labels) >= 2:
            row_dict["species2"] = labels[1]
            row_dict["mean_species2"] = means_saved[labels[1]]
        else:
            row_dict["species2"] = ""
            row_dict["mean_species2"] = np.nan
        row = pd.DataFrame([row_dict])
        if os.path.exists(means_path):
            existing = pd.read_csv(means_path)
            mask = (existing["composition"] != composition) | (existing["pair"] != pair)
            existing = existing[mask]
            row = pd.concat([existing, row], ignore_index=True)
        else:
            os.makedirs(os.path.dirname(means_path), exist_ok=True)
        row.to_csv(means_path, index=False)


def plot_single_folder(
    folder: str,
    pair: str = None,
    outfile: str = "enrichment_single.png",
    save_means: bool = True,
):
    """
    One figure for one composition: up to 4 panels (POPC/POPS WRT POPS, POPC/POPS WRT POPC,
    POPC/CHOL, POPS/CHOL). If pair is set, only that pair is shown (one panel).
    """
    folder = os.path.abspath(folder)
    composition = os.path.basename(os.path.normpath(folder))
    pairs_to_try = [pair] if pair else PANEL_ORDER
    panels = []
    for p in pairs_to_try:
        if p not in PAIR_CONFIG:
            continue
        path = os.path.join(folder, "analysis_out", PAIR_CONFIG[p][0])
        if not os.path.exists(path):
            continue
        try:
            df = load_timeseries_for_pair(folder, p)
        except (FileNotFoundError, RuntimeError):
            continue
        panels.append((p, df))
    if not panels:
        raise FileNotFoundError(
            f"No enrichment timeseries found in {folder}/analysis_out for any of {pairs_to_try}. "
            f"Run analyze_enrichment.py first."
        )
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (p, df) in zip(axes, panels):
        subtitle = PANEL_TITLES.get(p, p.replace("_", " / "))
        title = f"{composition}\n{subtitle}"
        _draw_one_panel(ax, df, p, title, save_means, folder, composition)
    fig.tight_layout()
    outdir = os.path.join(folder, "analysis_out")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)
    if save_means:
        print("Updated enrichment_means.csv in folder.")


def plot_two_panels(
    folder_low: str,
    folder_high: str,
    pair: str = "POPC_POPS",
    label_low: str = "Low CHOL",
    label_high: str = "High CHOL",
    outfile: str = None,
    save_means: bool = True,
):
    if pair not in PAIR_CONFIG:
        raise ValueError(f"Unknown pair '{pair}'. Use one of: {list(PAIR_CONFIG.keys())}")

    dfL = load_timeseries_for_pair(folder_low, pair)
    dfH = load_timeseries_for_pair(folder_high, pair)
    series = pick_cols_available(dfL, pair)
    if not series:
        series = pick_cols_available(dfH, pair)
    if not series:
        raise RuntimeError(f"No enrichment columns for pair {pair} in either folder.")
    tL = dfL["time_us"].values
    tH = dfH["time_us"].values

    if outfile is None:
        outfile = f"enrichment_compare_{pair}.png"
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.0), sharey=True)
    ymin, ymax = 0.6, 1.4
    means_low, means_high = {}, {}

    for ax, df, title, means_out in [
        (axes[0], dfL, label_low, means_low),
        (axes[1], dfH, label_high, means_high),
    ]:
        for col, label in series:
            if col not in df.columns:
                continue
            color = COLORS.get(label, "#333333")
            t = df["time_us"].values
            _, _, mean_y = best_fit_and_mean(t, df[col].values)
            means_out[label] = mean_y
            ax.plot(df["time_us"], df[col], label=label, color=color, linewidth=1.2, alpha=0.65)
            if not np.isnan(mean_y):
                ax.axhline(mean_y, linestyle="--", color=color, linewidth=1.6, dashes=(6, 3), label=f"{label} (avg)")
        ax.axhline(1.0, linestyle="--", linewidth=1.5, color="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (µs)", fontsize=10)
        ax.set_ylabel(f"Enrichment ({' / '.join(l for c, l in series if c in df.columns)})", fontsize=10)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)

    fig.tight_layout()
    outdir = os.path.join(folder_high, "analysis_out")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)

    if save_means:
        comp_low = os.path.basename(os.path.normpath(folder_low))
        comp_high = os.path.basename(os.path.normpath(folder_high))
        labels = [l for _, l in series]
        for folder, comp, means in [
            (folder_low, comp_low, means_low),
            (folder_high, comp_high, means_high),
        ]:
            means_path = os.path.join(folder, "analysis_out", "enrichment_means.csv")
            row_dict = {"composition": comp, "pair": pair}
            row_dict["species1"] = labels[0] if len(labels) >= 1 else ""
            row_dict["mean_species1"] = means.get(labels[0], np.nan) if labels else np.nan
            row_dict["species2"] = labels[1] if len(labels) >= 2 else ""
            row_dict["mean_species2"] = means.get(labels[1], np.nan) if len(labels) >= 2 else np.nan
            row = pd.DataFrame([row_dict])
            if os.path.exists(means_path):
                existing = pd.read_csv(means_path)
                mask = (existing["composition"] != comp) | (existing["pair"] != pair)
                existing = existing[mask]
                row = pd.concat([existing, row], ignore_index=True)
            else:
                os.makedirs(os.path.dirname(means_path), exist_ok=True)
            row.to_csv(means_path, index=False)
        print("Updated enrichment_means.csv in both folders.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot enrichment time series: single composition, batch over many, or compare two."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="One path: single figure or batch (if directory, loop over composition subdirs). "
             "Two or more paths: batch (one figure per folder). Use --compare for side-by-side comparison.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparison mode: exactly two folders, plot side-by-side (default: batch over all given folders).",
    )
    parser.add_argument(
        "--pair",
        choices=list(PAIR_CONFIG.keys()),
        default=None,
        help="Show only this pair (default: all pairs for single/batch, POPC_POPS for comparison).",
    )
    parser.add_argument("--label-low", default="Low CHOL", help="Comparison mode: title for first panel.")
    parser.add_argument("--label-high", default="High CHOL", help="Comparison mode: title for second panel.")
    args = parser.parse_args()

    if len(args.folders) == 1:
        path = os.path.abspath(args.folders[0])
        if os.path.isdir(path):
            subdirs = list(_composition_subdirs(path))
            if subdirs:
                # Batch: loop over all composition folders under this directory
                for i, folder in enumerate(subdirs):
                    print(f"\n[{i+1}/{len(subdirs)}] {os.path.basename(folder)}")
                    try:
                        plot_single_folder(folder, pair=args.pair)
                    except Exception as e:
                        print(f"  Error: {e}")
                sys.exit(0)
        # Single composition folder
        plot_single_folder(path, pair=args.pair)
    elif args.compare and len(args.folders) == 2:
        low = os.path.abspath(args.folders[0])
        high = os.path.abspath(args.folders[1])
        pair = args.pair or "POPC_POPS"
        if pair not in PAIR_CONFIG:
            pair = "POPC_POPS"
        plot_two_panels(
            low, high,
            pair=pair,
            label_low=args.label_low,
            label_high=args.label_high,
        )
    else:
        # Batch: two or more folders (e.g. shell glob: Production_out/POPC_90*)
        folders = [os.path.abspath(p) for p in args.folders]
        for i, folder in enumerate(folders):
            if not os.path.isdir(folder):
                print(f"Skipping (not a directory): {folder}")
                continue
            print(f"\n[{i+1}/{len(folders)}] {os.path.basename(folder)}")
            try:
                plot_single_folder(folder, pair=args.pair)
            except Exception as e:
                print(f"  Error: {e}")
