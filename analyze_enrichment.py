#!/usr/bin/env python3
"""
analyze_enrichment.py

Runs LiPyphilic neighbour enrichment on ONE composition folder and writes:
  - analysis_out/neighbour_counts.csv
  - analysis_out/enrichment_full.csv
  - analysis_out/enrichment_pops_timeseries.csv   (POPS view: fePOPS, fePOPC; backward compatible)
  - analysis_out/enrichment_timeseries_POPS_view.csv  (POPS view, includes feCHOL when present)
  - analysis_out/enrichment_timeseries_POPC_view.csv  (POPC view: fePOPS, fePOPC, feCHOL when present)
  - analysis_out/enrichment_timeseries_CHOL_view.csv  (CHOL view: enrichment around cholesterol; when CHOL present)
  - analysis_out/enrichment_POPS_neighbours.png

How enrichment is calculated:
  - For each lipid we count neighbours within cutoff_A (default 12 Å) using beads GL1/GL2/ROH.
  - LiPyphilic computes enrichment for every lipid type (POPS, POPC, CHOL, ...).
  - Enrichment = (local fraction of type X around a lipid) / (global fraction of X). 1.0 = random;
    >1 = enriched, <1 = depleted.
  - Smaller cutoff = first shell only → often larger separation; larger cutoff → tends toward 1.0.
"""

import os
import sys
import numpy as np
import MDAnalysis as mda

# LiPyphilic
from lipyphilic.analysis.neighbours import Neighbours


def pick_topology_and_traj(folder: str):
    """Prefer step7_lipids_ions (lipids + ions), then step7_lipids, then production."""
    for xname, gname in [
        ("step7_lipids_ions.xtc", "step7_lipids_ions.gro"),
        ("step7_lipids.xtc", "step7_lipids.gro"),
    ]:
        xtc = os.path.join(folder, xname)
        gro = os.path.join(folder, gname)
        if os.path.exists(gro) and os.path.exists(xtc):
            return gro, xtc
        if os.path.exists(gro):
            xtc = os.path.join(folder, "step7_production.xtc")
            if os.path.exists(xtc):
                return gro, xtc

    tpr = os.path.join(folder, "step7_production.tpr")
    xtc = os.path.join(folder, "step7_production.xtc")
    if os.path.exists(tpr) and os.path.exists(xtc):
        return tpr, xtc

    # List what exists so the user can see what's missing
    try:
        found = [f for f in os.listdir(folder) if not f.startswith(".")]
    except OSError:
        found = []
    raise FileNotFoundError(
        f"Need step7_lipids_ions.gro/xtc or step7_lipids.gro/xtc or step7_production.tpr/xtc in folder.\n"
        f"Folder: {folder}\n"
        f"Contents (non-hidden): {found or '(empty or not readable)'}"
    )


def run_enrichment_lipyphilic(u, outdir, cutoff_A=12.0, step=1):
    """
    Compute neighbour enrichment (POPS-centred and POPC-centred) and save outputs.
    Writes timeseries for POPS view (fePOPS, fePOPC, feCHOL when present) and
    POPC view (fePOPS, fePOPC, feCHOL when present). The small summary plot uses POPS and POPC only.
    """
    os.makedirs(outdir, exist_ok=True)

    nb = Neighbours(
        universe=u,
        lipid_sel="name GL1 GL2 ROH",
        cutoff=cutoff_A,
    )
    nb.run(step=step, verbose=True)

    counts, enrich = nb.count_neighbours(return_enrichment=True)

    counts.to_csv(os.path.join(outdir, "neighbour_counts.csv"), index=False)
    enrich.to_csv(os.path.join(outdir, "enrichment_full.csv"), index=False)

    # LiPyphilic output: first column is Label/resname; Frame may be "Frame" or "frame"
    id_col = enrich.columns[0]
    frame_col = "Frame" if "Frame" in enrich.columns else "frame"
    if frame_col not in enrich.columns:
        raise RuntimeError(
            f"Expected 'Frame' or 'frame' column in enrichment output. Columns: {list(enrich.columns)}"
        )

    pops = enrich[enrich[id_col] == "POPS"].copy()
    popc_df = enrich[enrich[id_col] == "POPC"].copy()
    chol_df = enrich[enrich[id_col] == "CHOL"].copy()

    dt_ps = float(getattr(u.trajectory, "dt", 1.0))
    ts = None

    # POPS view: only if this system has POPS lipids; write whatever enrichment columns exist
    if len(pops) > 0:
        cols_pops = [c for c in ["fePOPS", "fePOPC", "feCHOL"] if c in pops.columns]
        if not cols_pops:
            cols_pops = [c for c in ["POPS", "POPC", "CHOL"] if c in pops.columns]
        if cols_pops:
            ts = pops.groupby(frame_col)[cols_pops].mean().reset_index()
            ts["time_us"] = ts[frame_col] * dt_ps * step / 1e6
            ts.to_csv(os.path.join(outdir, "enrichment_pops_timeseries.csv"), index=False)
            ts.to_csv(os.path.join(outdir, "enrichment_timeseries_POPS_view.csv"), index=False)

    # POPC view: only if this system has POPC lipids
    if len(popc_df) > 0:
        cols_popc = [c for c in ["fePOPS", "fePOPC", "feCHOL"] if c in popc_df.columns] or [c for c in ["POPS", "POPC", "CHOL"] if c in popc_df.columns]
        cols_popc = [c for c in cols_popc if c in popc_df.columns]
        if cols_popc:
            ts_popc = popc_df.groupby(frame_col)[cols_popc].mean().reset_index()
            ts_popc["time_us"] = ts_popc[frame_col] * dt_ps * step / 1e6
            ts_popc.to_csv(os.path.join(outdir, "enrichment_timeseries_POPC_view.csv"), index=False)

    # CHOL view: only if this system has cholesterol
    if len(chol_df) > 0:
        cols_chol = [c for c in ["fePOPS", "fePOPC", "feCHOL"] if c in chol_df.columns] or [c for c in ["POPS", "POPC", "CHOL"] if c in chol_df.columns]
        cols_chol = [c for c in cols_chol if c in chol_df.columns]
        if cols_chol:
            ts_chol = chol_df.groupby(frame_col)[cols_chol].mean().reset_index()
            ts_chol["time_us"] = ts_chol[frame_col] * dt_ps * step / 1e6
            ts_chol.to_csv(os.path.join(outdir, "enrichment_timeseries_CHOL_view.csv"), index=False)

    # Require at least one view so we wrote something (avoid silent skip when e.g. no lipids match)
    if ts is None and len(popc_df) == 0 and len(chol_df) == 0:
        raise RuntimeError(
            "No enrichment data: no POPS, POPC, or CHOL rows in output. "
            f"Columns: {list(enrich.columns)}"
        )

    # ---- Plot (paper-style panel H): only if we have POPS view with plottable columns
    if ts is not None and len(ts.columns) > 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.6, 3.6))
        for col, label in [("fePOPS", "POPS"), ("POPS", "POPS"),
                           ("fePOPC", "POPC"), ("POPC", "POPC"),
                           ("feCHOL", "CHOL"), ("CHOL", "CHOL")]:
            if col in ts.columns:
                plt.plot(ts["time_us"], ts[col], label=label)
        plt.axhline(1.0, linestyle="--", linewidth=1.5)
        plt.xlabel("Time (µs)")
        plt.ylabel("Neighbour enrichment")
        plt.ylim(0.6, 1.4)
        plt.legend(frameon=False, ncol=2, loc="upper left")
        plt.tight_layout()
        out_png = os.path.join(outdir, "enrichment_POPS_neighbours.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote:", out_png)


def main(folder: str, cutoff_A=12.0, step=1):
    folder = os.path.abspath(folder)
    topo, xtc = pick_topology_and_traj(folder)
    u = mda.Universe(topo, xtc)

    outdir = os.path.join(folder, "analysis_out")
    os.makedirs(outdir, exist_ok=True)

    run_enrichment_lipyphilic(u, outdir, cutoff_A=cutoff_A, step=step)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_enrichment.py /path/to/FOLDER [cutoff_A] [step]")
        print("       python analyze_enrichment.py FOLDER1 FOLDER2 ...   (multiple folders, same cutoff)")
        print("  cutoff_A: neighbour distance in Å (default 12). Try 6–8 for stronger enrichment contrast.")
        raise SystemExit(1)

    # If second arg is a number (cutoff), single-folder mode with optional cutoff/step
    if len(sys.argv) >= 3:
        try:
            cutoff = float(sys.argv[2])
            step = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
            main(sys.argv[1], cutoff_A=cutoff, step=step)
            sys.exit(0)
        except ValueError:
            pass

    # Multiple folders (e.g. shell glob: python analyze_enrichment.py Production_out/POPC_*)
    folders = [os.path.abspath(p) for p in sys.argv[1:]]
    cutoff = 12.0
    step = 1
    for i, folder in enumerate(folders):
        if not os.path.isdir(folder):
            print(f"Skipping (not a directory): {folder}")
            continue
        print(f"\n[{i+1}/{len(folders)}] {folder}")
        try:
            main(folder, cutoff_A=cutoff, step=step)
        except Exception as e:
            print(f"  Error: {e}")
            raise
