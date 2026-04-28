#!/usr/bin/env python3
"""
analyze_phase_rdf_shell.py

Run global 2D tail-bead RDF, detect first minimum, and per-lipid first-shell
neighbor counts for one or many composition folders. RDF plots show both raw
and normalized g(r), peak position, g(r) at peak, and first-shell coordination number.

Usage:
  # Single root: discover composition subfolders (e.g. POPC_*_POPS_*_CHOL_* or DOPC_*_DPPC_*)
  python analyze_phase_rdf_shell.py Production_out [pattern]
  python analyze_phase_rdf_shell.py Production_out "POPC_50_*"
  python analyze_phase_rdf_shell.py Production_out "DOPC_*"

  # Multiple composition folders: run on each and also produce overlay plots
  python analyze_phase_rdf_shell.py path/to/folder1 path/to/folder2 path/to/folder3 path/to/folder4

For each folder, writes to <folder>/analysis_out/:
  - <composition>_rdf.png (labeled with composition; raw + normalized g(r); peak and coord. no.)
  - <composition>_rdf.dat, _r.npy, _g_raw.npy, _g_norm.npy
  - <composition>_neighbors.csv, _neighbors.npy, _neighbors_hist.png (labeled with composition)

When multiple folders are given, also writes to <parent>/analysis_out/:
  - overlay_rdf.png (all RDFs overlaid, stacked with 0.05 vertical offset per curve)
  - overlay_neighbors_hist.png (all neighbor histograms overlaid, stacked similarly)

  Use --order-by SPECIES (e.g. --order-by CHOL) to order curves by that species ascending;
  the topmost line then corresponds to the composition with the highest value of that species.
  Use --overlay-offset to change the vertical shift (default 0.05).
"""

import re
import sys
import fnmatch
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import MDAnalysis as mda
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Config: lipid species and tail beads (extend here for new systems; logic unchanged)
# ---------------------------------------------------------------------------
# Resnames included in global 2D RDF and per-lipid first-shell neighbor analysis
TAIL_RESNAMES = "POPC POPS DOPC DPPC"
# Martini tail bead names (same for POPC/POPS/DOPC/DPPC)
TAIL_BEAD_NAMES = "D2A C2B"

# Patterns that identify a composition folder (for discovery under a root path)
_COMPOSITION_PATTERNS = [
    re.compile(r"POPC_\d+_POPS_\d+_CHOL_\d+", re.IGNORECASE),
    re.compile(r"DOPC_\d+_DPPC_\d+_CHOL_\d+", re.IGNORECASE),
    re.compile(r"DOPC_\d+_DPPC_\d+", re.IGNORECASE),
]


def get_species_value(composition_str, species_name):
    """Extract numeric value for a species from a composition string (e.g. CHOL from 'POPC_50_POPS_0_CHOL_50' -> 50).
    Returns float or None if not found."""
    if not composition_str or not species_name:
        return None
    # Match SPECIES_<digits> in the folder name (case-insensitive)
    pat = re.compile(rf"{re.escape(species_name.strip())}_(\d+)", re.IGNORECASE)
    m = pat.search(composition_str)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Topology/trajectory discovery (same as analyze_chol_heatmap)
# ---------------------------------------------------------------------------
def pick_topology_and_traj(folder):
    """Locate topology and trajectory files inside a composition folder."""
    folder = Path(folder)
    for tname, xname in [
        ("step7_lipids_ions.tpr", "step7_lipids_ions.xtc"),
        ("step7_production.tpr", "step7_lipids_ions.xtc"),
        ("step7_production.tpr", "step7_production.xtc"),
    ]:
        tpath = folder / tname
        xpath = folder / xname
        if tpath.exists() and xpath.exists():
            return str(tpath), str(xpath)
    for xname, gname in [
        ("step7_lipids_ions.xtc", "step7_lipids_ions.gro"),
        ("step7_lipids.xtc", "step7_lipids.gro"),
    ]:
        xpath = folder / xname
        gpath = folder / gname
        if gpath.exists() and xpath.exists():
            return str(gpath), str(xpath)
    raise FileNotFoundError(
        f"No topology+trajectory found in {folder}. "
        f"Need step7_lipids_ions.tpr/xtc or step7_production.tpr/xtc (or .gro/.xtc)."
    )


# ---------- utilities ----------
def detect_scale_from_box(Lz):
    return 0.1 if Lz > 50 else 1.0


def min_image_2d(dxy, Lx, Ly):
    dxy[:, 0] -= Lx * np.round(dxy[:, 0] / Lx)
    dxy[:, 1] -= Ly * np.round(dxy[:, 1] / Ly)
    return dxy


def wrap_xy_to_box(xy, Lx, Ly):
    out = np.empty_like(xy)
    out[:, 0] = np.mod(xy[:, 0], Lx)
    out[:, 1] = np.mod(xy[:, 1], Ly)
    return out


def normalize_tail(r, g, r_norm_min):
    mask = r >= r_norm_min
    if mask.sum() < 10:
        k = max(10, int(0.1 * len(r)))
        denom = np.mean(g[-k:])
    else:
        denom = np.mean(g[mask])
    return g / denom if denom > 0 else g


def find_first_minimum(r, g, r_peak_min=0.3, r_peak_max=1.2, r_min_search_max=2.0):
    g_s = gaussian_filter1d(g, sigma=2.0)
    peak_mask = (r >= r_peak_min) & (r <= r_peak_max)
    if peak_mask.sum() < 5:
        raise RuntimeError("Peak search window too small or r-range too small.")
    i_peak_local = np.argmax(g_s[peak_mask])
    i_peak = np.where(peak_mask)[0][0] + i_peak_local
    r_peak = r[i_peak]
    min_mask = (r > r_peak) & (r <= r_min_search_max)
    if min_mask.sum() < 5:
        raise RuntimeError("Min search window too small; increase r_max or r_min_search_max.")
    i_min_local = np.argmin(g_s[min_mask])
    i_min = np.where(min_mask)[0][0] + i_min_local
    r_min = r[i_min]
    return r_peak, r_min


# ---------- global 2D RDF ----------
def compute_global_rdf_2d(u, sel, r_max=4.5, dr=0.01, stride=1, t_start=None, t_end=None, n_expected=None):
    grp = u.select_atoms(sel)
    if len(grp) == 0:
        raise ValueError(f"Empty selection: {sel}")
    edges = np.arange(0.0, r_max + dr, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros_like(r, dtype=np.float64)
    n_frames = 0
    n = len(grp)
    traj = u.trajectory
    t0 = traj[0].time
    t_last = traj[-1].time
    if t_start is None:
        t_start = t0
    if t_end is None:
        t_end = t_last
    spinner = itertools.cycle(r"\|/-")
    pbar = tqdm(
        total=n_expected, unit="frame", desc=" ",
        bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )
    for ts in traj[::stride]:
        if ts.time < t_start or ts.time > t_end:
            continue
        pbar.set_description(f" {next(spinner)} RDF (t={ts.time/1e6:.2f} µs)")
        pbar.update(1)
        Lx, Ly, Lz = ts.dimensions[:3]
        scale = detect_scale_from_box(Lz)
        Lx *= scale
        Ly *= scale
        xy = grp.positions[:, :2] * scale
        xy = wrap_xy_to_box(xy, Lx, Ly)
        tree = cKDTree(xy, boxsize=(Lx, Ly))
        pairs = tree.query_pairs(r=r_max, output_type="ndarray")
        if pairs.size > 0:
            d = xy[pairs[:, 0]] - xy[pairs[:, 1]]
            d = min_image_2d(d, Lx, Ly)
            rij = np.sqrt((d**2).sum(axis=1))
            h, _ = np.histogram(rij, bins=edges)
            counts += h
        n_frames += 1
    pbar.close()
    if n_frames == 0:
        raise RuntimeError("No frames selected in the requested time window.")
    neighbors_per_particle = (2.0 * counts) / (n_frames * n)
    Lx, Ly, Lz = u.trajectory[-1].dimensions[:3]
    scale = detect_scale_from_box(Lz)
    area = (Lx * scale) * (Ly * scale)
    rho_2d = n / area
    shell_area = 2.0 * np.pi * r * dr
    g = neighbors_per_particle / (rho_2d * shell_area + 1e-30)
    return r, g, counts, n_frames, (t_start, t_end)


# ---------- per-lipid first-shell neighbor counts ----------
def per_lipid_first_shell(u, tail_sel, r_cut, stride=1, t_start=None, t_end=None, n_expected=None):
    tails = u.select_atoms(tail_sel)
    if len(tails) == 0:
        raise ValueError(f"Empty tail selection: {tail_sel}")
    residues = tails.residues
    n_lip = len(residues)
    if n_lip == 0:
        raise RuntimeError("No lipid residues found for tail selection.")
    resids = np.array([r.resid for r in residues], dtype=int)
    resnames = np.array([r.resname for r in residues], dtype=object)
    neighbor_sum = np.zeros(n_lip, dtype=np.float64)
    n_frames = 0
    traj = u.trajectory
    t0 = traj[0].time
    t_last = traj[-1].time
    if t_start is None:
        t_start = t0
    if t_end is None:
        t_end = t_last
    spinner = itertools.cycle(r"\|/-")
    pbar = tqdm(
        total=n_expected, unit="frame", desc=" ",
        bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )
    for ts in traj[::stride]:
        if ts.time < t_start or ts.time > t_end:
            continue
        pbar.set_description(f" {next(spinner)} Shell (t={ts.time/1e6:.2f} µs)")
        pbar.update(1)
        Lx, Ly, Lz = ts.dimensions[:3]
        scale = detect_scale_from_box(Lz)
        Lx *= scale
        Ly *= scale
        xy = np.zeros((n_lip, 2), dtype=np.float64)
        for i, res in enumerate(residues):
            pos = res.atoms.positions[:, :2] * scale
            xy[i] = pos.mean(axis=0)
        xy = wrap_xy_to_box(xy, Lx, Ly)
        tree = cKDTree(xy, boxsize=(Lx, Ly))
        neigh_lists = tree.query_ball_point(xy, r=r_cut)
        for i, neigh in enumerate(neigh_lists):
            c = len(neigh) - (1 if i in neigh else 0)
            neighbor_sum[i] += c
        n_frames += 1
    pbar.close()
    if n_frames == 0:
        raise RuntimeError("No frames selected in the requested time window for per-lipid analysis.")
    mean_neighbors = neighbor_sum / n_frames
    return resids, resnames, mean_neighbors, n_frames, (t_start, t_end)


# ---------- coordination number from first peak (2D) ----------
def coordination_number_first_shell(r, g, r_min, rho_2d):
    """Integral 2*pi*rho*int_0^r_min r*g(r) dr = number of particles in first shell."""
    mask = (r > 0) & (r <= r_min)
    if mask.sum() < 2:
        return np.nan
    r_m = r[mask]
    g_m = g[mask]
    integrand = r_m * g_m
    # numpy 2.0+ removed trapz; use trapezoid (same API)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise RuntimeError("Need numpy.trapezoid or numpy.trapz for coordination number")
    cn = 2.0 * np.pi * rho_2d * _trapz(integrand, r_m)
    return cn


# ---------- plotting + saving ----------
def save_rdf(out_dir, prefix, r, g_raw, g_norm, meta, composition_label=None):
    """Save RDF data and figure to out_dir with given prefix. Plot both raw and normalized g(r)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / prefix
    arr = np.column_stack([r, g_raw, g_norm])
    header = "\n".join(
        ["# r_nm  g_raw  g_norm"] + [f"# {k}: {v}" for k, v in meta.items()]
    )
    np.savetxt(str(p) + "_rdf.dat", arr, header=header)
    np.save(str(p) + "_r.npy", r)
    np.save(str(p) + "_g_raw.npy", g_raw)
    np.save(str(p) + "_g_norm.npy", g_norm)
    
    title = composition_label if composition_label else prefix
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.plot(r, g_norm, label="g(r)", color='#0066CC', linewidth=2.0, linestyle='-', alpha=0.9)
    
    r_peak = meta.get('r_peak_nm', None)
    r_min = meta.get('r_first_min_nm', None)
    peak_g_norm = None
    peak_r = None
    if r_peak is not None:
        peak_mask = np.abs(r - r_peak) < 0.1
        if peak_mask.any():
            idx_peak = np.where(peak_mask)[0][np.argmax(g_norm[peak_mask])]
            peak_r = r[idx_peak]
            peak_g_norm = g_norm[idx_peak]
            ax.axvline(peak_r, color='#CC0000', linestyle='--', linewidth=2.0, alpha=0.8,
                      label=f'peak at r={peak_r:.2f} nm, g(r)={peak_g_norm:.3f}')
    
    # Coordination number (first shell) using raw g(r) — place lower-left so not under legend
    rho_2d = meta.get('rho_2d_per_nm2', None)
    cn = np.nan
    if r_min is not None and rho_2d is not None and rho_2d > 0:
        cn = coordination_number_first_shell(r, g_raw, r_min, rho_2d)
        if not np.isnan(cn):
            ax.text(0.02, 0.02, f"1st shell coord. no. = {cn:.2f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xlim(0, min(10.0, r.max()))
    y_min = g_norm.min()
    y_max = g_norm.max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.2 * y_range)
    
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("r (nm)", fontsize=11)
    ax.set_ylabel("g(r) (2D lateral)", fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    n_frames = meta.get('n_frames', 0)
    n_particles = meta.get('n_particles', 'unknown')
    if r_peak is not None and peak_g_norm is not None:
        cn_str = f", coord. no. = {cn:.2f}" if not np.isnan(cn) else ""
        print(f"  RDF peak: r={r_peak:.3f} nm, g(r)={peak_g_norm:.3f}{cn_str}")
    else:
        print(f"  RDF peak: not found")
    if not np.isnan(cn):
        print(f"  First-shell coordination number: {cn:.3f}")
    print(f"  RDF range: [{g_raw.min():.3f}, {g_raw.max():.3f}], frames={n_frames}, particles={n_particles}")
    
    if n_frames < 100:
        print(f"  Note: Low frame count ({n_frames}) may contribute to noise in RDF")
    if isinstance(n_particles, int) and n_particles < 100:
        print(f"  Note: Low particle count ({n_particles}) may contribute to noise in RDF")
    if n_frames >= 200 and isinstance(n_particles, int) and n_particles >= 200:
        print(f"  RDF quality: Good statistics ({n_frames} frames, {n_particles} particles) → smooth RDF expected")
    
    fig.tight_layout()
    fig.savefig(str(p) + "_rdf.png")
    plt.close(fig)


def _plot_neighbors_histogram(ax, mean_neighbors, bins=40, bar_color='#2E86AB', label=None):
    """Professional histogram with consistent color and clean styling."""
    counts, edges = np.histogram(mean_neighbors, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    for i in range(len(counts)):
        if counts[i] <= 0:
            continue
        ax.bar(
            centers[i], counts[i], width=widths[i] * 0.95, align="center",
            color=bar_color, edgecolor="white", linewidth=0.5,
        )
    ax.set_xlim(edges[0], edges[-1])
    y_max = max(counts) * 1.05 if counts.max() > 0 else 1
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Mean neighbors in first shell", fontsize=11)
    ax.set_ylabel("Count (lipids)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    y_max_rounded = int(np.ceil(y_max / 10) * 10)
    ax.set_yticks(np.arange(0, y_max_rounded + 1, 10))


def save_neighbors(out_dir, prefix, resids, resnames, mean_neighbors, meta, bins=40, composition_label=None):
    """Save per-lipid neighbor data and histogram to out_dir with given prefix."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / prefix
    np.save(str(p) + "_neighbors.npy", mean_neighbors)
    out = np.column_stack([resids, resnames, mean_neighbors])
    header = "\n".join(
        ["# resid  resname  mean_neighbors_within_first_shell"]
        + [f"# {k}: {v}" for k, v in meta.items()]
    )
    np.savetxt(str(p) + "_neighbors.csv", out, fmt="%s", delimiter=",", header=header)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    _plot_neighbors_histogram(ax, mean_neighbors, bins=bins)
    title = composition_label if composition_label else prefix
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(p) + "_neighbors_hist.png")
    plt.close(fig)


def run_analysis_for_folder(
    folder,
    stride=1,
    rmax=10.0,
    dr=0.01,
    last_us=1.0,
    tail_norm_min=3.0,
):
    """Run full phase RDF + neighbor analysis for a single composition folder.
    Returns a dict of data for overlay plotting, or None on failure."""
    folder = Path(folder)
    top, traj = pick_topology_and_traj(folder)
    u = mda.Universe(top, traj)
    tail_beads = f"(resname {TAIL_RESNAMES}) and (name {TAIL_BEAD_NAMES})"
    global_sel = tail_beads

    t_end = u.trajectory[-1].time
    t_start = max(u.trajectory[0].time, t_end - last_us * 1e6)

    n_strided = (len(u.trajectory) + stride - 1) // stride
    n_frames_in_window = 0
    count_pbar = tqdm(total=n_strided, unit="frame", desc=" ", dynamic_ncols=True)
    for ts in u.trajectory[::stride]:
        if t_start <= ts.time <= t_end:
            n_frames_in_window += 1
        count_pbar.update(1)
    count_pbar.close()
    print(f"  → {n_frames_in_window} frames in time window [{t_start/1e6:.2f}, {t_end/1e6:.2f}] µs\n")

    r, g_raw, counts, n_frames_rdf, (t0, t1) = compute_global_rdf_2d(
        u, global_sel, r_max=rmax, dr=dr, stride=stride,
        t_start=t_start, t_end=t_end, n_expected=n_frames_in_window,
    )
    g_norm = normalize_tail(r, g_raw, r_norm_min=tail_norm_min)
    r_peak, r_min = find_first_minimum(r, g_norm)
    
    n_particles = len(u.select_atoms(global_sel))
    Lx, Ly, Lz = u.trajectory[-1].dimensions[:3]
    scale = detect_scale_from_box(Lz)
    area = (Lx * scale) * (Ly * scale)
    rho_2d = n_particles / area

    rdf_meta = dict(
        top=top, traj=traj, selection=global_sel, stride=stride,
        rmax_nm=rmax, dr_nm=dr, t_start_ps=t0, t_end_ps=t1,
        n_frames=n_frames_rdf, n_particles=n_particles,
        tail_norm_min_nm=tail_norm_min,
        r_peak_nm=r_peak, r_first_min_nm=r_min,
        rho_2d_per_nm2=rho_2d,
    )

    composition_label = folder.name
    out_dir = folder / "analysis_out"
    prefix = folder.name
    save_rdf(out_dir, prefix, r, g_raw, g_norm, rdf_meta, composition_label=composition_label)

    resids, resnames, mean_neigh, n_frames_shell, (s0, s1) = per_lipid_first_shell(
        u, tail_sel=tail_beads, r_cut=r_min, stride=stride,
        t_start=t_start, t_end=t_end, n_expected=n_frames_in_window,
    )
    shell_meta = dict(
        top=top, traj=traj, tail_sel=tail_beads, stride=stride,
        t_start_ps=s0, t_end_ps=s1, n_frames=n_frames_shell,
        r_cut_first_shell_nm=r_min,
    )
    save_neighbors(out_dir, prefix, resids, resnames, mean_neigh, shell_meta, composition_label=composition_label)

    print(f"[OK] Global RDF saved: {out_dir / prefix}_rdf.png / .dat")
    print(f"[OK] First peak at r={r_peak:.3f} nm, first minimum r_min={r_min:.3f} nm")
    print(f"[OK] Per-lipid neighbor data saved: {out_dir / prefix}_neighbors.csv / .npy")
    print(f"[OK] Histogram saved: {out_dir / prefix}_neighbors_hist.png")

    # Data for overlay plots
    peak_mask = np.abs(r - r_peak) < 0.1 if r_peak is not None else np.zeros(len(r), dtype=bool)
    peak_g_raw = float(g_raw[peak_mask].max()) if peak_mask.any() else None
    return dict(
        composition=composition_label,
        r=r, g_raw=g_raw, g_norm=g_norm,
        r_peak=r_peak, r_min=r_min, peak_g_raw=peak_g_raw,
        rho_2d=rho_2d, rdf_meta=rdf_meta,
        mean_neighbors=mean_neigh,
    )


# ---------- overlay plots (multiple compositions) ----------
OVERLAY_COLORS = ['#0066CC', '#CC6600', '#2E7D32', '#6A1B9A', '#C62828', '#00838F']


def _sort_results_by_species(results_list, order_by_species):
    """Sort results by species value ascending (bottom = lowest, top = highest). Compositions without the species go last."""
    if not order_by_species or not results_list:
        return results_list
    def key(res):
        v = get_species_value(res["composition"], order_by_species)
        return (v is None, v if v is not None else np.inf)
    return sorted(results_list, key=key)


def save_overlay_rdf(results_list, out_path, order_by_species=None, vertical_offset=0.05):
    """Save one figure with all normalized RDFs overlaid, stacked with vertical_offset between curves.
    If order_by_species is set (e.g. 'CHOL'), curves are ordered ascending by that species so the topmost line = highest value."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _sort_results_by_species(results_list, order_by_species)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    colors = OVERLAY_COLORS[: len(ordered)]
    for i, res in enumerate(ordered):
        c = colors[i % len(OVERLAY_COLORS)]
        lab = res["composition"]
        r, g_norm = res["r"], res["g_norm"]
        y = g_norm + i * vertical_offset
        ax.plot(r, y, label=lab, color=c, linewidth=1.0, linestyle='-', alpha=0.9)
    ax.set_xlim(0, min(10.0, max(res["r"].max() for res in ordered)))
    ax.set_xlabel("r (nm)", fontsize=11)
    ax.set_ylabel("RDF", fontsize=11)
    ax.set_title("Combined RDF", fontsize=11)
    ax.legend(fontsize=8, loc='upper right', ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_overlay_histogram(results_list, out_path, bins=40, order_by_species=None, vertical_offset=0.05):
    """Save one figure with all neighbor-count histograms overlaid, stacked with vertical_offset between curves.
    If order_by_species is set, histograms are ordered ascending by that species (topmost = highest value)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _sort_results_by_species(results_list, order_by_species)
    all_vals = np.concatenate([res["mean_neighbors"] for res in ordered])
    lo, hi = all_vals.min(), all_vals.max()
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    colors = OVERLAY_COLORS[: len(ordered)]
    max_count = 0
    for i, res in enumerate(ordered):
        mean_n = res["mean_neighbors"]
        counts, _ = np.histogram(mean_n, bins=edges)
        max_count = max(max_count, counts.max())
    # Stack: each histogram baseline shifted up by vertical_offset * (max_count) so curves don't overlap
    offset_step = vertical_offset * max_count if max_count > 0 else 1.0
    for i, res in enumerate(ordered):
        mean_n = res["mean_neighbors"]
        counts, _ = np.histogram(mean_n, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1] - edges[0]) * 0.85 / len(ordered)
        x_offset = (i - 0.5 * (len(ordered) - 1)) * widths * 0.3
        y_baseline = i * offset_step
        c = colors[i % len(OVERLAY_COLORS)]
        ax.bar(centers + x_offset, counts, width=widths, bottom=y_baseline, align='center',
               label=res["composition"], color=c, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, len(ordered) * offset_step + max_count * 1.15)
    ax.set_xlabel("Mean neighbors in first shell", fontsize=11)
    ax.set_ylabel("Count (lipids)", fontsize=11)
    ax.set_title("Combined first-shell neighbors", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Phase RDF and first-shell neighbor analysis for composition folders."
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="Root directory (one path) or list of composition folder paths. "
             "E.g. 'Production_out' or 'Production_out/POPC_50_POPS_0_CHOL_50 Production_out/POPC_60_POPS_0_CHOL_40'"
    )
    ap.add_argument("pattern", nargs="?", default=None,
                    help="Optional fnmatch pattern when one root path is given (e.g. POPC_50_*)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        help="Maximum distance (nm) for RDF computation and histogram bins. Pairs beyond rmax are not counted (default 10.0).",
    )
    ap.add_argument(
        "--dr",
        type=float,
        default=0.01,
        help="Bin width (nm) for the RDF histogram. Smaller dr = finer resolution, more bins (default 0.01).",
    )
    ap.add_argument("--last_us", type=float, default=1.0)
    ap.add_argument(
        "--tail_norm_min",
        type=float,
        default=3.0,
        help="Minimum r (nm) from which g(r) is averaged to normalize the RDF. The mean g(r) for r >= tail_norm_min is used as the baseline so normalized g(r) -> 1 at large r (default 3.0).",
    )
    ap.add_argument(
        "--order-by",
        type=str,
        default=None,
        metavar="SPECIES",
        help="Order overlay curves by this species (ascending); topmost line = highest value. E.g. --order-by CHOL",
    )
    ap.add_argument(
        "--overlay-offset",
        type=float,
        default=0.15,
        help="Vertical offset between stacked overlay curves (default 0.05).",
    )
    args = ap.parse_args()

    raw_paths = args.paths
    paths = [Path(p).resolve() for p in raw_paths]

    def has_composition_like(name):
        return any(p.search(name) for p in _COMPOSITION_PATTERNS)

    # Backward compat: "root pattern" (e.g. Production_out POPC_56_POPS_14_CHOL_30) → first is root, second is pattern
    if len(paths) == 2 and paths[0].is_dir() and (not paths[1].exists() or not paths[1].is_dir()):
        root = paths[0]
        pattern = raw_paths[1]
        try:
            candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            candidates = []
        candidates = [p for p in candidates if has_composition_like(p.name) and fnmatch.fnmatch(p.name, pattern)]
        overlay_root = root
    # Single path that is a directory → discover composition subfolders, or treat as single composition folder
    elif len(paths) == 1 and paths[0].is_dir():
        root = paths[0]
        try:
            candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            candidates = []
        candidates = [p for p in candidates if has_composition_like(p.name)]
        if args.pattern:
            candidates = [p for p in candidates if fnmatch.fnmatch(p.name, args.pattern)]
        # If nothing found and the directory itself looks like a composition folder, analyze it
        if not candidates and has_composition_like(root.name):
            candidates = [root]
        overlay_root = root
    # Multiple paths → treat each as a composition folder (must exist and be a directory)
    else:
        candidates = []
        for p in paths:
            if not p.exists():
                print(f"Warning: path does not exist: {p}", file=sys.stderr)
                continue
            if p.is_dir():
                candidates.append(p)
            else:
                print(f"Warning: not a directory, skipping: {p}", file=sys.stderr)
        overlay_root = candidates[0].parent if candidates else Path.cwd()

    if not candidates:
        print("No composition folders to analyze.")
        sys.exit(0)

    print(f"Found {len(candidates)} folder(s) to analyze: {[p.name for p in candidates]}\n")

    results_for_overlay = []
    for folder in sorted(candidates):
        try:
            print(f"Analyzing: {folder.name}")
            data = run_analysis_for_folder(
                folder,
                stride=args.stride,
                rmax=args.rmax,
                dr=args.dr,
                last_us=args.last_us,
                tail_norm_min=args.tail_norm_min,
            )
            if data is not None:
                results_for_overlay.append(data)
        except FileNotFoundError as e:
            print(f"Skipping {folder.name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            raise

    # Save overlay figures when we have more than one composition (one curve per composition)
    if len(results_for_overlay) > 1:
        out_overlay = overlay_root / "analysis_out"
        out_overlay.mkdir(parents=True, exist_ok=True)
        save_overlay_rdf(
            results_for_overlay,
            out_overlay / "overlay_rdf.png",
            order_by_species=args.order_by,
            vertical_offset=args.overlay_offset,
        )
        save_overlay_histogram(
            results_for_overlay,
            out_overlay / "overlay_neighbors_hist.png",
            order_by_species=args.order_by,
            vertical_offset=args.overlay_offset,
        )
        print(f"\n[OK] Overlay RDF saved: {out_overlay / 'overlay_rdf.png'}")
        print(f"[OK] Overlay neighbors histogram saved: {out_overlay / 'overlay_neighbors_hist.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
