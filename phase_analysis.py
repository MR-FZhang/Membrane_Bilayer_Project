#!/usr/bin/env python3
"""
phase_analysis.py

Probe-cell grid phase separation analysis for lipid bilayers (POPC/POPS/CHOL or DOPC/DPPC/CHOL etc.).
Based on: "Detecting Solution Phase Behavior in Particle Simulations" (McDonagh et al.).
Divides the XY plane into N×N cells, counts headgroup beads per cell per frame (per leaflet),
builds histograms and compares to binomial (ideal mixing) reference.

Usage:
  python phase_analysis.py <path_or_pattern> [pattern]
  python phase_analysis.py <folder1> [folder2 ...]

Examples:
  python phase_analysis.py Production_out/POPC_30_POPS_40_CHOL_30
  python phase_analysis.py "Production_out/POPC_30*"
  python phase_analysis.py Production_out POPC_30*
  # Equilibrated segment only: last 2000 frames, or from frame 3000 onward
  python phase_analysis.py Production_out/POPC_30_POPS_40_CHOL_30 --last-n-frames 2000
  python phase_analysis.py Production_out/CHOL_10_DOPC_30_DPPC_60 --start-frame 3000

Output: <folder>/analysis_out/phase_analysis_<folder_name>.png (per folder)
"""

import sys
import re
import fnmatch
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable parameters (edit these at the top of the script)
# ---------------------------------------------------------------------------
GRO_FILENAME = "step7_lipids_ions.gro"
XTC_FILENAME = "step7_lipids_ions.xtc"
N_GRID = 7  # number of cells per side (8×8 = 64 cells)
HEADGROUP_BEADS = {
    "POPC": "PO4",
    "POPS": "PO4",
    "DOPC": "PO4",
    "DPPC": "PO4",
    "CHOL": "ROH",
}
# Column order in plots (only species present are shown)
SPECIES_PLOT_ORDER = ["POPC", "POPS", "DOPC", "DPPC", "CHOL"]
BETA_THRESHOLD = 5 / 9  # bimodal coefficient threshold (for future use)
STRIDE = 1  # analyse every frame (set to 5 or 10 to speed up)
START_FRAME = 0  # first frame to use (0 = from start); use --start-frame or --last-n-frames to override
ANALYSIS_OUT_DIR = "analysis_out"

# ---------------------------------------------------------------------------
# Step 1 — Environment and imports
# ---------------------------------------------------------------------------
def _check_imports():
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import MDAnalysis
    except ImportError:
        missing.append("MDAnalysis")
    if missing:
        print("Missing required libraries:", ", ".join(missing), file=sys.stderr)
        print("Install with: pip install numpy scipy matplotlib MDAnalysis", file=sys.stderr)
        sys.exit(1)


_check_imports()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import MDAnalysis as mda

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def pick_topology_and_traj(folder):
    """Find topology and trajectory in folder. Prefer .gro+.xtc, then .tpr+.xtc."""
    folder = Path(folder)
    # User config: gro + xtc first
    gpath = folder / GRO_FILENAME
    xpath = folder / XTC_FILENAME
    if gpath.exists() and xpath.exists():
        return str(gpath), str(xpath)
    # Fallbacks
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
        f"Tried {GRO_FILENAME}/{XTC_FILENAME} and common .tpr/.gro/.xtc names."
    )


def detect_scale(Lz):
    """Box in Å if Lz > 50, else nm."""
    return 0.1 if Lz > 50 else 1.0


def wrap_xy(xy, Lx, Ly):
    """Wrap xy into [0, Lx) x [0, Ly)."""
    out = np.empty_like(xy)
    out[:, 0] = np.mod(xy[:, 0], Lx)
    out[:, 1] = np.mod(xy[:, 1], Ly)
    return out


# ---------------------------------------------------------------------------
# Step 2 — Load and verify
# ---------------------------------------------------------------------------
def load_and_verify(top_path, traj_path):
    """Load universe and print summary."""
    u = mda.Universe(top_path, traj_path)
    ts = u.trajectory[0]
    Lx, Ly, Lz = ts.dimensions[:3]
    scale = detect_scale(Lz)
    n_atoms = u.atoms.n_atoms
    resnames = np.unique(u.atoms.resnames)
    bead_names = np.unique(u.atoms.names)
    n_frames = len(u.trajectory)

    print("--- File loading and verification ---")
    print(f"  Topology: {top_path}")
    print(f"  Trajectory: {traj_path}")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Residue names: {list(resnames)}")
    print(f"  Unique bead names: {list(bead_names)}")
    print(f"  Box (native): Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    print(f"  Scale (to nm): {scale}")
    print(f"  Frames in trajectory: {n_frames}")
    return u, scale


# ---------------------------------------------------------------------------
# Step 3 — Headgroup bead selection (by resname + bead name)
# ---------------------------------------------------------------------------
def select_headgroup_atoms(u, species_present):
    """
    For each species in species_present, select atoms matching resname AND bead name.
    Returns dict: resname -> AtomGroup of head beads.
    """
    head = {}
    for resname in species_present:
        bead_name = HEADGROUP_BEADS.get(resname)
        if not bead_name:
            continue
        sel = u.select_atoms(f"resname {resname} and name {bead_name}")
        if len(sel) > 0:
            head[resname] = sel
    return head


# ---------------------------------------------------------------------------
# Step 4 — Leaflet assignment at frame 0
# ---------------------------------------------------------------------------
def assign_leaflets(u, head_groups, scale, stride=1):
    """
    At frame 0, compute mean Z of all selected head beads; assign each bead to
    upper (Z > mean) or lower (Z <= mean). Return leaflet index per residue.
    Returns: (leaflet_per_residue, counts_per_species_leaflet).
    leaflet_per_residue: dict resname -> array of 0=lower, 1=upper per bead index in that group.
    """
    u.trajectory[0]
    all_z = []
    resname_per_idx = []  # which resname each position belongs to
    idx_in_resname = []   # index within that resname's AtomGroup
    for resname, ag in head_groups.items():
        pos = ag.positions * scale
        z = pos[:, 2]
        for i in range(len(z)):
            all_z.append(z[i])
            resname_per_idx.append(resname)
            idx_in_resname.append(i)
    all_z = np.array(all_z)
    z_mean = np.mean(all_z)
    upper = (all_z > z_mean).astype(int)  # 1 = upper, 0 = lower

    leaflet_per_residue = {}
    for resname in head_groups:
        mask = [r == resname for r in resname_per_idx]
        leaflet_per_residue[resname] = upper[np.array(mask)]

    counts = {}
    for resname, ag in head_groups.items():
        lt = leaflet_per_residue[resname]
        n_upper = int(np.sum(lt == 1))
        n_lower = int(np.sum(lt == 0))
        counts[resname] = {"upper": n_upper, "lower": n_lower}
        print(f"  {resname}: upper={n_upper}, lower={n_lower}")

    return leaflet_per_residue, counts


# ---------------------------------------------------------------------------
# Step 5 — Probe cell grid: count per cell per frame per species/leaflet
# ---------------------------------------------------------------------------
def run_probe_cell_counts(u, head_groups, leaflet_per_residue, scale, stride=1, start_frame=0):
    """
    For each frame (from start_frame onward, stride), each leaflet, each species: count head beads
    in each of N_GRID×N_GRID cells. Return dict (resname, leaflet) -> array of shape (n_frames_used, N_GRID*N_GRID).
    start_frame: first frame index to use (0-based); skip frames before this (e.g. non-equilibrated).
    """
    n_cells = N_GRID * N_GRID
    species_leaflet_data = {}
    for resname in head_groups:
        for leaf in ("upper", "lower"):
            species_leaflet_data[(resname, leaf)] = []

    n_traj = len(u.trajectory)
    start_frame = max(0, min(start_frame, n_traj - 1))
    n_frames_in_range = n_traj - start_frame
    n_to_process = max(0, (n_frames_in_range + stride - 1) // stride)
    frames_used = 0
    for ts in tqdm(
        u.trajectory[start_frame::stride],
        desc="Frames",
        unit="frame",
        total=n_to_process,
    ):
        Lx, Ly, Lz = ts.dimensions[:3] * scale
        dx = Lx / N_GRID
        dy = Ly / N_GRID

        for resname, ag in head_groups.items():
            pos = ag.positions * scale
            xy = pos[:, :2]
            xy = wrap_xy(xy, Lx, Ly)
            leaflet = leaflet_per_residue[resname]  # 0=lower, 1=upper

            for leaf_id, leaf_name in enumerate(("lower", "upper")):
                mask = leaflet == leaf_id
                xy_leaf = xy[mask]
                counts_flat = np.zeros(n_cells, dtype=np.int32)
                if len(xy_leaf) == 0:
                    species_leaflet_data[(resname, leaf_name)].append(counts_flat.copy())
                    continue
                # cell index: i = ix + iy * N_GRID, with ix, iy in [0, N_GRID)
                ix = np.clip(np.floor(xy_leaf[:, 0] / dx).astype(int), 0, N_GRID - 1)
                iy = np.clip(np.floor(xy_leaf[:, 1] / dy).astype(int), 0, N_GRID - 1)
                cell_idx = ix + iy * N_GRID
                np.add.at(counts_flat, cell_idx, 1)
                species_leaflet_data[(resname, leaf_name)].append(counts_flat)

        frames_used += 1

    # Stack into (n_frames, n_cells) per (resname, leaflet)
    out = {}
    for key in species_leaflet_data:
        out[key] = np.array(species_leaflet_data[key], dtype=np.int32)
    return out, frames_used


# ---------------------------------------------------------------------------
# Step 6 — Flatten and build normalised histogram
# ---------------------------------------------------------------------------
def build_histogram(counts_2d):
    """
    counts_2d shape (n_frames, n_cells). Flatten to 1D, then normalised histogram.
    X-axis of the histogram = count per cell (0, 1, 2, ...); many of the 64 cells
    (across frames) can share the same count, so the number of histogram bars
    is the number of distinct counts observed, not 64.
    """
    flat = counts_2d.ravel()
    max_count = int(flat.max()) if len(flat) > 0 else 0
    bins = np.arange(max_count + 2) - 0.5
    hist, _ = np.histogram(flat, bins=bins)
    hist = hist.astype(float) / (hist.sum() + 1e-20)
    bin_centers = np.arange(max_count + 1)
    return bin_centers, hist[: len(bin_centers)]


# ---------------------------------------------------------------------------
# Step 7 — Binomial reference B(x; n, p), p = 1/N_cells
# ---------------------------------------------------------------------------
def binomial_reference(n_molecules, n_cells):
    """
    Ideal-mixing reference: B(x; n, p) with
      n = total molecules of this species in this leaflet,
      p = 1/N_cells (probability a molecule is in any one cell under ideal mixing),
      x = count per cell (0, 1, ..., n).
    Computed via scipy.stats.binom.pmf(x, n, p) at each integer x.
    """
    x = np.arange(n_molecules + 1, dtype=float)
    p = 1.0 / n_cells
    pmf = scipy_stats.binom.pmf(x, n_molecules, p)
    return x, pmf


# ---------------------------------------------------------------------------
# Step 8 — Plot: 2 rows (upper, lower) × N columns (species present)
# ---------------------------------------------------------------------------
def plot_histograms_and_binomial(
    species_leaflet_histograms,
    species_leaflet_binomial,
    species_leaflet_counts,
    composition_label,
    out_path,
):
    """
    species_leaflet_histograms: dict (resname, leaflet) -> (bin_centers, prob)
    species_leaflet_binomial: dict (resname, leaflet) -> (x, pmf)
    species_leaflet_counts: dict (resname, leaflet) -> n_molecules (for title)
    """
    species_order = [s for s in SPECIES_PLOT_ORDER if any((s, leaf) in species_leaflet_histograms for leaf in ("upper", "lower"))]
    if not species_order:
        print("No species data to plot.")
        return
    leaflets = ["upper", "lower"]
    n_cols = len(species_order)
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    for row, leaf in enumerate(leaflets):
        for col, resname in enumerate(species_order):
            ax = axes[row, col]
            key = (resname, leaf)
            if key not in species_leaflet_histograms:
                ax.set_visible(False)
                continue
            x_emp, p_emp = species_leaflet_histograms[key]
            x_bin, p_bin = species_leaflet_binomial[key]
            n_mol = species_leaflet_counts.get(key, 0)
            ax.bar(x_emp, p_emp, color="steelblue", alpha=0.8, width=0.7, label="Data")
            # Binomial as step function (blocky, paper-style) rather than smooth line
            ax.step(
                x_bin, p_bin, where="post", color="red", lw=2, label="Binomial (ideal)"
            )
            ax.set_xlabel("Count per cell")
            ax.set_ylabel("Probability")
            ax.set_title(f"Phase analysis {resname} {leaf}\n(n={n_mol})")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(0, None)
            # Narrow x-axis so distribution shape is visible (data is ~0 to mean+few std, not 0 to n_mol)
            max_emp = int(x_emp[-1]) if len(x_emp) > 0 else 0
            visible_bin = x_bin[p_bin > 1e-6]
            max_bin_visible = int(visible_bin[-1]) if len(visible_bin) > 0 else 0
            x_max = min(50, max(max_emp, max_bin_visible) + 5)
            x_max = max(x_max, 15)
            # Left margin: gap between y-axis and first bar (paper-style)
            x_left = -4
            ax.set_xlim(x_left, x_max)
    fig.suptitle(f"Probe-cell occupancy — {composition_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_composition(folder_name):
    """Extract composition string from folder name for plot title."""
    m = re.search(r"POPC_(\d+)_POPS_(\d+)_CHOL_(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"POPC {m.group(1)} / POPS {m.group(2)} / CHOL {m.group(3)}"
    m = re.search(r"DOPC_(\d+)_DPPC_(\d+)_CHOL_(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"DOPC {m.group(1)} / DPPC {m.group(2)} / CHOL {m.group(3)}"
    m = re.search(r"DOPC_(\d+)_DPPC_(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"DOPC {m.group(1)} / DPPC {m.group(2)}"
    return folder_name


# Patterns that identify a composition folder (POPC/POPS/CHOL or DOPC/DPPC/CHOL etc.)
_COMPOSITION_PATTERNS = [
    re.compile(r"POPC_\d+_POPS_\d+_CHOL_\d+", re.IGNORECASE),
    re.compile(r"DOPC_\d+_DPPC_\d+_CHOL_\d+", re.IGNORECASE),
    re.compile(r"DOPC_\d+_DPPC_\d+", re.IGNORECASE),
]


def _has_composition(name):
    """True if folder name looks like a known composition (POPC/POPS/CHOL or DOPC/DPPC/CHOL etc.)."""
    return any(p.search(name) for p in _COMPOSITION_PATTERNS)


def resolve_folders(path_args, pattern_arg=None):
    """
    Resolve command-line path/pattern into a list of composition folders.
    - path_args: list of strings (one or more).
    - One path_arg + pattern_arg: path is parent dir, pattern is fnmatch (e.g. Production_out POPC_30*).
    - Multiple path_args: if first is a dir and second is a subdir name, treat as root + list of subdirs
      (shell-expanded "Production_out POPC_30*"); else treat each as a full path.
    - One path_arg, no pattern: if path contains * or ?, glob in parent; else single folder.
    """
    if not path_args:
        return []

    # If two args and second looks like a pattern (contains * or ?), treat as (root, pattern)
    if len(path_args) == 2 and ("*" in path_args[1] or "?" in path_args[1]):
        pattern_arg = path_args[1].strip()
        path_args = [path_args[0].strip()]
    elif pattern_arg is not None and path_args:
        path_args = [path_args[0].strip()]

    path_arg = path_args[0].strip()
    if pattern_arg is not None:
        # Two args: path + pattern (e.g. Production_out POPC_30*)
        root = Path(path_arg).resolve()
        if not root.is_dir():
            return []
        pattern = pattern_arg.strip()
        candidates = [p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern)]
        candidates = [p for p in candidates if _has_composition(p.name)]
        return sorted(candidates, key=lambda p: p.name)

    if len(path_args) > 1:
        # Multiple args: either "Production_out POPC_30_POPS_40_CHOL_30 ..." (root + subdirs) or full paths
        root = Path(path_arg).resolve()
        rest = [p.strip() for p in path_args[1:] if p.strip()]
        if root.is_dir() and rest:
            # Check if rest look like subdir names (no path sep) and exist under root
            subdirs = [root / n for n in rest if "/" not in n and "\\" not in n]
            if subdirs and subdirs[0].is_dir():
                return sorted([p for p in subdirs if p.is_dir()], key=lambda p: p.name)
        # Treat each as full path (e.g. Production_out/POPC_30_... Production_out/POPC_30_...)
        folders = [Path(p).resolve() for p in path_args if p.strip()]
        return [f for f in folders if f.is_dir()]

    # Single path (may contain glob)
    path_str = Path(path_arg)
    if "*" in path_arg or "?" in path_arg:
        root = path_str.parent.resolve()
        if not root.is_dir():
            return []
        pattern = path_str.name
        candidates = [p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern)]
        candidates = [p for p in candidates if _has_composition(p.name)]
        return sorted(candidates, key=lambda p: p.name)
    folder = Path(path_arg).resolve()
    if folder.is_dir():
        return [folder]
    return []


def run_one_folder(folder, stride, start_frame=0, last_n_frames=None, verbose=True):
    """
    Run full phase analysis for one composition folder.
    Returns True on success, False on skip/error.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        if verbose:
            print(f"Skipping (not a directory): {folder}", file=sys.stderr)
        return False
    try:
        top_path, traj_path = pick_topology_and_traj(folder)
    except FileNotFoundError as e:
        if verbose:
            print(f"Skipping {folder.name}: {e}")
        return False

    if verbose:
        print(f"\n{'='*60}\nAnalyzing: {folder.name}\n{'='*60}")
    u, scale = load_and_verify(top_path, traj_path)

    all_resnames = set(u.atoms.resnames)
    species_present = [s for s in HEADGROUP_BEADS if s in all_resnames]
    if not species_present:
        if verbose:
            print("No known lipid species (see HEADGROUP_BEADS) found in system.", file=sys.stderr)
        return False

    if verbose:
        print("\n--- Headgroup bead selection ---")
        print("  Species present:", species_present)
    head_groups = select_headgroup_atoms(u, species_present)
    if not head_groups:
        if verbose:
            print("No headgroup beads found for any species.", file=sys.stderr)
        return False

    if verbose:
        print("\n--- Leaflet assignment (frame 0) ---")
    leaflet_per_residue, counts_per_species_leaflet = assign_leaflets(
        u, head_groups, scale
    )

    n_traj = len(u.trajectory)
    if last_n_frames is not None:
        effective_start = max(0, n_traj - last_n_frames)
    else:
        effective_start = min(max(0, start_frame), n_traj - 1) if n_traj else 0
    n_in_range = n_traj - effective_start
    n_analyze = max(0, (n_in_range + stride - 1) // stride)
    if verbose:
        print(f"\n--- Probe cell grid (counting per frame) ---")
        print(f"  Trajectory has {n_traj} frames; using from frame {effective_start} (stride={stride}) → analyzing {n_analyze} frames.")
    species_leaflet_data, n_frames = run_probe_cell_counts(
        u, head_groups, leaflet_per_residue, scale, stride=stride, start_frame=effective_start
    )
    if verbose:
        print(f"  Frames analyzed: {n_frames}, cells: {N_GRID}×{N_GRID} = {N_GRID*N_GRID}")

    species_leaflet_histograms = {}
    species_leaflet_binomial = {}
    species_leaflet_n_molecules = {}
    n_cells = N_GRID * N_GRID
    for resname in species_present:
        for leaf in ("upper", "lower"):
            key = (resname, leaf)
            arr = species_leaflet_data.get(key)
            if arr is None or arr.size == 0:
                continue
            bin_centers, hist = build_histogram(arr)
            species_leaflet_histograms[key] = (bin_centers, hist)
            n_mol = counts_per_species_leaflet[resname][leaf]
            x_bin, p_bin = binomial_reference(n_mol, n_cells)
            species_leaflet_binomial[key] = (x_bin, p_bin)
            species_leaflet_n_molecules[key] = n_mol

    out_dir = folder / ANALYSIS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    composition_label = parse_composition(folder.name)
    out_name = f"phase_analysis_{folder.name}.png"
    out_path = out_dir / out_name

    plot_histograms_and_binomial(
        species_leaflet_histograms,
        species_leaflet_binomial,
        species_leaflet_n_molecules,
        composition_label,
        out_path,
    )
    if verbose:
        print("Done.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Probe-cell phase analysis for one or more composition folders."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="*",
        help="Folder path(s), or glob (e.g. Production_out/POPC_30*), or root dir if pattern given",
    )
    parser.add_argument(
        "pattern",
        type=str,
        nargs="?",
        default=None,
        help="Optional fnmatch pattern (e.g. POPC_30*) when path is a single directory",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help=f"Analyse every Nth frame (default {STRIDE})",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=START_FRAME,
        metavar="N",
        help="First frame index to use (0-based). Skip frames before this (e.g. non-equilibrated). Default 0.",
    )
    parser.add_argument(
        "--last-n-frames",
        type=int,
        default=None,
        metavar="N",
        help="Use only the last N frames (overrides --start-frame). E.g. 1000 or 2000 for equilibrated segment.",
    )
    args = parser.parse_args()
    path_args = args.path if args.path else []
    pattern_arg = args.pattern
    stride = args.stride
    start_frame = args.start_frame
    last_n_frames = args.last_n_frames

    if not path_args:
        print("Usage: phase_analysis.py <path_or_pattern> [pattern]", file=sys.stderr)
        print("Examples: phase_analysis.py Production_out/POPC_30_POPS_40_CHOL_30", file=sys.stderr)
        print('         phase_analysis.py "Production_out/POPC_30*"', file=sys.stderr)
        print("         phase_analysis.py Production_out POPC_30*", file=sys.stderr)
        sys.exit(1)

    folders = resolve_folders(path_args, pattern_arg)

    if not folders:
        print("No matching composition folders found.", file=sys.stderr)
        print("Examples: phase_analysis.py Production_out/POPC_30_POPS_40_CHOL_30", file=sys.stderr)
        print('         phase_analysis.py "Production_out/POPC_30*"', file=sys.stderr)
        print("         phase_analysis.py Production_out POPC_30*", file=sys.stderr)
        sys.exit(1)

    if len(folders) > 1:
        print(f"Found {len(folders)} folder(s). Analyzing each.\n")
    ok = 0
    for folder in folders:
        if run_one_folder(folder, stride, start_frame=start_frame, last_n_frames=last_n_frames, verbose=True):
            ok += 1
    if len(folders) > 1:
        print(f"\nCompleted: {ok}/{len(folders)} folders.")


if __name__ == "__main__":
    main()
