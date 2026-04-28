"""
Ternary phase analysis: GMM on first-shell packing metric (and optional tail order).

METRIC AND UNITS (first-shell peak metric):
- What it is: For each lipid we build a distance histogram of neighbors (same leaflet)
  using reference beads (C2A, C2B, D2A, D2B, R1). Each bin counts how many neighbor
  pairs fall in that distance bin; we average over frames, then sum over bins in
  the "first shell" window.
- Units: The first-shell peak metric is in "average neighbor count (per frame)"
  summed over the first-shell window. So it's a dimensionless count (number of
  neighbor contributions, not physical length).
- Why histogram "Density" is high (e.g. 5000): We plot the distribution of this
  metric across lipids with density=True and bins=200. The y-axis is probability
  density = (fraction of lipids in bin) / (bin width in metric units). Because
  the metric range is small (~0.004), bin width is tiny, so density = fraction/0.00002
  can be thousands. So the "5000" is not number of lipids—it's probability per unit
  metric (1/count).

BIN WIDTH AND R RANGE:
- bin_width (default 0.01 nm): Width of each distance bin when building the
  RDF-like histogram. Smaller = finer resolution but noisier (spiky) curves.
- r_max (default 1.2 nm): Maximum distance for neighbor search.
- First-shell window: r in [0.3, 0.9] nm (hardcoded). Only bins in this range
  are summed to get the "first-shell peak metric". You can tune these in the code.
"""
import re
import fnmatch
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        return iterable

# -----------------------------
# Parameters (tune these if needed)
# -----------------------------
BIN_WIDTH_RDF = 0.01   # nm; larger = smoother RDF, less spiky
R_MAX = 1.2            # nm; max distance for neighbor search
PEAK_WINDOW_R = (0.3, 0.9)  # nm; first-shell window for summing neighbor count
HIST_BINS_METRIC = 200   # bins for histogram of the metric (density plot)
VALIDATION_RDF_REBIN = 0.05  # nm; rebin validation RDF curves for smoother plot (0 = no rebin)

# -----------------------------
# Helpers
# -----------------------------
def parse_composition(folder_name: str):
    # expects e.g. POPC_40_POPS_10_CHOL_50
    m = re.search(r"POPC_(\d+)_POPS_(\d+)_CHOL_(\d+)", folder_name)
    if not m:
        return None
    popc, pops, chol = map(int, m.groups())
    s = popc + pops + chol
    return (popc/s, pops/s, chol/s)

def get_last_1us_frames(u):
    # MDAnalysis uses whatever time is stored (often ps for GROMACS)
    # 1 microsecond = 1e6 ps
    t_end = u.trajectory[-1].time
    t_start = t_end - 1*1e6
    return [ts.frame for ts in u.trajectory if ts.time >= t_start]

def fit_gmm_thresholds(values, k_min=1, k_max=3):
    """
    Fit a GMM to the metric values and compute thresholds between components.
    Tries K = k_min .. k_max (default 1, 2, 3) and picks the K with lowest BIC.

    Interpretation (important):
    - K is chosen by BIC: "How many distinct groups in these metric values
      are statistically justified?" It is the number of *metric populations*,
      not a direct phase detector. Interpret as phases (e.g. Ld vs Lo) only
      if you validate that the groups differ in something physical (e.g. tail
      order, RDF, local packing).
    - K=1 → single population (no evidence for phase separation in this metric).
    - K=2 → often two local environments (e.g. Ld vs Lo).
    - K=3 → possibly three (e.g. Ld / Lo / gel-like); accept only if
      validation agrees.

    Returns:
        best_k, thresholds, (means, covs, weights), bic_by_k
    where bic_by_k is a dict mapping k -> BIC value for all tried k.
    """
    values = values.reshape(-1, 1)
    bic_by_k = {}

    best = None
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(values)
        bic = gmm.bic(values)
        bic_by_k[k] = bic
        if best is None or bic < best["bic"]:
            best = {"k": k, "gmm": gmm, "bic": bic}

    gmm = best["gmm"]
    means = gmm.means_.flatten()
    order = np.argsort(means)
    means = means[order]
    covs = gmm.covariances_.flatten()[order]
    weights = gmm.weights_.flatten()[order]

    # Intersection of neighboring Gaussians (1D) gives thresholds (K-1 of them)
    # Solve w1*N(m1,s1)=w2*N(m2,s2)
    thresholds = []
    for i in range(len(means) - 1):
        m1, m2 = means[i], means[i+1]
        s1, s2 = np.sqrt(covs[i]), np.sqrt(covs[i+1])
        w1, w2 = weights[i], weights[i+1]

        a = 1/(2*s2*s2) - 1/(2*s1*s1)
        b = m1/(s1*s1) - m2/(s2*s2)
        c = (m2*m2)/(2*s2*s2) - (m1*m1)/(2*s1*s1) + np.log((w2*s1)/(w1*s2))

        roots = np.roots([a, b, c])
        real_roots = [float(np.real(r)) for r in roots if np.isreal(r)]
        midpoint = (m1 + m2) / 2.0
        if len(real_roots) == 0:
            threshold = midpoint
        elif len(real_roots) == 1:
            threshold = real_roots[0]
        else:
            threshold = min(real_roots, key=lambda x: abs(x - midpoint))
        thresholds.append(threshold)

    return best["k"], thresholds, (means, covs, weights), bic_by_k

# -----------------------------
# Core: compute per-lipid first-shell peak metric
# -----------------------------
def compute_peak_metric_for_run(tpr_path, xtc_path, bin_width=None, r_max=None, max_frames=None, compute_order=False):
    """
    max_frames: if set, only process this many frames (for testing). 
                If None, process all frames in last 1 µs.
    compute_order: if True, also compute per-lipid S_chain order and return as 5th element.
    """
    if bin_width is None:
        bin_width = BIN_WIDTH_RDF
    if r_max is None:
        r_max = R_MAX
    u = mda.Universe(str(tpr_path), str(xtc_path))

    # Lipid residues (adjust if your resnames differ)
    lip = u.select_atoms("resname POPC POPS CHOL").residues
    n_res = len(lip)

    # reference beads: second tail beads + cholesterol R1 (as in SI) :contentReference[oaicite:4]{index=4}
    # We build a "reference point per residue" as the mean position of all matching beads.
    ref_names = {"C2A", "C2B", "D2A", "D2B", "R1"}
    ref_atom_ids = []
    ref_res_ids = []

    head_atom_ids = np.full(n_res, -1, dtype=int)  # for leaflet assignment
    po4_ids = []

    for ri, res in enumerate(lip):
        # head bead for leaflet assignment
        head = res.atoms.select_atoms("name PO4 ROH")
        if len(head) > 0:
            head_atom_ids[ri] = head.atoms.indices[0]
            if head.atoms.names[0] == "PO4":
                po4_ids.append(head_atom_ids[ri])

        # reference beads
        rsel = res.atoms.select_atoms("name " + " ".join(sorted(ref_names)))
        for a in rsel:
            ref_atom_ids.append(a.index)
            ref_res_ids.append(ri)

    ref_atom_ids = np.array(ref_atom_ids, dtype=int)
    ref_res_ids = np.array(ref_res_ids, dtype=int)
    po4_ids = np.array(po4_ids, dtype=int)

    n_bins = int(np.ceil(r_max / bin_width))
    hist = np.zeros((n_res, n_bins), dtype=np.float32)

    print("  Finding frames in last 1 µs...", flush=True)
    frames = get_last_1us_frames(u)
    if not frames:
        raise RuntimeError("No frames found for last 1 µs window (check time units in XTC).")

    # Limit frames if max_frames is set (for testing)
    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]
        print(f"  Limiting to {max_frames} frames for testing...", flush=True)

    n_frames = len(frames)
    print(f"  Processing {n_frames} frames (last 1 µs)...", flush=True)

    # Preallocate for fast accumulation
    sum_pos = np.zeros((n_res, 3), dtype=np.float32)
    counts = np.bincount(ref_res_ids, minlength=n_res).astype(np.float32)
    counts[counts == 0] = 1.0

    # Neighbor search (fast C-based)
    from MDAnalysis.lib.nsgrid import FastNS

    # Progress bar for frame processing
    for fi in tqdm(frames, desc=f"  Processing {n_frames} frames", unit="frame"):
        ts = u.trajectory[fi]
        box = ts.dimensions  # [lx, ly, lz, alpha, beta, gamma]

        # Compute reference positions (mean of ref beads per residue)
        sum_pos[:] = 0.0
        pos = u.atoms.positions[ref_atom_ids]
        np.add.at(sum_pos, ref_res_ids, pos)
        ref_pos = sum_pos / counts[:, None]

        # Leaflet split using head bead z relative to PO4 midplane
        if len(po4_ids) > 0:
            z_mid = np.mean(u.atoms.positions[po4_ids, 2])
        else:
            z_mid = np.mean(ref_pos[:, 2])

        head_z = np.where(
            head_atom_ids >= 0,
            u.atoms.positions[head_atom_ids, 2],
            ref_pos[:, 2],
        )

        up = np.where(head_z > z_mid)[0]
        dn = np.where(head_z <= z_mid)[0]

        # run neighbor search within each leaflet
        for idx in (up, dn):
            if len(idx) < 2:
                continue
            coords = ref_pos[idx].astype(np.float32)

            ns = FastNS(r_max, coords, box)
            res = ns.self_search()
            pairs = res.get_pairs()      # shape (n_pairs, 2) in leaflet-local indexing
            dists = res.get_pair_distances()

            if len(dists) == 0:
                continue

            bins = np.minimum((dists / bin_width).astype(int), n_bins - 1)

            gi = idx[pairs[:, 0]]
            gj = idx[pairs[:, 1]]

            # symmetric update (each pair contributes to both lipids)
            np.add.at(hist, (gi, bins), 1.0)
            np.add.at(hist, (gj, bins), 1.0)

    # Average over frames (so peak height isn’t just “more frames = bigger number”)
    hist /= float(len(frames))

    r = (np.arange(n_bins) + 0.5) * bin_width

    # Identify a “first peak window” robustly (typical nearest-neighbor distances)
    # You can tune these bounds after you look at one global histogram.
    peak_window = (r >= PEAK_WINDOW_R[0]) & (r <= PEAK_WINDOW_R[1])

    peak_metric = hist[:, peak_window].sum(axis=1)

    # Residue labels for validation plots (resname, resid) per index
    res_labels = [(res.resname, res.resid) for res in lip]

    order_metric = None
    if compute_order:
        order_metric = compute_per_lipid_order(u, lip, frames)
    return peak_metric, hist, r, res_labels, order_metric


def compute_per_lipid_order(u, lip, frames, sn1_beads=None, sn2_beads=None):
    """
    Compute S_chain = (3*cos²θ - 1)/2 per lipid, averaged over frames and tail bonds.
    θ = angle between tail bond vector and membrane normal (z).
    Returns array of length n_res. For CHOL (no acyl tails) we use 0.
    """
    if sn1_beads is None:
        sn1_beads = ["D2A", "C2A", "C3A", "C4A"]
    if sn2_beads is None:
        sn2_beads = ["C2B", "C3B", "C4B"]
    
    n_res = len(lip)
    order_per_res = np.zeros(n_res, dtype=np.float64)
    count_per_res = np.zeros(n_res, dtype=np.float64)
    
    for fi in frames:
        ts = u.trajectory[fi]
        Lx, Ly, Lz = ts.dimensions[:3]
        for ri, res in enumerate(lip):
            rn = res.resname
            if rn == "CHOL":
                # Cholesterol: no S_chain; use 0 so we still have 2D (packing, 0)
                order_per_res[ri] += 0.0
                count_per_res[ri] += 1.0
                continue
            atoms = res.atoms
            names = atoms.names
            pos = atoms.positions.copy()
            vectors = []
            for bead_list in [sn1_beads, sn2_beads]:
                coords = []
                for b in bead_list:
                    idx = np.where(names == b)[0]
                    if len(idx) > 0:
                        coords.append(pos[idx[0]])
                for i in range(len(coords) - 1):
                    d = coords[i + 1] - coords[i]
                    # PBC
                    d[0] -= Lx * round(d[0] / Lx)
                    d[1] -= Ly * round(d[1] / Ly)
                    d[2] -= Lz * round(d[2] / Lz)
                    nrm = np.linalg.norm(d)
                    if nrm > 1e-6:
                        vectors.append(d / nrm)
            if not vectors:
                continue
            vectors = np.array(vectors)
            cos_theta = np.abs(vectors[:, 2])
            s_bonds = 0.5 * (3 * cos_theta**2 - 1)
            s_chain = np.mean(s_bonds)
            order_per_res[ri] += s_chain
            count_per_res[ri] += 1.0
    
    count_per_res[count_per_res == 0] = 1.0
    return order_per_res / count_per_res


def fit_gmm_2d(X, k_min=1, k_max=3):
    """
    Fit 2D GMM (e.g. packing + order). Returns best_k, gmm, bic_by_k.
    Phase assignment is by gmm.predict(X), not by thresholds.
    """
    bic_by_k = {}
    best = None
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_by_k[k] = bic
        if best is None or bic < best["bic"]:
            best = {"k": k, "gmm": gmm, "bic": bic}
    return best["k"], best["gmm"], bic_by_k


def plot_and_save_histogram(metrics, run_name, out_dir, title_suffix="", global_thresholds=None):
    """
    Plot histogram + KDE + GMM for metrics and save to out_dir.
    If global_thresholds is provided, overlay them as vertical lines (for per-composition plots).
    Returns k, thresholds, bic_by_k for the GMM fit.
    """
    k, thresholds, params, bic_by_k = fit_gmm_thresholds(metrics, k_min=1, k_max=3)
    
    fig, ax = plt.subplots()
    ax.hist(metrics, bins=HIST_BINS_METRIC, density=True, alpha=0.5, color="steelblue",
            label="Histogram (data)", edgecolor="none")
    
    x_min, x_max = metrics.min(), metrics.max()
    xs = np.linspace(x_min, x_max, 500)
    try:
        kde = gaussian_kde(metrics)
        ax.plot(xs, kde(xs), color="darkblue", linewidth=2, label="KDE (smooth)")
    except np.linalg.LinAlgError:
        pass
    
    means, covs, weights = params
    stds = np.sqrt(covs)
    for i in range(len(means)):
        ax.plot(xs, weights[i] * norm.pdf(xs, means[i], stds[i]),
                linestyle="-", linewidth=1.5, label=f"GMM component {i+1}")
    
    mixture = np.zeros_like(xs)
    for i in range(len(means)):
        mixture += weights[i] * norm.pdf(xs, means[i], stds[i])
    ax.plot(xs, mixture, color="black", linestyle="--", linewidth=2, label="GMM mixture")
    
    # Plot thresholds: local (from this composition's GMM) in gray, global (if provided) in red
    for th in thresholds:
        ax.axvline(th, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Local thresholds" if th == thresholds[0] else "")
    
    if global_thresholds is not None:
        for th in global_thresholds:
            ax.axvline(th, color="red", linestyle="-", linewidth=2, alpha=0.8, label="Global thresholds" if th == global_thresholds[0] else "")
    
    ax.set_xlabel("First-shell peak metric")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8)
    title = f"{run_name}: First-shell peak metric (best K={k}){title_suffix}"
    if global_thresholds is not None:
        title += "\n(Red lines = global thresholds applied to all compositions)"
    ax.set_title(title)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "ternary_phase_histogram.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    
    return k, thresholds, bic_by_k


def plot_validation_rdfs(hist, r, peak_metric, res_labels, run_name, out_dir, percentile=5, rebin_nm=None):
    """
    Validation: plot mean ± std RDF-like curves for lowest/highest percentile of lipids.
    rebin_nm: if > 0, rebin curves into this bin width (nm) for smoother plots (e.g. 0.05 reduces spikiness).
    """
    n_res = len(peak_metric)
    if n_res < 20:
        print(f"  Skipping validation plots: only {n_res} lipids (need ≥20 for 5% groups)")
        return

    n_group = max(10, n_res * percentile // 100)
    order = np.argsort(peak_metric)
    lowest_idx = order[:n_group]
    highest_idx = order[-n_group:]

    low_mean = hist[lowest_idx, :].mean(axis=0)
    low_std = hist[lowest_idx, :].std(axis=0)
    high_mean = hist[highest_idx, :].mean(axis=0)
    high_std = hist[highest_idx, :].std(axis=0)
    
    if rebin_nm is None:
        rebin_nm = VALIDATION_RDF_REBIN
    # Optional rebin for smoother plot (reduces spikiness from fine bin_width)
    if rebin_nm > 0 and len(r) > 1:
        r_edges = np.arange(0, r.max() + rebin_nm, rebin_nm)
        r_plot = 0.5 * (r_edges[:-1] + r_edges[1:])
        low_mean_r = np.zeros(len(r_plot))
        low_std_r = np.zeros(len(r_plot))
        high_mean_r = np.zeros(len(r_plot))
        high_std_r = np.zeros(len(r_plot))
        for i in range(len(r_edges) - 1):
            mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
            if mask.any():
                low_mean_r[i] = np.mean(low_mean[mask])
                low_std_r[i] = np.mean(low_std[mask])
                high_mean_r[i] = np.mean(high_mean[mask])
                high_std_r[i] = np.mean(high_std[mask])
        r, low_mean, low_std, high_mean, high_std = r_plot, low_mean_r, low_std_r, high_mean_r, high_std_r

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Plot both groups on one figure for comparison
    fig, ax = plt.subplots()
    ax.plot(r, low_mean, color="blue", linewidth=2, label=f"Lowest {percentile}% (n={n_group})")
    ax.fill_between(r, low_mean - low_std, low_mean + low_std, alpha=0.3, color="blue")
    
    ax.plot(r, high_mean, color="red", linewidth=2, label=f"Highest {percentile}% (n={n_group})")
    ax.fill_between(r, high_mean - high_std, high_mean + high_std, alpha=0.3, color="red")
    
    ax.set_xlim(0.0, 1.2)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("Neighbor count (avg per frame)")
    ax.set_title(f"{run_name}: Validation (lowest vs highest {percentile}%)\n"
                 f"If metric is meaningful: highest group should show stronger first-shell peak")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(out_path / "validation_rdf_comparison.png", dpi=150)
    plt.close(fig)

    print(f"  Saved validation plot to: {out_path / 'validation_rdf_comparison.png'}")
    print(f"  Validation groups: lowest {percentile}% (n={n_group}, metric range [{peak_metric[lowest_idx].min():.4f}, {peak_metric[lowest_idx].max():.4f}])")
    print(f"                      highest {percentile}% (n={n_group}, metric range [{peak_metric[highest_idx].min():.4f}, {peak_metric[highest_idx].max():.4f}])")


def assign_phases_with_thresholds(metrics, thresholds):
    """
    Assign each lipid to a phase based on global thresholds (literature approach):
    - Below lower threshold → Ld (disordered phase)
    - Between lower and upper threshold → Lo (ordered phase)  
    - Above upper threshold → Lb (gel phase)
    
    Returns phase assignments: 0=Ld, 1=Lo, 2=Lb
    """
    if len(thresholds) == 0:
        # K=1: all lipids in same phase (cannot distinguish phases)
        return np.zeros(len(metrics), dtype=int)
    elif len(thresholds) == 1:
        # K=2: binary split (one threshold)
        # Below threshold = Ld, above threshold = Lo (or Lb if high metric = gel)
        phases = np.zeros(len(metrics), dtype=int)
        phases[metrics > thresholds[0]] = 1  # Above threshold = Lo (ordered)
        # Below threshold = Ld (disordered) remains 0
        return phases
    else:
        # K=3: three phases with two thresholds
        # thresholds[0] = lower threshold, thresholds[-1] = upper threshold
        phases = np.zeros(len(metrics), dtype=int)
        phases[metrics < thresholds[0]] = 0  # Ld (below lower threshold)
        phases[(metrics >= thresholds[0]) & (metrics <= thresholds[-1])] = 1  # Lo (between thresholds)
        phases[metrics > thresholds[-1]] = 2  # Lb (above upper threshold)
        return phases


def compute_phase_fractions(metrics, thresholds):
    """
    Compute phase fractions using global thresholds (literature approach):
    - Ld (disordered): below lower threshold
    - Lo (ordered): between lower and upper threshold
    - Lb (gel): above upper threshold
    
    Returns dict with fractions for each phase.
    """
    phases = assign_phases_with_thresholds(metrics, thresholds)
    n_total = len(phases)
    
    if len(thresholds) == 0:
        return {"single_phase": 1.0, "note": "K=1: cannot distinguish phases with this metric"}
    elif len(thresholds) == 1:
        # Binary: Ld vs Lo (or Ld vs Lb if high metric = gel)
        frac_ld = np.sum(phases == 0) / n_total
        frac_lo = np.sum(phases == 1) / n_total
        return {"Ld": frac_ld, "Lo": frac_lo}
    else:
        # Ternary: Ld, Lo, Lb (two thresholds)
        frac_ld = np.sum(phases == 0) / n_total
        frac_lo = np.sum(phases == 1) / n_total
        frac_lb = np.sum(phases == 2) / n_total
        return {"Ld": frac_ld, "Lo": frac_lo, "Lb": frac_lb}


# -----------------------------
# Example: run on a root directory
# -----------------------------
def main(root_dir, composition_filter=None, max_frames=None, use_order=False):
    """
    root_dir: path to folder containing composition subdirs (e.g. Production_out).
    composition_filter: optional.
    max_frames: if set, limit frames per run (for testing).
    use_order: if True, compute per-lipid tail order (S_chain) and fit 2D GMM (packing + order).

    Methodology:
    1. Collect metrics (and optionally order) from all compositions
    2. Fit GLOBAL GMM (1D packing or 2D packing+order) → get thresholds or component assignment
    3. Apply to each composition → compute phase fractions
    4. Generate validation plots (mean ± std for lowest/highest 5%)
    """
    # Print parameters so user knows bin width, R range, and metric meaning
    print("\n--- Parameters ---")
    print(f"  bin_width (RDF distance bins): {BIN_WIDTH_RDF} nm")
    print(f"  r_max (neighbor search):       {R_MAX} nm")
    print(f"  first-shell window:            [{PEAK_WINDOW_R[0]}, {PEAK_WINDOW_R[1]}] nm")
    print(f"  First-shell peak metric units: average neighbor count (per frame) summed over shell")
    print(f"  Histogram y-axis 'Density':    probability per unit metric (high values = narrow metric range)")
    if use_order:
        print(f"  Second metric:                per-lipid tail order S_chain (P2-like); 2D GMM")
    print("---\n")

    root = Path(root_dir)
    all_composition_dirs = [p for p in root.iterdir() if p.is_dir() and parse_composition(p.name)]

    if composition_filter is None:
        run_dirs = all_composition_dirs
    elif "*" in composition_filter or "?" in composition_filter:
        run_dirs = [p for p in all_composition_dirs if fnmatch.fnmatch(p.name, composition_filter)]
        if not run_dirs:
            available = [p.name for p in all_composition_dirs]
            raise SystemExit(
                f"No folders matched pattern '{composition_filter}' under {root}. "
                f"Available ({len(available)}): {available[:15]}{'...' if len(available) > 15 else ''}"
            )
        print(f"Pattern '{composition_filter}' matched {len(run_dirs)} folder(s): {[p.name for p in run_dirs]}")
    else:
        run_dirs = [p for p in all_composition_dirs if p.name == composition_filter]
        if not run_dirs:
            available = [p.name for p in all_composition_dirs]
            raise SystemExit(
                f"No folder named '{composition_filter}' found under {root}. "
                f"Available: {available}"
            )

    # ====================================================================
    # STEP 1: Collect metrics from all compositions
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: Collecting metrics from {len(run_dirs)} composition(s)")
    print(f"{'='*70}")
    
    all_metrics = []
    all_order = []  # used when use_order=True
    run_data = {}   # Store (metrics, hist, r, res_labels, order) for each run

    for rd in run_dirs:
        tpr = rd / "step7_lipids_ions.tpr"
        xtc = rd / "step7_lipids_ions.xtc"
        if not (tpr.exists() and xtc.exists()):
            print(f"Skipping {rd.name} (missing .tpr or .xtc)", flush=True)
            continue

        print(f"\nCollecting: {rd.name}")
        out = compute_peak_metric_for_run(tpr, xtc, max_frames=max_frames, compute_order=use_order)
        metrics, hist, r_axis, res_labels, order_metric = out
        all_metrics.append(metrics)
        if order_metric is not None:
            all_order.append(order_metric)
        run_data[rd.name] = {
            "metrics": metrics,
            "hist": hist,
            "r": r_axis,
            "res_labels": res_labels,
            "path": rd,
            "order": order_metric
        }

    if not all_metrics:
        raise SystemExit("No runs had both step7_lipids_ions.tpr and step7_lipids_ions.xtc. Nothing to analyze.")

    # ====================================================================
    # STEP 2: Fit GLOBAL GMM (1D packing or 2D packing+order)
    # ====================================================================
    if use_order and all_order:
        # 2D GMM: packing + tail order
        print(f"\n{'='*70}")
        print(f"STEP 2: Fitting GLOBAL 2D GMM (packing + tail order)")
        print(f"{'='*70}")
        all_metrics_pooled = np.concatenate(all_metrics)
        all_order_pooled = np.concatenate(all_order)
        X_2d = np.column_stack([all_metrics_pooled, all_order_pooled])
        k_global, gmm_2d, bic_by_k_global = fit_gmm_2d(X_2d, k_min=1, k_max=3)
        # Order components by mean order (low → high → Ld, Lo, Lb)
        order_means = [gmm_2d.means_[i, 1] for i in range(k_global)]
        order_perm = np.argsort(order_means)
        thresholds_global = []  # no 1D thresholds in 2D mode
        print("\n--- BIC for GLOBAL 2D (packing + order) ---")
        for kk in sorted(bic_by_k_global.keys()):
            marker = "  <-- best" if kk == k_global else ""
            print(f"  K={kk}: BIC = {bic_by_k_global[kk]:.2f}{marker}")
        print(f"\n*** GLOBAL Best K = {k_global} (2D GMM) ***")
        print("  Phases assigned by component; components ordered by mean tail order (low→Ld, high→Lo/Lb).")
        for i in range(k_global):
            j = order_perm[i]
            print(f"     Component {j}: mean packing={gmm_2d.means_[j,0]:.6f}, mean order={gmm_2d.means_[j,1]:.4f}")
        thresholds_global = []  # no 1D thresholds in 2D mode
        global_thresh_file = root / "analysis_out" / "global_thresholds.txt"
        global_thresh_file.parent.mkdir(parents=True, exist_ok=True)
        with open(global_thresh_file, "w") as f:
            f.write(f"Global 2D GMM: K={k_global}\n")
            f.write(f"BIC values:\n")
            for kk in sorted(bic_by_k_global.keys()):
                f.write(f"  K={kk}: BIC = {bic_by_k_global[kk]:.2f}\n")
        print(f"\nSaved: {global_thresh_file}")
    else:
        # 1D GMM: packing only
        print(f"\n{'='*70}")
        print(f"STEP 2: Fitting GLOBAL GMM on pooled metrics (all compositions)")
        print(f"{'='*70}")
        all_metrics_pooled = np.concatenate(all_metrics)
        k_global, thresholds_global, bic_by_k_global = plot_and_save_histogram(
            all_metrics_pooled, "GLOBAL_THRESHOLDS", root / "analysis_out",
            title_suffix=" (GLOBAL - applied to all compositions)"
        )
        
        print("\n--- BIC for GLOBAL pooled data (lower = better fit) ---")
        for kk in sorted(bic_by_k_global.keys()):
            marker = "  <-- best" if kk == k_global else ""
            print(f"  K={kk}: BIC = {bic_by_k_global[kk]:.2f}{marker}")
        print(f"\n*** GLOBAL Best K = {k_global}  |  GLOBAL thresholds = {thresholds_global} ***")
        print("  These thresholds will be applied to ALL compositions for consistent phase assignment.")
        
        if k_global == 1:
            print("\n  ⚠️  NOTE: K=1 means the metric shows a SINGLE, UNIMODAL distribution.")
            print("     Try --use-order for 2D GMM (packing + tail order) to improve phase detection.")
        elif k_global == 2:
            print(f"\n  ✓ K=2: Two phases. Threshold = {thresholds_global[0]:.6f}")
        elif k_global == 3:
            print(f"\n  ✓ K=3: Three phases. Thresholds = {thresholds_global}")
        
        if 1 in bic_by_k_global and 2 in bic_by_k_global:
            delta_2_vs_1 = bic_by_k_global[1] - bic_by_k_global[2]
            if k_global == 2 and delta_2_vs_1 < 10:
                print(f"\n  Note: K=2 is only {delta_2_vs_1:.1f} BIC points better than K=1.")
            elif k_global == 2 and delta_2_vs_1 >= 10:
                print(f"\n  K=2 is {delta_2_vs_1:.1f} BIC points better than K=1.")

        global_thresh_file = root / "analysis_out" / "global_thresholds.txt"
        global_thresh_file.parent.mkdir(parents=True, exist_ok=True)
        with open(global_thresh_file, "w") as f:
            f.write(f"Global GMM fit: K={k_global}\n")
            f.write(f"Global thresholds: {thresholds_global}\n")
            f.write(f"BIC values:\n")
            for kk in sorted(bic_by_k_global.keys()):
                f.write(f"  K={kk}: BIC = {bic_by_k_global[kk]:.2f}\n")
        print(f"\nSaved global thresholds to: {global_thresh_file}")
        gmm_2d = None
        order_perm = None

    # ====================================================================
    # STEP 3: Apply global thresholds to each composition
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"STEP 3: Applying global thresholds to each composition")
    print(f"{'='*70}")
    
    phase_fractions_by_composition = {}
    
    for run_name, data in run_data.items():
        print(f"\n{'-'*60}")
        print(f"Composition: {run_name}")
        print(f"{'-'*60}")
        
        metrics = data["metrics"]
        hist = data["hist"]
        r_axis = data["r"]
        res_labels = data["res_labels"]
        rd = data["path"]
        order_metric = data.get("order")
        
        if gmm_2d is not None and order_metric is not None:
            # 2D GMM: assign phase by component (ordered by mean order: 0=Ld, 1=Lo, 2=Lb)
            X_run = np.column_stack([metrics, order_metric])
            comp_labels = gmm_2d.predict(X_run)  # 0, 1, or 2 (raw component index)
            # Map to phase by order_perm: component order_perm[0] = Ld, etc.
            phase_labels = np.zeros(len(comp_labels), dtype=int)
            for c in range(k_global):
                phase_labels[comp_labels == order_perm[c]] = c
            n_total = len(phase_labels)
            phase_fracs = {}
            names = ["Ld", "Lo", "Lb"]  # up to 3 phases
            for c in range(min(k_global, 3)):
                phase_fracs[names[c]] = np.sum(phase_labels == c) / n_total
            phase_fractions_by_composition[run_name] = phase_fracs
            print(f"\nPhase fractions (2D GMM, packing + order):")
            for name, frac in phase_fracs.items():
                print(f"  {name}: {frac:.3f} ({frac*100:.1f}%)")
        else:
            # 1D: use global thresholds
            phase_fracs = compute_phase_fractions(metrics, thresholds_global)
            phase_fractions_by_composition[run_name] = phase_fracs
            
            print(f"\nPhase fractions (using global thresholds):")
            if "note" in phase_fracs:
                print(f"  {phase_fracs['note']}")
            for phase, frac in phase_fracs.items():
                if phase != "note":
                    print(f"  {phase}: {frac:.3f} ({frac*100:.1f}%)")
        
        # Show threshold interpretation (1D only)
        if gmm_2d is None and len(thresholds_global) > 0:
            metric_min, metric_max = metrics.min(), metrics.max()
            print(f"\n  Metric range for this composition: [{metric_min:.6f}, {metric_max:.6f}]")
            if len(thresholds_global) == 1:
                print(f"  Global threshold: {thresholds_global[0]:.6f}")
                n_below = np.sum(metrics < thresholds_global[0])
                n_above = np.sum(metrics >= thresholds_global[0])
                print(f"    {n_below} lipids below (Ld), {n_above} lipids above (Lo)")
            else:
                print(f"  Global thresholds: lower={thresholds_global[0]:.6f}, upper={thresholds_global[-1]:.6f}")
                n_ld = np.sum(metrics < thresholds_global[0])
                n_lo = np.sum((metrics >= thresholds_global[0]) & (metrics <= thresholds_global[-1]))
                n_lb = np.sum(metrics > thresholds_global[-1])
                print(f"    {n_ld} lipids in Ld, {n_lo} lipids in Lo, {n_lb} lipids in Lb")
        
        # Plot histogram with global thresholds overlaid (1D only; 2D has no 1D thresholds)
        val_out = rd / "analysis_out"
        k_local, thresholds_local, bic_by_k_local = plot_and_save_histogram(
            metrics, run_name, val_out, title_suffix="",
            global_thresholds=thresholds_global if gmm_2d is None else None
        )
        
        # Print local BIC for comparison (but we use global thresholds)
        print(f"\nLocal GMM fit (for comparison): K={k_local}, thresholds={thresholds_local}")
        print(f"  Note: Using GLOBAL thresholds for phase assignment, not local ones.")
        
        # Validation: plot mean ± std for lowest/highest 5%
        plot_validation_rdfs(hist, r_axis, metrics, res_labels, run_name, val_out)
        print(f"Saved plots for {run_name} to: {val_out}")

    # ====================================================================
    # Summary: Phase fractions for all compositions
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY: Phase fractions for all compositions")
    print(f"{'='*70}")
    if gmm_2d is not None:
        print(f"Mode: 2D GMM (packing + tail order)")
    else:
        print(f"Global thresholds: {thresholds_global}")
    print(f"\nComposition | Phase fractions")
    print(f"{'-'*70}")
    for run_name, fracs in phase_fractions_by_composition.items():
        frac_str = " | ".join([f"{k}: {v:.3f}" for k, v in fracs.items() if k != "note" and isinstance(v, (int, float))])
        print(f"{run_name:30s} | {frac_str}")
    
    # Save summary to file
    summary_file = root / "analysis_out" / "phase_fractions_summary.txt"
    with open(summary_file, "w") as f:
        if gmm_2d is not None:
            f.write("Mode: 2D GMM (packing + tail order)\n")
        else:
            f.write(f"Global thresholds: {thresholds_global}\n")
        f.write(f"Global K: {k_global}\n\n")
        f.write("Composition | Phase fractions\n")
        f.write("-" * 70 + "\n")
        for run_name, fracs in phase_fractions_by_composition.items():
            frac_str = " | ".join([f"{k}: {v:.3f}" for k, v in fracs.items() if k != "note" and isinstance(v, (int, float))])
            f.write(f"{run_name:30s} | {frac_str}\n")
    print(f"\nSaved phase fractions summary to: {summary_file}")
    
    # Copy global plot to current working directory
    import shutil
    cwd = Path.cwd()
    global_source = root / "analysis_out" / "ternary_phase_histogram.png"
    global_dest = cwd / "ternary_phase_histogram_GLOBAL.png"
    if global_source.exists():
        shutil.copy2(global_source, global_dest)
        print(f"Saved global plot copy to: {global_dest}")

if __name__ == "__main__":
    import sys
    import argparse
    
    default_root = "/Users/fuchunzhang/Downloads/Data_Extract/Production_out"
    
    # Parse arguments manually to handle legacy usage patterns
    max_frames = None
    root_dir = default_root
    composition_filter = None
    
    use_order = "--use-order" in sys.argv
    if use_order:
        sys.argv.remove("--use-order")
    
    # Check for --max-frames flag
    if "--max-frames" in sys.argv:
        idx = sys.argv.index("--max-frames")
        if idx + 1 < len(sys.argv):
            try:
                max_frames = int(sys.argv[idx + 1])
                sys.argv.pop(idx)
                sys.argv.pop(idx)
            except (ValueError, IndexError):
                pass
    
    # Handle positional arguments (legacy style)
    # Pattern: [python script.py [root_dir] [filter]]
    # or: [python script.py [filter]] (uses default root)
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    
    if len(positional_args) == 0:
        # No args: use defaults
        root_dir, composition_filter = default_root, None
    elif len(positional_args) == 1:
        # One arg: could be root_dir or filter
        arg = positional_args[0]
        if Path(arg).exists() and Path(arg).is_dir():
            # It's a directory path
            root_dir = arg
            composition_filter = None
        else:
            # It's a filter pattern (composition name or wildcard)
            root_dir = default_root
            composition_filter = arg
    elif len(positional_args) >= 2:
        # Two args: root_dir, then filter
        root_dir = positional_args[0]
        composition_filter = positional_args[1]
    
    main(root_dir, composition_filter, max_frames=max_frames, use_order=use_order)
