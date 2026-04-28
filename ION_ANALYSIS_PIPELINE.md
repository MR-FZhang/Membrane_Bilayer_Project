# Ion Distribution & Electrical Double Layer Analysis Pipeline
## Cursor Implementation Instructions

> **Purpose:** This document provides complete, self-contained instructions for Cursor to implement a
> modular Python analysis pipeline for characterising ion distributions and the electrical double layer
> around Martini 3 coarse-grained lipid bilayer simulations. Read every section before writing any code.

---

## 0. Background & Scientific Context

This pipeline analyses how Na⁺ and Cl⁻ ions distribute around lipid bilayers of three compositions:

| System tag | Composition | Net bilayer charge |
|---|---|---|
| `POPC100` | Pure POPC | 0 |
| `POPC_POPS` | POPC + POPS (mole ratio TBD) | −N_POPS × e |
| `POPC_POPS_CHOL` | POPC + POPS + Cholesterol | −N_POPS × e |

The methodology follows two references:
1. **Saurabh et al., Mol. Phys. 121 e2236248 (2023)** — surface-distance RDF and Debye length fitting
2. **Gonzalez & Bresme, J. Phys. Chem. B 124, 5156 (2020)** — number density profiles ρ(z)

**Key methodological choice:** The Q_net(z) and Debye length analysis uses the **1D number density
profile ρ(z)** method (integrating along the bilayer normal), not the surface-distance RDF integration.
The surface-distance RDF is also computed for visualisation and comparison.

---

## 1. Repository / Folder Structure to Create

Cursor must create the following directory structure. Do **not** deviate from these names:

```
ion_analysis/
├── ION_ANALYSIS_PIPELINE.md        ← this file (already exists)
├── config.yaml                     ← user-editable configuration (see Section 2)
├── run_all.sh                      ← master shell script that runs the full pipeline
│
├── scripts/
│   ├── 00_prepare_gromacs.sh       ← GROMACS pre-processing commands
│   ├── 01_run_gromacs_density.sh   ← gmx density commands for all species
│   ├── 02_run_gromacs_rdf.sh       ← gmx rdf -surf commands for Na+ and Cl-
│   ├── 03_plot_density_profiles.py ← Plot ρ(z) for all species
│   ├── 04_compute_qnet.py          ← Compute Q_net(z) from ρ(z) profiles
│   ├── 05_fit_debye_length.py      ← Interactive Debye length exponential fitting
│   ├── 06_plot_rdf_normalised.py   ← Load, normalise, and plot surface-distance RDFs
│   └── utils.py                    ← Shared utility functions (file I/O, plotting style)
│
└── {SYSTEM_TAG}/                   ← One folder per composition (created by user)
    ├── topol.tpr
    ├── traj_nopbc.xtc              ← pre-processed trajectory (centred, no PBC)
    ├── analysis.ndx                ← GROMACS index file (see Section 4)
    └── analysis_out/               ← ALL output files go here (created by scripts)
        ├── density_Na.xvg
        ├── density_Cl.xvg
        ├── density_PO4.xvg
        ├── density_CHOL.xvg
        ├── RDF_Na_surf.xvg
        ├── RDF_Cl_surf.xvg
        ├── fig_density_profiles.png
        ├── fig_qnet.png
        ├── fig_debye_fit.png
        ├── fig_rdf_normalised.png
        └── results_summary.txt
```

---

## 2. Configuration File: `config.yaml`

Create `config.yaml` in the `ion_analysis/` root. This is the **only file the user ever needs to edit**.
All scripts read from this file. Use the `pyyaml` library to load it.

```yaml
# ============================================================
# USER CONFIGURATION — edit these values for your system
# ============================================================

# Path to the folder containing topol.tpr, traj_nopbc.xtc, analysis.ndx
# Example: "/home/user/simulations/POPC_POPS"
system_path: "/path/to/your/system"

# Human-readable label for this system (used in plot titles and filenames)
system_label: "POPC+POPS (Martini 3)"

# Number of POPS molecules in the ENTIRE bilayer (both leaflets combined)
# Set to 0 for pure POPC system
n_pops: 40

# Number of POPC molecules (both leaflets)
n_popc: 120

# Number of CHOL molecules (both leaflets); set to 0 if no cholesterol
n_chol: 0

# Martini 3 bead names — verify with: gmx dump -s topol.tpr | grep -iE '"NA|"CL'
ion_name_na: "NA"     # Na+ residue name in your topology
ion_name_cl: "CL"     # Cl- residue name in your topology
lipid_head_bead: "PO4"          # phosphate bead name (same for POPC and POPS)
chol_head_bead: "ROH"           # cholesterol head bead (set to null if no CHOL)

# GROMACS index group names (must match names in analysis.ndx)
ndx_group_po4: "PO4_all"        # PO4 beads of all POPC + POPS (reference surface)
ndx_group_na: "NA_ions"
ndx_group_cl: "CL_ions"
ndx_group_membrane: "Membrane"  # all lipid residues, used for centering
ndx_group_chol_head: "CHOL_head"  # set to null if no cholesterol

# Trajectory time settings
# Start time (ps) for analysis — use last 50-60% of trajectory for equilibrium
traj_begin_ps: 100000

# Number of slabs for gmx density along z-axis
density_n_slabs: 200

# RDF settings
rdf_rmax_nm: 2.5    # maximum distance for surface-distance RDF (nm)
rdf_bin_nm: 0.01    # bin width for RDF (nm)

# ============================================================
# ANALYSIS PARAMETERS — adjust after inspecting initial plots
# ============================================================

# Region for linear baseline fitting of the normalised RDF
# These are distances (nm) from the PO4 surface
rdf_bulk_fit_min_nm: 1.0
rdf_bulk_fit_max_nm: 2.0

# Region for Debye length exponential fitting of Q_net(z)
# z_fit_min: start of fitting (nm from bilayer centre), should be BEYOND first ion peak
# z_fit_max: end of fitting (nm from bilayer centre), should be before bulk flatline noise
# NOTE: these are z values in the FULL bilayer frame (bilayer COM = z=0)
# The fitting is done on the FOLDED profile (distance from nearest leaflet surface)
debye_fit_min_nm: 0.7   # nm from PO4 peak position outward into bulk water
debye_fit_max_nm: 1.8   # nm from PO4 peak position

# Poisson-Boltzmann parameters
salt_concentration_M: 0.15   # NaCl concentration in mol/L
temperature_K: 310.0         # simulation temperature in Kelvin
epsilon_r: 84.0              # effective dielectric constant for Martini 3 water (W beads)
```

---

## 3. Shared Utilities: `scripts/utils.py`

Create this file first. All other scripts import from it.

### 3.1 Functions to implement

```python
# scripts/utils.py

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

def load_config(config_path="config.yaml") -> dict:
    """Load and validate config.yaml. Raise clear errors for missing keys."""
    ...

def get_output_dir(cfg: dict) -> Path:
    """Return Path to analysis_out/ inside system_path. Create it if absent."""
    out = Path(cfg["system_path"]) / "analysis_out"
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_xvg(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GROMACS .xvg file, skipping all lines beginning with '#' or '@'.
    Returns (x_array, y_array) as float64 numpy arrays.
    Handles both 2-column and multi-column files (returns first two columns).
    """
    ...

def set_publication_style():
    """
    Apply a clean, publication-quality matplotlib style.
    - Font: Arial or DejaVu Sans, 11pt
    - Figure size: (8, 5) inches default
    - DPI: 300 for saved figures
    - Spine: top and right spines removed
    - Grid: subtle horizontal dotted grid
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def compute_pb_debye_length(epsilon_r: float, C_mol_L: float, T_K: float) -> float:
    """
    Compute theoretical Poisson-Boltzmann Debye screening length in nm.

    Formula: zeta = sqrt(epsilon_r * epsilon_0 * k_B * T / (2 * e^2 * N_A * C))

    Parameters
    ----------
    epsilon_r : relative permittivity (use 84 for Martini 3 W beads)
    C_mol_L   : salt concentration in mol/L (e.g. 0.15)
    T_K       : temperature in Kelvin

    Returns
    -------
    Debye length in nm
    """
    import scipy.constants as const
    C_SI = C_mol_L * 1000.0  # mol/m^3
    zeta_m = np.sqrt(
        (epsilon_r * const.epsilon_0 * const.k * T_K) /
        (2.0 * const.e**2 * const.Avogadro * C_SI)
    )
    return zeta_m * 1e9  # convert m → nm

def write_summary(output_dir: Path, content: str):
    """Append content to results_summary.txt in output_dir."""
    with open(output_dir / "results_summary.txt", "a") as f:
        f.write(content + "\n")
```

---

## 4. GROMACS Shell Scripts

### 4.1 `scripts/00_prepare_gromacs.sh`

This script builds the GROMACS index file. It uses `gmx select` (GROMACS 2021+) for string-based
selection. The script reads `SYSTEM_PATH` from the first command-line argument.

```bash
#!/bin/bash
# Usage: bash scripts/00_prepare_gromacs.sh /path/to/system
# Creates analysis.ndx in the system folder

SYSTEM_PATH=$1
TPR="${SYSTEM_PATH}/topol.tpr"
NDX_OUT="${SYSTEM_PATH}/analysis.ndx"

# Read config values (ion and bead names) — these must match config.yaml
ION_NA="NA"
ION_CL="CL"
LIPID_HEAD="PO4"
MEMBRANE_RESNAMES="POPC POPS CHOL"

echo "Building index file for: ${SYSTEM_PATH}"

# Create index groups using gmx select
gmx select -s "${TPR}" \
  -select \
    "\"PO4_all\" resname POPC POPS and name ${LIPID_HEAD}; \
     \"NA_ions\" resname ${ION_NA}; \
     \"CL_ions\" resname ${ION_CL}; \
     \"Membrane\" resname ${MEMBRANE_RESNAMES}; \
     \"CHOL_head\" resname CHOL and name ROH" \
  -on "${NDX_OUT}"

echo "Index file written to: ${NDX_OUT}"
echo "IMPORTANT: Verify ion names with:"
echo "  gmx dump -s ${TPR} | grep -iE '\"NA|\"CL' | sort -u"
```

### 4.2 `scripts/01_run_gromacs_density.sh`

Runs `gmx density` for each species. All output goes to `analysis_out/`.

```bash
#!/bin/bash
# Usage: bash scripts/01_run_gromacs_density.sh /path/to/system [begin_time_ps]
# Outputs: density_Na.xvg, density_Cl.xvg, density_PO4.xvg, density_CHOL.xvg

SYSTEM_PATH=$1
BEGIN=${2:-100000}    # default: 100000 ps; override from config.yaml
TPR="${SYSTEM_PATH}/topol.tpr"
TRJ="${SYSTEM_PATH}/traj_nopbc.xtc"
NDX="${SYSTEM_PATH}/analysis.ndx"
OUT="${SYSTEM_PATH}/analysis_out"
mkdir -p "${OUT}"

# Common flags
COMMON="-s ${TPR} -f ${TRJ} -n ${NDX} -center -d Z -sl 200 -dens number -b ${BEGIN}"

echo "Running gmx density for Na+ ..."
echo "NA_ions
Membrane" | gmx density ${COMMON} -o "${OUT}/density_Na.xvg" 2>&1 | tail -5

echo "Running gmx density for Cl- ..."
echo "CL_ions
Membrane" | gmx density ${COMMON} -o "${OUT}/density_Cl.xvg" 2>&1 | tail -5

echo "Running gmx density for PO4 (PHOS) ..."
echo "PO4_all
Membrane" | gmx density ${COMMON} -o "${OUT}/density_PO4.xvg" 2>&1 | tail -5

echo "Running gmx density for CHOL head (ROH) ..."
echo "CHOL_head
Membrane" | gmx density ${COMMON} -o "${OUT}/density_CHOL.xvg" 2>&1 | tail -5

echo "Done. Output in: ${OUT}"
```

### 4.3 `scripts/02_run_gromacs_rdf.sh`

Runs `gmx rdf -surf` for Na⁺ and Cl⁻.

```bash
#!/bin/bash
# Usage: bash scripts/02_run_gromacs_rdf.sh /path/to/system [begin_time_ps]
# Outputs: RDF_Na_surf.xvg, RDF_Cl_surf.xvg

SYSTEM_PATH=$1
BEGIN=${2:-100000}
TPR="${SYSTEM_PATH}/topol.tpr"
TRJ="${SYSTEM_PATH}/traj_nopbc.xtc"
NDX="${SYSTEM_PATH}/analysis.ndx"
OUT="${SYSTEM_PATH}/analysis_out"
mkdir -p "${OUT}"

COMMON="-s ${TPR} -f ${TRJ} -n ${NDX} -surf -ref PO4_all -rmax 2.5 -bin 0.01 -b ${BEGIN}"

echo "Running gmx rdf -surf for Na+ ..."
gmx rdf ${COMMON} -sel NA_ions -o "${OUT}/RDF_Na_surf.xvg" 2>&1 | tail -5

echo "Running gmx rdf -surf for Cl- ..."
gmx rdf ${COMMON} -sel CL_ions -o "${OUT}/RDF_Cl_surf.xvg" 2>&1 | tail -5

echo "Done. Output in: ${OUT}"
```

---

## 5. Python Script 1: Density Profiles — `scripts/03_plot_density_profiles.py`

### Purpose
Load ρ(z) xvg files and produce Figure 1: number density profiles of Na⁺, Cl⁻, PO4 (PHOS),
and optionally CHOL head group. Mirrors Figure 3 of Gonzalez & Bresme (2020).

### Implementation requirements

1. Load `config.yaml` using `utils.load_config()`.
2. Set output dir using `utils.get_output_dir()`.
3. Load these files from `analysis_out/`:
   - `density_Na.xvg` → z (nm), ρ_Na (nm⁻³)
   - `density_Cl.xvg` → z (nm), ρ_Cl (nm⁻³)
   - `density_PO4.xvg` → z (nm), ρ_PO4 (nm⁻³)
   - `density_CHOL.xvg` → z (nm), ρ_CHOL (nm⁻³) — **skip silently if file absent**

4. **Re-centre z so that bilayer centre of mass is at z = 0.** The GROMACS `-center` flag centres
   on the Membrane group, so z = 0 should already correspond to box centre after centering.
   Verify by checking that the PO4 density has two symmetric peaks (one at positive z, one at
   negative z) equidistant from zero. If not, the user needs to re-run the GROMACS step.

5. **Fold the profile to show only one leaflet:** Average the two leaflets by folding:
   `rho_folded(z) = 0.5 * (rho(z) + rho(-z))` for z >= 0. This improves statistics.
   Plot from z = 0 (bilayer centre) to z = z_max (bulk water).

6. **Y-axis:** Use a **log scale** (matching Gonzalez & Bresme Figure 3) to show both the
   large PO4 peak and the small bulk ion density on the same plot.

7. **Plot specification:**
   - Single panel (not 3 stacked like the paper — that is for multiple cation types)
   - Lines: Na⁺ (blue, solid), Cl⁻ (red, solid), PO4 (green, dashed), CHOL (purple, dash-dot, if present)
   - Y-axis label: `ρ(z)  (nm⁻³)`
   - X-axis label: `z  (nm)`
   - Title: `{system_label}` from config
   - Legend inside plot, upper right
   - Y-axis: log scale, range [0.005, max_density * 2]
   - X-axis: [0, z_max] where z_max = half the box length in z direction
   - Add a vertical dashed grey line at the z position of the PO4 peak (detect automatically
     as the argmax of ρ_PO4 in the positive-z half)
   - Annotate this line with "PO4 interface"

8. Save as `analysis_out/fig_density_profiles.png` and `analysis_out/fig_density_profiles.pdf`.

9. Print to stdout: the z-position of the PO4 peak (this is needed as input to script 05).

---

## 6. Python Script 2: Q_net(z) — `scripts/04_compute_qnet.py`

### Purpose
Compute and plot the net accumulated charge Q_net as a function of distance z from the bilayer
surface. This directly follows Eq. 2 of Saurabh et al. adapted for the z-coordinate.

### Scientific background

For a planar bilayer, Q_net(d) is the net charge per unit area accumulated within distance d from
the PO4 surface outward into the bulk water:

```
Q_net(d) = Q_bilayer + ∫₀ᵈ [ρ_Na(z_surf + d') - ρ_Cl(z_surf + d')] * e * A_box  dd'
```

Where:
- `z_surf` = z-position of PO4 peak (from script 03)
- `d` = distance from PO4 peak into the aqueous phase
- `ρ_Na`, `ρ_Cl` = number density profiles in nm⁻³
- `e` = elementary charge
- `A_box` = box area in x-y plane in nm² (read from the tpr/trajectory header, or hard-code from
  the user's system)
- `Q_bilayer` = −n_pops (in units of elementary charge; read from config `n_pops`)

**Important geometry note:** Because the bilayer has two leaflets, integrate outward from BOTH
leaflet surfaces and average, or equivalently use the folded profile (z >= 0) and integrate
outward from the PO4 peak position. This ensures each ion is associated with its nearest
leaflet only, avoiding double-counting across the bilayer midplane.

### Implementation requirements

1. Load config. Load the folded density profiles produced by script 03 (or recompute them here
   by calling the same load + fold logic — factor this into `utils.py`).

2. Detect `z_PO4` = z-position of PO4 peak in the folded profile. Store this for later use.

3. **Define the integration coordinate:** `d = z - z_PO4` where d > 0 means outside the bilayer
   (in water). Select only the region d > 0.

4. **Compute cumulative ion numbers** (per nm² of bilayer area):
   ```
   n_plus(d)  = ∫₀ᵈ ρ_Na(z_PO4 + d') dd'    [nm⁻²]
   n_minus(d) = ∫₀ᵈ ρ_Cl(z_PO4 + d') dd'    [nm⁻²]
   ```
   Use `scipy.integrate.cumulative_trapezoid`.

5. **Compute Q_net(d):**
   ```
   Q_bilayer_per_area = -n_pops / (2 * A_box)   # divide by 2: one leaflet only
   Q_net(d) = Q_bilayer_per_area + n_plus(d) - n_minus(d)
   ```
   Units: elementary charges per nm².

6. **Bulk neutrality check:** Print the mean of Q_net over the last 10% of d values.
   Warn if |mean| > 0.05 e/nm².

7. **Plot specification:**
   - Single panel
   - X-axis: d (nm from PO4 surface), range [0, 2.5 nm]
   - Y-axis: Q_net (e nm⁻²)
   - Line: dark blue, solid, linewidth 2
   - Title: `{system_label} — Net charge accumulation`
   - Add a horizontal dashed grey line at Q_net = 0
   - Add a vertical dashed orange line at d = config `debye_fit_min_nm` (start of fitting region)
   - Add a vertical dashed orange line at d = config `debye_fit_max_nm` (end of fitting region)
   - Label these two lines "fit window"
   - Annotate with the value of `Q_bilayer_per_area` at d = 0

8. Save as `analysis_out/fig_qnet.png` and `analysis_out/fig_qnet.pdf`.

9. **Save the Q_net data** as `analysis_out/qnet_data.npz` containing:
   - `d`: distance array (nm)
   - `qnet`: Q_net array (e/nm²)
   - `z_PO4`: float (nm)
   - `Q_bilayer_per_area`: float

---

## 7. Python Script 3: Debye Length Fitting — `scripts/05_fit_debye_length.py`

### Purpose
Fit an exponential decay to Q_net(d) to extract the Debye screening length ζ, compare it to
the Poisson-Boltzmann prediction, and produce a publication-quality figure.

### Implementation requirements

1. Load config. Load `analysis_out/qnet_data.npz`.

2. **Fitting region:** Use `debye_fit_min_nm` and `debye_fit_max_nm` from config as the fitting
   window. The user can edit these values in config.yaml after inspecting the Q_net plot, then
   re-run this script. No code changes needed.

3. **Fitting model:**
   ```python
   def exponential_decay(d, A0, zeta):
       return A0 * np.exp(-d / zeta)
   ```
   Use `scipy.optimize.curve_fit`. Initial guess: `A0 = Q_bilayer_per_area`, `zeta = 0.8`.
   Bounds: `A0` unconstrained, `zeta` in [0.05, 5.0] nm.

4. **Extract results:**
   - `zeta_fit` ± `zeta_err` (from square root of diagonal of covariance matrix)
   - `A0_fit` ± `A0_err`
   - R² of the fit within the fitting window

5. **Compute theoretical PB Debye length** using `utils.compute_pb_debye_length()` with values
   from config (`epsilon_r`, `salt_concentration_M`, `temperature_K`).

6. **Plot specification — Figure 2 (Debye fit):**
   - X-axis: d (nm), range [0, 2.5 nm]
   - Y-axis: Q_net (e nm⁻²)
   - Trace 1: Q_net data, dark blue solid line, label = "Q_net(d)"
   - Trace 2: Exponential fit in the fitting window only, plotted as **orange dashed line**,
     label = f"Fit: ζ = {zeta_fit:.3f} ± {zeta_err:.3f} nm"
   - Extend the fitted curve slightly beyond the fitting window (10%) as a lighter dotted line
     to show where the extrapolation goes
   - Shade the fitting window region with a light orange transparent band (alpha=0.15)
   - Add a horizontal dashed grey line at Q_net = 0
   - Text box (upper right, inside axes): display
     ```
     ζ_fit  = {zeta_fit:.3f} ± {zeta_err:.3f} nm
     ζ_PB   = {zeta_PB:.3f} nm
     Ratio  = {zeta_fit/zeta_PB:.2f}
     R²     = {r_squared:.4f}
     ```
   - Title: `{system_label} — Debye Length Fitting`

7. **Figure 3 (comparison bar chart):** If results from all three systems are available
   (check for qnet_data.npz in all three system folders listed in config), produce a grouped
   bar chart comparing ζ_fit for each composition with a horizontal line at ζ_PB.
   - Otherwise, produce a single-system bar chart showing ζ_fit vs ζ_PB side by side.
   - Error bars from zeta_err.

8. Save:
   - `analysis_out/fig_debye_fit.png` and `.pdf`
   - `analysis_out/fig_debye_comparison.png` and `.pdf` (if multi-system)
   - Append to `analysis_out/results_summary.txt`:
     ```
     === Debye Length Results ===
     System:   {system_label}
     ζ_fit:    {zeta_fit:.4f} ± {zeta_err:.4f} nm
     ζ_PB:     {zeta_PB:.4f} nm
     Ratio:    {ratio:.4f}
     R²:       {r2:.4f}
     Fit window: [{debye_fit_min_nm}, {debye_fit_max_nm}] nm
     ```

---

## 8. Python Script 4: Normalised RDF — `scripts/06_plot_rdf_normalised.py`

### Purpose
Load the raw (un-normalised) surface-distance RDF output from `gmx rdf -surf`, apply the
linear baseline normalisation, and plot the normalised RDF for Na⁺ and Cl⁻.

### Implementation requirements

1. Load config. Load:
   - `analysis_out/RDF_Na_surf.xvg` → r (nm), RDF_Na (un-normalised)
   - `analysis_out/RDF_Cl_surf.xvg` → r (nm), RDF_Cl (un-normalised)

2. **Linear normalisation for each ion species separately:**
   ```python
   def normalise_rdf(r, rdf, bulk_min_nm, bulk_max_nm):
       """
       Fit a linear function to rdf in the bulk region [bulk_min_nm, bulk_max_nm].
       Divide the entire rdf by the fitted function evaluated at all r.
       Return rdf_norm where bulk region ≈ 1.0.
       """
       mask = (r >= bulk_min_nm) & (r <= bulk_max_nm)
       coeffs = np.polyfit(r[mask], rdf[mask], deg=1)
       baseline = np.polyval(coeffs, r)
       # Avoid division by zero or negative baseline
       baseline = np.clip(baseline, 1e-10, None)
       return rdf / baseline
   ```
   Use `rdf_bulk_fit_min_nm` and `rdf_bulk_fit_max_nm` from config.

3. **Plot specification:**
   - Single panel with Na⁺ and Cl⁻ on same axes
   - Na⁺: blue solid line
   - Cl⁻: red solid line
   - Horizontal dashed grey line at y = 1.0 (bulk reference)
   - X-axis: r (nm from PO4 surface), range [0, 2.5 nm]
   - Y-axis: `Normalised RDF (a.u.)`, range [0, max(rdf_norm) * 1.15]
   - Title: `{system_label} — Surface-Distance RDF (Na⁺ and Cl⁻)`
   - Shade the bulk fitting region with a light grey band, label "normalisation region"
   - Annotate the first peak of Na⁺ with an arrow and text: "first adsorption peak"
     (detect as argmax of rdf_norm_Na in range r = 0.2 to 0.8 nm)

4. Save as `analysis_out/fig_rdf_normalised.png` and `analysis_out/fig_rdf_normalised.pdf`.

---

## 9. Master Script: `run_all.sh`

Create a master bash script that runs the entire pipeline in order for a given system path.

```bash
#!/bin/bash
# Usage: bash run_all.sh /path/to/system [begin_time_ps]
# Runs the full ion analysis pipeline for one system.

set -e  # exit on any error

SYSTEM_PATH=${1:?"ERROR: Provide system path as first argument"}
BEGIN=${2:-100000}

echo "============================================"
echo " Ion Analysis Pipeline"
echo " System: ${SYSTEM_PATH}"
echo " Analysis start time: ${BEGIN} ps"
echo "============================================"

echo ""
echo "[Step 0] Building GROMACS index file..."
bash scripts/00_prepare_gromacs.sh "${SYSTEM_PATH}"

echo ""
echo "[Step 1] Running gmx density..."
bash scripts/01_run_gromacs_density.sh "${SYSTEM_PATH}" "${BEGIN}"

echo ""
echo "[Step 2] Running gmx rdf -surf..."
bash scripts/02_run_gromacs_rdf.sh "${SYSTEM_PATH}" "${BEGIN}"

echo ""
echo "[Step 3] Plotting density profiles..."
python3 scripts/03_plot_density_profiles.py

echo ""
echo "[Step 4] Computing Q_net(z)..."
python3 scripts/04_compute_qnet.py

echo ""
echo "[Step 5] Fitting Debye length..."
python3 scripts/05_fit_debye_length.py

echo ""
echo "[Step 6] Plotting normalised RDF..."
python3 scripts/06_plot_rdf_normalised.py

echo ""
echo "============================================"
echo " Pipeline complete."
echo " All outputs in: ${SYSTEM_PATH}/analysis_out/"
echo "============================================"
```

---

## 10. Single-Leaflet / Two-Leaflet Geometry: Critical Implementation Notes

This is the most important correctness concern for a bilayer simulation.

### The problem
A bilayer has two leaflets, each facing a bulk water region on opposite sides. If you naively
integrate ρ_Na(z) from z = 0 (box edge) to z = L (other box edge), you will include ions from
BOTH water regions, which belong to different leaflets. This would double-count and mix up the
two double layers.

### The solution used here (ρ(z) folding)
Because `-center` in `gmx density` places the bilayer COM at z = 0 (box centre):
- The two PO4 peaks sit symmetrically at ±z_PO4
- The two water regions are at z < −z_PO4 and z > +z_PO4
- The bilayer interior (hydrophobic core) is at −z_PO4 < z < +z_PO4

**Folding procedure (implemented in utils.py as `fold_density_profile`):**
```python
def fold_density_profile(z, rho):
    """
    Fold a symmetric bilayer density profile onto the positive-z half.
    Assumes bilayer COM is at z = 0 (ensured by gmx density -center).
    Returns (z_pos, rho_folded) where z_pos >= 0.
    rho_folded(z) = 0.5 * (rho(z) + rho(-z)) for z >= 0.
    """
    # Find the index of z = 0 (or closest to 0)
    z_pos_mask = z >= 0
    z_pos = z[z_pos_mask]
    rho_pos = rho[z_pos_mask]
    # Mirror: for each z > 0, find corresponding -z bin
    rho_neg_mirrored = np.interp(z_pos, -z[::-1], rho[::-1])
    rho_folded = 0.5 * (rho_pos + rho_neg_mirrored)
    return z_pos, rho_folded
```

After folding, integrate outward from z_PO4 (the PO4 peak) toward positive z (into the water).
This integration correctly captures only one leaflet's ionic double layer.

### Does gmx rdf -surf have the same problem?
For the surface-distance RDF: yes and no. The `-surf` flag computes the minimum distance between
each ion and the nearest PO4 bead. An ion in the upper water region will find its nearest PO4 in
the upper leaflet; an ion in the lower water region will find its nearest PO4 in the lower leaflet.
So there is NO cross-leaflet mixing in the RDF — the minimum-distance assignment naturally
assigns each ion to its own leaflet. This is one of the reasons `-surf` is well-suited to bilayer
analysis.

---

## 11. Python Dependencies

Add a `requirements.txt` to `ion_analysis/`:

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
pathlib  # stdlib, listed for clarity
```

Install with: `pip install -r requirements.txt`

---

## 12. Error Handling and Validation

Every Python script must include the following checks. Raise informative `FileNotFoundError` or
`ValueError` with clear messages — never silent failures.

| Check | Where | Error message template |
|---|---|---|
| `config.yaml` exists | All scripts | `"config.yaml not found. Run from ion_analysis/ root."` |
| `system_path` folder exists | All scripts | `"system_path '{path}' does not exist. Check config.yaml."` |
| `analysis_out/*.xvg` files exist | Scripts 03–06 | `"'{file}' not found. Run step N first."` |
| PO4 peak detected | Scripts 04–05 | `"Cannot detect PO4 peak. Check density_PO4.xvg is non-empty."` |
| Q_net bulk neutrality | Script 04 | `"WARNING: Q_net bulk residual = {val:.3f} e/nm². Expected < 0.05."` |
| Curve fit convergence | Script 05 | `"WARNING: curve_fit did not converge. Adjust debye_fit_min/max_nm in config.yaml."` |
| Both leaflets symmetric | Script 04 | Print symmetry diagnostic: difference between left and right PO4 peak positions. |

---

## 13. Example Usage Workflow (for the user)

```bash
# 1. Clone or create the ion_analysis/ folder
cd ion_analysis/

# 2. Edit config.yaml with your system path and parameters

# 3. Verify GROMACS ion bead names FIRST:
gmx dump -s /path/to/system/topol.tpr | grep -iE '"NA|"CL|"SOD|"CLA' | sort -u

# 4. Run the full pipeline
bash run_all.sh /path/to/POPC_POPS_system 100000

# 5. Inspect fig_density_profiles.png and fig_qnet.png

# 6. If needed, adjust fitting windows in config.yaml:
#    debye_fit_min_nm: 0.7   ← increase if fit includes contact layer
#    debye_fit_max_nm: 1.8   ← decrease if noisy at large d

# 7. Re-run only the fitting script (no need to re-run GROMACS):
python3 scripts/05_fit_debye_length.py

# 8. Repeat for each composition by changing system_path in config.yaml
```

---

## 14. Output Summary for Each Run

After a successful run, `analysis_out/` will contain:

| File | Contents |
|---|---|
| `density_Na.xvg` | Raw gmx density output for Na⁺ |
| `density_Cl.xvg` | Raw gmx density output for Cl⁻ |
| `density_PO4.xvg` | Raw gmx density output for PO4 |
| `density_CHOL.xvg` | Raw gmx density output for CHOL head (if present) |
| `RDF_Na_surf.xvg` | Raw gmx rdf -surf output for Na⁺ |
| `RDF_Cl_surf.xvg` | Raw gmx rdf -surf output for Cl⁻ |
| `fig_density_profiles.png/.pdf` | ρ(z) plot — log scale |
| `fig_qnet.png/.pdf` | Q_net(d) with fitting window marked |
| `fig_debye_fit.png/.pdf` | Exponential fit on Q_net with text box |
| `fig_debye_comparison.png/.pdf` | Bar chart: ζ_fit vs ζ_PB (all systems) |
| `fig_rdf_normalised.png/.pdf` | Normalised surface-distance RDF |
| `qnet_data.npz` | Numpy archive of Q_net arrays |
| `results_summary.txt` | Text log of all numerical results |

---

## 15. Notes for Cursor

- **Do not** hardcode any paths, bead names, or numerical parameters — all must come from `config.yaml`.
- **Do not** merge scripts 03–06 into one file. The modular structure is intentional and required.
- All scripts must be runnable as `python3 scripts/NN_scriptname.py` from the `ion_analysis/` root
  directory (no command-line arguments needed; they read from `config.yaml`).
- Shell scripts (`.sh`) are callable with a system path as argument.
- Use f-strings throughout. Do not use `%` formatting.
- All matplotlib figures must call `utils.set_publication_style()` before creating any axes.
- Every script must print a clear start and end message to stdout, e.g.:
  `"[03] Plotting density profiles..." ` and `"[03] Done. Saved to analysis_out/fig_density_profiles.png"`
- Add a `if __name__ == "__main__":` guard to every Python script.
- Use `pathlib.Path` for all file path operations — no `os.path.join`.
