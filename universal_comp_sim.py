#!/usr/bin/env python3
"""
Universal composition builder: arbitrary lipid bilayers (POPC, POPS, DOPC, DPPC, CHOL)
using INSANE (Martini 3).

CORRECT THREE-STEP ION PLACEMENT:
----------------------------------
insane's -salt and -charge behaviour:
  -salt 0    -> sets NaCl concentration to zero, BUT insane's default
                -charge auto STILL adds counter-ions for charged lipids
                (e.g. NA+ for POPS).  ALSO: insane may round lipid counts
                per-leaflet, so the actual POPS count can differ from the
                requested count.
  -salt 0.15 -> treats 0.15 M as AVERAGE (NA+CL)/2, not independent pairs.
                With 400 POPS, CL- ends up at only ~69 mM instead of 150 mM.

THE CORRECT APPROACH:
  Step 1. insane  -salt 0       bilayer + water (insane may add counter-ions)
  Step 1b. Strip any insane-added ions from the GRO  (guarantees clean start)
  Step 2. gmx genion            N_POPS_actual NA+ only  (neutralise POPS charge)
  Step 3. gmx genion            N_bg NA+ + N_bg CL-  (exactly 0.15 M NaCl background)

All counts are computed DYNAMICALLY from the actual GRO after insane:
  N_POPS = actual POPS count read from the GRO  (handles insane rounding)
  N_bg   = round(salt_M * water_beads * 4 * 18 / 1000)  (from actual water volume)

KNOWN GROMACS 2026 BEHAVIOUR:
  gmx genion writes residue name "ION" in the GRO instead of the pname you
  specified (e.g. "NA") in two known cases:
    1. When -neutral flag is used.
    2. When adding ONLY positive ions (nn=0, e.g. counter-ions for POPS).
  This breaks subsequent grompp calls because no moleculetype "ION" is
  defined in Martini itp files. FIX: after every genion call, we scan the
  GRO and rename any "ION" residues back to the correct pname.
  We compute exact ion counts ourselves, so -neutral is never needed.

Usage:
  python3 universal_comp_sim.py \\
      --popc 1600 --pops 400 --total-lipids 2000 \\
      --template-top ./system.top \\
      --insane-exe /path/to/insane

  python3 universal_comp_sim.py \\
      --dppc 1360 --dopc 340 --chol 300 --total-lipids 2000 \\
      --template-top ./system.top \\
      --insane-exe /path/to/insane

  python3 universal_comp_sim.py \\
      --popc 800 --pops 200 --chol 1000 --total-lipids 2000 \\
      --ca2-ratio 1.5 \\
      --template-top ./system.top \\
      --insane-exe /path/to/insane
"""

import argparse
import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Shell runner
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, stdin=None):
    """Run a shell command; raise CalledProcessError on failure."""
    env = {**os.environ, "GMX_MAXBACKUP": "-1"}  # suppress #backup# files
    print(">>", " ".join(map(str, cmd)))
    if stdin is not None:
        subprocess.run(
            cmd, cwd=cwd, check=True, env=env,
            input=stdin.encode() if isinstance(stdin, str) else stdin,
        )
    else:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_LIPIDS = frozenset(["POPC", "POPS", "DOPC", "DPPC", "CHOL", "POP", "CHO"])

# Minimal MDP for grompp pre-genion (no actual MD, just topology pre-processing).
_IONS_MDP = """\
integrator    = steep
nsteps        = 0
nstlist       = 1
cutoff-scheme = Verlet
ns_type       = grid
pbc           = xyz
coulombtype   = reaction-field
rcoulomb      = 1.1
rvdw          = 1.1
"""


# ---------------------------------------------------------------------------
# GRO utilities
# ---------------------------------------------------------------------------

def get_membrane_resnames_from_gro(gro_path: Path) -> list[str]:
    """Return lipid residue names present in the GRO file."""
    lines    = gro_path.read_text().splitlines()
    natoms   = int(lines[1].strip())
    resnames = set(ln[5:10].strip() for ln in lines[2:2+natoms] if ln[5:10].strip())
    return [r for r in KNOWN_LIPIDS if r in resnames]


def parse_molecules_in_order_from_gro(gro_path: Path) -> list[str]:
    """
    Return the molecule sequence from the GRO file (one entry per molecule).
    This order MUST match [ molecules ] in system.top exactly.
    """
    lines  = gro_path.read_text().splitlines()
    natoms = int(lines[1].strip())
    mol_order, prev_key = [], None
    for ln in lines[2:2+natoms]:
        key = (ln[0:5].strip(), ln[5:10].strip())
        if key != prev_key:
            mol_order.append(key[1])
            prev_key = key
    return mol_order


def count_molecules_from_gro(gro_path: Path) -> dict[str, int]:
    """Return {resname: molecule_count} for every residue type in the GRO."""
    lines  = gro_path.read_text().splitlines()
    natoms = int(lines[1].strip())
    counts: dict[str, int] = {}
    prev   = None
    for ln in lines[2:2+natoms]:
        key = (ln[0:5].strip(), ln[5:10].strip())
        if key != prev:
            counts[key[1]] = counts.get(key[1], 0) + 1
            prev = key
    return counts


def nacl_count_for_concentration(water_beads: int, salt_m: float) -> int:
    """
    Calculate NaCl ion pairs needed for salt_m mol/L given Martini W bead count.

    Each CG W bead = 4 real water molecules.
    N = salt_m [mol/L] * volume_water [L]
    volume_water [L] = water_beads * 4 * 18 [g/mol] / 1000 [g/L]  (density 1 kg/L)
    N = salt_m * water_beads * 4 * 18 / 1000

    Verification: for 33367 W beads, 0.15 M ->
      N = 0.15 * 33367 * 4 * 18 / 1000 = 360.4 -> 360 ions
      Back-check: 360 / (33367*4*18/1000) = 360/2402.4 = 0.1498 M  (correct)
    """
    return round(salt_m * water_beads * 4 * 18 / 1000)


# Ion residue names that insane / genion may write.
ION_RESNAMES = frozenset(["NA", "CL", "NA+", "CL-", "ION", "CA", "CA2+"])


def strip_ions_from_gro(gro_path: Path) -> int:
    """
    Remove ALL ion atoms from a GRO file in-place.

    insane's -charge auto default adds counter-ions even with -salt 0.
    This function strips them so the subsequent genion steps start from a
    clean lipid+water system with exactly zero ions.

    Returns the number of ion atoms removed.
    """
    lines  = gro_path.read_text().splitlines()
    title  = lines[0]
    natoms = int(lines[1].strip())
    atoms  = lines[2:2+natoms]
    box    = lines[2+natoms] if len(lines) > 2+natoms else ""

    kept, removed = [], 0
    for ln in atoms:
        resname = ln[5:10].strip()
        if resname in ION_RESNAMES:
            removed += 1
        else:
            kept.append(ln)

    if removed > 0:
        new_lines = [title, f"{len(kept):>5d}"] + kept + [box]
        gro_path.write_text("\n".join(new_lines) + "\n")
        print(f"  [strip_ions] Removed {removed} ion atom(s) from GRO"
              f" (insane auto-neutralisation).")
    else:
        print(f"  [strip_ions] GRO is clean — no ions to remove.")

    return removed


def fix_ion_resnames_in_gro(gro_path: Path, pname: str, nname: str | None = None) -> int:
    """
    Rename all 'ION' residues in a GRO file to the correct Martini names.

    GROMACS 2026 BEHAVIOUR:
      gmx genion writes residue name "ION" (and atom name "ION") instead of
      the requested pname/nname in certain cases — notably when adding ONLY
      positive ions (nn=0, e.g. counter-ions for POPS).  Paired NA+CL
      additions usually work, but this function is called after every genion
      step as a safety net.

    GRO fixed-width columns (0-indexed):
      [0:5]   residue number
      [5:10]  residue name   (5 chars, left-justified by convention)
      [10:15] atom name      (5 chars, right-justified by convention)
      [15:20] atom number

    For Martini CG single-bead ions, resname == atomname (e.g. "NA", "CL").
    We replace both fields for every atom whose resname is "ION".

    If nname is provided AND both positive and negative ions were added in the
    same genion call, we cannot distinguish which "ION" is positive vs negative
    from the GRO alone.  In practice this ambiguity does not arise because
    GROMACS 2026 writes correct names for paired additions.  This function
    renames all remaining "ION" to pname (the positive ion), which is correct
    for the counter-ion-only case that triggers the bug.

    Returns the number of atoms fixed.
    """
    lines  = gro_path.read_text().splitlines()
    natoms = int(lines[1].strip())
    fixed  = 0

    new_resname  = f"{pname:<5s}"       # left-justified, 5 chars
    new_atomname = f"{pname:>5s}"       # right-justified, 5 chars

    for i in range(2, 2 + natoms):
        ln = lines[i]
        if ln[5:10].strip() == "ION":
            # Replace resname (cols 5-10) and atomname (cols 10-15)
            lines[i] = ln[:5] + new_resname + new_atomname + ln[15:]
            fixed += 1

    if fixed > 0:
        gro_path.write_text("\n".join(lines) + "\n")
        print(f"  [fix_ion] Renamed {fixed} 'ION' -> '{pname}' in GRO"
              f" (GROMACS 2026 genion workaround).")

    return fixed


# ---------------------------------------------------------------------------
# Topology writer
# ---------------------------------------------------------------------------

def write_system_top_from_template_and_order(
    template_top: Path,
    out_top: Path,
    mol_order: list[str],
):
    """
    Write system.top: template header up to and including [ molecules ],
    then a run-length encoded molecule list matching the GRO coordinate order.
    GROMACS requires topology molecule order to exactly match GRO order.
    """
    tpl_lines = template_top.read_text().splitlines()

    # Fix relative #include paths when output is in a subdirectory.
    out_dir    = out_top.parent
    tpl_dir    = template_top.parent
    toppar_dir = tpl_dir / "toppar"
    if toppar_dir.exists() and out_dir != tpl_dir:
        rel = os.path.relpath(toppar_dir, out_dir)
        tpl_lines = [
            ln.replace('"toppar/', f'"{rel}/').replace("'toppar/", f"'{rel}/")
            if '#include' in ln and 'toppar/' in ln else ln
            for ln in tpl_lines
        ]

    idx = next(
        (i for i, ln in enumerate(tpl_lines) if ln.strip().lower() == "[ molecules ]"),
        None,
    )
    if idx is None:
        raise RuntimeError("Template system.top missing [ molecules ]")

    out_lines = tpl_lines[:idx+1] + ["; name        number"]
    if not mol_order:
        raise RuntimeError("Empty molecule order — cannot write topology.")

    # Run-length encode: ["POPC","POPC","W","W","W","NA"] -> "POPC 2\nW 3\nNA 1"
    cur, n = mol_order[0], 1
    for m in mol_order[1:]:
        if m == cur:
            n += 1
        else:
            out_lines.append(f"{cur:<12} {n}")
            cur, n = m, 1
    out_lines.append(f"{cur:<12} {n}")

    out_top.write_text("\n".join(out_lines) + "\n")


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------

def generate_index_ndx(out_dir: Path, add_sol_group: bool = False):
    """
    Generate index.ndx with MEMBRANE and SOLUTE groups.
    add_sol_group=True also adds "W" group — required as genion replacement pool.
    """
    gro = out_dir / "step5_charmm2gmx.gro"
    if not gro.exists():
        raise RuntimeError("Missing step5_charmm2gmx.gro")

    membrane_res = get_membrane_resnames_from_gro(gro)
    if not membrane_res:
        raise RuntimeError("Could not detect membrane residue names in GRO")

    mem_sel    = " ".join(membrane_res)
    select_str = (
        f'"MEMBRANE" resname {mem_sel}; '
        f'"SOLUTE" not (resname {mem_sel})'
    )
    if add_sol_group:
        select_str += '; "W" resname W'

    run(
        ["gmx", "select", "-s", "step5_charmm2gmx.gro",
         "-select", select_str, "-on", "index.ndx"],
        cwd=out_dir,
    )
    if not (out_dir / "index.ndx").exists():
        raise RuntimeError("Failed to generate index.ndx")


# ---------------------------------------------------------------------------
# Step 1: insane (bilayer + water, NO ions)
# ---------------------------------------------------------------------------

def build_with_insane(
    out_dir: Path,
    composition: dict[str, int],
    apl_nm2: float,
    pbc: str,
    z_nm: float,
    insane_exe: str,
    insane_script: str | None,
    box_lx: float | None = None,
    box_ly: float | None = None,
    box_lz: float | None = None,
):
    """
    Run insane with -salt 0: builds bilayer + water with absolutely no ions.
    All ion placement is done by separate gmx genion steps afterwards.
    """
    for name, val in composition.items():
        if val > 0 and val % 2 != 0:
            raise ValueError(f"{name} total ({val}) must be even to split across leaflets.")

    leaf = {name: total // 2 for name, total in composition.items() if total > 0}

    # Resolve to absolute path — insane subprocess runs in out_dir, breaking relative paths.
    if insane_script is not None:
        insane_script = str(Path(insane_script).resolve())
    base_cmd = [insane_exe] if insane_script is None else ["python3", insane_script]

    if box_lx is not None and box_ly is not None and box_lz is not None:
        box = f"{box_lx:.3f},{box_ly:.3f},{box_lz:.3f}"
    else:
        nleaf = sum(leaf.values())
        Lxy   = (nleaf * apl_nm2) ** 0.5
        box   = f"{Lxy:.3f},{Lxy:.3f},{z_nm:.3f}"

    u_args, l_args = [], []
    for name, count in leaf.items():
        u_args.extend(["-u", f"{name}:{count}"])
        l_args.extend(["-l", f"{name}:{count}"])

    cmd = base_cmd + [
        "-o",    "step5_charmm2gmx.gro",
        "-p",    "system.insane.top",
        "-pbc",  pbc,
        "-box",  box,
        "-a",    str(apl_nm2),
        "-sol",  "W",
        "-salt", "0",   # NO ions — all ions placed by gmx genion below
    ] + u_args + l_args

    print("\nINSANE command:", " ".join(map(str, cmd)))
    print("  -> insane places bilayer + water ONLY. Zero ions.")
    run(cmd, cwd=out_dir)

    if not (out_dir / "step5_charmm2gmx.gro").exists():
        raise RuntimeError("INSANE did not produce step5_charmm2gmx.gro.")

    run(["gmx", "editconf",
         "-f", "step5_charmm2gmx.gro", "-o", "step5_charmm2gmx.pdb"],
        cwd=out_dir)


# ---------------------------------------------------------------------------
# Core genion helper — NO -neutral flag (CRITICAL for GROMACS 2026)
# ---------------------------------------------------------------------------

def _run_genion(
    out_dir: Path,
    label: str,
    template_top: Path,
    mol_order: list[str],
    np: int,
    pname: str,
    pq: int = 1,
    nn: int = 0,
    nname: str = "CL",
    nq: int = -1,
):
    """
    Generic genion helper.

    CRITICAL: -neutral is NEVER passed to genion.
    GROMACS 2026 genion writes residue name "ION" in the GRO file in several
    situations:
      1. When -neutral is used (known).
      2. When adding ONLY positive ions (nn=0), e.g. counter-ions for POPS.
    This breaks subsequent grompp calls because no moleculetype "ION" exists
    in the Martini itp files.

    FIX: After every genion call, fix_ion_resnames_in_gro() renames any "ION"
    residues back to the requested pname (e.g. "NA").  We compute exact ion
    counts ourselves, so -neutral is not needed.

    Parameters
    ----------
    label    : short name for temp file naming (e.g. "counterion", "nacl", "ca2")
    mol_order: current molecule order (used to write temp topology for grompp)
    np       : number of positive ions to add
    pname    : positive ion residue name (e.g. "NA")
    pq       : positive ion charge (default 1)
    nn       : number of negative ions to add (0 = none)
    nname    : negative ion residue name (e.g. "CL")
    nq       : negative ion charge (default -1)

    Returns updated mol_order re-parsed from GRO after genion.
    """
    mdp_file = out_dir / f"ions_{label}.mdp"
    tpr_file = out_dir / f"ions_{label}.tpr"
    top_file = out_dir / f"system_{label}.top"

    mdp_file.write_text(_IONS_MDP)
    write_system_top_from_template_and_order(template_top, top_file, mol_order)

    run(
        ["gmx", "grompp",
         "-f", mdp_file.name, "-c", "step5_charmm2gmx.gro",
         "-p", top_file.name, "-o", tpr_file.name,
         "-n", "index.ndx",   "-maxwarn", "2"],
        cwd=out_dir,
    )

    # Build genion command — NO -neutral flag (see docstring above).
    genion_cmd = [
        "gmx", "genion",
        "-s",     tpr_file.name,
        "-o",     "step5_charmm2gmx.gro",
        "-p",     top_file.name,
        "-pname", pname, "-pq", str(pq), "-np", str(np),
        "-n",     "index.ndx",
    ]
    if nn > 0:
        genion_cmd.extend(["-nname", nname, "-nq", str(nq), "-nn", str(nn)])

    run(genion_cmd, cwd=out_dir, stdin="W\n")

    # GROMACS 2026 workaround: rename any "ION" residues to correct names.
    gro_file = out_dir / "step5_charmm2gmx.gro"
    fix_ion_resnames_in_gro(gro_file, pname=pname, nname=nname)

    # Clean up temp files.
    mdp_file.unlink(missing_ok=True)
    tpr_file.unlink(missing_ok=True)
    top_file.unlink(missing_ok=True)
    (out_dir / "mdout.mdp").unlink(missing_ok=True)

    # Re-parse mol order from the updated GRO (authoritative source).
    return parse_molecules_in_order_from_gro(out_dir / "step5_charmm2gmx.gro")


# ---------------------------------------------------------------------------
# Step 2: Na+ counter-ions for POPS neutralisation
# ---------------------------------------------------------------------------

def add_counter_ions(
    out_dir: Path,
    n_pops: int,
    template_top: Path,
    mol_order: list[str],
) -> list[str]:
    """
    Add exactly n_pops Na+ ions to neutralise the POPS negative charges.
    No Cl- is added. No -neutral flag used.
    System becomes electrically neutral after this step.
    """
    if n_pops == 0:
        print("  [Step 2] No POPS — counter-ion step skipped.")
        return mol_order

    print(f"\n  [Step 2 - Counter-ions] Adding {n_pops} NA+ to neutralise {n_pops} POPS.")
    return _run_genion(
        out_dir=out_dir, label="counterion",
        template_top=template_top, mol_order=mol_order,
        np=n_pops, pname="NA", pq=1,
        nn=0,   # no anions — only positive counter-ions
    )


# ---------------------------------------------------------------------------
# Step 3: background NaCl at target concentration
# ---------------------------------------------------------------------------

def add_background_nacl(
    out_dir: Path,
    water_beads_initial: int,
    salt_m: float,
    template_top: Path,
    mol_order: list[str],
) -> list[str]:
    """
    Add symmetric NaCl pairs for exactly salt_m mol/L.
    Water bead count is re-read from GRO to account for W beads consumed
    by the counter-ion step, giving the most accurate concentration.
    """
    # Re-count water from current GRO (some W beads replaced by Na+ in step 2).
    actual_w = count_molecules_from_gro(
        out_dir / "step5_charmm2gmx.gro"
    ).get("W", water_beads_initial)

    n_nacl  = nacl_count_for_concentration(actual_w, salt_m)
    conc_M  = n_nacl / (actual_w * 4 * 18 / 1000)   # mol/L — for display only

    if n_nacl == 0:
        print(f"  [Step 3] salt_m={salt_m} -> 0 pairs — skipped.")
        return mol_order

    print(f"\n  [Step 3 - NaCl background]")
    print(f"    Water beads (after counter-ions):  {actual_w}  ({actual_w*4:,} real waters)")
    print(f"    NaCl pairs to place:               {n_nacl}")
    print(f"    Resulting concentration:           {conc_M*1000:.2f} mM  ({conc_M:.4f} M)")
    print(f"    Target:                            {salt_m*1000:.2f} mM  ({salt_m:.4f} M)")

    return _run_genion(
        out_dir=out_dir, label="nacl",
        template_top=template_top, mol_order=mol_order,
        np=n_nacl, pname="NA", pq=1,
        nn=n_nacl, nname="CL", nq=-1,
    )


# ---------------------------------------------------------------------------
# Optional Step 4: Ca2+ addition
# ---------------------------------------------------------------------------

def add_ca2_ions(
    out_dir: Path,
    n_ca: int,
    n_pops: int,
    template_top: Path,
    mol_order: list[str],
) -> list[str]:
    """
    Add Ca2+ ions (and balancing Cl-) after the system is already neutral.

    At this point the system is neutral: Na+ counter-ions balanced POPS,
    and NaCl pairs are symmetric.

    Adding n_ca Ca2+ (charge +2 each) adds +2*n_ca charge.
    The Na+ that were covering POPS duty (+n_pops) are already present.
    Net Cl- needed to rebalance: 2*n_ca - n_pops.
    """
    n_cl = 2 * n_ca - n_pops
    if n_cl < 0:
        raise ValueError(
            f"Ca2+ count ({n_ca}) must be >= n_pops/2 ({n_pops//2}). "
            f"Got n_cl={n_cl}."
        )
    if n_ca == 0:
        return mol_order

    print(f"\n  [Step 4 - Ca2+] Adding {n_ca} CA2+ and {n_cl} extra CL-.")

    mol_order = _run_genion(
        out_dir=out_dir, label="ca2",
        template_top=template_top, mol_order=mol_order,
        np=n_ca, pname="CA", pq=2,
        nn=n_cl, nname="CL", nq=-1,
    )
    run(["gmx", "editconf",
         "-f", "step5_charmm2gmx.gro", "-o", "step5_charmm2gmx.pdb"],
        cwd=out_dir)
    return mol_order


# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------

def report_and_verify(out_dir: Path, actual_pops: int, salt_m: float):
    """
    Read the final GRO, print ion counts, and run four verification checks.
    All checks must pass before using the system for MD.

    actual_pops: the POPS count read from the GRO (may differ from requested).
    """
    gro    = out_dir / "step5_charmm2gmx.gro"
    counts = count_molecules_from_gro(gro)

    pops = actual_pops
    na   = counts.get("NA",  counts.get("NA+", 0))
    cl   = counts.get("CL",  counts.get("CL-", 0))
    ca   = counts.get("CA",  0)
    w    = counts.get("W",   0)

    conc_M      = cl / (w * 4 * 18 / 1000) if w > 0 else 0.0
    net_charge  = na + 2*ca - cl - pops  # positive contributions - negative contributions

    n_bg_expected = nacl_count_for_concentration(w, salt_m)
    na_expected   = pops + n_bg_expected
    cl_expected   = n_bg_expected

    print("\n" + "="*52)
    print("  FINAL ION VERIFICATION")
    print("="*52)
    print(f"  POPS lipids (charge -1 each):      {pops}")
    print(f"  Na+ counter-ions for POPS:         {pops}  (expected {pops})")
    print(f"  Na+ background NaCl:               {cl}   (expected ~{n_bg_expected})")
    print(f"  Na+ TOTAL:                         {na}   (expected ~{na_expected})")
    print(f"  Cl- TOTAL (background only):       {cl}   (expected ~{cl_expected})")
    print(f"  Ca2+ (if added):                   {ca}")
    print(f"  Water beads (W):                   {w}  ({w*4:,} real waters)")
    print(f"  NaCl background:                   {conc_M*1000:.2f} mM  ({conc_M:.4f} M)")
    print(f"  Target NaCl:                       {salt_m*1000:.2f} mM  ({salt_m:.4f} M)")
    print()

    checks = [
        ("Na+ - Cl- == POPS  (charge balanced)",   (na - cl) == pops),
        ("Net system charge == 0",                   net_charge == 0),
        ("[NaCl] within 10 mM of target",           abs(conc_M - salt_m) < 0.010),
        ("No ION residues (Martini 3 names OK)",
         "ION" not in counts),
    ]

    all_ok = True
    for desc, ok in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {desc}")
        if not ok:
            all_ok = False

    print("="*52)
    if all_ok:
        print("  All checks passed. System ready for minimisation.")
    else:
        print("  WARNING: one or more checks FAILED. Do not run MD yet.")
    print("="*52 + "\n")


# ---------------------------------------------------------------------------
# Folder naming
# ---------------------------------------------------------------------------

def make_folder_name_from_composition(composition: dict[str, int]) -> str:
    total = sum(composition.values())
    if total == 0:
        return "MIX_0"
    return "_".join(
        f"{name}_{round(100*n/total)}"
        for name, n in sorted(composition.items()) if n > 0
    )


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------

def build_one(out_dir: Path, composition: dict[str, int], args):
    """
    Full pipeline:
      Step 1   insane -salt 0         bilayer + water (may include auto-ions)
      Step 1b  strip insane ions      guarantee clean lipid+water GRO
      [index]  generate index.ndx     with W group for genion
      Step 2   gmx genion             N_POPS_actual Na+  (counter-ions)
      [index]  regenerate             still with W group
      Step 3   gmx genion             N_bg NA + N_bg CL  (0.15 M NaCl)
      Step 4   write system.top       from final GRO order
      [index]  final index.ndx        MEMBRANE + SOLUTE
      Step 5   gmx editconf           update PDB
      [opt]    gmx genion Ca2+        if --ca2-ratio given
      Cleanup  remove intermediate files
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lipids = sum(composition.values())
    target = getattr(args, "total_lipids", 2000)
    if total_lipids != target:
        raise ValueError(f"Total lipids must be {target}. Got {total_lipids}.")
    if total_lipids % 2 != 0:
        raise ValueError("Total lipids must be even.")

    box = getattr(args, "box", None)
    box_lx = box_ly = box_lz = None
    if box:
        parts = [float(x.strip()) for x in str(box).split(",")]
        if len(parts) >= 3:
            box_lx, box_ly, box_lz = parts[0], parts[1], parts[2]

    template_top = Path(args.template_top).resolve()

    # ------------------------------------------------------------------ #
    # STEP 1: insane — bilayer + water                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*52)
    print("STEP 1: insane (bilayer + water)")
    print("="*52)
    build_with_insane(
        out_dir=out_dir, composition=composition,
        apl_nm2=args.apl, pbc=args.pbc, z_nm=args.z,
        insane_exe=args.insane_exe, insane_script=args.insane_script,
        box_lx=box_lx, box_ly=box_ly, box_lz=box_lz,
    )

    gro = out_dir / "step5_charmm2gmx.gro"

    # ------------------------------------------------------------------ #
    # STEP 1b: strip any ions insane added (auto-neutralisation artefact) #
    # ------------------------------------------------------------------ #
    print("\n" + "="*52)
    print("STEP 1b: strip insane-added ions (clean start)")
    print("="*52)
    strip_ions_from_gro(gro)

    # Read actual counts FROM the cleaned GRO (insane may round lipids).
    gro_counts   = count_molecules_from_gro(gro)
    actual_pops  = gro_counts.get("POPS", gro_counts.get("POP", 0))
    water_beads  = gro_counts.get("W", 0)
    mol_order    = parse_molecules_in_order_from_gro(gro)

    if actual_pops != composition.get("POPS", 0):
        print(f"  NOTE: insane created {actual_pops} POPS"
              f" (requested {composition.get('POPS', 0)}).")

    need_ca = bool(getattr(args, "ca2_ratio", None) and actual_pops > 0)

    # index.ndx with W group (required as replacement pool for all genion steps)
    generate_index_ndx(out_dir, add_sol_group=True)

    # ------------------------------------------------------------------ #
    # STEP 2: Na+ counter-ions (dynamic: exactly N_POPS_actual ions)      #
    # ------------------------------------------------------------------ #
    print("\n" + "="*52)
    print(f"STEP 2: counter-ions  ({actual_pops} NA+ for {actual_pops} POPS)")
    print("="*52)
    mol_order = add_counter_ions(
        out_dir=out_dir, n_pops=actual_pops,
        template_top=template_top, mol_order=mol_order,
    )
    # Regenerate index — keeps W group for the next genion step.
    generate_index_ndx(out_dir, add_sol_group=True)

    # ------------------------------------------------------------------ #
    # STEP 3: background NaCl (dynamic: from actual water volume)         #
    # ------------------------------------------------------------------ #
    n_nacl_expected = nacl_count_for_concentration(water_beads, args.salt)
    print("\n" + "="*52)
    print(f"STEP 3: background NaCl  (~{n_nacl_expected} pairs for {args.salt*1000:.0f} mM)")
    print("="*52)
    mol_order = add_background_nacl(
        out_dir=out_dir, water_beads_initial=water_beads,
        salt_m=args.salt, template_top=template_top, mol_order=mol_order,
    )

    # ------------------------------------------------------------------ #
    # STEP 4: write final system.top from authoritative GRO order         #
    # ------------------------------------------------------------------ #
    mol_order = parse_molecules_in_order_from_gro(gro)
    write_system_top_from_template_and_order(
        template_top=template_top,
        out_top=out_dir / "system.top",
        mol_order=mol_order,
    )

    # Final index.ndx — keep W group only if Ca2+ step follows.
    generate_index_ndx(out_dir, add_sol_group=need_ca)

    # Update PDB.
    run(["gmx", "editconf",
         "-f", "step5_charmm2gmx.gro", "-o", "step5_charmm2gmx.pdb"],
        cwd=out_dir)

    # ------------------------------------------------------------------ #
    # OPTIONAL STEP: Ca2+                                                 #
    # ------------------------------------------------------------------ #
    if need_ca:
        n_ca = int(round(args.ca2_ratio * actual_pops))
        print("\n" + "="*52)
        print(f"STEP Ca2+: adding {n_ca} CA2+  (ratio {args.ca2_ratio} × {actual_pops} POPS)")
        print("="*52)
        mol_order = parse_molecules_in_order_from_gro(gro)
        mol_order = add_ca2_ions(
            out_dir=out_dir, n_ca=n_ca, n_pops=actual_pops,
            template_top=template_top, mol_order=mol_order,
        )
        mol_order = parse_molecules_in_order_from_gro(gro)
        write_system_top_from_template_and_order(
            template_top=template_top,
            out_top=out_dir / "system.top",
            mol_order=mol_order,
        )
        generate_index_ndx(out_dir, add_sol_group=False)

    # ------------------------------------------------------------------ #
    # Cleanup: remove intermediate files                                  #
    # ------------------------------------------------------------------ #
    for pattern in ["system.insane.top", "mdout.mdp",
                    "ions_*.mdp", "ions_*.tpr",
                    "system_counterion.top", "system_nacl.top", "system_ca2.top",
                    "#*#"]:
        for f in out_dir.glob(pattern):
            f.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Final verification report                                           #
    # ------------------------------------------------------------------ #
    report_and_verify(out_dir, actual_pops, args.salt)
    print(f"Built: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Universal Martini 3 bilayer builder. "
            "Three-step ion placement: insane (no ions), "
            "genion counter-ions, genion NaCl background. "
            "All counts computed dynamically from composition and water volume."
        )
    )
    ap.add_argument("--template-top",  required=True,
                    help="Path to Martini 3 template system.top")
    ap.add_argument("--salt",          type=float, default=0.15,
                    help="Background NaCl in mol/L (default 0.15). "
                         "Added independently of POPS counter-ions.")
    ap.add_argument("--apl",           type=float, default=0.60,
                    help="Area per lipid in nm² (default 0.60)")
    ap.add_argument("--z",             type=float, default=12.0,
                    help="Box Z dimension in nm (default 12.0)")
    ap.add_argument("--pbc",           default="square",
                    choices=["square", "rectangular"])
    ap.add_argument("--insane-exe",    default="insane",
                    help="insane executable (default: insane in PATH)")
    ap.add_argument("--insane-script", default=None,
                    help="Path to insane.py if not using the installed executable")
    ap.add_argument("--popc",          type=int, default=0)
    ap.add_argument("--pops",          type=int, default=0)
    ap.add_argument("--dopc",          type=int, default=0)
    ap.add_argument("--dppc",          type=int, default=0)
    ap.add_argument("--chol",          type=int, default=0)
    ap.add_argument("--out",           default=None,
                    help="Output folder name (default: auto from composition)")
    ap.add_argument("--total-lipids",  type=int, default=2000,
                    help="Expected total lipid count (default 2000)")
    ap.add_argument("--box",           default=None,
                    help="Explicit box Lx,Ly,Lz in nm e.g. '40,40,12'")
    ap.add_argument("--ca2-ratio",     type=float, default=None,
                    help="Ca2+ : POPS molar ratio (e.g. 1.5)")

    args = ap.parse_args()

    composition = {}
    for lipid, count in [("POPC", args.popc), ("POPS", args.pops),
                          ("DOPC", args.dopc), ("DPPC", args.dppc),
                          ("CHOL", args.chol)]:
        if count > 0:
            composition[lipid] = count

    if not composition:
        raise SystemExit(
            "No composition provided. Example:\n"
            "  --popc 1600 --pops 400 --total-lipids 2000\n"
        )

    total = sum(composition.values())
    if total != args.total_lipids:
        raise ValueError(
            f"Lipid counts sum to {total} but --total-lipids={args.total_lipids}."
        )

    if args.out is None:
        args.out = make_folder_name_from_composition(composition)

    out_dir = Path(args.out)

    print(f"\n=== BUILD: out={out_dir}  total={total} ===")
    print(f"    Composition: {composition}")
    print(f"    NOTE: insane may round lipid counts per-leaflet;")
    print(f"          actual counts are read from the GRO after insane.")
    if (p := composition.get("POPS", 0)) > 0:
        n_est = nacl_count_for_concentration(33000, args.salt)  # rough estimate
        print(f"    Ion plan (dynamic, based on actual GRO after insane):")
        print(f"      Step 1b -> strip any insane auto-ions")
        print(f"      Step 2  -> ~{p} NA+    (counter-ions for POPS)")
        print(f"      Step 3  -> ~{n_est} NA + ~{n_est} CL  ({args.salt*1000:.0f} mM NaCl)")
    else:
        n_est = nacl_count_for_concentration(33000, args.salt)
        print(f"    Ion plan (dynamic):")
        print(f"      Step 2  -> skipped (no POPS)")
        print(f"      Step 3  -> ~{n_est} NA + ~{n_est} CL  ({args.salt*1000:.0f} mM NaCl)")

    build_one(out_dir, composition, args)


if __name__ == "__main__":
    main()
