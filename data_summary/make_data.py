#!/usr/bin/env python3
"""Build the slim reference tables in data_summary/ from the full data package data_JPCL/.

Each output CSV keeps only identifiers plus the three strong-coupling reference quantities
    W_0     (weak-coupling limit W_0[rho] = E_x)
    W_inf   (exact SCE W_inf[rho])
    W'_inf  (exact SCE zero-point coefficient W'_inf[rho]; column T_ZP_prl in data_JPCL)
All model columns (LDA, PBE, TPSS, r2SCAN, PC, mPC, hPC, fPC, ePC, ...) are dropped.
Energies in Hartree, 6 decimals.

usage:  python3 make_data.py [--src ../data_JPCL] [--out .]
"""
import argparse, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(Path(__file__).resolve().parent.parent / "data_JPCL"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
Z = {s: i + 1 for i, s in enumerate("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
     "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())}
W = {"PYOEP_W0": "W_0", "PYOEP_Winf": "W_inf", "PYOEP_T_ZP_prl": "W'_inf"}


def save(df, rel):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format="%.6f")
    print(f"{rel:40s} {len(df):4d} rows  {list(df.columns)}")


def split_folder(folder):
    """'Be2_plus_5ZaPa-NR-CV' -> ('Be2+', '5ZaPa-NR-CV');  'H_minus_aug-cc-pV6Z' -> ('H-', ...)"""
    m = re.match(r"^([A-Z][a-z]?)(\d*)_(plus|minus)_(.+)$", folder)
    if m:
        el, q, sign, basis = m.groups()
        return el + q + ("+" if sign == "plus" else "-"), basis
    el, basis = folder.split("_", 1)
    return el, basis


# Hookium (two electrons in a harmonic trap), FCI target density
d = pd.read_csv(src / "hookium/merged_hooke_results.csv")
save(d[["omega", "density_type"] + list(W)].rename(columns=W)
     .sort_values("omega"), "hookium/hookium_W0_Winf_Wpinf.csv")

# He isoelectronic series H- ... Ar16+, FCI target density
d = pd.read_csv(src / "he_series/merged_results.csv")
sp = d.source_folder.map(split_folder)
d["system"], d["basis"] = sp.str[0], sp.str[1]
d["Z"] = d.source_folder.str.extract(r"^([A-Z][a-z]?)")[0].map(Z)
save(d[["Z", "system", "basis", "density_type"] + list(W)].rename(columns=W)
     .sort_values("Z"), "he_series/he_series_W0_Winf_Wpinf.csv")

# Atoms and ions H ... Zn, CCSD(T) target density (CCSD for Zn)
d = pd.read_csv(src / "atoms_ions/merged_results.csv")
d["basis"] = d.source_folder.map(lambda f: split_folder(f)[1])
d["Z"] = d.system.str.rstrip("+-").map(Z)
d["N_electrons"] = d.Z - d.system.str.count(r"\+") + d.system.str.count("-")
save(d[["Z", "system", "N_electrons", "basis", "density_type"] + list(W)].rename(columns=W)
     .sort_values(["Z", "N_electrons"]), "atoms_ions/atoms_ions_W0_Winf_Wpinf.csv")

# H2 dissociation, FCI target density; W_inf = dual-Kantorovich (cylindrical),
# W'_inf = restricted-mode zero-point (rmZPE) value, as in the manuscript
d = pd.read_csv(src / "h2_dissociation/final_merged_dou_sig.csv")
d = d[["R_bohr", "density_type", "PYOEP_W0", "PYOEP_winf_cyl", "PYOEP_winf_prime_rmzpe_cyl"]]
save(d.rename(columns={"PYOEP_W0": "W_0", "PYOEP_winf_cyl": "W_inf",
                       "PYOEP_winf_prime_rmzpe_cyl": "W'_inf"}).sort_values("R_bohr"),
     "h2_dissociation/h2_W0_Winf_Wpinf.csv")

# Ne basis-set series (SI Table S4), CCSD(T) target density; W_0 not stored for these runs
rows = []
for f in sorted((src / "tables/tableS4_ne_ccsdt_runs").glob("Ne_*/data/batch_summary_pc4.csv")):
    r = pd.read_csv(f).iloc[0]
    rows.append({"system": "Ne", "basis": f.parts[-3].split("_", 1)[1], "density_type": "CCSDT",
                 "W_inf": r.W_inf_SCE, "W'_inf": r.Wp_inf_SCE})
save(pd.DataFrame(rows), "ne_basis/ne_basis_Winf_Wpinf.csv")
