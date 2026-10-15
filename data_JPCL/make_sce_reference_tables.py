#!/usr/bin/env python3
"""SI tables of the reference W_inf and W'_inf for Hookium, the He series and H2 (R2-iii, revision 1).

Purpose: tabulate the SCE reference data for Hookium, the He series and H2 in the SI, with literature
values, in addition to the CSV files. This script writes
../SI_sce_reference_tables.tex (\\input by SI.tex), defining \\sceTabHook, \\sceTabHe, \\sceTabHtwo:
  Hookium  all 20 omega, FCI rows of hookium/merged_hooke_results.csv, PyOEP density (PYOEP_Winf,
           PYOEP_T_ZP_prl); exact analytic (Taut) density from hookium/taut_exact_density_sce.py; literature: Smiga et al. 2022 (SCE data of Kooi and Gori-Giorgi 2018) at
           Taut's analytic omega.
  He       H- ... Ar16+, FCI rows, PyOEP density, 5ZaPa-NR-CV (--he CSV, default he_series/merged_results.csv);
           ions missing from the CSV are printed as "--"; literature as in main-text Table 1.
  H2       FCI density (aug-cc-pV5Z); W_inf dual-Kantorovich (winf_cyl), W'_inf rmZPE
           (winf_prime_rmzpe_cyl); selected R in the window 0.3 <= R < 7.66262 bohr used for Figs. 7-9.
Values are printed to 6 decimals, as in Table 1 of the main text.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HE_IONS = [("H-", 1, r"H$^-$"), ("He", 2, "He"), ("Li", 3, r"Li$^+$"), ("Be", 4, r"Be$^{2+}$"),
           ("B", 5, r"B$^{3+}$"), ("C", 6, r"C$^{4+}$"), ("N", 7, r"N$^{5+}$"), ("O", 8, r"O$^{6+}$"),
           ("F", 9, r"F$^{7+}$"), ("Ne", 10, r"Ne$^{8+}$"), ("Na", 11, r"Na$^{9+}$"), ("Mg", 12, r"Mg$^{10+}$"),
           ("Al", 13, r"Al$^{11+}$"), ("Si", 14, r"Si$^{12+}$"), ("P", 15, r"P$^{13+}$"), ("S", 16, r"S$^{14+}$"),
           ("Cl", 17, r"Cl$^{15+}$"), ("Ar", 18, r"Ar$^{16+}$")]
# literature values (E_h); keys are the bib keys used in main.tex
HOOK_LIT = {0.0365373: ("-0.170", "0.022"), 0.1: ("-0.304", "0.054"), 0.5: ("-0.743", "0.208")}
HOOK_LIT_CITE = r"\cite{SCFISI}"
# this work, exact SCE on Taut's analytic density (hookium/taut_exact_density_sce.py)
HOOK_EXACT = {0.0365373: (-0.169884, 0.022415), 0.1: (-0.304219, 0.054142), 0.5: (-0.743153, 0.208057)}
HE_LIT = {"H-": (r"$-0.569$\cite{mirtschink2012energy}, $-0.5689$\cite{seidl07,SCE_CCSD}", ""),
          "He": (r"$-1.5000$\cite{seidl07}, $-1.498$\cite{mirtschink2012energy}, $-1.4982$\cite{SCE_CCSD}",
                 r"$0.62084$\cite{gorigiorgi09}")}
H2_R = [0.3, 0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
H2_CUTOFF = 7.66262


def num(x):
    return f"${x:.6f}$"


def hookium():
    d = pd.read_csv(HERE / "hookium" / "merged_hooke_results.csv")
    d.columns = d.columns.str.strip()
    d = d[d.density_type.astype(str).str.upper() == "FCI"].drop_duplicates("omega").sort_values("omega")
    rows = []
    for _, r in d.iterrows():
        w = float(r.omega)
        lit = next((v for k, v in HOOK_LIT.items() if abs(k - w) < 1e-6), None)
        lw, lp = (f"${lit[0]}$", f"${lit[1]}$") if lit else ("", "")
        ex = next((v for k, v in HOOK_EXACT.items() if abs(k - w) < 1e-6), None)
        ew, ep = (num(ex[0]), num(ex[1])) if ex else ("", "")
        rows.append(f"{w:g} & {num(float(r.PYOEP_Winf))} & {num(float(r.PYOEP_T_ZP_prl))} & {ew} & {ep} & {lw} & {lp} \\\\")
    print(f"Hookium: {len(rows)} omega")
    head = [r"\begin{tabular}{r cc cc cc}", r"\toprule",
            r" & \multicolumn{2}{c}{this work, PyOEP density} & \multicolumn{2}{c}{this work, exact density}"
            r" & \multicolumn{2}{c}{literature" + HOOK_LIT_CITE + r"} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"$\omega$ & $W_\infty$ & $W'_\infty$ & $W_\infty$ & $W'_\infty$ & $W_\infty$ & $W'_\infty$ \\", r"\midrule"]
    return head + rows + [r"\bottomrule", r"\end{tabular}"]


def he_series(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()
    d = d[d.density_type == "FCI"]
    if "source_folder" in d:
        d = d[d.source_folder.ne("he_se")]
    d = d.drop_duplicates("system").set_index("system")
    rows, missing = [], []
    for key, z, label in HE_IONS:
        lw, lp = HE_LIT.get(key, ("", ""))
        if key in d.index:
            r = d.loc[key]
            mark = r"\textsuperscript{a}" if key == "H-" else ""
            rows.append(f"{z} & {label}{mark} & {num(float(r.PYOEP_Winf))} & {num(float(r.PYOEP_T_ZP_prl))} & {lw} & {lp} \\\\")
        else:
            missing.append(key)
            rows.append(f"{z} & {label} & -- & -- & {lw} & {lp} \\\\")
    print(f"He series: {len(HE_IONS) - len(missing)} ions from {path}; missing: {missing or 'none'}")
    head = [r"\begin{tabular}{r l cc cc}", r"\toprule",
            r" & & \multicolumn{2}{c}{this work} & \multicolumn{2}{c}{literature} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"$Z$ & ion & $W_\infty$ & $W'_\infty$ & $W_\infty$ & $W'_\infty$ \\", r"\midrule"]
    return head + rows + [r"\bottomrule", r"\end{tabular}"], missing


def h2():
    sys.path.insert(0, str(HERE / "h2_dissociation"))
    import plot_final_wy_pyoep as pw
    d = pd.read_csv(HERE / "h2_dissociation" / "final_merged_dou_sig.csv", comment="#", skip_blank_lines=True)
    with contextlib.redirect_stdout(io.StringIO()):
        fr = pw.solver_frame(d, "PYOEP")
    fr = fr[(fr.R_bohr >= 0.3) & (fr.R_bohr < H2_CUTOFF)].sort_values("R_bohr")
    rows = []
    for R in H2_R:
        r = fr.iloc[int(np.argmin(np.abs(fr.R_bohr.to_numpy() - R)))]
        wi, wp = float(r.winf_cyl), float(r.winf_prime_rmzpe_cyl)
        rows.append(f"{float(r.R_bohr):.3f} & {num(wi)} & {num(wp)} \\\\")
    print(f"H2: {len(rows)} of {len(fr)} R points in 0.3 <= R < {H2_CUTOFF} bohr")
    head = [r"\begin{tabular}{r cc}", r"\toprule",
            r"$R$ ($a_0$) & $W_\infty$ (dual Kantorovich) & $W'_\infty$ (rmZPE) \\", r"\midrule"]
    tail = [r"\midrule", r"$\infty$ (exact) & $-0.625000$\cite{seidl07} & 0 \\"]
    return head + rows + tail + [r"\bottomrule", r"\end{tabular}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default=str(HERE / "he_series" / "merged_results.csv"))
    a = ap.parse_args()
    th, missing = he_series(a.he)
    out = HERE.parent / "SI_sce_reference_tables.tex"
    out.write_text("% Generated by data_JPCL/make_sce_reference_tables.py -- do not edit by hand.\n"
                   + "\\def\\sceTabHook{%\n" + "\n".join(hookium()) + "}\n"
                   + "\\def\\sceTabHe{%\n" + "\n".join(th) + "}\n"
                   + "\\def\\sceTabHtwo{%\n" + "\n".join(h2()) + "}\n")
    print("written:", out)


if __name__ == "__main__":
    main()
