#!/usr/bin/env python3
"""Hookium error statistics by correlation regime (R2-1, revision 1).

Purpose: an unweighted mean over the twenty frequencies could be dominated by the largest omega.
This script writes three SI tables (\\input by SI.tex):
  Table S7  MAREs of W_inf / W'_inf for all 20 omega, strongly (omega < 0.04) and weakly
            (omega >= 0.04) correlated subsets, and the log-omega-integrated mean
            (1/ln(w_max/w_min)) * int |err| d ln(omega) (trapezoid), which weights every decade equally.
  Table S8  the same four statistics for the SPL and genISI2 correlation energies vs FCI.
  Table S9  numerical accuracy: (a) FCI vs Taut exact energies (mE_h and %); (b) W_inf/sqrt(omega),
            W'_inf/omega^(3/4), their % deviation from the exact Gaussian (omega -> infinity) limits, E_c.
Same definitions and data as Figs. 1 and 2 (PyOEP density, FCI rows of merged_hooke_results.csv).
Output: ../../SI_hookium_regime_tables.tex. Prints the numbers quoted in the manuscript.
"""
from pathlib import Path
import numpy as np

import plot_hook_wy_pyoep as h

HERE = Path(__file__).resolve().parent
MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
NAMES = {"R2SCAN": r"r$^2$SCAN"}
W_CRIT = 0.04
# exact omega -> infinity limits for the Gaussian density 2(omega/pi)^(3/2) exp(-omega r^2)
WINF_LIM, WP_LIM = -1.151617, 0.403058  # exact: Vee_SCE 0.444152 - U 2*sqrt(2/pi); see SI S15
# Taut's analytic ground states: omega_n and E = (n + 3) omega
TAUT = {0.5: 2.0, 0.1: 0.5, 0.0365373: 6 * 0.036537265599264124}


def stats(omega, relerr):
    """MARE (%) over all, strong, weak, and log-omega integrated."""
    e = 100 * np.abs(relerr)
    lw = np.log(omega)
    strong, weak = omega < W_CRIT, omega >= W_CRIT
    integ = np.trapezoid(e, lw) / (lw[-1] - lw[0])
    return [e.mean(), e[strong].mean(), e[weak].mean(), integ]


def fmt(x):
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def main():
    d = h.load_block(HERE / "merged_hooke_results.csv").drop_duplicates("omega").sort_values("omega")
    om = h.numeric(d, "omega")
    n = lambda c: h.numeric(d, "PYOEP_" + c)
    w0, gl2, ec = n("W0"), 0.5 * n("W0_prime"), n("E_c")
    wi, wp = n("Winf"), n("T_ZP_prl")
    spl = lambda a: np.array([h.spl_corr(x, y, z) for x, y, z in zip(w0, gl2, a)])
    gen = lambda a, b: np.array([h.genisi2_corr(x, y, z, q) for x, y, z, q in zip(w0, gl2, a, b)])
    print(f"omega: {len(om)} points, {np.sum(om < W_CRIT)} strong (< {W_CRIT}), {np.sum(om >= W_CRIT)} weak")

    ing, ene = [], []
    print("ingredient MARE %  [all, strong, weak, log-int]  W_inf | W'_inf ; share of omega=1000 in the 20-point mean")
    for m in MODELS:
        a, b = n("W_inf_" + m), n("Wp_inf_" + m)
        rw, rp = (a - wi) / np.abs(wi), (b - wp) / np.abs(wp)
        sw, sp = stats(om, rw), stats(om, rp)
        ing.append((m, sw + sp))
        print(f"  {m:7s} " + " ".join(f"{x:6.2f}" for x in sw) + " | " + " ".join(f"{x:6.1f}" for x in sp)
              + f" ; share {100*abs(rw[-1])/np.sum(np.abs(rw)):4.1f}% / {100*abs(rp[-1])/np.sum(np.abs(rp)):4.1f}%")
    print("correlation-energy MARE %  [all, strong, weak, log-int]  SPL | genISI2")
    rows = [("SCE", wi, wp)] + [(m, n("W_inf_" + m), n("Wp_inf_" + m)) for m in MODELS]
    for m, a, b in rows:
        ss, sg = stats(om, (spl(a) - ec) / np.abs(ec)), stats(om, (gen(a, b) - ec) / np.abs(ec))
        ene.append((m, ss + sg))
        print(f"  {m:7s} " + " ".join(f"{x:6.2f}" for x in ss) + " | " + " ".join(f"{x:6.2f}" for x in sg))

    print("high-omega: omega, E_FCI, W_inf/sqrt(w), W'_inf/w^0.75, E_c")
    hi = [1.0, 10.0, 100.0, 1000.0]
    hrows = []
    for w in hi:
        i = int(np.argmin(np.abs(om - w)))
        hrows.append((w, wi[i] / np.sqrt(w), wp[i] / w**0.75, 1000 * ec[i]))
        print(f"  {w:7g} {wi[i]/np.sqrt(w):9.4f} ({100*(wi[i]/np.sqrt(w)-WINF_LIM)/abs(WINF_LIM):+.2f} %) {wp[i]/w**0.75:8.4f} ({100*(wp[i]/w**0.75-WP_LIM)/WP_LIM:+.2f} %) {1000*ec[i]:8.2f}")
    print("Taut exact energies vs FCI (mE_h):")
    E = h.numeric(d, "E_ref")
    trows = []
    for w, ex in TAUT.items():
        i = int(np.argmin(np.abs(om - w)))
        trows.append((w, ex, E[i], 1000 * (E[i] - ex)))
        print(f"  omega {w}: exact {ex:.6f} FCI {E[i]:.6f} diff {1000*(E[i]-ex):+.3f} mE_h ({100*(E[i]-ex)/ex:+.4f} %)")

    sub = r" & all & $\omega<0.04$ & $\omega\ge0.04$ & $\ln\omega$-int."
    head = lambda a, b: [
        r"\begin{tabular}{l cccc cccc}", r"\toprule",
        rf" & \multicolumn{{4}}{{c}}{{{a}}} & \multicolumn{{4}}{{c}}{{{b}}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        "Model" + sub + sub.replace(" & all", " & all", 1) + r" \\", r"\midrule"]
    t7 = head(r"$W_\infty$", r"$W'_\infty$") + [
        f"{NAMES.get(m, m):9s} & " + " & ".join(fmt(x) for x in v) + r" \\" for m, v in ing] + [r"\bottomrule", r"\end{tabular}"]
    t8 = head("SPL", "genISI2") + [
        f"{('exact SCE' if m == 'SCE' else NAMES.get(m, m)):9s} & " + " & ".join(fmt(x) for x in v) + r" \\"
        + (r" \midrule" if m == "SCE" else "") for m, v in ene] + [r"\bottomrule", r"\end{tabular}"]
    t9 = [r"\begin{tabular}{r ccc ccc c}", r"\toprule",
          r" & \multicolumn{3}{c}{$W_\infty/\sqrt{\omega}$} & \multicolumn{3}{c}{$W'_\infty/\omega^{3/4}$} & \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"$\omega$ & this work & analytic & dev.\ (\%) & this work & analytic & dev.\ (\%) & $E_c^{\rm FCI}$ (m$E_h$) \\",
          r"\midrule"]
    t9 += [f"{w:g} & ${a:.4f}$ & ${WINF_LIM:.4f}$ & ${100*(a-WINF_LIM)/abs(WINF_LIM):+.2f}$ & ${b:.4f}$ & ${WP_LIM:.4f}$ & ${100*(b-WP_LIM)/WP_LIM:+.2f}$ & ${c:.2f}$ \\\\"
           for w, a, b, c in hrows]
    t9 += [r"\bottomrule", r"\end{tabular}"]
    t9b = [r"\begin{tabular}{r cccc}", r"\toprule",
           r"$\omega$ & $E$ this work, FCI ($E_h$) & $E$ analytic ($E_h$) & error (m$E_h$) & error (\%) \\", r"\midrule"]
    t9b += [f"{w:g} & {e:.6f} & {ex:.6f} & ${dd:+.3f}$ & ${100*dd/1000/ex:+.3f}$ \\\\" for w, ex, e, dd in trows]
    t9b += [r"\bottomrule", r"\end{tabular}"]
    out = HERE.parent.parent / "SI_hookium_regime_tables.tex"
    out.write_text("% Generated by data_JPCL/hookium/make_hookium_regime_tables.py -- do not edit by hand.\n"
                   + "\\def\\hookTabSeven{%\n" + "\n".join(t7) + "}\n"
                   + "\\def\\hookTabEight{%\n" + "\n".join(t8) + "}\n"
                   + "\\def\\hookTabNineA{%\n" + "\n".join(t9b) + "}\n"
                   + "\\def\\hookTabNineB{%\n" + "\n".join(t9) + "}\n")
    print("written:", out)


if __name__ == "__main__":
    main()
