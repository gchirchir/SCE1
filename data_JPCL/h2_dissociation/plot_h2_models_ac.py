#!/usr/bin/env python3
"""H2: impact of exact vs model strong-coupling ingredients on SPL/genISI2 (R2-9, revision 1).

Same pipeline as Fig. 7 (plot_final_wy_pyoep.solver_frame -> plot_final's exchange
base, GL2 term, and SPL/genISI2 formulas); only the (W_inf, W'_inf) pair is replaced
by each semilocal model's values on the same PyOEP density.  Row selection is that
of Fig. 7: FCI rows, 0.3 <= R < 7.66262 bohr (PyOEP reliability cutoff), and the
same isolated-spike residual filter (R >= 4.5 bohr, 0.01 Ha), applied to every
curve independently on the full R range before the window is applied.
The exact-SCE row reproduces the manuscript values (SPL 3.51, genISI2 7.61 mE_H).

Error split, as for Hookium and the He series:
  ACME = E(exact SCE ingredients) - E_FCI          (interpolation error)
  PCME = E(model ingredients) - E(exact SCE)       (ingredient error propagated)
Outputs: H2_models_AC_MAE_PYOEP.{eps,png}; table printed to stdout.
No in-figure title (description goes in the caption).
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_final as original
import plot_final_wy_pyoep as pw

HERE = Path(__file__).resolve().parent
CUTOFF = 7.66262          # PyOEP reliability cutoff used for Figs. 7-9
MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
LABELS = ["SCE", "LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]


def model_curves(fr):
    """Total SPL/genISI2 energies with exact and with each model's ingredients."""
    with contextlib.redirect_stdout(io.StringIO()):
        exx = original.construct_fci_exact_exchange(fr)
    gl2 = (pd.to_numeric(fr["gl2"], errors="coerce") if "gl2" in fr
           else 0.5 * pd.to_numeric(fr["W0_prime"], errors="coerce"))
    base = fr["E_ac_base_cyl"]
    out = {"SCE": (fr["E_spl_cyl"], fr["E_gisi2_cyl"])}
    for m in MODELS:
        wi = pd.to_numeric(fr[f"W_inf_{m}"], errors="coerce")
        wp = pd.to_numeric(fr[f"Wp_inf_{m}"], errors="coerce")
        spl = [original.spl_correlation(a, b, c) for a, b, c in zip(exx, gl2, wi)]
        gen = [original.genisi2_correlation(a, b, c, d)
               for a, b, c, d in zip(exx, gl2, wi, wp)]
        out[m] = (base + np.asarray(spl, float), base + np.asarray(gen, float))
    return out


def residual_filter(fr, series):
    clean, _ = original.filter_isolated_oscillations(
        fr["R_plot"], pd.to_numeric(series, errors="coerce"), start_x=4.5,
        absolute_tolerance=0.01, scale_factor=0.0, max_passes=20)
    return pd.Series(np.asarray(clean, float), index=fr.index)


def main():
    data = pd.read_csv(HERE / "final_merged_dou_sig.csv", comment="#",
                       skip_blank_lines=True)
    with contextlib.redirect_stdout(io.StringIO()):
        fr = pw.solver_frame(data, "PYOEP")
    # filter the full curves first (as Fig. 7 does), then apply the R window
    curves = {k: tuple(residual_filter(fr, s) for s in v)
              for k, v in model_curves(fr).items()}
    keep = (fr["R_plot"] >= 0.3) & (fr["R_plot"] < CUTOFF)
    fr = fr[keep].copy()
    curves = {k: tuple(s[keep] for s in v) for k, v in curves.items()}
    ref = pd.to_numeric(fr["E_ref"], errors="coerce")
    sce_spl, sce_gen = curves["SCE"]

    rows = []
    for k in ["SCE"] + MODELS:
        spl, gen = curves[k]
        e_spl, e_gen = 1000 * (spl - ref), 1000 * (gen - ref)
        p_spl, p_gen = 1000 * (spl - sce_spl), 1000 * (gen - sce_gen)
        rows.append(dict(model=k,
                         MAE_SPL=e_spl.abs().mean(), MAE_gen=e_gen.abs().mean(),
                         ME_SPL=e_spl.mean(), ME_gen=e_gen.mean(),
                         PCME_SPL=p_spl.mean(), PCME_gen=p_gen.mean(),
                         N=int(np.isfinite(e_spl).sum())))
    t = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 140)
    print("H2, 0.3 <= R < %.5f bohr, energies in mE_H (ME = mean signed error vs FCI;"
          " PCME = mean of E(model) - E(exact SCE))" % CUTOFF)
    print(t.round(2).to_string())

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 15, "axes.linewidth": 1.3})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(t)); w = 0.38
    ax.bar(x - w / 2, t["MAE_SPL"], w, color="#334155", edgecolor="black",
           hatch="////", label="SPL", zorder=3)
    ax.bar(x + w / 2, t["MAE_gen"], w, color="#d81b60", edgecolor="black",
           hatch="\\\\", label="genISI2", zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel(r"MAE vs FCI  [m$E_H$]")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.axvline(0.5, color="#333333", ls="--", lw=1.5)
    ax.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("eps", "png"):
        fig.savefig(HERE / f"H2_models_AC_MAE_PYOEP.{ext}", dpi=300)


if __name__ == "__main__":
    main()
