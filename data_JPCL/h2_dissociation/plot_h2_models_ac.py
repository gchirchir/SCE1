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
LABELS = ["SCE", "LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]  # as in Figs. 2 and 5


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

    # same style as main-text Figs. 2 and 5 (plot_he_series_mare_spl_genisi2_wy_pyoep.py)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": 40, "xtick.labelsize": 34,
        "ytick.labelsize": 34, "legend.fontsize": 30, "axes.linewidth": 2.5,
        "xtick.major.width": 2.5, "ytick.major.width": 2.5, "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5, "xtick.major.size": 12, "ytick.major.size": 12,
        "xtick.minor.size": 6, "ytick.minor.size": 6, "xtick.direction": "out",
        "ytick.direction": "out", "hatch.linewidth": 1.8, "savefig.dpi": 800,
        "savefig.bbox": "tight"})
    from matplotlib import ticker, transforms
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 12))
    x = np.arange(len(t)); w = 0.38
    ax.bar(x - w / 2, t["MAE_SPL"], w, color="#334155", edgecolor="black", linewidth=1.5,
           hatch="////", alpha=0.94, zorder=3)
    ax.bar(x + w / 2, t["MAE_gen"], w, color="#d81b60", edgecolor="black", linewidth=1.5,
           hatch="\\\\", alpha=0.94, zorder=3)
    ax.set_xlim(-0.5, len(t) - 0.5)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 700.0)  # headroom above the bars for the legend
    ax.yaxis.set_major_locator(ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=2.6, zorder=4)
    ax.axvline(4.5, color="#333333", linestyle="--", linewidth=2.6, zorder=4)
    ax.tick_params(axis="x", which="major", top=False, bottom=True)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(2.5, 1.04, "Semilocal", transform=trans, ha="center", va="bottom",
            fontsize=40, fontweight="bold")
    ax.text(7.0, 1.04, "PC", transform=trans, ha="center", va="bottom",
            fontsize=40, fontweight="bold")
    ax.set_ylabel(r"MAE vs FCI [m$E_h$]", labelpad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=40, ha="right", rotation_mode="anchor")
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=0)
    p_spl = mpatches.Patch(facecolor="#334155", edgecolor="black", linewidth=1.5, hatch="////",
                           alpha=0.94, label="SPL")
    p_gen = mpatches.Patch(facecolor="#d81b60", edgecolor="black", linewidth=1.5,
                           hatch="\\\\", alpha=0.94, label="genISI2")
    leg = ax.legend(handles=[p_spl, p_gen], loc="upper left", bbox_to_anchor=(0.10, 1.0),
                    ncol=2, frameon=True, framealpha=0.0, handlelength=2.0, handleheight=1.2,
                    title=None, title_fontsize=30)
    leg.get_frame().set_linewidth(0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("eps", "png"):
        fig.savefig(HERE / f"H2_models_AC_MAE_PYOEP.{ext}", dpi=300)


if __name__ == "__main__":
    main()
