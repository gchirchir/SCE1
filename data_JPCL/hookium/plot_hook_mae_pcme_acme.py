#!/usr/bin/env python3
"""Absolute-error (MAE) counterpart of Fig. 3 for Hookium (R2-4, revision 1).

Same quantities as plot_hook_wy_pyoep.py (Fig. 3), but in absolute units instead of
relative to |E_c^FCI|:
    PCME_abs = -(E_c[model ingredients] - E_c[exact SCE ingredients])   (main panels)
    ACME_abs = -(E_c[exact SCE ingredients] - E_c^FCI)                   (insets)
in mE_h (left axis) and kcal/mol (right axis); the shaded band marks +-1 kcal/mol.
Uses the corrected merged_hooke_results.csv (omega = 0.05, 0.06 W'_inf fixed, R2-2).
Prints the per-model MAEs quoted in the manuscript. No in-figure title.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullFormatter

import plot_hook_wy_pyoep as h

HERE = Path(__file__).resolve().parent
KCAL = 627.509474           # kcal/mol per E_h
MEH_PER_KCAL = 1000.0 / KCAL  # 1 kcal/mol = 1.5936 mE_h


def absolute_errors(data, source="PYOEP"):
    p = f"{source}_"
    num = lambda c: h.numeric(data, p + c)
    omega = h.numeric(data, "omega")
    w0, gl2, ec = num("W0"), 0.5 * num("W0_prime"), num("E_c")
    wi, wp = num("Winf"), num("T_ZP_prl")
    spl = lambda a: np.array([h.spl_corr(x, y, z) for x, y, z in zip(w0, gl2, a)])
    gen = lambda a, b: np.array([h.genisi2_corr(x, y, z, q) for x, y, z, q in zip(w0, gl2, a, b)])
    s0, g0 = spl(wi), gen(wi, wp)
    out = {"omega": omega,
           "acme_spl": -1000 * (s0 - ec), "acme_gen": -1000 * (g0 - ec),
           "total": {"SCE": (1000 * (s0 - ec), 1000 * (g0 - ec))}, "models": {}}
    for m in h.MODELS:
        a, b = num("W_inf_" + m), num("Wp_inf_" + m)
        s, g = spl(a), gen(a, b)
        out["models"][m] = {"spl": -1000 * (s - s0), "gen": -1000 * (g - g0)}
        out["total"][m] = (1000 * (s - ec), 1000 * (g - ec))
    return out


def main():
    data = h.load_block(HERE / "merged_hooke_results.csv").drop_duplicates("omega")
    r = absolute_errors(data)
    om = r["omega"]

    # ---- numbers for the text ----
    print("Total error vs FCI, MAE in mE_h (kcal/mol); 1 kcal/mol = %.4f mE_h" % MEH_PER_KCAL)
    for k, (es, eg) in r["total"].items():
        ms, mg = np.mean(np.abs(es)), np.mean(np.abs(eg))
        print(f"  {k:7s} SPL {ms:5.2f} ({ms/MEH_PER_KCAL:4.2f})  genISI2 {mg:5.2f} ({mg/MEH_PER_KCAL:4.2f})"
              f"  max|err| SPL {np.max(np.abs(es)):5.2f} gen {np.max(np.abs(eg)):5.2f}")
    print("ACME (exact SCE) mE_h: SPL range %.2f..%.2f, genISI2 range %.2f..%.2f" % (
        r["acme_spl"].min(), r["acme_spl"].max(), r["acme_gen"].min(), r["acme_gen"].max()))
    print("PCME MAE (mE_h) SPL / genISI2: " + ", ".join(
        f"{m} {np.mean(np.abs(v['spl'])):.2f}/{np.mean(np.abs(v['gen'])):.2f}" for m, v in r["models"].items()))

    # ---- figure (same layout as Fig. 3) ----
    h.set_style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 28), sharex=False)
    handles = []
    for model, label, marker, ls in zip(h.MODELS, h.LABELS, h.MARKERS, h.LINESTYLES):
        v = r["models"][model]
        line, = ax_top.plot(om, v["spl"], marker=marker, linestyle=ls, linewidth=4.0,
                            markersize=14, markeredgewidth=1.5, label=label)
        handles.append(line)
        ax_bot.plot(om, v["gen"], marker=marker, linestyle=ls, linewidth=4.0,
                    markersize=14, markeredgewidth=1.5, label=label)
    for ax in (ax_top, ax_bot):
        ax.set_xscale("log")
        ax.set_xlim(0.025, 1200)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(0.0, color="black", linewidth=1.8)
        ax.axhspan(-MEH_PER_KCAL, MEH_PER_KCAL, color="0.85", zorder=0)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"absolute error [m$E_h$]")
        sec = ax.secondary_yaxis("right", functions=(lambda y: y / MEH_PER_KCAL,
                                                     lambda y: y * MEH_PER_KCAL))
        sec.set_ylabel("[kcal/mol]")
    ax_top.text(0.02, 0.96, "(a) SPL", transform=ax_top.transAxes, ha="left", va="top",
                fontsize=30, fontweight="bold")
    ax_bot.text(0.02, 0.96, "(b) genISI2", transform=ax_bot.transAxes, ha="left", va="top",
                fontsize=30, fontweight="bold")

    for ax, vals, method, color, pos in (
            (ax_top, r["acme_spl"], "SPL", "tab:blue", [0.46, 0.62, 0.40, 0.33]),
            (ax_bot, r["acme_gen"], "genISI2", "tab:red", [0.63, 0.04, 0.34, 0.21])):
        ins = ax.inset_axes(pos)
        ins.axhline(0.0, color="black", linewidth=1.6)
        ins.axhspan(-MEH_PER_KCAL, MEH_PER_KCAL, color="0.85", zorder=0)
        ins.plot(om, vals, "o-", linewidth=3.0, markersize=9, markerfacecolor="white",
                 markeredgecolor="tab:blue", markeredgewidth=1.3, color=color, label=method)
        ins.set_xscale("log")
        ins.set_xlabel(r"$\omega$", fontsize=26, labelpad=6)
        ins.set_ylabel(r"m$E_h$", fontsize=26, labelpad=6)
        ins.tick_params(axis="both", which="major", labelsize=22, direction="in", length=7, width=1.8)
        ins.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black",
                   fontsize=17, handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    ax_top.legend(handles=handles, labels=[x.get_label() for x in handles], loc="lower right",
                  frameon=True, facecolor="white", edgecolor="black", ncol=2, fontsize=28,
                  handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()
    out = HERE / "HOOK_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps"
    fig.savefig(out, format="eps")
    fig.savefig(out.with_suffix(".png"), dpi=60)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
