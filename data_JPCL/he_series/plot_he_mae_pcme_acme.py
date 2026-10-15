#!/usr/bin/env python3
"""Absolute-error (MAE) counterpart of Fig. 6 for the He isoelectronic series (R2-8, revision 1).

Same quantities as plot_oep_wy.py (Fig. 6), taken from its calculate(), but in absolute units:
    PCME_abs = -(E_c[model ingredients] - E_c[exact SCE ingredients])   (main panels)
    ACME_abs = -(E_c[exact SCE ingredients] - E_c^FCI)                   (insets)
in mE_h (left axis) and kcal/mol (right axis); the grey band marks +-1 kcal/mol.
FCI rows, PyOEP density, all ions present in merged_results.csv. Prints the numbers quoted in the
manuscript. No in-figure title.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import MODELS, LABELS, MARKERS, LINESTYLES, style
from plot_oep_wy import Z_MAP, calculate

HERE = Path(__file__).resolve().parent
KCAL = 627.509474
MEH_PER_KCAL = 1000.0 / KCAL


def main():
    df = pd.read_csv(HERE / "merged_results.csv")
    df = df[df.system.isin(Z_MAP) & df.source_folder.ne("he_se")].copy()
    df["Z"] = df.system.map(Z_MAP)
    block = df[df.density_type.eq("FCI")].sort_values("Z").copy()
    v = calculate(block, "PYOEP")
    first = v[v.model.eq(MODELS[0])].sort_values("Z")
    Z = first.Z.to_numpy()
    acme_spl = -1000 * first.ACME_SPL.to_numpy()
    acme_gen = -1000 * first.ACME_genISI2.to_numpy()

    # ---- numbers for the text ----
    ec = first.E_c_ref.to_numpy()
    print("ions:", list(first.system))
    print("1 kcal/mol = %.4f mE_h; |E_c^FCI| range %.1f - %.1f mE_h" % (MEH_PER_KCAL, 1000*np.abs(ec).min(), 1000*np.abs(ec).max()))
    for lab, a in (("SPL", acme_spl), ("genISI2", acme_gen)):
        print(f"exact SCE {lab:7s}: MAE {np.mean(np.abs(a)):.2f} mE_h ({np.mean(np.abs(a))/MEH_PER_KCAL:.2f} kcal/mol), "
              f"max |err| {np.max(np.abs(a)):.2f} mE_h at Z={Z[np.argmax(np.abs(a))]}, range {a.min():+.2f}..{a.max():+.2f}")
    print("model totals vs FCI, MAE mE_h (kcal/mol) | max |err| mE_h  [SPL ; genISI2]  | PCME MAE SPL/gen")
    for m in MODELS:
        q = v[v.model.eq(m)].sort_values("Z")
        ts = 1000 * (q.E_c_spl - q.E_c_ref).to_numpy(); tg = 1000 * (q.E_c_genISI2 - q.E_c_ref).to_numpy()
        ps = 1000 * q.PCME_SPL.to_numpy(); pg = 1000 * q.PCME_genISI2.to_numpy()
        print(f"  {m:7s} SPL {np.mean(np.abs(ts)):5.2f} ({np.mean(np.abs(ts))/MEH_PER_KCAL:4.2f}) max {np.max(np.abs(ts)):5.2f} ;"
              f" gen {np.mean(np.abs(tg)):5.2f} ({np.mean(np.abs(tg))/MEH_PER_KCAL:4.2f}) max {np.max(np.abs(tg)):5.2f} |"
              f" PCME {np.mean(np.abs(ps)):.2f}/{np.mean(np.abs(pg)):.2f}")

    # ---- figure (same layout as Fig. 6) ----
    style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 28), sharex=True)
    handles = []
    for m, lab, mk, ls in zip(MODELS, LABELS, MARKERS, LINESTYLES):
        q = v[v.model.eq(m)].sort_values("Z")
        h, = ax_top.plot(q.Z, -1000 * q.PCME_SPL, marker=mk, linestyle=ls, linewidth=4.0,
                         markersize=14, markeredgewidth=1.5, label=lab)
        handles.append(h)
        ax_bot.plot(q.Z, -1000 * q.PCME_genISI2, marker=mk, linestyle=ls, linewidth=4.0,
                    markersize=14, markeredgewidth=1.5, label=lab)
    for ax in (ax_top, ax_bot):
        ax.axhline(0.0, color="black", linewidth=1.8)
        ax.axhspan(-MEH_PER_KCAL, MEH_PER_KCAL, color="0.85", zorder=0)
        ax.set_xticks(range(int(Z.min()), int(Z.max()) + 1))
        ax.tick_params(axis="x", labelsize=24)  # REV1 (R2-6): 18 labels must not collide
        ax.set_ylabel(r"absolute error [m$E_h$]")
        sec = ax.secondary_yaxis("right", functions=(lambda y: y / MEH_PER_KCAL, lambda y: y * MEH_PER_KCAL))
        sec.set_ylabel("[kcal/mol]")
    ax_bot.set_xlabel(r"Nuclear charge, $Z$")
    ax_top.text(0.02, 0.96, "(a) SPL", transform=ax_top.transAxes, ha="left", va="top", fontsize=30, fontweight="bold")
    ax_bot.text(0.02, 0.96, "(b) genISI2", transform=ax_bot.transAxes, ha="left", va="top", fontsize=30, fontweight="bold")
    for ax, vals, method, color, pos in ((ax_top, acme_spl, "SPL", "tab:blue", [0.50, 0.55, 0.46, 0.33]),
                                         (ax_bot, acme_gen, "genISI2", "tab:red", [0.50, 0.09, 0.46, 0.24])):
        ins = ax.inset_axes(pos)
        ins.axhline(0.0, color="black", linewidth=1.6)
        ins.axhspan(-MEH_PER_KCAL, MEH_PER_KCAL, color="0.85", zorder=0)
        ins.plot(Z, vals, "o-", linewidth=3.0, markersize=9, markerfacecolor="white",
                 markeredgecolor="tab:blue", markeredgewidth=1.3, color=color, label=method)
        ins.set_xlabel(r"$Z$", fontsize=26, labelpad=4)
        ins.set_ylabel(r"m$E_h$", fontsize=26, labelpad=4)
        ins.tick_params(axis="both", which="major", labelsize=22, direction="in", length=7, width=1.8)
        ins.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black",
                   fontsize=17, handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    ax_top.legend(handles=handles, labels=[x.get_label() for x in handles], loc="lower right",
                  frameon=True, facecolor="white", edgecolor="black", ncol=2, fontsize=26,
                  handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()
    out = HERE / "He_iso_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps"
    fig.savefig(out, format="eps")
    fig.savefig(out.with_suffix(".png"), dpi=60)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
