#!/usr/bin/env python3
"""Signed deviations of the semilocal strong-coupling ingredients from the
exact SCE reference for the Hookium atom, as a function of the confinement
frequency omega.

Produces one two-panel figure per reference density:
    Hook_delta_ingredients_WY.eps      (top: dW_inf,  bottom: dW'_inf)
    Hook_delta_ingredients_PYOEP.eps

Reads merged_hooke_results.csv (the current curated data, FCI rows).
This replaces the older FCI-only Hook_delta_Winf_all_model / Hook_Wprime_inf
scripts so that the SI signed-deviation figures match the WY / PyOEP panels
of the main-text MARE figure.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
LABELS = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
MARKERS = ["s", "^", "D", "v", "P", "X", "*", "h", ">"]
LSTYLE = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def clean_omega(s):
    return pd.to_numeric(s.astype(str).str.strip("[] "), errors="coerce")


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 22, "axes.titlesize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 14, "axes.linewidth": 1.6,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 600, "savefig.bbox": "tight",
    })


def panel(ax, x, dev, ylabel):
    for m, lab, mk, ls, c in zip(MODELS, LABELS, MARKERS, LSTYLE, COLORS):
        ax.plot(x, dev[m], marker=mk, ls=ls, color=c, lw=2.0, ms=7,
                markeredgewidth=0.8, label=lab)
    ax.axhline(0.0, color="0.3", lw=1.2, ls="-")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$ (a.u.)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)


def make(df, source, outdir):
    style()
    pre = f"{source}_"
    x = clean_omega(df["omega"]).to_numpy()
    o = np.argsort(x)
    x = x[o]
    df = df.iloc[o]
    w_sce = pd.to_numeric(df[pre + "Winf"], errors="coerce").to_numpy()
    wp_sce = pd.to_numeric(df[pre + "T_ZP_prl"], errors="coerce").to_numpy()
    dW = {m: pd.to_numeric(df[pre + "W_inf_" + m], errors="coerce").to_numpy() - w_sce
          for m in MODELS}
    dWp = {m: pd.to_numeric(df[pre + "Wp_inf_" + m], errors="coerce").to_numpy() - wp_sce
           for m in MODELS}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 11), sharex=True)
    panel(a1, x, dW, r"$W_\infty^{\rm model}-W_\infty^{\rm SCE}$ (E$_H$)")
    panel(a2, x, dWp, r"$W_\infty^{\prime\,\rm model}-W_\infty^{\prime\,\rm SCE}$ (E$_H$)")
    a1.set_xlabel("")
    a1.legend(ncol=3, frameon=True, framealpha=0.92, loc="best")
    a1.text(0.02, 0.94, "(a)", transform=a1.transAxes, fontsize=20, fontweight="bold")
    a2.text(0.02, 0.94, "(b)", transform=a2.transAxes, fontsize=20, fontweight="bold")
    # No figure title: the panel is described in the SI caption / subsection.
    fig.tight_layout()
    out = Path(outdir) / f"Hook_delta_ingredients_{source}.eps"
    fig.savefig(out, format="eps")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="merged_hooke_results.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    df = df[df["density_type"].astype(str).str.upper().str.strip().eq("FCI")].copy()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for src in ("PYOEP",):
        make(df, src, args.outdir)


if __name__ == "__main__":
    main()
