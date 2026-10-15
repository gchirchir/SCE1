#!/usr/bin/env python3
"""Separate helium-series SPL/genISI2 MARE plots for WY and PyOEP."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from plot import spl_corr, genisi2_corr

FUNCTIONALS = ["SCE", "LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
DISPLAY_LABELS = ["SCE", "LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
SYSTEMS = {"H-", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"}  # REV1: N5+-F7+ added (R2-5), Na9+-Ar16+ (R2-6)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def calculate(data, source):
    prefix = f"{source}_"
    w0 = numeric(data, prefix + "W0")
    gl2 = 0.5 * numeric(data, prefix + "W0_prime")
    ec_ref = numeric(data, prefix + "E_c")
    mare_spl, mare_gen = [], []
    for functional in FUNCTIONALS:
        if functional == "SCE":
            winf = numeric(data, prefix + "Winf")
            wpinf = numeric(data, prefix + "T_ZP_prl")
        else:
            winf = numeric(data, prefix + "W_inf_" + functional)
            wpinf = numeric(data, prefix + "Wp_inf_" + functional)
        ec_spl = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf)])
        ec_gen = np.array([genisi2_corr(a, b, c, d)
                           for a, b, c, d in zip(w0, gl2, winf, wpinf)])
        valid_spl = np.isfinite(ec_spl) & np.isfinite(ec_ref) & (np.abs(ec_ref) > 1e-14)
        valid_gen = np.isfinite(ec_gen) & np.isfinite(ec_ref) & (np.abs(ec_ref) > 1e-14)
        mare_spl.append(100.0 * np.mean(np.abs(ec_spl[valid_spl] - ec_ref[valid_spl]) /
                                        np.abs(ec_ref[valid_spl])))
        mare_gen.append(100.0 * np.mean(np.abs(ec_gen[valid_gen] - ec_ref[valid_gen]) /
                                        np.abs(ec_ref[valid_gen])))
    return np.asarray(mare_spl), np.asarray(mare_gen)


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 40, "axes.titlesize": 40,
        "xtick.labelsize": 34, "ytick.labelsize": 34,
        "legend.fontsize": 30, "axes.linewidth": 2.5,
        "xtick.major.width": 2.5, "ytick.major.width": 2.5,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "xtick.major.size": 12, "ytick.major.size": 12,
        "xtick.minor.size": 6, "ytick.minor.size": 6,
        "xtick.direction": "out", "ytick.direction": "out",
        "hatch.linewidth": 1.8, "savefig.dpi": 800,
        "savefig.bbox": "tight",
    })


def plot_source(mare_spl, mare_gen, source, output_dir):
    style()
    x = np.arange(len(FUNCTIONALS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.bar(x - width / 2, mare_spl, width, color="#334155", edgecolor="black",
           linewidth=1.5, hatch="////", alpha=0.94, zorder=3)
    ax.bar(x + width / 2, mare_gen, width, color="#d81b60", edgecolor="black",
           linewidth=1.5, hatch="\\\\", alpha=0.94, zorder=3)
    ax.set_xlim(-0.5, len(FUNCTIONALS) - 0.5)
    ymax = max(8.0, float(np.ceil(max(np.nanmax(mare_spl), np.nanmax(mare_gen)))))
    ax.set_ylim(0.0, ymax)
    ax.set_yticks(np.arange(0.0, ymax + 1.0, 1.0))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=2.6, zorder=4)
    ax.axvline(4.5, color="#333333", linestyle="--", linewidth=2.6, zorder=4)
    ax.tick_params(axis="x", which="major", top=False, bottom=True)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(3.0, 1.08, "Semilocal", transform=trans, ha="center", va="bottom",
            fontsize=40, fontweight="bold")
    ax.text(7.0, 1.08, "PC", transform=trans, ha="center", va="bottom",
            fontsize=40, fontweight="bold")
    ax.set_ylabel("MARE [%]", labelpad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS, rotation=40, ha="right", rotation_mode="anchor")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.7,
            alpha=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.4,
            alpha=0.25, zorder=0)

    patch_spl = mpatches.Patch(facecolor="#334155", edgecolor="black", linewidth=1.5,
                               hatch="////", alpha=0.94, label="SPL")
    patch_gen = mpatches.Patch(facecolor="#d81b60", edgecolor="black", linewidth=1.5,
                               hatch="\\\\", alpha=0.94, label="genISI2")
    legend = ax.legend(handles=[patch_spl, patch_gen], loc="upper left",
                       bbox_to_anchor=(0.10, 1.0), ncol=1, frameon=True,
                       edgecolor="black", framealpha=0.0, handlelength=2.0,
                       handleheight=1.2, handletextpad=0.5, columnspacing=2.0,
                       borderpad=0.55, labelspacing=0.6,
                       title="PyOEP" if source == "PYOEP" else "WY", title_fontsize=30)
    legend.get_frame().set_linewidth(0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / f"He_series_MARE_SPL_genISI2_{source}.eps"
    fig.savefig(out, format="eps")
    plt.close(fig)
    for functional, spl, gen in zip(FUNCTIONALS, mare_spl, mare_gen):
        print(f"{source:5s} {functional:8s} SPL={spl:10.6f}%  genISI2={gen:10.6f}%")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="merged_results.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    fci = df[df["density_type"].astype(str).str.upper().str.strip().eq("FCI")].copy()
    fci = fci[fci["system"].isin(SYSTEMS) & fci["source_folder"].ne("he_se")]
    fci = fci.sort_values("system").drop_duplicates("system")
    if fci.empty:
        raise ValueError("No unique FCI helium-series rows found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in ("PYOEP",):
        plot_source(*calculate(fci, source), source, output_dir)


if __name__ == "__main__":
    main()
