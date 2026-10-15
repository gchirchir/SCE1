#!/usr/bin/env python3
"""Separate helium-series W_inf/Wp_inf FCI MARE plots for WY and PyOEP."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

MODELS = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
MODEL_COLUMNS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
SYSTEMS = {"H-", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"}  # REV1: N5+-F7+ added (R2-5), Na9+-Ar16+ (R2-6)
COLOR_W, COLOR_WP = "#c00000", "#008000"
HATCH_W, HATCH_WP = "////", "\\\\"


def calculate(data, source):
    prefix = f"{source}_"
    w_ref = pd.to_numeric(data[prefix + "Winf"], errors="coerce")
    wp_ref = pd.to_numeric(data[prefix + "T_ZP_prl"], errors="coerce")
    w_mare, wp_mare = [], []
    for model in MODEL_COLUMNS:
        w = pd.to_numeric(data[prefix + "W_inf_" + model], errors="coerce")
        wp = pd.to_numeric(data[prefix + "Wp_inf_" + model], errors="coerce")
        valid_w = w.notna() & w_ref.notna() & w_ref.ne(0)
        valid_wp = wp.notna() & wp_ref.notna() & wp_ref.ne(0)
        w_mare.append(100.0 * np.mean(np.abs(w[valid_w] - w_ref[valid_w]) /
                                      np.abs(w_ref[valid_w])))
        wp_mare.append(100.0 * np.mean(np.abs(wp[valid_wp] - wp_ref[valid_wp]) /
                                       np.abs(wp_ref[valid_wp])))
    return np.asarray(w_mare), np.asarray(wp_mare)


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


def plot_source(w_mare, wp_mare, source, output_dir):
    style()
    fig, ax = plt.subplots(figsize=(12, 12))
    x = np.arange(len(MODELS))
    width, floor = 0.38, 1e-3
    w_vals, wp_vals = np.maximum(w_mare, floor), np.maximum(wp_mare, floor)
    ax.bar(x - width / 2, w_vals, width, color=COLOR_W, edgecolor="black",
           linewidth=1.5, hatch=HATCH_W, alpha=0.94, zorder=3)
    ax.bar(x + width / 2, wp_vals, width, color=COLOR_WP, edgecolor="black",
           linewidth=1.5, hatch=HATCH_WP, alpha=0.94, zorder=3)
    ax.set_yscale("log")
    y_top = max(np.max(w_vals), np.max(wp_vals)) * 2.5
    ax.set_ylim(0.15, y_top)
    ax.set_ylabel("MARE  [%]", labelpad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=40, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0,
                                                  subs=np.arange(2, 10) * 0.1,
                                                  numticks=20))
    ax.axvline(3.5, color="#333333", linestyle="--", linewidth=2.6, zorder=4)
    y_label, y_bracket = y_top * 1.07, y_top * 1.035
    ax.text(1.5, y_label, "Semilocal", ha="center", va="bottom", fontsize=40,
            fontweight="bold", clip_on=False)
    ax.text(6.0, y_label, "PC-based", ha="center", va="bottom", fontsize=40,
            fontweight="bold", clip_on=False)
    for start, end in ((0, 3), (4, 8)):
        ax.annotate("", xy=(start - 0.35, y_bracket), xytext=(end + 0.35, y_bracket),
                    xycoords="data", arrowprops=dict(arrowstyle="-", color="#333333",
                    lw=2.0, connectionstyle="arc3,rad=0"), annotation_clip=False)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.7,
            alpha=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.4,
            alpha=0.25, zorder=0)

    patch_w = mpatches.Patch(facecolor=COLOR_W, edgecolor="black", linewidth=1.5,
                             hatch=HATCH_W, alpha=0.94, label=r"$W_\infty$")
    patch_wp = mpatches.Patch(facecolor=COLOR_WP, edgecolor="black", linewidth=1.5,
                              hatch=HATCH_WP, alpha=0.94, label=r"$W'_\infty$")
    legend = ax.legend(handles=[patch_w, patch_wp], bbox_to_anchor=(0.32, 1.03),
                       loc="upper left", ncol=1, frameon=True, facecolor="white",
                       edgecolor="black", framealpha=0.92, handlelength=1.0,
                       handleheight=0.3, handletextpad=0.5, columnspacing=2.0,
                       borderpad=0.55, labelspacing=0.6,
                       title="PyOEP" if source == "PYOEP" else "WY", title_fontsize=30)
    legend.get_frame().set_linewidth(0.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / f"HE_SERIES_1_{source}.eps"
    fig.savefig(out, format="eps")
    plt.close(fig)
    print(f"{source} MARE W_inf:  " + ", ".join(f"{m}={v:.4f}" for m, v in zip(MODEL_COLUMNS, w_mare)))
    print(f"{source} MARE Wp_inf: " + ", ".join(f"{m}={v:.4f}" for m, v in zip(MODEL_COLUMNS, wp_mare)))
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
