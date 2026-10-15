#!/usr/bin/env python3
"""Reference-style Hookium PCME plots for WY and PyOEP FCI densities."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullFormatter
from scipy import integrate

MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
LABELS = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
MARKERS = ["s", "^", "D", "v", "P", "X", "*", "h", ">"]
LINESTYLES = ["-", "--", "-.", ":", "-", "-.", "-.", ":", "-."]


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def clean_omega(series):
    return pd.to_numeric(series.astype(str).str.strip("[] "), errors="coerce")


def spl_corr(w0, gl2, winf):
    if not np.all(np.isfinite([w0, gl2, winf])):
        return np.nan
    if abs(gl2) < 1e-12 or abs(winf - w0) < 1e-12:
        return 0.0
    chi = 2.0 * gl2 / (winf - w0)
    arg = 1.0 + 2.0 * chi
    if arg <= 0.0:
        return np.nan
    return (w0 - winf) * ((np.sqrt(arg) - 1.0 - chi) / chi)


def genisi2_corr(w0, gl2, winf, wpinf):
    if not np.all(np.isfinite([w0, gl2, winf, wpinf])):
        return np.nan
    if abs(gl2) < 1e-12 or abs(wpinf) < 1e-30:
        return 0.0
    if abs(w0) < 1e-30 or abs(winf) < 1e-30:
        return np.nan

    def integrand(lam):
        d = 3.5
        b = (w0 - winf) * (1.0 + d)
        c = b**2 / (4.0 * wpinf**2)
        sq = np.sqrt(1.0 + c * lam)
        uisi = winf + b * (2.0 + c * lam + 2.0 * d * sq) / (2.0 * sq * (d + sq)**2)
        pl = (2.0 * gl2 / w0) * lam
        a2 = -(uisi - w0) / pl if abs(pl) > 1e-30 else 0.0
        ratio = (w0 / winf)**3
        wa = (w0 * pl) / (1.0 + 10.65 * ratio * pl)**3
        wb = (a2 * pl) / (1.0 + 3.6 * ratio * pl)**3
        return uisi + wa + wb - w0

    try:
        value, _ = integrate.quad(integrand, 0.0, 1.0, limit=300,
                                  epsabs=1e-10, epsrel=1e-10)
        return float(value)
    except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
        return np.nan


def load_block(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    data = df[df["density_type"].astype(str).str.upper().str.strip().eq("FCI")].copy()
    data["omega"] = clean_omega(data["omega"])
    data = data.dropna(subset=["omega"]).sort_values("omega")
    if data.empty:
        raise ValueError(f"No FCI rows found in {path}")
    return data


def calculate(data, source):
    prefix = f"{source}_"
    omega = numeric(data, "omega")
    w0 = numeric(data, prefix + "W0")
    gl2 = 0.5 * numeric(data, prefix + "W0_prime")
    ec_ref = numeric(data, prefix + "E_c")
    if np.any(np.abs(ec_ref) < 1e-14):
        raise ValueError("FCI reference correlation energy contains zero values")
    winf_sce = numeric(data, prefix + "Winf")
    wpinf_sce = numeric(data, prefix + "T_ZP_prl")
    ec_spl_sce = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf_sce)])
    ec_gen_sce = np.array([genisi2_corr(a, b, c, d)
                           for a, b, c, d in zip(w0, gl2, winf_sce, wpinf_sce)])
    result = {
        "omega": omega,
        "acme_spl": -100.0 * (ec_spl_sce - ec_ref) / np.abs(ec_ref),
        "acme_gen": -100.0 * (ec_gen_sce - ec_ref) / np.abs(ec_ref),
        "models": {},
    }
    for model in MODELS:
        wc, wpc = f"{prefix}W_inf_{model}", f"{prefix}Wp_inf_{model}"
        if wc not in data or wpc not in data:
            continue
        winf, wpinf = numeric(data, wc), numeric(data, wpc)
        ec_spl = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf)])
        ec_gen = np.array([genisi2_corr(a, b, c, d)
                           for a, b, c, d in zip(w0, gl2, winf, wpinf)])
        result["models"][model] = {
            "spl": -100.0 * (ec_spl - ec_spl_sce) / np.abs(ec_ref),
            "gen": -100.0 * (ec_gen - ec_gen_sce) / np.abs(ec_ref),
        }
    return result


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": 40, "axes.titlesize": 40,
        "xtick.labelsize": 34, "ytick.labelsize": 34, "legend.fontsize": 30,
        "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "xtick.major.size": 12, "ytick.major.size": 12,
        "xtick.minor.size": 6, "ytick.minor.size": 6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 800, "savefig.bbox": "tight",
    })


def plot_source(result, source, output_dir):
    set_style()
    omega = result["omega"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 28), sharex=False)
    handles = []
    for model, label, marker, ls in zip(MODELS, LABELS, MARKERS, LINESTYLES):
        if model not in result["models"]:
            continue
        values = result["models"][model]
        line, = ax_top.plot(omega, values["spl"], marker=marker, linestyle=ls,
                            linewidth=4.0, markersize=14, markeredgewidth=1.5,
                            label=label)
        handles.append(line)
        ax_bot.plot(omega, values["gen"], marker=marker, linestyle=ls,
                    linewidth=4.0, markersize=14, markeredgewidth=1.5, label=label)

    for ax in (ax_top, ax_bot):
        ax.set_xscale("log")
        ax.set_xlim(0.025, 1200)  # REV1 (R2-3): whole omega range, 0.03-1000
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(0.0, color="black", linewidth=1.8)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("RE [%]")

    ax_top.text(0.02, 0.96, "(a) SPL", transform=ax_top.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")
    ax_bot.text(0.02, 0.96, "(b) genISI2", transform=ax_bot.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")
    # REV1 (R2-2/R2-3): the former ylim clip hid the omega = 0.06 W'_inf artifact;
    # with the corrected data no clip is needed.

    axins_top = ax_top.inset_axes([0.46, 0.62, 0.52, 0.36])
    axins_bot = ax_bot.inset_axes([0.58, 0.06, 0.38, 0.24])
    source_label = "PyOEP" if source == "PYOEP" else "WY"
    for inset, vals, method, color in (
        (axins_top, result["acme_spl"], "SPL", "tab:blue"),
        (axins_bot, result["acme_gen"], "genISI2", "tab:red"),
    ):
        inset.axhline(0.0, color="black", linewidth=1.6)
        inset.plot(omega, vals, "o-", linewidth=3.0, markersize=9,
                   markerfacecolor="white", markeredgecolor="tab:blue",
                   markeredgewidth=1.3, color=color, label=method)
        inset.set_xscale("log")
        inset.set_xlabel(r"$\omega$", fontsize=30, labelpad=6)
        inset.set_ylabel("RE [%]", fontsize=30, labelpad=6)
        inset.tick_params(axis="both", which="major", labelsize=24,
                          direction="in", length=7, width=1.8)
        inset.tick_params(axis="both", which="minor", direction="in", length=4, width=1.2)
        for spine in inset.spines.values():
            spine.set_linewidth(1.8)
        legend = inset.legend(loc="best", frameon=True, facecolor="white",
                              edgecolor="black", fontsize=24, handlelength=1.6,
                              title=source_label)
        legend.get_title().set_fontsize(24)
        legend.get_title().set_fontfamily("serif")
        for text in legend.get_texts():
            text.set_fontsize(24)
            text.set_fontfamily("serif")

    ax_top.legend(handles=handles, labels=[h.get_label() for h in handles],
                  loc="lower right", frameon=True, facecolor="white",
                  edgecolor="black", ncol=2, fontsize=30, handlelength=2.0,
                  columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()
    out = output_dir / f"HOOK_FCI_{source}_SPL_genISI2_PCME_combined.eps"
    fig.savefig(out, format="eps")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="merged_hooke_results.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_block(args.csv)
    for source in ("PYOEP",):
        plot_source(calculate(data, source), source, output_dir)


if __name__ == "__main__":
    main()
