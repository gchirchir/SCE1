#!/usr/bin/env python3
"""Generate SPL/genISI2 data and plots for WY and PyOEP results."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot import (MODELS, LABELS, MARKERS, LINESTYLES, genisi2_corr,
                  padded_limits, spl_corr, style)

Z_MAP = {"H-": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
         "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18}  # REV1 (R2-5, R2-6)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def calculate(data, source):
    p = "WY_" if source == "WY" else "PYOEP_"
    # The SCE reference fields are stored separately from the model W_inf fields.
    w0 = numeric(data, p + "W0")
    gl2 = 0.5 * numeric(data, p + "W0_prime")
    ec_ref = numeric(data, p + "E_c")
    winf_sce = numeric(data, p + "Winf")
    wpinf_sce = numeric(data, p + "T_ZP_prl")
    ec_spl_sce = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf_sce)])
    ec_gen_sce = np.array([genisi2_corr(a, b, c, d)
                           for a, b, c, d in zip(w0, gl2, winf_sce, wpinf_sce)])
    rows = []
    for model, label in zip(MODELS, LABELS):
        winf = numeric(data, p + "W_inf_" + model)
        wpinf = numeric(data, p + "Wp_inf_" + model)
        ec_spl = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf)])
        ec_gen = np.array([genisi2_corr(a, b, c, d)
                           for a, b, c, d in zip(w0, gl2, winf, wpinf)])
        for i, (_, row) in enumerate(data.iterrows()):
            rows.append({
                "source": source, "density": row["density_type"],
                "system": row["system"], "Z": row["Z"], "model": model,
                "model_label": label, "E_c_ref": ec_ref[i],
                "E_c_spl": ec_spl[i], "E_c_genISI2": ec_gen[i],
                "PCME_SPL": ec_spl[i] - ec_spl_sce[i],
                "PCME_genISI2": ec_gen[i] - ec_gen_sce[i],
                "ACME_SPL": ec_spl_sce[i] - ec_ref[i],
                "ACME_genISI2": ec_gen_sce[i] - ec_ref[i],
                "W0": w0[i], "W0_prime": 2 * gl2[i], "W_inf": winf[i],
                "Wp_inf": wpinf[i],
            })
    return pd.DataFrame(rows)


def plot_source(values, density, source, outdir):
    """Make the reference-style, two-panel plot for one OEP source."""
    d = values[values.source.eq(source)].copy()
    if d.empty:
        return

    # E_c_ref is the matching nonzero FCI correlation reference for both plots.
    relative = True
    ylabel = "RE [%]"

    style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 28), sharex=True)
    line_handles = []
    top_values, bot_values = [], []

    for model, label, marker, ls in zip(MODELS, LABELS, MARKERS, LINESTYLES):
        q = d[d.model.eq(model)].sort_values("Z")
        if q.empty:
            continue
        qscale = -100.0 / q["E_c_ref"].abs() if relative else 1.0
        y_gen = (q.PCME_genISI2 * qscale).to_numpy(float)
        y_spl = (q.PCME_SPL * qscale).to_numpy(float)
        line, = ax_top.plot(
            q.Z, y_spl, marker=marker, linestyle=ls, linewidth=4.0,
            markersize=14, markeredgewidth=1.5, label=label,
        )
        line_handles.append(line)
        ax_bot.plot(
            q.Z, y_gen, marker=marker, linestyle=ls, linewidth=4.0,
            markersize=14, markeredgewidth=1.5, label=label,
        )
        top_values.append(y_spl)
        bot_values.append(y_gen)

    zvals = np.sort(d.Z.unique())
    for ax in (ax_top, ax_bot):
        ax.axhline(0.0, color="black", linewidth=1.8)
        ax.set_xlim(max(0.0, zvals.min() - 0.25), zvals.max() + 0.25)
        ax.set_xticks(zvals)
        ax.set_ylabel(ylabel)
        ax.grid(False)
    ax_top.set_ylim(*padded_limits(top_values, minimum_span=1.0 if relative else 0.001))
    ax_bot.set_ylim(*padded_limits(bot_values, minimum_span=1.0 if relative else 0.001))
    ax_top.set_xlabel(r"Nuclear charge, $Z$")
    ax_bot.set_xlabel(r"Nuclear charge, $Z$")
    ax_top.text(0.02, 0.96, "(a) SPL", transform=ax_top.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")
    ax_bot.text(0.02, 0.96, "(b) genISI2", transform=ax_bot.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")

    # ACME is independent of the semilocal model, hence one value per Z.
    qref = d[d.model.eq("LDA")].sort_values("Z").drop_duplicates("Z")
    refscale = -100.0 / qref["E_c_ref"].abs() if relative else 1.0
    inset_gen_values = (qref.ACME_genISI2 * refscale).to_numpy(float)
    inset_spl_values = (qref.ACME_SPL * refscale).to_numpy(float)
    axins_top = ax_top.inset_axes([0.47, 0.62, 0.51, 0.34])
    axins_bot = ax_bot.inset_axes([0.60, 0.06, 0.36, 0.22])
    for inset, vals, label, color in (
        (axins_top, inset_spl_values, "SPL", "tab:blue"),
        (axins_bot, inset_gen_values, "genISI2", "tab:red"),
    ):
        inset.plot(qref.Z, vals, "o-", linewidth=3.0, markersize=9,
                   markerfacecolor="white", markeredgecolor="tab:blue",
                   markeredgewidth=1.3, color=color, label=label)
        inset.axhline(0.0, color="black", linewidth=1.4)
        inset.set_xlim(zvals.min() - 0.2, zvals.max() + 0.2)
        inset.set_ylim(*padded_limits([vals], minimum_span=0.25 if relative else 0.001))
        inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        inset.set_xlabel(r"$Z$", fontsize=27)
        inset.set_ylabel(ylabel, fontsize=27)
        inset.tick_params(axis="both", which="major", labelsize=21,
                          direction="in", length=7, width=1.8)
        for spine in inset.spines.values():
            spine.set_linewidth(1.8)
        source_label = "PyOEP" if source == "PYOEP" else "WY"
        inset_legend = inset.legend(
            loc="best", frameon=True, facecolor="white", edgecolor="black",
            fontsize=24, handlelength=1.6, title=None,
        )
        # Matplotlib styles legend titles separately; force identical text
        # properties for the source name and SPL/genISI2 method label.
        inset_legend.get_title().set_fontsize(24)
        inset_legend.get_title().set_fontfamily("serif")
        inset_legend.get_title().set_fontweight("normal")
        for legend_text in inset_legend.get_texts():
            legend_text.set_fontsize(24)
            legend_text.set_fontfamily("serif")
            legend_text.set_fontweight("normal")

    legend_kwargs = dict(
        handles=line_handles, labels=[h.get_label() for h in line_handles],
        frameon=True, facecolor="white", edgecolor="black", ncol=2,
        fontsize=27, handlelength=2.0, columnspacing=1.0, labelspacing=0.35,
    )
    if density == "HF":
        # Anchor the box in the empty deep-negative part of the HF SPL panel.
        ax_top.legend(loc="lower right", bbox_to_anchor=(0.98, 0.015),
                      borderaxespad=0.0, **legend_kwargs)
    else:
        ax_top.legend(loc="lower right", **legend_kwargs)
    fig.tight_layout()
    stem = f"He_iso_{density}_{source}_genISI2_SPL_PCME_combined"
    fig.savefig(outdir / f"{stem}.eps", format="eps")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="merged_results.csv")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    # Remove duplicated header rows and the he_se summary rows.
    df = df[df.system.isin(Z_MAP) & df.source_folder.ne("he_se")].copy()
    df["Z"] = df.system.map(Z_MAP)
    # Requested output set: only FCI/WY and HF/PyOEP, in EPS format.
    requested_plots = (("FCI", "PYOEP"),)
    for density, source in requested_plots:
        block = df[df.density_type.eq(density)].sort_values("Z").copy()
        if block.empty:
            continue
        if density == "HF":
            # HF rows have E_c=0.  Use the matching FCI correlation energies
            # as the exact reference so the requested relative error is
            # finite and directly comparable with the FCI/WY plot.
            fci = df[df.density_type.eq("FCI")].drop_duplicates("system")
            reference = fci.set_index("system")[f"{source}_E_c"]
            block[f"{source}_E_c"] = block["system"].map(reference)
            if block[f"{source}_E_c"].isna().any():
                missing = block.loc[block[f"{source}_E_c"].isna(), "system"].tolist()
                raise ValueError(f"Missing FCI correlation references for: {missing}")
        values = calculate(block, source)
        plot_source(values, density, source, outdir)


if __name__ == "__main__":
    main()
