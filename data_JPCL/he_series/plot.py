

#!/usr/bin/env python3
"""Plot the SPL/genISI2 PCME decomposition for the He isoelectronic series.

The construction is identical to the Hookium analysis:

    FD   = E_c(model inputs) - E_c(reference)
    PCME = E_c(model inputs) - E_c(SCE inputs)
    ACME = E_c(SCE inputs)   - E_c(reference)

and therefore FD = PCME + ACME (up to floating-point roundoff).

By default the FCI rows are used.  A different density block can be selected,
for example:  python He_iso_PCME_SPL_genISI2.py --density HF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import integrate


MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
LABELS = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
MARKERS = ["s", "^", "D", "v", "P", "X", "*", "h", ">"]
LINESTYLES = ["-", "--", "-.", ":", "-", "-.", "-.", ":", "-."]


def spl_corr(w0: float, gl2: float, winf: float) -> float:
    """SPL correlation energy."""
    if not np.all(np.isfinite([w0, gl2, winf])):
        return np.nan
    if abs(gl2) < 1.0e-12 or abs(winf - w0) < 1.0e-12:
        return 0.0

    chi = 2.0 * gl2 / (winf - w0)
    arg = 1.0 + 2.0 * chi
    if arg <= 0.0:
        return np.nan
    return (w0 - winf) * ((np.sqrt(arg) - 1.0 - chi) / chi)


def genisi2_corr(w0: float, gl2: float, winf: float, wpinf: float) -> float:
    """genISI2 correlation energy."""
    if not np.all(np.isfinite([w0, gl2, winf, wpinf])):
        return np.nan
    if abs(gl2) < 1.0e-12 or abs(wpinf) < 1.0e-30:
        return 0.0
    if abs(w0) < 1.0e-30 or abs(winf) < 1.0e-30:
        return np.nan

    def integrand(lam: float) -> float:
        d = 3.5
        b = (w0 - winf) * (1.0 + d)
        c = b**2 / (4.0 * wpinf**2)
        l1, l2 = 10.65, 3.6
        w0p = 2.0 * gl2

        sq = np.sqrt(1.0 + c * lam)
        uisi = winf + b * (2.0 + c * lam + 2.0 * d * sq) / (
            2.0 * sq * (d + sq) ** 2
        )

        p = w0p / w0
        pl = p * lam
        a1 = w0
        a2 = -(uisi - w0) / pl if abs(pl) > 1.0e-30 else 0.0
        r1 = l1 * (w0 / winf) ** 3
        r2 = l2 * (w0 / winf) ** 3
        wa = (a1 * pl) / (1.0 + r1 * pl) ** 3
        wb = (a2 * pl) / (1.0 + r2 * pl) ** 3
        return uisi + wa + wb - w0

    try:
        value, _ = integrate.quad(
            integrand, 0.0, 1.0, limit=300, epsabs=1.0e-10, epsrel=1.0e-10
        )
        return float(value)
    except (FloatingPointError, ValueError, OverflowError, ZeroDivisionError):
        return np.nan


def padded_limits(arrays: list[np.ndarray], minimum_span: float = 1.0) -> tuple[float, float]:
    """Return readable limits that include zero and all finite plotted values."""
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays if np.any(np.isfinite(a))])
    lo = min(0.0, float(np.min(finite)))
    hi = max(0.0, float(np.max(finite)))
    span = max(hi - lo, minimum_span)
    return lo - 0.10 * span, hi + 0.10 * span


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 40,
        "axes.titlesize": 40,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 27,
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 800,
        "savefig.bbox": "tight",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="He_iso_electronic_series_5ZaPa-NR-CV.csv",
        help="Input He-isoelectronic-series CSV file.",
    )
    parser.add_argument(
        "--density", default="FCI", choices=["FCI", "CCSD", "HF"],
        help="Density block to plot (default: FCI).",
    )
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    input_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df["density_type"] = df["density_type"].astype(str).str.upper().str.strip()
    data = df[df["density_type"] == args.density].copy()
    if data.empty:
        raise ValueError(f"No {args.density} rows found in {input_path}.")

    # For a two-electron ion, nuclear charge Z = ionic charge + 2.
    data["Z"] = pd.to_numeric(data["charge"], errors="coerce") + 2.0
    data = data.sort_values("Z").dropna(subset=["Z"])
    z = data["Z"].to_numpy(float)
    system_names = data["system"].astype(str).to_list()

    w0 = pd.to_numeric(data["W0"], errors="coerce").to_numpy(float)
    ec_ref = pd.to_numeric(data["E_c"], errors="coerce").to_numpy(float)
    if np.any(np.abs(ec_ref) < 1.0e-14):
        raise ValueError(
            "The selected density block contains zero E_c values. "
            "Use --density FCI or --density CCSD for relative-error plots."
        )

    # W0_prime = 2 E_c^GL2 in this dataset, so gl2 = W0_prime/2.
    if "gl2" in data.columns:
        gl2 = pd.to_numeric(data["gl2"], errors="coerce").to_numpy(float)
    elif "W0_prime" in data.columns:
        gl2 = 0.5 * pd.to_numeric(data["W0_prime"], errors="coerce").to_numpy(float)
    else:
        raise ValueError("Missing both gl2 and W0_prime columns.")

    winf_sce = pd.to_numeric(data["W_inf_SCE"], errors="coerce").to_numpy(float)
    wpinf_sce = pd.to_numeric(data["Wp_inf_SCE"], errors="coerce").to_numpy(float)

    ec_spl_sce = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf_sce)])
    ec_gen_sce = np.array([
        genisi2_corr(a, b, c, d)
        for a, b, c, d in zip(w0, gl2, winf_sce, wpinf_sce)
    ])
    acme_spl = ec_spl_sce - ec_ref
    acme_gen = ec_gen_sce - ec_ref

    results: dict[str, dict[str, np.ndarray | str]] = {}
    for model, label in zip(MODELS, LABELS):
        wc, wpc = f"W_inf_{model}", f"Wp_inf_{model}"
        if wc not in data.columns or wpc not in data.columns:
            print(f"Skipping {model}: missing {wc} or {wpc}")
            continue

        winf = pd.to_numeric(data[wc], errors="coerce").to_numpy(float)
        wpinf = pd.to_numeric(data[wpc], errors="coerce").to_numpy(float)
        ec_spl = np.array([spl_corr(a, b, c) for a, b, c in zip(w0, gl2, winf)])
        ec_gen = np.array([
            genisi2_corr(a, b, c, d) for a, b, c, d in zip(w0, gl2, winf, wpinf)
        ])
        results[model] = {
            "label": label,
            "fd_spl": ec_spl - ec_ref,
            "pcme_spl": ec_spl - ec_spl_sce,
            "fd_gen": ec_gen - ec_ref,
            "pcme_gen": ec_gen - ec_gen_sce,
        }

    print("\nConsistency check: max |FD - (PCME + ACME)| [Ha]")
    for model, r in results.items():
        res_spl = np.nanmax(np.abs(r["fd_spl"] - (r["pcme_spl"] + acme_spl)))
        res_gen = np.nanmax(np.abs(r["fd_gen"] - (r["pcme_gen"] + acme_gen)))
        print(f"  {model:8s} SPL={res_spl:.3e}  genISI2={res_gen:.3e}")

    # Same sign convention as the supplied Hookium plot.
    re_gen = {m: -100.0 * r["pcme_gen"] / np.abs(ec_ref) for m, r in results.items()}
    re_spl = {m: -100.0 * r["pcme_spl"] / np.abs(ec_ref) for m, r in results.items()}
    re_acme_gen = -100.0 * acme_gen / np.abs(ec_ref)
    re_acme_spl = -100.0 * acme_spl / np.abs(ec_ref)

    style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 28), sharex=True)
    line_handles = []

    for model, label, marker, ls in zip(MODELS, LABELS, MARKERS, LINESTYLES):
        if model not in results:
            continue
        line, = ax_top.plot(
            z, re_gen[model], marker=marker, linestyle=ls, linewidth=4.0,
            markersize=14, markeredgewidth=1.5, label=label,
        )
        line_handles.append(line)
        ax_bot.plot(
            z, re_spl[model], marker=marker, linestyle=ls, linewidth=4.0,
            markersize=14, markeredgewidth=1.5, label=label,
        )

    for ax in (ax_top, ax_bot):
        ax.axhline(0.0, color="black", linewidth=1.8)
        ax.set_xlim(0.0, 10.25)
        ax.set_xticks(np.arange(1, 11, 1))
        ax.set_ylabel("RE [%]")
        ax.grid(False)

    ax_top.set_ylim(*padded_limits(list(re_gen.values())))
    ax_bot.set_ylim(*padded_limits(list(re_spl.values())))
    ax_top.set_xlabel(r"Nuclear charge, $Z$")
    ax_bot.set_xlabel(r"Nuclear charge, $Z$")

    ax_top.text(0.02, 0.96, "(a) genISI2", transform=ax_top.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")
    ax_bot.text(0.02, 0.96, "(b) SPL", transform=ax_bot.transAxes,
                ha="left", va="top", fontsize=30, fontweight="bold")

    # Insets contain the ACM error obtained with reference SCE inputs.
    axins_top = ax_top.inset_axes([0.47, 0.08, 0.51, 0.34])
    axins_top.plot(z, re_acme_gen, "o-", linewidth=3.0, markersize=9,
                   markerfacecolor="white", markeredgecolor="tab:blue",
                   markeredgewidth=1.3, color="tab:red", label="genISI2")
    axins_bot = ax_bot.inset_axes([0.47, 0.62, 0.51, 0.34])
    axins_bot.plot(z, re_acme_spl, "o-", linewidth=3.0, markersize=9,
                   markerfacecolor="white", markeredgecolor="tab:blue",
                   markeredgewidth=1.3, color="tab:blue", label="SPL")

    for inset, vals in ((axins_top, re_acme_gen), (axins_bot, re_acme_spl)):
        inset.axhline(0.0, color="black", linewidth=1.4)
        inset.set_xlim(0.8, 10.2)
        inset.set_ylim(*padded_limits([vals], minimum_span=0.25))
        inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        inset.set_xlabel(r"$Z$", fontsize=27)
        inset.set_ylabel("RE [%]", fontsize=27)
        inset.tick_params(axis="both", which="major", labelsize=21,
                          direction="in", length=7, width=1.8)
        for spine in inset.spines.values():
            spine.set_linewidth(1.8)
        inset.legend(loc="best", frameon=True, facecolor="white",
                     edgecolor="black", fontsize=24, handlelength=1.6)

    ax_bot.legend(
        handles=line_handles, labels=[h.get_label() for h in line_handles],
        loc="lower right", frameon=True, facecolor="white", edgecolor="black",
        ncol=2, fontsize=27, handlelength=2.0, columnspacing=1.0,
        labelspacing=0.35,
    )

    fig.tight_layout()
    stem = f"He_iso_{args.density}_genISI2_SPL_PCME_combined"
    eps_path = output_dir / f"{stem}.eps"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(eps_path, format="eps")
    fig.savefig(png_path, format="png", dpi=400)
    plt.close(fig)

    # Export every plotted number for reproducibility.
    out = pd.DataFrame({"Z": z, "system": system_names, "E_c_ref": ec_ref,
                        "ACME_RE_genISI2_percent": re_acme_gen,
                        "ACME_RE_SPL_percent": re_acme_spl})
    for model in results:
        out[f"PCME_RE_genISI2_{model}_percent"] = re_gen[model]
        out[f"PCME_RE_SPL_{model}_percent"] = re_spl[model]
    values_path = output_dir / f"{stem}_values.csv"
    out.to_csv(values_path, index=False)

    print(f"\nSaved: {eps_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {values_path}")


if __name__ == "__main__":
    main()


