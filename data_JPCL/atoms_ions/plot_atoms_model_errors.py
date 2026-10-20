#!/usr/bin/env python3
"""Per-atom signed relative errors of the strong-interaction models (R2-12, revision 1).

For every atom/ion of the statistics set (H-Zn incl. anions, ordered by Z) this
plots 100*(W_model - W_SCE)/|W_SCE| for W_inf (panel a) and W'_inf (panel b),
for the nine models of Fig. 10, on the PyOEP reference density.  Input is the
same merged_results.csv used for Fig. 10, so the averages over the systems equal
the MAREs of Fig. 10(a).  H is omitted in panel (b) because W'_inf = 0 for one
electron.  No in-figure title (description goes in the caption).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = ["H", "H-", "He", "Li", "Li-", "Be", "B", "C", "N", "O", "F", "F-",
         "Ne", "Na", "Mg", "Cl-", "Ar", "Ca", "Zn"]
LABELS = {s: (s[:-1] + r"$^-$" if s.endswith("-") else s) for s in ORDER}
MODELS = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
# same colours, markers, line styles and fonts as SI Figs. S1/S2 (plot_he_delta_ingredients.py)
NAMES = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
MARKERS = ["s", "^", "D", "v", "P", "X", "*", "h", ">"]
LSTYLE = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 14, "axes.linewidth": 1.6,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 600, "savefig.bbox": "tight",
    })


def load():
    d = pd.read_csv(HERE / "merged_results.csv")
    d.columns = d.columns.str.strip()
    d = d[d["density_type"].astype(str).str.upper().isin({"FCI", "CCSDT", "CCSD"})]
    return d.drop_duplicates("system").set_index("system").loc[ORDER]


def rel_err(d, q, ref):
    r = pd.to_numeric(d[ref], errors="coerce")
    out = {}
    for m in MODELS:
        v = pd.to_numeric(d[f"PYOEP_{q}_{m}"], errors="coerce")
        e = 100.0 * (v - r) / r.abs()
        out[m] = e.where(r.abs() > 0)
    return pd.DataFrame(out)


def main():
    d = load()
    ew = rel_err(d, "W_inf", "PYOEP_Winf")
    ewp = rel_err(d, "Wp_inf", "PYOEP_T_ZP_prl")
    style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True)
    x = np.arange(len(ORDER))
    for ax, e, lab in ((axes[0], ew, r"$\Delta W_\infty/|W_\infty^{\rm SCE}|$ (%)"),
                       (axes[1], ewp, r"$\Delta W'_\infty/W_\infty^{\prime\,\rm SCE}$ (%)")):
        for m, name, mk, ls, c in zip(MODELS, NAMES, MARKERS, LSTYLE, COLORS):
            ax.plot(x, e[m].values, marker=mk, ls=ls, color=c, lw=2.0, ms=8,
                    markeredgewidth=0.8, label=name)
        ax.axhline(0.0, color="0.3", lw=1.2)
        ax.set_ylabel(lab)
        ax.grid(True, ls=":", lw=0.5, alpha=0.4)
        for xv in (2.5, 12.5):  # rows of the periodic table: H-He | Li-Ne | Na-Zn
            ax.axvline(xv, color="0.5", lw=1.0, ls=":", zorder=0)
    axes[0].text(0.02, 0.94, "(a)", transform=axes[0].transAxes, fontsize=20, fontweight="bold")
    axes[1].text(0.02, 0.94, "(b)", transform=axes[1].transAxes, fontsize=20, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[s] for s in ORDER], rotation=90)
    axes[1].set_xlabel("system (ordered by nuclear charge $Z$)")
    axes[0].legend(ncol=3, frameon=True, framealpha=0.92, loc="lower right")
    # No figure title: the panel is described in the SI caption.
    fig.tight_layout()
    for ext in ("eps", "png"):
        fig.savefig(HERE / f"atoms_model_errors_PYOEP.{ext}", dpi=300)
    # printed check: the means over systems must equal the Fig. 10(a) MAREs
    print("MARE W_inf :", {m: round(float(ew[m].abs().mean()), 2) for m in MODELS})
    print("MARE Wp_inf:", {m: round(float(ewp[m].abs().mean()), 2) for m in MODELS})


if __name__ == "__main__":
    main()
