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
NAMES = {"R2SCAN": r"r$^2$SCAN"}
STYLE = {  # semilocal: dashed/open markers; PC family: solid/filled markers
    "LDA": ("#7f7f7f", "--", "o", "none"), "PBE": ("#17becf", "--", "s", "none"),
    "TPSS": ("#2ca02c", "--", "^", "none"), "R2SCAN": ("#9467bd", "--", "v", "none"),
    "PC": ("#ff7f0e", "-", "o", None), "mPC": ("#8c564b", "-", "s", None),
    "hPC": ("#d62728", "-", "D", None), "fPC": ("#e377c2", "-", "^", None),
    "ePC": ("#1f77b4", "-", "P", None),
}


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
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 13, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    x = np.arange(len(ORDER))
    for ax, e, lab in ((axes[0], ew, r"$\Delta W_\infty\,/\,|W_{\infty,\rm SCE}|$  [%]"),
                       (axes[1], ewp, r"$\Delta W'_\infty\,/\,W'_{\infty,\rm SCE}$  [%]")):
        ax.axhline(0, color="black", lw=1.0, zorder=1)
        for m in MODELS:
            c, ls, mk, fc = STYLE[m]
            ax.plot(x, e[m].values, ls=ls, marker=mk, color=c, lw=1.6, ms=7,
                    mfc=(fc if fc is not None else c), mec=c, label=NAMES.get(m, m),
                    zorder=3)
        ax.set_ylabel(lab)
        ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
        for xv in (2.5, 12.5):  # shell closures He | Li..Ne | Na..
            ax.axvline(xv, color="#999999", lw=0.8, ls=":", zorder=0)
    axes[0].text(0.01, 0.95, "(a)", transform=axes[0].transAxes, va="top", fontsize=15)
    axes[1].text(0.01, 0.95, "(b)", transform=axes[1].transAxes, va="top", fontsize=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[s] for s in ORDER])
    axes[1].set_xlabel("system (ordered by nuclear charge $Z$)")
    axes[0].legend(ncol=5, fontsize=11, frameon=True, loc="lower right")
    fig.tight_layout()
    for ext in ("eps", "png"):
        fig.savefig(HERE / f"atoms_model_errors_PYOEP.{ext}", dpi=300)
    # printed check: the means over systems must equal the Fig. 10(a) MAREs
    print("MARE W_inf :", {m: round(float(ew[m].abs().mean()), 2) for m in MODELS})
    print("MARE Wp_inf:", {m: round(float(ewp[m].abs().mean()), 2) for m in MODELS})


if __name__ == "__main__":
    main()
