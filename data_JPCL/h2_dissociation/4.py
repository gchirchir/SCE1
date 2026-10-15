#!/usr/bin/env python3
"""
Publication-quality plot of ΔW'∞ vs R for H2.
Upgraded with a 100x enhanced professional palette and dual internal legends.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# 1. LOAD & FILTER DATA
# ============================================================
# PyOEP-only: WY is dropped entirely. sce_col uses the restricted-mode ZPE
# quantity (PYOEP_winf_prime_rmzpe_cyl) rather than PYOEP_Wp_inf_SCE -- the
# latter is currently broken (O(10^5) E_H, a pipeline bug in the raw T_ZP
# Hessian route) while winf_prime_rmzpe_cyl is sane (0.01-0.38 E_H range).
# PYOEP_Wp_inf_{model} is still 100% empty for every semilocal model as of
# 2026-09-11 -- a known gap, not a plotting bug; models with no data simply
# produce no line below.
main_file = "final_merged_dou_sig.csv"
df_all = pd.read_csv(main_file)
df_all.columns = df_all.columns.str.strip()
df_all["density_type"] = df_all["density_type"].astype(str).str.strip().str.upper()
df_all["R"] = pd.to_numeric(df_all["R"], errors="coerce")

df = df_all[df_all["density_type"] == "FCI"].copy()
df = df.sort_values("R").drop_duplicates("R", keep="first")
df = df[(df["R"] >= 0.3) & (df["R"] <= 10.0)].copy()

# ============================================================
# 2. MODELS & LABELS
# ============================================================
models = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
labels = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]

sce_col = "PYOEP_winf_prime_rmzpe_cyl"
if sce_col not in df.columns:
    raise KeyError(f"Column '{sce_col}' not found.")
df[sce_col] = pd.to_numeric(df[sce_col], errors="coerce")

# Single-row convergence glitches: winf_prime_rmzpe_cyl is smooth in R (its
# physical trend decreases toward the asymptotic plateau as R grows), but a
# handful of isolated rows (e.g. R=8.9667 -> 0.1277 sitting between two
# neighbors of ~0.03) spike the whole delta plot through one bad geometry.
# A rolling-window/MAD test false-positives on ordinary sampling gaps (R
# jumps unevenly here) and on normal point-to-point noise, so instead flag
# only a genuine local spike: a row whose value is more than double BOTH of
# its immediate R-sorted neighbors (which is exactly the R=8.9667 shape --
# neighbors ~0.031/0.031, center 0.128 -- and nothing else in this data).
_v = df[sce_col].to_numpy()
_left, _right = np.roll(_v, 1), np.roll(_v, -1)
_is_bad = (_v > 2 * np.maximum(_left, _right)) & (_v > 2 * np.minimum(_left, _right))
_is_bad[0] = _is_bad[-1] = False  # no wraparound neighbor at the array ends
if _is_bad.any():
    print(f"WARNING: dropping {int(_is_bad.sum())} row(s), isolated spike "
          f"vs both immediate neighbors in {sce_col} (likely PyOEP "
          f"convergence glitch): R = {sorted(df.loc[_is_bad, 'R'].tolist())}")
    df = df[~_is_bad].copy()

for model in models:
    col = f"PYOEP_Wp_inf_{model}"
    df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df[f"dWp_{model}"] = df[col] - df[sce_col]

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", sce_col])

# ============================================================
# 3. GLOBAL STYLE (High-Fidelity Journal Settings)
# ============================================================
SPINE_LW = 2.0
TICK_MAJ  = 10
TICK_MIN  = 6
TICK_W_M  = 2.0
TICK_W_m  = 1.4
LABEL_FS  = 34
TICK_FS   = 28
LEG_FS    = 16  # Perfectly scaled to prevent box boundary collision

plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "stix",
    "axes.linewidth":       SPINE_LW,
    "xtick.major.width":    TICK_W_M,  "ytick.major.width":  TICK_W_M,
    "xtick.minor.width":    TICK_W_m,  "ytick.minor.width":  TICK_W_m,
    "xtick.major.size":     TICK_MAJ,  "ytick.major.size":   TICK_MAJ,
    "xtick.minor.size":     TICK_MIN,  "ytick.minor.size":   TICK_MIN,
    "xtick.direction":      "in",      "ytick.direction":    "in",
    "xtick.top":            True,      "ytick.right":        True,
    "xtick.minor.visible":  True,      "ytick.minor.visible": True,
    "axes.labelsize":       LABEL_FS,
    "xtick.labelsize":      TICK_FS,
    "ytick.labelsize":      TICK_FS,
    "legend.fontsize":      LEG_FS,
    "savefig.dpi":          1000,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# ============================================================
# 4. 100x ENHANCED HIGH-FIDELITY COLOR PALETTE
# ============================================================
enhanced_style = {
    "LDA":    dict(color="#005AB5", lw=3.0, ls="-",  marker="s", ms=8.5,  mew=1.2), # Deep Royal Blue
    "PBE":    dict(color="#24B2E6", lw=3.0, ls="--", marker="^", ms=9.5,  mew=1.2), # Electric Cyan
    "TPSS":   dict(color="#11A050", lw=3.0, ls=":",  marker="D", ms=7.5,  mew=1.2), # Vivid Emerald Green
    "R2SCAN": dict(color="#FFAD00", lw=3.0, ls="-.", marker="p", ms=9.0,  mew=1.2), # Amber Gold
    "PC":     dict(color="#D62728", lw=3.2, ls="-",  marker="*", ms=13.0, mew=1.2), # Crimson Red
    "mPC":    dict(color="#B5127B", lw=3.0, ls="--", marker="P", ms=9.5,  mew=1.2), # Rich Orchid Magenta
    "hPC":    dict(color="#4A00A8", lw=3.0, ls=":",  marker="<", ms=8.5,  mew=1.2), # Deep Velvet Indigo
    "fPC":    dict(color="#FF6F00", lw=3.0, ls="-.", marker=">", ms=8.5,  mew=1.2), # Solar Orange
    "ePC":    dict(color="#7F7F7F", lw=2.8, ls="-",  marker="d", ms=8.5,  mew=1.2), # Premium Slate Grey
}

# ============================================================
# 5. GENERATE PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

lines = []
line_labels = []
for model, label in zip(models, labels):
    dcol = f"dWp_{model}"
    sub  = df[["R", dcol]].dropna()
    if len(sub) == 0:
        continue

    st = enhanced_style[model]
    line, = ax.plot(
        sub["R"], sub[dcol],
        label=label,
        color=st["color"],
        lw=st["lw"],
        ls=st["ls"],
        marker=st["marker"],
        markersize=st["ms"],
        markerfacecolor=st["color"],
        markeredgecolor="white",           # Sharp crisp outline boundary
        markeredgewidth=st["mew"],
        alpha=0.95,
        markevery=max(1, len(sub) // 14),
        zorder=4,
    )
    lines.append(line)
    line_labels.append(label)

# ============================================================
# 6. AXES, LIMITS & TICKS
# ============================================================
ax.set_ylim(-0.02, 0.72)
ax.set_xlim(0.2, 10.2)

ax.set_xlabel(r"$R$ ", labelpad=15)
ax.set_ylabel(
    r"$\Delta W_\infty^\prime = W_\infty^{\prime\,\mathrm{model}} "
    r"- W_\infty^{\prime\,\mathrm{SCE}}\ $",
    labelpad=15,
)

ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 6, 8, 10]))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.tick_params(axis="both", which="major", pad=10)

# ============================================================
# 7. REFERENCE LINES
# ============================================================
ax.axhline(0.0, color="black", lw=1.5, zorder=2)
ax.axvline(1.4, color="#C04030", lw=1.8, ls=(0, (6, 5)), zorder=2)

# ============================================================
# 8. DUAL INTERNAL SPLIT LEGENDS
# ============================================================
if not lines:
    print("WARNING: no model has any usable PYOEP_Wp_inf_{model} data in "
          "this R range -- plot has no curves (known pipeline gap).")
else:
    # Legend 1: Positioned flawlessly within the center-right empty data corridor
    leg1 = ax.legend(
        lines[:6], line_labels[:6],
        loc="upper right",
        bbox_to_anchor=(0.94, 0.98),
        ncol=3,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        columnspacing=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        handlelength=1.5,
    )
    leg1.get_frame().set_linewidth(0.8)
    leg1.set_zorder(5)
    ax.add_artist(leg1)  # Fixes legend 1 onto canvas so it doesn't get cleared

    # Legend 2: Positioned elegantly up in the top-right open area
    leg2 = ax.legend(
        lines[6:], line_labels[6:],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.55),
        ncol=3,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        labelspacing=0.3,
        handlelength=1.5,
    )
    leg2.get_frame().set_linewidth(0.8)
    leg2.set_zorder(5)

plt.tight_layout()

# ============================================================
# 9. EXPORT EXCELLENCE
# ============================================================
outname = "H2_Wprime_inf_enhanced"
plt.savefig(outname + ".eps", format="eps", bbox_inches="tight")
# plt.savefig(outname + ".pdf", format="pdf", bbox_inches="tight")
# plt.savefig(outname + ".png", format="png", bbox_inches="tight", dpi=300)
plt.show()