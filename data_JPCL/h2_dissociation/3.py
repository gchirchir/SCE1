# #!/usr/bin/env python3
# """
# Publication-quality plot of ΔW∞ vs R for H2.
# Improved layout, colour palette, legend, and axis styling.
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # ============================================================
# # 1. LOAD & FILTER DATA
# # ============================================================
# try:
#     df_all = pd.read_csv("final_merged_sorted1.csv")
# except FileNotFoundError:
#     df_all = pd.read_csv("H2_Stretch_rm_ZPE_SPL_GENIS2.csv")

# df_all.columns = df_all.columns.str.strip()
# df_all["density_type"] = df_all["density_type"].astype(str).str.strip().str.upper()
# df_all["R"] = pd.to_numeric(df_all["R"], errors="coerce")

# df = df_all[df_all["density_type"] == "FCI"].copy()
# df = df.sort_values("R").drop_duplicates("R", keep="first")
# df = df[(df["R"] >= 0.7) & (df["R"] <= 10.0)].copy()

# # ============================================================
# # 2. MODELS & LABELS
# # ============================================================
# models = ["LDA", "PBE", "TPSS", "R2SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]
# labels = ["LDA", "PBE", "TPSS", r"R$^2$SCAN", "PC", "mPC", "hPC", "fPC", "ePC"]

# sce_col = "W_inf_SCE" if "W_inf_SCE" in df.columns else "winf_cyl"
# df[sce_col] = pd.to_numeric(df[sce_col], errors="coerce")

# for model in models:
#     col = f"W_inf_{model}"
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#     df[f"dW_{model}"] = df[col] - df[sce_col]

# df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", sce_col])

# # ============================================================
# # 3. TYPOGRAPHY & GEOMETRY CONSTANTS
# # ============================================================
# LABEL_FS = 34
# TICK_FS  = 28
# LEG_FS   = 22
# SPINE_LW = 1.8
# TICK_W_M = 1.8
# TICK_W_m = 1.2
# TICK_MAJ = 10
# TICK_MIN = 5.5

# plt.rcParams.update({
#     "font.family":          "serif",
#     "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
#     "mathtext.fontset":     "stix",
#     "axes.linewidth":       SPINE_LW,
#     "xtick.major.width":    TICK_W_M,   "ytick.major.width":  TICK_W_M,
#     "xtick.minor.width":    TICK_W_m,   "ytick.minor.width":  TICK_W_m,
#     "xtick.major.size":     TICK_MAJ,   "ytick.major.size":   TICK_MAJ,
#     "xtick.minor.size":     TICK_MIN,   "ytick.minor.size":   TICK_MIN,
#     "xtick.direction":      "in",       "ytick.direction":    "in",
#     "xtick.top":            True,       "ytick.right":        True,
#     "xtick.minor.visible":  True,       "ytick.minor.visible": True,
#     "axes.labelsize":       LABEL_FS,
#     "xtick.labelsize":      TICK_FS,
#     "ytick.labelsize":      TICK_FS,
#     "legend.fontsize":      LEG_FS,
#     "savefig.dpi":          1000,
#     "pdf.fonttype":         42,
#     "ps.fonttype":          42,
# })

# # ============================================================
# # 4. COLOUR & LINE STYLE PALETTE
# #    Chosen to be perceptually distinct, colour-blind safe,
# #    and consistent with the companion W'∞ figure.
# #    Markers are OPEN (white face) to match the companion plot.
# # ============================================================
# style = {
#     # ── Semi-local functionals ──────────────────────────────
#     "LDA":    dict(color="#1A6FBF", lw=2.8, ls="-",
#                    marker="s",  ms=9.5,  mew=1.6),   # steel blue  / solid
#     "PBE":    dict(color="#38C0E8", lw=2.8, ls=(0,(8,3)),
#                    marker="^",  ms=10.5, mew=1.6),   # sky cyan    / long-dash
#     "TPSS":   dict(color="#00A880", lw=2.8, ls=(0,(5,2.5,1.5,2.5)),
#                    marker="D",  ms=8.5,  mew=1.6),   # teal        / dash-dot
#     "R2SCAN": dict(color="#E8366E", lw=2.5, ls=(0,(2,2)),
#                    marker="p",  ms=10.0, mew=1.6),   # vivid pink  / dotted
#     # ── PC functional family ────────────────────────────────
#     "PC":     dict(color="#3B2FA8", lw=3.2, ls="-",
#                    marker="*",  ms=13.5, mew=1.6),   # indigo      / solid
#     "mPC":    dict(color="#B02878", lw=2.8, ls=(0,(6,2,1,2)),
#                    marker="P",  ms=10.5, mew=1.6),   # magenta     / dash-dot
#     "hPC":    dict(color="#1A8040", lw=2.8, ls=(0,(6,3)),
#                    marker="<",  ms=9.5,  mew=1.6),   # forest grn  / dashed
#     "fPC":    dict(color="#A89020", lw=2.8, ls=(0,(6,2,1,2,1,2)),
#                    marker=">",  ms=9.5,  mew=1.6),   # olive gold  / dash-dot-dot
#     "ePC":    dict(color="#9E4060", lw=2.5, ls=(0,(2,3)),
#                    marker="d",  ms=9.5,  mew=1.6),   # dusty rose  / loose dot
# }

# # ============================================================
# # 5. FIGURE
# # ============================================================
# fig, ax = plt.subplots(figsize=(8, 8))

# lines = []
# for model, label in zip(models, labels):
#     dcol = f"dW_{model}"
#     sub  = df[["R", dcol]].dropna()
#     if len(sub) == 0:
#         continue
#     st = style[model]
#     line, = ax.plot(
#         sub["R"], sub[dcol],
#         label=label,
#         color=st["color"],
#         lw=st["lw"],
#         ls=st["ls"],
#         marker=st["marker"],
#         markersize=st["ms"],
#         markerfacecolor="white",
#         markeredgecolor=st["color"],
#         markeredgewidth=st["mew"],
#         alpha=0.95,
#         markevery=max(1, len(sub) // 14),
#         zorder=4,
#     )
#     lines.append((line, label))

# # ============================================================
# # 6. AXES, LIMITS & TICKS
# # ============================================================
# ax.set_xlim(0.7, 10.1)
# ax.set_ylim(-0.22, 0.08)

# ax.set_xlabel(r"$R$ ", labelpad=15)
# ax.set_ylabel(
#     r"$\Delta W_\infty = W_\infty^{\mathrm{model}} - W_\infty^{\mathrm{SCE}})$",
#     labelpad=15,
# )

# ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 6, 8, 10]))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
# ax.tick_params(axis="both", which="major", pad=10)

# # ============================================================
# # 7. REFERENCE LINES
# # ============================================================
# ax.axhline(0.0, color="black", lw=1.5, zorder=2)
# ax.axvline(1.4, color="#C04030", lw=1.8, ls=(0,(6,5)), zorder=2)

# # ============================================================
# # 8. GRID
# # ============================================================
# ax.grid(True, which="major", ls=":", lw=0.7, color="0.75", zorder=0)
# ax.grid(True, which="minor", ls=":", lw=0.4, color="0.88", zorder=0)

# # ============================================================
# # 9. SINGLE UNIFIED LEGEND  — 3 columns, upper-right
# #    Uses bbox_to_anchor so it can float into the empty
# #    upper-right data channel without masking any curves.
# # ============================================================
# hs, ls_ = zip(*lines)
# legend = ax.legend(
#     hs, ls_,
#     loc="center right",
#     bbox_to_anchor=(0.995, 0.395),
#     bbox_transform=ax.transAxes,
#     ncol=3,
#     frameon=True,
#     framealpha=1.0,
#     edgecolor="black",
#     facecolor="white",
#     handlelength=2.2,
#     handleheight=1.0,
#     handletextpad=0.5,
#     columnspacing=1.0,
#     labelspacing=0.45,
#     borderpad=0.7,
# )
# legend.get_frame().set_linewidth(1.2)
# legend.set_zorder(1)

# plt.tight_layout()

# # ============================================================
# # 10. EXPORT
# # ============================================================
# outname = "H2_delta_Winf_all_models"
# plt.savefig(outname + ".eps", format="eps", bbox_inches="tight")
# # plt.savefig(outname + ".pdf", format="pdf", bbox_inches="tight")
# plt.show()



#!/usr/bin/env python3
"""
Publication-quality plot of ΔW∞ vs R for H2.
Improved with dual internal split legends and a 100x enhanced color palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# 1. LOAD & FILTER DATA
# ============================================================
# PyOEP-only: WY is dropped entirely (deprecated everywhere else in this
# manuscript). final_merged_dou_sig.csv carries PYOEP_W_inf_{model}
# columns, but as of 2026-09-11 those are still empty for every semilocal
# model (only the exact PYOEP_W_inf_SCE ingredient is populated) -- a known
# pipeline gap, not a plotting bug. Models with no data simply produce no
# line below (see the "if len(sub) == 0: continue" guard); that is expected
# until the cluster run that fills PYOEP_W_inf_{model} completes.
df_all = pd.read_csv("final_merged_dou_sig.csv")

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

# Exact reference: the dual-Kantorovich W_inf for the cylindrical H2 density
# (PYOEP_winf_cyl), as described in the manuscript. It tends to the correct
# dissociation limit (-0.625 E_H); PYOEP_W_inf_SCE does not.
sce_col = "PYOEP_winf_cyl"
df[sce_col] = pd.to_numeric(df[sce_col], errors="coerce")

# Drop isolated single-row glitches in the reference: a point more than 10 mE_H
# from the mean of its two neighbours while those neighbours agree within
# 5 mE_H (the same R = 8.9667 row that 4.py drops for W'_inf). No point is
# interpolated or smoothed.
_w = df[sce_col].to_numpy()
_l, _r = np.roll(_w, 1), np.roll(_w, -1)
_spike = (np.abs(_w - 0.5 * (_l + _r)) > 0.010) & (np.abs(_l - _r) < 0.005)
_spike[0] = _spike[-1] = False
if _spike.any():
    print(f"WARNING: dropping {int(_spike.sum())} isolated spike(s) in {sce_col}: "
          f"R = {sorted(df.loc[_spike, 'R'].tolist())}")
    df = df[~_spike].copy()

for model in models:
    col = f"PYOEP_W_inf_{model}"
    df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df[f"dW_{model}"] = df[col] - df[sce_col]

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", sce_col])

# ============================================================
# 3. TYPOGRAPHY & GEOMETRY CONSTANTS
# ============================================================
LABEL_FS = 34
TICK_FS  = 28
LEG_FS   = 16  # Optimized to prevent internal text box boundary clipping
SPINE_LW = 1.8
TICK_W_M = 1.8
TICK_W_m = 1.2
TICK_MAJ = 10
TICK_MIN = 5.5

plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "stix",
    "axes.linewidth":       SPINE_LW,
    "xtick.major.width":    TICK_W_M,   "ytick.major.width":  TICK_W_M,
    "xtick.minor.width":    TICK_W_m,   "ytick.minor.width":  TICK_W_m,
    "xtick.major.size":     TICK_MAJ,   "ytick.major.size":   TICK_MAJ,
    "xtick.minor.size":     TICK_MIN,   "ytick.minor.size":   TICK_MIN,
    "xtick.direction":      "in",       "ytick.direction":    "in",
    "xtick.top":            True,       "ytick.right":        True,
    "xtick.minor.visible":  True,       "ytick.minor.visible": True,
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
style = {
    # ── Semilocal Functionals (Cool Vibrant Coordinated Tones) ──
    "LDA":    dict(color="#005AB5", lw=2.8, ls="-",
                   marker="s",  ms=9.5,  mew=1.6),   # Premium Sapphire Blue
    "PBE":    dict(color="#24B2E6", lw=2.8, ls=(0,(8,3)),
                   marker="^",  ms=10.5, mew=1.6),   # Electric Sky Blue
    "TPSS":   dict(color="#11A050", lw=2.8, ls=(0,(5,2.5,1.5,2.5)),
                   marker="D",  ms=8.5,  mew=1.6),   # Rich Emerald Green
    "R2SCAN": dict(color="#FFAD00", lw=2.5, ls=(0,(2,2)),
                   marker="p",  ms=10.0, mew=1.6),   # Radiant Amber Gold
                   
    # ── PC Functional Family (Warm Distinct Accent Tones) ───────
    "PC":     dict(color="#D62728", lw=3.2, ls="-",
                   marker="*",  ms=13.5, mew=1.6),   # Crimson Terracotta
    "mPC":    dict(color="#B5127B", lw=2.8, ls=(0,(6,2,1,2)),
                   marker="P",  ms=10.5, mew=1.6),   # Deep Orchid Velvet
    "hPC":    dict(color="#4A00A8", lw=2.8, ls=(0,(6,3)),
                   marker="<",  ms=9.5,  mew=1.6),   # Midnight Royal Purple
    "fPC":    dict(color="#FF6F00", lw=2.8, ls=(0,(6,2,1,2,1,2)),
                   marker=">",  ms=9.5,  mew=1.6),   # Solar Copper Orange
    "ePC":    dict(color="#7F7F7F", lw=2.5, ls=(0,(2,3)),
                   marker="d",  ms=9.5,  mew=1.6),   # Premium Slate Grey
}

# ============================================================
# 5. GENERATE PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(9, 8))

lines = []
for model, label in zip(models, labels):
    dcol = f"dW_{model}"
    sub  = df[["R", dcol]].dropna()
    if len(sub) == 0:
        continue
    st = style[model]
    line, = ax.plot(
        sub["R"], sub[dcol],
        label=label,
        color=st["color"],
        lw=st["lw"],
        ls=st["ls"],
        marker=st["marker"],
        markersize=st["ms"],
        markerfacecolor="white",            # Keeps markers open to match companion plot
        markeredgecolor=st["color"],
        markeredgewidth=st["mew"],
        alpha=0.95,
        markevery=max(1, len(sub) // 14),
        zorder=4,
    )
    lines.append((line, label))

# ============================================================
# 6. AXES, LIMITS & TICKS
# ============================================================
ax.set_xlim(0.2, 10.2)
ax.set_ylim(-0.22, 0.08)

ax.set_xlabel(r"$R$ ", labelpad=15)
ax.set_ylabel(
    r"$\Delta W_\infty = W_\infty^{\mathrm{model}} - W_\infty^{\mathrm{SCE}})$",
    labelpad=15,
)

ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 6, 8, 10]))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.tick_params(axis="both", which="major", pad=10)

# ============================================================
# 7. REFERENCE LINES
# ============================================================
ax.axhline(0.0, color="black", lw=1.5, zorder=2)
ax.axvline(1.4, color="#C04030", lw=1.8, ls=(0,(6,5)), zorder=2)

# ============================================================
# 8. TECHNICAL GRID
# ============================================================
# ax.grid(True, which="major", ls=":", lw=0.7, color="0.75", zorder=0)
# ax.grid(True, which="minor", ls=":", lw=0.4, color="0.88", zorder=0)

# ============================================================
# 9. DUAL INTERNAL SPLIT LEGENDS
# ============================================================
if not lines:
    print("WARNING: no model has any usable PYOEP_W_inf_{model} data in "
          "this R range -- plot has no curves (known pipeline gap).")
else:
    hs, ls_ = zip(*lines)

    # Legend 1: LDA, PBE group (Semilocals) - Positioned cleanly in the central-right whitespace corridor
    leg1 = ax.legend(
        hs[:4], ls_[:4],
        loc="lower center",
        bbox_to_anchor=(0.58, 0.34),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        columnspacing=0.8,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    leg1.get_frame().set_linewidth(0.8)
    leg1.set_zorder(5)
    ax.add_artist(leg1)  # Pins the first legend block down so it doesn't get cleared

    # Legend 2: PC family group - Positioned up in the clear top-right open area
    leg2 = ax.legend(
        hs[4:], ls_[4:],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.12),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        columnspacing=0.8,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    leg2.get_frame().set_linewidth(0.8)
    leg2.set_zorder(5)

plt.tight_layout()

# ============================================================
# 10. EXPORT
# ============================================================
outname = "H2_delta_Winf_all_models_split"
plt.savefig(outname + ".eps", format="eps", bbox_inches="tight")
# plt.savefig(outname + ".pdf", format="pdf", bbox_inches="tight")
# plt.savefig(outname + ".png", format="png", bbox_inches="tight", dpi=300)
plt.show()