"""
TOC graphic for the JPCL manuscript, sized to the journal's real TOC-entry
box (3.25 in x 1.75 in, per jpclcd_authguide.pdf -- see main_jpcl.tex's
\\setlength\\acs@tocentry@width/height).

Visual language borrowed from an earlier figure script of the authors
(the AC_drugs figure): bold sans-serif, thick rounded-box panels, a
FancyArrowPatch connector, high-contrast flat colors -- but the CONTENT here
is specific to this manuscript, not the generic AC/drug-discovery story.

Panel A: the adiabatic-connection curve's strong-coupling endpoint
(lambda -> 1, giving W_inf/W'_inf), with the four exact benchmark systems
feeding into it.
Panel B: REAL numbers from main_jpcl.tex's Results ("Atoms and ions"
subsection, ~lines 1461-1481): the W'_inf ingredient MARE varies enormously
across semilocal models (2.5% for hPC up to ~86% for LDA/PBE/mPC), yet the
genISI2-propagated total correlation-energy MARE is reported as flat at
7-9% for every one of those inputs. This is the manuscript's actual
punchline: ingredient accuracy alone does not predict total-energy
accuracy.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.0

FIG_W, FIG_H = 3.25, 1.75  # ACS JPC Letters tocentry box, inches
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=600)

gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.14, 1.35],
                       wspace=0.05, left=0.05, right=0.985, top=0.90, bottom=0.20)
gs_left = gs[0].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.08)
ax_ac = fig.add_subplot(gs_left[0])
ax_sys = fig.add_subplot(gs_left[1])
ax_arrow = fig.add_subplot(gs[1])
ax_bar = fig.add_subplot(gs[2])

# ---------- Panel A (top): AC curve endpoint ----------
lam = np.linspace(0, 1, 300)
W = -0.35 - 1.55 * (1 - np.exp(-2.4 * lam))
ax_ac.plot(lam, W, color='black', lw=1.6, solid_capstyle='round')
ax_ac.plot([1, 1], [0, W[-1]], color='#4361EE', lw=2.2)
ax_ac.annotate(r"$W_\infty,\,W'_\infty$", xy=(1, W[-1]), xytext=(0.10, W[-1] - 0.02),
               fontsize=6.0, fontweight='bold', color='#4361EE',
               arrowprops=dict(arrowstyle='->', lw=1.1, color='#4361EE'))
ax_ac.set_xlim(-0.08, 1.25)
ax_ac.set_ylim(min(W) - 0.15, 0.18)
ax_ac.axis('off')
ax_ac.text(0.5, 1.06, 'Four exact benchmarks', fontsize=5.9, fontweight='bold', ha='center',
           transform=ax_ac.transAxes,
           bbox=dict(boxstyle='round,pad=0.26', facecolor='#E3F2FD',
                     edgecolor='#1976D2', linewidth=1.0))

# ---------- Panel A (bottom): the four benchmark systems, in-bounds ----------
systems = ['Hookium', 'He series', r'H$_2$', 'Atoms H$-$Xe']
colors_sys = ['#E63946', '#F77F00', '#06A77D', '#4361EE']
ys = np.linspace(0.86, 0.12, len(systems))
for y, s, c in zip(ys, systems, colors_sys):
    ax_sys.text(0.04, y, '●', color=c, fontsize=6.3,
               transform=ax_sys.transAxes, va='center')
    ax_sys.text(0.17, y, s, color='black', fontsize=5.1, fontweight='bold',
               transform=ax_sys.transAxes, va='center')
ax_sys.set_xlim(0, 1)
ax_sys.set_ylim(0, 1)
ax_sys.axis('off')

# ---------- Arrow ----------
arrow = FancyArrowPatch((0.05, 0.5), (0.95, 0.5), arrowstyle='-|>', mutation_scale=13,
                        linewidth=2.0, color='#333333', transform=ax_arrow.transAxes)
ax_arrow.add_patch(arrow)
ax_arrow.axis('off')

# ---------- Panel B: W'_inf ingredient MARE (real numbers) vs. the flat ----------
# ---------- genISI2 total-energy MARE band (real numbers), atoms H-Zn ----------
# Source: main_jpcl.tex Results, "Atoms and ions" subsection (lines ~1461-1481):
#   hPC 2.5%, ePC 4.7% (added for R2-vii, revision 1), TPSS 5.8%, PC 8.9%, r^2SCAN 22% (all exact, quoted MARE for W'_inf);
#   LDA/PBE/mPC lumped in the text as "76-96%" (not resolved per-model there) --
#   plotted here at the band midpoint, 86%, labeled accordingly.
#   genISI2 propagated correlation-energy MARE is reported as "within 7-9%"
#   for EVERY input tested, including the exact SCE ingredient itself -- shown
#   below as a flat reference band, not fabricated per-model bars.
models = ['hPC', 'ePC', 'TPSS', 'PC', 'r$^2$SCAN', 'LDA/PBE/\nmPC (mid)']
wprime_mare = [2.5, 4.7, 5.8, 8.9, 22, 86]
bar_colors = ['#2E7D32', '#8E44AD', '#4361EE', '#F77F00', '#E63946', '#7A7A7A']

x = np.arange(len(models))
ax_bar.set_yscale('log')
ax_bar.axhspan(7, 9, color='#06A77D', alpha=0.28, zorder=1)
ax_bar.axhline(8, color='#06A77D', lw=1.1, ls='--', zorder=2)
bars = ax_bar.bar(x, wprime_mare, width=0.55, color=bar_colors, zorder=3)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(models, fontsize=4.1, rotation=20, ha='right', fontweight='bold')
ax_bar.set_yticks([3, 10, 30, 100])
ax_bar.set_yticklabels(['3', '10', '30', '100%'], fontsize=4.4)
ax_bar.tick_params(axis='y', length=2, pad=1)
for spine in ('top', 'right'):
    ax_bar.spines[spine].set_visible(False)
ax_bar.spines['left'].set_linewidth(1.0)
ax_bar.spines['bottom'].set_linewidth(1.0)
ax_bar.set_ylim(1.8, 130)
ax_bar.set_title(r"$W'_\infty$ ingredient MARE (atoms H$-$Zn)", fontsize=5.3,
                 fontweight='bold', pad=2)

ax_bar.annotate('genISI2 total energy:\nflat, 7$-$9%, any input',
                xy=(1.5, 8), xytext=(2.0, 40), fontsize=4.0, fontweight='bold',
                color='#1B5E20', ha='left',
                arrowprops=dict(arrowstyle='->', lw=1.0, color='#1B5E20'))

for ext in ('eps', 'pdf', 'png'):
    fig.savefig(f'toc_jpcl.{ext}', format=ext, dpi=600, facecolor='white')

plt.close(fig)
print("TOC draft saved: toc_jpcl.eps / .pdf / .png (3.25in x 1.75in target)")
