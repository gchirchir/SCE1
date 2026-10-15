# Data and plotting scripts for

**"Probing the Accuracy of Adiabatic-Connection Exchange–Correlation Functionals
with Highly Accurate, Strongly Correlated Electron Data"**
G. Chirchir, A. Bhattacharjee, S. Jana, S. Śmiga

This folder contains the data behind every figure in the manuscript and its
Supporting Information, the exact scripts that plotted them, and the `star.py`
scripts that generated the data.
Running the scripts on these files regenerates all 18 figures, and a check
script confirms that each one is pixel-for-pixel identical to the version in
the manuscript.

## Quick start

The generated figures are already included: each folder contains its data,
its scripts, and the EPS figures those scripts produce (for example,
`atoms_ions/neutral_1_PYOEP.eps`). A single script can be re-run in its own
folder, e.g.

```bash
cd atoms_ions && python3 plot_he_series_mare_spl_genisi2_wy_pyoep.py
```

which prints the MARE values and rewrites the figure. To regenerate and check
everything at once:

```bash
./run_all.sh
```

This runs every plotting script in its own folder and then `verify_figures.py`,
which rasterizes each regenerated EPS and the manuscript copy in
`reference_figures/` (Ghostscript, 100 dpi) and compares them pixel by pixel.
Expected last line:

```
18/18 figures reproduce the manuscript exactly.
```

Requirements: Python 3 with numpy, pandas, matplotlib, scipy, and Pillow, plus
Ghostscript (`gs`) for the check. Tested with Python 3.14.7, numpy 2.5.2,
pandas 3.0.5, matplotlib 3.11.1, scipy 1.18.1, Ghostscript 10.06.0.

## Figure → script → data

| Manuscript figure | Output file | Folder | Script | Data file |
|---|---|---|---|---|
| Figure 1 | `Fig_MARE_Hook_FCI_PYOEP_RED_GREEN_R2SCAN.eps` | `hookium/` | `plot_hook_mare_wy_pyoep.py` | `merged_hooke_results.csv` |
| Figure 2 | `Hook_MARE_SPL_genISI2_PYOEP.eps` | `hookium/` | `plot_hook_mare_spl_genisi2_wy_pyoep.py` | `merged_hooke_results.csv` |
| Figure 3 | `HOOK_FCI_PYOEP_SPL_genISI2_PCME_combined.eps` | `hookium/` | `plot_hook_wy_pyoep.py` | `merged_hooke_results.csv` |
| Figure 4 | `HE_SERIES_1_PYOEP.eps` | `he_series/` | `plot_he_series_winf_mare_wy_pyoep.py` | `merged_results.csv` |
| Figure 5 | `He_series_MARE_SPL_genISI2_PYOEP.eps` | `he_series/` | `plot_he_series_mare_spl_genisi2_wy_pyoep.py` | `merged_results.csv` |
| Figure 6 | `He_iso_FCI_PYOEP_genISI2_SPL_PCME_combined.eps` | `he_series/` | `plot_oep_wy.py` | `merged_results.csv` |
| Figure 7 | `Fig_H2_total_energy_journal_cylAC_PYOEP.eps` | `h2_dissociation/` | `plot_final_wy_pyoep.py` (uses `plot_final.py`) | `final_merged_dou_sig.csv` |
| Figure 8 | `H2_delta_Winf_all_models_split.eps` | `h2_dissociation/` | `3.py` | `final_merged_dou_sig.csv` |
| Figure 9 | `H2_Wprime_inf_enhanced.eps` | `h2_dissociation/` | `4.py` | `final_merged_dou_sig.csv` |
| Figure 10a | `neutral_1_PYOEP.eps` | `atoms_ions/` | `plot_he_series_winf_mare_wy_pyoep.py` | `merged_results.csv` |
| Figure 10b | `neutral_MARE_SPL_genISI2_PYOEP.eps` | `atoms_ions/` | `plot_he_series_mare_spl_genisi2_wy_pyoep.py` | `merged_results.csv` |
| TOC graphic | `toc_jpcl.eps` | `toc_graphic/` | `plot_toc_jpcl.py` | none: MARE values quoted in the Results text (from the Figure 10 data) |
| SI Figure S1 | `Hook_delta_ingredients_PYOEP.eps` | `hookium/` | `plot_hook_delta_ingredients.py` | `merged_hooke_results.csv` |
| SI Figure S2 | `He_series_delta_ingredients_PYOEP.eps` | `he_series/` | `plot_he_delta_ingredients.py` | `merged_results.csv` |

Helper modules imported by the scripts are included next to them:
`hookium/plot_hook_wy_pyoep.py` (SPL/genISI2 formulas for Figures 2–3),
`he_series/plot.py` (model list and SPL/genISI2 formulas for Figures 5–6), and
`h2_dissociation/plot_final.py` (the H₂ curve construction for Figure 7).

## Revision 1 additions (SI figures and tables)

Added in the revision of the manuscript. Run each script from its own folder
(`python3 <script>`); the table scripts write a LaTeX fragment next to `SI.tex`, one level
above this folder.

| SI item | Output | Folder | Script | Data |
|---|---|---|---|---|
| SI Figure S3 | `atoms_model_errors_PYOEP.eps` | `atoms_ions/` | `plot_atoms_model_errors.py` | `merged_results.csv` |
| SI Figure S4 | `H2_models_AC_MAE_PYOEP.eps` | `h2_dissociation/` | `plot_h2_models_ac.py` | `final_merged_dou_sig.csv` |
| SI Figure S5 | `HOOK_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps` | `hookium/` | `plot_hook_mae_pcme_acme.py` | `merged_hooke_results.csv` |
| SI Figure S6 | `He_iso_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps` | `he_series/` | `plot_he_mae_pcme_acme.py` | `merged_results.csv` |
| SI Table S6 | `../SI_mare_colour_table.tex` | `.` | `make_mare_colour_table.py` | Hookium, He-series and atoms CSVs |
| SI Tables S7–S9 | `../SI_hookium_regime_tables.tex` | `hookium/` | `make_hookium_regime_tables.py` | `merged_hooke_results.csv` |
| SI Table S9(b), analytic column | printed values | `hookium/` | `gaussian_limit.py` | none (exact Gaussian density) |
| SI Table S10, exact-density columns | printed values | `hookium/` | `taut_exact_density_sce.py` | none (Taut's analytic densities) |
| SI Tables S10–S12 | `../SI_sce_reference_tables.tex` | `.` | `make_sce_reference_tables.py` (`--he <csv>` selects the He-series file) | Hookium, He-series and H₂ CSVs |
| SI Tables S13–S14 | `../SI_he_regime_tables.tex` | `he_series/` | `make_he_regime_tables.py` | `merged_results.csv` |

`gaussian_limit.py` evaluates the exact two-electron W∞ and W′∞ of the ω → ∞ Gaussian
density (W∞/√ω = −1.151617, W′∞/ω^(3/4) = 0.403058). `taut_exact_density_sce.py` builds
Taut's analytic Hookium densities (ω = 1/2, 1/10, 0.0365373) and evaluates the same
expressions on them; it reproduces the published SCE values.

## Data files

All energies are in Hartree (E_H). One row per system (and per reference-density
level); columns are grouped by prefix.

| File | Rows | Contents |
|---|---|---|
| `hookium/merged_hooke_results.csv` | 39 | Hookium atom at 20 confinement frequencies ω = 0.03–1000 a.u. (`omega`), HF and FCI reference densities (`density_type`) |
| `he_series/merged_results.csv` | 36 | He isoelectronic series, H⁻ to Ar¹⁶⁺ (Z = 1–18; `system` gives the element: H-, He, Li, …, Ar), HF and FCI densities, 5ZaPa-NR-CV basis, recomputed in the revision with the density-adapted radial grid (see below) |
| `atoms_ions/merged_results.csv` | 19 | Neutral atoms and anions H to Zn (the 19 systems in the Figure 10 statistics), relaxed CCSD(T) or CCSD densities |
| `h2_dissociation/final_merged_dou_sig.csv` | 592 | H₂ at bond lengths R = 0.3 a₀ upward (`R_bohr`), HF and FCI densities |

Key columns. `<model>` is one of LDA, PBE, TPSS, R2SCAN, PC, mPC, hPC, fPC, ePC.
The prefix `PYOEP_` marks quantities evaluated on the PyOEP-inverted reference
density, which is used for every figure.

| Quantity | Hookium, He series, atoms/ions | H₂ |
|---|---|---|
| Model W∞[ρ], W′∞[ρ] | `PYOEP_W_inf_<model>`, `PYOEP_Wp_inf_<model>` | same |
| Exact SCE W∞[ρ] | `PYOEP_Winf` | `PYOEP_winf_cyl` (dual-Kantorovich; used by Figure 8, `3.py`) |
| Exact SCE W′∞[ρ] | `PYOEP_T_ZP_prl` (despite the name, computed with the ¼ prefactor, the PRA convention of SI Section S9, as used throughout the manuscript) | `PYOEP_winf_prime_rmzpe_cyl` (restricted-mode ZPE, used by Figure 9, `4.py`) |
| Weak-coupling input W₀ = E_x, W′₀ = 2E_c^GL2 | `PYOEP_W0`, `PYOEP_W0_prime` | same |
| Reference correlation energy | `PYOEP_E_c` (FCI or CCSD(T)/CCSD) | `E_ref` (FCI total energy) |
| Inversion quality | `PYOEP_converged`, `PYOEP_rho_diff` | same |

The reference-density level is in `density_type` (HF, FCI, CCSD, CCSDT). The
figures use the FCI rows (Hookium, He series, H₂) and the CCSD(T)/CCSD rows
(atoms/ions).

For H₂, use `PYOEP_winf_cyl` rather than `PYOEP_W_inf_SCE` as the exact
W∞: only the dual-Kantorovich value approaches the correct dissociation limit
(−0.625 E_H). Both `3.py` and `4.py` drop one isolated single-row glitch in their
reference column (R = 8.9667 a₀) and print a warning when they do.

**Do not use `PYOEP_Wp_inf_SCE` in the H₂ file.** It is broken (values up to
~1.5 × 10⁵ E_H from a known problem in that Hessian route). Figure 9 uses
`PYOEP_winf_prime_rmzpe_cyl` instead, as noted in `4.py`. For H₂ the PyOEP
inversion is reliable only up to R ≈ 7.7 a₀; beyond that, the curves in Figure 7
are cut, as stated in the manuscript.

The manuscript uses the **PyOEP** density inversion throughout, and the scripts
here produce PyOEP figures only. The CSV files also contain columns from an
alternative Wu–Yang inversion (`WY_` prefix); they are not used in the
manuscript or by any script in this folder. (The scripts are the authors'
original plotting scripts; the only change is that the WY/PyOEP loop in each
now runs over PyOEP alone.)

**Note on `he_series/merged_results.csv`:** recomputed in the revision for all 18 ions with
`he_series/star.py`, whose two-electron SCE integration normalizes N_e(r) to exactly 2 and uses a radial
grid adapted to the extent of the density (≥ 40001 points, cut where 4πr²ρ < 10⁻¹⁴ of its maximum).
Z = 1–11 and 15–18 were run with PySCF 2.3.0, Z = 12–14 with PySCF 2.7.0 (the script contains a
compatibility line for PySCF ≥ 2.5); on the 14 ions run with both versions W∞ agrees to ≤ 0.06 mE_h.
The file has no repeated header rows any more.

## How the data were generated (`star.py`)

Each system folder also contains the `star.py` script that produced its CSV
file (from PySCF reference densities, PyOEP density inversion, the exact SCE
construction, and the semilocal models), together with the local modules it
imports. These scripts are included so that the calculations can be inspected
and rerun; they are **not** needed to regenerate the figures.

| Folder | Script | Local modules included | What it computes |
|---|---|---|---|
| `hookium/` | `star.py` | `wu_yang.py` | Hookium at each ω: HF and two-electron correlated target densities (CCSD, exact for two electrons, i.e. FCI), PyOEP inversion, analytic two-electron SCE W∞/W′∞, semilocal W∞/W′∞, W₀/W′₀, SPL/genISI2 |
| `he_series/` | `star.py` | `wu_yang.py` (`relaxed_dm.py`, used by the original runs, is kept but no longer imported) | He isoelectronic series, HF/CCSD/FCI densities, same quantities |
| `atoms_ions/` | `star.py` | `jaxsce/` (the locally modified jaxsce SCE code) | Atoms and ions: CCSD(T)/CCSD densities, numerical co-motion SCE W∞ and W′∞ (Hessian normal modes), semilocal models |
| `h2_dissociation/` | `star.py` (+ `collect_h2_range.py`, which gathers the per-R cluster outputs of `star.py` into one CSV) | `wu_yang.py` | H₂ scan in R: FCI densities, PyOEP inversion, dual-Kantorovich W∞ and restricted-mode W′∞, semilocal models, SPL/genISI2 |

External requirements for these scripts (not included): PySCF, KS-pies
(`kspies`), basis-set-exchange, JAX (for `jaxsce`), and the PyOEP package, which
provides the `methods` module (`exxoep`, `KSINV`); point the `PYOEP_ROOT`
environment variable at the PyOEP installation. Each `star.py` takes the system
and basis on its command line (`--atom`, `--charge`, `--spin`, `--basis`, …).

Note: `hookium/star.py` and `he_series/star.py` import `wu_yang` (the Wu–Yang inversion helper;
its functions are not used for the PyOEP results in the manuscript, but the import is
unconditional). Both drivers are the revision versions that produced the current CSV files
(N_e(r) normalized to 2 and density-adapted radial grid in the two-electron SCE integration;
the earlier versions are not included because their results are superseded).

## Other files

- `reference_figures/`: the exact EPS files included in the manuscript, used
  by `verify_figures.py`.
- `MANIFEST.sha256`: SHA-256 checksums of every data file, script, and reference
  figure (`sha256sum -c MANIFEST.sha256` to check integrity).
- `tables/`: the sources of the values printed in Table 1 and SI Tables S4–S5 (formerly main-text Tables 2–3). `tables/table_sources.csv`
  maps each of the 90 values to its source file and line, and `tables/check_tables.py`
  verifies them (70 verified; 20 have no archived raw output, listed in
  `tables/README.md`).
