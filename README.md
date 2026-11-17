# SCE benchmark data, scripts, and code

This repository holds the data, analysis scripts, and code that accompany:

> **Probing the Accuracy of Adiabatic-Connection Exchange–Correlation Functionals with Highly Accurate, Strongly Correlated Electron Data**

Authors: Gabriel Chirchir, Abhishek Bhattacharjee, Subrata Jana, and Szymon Śmiga.

It provides exact strong-coupling reference values W₀[ρ], W∞[ρ] and W′∞[ρ] computed on correlated reference densities for four benchmark sets. It also provides the semilocal strong-coupling models and the SPL and genISI2 adiabatic-connection interpolations, evaluated on the same densities.

## Repository layout

| Folder | Purpose | How to use it |
|---|---|---|
| `data_JPCL/` | **Full data.** Every CSV with all model values, the scripts that draw every figure and build the SI tables, and the `star.py` drivers that produced the data. | `cd data_JPCL && ./run_all.sh` regenerates all 18 figures in a single run and checks them against the published ones. See `data_JPCL/README.md`. |
| `data_summary/` | **Quick look.** For each system, the reference values W₀, W∞ and W′∞ used in the manuscript, one small table each. | Open the CSVs directly; nothing has to be run. See `data_summary/README.md`. |
| `JAXSCE_SCE/` | **Code.** Installable SCE implementation that computes exact W∞ and W′∞ from correlated PySCF densities, the benchmark drivers, and a He test case. | `cd JAXSCE_SCE && pip install -e . && ./examples/run_he.sh`. See `JAXSCE_SCE/README.md`. |

---

## Benchmark sets

| System | Data folder | Systems covered | Target density | Main-text figures | SI |
|---|---|---|---|---|---|
| Hookium (two electrons in a harmonic trap) | `hookium/` | 20 confinement frequencies, ω = 0.03–1000 | FCI | 1, 2, 3 | Figs. S1, S5 |
| He isoelectronic series | `he_series/` | H⁻ … Ar¹⁶⁺ (Z = 1–18) | FCI, 5ZaPa-NR-CV | 4, 5, 6 | Figs. S2, S6 |
| H₂ dissociation | `h2_dissociation/` | R = 0.3–11 a₀ | FCI | 7, 8, 9 | Fig. S4 |
| Neutral atoms and ions | `atoms_ions/` | H … Zn in the figures, H … Xe in Table 1 | CCSD(T) | 10a, 10b | Fig. S3 |
| Ne basis-set series | `tables/` | 2Z–5ZaPa-NR-CV | CCSD(T), CCSD, HF | – | Tables S4, S5 |

Each data folder is in `data_JPCL/` (full data). `data_summary/` has the same folders with the reference values only; the Ne series is in `data_summary/ne_basis/`.

### Quantities

| Symbol | Meaning | Column in `data_JPCL/` | Column in `data_summary/` |
|---|---|---|---|
| W₀[ρ] | weak-coupling limit (= exact exchange E_x) | `PYOEP_W0` | `W_0` |
| W∞[ρ] | strictly-correlated-electrons (SCE) strong-coupling limit | `PYOEP_Winf` (H₂: `PYOEP_winf_cyl`) | `W_inf` |
| W′∞[ρ] | zero-point coefficient of the strong-coupling expansion, with the ¼ prefactor (SI Section S9) | `PYOEP_T_ZP_prl` (H₂: `PYOEP_winf_prime_rmzpe_cyl`) | `W'_inf` |

All energies are in **Hartree**. Model columns in `data_JPCL/` are named `PYOEP_W_inf_<model>` and `PYOEP_Wp_inf_<model>`, for LDA, PBE, TPSS, r²SCAN, PC, mPC, hPC, fPC and ePC.

---

## Quick start

**Look at the reference values** (needs only pandas):

```python
import pandas as pd

he = pd.read_csv("data_summary/he_series/he_series_W0_Winf_Wpinf.csv")
print(he[["system", "W_0", "W_inf", "W'_inf"]])
```

**Reproduce every figure** (needs Python 3 with numpy, pandas, matplotlib, scipy and Pillow, plus Ghostscript):

```bash
cd data_JPCL
./run_all.sh
```

The expected last line is `18/18 figures reproduce the manuscript exactly.`

**Re-run the calculations** (also needs PySCF, JAX, KS-pies and PyOEP; see `JAXSCE_SCE/README.md` for installation):

```bash
cd JAXSCE_SCE/drivers/atoms_ions
python star.py --atom Ne --basis 5ZaPa-NR-CV --density ccsdt --N-grid 1025
python star.py --help        # all options
```

## Integrity checks

| Check | Command | What it confirms |
|---|---|---|
| Checksums | `cd data_JPCL && sha256sum -c MANIFEST.sha256` | every data file, script and reference figure is unchanged |
| Figures | `cd data_JPCL && ./run_all.sh` | all 18 figures match `data_JPCL/reference_figures/` |
| Tables | `cd data_JPCL/tables && python3 check_tables.py` | manuscript table values match their archived raw outputs |
| Summary | `cd data_summary && python3 make_data.py` | `data_summary/` is rebuilt unchanged from `data_JPCL/` |
| Code | `cd JAXSCE_SCE && python -m pytest -q tests` | the SCE pipeline runs on the He test case |
