# SCE reference data: W₀, W∞ and W′∞

This folder holds the strong-coupling reference data that accompany:

> **Probing the Accuracy of Adiabatic-Connection Exchange–Correlation Functionals with Highly Accurate, Strongly Correlated Electron Data**

Authors: Gabriel Chirchir, Abhishek Bhattacharjee, Subrata Jana, and Szymon Śmiga.

## Two data folders

| Folder | Purpose | How to use it |
|---|---|---|
| `../data_JPCL/` | **Full data.** Every CSV with all model values, plus the plotting scripts and calculation drivers. | `cd ../data_JPCL && ./run_all.sh` regenerates all 18 manuscript figures in a single run and checks them against the published ones. |
| `data_summary/` (this folder) | **Quick look.** For each system, the reference values used in the manuscript, one small table each. | Open the CSVs directly; nothing has to be run. |

The subfolders here follow the system folders in `../data_JPCL/` (`hookium`, `he_series`, `atoms_ions`, `h2_dissociation`, plus `ne_basis` from `tables/`). Every value is taken unchanged from the full data by `make_data.py`.

## Columns

Each table contains only the three adiabatic-connection ingredients, one row per system:

| Column | Quantity | Meaning |
|---|---|---|
| `W_0` | W₀[ρ] | weak-coupling limit (= exact exchange E_x) |
| `W_inf` | W∞[ρ] | strictly-correlated-electrons (SCE) strong-coupling limit |
| `W'_inf` | W′∞[ρ] | zero-point coefficient of the strong-coupling expansion, with the ¼ prefactor (PRA convention, SI Section S9) |

All energies are in **Hartree**, rounded to 6 decimals. Model values (LDA, PBE, TPSS, r²SCAN, PC, mPC, hPC, fPC, ePC) are not included here. The full tables with all models are in `../data_JPCL/`.

---

## Contents

| Path | System | Rows | Identifier columns | Target density |
|---|---|---:|---|---|
| `hookium/hookium_W0_Winf_Wpinf.csv` | Hookium (two electrons in a harmonic trap) | 20 | `omega` (0.03–1000) | FCI |
| `he_series/he_series_W0_Winf_Wpinf.csv` | He isoelectronic series, H⁻ … Ar¹⁶⁺ | 18 | `Z`, `system`, `basis` | FCI, 5ZaPa-NR-CV |
| `atoms_ions/atoms_ions_W0_Winf_Wpinf.csv` | Neutral atoms and ions, H … Zn | 19 | `Z`, `system`, `N_electrons`, `basis` | CCSD(T) (CCSD for Zn) |
| `h2_dissociation/h2_W0_Winf_Wpinf.csv` | H₂ bond stretching | 296 | `R_bohr` (0.3–11 a₀) | FCI |
| `ne_basis/ne_basis_Winf_Wpinf.csv` | Ne, basis-set series 2Z–5ZaPa-NR-CV (SI Table S4) | 4 | `system`, `basis` | CCSD(T) |
| `make_data.py` | script that rebuilds every table above from `../data_JPCL/` | | | |

The `density_type` column in every file names the correlated method that produced the target density.

### Notes by system

- **Hookium, He series, atoms and ions:** the values are those of the manuscript tables and figures, taken from the same data used in `../data_JPCL/`.
- **H₂:** `W_inf` is the dual-Kantorovich (cylindrical) value and `W'_inf` is the restricted-mode zero-point value (rmZPE), as in Figs. 8 and 9 of the manuscript. Total energies in the manuscript use only R ≤ 7.66 a₀, where the density inversion converges. At R = 2.39292 a₀ the SCE values were not computed, so `W_inf` and `W'_inf` are empty there.
- **Ne basis series:** W₀ was not stored for these runs, so the file has only `W_inf` and `W'_inf`.

---

## Quick start

The files are plain UTF-8 CSV with one header line.

```python
import pandas as pd

he = pd.read_csv("he_series/he_series_W0_Winf_Wpinf.csv")
print(he[["system", "W_0", "W_inf", "W'_inf"]])
```

Note that the header `W'_inf` contains an apostrophe, so it must be quoted as `"W'_inf"` in Python.

## Rebuilding the tables

```bash
python3 make_data.py                         # reads ../data_JPCL, writes here
python3 make_data.py --src /path/to/data_JPCL --out .
```

The script needs only pandas.

## Related material

| Path | Contents |
|---|---|
| `../data_JPCL/` | full data with all model values, plotting scripts, `run_all.sh` (reproduces all 18 figures), `MANIFEST.sha256` |
| `../JAXSCE_SCE/` | SCE implementation and calculation drivers |
| https://github.com/gchirchir/SCE1 | public repository with this folder, the full data and the code |

## License

MIT. See `../LICENSE`.
