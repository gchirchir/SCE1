# SCE benchmark data, scripts, and code

Data, analysis scripts, and code accompanying

> G. Chirchir, A. Bhattacharjee, S. Jana, and S. Śmiga,
> *Probing the Accuracy of Adiabatic-Connection Exchange–Correlation Functionals
> with Highly Accurate, Strongly Correlated Electron Data*.

The repository provides reference values of the strong-coupling coefficients
W∞[ρ] and W′∞[ρ] from correlated reference densities for Hookium (20
confinement frequencies), the He isoelectronic series (H⁻–Ar¹⁶⁺), H₂
dissociation, and neutral atoms and ions (H–Xe), together with the semilocal
strong-coupling models and the SPL and genISI2 adiabatic-connection
interpolations evaluated on the same densities.

## Contents

| Path | Contents |
|---|---|
| `data_JPCL/` | Reference CSV data for all four benchmark sets, the scripts that plot every figure of the paper and its Supporting Information and that generate the SI tables, and the `star.py` drivers that produced the data. `./run_all.sh` regenerates all 18 figures and checks them pixel by pixel against the published versions in `data_JPCL/reference_figures/`. See `data_JPCL/README.md`. |
| `JAXSCE_SCE/` | Installable SCE code computing the exact W∞ and W′∞ of correlated PySCF densities, the benchmark drivers, and a He test case. See `JAXSCE_SCE/README.md`. |

Integrity: `sha256sum -c MANIFEST.sha256` inside `data_JPCL/` checks every data
file, script, and reference figure.

## Quick start

```bash
cd data_JPCL
./run_all.sh          # needs Python 3 with numpy, pandas, matplotlib, scipy, Pillow, and Ghostscript
```

Expected last line: `18/18 figures reproduce the manuscript exactly.`

Re-running the electronic-structure and SCE calculations additionally needs
PySCF, JAX, KS-pies, and PyOEP (see `JAXSCE_SCE/README.md`).

## Citation and license

Please cite the paper above; citation metadata are in `CITATION.cff`. The code
is released under the MIT License (`LICENSE`); `JAXSCE_SCE/` builds on the
jaxsce code of D. P. Kooi and K. J. Daas (see `JAXSCE_SCE/LICENSE`).
