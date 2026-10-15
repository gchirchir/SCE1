# JAXSCE_SCE: exact strong-coupling functionals from correlated PySCF densities

JAXSCE_SCE computes the exact strictly-correlated-electron (SCE) limit of the
adiabatic connection, the leading coefficient W∞[ρ] and the zero-point
coefficient W′∞[ρ], for densities obtained from correlated wave-function
calculations in PySCF. It is the code behind the reference data of

> G. Chirchir, A. Bhattacharjee, S. Jana, S. Śmiga, *Probing the Accuracy of
> Adiabatic-Connection Exchange–Correlation Functionals with Highly Accurate,
> Strongly Correlated Electron Data*.

## What the code does

- **Reference densities from PySCF.** Hartree–Fock, CCSD, orbital-relaxed
  CCSD(T) (from the Λ-amplitude equations, for closed- and open-shell
  references), and FCI one-particle density matrices are generated with PySCF,
  transformed to the atomic-orbital basis, spin-summed, and turned into the
  spherically averaged radial density and its cumulant N_e(r), together with the
  Hartree energy U[ρ] (`jaxsce/densities/pyscf.py`, `relaxed_dm.py`).
- **Exact SCE ingredients.** From N_e(r), the co-motion functions are obtained
  by numerical inversion of the cumulant. The electron–electron repulsion at each
  radial point is minimized over the electron angles, and the result is
  integrated (with interpolation and extrapolation) to give V_ee^SCE[ρ] and
  W∞[ρ] = V_ee^SCE[ρ] − U[ρ].
- **Zero-point coefficient W′∞.** The drivers evaluate W′∞ from the normal-mode
  frequencies around the SCE configuration: in closed form for two electrons,
  with a planar co-motion treatment for four electrons, and from the local Hessian
  of the angular problem for other electron numbers.
- **Benchmark drivers.** For each benchmark set, the `star.py` drivers also
  evaluate the semilocal strong-coupling models (LDA, PBE, TPSS, r²SCAN, PC,
  mPC, hPC, fPC, ePC), the weak-coupling input from Kohn–Sham orbitals of the
  PyOEP-inverted density, and the SPL and genISI2 adiabatic-connection
  interpolations. An optional Wu–Yang inversion (KS-pies) is also provided.

## Layout

| Path | Contents |
|---|---|
| `jaxsce/` | SCE co-motion optimization, Hessian/zero-point modes, integration, PySCF density classes |
| `drivers/atoms_ions/star.py` | Atoms and ions (Table 1, Figure 10): HF/CCSD/CCSD(T)/FCI density → W∞, W′∞ |
| `drivers/ne_series/star.py` | Ne basis-set series (SI Tables S4–S5) |
| `drivers/he_series/star.py` (+ `wu_yang.py`) | He isoelectronic series H⁻–Ar¹⁶⁺ (Figures 4–6, S2, S6) |
| `drivers/hookium/star.py` (+ `wu_yang.py`) | Hookium atom (Figures 1–3, S1, S5) |
| `drivers/h2_dissociation/star.py` (+ `wu_yang.py`, `collect_h2_range.py`) | H₂ dissociation (Figures 7–9) |
| `examples/run_he.sh` | 20-second example (He atom) |
| `tests/test_he_smoke.py` | Smoke test of the full SCE pipeline |

## Installation

Python 3.10–3.12. The versions in `requirements.txt` were tested together:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes `import jaxsce` available
```

Version constraints that matter: TensorFlow Probability 0.25 needs JAX 0.4.x
(tested with 0.4.38), and SciPy must be < 1.17 because KS-pies imports
`scipy.special.sph_harm`.

### Installing KS-pies (only for the Hookium, He-series and H₂ drivers)

KS-pies 1.0.3 on PyPI builds with `numpy.distutils`, which recent NumPy no
longer provides. Its Fortran extension is optional, so the pure-Python modules
can be installed directly:

```bash
pip download --no-deps --no-binary :all: kspies==1.0.3   # or fetch kspies_source-1.0.3.tar.gz from PyPI
tar xzf kspies_source-1.0.3.tar.gz
cp -r kspies-1.0.3/kspies "$(python -c 'import site; print(site.getsitepackages()[0])')"/
```

The Hookium, He-series and H₂ drivers additionally use the **PyOEP** package
(density inversion; it provides the `methods` module with `exxoep` and `KSINV`).
Point the `PYOEP_ROOT` environment variable to your PyOEP checkout.

## Quick check

```bash
./examples/run_he.sh              # or: python -m pytest -q tests
```

Expected output (He, HF density, aug-cc-pVTZ): `W_inf ≈ -1.4993`,
`T_ZP (W'∞) ≈ 0.6207` E_H (literature −1.5008 and 0.621). This was verified on
a CPU-only workstation (Python 3.12, JAX 0.4.38, PySCF 2.14) on 2026-09-24.

## Production runs

All numbers in the paper were produced with these drivers on the WCSS (Wrocław) and
UMK (Toruń) clusters. The Hookium and He-series drivers are the revised versions: their
two-electron SCE integration normalizes N_e(r) to exactly 2 and uses a radial grid adapted
to the extent of the density.
Each `star.py` takes the system on the command line, for example

```bash
cd drivers/atoms_ions
python star.py --atom Ne --basis 5ZaPa-NR-CV --density ccsdt --N-grid 1025
python star.py --help           # all options; without --atom, runs the built-in batch list
```

JAX uses a GPU if one is available and falls back to the CPU otherwise.

## W′∞ convention

All W′∞ values use the prefactor 1/4, W′∞ = 1/(4N) ∫ n(r) Σ_μ ω_μ(r) dr, called
the "PRA" convention in the Supporting Information (Section S9). Output
columns named `T_ZP_prl` hold values in this convention despite their name.

## License

The package is released under the MIT License. The included jaxsce
implementation retains the attribution to its original authors in `LICENSE`.
