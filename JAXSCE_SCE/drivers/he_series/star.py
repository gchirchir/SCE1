
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STREAMLINED SCE OPTIMIZER with WU-YANG + PYOEP DENSITY INVERSION

Architecture:
  1. Generate the two-electron target density directly with PySCF
  2. Independently invert the same target with Wu-Yang and PyOEP
  3. Prefer Wu-Yang; use converged PyOEP if Wu-Yang fails
  4. Evaluate analytical SCE, DFT, W_inf and W'_inf on that density

Key optimization: Reuse existing densities - just copy and overwrite with WY result
No need to recompute CCSD/CCSDT/FCI from scratch for WY runs.
"""

import os
import argparse
import csv
import math
import re as _re
import sys
import copy as _copy
from types import SimpleNamespace
from timeit import default_timer as timer
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d
from pyscf import cc, scf, gto, dft, mp, fci

from wu_yang import (
    DFT_MODELS,
    compute_W_for_density,
    make_exc_func,
    model_xctype,
)

# Independent PyOEP inversion.  The launcher copies PyOEP/methods beside this
# script.  PyOEP is attempted even when kspies/Wu-Yang is unavailable or fails.
_PYOEP_ROOT = os.environ.get(
    "PYOEP_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "PyOEP")
)
if os.path.isdir(_PYOEP_ROOT) and _PYOEP_ROOT not in sys.path:
    sys.path.insert(0, _PYOEP_ROOT)
try:
    from methods import exxoep as _PYOEP_EXXOEP_MODULE
    PYOEP_EXXOEP_AVAILABLE = True
except ImportError:
    _PYOEP_EXXOEP_MODULE = None
    PYOEP_EXXOEP_AVAILABLE = False
try:
    from methods.ksinv import KSINV as _KSINV
    from methods.osksinv import OSKSINV as _OSKSINV
    PYOEP_AVAILABLE = True
    _PYOEP_IMPORT_ERROR = ""
    print("[pyoep] EXXOEP/KSINV/OSKSINV sources detected", flush=True)
except ImportError as exc:
    _KSINV = _OSKSINV = None
    PYOEP_AVAILABLE = False
    _PYOEP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[pyoep] WARNING: comparison unavailable: {_PYOEP_IMPORT_ERROR}",
          flush=True)

# -----------------------------------------------------------------------------
# WU-YANG KS INVERSION (kspies)
# -----------------------------------------------------------------------------
try:
    from kspies import wy as _wy
    KSPIES_AVAILABLE = True
    print("[wy] kspies available", flush=True)
except ImportError:
    KSPIES_AVAILABLE = False
    print("[wy] WARNING: kspies not installed; --wy disabled", flush=True)

INVERSION_RHO_L2_TOL = float(os.environ.get("INVERSION_RHO_L2_TOL", "0.05"))

def _collapse_dm(dm):
    dm = np.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm

# -----------------------------------------------------------------------------
# PYSCF 2.x TREUTLER COMPATIBILITY PATCH
# -----------------------------------------------------------------------------
import inspect as _inspect
from pyscf.dft import radi as _radi

if not getattr(_radi, "_treutler_patched", False):
    _real_treutler = _radi.treutler_ahlrichs
    _n_req = sum(
        1 for p in _inspect.signature(_real_treutler).parameters.values()
        if p.default is _inspect.Parameter.empty
    )

    def _treutler_compat(n, chg=1):
        return _real_treutler(n, chg) if _n_req >= 2 else _real_treutler(n)

    _radi.treutler = _radi.treutler_ahlrichs = _treutler_compat
    _radi._treutler_patched = True

# -----------------------------------------------------------------------------
# BASIS-SET-EXCHANGE LAYER
# -----------------------------------------------------------------------------
try:
    import basis_set_exchange as bse
    BSE_AVAILABLE = True
    print("[basis] basis_set_exchange available", flush=True)
except ImportError:
    bse = None
    BSE_AVAILABLE = False
    print("[basis] WARNING: basis_set_exchange not installed.", flush=True)

_BASIS_CACHE: dict = {}
_BASIS_ALIASES: dict = {
    "jorge-6pz": "jorge-6ZP", "jorge-5pz": "jorge-5ZP", "jorge-4pz": "jorge-4ZP",
    "jorge-3pz": "jorge-3ZP", "jorge-2pz": "jorge-2ZP", "jorge-dzp": "jorge-DZP",
    "jorge-tzp": "jorge-TZP", "jorge-qzp": "jorge-QZP",
    "jorge-aqzp": "jorge-AQZP",
    "aug-cc-pvdz-pp": "aug-cc-pVDZ-PP", "aug-cc-pvtz-pp": "aug-cc-pVTZ-PP",
    "aug-cc-pvqz-pp": "aug-cc-pVQZ-PP", "aug-cc-pv5z-pp": "aug-cc-pV5Z-PP",
    "aug-cc-pv6z-pp": "aug-cc-pV6Z-PP", "aug-cc-pcvdz": "aug-cc-pCVDZ",
    "aug-cc-pcvtz": "aug-cc-pCVTZ", "aug-cc-pcv4z": "aug-cc-pCV4Z",
    "aug-cc-pcv5z": "aug-cc-pCV5Z", "aug-cc-pcv6z": "aug-cc-pCV6Z",
}
_UNC_PREFIX = "unc-"

def _normalise_basis_name(name: str) -> str:
    lo = name.lower()
    if lo in _BASIS_ALIASES:
        return _BASIS_ALIASES[lo]
    if lo.endswith("pz"):
        return name[:-2] + "ZP"
    return name

def _is_uncontracted(name: str):
    if name.lower().startswith(_UNC_PREFIX):
        return True, name[len(_UNC_PREFIX):]
    return False, name

def fetch_basis_nwchem(basis_name: str, atom_symbol: str):
    if not BSE_AVAILABLE:
        return None
    is_unc, base_name = _is_uncontracted(basis_name)
    base_name = _normalise_basis_name(base_name)
    cache_key = (basis_name.lower(), atom_symbol.upper())
    if cache_key in _BASIS_CACHE:
        return _BASIS_CACHE[cache_key]
    try:
        kw = {"elements": [atom_symbol], "fmt": "nwchem", "header": False}
        if is_unc:
            kw.update({"uncontract_general": True, "uncontract_spdf": True,
                       "uncontract_segmented": True})
        result = bse.get_basis(base_name, **kw)
        _BASIS_CACHE[cache_key] = result
        print(f"  [basis] fetched '{basis_name}' for {atom_symbol}", flush=True)
        return result
    except Exception as exc:
        print(f"  [basis] WARNING: Could not fetch '{basis_name}' "
              f"for {atom_symbol}: {exc}", flush=True)
        _BASIS_CACHE[cache_key] = None
        return None

def _split_basis_ecp(basis_string: str):
    basis_match = _re.search(r"(^BASIS\b.*?^END\b)", basis_string,
                              _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
    ecp_match = _re.search(r"(^ECP\b.*?^END\b)", basis_string,
                            _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
    return (
        basis_match.group(1) if basis_match else basis_string,
        ecp_match.group(1) if ecp_match else None,
    )

def load_basis_for_pyscf(basis_name: str, atom_symbol: str):
    from pyscf.gto.basis import parse_nwchem
    basis_string = fetch_basis_nwchem(basis_name, atom_symbol)
    if basis_string is None:
        print(f"  [basis] using PySCF internal '{basis_name}' "
              f"for {atom_symbol}", flush=True)
        return basis_name, False, None
    basis_block, ecp_block = _split_basis_ecp(basis_string)
    try:
        basis_dict = {atom_symbol: parse_nwchem.parse(basis_block)}
    except Exception as exc:
        print(f"  [basis] WARNING: Failed to parse basis: {exc}", flush=True)
        return basis_name, False, None
    ecp_dict = None
    has_ecp = False
    if ecp_block is not None:
        from pyscf.gto.basis import parse_nwchem_ecp
        for parser_name in ("parse_ecp", "parse"):
            parser = getattr(parse_nwchem_ecp, parser_name, None)
            if parser is None:
                continue
            try:
                ecp_dict = {atom_symbol: parser(ecp_block)}
                has_ecp = True
                print(f"  [basis] ECP loaded for {atom_symbol}", flush=True)
                break
            except Exception:
                continue
        if not has_ecp:
            print(f"  [basis] WARNING: ECP parse failed for {atom_symbol}",
                  flush=True)
    return basis_dict, has_ecp, ecp_dict

# -----------------------------------------------------------------------------
# REFERENCE DATA
# -----------------------------------------------------------------------------
MOLE_REFERENCES = {
    "H": {"Nel": 1, "spin": 1, "W_ref": -0.3125, "Wp_ref": 0.0},
    "H-": {"Nel": 2, "spin": 0, "W_ref": -0.569, "Wp_ref": None},
    "He": {"Nel": 2, "spin": 0, "W_ref": -1.5008, "Wp_ref": 0.621},
    "He-": {"Nel": 3, "spin": 1, "W_ref": None, "Wp_ref": None},
    "Li": {"Nel": 3, "spin": 1, "W_ref": -2.603, "Wp_ref": 1.38},
    "Li-": {"Nel": 4, "spin": 0, "W_ref": -2.7308, "Wp_ref": 1.52075},
    "Be": {"Nel": 4, "spin": 0, "W_ref": -4.021, "Wp_ref": 2.59},
    "B": {"Nel": 5, "spin": 1, "W_ref": -5.706, "Wp_ref": 4.2},
    "C": {"Nel": 6, "spin": 2, "W_ref": -7.782, "Wp_ref": 6.3},
    "N": {"Nel": 7, "spin": 3, "W_ref": -10.246, "Wp_ref": None},
    "O": {"Nel": 8, "spin": 2, "W_ref": -10.335, "Wp_ref": 13.0},
    "F": {"Nel": 9, "spin": 1, "W_ref": None, "Wp_ref": None},
    "F-": {"Nel": 10, "spin": 0, "W_ref": -16.940, "Wp_ref": None},
    "Ne": {"Nel": 10, "spin": 0, "W_ref": -20.035, "Wp_ref": 22.0},
    "Na": {"Nel": 11, "spin": 1, "W_ref": None, "Wp_ref": None},
    "Mg": {"Nel": 12, "spin": 0, "W_ref": -26.7297, "Wp_ref": None},
    "Cl-": {"Nel": 18, "spin": 0, "W_ref": -47.26, "Wp_ref": None},
    "Ar": {"Nel": 18, "spin": 0, "W_ref": -51.5550, "Wp_ref": None},
    "Ca": {"Nel": 20, "spin": 0, "W_ref": -60.485716, "Wp_ref": None},
    "Zn": {"Nel": 30, "spin": 0, "W_ref": None, "Wp_ref": None},
    "Kr": {"Nel": 36, "spin": 0, "W_ref": -166.8500, "Wp_ref": None},
    "Xe": {"Nel": 54, "spin": 0, "W_ref": -322.8348, "Wp_ref": None},
}

# -----------------------------------------------------------------------------
# PER-SYSTEM BASIS SETS (batch mode)
# -----------------------------------------------------------------------------
SYSTEM_BASIS = [
     ("H",  -1, 0, "5ZaPa-NR-CV"),
    ("He",  0, 0, "5ZaPa-NR-CV"),
     ("Li",  1, 0, "5ZaPa-NR-CV"),
     ("Be",  2, 0, "5ZaPa-NR-CV"),
     ("B",   3, 0, "5ZaPa-NR-CV"),
     ("C",   4, 0, "5ZaPa-NR-CV"),
     ("N",   5, 0, "5ZaPa-NR-CV"),
     ("Ne",  8, 0, "5ZaPa-NR-CV"),
     ("Na",  9, 0, "5ZaPa-NR-CV"),
     ("Mg", 10, 0, "5ZaPa-NR-CV"),
     ("Al", 11, 0, "5ZaPa-NR-CV"),
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def atom_label(atom: str, charge: int) -> str:
    return atom.strip() + "-" if charge == -1 else atom.strip()

def _density_folder(density_type: str) -> str:
    return density_type.lower().replace("(", "").replace(")", "")

def _filename_suffix(density_type: str) -> str:
    if density_type == "ccsd":
        return "_ccsd"
    if density_type in ("ccsd(t)", "ccsdt"):
        return "_ccsdt"
    if density_type == "fci":
        return "_fci"
    return ""

# -----------------------------------------------------------------------------
# WU-YANG INVERSION FUNCTIONS
# -----------------------------------------------------------------------------
def _target_dm_for_wy(density, mol):
    """Prepare target DM for WY inversion: 2-D for RHF, (dm_a, dm_b) for UHF."""
    dm = np.asarray(density.dm)
    if mol.spin == 0:
        return dm[0] + dm[1] if dm.ndim == 3 else dm
    if dm.ndim == 3 and dm.shape[0] == 2:
        return dm[0], dm[1]
    # Correlated open-shell DM stored spin-summed: borrow UHF magnetisation
    dm_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    da, db = mf.make_rdm1()
    mag = da - db
    return 0.5 * (dm_tot + mag), 0.5 * (dm_tot - mag)

def _hartree_U(mol, dm2d):
    """U = 1/2 Tr(D J[D]) from spin-summed AO density."""
    j = np.asarray(scf.hf.get_jk(mol, np.asarray(dm2d), hermi=1, with_k=False)[0])
    if j.ndim == 3:
        j = j[0] + j[1]
    return float(np.einsum("ij,ji->", dm2d, j) * 0.5)

def _rho_l2_diff(mol, dm1, dm2):
    """L2 norm of the total-density difference on a level-9 DFT grid."""
    grids = dft.gen_grid.Grids(mol)
    grids.level = 9
    grids.build()
    ao = dft.numint.eval_ao(mol, grids.coords)

    def total_dm(dm):
        arr = np.asarray(dm)
        return arr[0] + arr[1] if arr.ndim == 3 else arr

    rho1 = np.einsum("gi,ij,gj->g", ao, total_dm(dm1), ao)
    rho2 = np.einsum("gi,ij,gj->g", ao, total_dm(dm2), ao)
    return float(np.sqrt(np.dot(grids.weights, (rho1 - rho2) ** 2)))


def run_wy_inversion(density, mol, verbose=False):
    """
    Wu-Yang inversion of density.dm.
    Returns the inverted spin-summed 2-D DM.
    """
    if not KSPIES_AVAILABLE:
        raise RuntimeError("kspies unavailable — cannot run --wy")

    target = _target_dm_for_wy(density, mol)
    if mol.spin == 0:
        inv = _wy.RWY(mol, target)
        wy_name = "RWY"
    else:
        da, db = target
        inv = _wy.UWY(mol, np.array([da, db]))
        wy_name = "UWY"

    if verbose:
        print(f"  [WY] Running {wy_name} inversion...", flush=True)

    inv.run()
    if not bool(getattr(inv, "converged", False)):
        raise RuntimeError(f"{wy_name} did not converge for target density")

    if verbose:
        inv.info()

    rho_diff = _rho_l2_diff(mol, target, inv.dm)
    return inv, rho_diff

def apply_wy_to_density(density, mol, verbose=False):
    """
    In-place Wu-Yang inversion of density.dm.
    Updates density.dm and density.U in place.
    Returns (wy_dm, converged).
    """
    inv, rho_diff = run_wy_inversion(density, mol, verbose=verbose)
    if not inv.converged:
        raise RuntimeError(
            f"Wu-Yang inversion did not converge (rho_diff={rho_diff:.3e}); "
            "SCE and DFT were not run"
        )

    dm_raw = np.asarray(inv.dm, dtype=np.float64)
    dm_spin_wy = None
    if isinstance(inv.dm, (tuple, list)) or dm_raw.ndim == 3:
        da, db = np.asarray(inv.dm[0]), np.asarray(inv.dm[1])
        dm_spin_wy = (da, db)
        wy_dm = da + db
    else:
        wy_dm = dm_raw

    density.dm = np.asarray(wy_dm, dtype=np.float64)
    density.U = _hartree_U(mol, density.dm)
    density.Nel = int(density.Nel)
    print(f"  [WY] Inversion converged. U(WY) = {density.U:.10f}", flush=True)
    return density.dm, dm_spin_wy, inv, rho_diff


def run_pyoep_comparison(mol, dm_target, e_ref, system_basis,
                         verbose=False):
    """Run independent KSINV/OSKSINV in the working system-basis setup."""
    method = "OSKSINV" if mol.spin else "KSINV"
    space_sym = os.environ.get("PYOEP_SPACE_SYM", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )
    outcome = {
        "pyoep_available": bool(PYOEP_AVAILABLE),
        "pyoep_exxoep_available": bool(PYOEP_EXXOEP_AVAILABLE),
        "pyoep_method": method,
        "pyoep_orbital_basis": system_basis,
        "pyoep_oep_basis": system_basis,
        "pyoep_same_system_basis": True,
        "pyoep_space_sym": space_sym,
        "pyoep_maxiter": int(os.environ.get("PYOEP_MAXITER", "300")),
        "pyoep_conv_thr": float(os.environ.get("PYOEP_CONV_THR", "1e-10")),
        "pyoep_thr_fai_oep": float(
            os.environ.get("PYOEP_THR_FAI_OEP", "5e-2")
        ),
        "pyoep_converged": False,
        "pyoep_rho_diff": float("nan"),
        "pyoep_error": _PYOEP_IMPORT_ERROR,
    }
    if not PYOEP_AVAILABLE:
        return None, None, None, outcome

    try:
        pmol = mol.copy()
        pmol.symmetry = False
        pmol.verbose = 0
        mf = scf.UHF(pmol) if pmol.spin else scf.RHF(pmol)
        mf.verbose = 0
        mf.conv_tol = 1e-10
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("PyOEP reference HF did not converge")

        # Working configuration validated for ZaPa: use the exact parsed
        # system basis as the PyOEP potential basis with atomic space symmetry.
        oep_basis = {key: value for key, value in pmol._basis.items()}
        inv_cls = _OSKSINV if pmol.spin else _KSINV
        # Own copy: PyOEP must never receive or mutate the Wu-Yang object.
        target = np.array(dm_target, dtype=np.float64, copy=True)
        # PyOEP's EXXOEP.init_common calls a module-level deepcopy(mf).
        # A completed PySCF object contains unpicklable stdout, DIIS and
        # temporary integral/checkpoint streams.  Install a narrowly-scoped
        # PySCF-aware copier for this constructor call.  It creates a distinct
        # molecule and distinct orbital arrays, while forcing all integral and
        # DIIS state to be rebuilt inside the independent PyOEP calculation.
        def _copy_array_tree(value):
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True)
            if isinstance(value, tuple):
                return tuple(_copy_array_tree(v) for v in value)
            if isinstance(value, list):
                return [_copy_array_tree(v) for v in value]
            return value

        def _independent_pyscf_copy(source_mf):
            copied = _copy.copy(source_mf)
            copied.mol = source_mf.mol.copy()
            copied.mo_coeff = _copy_array_tree(source_mf.mo_coeff)
            copied.mo_occ = _copy_array_tree(source_mf.mo_occ)
            copied.mo_energy = _copy_array_tree(source_mf.mo_energy)
            copied._eri = None
            # REV1 compat (2026-10-14): PySCF >= 2.5 keeps _opt as a dict {omega: vhfopt} and calls
            # self._opt.get(omega); None crashes there ('NoneType' has no attribute 'get'). PySCF 2.3 uses None.
            copied._opt = {None: None} if isinstance(getattr(source_mf, "_opt", None), dict) else None
            copied.diis = None
            copied.chkfile = None
            copied.stdout = None
            copied.mol.stdout = None
            return copied

        mf_stdout = getattr(mf, "stdout", None)
        mol_stdout = getattr(mf.mol, "stdout", None)
        mf.stdout = None
        mf.mol.stdout = None
        original_pyoep_deepcopy = _PYOEP_EXXOEP_MODULE.deepcopy
        _PYOEP_EXXOEP_MODULE.deepcopy = _independent_pyscf_copy
        try:
            inv = inv_cls(
                mf, oep_basis, target, float(e_ref),
                space_sym=space_sym,
            )
        finally:
            _PYOEP_EXXOEP_MODULE.deepcopy = original_pyoep_deepcopy
            mf.stdout = mf_stdout
            mf.mol.stdout = mol_stdout

        # Restore usable handles on PyOEP's private deep-copied objects.
        inv.mf.stdout = mf_stdout if mf_stdout is not None else sys.stdout
        inv.mf.mol.stdout = (
            mol_stdout if mol_stdout is not None else inv.mf.stdout
        )
        inv.run(maxit=outcome["pyoep_maxiter"],
                thr_fai_oep=outcome["pyoep_thr_fai_oep"],
                conv_thr=outcome["pyoep_conv_thr"],
                mixing_a=0.1, mixing_i=3)

        raw = np.asarray(inv.mf.make_rdm1(), dtype=np.float64)
        dm_spin = (raw[0], raw[1]) if raw.ndim == 3 else None
        dm_total = raw[0] + raw[1] if raw.ndim == 3 else raw
        rho_diff = _rho_l2_diff(pmol, target, raw)
        converged = bool(getattr(inv, "converged", False))
        outcome.update({
            "pyoep_converged": converged,
            "pyoep_rho_diff": float(rho_diff),
            "pyoep_error": "" if converged else "KS inversion did not converge",
        })
        if verbose:
            print(f"  [PyOEP/{method}] system/OEP basis={system_basis}, "
                  f"rho_L2={rho_diff:.3e}, converged={converged}", flush=True)
        return inv, dm_total, dm_spin, outcome
    except Exception as exc:
        outcome["pyoep_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [PyOEP/{method}] FAILED independently: "
              f"{outcome['pyoep_error']}", flush=True)
        return None, None, None, outcome


def compute_gl2(mol, wy_mw):
    """MP2 doubles energy evaluated on the converged WY KS orbitals."""
    source = getattr(wy_mw, "mf", wy_mw)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    mf.kernel()
    mf.mo_coeff = np.asarray(source.mo_coeff)
    mf.mo_occ = np.asarray(source.mo_occ)
    mf.mo_energy = np.asarray(source.mo_energy)
    solver = mp.MP2(mf) if mol.spin == 0 else mp.UMP2(mf)
    result = solver.kernel()
    return float(np.asarray(result[0] if isinstance(result, tuple) else result).flat[0])


def _total_energies(mol, density_type, datadir):
    """Return cached (method, HF) total energies for E_c using CSV."""
    method = density_type.lower().replace("(", "").replace(")", "")
    cache_path = os.path.join(datadir, f"total_energies_{method}.csv")
    try:
        with open(cache_path, newline="", encoding="utf-8") as handle:
            cached = next(csv.DictReader(handle))
        return float(cached["E_method"]), float(cached["E_hf"])
    except (OSError, StopIteration, KeyError, TypeError, ValueError):
        pass

    mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 2000
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF energy calculation did not converge")
    e_hf = float(mf.e_tot)
    if method in ("hf", "hartree-fock"):
        e_method = e_hf
    elif method == "fci":
        from pyscf import fci
        e_method = float(np.atleast_1d(fci.FCI(mf).kernel(nroots=1)[0])[0])
    elif method in ("ccsd", "ccsdt"):
        solver = cc.CCSD(mf)
        solver.kernel()
        e_method = float(solver.e_tot)
        if method == "ccsdt":
            e_method += float(solver.ccsd_t())
    else:
        raise ValueError(f"Unsupported energy method: {density_type}")
    with open(cache_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["density_type", "E_method", "E_hf"]
        )
        writer.writeheader()
        writer.writerow({
            "density_type": density_type,
            "E_method": f"{e_method:.16g}",
            "E_hf": f"{e_hf:.16g}",
        })
    return e_method, e_hf


def compute_extra_quantities(mol, dm_inverted, density_type, datadir,
                             inversion_mw, wy_ok, wy_rho_diff,
                             inversion_source, verbose=False):
    """Return EXX/GL2 quantities for the selected production inversion."""
    U_inverted = _hartree_U(mol, np.asarray(dm_inverted))
    # All systems handled by this driver are closed-shell two-electron ions,
    # for which exact exchange is exactly -U/2.
    W0 = -U_inverted / 2.0
    e_method, e_hf = _total_energies(mol, density_type, datadir)
    E_c = float(e_method - e_hf)
    E_xc = float(W0 + E_c)
    gl2 = float("nan")
    W0_prime = float("nan")
    if inversion_mw is not None and bool(getattr(inversion_mw, "converged", False)):
        try:
            gl2 = compute_gl2(mol, inversion_mw)
            W0_prime = 2.0 * gl2
            if verbose:
                print(f"  [GL2] GL2={gl2:.10f}, W0_prime={W0_prime:.10f}",
                      flush=True)
        except Exception as exc:
            if verbose:
                print(f"  [GL2] failed: {exc}", flush=True)
    return {
        "W0": W0,
        "E_c": E_c,
        "E_xc": E_xc,
        "W0_prime": W0_prime,
        "gl2_2x": 2.0 * gl2 if math.isfinite(gl2) else float("nan"),
        "inversion_source": inversion_source,
        "inversion_ok": bool(getattr(inversion_mw, "converged", False)),
        "WY_ok": bool(wy_ok),
        "wy_rho_diff": float(wy_rho_diff),
    }


# Adiabatic-connection models.  Here gl2 is the raw GL2 value, so
# W0_prime = 2*gl2.
def compute_isi_derivatives(ex, gl2, w, w1):
    if abs(gl2) < 1e-12:
        return 0.0
    x, z = -4.0 * gl2, ex - w
    xx = x * w1**2 / z**2
    yy = x**2 * w1**2 / z**4
    zz = x * w1**2 / z**3 - 1.0
    root = np.sqrt(1.0 + yy)
    return w + 2.0 * xx * (
        root - 1.0 - zz * np.log((root + zz) / (1.0 + zz))
    ) / yy - ex


def compute_spl_derivatives(ex, gl2, w):
    if abs(gl2) < 1e-12:
        return 0.0
    chi = 2.0 * gl2 / (w - ex)
    return (ex - w) * (np.sqrt(1.0 + 2.0 * chi) - 1.0 - chi) / chi


def compute_rev_derivatives(ex, gl2, w, w1):
    if abs(gl2) < 1e-12:
        return 0.0
    aa = ex - w
    b = -8.0 * gl2 * w1**2 / aa**2
    c = -2.0 * b * gl2 / aa**2
    d = -1.0 + b / aa
    return w + b / (np.sqrt(1.0 + c) + d) - ex


def compute_gisi2_derivatives(ex, gl2, w, w1):
    from scipy.integrate import quad
    if abs(gl2) < 1e-12:
        return 0.0

    def integrand(lam):
        d = 3.5
        b = (ex - w) * (1.0 + d)
        c = b**2 / (4.0 * w1**2)
        root = np.sqrt(1.0 + c * lam)
        uisi = w + b * (2.0 + c * lam + 2.0 * d * root) / (
            2.0 * root * (d + root)**2
        )
        p = 2.0 * gl2 / ex
        a2 = -(uisi - ex) / (p * lam) if abs(p * lam) > 1e-30 else 0.0
        ratio3 = (ex / w)**3
        wa = ex * p * lam / (1.0 + 10.65 * ratio3 * p * lam)**3
        wb = a2 * p * lam / (1.0 + 3.6 * ratio3 * p * lam)**3
        return uisi + wa + wb - ex

    value, _ = quad(integrand, 0.0, 1.0, limit=200)
    return float(value)


def compute_spl2(ex, w, gl2):
    if abs(gl2) < 1e-12:
        return 0.0
    m2, b2 = 10.68, 0.117
    wab = 1.1472 * w - 1.7397 * ex
    m1 = wab - m2
    b1 = (b2 * m2 - 4.0 * gl2) / (m2 - wab)
    return (wab
            - 2.0 * m1 * (np.sqrt(1.0 + b1) - 1.0) / b1
            - 2.0 * m2 * (np.sqrt(1.0 + b2) - 1.0) / b2)


def compute_all_ac(ex, gl2, w, w1, e_hf):
    keys = (
        "eisi_corr", "spl_corr", "gisi2_corr", "rvisl_corr", "spl2_corr",
        "E_isi", "E_spl", "E_gisi2", "E_rvisl", "E_spl2", "E_gl2_tot",
    )
    if not all(np.isfinite(v) for v in (ex, gl2, w, w1, e_hf)):
        return {key: float("nan") for key in keys}
    try:
        corr = {
            "eisi_corr": compute_isi_derivatives(ex, gl2, w, w1),
            "spl_corr": compute_spl_derivatives(ex, gl2, w),
            "gisi2_corr": compute_gisi2_derivatives(ex, gl2, w, w1),
            "rvisl_corr": compute_rev_derivatives(ex, gl2, w, w1),
            "spl2_corr": compute_spl2(ex, w, gl2),
        }
        return {
            **{key: float(value) for key, value in corr.items()},
            "E_isi": float(e_hf + corr["eisi_corr"]),
            "E_spl": float(e_hf + corr["spl_corr"]),
            "E_gisi2": float(e_hf + corr["gisi2_corr"]),
            "E_rvisl": float(e_hf + corr["rvisl_corr"]),
            "E_spl2": float(e_hf + corr["spl2_corr"]),
            "E_gl2_tot": float(e_hf + gl2),
        }
    except (ArithmeticError, FloatingPointError, ValueError, ZeroDivisionError):
        return {key: float("nan") for key in keys}


def compute_dft_models(mol, dm_wy, dm_spin_wy=None, verbose=False):
    """Evaluate every wu_yang.py DFT model on one converged WY density."""
    spin = 0 if mol.spin == 0 else 1
    dm_dft = dm_wy if spin == 0 else dm_spin_wy
    if spin and dm_dft is None:
        raise RuntimeError("open-shell DFT requires the alpha/beta UWY density")

    mf_int = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf_int.grids.level = 9
    mf_int.grids.prune = None
    results = {}
    for model in DFT_MODELS:
        try:
            xctype = model_xctype(model)
            W = compute_W_for_density(
                mf_int, dm_dft, make_exc_func(model, "W"), xctype
            )
            Wp = compute_W_for_density(
                mf_int, dm_dft, make_exc_func(model, "Wp"), xctype
            )
        except Exception as exc:
            if verbose:
                print(f"  [DFT/{model}] FAILED: {exc}", flush=True)
            W = Wp = float("nan")
        results[f"W_inf_{model}"] = float(W)
        results[f"Wp_inf_{model}"] = float(Wp)
    return results

# -----------------------------------------------------------------------------
# DENSITY FACTORY WITH CACHING
# -----------------------------------------------------------------------------
_DENSITY_CACHE = {}

def get_density(atom, charge, spin, basis, density_type, datadir, args, force_recompute=False):
    """Build a two-electron AO density directly with PySCF."""
    cache_key = f"{atom}_{charge}_{spin}_{basis}_{density_type}"

    if cache_key in _DENSITY_CACHE and not force_recompute:
        print(f"  [cache] Reusing cached {density_type} density for {atom}", flush=True)
        return _DENSITY_CACHE[cache_key]

    print(f"  [cache] Creating {density_type} density for {atom}", flush=True)

    basis_value, has_ecp, ecp_value = load_basis_for_pyscf(basis, atom)

    # Build molecule
    mol = gto.M(atom=atom, charge=charge, spin=spin, basis=basis_value)
    if has_ecp:
        mol.ecp = ecp_value
    mol.build()

    if mol.nelectron != 2 or mol.spin != 0:
        raise ValueError(
            f"The analytical SCE implementation requires a closed-shell "
            f"two-electron system; got Nel={mol.nelectron}, spin={mol.spin}"
        )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 2000
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    if density_type in ("hartree-fock", "hf"):
        dm = mf.make_rdm1()
    elif density_type in ("ccsd", "ccsd(t)", "ccsdt"):
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.kernel()
        dm = mycc.make_rdm1(ao_repr=True)
    elif density_type == "fci":
        cisolver = fci.FCI(mf)
        _, ci_vec = cisolver.kernel()
        dm_mo = cisolver.make_rdm1(
            ci_vec, mf.mo_coeff.shape[1], mol.nelec
        )
        dm = mf.mo_coeff @ dm_mo @ mf.mo_coeff.T
    else:
        raise ValueError(f"Unsupported density type: {density_type!r}")

    dm = np.asarray(dm, dtype=np.float64)
    density = SimpleNamespace(dm=dm, Nel=2, U=_hartree_U(mol, dm))

    # Cache the density object
    _DENSITY_CACHE[cache_key] = (density, mol)

    return density, mol


def compute_sce_n2(mol, dm, n_grid=8001, r_max=30.0):
    """Analytical spherical two-electron SCE values from an AO density."""
    # REV1 fix 2 (2026-10-14, R2-1): density-adapted radial grid. The fixed grid (r_max = 30 bohr,
    # 4001 points, dr = 0.0075 bohr) under-resolves compact densities: on an exact Gaussian it gives
    # +4 % (Vee) at omega = 100 and +10 % at omega = 1000. Cut the grid where 4 pi r^2 rho has dropped
    # below 1e-14 of its maximum and use >= 40001 points; accuracy <= 0.03 % for omega = 0.03-10^4 and
    # hydrogenic Z = 1-18.
    _rp = np.logspace(-6, np.log10(200.0), 4000)
    _aop = np.asarray(dft.numint.eval_ao(mol, np.column_stack((np.zeros_like(_rp), np.zeros_like(_rp), _rp))))
    if _aop.ndim == 3:
        _aop = _aop[0]
    _fp = 4.0 * np.pi * _rp**2 * np.maximum(np.einsum("gi,ij,gj->g", _aop, dm, _aop), 0.0)
    _keep = np.where(_fp > 1e-14 * _fp.max())[0]
    r_max = float(_rp[_keep[-1]]) * 1.2
    n_grid = max(int(n_grid), 40001)
    n_grid = n_grid if n_grid % 2 else n_grid + 1
    r_grid = np.linspace(1e-6, r_max, n_grid)
    dr = r_grid[1] - r_grid[0]
    ao = np.asarray(dft.numint.eval_ao(
        mol, np.column_stack((np.zeros(n_grid), np.zeros(n_grid), r_grid))
    ))
    if ao.ndim == 3:
        ao = ao[0]
    rho = np.maximum(np.einsum("gi,ij,gj->g", ao, dm, ao), 0.0)
    # REV1 fix (2026-10-14, R2-2): normalise the cumulative electron number to exactly 2 before
    # inverting it for the co-motion function. Without this, N_e(r_max) < 2 along the sampled axis
    # clamps f(r) at r_max for the innermost points (rho_f ~ 1e-25) and inflates W'_inf
    # (Hookium omega = 0.06: 0.5593 -> 0.0343). W_inf changes by < 1e-6 E_h.
    ne_raw = np.cumsum(4.0 * np.pi * r_grid**2 * rho) * dr
    ne = np.clip(ne_raw * (2.0 / ne_raw[-1]), 0.0, 2.0)
    unique = np.concatenate(([True], np.diff(ne) > 1e-12))
    if np.count_nonzero(unique) < 3:
        raise RuntimeError("Insufficient radial density support for SCE integration")
    ne_inv = interp1d(
        np.concatenate(([0.0], ne[unique])),
        np.concatenate(([r_grid[0]], r_grid[unique])),
        bounds_error=False, fill_value=(r_grid[0], r_grid[-1]),
    )
    rho_at = interp1d(r_grid, rho, bounds_error=False, fill_value=0.0)
    win_sum = wp_sum = 0.0
    used = 0
    for i in range(int(np.searchsorted(ne, 1.0))):
        r, rho_r = float(r_grid[i]), float(rho[i])
        if r < 1e-12 or rho_r < 1e-20:
            continue
        f_r = float(ne_inv(2.0 - float(ne[i])))
        distance = r + f_r
        if f_r < 1e-12 or distance < 1e-12:
            continue
        weight = 4.0 * np.pi * r**2 * rho_r * dr
        win_sum += weight / distance
        rho_f = float(rho_at(f_r))
        lb = (r*r + f_r*f_r) / (r * f_r * distance**3)
        if rho_f < 1e-30 or lb <= 0.0:
            continue
        p = (r / f_r)**2 * rho_r / rho_f
        if p < 1e-12:
            continue
        ls = 2.0 * (p*p + 1.0) / (p * distance**3)
        wp_sum += weight * (2.0 * np.sqrt(lb) + np.sqrt(max(ls, 0.0)))
        used += 1
    if used == 0:
        raise RuntimeError("No valid radial points in analytical SCE integration")
    # The r integration covers Ne(r) in [0, 1].  For N=2 this half-domain
    # already counts the single electron pair exactly once, so Vee_SCE has no
    # additional 1/2 prefactor.  The zero-point expression retains its 1/4.
    return win_sum, 0.25 * wp_sum

# -----------------------------------------------------------------------------
# ROBUST W_inf ESTIMATOR
# -----------------------------------------------------------------------------
_EST_ORDER = ["akima", "pchip", "cubic", "bpoly_1", "bpoly_0",
              "simpson", "romb", "trapezoid", "bpoly_2"]

def best_winf_fixed(U, integrals, extrapolated):
    """Return (W_inf, estimator_label) using the most reliable available value."""
    def _pick(k, ext):
        if not (ext and isinstance(ext, (list, tuple)) and len(ext) >= 2):
            return None
        try:
            v0, v1 = float(ext[0]), float(ext[1])
        except (TypeError, ValueError):
            return None
        if k in ("akima", "pchip"):
            if math.isfinite(v0) and v0 >= 0.0:
                return v0
            return None
        denom = abs(v0) + 1e-10
        if math.isfinite(v1) and v1 >= 0.0 and abs(v1 - v0) / denom < 2.0:
            return v1
        if math.isfinite(v0) and v0 >= 0.0:
            return v0
        return None

    cands = {
        k: _pick(k, extrapolated.get(k))
        for k in _EST_ORDER
        if _pick(k, extrapolated.get(k)) is not None
    }

    if not cands:
        for k in _EST_ORDER:
            ext = integrals.get(k)
            if ext and isinstance(ext, (list, tuple)) and len(ext) >= 2:
                try:
                    v = float(ext[1])
                    if math.isfinite(v) and v >= 0.0:
                        return v - U, f"{k}[raw]"
                except Exception:
                    pass
        return float("nan"), "none"

    if len(cands) == 1:
        k = next(iter(cands))
        return cands[k] - U, k

    vals = list(cands.values())
    med = sorted(vals)[len(vals) // 2]
    thr = max(3.0 * np.std(vals), 0.05 * abs(med) + 1e-6)
    stable = {k: v for k, v in cands.items() if abs(v - med) <= thr} or cands
    bk = next((k for k in _EST_ORDER if k in stable), next(iter(stable)))
    return stable[bk] - U, bk

# -----------------------------------------------------------------------------
# T_ZP / W'inf COMPUTATION
# -----------------------------------------------------------------------------
_NEL_ZERO_MODES: dict = {
    3: 0, 5: 7, 6: 9, 7: 11, 8: 13, 9: 15, 10: 17, 11: 19, 12: 21,
    13: 23, 14: 25, 15: 27, 16: 29, 17: 31, 18: 33, 20: 37, 30: 57, 36: 69, 54: 105,
}
_NEL_ZERO_MODES_DEFAULT = 8

def _n_skip_for_nel(nel: int) -> int:
    return _NEL_ZERO_MODES.get(nel, _NEL_ZERO_MODES_DEFAULT)

class _TruncatedResult:
    _TRUNC = frozenset(("angles", "f", "df", "d2f", "r", "Ne",
                        "rho", "drho", "d2rho", "Vee", "dVee", "d2Vee"))

    def __init__(self, res, n: int):
        object.__setattr__(self, "_res", res)
        object.__setattr__(self, "_n", n)

    def __getattr__(self, name):
        val = getattr(self._res, name)
        if name in self._TRUNC:
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] > self._n:
                return arr[:self._n]
        return val

    def __setattr__(self, name, value):
        setattr(self._res, name, value)

def _integrate_omega(res, omega_sums: np.ndarray, integrator: str = "simpson") -> float:
    from scipy.integrate import simpson as _simp, trapezoid as _trap
    N = omega_sums.shape[0]
    if res.opt.grid == "r":
        x = np.asarray(res.r[:N], dtype=float)
        y = (4.0 * np.pi * x**2 * np.asarray(res.rho[:N], dtype=float) * omega_sums)
    elif res.opt.grid == "Ne":
        x = np.asarray(res.Ne[:N], dtype=float)
        y = omega_sums
    else:
        raise ValueError(f"Unknown grid type '{res.opt.grid}'")
    integral = _trap(y, x=x) if integrator == "trapezoid" else _simp(y, x=x)
    return 0.5 * float(integral)

def compute_tzp_nel1() -> float:
    print("  [T_ZP] Nel=1 -> T_ZP = 0.0 (no e-e interaction)", flush=True)
    return 0.0

def compute_tzp_nel2(density, result=None, N_grid: int = 4001) -> float:
    from scipy.interpolate import interp1d
    r_grid = np.linspace(1e-6, 20.0, N_grid)
    dr = r_grid[1] - r_grid[0]
    rho_1d = np.maximum(
        np.array([float(density.rho(r)) for r in r_grid], dtype=float), 0.0
    )
    Ne = np.clip(np.cumsum(4.0 * np.pi * r_grid**2 * rho_1d) * dr, 0.0, 2.0)
    _, uid = np.unique(Ne, return_index=True)

    use_result = False
    f_interp = None
    if result is not None and hasattr(result, "f") and hasattr(result, "r"):
        try:
            r_sce = np.asarray(result.r).ravel()
            f_sce = np.asarray(result.f).ravel()
            if r_sce.shape == f_sce.shape and r_sce.size > 1:
                f_interp = interp1d(r_sce, f_sce, kind="linear",
                                    bounds_error=False,
                                    fill_value=(f_sce[0], f_sce[-1]))
                use_result = True
                print("  [T_ZP] Nel=2: using SCE co-motion f(r) from result", flush=True)
        except Exception as exc:
            print(f"  [T_ZP] Nel=2: result setup failed ({exc})", flush=True)

    Ne_inv = interp1d(
        np.concatenate([[0.0], Ne[uid]]),
        np.concatenate([[r_grid[0]], r_grid[uid]]),
        kind="linear", bounds_error=False,
        fill_value=(r_grid[0], 20.0),
    )

    i_half = int(np.searchsorted(Ne, 1.0))
    Wp_sum = 0.0
    for i in range(i_half):
        r, rho_r = float(r_grid[i]), float(rho_1d[i])
        if rho_r < 1e-20 or r < 1e-12:
            continue
        f_r = (float(f_interp(r)) if use_result else float(Ne_inv(2.0 - float(Ne[i]))))
        d = r + f_r
        if d < 1e-12 or f_r < 1e-12:
            continue
        lb = (r*r + f_r*f_r) / (r * f_r * d**3)
        if lb <= 0.0:
            continue
        rho_f = float(np.interp(f_r, r_grid, rho_1d))
        if rho_f < 1e-30:
            continue
        p = (r / f_r)**2 * rho_r / rho_f
        ls = 2.0 * (p*p + 1.0) / (p * d**3)
        Wp_sum += (4.0 * np.pi * r**2 * rho_r
                   * (2.0 * np.sqrt(lb) + np.sqrt(max(ls, 0.0))) * dr)

    val = 0.25 * Wp_sum
    print(f"  [T_ZP] Nel=2 analytical W'inf = {val:.10f}", flush=True)
    return val

def compute_tzp_for_planar(result, density=None) -> float:
    res = result

    Nel = res.f.shape[1]
    coordinates = res.opt.coordinates
    N_grid_end = res.angles.shape[0]
    n_angles = res.angles.shape[1]
    has_radials = (hasattr(coordinates, "Vee_hessian_radials") and
                   hasattr(coordinates, "Vee_hessian_angles_radials"))
    n_radials = Nel - 1 if has_radials else 0
    n_dof = n_angles + n_radials
    n_zero = 3
    n_vib = max(n_dof - n_zero, 1)

    omega_sums = np.zeros(N_grid_end)
    for idx in range(N_grid_end):
        try:
            if hasattr(coordinates, "Vee_hessian_angles"):
                h_aa = np.asarray(coordinates.Vee_hessian_angles(res.angles[idx], res.f[idx]))
                if has_radials:
                    h_rr = np.asarray(coordinates.Vee_hessian_radials(res.angles[idx], res.f[idx]))
                    h_ar = np.asarray(coordinates.Vee_hessian_angles_radials(
                        res.angles[idx], res.f[idx]))
                    H = np.block([[h_aa, h_ar], [h_ar.T, h_rr]])
                else:
                    H = h_aa
            else:
                raise RuntimeError("Analytical two-electron path has no JAX Hessian")

            H = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H)
            vib_pos = eigvals[-n_vib:] if len(eigvals) >= n_vib else eigvals
            vib_pos = vib_pos[vib_pos > 1e-8]
            omega_sums[idx] = float(np.sum(np.sqrt(vib_pos)))
        except Exception as exc:
            omega_sums[idx] = 0.0

    val = _integrate_omega(res, omega_sums)
    print(f"  [T_ZP] Planar W'inf = {val:.10f}", flush=True)
    return val

def compute_tzp_local(result, density=None, integrator: str = "simpson") -> float:
    res = result

    if not hasattr(res, "opt") or not hasattr(res.opt, "coordinates"):
        raise ValueError("Result object does not expose a coordinate system.")

    coordinates = res.opt.coordinates
    N_grid_end = res.angles.shape[0]
    Nel = res.f.shape[1]
    n_skip = _n_skip_for_nel(Nel)

    omega_sums = np.zeros(N_grid_end)
    for idx in range(N_grid_end):
        try:
            raise RuntimeError("Analytical two-electron path has no JAX Hessian")
        except Exception as exc:
            omega_sums[idx] = 0.0

    val = _integrate_omega(res, omega_sums, integrator)
    print(f"  [T_ZP] Local Hess W'inf = {val:.10f}", flush=True)
    return val

def compute_tzp_from_result(result, density, nel: int) -> float:
    if nel == 1:
        return compute_tzp_nel1()

    if nel == 2:
        print("  [T_ZP] Nel=2 -> exact analytical formula", flush=True)
        return compute_tzp_nel2(density, result=result)

    res = result

    if nel == 3:
        print(f"  [T_ZP] Nel=3 -> local Hessian (n_skip={_n_skip_for_nel(3)})", flush=True)
        val = compute_tzp_local(res, density)
        return val if (math.isfinite(val) and val > 0) else float("nan")
    elif nel == 4:
        n_pts = res.f.shape[0]
        if n_pts > 1:
            res = _TruncatedResult(res, n_pts - 1)
        return compute_tzp_for_planar(res, density)
    else:
        n_skip = _n_skip_for_nel(nel)
        print(f"  [T_ZP] Nel={nel} -> local Hessian (n_skip={n_skip})", flush=True)
        val = compute_tzp_local(res, density)
        if math.isfinite(val) and val > 0:
            return val
        return float("nan")

# -----------------------------------------------------------------------------
# MAIN RUN FUNCTION
# -----------------------------------------------------------------------------
def run_system(atom, charge, spin, basis, density_type, args):
    """Run optimization + T_ZP + W_inf integration for one system."""

    start_time = timer()

    atom_key = atom_label(atom, charge)
    ref = MOLE_REFERENCES.get(atom_key, {})
    W_inf_ref = ref.get("W_ref")
    T_ZP_ref = ref.get("Wp_ref")

    if spin is None:
        spin = ref.get("spin", 0)

    datadir = f"data/{atom}/{basis}/{_density_folder(density_type)}"
    os.makedirs(datadir, exist_ok=True)

    # Get density (with caching)
    density, mol = get_density(atom, charge, spin, basis, density_type, datadir, args)

    # Preserve one untouched target for both independent inversion methods.
    inversion_target = _target_dm_for_wy(density, mol)
    e_method, e_hf = _total_energies(mol, density_type, datadir)

    # -------------------------------------------------------------------------
    # INDEPENDENT WU-YANG ATTEMPT
    # -------------------------------------------------------------------------
    wy_mw = None
    wy_dm = wy_dm_spin = None
    wy_rho_diff = float("nan")
    wy_error = ""
    print(f"\n  [WY] Inverting {density_type.upper()} target density for {atom}...",
          flush=True)
    try:
        wy_mw, wy_rho_diff = run_wy_inversion(
            SimpleNamespace(dm=np.asarray(density.dm).copy()), mol,
            verbose=args.verbose,
        )
        raw = np.asarray(wy_mw.dm, dtype=np.float64)
        wy_dm_spin = (raw[0], raw[1]) if raw.ndim == 3 else None
        wy_dm = raw[0] + raw[1] if raw.ndim == 3 else raw
        if (not np.isfinite(wy_rho_diff)
                or wy_rho_diff > INVERSION_RHO_L2_TOL):
            raise RuntimeError(
                f"density error {wy_rho_diff:.3e} exceeds "
                f"{INVERSION_RHO_L2_TOL:.3e}"
            )
    except Exception as exc:
        wy_error = f"{type(exc).__name__}: {exc}"
        wy_mw = None
        wy_dm = wy_dm_spin = None
        print(f"  [WY] FAILED: {wy_error}", flush=True)

    wy_ok = wy_mw is not None

    # -------------------------------------------------------------------------
    # INDEPENDENT PYOEP ATTEMPT — SAME SYSTEM BASIS
    # -------------------------------------------------------------------------
    print(f"  [PyOEP] Running {'OSKSINV' if mol.spin else 'KSINV'} "
          f"independently with the system/OEP basis {basis}...", flush=True)
    pyoep_mw, pyoep_dm, pyoep_dm_spin, pyoep_results = run_pyoep_comparison(
        mol, inversion_target, e_method, basis, verbose=args.verbose
    )
    if (pyoep_results["pyoep_converged"]
            and (not np.isfinite(pyoep_results["pyoep_rho_diff"])
                 or pyoep_results["pyoep_rho_diff"] > INVERSION_RHO_L2_TOL)):
        pyoep_results["pyoep_converged"] = False
        pyoep_results["pyoep_error"] = (
            f"density error {pyoep_results['pyoep_rho_diff']:.3e} exceeds "
            f"{INVERSION_RHO_L2_TOL:.3e}"
        )
    pyoep_ok = bool(pyoep_results["pyoep_converged"] and pyoep_dm is not None)
    pyoep_results["wy_pyoep_rho_diff"] = (
        _rho_l2_diff(mol, wy_dm, pyoep_dm)
        if wy_ok and pyoep_dm is not None else float("nan")
    )

    # Wu-Yang is preferred.  PyOEP becomes the production inversion only when
    # Wu-Yang fails, so a valid density can still reach SCE/DFT/EXX/GL2.
    if wy_ok:
        production_source = "WY"
        production_dm, production_dm_spin, production_mw = (
            wy_dm, wy_dm_spin, wy_mw
        )
    elif pyoep_ok:
        production_source = "PYOEP"
        production_dm, production_dm_spin, production_mw = (
            pyoep_dm, pyoep_dm_spin, pyoep_mw
        )
        print("  [fallback] Wu-Yang failed; using converged PyOEP density.",
              flush=True)
    else:
        raise RuntimeError(
            "Both density inversions failed. "
            f"Wu-Yang: {wy_error or 'unavailable'}; "
            f"PyOEP: {pyoep_results['pyoep_error'] or 'unavailable'}"
        )

    density.dm = np.asarray(production_dm, dtype=np.float64)
    density.U = _hartree_U(mol, density.dm)

    # Store method-specific DFT values for direct comparison, plus unprefixed
    # production values used by all downstream calculations.
    dft_results = compute_dft_models(
        mol, production_dm, production_dm_spin, verbose=args.verbose
    )
    wy_dft_results = {
        f"WY_{key}": value for key, value in (
            compute_dft_models(mol, wy_dm, wy_dm_spin, verbose=args.verbose)
            if wy_ok else {
                **{f"W_inf_{m}": float("nan") for m in DFT_MODELS},
                **{f"Wp_inf_{m}": float("nan") for m in DFT_MODELS},
            }
        ).items()
    }
    pyoep_dft_results = {
        f"PYOEP_{key}": value for key, value in (
            compute_dft_models(
                mol, pyoep_dm, pyoep_dm_spin, verbose=args.verbose
            ) if pyoep_ok else {
                **{f"W_inf_{m}": float("nan") for m in DFT_MODELS},
                **{f"Wp_inf_{m}": float("nan") for m in DFT_MODELS},
            }
        ).items()
    }
    extra_results = compute_extra_quantities(
        mol, production_dm, density_type, datadir, production_mw,
        wy_ok, wy_rho_diff, production_source, verbose=args.verbose,
    )

    # Method-specific EXX/GL2 comparison fields.
    pyoep_results.update({
        "pyoep_W0": (-_hartree_U(mol, pyoep_dm) / 2.0
                       if pyoep_ok else float("nan")),
        "pyoep_W0_prime": float("nan"),
        "pyoep_gl2_2x": float("nan"),
    })
    if pyoep_ok:
        try:
            pyoep_gl2 = compute_gl2(mol, pyoep_mw)
            pyoep_results["pyoep_W0_prime"] = 2.0 * pyoep_gl2
            pyoep_results["pyoep_gl2_2x"] = 2.0 * pyoep_gl2
        except Exception as exc:
            if args.verbose:
                print(f"  [PyOEP/GL2] failed: {exc}", flush=True)

    # Compute complete SCE results independently for every converged inversion.
    def _independent_sce(prefix, dm_value, ok):
        values = {
            f"{prefix}_U": float("nan"),
            f"{prefix}_Vee_SCE": float("nan"),
            f"{prefix}_Winf": float("nan"),
            f"{prefix}_T_ZP_prl": float("nan"),
        }
        if not ok:
            return values
        print(f"  [{prefix}] Computing independent analytical SCE values...",
              flush=True)
        u_value = _hartree_U(mol, dm_value)
        vee_value, tzp_value = compute_sce_n2(
            mol, dm_value, n_grid=args.N_grid
        )
        values.update({
            f"{prefix}_U": float(u_value),
            f"{prefix}_Vee_SCE": float(vee_value),
            f"{prefix}_Winf": float(vee_value - u_value),
            f"{prefix}_T_ZP_prl": float(tzp_value),
        })
        return values

    wy_sce_results = _independent_sce("WY", wy_dm, wy_ok)
    pyoep_sce_results = _independent_sce("PYOEP", pyoep_dm, pyoep_ok)
    selected_sce = (wy_sce_results if production_source == "WY"
                    else pyoep_sce_results)
    vee_sce = selected_sce[f"{production_source}_Vee_SCE"]
    Winf = selected_sce[f"{production_source}_Winf"]
    T_ZP_prl = selected_sce[f"{production_source}_T_ZP_prl"]
    T_ZP_legacy = T_ZP_prl / 2.0
    ac_results = compute_all_ac(
        extra_results["W0"],
        0.5 * extra_results["W0_prime"],  # raw GL2
        Winf,
        T_ZP_prl,
        e_hf,
    )
    # Independent PyOEP AC interpolation: do not reuse any Wu-Yang W0, GL2,
    # Winf, or Wpinf value.  Non-converged/missing PyOEP inputs produce NaN.
    pyoep_ac_results = {
        f"PYOEP_{key}": value for key, value in compute_all_ac(
            pyoep_results["pyoep_W0"],
            0.5 * pyoep_results["pyoep_W0_prime"],  # raw PyOEP GL2
            pyoep_sce_results["PYOEP_Winf"],
            pyoep_sce_results["PYOEP_T_ZP_prl"],
            e_hf,
        ).items()
    }
    ints_time = timer()
    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    error_W = ((Winf - W_inf_ref) if (W_inf_ref is not None and math.isfinite(Winf)) else None)
    error_TZP = ((T_ZP_prl - T_ZP_ref)
                 if (T_ZP_ref is not None and math.isfinite(T_ZP_prl)) else None)

    result = {
        "density_type": density_type.upper(),
        "system": atom_key,
        "omega": float("nan"),
        "basis": basis,
        "spin": int(spin),
        "charge": int(charge),
        "E_ref": float(e_method),
        "E_HF": float(e_hf),
        "U": float(density.U),
        "Vee_SCE": float(vee_sce),
        "Winf": float(Winf),
        "T_ZP_prl": float(T_ZP_prl),
        "T_ZP_legacy": float(T_ZP_legacy),
        "estimator": "analytical-N2",
        "density_used": f"{production_source}[{density_type.upper()}]",
        "inversion_source": production_source,
        "wy_inverted": bool(wy_ok),
        "wy_converged": bool(wy_ok),
        "wy_error": wy_error,
        **extra_results,
        **ac_results,
        **dft_results,
        **wy_dft_results,
        **pyoep_results,
        **pyoep_dft_results,
        **wy_sce_results,
        **pyoep_sce_results,
        **pyoep_ac_results,
    }

    def _fs(v, fmt=".10f"):
        return f"{v:{fmt}}" if (v is not None and math.isfinite(v)) else "N/A"

    W = 80
    print(f"\n{'='*W}")
    print(f"RESULTS for {atom_key}  (Nel={density.Nel})")
    print(f"{'='*W}")
    print(f"  Basis            : {basis}")
    print(f"  Density          : {production_source}[{density_type.upper()}]")
    print(f"  Grid             : analytical radial")
    print(f"  Estimator        : analytical-N2")
    print(f"  U                : {density.U:.10f}")
    print(f"  W_inf            : {_fs(Winf)}")
    print(f"  W'inf PRL (T_ZP) : {_fs(T_ZP_prl)}")
    print(f"  W'inf legacy     : {_fs(T_ZP_legacy)}")
    print(f"  W0 ({production_source} EXX) : {_fs(extra_results['W0'])}")
    print(f"  E_c              : {_fs(extra_results['E_c'])}")
    print(f"  E_xc             : {_fs(extra_results['E_xc'])}")
    print(f"  W0_prime         : {_fs(extra_results['W0_prime'])}")
    print(f"  gl2_2x           : {_fs(extra_results['gl2_2x'])}")
    print(f"  SPL correlation  : {_fs(ac_results['spl_corr'])}")
    print(f"  genISI2 corr.    : {_fs(ac_results['gisi2_corr'])}")
    print(f"  E_SPL            : {_fs(ac_results['E_spl'])}")
    print(f"  E_genISI2        : {_fs(ac_results['E_gisi2'])}")
    print(f"  WY rho L2 error  : {_fs(extra_results['wy_rho_diff'], '.6e')}")
    print(f"  WY converged     : {wy_ok}")
    if wy_error:
        print(f"  WY error         : {wy_error}")
    print(f"  PyOEP method     : {pyoep_results['pyoep_method']}")
    print(f"  PyOEP orbital bs : {pyoep_results['pyoep_orbital_basis']}")
    print(f"  PyOEP OEP basis  : {pyoep_results['pyoep_oep_basis']}")
    print(f"  PyOEP converged  : {pyoep_results['pyoep_converged']}")
    print(f"  PyOEP space sym  : {pyoep_results['pyoep_space_sym']}")
    print(f"  PyOEP rho L2     : {_fs(pyoep_results['pyoep_rho_diff'], '.6e')}")
    print(f"  WY-PyOEP rho L2  : {_fs(pyoep_results['wy_pyoep_rho_diff'], '.6e')}")
    print(f"  PyOEP W0         : {_fs(pyoep_results['pyoep_W0'])}")
    print(f"  PyOEP W0_prime   : {_fs(pyoep_results['pyoep_W0_prime'])}")
    print(f"  PyOEP SPL corr.  : {_fs(pyoep_ac_results['PYOEP_spl_corr'])}")
    print(f"  PyOEP genISI2 corr: {_fs(pyoep_ac_results['PYOEP_gisi2_corr'])}")
    print(f"  PyOEP E_SPL      : {_fs(pyoep_ac_results['PYOEP_E_spl'])}")
    print(f"  PyOEP E_genISI2  : {_fs(pyoep_ac_results['PYOEP_E_gisi2'])}")
    print(f"  WY W_inf         : {_fs(wy_sce_results['WY_Winf'])}")
    print(f"  WY W'inf         : {_fs(wy_sce_results['WY_T_ZP_prl'])}")
    print(f"  PyOEP W_inf      : {_fs(pyoep_sce_results['PYOEP_Winf'])}")
    print(f"  PyOEP W'inf      : {_fs(pyoep_sce_results['PYOEP_T_ZP_prl'])}")
    if pyoep_results.get('pyoep_error'):
        print(f"  PyOEP error      : {pyoep_results['pyoep_error']}")
    if W_inf_ref is not None:
        print(f"  Reference W_inf  : {W_inf_ref:.6f}")
        print(f"  Error W_inf      : {_fs(error_W, '.6f')}")
    if T_ZP_ref is not None:
        print(f"  Reference W'inf  : {T_ZP_ref:.6f}")
        print(f"  Error W'inf(PRL) : {_fs(error_TZP, '.6f')}")
    print(f"  Time             : {ints_time - start_time:.1f}s")
    print(f"{'='*W}\n")

    return result


# -----------------------------------------------------------------------------
# SECTIONED CSV OUTPUT
# -----------------------------------------------------------------------------
def save_results_csv(results, output_filename=None):
    """Write calculation rows in the established sectioned CSV layout."""
    rows = list(results.values()) if isinstance(results, dict) else list(results)
    if not rows:
        return None
    if output_filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"density_loop_results_{stamp}.csv"

    id_cols = [
        "density_type", "density_used", "system", "omega", "basis",
        "spin", "charge", "E_ref", "E_HF",
    ]
    dft_cols = [
        key for model in DFT_MODELS
        for key in (f"W_inf_{model}", f"Wp_inf_{model}")
    ]
    sce_cols = ["W_inf_SCE", "Wp_inf_SCE", "U"]
    wy_cols = [
        "W0", "E_c", "E_xc", "W0_prime", "gl2_2x", "WY_ok",
        "wy_rho_diff", "wy_error", "inversion_source", "inversion_ok",
    ]
    ac_cols = [
        "eisi_corr", "spl_corr", "gisi2_corr", "rvisl_corr", "spl2_corr",
        "E_isi", "E_spl", "E_gisi2", "E_rvisl", "E_spl2", "E_gl2_tot",
    ]
    pyoep_ac_cols = [f"PYOEP_{key}" for key in ac_cols]
    pyoep_meta_cols = [
        "pyoep_available", "pyoep_exxoep_available", "pyoep_method",
        "pyoep_oep_basis", "pyoep_same_system_basis", "pyoep_space_sym",
        "pyoep_maxiter", "pyoep_conv_thr", "pyoep_thr_fai_oep",
        "pyoep_converged", "pyoep_rho_diff", "wy_pyoep_rho_diff",
        "pyoep_W0", "pyoep_W0_prime", "pyoep_gl2_2x", "pyoep_error",
    ]
    wy_dft_cols = [
        key for model in DFT_MODELS
        for key in (f"WY_W_inf_{model}", f"WY_Wp_inf_{model}")
    ]
    pyoep_dft_cols = [
        key for model in DFT_MODELS
        for key in (f"PYOEP_W_inf_{model}", f"PYOEP_Wp_inf_{model}")
    ]
    comparison_sce_cols = [
        "WY_U", "WY_Vee_SCE", "WY_Winf", "WY_T_ZP_prl",
        "PYOEP_U", "PYOEP_Vee_SCE", "PYOEP_Winf", "PYOEP_T_ZP_prl",
    ]
    columns = (
        id_cols + dft_cols + sce_cols + wy_cols + ac_cols +
        wy_dft_cols + pyoep_meta_cols + pyoep_dft_cols +
        comparison_sce_cols + pyoep_ac_cols
    )

    aliases = {
        "W_inf_SCE": "Winf",
        "Wp_inf_SCE": "T_ZP_prl",
    }

    def value_for(row, column):
        return row.get(column, row.get(aliases.get(column, ""), ""))

    def format_value(value):
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{float(value):.10f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return "" if value is None else str(value)

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get("density_type", "UNKNOWN")).upper(), []).append(row)

    with open(output_filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        first = True
        for density_name, density_rows in grouped.items():
            if not first:
                handle.write("\n" * 5)
            handle.write(f"# ===== {density_name} DENSITY =====\n")
            writer.writerow(columns)
            for row in density_rows:
                writer.writerow([
                    format_value(value_for(row, column)) for column in columns
                ])
            first = False

    print(f"\n  CSV results saved: {output_filename}", flush=True)
    return output_filename


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SCE optimization for atoms with Wu-Yang density inversion"
    )
    parser.add_argument("--atom", type=str, help="Atom symbol for single system run")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=None,
                        help="Spin (auto-detected from reference table if omitted)")
    parser.add_argument("--basis", type=str, default="cc-pVDZ")
    parser.add_argument(
        "--density", type=str, default="all",
        choices=["all", "hartree-fock", "hf", "ccsd", "ccsd(t)", "ccsdt", "fci"],
        help="Density layer; 'all' runs HF first and FCI second",
    )
    parser.add_argument(
        "--wy", action="store_true", default=True,
        help="Deprecated compatibility flag; Wu-Yang is always enabled"
    )
    parser.add_argument("--N-grid", type=int, default=4001,
                        help="Odd radial-grid size for analytical N=2 SCE")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Verbose output")
    parser.add_argument("--clear-cache", action="store_true", default=False,
                        help="Clear density cache before running")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV filename")

    args = parser.parse_args()
    args.density = args.density.lower()
    if args.density == "hartree-fock":
        args.density = "hf"
    density_sequence = ["hf", "fci"] if args.density == "all" else [args.density]
    args.wy = True

    if args.clear_cache:
        _DENSITY_CACHE.clear()
        print("[cache] Density cache cleared", flush=True)

    if args.atom is not None:
        print(f"\n{'='*60}")
        print(f"  Running single system: {args.atom}")
        print(f"  Charge: {args.charge}, Basis: {args.basis}, "
              f"Density layers: {' -> '.join(d.upper() for d in density_sequence)}")
        print(f"{'='*60}")
        results = []
        single_failures = []
        for density_type in density_sequence:
            print(f"\n{'#'*60}\n"
                  f"  DENSITY LAYER: {density_type.upper()}\n"
                  f"{'#'*60}", flush=True)
            try:
                results.append(run_system(
                    atom=args.atom, charge=args.charge, spin=args.spin,
                    basis=args.basis, density_type=density_type, args=args,
                ))
            except Exception as exc:
                single_failures.append((density_type, exc))
                print(f"  x {density_type.upper()} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        if results:
            save_results_csv(results, args.output)
        if single_failures:
            print("\n  Failed density layers:")
            for density_type, exc in single_failures:
                print(f"    x {density_type.upper()}: "
                      f"{type(exc).__name__}: {exc}")
        return

    if not SYSTEM_BASIS:
        print("Error: SYSTEM_BASIS is empty. Use --atom for single system.")
        sys.exit(1)

    batch_start = timer()
    all_results = {}
    failures = []

    print(f"\n{'='*60}")
    print(f"  Batch density layers: "
          f"{' -> '.join(d.upper() for d in density_sequence)}")
    print(f"  analytical radial N_grid={args.N_grid}")
    print(f"  Systems: {len(SYSTEM_BASIS)}")
    print(f"{'='*60}")

    # Density is the outer loop: complete every HF calculation first, then FCI.
    for density_type in density_sequence:
        print(f"\n{'#'*60}\n"
              f"  DENSITY LAYER: {density_type.upper()}\n"
              f"{'#'*60}", flush=True)
        for atom, charge, spin, basis in SYSTEM_BASIS:
            system_tag = f"{atom_label(atom, charge)}/{basis}"
            result_tag = f"{density_type.upper()}:{system_tag}"
            print(f"\n{'─'*60}")
            print(f"  > Running {system_tag}  density={density_type.upper()} "
                  f"(charge={charge}, spin={spin})")
            print(f"{'─'*60}")
            try:
                res = run_system(
                    atom=atom, charge=charge, spin=spin, basis=basis,
                    density_type=density_type, args=args,
                )
                all_results[result_tag] = res
            except Exception as exc:
                print(f"\n  x  FAILED: {result_tag}")
                print(f"     {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                failures.append((result_tag, exc))

    total_time = timer() - batch_start

    save_results_csv(all_results, args.output)

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY   (total wall time: {total_time:.1f}s)")
    print(f"{'='*60}")

    if all_results:
        header = (f"  {'Density:System':<28} {'W_inf':>12} {'Err_W':>10} "
                  f"{'W_prime(PRL)':>14} {'Err_W_prime':>12} "
                  f"{'Source':>8} {'WY':>5} {'PyOEP':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for tag, res in all_results.items():
            atom_key = str(res.get("system", ""))
            ref_data = MOLE_REFERENCES.get(atom_key, {})
            W_inf_ref = ref_data.get("W_ref")
            T_ZP_ref = ref_data.get("Wp_ref")

            Winf = res.get("Winf", float("nan"))
            T_ZP = res.get("T_ZP_prl", float("nan"))
            wy_ok = "Y" if res.get("wy_inverted", False) and res.get("wy_converged", False) else "N"
            pyoep_ok = "Y" if res.get("pyoep_converged", False) else "N"
            source = str(res.get("inversion_source", "?"))

            err_W = ((Winf - W_inf_ref)
                     if (W_inf_ref is not None and math.isfinite(Winf))
                     else float("nan"))
            err_T = ((T_ZP - T_ZP_ref)
                     if (T_ZP_ref is not None and math.isfinite(T_ZP))
                     else float("nan"))

            def _f(v, w):
                return f"{v:>{w}.6f}" if math.isfinite(v) else f"{'N/A':>{w}}"

            print(f"  {tag:<28} {_f(Winf,12)} {_f(err_W,10)} "
                  f"{_f(T_ZP,14)} {_f(err_T,12)} "
                  f"{source:>8} {wy_ok:>5} {pyoep_ok:>7}")

    if failures:
        print(f"\n  Failed systems ({len(failures)}):")
        for tag, exc in failures:
            print(f"    x  {tag}: {type(exc).__name__}: {exc}")

    expected_runs = len(SYSTEM_BASIS) * len(density_sequence)
    print(f"\n  Done. {len(all_results)}/{expected_runs} density-system runs completed.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
