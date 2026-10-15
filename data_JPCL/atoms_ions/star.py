

"""Batch optimization over SYSTEM_BASIS with dynamic reference value lookup.

T_ZP / W'∞ dispatch (correct, not a placeholder):
  Nel == 2  →  exact analytical formula  (compute_tzp_nel2)
  Nel == 4  →  planar co-motion method   (compute_tzp_for_planar)
  Nel >= 3  →  local Hessian method      (compute_tzp_local)
"""

import os

import argparse
import json
import math
import re as _re
import sys
from functools import reduce
from timeit import default_timer as timer
from typing import Dict, Union

# ── JAX platform: GPU first, automatic CPU fallback ───────────────────────────
# Never hard-code JAX_PLATFORMS.  Try GPU variants in order; if every attempt
# fails (OOM, driver mismatch, no device) we silently fall back to CPU so the
# rest of the script runs unchanged.
def _init_jax_platform() -> None:
    import jax
    for _platform in ("gpu", "cuda", "rocm"):
        try:
            jax.config.update("jax_platform_name", _platform)
            import jax.numpy as _jnp
            _ = _jnp.array(1.0) + 1.0          # force real initialisation
            print(f"[jax] Using platform : {_platform.upper()}", flush=True)
            return
        except Exception as _e:
            print(f"[jax] {_platform.upper()} unavailable "
                  f"({type(_e).__name__}); trying next...", flush=True)
    jax.config.update("jax_platform_name", "cpu")
    print("[jax] Falling back to CPU", flush=True)

_init_jax_platform()

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
print(f"[jax] Active backend : {jax.default_backend().upper()}", flush=True)

import numpy as np
from pyscf import cc, scf
from pyscf.cc import (ccsd_t_lambda_slow, ccsd_t_rdm_slow,
                      ccsd_t_slow, uccsd_t)
from pyscf.grad import ccsd as ccsd_grad
from pyscf import lib as pyscf_lib

from jaxsce.constants import A
from jaxsce.coordinates_3d import get_coordinate_system
from jaxsce.densities.pyscf import CCSDDensity, CCSDTDensity, FCIDensity, HFDensity
from jaxsce.integrate import VeeIntegration
from jaxsce.optimize import AngularOptimization, TwoElectron

try:
    from pyscf.cc import uccsd_t_lambda
except ImportError:
    uccsd_t_lambda = None

try:
    from pyscf.cc import uccsd_t_rdm
except ImportError:
    uccsd_t_rdm = None


# ──────────────────────────────────────────────────────────────────────────────
# UHF DENSITY-MATRIX COLLAPSE PATCH
#
# For open-shell atoms PySCF returns dm with shape (2, nao, nao).
# jaxsce's vH_deriv expects (nao, nao) and crashes with a shape error in
# np.einsum("ijp,ij->p", ...).  We monkey-patch vH_deriv to collapse
# alpha+beta on first call.
# ──────────────────────────────────────────────────────────────────────────────

import jaxsce.densities.pyscf as _pyscf_mod


def _collapse_dm(dm):
    dm = np.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm


def _recompute_U(density):
    eris = density.mol.intor("int2e")
    density.U = 0.5 * float(
        np.einsum("pqrs,pq,rs->", eris, density.dm, density.dm)
    )


_orig_vH_deriv = (
    _pyscf_mod.HFDensity.vH_deriv.__func__
    if hasattr(_pyscf_mod.HFDensity.vH_deriv, "__func__")
    else _pyscf_mod.HFDensity.vH_deriv
)


def _patched_vH_deriv(self, r):
    if np.asarray(self.dm).ndim == 3:
        self.dm = _collapse_dm(self.dm)
        _recompute_U(self)
    return _orig_vH_deriv(self, r)


_pyscf_mod.HFDensity.vH_deriv = _patched_vH_deriv


# ──────────────────────────────────────────────────────────────────────────────
# PYSCF 2.x TREUTLER COMPATIBILITY PATCH
#
# Newer PySCF renamed / changed the signature of treutler_ahlrichs.
# This shim accepts both the old 1-arg and new 2-arg forms.
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# BASIS-SET-EXCHANGE LAYER
# ──────────────────────────────────────────────────────────────────────────────

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
    "jorge-aqzp": "jorge-aQZP",
    "aug-cc-pvdz-pp": "aug-cc-pVDZ-PP", "aug-cc-pvtz-pp": "aug-cc-pVTZ-PP",
    "aug-cc-pvqz-pp": "aug-cc-pVQZ-PP", "aug-cc-pv5z-pp": "aug-cc-pV5Z-PP",
    "aug-cc-pv6z-pp": "aug-cc-pV6Z-PP", "aug-cc-pcvdz": "aug-cc-pCVDZ",
    "aug-cc-pcvtz": "aug-cc-pCVTZ",     "aug-cc-pcv4z": "aug-cc-pCV4Z",
    "aug-cc-pcv5z": "aug-cc-pCV5Z",     "aug-cc-pcv6z": "aug-cc-pCV6Z",
}
_UNC_PREFIX = "unc-"
_F_EPS = 1e-12


def _normalise_basis_name(name):
    lo = name.lower()
    if lo in _BASIS_ALIASES:
        return _BASIS_ALIASES[lo]
    if lo.endswith("pz"):
        return name[:-2] + "ZP"
    return name


def _is_uncontracted(name):
    if name.lower().startswith(_UNC_PREFIX):
        return True, name[len(_UNC_PREFIX):]
    return False, name


def fetch_basis_nwchem(basis_name, atom_symbol):
    if not BSE_AVAILABLE:
        return None
    is_unc, base_name = _is_uncontracted(basis_name)
    base_name = _normalise_basis_name(base_name)
    cache_key = (basis_name.lower(), atom_symbol.upper())
    if cache_key in _BASIS_CACHE:
        return _BASIS_CACHE[cache_key]
    try:
        kw = {'elements': [atom_symbol], 'fmt': 'nwchem', 'header': False}
        if is_unc:
            kw.update({'uncontract_general': True, 'uncontract_spdf': True,
                       'uncontract_segmented': True})
        result = bse.get_basis(base_name, **kw)
        _BASIS_CACHE[cache_key] = result
        print(f"  [basis] fetched '{basis_name}' for {atom_symbol}", flush=True)
        return result
    except Exception as e:
        print(f"  [basis] WARNING: Could not fetch '{basis_name}' "
              f"for {atom_symbol}: {e}", flush=True)
        _BASIS_CACHE[cache_key] = None
        return None


def _split_basis_ecp(basis_string):
    basis_match = _re.search(r'(^BASIS\b.*?^END\b)', basis_string,
                              _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
    ecp_match   = _re.search(r'(^ECP\b.*?^END\b)',   basis_string,
                              _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
    return (
        basis_match.group(1) if basis_match else basis_string,
        ecp_match.group(1)   if ecp_match   else None,
    )


def load_basis_for_pyscf(basis_name, atom_symbol):
    from pyscf.gto.basis import parse_nwchem
    basis_string = fetch_basis_nwchem(basis_name, atom_symbol)
    if basis_string is None:
        print(f"  [basis] using PySCF internal '{basis_name}' "
              f"for {atom_symbol}", flush=True)
        return basis_name, False, None
    basis_block, ecp_block = _split_basis_ecp(basis_string)
    try:
        basis_dict = {atom_symbol: parse_nwchem.parse(basis_block)}
    except Exception as e:
        print(f"  [basis] WARNING: Failed to parse basis: {e}", flush=True)
        return basis_name, False, None
    ecp_dict = None
    has_ecp  = False
    if ecp_block is not None:
        from pyscf.gto.basis import parse_nwchem_ecp
        try:
            for parser_name in ['parse_ecp', 'parse']:
                parser = getattr(parse_nwchem_ecp, parser_name, None)
                if parser is not None:
                    try:
                        ecp_dict = {atom_symbol: parser(ecp_block)}
                        has_ecp  = True
                        print(f"  [basis] ECP loaded for {atom_symbol}", flush=True)
                        break
                    except Exception:
                        continue
        except Exception as e:
            print(f"  [basis] WARNING: Failed to parse ECP: {e}", flush=True)
    return basis_dict, has_ecp, ecp_dict


# ──────────────────────────────────────────────────────────────────────────────
# REFERENCE DATA
# ──────────────────────────────────────────────────────────────────────────────

MOLE_REFERENCES = {
    "H":   {"Nel": 1,  "spin": 1, "W_ref": -0.3125,      "Wp_ref": 0.0},
    "H-":  {"Nel": 2,  "spin": 0, "W_ref": -0.569,       "Wp_ref": None},
    "He":  {"Nel": 2,  "spin": 0, "W_ref": -1.5008,      "Wp_ref": 0.621},
    "He-": {"Nel": 3,  "spin": 1, "W_ref": None,         "Wp_ref": None},
    "Li":  {"Nel": 3,  "spin": 1, "W_ref": -2.603,       "Wp_ref": 1.38},
    "Li-": {"Nel": 4,  "spin": 0, "W_ref": -2.7308,      "Wp_ref": 1.52075},
    "Be":  {"Nel": 4,  "spin": 0, "W_ref": -4.021,       "Wp_ref": 2.59},
    "B":   {"Nel": 5,  "spin": 1, "W_ref": -5.706,       "Wp_ref": 4.2},
    "C":   {"Nel": 6,  "spin": 0, "W_ref": -7.782,       "Wp_ref": 6.3},
    "N":   {"Nel": 7,  "spin": 1, "W_ref": -10.246,      "Wp_ref": None},
    "O":   {"Nel": 8,  "spin": 0, "W_ref": -10.335,      "Wp_ref": 13.0},
    "F":   {"Nel": 9,  "spin": 1, "W_ref": None,         "Wp_ref": None},
    "F-":  {"Nel": 10, "spin": 0, "W_ref": -16.940,      "Wp_ref": None},
    "Ne":  {"Nel": 10, "spin": 0, "W_ref": -20.035,      "Wp_ref": 22.0},
    "Na":  {"Nel": 11, "spin": 1, "W_ref": None,         "Wp_ref": None},
    "Mg":  {"Nel": 12, "spin": 0, "W_ref": -26.7297,     "Wp_ref": None},
    "Cl-": {"Nel": 18, "spin": 0, "W_ref": -47.26,       "Wp_ref": None},
    "Ar":  {"Nel": 18, "spin": 0, "W_ref": -51.5550,     "Wp_ref": None},
    "Ca":  {"Nel": 20, "spin": 0, "W_ref": -60.485716,   "Wp_ref": None},
    "Zn":  {"Nel": 30, "spin": 0, "W_ref": None,         "Wp_ref": None},
    "Kr":  {"Nel": 36, "spin": 0, "W_ref": -166.8500,    "Wp_ref": None},
    "Xe":  {"Nel": 54, "spin": 0, "W_ref": -322.8348,    "Wp_ref": None},
}

# ──────────────────────────────────────────────────────────────────────────────
# PER-SYSTEM BASIS SETS (for batch mode)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_BASIS = [
    # Hydrogen
    ("H",   0, 1, "aug-cc-pv6z"),
    ("H",  -1, 0, "aug-cc-pv6z"),
    # Helium
    ("He",  0, 0, "cc-pVDZ"),
    ("He",  0, 0, "aug-cc-pVQZ"),
    ("He",  0, 0, "aug-cc-pV6Z"),
    ("He", -1, 1, "aug-cc-pv6z"),
    # Lithium
    ("Li",  0, 1, "aug-cc-pv5z"),
    ("Li", -1, 0, "aug-cc-pv5z"),
    # Beryllium
    ("Be",  0, 0, "aug-cc-pV5Z"),
    # Boron → Oxygen
    ("B",   0, 1, "aug-cc-pv5z"),
    ("C",   0, 2, "aug-cc-pcvqz"),
    ("N",   0, 3, "aug-cc-pcv5z"),
    ("O",   0, 2, "aug-cc-pcv5z"),
    # Fluorine
    ("F",   0, 1, "aug-cc-pcv5z"),
    ("F",  -1, 0, "aug-cc-pcvQz"),
    # Neon
    ("Ne",  0, 0, "aug-cc-pVQZ"),
    ("Ne",  0, 0, "aug-cc-pcv5z"),
    # Sodium → Argon
    ("Na",  0, 1, "aug-cc-pcv5z"),
    ("Mg",  0, 0, "aug-cc-pVQZ"),
    ("Mg",  0, 0, "aug-cc-pcv5z"),
    ("Ar",  0, 0, "aug-cc-pVQZ"),
    ("Ar",  0, 0, "aug-cc-pcv5z"),
    # Calcium
    ("Ca",  0, 0, "jorge-qzp"),
    ("Ca",  0, 0, "jorge-QZP"),
    # Zinc
    ("Zn",  0, 0, "aug-cc-pvqz"),
    # Krypton
    ("Kr",  0, 0, "cc-pVDZ"),
    ("Kr",  0, 0, "aug-cc-pVQZ"),
    ("Kr",  0, 0, "aug-cc-pvqz"),
    # Xenon
    ("Xe",  0, 0, "jorge-aqzp"),
    ("Xe",  0, 0, "aug-cc-pvqz"),
    ("Xe",  0, 0, "aug-cc-pVQZ-PP"),
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def atom_label(atom: str, charge: int) -> str:
    if charge == -1:
        return atom.strip() + "-"
    return atom.strip()


def fmt_val(val, fmt=".6f"):
    return f"{val:{fmt}}" if val is not None else "N/A"


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


# ──────────────────────────────────────────────────────────────────────────────
# ROBUST W_inf ESTIMATOR
#
# bpoly_2[1] can be NaN/unstable for small grids or difficult systems.
# best_winf_fixed tries all estimators in priority order and returns the
# first finite, non-negative, non-divergent value together with its label.
# ──────────────────────────────────────────────────────────────────────────────

_EST_ORDER = ["akima", "pchip", "cubic", "bpoly_1", "bpoly_0",
              "simpson", "romb", "trapezoid", "bpoly_2"]


def best_winf_fixed(U, integrals, extrapolated):
    """Return (W_inf, estimator_label) using the most reliable available value."""

    def _pick(ext):
        if not (ext and isinstance(ext, (list, tuple)) and len(ext) >= 2):
            return None
        try:
            v0, v1 = float(ext[0]), float(ext[1])
        except (TypeError, ValueError):
            return None
        denom = abs(v0) + 1e-10
        if math.isfinite(v1) and v1 >= 0.0 and abs(v1 - v0) / denom < 2.0:
            return v1
        if math.isfinite(v0) and v0 >= 0.0:
            return v0
        return None

    cands = {k: _pick(extrapolated.get(k)) for k in _EST_ORDER
             if _pick(extrapolated.get(k)) is not None}

    if not cands:
        # last resort: raw integral values
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

    vals   = list(cands.values())
    med    = sorted(vals)[len(vals) // 2]
    thr    = max(3.0 * np.std(vals), 0.05 * abs(med) + 1e-6)
    stable = {k: v for k, v in cands.items() if abs(v - med) <= thr} or cands
    bk     = next((k for k in _EST_ORDER if k in stable), next(iter(stable)))
    return stable[bk] - U, bk


# ──────────────────────────────────────────────────────────────────────────────
# T_ZP / W'∞  ——  three separate methods, one dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class _TruncatedResult:
    """Proxy exposing only the first *n* rows of array attributes."""
    _TRUNC = frozenset(("angles", "f", "df", "d2f", "r", "Ne",
                        "rho", "drho", "d2rho", "Vee", "dVee", "d2Vee"))

    def __init__(self, res, n: int):
        object.__setattr__(self, "_res", res)
        object.__setattr__(self, "_n",   n)

    def __getattr__(self, name):
        val = getattr(self._res, name)
        if name in self._TRUNC:
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] > self._n:
                return arr[:self._n]
        return val

    def __setattr__(self, name, value):
        setattr(self._res, name, value)


# ── Nel = 2 : exact analytic ──────────────────────────────────────────────────

def compute_tzp_nel2(density, result=None, N_grid: int = 4001) -> float:
    """Exact analytic T_ZP for Nel=2 (Gori-Giorgi, Vignale & Seidl 2009).

    The *result* from TwoElectron may have r and f on different grids
    (Ne-grid vs r-grid), so we validate shape compatibility before trying
    to use the co-motion function; on any mismatch we fall back silently
    to pure numerical inversion of Ne(r) from the density.
    """
    from scipy.interpolate import interp1d

    # build r-grid and density
    r_grid = np.linspace(1e-6, 20.0, N_grid)
    dr     = r_grid[1] - r_grid[0]
    rho_1d = np.maximum(
        np.array([float(density.rho(r)) for r in r_grid], dtype=float), 0.0
    )
    Ne = np.clip(np.cumsum(4.0 * np.pi * r_grid**2 * rho_1d) * dr, 0.0, 2.0)
    _, uid = np.unique(Ne, return_index=True)

    # attempt to use SCE co-motion function —— guarded against shape mismatch
    use_result = False
    f_interp   = None
    if result is not None and hasattr(result, 'f') and hasattr(result, 'r'):
        try:
            r_sce = np.asarray(result.r).ravel()
            f_sce = np.asarray(result.f).ravel()
            if r_sce.shape == f_sce.shape and r_sce.size > 1:
                f_interp   = interp1d(r_sce, f_sce, kind="linear",
                                      bounds_error=False,
                                      fill_value=(f_sce[0], f_sce[-1]))
                use_result = True
                print("  [T_ZP] Nel=2: using SCE co-motion f(r) from result",
                      flush=True)
            else:
                print(f"  [T_ZP] Nel=2: r/f shape mismatch "
                      f"({r_sce.shape} vs {f_sce.shape}) — "
                      "using density-only inversion", flush=True)
        except Exception as exc:
            print(f"  [T_ZP] Nel=2: result setup failed ({exc}) — "
                  "using density-only inversion", flush=True)

    if not use_result:
        print("  [T_ZP] Nel=2: density-only Ne(r) inversion", flush=True)

    # Ne(r) inversion (fallback path)
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
        f_r = (float(f_interp(r)) if use_result
               else float(Ne_inv(2.0 - float(Ne[i]))))
        d   = r + f_r
        if d < 1e-12 or f_r < 1e-12:
            continue
        lb = (r * r + f_r * f_r) / (r * f_r * d ** 3)
        if lb <= 0.0:
            continue
        rho_f = float(np.interp(f_r, r_grid, rho_1d))
        if rho_f < 1e-30:
            continue
        p  = (r / f_r) ** 2 * rho_r / rho_f
        ls = 2.0 * (p * p + 1.0) / (p * d ** 3)
        Wp_sum += (4.0 * np.pi * r ** 2 * rho_r
                   * (2.0 * np.sqrt(lb) + np.sqrt(max(ls, 0.0))) * dr)

    val = 0.25 * Wp_sum
    print(f"  [T_ZP] Nel=2 exact analytical W'∞ = {val:.8f}", flush=True)
    return val


# ── Nel = 4 : planar co-motion ────────────────────────────────────────────────

def compute_tzp_for_planar(result, density=None) -> float:
    """T_ZP for planar 4-electron configurations (Be, Li-).

    SCE minimum for Nel=4 is planar → one extra zero mode vs non-planar,
    so n_zero = 3  (2 already gauged + 1 out-of-plane rotation).
    """
    from scipy.integrate import simpson
    from jaxsce.optimize import AngularOptimizationResult

    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    Nel         = res.f.shape[1]
    coordinates = res.opt.coordinates
    N_grid_end  = res.angles.shape[0]

    n_angles    = res.angles.shape[1]
    has_radials = (hasattr(coordinates, 'Vee_hessian_radials') and
                   hasattr(coordinates, 'Vee_hessian_angles_radials'))
    n_radials   = Nel - 1 if has_radials else 0
    n_dof       = n_angles + n_radials
    n_zero      = 3                        # planar: 2 gauged + 1 out-of-plane
    n_vib       = max(n_dof - n_zero, 1)

    print(f"  [T_ZP] Planar Nel={Nel}: dim={n_dof} "
          f"(ang={n_angles}, rad={n_radials}), zeros={n_zero}, vib={n_vib}",
          flush=True)

    omega_sums = np.zeros(N_grid_end)

    for idx in range(N_grid_end):
        if idx % 50 == 0 and idx > 0:
            print(f"  [T_ZP] Progress: {idx}/{N_grid_end}", flush=True)
        try:
            if hasattr(coordinates, 'Vee_hessian_angles'):
                h_aa = np.asarray(coordinates.Vee_hessian_angles(
                    res.angles[idx], res.f[idx]))
                if has_radials:
                    h_rr = np.asarray(coordinates.Vee_hessian_radials(
                        res.angles[idx], res.f[idx]))
                    h_ar = np.asarray(coordinates.Vee_hessian_angles_radials(
                        res.angles[idx], res.f[idx]))
                    H = np.block([[h_aa, h_ar], [h_ar.T, h_rr]])
                else:
                    H = h_aa
            else:
                from jax import grad, jacfwd
                x0 = np.concatenate((np.asarray(res.f[idx]),
                                     np.asarray(res.angles[idx])))
                def vee_sph(x):
                    return coordinates.Vee(x[Nel:], x[:Nel])
                H = np.asarray(jacfwd(grad(vee_sph))(x0))

            H       = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H)        # ascending
            vib_pos = eigvals[-n_vib:] if len(eigvals) >= n_vib else eigvals
            vib_pos = vib_pos[vib_pos > 1e-8]
            omega_sums[idx] = float(np.sum(np.sqrt(vib_pos)))

        except Exception as exc:
            print(f"  [T_ZP] Warning at idx {idx}: {exc}", flush=True)
            omega_sums[idx] = 0.0

    if res.opt.grid == "r":
        x         = res.r[:N_grid_end]
        integrand = 4 * np.pi * x ** 2 * res.rho[:N_grid_end] * omega_sums
    elif res.opt.grid == "Ne":
        x         = res.Ne[:N_grid_end]
        integrand = omega_sums
    else:
        raise ValueError(f"Unknown grid type '{res.opt.grid}'")

    val = 0.5 * float(simpson(integrand, x=x))
    print(f"  [T_ZP] Planar W'∞ = {val:.8f}", flush=True)
    return val


# ── Nel = 3 and Nel >= 5 : local Hessian ─────────────────────────────────────

def compute_tzp_local(result, density=None, integrator="simpson") -> float:
    """T_ZP via local Hessian eigenvalues — general Nel != 2 and != 4."""
    from scipy.integrate import simpson, trapezoid
    from jaxsce.optimize import AngularOptimizationResult

    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    if not hasattr(res, "opt") or not hasattr(res.opt, "coordinates"):
        raise ValueError("Result object does not expose a coordinate system.")

    coordinates = res.opt.coordinates
    N_grid_end  = res.angles.shape[0]
    Nel         = res.f.shape[1]

    print(f"  [T_ZP] Local Hessian for {N_grid_end} grid points "
          f"(Nel={Nel})...", flush=True)

    omega_sums = np.zeros(N_grid_end)

    for idx in range(N_grid_end):
        if idx % 50 == 0 and idx > 0:
            print(f"  [T_ZP] Progress: {idx}/{N_grid_end}", flush=True)
        try:
            from jax import grad, jacfwd
            x0 = np.concatenate((np.asarray(res.f[idx]),
                                  np.asarray(res.angles[idx])))
            def vee_sph(x):
                return coordinates.Vee(x[Nel:], x[:Nel])
            H       = np.asarray(jacfwd(grad(vee_sph))(x0))
            eigvals = np.linalg.eigvalsh(H)
            # clip small negatives (numerical noise) to zero before sqrt
            omega_sums[idx] = float(
                np.sum(np.sqrt(np.clip(eigvals, 0.0, None)))
            )
        except Exception as exc:
            print(f"  [T_ZP] Warning at idx {idx}: {exc}", flush=True)
            omega_sums[idx] = 0.0

    if res.opt.grid == "r":
        x = res.r[:N_grid_end]
        y = 4 * np.pi * x ** 2 * res.rho[:N_grid_end] * omega_sums
    elif res.opt.grid == "Ne":
        x = res.Ne[:N_grid_end]
        y = omega_sums
    else:
        raise ValueError(f"Unknown grid type '{res.opt.grid}'")

    integral = (trapezoid(y, x=x) if integrator == "trapezoid"
                else simpson(y, x=x))

    val = 0.5 * float(integral)
    print(f"  [T_ZP] Local Hessian W'∞ = {val:.8f}", flush=True)
    return val


# ── Master dispatcher ─────────────────────────────────────────────────────────

def compute_tzp_from_result(result, density, nel: int) -> float:
    """Route T_ZP computation to the correct method for this *nel*.

    Nel == 2  →  exact analytical  (compute_tzp_nel2)
    Nel == 4  →  planar co-motion  (compute_tzp_for_planar, last point dropped)
    Nel == 3  →  local Hessian     (compute_tzp_local)
    Nel >= 5  →  local Hessian     (compute_tzp_local)
    """
    from jaxsce.optimize import AngularOptimizationResult

    # ── Nel = 2: exact ────────────────────────────────────────────────────────
    if nel == 2:
        print(f"  [T_ZP] Nel=2: exact analytical formula", flush=True)
        return compute_tzp_nel2(density, result=result)

    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    # ── Nel = 4: planar ───────────────────────────────────────────────────────
    if nel == 4:
        n_pts = res.f.shape[0]
        if n_pts > 1:
            print(f"  [T_ZP] Nel=4 planar: dropping last grid point "
                  f"({n_pts} → {n_pts - 1})", flush=True)
            res_use = _TruncatedResult(res, n_pts - 1)
        else:
            res_use = res
        return compute_tzp_for_planar(res_use, density)

    # ── Nel = 3 and Nel >= 5: local Hessian ───────────────────────────────────
    print(f"  [T_ZP] Nel={nel}: local Hessian method", flush=True)
    val = compute_tzp_local(result, density)
    if math.isfinite(val) and val > 0:
        return val

    print("  [T_ZP] WARNING: Hessian gave non-positive/NaN → returning NaN",
          flush=True)
    return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# CORE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_system(atom, charge, spin, basis, density_type, args):
    """Run optimization + T_ZP + integration for one system."""

    basis_value, has_ecp, ecp_value = load_basis_for_pyscf(basis, atom)

    start_time = timer()

    atom_key  = atom_label(atom, charge)
    ref       = MOLE_REFERENCES.get(atom_key, {})
    W_inf_ref = ref.get("W_ref")
    T_ZP_ref  = ref.get("Wp_ref")

    if spin is None:
        spin = ref.get("spin", 0)

    # grid type
    from pyscf import gto
    mol       = gto.M(atom=atom, charge=charge, spin=spin)
    nel       = mol.nelectron
    grid_type = args.grid
    if nel == 2 and grid_type == "r":
        print("  WARNING: Grid 'r' not supported for Nel=2, "
              "switching to 'Ne'", flush=True)
        grid_type = "Ne"

    # starting indices
    if args.nstart:
        nstart = jnp.array(args.nstart, dtype=jnp.int32)
    else:
        nstart = jnp.array([1, args.N_grid // 2, args.N_grid - 2],
                            dtype=jnp.int32)

    # data directory
    datadir = f"data/{atom}/{basis}/{_density_folder(density_type)}"
    chkfile = "chkfile.chk"
    os.makedirs(datadir, exist_ok=True)

    density_kwargs = dict(
        atom=atom,
        charge=charge,
        spin=spin,
        N_points_Ne=args.N_points_Ne,
        basis=basis_value,
        cumulant_threshold=args.cumulant_threshold,
        invNe_tol=args.invNe_tol,
    )
    if has_ecp and ecp_value is not None:
        density_kwargs["ecp"] = ecp_value
        print(f"  Using ECP for {atom}", flush=True)

    # ── density selection ────────────────────────────────────────────────────
    if density_type in ("hartree-fock", "hf"):
        print("  Using Hartree-Fock density", flush=True)
        density = HFDensity(chkfile_name=chkfile, chkfile_dir=datadir,
                            **density_kwargs)

    elif density_type == "ccsd":
        print("  Using CCSD density", flush=True)
        density = CCSDDensity(dm_file_name="dm_ccsd.npy", dm_file_dir=datadir,
                              chkfile_name=chkfile, chkfile_dir=datadir,
                              **density_kwargs)

    elif density_type in ("ccsd(t)", "ccsdt"):
        print("  Using CCSD(T) relaxed density", flush=True)
        density = CCSDTDensity(dm_file_name="dm_ccsdt.npy", dm_file_dir=datadir,
                               chkfile_name=chkfile, chkfile_dir=datadir,
                               **density_kwargs)

    elif density_type == "fci":
        print("  Using FCI density", flush=True)
        density = FCIDensity(dm_file_name="dm_fci.npy", dm_file_dir=datadir,
                             **density_kwargs)

    else:
        raise ValueError(f"Unsupported density type: {density_type!r}")

    print(f"  Nel = {density.Nel}, U = {density.U:.8f}", flush=True)

    # ── optimizer ────────────────────────────────────────────────────────────
    if density.Nel == 2:
        opt = TwoElectron(N_grid=args.N_grid, grid=grid_type)
    else:
        coordinates      = get_coordinate_system(args.coordinate_system, args.seed)
        coordinates_last = get_coordinate_system(args.coordinate_system_last, args.seed)
        opt = AngularOptimization(
            coordinates=coordinates,
            coordinates_last=coordinates_last,
            N_grid=args.N_grid,
            N_random=args.N_random,
            N_random_last=args.N_random_last,
            N_select=args.N_select,
            N_sweeps=args.N_sweeps,
            strategy=args.strategy,
            grid=grid_type,
        )

    print("  Running optimization...", flush=True)
    opt_result = opt.run(density, nstart)

    filename = f"N{args.N_grid}{_filename_suffix(density_type)}"
    opt_result.save(filename, datadir)

    # ── T_ZP / W'∞  (correct physics) ───────────────────────────────────────
    print("  Computing T_ZP (W'∞)...", flush=True)
    try:
        T_ZP = compute_tzp_from_result(opt_result, density, nel=density.Nel)
    except Exception as exc:
        print(f"  [T_ZP] FAILED: {type(exc).__name__}: {exc}", flush=True)
        T_ZP = float("nan")

    # ── integrals & W_inf ────────────────────────────────────────────────────
    print("  Computing integrals...", flush=True)
    ints = VeeIntegration(opt_result)
    ints.interpolate_all(4)
    integrals, extrapolated = ints.integrate(4)
    ints_time = timer()

    # W_inf — use robust multi-estimator selector instead of hard bpoly_2 lookup
    Winf, estimator = best_winf_fixed(density.U, integrals, extrapolated)
    Lambda = (-Winf / density.LDA_int if density.LDA_int != 0
              else float("nan"))
    B      = ((Winf - A * density.LDA_int) / density.GEA_int
              if density.GEA_int != 0 else float("nan"))

    error_W   = ((Winf - W_inf_ref)
                 if (W_inf_ref is not None and math.isfinite(Winf)) else None)
    error_TZP = ((T_ZP - T_ZP_ref)
                 if (T_ZP_ref is not None and math.isfinite(T_ZP)) else None)

    result = {
        "integrals":    integrals,
        "extrapolated": extrapolated,
        "U":            float(density.U),
        "LDA_int":      float(density.LDA_int),
        "GEA_int":      float(density.GEA_int),
        "Winf":         float(Winf),
        "T_ZP":         float(T_ZP),
        "Lambda":       float(Lambda),
        "B":            float(B),
        "estimator":    estimator,
    }

    with open(f"{datadir}/{filename}_integrals.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ── printout ─────────────────────────────────────────────────────────────
    winf_str = f"{Winf:.10f}"    if math.isfinite(Winf)  else "N/A"
    lam_str  = f"{Lambda:.10f}"  if math.isfinite(Lambda) else "N/A"
    b_str    = f"{B:.10f}"       if math.isfinite(B)      else "N/A"
    t_str    = f"{T_ZP:.10f}"    if math.isfinite(T_ZP)   else "N/A"
    ew_str   = f"{error_W:.6f}"  if error_W   is not None else "N/A"
    et_str   = f"{error_TZP:.6f}" if error_TZP is not None else "N/A"

    print(f"\n{'='*60}")
    print(f"RESULTS for {atom_key} (Nel={density.Nel})")
    print(f"{'='*60}")
    print(f"  Basis            : {basis}")
    print(f"  Density          : {density_type.upper()}")
    print(f"  Grid             : {grid_type}")
    print(f"  Estimator        : {estimator}")
    print(f"  U                : {density.U:.10f}")
    print(f"  W_inf            : {winf_str}")
    print(f"  T_ZP (W'∞)       : {t_str}")
    print(f"  Lambda           : {lam_str}")
    print(f"  B                : {b_str}")
    if W_inf_ref is not None:
        print(f"  Reference W_inf  : {W_inf_ref:.6f}")
        print(f"  Error W_inf      : {ew_str}")
    if T_ZP_ref is not None:
        print(f"  Reference T_ZP   : {T_ZP_ref:.6f}")
        print(f"  Error T_ZP       : {et_str}")
    print(f"  Time             : {ints_time - start_time:.1f}s")
    print(f"{'='*60}\n")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCE optimization for atoms (single system or batch mode)"
    )

    parser.add_argument("--atom",   type=str,
                        help="Atom symbol for single system run")
    parser.add_argument("--charge", type=int,  default=0)
    parser.add_argument("--spin",   type=int,  default=None,
                        help="Spin (auto-detected from reference table if omitted)")
    parser.add_argument("--basis",  type=str,  default="cc-pVDZ")

    parser.add_argument(
        "--density", type=str, default="hf",
        choices=["hartree-fock", "hf", "ccsd", "ccsd(t)", "ccsdt", "fci"],
        help="Density type (default: hf)",
    )

    parser.add_argument("--grid",               type=str,   default="Ne",
                        choices=["r", "Ne"])
    parser.add_argument("--N-grid",             type=int,   default=1025)
    parser.add_argument("--N-random",           type=int,   default=10000)
    parser.add_argument("--N-random-last",      type=int,   default=30000)
    parser.add_argument("--N-select",           type=int,   default=30)
    parser.add_argument("--N-sweeps",           type=int,   default=0)
    parser.add_argument("--strategy",           type=str,   default="sweeps")
    parser.add_argument("--nstart",             type=int,   nargs="+", default=[])
    parser.add_argument("--N-points-Ne",        type=int,   default=10000)
    parser.add_argument("--invNe-tol",          type=float, default=1e-10)
    parser.add_argument("--cumulant-threshold", type=float, default=1e-8)
    parser.add_argument("--coordinate-system",      type=str, default="reduced")
    parser.add_argument("--coordinate-system-last", type=str, default="double_reduced")
    parser.add_argument("--seed", type=int, default=5)

    args = parser.parse_args()
    args.density = args.density.lower()

    # ── single system ────────────────────────────────────────────────────────
    if args.atom is not None:
        print(f"\n{'='*60}")
        print(f"  Running single system: {args.atom}")
        print(f"  Charge: {args.charge}, Basis: {args.basis}, "
              f"Density: {args.density.upper()}")
        print(f"{'='*60}")
        run_system(atom=args.atom, charge=args.charge, spin=args.spin,
                   basis=args.basis, density_type=args.density, args=args)
        return

    # ── batch mode ───────────────────────────────────────────────────────────
    if not SYSTEM_BASIS:
        print("Error: SYSTEM_BASIS is empty. Use --atom for single system.")
        sys.exit(1)

    batch_start = timer()
    all_results: dict = {}
    failures: list    = []

    print(f"\n{'='*60}")
    print(f"  Batch run — density={args.density.upper()}  "
          f"grid={args.grid}  N_grid={args.N_grid}")
    print(f"  Systems: {len(SYSTEM_BASIS)}")
    print(f"{'='*60}")

    for atom, charge, spin, basis in SYSTEM_BASIS:
        tag = f"{atom_label(atom, charge)}/{basis}"
        print(f"\n{'─'*60}")
        print(f"  ▶  Running {tag}  (charge={charge}, spin={spin})")
        print(f"{'─'*60}")
        try:
            res = run_system(atom=atom, charge=charge, spin=spin,
                             basis=basis, density_type=args.density, args=args)
            all_results[tag] = res
        except Exception as exc:
            print(f"\n  ✗  FAILED: {tag}")
            print(f"     {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()
            failures.append((tag, exc))

    total_time = timer() - batch_start

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY   (total wall time: {total_time:.1f}s)")
    print(f"{'='*60}")

    if all_results:
        header = (f"  {'System':<22} {'W_inf':>12} {'Err_W':>10} "
                  f"{'T_ZP':>12} {'Err_TZP':>10}")
        print(header)
        print("  " + "─" * (len(header) - 2))

        for tag, res in all_results.items():
            atom_key  = tag.split("/")[0]
            ref       = MOLE_REFERENCES.get(atom_key, {})
            W_inf_ref = ref.get("W_ref")
            T_ZP_ref  = ref.get("Wp_ref")

            Winf = res["Winf"]
            T_ZP = res.get("T_ZP", float("nan"))

            err_W   = ((Winf - W_inf_ref)
                       if (W_inf_ref is not None and math.isfinite(Winf))
                       else float("nan"))
            err_TZP = ((T_ZP - T_ZP_ref)
                       if (T_ZP_ref is not None and math.isfinite(T_ZP))
                       else float("nan"))

            w_str = (f"{Winf:>12.6f}" if math.isfinite(Winf)
                     else f"{'N/A':>12}")
            ew_str = (f"{err_W:>10.6f}" if math.isfinite(err_W)
                      else f"{'N/A':>10}")
            t_str = (f"{T_ZP:>12.6f}" if math.isfinite(T_ZP)
                     else f"{'N/A':>12}")
            e_str = (f"{err_TZP:>10.6f}" if math.isfinite(err_TZP)
                     else f"{'N/A':>10}")

            print(f"  {tag:<22} {w_str} {ew_str} {t_str} {e_str}")

    if failures:
        print(f"\n  Failed systems ({len(failures)}):")
        for tag, exc in failures:
            print(f"    ✗  {tag}: {type(exc).__name__}: {exc}")

    print(f"\n  Done. {len(all_results)}/{len(SYSTEM_BASIS)} systems completed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
