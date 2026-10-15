#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 DISSOCIATION SCAN: DENSITY LOOP + DFT W_inf MODELS + SCE (radial + 3D cyl)
=============================================================================
version 2026-09-08-reference-density-stable-stretch

Cleaned:
  * single GL2 scenario only (dou_sig.csv); the singles.csv loop is gone
  * PC4 removed from the DFT W_inf model set
  * unused _regularise_ovlp() helper removed
  * XC-components block (calc_exc / xc_all + ex_*/ec_* columns) removed
  * SPL / gISI2 correlations are built ONLY from SCE data
    (W_inf_SCE, Wp_inf_SCE); the exact-2RDM diagnostic pair is gone
  * density loop restored as HF followed by FCI
  * each HF/FCI target has two fully independent inversions: Wu--Yang and
    PyOEP (KSINV for the closed-shell H2 scan)
  * HF/FCI reference densities feed DFT, radial-SCE, cylindrical-SCE and ACM;
    independently inverted WY/PyOEP orbitals and eigenvalues feed GL2 only
  * E_c = E_FCI(FCI_DM) - E_HF(HF_DM), stored with both total energies
  * W0/exact exchange is evaluated on the selected HF/FCI reference density:
    E_xx_density = -J[reference density]/2 for two-electron singlet H2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy as _copy
import csv
import functools
import math
import multiprocessing as mp_ctx
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, optimize, special

warnings.filterwarnings("ignore")

from pyscf import ao2mo, cc, dft, fci, gto, lib as pyscf_lib_top, mp, scf
from pyscf.dft import numint as pyscf_numint

from wu_yang import (
    DFT_MODELS,
    compute_W_for_density,
    make_exc_func,
    model_xctype,
)

# Jobs created by create3 run from nested R_* directories.  Make local
# checkouts visible without requiring create3 to copy these large packages.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = []
_here = _SCRIPT_DIR
for _ in range(6):
    if _here not in _SEARCH_DIRS:
        _SEARCH_DIRS.append(_here)
    _parent = os.path.dirname(_here)
    if _parent == _here:
        break
    _here = _parent
for _base in _SEARCH_DIRS:
    if os.path.isdir(os.path.join(_base, 'kspies')) and _base not in sys.path:
        sys.path.insert(0, _base)

try:
    from kspies import wy as wy_module
    KSPIES_AVAILABLE = True
except ImportError:
    KSPIES_AVAILABLE = False


# Independent PyOEP inversion.  A launcher may either put ``methods`` on
# PYTHONPATH or place the PyOEP checkout beside this script.  PyOEP is still
# attempted when Wu--Yang is unavailable or fails, and vice versa.
_PYOEP_ROOT = os.environ.get('PYOEP_ROOT', '').strip()
if not _PYOEP_ROOT:
    _PYOEP_ROOT = next(
        (os.path.join(base, 'PyOEP') for base in _SEARCH_DIRS
         if os.path.isdir(os.path.join(base, 'PyOEP', 'methods'))),
        os.path.join(_SCRIPT_DIR, 'PyOEP'),
    )
if os.path.isdir(_PYOEP_ROOT) and _PYOEP_ROOT not in sys.path:
    sys.path.insert(0, _PYOEP_ROOT)
try:
    from methods import exxoep as _PYOEP_EXXOEP_MODULE
    from methods.ksinv import KSINV as _KSINV
    PYOEP_AVAILABLE = True
    _PYOEP_IMPORT_ERROR = ''
except ImportError as exc:
    _PYOEP_EXXOEP_MODULE = None
    _KSINV = None
    PYOEP_AVAILABLE = False
    _PYOEP_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

R_UNIT      = 'Bohr'          # EXPLICIT. Was implicitly Angstrom before.
R_MIN_DEF   = 0.03            # R = 0 exactly is singular; this is the floor.
R_MAX_DEF   = 25.0
BASIS_DEF   = 'aug-cc-pV5z'

N_ROOTS      = 1
N_GRID       = 8001
N_ANG        = 64
ANG_OPT_ITER = 100
LINDEP_THR   = 1e-8
WY_ETA       = 1e-5

# Number of numerical-integration points handled per AO-density batch.
# This must be defined before _rho_l1_diff is declared because Python
# evaluates default argument expressions while importing the module.
GRID_BLKSIZE = int(os.environ.get('GRID_BLKSIZE', '8192'))

# ── WY inversion ───────────────────────────────────────────────────────────
WY_TOL          = 1e-5
WY_MAXITER      = 5000
WY_RHO_L1_TOL   = 1e-3    # validated Ne/CO criterion (integrated |drho|)
WY_STRICT       = False   # False: record the L1 error, keep the orbitals

# ── independent PyOEP / KSINV inversion ───────────────────────────────────
PYOEP_RHO_L1_TOL = 1e-3
PYOEP_MAXITER    = int(os.environ.get('PYOEP_MAXITER', '300'))
PYOEP_CONV_THR   = float(os.environ.get('PYOEP_CONV_THR', '1e-10'))
# PyOEP recommends a separate, balanced auxiliary potential basis.  For an
# aug-cc-pV5Z orbital basis, the documented preprocessing threshold is 5e-3.
PYOEP_OEP_BASIS  = os.environ.get(
    'PYOEP_OEP_BASIS', 'aug-cc-pV5z'
    #'PYOEP_OEP_BASIS', 'aug-cc-pVDZ-RIFIT'
).strip()
PYOEP_THR_FAI    = float(os.environ.get('PYOEP_THR_FAI_OEP', '5e-3'))
PYOEP_SPACE_SYM  = os.environ.get('PYOEP_SPACE_SYM', '1').strip().lower() in (
    '1', 'true', 'yes', 'on'
)

# ── GL2 ────────────────────────────────────────────────────────────────────
GL2_GAP_TOL         = 1e-8
GL2_STRICT          = False   # False: floor denominators, keep the row
SINGLES_SPIN_FACTOR = 2.0     # alpha/beta spin sum

# ── scenarios: master gl2 scalar -> output file ────────────────────────────
GL2_SCENARIOS = (
    ('dou_sig', 'dou_sig.csv'),   # gl2 = doubles + singles
)

# Process the HF density first, then the FCI density.  Python preserves this
# list order throughout the per-system Wu--Yang/GL2/SCE loops and CSV sections.
DENSITY_TYPES = ['HF', 'FCI']
INVERSION_METHODS = ['WY', 'PYOEP']
#INVERSION_METHODS = ['PYOEP']


def parse_density_types(values, parser):
    """Return canonical density names in the fixed HF-then-FCI order.

    H2 has only two electrons, so CCSD and CCSD(T) coincide with FCI in a
    finite one-particle basis; accept those common spellings as FCI aliases.
    "all" selects both implemented density blocks.
    """
    aliases = {
        'hf': ('HF',),
        'fci': ('FCI',),
        'ccsd': ('FCI',),
        'ccsdt': ('FCI',),
        'ccsd(t)': ('FCI',),
        'all': tuple(DENSITY_TYPES),
    }
    requested = set()
    for value in values:
        key = value.strip().lower()
        if key not in aliases:
            parser.error(
                "argument --density-types: invalid choice: "
                f"{value!r} (choose from HF, FCI, CCSD, CCSDT, CCSD(T), all)"
            )
        requested.update(aliases[key])
    return [dt for dt in DENSITY_TYPES if dt in requested]

# ── MASTER ON/OFF SWITCHES ──────────────────────────────────────────────────
# Edit these two booleans to turn whole blocks of the calculation on or off.
#   True  -> that block runs
#   False -> that block is skipped and its columns are written blank/NaN
# The corresponding CLI flags (--no-dft-models, --no-sce3d) can still force a
# block OFF for a single run, but they cannot turn ON a block set False here.
COMPUTE_DFT_MODELS = True    # 13 DFT W_inf / Wp_inf models (LDA … ePC)
COMPUTE_SCE3D      = True    # 3D cylindrical SCE / rm-ZPE (the *_cyl columns)

SCE3D_CFG: Dict = dict(
    grid_level       = 5,
    m                = 2,
    max_outer_iter   = 200,
    zpe_max_points   = None,
    zpe_density_mass = 0.9995,
    partner_mode     = 'gradient',
    seed             = 17,
)

pi           = np.pi
_BLANK_LINES = 5

R_VALS = np.linspace(7.7, 11, 233) #[1.4, 2.0]

# ── GEOMETRY GRID ──────────────────────────────────────────────────────────
# EDIT THESE TWO BLOCKS to change the scan. R is in BOHR (see R_UNIT above).
# R = 0 exactly is singular (coincident nuclei, E_nn -> inf), so the first
# point is the practical stand-in for zero.
#R_VALS = [0.2973000000]  # patched by create2_flexible

#R_VALS = [1.4]
print(R_VALS)
print("Number of points:", len(R_VALS))




SYSTEM_BASIS = [
    (
        f"H2_{R:.2f}",
        f"H 0 0 {-R/2:.5f}; H 0 0 {R/2:.5f}",
        0,
        0,
        "aug-cc-pV5Z",
        0.0,
    )
    for R in R_VALS
]


def _ensure_unique_labels(systems):
    """The label is the checkpoint key, so a collision would silently
    overwrite a geometry. f"{R:.2f}" is unique for the grid above, but if
    you densify R_VALS it may not be -- widen the precision automatically
    rather than losing points."""
    labels = [s[0] for s in systems]
    if len(set(labels)) == len(labels):
        return systems
    for prec in (3, 4, 5, 6):
        relabelled = []
        for s, R in zip(systems, R_VALS):
            relabelled.append((f"H2_{R:.{prec}f}",) + tuple(s[1:]))
        if len({x[0] for x in relabelled}) == len(relabelled):
            print(f"  ⚠ label collision at 2 dp — widened to {prec} dp")
            return relabelled
    raise ValueError("cannot make geometry labels unique")


def build_r_vals(r_min: float = R_MIN_DEF, r_max: float = R_MAX_DEF,
                 n_short: int = 60, n_mid: int = 60, n_long: int = 30):
    """CLI override for R_VALS, same three-segment shape."""
    return np.concatenate([
        np.linspace(r_min, 2.0, n_short, endpoint=False),
        np.linspace(2.0, 10.0, n_mid, endpoint=False),
        np.linspace(10.0, r_max, n_long, endpoint=True),
    ])


def build_systems(r_vals, basis_name: str = BASIS_DEF):
    return [
        (f"H2_{R:.2f}",
         f"H 0 0 {-R/2:.5f}; H 0 0 {R/2:.5f}",
         0, 0, basis_name, 0.0)
        for R in r_vals
    ]



# ═══════════════════════════════════════════════════════════════════════════
#  3D CYLINDRICAL SCE / rm-ZPE
# ═══════════════════════════════════════════════════════════════════════════

_SQRT_PI = math.sqrt(math.pi)
_EPS_SCE = 1.0e-12


@dataclass
class SCEParams:
    A: np.ndarray
    p: np.ndarray
    q: np.ndarray
    a: float


def _decode_params(raw: np.ndarray, m: int) -> SCEParams:
    raw = np.asarray(raw, dtype=float)
    return SCEParams(A=raw[:m], p=np.exp(raw[m:2*m]),
                     q=np.exp(raw[2*m:3*m]), a=float(np.exp(raw[-1])))


def _erf_over_r(a, r):
    r = np.asarray(r); out = np.empty_like(r, dtype=float)
    sm = r < 1e-8
    out[sm]  = 2.0*a/_SQRT_PI - 2.0*a**3*r[sm]**2/(3.0*_SQRT_PI)
    out[~sm] = special.erf(a*r[~sm])/r[~sm]
    return out


def _d_erf_over_r_dr(a, r):
    r = np.asarray(r); out = np.empty_like(r, dtype=float)
    sm = r < 1e-8
    out[sm] = -4.0*a**3*r[sm]/(3.0*_SQRT_PI)
    rr = r[~sm]
    out[~sm] = (2.0*a/_SQRT_PI)*np.exp(-(a*rr)**2)/rr - special.erf(a*rr)/rr**2
    return out


def v_sce_cyl(z, h, pars):
    z, h = np.asarray(z), np.asarray(h)
    r = np.sqrt(z*z + h*h); v = _erf_over_r(pars.a, r)
    for A, p, q in zip(pars.A, pars.p, pars.q):
        v = v + A*np.exp(-p*z*z - q*h*h)
    return v


def _grad_v_sce_cyl(z, h, pars):
    z, h = np.asarray(z), np.asarray(h)
    r = np.sqrt(z*z + h*h)
    dz = np.zeros_like(z); dh = np.zeros_like(h)
    dr = _d_erf_over_r_dr(pars.a, r)
    inv_r = np.where(r > 1e-10, 1.0/np.maximum(r, 1e-300), 0.0)
    dz += dr*z*inv_r; dh += dr*h*inv_r
    for A, p, q in zip(pars.A, pars.p, pars.q):
        e = np.exp(-p*z*z - q*h*h)
        dz += A*e*(-2.0*p*z); dh += A*e*(-2.0*q*h)
    return dz, dh


def _pot_h2_restricted(x, pars):
    z1, h1, z2, h2 = map(float, x)
    d = math.sqrt(max((h1+h2)**2 + (z1-z2)**2, _EPS_SCE))
    return 1.0/d - float(v_sce_cyl(np.array([z1]), np.array([h1]), pars)[0]) \
                 - float(v_sce_cyl(np.array([z2]), np.array([h2]), pars)[0])


def _clip(x, bounds, eps=1e-10):
    arr = np.asarray(x, dtype=float).flatten()
    for i, (lo, hi) in enumerate(bounds):
        if not np.isfinite(arr[i]):
            arr[i] = 0.5*(lo+hi)
        arr[i] = float(np.clip(arr[i], lo+eps, hi-eps if hi > lo else lo))
    return arr


def _dedup(starts):
    out, seen = [], set()
    for x in starts:
        key = tuple(np.round(np.asarray(x, dtype=float), 10))
        if key not in seen:
            seen.add(key); out.append(np.asarray(x, dtype=float))
    return out


def _make_inner_min(R, zmax, hmax, n_rand=8, seed=17):
    rng = np.random.default_rng(seed)
    bnds = [(-zmax, zmax), (0.0, hmax), (-zmax, zmax), (0.0, hmax)]

    def run(pars, guess=None):
        starts = []
        if guess is not None:
            starts.append(_clip(guess, bnds))
        starts += [np.array(s) for s in [[+R/2, 0.0, -R/2, 0.0], [+R/2, 0.2, -R/2, 0.2],
                                         [+R/2, 0.6, -R/2, 0.6], [0.0, 0.2, 0.0, 0.2],
                                         [0.0, 0.8, 0.0, 0.8]]]
        for _ in range(n_rand):
            starts.append(np.array([rng.uniform(-zmax, zmax), rng.uniform(0, hmax),
                                    rng.uniform(-zmax, zmax), rng.uniform(0, hmax)]))
        best_v, best_x = np.inf, None
        for x0 in _dedup([_clip(s, bnds) for s in starts]):
            try:
                res = optimize.minimize(lambda xx: _pot_h2_restricted(xx, pars),
                                        x0=_clip(x0, bnds), method='L-BFGS-B', bounds=bnds,
                                        options={'maxiter': 600, 'ftol': 1e-13})
            except ValueError:
                continue
            if np.isfinite(res.fun) and res.fun < best_v:
                best_v = res.fun; best_x = _clip(res.x, bnds)
        if best_x is None:
            raise RuntimeError("Inner SCE minimisation failed.")
        return best_v, best_x
    return run


def _optimize_sce_potential(R, z, h, weights, rho, m, max_outer_iter, zmax, hmax, seed):
    wrho = weights * rho
    inner = _make_inner_min(R, zmax, hmax, seed=seed)
    ib = [(-zmax, zmax), (0.0, hmax), (-zmax, zmax), (0.0, hmax)]

    def mk(A0, p0, q0, a0):
        A = np.full(m, A0)
        p = np.array([p0*(1+0.5*i) for i in range(m)])
        q = np.array([q0*(1+0.7*i) for i in range(m)])
        return np.concatenate([A, np.log(p), np.log(q), [math.log(a0)]])

    g0s = [mk(0.10, 0.8, 0.8, 0.7), mk(0.30, 0.7, 1.0, 0.8),
           mk(0.50, 0.5, 1.5, 0.9), mk(-0.10, 0.8, 1.2, 0.7),
           mk(-0.40, 1.0, 1.5, 0.5)]
    best_obj, best_raw, best_ix = np.inf, None, None
    for raw0 in g0s:
        state = {'last': _clip([+R/2, 0.2, -R/2, 0.2], ib)}

        def obj(raw):
            pars = _decode_params(raw, m)
            gv, xopt = inner(pars, guess=state['last'])
            state['last'] = _clip(xopt, ib)
            return -float(np.dot(wrho, v_sce_cyl(z, h, pars)) + gv)

        res = optimize.minimize(obj, x0=raw0, method='Powell',
                                options={'maxiter': max_outer_iter, 'xtol': 5e-4, 'ftol': 1e-6})
        if np.isfinite(res.fun) and res.fun < best_obj:
            best_obj = float(res.fun); best_raw = np.array(res.x); best_ix = np.array(state['last'])
    if best_raw is None:
        raise RuntimeError("Outer SCE optimisation failed.")
    return _decode_params(best_raw, m), -best_obj, best_ix


def _comotion(z, h, pars):
    vz, vh = _grad_v_sce_cyl(z, h, pars)
    g2 = vz*vz + vh*vh
    d = np.maximum(g2, 1e-16)**0.75
    return z + vz/d, h + vh/d


def _fd_hessian(fun, x0, step=1e-3):
    n = len(x0); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = step; ej[j] = step
            H[i, j] = H[j, i] = (fun(x0+ei+ej)-fun(x0+ei-ej)-fun(x0-ei+ej)+fun(x0-ei-ej))/(4*step**2)
    return H


def _rm_zpe_winf_prime(R, z, h, weights, rho, pars, zmax, hmax,
                       max_points=None, density_mass=0.9995, partner_mode='gradient'):
    imp = weights * rho; order = np.argsort(imp)[::-1]
    if max_points is not None or density_mass < 1.0 - 1e-12:
        tot = float(np.sum(imp)); tgt = min(max(density_mass, 0), 1)*tot
        mk  = len(order) if max_points is None else min(int(max_points), len(order))
        cs  = np.cumsum(imp[order[:mk]])
        nk  = min(max(int(np.searchsorted(cs, tgt, 'left'))+1, 1), mk)
        order = order[:nk]
    zz, hh, ww, rr = z[order], h[order], weights[order], rho[order]
    fz0, fh0 = _comotion(zz, hh, pars)
    local_omega = np.zeros_like(zz)
    ib2 = [(-zmax, zmax), (0.0, hmax)]
    last_p = None
    for i in range(len(zz)):
        z1, h1 = float(zz[i]), float(hh[i])
        if partner_mode == 'localmin':
            g = last_p if last_p is not None else (float(fz0[i]), abs(float(fh0[i])))
            starts = _dedup([_clip(s, ib2) for s in [np.array(g), np.array([-z1, h1]),
                             np.array([-R/2, max(h1, 0.1)]), np.array([0.0, max(h1, 0.2)])]])
            bv, bx = np.inf, None
            for x0 in starts:
                try:
                    res = optimize.minimize(lambda x: _pot_h2_restricted([z1, h1, x[0], x[1]], pars),
                                            x0=x0, method='L-BFGS-B', bounds=ib2,
                                            options={'maxiter': 300, 'ftol': 1e-13})
                    if np.isfinite(res.fun) and res.fun < bv:
                        bv = res.fun; bx = _clip(res.x, ib2)
                except ValueError:
                    pass
            if bx is None:
                continue
            z2, h2 = float(bx[0]), float(bx[1]); last_p = (z2, h2)
        else:
            z2, h2 = float(fz0[i]), abs(float(fh0[i]))
        x0 = np.array([z1, h1, z2, h2])
        H = _fd_hessian(lambda x: _pot_h2_restricted(x, pars), x0)
        H = 0.5*(H+H.T); eigs = np.linalg.eigvalsh(H)
        pos = eigs[eigs > 5e-6]
        if len(pos):
            local_omega[i] = np.sum(np.sqrt(np.sort(pos)[-2:]))
    return float(0.125 * np.dot(ww, rr * local_omega))


def compute_sce3d_rmzpe(mol, dm1_ao, R: float, grid_level=5, m=2,
                        max_outer_iter=200, zpe_max_points=None,
                        zpe_density_mass=0.9995, partner_mode='gradient', seed=17):
    dm = np.asarray(dm1_ao)
    if dm.ndim == 3:
        dm = dm[0]+dm[1]
    T_mat = mol.intor('int1e_kin'); V_nuc = mol.intor('int1e_nuc')
    ts   = float(np.einsum('ij,ij->', T_mat, dm))
    vext = float(np.einsum('ij,ij->', V_nuc, dm))
    vnn  = float(mol.energy_nuc())
    # J_H via get_jk instead of building the full nao^4 AO ERI (100 MB for
    # aug-cc-pV5Z H2, and it was being rebuilt for every density type).
    J_ao = scf.hf.get_jk(mol, dm)[0]
    J_H  = 0.5*float(np.einsum('ij,ij->', J_ao, dm))
    grids = dft.gen_grid.Grids(mol); grids.level = grid_level
    grids.build(with_non0tab=True)
    coords  = np.asarray(grids.coords); weights = np.asarray(grids.weights)
    ao0 = dft.numint.eval_ao(mol, coords, deriv=0)
    rho = np.clip(np.einsum('pi,ij,pj->p', ao0, dm, ao0), 0.0, None)
    xc, yc, zc = coords[:, 0], coords[:, 1], coords[:, 2]
    hc = np.sqrt(xc*xc + yc*yc)
    zmax = 1.25*float(np.max(np.abs(zc))); hmax = 1.25*float(np.max(hc))
    pars, vee_sce, inner_x = _optimize_sce_potential(
        R=R, z=zc, h=hc, weights=weights, rho=rho,
        m=m, max_outer_iter=max_outer_iter, zmax=zmax, hmax=hmax, seed=seed)
    winf = vee_sce - J_H
    winf_prime = _rm_zpe_winf_prime(R=R, z=zc, h=hc, weights=weights, rho=rho, pars=pars,
                                    zmax=zmax, hmax=hmax, max_points=zpe_max_points,
                                    density_mass=zpe_density_mass, partner_mode=partner_mode)
    total_sce       = ts + vext + vnn + vee_sce
    total_sce_rmzpe = total_sce + winf_prime
    c_lz = 0.5*float(v_sce_cyl(np.array([0.0]), np.array([0.0]), pars)[0])
    z1, h1, z2, h2 = map(float, inner_x)
    return {
        'ts_cyl': ts, 'vext_cyl': vext, 'vnn_cyl': vnn, 'hartree_cyl': J_H,
        'vee_sce_cyl': vee_sce, 'winf_cyl': winf,
        'winf_prime_rmzpe_cyl': winf_prime,
        'total_sce_cyl': total_sce, 'total_sce_rmzpe_cyl': total_sce_rmzpe,
        'levy_zahariev_c_cyl': c_lz,
        'inner_z1': z1, 'inner_h1': h1, 'inner_z2': z2, 'inner_h2': h2,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  RADIAL SCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def angular_objective(pts_flat, n):
    pts = pts_flat.reshape(n, 3); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    e = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i]-pts[j])
            if d > 1e-12:
                e += 1/d
    return e


def optimize_angular_points(n, iters=1000):
    i   = np.arange(n, dtype=float)
    phi = np.arccos(np.clip(1-2*i/n, -1, 1)); th = pi*(1+5**0.5)*i
    init = np.column_stack([np.sin(phi)*np.cos(th), np.sin(phi)*np.sin(th), np.cos(phi)]).flatten()
    res = optimize.minimize(angular_objective, init, args=(n,), method='L-BFGS-B',
                            bounds=[(-2, 2)]*3*n, options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8})
    pts = res.x.reshape(n, 3); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, res.success


@functools.lru_cache(maxsize=None)
def get_optimized_angular_vectors(n):
    """Deterministic Thomson-style angular grid; pure function of n, so cache
    it. Un-cached this re-ran a 1000-iteration L-BFGS-B (~75 s at N_ANG=64)
    on every call to get_density_radial for bit-identical output."""
    pts, _ = optimize_angular_points(n, ANG_OPT_ITER)
    return pts


def V_harm_system(mol, omega):
    if omega == 0.0:
        return np.zeros((mol.nao_nr(), mol.nao_nr()))
    grids = dft.gen_grid.Grids(mol); grids.level = 9; grids.build()
    coords = grids.coords; w = grids.weights; ao = dft.numint.eval_ao(mol, coords)
    ao = np.asarray(ao)
    if ao.ndim == 3:
        ao = ao[0]
    v = 0.5*omega**2*(coords[:, 0]**2+coords[:, 1]**2+coords[:, 2]**2)
    return np.einsum("gi,g,g,gj->ij", ao, v, w, ao)


def get_density_radial(mol, dm1_ao, r_grid):
    dm = np.asarray(dm1_ao)
    if dm.ndim == 3:
        dm = dm[0]+dm[1]
    ang = get_optimized_angular_vectors(N_ANG)
    ni  = pyscf_numint.NumInt()
    n_r, n_ang = len(r_grid), len(ang)
    coords_all = np.concatenate([np.outer(r_grid, a) for a in ang], axis=0)
    ao_all = np.asarray(ni.eval_ao(mol, coords_all))
    if ao_all.ndim == 3:
        ao_all = ao_all[0]
    ao_all = ao_all.reshape(n_ang, n_r, -1)
    rho_rays = np.einsum('agm,mn,agn->ag', ao_all, dm, ao_all)
    return np.maximum(rho_rays.mean(axis=0), 0.0)


def build_r_grid(r_max=None):
    """Radial grid for the spherically averaged density. Must comfortably
    enclose BOTH nuclei, so it has to grow with the bond length: at R = 25
    Bohr the atoms sit at +-12.5 and a 30-Bohr box is already tight."""
    n = N_GRID if N_GRID % 2 == 1 else N_GRID+1
    rmax = 30.0 if r_max is None else max(30.0, float(r_max))
    return np.linspace(1e-6, rmax, n)


def compute_T_ZP_and_Winf_N2(rho_1d, r_grid):
    from scipy.interpolate import interp1d
    dr = r_grid[1]-r_grid[0]; rho_1d = np.maximum(rho_1d, 0)
    Ne = np.clip(np.cumsum(4*pi*r_grid**2*rho_1d)*dr, 0, 2)
    umask = np.concatenate([[True], np.diff(Ne) > 1e-12])
    Ne_u, r_u = Ne[umask], r_grid[umask]
    if len(Ne_u) < 3:
        return 0.0, 0.0, 0
    Ne_inv = interp1d(np.concatenate([[0], Ne_u]), np.concatenate([[r_grid[0]], r_u]),
                      kind='linear', bounds_error=False, fill_value=(r_grid[0], r_grid[-1]))
    rho_i  = interp1d(r_grid, rho_1d, kind='linear', bounds_error=False, fill_value=0)
    i_half = int(np.searchsorted(Ne, 1.0))
    if i_half < 2:
        return 0.0, 0.0, 0
    Wp_sum = Win_sum = 0.0; n_pts = 0
    for i in range(i_half):
        r = float(r_grid[i]); rho_r = float(rho_1d[i])
        if rho_r < 1e-20 or r < 1e-12:
            continue
        f_r = float(Ne_inv(2-float(Ne[i]))); d = r+f_r
        if d < 1e-12 or f_r < 1e-12:
            continue
        w = 4*pi*r**2*rho_r*dr; Win_sum += w/d
        lb = (r**2+f_r**2)/(r*f_r*d**3)
        if lb <= 0:
            continue
        wb = np.sqrt(lb); rho_f = float(rho_i(f_r))
        if rho_f < 1e-30:
            continue
        p = (r/f_r)**2*rho_r/rho_f
        if p < 1e-12:
            continue
        ls = 2*(p**2+1)/(p*d**3); ws = np.sqrt(max(ls, 0))
        Wp_sum += w*(2*wb+ws); n_pts += 1
    return 0.25*Wp_sum, Win_sum, n_pts


# ═══════════════════════════════════════════════════════════════════════════
#  SPL / gISI2 MODELS
# ═══════════════════════════════════════════════════════════════════════════

def _spl_corr(W0, gl2, W_inf):
    if np.isnan(W0) or np.isnan(gl2) or np.isnan(W_inf):
        return np.nan
    if abs(gl2) < 1e-12 or abs(W_inf - W0) < 1e-12:
        return 0.0
    chi = 2.0 * gl2 / (W_inf - W0)
    arg = 1.0 + 2.0 * chi
    if arg <= 0.0:
        return np.nan
    return (W0 - W_inf) * ((np.sqrt(arg) - 1.0 - chi) / chi)


def _gisi2_corr(W0, gl2, W_inf, Wp_inf):
    if np.isnan(W0) or np.isnan(gl2) or np.isnan(W_inf) or np.isnan(Wp_inf):
        return np.nan
    if abs(gl2) < 1e-12 or abs(Wp_inf) < 1e-30:
        return 0.0
    if abs(W0) < 1e-30 or abs(W_inf) < 1e-30:
        return np.nan

    def _integrand(lam):
        d = 3.5
        b = (W0 - W_inf) * (1.0 + d)
        c = b**2 / (4.0 * Wp_inf**2)
        l1, l2 = 10.65, 3.6
        W0p = 2.0 * gl2
        sq = np.sqrt(1.0 + c * lam)
        uISI = (W_inf + b * (2.0 + c * lam + 2.0 * d * sq) /
                (2.0 * sq * (d + sq) ** 2))
        p = W0p / W0 if abs(W0) > 1e-30 else 0.0
        pl = p * lam
        a1 = W0
        a2 = -(uISI - W0) / pl if abs(pl) > 1e-30 else 0.0
        r1 = l1 * (W0 / W_inf) ** 3 if abs(W_inf) > 1e-30 else 0.0
        r2 = l2 * (W0 / W_inf) ** 3 if abs(W_inf) > 1e-30 else 0.0
        Wa = (a1 * pl) / (1.0 + r1 * pl) ** 3
        Wb = (a2 * pl) / (1.0 + r2 * pl) ** 3
        return uISI + Wa + Wb - W0

    try:
        val, _ = integrate.quad(_integrand, 0.0, 1.0, limit=300,
                                epsabs=1e-10, epsrel=1e-10)
        return float(val)
    except Exception:
        return np.nan


# ── lambda=1 ENDPOINT VALUES  W_lambda(1)  (density-based / 7.py construction) ─
#
# The functions above return the CORRELATION INTEGRAL  E_c = int_0^1 (W_l - W0) dl,
# used in the standard post-HF form  E = E_HF + E_c.
#
# The two functions below instead return the AC integrand AT lambda = 1, i.e.
# W_1^model = W_{lambda=1}^model — the model's prediction of the *physical*
# (lambda=1) XC potential energy  V_ee - J.  These pair with the density-based
# total energy
#
#     E = T[FCI] + V_ne[FCI] + J[FCI] + V_nn + W_1^model ,
#
# exactly as AC1/AC6 are assembled in the reference 7.py script. With the exact
# endpoint W_1 = W1_exact = V_ee - J this construction returns E_FCI identically
# (the "AC1 = FCI" self-consistency), because T + V_ne + J + V_nn + (V_ee - J)
# = T + V_ne + V_ee + V_nn = E_FCI.

def _spl_W1(W0, gl2, W_inf):
    """SPL adiabatic-connection integrand evaluated at lambda = 1."""
    if np.isnan(W0) or np.isnan(gl2) or np.isnan(W_inf):
        return np.nan
    if abs(W_inf - W0) < 1e-12:
        return W0
    chi = 2.0 * gl2 / (W_inf - W0)
    arg = 1.0 + 2.0 * chi
    if arg <= 0.0:
        return np.nan
    return W_inf + (W0 - W_inf) / np.sqrt(arg)


def _gisi2_W1(W0, gl2, W_inf, Wp_inf):
    """gISI2 adiabatic-connection integrand evaluated at lambda = 1
    (the same u^ISI + W_a + W_b used inside _gisi2_corr, taken at lambda = 1)."""
    if np.isnan(W0) or np.isnan(gl2) or np.isnan(W_inf) or np.isnan(Wp_inf):
        return np.nan
    if abs(Wp_inf) < 1e-30 or abs(W0) < 1e-30 or abs(W_inf) < 1e-30:
        return np.nan
    lam = 1.0
    d = 3.5
    b = (W0 - W_inf) * (1.0 + d)
    c = b**2 / (4.0 * Wp_inf**2)
    l1, l2 = 10.65, 3.6
    W0p = 2.0 * gl2
    sq = np.sqrt(1.0 + c * lam)
    uISI = (W_inf + b * (2.0 + c * lam + 2.0 * d * sq) /
            (2.0 * sq * (d + sq) ** 2))
    p = W0p / W0 if abs(W0) > 1e-30 else 0.0
    pl = p * lam
    a2 = -(uISI - W0) / pl if abs(pl) > 1e-30 else 0.0
    r1 = l1 * (W0 / W_inf) ** 3 if abs(W_inf) > 1e-30 else 0.0
    r2 = l2 * (W0 / W_inf) ** 3 if abs(W_inf) > 1e-30 else 0.0
    Wa = (W0 * pl) / (1.0 + r1 * pl) ** 3
    Wb = (a2 * pl) / (1.0 + r2 * pl) ** 3
    return uISI + Wa + Wb


# ═══════════════════════════════════════════════════════════════════════════
#  WY INVERSION + GL2
# ═══════════════════════════════════════════════════════════════════════════

def _rho_l1_diff(mol, dm1, dm2, grid_level=9, blksize=GRID_BLKSIZE):
    """Integrated |rho_KS - rho_target| (L1). The validated Ne/CO benchmark
    used this criterion at 1e-3; the previous L2 form with a 0.05 threshold
    admitted inversions far too poor to trust.

    Batched over the grid for the same memory reason as _build_grid_and_rho."""
    grids = dft.gen_grid.Grids(mol); grids.level = grid_level; grids.build()
    coords = np.asarray(grids.coords); w = np.asarray(grids.weights)
    total = 0.0
    for p0 in range(0, coords.shape[0], blksize):
        sl = slice(p0, p0+blksize)
        ao = dft.numint.eval_ao(mol, coords[sl])
        r1 = np.einsum('gi,ij,gj->g', ao, dm1, ao)
        r2 = np.einsum('gi,ij,gj->g', ao, dm2, ao)
        total += float(np.dot(w[sl], np.abs(r1-r2)))
        ao = None
    return total


def run_wy_inversion(mol, dm_target, rho_l1_tol=WY_RHO_L1_TOL,
                     strict=WY_STRICT, grid_level=9, verbose=False):
    """Run the Wu-Yang inversion and RECORD its fidelity.

    With strict = False the L1 density error is reported and stored in
    mw.rho_ok but the inversion is NOT marked failed, so the caller still
    gets orbitals to build GL2 from. strict = True reproduces the old
    behaviour where a large L1 error blanked the whole GL2 block."""
    if not KSPIES_AVAILABLE:
        class _Fake:
            converged = False
            rho_ok    = False
        return _Fake(), np.inf
    # NOTE: the overlap is deliberately NOT regularized here — WY runs against
    # been removed.)
    mw = wy_module.RWY(mol, dm_target); mw.guide = 'faxc'; mw.tol = WY_TOL; mw.eta= WY_ETA;
    mw.method = 'trust-exact'; mw.max_cycle = WY_MAXITER
    try:
        mw.run()
    except Exception as e:
        if verbose:
            print(f"      WY exception: {e}")
        mw.converged = False
    rho_diff = (_rho_l1_diff(mol, dm_target, mw.dm, grid_level=grid_level)
                if hasattr(mw, 'dm') else np.inf)
    mw.rho_ok = bool(np.isfinite(rho_diff) and rho_diff <= rho_l1_tol)
    if not mw.rho_ok:
        print(f"      WY density L1 error={rho_diff:.3e} > {rho_l1_tol:.0e}"
              f" → {'marking inversion failed' if strict else 'KEEPING (WY_STRICT=False)'}")
        if strict:
            mw.converged = False
    return mw, rho_diff


def run_pyoep_inversion(mol, dm_target, e_ref, system_basis,
                        rho_l1_tol=PYOEP_RHO_L1_TOL, grid_level=9,
                        verbose=False):
    """Run an independent closed-shell PyOEP/KSINV calculation.

    The PyOEP reference SCF, molecule, target array, potential basis and
    inversion object are all distinct from Wu--Yang.  The orbital calculation
    retains the requested system basis, while the local potential is expanded
    in the balanced PYOEP_OEP_BASIS auxiliary basis recommended by PyOEP.
    The returned adapter exposes the same four orbital fields used by
    :func:`compute_gl2_from_wy`.
    """
    meta = {
        'method': 'KSINV',
        'available': bool(PYOEP_AVAILABLE),
        'orbital_basis': system_basis,
        'oep_basis': PYOEP_OEP_BASIS,
        'same_system_basis': False,
        'space_sym': bool(PYOEP_SPACE_SYM),
        'maxiter': int(PYOEP_MAXITER),
        'conv_thr': float(PYOEP_CONV_THR),
        'thr_fai_oep': float(PYOEP_THR_FAI),
        'error': _PYOEP_IMPORT_ERROR,
    }
    if not PYOEP_AVAILABLE:
        failed = SimpleNamespace(converged=False, rho_ok=False,
                                 error=_PYOEP_IMPORT_ERROR)
        return failed, np.inf, meta
    if mol.spin != 0:
        meta['error'] = 'star.katrovich supports PyOEP KSINV only for spin=0'
        failed = SimpleNamespace(converged=False, rho_ok=False,
                                 error=meta['error'])
        return failed, np.inf, meta

    try:
        pmol = mol.copy()
        pmol.symmetry = False
        pmol.verbose = 0
        mf = scf.RHF(pmol)
        mf.verbose = 0
        mf.conv_tol = 1e-5
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('PyOEP reference HF did not converge')

        # Keep the large orbital basis for the density/wavefunction, but use a
        # distinct balanced auxiliary basis to represent the OEP potential.
        # Reusing aug-cc-pV5Z for both roles is prone to ill-conditioned OEP
        # response matrices, especially for stretched H2.
        oep_basis = PYOEP_OEP_BASIS
        target = np.array(dm_target, dtype=np.float64, copy=True)

        # EXXOEP.init_common performs a module-level deepcopy(mf).  Completed
        # PySCF objects contain unpicklable stdout/DIIS/checkpoint handles, so
        # temporarily install a PySCF-aware copier for this constructor only.
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
            copied._opt = None
            copied.diis = None
            copied.chkfile = None
            copied.stdout = None
            copied.mol.stdout = None
            return copied

        mf_stdout = getattr(mf, 'stdout', None)
        mol_stdout = getattr(mf.mol, 'stdout', None)
        original_deepcopy = _PYOEP_EXXOEP_MODULE.deepcopy
        mf.stdout = None
        mf.mol.stdout = None
        _PYOEP_EXXOEP_MODULE.deepcopy = _independent_pyscf_copy
        try:
            inv = _KSINV(
                mf, oep_basis, target, float(e_ref),
                space_sym=PYOEP_SPACE_SYM,
            )
        finally:
            _PYOEP_EXXOEP_MODULE.deepcopy = original_deepcopy
            mf.stdout = mf_stdout
            mf.mol.stdout = mol_stdout

        inv.mf.stdout = mf_stdout if mf_stdout is not None else sys.stdout
        inv.mf.mol.stdout = (
            mol_stdout if mol_stdout is not None else inv.mf.stdout
        )
        inv.run(maxit=PYOEP_MAXITER,
                thr_fai_oep=PYOEP_THR_FAI,
                conv_thr=PYOEP_CONV_THR,
                mixing_a=0.1, mixing_i=3)

        raw_dm = np.asarray(inv.mf.make_rdm1(), dtype=np.float64)
        dm_total = raw_dm[0] + raw_dm[1] if raw_dm.ndim == 3 else raw_dm
        dm_total = 0.5 * (dm_total + dm_total.T)
        rho_diff = _rho_l1_diff(
            pmol, target, dm_total, grid_level=grid_level
        )
        converged = bool(getattr(inv, 'converged', True))
        rho_ok = bool(np.isfinite(rho_diff) and rho_diff <= rho_l1_tol)
        adapter = SimpleNamespace(
            mo_coeff=np.array(inv.mf.mo_coeff, copy=True),
            mo_occ=np.array(inv.mf.mo_occ, dtype=float, copy=True),
            mo_energy=np.array(inv.mf.mo_energy, dtype=float, copy=True),
            dm=dm_total,
            converged=converged,
            rho_ok=rho_ok,
            inversion=inv,
            error='' if converged else 'KS inversion did not converge',
        )
        meta['error'] = adapter.error
        if verbose:
            print(f"      PyOEP/KSINV rho_L1={rho_diff:.3e}, "
                  f"converged={converged}, rho_ok={rho_ok}")
        return adapter, float(rho_diff), meta
    except Exception as exc:
        meta['error'] = f'{type(exc).__name__}: {exc}'
        print(f"      PyOEP/KSINV FAILED independently: {meta['error']}")
        failed = SimpleNamespace(converged=False, rho_ok=False,
                                 error=meta['error'])
        return failed, np.inf, meta


def _wy_has_finite_data(mw):
    """Check whether a WY object contains enough finite data to evaluate GL2.

    Optimizer convergence and density fidelity are deliberately not tested
    here.  They remain independent quality diagnostics.  This function only
    answers whether a numerical GL2 evaluation can be attempted.
    """
    required = ('mo_coeff', 'mo_occ', 'mo_energy', 'dm')
    missing = [name for name in required if not hasattr(mw, name)]
    if missing:
        return False, f"missing WY fields: {','.join(missing)}"
    try:
        coeff = np.asarray(mw.mo_coeff)
        occ = np.asarray(mw.mo_occ)
        energy = np.asarray(mw.mo_energy)
        dm = np.asarray(mw.dm)
    except Exception as exc:
        return False, f"unreadable WY arrays: {exc}"

    if coeff.ndim != 2 or dm.ndim != 2 or occ.ndim != 1 or energy.ndim != 1:
        return False, "invalid WY array dimensions"
    if coeff.shape[1] != occ.size or occ.size != energy.size:
        return False, "inconsistent WY orbital dimensions"
    if dm.shape != (coeff.shape[0], coeff.shape[0]):
        return False, "inconsistent WY density-matrix dimensions"
    if not all(np.all(np.isfinite(a)) for a in (coeff, occ, energy, dm)):
        return False, "non-finite WY orbital/density data"
    return True, "finite WY orbital/density data"


def compute_gl2_from_wy(mol, mw, gap_tol=GL2_GAP_TOL, strict=GL2_STRICT):
    """
    GL2 correlation energy for a two-electron singlet, evaluated directly on
    the converged Wu-Yang KS orbitals.

    This REPLACES the previous mp.MP2(mf_c) approach, which

      (1) returned only the doubles-like term (the GL2 singles term was
          silently dropped), and
      (2) under linear dependency re-projected/truncated the WY orbitals onto
          the RHF-retained subspace and then paired those non-canonical
          columns with the original WY orbital energies -- corrupting the
          denominators.

    Both doubles and singles are formed explicitly on the WY orbitals:

        E_doubles = sum_{iajb} (ia|jb)[2(ia|jb)-(ib|ja)] / (e_i+e_j-e_a-e_b)
        E_singles = 2 sum_{ia} |<i| Sigma_x - v_x |a>|^2 / (e_i - e_a)
        Sigma_x   = -K[D]/2 ,  v_x = -J[D]/2      (TWO-ELECTRON SINGLET ONLY --
        the v_x = -J/2 identity is exact for N=2 and wrong for anything else;
        the leading factor of 2 is the alpha/beta spin sum.)

    *** This function returns ONLY the two components (e_doubles, e_singles).
        Which scalar becomes the master "gl2" is decided by the scenario in
        merge_row(), so the selection lives in exactly one place. ***

    For a two-electron singlet with an EXACT inversion, Sigma_x and v_x act
    identically on the single occupied orbital, so E_singles -> 0. With a
    real WY inversion it is the (small, nonzero) residual measuring how far
    the inverted local v_x sits from the exact -J/2, i.e. a probe of the
    exchange-potential inversion quality.

    strict=True  : a vanishing/inverted KS gap or any non-negative denominator
                   returns (nan, nan) -- the old all-or-nothing behaviour.
    strict=False : denominators are FLOORED with their sign preserved, so the
                   pathology shows up as a huge (or wrong-signed) GL2 that can
                   be inspected, rather than a discarded row.

    Returns (e_doubles, e_singles). Only a structurally impossible case (no
    occupied or no virtual orbitals, non-finite eigenvalues) gives nan.
    """
    _NAN2 = (np.nan, np.nan)

    def _floor_signed(d, tol):
        """Push |d| off zero without changing its sign."""
        s = np.where(d >= 0.0, 1.0, -1.0)
        return np.where(np.abs(d) < tol, s * tol, d)

    C     = np.asarray(mw.mo_coeff)
    occ   = np.asarray(mw.mo_occ,    dtype=float)
    eps   = np.asarray(mw.mo_energy, dtype=float)
    dm_ks = np.asarray(mw.dm)

    if not np.all(np.isfinite(eps)):
        return _NAN2

    occ_mask = occ > 1.0
    vir_mask = occ < 1e-8
    if int(np.count_nonzero(occ_mask)) == 0 or not np.any(vir_mask):
        return _NAN2
    # More than one doubly occupied spatial orbital invalidates the N=2
    # singles identity (v_x = -J/2). Rejected only in strict mode.
    if strict and int(np.count_nonzero(occ_mask)) != 1:
        return _NAN2

    c_occ, c_vir = C[:, occ_mask], C[:, vir_mask]
    e_occ, e_vir = eps[occ_mask], eps[vir_mask]
    nocc, nvir   = c_occ.shape[1], c_vir.shape[1]

    gap = float(np.min(e_vir) - np.max(e_occ))
    if not np.isfinite(gap):
        return _NAN2
    if gap <= gap_tol:
        print(f"      ⚠ WY KS gap = {gap:.3e} ≤ {gap_tol:.0e}"
              f"{' → GL2 rejected' if strict else ' → GL2 computed anyway'}")
        if strict:
            return _NAN2

    # ── GL2 doubles ──────────────────────────────────────────────────────
    eri_iajb = ao2mo.general(mol, (c_occ, c_vir, c_occ, c_vir),
                             compact=False).reshape(nocc, nvir, nocc, nvir)
    eri_ibja = eri_iajb.transpose(0, 3, 2, 1)
    d_denom  = (e_occ[:, None, None, None] + e_occ[None, None, :, None]
                - e_vir[None, :, None, None] - e_vir[None, None, None, :])
    if np.any(d_denom >= -2.0 * gap_tol):
        if strict:
            return _NAN2
        d_denom = _floor_signed(d_denom, 2.0 * gap_tol)
    e_doubles = float(np.sum(eri_iajb * (2.0 * eri_iajb - eri_ibja) / d_denom))

    # ── GL2 singles (≈0 for a two-electron singlet) ──────────────────────
    j_ao, k_ao = scf.hf.get_jk(mol, dm_ks)
    dv_ao      = (-0.5 * k_ao) - (-0.5 * j_ao)          # Sigma_x - v_x
    dv_ov      = (C.T @ dv_ao @ C)[np.ix_(occ_mask, vir_mask)]
    s_denom    = e_occ[:, None] - e_vir[None, :]
    if np.any(s_denom >= -gap_tol):
        if strict:
            return _NAN2
        s_denom = _floor_signed(s_denom, gap_tol)
    e_singles = SINGLES_SPIN_FACTOR * float(np.sum(np.abs(dv_ov) ** 2 / s_denom))

    if strict and not (np.isfinite(e_doubles) and np.isfinite(e_singles)):
        return _NAN2
    return e_doubles, e_singles


# ═══════════════════════════════════════════════════════════════════════════
#  STARTUP BENCHMARK  —  H2 / decontracted aug-cc-pV5Z (Cartesian, nAO=216)
#                        at R = 1.4 Bohr
#
#  No published reference data exist for this system; validation is against
#  the EXACT two-electron identities:
#
#    (i)  v_x = -J/2 holds exactly for an N=2 singlet, so Sigma_x and v_x act
#         identically on the single occupied orbital and the GL2/SC SINGLES
#         must VANISH.
#    (ii) a faithful inversion returns the HF occupied orbital, so
#         semicanonicalising in the virtual space recovers the canonical HF
#         virtuals and the SC DOUBLES must equal MP2.
#    (iii) the GL2 doubles must NOT equal MP2 -- the KS gap is smaller than
#         the HF gap, so |GL2| > |MP2|. This is physics, not error.
#
#  Reference values (R = 1.4 a0):
#      E_HF              -1.1336167587
#      GL2 Single        ~0            (published table quoted -5.4e-09;
#      SC  Single        ~0             a cleaner inversion gives ~1e-17)
#      GL2 Double AA+AB  -0.0497921079 .. -0.0497924963
#      SC  Double AA+AB  -0.0338275378 .. -0.0338278968
#      MP2 (reference)   -0.0338278969
#      GL2 Double - MP2  -0.01596
#
#  Self-contained Wu-Yang inversion, so the check runs whether or not kspies
#  is installed. Skip with --skip-benchmark.
# ═══════════════════════════════════════════════════════════════════════════

# IMPORTANT — how tight can these tolerances legitimately be?
# The decontracted aug-cc-pV5Z overlap has min eigenvalue 5.3e-08, so the
# near-null space is resolved differently by different LAPACK/BLAS builds.
# The SAME molecule gives E_HF = -1.1336155273 on one machine and
# -1.1336167587 on another (1.2e-06 apart), independent of conv_tol. Absolute
# energies are therefore only reproducible to ~1e-06 here, and tolerances
# tighter than that would fail for reasons that have nothing to do with the
# implementation. The physically meaningful checks are the IDENTITIES
# (singles vanish; SC doubles == MP2), which are evaluated within a single
# run and so are unaffected by this.
BENCH_REF = dict(
    E_HF        = -1.1336161,      # machine-dependent at the 1e-6 level
    GL2_d       = -0.0497923,      # midpoint of the two reported runs
    SC_d        = -0.0338277,
    MP2         = -0.0338271,
    tol_energy  = 5.0e-5,          # >> the 1e-6 cross-machine spread
    tol_singles = 1.0e-6,          # singles must vanish (identity)
    tol_sc_mp2  = 1.0e-5,          # SC doubles == MP2 (identity, same run)
)


def _bench_decontract(basname, symbols):
    return {x: gto.uncontract(gto.basis.load(basname, x)) for x in symbols}


def _bench_build_Vt(mol, level, pot_basis):
    """3-centre integrals <mu|g_t|nu>. An einsum('gm,g,gt,gn->tmn') picks a
    pathological contraction path here (unusable at nAO=216); the explicit
    loop is the same arithmetic through BLAS."""
    g = dft.gen_grid.Grids(mol); g.level = level; g.build()
    ao = dft.numint.eval_ao(mol, g.coords)
    symbols = sorted({mol.atom_symbol(i) for i in range(mol.natm)})
    molp = gto.M(atom=mol._atom, basis=_bench_decontract(pot_basis, symbols),
                 unit='Bohr', cart=mol.cart, verbose=0)
    aop = dft.numint.eval_ao(molp, g.coords)
    w = g.weights
    Vt = np.empty((aop.shape[1], ao.shape[1], ao.shape[1]))
    for t in range(aop.shape[1]):
        Vt[t] = ao.T @ (ao * (w * aop[:, t])[:, None])
    return Vt


def _bench_wy(mol, dm_tgt, grid_level=9, pot_basis='aug-cc-pvtz', maxiter=4000):
    """Maximise the concave Wu-Yang functional
         W(b) = 2*sum_i eps_i - Tr[V_s(b) D_tgt],
       dW/db_t = Tr[V_t (D_KS - D_tgt)],  Fermi-Amaldi guide."""
    S = mol.intor('int1e_ovlp')
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    Vne = mol.intor('int1e_nuc')
    N = mol.nelectron; nocc = N // 2
    J = scf.hf.get_jk(mol, dm_tgt)[0]
    Vguide = -(1.0 / N) * J
    base = hcore + J + Vguide
    const = Vne + J + Vguide
    Vt = _bench_build_Vt(mol, grid_level, pot_basis)

    def negW(b):
        F = base + np.einsum('t,tmn->mn', b, Vt)
        eps, C = scf.hf.eig(F, S)
        Dks = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        W = 2.0 * np.sum(eps[:nocc]) - np.einsum(
            'mn,mn->', const + np.einsum('t,tmn->mn', b, Vt), dm_tgt)
        return -W, -np.einsum('tmn,mn->t', Vt, Dks - dm_tgt)

    res = optimize.minimize(negW, np.zeros(Vt.shape[0]), jac=True,
                            method='L-BFGS-B',
                            options={'maxiter': maxiter, 'ftol': 1e-16,
                                     'gtol': 1e-10})
    F = base + np.einsum('t,tmn->mn', res.x, Vt)
    eps, C = scf.hf.eig(F, S)
    return np.asarray(C), np.asarray(eps)


def _bench_gl2_sc(mol, C, eps, nocc):
    """GL2 (canonical KS eigenvalues) and SC (semicanonical) singles+doubles.
       Uses <i|Sigma_x - v_x|a> = F_ia, F = h + J[D] - K[D]/2, valid for any N."""
    Co, Cv = C[:, :nocc], C[:, nocc:]
    eo, ev = eps[:nocc], eps[nocc:]
    no, nv = nocc, Cv.shape[1]
    D = 2.0 * Co @ Co.T
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    J, K = scf.hf.get_jk(mol, D)
    Fmo = C.T @ (h + J - 0.5 * K) @ C

    def _sing(Fov, a, b):
        return SINGLES_SPIN_FACTOR * float(
            np.sum(np.abs(Fov) ** 2 / (a[:, None] - b[None, :])))

    def _doub(Cq, Cw, a, b):
        e = ao2mo.general(mol, (Cq, Cw, Cq, Cw),
                          compact=False).reshape(no, nv, no, nv)
        d = (a[:, None, None, None] + a[None, None, :, None]
             - b[None, :, None, None] - b[None, None, None, :])
        return float(np.sum(e * (2.0 * e - e.transpose(0, 3, 2, 1)) / d))

    gl2_s, gl2_d = _sing(Fmo[:no, no:], eo, ev), _doub(Co, Cv, eo, ev)
    eo2, Uo = np.linalg.eigh(Fmo[:no, :no])
    ev2, Uv = np.linalg.eigh(Fmo[no:, no:])
    sc_s = _sing(Uo.T @ Fmo[:no, no:] @ Uv, eo2, ev2)
    sc_d = _doub(Co @ Uo, Cv @ Uv, eo2, ev2)
    return gl2_s, gl2_d, sc_s, sc_d


def run_startup_benchmark(R=1.4, basis='aug-cc-pv5z', pot_basis='aug-cc-pvtz',
                          grid_level=9, verbose=True):
    """Returns (all_passed, results_dict). Prints a PASS/FAIL table."""
    t0 = time.time()
    print("\n" + "=" * 78)
    print("  STARTUP BENCHMARK — H2 / decontracted aug-cc-pV5Z, R = 1.4 Bohr")
    print("  Validating against the EXACT N=2 identities before production.")
    print("=" * 78)

    bas = _bench_decontract(basis, ['H'])
    mol = gto.M(atom=f'H 0 0 {-R/2:.10f}; H 0 0 {R/2:.10f}', unit='Bohr',
                basis=bas, cart=True, verbose=0)
    print(f"  nAO = {mol.nao_nr()}   (Cartesian, decontracted)")

    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.max_cycle = 200; mf.kernel()
    e_hf = float(mf.e_tot)
    dm_tgt = mf.make_rdm1()
    e_mp2 = float(mp.MP2(mf).kernel()[0])
    print(f"  E_HF  = {e_hf:.10f}     MP2 = {e_mp2:.10f}   ({time.time()-t0:.1f}s)")

    print("  Running self-contained Wu-Yang inversion …")
    C, eps = _bench_wy(mol, dm_tgt, grid_level=grid_level, pot_basis=pot_basis)
    gl2_s, gl2_d, sc_s, sc_d = _bench_gl2_sc(mol, C, eps, nocc=mol.nelectron // 2)

    res = dict(E_HF=e_hf, MP2=e_mp2, GL2_s=gl2_s, GL2_d=gl2_d,
               SC_s=sc_s, SC_d=sc_d)

    checks = [
        ("E_HF vs reference",
         abs(e_hf - BENCH_REF['E_HF']) < BENCH_REF['tol_energy'],
         f"{e_hf:.10f} vs {BENCH_REF['E_HF']:.7f}"),
        ("GL2 singles vanish (identity)",
         abs(gl2_s) < BENCH_REF['tol_singles'], f"{gl2_s:.3e}"),
        ("SC  singles vanish (identity)",
         abs(sc_s) < BENCH_REF['tol_singles'], f"{sc_s:.3e}"),
        ("SC doubles == MP2 (identity, same run)",
         abs(sc_d - e_mp2) < BENCH_REF['tol_sc_mp2'],
         f"{sc_d:.10f} vs {e_mp2:.10f}  Δ={sc_d-e_mp2:.2e}"),
        ("GL2 doubles vs reference",
         abs(gl2_d - BENCH_REF['GL2_d']) < BENCH_REF['tol_energy'],
         f"{gl2_d:.10f} vs {BENCH_REF['GL2_d']:.7f}"),
        ("|GL2 doubles| > |MP2|  (smaller KS gap — physics)",
         abs(gl2_d) > abs(e_mp2), f"{gl2_d:.6f} vs {e_mp2:.6f}"),
    ]

    print("-" * 78)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<42s} {detail}")
    print("-" * 78)
    print(f"  Benchmark {'PASSED' if all_ok else 'FAILED'} in {time.time()-t0:.1f}s")
    print("=" * 78 + "\n")
    return all_ok, res


# ═══════════════════════════════════════════════════════════════════════════
#  MOL BUILDER + HF
# ═══════════════════════════════════════════════════════════════════════════

def make_mol(atom_str, basis_name, spin, charge, unit=R_UNIT):
    """NOTE: unit is now EXPLICIT. Previously omitted, so PySCF defaulted
    to Angstrom while everything on the SCE side assumed Bohr."""
    mol = gto.M(atom=atom_str, spin=spin, charge=charge, basis=basis_name,
                unit=unit, cart=False, symmetry=False, verbose=0)
    mol.build()
    return mol


def _run_hf(mol, spin, omega):
    mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf.max_cycle = 400; mf.conv_tol = 1e-10; mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        # Stretched H2 in a big diffuse basis is a classic SCF headache.
        mf = scf.addons.remove_linear_dep_(mf)
        mf = mf.newton(); mf.max_cycle = 200; mf.kernel()
    return mf


# ═══════════════════════════════════════════════════════════════════════════
#  DENSITY GETTERS
# ═══════════════════════════════════════════════════════════════════════════

def _norm_rdm(rdm):
    rdm = np.asarray(rdm)
    return rdm[0]+rdm[1] if rdm.ndim == 3 and rdm.shape[0] == 2 else rdm


def get_dm_hf(mf, verbose=False):
    """Return the HF total 1-RDM in AO/MO representations.

    For the present closed-shell H2 calculations, ``mf.mo_occ`` is the
    spin-summed diagonal MO-basis 1-RDM.  No FCI object enters this path.
    ``dm2_mo`` is returned as None; _pass1_single() evaluates the two-electron
    HF endpoint analytically for the two-electron singlet.
    """
    if np.asarray(mf.mo_coeff).ndim != 2:
        raise NotImplementedError(
            "The current H2 HF-density path expects a restricted, "
            "closed-shell reference."
        )

    dm1_ao = _norm_rdm(mf.make_rdm1())
    dm1_mo = np.diag(np.asarray(mf.mo_occ, dtype=float))
    E_hf = float(mf.e_tot)

    if verbose:
        print(f"    HF  E_tot = {E_hf:.10f}")
    return dm1_ao, dm1_mo, None, E_hf


def get_dm_fci(mf, verbose=False):
    """Return the FCI 1-/2-RDMs and FCI total energy.

    The FCI density is transformed to AO form only after the correlated
    MO-basis RDM has been constructed.
    """
    norb = mf.mo_coeff.shape[1]; nelec = mf.mol.nelec
    myfci = fci.FCI(mf); myfci.max_memory = 4000; myfci.verbose = 0
    vals, vecs = myfci.kernel(nroots=N_ROOTS)
    E_fci = float(np.atleast_1d(vals)[0]); ci_gs = vecs[0] if N_ROOTS > 1 else vecs
    dm1_mo = _norm_rdm(myfci.make_rdm1(ci_gs, norb=norb, nelec=nelec))
    dm2_mo = _norm_rdm(myfci.make_rdm2(ci_gs, norb=norb, nelec=nelec))
    dm1_ao = mf.mo_coeff@dm1_mo@mf.mo_coeff.T
    if verbose:
        print(f"    FCI E_tot = {E_fci:.10f}")
    return dm1_ao, dm1_mo, dm2_mo, E_fci


# ═══════════════════════════════════════════════════════════════════════════
#  PASS 1  ─  W1_exact + DFT W_inf MODELS + WY/GL2
# ═══════════════════════════════════════════════════════════════════════════

_CYL_KEYS = [
    'ts_cyl', 'vext_cyl', 'vnn_cyl', 'hartree_cyl',
    'vee_sce_cyl', 'winf_cyl', 'winf_prime_rmzpe_cyl',
    'total_sce_cyl', 'total_sce_rmzpe_cyl', 'levy_zahariev_c_cyl',
    'inner_z1', 'inner_h1', 'inner_z2', 'inner_h2',
]


def _pass1_single(mol, dm1_ao, dm1_mo, dm2_mo, mo_coeff,
                  spin, density_type, label, basis_label,
                  E_ref, E_hf, omega, cfg, dm_reference, inversion_obj,
                  inversion_rho_diff, inversion_method,
                  inversion_meta=None, verbose=False):
    """SINGLE definition (the file previously carried two).

    Stores gl2_doubles and gl2_singles SEPARATELY and does NOT pick a master
    scalar; that choice belongs to merge_row()."""
    # Density-derived quantities use the requested HF/FCI reference density.
    # The independent inversion object supplies only KS orbitals/eigenvalues
    # (and its own KS density inside compute_gl2_from_wy) for GL2.
    dm1_2d = np.asarray(dm_reference, dtype=np.float64)
    nmo = mo_coeff.shape[1]
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), nmo)
    # dm1_mo belongs to the density_type currently being processed.  The
    # common HF molecular-orbital basis is only a representation for the
    # contractions; it does not change an FCI RDM into an HF density.
    eri_ao = ao2mo.restore(1, mol.intor('int2e'), mol.nao_nr())
    J_H = 0.5*float(np.einsum('pqrs,pq,rs->', eri_ao, dm1_2d, dm1_2d))

    # Exact exchange of a two-electron singlet evaluated on THIS row's
    # density.  Keep separate named HF/FCI columns to prevent cross-use.
    E_xx_density = -J_H/2.0
    E_xx_HF = E_xx_density if density_type == 'HF' else np.nan
    E_xx_FCI = E_xx_density if density_type == 'FCI' else np.nan
    W0 = E_xx_density

    if dm2_mo is not None:
        Vee = 0.5*float(np.einsum('pqrs,pqrs->', eri_mo, dm2_mo))
        W1_exact = Vee-J_H
    elif density_type == 'HF':
        # For closed-shell two-electron H2, the determinant endpoint is
        # Vee = J + Ex and therefore W1_exact = Vee-J = Ex = W0.
        W1_exact = E_xx_density
        Vee = J_H + W1_exact
    else:
        Vee = W1_exact = np.nan
    T_mat = mol.intor('int1e_kin'); Vh = V_harm_system(mol, omega)
    V_nuc = mol.intor('int1e_nuc')
    E_kin = float(np.einsum('ij,ij->', T_mat, dm1_2d))
    E_ext = float(np.einsum('ij,ij->', Vh, dm1_2d))
    # Nuclear-attraction (electron-nucleus) energy on the correlated 1-RDM.
    # Needed for the density-based total energy E = (T + V_ne + J + V_nn) + E_xc
    # used in merge_row (mirrors the 7.py AC1/AC6 construction). E_ext above is
    # the harmonic external potential and is zero for molecular H2 (omega=0),
    # so it is NOT the electron-nucleus term.
    E_vne = float(np.einsum('ij,ij->', V_nuc, dm1_2d))

    dft_winf: Dict = {}
    if cfg['compute_dft_models']:
        if verbose:
            print(f"    {density_type}/{inversion_method}: "
                  "DFT W_inf/Wp_inf models …")
        # compute_W_for_density() reads mf_int.mol / mf_int.grids / mf_int._numint,
        # so it needs an actual dft.RKS/UKS object with grids attached -- not the
        # bare gto.Mole. Passing mol directly (as before) raised AttributeError on
        # every call, silently caught below and recorded as NaN for every model.
        mf_int = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf_int.grids.level = 9
        mf_int.grids.prune = None
        for model in DFT_MODELS:
            try:
                xctype = model_xctype(model)
                W = compute_W_for_density(
                    mf_int, dm1_2d, make_exc_func(model, "W"), xctype
                )
                Wp = compute_W_for_density(
                    mf_int, dm1_2d, make_exc_func(model, "Wp"), xctype
                )
            except Exception as e:
                if verbose:
                    print(f"      {model}: {e}")
                W = Wp = np.nan
            dft_winf[f'W_inf_{model}']  = W
            dft_winf[f'Wp_inf_{model}'] = Wp
            if verbose:
                print(f"      {model:6s}: W={W:.6f}  Wp={Wp:.6f}")
    else:
        for model in DFT_MODELS:
            dft_winf[f'W_inf_{model}']  = np.nan
            dft_winf[f'Wp_inf_{model}'] = np.nan

    # ── method-specific inversion diagnostics + GL2 components ──────────
    gl2_doubles = np.nan
    gl2_singles = np.nan
    inversion_ok = bool(getattr(inversion_obj, 'converged', False))
    inversion_rho_ok = bool(getattr(inversion_obj, 'rho_ok', False))
    inversion_rho_diff = float(inversion_rho_diff)
    inversion_data_used = False
    inversion_status = 'not attempted (open-shell density)'
    ks_gap = np.nan

    if spin == 0:
        usable, data_reason = _wy_has_finite_data(inversion_obj)
        if usable:
            _occ = np.asarray(inversion_obj.mo_occ, dtype=float)
            _eps = np.asarray(inversion_obj.mo_energy, dtype=float)
            _eo = _eps[_occ > 1.0]
            _ev = _eps[_occ < 1.0e-8]
            if _eo.size and _ev.size:
                ks_gap = float(np.min(_ev) - np.max(_eo))

        # The run loop admits a row only when the method has finite orbital
        # data.  Strict mode additionally makes convergence a GL2 gate.
        method_strict = (cfg['wy_strict'] if inversion_method == 'WY'
                         else cfg['pyoep_strict'])
        allow_values = usable and (inversion_ok or not method_strict)
        if allow_values:
            try:
                gl2_doubles, gl2_singles = compute_gl2_from_wy(
                    mol, inversion_obj, gap_tol=GL2_GAP_TOL,
                    strict=cfg['gl2_strict'])
                inversion_data_used = bool(
                    np.isfinite(gl2_doubles) and np.isfinite(gl2_singles)
                )
                quality = []
                if not inversion_ok:
                    quality.append('optimizer_not_converged')
                if not inversion_rho_ok:
                    tol = (cfg['wy_rho_l1_tol'] if inversion_method == 'WY'
                           else cfg['pyoep_rho_l1_tol'])
                    quality.append(
                        f'rho_L1_above_{tol:.1e}'
                    )
                if inversion_data_used:
                    inversion_status = (
                        'values kept'
                        + (f' ({"; ".join(quality)})' if quality else
                           f' (all {inversion_method} diagnostics passed)')
                    )
                else:
                    inversion_status = (
                        'GL2 returned non-finite components despite finite '
                        'inversion orbital data'
                    )
                if verbose:
                    print(f"      GL2 doubles = {gl2_doubles:.10f}   "
                          f"singles = {gl2_singles:.3e}")
                    print(f"      {inversion_method} status: "
                          f"{inversion_status}")
            except Exception as e:
                inversion_status = (
                    f'GL2 evaluation failed: {type(e).__name__}: {e}'
                )
                if verbose:
                    print(f"      GL2 failed: {e}")
                gl2_doubles = gl2_singles = np.nan
        elif method_strict and not inversion_ok:
            inversion_status = (
                f'discarded because {inversion_method} strict mode and '
                'inversion_ok=False'
            )
        else:
            inversion_status = f'cannot evaluate GL2: {data_reason}'

    gl2_total = (gl2_doubles + gl2_singles
                 if (np.isfinite(gl2_doubles) and
                     np.isfinite(gl2_singles)) else np.nan)

    return {
        'density_type': density_type, 'system': label, 'omega': omega,
        'basis': basis_label, 'spin': spin, 'charge': mol.charge,
        'inversion_method': inversion_method,
        'density_used': f'{density_type}_reference',
        'orbital_source': inversion_method,
        'E_ref': E_ref,
        # Filled consistently for both density blocks after FCI finishes:
        # E_c = E_FCI[FCI DM] - E_HF[HF DM].
        'E_c': np.nan,
        'E_hf_tot': float(E_hf),
        'E_fci_tot': np.nan,
        'E_c_definition': 'E_FCI(FCI_DM)-E_HF(HF_DM)',
        'W0': W0, 'E_xx_density': E_xx_density,
        'E_xx_HF': E_xx_HF, 'E_xx_FCI': E_xx_FCI,
        'J': J_H, 'Vee': Vee, 'W1_exact': W1_exact,
        'E_kin': E_kin, 'E_ext': E_ext, 'E_vne': E_vne, 'E_nn': float(mol.energy_nuc()),
        'inversion_ok': inversion_ok,
        'inversion_rho_ok': inversion_rho_ok,
        'inversion_data_used': inversion_data_used,
        'inversion_status': inversion_status,
        'inversion_rho_l1': inversion_rho_diff,
        'WY_ok': inversion_ok if inversion_method == 'WY' else np.nan,
        'WY_rho_ok': (inversion_rho_ok
                      if inversion_method == 'WY' else np.nan),
        'WY_data_used': (inversion_data_used
                         if inversion_method == 'WY' else np.nan),
        'WY_status': (inversion_status
                      if inversion_method == 'WY' else ''),
        'wy_rho_diff': (inversion_rho_diff
                        if inversion_method == 'WY' else np.nan),
        'PYOEP_ok': (inversion_ok
                     if inversion_method == 'PYOEP' else np.nan),
        'PYOEP_rho_ok': (inversion_rho_ok
                         if inversion_method == 'PYOEP' else np.nan),
        'PYOEP_data_used': (inversion_data_used
                            if inversion_method == 'PYOEP' else np.nan),
        'PYOEP_status': (inversion_status
                         if inversion_method == 'PYOEP' else ''),
        'pyoep_rho_diff': (inversion_rho_diff
                           if inversion_method == 'PYOEP' else np.nan),
        'pyoep_error': ((inversion_meta or {}).get('error', '')
                        if inversion_method == 'PYOEP' else ''),
        'pyoep_available': ((inversion_meta or {}).get('available', False)
                            if inversion_method == 'PYOEP' else np.nan),
        'pyoep_method': ((inversion_meta or {}).get('method', '')
                         if inversion_method == 'PYOEP' else ''),
        'pyoep_orbital_basis': ((inversion_meta or {}).get(
            'orbital_basis', '') if inversion_method == 'PYOEP' else ''),
        'pyoep_oep_basis': ((inversion_meta or {}).get('oep_basis', '')
                            if inversion_method == 'PYOEP' else ''),
        'pyoep_same_system_basis': ((inversion_meta or {}).get(
            'same_system_basis', False)
            if inversion_method == 'PYOEP' else np.nan),
        'pyoep_space_sym': ((inversion_meta or {}).get('space_sym', False)
                            if inversion_method == 'PYOEP' else np.nan),
        'pyoep_maxiter': ((inversion_meta or {}).get('maxiter', np.nan)
                          if inversion_method == 'PYOEP' else np.nan),
        'pyoep_conv_thr': ((inversion_meta or {}).get('conv_thr', np.nan)
                           if inversion_method == 'PYOEP' else np.nan),
        'pyoep_thr_fai_oep': ((inversion_meta or {}).get(
            'thr_fai_oep', np.nan)
            if inversion_method == 'PYOEP' else np.nan),
        'gl2_doubles': gl2_doubles,
        'gl2_singles': gl2_singles,
        'gl2_total': gl2_total,
        'ks_gap': ks_gap,
        **dft_winf,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PASS 2  ─  W_inf_SCE + 3D CYLINDRICAL SCE / rm-ZPE
# ═══════════════════════════════════════════════════════════════════════════

def _pass2_single(mol, dm1_ao, dm1_mo, mo_coeff,
                  spin, density_type, label, omega,
                  cfg, dm_reference, inversion_method, verbose=False):
    # SCE is a density functional: evaluate it on the target HF/FCI density,
    # never on numerical noise left by a finite-basis inversion.
    dm1_2d = np.asarray(dm_reference, dtype=np.float64)

    # U via get_jk, not a full nao^4 AO ERI (see compute_sce3d_rmzpe).
    J_ao = scf.hf.get_jk(mol, dm1_2d)[0]
    U    = 0.5*float(np.einsum('ij,ij->', J_ao, dm1_2d))

    atom_c = mol.atom_coords()
    R_bohr = float(np.linalg.norm(atom_c[1]-atom_c[0])) if mol.natm == 2 else 0.0

    r_grid = build_r_grid(r_max=1.5*R_bohr + 10.0)
    rho_1d = get_density_radial(mol, dm1_2d, r_grid)
    T_ZP, V_SCE, _ = compute_T_ZP_and_Winf_N2(rho_1d, r_grid)
    W_inf_SCE = V_SCE-U; Wp_inf_SCE = T_ZP
    if verbose:
        print(f"    {density_type}/{inversion_method}: "
              f"W_inf_SCE(radial)={W_inf_SCE:.6f}  Wp={Wp_inf_SCE:.6f}")

    cyl: Dict = {k: np.nan for k in _CYL_KEYS}
    do_3d = (cfg['compute_sce3d'] and spin == 0 and mol.natm == 2
             and (cfg['sce3d_dts'] is None or density_type in cfg['sce3d_dts']))
    if do_3d:
        try:
            if verbose:
                print(f"    {density_type}/{inversion_method}: "
                      f"3D SCE/rmZPE  R={R_bohr:.4f} Bohr …")
            cyl = compute_sce3d_rmzpe(mol, dm1_2d, R_bohr, **cfg['sce3d_cfg'])
            if verbose:
                print(f"      vee_sce_cyl          = {cyl['vee_sce_cyl']:.8f}")
                print(f"      winf_cyl             = {cyl['winf_cyl']:.8f}")
                print(f"      winf_prime_rmzpe_cyl = {cyl['winf_prime_rmzpe_cyl']:.8f}")
        except Exception as e:
            print(f"  ⚠ 3D SCE failed "
                  f"({density_type}/{inversion_method}/{label}): {e}")

    return {
        'density_type': density_type, 'inversion_method': inversion_method,
        'system': label, 'omega': omega,
        'R_bohr': R_bohr,
        'U': U, 'W_inf_SCE': W_inf_SCE, 'Wp_inf_SCE': Wp_inf_SCE,
        **cyl,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PER-SYSTEM TWO-PASS LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_system_two_pass(system_info, cfg, verbose=False):
    label, atom_str, spin, charge, basis_name, omega = system_info
    basis_label = basis_name if isinstance(basis_name, str) else f"ET_w{omega:.4g}"
    mol = make_mol(atom_str, basis_name, spin, charge, unit=cfg['unit'])
    mol.incore_anyway = True
    mf = _run_hf(mol, spin, omega)
    if not mf.converged:
        print(f"  ❌ HF not converged for {label}")
        return None, None
    E_hf = mf.e_tot
    dts = cfg['density_types']

    def _get(dt):
        """Build one density independently.

        The caller iterates in cfg['density_types'] order, so the default
        sequence is HF followed by FCI.  Each returned 1-RDM is subsequently
        passed through its own Wu--Yang inversion.
        """
        try:
            if dt == 'HF':
                return get_dm_hf(mf, verbose)
            if dt == 'FCI':
                return get_dm_fci(mf, verbose)
            print(f"  ⚠ density type {dt!r} not supported")
        except Exception as e:
            print(f"  ⚠ {dt} failed for {label}: {e}")
        return None

    # Keep a separate (density type, inversion method) row.  This guarantees
    # that a failed WY branch cannot block, seed, or replace PyOEP and that a
    # failed PyOEP branch cannot affect WY.
    pass1_rows, pass2_rows = {}, {}
    target_energies = {}
    overlap_ao = mol.intor('int1e_ovlp')

    def _failed_row(dt, method, e_tot, message, rho_diff=np.inf, meta=None):
        is_wy = method == 'WY'
        meta = meta or {}
        return {
            'density_type': dt,
            'inversion_method': method,
            'density_used': f'{dt}_reference',
            'orbital_source': method,
            'system': label, 'omega': omega, 'basis': basis_label,
            'spin': spin, 'charge': charge,
            'E_ref': float(e_tot), 'E_hf_tot': float(E_hf),
            'E_fci_tot': np.nan, 'E_c': np.nan,
            'E_c_definition': 'E_FCI(FCI_DM)-E_HF(HF_DM)',
            'inversion_ok': False, 'inversion_rho_ok': False,
            'inversion_data_used': False,
            'inversion_status': message,
            'inversion_rho_l1': float(rho_diff),
            'WY_ok': False if is_wy else np.nan,
            'WY_rho_ok': False if is_wy else np.nan,
            'WY_data_used': False if is_wy else np.nan,
            'WY_status': message if is_wy else '',
            'wy_rho_diff': float(rho_diff) if is_wy else np.nan,
            'PYOEP_ok': False if not is_wy else np.nan,
            'PYOEP_rho_ok': False if not is_wy else np.nan,
            'PYOEP_data_used': False if not is_wy else np.nan,
            'PYOEP_status': message if not is_wy else '',
            'pyoep_rho_diff': float(rho_diff) if not is_wy else np.nan,
            'pyoep_error': message if not is_wy else '',
            'pyoep_available': (meta.get('available', PYOEP_AVAILABLE)
                                if not is_wy else np.nan),
            'pyoep_method': meta.get('method', 'KSINV') if not is_wy else '',
            'pyoep_orbital_basis': (meta.get('orbital_basis', basis_label)
                                     if not is_wy else ''),
            'pyoep_oep_basis': (meta.get('oep_basis', PYOEP_OEP_BASIS)
                                if not is_wy else ''),
            'pyoep_same_system_basis': (meta.get('same_system_basis', False)
                                         if not is_wy else np.nan),
            'pyoep_space_sym': (meta.get('space_sym', PYOEP_SPACE_SYM)
                                if not is_wy else np.nan),
            'pyoep_maxiter': (meta.get('maxiter', PYOEP_MAXITER)
                              if not is_wy else np.nan),
            'pyoep_conv_thr': (meta.get('conv_thr', PYOEP_CONV_THR)
                               if not is_wy else np.nan),
            'pyoep_thr_fai_oep': (meta.get('thr_fai_oep', PYOEP_THR_FAI)
                                  if not is_wy else np.nan),
        }

    for dt in dts:
        tup = _get(dt)
        if tup is None:
            for method in cfg['inversion_methods']:
                pass1_rows[(dt, method)] = None
                pass2_rows[(dt, method)] = None
            continue
        dm1_ao, dm1_mo, dm2_mo, E_tot = tup
        target_energies[dt] = float(E_tot)

        # Density normalization check before Wu--Yang.  This catches an
        # accidental spin-block, AO/MO, or HF/FCI density mix-up immediately.
        dm1_total_ao = _norm_rdm(dm1_ao)
        electrons_from_dm = float(
            np.einsum('ij,ji->', overlap_ao, dm1_total_ao)
        )
        if abs(electrons_from_dm - mol.nelectron) > 1.0e-6:
            raise ValueError(
                f"{dt} density integrates to {electrons_from_dm:.10f} "
                f"electrons; expected {mol.nelectron}."
            )
        if verbose:
            print(
                f"    {dt} density trace Tr[S D] = "
                f"{electrons_from_dm:.10f}"
            )

        # Both inversion methods see separate copies of the same untouched
        # target density.  Nothing below falls back to the target or to the
        # other inversion's density.
        if spin != 0:
            raise RuntimeError(
                "star.katrovich currently requires a closed-shell H2 target"
            )

        for method in cfg['inversion_methods']:
            key = (dt, method)
            target_for_method = np.array(
                dm1_total_ao, dtype=np.float64, copy=True
            )
            inversion_meta = {}
            dm_reference = np.asarray(
                dm1_total_ao, dtype=np.float64
            )
            dm_reference = 0.5 * (dm_reference + dm_reference.T)
            if verbose:
                print(f"    {dt}: independent {method} inversion …")
            try:
                if method == 'WY':
                    inversion_obj, rho_diff = run_wy_inversion(
                        mol, target_for_method,
                        rho_l1_tol=cfg['wy_rho_l1_tol'],
                        strict=cfg['wy_strict'],
                        grid_level=cfg['grid_level'],
                        verbose=verbose,
                    )
                    method_strict = cfg['wy_strict']
                elif method == 'PYOEP':
                    inversion_obj, rho_diff, inversion_meta = (
                        run_pyoep_inversion(
                            mol, target_for_method, E_tot, basis_label,
                            rho_l1_tol=cfg['pyoep_rho_l1_tol'],
                            grid_level=cfg['grid_level'],
                            verbose=verbose,
                        )
                    )
                    method_strict = cfg['pyoep_strict']
                else:
                    raise ValueError(f'unknown inversion method {method!r}')

                inversion_ok = bool(
                    getattr(inversion_obj, 'converged', False)
                )
                rho_ok = bool(getattr(inversion_obj, 'rho_ok', False))
                usable, reason = _wy_has_finite_data(inversion_obj)
                accepted = usable and (
                    (inversion_ok and rho_ok) or not method_strict
                )
                if not accepted:
                    message = (
                        f'{method} inversion unavailable: '
                        f'converged={inversion_ok}, rho_ok={rho_ok}, '
                        f'rho_L1={rho_diff:.3e}, data={reason}'
                    )
                    extra_error = inversion_meta.get('error', '')
                    if extra_error:
                        message += f', error={extra_error}'
                    # Inversion failure invalidates only orbital-dependent
                    # GL2.  Reference-density DFT/SCE/ACM data remain valid
                    # and must still be evaluated and written.
                    pass1_rows[key] = _pass1_single(
                        mol, dm1_ao, dm1_mo, dm2_mo, mf.mo_coeff,
                        spin, dt, label, basis_label,
                        E_tot, E_hf, omega, cfg,
                        dm_reference, inversion_obj, rho_diff, method,
                        inversion_meta, verbose,
                    )
                    pass1_rows[key]['inversion_status'] = message
                    if method == 'WY':
                        pass1_rows[key]['WY_status'] = message
                    else:
                        pass1_rows[key]['PYOEP_status'] = message
                    pass2_rows[key] = _pass2_single(
                        mol, dm1_ao, dm1_mo, mf.mo_coeff, spin, dt,
                        label, omega, cfg, dm_reference, method, verbose,
                    )
                    print(f"  ⚠ {dt}/{method}/{label}: {message}")
                    continue

                dm_ks = np.asarray(
                    inversion_obj.dm, dtype=np.float64
                )
                dm_ks = 0.5 * (dm_ks + dm_ks.T)
                inversion_obj.dm = dm_ks

                pass1_rows[key] = _pass1_single(
                    mol, dm1_ao, dm1_mo, dm2_mo, mf.mo_coeff,
                    spin, dt, label, basis_label,
                    E_tot, E_hf, omega, cfg,
                    dm_reference, inversion_obj, rho_diff, method,
                    inversion_meta, verbose,
                )
                pass2_rows[key] = _pass2_single(
                    mol, dm1_ao, dm1_mo, mf.mo_coeff, spin, dt,
                    label, omega, cfg, dm_reference, method, verbose,
                )
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                pass1_rows[key] = _failed_row(
                    dt, method, E_tot, message, meta=inversion_meta
                )
                pass2_rows[key] = None
                print(f"  ⚠ {dt}/{method}/{label} failed independently: "
                      f"{message}")

        dm2_mo = None; tup = None
        dm1_ao = dm1_mo = None

    # Define correlation once from independently evaluated total energies:
    #
    #   E_c = E_FCI[FCI density matrix] - E_HF[HF density matrix].
    #
    # Store the same reference difference in both density sections, while
    # retaining each row's own E_ref, W0, J, exchange, WY orbitals, and GL2.
    E_fci = target_energies.get('FCI', np.nan)
    E_corr_fci_hf = (
        float(E_fci - E_hf)
        if np.isfinite(E_fci) and np.isfinite(E_hf)
        else np.nan
    )
    for dt in dts:
        for method in cfg['inversion_methods']:
            row = pass1_rows.get((dt, method))
            if row is None:
                continue
            row['E_hf_tot'] = float(E_hf)
            row['E_fci_tot'] = (
                float(E_fci) if np.isfinite(E_fci) else np.nan
            )
            row['E_c'] = E_corr_fci_hf
            row['E_c_definition'] = 'E_FCI(FCI_DM)-E_HF(HF_DM)'

    if verbose:
        print(
            f"    E_c = E_FCI(FCI_DM) - E_HF(HF_DM) "
            f"= {E_corr_fci_hf:.12f}"
        )

    return pass1_rows, pass2_rows


# ═══════════════════════════════════════════════════════════════════════════
#  MERGE  —  single scenario, SCE-only correlations
#
#  gl2_mode = 'dou_sig'  ->  gl2_master = gl2_doubles + gl2_singles
#
#  W0_prime = 2 * gl2_master
#  E_spl    = E_HF(HF_DM) + _spl_corr
#             (W0[density_type], gl2_master, W_inf_SCE[density_type])
#  E_gisi2  = E_HF(HF_DM) + _gisi2_corr
#             (W0[density_type], gl2_master,
#              W_inf_SCE[density_type], Wp_inf_SCE[density_type])
#
#  Both SPL and gISI2 use ONLY SCE data (W_inf_SCE and Wp_inf_SCE); the
#  exact-2RDM diagnostic pair has been removed.
#  E_c is stored as E_FCI(FCI_DM)-E_HF(HF_DM) in BOTH density sections.
# ═══════════════════════════════════════════════════════════════════════════

def _select_gl2(p1, gl2_mode):
    d = p1.get('gl2_doubles', np.nan) if p1 else np.nan
    s = p1.get('gl2_singles', np.nan) if p1 else np.nan
    if gl2_mode == 'dou_sig':
        return d + s if not (np.isnan(d) or np.isnan(s)) else np.nan
    raise ValueError(f"unknown gl2_mode {gl2_mode!r}")


def merge_row(p1, p2, gl2_mode):
    if p1 is None and p2 is None:
        return None

    row = {}
    src = p1 if p1 is not None else p2

    for k in (
        'density_type', 'inversion_method', 'system', 'omega', 'basis',
        'spin', 'charge',
        'density_used', 'orbital_source', 'E_ref', 'E_hf_tot', 'E_fci_tot',
        'E_c', 'E_c_definition',
    ):
        row[k] = src.get(k, np.nan)
    row['gl2_mode'] = gl2_mode
    row['R_bohr'] = p2.get('R_bohr', np.nan) if p2 else np.nan

    def _g1(k):
        return p1.get(k, np.nan) if p1 else np.nan

    def _g2(k):
        return p2.get(k, np.nan) if p2 else np.nan

    for k in ('W0', 'E_xx_density', 'E_xx_HF', 'E_xx_FCI',
              'J', 'Vee', 'W1_exact',
              'E_kin', 'E_ext', 'E_vne', 'E_nn',
              'inversion_ok', 'inversion_rho_ok',
              'inversion_data_used', 'inversion_status',
              'inversion_rho_l1',
              'WY_ok', 'WY_rho_ok', 'WY_data_used', 'WY_status',
              'wy_rho_diff',
              'PYOEP_ok', 'PYOEP_rho_ok', 'PYOEP_data_used',
              'PYOEP_status', 'pyoep_rho_diff', 'pyoep_error',
              'pyoep_available', 'pyoep_method',
              'pyoep_orbital_basis', 'pyoep_oep_basis',
              'pyoep_same_system_basis', 'pyoep_space_sym',
              'pyoep_maxiter', 'pyoep_conv_thr', 'pyoep_thr_fai_oep',
              'gl2_doubles', 'gl2_singles', 'gl2_total'):
        row[k] = _g1(k)

    for model in DFT_MODELS:
        row[f'W_inf_{model}']  = _g1(f'W_inf_{model}')
        row[f'Wp_inf_{model}'] = _g1(f'Wp_inf_{model}')

    for k in ('U', 'W_inf_SCE', 'Wp_inf_SCE', *_CYL_KEYS):
        row[k] = _g2(k)

    E_ref = row.get('E_ref', np.nan)
    E_c   = row.get('E_c',   np.nan)
    E_hf  = row.get('E_hf_tot', np.nan)

    gl2 = _select_gl2(p1, gl2_mode)
    row['gl2']      = gl2
    row['W0_prime'] = 2.0*gl2 if not np.isnan(gl2) else np.nan
    row['gl2_2x']   = row['W0_prime']
    row['E_xc']     = (row['W0'] + E_c
                       if not (np.isnan(row['W0']) or np.isnan(E_c))
                       else np.nan)
    row['E_gl2_tot'] = float(E_hf + gl2) if not (np.isnan(gl2) or np.isnan(E_hf)) else np.nan

    # Use the exchange evaluated on the current row's density.  The separate
    # E_xx_HF/E_xx_FCI columns make the density provenance explicit.
    W0         = _g1('E_xx_density')
    if np.isnan(W0):
        W0 = _g1('W0')
    W_inf_SCE  = _g2('W_inf_SCE')
    Wp_inf_SCE = _g2('Wp_inf_SCE')

    # ── (A) post-HF form:  E = E_HF + int_0^1 (W_l - W0) dl  ────────────
    #     the standard ACM-on-HF total energy (SCE endpoints only).
    if not any(np.isnan(v) for v in (W0, gl2, E_hf, W_inf_SCE, Wp_inf_SCE)):
        spl_corr_sce   = _spl_corr(W0, gl2, W_inf_SCE)
        gisi2_corr_sce = _gisi2_corr(W0, gl2, W_inf_SCE, Wp_inf_SCE)
        row['E_spl']   = float(E_hf + spl_corr_sce)   if not np.isnan(spl_corr_sce)   else np.nan
        row['E_gisi2'] = float(E_hf + gisi2_corr_sce) if not np.isnan(gisi2_corr_sce) else np.nan
    else:
        row['E_spl'] = row['E_gisi2'] = np.nan

    # ── (B) density-based form (mirrors 7.py AC1/AC6) ──────────────────
    #     E = T[FCI] + V_ne[FCI] + J[FCI] + V_nn + W_1^model ,
    #     with W_1^model the AC integrand at lambda = 1 (the physical XC
    #     potential energy V_ee - J that the model predicts).
    #
    #     E_1e   = T + V_ne           (one-electron energy on the FCI 1-RDM)
    #     E_base = E_1e + J + V_nn     (everything except E_xc)
    #     E_acm_exact = E_base + W1_exact  ==  E_ref   (the AC1 = FCI identity)
    E_kin  = _g1('E_kin')
    E_vne  = _g1('E_vne')
    E_nn   = _g1('E_nn')
    J_val  = _g1('J')
    W1_ex  = _g1('W1_exact')

    E_1e   = (E_kin + E_vne) if not (np.isnan(E_kin) or np.isnan(E_vne)) else np.nan
    E_base = (E_1e + J_val + E_nn
              if not any(np.isnan(v) for v in (E_1e, J_val, E_nn)) else np.nan)
    row['E_1e']   = E_1e
    row['E_base'] = E_base

    # exact endpoint -> reconstructs FCI (self-consistency check; == E_ref)
    row['E_acm_exact'] = (float(E_base + W1_ex)
                          if not (np.isnan(E_base) or np.isnan(W1_ex)) else np.nan)

    # SPL / gISI2 lambda=1 endpoints and their density-based totals
    W1_spl   = _spl_W1(W0, gl2, W_inf_SCE)
    W1_gisi2 = _gisi2_W1(W0, gl2, W_inf_SCE, Wp_inf_SCE)
    row['W1_spl']   = W1_spl
    row['W1_gisi2'] = W1_gisi2
    row['E_spl_acm'] = (float(E_base + W1_spl)
                        if not (np.isnan(E_base) or np.isnan(W1_spl)) else np.nan)
    row['E_gisi2_acm'] = (float(E_base + W1_gisi2)
                          if not (np.isnan(E_base) or np.isnan(W1_gisi2)) else np.nan)

    return row


# ═══════════════════════════════════════════════════════════════════════════
#  PARALLEL DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def _init_worker(nthreads, mem_limit_gb=0.0):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(nthreads)
    try:
        from pyscf import lib as _lib
        _lib.num_threads(nthreads)
    except Exception:
        pass
    # A soft address-space cap turns an OOM-kill (SIGKILL -> opaque
    # BrokenProcessPool) into a catchable MemoryError with a traceback that
    # actually says which geometry and which step blew up.
    if mem_limit_gb and mem_limit_gb > 0:
        try:
            import resource
            nbytes = int(mem_limit_gb * (1024**3))
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except Exception:
            pass


def _worker(task):
    sys_info, cfg = task
    label = sys_info[0]
    t0 = time.time()
    try:
        p1, p2 = run_system_two_pass(sys_info, cfg, verbose=cfg['verbose'])
        return label, p1, p2, None, time.time()-t0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return label, None, None, f"{type(e).__name__}: {e}", time.time()-t0


def _executor_kwargs(nproc, ctx, threads, mem_limit_gb):
    kw = dict(max_workers=nproc, mp_context=ctx,
              initializer=_init_worker, initargs=(threads, mem_limit_gb))
    # Python >= 3.11: recycle the worker after every geometry so a leak or a
    # high-water mark in one point cannot accumulate across the run.
    import inspect
    if 'max_tasks_per_child' in inspect.signature(cf.ProcessPoolExecutor).parameters:
        kw['max_tasks_per_child'] = 1
    return kw


def run_all_parallel(systems, cfg, nproc=1, threads_per_worker=1,
                     ckpt_path=None, resume=False, mem_limit_gb=0.0,
                     max_retries=2):
    """Wave-based driver.

    A worker killed by the cgroup OOM reaper takes the WHOLE executor down
    (BrokenProcessPool), and every still-pending future with it. Rather than
    losing the run, each wave records whatever finished, halves the worker
    count, and retries the remainder; the last attempt runs serially. Any
    geometry that still fails is reported and skipped, not fatal.
    """
    raw_pairs: Dict[Tuple[str, str, str], Tuple] = {}
    done_labels = set()

    if resume and ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as fh:
            raw_pairs = pickle.load(fh)
        labels_in_checkpoint = {k[0] for k in raw_pairs if len(k) == 3}

        def _checkpoint_method_completed(label, density_type, method):
            pair = raw_pairs.get((label, density_type, method))
            if not pair or pair[0] is None:
                return False
            row = pair[0]
            # New checkpoints explicitly record whether finite inversion
            # orbitals produced GL2.  Retain compatibility with older files,
            # whose failed rows were marked by a :FAILED density label.
            if 'inversion_data_used' in row:
                return bool(row.get('inversion_data_used', False))
            density_used = str(row.get('density_used', ''))
            return not density_used.endswith(':FAILED')

        done_labels = {
            label for label in labels_in_checkpoint
            if all(_checkpoint_method_completed(label, dt, method)
                   for dt in cfg['density_types']
                   for method in cfg['inversion_methods'])
        }
        print(f"  ↻ resumed {len(done_labels)} completed geometries from {ckpt_path}")

    todo = [s for s in systems if s[0] not in done_labels]
    print(f"\n{'='*70}")
    print(f"  H2 SCE / DFT-W_inf SCAN")
    print(f"  Geometries   : {len(systems)}  ({len(todo)} to run)")
    print(f"  R range      : {cfg['r_min']:.3f} .. {cfg['r_max']:.3f} {cfg['unit']}")
    print(f"  Density types: {' → '.join(cfg['density_types'])}")
    print(f"  Inversions   : {' + '.join(cfg['inversion_methods'])} "
          "(fully independent)")
    if 'PYOEP' in cfg['inversion_methods']:
        print(f"  PyOEP bases  : orbital={systems[0][4]}  "
              f"OEP={PYOEP_OEP_BASIS}")
        print(f"  PyOEP setup  : thr_fai_oep={PYOEP_THR_FAI:.1e}  "
              f"space_sym={PYOEP_SPACE_SYM}")
    print(f"  DFT models   : {len(DFT_MODELS)}  ({', '.join(DFT_MODELS)})")
    print(f"  DFT W_inf    : {'ENABLED' if cfg['compute_dft_models'] else 'DISABLED'}")
    print(f"  Property grid: level {cfg['grid_level']}")
    print(f"  3D SCE/ZPE   : {'ENABLED' if cfg['compute_sce3d'] else 'DISABLED'}"
          f"{'' if cfg['sce3d_dts'] is None else '  [' + ','.join(cfg['sce3d_dts']) + ']'}")
    print(f"  Parallel     : {nproc} process(es) x {threads_per_worker} thread(s)"
          f"{'' if not mem_limit_gb else f',  cap {mem_limit_gb:g} GB/worker'}")
    print(f"  NOTE: budget ~1 GB RSS per worker at grid level 9. If SLURM gives")
    print(f"        you less than nproc x 1 GB, lower -n or --grid-level.")
    print(f"{'='*70}\n")

    t0 = time.time()
    total = len(todo)
    ndone = 0
    pending = list(todo)
    attempt = 0

    def _absorb(result):
        nonlocal ndone
        label, p1, p2, err, dt = result
        _record(raw_pairs, label, p1, p2, cfg['density_types'],
                cfg['inversion_methods'])
        done_labels.add(label)
        ndone += 1
        print(f"  [{ndone}/{total}] {label:<14s} {dt:8.1f}s"
              f"{'' if err is None else '  ⚠ ' + err}", flush=True)
        _checkpoint(raw_pairs, ckpt_path)

    while pending and attempt <= max_retries:
        nproc_eff = 1 if attempt == max_retries else max(1, nproc // (2 ** attempt))
        if attempt > 0:
            print(f"\n  ↻ retry {attempt}: {len(pending)} geometrie(s) left, "
                  f"dropping to {nproc_eff} worker(s)\n", flush=True)

        if nproc_eff <= 1:
            _init_worker(threads_per_worker, mem_limit_gb)
            for s in list(pending):
                try:
                    _absorb(_worker((s, cfg)))
                except Exception as e:
                    print(f"  ✗ {s[0]}: {type(e).__name__}: {e}", flush=True)
        else:
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[v] = str(threads_per_worker)
            ctx = mp_ctx.get_context('spawn')
            kw = _executor_kwargs(nproc_eff, ctx, threads_per_worker, mem_limit_gb)
            try:
                with cf.ProcessPoolExecutor(**kw) as ex:
                    futs = {ex.submit(_worker, (s, cfg)): s[0] for s in pending}
                    for fut in cf.as_completed(futs):
                        try:
                            _absorb(fut.result())
                        except cf.process.BrokenProcessPool:
                            raise
                        except Exception as e:
                            print(f"  ✗ {futs[fut]}: {type(e).__name__}: {e}", flush=True)
            except cf.process.BrokenProcessPool:
                print("\n  ⚠ BrokenProcessPool — a worker was killed (almost always "
                      "the cgroup OOM reaper).\n"
                      "    Completed points are checkpointed; retrying the rest "
                      "with fewer workers.\n", flush=True)

        pending = [s for s in pending if s[0] not in done_labels]
        attempt += 1

    if pending:
        print(f"\n  ✗ {len(pending)} geometry point(s) never completed: "
              f"{', '.join(s[0] for s in pending)}")
        print("    They are simply absent from the CSVs; --resume will retry them.")

    print(f"\n✓ {ndone}/{total} geometry points done in {time.time()-t0:.1f}s")
    return raw_pairs


def _record(raw_pairs, label, p1_rows, p2_rows, dts, methods):
    for dt in dts:
        for method in methods:
            key = (dt, method)
            p1 = p1_rows.get(key) if p1_rows else None
            p2 = p2_rows.get(key) if p2_rows else None
            raw_pairs[(label, dt, method)] = (p1, p2)


def _checkpoint(raw_pairs, ckpt_path):
    if not ckpt_path:
        return
    tmp = ckpt_path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(raw_pairs, fh, protocol=4)
    os.replace(tmp, ckpt_path)


def build_merged_store(raw_pairs, systems, dts, methods, gl2_mode):
    merged_store = {(dt, method): [] for dt in dts for method in methods}
    for s in systems:
        label = s[0]
        for dt in dts:
            for method in methods:
                p1, p2 = raw_pairs.get(
                    (label, dt, method), (None, None)
                )
                merged = merge_row(p1, p2, gl2_mode)
                if merged is not None:
                    merged_store[(dt, method)].append(merged)
    return merged_store


# ═══════════════════════════════════════════════════════════════════════════
#  CSV WRITER
# ═══════════════════════════════════════════════════════════════════════════

def save_sectioned_csv(merged_store, dts, methods, output_filename, gl2_mode):
    """Write one section per target density with WY and PyOEP side by side.

    The former four-section format made comparison difficult and repeated the
    complete header for every method.  Here each geometry occupies one row in
    ``HF DENSITY`` or ``FCI DENSITY``.  Every inversion-derived field is
    method-prefixed, so the two calculations remain independent and cannot be
    mistaken for a fallback.  The original unprefixed Katrovich cylindrical
    and ACM columns are retained at the far right and are sourced from WY only.
    """
    common_cols = [
        'gl2_mode', 'density_type', 'system', 'R_bohr', 'omega', 'basis',
        'spin', 'charge', 'E_ref', 'E_HF', 'E_FCI', 'E_c',
        'E_c_definition', 'density_used',
    ]

    diagnostic_fields = [
        'available', 'converged', 'rho_ok', 'data_used',
        'rho_diff', 'error', 'status',
    ]
    energy_fields = [
        'W0', 'E_c', 'E_xc', 'W0_prime', 'gl2_2x',
        'gl2_doubles', 'gl2_singles', 'gl2_total', 'gl2', 'E_gl2_tot',
        'E_xx_density', 'E_xx_HF', 'E_xx_FCI',
        'J', 'Vee', 'W1_exact', 'E_kin', 'E_ext', 'E_vne', 'E_nn',
        'ks_gap',
    ]
    dft_fields = []
    for model in DFT_MODELS:
        dft_fields.extend((f'W_inf_{model}', f'Wp_inf_{model}'))
    radial_fields = ['U', 'W_inf_SCE', 'Wp_inf_SCE']
    method_cyl_fields = list(_CYL_KEYS)
    acm_fields = [
        'E_spl', 'E_gisi2', 'E_1e', 'E_base',
        'W1_spl', 'W1_gisi2', 'E_acm_exact',
        'E_spl_acm', 'E_gisi2_acm',
    ]
    method_value_fields = (
        energy_fields + dft_fields + radial_fields
        + method_cyl_fields + acm_fields
    )
    method_cols = []
    for method in methods:
        method_cols.extend(
            f'{method}_{field}'
            for field in diagnostic_fields + method_value_fields
        )

    # Requested Katrovich block, kept in this exact order.  These unprefixed
    # compatibility columns are copied from WY only; there is no PyOEP fallback.
    katrovich_cols = [
        'ts_cyl', 'vext_cyl', 'vnn_cyl', 'hartree_cyl',
        'vee_sce_cyl', 'winf_cyl', 'winf_prime_rmzpe_cyl',
        'total_sce_cyl', 'total_sce_rmzpe_cyl',
        'levy_zahariev_c_cyl',
        'inner_z1', 'inner_h1', 'inner_z2', 'inner_h2',
        'E_spl', 'E_gisi2', 'E_1e', 'E_base',
        'W1_spl', 'W1_gisi2', 'E_acm_exact',
        'E_spl_acm', 'E_gisi2_acm',
    ]
    all_cols = common_cols + method_cols + katrovich_cols

    def _missing(value):
        if value is None:
            return True
        try:
            return bool(np.isscalar(value) and np.isnan(value))
        except (TypeError, ValueError):
            return False

    def _fmt(value):
        if _missing(value):
            return ''
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (float, np.floating)):
            return f'{float(value):.10f}'
        return str(value)

    def _first_present(rows, key, default=np.nan):
        for row in rows:
            if row is None:
                continue
            value = row.get(key, default)
            if not _missing(value) and value != '':
                return value
        return default

    def _method_diagnostic(row, method, field):
        if row is None:
            return np.nan if field not in ('error', 'status') else ''
        if method == 'WY':
            mapping = {
                'available': bool(KSPIES_AVAILABLE),
                'converged': row.get('WY_ok', row.get('inversion_ok', False)),
                'rho_ok': row.get('WY_rho_ok',
                                  row.get('inversion_rho_ok', False)),
                'data_used': row.get('WY_data_used',
                                     row.get('inversion_data_used', False)),
                'rho_diff': row.get('wy_rho_diff',
                                    row.get('inversion_rho_l1', np.nan)),
                'status': row.get('WY_status',
                                  row.get('inversion_status', '')),
            }
            used = bool(mapping['data_used'])
            mapping['error'] = '' if used else mapping['status']
            return mapping[field]
        mapping = {
            'available': row.get('pyoep_available', PYOEP_AVAILABLE),
            'converged': row.get('PYOEP_ok', row.get('inversion_ok', False)),
            'rho_ok': row.get('PYOEP_rho_ok',
                              row.get('inversion_rho_ok', False)),
            'data_used': row.get('PYOEP_data_used',
                                 row.get('inversion_data_used', False)),
            'rho_diff': row.get('pyoep_rho_diff',
                                row.get('inversion_rho_l1', np.nan)),
            'status': row.get('PYOEP_status',
                              row.get('inversion_status', '')),
        }
        used = bool(mapping['data_used'])
        raw_error = row.get('pyoep_error', '')
        mapping['error'] = '' if used else (raw_error or mapping['status'])
        return mapping[field]

    def _combined_rows(dt):
        by_method = {}
        system_order = []
        for method in methods:
            method_map = {}
            for row in merged_store.get((dt, method), []):
                system = row.get('system', '')
                method_map[system] = row
                if system not in system_order:
                    system_order.append(system)
            by_method[method] = method_map

        combined = []
        for system in system_order:
            rows = [by_method[m].get(system) for m in methods]
            out = {
                'gl2_mode': gl2_mode,
                'density_type': dt,
                'system': system,
                'R_bohr': _first_present(rows, 'R_bohr'),
                'omega': _first_present(rows, 'omega'),
                'basis': _first_present(rows, 'basis'),
                'spin': _first_present(rows, 'spin'),
                'charge': _first_present(rows, 'charge'),
                'E_ref': _first_present(rows, 'E_ref'),
                'E_HF': _first_present(rows, 'E_hf_tot'),
                'E_FCI': _first_present(rows, 'E_fci_tot'),
                'E_c': _first_present(rows, 'E_c'),
                'E_c_definition': _first_present(rows, 'E_c_definition', ''),
                'density_used': _first_present(
                    rows, 'density_used', f'{dt}_reference'
                ),
            }
            for method in methods:
                row = by_method[method].get(system)
                for field in diagnostic_fields:
                    out[f'{method}_{field}'] = _method_diagnostic(
                        row, method, field
                    )
                for field in method_value_fields:
                    out[f'{method}_{field}'] = (
                        row.get(field, np.nan) if row else np.nan
                    )

            # Backward-compatible Katrovich columns: always WY, never fallback.
            wy_row = by_method.get('WY', {}).get(system)
            for field in katrovich_cols:
                out[field] = (
                    wy_row.get(field, np.nan) if wy_row else np.nan
                )
            combined.append(out)
        return combined

    with open(output_filename, 'w', newline='') as f:
        f.write(f'# scenario: gl2_mode = {gl2_mode}  (doubles + singles), '
                f'W0_prime = 2 * gl2\n')
        f.write('# SPL and gISI2 correlations use ONLY SCE data '
                '(W_inf_SCE, Wp_inf_SCE).\n')
        f.write('# Density order: HF first, FCI second. Each target has fully '
                'independent side-by-side WY_* and PYOEP_* evaluations.\n')
        f.write('# Density-L1 is a quality diagnostic by default; use '
                '--wy-strict or --pyoep-strict to reject a method.\n')
        f.write('# No fallback: WY/PyOEP orbitals are independent and feed '
                'only their corresponding GL2 columns; all density-derived '
                'quantities use the untouched HF/FCI reference density.\n')
        f.write('# E_c = E_FCI(FCI_DM) - E_HF(HF_DM); E_HF and E_FCI are '
                'stored explicitly in both density sections.\n')
        f.write('# W0 = E_xx_density = -J[density]/2 for two-electron '
                'singlet H2; E_xx_HF/E_xx_FCI preserve provenance.\n')
        f.write('# Two total-energy constructions per model:\n')
        f.write('#   E_spl / E_gisi2         = E_HF + int_0^1 (W_l - W0) dl   '
                '(post-HF ACM).\n')
        f.write('#   E_spl_acm / E_gisi2_acm = E_base + W1_model,  '
                'E_base = T[density]+V_ne[density]+J[density]+V_nn,\n')
        f.write('#     W1_model = AC integrand at lambda=1 (density-based, '
                'mirrors 7.py AC1/AC6).\n')
        f.write('#   E_acm_exact = E_base + W1_exact == E_ref for the '
                'corresponding HF or FCI density block.\n')
        f.write('# The final unprefixed Katrovich cylindrical/ACM block is '
                'copied from WY only and never falls back to PyOEP.\n')
        f.write(f'# generated {datetime.now().isoformat(timespec="seconds")}\n')
        writer = csv.writer(f, lineterminator='\n')
        first = True
        for dt in dts:
            rows = _combined_rows(dt)
            if not rows:
                continue
            if not first:
                for _ in range(_BLANK_LINES):
                    f.write('\n')
            f.write(f'# ===== {dt} DENSITY =====\n')
            writer.writerow(all_cols)
            for row in rows:
                writer.writerow([_fmt(row.get(col, np.nan)) for col in all_cols])
            first = False

    print(f"✓ [{gl2_mode}] results saved → {output_filename}")
    return output_filename


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(merged_store, dts, methods, gl2_mode):
    sep = "─"*180
    for dt in dts:
        for method in methods:
            rows = merged_store.get((dt, method), [])
            if not rows:
                continue
            print(f"\n{sep}")
            print(f"  [{gl2_mode}]  {dt} DENSITY / {method}  "
                  f"({len(rows)} pts)")
            print(sep)
            print(f"  {'system':>12}  {'R(a0)':>8}  {'E_ref':>14}  {'gl2':>13}"
                  f"  {'W0_prime':>13}  {'W_inf(rad)':>13}  {'W_inf_LDA':>13}"
                  f"  {'vee_sce_cyl':>13}  {'winf_cyl':>13}  {'Wp_rmZPE_cyl':>13}"
                  f"  {'E_spl':>14}  {'E_gisi2':>14}")
            for row in rows:
                def _g(k):
                    v = row.get(k, np.nan)
                    return (float('nan')
                            if not isinstance(v, float) else v)

                def _f(v, w=13):
                    return (f'{v:{w}.8f}' if v == v
                            else f"{'NaN':>{w}}")

                print(f"  {row.get('system','?'):>12}  {_g('R_bohr'):8.3f}"
                      f"  {_f(_g('E_ref'), 14)}"
                      f"  {_f(_g('gl2'))}  {_f(_g('W0_prime'))}"
                      f"  {_f(_g('W_inf_SCE'))}  {_f(_g('W_inf_LDA'))}"
                      f"  {_f(_g('vee_sce_cyl'))}  {_f(_g('winf_cyl'))}"
                      f"  {_f(_g('winf_prime_rmzpe_cyl'))}"
                      f"  {_f(_g('E_spl'), 14)}  {_f(_g('E_gisi2'), 14)}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='H2 scan: DFT W_inf models + radial/3D SCE + independent '
                    'WY/PyOEP GL2, '
                    'one GL2 scenario (dou_sig.csv)')
    p.add_argument('-v', '--verbose', action='store_true')

    g = p.add_argument_group('geometry (default: the module-level R_VALS block)')
    g.add_argument('--r-min', type=float, default=None,
                   help='override R_VALS: smallest R in Bohr (R=0 is singular)')
    g.add_argument('--r-max', type=float, default=None)
    g.add_argument('--r-points', type=int, nargs=3, default=None,
                   metavar=('N_SHORT', 'N_MID', 'N_LONG'))
    g.add_argument('--basis', type=str, default=BASIS_DEF)
    g.add_argument('--unit', type=str, default=R_UNIT, choices=['Bohr', 'Angstrom'])

    b = p.add_argument_group('benchmark')
    b.add_argument('--skip-benchmark', action='store_true')
    b.add_argument('--force', action='store_true',
                   help='continue even if the startup benchmark fails')
    b.add_argument('--bench-pot-basis', type=str, default='aug-cc-pvtz')
    b.add_argument('--bench-grid-level', type=int, default=9)
    b.add_argument('--benchmark-only', action='store_true')

    c = p.add_argument_group('what to compute')
    c.add_argument(
        '--density-types',
        type=str,
        nargs='+',
        default=DENSITY_TYPES,
        help=(
            "Density blocks to evaluate: HF, FCI, or all. For two-electron "
            "H2, CCSD/CCSDT/CCSD(T) are accepted as aliases for FCI. Default "
            "order is 'HF FCI'; each density is sent independently through "
            "Wu--Yang/GL2 and SCE."
        ),
    )
    c.add_argument('--no-sce3d', action='store_true')
    c.add_argument('--no-dft-models', action='store_true')
    c.add_argument('--sce3d-dts', type=str, nargs='+', default=None,
                   help='restrict the expensive 3D SCE to these density types '
                        '(e.g. --sce3d-dts FCI). Default: all.')
    c.add_argument('--sce3d-grid-level', type=int, default=SCE3D_CFG['grid_level'])
    c.add_argument('--sce3d-m', type=int, default=SCE3D_CFG['m'])
    c.add_argument('--sce3d-outer-iter', type=int, default=SCE3D_CFG['max_outer_iter'])
    c.add_argument('--sce3d-zpe-max-points', type=int, default=SCE3D_CFG['zpe_max_points'],
                   help='cap the number of grid points entering the rm-ZPE '
                        'Hessian loop — the single biggest cost knob')
    c.add_argument('--sce3d-partner-mode', choices=['gradient', 'localmin'],
                   default=SCE3D_CFG['partner_mode'])
    c.add_argument('--grid-level', type=int, default=9,
                   help='Becke grid level for the DFT W_inf models and the WY '
                        'L1 check. 9 is the production value; 7 roughly halves '
                        'memory and time if the node is tight.')
    c.add_argument('--wy-strict', action='store_true')
    c.add_argument('--no-pyoep', action='store_true',
                   help='disable the independent PyOEP/KSINV branch')
    c.add_argument('--pyoep-strict', action='store_true',
                   help='reject PyOEP data when convergence or density-L1 '
                        'diagnostics fail; default keeps finite data')
    c.add_argument('--gl2-strict', action='store_true')
    c.add_argument('--wy-rho-l1-tol', type=float, default=WY_RHO_L1_TOL)
    c.add_argument('--pyoep-rho-l1-tol', type=float,
                   default=PYOEP_RHO_L1_TOL)

    r = p.add_argument_group('runtime')
    r.add_argument('-n', '--nproc', type=int, default=max(1, (os.cpu_count() or 2)//2))
    r.add_argument('-t', '--threads-per-worker', type=int, default=1)
    r.add_argument('--mem-limit-gb', type=float, default=0.0,
                   help='per-worker address-space cap. Turns a silent OOM kill '
                        'into a catchable MemoryError with a real traceback. '
                        'Set to about (SLURM mem) / nproc.')
    r.add_argument('--max-retries', type=int, default=2,
                   help='waves of retry after a BrokenProcessPool; the last '
                        'one runs serially')
    r.add_argument('--outdir', type=str, default='.')
    r.add_argument('--checkpoint', type=str, default='h2_scan_ckpt.pkl')
    r.add_argument('--resume', action='store_true')

    args = p.parse_args()
    # Validate aliases before starting the potentially expensive benchmark.
    ordered_density_types = parse_density_types(args.density_types, p)

    # ── 1. benchmark first ─────────────────────────────────────────────
    if not args.skip_benchmark:
        ok, _ = run_startup_benchmark(pot_basis=args.bench_pot_basis,
                                      grid_level=args.bench_grid_level)
        if not ok and not args.force:
            print("Benchmark FAILED — refusing to start production. "
                  "Re-run with --force to override, or --skip-benchmark.")
            return 1
    if args.benchmark_only:
        return 0

    if not KSPIES_AVAILABLE:
        print("⚠ kspies not available — WY sections will record failures; "
              "PyOEP remains independent.")
    if not PYOEP_AVAILABLE and not args.no_pyoep:
        print(f"⚠ PyOEP not available ({_PYOEP_IMPORT_ERROR}) — PyOEP "
              "sections will record failures; WY remains independent.")

    # ── 2. geometry grid ───────────────────────────────────────────────
    # Default: the module-level R_VALS / SYSTEM_BASIS block near the top of
    # this file. Edit that block to change the scan. The --r-* flags are an
    # optional override for one-off runs.
    if args.r_min is None and args.r_max is None and args.r_points is None \
            and args.basis == BASIS_DEF:
        systems = _ensure_unique_labels(SYSTEM_BASIS)
        r_vals = R_VALS
    else:
        r_vals = build_r_vals(
            R_MIN_DEF if args.r_min is None else args.r_min,
            R_MAX_DEF if args.r_max is None else args.r_max,
            *(args.r_points or [60, 60, 30]))
        systems = build_systems(r_vals, args.basis)

    cfg = dict(
        unit               = args.unit,
        r_min              = float(np.min(r_vals)),
        r_max              = float(np.max(r_vals)),
        grid_level         = int(args.grid_level),
        density_types      = ordered_density_types,
        inversion_methods  = (["WY"] if args.no_pyoep
                              else list(INVERSION_METHODS)),
        compute_dft_models = COMPUTE_DFT_MODELS and not args.no_dft_models,
        compute_sce3d      = COMPUTE_SCE3D      and not args.no_sce3d,
        sce3d_dts          = list(args.sce3d_dts) if args.sce3d_dts else None,
        sce3d_cfg          = dict(
            grid_level       = args.sce3d_grid_level,
            m                = args.sce3d_m,
            max_outer_iter   = args.sce3d_outer_iter,
            zpe_max_points   = args.sce3d_zpe_max_points,
            zpe_density_mass = SCE3D_CFG['zpe_density_mass'],
            partner_mode     = args.sce3d_partner_mode,
            seed             = SCE3D_CFG['seed'],
        ),
        wy_strict          = bool(args.wy_strict),
        pyoep_strict       = bool(args.pyoep_strict),
        gl2_strict         = bool(args.gl2_strict),
        wy_rho_l1_tol      = float(args.wy_rho_l1_tol),
        pyoep_rho_l1_tol   = float(args.pyoep_rho_l1_tol),
        verbose            = bool(args.verbose),
    )

    os.makedirs(args.outdir, exist_ok=True)
    ckpt = os.path.join(args.outdir, args.checkpoint) if args.checkpoint else None

    # ── 3. one expensive pass over all geometries ──────────────────────
    raw_pairs = run_all_parallel(systems, cfg,
                                 nproc=args.nproc,
                                 threads_per_worker=args.threads_per_worker,
                                 ckpt_path=ckpt, resume=args.resume,
                                 mem_limit_gb=args.mem_limit_gb,
                                 max_retries=args.max_retries)

    # ── 4. single scenario -> CSV, from the raw results ────────────────
    print("\n" + "="*70)
    print("  SCENARIO — master GL2 = doubles + singles")
    print("="*70)
    for gl2_mode, fname in GL2_SCENARIOS:
        merged = build_merged_store(
            raw_pairs, systems, cfg['density_types'],
            cfg['inversion_methods'], gl2_mode
        )
        out = os.path.join(args.outdir, fname)
        save_sectioned_csv(
            merged, cfg['density_types'], cfg['inversion_methods'],
            out, gl2_mode
        )
        if args.verbose:
            print_summary(
                merged, cfg['density_types'], cfg['inversion_methods'],
                gl2_mode
            )

    print("\n" + "="*70)
    print("  DONE")
    print("  dou_sig.csv : gl2 = doubles + singles,  W0' = 2*(doubles+singles)")
    print("  SPL / gISI2 built from SCE data only (W_inf_SCE, Wp_inf_SCE).")
    print("="*70)
    return 0


if __name__ == '__main__':
    sys.exit(main())

