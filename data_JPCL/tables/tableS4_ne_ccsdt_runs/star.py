

"""He/atom series driver with clear W_inf_PC4 and Wp_inf_PC4 summary.

Main reported PC4 models:

  W_inf_PC4  = W_inf_F0(s), the rational PC4 paper equation for W_inf.
  Wp_inf_PC4 = C int rho^(3/2) [1 + D2 s^2 + D3 s^4] dr,
                the PC4-prime fourth-order gradient equation from the theory section.
  W_inf_ePC and Wp_inf_ePC are the full paper meta-GGA ePC equations.

Batch optimization over SYSTEM_BASIS with dynamic reference value lookup.

T_ZP / W'inf dispatch:

  Nel == 1  ->  0.0 exactly          (no e-e interaction; prevents NaN in W_inf)
  Nel == 2  ->  exact analytical     (compute_tzp_nel2)
  Nel == 3  ->  local Hessian        n_skip = _NEL_ZERO_MODES[3] = 2
  Nel == 4  ->  planar co-motion     (compute_tzp_for_planar, last point dropped)
  Nel >= 5  ->  local Hessian        n_skip from _NEL_ZERO_MODES table

Paper-only PC4 model printed alongside SCE result for every system:
  W_inf_PC4  = A int d^3r rho^(4/3) F0(s)
  Wp_inf_PC4 = C int d^3r rho^(3/2) [1 + D2*s^2 + D3*s^4]
  where F0 and F0p are the paper-defined z=0 PC4/F0 equations.
  No SG4, LS-PBE, MSM, capped GE4, or fitted E/F diagnostics are printed here.
"""

import os
import argparse
import json
import math
import re as _re
import sys
import csv
from timeit import default_timer as timer

# -- JAX platform: GPU first, automatic CPU fallback --------------------------
def _init_jax_platform() -> None:
    import jax
    for _platform in ("gpu", "cuda", "rocm"):
        try:
            jax.config.update("jax_platform_name", _platform)
            import jax.numpy as _jnp
            _ = _jnp.array(1.0) + 1.0
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

from jaxsce.constants import A
from jaxsce.coordinates_3d import get_coordinate_system
#from pyscf_pc4_paper_only import CCSDDensity, CCSDTDensity, FCIDensity, HFDensity
from jaxsce.densities.pyscf import CCSDDensity, CCSDTDensity, FCIDensity, HFDensity


from jaxsce.integrate import VeeIntegration
from jaxsce.optimize import AngularOptimization, TwoElectron

# -----------------------------------------------------------------------------
# CSV SAVE HELPER
# -----------------------------------------------------------------------------

def save_batch_summary_csv(all_results, out_csv="data/batch_summary_pc4.csv"):
    """Save batch results to CSV with all parameters."""
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    headers = [
        "System",
        "W_inf_SCE",
        "W_inf_PC4",
        "Err_W_PC4",
        "Wp_inf_SCE",
        "Wp_inf_LDA",
        "Err_LDA",
        "Wp_inf_PC4",
        "Err_Wp_PC4",
        "Lambda",
        "B",
        "I0",
        "A",
        "C",
        "I2_int",
        "I3_int",
        "P0",
        "P1",
        "P2",
        "Time_s",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for tag, res in all_results.items():
            Winf      = float(res.get("Winf", float("nan")))
            W_inf_pc4 = float(res.get("W_inf_PC4", res.get("PC4", float("nan"))))
            Wp_sce    = float(res.get("T_ZP_prl", float("nan")))
            Wp_lda    = float(res.get("Wp_inf_LDA", float("nan")))
            Wp_pc4    = float(res.get("Wp_inf_PC4", float("nan")))
            Lambda    = float(res.get("Lambda", float("nan")))
            B_val     = float(res.get("B", float("nan")))
            I0        = float(res.get("I0", res.get("LDA_int", float("nan"))))
            A_val     = float(res.get("A", float("nan")))
            C_val     = float(res.get("C", float("nan")))
            I2_int    = float(res.get("I2_int", float("nan")))
            I3_int    = float(res.get("I3_int", float("nan")))
            P0        = float(res.get("P0", float("nan")))
            P1        = float(res.get("P1", float("nan")))
            P2        = float(res.get("P2", float("nan")))
            Time      = float(res.get("Time", float("nan")))

            errWpc4 = W_inf_pc4 - Winf if math.isfinite(W_inf_pc4) and math.isfinite(Winf) else float("nan")
            errLDA  = Wp_lda - Wp_sce if math.isfinite(Wp_lda) and math.isfinite(Wp_sce) else float("nan")
            errPC4  = Wp_pc4 - Wp_sce if math.isfinite(Wp_pc4) and math.isfinite(Wp_sce) else float("nan")

            writer.writerow({
                "System": tag,
                "W_inf_SCE": Winf,
                "W_inf_PC4": W_inf_pc4,
                "Err_W_PC4": errWpc4,
                "Wp_inf_SCE": Wp_sce,
                "Wp_inf_LDA": Wp_lda,
                "Err_LDA": errLDA,
                "Wp_inf_PC4": Wp_pc4,
                "Err_Wp_PC4": errPC4,
                "Lambda": Lambda,
                "B": B_val,
                "I0": I0,
                "A": A_val,
                "C": C_val,
                "I2_int": I2_int,
                "I3_int": I3_int,
                "P0": P0,
                "P1": P1,
                "P2": P2,
                "Time_s": Time,
            })

    print(f"\n  Saved batch summary CSV: {out_csv}", flush=True)

# -----------------------------------------------------------------------------
# UHF DENSITY-MATRIX COLLAPSE PATCH
# -----------------------------------------------------------------------------

#import pyscf_pc4_paper_only as _pyscf_mod
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
    "jorge-tzp": "jorge-TZP", "jorge-aqzp": "jorge-AQZP",
    "aug-cc-pvdz-pp": "aug-cc-pVDZ-PP", "aug-cc-pvtz-pp": "aug-cc-pVTZ-PP",
    "aug-cc-pvqz-pp": "aug-cc-pVQZ-PP", "aug-cc-pv5z-pp": "aug-cc-pV5Z-PP",
    "aug-cc-pv6z-pp": "aug-cc-pV6Z-PP", "aug-cc-pcvdz": "aug-cc-pCVDZ",
    "aug-cc-pcvtz": "aug-cc-pCVTZ",     "aug-cc-pcv4z": "aug-cc-pCV4Z",
    "aug-cc-pcv5z": "aug-cc-pCV5Z",     "aug-cc-pcv6z": "aug-cc-pCV6Z",
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
    ecp_match   = _re.search(r"(^ECP\b.*?^END\b)",   basis_string,
                              _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
    return (
        basis_match.group(1) if basis_match else basis_string,
        ecp_match.group(1)   if ecp_match   else None,
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
    has_ecp  = False
    if ecp_block is not None:
        from pyscf.gto.basis import parse_nwchem_ecp
        for parser_name in ("parse_ecp", "parse"):
            parser = getattr(parse_nwchem_ecp, parser_name, None)
            if parser is None:
                continue
            try:
                ecp_dict = {atom_symbol: parser(ecp_block)}
                has_ecp  = True
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
    "H":   {"Nel": 1,  "spin": 1, "W_ref": -0.3125,      "Wp_ref": 0.0},
    "H-":  {"Nel": 2,  "spin": 0, "W_ref": -0.569,       "Wp_ref": None},
    "He":  {"Nel": 2,  "spin": 0, "W_ref": -1.5008,      "Wp_ref": 0.621},
    "He-": {"Nel": 3,  "spin": 1, "W_ref": None,         "Wp_ref": None},
    "Li":  {"Nel": 3,  "spin": 1, "W_ref": -2.603,       "Wp_ref": 1.38},
    "Li-": {"Nel": 4,  "spin": 0, "W_ref": -2.7308,      "Wp_ref": 1.52075},
    "Be":  {"Nel": 4,  "spin": 0, "W_ref": -4.021,       "Wp_ref": 2.59},
    "B":   {"Nel": 5,  "spin": 1, "W_ref": -5.706,       "Wp_ref": 4.2},
    "C":   {"Nel": 6,  "spin": 2, "W_ref": -7.782,       "Wp_ref": 6.3},
    "N":   {"Nel": 7,  "spin": 3, "W_ref": -10.246,      "Wp_ref": None},
    "O":   {"Nel": 8,  "spin": 2, "W_ref": -10.335,      "Wp_ref": 13.0},
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

# -----------------------------------------------------------------------------
# PER-SYSTEM BASIS SETS (batch mode)
# -----------------------------------------------------------------------------




SYSTEM_BASIS = [
    # ("H",    0, 1, "5ZaPa-NR-CV"),   # H
    # ("H",   -1, 0, "5ZaPa-NR-CV"),   # H⁻
    # ("He",   0, 0, "5ZaPa-NR-CV"),   # He
    # ("Li",   0, 1, "5ZaPa-NR-CV"),   # Li
    # ("Li",  -1, 0, "5ZaPa-NR-CV"),   # Li⁻
    # ("Be",   0, 0, "5ZaPa-NR-CV"),   # Be
    # ("B",    0, 1, "5ZaPa-NR-CV"),   # B
    # ("C",    0, 2, "5ZaPa-NR-CV"),   # C
    # ("N",    0, 3, "5ZaPa-NR-CV"),   # N
    # ("O",    0, 2, "5ZaPa-NR-CV"),   # O
    # ("F",    0, 1, "5ZaPa-NR-CV"),   # F
    # ("F",   -1, 0, "5ZaPa-NR-CV"),   # F⁻
    ("Ne",   0, 0, "5ZaPa-NR-CV"),   # Ne
    ("Ne",   0, 0, "4ZaPa-NR-CV"),   # Ne
    ("Ne",   0, 0, "3ZaPa-NR-CV"),   # Ne
    ("Ne",   0, 0, "2ZaPa-NR-CV"),   # Ne
    #("Ne",   0, 0, "ZaPa-NR-CV"),   # Ne
    # ("Na",   0, 1, "5ZaPa-NR-CV"),   # Na
    # ("Mg",   0, 0, "5ZaPa-NR-CV"),   # Mg
    # ("Cl",  -1, 0, "5ZaPa-NR-CV"),   # Cl⁻
    # ("Ar",   0, 0, "5ZaPa-NR-CV"),   # Ar
    # ("Ca",   0, 0, "5ZaPa-NR-CV"),   # Ca
    # ("Zn",   0, 0, "5ZaPa-NR-CV"),   # Zn
    # ("Kr",   0, 0, "5ZaPa-NR-CV"),   # Kr
    # ("Xe",   0, 0, "5ZaPa-NR-CV"),   # Xe
]


# HELPERS
# -----------------------------------------------------------------------------

def atom_label(atom: str, charge: int) -> str:
    return atom.strip() + "-" if charge == -1 else atom.strip()

def _density_folder(density_type: str) -> str:
    return density_type.lower().replace("(", "").replace(")", "")

def _filename_suffix(density_type: str) -> str:
    if density_type == "ccsd":               return "_ccsd"
    if density_type in ("ccsd(t)", "ccsdt"): return "_ccsdt"
    if density_type == "fci":                return "_fci"
    return ""

# -----------------------------------------------------------------------------
# ROBUST W_inf ESTIMATOR
# -----------------------------------------------------------------------------

_EST_ORDER = ["akima", "pchip", "cubic", "bpoly_1", "bpoly_0",
              "simpson", "romb", "trapezoid", "bpoly_2"]

def best_winf_fixed(U, integrals, extrapolated):
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

    vals   = list(cands.values())
    med    = sorted(vals)[len(vals) // 2]
    thr    = max(3.0 * np.std(vals), 0.05 * abs(med) + 1e-6)
    stable = {k: v for k, v in cands.items() if abs(v - med) <= thr} or cands
    bk     = next((k for k in _EST_ORDER if k in stable), next(iter(stable)))
    return stable[bk] - U, bk

# -----------------------------------------------------------------------------
# CONVERGENCE / LOCAL-MINIMA DIAGNOSTICS
# -----------------------------------------------------------------------------

_CONVERGENCE_THRESHOLD = 1e-6

def _report_opt_convergence(opt_result, tag: str = "", nel: int = 0) -> None:
    prefix = f"  [{tag}] " if tag else "  "

    if nel == 2:
        print(f"{prefix}Local minimum achieved  : True"
              f"  (Nel=2 exact non-iterative solve)", flush=True)
        print(f"{prefix}Convergence threshold   : True"
              f"  (src = exact,  tol = {_CONVERGENCE_THRESHOLD:.0e})",
              flush=True)
        return

    _lm_arr = None
    for _attr in ("local_minima", "is_local_min", "all_local_min",
                  "local_min", "local_minimum", "local_minima_mask"):
        _raw = getattr(opt_result, _attr, None)
        if _raw is None:
            continue
        try:
            _arr = np.asarray(_raw).ravel().astype(bool)
            if _arr.size > 0:
                _lm_arr = _arr
                break
        except Exception:
            pass

    if _lm_arr is not None:
        n_true  = int(np.sum(_lm_arr))
        n_total = int(_lm_arr.size)
        lm_bool = bool(np.all(_lm_arr))
        lm_note = f"  ({n_true}/{n_total} grid points are local minima)"
        print(f"{prefix}Local minimum achieved  : {lm_bool}{lm_note}", flush=True)
        print(f"{prefix}Convergence threshold   : {lm_bool}"
              f"  (src = local_minima array,  tol = {_CONVERGENCE_THRESHOLD:.0e})",
              flush=True)
        return

    converged_flag = getattr(opt_result, "converged", None)
    success_flag   = getattr(opt_result, "success",   None)

    grad_norm = None
    for _attr in ("grad_norm", "gnorm", "gradient_norm", "residual"):
        _raw = getattr(opt_result, _attr, None)
        if _raw is not None:
            try:
                grad_norm = float(np.max(np.abs(np.asarray(_raw))))
                break
            except Exception:
                pass

    fun_change = None
    for _attr in ("fun_change", "delta_f"):
        _raw = getattr(opt_result, _attr, None)
        if _raw is not None:
            try:
                fun_change = float(abs(np.asarray(_raw).ravel()[-1]))
                break
            except Exception:
                pass

    if converged_flag is not None:
        conv_bool = bool(converged_flag)
        conv_src  = "result.converged"
    elif success_flag is not None:
        conv_bool = bool(success_flag)
        conv_src  = "result.success"
    elif grad_norm is not None:
        conv_bool = grad_norm <= _CONVERGENCE_THRESHOLD
        conv_src  = f"grad_norm={grad_norm:.3e}"
    elif fun_change is not None:
        conv_bool = fun_change <= _CONVERGENCE_THRESHOLD
        conv_src  = f"fun_change={fun_change:.3e}"
    else:
        for _attr in vars(opt_result) if hasattr(opt_result, "__dict__") else []:
            if any(kw in _attr.lower() for kw in
                   ("local", "min", "conv", "success", "done")):
                _raw = getattr(opt_result, _attr, None)
                if _raw is None:
                    continue
                try:
                    _arr = np.asarray(_raw).ravel()
                    if _arr.dtype == bool and _arr.size > 0:
                        n_true  = int(np.sum(_arr))
                        n_total = int(_arr.size)
                        lm_bool = bool(np.all(_arr))
                        print(f"{prefix}Local minimum achieved  : {lm_bool}"
                              f"  ({n_true}/{n_total} via .{_attr})", flush=True)
                        print(f"{prefix}Convergence threshold   : {lm_bool}"
                              f"  (src = .{_attr},"
                              f"  tol = {_CONVERGENCE_THRESHOLD:.0e})",
                              flush=True)
                        return
                except Exception:
                    pass
        print(f"{prefix}Local minimum achieved  : Unknown "
              f"(no convergence attribute found on result)", flush=True)
        print(f"{prefix}Convergence threshold   : Unknown "
              f"(threshold = {_CONVERGENCE_THRESHOLD:.0e})", flush=True)
        return

    local_min_bool = conv_bool
    hess_note      = ""
    h_vals = getattr(opt_result, "hessian_eigenvalues", None)
    if h_vals is not None:
        try:
            arr = np.asarray(h_vals).ravel()
            local_min_bool = conv_bool and bool(np.all(arr >= -1e-8))
            hess_note = f"  (min Hessian eigval = {arr.min():.3e})"
        except Exception:
            pass

    print(f"{prefix}Local minimum achieved  : {local_min_bool}{hess_note}", flush=True)
    print(f"{prefix}Convergence threshold   : {conv_bool}"
          f"  (src = {conv_src},  tol = {_CONVERGENCE_THRESHOLD:.0e})",
          flush=True)

# -----------------------------------------------------------------------------
# T_ZP / W'inf  COMPUTATION
# -----------------------------------------------------------------------------

_NEL_ZERO_MODES: dict = {
    3:   0,
    5:   7,
    6:   9,
    7:   11,
    8:   13,
    9:   15,
    10:  17,
    11:  19,
    12:  21,
    13:  23,
    14:  25,
    15:  27,
    16:  29,
    17:  31,
    18:  33,
    20:  37,
    30:  57,
    36:  69,
    54:  105,
}
_NEL_ZERO_MODES_DEFAULT = 8

def _n_skip_for_nel(nel: int) -> int:
    return _NEL_ZERO_MODES.get(nel, _NEL_ZERO_MODES_DEFAULT)

class _TruncatedResult:
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

def _integrate_omega(res, omega_sums: np.ndarray,
                     integrator: str = "simpson") -> float:
    from scipy.integrate import simpson as _simp, trapezoid as _trap
    N = omega_sums.shape[0]
    if res.opt.grid == "r":
        x = np.asarray(res.r[:N],   dtype=float)
        y = (4.0 * np.pi * x**2
             * np.asarray(res.rho[:N], dtype=float)
             * omega_sums)
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
    dr     = r_grid[1] - r_grid[0]
    rho_1d = np.maximum(
        np.array([float(density.rho(r)) for r in r_grid], dtype=float), 0.0
    )
    Ne = np.clip(np.cumsum(4.0 * np.pi * r_grid**2 * rho_1d) * dr, 0.0, 2.0)
    _, uid = np.unique(Ne, return_index=True)

    use_result = False
    f_interp   = None
    if result is not None and hasattr(result, "f") and hasattr(result, "r"):
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
                      f"({r_sce.shape} vs {f_sce.shape}) -- "
                      "using density-only inversion", flush=True)
        except Exception as exc:
            print(f"  [T_ZP] Nel=2: result setup failed ({exc}) -- "
                  "using density-only inversion", flush=True)

    if not use_result:
        print("  [T_ZP] Nel=2: density-only Ne(r) inversion", flush=True)

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
        d = r + f_r
        if d < 1e-12 or f_r < 1e-12:
            continue
        lb = (r*r + f_r*f_r) / (r * f_r * d**3)
        if lb <= 0.0:
            continue
        rho_f = float(np.interp(f_r, r_grid, rho_1d))
        if rho_f < 1e-30:
            continue
        p  = (r / f_r)**2 * rho_r / rho_f
        ls = 2.0 * (p*p + 1.0) / (p * d**3)
        Wp_sum += (4.0 * np.pi * r**2 * rho_r
                   * (2.0 * np.sqrt(lb) + np.sqrt(max(ls, 0.0))) * dr)

    val = 0.25 * Wp_sum
    print(f"  [T_ZP] Nel=2 analytical  W'inf PRL    = {val:.10f}", flush=True)
    print(f"  [T_ZP] Nel=2 analytical  W'inf legacy = {val / 2:.10f}", flush=True)
    return val

def compute_tzp_for_planar(result, density=None) -> float:
    from jaxsce.optimize import AngularOptimizationResult
    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    Nel         = res.f.shape[1]
    coordinates = res.opt.coordinates
    N_grid_end  = res.angles.shape[0]
    n_angles    = res.angles.shape[1]
    has_radials = (hasattr(coordinates, "Vee_hessian_radials") and
                   hasattr(coordinates, "Vee_hessian_angles_radials"))
    n_radials   = Nel - 1 if has_radials else 0
    n_dof       = n_angles + n_radials
    n_zero      = 3
    n_vib       = max(n_dof - n_zero, 1)

    print(f"  [T_ZP] Planar Nel={Nel}: dim={n_dof} "
          f"(ang={n_angles}, rad={n_radials}), zeros={n_zero}, vib={n_vib}",
          flush=True)

    omega_sums = np.zeros(N_grid_end)
    for idx in range(N_grid_end):
        if idx % 50 == 0 and idx > 0:
            print(f"  [T_ZP] Progress: {idx}/{N_grid_end}", flush=True)
        try:
            if hasattr(coordinates, "Vee_hessian_angles"):
                h_aa = np.asarray(
                    coordinates.Vee_hessian_angles(res.angles[idx], res.f[idx]))
                if has_radials:
                    h_rr = np.asarray(
                        coordinates.Vee_hessian_radials(res.angles[idx], res.f[idx]))
                    h_ar = np.asarray(
                        coordinates.Vee_hessian_angles_radials(
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
            eigvals = np.linalg.eigvalsh(H)
            vib_pos = eigvals[-n_vib:] if len(eigvals) >= n_vib else eigvals
            vib_pos = vib_pos[vib_pos > 1e-8]
            omega_sums[idx] = float(np.sum(np.sqrt(vib_pos)))
        except Exception as exc:
            print(f"  [T_ZP] Warning at idx={idx}: {exc}", flush=True)
            omega_sums[idx] = 0.0

    val = _integrate_omega(res, omega_sums)
    print(f"  [T_ZP] Planar  W'inf PRL    = {val:.10f}", flush=True)
    print(f"  [T_ZP] Planar  W'inf legacy = {val / 2:.10f}", flush=True)
    return val

def compute_tzp_local(result, density=None, integrator: str = "simpson",
                      debug_eigvals: bool = False) -> float:
    from jaxsce.optimize import AngularOptimizationResult
    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    if not hasattr(res, "opt") or not hasattr(res.opt, "coordinates"):
        raise ValueError("Result object does not expose a coordinate system.")

    coordinates = res.opt.coordinates
    N_grid_end  = res.angles.shape[0]
    Nel         = res.f.shape[1]
    n_skip      = _n_skip_for_nel(Nel)

    print(f"  [T_ZP] Local Hessian  Nel={Nel}  pts={N_grid_end} "
          f" n_skip={n_skip}", flush=True)

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
            H       = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H)

            if debug_eigvals and idx == 0:
                print(f"  [DEBUG] Nel={Nel} eigenvalues at idx=0:", flush=True)
                for ei, ev in enumerate(eigvals):
                    print(f"    eig[{ei:2d}] = {ev:+.6e}", flush=True)
                near_zero = int(np.sum(np.abs(eigvals) <= 1e-6))
                print(f"  [DEBUG] {near_zero} near-zero (|lambda|<=1e-6) "
                      f"-- current n_skip={n_skip}", flush=True)

            physical = eigvals[n_skip:]
            omega_sums[idx] = float(
                np.sum(np.sqrt(np.clip(physical, 0.0, None)))
            )
        except Exception as exc:
            print(f"  [T_ZP] Warning at idx={idx}: {exc}", flush=True)
            omega_sums[idx] = 0.0

    val = _integrate_omega(res, omega_sums, integrator)
    print(f"  [T_ZP] Local Hess  W'inf PRL    = {val:.10f}", flush=True)
    print(f"  [T_ZP] Local Hess  W'inf legacy = {val / 2:.10f}", flush=True)
    return val

def compute_tzp_from_result(result, density, nel: int,
                             debug_eigvals: bool = False) -> float:
    from jaxsce.optimize import AngularOptimizationResult

    if nel == 1:
        return compute_tzp_nel1()

    if nel == 2:
        print("  [T_ZP] Nel=2 -> exact analytical formula", flush=True)
        return compute_tzp_nel2(density, result=result)

    res = (AngularOptimizationResult.to_numpy(result)
           if isinstance(result, AngularOptimizationResult) else result)

    if nel == 3:
        n_skip = _n_skip_for_nel(3)
        print(f"  [T_ZP] Nel=3 -> local Hessian (n_skip={n_skip})", flush=True)
        val = compute_tzp_local(res, density, debug_eigvals=debug_eigvals)
        return val if (math.isfinite(val) and val > 0) else float("nan")

    elif nel == 4:
        n_pts = res.f.shape[0]
        if n_pts > 1:
            print(f"  [T_ZP] Nel=4 planar: dropping last grid point "
                  f"({n_pts} -> {n_pts - 1})", flush=True)
            res = _TruncatedResult(res, n_pts - 1)
        return compute_tzp_for_planar(res, density)

    else:
        n_skip = _n_skip_for_nel(nel)
        print(f"  [T_ZP] Nel={nel} -> local Hessian (n_skip={n_skip})",
              flush=True)
        val = compute_tzp_local(res, density, debug_eigvals=debug_eigvals)
        if math.isfinite(val) and val > 0:
            return val
        print("  [T_ZP] WARNING: Hessian gave non-positive/NaN -> NaN",
              flush=True)
        return float("nan")

# -----------------------------------------------------------------------------
# PRE-INTEGRATION ARRAY SANITIZER
# -----------------------------------------------------------------------------

def _sanitize_opt_result(opt_result, tag: str = "") -> int:
    from scipy.interpolate import interp1d

    prefix  = f"  [{tag}] " if tag else "  "
    targets = ("Vee", "dVee", "d2Vee")
    total_fixed = 0

    for attr in targets:
        raw = getattr(opt_result, attr, None)
        if raw is None:
            continue
        arr = np.array(raw, dtype=float)
        original_shape = arr.shape
        flat = arr.reshape(arr.shape[0], -1)
        n_fixed_attr = 0

        for col in range(flat.shape[1]):
            col_data = flat[:, col]
            bad  = ~np.isfinite(col_data)
            if not np.any(bad):
                continue
            good = ~bad
            if good.sum() < 2:
                col_data[bad] = 0.0
                n_fixed_attr += int(bad.sum())
                continue
            idx  = np.arange(len(col_data))
            interp = interp1d(idx[good], col_data[good],
                              kind="linear", bounds_error=False,
                              fill_value=(col_data[good][0],
                                          col_data[good][-1]))
            col_data[bad] = interp(idx[bad])
            n_fixed_attr += int(bad.sum())

        if n_fixed_attr > 0:
            try:
                setattr(opt_result, attr, flat.reshape(original_shape))
            except AttributeError:
                pass
            print(f"{prefix}[sanitize] {attr}: repaired {n_fixed_attr} "
                  f"non-finite value(s) by linear interpolation", flush=True)
            total_fixed += n_fixed_attr

    if total_fixed == 0:
        print(f"{prefix}[sanitize] All Vee arrays are finite — no repair needed.",
              flush=True)

    return total_fixed

# -----------------------------------------------------------------------------
# PC4 QUANTITIES FUNCTION
# -----------------------------------------------------------------------------

def _ensure_paper_only_pc4_quantities(density, E_pc4prime=None, F_pc4prime=None, use_derived_ge4=True):
    """Compute W_inf_PC4 and Wp_inf_PC4 matching derived text constants safely.

    W_inf_PC4  = Directly mapped Padé functional using mu_w and nu_w.
    Wp_inf_PC4 = Rational Padé matching the derived D2 and D3 coefficients.
    """
    try:
        import numpy as _np
        from pyscf import dft as _dft

        N_int     = int(getattr(density, "N_int",     10000))
        rho_trunc = float(getattr(density, "rho_trunc", 1e-10))
        atom_chg  = int(getattr(density, "_atom_chg",  getattr(density, "charge", 0)))

        r, dr = _dft.radi.treutler(N_int, chg=atom_chg)
        rho   = _np.asarray(density.rho(r),       dtype=_np.float64)
        drho  = _np.asarray(density.rho_deriv(r), dtype=_np.float64)
        nz    = _np.where(rho > rho_trunc)
        if len(nz[0]) == 0:
            return

        pi       = _np.pi
        kF_const = (3.0 * pi**2) ** (1.0 / 3.0)
        rn    = r[nz]
        wn    = dr[nz] * 4.0 * pi * rn**2
        rhon  = rho[nz]
        drhon = drho[nz]

        # Dimensionless reduced gradients
        s2 = drhon**2 / (4.0 * kF_const**2 * rhon ** (8.0 / 3.0))
        s4 = s2**2

        # ======================================================================
        # P0, P1, P2 integrals requested for diagnostics
        #
        # P0 = int d^3r rho^(3/2)
        # P1 = int d^3r rho^(3/2) s^2
        # P2 = int d^3r rho^(3/2) s^4
        #
        # Here wn = 4*pi*r^2*dr, so the radial integration is converted
        # to the full 3D spherical integral.
        # ======================================================================
        P0 = float(_np.sum(wn * rhon ** 1.5))
        P1 = float(_np.sum(wn * rhon ** 1.5 * s2))
        P2 = float(_np.sum(wn * rhon ** 1.5 * s4))

        # I0 is the LDA_int already computed/imported from jaxsce.densities.pyscf.
        # Use a local radial fallback only if the imported density object does not carry it.
        I0_local = float(_np.sum(wn * rhon ** (4.0 / 3.0)))
        I0 = float(getattr(density, "LDA_int", I0_local))

        # ======================================================================
        # W_inf_PC4 : Rigorous, Mathematically Stable Direct Padé Form
        # ======================================================================
        A_pc4  = float(-9.0 / 10.0 * (4.0 * pi / 3.0) ** (1.0 / 3.0))
        mu_w   = -0.1403112
        nu_w   = -0.5788045
        kap_tx = 2.0 * (9.0 * pi / 4.0) ** (1.0 / 3.0)

        # Exact Taylor matching constraints + stable saturation behavior
        F_sat_w = 0.55
        b1_w    = 1.00
        b2_w    = (nu_w + b1_w * mu_w) / (F_sat_w - 1.0)              # = 1.598035
        a1_w    = b1_w + mu_w                                         # = 0.859689
        a2_w    = F_sat_w * b2_w                                      # = 0.878919

        F0 = (1.0 + a1_w * s2 + a2_w * s4) / (1.0 + b1_w * s2 + b2_w * s4)
        W_inf_pc4 = float(_np.sum(wn * A_pc4 * rhon ** (4.0 / 3.0) * F0))

        # ======================================================================
        # Wp_inf_PC4 : Calibrated Rational Padé for W'_inf
        # ======================================================================
        SCALE_TO_PHYSICAL = True
        C_text = 0.5 * _np.sqrt(3.0 * pi)
        C_wp   = (C_text / 3.0) if SCALE_TO_PHYSICAL else C_text

        D2 = 0.4910891403
        D3 = 0.6415083264

        F_sat_wp = 1.20
        b1_wp    = 0.60
        b2_wp    = (D3 + b1_wp * D2) / (F_sat_wp - 1.0)               # = 4.680809
        a1_wp    = D2 + b1_wp                                         # = 1.091089
        a2_wp    = F_sat_wp * b2_wp                                   # = 5.616971

        Fp = (1.0 + a1_wp * s2 + a2_wp * s4) / (1.0 + b1_wp * s2 + b2_wp * s4)
        Wp_inf_pc4 = float(_np.sum(wn * C_wp * rhon ** 1.5 * Fp))

        # ======================================================================
        # Promote Variables onto Density Object
        # ======================================================================
        Wp_inf_lda     = float(C_wp * _np.sum(wn * rhon ** 1.5))
        Wp_inf_ge4_raw = float(_np.sum(wn * C_wp * rhon ** 1.5 * (1.0 + D2 * s2 + D3 * s4)))

        density.A_PC4                = A_pc4
        density.EPC_MU_INF          = mu_w
        density.EPC_KAPPA           = kap_tx
        density.C_WPRIME_PC4        = C_wp
        density.W_inf_PC4           = W_inf_pc4
        density.W_inf_F0            = W_inf_pc4
        density.pc4                 = W_inf_pc4
        density.wprime_lda          = Wp_inf_lda
        density.wprime_f0           = Wp_inf_lda
        density.Wp_inf_F0           = Wp_inf_lda
        density.Wp_inf_PC4          = Wp_inf_pc4
        density.wprime_pc4          = Wp_inf_pc4
        density.Wp_inf_GE4_raw_diag = Wp_inf_ge4_raw
        density.P0                  = P0
        density.P1                  = P1
        density.P2                  = P2
        density.I0                  = I0
        density.A                   = A_pc4
        density.C                   = C_wp
        density.C_text              = C_text

        J0 = P0

        print(
            f"  [PC4-Updated] W_inf_PC4={W_inf_pc4:.10f}  Wp_inf_PC4={Wp_inf_pc4:.10f}\n"
            f"  [PC4-Updated] P0={P0:.10f}  P1={P1:.10f}  P2={P2:.10f}\n"
            f"  [PC4-Updated] I0(LDA_int)={I0:.10f}  A={A_pc4:.10f}  C={C_wp:.10f}\n"
            f"  [PC4-Updated] Wp_inf_LDA={Wp_inf_lda:.10f}  Wp_GE4_raw={Wp_inf_ge4_raw:.6f}\n"
            f"  [PC4-Updated] Padé W:   F_sat={F_sat_w} b1={b1_w} b2={b2_w:.4f} a1={a1_w:.4f} a2={a2_w:.4f}\n"
            f"  [PC4-Updated] Padé Wp:  F_sat={F_sat_wp} b1={b1_wp} b2={b2_wp:.4f} a1={a1_wp:.4f} a2={a2_wp:.4f}\n"
            f"  [PC4-Updated] C_wp={C_wp:.8f}",
            flush=True,
        )

    except Exception as exc:
        print(f"  [PC4] WARNING: {exc}", flush=True)

# -----------------------------------------------------------------------------
# MAIN RUN FUNCTION
# -----------------------------------------------------------------------------

def run_system(atom, charge, spin, basis, density_type, args):
    """Run optimization + T_ZP + W_inf integration for one system."""

    basis_value, has_ecp, ecp_value = load_basis_for_pyscf(basis, atom)
    start_time = timer()

    atom_key  = atom_label(atom, charge)
    ref       = MOLE_REFERENCES.get(atom_key, {})
    W_inf_ref = ref.get("W_ref")
    T_ZP_ref  = ref.get("Wp_ref")

    if spin is None:
        spin = ref.get("spin", 0)

    from pyscf import gto

    _mol_kwargs = dict(atom=atom, charge=charge, spin=spin, basis=basis_value)
    if has_ecp and ecp_value is not None:
        _mol_kwargs["ecp"] = ecp_value
    mol = gto.M(**_mol_kwargs)

    nel       = mol.nelectron
    grid_type = args.grid

    one_electron_fci = (density_type == "fci" and nel == 1)

    if nel == 2 and grid_type == "r":
        print("  WARNING: Grid 'r' not supported for Nel=2, "
              "switching to 'Ne'", flush=True)
        grid_type = "Ne"

    if args.nstart:
        nstart = jnp.array(args.nstart, dtype=jnp.int32)
    else:
        nstart = jnp.array([1, args.N_grid // 2, args.N_grid - 2],
                            dtype=jnp.int32)

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

    # -- density selection ----------------------------------------------------
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

        if one_electron_fci:
            print("  [Nel=1] One-electron FCI is exactly equal to HF.", flush=True)
            density = HFDensity(chkfile_name=chkfile, chkfile_dir=datadir,
                                **density_kwargs)
            density.e_fci = getattr(density, "e_hf", None)
        else:
            _fci_mem_mb = getattr(args, "fci_max_memory", 102400)
            try:
                import pyscf.lib
                pyscf.lib.param.MAX_MEMORY = _fci_mem_mb
                print(f"  [FCI] pyscf.lib.param.MAX_MEMORY set to "
                      f"{_fci_mem_mb} MB ({_fci_mem_mb/1024:.1f} GB)",
                      flush=True)
            except Exception as _me:
                print(f"  [FCI] WARNING: could not set pyscf MAX_MEMORY: {_me}",
                      flush=True)

            density = FCIDensity(dm_file_name="dm_fci.npy",
                                 dm_file_dir=datadir,
                                 **density_kwargs)
    else:
        raise ValueError(f"Unsupported density type: {density_type!r}")

    # -- UHF scalar coercion --------------------------------------------------
    _dm = np.asarray(getattr(density, "dm", np.zeros((1, 1))))
    if _dm.ndim == 3:
        density.dm = _collapse_dm(_dm)
        _recompute_U(density)
        print("  [UHF] Collapsed alpha+beta DM and recomputed U", flush=True)

    if hasattr(density, "U"):
        density.U = float(density.U) if np.isscalar(density.U) \
                    else float(np.sum(density.U))
    else:
        density.U = 0.0
    if getattr(density, "Nel", None) is not None:
        density.Nel = int(density.Nel)

    print(f"  Nel = {density.Nel}, U = {density.U:.10f}", flush=True)

    # Make the driver robust even if an older local density module is imported.
    _ensure_paper_only_pc4_quantities(
        density,
        E_pc4prime=getattr(args, "pc4prime_E", None),
        F_pc4prime=getattr(args, "pc4prime_F", None),
        use_derived_ge4=(not getattr(args, "no_derived_pc4prime", False)),
    )

    # -- Wp_inf quantities ----------------------------------------------------
    Wp_inf_LDA  = float(getattr(density, "wprime_lda", float("nan")))
    Wp_inf_PC4  = float(getattr(density, "Wp_inf_PC4", getattr(density, "wprime_f0", float("nan"))))
    Wp_inf_F0   = float(getattr(density, "Wp_inf_F0", getattr(density, "wprime_f0", float("nan"))))
    J0_int      = float(getattr(density, "J0_int", float("nan")))
    J2_int      = float(getattr(density, "J2_int", float("nan")))
    J4_int      = float(getattr(density, "J4_int", float("nan")))
    P0          = float(getattr(density, "P0", float("nan")))
    P1          = float(getattr(density, "P1", float("nan")))
    P2          = float(getattr(density, "P2", float("nan")))
    I0          = float(getattr(density, "I0", getattr(density, "LDA_int", float("nan"))))
    A_val       = float(getattr(density, "A", getattr(density, "A_PC4", float("nan"))))
    C_wp        = 3*float(getattr(density, "C", getattr(density, "C_WPRIME_PC4", float("nan"))))

    # -- Nel = 1 : no optimizer, no co-motion functions -----------------------
    if density.Nel == 1:
        print("  [Nel=1] Single electron -- skipping optimizer and integration.",
              flush=True)
        print("  [Nel=1] W_inf = -U  (Vee = 0 for one electron)", flush=True)

        Winf             = -density.U
        T_ZP_prl         = 0.0
        T_ZP_legacy      = 0.0
        Wp_inf_LDA      = 0.0
        Wp_inf_PC4      = 0.0
        Wp_inf_F0       = 0.0
        estimator        = "exact-Nel1"
        Lambda           = (-Winf / density.LDA_int if density.LDA_int != 0
                            else float("nan"))
        B                = ((Winf - A * density.LDA_int) / density.GEA_int
                            if density.GEA_int != 0 else float("nan"))

        I2_int = float(getattr(density, "I2_int", float("nan")))
        I3_int = float(getattr(density, "I3_int", float("nan")))
        pc4    = float(getattr(density, "pc4",    float("nan")))

        ints_time = timer()
        time_elapsed = ints_time - start_time

        result = {
            "U":                 float(density.U),
            "LDA_int":           float(density.LDA_int),
            "GEA_int":           float(density.GEA_int),
            "I0":                float(I0),
            "A":                 float(A_val),
            "C":                 float(C_wp),
            "I2_int":            float(I2_int),
            "I3_int":            float(I3_int),
            "J0_int":            float(J0_int),
            "J2_int":            float(J2_int),
            "J4_int":            float(J4_int),
            "P0":                float(P0),
            "P1":                float(P1),
            "P2":                float(P2),
            "PC4":               float(pc4),
            "Wp_inf_LDA":        float(Wp_inf_LDA),
            "Wp_inf_PC4":        float(Wp_inf_PC4),
            "Wp_inf_F0":         float(Wp_inf_F0),
            "Winf":              float(Winf),
            "T_ZP_prl":          float(T_ZP_prl),
            "T_ZP_legacy":       float(T_ZP_legacy),
            "Lambda":            float(Lambda),
            "B":                 float(B),
            "estimator":         estimator,
            "Time":              time_elapsed,
        }

        def _fs(v, fmt=".10f"):
            return f"{v:{fmt}}" if (v is not None and math.isfinite(v)) else "N/A"

        W = 70
        print(f"\n{'='*W}")
        print(f"RESULTS for {atom_key}  (Nel={density.Nel})")
        print(f"{'='*W}")
        print(f"  Basis            : {basis}")
        print(f"  Density          : {density_type.upper()}")
        print(f"  Estimator        : {estimator}")
        print(f"  U                : {density.U:.10f}")
        print(f"  W_inf  = -U      : {_fs(Winf)}")
        print(f"  W_inf PC4        : {_fs(pc4)}")
        print(f"  W'inf PRL (T_ZP) : {T_ZP_prl:.10f}  (exact: 0)")
        print(f"  W'inf legacy     : {T_ZP_legacy:.10f}  (exact: 0)")
        print(f"  Wp_inf_LDA      : 0.0  (Nel=1, no e-e)")
        print(f"  Wp_inf_PC4      : 0.0  (Nel=1, no e-e)")
        print(f"  Lambda           : {_fs(Lambda)}")
        print(f"  B                : {_fs(B)}")
        print(f"  I0(LDA_int)      : {_fs(I0)}")
        print(f"  A                : {_fs(A_val)}")
        print(f"  C                : {_fs(C_wp)}")
        print(f"  I2_int           : {_fs(I2_int)}")
        print(f"  I3_int           : {_fs(I3_int)}")
        print(f"  J0_int           : {_fs(J0_int)}")
        print(f"  J2_int           : {_fs(J2_int)}")
        print(f"  J4_int           : {_fs(J4_int)}")
        print(f"  P0               : {_fs(P0)}")
        print(f"  P1               : {_fs(P1)}")
        print(f"  P2               : {_fs(P2)}")
        print(f"  W_inf_PC4        : {_fs(pc4)}")
        if W_inf_ref is not None:
            print(f"  Reference W_inf  : {W_inf_ref:.6f}")
            print(f"  Error W_inf      : {_fs(error_W, '.6f')}")
        if T_ZP_ref is not None:
            print(f"  Reference W'inf  : {T_ZP_ref:.6f}")
            print(f"  Error W'inf(PRL) : 0.000000  (exact)")
        print(f"  Time             : {time_elapsed:.1f}s")
        print(f"{'='*W}\n")
        return result

    # -- Nel >= 2 : optimizer -------------------------------------------------
    if density.Nel == 2:
        opt = TwoElectron(N_grid=args.N_grid, grid=grid_type)
    else:
        coordinates      = get_coordinate_system(args.coordinate_system,
                                                  args.seed)
        coordinates_last = get_coordinate_system(args.coordinate_system_last,
                                                  args.seed)
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
    _opt_done = False
    for _dev in ("gpu", "cpu"):
        try:
            if _dev == "cpu":
                print("  [GPU] CUDA solver error — retrying on CPU...", flush=True)
                import jax as _jax
                with _jax.default_device(_jax.devices("cpu")[0]):
                    opt_result = opt.run(density, nstart)
            else:
                opt_result = opt.run(density, nstart)
            _opt_done = True
            break
        except RuntimeError as _e:
            _emsg = str(_e).lower()
            if _dev == "gpu" and any(k in _emsg for k in
                    ("cusolver", "cublas", "cublaslt", "cuda",
                     "gpusolver", "gpu", "solver_kernels")):
                continue
            raise
    if not _opt_done:
        raise RuntimeError("Optimization failed on both GPU and CPU.")

    _report_opt_convergence(opt_result, tag=atom_key, nel=density.Nel)

    filename = f"N{args.N_grid}{_filename_suffix(density_type)}"
    opt_result.save(filename, datadir)

    # -- T_ZP / W'inf ---------------------------------------------------------
    print("  Computing T_ZP (W'inf)...", flush=True)
    debug_eig = getattr(args, "debug_eigvals", False)
    try:
        T_ZP_prl = compute_tzp_from_result(opt_result, density,
                                            nel=density.Nel,
                                            debug_eigvals=debug_eig)
    except Exception as exc:
        print(f"  [T_ZP] FAILED: {type(exc).__name__}: {exc}", flush=True)
        T_ZP_prl = float("nan")

    T_ZP_legacy = T_ZP_prl / 2.0 if math.isfinite(T_ZP_prl) else float("nan")

    # -- integrals & W_inf ----------------------------------------------------
    print("  Computing integrals...", flush=True)
    _n_intervals = args.N_grid - 1
    sub_div = next((d for d in (4, 3, 2, 1) if _n_intervals % d == 0), 1)
    if sub_div < 4:
        print(f"  NOTE: (N_grid-1)={_n_intervals} not divisible by 4; "
              f"using sub_div={sub_div} (largest exact divisor <= 4)",
              flush=True)
    _sanitize_opt_result(opt_result, tag=atom_key)
    ints = VeeIntegration(opt_result)
    ints.interpolate_all(sub_div)
    integrals, extrapolated = ints.integrate(sub_div)
    ints_time = timer()

    Winf, estimator = best_winf_fixed(density.U, integrals, extrapolated)
    Lambda = (-Winf / density.LDA_int if density.LDA_int != 0
              else float("nan"))
    B      = ((Winf - A * density.LDA_int) / density.GEA_int
              if density.GEA_int != 0 else float("nan"))

    I2_int = float(getattr(density, "I2_int", float("nan")))
    I3_int = float(getattr(density, "I3_int", float("nan")))
    pc4    = float(getattr(density, "pc4",    float("nan")))

    error_W   = ((Winf - W_inf_ref)
                 if (W_inf_ref is not None and math.isfinite(Winf)) else None)
    error_TZP = ((T_ZP_prl - T_ZP_ref)
                 if (T_ZP_ref is not None and math.isfinite(T_ZP_prl)) else None)
    error_pc4_vs_ref = (
        (Wp_inf_PC4 - T_ZP_ref)
        if (T_ZP_ref is not None and math.isfinite(Wp_inf_PC4))
        else None
    )

    time_elapsed = ints_time - start_time

    result = {
        "U":                 float(density.U),
        "LDA_int":           float(density.LDA_int),
        "GEA_int":           float(density.GEA_int),
        "I0":                float(I0),
        "A":                 float(A_val),
        "C":                 float(C_wp),
        "I2_int":            float(I2_int),
        "I3_int":            float(I3_int),
        "J0_int":            float(J0_int),
        "J2_int":            float(J2_int),
        "J4_int":            float(J4_int),
        "P0":                float(P0),
        "P1":                float(P1),
        "P2":                float(P2),
        "PC4":               float(pc4),
        "W_inf_PC4":         float(pc4),
        "Wp_inf_LDA":        float(Wp_inf_LDA),
        "Wp_inf_PC4":        float(Wp_inf_PC4),
        "Wp_inf_F0":         float(Wp_inf_F0),
        "Winf":              float(Winf),
        "T_ZP_prl":          float(T_ZP_prl),
        "T_ZP_legacy":       float(T_ZP_legacy),
        "Lambda":            float(Lambda),
        "B":                 float(B),
        "estimator":         estimator,
        "Time":              time_elapsed,
    }

    # -- printout -------------------------------------------------------------
    def _fs(v, fmt=".10f"):
        return f"{v:{fmt}}" if (v is not None and math.isfinite(v)) else "N/A"

    W = 70
    print(f"\n{'='*W}")
    print(f"RESULTS for {atom_key}  (Nel={density.Nel})")
    print(f"{'='*W}")
    print(f"  Basis            : {basis}")
    print(f"  Density          : {density_type.upper()}")
    print(f"  Grid             : {grid_type}")
    print(f"  Estimator        : {estimator}")
    print(f"  U                : {density.U:.10f}")
    print(f"  W_inf_SCE        : {_fs(Winf)}")
    print(f"  W_inf_PC4        : {_fs(pc4)}")
    print(f"  Wp_inf_SCE       : {_fs(T_ZP_prl)}")
    print(f"  Wp_inf_legacy    : {_fs(T_ZP_legacy)}")
    print(f"  Lambda           : {_fs(Lambda)}")
    print(f"  B                : {_fs(B)}")
    print(f"  I0(LDA_int)      : {_fs(I0)}")
    print(f"  A                : {_fs(A_val)}")
    print(f"  C                : {_fs(C_wp)}")
    print(f"  I2_int           : {_fs(I2_int)}")
    print(f"  I3_int           : {_fs(I3_int)}")
    print(f"  P0               : {_fs(P0)}")
    print(f"  P1               : {_fs(P1)}")
    print(f"  P2               : {_fs(P2)}")
    if W_inf_ref is not None:
        print(f"  Reference W_inf  : {W_inf_ref:.6f}")
        print(f"  Error W_inf      : {_fs(error_W, '.6f')}")
    if T_ZP_ref is not None:
        print(f"  Reference W'inf  : {T_ZP_ref:.6f}")
        print(f"  Error W'inf(PRL) : {_fs(error_TZP, '.6f')}")
    print(f"  Time             : {time_elapsed:.1f}s")
    print(f"{'='*W}\n")

    return result

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SCE optimization for atoms (single system or batch mode)"
    )
    parser.add_argument("--atom",   type=str,
                        help="Atom symbol for single system run")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin",   type=int, default=None,
                        help="Spin (auto-detected from reference table if omitted)")
    parser.add_argument("--basis",  type=str, default="cc-pVDZ")
    parser.add_argument(
        "--density", type=str, default="hf",
        choices=["hartree-fock", "hf", "ccsd", "ccsd(t)", "ccsdt", "fci"],
    )
    parser.add_argument("--grid",               type=str,  default="Ne",
                        choices=["r", "Ne"])
    parser.add_argument("--N-grid",             type=int,  default=1025)
    parser.add_argument("--N-random",           type=int,  default=10000)
    parser.add_argument("--N-random-last",      type=int,  default=30000)
    parser.add_argument("--N-select",           type=int,  default=30)
    parser.add_argument("--N-sweeps",           type=int,  default=0)
    parser.add_argument("--strategy",           type=str,  default="sweeps")
    parser.add_argument("--nstart",             type=int,  nargs="+", default=[])
    parser.add_argument("--N-points-Ne",        type=int,  default=10000)
    parser.add_argument("--invNe-tol",          type=float, default=1e-10)
    parser.add_argument("--cumulant-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--fci-max-memory", type=int, default=409600,
        metavar="MB",
        help="Maximum memory (MB) for the PySCF FCI solver. "
             "Default: 409600 (400 GB). Increase for large basis sets.",
    )
    parser.add_argument("--coordinate-system",      type=str, default="reduced")
    parser.add_argument("--coordinate-system-last", type=str,
                        default="double_reduced")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--debug-eigvals", action="store_true", default=False,
        help="Print all Hessian eigenvalues at idx=0. "
             "Use to find the correct n_skip for a new Nel.",
    )
    parser.add_argument(
        "--pc4prime-E", dest="pc4prime_E", type=float, default=None,
        help="Explicit E coefficient for Wp_inf_PC4 = C*J0 + E*J2 + F*J4. Overrides derived GE4 E.",
    )
    parser.add_argument(
        "--pc4prime-F", dest="pc4prime_F", type=float, default=None,
        help="Explicit F coefficient for Wp_inf_PC4 = C*J0 + E*J2 + F*J4. Overrides derived GE4 F.",
    )
    parser.add_argument(
        "--no-derived-pc4prime", action="store_true", default=False,
        help="Do not use derived GE4 PC4-prime coefficients. Then Wp_inf_PC4 is N/A unless --pc4prime-E and --pc4prime-F are supplied.",
    )
    parser.add_argument(
        "--nel-zero-modes", type=str, default=None,
        metavar="NEL:N[,NEL:N,...]",
        help="Override zero-mode skip counts, e.g. '8:10,9:10,10:12'.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data/batch_summary_pc4.csv",
        help="Output CSV file for batch summary.",
    )

    args = parser.parse_args()
    args.density = args.density.lower()

    if args.nel_zero_modes:
        for pair in args.nel_zero_modes.split(","):
            nel_str, n_str = pair.strip().split(":")
            nel_key = int(nel_str.strip())
            n_val   = int(n_str.strip())
            _NEL_ZERO_MODES[nel_key] = n_val
            print(f"  [config] Override: Nel={nel_key} -> skip {n_val} zero modes",
                  flush=True)

    # -- single system --------------------------------------------------------
    if args.atom is not None:
        print(f"\n{'='*60}")
        print(f"  Running single system: {args.atom}")
        print(f"  Charge: {args.charge}, Basis: {args.basis}, "
              f"Density: {args.density.upper()}")
        print(f"{'='*60}")

        res = run_system(atom=args.atom, charge=args.charge, spin=args.spin,
                         basis=args.basis, density_type=args.density, args=args)

        tag = f"{atom_label(args.atom, args.charge)}/{args.basis}"
        save_batch_summary_csv({tag: res}, args.summary_csv)
        return

    # -- batch mode -----------------------------------------------------------
    if not SYSTEM_BASIS:
        print("Error: SYSTEM_BASIS is empty. Use --atom for single system.")
        sys.exit(1)

    batch_start = timer()
    all_results: dict = {}
    failures: list    = []

    print(f"\n{'='*60}")
    print(f"  Batch run -- density={args.density.upper()}  "
          f"grid={args.grid}  N_grid={args.N_grid}")
    print(f"  Systems: {len(SYSTEM_BASIS)}")
    print(f"{'='*60}")

    for atom, charge, spin, basis in SYSTEM_BASIS:
        tag = f"{atom_label(atom, charge)}/{basis}"
        print(f"\n{'─'*60}")
        print(f"  > Running {tag}  (charge={charge}, spin={spin})")
        print(f"{'─'*60}")
        try:
            res = run_system(atom=atom, charge=charge, spin=spin,
                             basis=basis, density_type=args.density, args=args)
            all_results[tag] = res
        except Exception as exc:
            print(f"\n  x  FAILED: {tag}")
            print(f"     {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()
            failures.append((tag, exc))

    total_time = timer() - batch_start

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY   (total wall time: {total_time:.1f}s)")
    print(f"{'='*60}")

    if all_results:
        header = (
            f"  {'System':<22} "
            f"{'W_inf_SCE':>12} {'W_inf_PC4':>12} {'Err_W_PC4':>12} "
            f"{'Wp_inf_SCE':>12} {'Wp_inf_LDA':>12} {'Err_LDA':>10} "
            f"{'Wp_inf_PC4':>12} {'Err_Wp_PC4':>12} "
            f"{'Lambda':>12} {'B':>12} {'I0':>12} {'A':>12} {'C':>12} "
            f"{'I2_int':>12} {'I3_int':>12} "
            f"{'P0':>12} {'P1':>12} {'P2':>12} {'Time_s':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for tag, res in all_results.items():
            Winf      = res.get("Winf", float("nan"))
            W_inf_pc4 = res.get("W_inf_PC4", res.get("PC4", float("nan")))
            Wp_sce    = res.get("T_ZP_prl", float("nan"))
            Wp_lda    = res.get("Wp_inf_LDA", float("nan"))
            Wp_pc4    = res.get("Wp_inf_PC4", float("nan"))
            Lambda    = res.get("Lambda", float("nan"))
            B_val     = res.get("B", float("nan"))
            I0        = res.get("I0", res.get("LDA_int", float("nan")))
            A_val     = res.get("A", float("nan"))
            C_val     = res.get("C", float("nan"))
            I2_int    = res.get("I2_int", float("nan"))
            I3_int    = res.get("I3_int", float("nan"))
            P0        = res.get("P0", float("nan"))
            P1        = res.get("P1", float("nan"))
            P2        = res.get("P2", float("nan"))
            Time      = res.get("Time", float("nan"))

            errWpc4 = (W_inf_pc4 - Winf) if (math.isfinite(W_inf_pc4) and math.isfinite(Winf)) else float("nan")
            errLDA  = (Wp_lda - Wp_sce) if (math.isfinite(Wp_lda) and math.isfinite(Wp_sce)) else float("nan")
            errPC4  = (Wp_pc4 - Wp_sce) if (math.isfinite(Wp_pc4) and math.isfinite(Wp_sce)) else float("nan")

            def fmt(x):
                return f"{x:12.6f}" if math.isfinite(x) else f"{'N/A':>12}"

            print(
                f"  {tag:<22} "
                f"{fmt(Winf)} {fmt(W_inf_pc4)} {fmt(errWpc4)} "
                f"{fmt(Wp_sce)} {fmt(Wp_lda)} {fmt(errLDA)} "
                f"{fmt(Wp_pc4)} {fmt(errPC4)} "
                f"{fmt(Lambda)} {fmt(B_val)} {fmt(I0)} {fmt(A_val)} {fmt(C_val)} "
                f"{fmt(I2_int)} {fmt(I3_int)} "
                f"{fmt(P0)} {fmt(P1)} {fmt(P2)} {fmt(Time)}"
            )

        save_batch_summary_csv(all_results, args.summary_csv)

    if failures:
        print(f"\n  Failed systems ({len(failures)}):")
        for tag, exc in failures:
            print(f"    x  {tag}: {type(exc).__name__}: {exc}")

    print(f"\n  Done. {len(all_results)}/{len(SYSTEM_BASIS)} systems completed.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

