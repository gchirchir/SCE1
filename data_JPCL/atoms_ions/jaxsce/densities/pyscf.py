import abc
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Dict, Tuple, Union

import basis_set_exchange as bse
import numpy as np
import scipy
from pyscf import cc, df, dft, gto, scf
from pyscf.data.elements import NUC
from pyscf.gto.basis import BasisNotFoundError

from .classes import Density


def to_3D_vector(r: Union[float, np.ndarray]) -> np.ndarray:
    if isinstance(r, float):
        return np.array([[r, 0, 0]])
    return np.concatenate([r[..., None], np.zeros((*r.shape, 2))], axis=-1)


def _transform_dm_to_ao(dm_mo, mo_coeff) -> np.ndarray:
    if isinstance(dm_mo, tuple):
        dm_mo = np.stack(dm_mo, axis=0)
    dm_mo = np.asarray(dm_mo)
    if isinstance(mo_coeff, tuple):
        mo_coeff = np.stack(mo_coeff, axis=0)
    mo_coeff = np.asarray(mo_coeff)
    if dm_mo.ndim == 2 and mo_coeff.ndim == 2:
        return mo_coeff @ dm_mo @ mo_coeff.T
    if dm_mo.ndim == 3 and mo_coeff.ndim == 3:
        return np.einsum("xpi,xij,xqj->xpq", mo_coeff, dm_mo, mo_coeff)
    if dm_mo.ndim == 3 and mo_coeff.ndim == 2:
        return np.einsum("pi,xij,qj->xpq", mo_coeff, dm_mo, mo_coeff)
    raise ValueError(
        "Incompatible MO density/molecular-orbital coefficient shapes: "
        f"dm={dm_mo.shape}, mo_coeff={mo_coeff.shape}"
    )





# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _collapse_to_total_dm(dm) -> np.ndarray:
    """Return a spin-summed (nao, nao) float64 density matrix.

    Accepts:
      - tuple / list of two matrices  (UHF / ROHF in some PySCF versions)
      - ndarray with ndim == 3, shape (2, nao, nao)
      - ndarray with ndim == 2, shape (nao, nao)   [RHF, already correct]
    """
    if isinstance(dm, (list, tuple)):
        return (np.asarray(dm[0], dtype=np.float64)
                + np.asarray(dm[1], dtype=np.float64))
    dm = np.asarray(dm, dtype=np.float64)
    if dm.ndim == 3:
        return dm[0] + dm[1]
    return dm          # already (nao, nao)


def _hartree_energy(dm2d: np.ndarray, mf) -> float:
    """Compute the Hartree energy U = 0.5 * Tr(D * J) as a Python float.

    `mf.get_j()` called without arguments uses the SCF object's own dm,
    which for ROHF/UHF is (2, nao, nao).  Passing the collapsed 2-D dm
    explicitly guarantees a (nao, nao) J matrix and a scalar result.
    """
    j = np.asarray(mf.get_j(dm=dm2d), dtype=np.float64)
    if j.ndim == 3:          # some PySCF builds still return (2,nao,nao)
        j = j[0] + j[1]
    return float(np.trace(dm2d.dot(j)) / 2)


def mo_to_ao(mo_coeff, dm_mo) -> np.ndarray:
    """Transform DM from MO to AO basis; handles restricted and unrestricted."""
    if isinstance(mo_coeff, (list, tuple)) or (
        isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3
    ):
        mo_a = np.array(mo_coeff[0], dtype=np.float64)
        mo_b = np.array(mo_coeff[1], dtype=np.float64)
        dm_a = np.array(dm_mo[0],    dtype=np.float64)
        dm_b = np.array(dm_mo[1],    dtype=np.float64)
        return mo_a.dot(dm_a.dot(mo_a.T)) + mo_b.dot(dm_b.dot(mo_b.T))
    mo_coeff = np.array(mo_coeff, dtype=np.float64)
    dm_mo    = np.array(dm_mo,    dtype=np.float64)
    return mo_coeff.dot(dm_mo.dot(mo_coeff.T))


# ──────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ──────────────────────────────────────────────────────────────────────────────
class PyscfDensity(Density, metaclass=abc.ABCMeta):
    """
    Base class for densities from PySCF.

    Additional attributes:
    ---------------------
    atom : str
        Atom symbol.
    charge : int
        Charge of the atom.
    spin : int
        Spin of the atom (2S = Nalpha - Nbeta).
    N_points_Ne : int
        Number of radial points for inverting Ne and numerical integration.
    cumulant_threshold : float
        Threshold for the root solver.
    invNe_tol : float
        Tolerance for the root solver.
    rho_trunc : float
        Truncation threshold on the density for integration.
    maxiter : int
        Maximum number of iterations for the root solver.
    basis : str
        Basis set to use for PySCF.
    method : str
        Root solving method: 'newton' or 'halley'. Default is 'newton'.
    mol : pyscf.gto.Mole
        PySCF molecule object.
    """
    __doc__ = Density.__doc__ + __doc__

    
    _R_EPS: float = 1e-6

    def __init__(
        self,
        atom: str = "",
        charge: int = 0,
        spin: int = 0,
        N_points_Ne: int = 10000,
        N_int: int = 100000,
        cumulant_threshold: float = 1e-10,
        invNe_tol: float = 1e-12,
        #invNe_tol: float = 1e-8,
        rho_trunc: float = 1e-10,
        maxiter: int = 100,
        basis: str = "sto-3g",
        method: str = "newton",
    ):
        if atom == "":
            raise ValueError("atom must be specified")
        self.atom = atom[:2].strip()
        self.charge = charge
        self.spin = spin
        self.N_points_Ne = N_points_Ne
        self.N_int = N_int
        self.cumulant_threshold = cumulant_threshold
        self.invNe_tol = invNe_tol
        self.rho_trunc = rho_trunc
        self.maxiter = maxiter
        self.basis = basis
        self.method = method
        try:
            self.mol = gto.M(atom=atom, charge=charge, spin=spin, basis=basis, symmetry=True)
        except BasisNotFoundError:
            self.mol = gto.M(
                atom=atom,
                charge=charge,
                spin=spin,
                basis=bse.get_basis(basis, fmt="nwchem", elements=[NUC[atom]]),
                symmetry=True,
            )
        Nel = self.mol.nelectron
        super().__init__(Nel=Nel)
        # Nuclear charge for treutler() calls (required in PySCF >= 2.3)
        self._atom_chg = self.mol.atom_charge(0)
        self.f = lambda r, n: self.Ne(r) - n
        self.fprime = lambda r, n: self.Ne_deriv(r)
        if method == "newton":
            self.fprime2 = None
        elif method == "halley":
            self.fprime2 = lambda r, n: self.Ne_deriv2(r)
        self.r_grid: np.ndarray = None
        self.dr: np.ndarray = None
        self.Ne_grid: np.ndarray = None

    def __post_init__(self):
        self.r_grid, self.Ne_grid, self.dr = self.build_grid_guess()
        super().__post_init__()
        self.add_jvps()
        r, dr = self._radial_grid(self.N_int)
        rho_r = self.rho(r)
        self.LDA_int = np.sum(dr * 4 * np.pi * r**2 * rho_r ** (4 / 3))
        non_zero = np.where(rho_r > self.rho_trunc)
        self.GEA_int = np.sum(
            dr[non_zero]
            * 4
            * np.pi
            * r[non_zero] ** 2
            * self.rho_deriv(r[non_zero]) ** 2
            / (rho_r[non_zero] ** (4 / 3))
        )

    def encode(self) -> Dict[str, Union[int, float, str]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "atom": self.atom,
                "charge": self.charge,
                "spin": self.spin,
                "N_points_Ne": self.N_points_Ne,
                "N_int": self.N_int,
                "cumulant_threshold": self.cumulant_threshold,
                "invNe_tol": self.invNe_tol,
                "rho_trunc": self.rho_trunc,
                "maxiter": self.maxiter,
                "basis": self.basis,
                "method": self.method,
            }
        )
        return encode_dict

    def _radial_grid(self, npoints: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return dft.radi.treutler(npoints)
        except TypeError:
            return dft.radi.treutler(npoints, NUC[self.atom])

    def _dm_total(self) -> np.ndarray:
        """Return a spin-summed 2D AO density matrix."""
        dm = np.asarray(self.dm)
        if dm.ndim == 2:
            return dm
        if dm.ndim == 3:
            return np.sum(dm, axis=0)
        raise ValueError(f"Unsupported density matrix shape {dm.shape}")

    # kept as an alias so any call site (old or new) works regardless of
    # which name it was written against.
    @property
    def _dm2d(self) -> np.ndarray:
        """Always return a 2-D spin-summed DM, regardless of how self.dm was set."""
        return _collapse_to_total_dm(self.dm)

    def build_grid_guess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_grid, dr = self._radial_grid(self.N_points_Ne)
        Ne_grid = self.Ne(r_grid)
        return r_grid, Ne_grid, dr

    # ── invNe: bounded/regularised Newton, patched earlier ─────────────────
    def invNe(self, n: np.ndarray) -> np.ndarray:
        # Newton with a floored derivative first; anything that still fails
        # to land inside the tabulated [r_grid[0], r_grid[-1]] domain is
        # re-solved with brentq, which cannot leave that bracket, so it
        # cannot diverge into the region where Ne()/vH_deriv() overflow.
        _INVNE_DERIV_EPS = 1e-12

        def _floored_fprime(r_val, n_val):
            d = np.asarray(self.fprime(r_val, n_val), dtype=float)
            return np.where(d < _INVNE_DERIV_EPS, _INVNE_DERIV_EPS, d)

        n_arr = np.asarray(n, dtype=float)
        scalar_input = n_arr.ndim == 0
        orig_shape = n_arr.shape
        n_flat = n_arr.reshape(-1)

        r0 = np.interp(n_flat, self.Ne_grid, self.r_grid)
        with np.errstate(over="ignore", invalid="ignore"):
            r = np.atleast_1d(
                scipy.optimize.newton(
                    self.f, r0, fprime=_floored_fprime, args=(n_flat,),
                    tol=self.invNe_tol, maxiter=self.maxiter,
                    fprime2=self.fprime2,
                )
            ).astype(float)

        lo_bound, hi_bound = float(self.r_grid[0]), float(self.r_grid[-1])
        bad = ~np.isfinite(r) | (r < lo_bound) | (r > hi_bound)
        for idx in np.flatnonzero(bad):
            n_i = float(n_flat[idx])
            try:
                r[idx] = scipy.optimize.brentq(
                    self.f, lo_bound, hi_bound, args=(n_i,),
                    xtol=self.invNe_tol, maxiter=max(self.maxiter * 10, 200),
                )
            except ValueError:
                # f(lo_bound) and f(hi_bound) did not bracket a root, i.e.
                # n itself was outside the grid's actual Ne(r) range.
                # Clip to the nearest valid endpoint rather than return an
                # overflowed or non-finite r.
                r[idx] = lo_bound if self.f(lo_bound, n_i) > 0 else hi_bound

        if scalar_input:
            return float(r[0])
        return r.reshape(orig_shape)

    # ── singularity-safe radius, used wherever we divide by r ──────────────
    def _safe_r(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return np.sqrt(r * r + self._R_EPS * self._R_EPS)

    _R_MAX: float = 1.0e6

    def _capped_r(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return np.clip(r, -self._R_MAX, self._R_MAX)

    def rho(self, r: np.ndarray) -> np.ndarray:
        r = self._capped_r(r)
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=0)
        if isinstance(r, (int, float)):
            return dft.numint.eval_rho(self.mol, ao_value, self._dm_total(), xctype="LDA")[0]
        return dft.numint.eval_rho(self.mol, ao_value, self._dm_total(), xctype="LDA").reshape(r.shape)

    def rho_deriv(self, r: np.ndarray) -> np.ndarray:
        r = self._capped_r(r)
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=1)
        if isinstance(r, (int, float)):
            return dft.numint.eval_rho(self.mol, ao_value, self._dm_total(), xctype="GGA")[1][0]
        return dft.numint.eval_rho(self.mol, ao_value, self._dm_total(), xctype="GGA")[1].reshape(r.shape)

    def rho_deriv2(self, r: np.ndarray) -> np.ndarray:
        # d²ρ/dr² = laplacian − 2*(dρ/dr)/r  -- the /r is a genuine 0/0-shaped
        # singularity at the nucleus; regularised via _safe_r so a tiny
        # float-error numerator near r=0 doesn't get blown up into a huge
        # (or inf/nan) Hessian entry two derivatives downstream.
        r = self._capped_r(r)
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=2)
        rho_grad_lap = dft.numint.eval_rho(self.mol, ao_value, self._dm_total(), xctype="MGGA")
        if isinstance(r, (int, float)):
            r_safe = self._safe_r(r)
            return rho_grad_lap[4][0] - 2 * rho_grad_lap[1][0] / r_safe
        r_safe = self._safe_r(r)
        return rho_grad_lap[4].reshape(r.shape) - 2 * rho_grad_lap[1].reshape(r.shape) / r_safe

    def vH(self, r: np.ndarray) -> np.ndarray:
        r = self._capped_r(r)
        if isinstance(r, (int, float)):
            fakemol = gto.fakemol_for_charges(to_3D_vector(r))
        else:
            fakemol = gto.fakemol_for_charges(to_3D_vector(r.reshape(-1)))
        ints = df.incore.aux_e2(self.mol, fakemol, intor="int3c2e")
        ints = (ints + ints.transpose(1, 0, 2)) / 2
        if isinstance(r, (int, float)):
            return np.einsum("ijp,ij->p", ints, self._dm_total())[0]
        return np.einsum("ijp,ij->p", ints, self._dm_total()).reshape(r.shape)

    def vH_deriv(self, r: np.ndarray) -> np.ndarray:
        r = self._capped_r(r)
        if isinstance(r, (int, float)):
            fakemol = gto.fakemol_for_charges(to_3D_vector(r))
        else:
            fakemol = gto.fakemol_for_charges(to_3D_vector(r.reshape(-1)))
        ints = df.incore.aux_e2(self.mol, fakemol, intor="int3c2e_ip1")[0]
        ints = ints + ints.transpose(1, 0, 2)
        if isinstance(r, (int, float)):
            return np.einsum("ijp,ij->p", ints, self._dm_total())[0]
        return np.einsum("ijp,ij->p", ints, self._dm_total()).reshape(r.shape)

    def Ne(self, r: np.ndarray) -> np.ndarray:
        r = self._capped_r(r)
        return -(r**2) * self.vH_deriv(r)

    def coNe(self, r: np.ndarray) -> np.ndarray:
        # Ne(r)/r is analytically finite as r->0 (Ne ~ O(r^2) there) but
        # numerically 0/0-shaped; regularised the same way as rho_deriv2.
        r = self._capped_r(r)
        return self.vH(r) - self.Ne(r) / self._safe_r(r)


class HFDensity(PyscfDensity):
    __doc__ = PyscfDensity.__doc__ + "\n(Restricted) Hartree-Fock density class."

    def __init__(self, chkfile_name: str = "", chkfile_dir="",
                 fractional_occ: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = "hartree-fock"
        self.fractional_occ = fractional_occ
        mf = scf.ROHF(self.mol) if self.spin != 0 else scf.RHF(self.mol)
        if self.spin != 0 and fractional_occ:
            mf = scf.addons.frac_occ(mf)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 1000
        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir
        chkfile = os.path.join(chkfile_dir, chkfile_name)
        if os.path.isfile(chkfile):
            mf.chkfile = chkfile
            mf.update()
        else:
            with NamedTemporaryFile() as tempchk:
                mf.chkfile = tempchk.name
                mf.kernel()
                shutil.copyfile(tempchk.name, chkfile)
        self.dm = mf.make_rdm1()
        jmat = np.asarray(mf.get_j())
        if jmat.ndim == 3:
            jmat = np.sum(jmat, axis=0)
        self.U = np.trace(self._dm_total().dot(jmat)) / 2
        self.__post_init__()

    def encode(self) -> Dict[str, Union[str, float, int]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "chkfile_name": self.chkfile_name,
                "chkfile_dir": self.chkfile_dir,
                "fractional_occ": self.fractional_occ,
            }
        )
        return encode_dict


##################################################
########    CCSD
#####################################################
class CCSDDensity(PyscfDensity):
    __doc__ = PyscfDensity.__doc__ + "\nRelaxed CCSD density class."

    def __init__(self, dm_file_name: str = "", dm_file_dir: str = "",
                 chkfile_name: str = "", chkfile_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.name = "ccsd"
        if dm_file_name == "":
            raise ValueError("dm_file_name must be given")
        self.dm_file_name = dm_file_name
        self.dm_file_dir = dm_file_dir
        dm_file = os.path.join(dm_file_dir, dm_file_name)
        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir
        chkfile = os.path.join(chkfile_dir, chkfile_name)
        if os.path.isfile(dm_file):
            self.dm = np.load(dm_file)
        else:
            try:
                import psutil
                available_memory_mb = psutil.virtual_memory().available / 1024**2
                max_memory = min(int(0.7 * available_memory_mb), 100000)
            except ImportError:
                max_memory = 92000
            # Gradients and UGradients require matching RHF/UHF references.
            mf = scf.UHF(self.mol) if self.spin != 0 else scf.RHF(self.mol)
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-11
            mf.max_cycle = 100
            mf.max_memory = max_memory
            if os.path.exists(chkfile):
                mf.chkfile = chkfile
                mf.init_guess = "chkfile"
                mf.kernel()
            else:
                with NamedTemporaryFile() as tempchk:
                    mf.chkfile = tempchk.name
                    mf.kernel()
                    shutil.copyfile(tempchk.name, chkfile)
            if not mf.converged:
                raise RuntimeError("HF calculation for relaxed CCSD did not converge")
            ccsd = cc.CCSD(mf)
            ccsd.max_memory = max_memory
            ccsd.kernel()
            if not ccsd.converged:
                raise RuntimeError("CCSD calculation did not converge")
            try:
                from .relaxed_dm import Gradients, UGradients
                dm1_ao = (UGradients(ccsd).kernel() if self.spin != 0
                          else Gradients(ccsd).kernel())
            except Exception as exc:
                method = "UCCSD" if self.spin != 0 else "RCCSD"
                raise RuntimeError(
                    f"Failed to compute relaxed {method} density: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            # RHF returns a 2D AO density; UHF returns alpha/beta AO blocks.
            self.dm = np.asarray(dm1_ao, dtype=np.float64)
            print(
                f"  CCSD relaxed density calculated: shape {self.dm.shape}",
                flush=True,
            )
            np.save(dm_file, self.dm)
        eris = self.mol.intor("int2e")
        dm_tot = self._dm_total()
        self.U = 1 / 2 * np.einsum("pqrs,pq,rs->", eris, dm_tot, dm_tot)
        self.__post_init__()

    def encode(self) -> Dict[str, Union[str, float, int]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "dm_file_name": self.dm_file_name,
                "dm_file_dir": self.dm_file_dir,
                "chkfile_name": self.chkfile_name,
                "chkfile_dir": self.chkfile_dir,
            }
        )
        return encode_dict


# =============================================================================
# CCSD(T) ORBITAL-RELAXED DENSITY  (RHF Z-vector; UHF Z-vector with fallback)
# =============================================================================
def _rhf_ccsd_t_energy(mycc, eris):
    """True (T) correction for an RHF reference."""
    try:
        from pyscf.cc import ccsd_t
        return float(ccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2))
    except Exception:
        from pyscf.cc import ccsd_t_slow
        return float(ccsd_t_slow.kernel(mycc, eris, mycc.t1, mycc.t2))


def _uhf_ccsd_t_energy(mycc, eris):
    """True (T) correction for a UHF reference."""
    from pyscf.cc import uccsd_t
    return float(uccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2))


def make_ccsd_t_relaxed_dm1_ao(mycc, t1, t2, l1, l2, eris):
    """Build the standalone-equivalent relaxed RHF CCSD(T) AO 1-RDM."""
    from functools import reduce
    from pyscf import lib
    from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
    from pyscf.grad import ccsd as ccsd_grad

    mol = mycc.mol
    mo_coeff = mycc.mo_coeff
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mycc.mo_occ > 0)

    d1 = ccsd_t_rdm._gamma1_intermediates(
        mycc, t1, t2, l1, l2, eris, for_grad=True
    )
    doo, dov, dvo, dvv = d1

    fd2 = lib.H5TmpFile()
    d2 = ccsd_t_rdm._gamma2_outcore(
        mycc, t1, t2, l1, l2, eris, fd2, True
    )
    ccsd_grad._rdm2_mo2ao(mycc, d2, mo_coeff, fd2)

    Imat = np.zeros((nao, nao))
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * .9e6 / 8 / (nao**3 * 2.5)))
    offsetdic = mol.offset_nr_by_atom()
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        for b0, b1, nf in ccsd_grad._shell_prange(
            mol, shl0, shl1, blksize
        ):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = ccsd_grad._load_block_tril(
                fd2["dm2"], ip0, ip1, nao
            )
            dm2buf[:, :, diagidx] *= .5
            shls_slice = (
                b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas
            )
            eri0 = mol.intor(
                "int2e", aosym="s2kl", shls_slice=shls_slice
            )
            Imat += lib.einsum(
                "ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf
            )
    Imat = reduce(
        np.dot,
        (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff),
    ) * -1

    dm1mo = np.zeros((nmo, nmo))
    dm1mo[:nocc, :nocc] = doo + doo.T
    dm1mo[nocc:, nocc:] = dvv + dvv.T

    dm1_ao_tmp = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1_ao_tmp) * 2
    Xvo = reduce(
        np.dot,
        (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]),
    )
    Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

    dm1mo += ccsd_grad._response_dm1(mycc, Xvo, eris)
    for i in range(nocc):
        dm1mo[i, i] += 2.0

    return reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))


def _run_ccsdt_relaxed_density_rhf(mol):
    """RHF closed-shell T3-relaxed Z-vector orbital-relaxed CCSD(T) AO density.

    Validated path (unchanged). Returns (dm_ao, e_ccsdt).
    """
    from pyscf.cc import ccsd_t_lambda_slow

    print("    Running HF for CCSD(T)...", flush=True)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF did not converge for CCSD(T) density")

    print("    Running CCSD for CCSD(T)...", flush=True)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.conv_tol_normt = 1e-7
    mycc.max_cycle = 200
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD amplitudes did not converge for CCSD(T) density")

    print("    Computing ERIs...", flush=True)
    eris = mycc.ao2mo()
    et = _rhf_ccsd_t_energy(mycc, eris)
    e_ccsdt = float(mycc.e_tot + et)
    print(f"    E(CCSD)    = {mycc.e_tot:.10f}", flush=True)
    print(f"    E(T)       = {et:.10f}", flush=True)
    print(f"    E(CCSD(T)) = {e_ccsdt:.10f}", flush=True)

    print("    Solving CCSD(T) lambda equations...", flush=True)
    conv, l1, l2 = ccsd_t_lambda_slow.kernel(
        mycc, eris, mycc.t1, mycc.t2, max_cycle=200, tol=1e-8
    )
    if not conv:
        raise RuntimeError("CCSD(T) lambda did not converge")

    print("    Building CCSD(T) orbital-relaxed density (RHF Z-vector)...", flush=True)
    dm_ao = make_ccsd_t_relaxed_dm1_ao(
        mycc, mycc.t1, mycc.t2, l1, l2, eris
    )
    return dm_ao, e_ccsdt


def make_uccsd_t_relaxed_dm1_ao(mycc, t1, t2, l1, l2, eris):
    """Build the relaxed UHF CCSD(T) AO 1-RDM, summed over spin.

    This is the UHF analogue of :func:`make_ccsd_t_relaxed_dm1_ao`, extracted
    from the density/Z-vector part of ``pyscf.grad.uccsd.grad_elec``.  PySCF's
    UHF gradient intermediates represent one half of the correlation density;
    the final factor of two below converts them to the physical spin densities.
    """
    from functools import reduce
    from pyscf.cc import uccsd_t_rdm
    from pyscf.grad import uccsd as uccsd_grad
    from pyscf.grad import ccsd as ccsd_grad
    from pyscf import lib

    mol = mycc.mol
    mo_a, mo_b = mycc.mo_coeff
    nao = mo_a.shape[0]
    nmoa, nmob = mo_a.shape[1], mo_b.shape[1]
    nocca = int(np.count_nonzero(mycc.mo_occ[0] > 0))
    noccb = int(np.count_nonzero(mycc.mo_occ[1] > 0))

    d1 = uccsd_t_rdm._gamma1_intermediates(
        mycc, t1, t2, l1, l2, eris, for_grad=True
    )
    fd2 = lib.H5TmpFile()
    d2 = uccsd_t_rdm._gamma2_outcore(
        mycc, t1, t2, l1, l2, eris, fd2, True
    )
    uccsd_grad._rdm2_mo2ao(mycc, d2, (mo_a, mo_b), fd2)

    (dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb) = d1

    Imata = np.zeros((nao, nao))
    Imatb = np.zeros((nao, nao))
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * .9e6 / 8 / (nao**3 * 2.5)))
    offsetdic = mol.offset_nr_by_atom()
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        for b0, b1, nf in ccsd_grad._shell_prange(
            mol, shl0, shl1, blksize
        ):
            ip0, ip1 = ip1, ip1 + nf
            dm2bufa = ccsd_grad._load_block_tril(
                fd2["dm2aa+ab"], ip0, ip1, nao
            )
            dm2bufb = ccsd_grad._load_block_tril(
                fd2["dm2bb+ab"], ip0, ip1, nao
            )
            dm2bufa[:, :, diagidx] *= .5
            dm2bufb[:, :, diagidx] *= .5
            shls_slice = (
                b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas
            )
            eri0 = mol.intor(
                "int2e", aosym="s2kl", shls_slice=shls_slice
            )
            eri0 = eri0.reshape(nf, nao, -1)
            Imata += lib.einsum("ipx,iqx->pq", eri0, dm2bufa)
            Imatb += lib.einsum("ipx,iqx->pq", eri0, dm2bufb)

    s0 = mycc._scf.get_ovlp()
    Imata = reduce(np.dot, (mo_a.T, Imata, s0, mo_a)) * -1
    Imatb = reduce(np.dot, (mo_b.T, Imatb, s0, mo_b)) * -1

    # These are half-correlation density matrices in PySCF's UHF gradient
    # convention. Keep that convention through the response solve.
    dm1a = np.zeros((nmoa, nmoa))
    dm1b = np.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = (dooa + dooa.T) * .5
    dm1a[nocca:, nocca:] = (dvva + dvva.T) * .5
    dm1b[:noccb, :noccb] = (doob + doob.T) * .5
    dm1b[noccb:, noccb:] = (dvvb + dvvb.T) * .5

    dm1a_ao = reduce(np.dot, (mo_a, dm1a, mo_a.T))
    dm1b_ao = reduce(np.dot, (mo_b, dm1b, mo_b.T))
    vhf = mycc._scf.get_veff(mol, (dm1a_ao, dm1b_ao))
    Xvo = reduce(np.dot, (mo_a[:, nocca:].T, vhf[0], mo_a[:, :nocca]))
    XVO = reduce(np.dot, (mo_b[:, noccb:].T, vhf[1], mo_b[:, :noccb]))
    Xvo += Imata[:nocca, nocca:].T - Imata[nocca:, :nocca]
    XVO += Imatb[:noccb, noccb:].T - Imatb[noccb:, :noccb]

    dm1a_resp, dm1b_resp = uccsd_grad._response_dm1(
        mycc, (Xvo, XVO), eris
    )
    dm1a += dm1a_resp
    dm1b += dm1b_resp

    # Convert the half-correlation convention to physical spin densities and
    # restore the UHF reference occupations.
    dm1a *= 2
    dm1b *= 2
    for i in range(nocca):
        dm1a[i, i] += 1.0
    for i in range(noccb):
        dm1b[i, i] += 1.0

    dm_ao_a = reduce(np.dot, (mo_a, dm1a, mo_a.T))
    dm_ao_b = reduce(np.dot, (mo_b, dm1b, mo_b.T))
    return np.asarray(dm_ao_a + dm_ao_b, dtype=np.float64)


def _run_ccsdt_relaxed_density_uhf(mol):
    """UHF open-shell orbital-relaxed CCSD(T) AO density (spin-summed).

    Attempts the true orbital-relaxed (Z-vector) density. If PySCF's UHF grad
    internals don't line up on this version, falls back to the UNRELAXED (T)
    density and prints a prominent warning (theory level then differs from RHF).

    Returns (dm_ao, e_ccsdt).
    """
    from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm

    print("    Running UHF for CCSD(T)...", flush=True)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 400
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("UHF did not converge for CCSD(T) density")

    ss, s2p1 = mf.spin_square()
    print(f"    UHF <S^2> = {ss:.6f}  (2S+1 = {s2p1:.4f})", flush=True)

    print("    Running UCCSD for CCSD(T)...", flush=True)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.conv_tol_normt = 1e-7
    mycc.max_cycle = 300
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("UCCSD amplitudes did not converge for CCSD(T) density")

    print("    Computing ERIs...", flush=True)
    eris = mycc.ao2mo()
    et = _uhf_ccsd_t_energy(mycc, eris)
    e_ccsdt = float(mycc.e_tot + et)
    print(f"    E(UCCSD)    = {mycc.e_tot:.10f}", flush=True)
    print(f"    E(T)        = {et:.10f}", flush=True)
    print(f"    E(UCCSD(T)) = {e_ccsdt:.10f}", flush=True)

    print("    Solving UCCSD(T) lambda equations...", flush=True)
    conv, l1, l2 = uccsd_t_lambda.kernel(
        mycc, eris, mycc.t1, mycc.t2, max_cycle=300, tol=1e-8
    )
    if not conv:
        raise RuntimeError("UCCSD(T) lambda did not converge")

    # --- Try the ORBITAL-RELAXED (Z-vector) density -----------------------
    try:
        print("    Building CCSD(T) orbital-relaxed density (UHF Z-vector)...",
              flush=True)
        dm_ao = make_uccsd_t_relaxed_dm1_ao(
            mycc, mycc.t1, mycc.t2, l1, l2, eris
        )
        # basic finiteness + trace sanity, else treat as failure
        if not np.all(np.isfinite(dm_ao)):
            raise FloatingPointError("relaxed UHF density non-finite")
        print("    [UHF] density = ORBITAL-RELAXED (matches RHF theory level)",
              flush=True)
        return dm_ao, e_ccsdt
    except Exception as exc:
        print("    " + "!" * 66, flush=True)
        print(f"    [UHF] WARNING: orbital-relaxed path unavailable on this "
              f"PySCF ({type(exc).__name__}: {exc}).", flush=True)
        print("    [UHF] FALLING BACK to the UNRELAXED (T) density. This is a "
              "LOWER", flush=True)
        print("    [UHF] relaxation level than the RHF path -- do NOT mix the "
              "two in a", flush=True)
        print("    [UHF] published series without noting it. Pin the relaxed "
              "path first.", flush=True)
        print("    " + "!" * 66, flush=True)
        dm1 = uccsd_t_rdm.make_rdm1(
            mycc, mycc.t1, mycc.t2, l1, l2, eris, ao_repr=True
        )
        dm_ao = np.asarray(dm1[0], dtype=np.float64) + np.asarray(dm1[1], dtype=np.float64)
        return dm_ao, e_ccsdt


def _run_ccsdt_relaxed_density(mol):
    """Build a consistently orbital-relaxed RHF/UHF CCSD(T) density.

    The implementation in :mod:`.ccsdt_relaxed` mirrors the PySCF gradient
    Z-vector construction for both references and deliberately has no
    unrelaxed fallback.  Returning a spin-summed AO matrix keeps the public
    jaxsce density interface unchanged.
    """
    from .ccsdt_relaxed import make_ccsdt_density, validate_density

    is_uhf = int(mol.spin) != 0
    mf = scf.UHF(mol) if is_uhf else scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 400 if is_uhf else 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF reference did not converge for CCSD(T) density")

    mycc = cc.UCCSD(mf) if is_uhf else cc.CCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.conv_tol_normt = 1e-7
    mycc.max_cycle = 300 if is_uhf else 200
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD amplitudes did not converge for CCSD(T) density")

    eris = mycc.ao2mo()
    e_ccsdt = float(mycc.e_tot + mycc.ccsd_t(eris=eris))

    dm_spin, info = make_ccsdt_density(
        mycc, eris, max_cycle=300 if is_uhf else 200, tol=1e-8,
        return_info=True,
    )
    density_n, density_herm = validate_density(
        mol, dm_spin, mf.get_ovlp(), atol=5e-5)

    dm_spin = np.asarray(dm_spin, dtype=np.float64)
    dm_ao = dm_spin.sum(axis=0) if dm_spin.ndim == 3 else dm_spin
    dm_ao = np.asarray(0.5 * (dm_ao + dm_ao.T), dtype=np.float64)

    print(
        f"    CCSD(T) density = ORBITAL-RELAXED "
        f"({'UHF' if is_uhf else 'RHF'} Z-vector); "
        f"Tr(DS)={density_n:.10f} herm={density_herm:.3e} "
        f"one_electron={info['exact_one_electron']}",
        flush=True,
    )
    return dm_ao, e_ccsdt


class CCSDTDensity(PyscfDensity):
    """Orbital-relaxed CCSD(T) density (RHF closed-shell or UHF open-shell)."""

    __doc__ = PyscfDensity.__doc__ + __doc__

    def __init__(self, dm_file_name: str = "", dm_file_dir: str = "",
                 chkfile_name: str = "", chkfile_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.name = "ccsd(t)-orbital-relaxed"

        if dm_file_name == "":
            raise ValueError("dm_file_name must be given")
        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.dm_file_name = dm_file_name
        self.dm_file_dir = dm_file_dir
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir

        dm_file = os.path.join(dm_file_dir, dm_file_name)
        os.makedirs(dm_file_dir, exist_ok=True)

        if os.path.isfile(dm_file):
            print("    Loading cached orbital-relaxed CCSD(T) density...", flush=True)
            self.dm = np.asarray(np.load(dm_file), dtype=np.float64)
            self.e_ccsdt = None
            self.e_ccsd_reference = None
        else:
            self.dm, self.e_ccsdt = _run_ccsdt_relaxed_density(self.mol)
            self.e_ccsd_reference = self.e_ccsdt  # backward-compat alias
            np.save(dm_file, self.dm)

        if self.dm.shape != (self.mol.nao_nr(), self.mol.nao_nr()):
            raise RuntimeError(
                f"CCSD(T) AO density shape {self.dm.shape} does not match "
                f"nAO={self.mol.nao_nr()}"
            )
        if not np.all(np.isfinite(self.dm)):
            raise FloatingPointError("Cached CCSD(T) density contains NaN/Inf")

        overlap = self.mol.intor_symmetric("int1e_ovlp")
        electron_count = float(np.einsum("ij,ji->", self.dm, overlap))
        if abs(electron_count - self.mol.nelectron) > 1e-6:
            raise RuntimeError(
                f"CCSD(T) density has Tr(DS)={electron_count:.10f}; "
                f"expected {self.mol.nelectron}"
            )

        jmat = np.asarray(scf.hf.get_jk(self.mol, self.dm, hermi=1, with_k=False)[0])
        self.U = float(0.5 * np.einsum("ij,ji->", self.dm, jmat))
        self.__post_init__()

    def encode(self) -> Dict[str, Union[str, float, int]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "dm_file_name": self.dm_file_name,
                "dm_file_dir": self.dm_file_dir,
                "chkfile_name": self.chkfile_name,
                "chkfile_dir": self.chkfile_dir,
            }
        )
        return encode_dict


##################################################
########    FCI
#####################################################
class FCIDensity(PyscfDensity):
    """
    FCI density.  direct_spin1 (closed-shell) or direct_uhf (open-shell).

    Additional attributes:
    ----------------------
    dm : numpy.ndarray  (nao, nao)
        Spin-summed FCI density matrix in AO basis.
    e_fci : float or None
        Total FCI energy (None if loaded from cache).
    U : float
        Hartree energy (Python float).
    """
    __doc__ = PyscfDensity.__doc__ + __doc__

    def __init__(
        self,
        dm_file_name: str = "",
        dm_file_dir:  str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "fci"
        if dm_file_name == "":
            raise ValueError("dm_file_name must be given")
        self.dm_file_name = dm_file_name
        self.dm_file_dir  = dm_file_dir

        dm_file = os.path.join(dm_file_dir, dm_file_name)
        if os.path.isfile(dm_file):
            self.dm    = _collapse_to_total_dm(np.load(dm_file))
            self.e_fci = None
        else:
            dm_raw, self.e_fci = self._compute_fci_density()
            self.dm = _collapse_to_total_dm(dm_raw)
            np.save(dm_file, self.dm)

        eris   = self.mol.intor("int2e")
        dm_2d = self._dm_total()
        self.U = float(
            0.5 * np.einsum("pqrs,pq,rs->", eris, dm_2d, dm_2d)
        )
        self.__post_init__()

    def _compute_fci_density(self):
        from pyscf.fci import direct_spin1, direct_uhf
        from pyscf import ao2mo as pyscf_ao2mo

        is_open = self.spin != 0
        mf      = scf.UHF(self.mol) if is_open else scf.RHF(self.mol)
        mf.conv_tol      = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle     = 1000
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("HF did not converge for FCI calculation")

        if not is_open:
            C           = mf.mo_coeff
            norb        = C.shape[1]
            nelec_tuple = self.mol.nelec
            h1_mo  = C.T @ mf.get_hcore() @ C
            eri_mo = pyscf_ao2mo.restore(1, pyscf_ao2mo.kernel(self.mol, C), norb)
            e_fci, ci = direct_spin1.kernel(
                h1_mo, eri_mo, norb, nelec_tuple, tol=1e-10)
            dm1_mo, _ = direct_spin1.make_rdm12(ci, norb, nelec_tuple)
            dm_ao = C @ dm1_mo @ C.T
        else:
            Ca, Cb      = mf.mo_coeff[0], mf.mo_coeff[1]
            norb        = Ca.shape[1]
            nelec_tuple = self.mol.nelec
            hcore       = mf.get_hcore()
            h1_a   = Ca.T @ hcore @ Ca
            h1_b   = Cb.T @ hcore @ Cb
            eri_aa = pyscf_ao2mo.restore(1, pyscf_ao2mo.kernel(self.mol, Ca), norb)
            eri_bb = pyscf_ao2mo.restore(1, pyscf_ao2mo.kernel(self.mol, Cb), norb)
            eri_ab = pyscf_ao2mo.kernel(self.mol, [Ca, Ca, Cb, Cb])
            e_fci, ci = direct_uhf.kernel(
                (h1_a, h1_b), (eri_aa, eri_ab, eri_bb),
                norb, nelec_tuple, tol=1e-10,
            )
            dm1_mo = direct_uhf.make_rdm1s(ci, norb, nelec_tuple)
            dm_ao  = Ca @ dm1_mo[0] @ Ca.T + Cb @ dm1_mo[1] @ Cb.T
        return dm_ao, float(e_fci)

    def encode(self) -> Dict[str, Union[str, float, int]]:
        d = super().encode()
        d.update({"dm_file_name": self.dm_file_name,
                  "dm_file_dir":  self.dm_file_dir})
        return d


# ══════════════════════════════════════════════════════════════════════════════
# WU-YANG DENSITY INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

# ── WY inversion hyperparameters ─────────────────────────────────────────────
_WY_DENSITY_LINDEP_THR: float = 1e-4   # eigenvalue floor for overlap regularisation
_WY_DENSITY_TOL:        float = 1e-4   # gradient convergence tolerance
_WY_DENSITY_ETA:        float = 1e-4   # Tikhonov regularisation strength
_WY_DENSITY_MAXITER:    int   = 5000   # maximum Wu-Yang iterations
_WY_DENSITY_RHO_THR:    float = 0.05   # L2 density-error ceiling for acceptance


def _run_wy_inversion_for_density(
        mol,
        dm_target: np.ndarray,
        wy_file:   str  = "",
        verbose:   bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Wu-Yang KS inversion targeting *dm_target*.

    The function finds the local Kohn-Sham potential v_s(r) whose KS
    ground-state density exactly reproduces ρ[dm_target].  The resulting
    KS density matrix is returned in place of the original correlated DM.

    Cache behaviour
    ~~~~~~~~~~~~~~~
    If *wy_file* is non-empty and already exists on disk, the cached KS DM
    is loaded and the inversion is skipped.  On a successful inversion the
    KS DM is written to *wy_file* for future reuse.

    Parameters
    ----------
    mol       : pyscf.gto.Mole
    dm_target : (nao, nao) spin-summed float64 AO density matrix
    wy_file   : path for caching the KS DM (``""`` disables caching)
    verbose   : print kspies intermediate output

    Returns
    -------
    dm_wy : (nao, nao) float64  — KS DM  (== dm_target on failure)
    meta  : dict with keys
              ``wy_converged``, ``wy_rho_diff``,
              ``wy_mo_coeff``, ``wy_mo_occ``, ``wy_mo_energy``
    """
    _fail_meta: dict = {
        "wy_converged" : False,
        "wy_rho_diff"  : np.inf,
        "wy_mo_coeff"  : None,
        "wy_mo_occ"    : None,
        "wy_mo_energy" : None,
    }

    # ── cache hit ─────────────────────────────────────────────────────────────
    if wy_file and os.path.isfile(wy_file):
        print(f"  [WY-density] Loading cached KS DM from {wy_file}", flush=True)
        dm_wy = _collapse_to_total_dm(np.load(wy_file))
        # Mark as previously converged; rho_diff unknown but was acceptable
        return dm_wy, {**_fail_meta, "wy_converged": True, "wy_rho_diff": 0.0}

    # ── kspies availability ───────────────────────────────────────────────────
    try:
        from kspies import wy as _wy_mod
    except ImportError:
        print("  [WY-density] kspies not available; "
              "retaining original DM.", flush=True)
        return dm_target.copy(), _fail_meta

    # ── regularise overlap in-place (restored afterwards) ────────────────────
    S = mol.intor("int1e_ovlp")
    s_val, s_vec = np.linalg.eigh(S)
    S_reg = (s_vec
             @ np.diag(np.maximum(s_val, _WY_DENSITY_LINDEP_THR))
             @ s_vec.T)
    _orig_get_ovlp = mol.get_ovlp          # save for restoration
    mol.get_ovlp   = lambda *_: S_reg      # patch mol in-place

    converged = False
    mw        = None
    try:
        mw           = _wy_mod.RWY(mol, dm_target)
        mw.guide     = "faxc"
        mw.tol       = _WY_DENSITY_TOL
        mw.method    = "trust-exact"
        mw.eta       = _WY_DENSITY_ETA
        mw.max_cycle = _WY_DENSITY_MAXITER
        if verbose:
            mw.verbose = 5
        mw.run()
        converged = bool(getattr(mw, "converged", False))
    except Exception as exc:
        if verbose:
            print(f"  [WY-density] Exception during inversion: {exc}", flush=True)
    finally:
        mol.get_ovlp = _orig_get_ovlp      # always restore

    if not converged or mw is None or not hasattr(mw, "dm"):
        print("  [WY-density] Inversion failed; retaining original DM.",
              flush=True)
        return dm_target.copy(), _fail_meta

    # ── L2 density-fidelity check ─────────────────────────────────────────────
    grids = dft.gen_grid.Grids(mol)
    grids.level = 9
    grids.build()
    ao   = dft.numint.eval_ao(mol, grids.coords)
    rho1 = np.einsum("gi,ij,gj->g", ao, dm_target,         ao)
    rho2 = np.einsum("gi,ij,gj->g", ao, np.asarray(mw.dm), ao)
    rho_diff = float(np.sqrt(np.dot(grids.weights, (rho1 - rho2) ** 2)))

    meta: dict = {
        "wy_converged" : True,
        "wy_rho_diff"  : rho_diff,
        "wy_mo_coeff"  : getattr(mw, "mo_coeff",  None),
        "wy_mo_occ"    : getattr(mw, "mo_occ",    None),
        "wy_mo_energy" : getattr(mw, "mo_energy", None),
    }

    if rho_diff > _WY_DENSITY_RHO_THR:
        print(f"  [WY-density] rho_diff={rho_diff:.3e} exceeds threshold "
              f"({_WY_DENSITY_RHO_THR}); retaining original DM.", flush=True)
        meta["wy_converged"] = False
        return dm_target.copy(), meta

    dm_wy = _collapse_to_total_dm(np.asarray(mw.dm))
    print(f"  [WY-density] Converged (rho_diff={rho_diff:.3e}); "
          "KS DM substituted.", flush=True)

    # ── persist to cache ──────────────────────────────────────────────────────
    if wy_file:
        os.makedirs(os.path.dirname(os.path.abspath(wy_file)), exist_ok=True)
        np.save(wy_file, dm_wy)
        print(f"  [WY-density] KS DM saved to {wy_file}", flush=True)

    return dm_wy, meta


# ──────────────────────────────────────────────────────────────────────────────
# _WYMixin  —  shared Wu-Yang injection logic
# ──────────────────────────────────────────────────────────────────────────────
class _WYMixin:
    """
    Mixin that intercepts ``__post_init__`` to replace the correlated DM
    with the Wu-Yang KS DM before the density grid and integral quantities
    (LDA_int, GEA_int, …) are computed.

    MRO requirement: place ``_WYMixin`` before the base density class, e.g.
    ``class WYHFDensity(_WYMixin, HFDensity)``.

    Attributes injected by ``__init__`` of each concrete WY subclass
    ------------------------------------------------------------------
    _wy_file    : str   — path for caching the KS DM
    _verbose_wy : bool  — verbose kspies output
    _wy_pending : bool  — flag consumed by the first __post_init__ call
    _wy_tag     : str   — log-message prefix, e.g. "[WY-CCSD]"

    Attributes set after successful inversion
    -----------------------------------------
    wy_meta : dict — keys: wy_converged, wy_rho_diff,
                           wy_mo_coeff, wy_mo_occ, wy_mo_energy
    """
    # Class-level defaults (overridden as instance attributes in each __init__)
    _wy_file:    str  = ""
    _verbose_wy: bool = False
    _wy_pending: bool = False
    _wy_tag:     str  = "[WY]"

    # -- internal helpers ------------------------------------------------------
    def _apply_wy(self) -> None:
        """
        Run WY inversion and replace ``self.dm`` / ``self.U`` with KS values.

        For open-shell systems (spin != 0) the inversion is skipped and the
        original DM is kept unchanged.
        """
        if self.spin != 0:
            print(f"  {self._wy_tag} Open-shell (spin={self.spin}): "
                  "WY inversion requires RWY (closed-shell only); "
                  "retaining original DM.", flush=True)
            self.wy_meta = {
                "wy_converged" : False,
                "wy_rho_diff"  : np.inf,
                "wy_mo_coeff"  : None,
                "wy_mo_occ"    : None,
                "wy_mo_energy" : None,
            }
            return

        print(f"  {self._wy_tag} Starting Wu-Yang inversion …", flush=True)
        dm_wy, meta = _run_wy_inversion_for_density(
            self.mol, self._dm2d, self._wy_file, self._verbose_wy,
        )
        self.dm      = dm_wy
        self.wy_meta = meta

        # Recompute Hartree energy U for the KS DM via full ERI.
        # (The original U was computed from the correlated DM.)
        eris   = self.mol.intor("int2e")
        self.U = float(
            0.5 * np.einsum("pqrs,pq,rs->", eris, self.dm, self.dm)
        )
        print(f"  {self._wy_tag} U (KS DM) = {self.U:.10f}", flush=True)

    # -- overridden __post_init__ ----------------------------------------------
    def __post_init__(self) -> None:
        """
        Intercept the first ``__post_init__`` call to inject the WY step.

        Flow (with MRO  WYXxx → _WYMixin → XxxDensity → PyscfDensity):
          1. XxxDensity.__init__ computes self.dm  (correlated / HF DM)
          2. XxxDensity.__init__ calls self.__post_init__()
          3. → _WYMixin.__post_init__  (this method)
          4.   → _apply_wy()  replaces self.dm with KS DM, recomputes self.U
          5.   → super().__post_init__()  = PyscfDensity.__post_init__
          6.     builds r_grid, LDA_int, GEA_int  using the KS density
        """
        if self._wy_pending:
            self._wy_pending = False
            self._apply_wy()
        super().__post_init__()   # → PyscfDensity.__post_init__

