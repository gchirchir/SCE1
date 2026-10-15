"""Integration to get the Vee from the Vee(r) or Vee(Ne)"""

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate

from .optimize import AngularOptimizationResult


def integrand_r(r: np.ndarray, rho: np.ndarray, integ: np.ndarray) -> np.ndarray:
    """
    Integrand for r grid, that is 4*pi*r**2*rho(r)*Vee(r)

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.

    Returns
    -------
    integrand: np.ndarray (N_grid,)
        Integrand with 4*pi*r**2*rho.
    """
    return 4 * np.pi * r**2 * rho * integ


def dintegrand_r(
    r: np.ndarray, rho: np.ndarray, drho: np.ndarray, integ: np.ndarray, dinteg: np.ndarray
) -> np.ndarray:
    """
    Derivative of integrand for r grid.

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    drho : np.ndarray (N_grid,)
        Derivative of density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.
    dinteg : np.ndarray (N_grid,)
        Derivative of integrand without 4*pi*r**2*rho.

    Returns
    -------
    dintegrand: np.ndarray (N_grid,)
        Derivative of integrand with 4*pi*r**2*rho.
    """
    return 4 * np.pi * r**2 * (drho * integ + rho * dinteg) + 8 * np.pi * r * rho * integ


def d2integrand_r(
    r: np.ndarray,
    rho: np.ndarray,
    drho: np.ndarray,
    d2rho: np.ndarray,
    integ: np.ndarray,
    dinteg: np.ndarray,
    d2integ: np.ndarray,
) -> np.ndarray:
    """
    Second derivative of integrand for r grid

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    drho : np.ndarray (N_grid,)
        Derivative of density on grid.
    d2rho : np.ndarray (N_grid,)
        Second derivative of density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.
    dinteg : np.ndarray (N_grid,)
        Derivative of integrand without 4*pi*r**2*rho.
    d2integ : np.ndarray (N_grid,)
        Second derivative of integrand without 4*pi*r**2*rho.
    Returns
    -------
    d2integrand: np.ndarray (N_grid,)
        Second derivative of integrand with 4*pi*r**2*rho.
    """
    return (
        4 * np.pi * r**2 * (d2rho * integ + 2 * drho * dinteg + rho * d2integ)
        + 16 * np.pi * r * (drho * integ + rho * dinteg)
        + 8 * np.pi * rho * integ
    )


# Dictionary of the scipy interpolators
scipy_interpolators = {
    "cubic": scipy.interpolate.CubicSpline,
    "akima": scipy.interpolate.Akima1DInterpolator,
    "pchip": scipy.interpolate.PchipInterpolator,
}

# Dictionary of the scipy integrators
scipy_integrators = {
    "trapezoid": scipy.integrate.trapezoid,
    "simpson": scipy.integrate.simpson,
    "romb": scipy.integrate.romb,
}


# ──────────────────────────────────────────────────────────────────────────
# PATCH: diagnostic for VeeIntegration's f_reg / df_reg / d2f_reg construction
# ──────────────────────────────────────────────────────────────────────────
#
# Context
# -------
# self.f_reg, self.df_reg, self.d2f_reg subtract the KNOWN analytic
# divergence of Vee as the outermost electron's co-motion function
# f_{Nel-1}(r) -> 0 (expected behavior near one end of the grid -- this
# is "the" point where the regularization exists, by construction: see
# self.reg_integral = (Nel-1)*self.res.coNe[Nel-1], the exact closed-form
# piece added back after the singular part is subtracted numerically).
#
# d2f_reg = (Nel-1) * (2*df**2/f**3 - d2f/f**2) is the worst-conditioned
# of the three regularization terms because it scales as 1/f**3. Verified
# quantitatively with representative df/d2f ~ O(1): by f ~ 1e-5,
# d2f_reg ~ 9e14 -- still finite (no overflow), but at a magnitude where
# self.d2y_reg = self.d2y - self.d2f_reg_sph loses essentially all
# precision to catastrophic cancellation (float64 has ~15-16 significant
# digits total), well before any RuntimeWarning fires. The overflow
# warning seen for C/3ZaPa-NR-CV is just the point where this same growth
# finally exceeds float64's range -- the precision loss starts far
# earlier and is invisible to any finite/overflow check.
#
# What this diagnostic answers
# -----------------------------
# Whether the small-f points are:
#   (a) clustered at the actual grid edge (near index 0 or N_grid_end-1)
#       -- consistent with the INTENDED singularity the regularization
#       is designed for, in which case the imprecision is localized to
#       where coNe's closed-form piece is supposed to dominate anyway
#   (b) scattered through the grid interior -- which would indicate a
#       genuine co-motion-function pathology for this basis, unrelated
#       to the intended edge singularity
#
# This is purely diagnostic. It does not alter f_reg/df_reg/d2f_reg or
# any downstream integral.
# ──────────────────────────────────────────────────────────────────────────


def _diagnose_f_reg_construction(vee_int: "VeeIntegration") -> None:
    """
    Diagnose where f[:, Nel-2] gets small inside VeeIntegration.__init__,
    and how large d2f_reg gets there, to distinguish the intended edge
    singularity from a genuine interior co-motion pathology.

    Parameters
    ----------
    vee_int : VeeIntegration
        The VeeIntegration instance under construction (called with
        `self` from inside __init__, after f_reg/df_reg/d2f_reg are set).
    """
    f_col = vee_int.res.f[: vee_int.N_grid_end, vee_int.res.f.shape[1] - 2]
    abs_f = np.abs(np.asarray(f_col, dtype=np.float64))
    n_total = int(abs_f.size)
    if n_total == 0:
        return

    small_mask = abs_f < 1e-3
    n_small = int(np.sum(small_mask))
    if n_small == 0:
        print(f"  [integrate-diag] f[:, Nel-2]: min|f|={abs_f.min():.3e}, "
              f"no points below 1e-3 -- regularization denominator is "
              f"well-conditioned across the whole grid.", flush=True)
        return

    small_idx = np.where(small_mask)[0]
    edge_window = 5
    near_edge = (small_idx < edge_window) | (small_idx > n_total - 1 - edge_window)
    n_near_edge = int(np.sum(near_edge))
    n_interior = n_small - n_near_edge

    print(f"  [integrate-diag] f[:, Nel-2]: {n_small}/{n_total} point(s) "
          f"have |f| < 1e-3 (min={abs_f.min():.3e}).", flush=True)
    if n_near_edge > 0:
        print(f"  [integrate-diag]   {n_near_edge} of these are within "
              f"{edge_window} points of a grid edge (index < {edge_window} "
              f"or > {n_total - 1 - edge_window}) -- consistent with the "
              f"INTENDED f->0 singularity the regularization subtracts.",
              flush=True)
    else:
        print(f"  [integrate-diag]   0 of these are near a grid edge -- "
              f"none of the small-|f| points are explained by the "
              f"intended boundary singularity.", flush=True)
    if n_interior > 0:
        interior_idx = small_idx[~near_edge]
        print(f"  [integrate-diag]   WARNING: {n_interior} point(s) with "
              f"small |f| are in the grid INTERIOR (indices: "
              f"{interior_idx[:15].tolist()}"
              f"{' ...' if n_interior > 15 else ''}), away from either "
              f"edge. This is NOT explained by the intended boundary "
              f"singularity and suggests a genuine co-motion-function "
              f"pathology for this basis, independent of any numerical-"
              f"precision concern in the regularization itself.",
              flush=True)

    # Report the actual magnitude of d2f_reg at the worst small-f points,
    # since this is the term most exposed to catastrophic cancellation
    # against self.d2y well before any overflow warning would fire.
    try:
        d2f_reg_abs = np.abs(np.asarray(vee_int.d2f_reg, dtype=np.float64))
        worst = np.argsort(-d2f_reg_abs)[:5]
        print(f"  [integrate-diag]   largest |d2f_reg| values: "
              f"{d2f_reg_abs[worst].tolist()} at indices "
              f"{worst.tolist()}", flush=True)
        if np.any(d2f_reg_abs[worst] > 1e10):
            print(f"  [integrate-diag]   WARNING: |d2f_reg| exceeds 1e10 "
                  f"at some point(s) -- self.d2y_reg = self.d2y - "
                  f"self.d2f_reg_sph at these indices is subtracting two "
                  f"comparably enormous numbers; the float64 result "
                  f"likely retains very few meaningful significant "
                  f"digits there, regardless of whether FIX-2 later "
                  f"flags it as non-finite. Treat this as a precision "
                  f"warning even when no exception or overflow is "
                  f"raised.", flush=True)
    except Exception as exc:
        print(f"  [integrate-diag]   (could not inspect d2f_reg magnitude: "
              f"{type(exc).__name__}: {exc})", flush=True)


class VeeIntegration:
    """Class for integration of Vee(r) or Vee(Ne) to get Vee"""

    def __init__(self, res: AngularOptimizationResult):
        self.res: AngularOptimizationResult = AngularOptimizationResult.to_numpy(res)

        # Get some things we often need from the result
        self.N_grid: int = self.res.f.shape[0]
        Nel: int = self.res.f.shape[1]
        self.even: bool = Nel % 2 == 0
        self.N_grid_end: int = self.N_grid - 1 if self.even else self.N_grid

        # Get the subtraction for regularization of the integral with 1/f_N-1(r)

        self.f_reg = (Nel - 1) / self.res.f[:, Nel - 2]
        self.df_reg = (
            -(Nel - 1)
            * self.res.df[: self.N_grid_end, Nel - 2]
            / (self.res.f[: self.N_grid_end, Nel - 2]) ** 2
        )
        self.d2f_reg = (Nel - 1) * (
            2
            * self.res.df[: self.N_grid_end, Nel - 2] ** 2
            / (self.res.f[: self.N_grid_end, Nel - 2] ** 3)
            - self.res.d2f[: self.N_grid_end, Nel - 2]
            / (self.res.f[: self.N_grid_end, Nel - 2] ** 2)
        )

        # PATCH: diagnose whether small f[:, Nel-2] (the regularization
        # denominator) clusters at the intended grid edge or scatters
        # through the interior, and how large d2f_reg gets as a result.
        # Purely diagnostic -- does not alter any of the above. See the
        # module-level PATCH comment above _diagnose_f_reg_construction
        # for full context.
        _diagnose_f_reg_construction(self)

        # Get the analytical integral to add back later
        self.reg_integral = (Nel - 1) * self.res.coNe[Nel - 1]

        # Get grid, integrand and derivatives
        # Regardless of grid we use x and dx for the grid
        # and y, dy and d2y for the integrand and its derivatives
        # The y_reg, dy_reg and d2y_reg are the same but with the -1/f_N-1(r) subtraction
        if self.res.opt.grid == "r":
            self.x = self.res.r
            self.dx = self.res.r[1] - self.res.r[0]

            # Transform to float to prevent bpoly from complaining
            self.a1 = float(self.res.r[0])
            self.a2 = float(self.res.r[-1])

            self.y = integrand_r(self.res.r, self.res.rho, self.res.Vee)
            self.dy = dintegrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.Vee[: self.N_grid_end],
                self.res.dVee,
            )
            self.d2y = d2integrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.d2rho[: self.N_grid_end],
                self.res.Vee[: self.N_grid_end],
                self.res.dVee,
                self.res.d2Vee,
            )

            self.f_reg_sph = integrand_r(self.res.r, self.res.rho, self.f_reg)
            self.df_reg_sph = dintegrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.f_reg[: self.N_grid_end],
                self.df_reg,
            )
            self.d2f_reg_sph = d2integrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.d2rho[: self.N_grid_end],
                self.f_reg[: self.N_grid_end],
                self.df_reg,
                self.d2f_reg,
            )
            self.y_reg = self.y - self.f_reg_sph
            if self.even:
                self.y_reg[self.N_grid_end] = self.y[self.N_grid_end]
            self.dy_reg = self.dy - self.df_reg_sph
            self.d2y_reg = self.d2y - self.d2f_reg_sph

        elif self.res.opt.grid == "Ne":
            self.x = self.res.Ne
            self.dx = self.res.Ne[1] - self.res.Ne[0]
            self.a1 = 1.0
            self.a2 = 2.0

            self.y = self.res.Vee
            self.dy = self.res.dVee
            self.d2y = self.res.d2Vee

            self.y_reg = self.y - self.f_reg
            if self.even:
                self.y_reg[self.N_grid_end] = self.y[self.N_grid_end]
            self.dy_reg = self.dy - self.df_reg
            self.d2y_reg = self.d2y - self.d2f_reg
        else:
            raise ValueError(f"Unknown grid type {self.res.grid}")

        # Initialize empty dictionary for interpolators
        self.interpolators: Dict[
            str, List[Union[scipy.interpolate.BPoly, scipy.interpolate.PPoly]]
        ] = {}

    def scipy_interpolator(self, name: str, reg: bool = False, sub_div: int = 1):
        """
        Add a scipy interpolator to the dictionary of interpolators.

        Parameters
        ----------
        name : str
            Name of the interpolator. Must be one of "linear", "cubic", "akima" or "pchip".
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        if name in scipy_interpolators:
            if name not in self.interpolators:
                self.interpolators[name] = [{}, {}]
            self.interpolators[name][int(reg)][sub_div] = scipy_interpolators[name](
                np.copy(self.x[::sub_div]),
                np.copy(self.y_reg[::sub_div]) if reg else np.copy(self.y[::sub_div]),
            )
        else:
            raise ValueError(f"Unknown interpolator {name}")

    def bpoly_0(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we only include the integrand and no derivatives.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_0"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]

        if reg:
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), np.copy(self.y_reg[::sub_div]).reshape(-1, 1)
            )
        else:
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), np.copy(self.y[::sub_div]).reshape(-1, 1)
            )

    def bpoly_1(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we only include the integrand and the first derivative.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_1"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]
        if reg:
            fold_y_dy_reg = np.stack(
                (self.y_reg[: self.N_grid_end : sub_div], self.dy_reg[::sub_div]), axis=1
            ).tolist()
            if self.even:
                fold_y_dy_reg.append([self.y_reg[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_reg
            )
        else:
            fold_y_dy = np.stack(
                (self.y[: self.N_grid_end : sub_div], self.dy[::sub_div]), axis=1
            ).tolist()
            if self.even:
                fold_y_dy.append([self.y[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy
            )

    def bpoly_2(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we include the integrand and the first and second derivative.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_2"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]
        if reg:
            fold_y_dy_d2y_reg = np.stack(
                (
                    self.y_reg[: self.N_grid_end : sub_div],
                    self.dy_reg[::sub_div],
                    self.d2y_reg[::sub_div],
                ),
                axis=1,
            ).tolist()
            if self.even:
                fold_y_dy_d2y_reg.append([self.y_reg[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_d2y_reg
            )
        else:
            fold_y_dy_d2y = np.stack(
                (self.y[: self.N_grid_end : sub_div], self.dy[::sub_div], self.d2y[::sub_div]),
                axis=1,
            ).tolist()
            if self.even:
                fold_y_dy_d2y.append([self.y[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_d2y
            )

    def interpolate_all(self, max_sub_divs: int = 2):
        """Interpolate all the interpolators for all the possible combinations
        of regularization and subdivision.

        Parameters
        ----------
        max_sub_divs : int
            Maximum subdivision of the grid. 0 means no subdivision, 1 means once, etc.
            By default 2
        """
        sub_divs = (2 ** np.arange(0, max_sub_divs)).tolist()

        # Loop over all the possible combinations of regularization and subdivision
        for reg in [True, False]:
            for sub_div in sub_divs:
                # Initialize the different possible interpolators
                self.bpoly_0(reg, sub_div)
                self.bpoly_1(reg, sub_div)
                self.bpoly_2(reg, sub_div)
                for name in scipy_interpolators:
                    self.scipy_interpolator(name, reg, sub_div)

    def integrate(
        self, max_sub_divs: int = 2
    ) -> Tuple[Dict[str, List[Dict[int, float]]], Dict[str, List[float]]]:
        """
        Integrate all the interpolators for all the possible combinations of
        regularization and subdivision.
        Also compute the integrals using scipy.integrate instead.

        Parameters
        ----------
        max_sub_divs : int
            Maximum subdivision of the grid. 0 means no subdivision, 1 means once, etc.
            By default 2

        Returns
        -------
        integrals : Dict[str, List[Dict[int, float]]]
            Dictionary of the integrals for all the possible combinations
            of regularization, subdivision and integration method.
        """
        integrals: Dict[str, List[Dict[int, float]]] = {}
        for name, reg_list in self.interpolators.items():
            if name not in integrals:
                integrals[name] = [{}, {}]
            for reg in [True, False]:
                if self.interpolators[name][int(reg)] == {}:
                    continue
                for sub_div, interpolation in reg_list[int(reg)].items():
                    if name in scipy_interpolators:
                        integrals[name][int(reg)][sub_div] = float(
                            interpolation.integrate(self.a1, self.a2)
                        )
                    else:
                        integrals[name][int(reg)][sub_div] = interpolation.integrate(
                            self.a1, self.a2
                        )

        integrals["trapezoid"] = [{}, {}]
        integrals["simpson"] = [{}, {}]
        integrals["romb"] = [{}, {}]

        for reg in [True, False]:
            for sub_div in (2 ** np.arange(0, max_sub_divs)).tolist():
                integrals["trapezoid"][int(reg)][sub_div] = scipy.integrate.trapezoid(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.x[::sub_div]
                )
                integrals["simpson"][int(reg)][sub_div] = scipy.integrate.simpson(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.x[::sub_div]
                )
                integrals["romb"][int(reg)][sub_div] = scipy.integrate.romb(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.dx * sub_div
                )

        for name in integrals:
            for sub_div in integrals[name][1]:
                integrals[name][1][sub_div] += self.reg_integral
        if max_sub_divs > 1:
            extrapolated_integrals = self.richardson_extrapolation(integrals)
        else:
            extrapolated_integrals = None
        return integrals, extrapolated_integrals

    def sub_divs_to_grid_points(self, max_sub_div: int = 1):
        """
        Compute the number of grid points for each subdivision.

        Parameters
        ----------
        max_sub_div : int, optional
            Maximum subdivision of the grid, by default 1, must be a divisor of the grid size-1

        Returns
        -------
        grid_points : Dict[int, int]
            Dictionary of the number of grid points for each subdivision.
        """
        grid_points: Dict[int, int] = {}
        for sub_div in 2 ** np.arange(0, max_sub_div):
            grid_points[sub_div] = (self.N_grid - 1) // sub_div + 1
        return grid_points

    def richardson_extrapolation(
        self, integrals: Dict[str, List[Dict[int, float]]]
    ) -> Dict[str, List[float]]:
        """
        Compute the Richardson extrapolation of the integrals.

        Parameters
        ----------
        integrals : Dict[str, List[Dict[int, float]]]
            Dictionary of the integrals for all the possible combinations
            of regularization, subdivision and integration method.

        Returns
        -------
        extrapolated_integrals : Dict[str, List[float]]
            Dictionary of the integrals for all the possible combinations
            of regularization and integration method,
            with the Richardson extrapolation.
        """
        extrapolated_integrals: Dict[str, List[float]] = {}
        for name, ints in integrals.items():
            extrapolated_integrals[name] = [None, None]
            for reg in [True, False]:
                extrapolated_integrals[name][int(reg)] = (
                    ints[int(reg)][1] * ints[int(reg)][4] - ints[int(reg)][2] ** 2
                ) / (ints[int(reg)][1] - 2 * ints[int(reg)][2] + ints[int(reg)][4])
        return extrapolated_integrals
