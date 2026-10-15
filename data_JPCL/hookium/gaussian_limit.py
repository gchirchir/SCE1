#!/usr/bin/env python3
"""Analytic omega -> infinity limits of the Hookium SCE ingredients (SI Section S15, Table S9b).

For omega -> infinity the Hookium density tends to the Gaussian rho = 2 (omega/pi)^(3/2) exp(-omega r^2).
By uniform scaling (lambda = sqrt(omega)) W_inf / sqrt(omega) and W'_inf / omega^(3/4) are constants;
they are evaluated here at omega = 1 from the exact two-electron expressions of the main text (Eqs. 8-13):
  W_inf  = 1/2 int 4 pi r^2 rho [1/(r+f) - v_H],  N_e(f) = 2 - N_e(r),
  W'_inf = 1/2 int 4 pi r^2 rho (2 w1 + w2)/4,     w1^2 = (r^2+f^2)/(r f (r+f)^3),  w2^2 = -2(1+f'^2)/(f'(r+f)^3).
The integrals are taken in the variable u = N_e(r) (int 4 pi r^2 rho g dr = 2 int_0^1 g du by the symmetry of
the co-motion map), with U = 2 sqrt(2/pi) analytic. Result: W_inf -1.151617, W'_inf 0.403058.
"""
import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import brentq
from scipy.integrate import quad

SP = np.sqrt(np.pi)
rho = lambda r: 2 * np.pi**-1.5 * np.exp(-r * r)
ne = lambda r: 2 * (erf(r) - 2 / SP * r * np.exp(-r * r)) if r > 1e-3 else 2 * 4 / (3 * SP) * r**3 * (1 - 0.6 * r * r)
ne_c = lambda r: 2 * (erfc(r) + 2 / SP * r * np.exp(-r * r))  # 2 - N_e(r), accurate at large r


def pair(u):
    """Electron at r with N_e(r) = u and its partner at f with 2 - N_e(f) = u."""
    r = brentq(lambda x: ne(x) - u, 1e-12, 10, xtol=1e-15, rtol=1e-15)
    f = brentq(lambda x: ne_c(x) - u, 1e-6, 30, xtol=1e-15, rtol=1e-15)
    fp = -rho(r) * r * r / (rho(f) * f * f)
    w1 = np.sqrt((r * r + f * f) / (r * f * (r + f) ** 3))
    w2 = np.sqrt(-2 * (1 + fp * fp) / (fp * (r + f) ** 3))
    return r, f, w1, w2


def integral(g):
    return 2 * quad(lambda t: np.exp(t) * g(pair(np.exp(t))), -40, 0, limit=500, epsabs=1e-13, epsrel=1e-12)[0]


def main():
    vee = integral(lambda p: 0.5 / (p[0] + p[1]))
    wp = integral(lambda p: 0.5 * (2 * p[2] + p[3]) / 4)
    u = 2 * np.sqrt(2 / np.pi)
    print(f"V_ee^SCE/sqrt(w) = {vee:.6f}, U/sqrt(w) = {u:.6f}, W_inf/sqrt(w) = {vee - u:.6f}, W'_inf/w^0.75 = {wp:.6f}")


if __name__ == "__main__":
    main()
