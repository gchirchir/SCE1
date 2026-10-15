

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mandatory Wu-Yang inversions of RHF/UHF HF, PBE, PBE0, and CCSD target
densities (KSPies; same AO basis). All DFT models use only the converged
Wu-Yang density.
Compute W_inf and W'_inf for:
  LDA, PBE, TPSS, SCAN, R2SCAN, MSM, PC, PC4, nPC, mPC, hPC, fPC, ePC

Key convention (matches tpss_winf_wifp.f):
  W_inf  = ∫ n ( eps_x + eps_c_infty ) dr
  W'_inf = 1/2 ∫ n ( eps_c_2 ) dr
           (NO exchange in W'_inf for LDA/PBE/TPSS/SCAN/R2SCAN/MSM)

MSM definition:
  - MSM W_inf  == TPSS W_inf
  - MSM W'_inf == TPSS W'_inf but with GE2-corrected piece:
      d1/rs^(3/2)  ->  (d1/rs^(3/2)) * [1 + mu_bar * s^2/(1+s^2)]
    where s^2 = (4*phi(zeta)^2/pi) * t^2

SCAN / R2SCAN:
  - SCAN correlation-side low-density pieces are implemented from the formulas
    for H0^(1), H0^(2), H1^(1), H1^(2), eps~ and eps~~.
  - R2SCAN uses the upgraded H1^(1), H1^(2) expressions with Q, D, and Qtilde.
  - Exchange part in W_inf is evaluated through LibXC/PySCF:
      SCAN   -> exchange from SCAN
      R2SCAN -> exchange from R2SCAN
  - W'_inf keeps the same convention as above: no exchange contribution.

Notes:
- WY[...] requires: pip install kspies
- WY uses same AO basis (no pbas/tbas).
- CCSD: RCCSD for RHF, UCCSD for UHF.
- Use dense unpruned grid for stable results.

Implementation note:
- For r2SCAN, the regularized alpha-bar is kept in the standard code form below.
- For the upgraded r2SCAN H1^(2), the code uses the coefficient entering eps~~,
  i.e. no extra explicit alpha^(-3/2) factor is inserted inside eps~~ itself.
"""

import numpy as np
from pyscf import gto, scf, dft, cc

DFT_MODELS = (
    "LDA", "PBE", "TPSS", "SCAN", "R2SCAN", "MSM",
    "PC", "PC4", "nPC", "mPC", "hPC", "fPC", "ePC",
)

# ============================================================
#  Constants and parameters (atomic units)
# ============================================================

pi = np.pi
KF = (3.0 * pi**2)**(1.0 / 3.0)
S2_DEN = 4.0 * KF**2  # s^2 = |∇n|^2 / (4 kF^2 n^(8/3))

# PC prefactors
A_PC = -9.0 * (4.0 * pi / 3.0) ** (1.0 / 3.0) / 10.0
C_PC = 0.5 * np.sqrt(3.0 * pi)

# PC gradient expansion coefficients
MU_W_PC  = -(3.0 ** (1.0 / 3.0)) * (2.0 * pi) ** (2.0 / 3.0) / 35.0
MU_WP_PC = -0.638

# ---- PC4 coefficients
KAP_PC4 = 2.0 * (9.0 * pi / 4.0) ** (1.0 / 3.0)
MU_W_PC4 = -(1.0 / 105.0) * (KAP_PC4 ** 2)
NU_W_PC4 = -(1.0 / 375.0) * (KAP_PC4 ** 4)

print("PC4 coeffs:", KAP_PC4, MU_W_PC4, NU_W_PC4)

# ---- nPC parameters
NPC_KAPPA_W = 0.385
NPC_MU_W    = MU_W_PC4
NPC_NU_W    = 0.2425269624701989

NPC_KAPPA_WP = 0.804
NPC_MU_WP    = MU_WP_PC
NPC_NU_WP    = 0.0

# mPC parameters
A_mPC = 2.0
B_mPC = 1.3

# hPC parameters
KAPPA_W  = -7.11
KAPPA_WP = -99.11
MU_W_hPC  = MU_W_PC
MU_WP_hPC = -0.7222

# fPC parameter
F_fPC = 0.5

# ePC constants (MGGA, energy only)
EPC_KAPPA = 0.491
EPC_MU_INF = 0.14
EPC_p_W = 6.65
EPC_a1, EPC_a2, EPC_a3 = 0.1, 0.9342, 0.22447
EPC_MU_INF_PRIME = 0.491
EPC_p_WP = 11.0
EPC_b1, EPC_b2, EPC_b3 = 0.04865, 4.3217, 16.581

# MSM parameter (GE2 coefficient for W'_inf)
MSM_MU_BAR = 0.491

# ---- SCAN / R2SCAN correlation constants
SCAN_C1C = 0.64
SCAN_DC  = 0.7
SCAN_C2C = 1.5

SCAN_GAMMA    = 0.031091
SCAN_B1C      = 0.0285764
SCAN_B2C      = 0.0889
SCAN_B3C      = 0.125541
SCAN_BETA_INF = 0.0375281215   # low-density beta used in B1, B2, H1^(1), H1^(2)
SCAN_BETA_MB  = 0.066725
SCAN_CHI_INF  = 0.128026

R2SCAN_CC = np.array([
    1.0,
   -0.64,
   -0.4352,
   -1.535685604549,
    3.061560252175,
   -1.915710236206,
    0.516884468372,
   -0.051848879792
], dtype=float)

R2SCAN_DFC2 = float(np.dot(np.arange(1, 8, dtype=float), R2SCAN_CC[1:]))
R2SCAN_CHI_INF = 0.128025
R2SCAN_TAU_R   = 1.0e-4
R2SCAN_ALPHA_R = 1.0e-3
R2SCAN_ETA     = 1.0e-3
R2SCAN_DP2     = 0.361

# ============================================================
#  Helpers: density/grad/tau fields
# ============================================================

def _vec_norm(v):
    return np.sqrt(np.einsum("ij,ij->j", v, v))

def _get_spin_fields_from_rho(rho, spin, xctype):
    """
    rho formats (PySCF):
      - spin=0: rho shape (4,ngrids) for GGA, (6,ngrids) for MGGA
      - spin=1: rho is (rho_a, rho_b), each like above
    """
    if spin == 0:
        n_tot = rho[0]
        grad_tot = rho[1:4]
        n_up = 0.5 * n_tot
        n_dn = 0.5 * n_tot
        grad_up = 0.5 * grad_tot
        grad_dn = 0.5 * grad_tot
        if xctype.upper() == "MGGA":
            tau_tot = rho[5]
            tau_up = 0.5 * tau_tot
            tau_dn = 0.5 * tau_tot
        else:
            tau_up = tau_dn = None
    else:
        ra, rb = rho
        n_up = ra[0]
        n_dn = rb[0]
        grad_up = ra[1:4]
        grad_dn = rb[1:4]
        n_tot = n_up + n_dn
        grad_tot = grad_up + grad_dn
        if xctype.upper() == "MGGA":
            tau_up = ra[5]
            tau_dn = rb[5]
        else:
            tau_up = tau_dn = None

    gamma_tot = _vec_norm(grad_tot) ** 2
    return n_up, n_dn, grad_up, grad_dn, tau_up, tau_dn, n_tot, grad_tot, gamma_tot

def _s2_from_n_gamma(n, gamma):
    n_safe = np.maximum(n, 1e-20)
    n_83 = n_safe ** (8.0 / 3.0)
    return gamma / (S2_DEN * n_83)

def _safe_denom(x, eps=1e-20):
    return np.where(np.abs(x) < eps, np.where(x >= 0.0, eps, -eps), x)

def _safe_exp(x, clip=200.0):
    return np.exp(np.clip(x, -clip, clip))

def _tau_unif(n):
    n_safe = np.maximum(n, 1e-20)
    return (3.0 / 10.0) * (3.0 * pi * pi) ** (2.0 / 3.0) * n_safe ** (5.0 / 3.0)

# ============================================================
#  PC-family enhancement factors
# ============================================================

def _F_PC(s2, mu):
    return 1.0 + mu * s2

def _F_PC4(s2):
    return 1.0 + MU_W_PC4 * s2 + NU_W_PC4 * (s2 * s2)

def _F_nPC(s2, kappa, mu, nu):
    den = 1.0 + (mu / kappa) * s2 + (nu / (kappa * kappa)) * (s2 * s2)
    return 1.0 - kappa + kappa / den

def _F_mPC_W(s2, a):
    b = a + 0.14
    return (1.0 + a * s2) / (1.0 + b * s2)

def _F_mPC_Wp(s2, b):
    return (1.0 + b * s2) / (1.0 + s2)

def _F_hPC(s2, kappa, mu):
    den = 1.0 + (mu / kappa) * s2
    return 1.0 + kappa - kappa / den

def _g_fPC(s2, mu, f):
    omf = 1.0 - f
    return f + omf * np.exp(mu * s2 / omf)

def _Fexp(s2, mu):
    return np.exp(mu * s2)

# ============================================================
#  ePC enhancement factors (MGGA, energy only)
# ============================================================

def _ePC_F0_W(s2):
    k, mu = EPC_KAPPA, EPC_MU_INF
    denom = 1.0 + (mu / k) * s2 + (mu * mu / (k * k)) * (s2 * s2)
    return 1.0 - k + k / denom

def _ePC_F1_W(s2):
    s8 = (s2 * s2) * (s2 * s2)
    return EPC_a1 + EPC_a2 / (1.0 + EPC_a3 * s8)

def _ePC_Fe_W(s2, z):
    F0 = _ePC_F0_W(s2)
    F1 = _ePC_F1_W(s2)
    return F0 + (z * F1 - F0) * (z ** EPC_p_W)

def _ePC_F0_Wp(s2):
    mu_p = EPC_MU_INF_PRIME
    return (1.0 + ((mu_p + 1.0) * s2)) / (1.0 + s2)

def _ePC_F1_Wp(s2, zeta):
    s6 = s2 * s2 * s2
    core = (EPC_b1 + (EPC_b1 + EPC_b2 * s2) * np.exp(-EPC_b3 * s6))
    return core * (1.0 - zeta ** 10)

def _ePC_Fe_Wp(s2, z, zeta):
    F0 = _ePC_F0_Wp(s2)
    F1 = _ePC_F1_Wp(s2, zeta)
    return F0 + ((z ** EPC_p_WP) * F1 - F0) * (z ** 2)

# ============================================================
#  LDA/PBE/TPSS / SCAN / R2SCAN helpers
# ============================================================

def _phi(zeta):
    return 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))

def _ds_r2scan(zeta):
    z = np.clip(zeta, -1.0, 1.0)
    return 0.5 * ((1.0 + z) ** (5.0 / 3.0) + (1.0 - z) ** (5.0 / 3.0))

def _d0zeta(zeta):
    f = (1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0)
    fzeta = (f - 2.0) / ((2.0 ** (4.0 / 3.0)) - 2.0)
    dxczeta = 0.45820 * f / 2.0 + 0.4335 - 0.1310 * fzeta + 0.0262 * fzeta * (zeta ** 4)
    d0 = dxczeta - (3.0 / (8.0 * pi)) * ((9.0 * pi / 4.0) ** (1.0 / 3.0)) * f
    return d0

def _dx_scan(zeta):
    z = np.clip(zeta, -1.0, 1.0)
    return 0.5 * ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0))

def _Gc_scan(zeta):
    z = np.clip(zeta, -1.0, 1.0)
    return (1.0 - 2.3631 * (_dx_scan(z) - 1.0)) * (1.0 - z ** 12)

def _g_inf_from_s2(s2, chi_inf):
    return 1.0 / np.power(1.0 + 4.0 * chi_inf * s2, 0.25)

def _alpha_scan(n, gamma, tau):
    n_safe = np.maximum(n, 1e-20)
    tauW = gamma / (8.0 * n_safe)
    tauU = _tau_unif(n_safe)
    return (tau - tauW) / np.maximum(tauU, 1e-20)

def _alpha_bar_r2scan(n, gamma, tau):
    """
    Standard regularized alpha-bar used in the code.
    """
    n_safe = np.maximum(n, 1e-20)
    tauW = gamma / (8.0 * n_safe)
    tauU = _tau_unif(n_safe)
    alpha_tilde = (tau - tauW) / (tauU + R2SCAN_TAU_R)
    return alpha_tilde**3 / (alpha_tilde**2 + R2SCAN_ALPHA_R)

def _fc_scan(alpha):
    out = np.zeros_like(alpha)

    m1 = alpha < 1.0
    out[m1] = np.exp(-SCAN_C1C * alpha[m1] / np.maximum(1.0 - alpha[m1], 1e-20))

    m2 = alpha > 1.0
    out[m2] = -SCAN_DC * np.exp(SCAN_C2C / np.minimum(1.0 - alpha[m2], -1e-20))

    return out

def _fc_r2scan(alpha_bar):
    out = np.empty_like(alpha_bar)

    m1 = alpha_bar < 0.0
    out[m1] = np.exp(-SCAN_C1C * alpha_bar[m1] / (1.0 - alpha_bar[m1]))

    m2 = (alpha_bar >= 0.0) & (alpha_bar <= 2.5)
    out[m2] = np.polynomial.polynomial.polyval(alpha_bar[m2], R2SCAN_CC)

    m3 = alpha_bar > 2.5
    out[m3] = -SCAN_DC * np.exp(SCAN_C2C / (1.0 - alpha_bar[m3]))

    return out

def ECPW92(rs, zeta):
    return -_d0zeta(zeta) / rs

def ECPW922(rs, zeta):
    return 1.50 / (rs ** (3.0 / 2.0))

def ECPBE(rs, zeta, t):
    phi = _phi(zeta)
    ECU = ECPW92(rs, zeta)
    beta = 0.066725
    d0 = _d0zeta(zeta)
    B1 = beta * (phi ** 3) * rs / d0
    t = np.minimum(t, 100.0)
    Hpbe = beta * (phi ** 3) * (t ** 2) * (1.0 + B1 * (t ** 2)) / (1.0 + B1 * (t ** 2) + (B1 ** 2) * (t ** 4))
    return ECU + Hpbe

def ECPBE2(rs, zeta, t):
    phi = _phi(zeta)
    ECU = ECPW922(rs, zeta)
    beta = 0.066725
    d0 = _d0zeta(zeta)
    d1 = 1.50
    B1 = beta * (phi ** 3) * rs / d0
    B2 = beta * d1 * (phi ** 3) * (rs ** 0.5) / (d0 ** 2)
    t = np.minimum(t, 100.0)
    Hpbe = -beta * (phi ** 3) * B1 * B2 * (t ** 6) * (2.0 + B1 * (t ** 2)) / (1.0 + B1 * (t ** 2) + (B1 ** 2) * (t ** 4)) ** 2
    return ECU + Hpbe

def exchange2_unpol(dens, gradmag, tau, tauw):
    ak = 0.804
    amu = 0.21951
    kF = (3.0 * pi * pi * dens) ** (1.0 / 3.0)
    s = np.abs(gradmag) / (2.0 * kF * dens)
    FxPBE = 1.0 + ak - ak / (1.0 + (amu * s * s / ak))
    exLDA = -3.0 * kF / (4.0 * pi)
    exPBE = exLDA * FxPBE

    ab = 0.4
    ac = 1.59096
    ae = 1.537

    z = tauw / np.maximum(tau, 1e-20)
    tauunif = (3.0 / 10.0) * (kF ** 2) * dens
    alpha = (tau - tauw) / np.maximum(tauunif, 1e-20)

    p = s * s
    qtb = (9.0 / 20.0) * (alpha - 1.0) / np.sqrt(1.0 + ab * alpha * (alpha - 1.0)) + 2.0 * p / 3.0
    xx1 = (
        ((10.0 / 81.0) + ac * z * z / ((1.0 + z * z) ** 2)) * p
        + (146.0 / 2025.0) * qtb * qtb
        - (73.0 / 405.0) * qtb * np.sqrt(0.5 * (9.0 * z * z / 25.0) + 0.5 * p * p)
        + (100.0 / (ak * 6561.0)) * p * p
        + 2.0 * np.sqrt(ae) * (10.0 / 81.0) * (9.0 * z * z / 25.0)
        + ae * amu * p * p * p
    ) / ((1.0 + np.sqrt(ae) * p) ** 2)

    FxTPSS = 1.0 + ak - ak / (1.0 + (xx1 / ak))
    exTPSS = exLDA * FxTPSS
    return exLDA, exPBE, exTPSS

def ECTPSS(up, dn, grad_up_vec, grad_dn_vec, tau_tot):
    dens = up + dn
    dens_safe = np.maximum(dens, 1e-20)
    zeta = (up - dn) / dens_safe

    grad_tot = grad_up_vec + grad_dn_vec
    gr = _vec_norm(grad_tot)

    eta_vec = up * grad_dn_vec - dn * grad_up_vec
    eta3 = _vec_norm(eta_vec)
    eta = (2.0 / (dens_safe * dens_safe)) * eta3 / (2.0 * ((3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)))
    eta = np.abs(eta)

    c = (0.53 + 0.87 * zeta * zeta + 0.5 * (zeta ** 4) + 2.26 * (zeta ** 6)) / (
        (1.0 + 0.5 * eta * eta * (((1.0 + zeta) ** (-4.0 / 3.0)) + ((1.0 - zeta) ** (-4.0 / 3.0)))) ** 4
    )

    rs = (3.0 / (4.0 * pi * dens_safe)) ** (1.0 / 3.0)
    phi = _phi(zeta)
    kF = (3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)
    kS = np.sqrt(4.0 * kF / pi)
    t = np.abs(gr) / (2.0 * kS * dens_safe * phi)

    ECGGA = ECPBE(rs, zeta, t)

    up_safe = np.maximum(up, 1e-20)
    dn_safe = np.maximum(dn, 1e-20)
    gr_up = _vec_norm(grad_up_vec)
    gr_dn = _vec_norm(grad_dn_vec)

    rs_up = (3.0 / (4.0 * pi * up_safe)) ** (1.0 / 3.0)
    rs_dn = (3.0 / (4.0 * pi * dn_safe)) ** (1.0 / 3.0)

    kF_up = (3.0 * pi * pi * up_safe) ** (1.0 / 3.0)
    kF_dn = (3.0 * pi * pi * dn_safe) ** (1.0 / 3.0)

    kS_up = np.sqrt(4.0 * kF_up / pi)
    kS_dn = np.sqrt(4.0 * kF_dn / pi)

    t_up = np.abs(gr_up) / (2.0 * kS_up * up_safe * _phi(1.0))
    t_dn = np.abs(gr_dn) / (2.0 * kS_dn * dn_safe * _phi(-1.0))

    ECGGAup = ECPBE(rs_up, 1.0, t_up)
    ECGGAdn = ECPBE(rs_dn, -1.0, t_dn)

    tauW = (_vec_norm(grad_tot) ** 2) / (8.0 * dens_safe)
    z = tauW / np.maximum(tau_tot, 1e-20)

    ECrevPKZB = ECGGA * (1.0 + c * z * z) - (1.0 + c) * z * z * (
        (up / dens_safe) * ECGGAup + (dn / dens_safe) * ECGGAdn
    )
    return ECrevPKZB

def ECTPSS2(up, dn, grad_up_vec, grad_dn_vec, tau_tot):
    dens = up + dn
    dens_safe = np.maximum(dens, 1e-20)
    zeta = (up - dn) / dens_safe

    grad_tot = grad_up_vec + grad_dn_vec
    gr = _vec_norm(grad_tot)

    eta_vec = up * grad_dn_vec - dn * grad_up_vec
    eta3 = _vec_norm(eta_vec)
    eta = (2.0 / (dens_safe * dens_safe)) * eta3 / (2.0 * ((3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)))
    eta = np.abs(eta)

    c = (0.53 + 0.87 * zeta * zeta + 0.5 * (zeta ** 4) + 2.26 * (zeta ** 6)) / (
        (1.0 + 0.5 * eta * eta * (((1.0 + zeta) ** (-4.0 / 3.0)) + ((1.0 - zeta) ** (-4.0 / 3.0)))) ** 4
    )

    rs = (3.0 / (4.0 * pi * dens_safe)) ** (1.0 / 3.0)
    phi = _phi(zeta)
    kF = (3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)
    kS = np.sqrt(4.0 * kF / pi)
    t = np.abs(gr) / (2.0 * kS * dens_safe * phi)

    ECGGA = ECPBE2(rs, zeta, t)

    up_safe = np.maximum(up, 1e-20)
    dn_safe = np.maximum(dn, 1e-20)

    gr_up = _vec_norm(grad_up_vec)
    gr_dn = _vec_norm(grad_dn_vec)

    rs_up = (3.0 / (4.0 * pi * up_safe)) ** (1.0 / 3.0)
    rs_dn = (3.0 / (4.0 * pi * dn_safe)) ** (1.0 / 3.0)

    kF_up = (3.0 * pi * pi * up_safe) ** (1.0 / 3.0)
    kF_dn = (3.0 * pi * pi * dn_safe) ** (1.0 / 3.0)

    kS_up = np.sqrt(4.0 * kF_up / pi)
    kS_dn = np.sqrt(4.0 * kF_dn / pi)

    t_up = np.abs(gr_up) / (2.0 * kS_up * up_safe * _phi(1.0))
    t_dn = np.abs(gr_dn) / (2.0 * kS_dn * dn_safe * _phi(-1.0))

    ECGGAup = ECPBE2(rs_up, 1.0, t_up)
    ECGGAdn = ECPBE2(rs_dn, -1.0, t_dn)

    tauW = (_vec_norm(grad_tot) ** 2) / (8.0 * dens_safe)
    z = tauW / np.maximum(tau_tot, 1e-20)

    ECrevPKZB = ECGGA * (1.0 + c * z * z) - (1.0 + c) * z * z * (
        (up / dens_safe) * ECGGAup + (dn / dens_safe) * ECGGAdn
    )
    return ECrevPKZB

# ============================================================
# MSM: GE2-corrected ECPBE2 / ECTPSS2
# ============================================================

def ECPBE2_MSM(rs, zeta, t):
    """
    MSM modification of ECPBE2:
      ECU = d1/rs^(3/2)  ->  (d1/rs^(3/2)) * [1 + mu_bar * s^2/(1+s^2)]
      Hpbe (H2) is kept exactly the same as in ECPBE2 (starts at t^6)
    with s^2 = (4*phi(zeta)^2/pi) * t^2.
    """
    phi = _phi(zeta)

    d1 = 1.535
    ECU0 = d1 / (rs ** (3.0 / 2.0))

    s2 = (4.0 * phi * phi / pi) * (t * t)
    ECU = ECU0 * (1.0 + MSM_MU_BAR * (s2 / (1.0 + s2)))

    beta = 0.066725
    d0 = _d0zeta(zeta)
    B1 = beta * (phi ** 3) * rs / d0
    B2 = beta * d1 * (phi ** 3) * (rs ** 0.5) / (d0 ** 2)

    t = np.minimum(t, 100.0)
    Hpbe = -beta * (phi ** 3) * B1 * B2 * (t ** 6) * (2.0 + B1 * (t ** 2)) / (
        1.0 + B1 * (t ** 2) + (B1 ** 2) * (t ** 4)
    ) ** 2

    return ECU  # + Hpbe

def ECTPSS2_MSM(up, dn, grad_up_vec, grad_dn_vec, tau_tot):
    dens = up + dn
    dens_safe = np.maximum(dens, 1e-20)
    zeta = (up - dn) / dens_safe

    grad_tot = grad_up_vec + grad_dn_vec
    gr = _vec_norm(grad_tot)

    eta_vec = up * grad_dn_vec - dn * grad_up_vec
    eta3 = _vec_norm(eta_vec)
    eta = (2.0 / (dens_safe * dens_safe)) * eta3 / (2.0 * ((3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)))
    eta = np.abs(eta)

    c = 0.53

    rs = (3.0 / (4.0 * pi * dens_safe)) ** (1.0 / 3.0)
    phi = _phi(zeta)
    kF = (3.0 * pi * pi * dens_safe) ** (1.0 / 3.0)
    kS = np.sqrt(4.0 * kF / pi)
    t = np.abs(gr) / (2.0 * kS * dens_safe * phi)

    ECGGA = ECPBE2_MSM(rs, zeta, t)

    up_safe = np.maximum(up, 1e-20)
    dn_safe = np.maximum(dn, 1e-20)

    gr_up = _vec_norm(grad_up_vec)
    gr_dn = _vec_norm(grad_dn_vec)

    rs_up = (3.0 / (4.0 * pi * up_safe)) ** (1.0 / 3.0)
    rs_dn = (3.0 / (4.0 * pi * dn_safe)) ** (1.0 / 3.0)

    kF_up = (3.0 * pi * pi * up_safe) ** (1.0 / 3.0)
    kF_dn = (3.0 * pi * pi * dn_safe) ** (1.0 / 3.0)

    kS_up = np.sqrt(4.0 * kF_up / pi)
    kS_dn = np.sqrt(4.0 * kF_dn / pi)

    t_up = np.abs(gr_up) / (2.0 * kS_up * up_safe * _phi(1.0))
    t_dn = np.abs(gr_dn) / (2.0 * kS_dn * dn_safe * _phi(-1.0))

    ECGGAup = ECPBE2_MSM(rs_up, 1.0, t_up)
    ECGGAdn = ECPBE2_MSM(rs_dn, -1.0, t_dn)

    tauW = (_vec_norm(grad_tot) ** 2) / (8.0 * dens_safe)
    z = tauW / np.maximum(tau_tot, 1e-20)

    ECrevPKZB = ECGGA * (1.0 + c * z * z) - (1.0 + c) * z * z * (
        (up / dens_safe) * ECGGAup + (dn / dens_safe) * ECGGAdn
    )
    return ECrevPKZB

# ============================================================
#  SCAN / R2SCAN correlation-side low-density pieces
# ============================================================

def _libxc_mgga_x_eps(name, rho, spin):
    """
    Exchange energy per particle from LibXC via PySCF.
    Tries both alias form and explicit MGGA_X_* form.
    """
    trials = (f"{name},", f"MGGA_X_{name},")
    last_err = None
    for xc_code in trials:
        try:
            exc, _, _, _ = dft.libxc.eval_xc(
                xc_code, rho, spin=spin, relativity=0, deriv=1
            )
            return exc
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Could not evaluate exchange for {name} via LibXC: {last_err}")

def _scan_r2scan_ld_corr(up, dn, grad_up_vec, grad_dn_vec, tau_tot, variant="SCAN"):
    """
    Returns the low-density correlation-side quantities needed for:
      W_inf  : epsW  =  eps1_tilde  + f*(eps0_tilde  - eps1_tilde)
      W'_inf : epsWp = eps1_tt      + f*(eps0_tt     - eps1_tt)

    d1 is kept as 1.50 here to remain consistent with the rest of your code.
    """
    dens = np.maximum(up + dn, 1e-20)
    zeta = np.clip((up - dn) / dens, -1.0, 1.0)

    grad_tot = grad_up_vec + grad_dn_vec
    gr = _vec_norm(grad_tot)
    gamma = gr * gr

    s2 = _s2_from_n_gamma(dens, gamma)
    p = s2

    rs = (3.0 / (4.0 * pi * dens)) ** (1.0 / 3.0)
    phi = _phi(zeta)
    phi3 = np.maximum(phi ** 3, 1e-20)

    kF = (3.0 * pi * pi * dens) ** (1.0 / 3.0)
    kS = np.sqrt(4.0 * kF / pi)
    t = gr / (2.0 * kS * dens * np.maximum(phi, 1e-20))

    if variant.upper() == "SCAN":
        f_interp = _fc_scan(_alpha_scan(dens, gamma, tau_tot))
        chi_inf = SCAN_CHI_INF
    else:
        f_interp = _fc_r2scan(_alpha_bar_r2scan(dens, gamma, tau_tot))
        chi_inf = R2SCAN_CHI_INF

    ginf = _g_inf_from_s2(s2, chi_inf)
    Gc = _Gc_scan(zeta)

    # H0^(1), H0^(2)
    H01 = (SCAN_B1C / SCAN_B3C) * (1.0 - ginf) / rs
    H02 = -(SCAN_B1C * SCAN_B3C / (SCAN_B2C ** 2)) * (1.0 - ginf) / (rs ** 1.5)

    # d0(zeta), d1(zeta)
    d0 = _d0zeta(zeta)
    d1 = 1.50 * np.ones_like(rs)
    d0s = _safe_denom(d0)

    # Correct B1, B2 from the manuscript:
    #   B1 = (beta/d0) * phi^3 * rs
    #   B2 = (beta*d1/d0^2) * phi^3 * rs^(1/2)
    B1 = SCAN_BETA_INF * phi3 * rs / d0s
    B2 = SCAN_BETA_INF * d1 * phi3 * np.sqrt(rs) / np.maximum(d0 * d0, 1e-20)
    B1s = _safe_denom(B1)

    if variant.upper() == "SCAN":
        x = np.maximum(1.0 + 4.0 * B1 * (t ** 2), 1e-20)
        q = x ** 0.25

        H11 = SCAN_BETA_INF * phi3 * (q - 1.0) / (B1s * q)

        H12 = -SCAN_BETA_INF * phi3 * B2 * (
            -1.0 - 5.0 * B1 * (t ** 2) + q + 4.0 * B1 * (t ** 2) * q
        ) / np.maximum((B1s ** 2) * (x ** 1.25), 1e-20)

    else:
        # Upgraded r2SCAN low-density expressions
        ds = _ds_r2scan(zeta)

        p2 = p * p
        exp_p = _safe_exp(p2 / R2SCAN_DP2)
        exp_m = _safe_exp(-p2 / R2SCAN_DP2)

        Q = (
            5.0 * SCAN_B3C * d0 * p * R2SCAN_DFC2 * (4.0 + 9.0 * R2SCAN_ETA)
            - 5.0 * SCAN_B1C * p * R2SCAN_DFC2 * (4.0 * Gc + 9.0 * R2SCAN_ETA)
            + 27.0 * SCAN_B3C * ds * exp_p * rs * (t ** 2) * SCAN_BETA_INF * phi3
        )

        D = 27.0 + (
            4.0 * B1 * exp_m * Q
        ) / np.maximum(SCAN_B3C * ds * rs * SCAN_BETA_INF * phi3, 1e-20)

        D = np.maximum(D, 1e-20)
        D14 = D ** 0.25

        Qtil = (
            27.0 * SCAN_B3C * ds * exp_p * rs * SCAN_BETA_INF * phi3
            + 4.0 * B1 * Q
        )
        Qtils = _safe_denom(Qtil)

        c34 = 3.0 ** 0.75
        term = 1.0 - c34 / D14

        H11 = SCAN_BETA_INF * phi3 * term / B1s

        # Coefficient entering eps~~
        H12 = SCAN_BETA_INF * phi3 / np.maximum(B1s ** 2, 1e-20) * (
            c34 * B1 * B2 * Q / (Qtils * D14)
            - B2 * term
        )

    # eps~_0, eps~_1
    eps0_t = ((-SCAN_B1C / (SCAN_B3C * rs)) + H01) * Gc
    eps1_t = (-d0 / rs) + H11

    # eps~~_0, eps~~_1
    eps0_tt = ((SCAN_B1C * SCAN_B3C / (SCAN_B2C ** 2 * rs ** 1.5)) + H02) * Gc
    eps1_tt = (d1 / (rs ** 1.5)) + H12

    epsW  = eps1_t  + f_interp * (eps0_t  - eps1_t)
    epsWp = eps1_tt + f_interp * (eps0_tt - eps1_tt)

    return epsW, epsWp

# ============================================================
#  Model dispatch
# ============================================================

def model_xctype(model: str) -> str:
    m = model.upper()
    return "MGGA" if m in ("TPSS", "EPC", "MSM", "SCAN", "R2SCAN") else "GGA"

def make_exc_func(model: str, which: str):
    model = model.upper()
    assert which in ("W", "Wp")

    if which == "W":
        pref = A_PC
        pwr = 1.0 / 3.0
    else:
        pref = C_PC
        pwr = 1.0 / 2.0

    def exc(rho, spin):
        xctype = model_xctype(model)
        n_up, n_dn, grad_up, grad_dn, tau_up, tau_dn, n_tot, grad_tot, gamma_tot = _get_spin_fields_from_rho(
            rho, spin, xctype
        )
        n_safe = np.maximum(n_tot, 1e-20)

        # PC family (GGA-type)
        if model in ("PC", "PC4", "NPC", "MPC", "HPC", "FPC"):
            s2 = _s2_from_n_gamma(n_safe, gamma_tot)

            if model == "PC":
                F = _F_PC(s2, MU_W_PC if which == "W" else MU_WP_PC)

            elif model == "PC4":
                F = _F_PC4(s2) if which == "W" else _F_PC(s2, MU_WP_PC)

            elif model == "NPC":
                if which == "W":
                    F = _F_nPC(s2, NPC_KAPPA_W, NPC_MU_W, NPC_NU_W)
                else:
                    F = 0.0

            elif model == "MPC":
                F = _F_mPC_W(s2, A_mPC) if which == "W" else _F_mPC_Wp(s2, B_mPC)

            elif model == "HPC":
                F = _F_hPC(s2, KAPPA_W, MU_W_hPC) if which == "W" else _F_hPC(s2, KAPPA_WP, MU_WP_hPC)

            else:  # FPC
                F = _g_fPC(s2, MU_W_PC, F_fPC) if which == "W" else _Fexp(s2, MU_WP_PC)

            return pref * n_safe ** pwr * F

        # ePC (MGGA)
        if model == "EPC":
            s2 = _s2_from_n_gamma(n_safe, gamma_tot)
            tau_tot = np.maximum(tau_up + tau_dn, 1e-20)
            tauW = gamma_tot / (8.0 * n_safe)
            z = np.clip(tauW / tau_tot, 0.0, 1.0)
            zeta = (n_up - n_dn) / np.maximum(n_safe, 1e-20)
            F = _ePC_Fe_W(s2, z) if which == "W" else _ePC_Fe_Wp(s2, z, zeta)
            return pref * n_safe ** pwr * F

        # SCAN / R2SCAN
        if model in ("SCAN", "R2SCAN"):
            up = np.maximum(n_up, 1e-20)
            dn = np.maximum(n_dn, 1e-20)
            tau_tot = np.maximum(tau_up + tau_dn, 1e-20)

            ex = _libxc_mgga_x_eps(model, rho, spin)
            ecW, ecWp = _scan_r2scan_ld_corr(
                up, dn, grad_up, grad_dn, tau_tot, variant=model
            )

            if which == "W":
                return ex + ecW
            else:
                return 0.5 * ecWp

        # LDA/PBE/TPSS/MSM
        if model in ("LDA", "PBE", "TPSS", "MSM"):
            up = np.maximum(n_up, 1e-20)
            dn = np.maximum(n_dn, 1e-20)

            gr_up = _vec_norm(grad_up)
            gr_dn = _vec_norm(grad_dn)
            dens_u = 2.0 * up
            dens_d = 2.0 * dn
            gr_u = 2.0 * gr_up
            gr_d = 2.0 * gr_dn

            if model in ("TPSS", "MSM"):
                tauu = np.maximum(2.0 * tau_up, 1e-20)
                taud = np.maximum(2.0 * tau_dn, 1e-20)
            else:
                tauu = np.ones_like(up)
                taud = np.ones_like(dn)

            tauw_u = gr_u * gr_u / (8.0 * np.maximum(dens_u, 1e-20))
            tauw_d = gr_d * gr_d / (8.0 * np.maximum(dens_d, 1e-20))

            exLDA_u, exPBE_u, exTPSS_u = exchange2_unpol(dens_u, gr_u, tauu, tauw_u)
            exLDA_d, exPBE_d, exTPSS_d = exchange2_unpol(dens_d, gr_d, taud, tauw_d)

            rs = (3.0 / (4.0 * pi * n_safe)) ** (1.0 / 3.0)
            zeta = (up - dn) / np.maximum(n_safe, 1e-20)

            gr = _vec_norm(grad_tot)
            phi = _phi(zeta)
            kF = (3.0 * pi * pi * n_safe) ** (1.0 / 3.0)
            kS = np.sqrt(4.0 * kF / pi)
            t = np.abs(gr) / (2.0 * kS * n_safe * phi)

            if model == "LDA":
                if which == "W":
                    ex = 0.5 * (dens_u * exLDA_u + dens_d * exLDA_d) / np.maximum(n_safe, 1e-20)
                    ec = ECPW92(rs, zeta)
                    return ex + ec
                else:
                    return 0.5 * ECPW922(rs, zeta)

            if model == "PBE":
                if which == "W":
                    ex = 0.5 * (dens_u * exPBE_u + dens_d * exPBE_d) / np.maximum(n_safe, 1e-20)
                    ec = ECPBE(rs, zeta, t)
                    return ex + ec
                else:
                    return 0.5 * ECPBE2(rs, zeta, t)

            # TPSS / MSM
            tau_tot = np.maximum(tau_up + tau_dn, 1e-20)
            if which == "W":
                ex = 0.5 * (dens_u * exTPSS_u + dens_d * exTPSS_d) / np.maximum(n_safe, 1e-20)
                return ex + ECTPSS(up, dn, grad_up, grad_dn, tau_tot)
            else:
                if model == "TPSS":
                    return 0.5 * ECTPSS2(up, dn, grad_up, grad_dn, tau_tot)
                else:  # MSM
                    return 0.5 * ECTPSS2_MSM(up, dn, grad_up, grad_dn, tau_tot)

        raise ValueError(f"Unknown model {model}")

    return exc

# ============================================================
#  Integration and reference densities
# ============================================================

def compute_W_for_density(mf_int, dm, exc_func, xctype):
    mol = mf_int.mol
    ni = mf_int._numint
    grids = mf_int.grids
    grids.build(with_non0tab=True)

    coords = grids.coords
    weights = grids.weights

    ao_deriv = 2 if xctype.upper() == "MGGA" else 1
    ao = ni.eval_ao(mol, coords, deriv=ao_deriv)

    spin = 0 if mol.spin == 0 else 1

    if spin == 0:
        rho = ni.eval_rho(mol, ao, dm, xctype=xctype)
        n = rho[0]
    else:
        dm_a, dm_b = dm
        rho_a = ni.eval_rho(mol, ao, dm_a, xctype=xctype)
        rho_b = ni.eval_rho(mol, ao, dm_b, xctype=xctype)
        rho = (rho_a, rho_b)
        n = rho_a[0] + rho_b[0]

    exc = exc_func(rho, spin)
    return np.dot(weights, n * exc)

def get_wy_dm_from_target_same_basis(mol, dm_target, verbose=True):
    """
    Wu–Yang KS inversion targeting dm_target, using SAME AO basis.
    Returns:
      - closed shell: dm (nao,nao)
      - open shell: (dm_a, dm_b)
    """
    from kspies import wy

    if mol.spin == 0:
        inv = wy.RWY(mol, dm_target)
        inv.run()
        if not bool(getattr(inv, "converged", False)):
            raise RuntimeError("RWY did not converge for dm_target")
        if verbose:
            inv.info()
        return inv.dm

    if isinstance(dm_target, (tuple, list)):
        dm_a, dm_b = dm_target
        dm_tar = np.array([dm_a, dm_b])
    else:
        dm_tar = np.array(dm_target)
        if dm_tar.ndim != 3 or dm_tar.shape[0] != 2:
            raise ValueError("For open-shell WY, dm_target must be (dm_a,dm_b) or array (2,nao,nao).")

    inv = wy.UWY(mol, dm_tar)
    inv.run()
    if not bool(getattr(inv, "converged", False)):
        raise RuntimeError("UWY did not converge for dm_target")
    if verbose:
        inv.info()
    dm = inv.dm
    return (dm[0], dm[1])

def get_ccsd_dm_ao(mf, conv_tol=1e-10):
    """
    CCSD 1-RDM in AO representation.
    Returns:
      - closed shell: dm (nao,nao)
      - open shell: (dm_a, dm_b)
    """
    if mf.mol.spin == 0:
        mycc = cc.CCSD(mf)
        mycc.conv_tol = conv_tol
        mycc.kernel()
        dm = mycc.make_rdm1(ao_repr=True)
        return dm

    mycc = cc.UCCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.kernel()
    dm = mycc.make_rdm1(ao_repr=True)

    if isinstance(dm, np.ndarray) and dm.ndim == 3 and dm.shape[0] == 2:
        return (dm[0], dm[1])
    if isinstance(dm, (tuple, list)) and len(dm) == 2:
        return (dm[0], dm[1])

    raise RuntimeError("Unexpected UCCSD make_rdm1(ao_repr=True) format.")

# ============================================================
# Main driver
# ============================================================

def run_test(atom="Ne 0 0 0", basis="unc-cc-pcvqz", grid_level=9, prune_none=True, conv_tol=1e-12,
             models=None, do_pbe=True, do_pbe0=True, spin=0, charge=0):
    mol = gto.M(
        atom=atom,
        basis=basis,
        spin=spin,
        charge=charge,
        verbose=3,
    )

    # HF reference density
    mf_hf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    mf_hf.conv_tol = conv_tol
    mf_hf.kernel()
    dm_hf = mf_hf.make_rdm1()

    # PBE / PBE0 densities
    dm_pbe = dm_pbe0 = None
    if do_pbe:
        mf_pbe = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf_pbe.xc = "PBE"
        mf_pbe.conv_tol = conv_tol
        mf_pbe.grids.level = grid_level
        if prune_none:
            mf_pbe.grids.prune = None
        mf_pbe.kernel()
        dm_pbe = mf_pbe.make_rdm1()

    if do_pbe0:
        mf_pbe0 = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf_pbe0.xc = "PBE0"
        mf_pbe0.conv_tol = conv_tol
        mf_pbe0.grids.level = grid_level
        if prune_none:
            mf_pbe0.grids.prune = None
        mf_pbe0.kernel()
        dm_pbe0 = mf_pbe0.make_rdm1()

    # CCSD density (AO)
    dm_ccsd = get_ccsd_dm_ao(mf_hf, conv_tol=1e-10)

    # Every source density is a mandatory Wu-Yang target. Only converged WY
    # densities are exposed to the DFT-model loop.
    source_densities = [("HF", dm_hf)]
    if dm_pbe is not None:
        source_densities.append(("PBE", dm_pbe))
    if dm_pbe0 is not None:
        source_densities.append(("PBE0", dm_pbe0))
    source_densities.append(("CCSD", dm_ccsd))

    densities = []
    for source_name, dm_target in source_densities:
        dm_wy = get_wy_dm_from_target_same_basis(
            mol, dm_target, verbose=True
        )
        densities.append((f"WY[{source_name}]", dm_wy))

    # Integration driver
    mf_int = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf_int.grids.level = grid_level
    if prune_none:
        mf_int.grids.prune = None

    if models is None:
        models = DFT_MODELS

    nelec = mol.nelectron

    print("\n=== Reference densities ===")
    print(f"atom={atom.strip()}, basis={basis}, grid_level={grid_level}, prune={mf_int.grids.prune}, mol.spin={mol.spin}")
    print("\n=== Results: W_inf and W'_inf (a.u.) ===")

    for dens_name, dm in densities:
        print(f"\n--- Using {dens_name} density ---")
        for model in models:
            excW = make_exc_func(model, "W")
            excWp = make_exc_func(model, "Wp")
            xctype = model_xctype(model)

            W  = compute_W_for_density(mf_int, dm, excW,  xctype=xctype)
            Wp = compute_W_for_density(mf_int, dm, excWp, xctype=xctype)

            print(f"{model:7s}  W_inf={W: .12f}  W'_inf={Wp: .12f}  (W/N={W/nelec: .12f}, W'/N={Wp/nelec: .12f})")

    print("")

if __name__ == "__main__":
    # Example: H atom (open shell)
    run_test(atom="He 0 0 0",
             basis="aug-cc-pVQz",
             grid_level=9,
             prune_none=True,
             conv_tol=1e-8,
             spin=0,
             charge=0,
             models=["LDA", "PBE", "TPSS", "SCAN", "R2SCAN",
                     "PC", "mPC", "hPC", "fPC", "ePC"],
#             models=["LDA", "PBE", "TPSS", "SCAN", "R2SCAN", "MSM",
#                     "PC", "PC4", "nPC", "mPC", "hPC", "fPC", "ePC"],
             do_pbe=False,
             do_pbe0=False)

