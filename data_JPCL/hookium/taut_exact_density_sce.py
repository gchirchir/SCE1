#!/usr/bin/env python3
"""Exact SCE W_inf and W'_inf of Hookium on Taut's ANALYTIC densities (SI Table S10, R2-iii, revision 1).

At Taut's frequencies (omega = 1/2, 1/10, 0.0365373) the ground state is
Psi = exp(-omega R_cm^2) exp(-omega r12^2/4) sum_k a_k r12^k with the terminating recurrence
a_{k+1}(k+1)(k+2) = a_k + omega (k-n-1) a_{k-1}. The one-electron density is the convolution of the
centre-of-mass Gaussian with the relative distribution (evaluated by radial quadrature); W_inf and W'_inf
then follow from the exact two-electron expressions of the main text (Eqs. 8-13), integrated in the
variable u = N_e(r). Output reproduces the literature values (Smiga et al. 2022; Kooi & Gori-Giorgi 2018):
  0.5: -0.743153 / 0.208057;  0.1: -0.304219 / 0.054142;  0.0365373: -0.169884 / 0.022415.
"""
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.integrate import cumulative_trapezoid as ctz, quad
from scipy.interpolate import interp1d

def taut_poly(n, w):
    a=[1.0,0.5]
    for j in range(1,n+1):
        a.append((a[j]+w*(j-n-1)*a[j-1])/((j+1)*(j+2)))
    return np.array(a[:n+1])

def density(n, w, x):
    """exact one-electron density (normalised to 2) on grid x"""
    c=taut_poly(n,w)
    t=np.linspace(0,12/np.sqrt(w),20001)            # t = s/2
    s=2*t
    prel=np.exp(-w*s*s/2)*P.polyval(s,c)**2          # |phi_rel(s)|^2
    A=prel; A=A/np.trapezoid(4*np.pi*t*t*A,t)        # normalised distribution of t
    a=2*w                                            # CM |Phi|^2 ~ exp(-2 w R^2)
    G0=(a/np.pi)**1.5
    rho=np.empty_like(x)
    for i,xi in enumerate(x):
        if xi<1e-8: xi=1e-8
        inner=G0*(np.exp(-a*(xi-t)**2)-np.exp(-a*(xi+t)**2))/(2*a)
        rho[i]=2*(2*np.pi/xi)*np.trapezoid(t*A*inner,t)
    return rho

def sce(x,rho):
    ne=np.concatenate([[0],ctz(4*np.pi*x*x*rho,x)]); ne*=2/ne[-1]
    ok=np.concatenate([[True],np.diff(ne)>0])
    inv=interp1d(ne[ok],x[ok],bounds_error=False,fill_value=(x[0],x[-1]))
    R=interp1d(x,rho)
    def pair(u):
        r=float(inv(u)); f=float(inv(2-u)); fp=-R(r)*r*r/(R(f)*f*f)
        w1=np.sqrt((r*r+f*f)/(r*f*(r+f)**3)); w2=np.sqrt(-2*(1+fp*fp)/(fp*(r+f)**3)); return r,f,w1,w2
    I=lambda g: 2*quad(lambda t: np.exp(t)*g(pair(np.exp(t))),-25,0,limit=400)[0]
    vee=I(lambda p:0.5/(p[0]+p[1])); wp=I(lambda p:0.5*(2*p[2]+p[3])/4)
    # Hartree U = 1/2 int rho v_H
    inner=np.concatenate([[0],ctz(4*np.pi*x*x*rho,x)])
    outer=np.concatenate([[0],ctz(4*np.pi*x*rho,x)]); outer=outer[-1]-outer
    vH=inner/np.maximum(x,1e-12)+outer
    U=0.5*np.trapezoid(4*np.pi*x*x*rho*vH,x)
    return vee-U, wp

for n,w,lit in [(1,0.5,(-0.743,0.208)),(2,0.1,(-0.304,0.054)),(3,0.036537265599264124,(-0.170,0.022))]:
    x=np.linspace(1e-6,14/np.sqrt(w),6001)
    rho=density(n,w,x)
    print(f"omega {w:.7f}: norm {np.trapezoid(4*np.pi*x*x*rho,x):.6f}", end=' ')
    W,Wp=sce(x,rho)
    print(f"W_inf {W:.6f} W' {Wp:.6f}  lit {lit}")
