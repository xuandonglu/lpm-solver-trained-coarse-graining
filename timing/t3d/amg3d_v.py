"""I8b AMG-PCG (timing/t3d/amg3d.py, imported read-only) for specimens with voids (numerical environment, numba).

Unchanged: the geometry-only part (AMGGeom: 3x3x3 aggregates of the BC-free box of the full n^3 grid, rigid-body-mode
tentative prolongators of all levels), the V/W-cycle, smoothers, Galerkin products, PCG and its stopping rule.
Changed for voids (per specimen, timed with the solve):
  * the fine operator, its block diagonal and the level-1 Galerkin product use the specimen's neighbour table (a bond
    exists only if both particles are solid; void particles have no bonds) and free = BC-free & solid;
  * level-1 diagonal 6x6 blocks are made positive definite in their null directions only (an aggregate with few or no
    solid particles: the modes of P that vanish on the solid particles, or are zero-energy (floppy) modes of a porous
    aggregate, never appear in a residual, so any positive value is exact; lambda_ref 1e-3 is used);
  * the displacement of void particles is set to 0 at the end (the prolongation writes into them, they are decoupled).
"""
import math
import time
import numpy as np
import scipy.linalg as sla
import lin3d as L3
import amg3d as A3

AMGGeom = A3.AMGGeom


def specimen_nbr(lat, solid):
    """neighbour table of a specimen: -1 for bonds that touch a void particle."""
    nb = lat.nbr
    ok = (nb >= 0) & solid[:, None] & solid[np.where(nb >= 0, nb, 0)]
    return np.ascontiguousarray(np.where(ok, nb, -1))


def regularise_blocks(A, rel=1e-9, add=1e-3):
    """A (n, 27, 6, 6) stencil: null directions of the diagonal blocks A[:, 13] -> add * lambda_ref."""
    D = A[:, 13]
    w, V = np.linalg.eigh(0.5 * (D + D.transpose(0, 2, 1)))
    lref = float(np.median(w[:, -1][w[:, -1] > 0])) if np.any(w[:, -1] > 0) else 1.0
    bad = w < rel * lref
    nbad = int(bad.any(1).sum())
    if nbad:
        idx = np.nonzero(bad.any(1))[0]
        for i in idx:
            vb = V[i][:, bad[i]]
            A[i, 13] += add * lref * (vb @ vb.T)
    return nbad


class AMG3DV(A3.AMG3D):
    def __init__(self, geom, Kn, Tv, nbr, solid, nu=1, omega=1.32, power_its=10, Minv0=None, nu_c=None, gamma=1,
                 w_from=1, rho0=None):
        t0 = time.perf_counter()
        self.geom, self.nu, self.omega = geom, nu, omega
        self.fine_additive = False
        self.nu_c = nu if nu_c is None else nu_c
        self.gamma = gamma
        self.w_from = w_from
        lat = geom.lat
        self.nbr = nbr
        self.Kn, self.Tv = Kn, Tv
        self.solid = solid
        self.free = geom.free & solid
        self.N = lat.N
        if Minv0 is None:
            Minv0 = L3.k_block_jacobi(L3.k_block_diag(nbr, lat.ndir, Kn, Tv), self.free)
        else:
            Minv0 = Minv0 * self.free[:, None, None]
        self.M0 = Minv0
        self.rho0 = A3._power_rho(self.apply0, self.M0, self.N, 3, power_its) if rho0 is None else rho0
        g0 = geom.l0
        nagg = int(np.prod(g0["g"]))
        A1 = np.empty((nagg, 27, 6, 6))
        A3._galerkin0(g0["ptr"], g0["mem"], g0["agg_of"], g0["agg_ijk"], lat.ijk, nbr, lat.ndir, Kn, Tv, g0["site_of"],
                      lat.nx, lat.ny, lat.nz, g0["P"], A1)
        self.nreg = regularise_blocks(A1)
        self.A = [A1]
        self.Minv = []
        self.rho = []
        gs = [g0["g"]]
        for k, Lg in enumerate(geom.levels):
            A = self.A[-1]
            Mi = np.linalg.inv(A[:, 13])
            self.Minv.append(Mi)
            self.rho.append(A3._power_rho(lambda x, A=A, g=Lg["g"]: self._st(A, g, x), Mi, A.shape[0], 6, power_its))
            Ac = np.empty((int(np.prod(Lg["gc"])), 27, 6, 6))
            A3._st_galerkin(A, Lg["g"][0], Lg["g"][1], Lg["g"][2], Lg["Q"], Lg["agg_of"], Lg["ptr"], Lg["mem"],
                            Lg["agg_ijk"], Ac)
            self.A.append(Ac)
            gs.append(Lg["gc"])
        self.gs = gs
        Dn = A3._dense_from_stencil(self.A[-1], gs[-1])
        Dn = 0.5 * (Dn + Dn.T)
        self.cho = sla.cho_factor(Dn, lower=False, check_finite=False)
        self.t_setup = time.perf_counter() - t0
        self._x0 = np.zeros((self.N, 3))
        self._r0 = np.zeros((self.N, 3))

    def apply0(self, x, out=None):
        y = np.empty_like(x) if out is None else out
        A3._kx_free(x, x, self.nbr, self.geom.lat.ndir, self.Kn, self.Tv, self.free, y, False)
        return y

    def resid0(self, x, b, r):
        A3._kx_free(x, b, self.nbr, self.geom.lat.ndir, self.Kn, self.Tv, self.free, r, True)

    def pcg(self, b, tol, maxit=2000):
        x, it, rr = super().pcg(b, tol, maxit)
        x[~self.solid] = 0.0
        return x, it, rr
