"""Unpreconditioned CG from zero for the 2D benchmark problems (the LPM-C solverCG algorithm, ||r|| <= tol ||b||, tol = 1e-4),
matrix-free with the same K u kernel (solver2d._kmv) as the block-Jacobi PCG of solver2d; fp64 bond parameters."""
import a2common  # noqa: F401
import math
import numpy as np
from numba import njit, prange
import el_prob as EP
from solver2d import _kmv


@njit(parallel=True, cache=True)
def cg_mf(nbr, valid, ndir, shell, opp, Kn, Tv, mask, b, x, tol, maxit, g):
    n = b.shape[0]
    r = np.empty(n)
    p = np.empty(n)
    Ap = np.empty(n)
    _kmv(x, nbr, valid, ndir, shell, opp, Kn, Tv, g, Ap)
    bb = 0.0
    rr = 0.0
    for i in prange(n):
        r[i] = (b[i] - Ap[i]) * mask[i]
        bb += b[i] * b[i]
        rr += r[i] * r[i]
    bn = math.sqrt(bb)
    if math.sqrt(rr) <= tol * bn:
        return 0
    for i in prange(n):
        p[i] = r[i]
    it = 0
    while it < maxit:
        _kmv(p, nbr, valid, ndir, shell, opp, Kn, Tv, g, Ap)
        pAp = 0.0
        for i in prange(n):
            Ap[i] *= mask[i]
            pAp += p[i] * Ap[i]
        alpha = rr / pAp
        rn = 0.0
        for i in prange(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            rn += r[i] * r[i]
        it += 1
        if math.sqrt(rn) <= tol * bn:
            break
        beta = rn / rr
        rr = rn
        for i in prange(n):
            p[i] = r[i] + beta * p[i]
    return it


def solve_cg(sy, Kn, Tv, tol=1e-4, maxit=200000):
    """all 6 loads of system sy, zero start; returns ({load: u}, total iterations)."""
    arr = sy.mf_arrays(Kn, Tv, fp32=False)
    N = sy.lat.N
    g = np.empty((N, sy.lat.NN))
    U, its = {}, 0
    for bs, d in sy.sets.items():
        mask = np.repeat((~d["fixed"]).astype(np.float64), 2)
        for ld in d["loads"]:
            ub = sy.load[ld]["ub"]
            f = np.empty(2 * N)
            _kmv(ub, *arr, g, f)
            b = -f * mask
            x = np.zeros(2 * N)
            its += cg_mf(*arr, mask, b, x, tol, maxit, g)
            U[ld] = x + ub
    return U, its
