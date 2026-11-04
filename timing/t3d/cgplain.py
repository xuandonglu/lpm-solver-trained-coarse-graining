"""Unpreconditioned CG from x0 = 0 (the LPM-C solverCG algorithm: MKL dcg without the preconditioner switch, stopping
rule ||r|| <= tol ||r0||, r0 = b for the zero start; LPM-C uses tol = 1e-4 via dpar[0] = 1e-8 on squared norms).  The
absolute term of LPM-C (dpar[1] = 1e-12 on ||r||^2) is unit dependent and dropped here (pure relative 1e-4).
Same matrix-free K u kernel (lin3d.k_apply) as every other method of the benchmark, so only the algorithm differs."""
import math
import numpy as np
from numba import njit, prange
from lin3d import k_apply


@njit(cache=True, parallel=True)
def cg3(nbr, ndir, Kn, Tv, free, b, x, tol, maxit):
    """K_ff x_f = b_f on the free particles; b, x (N, 3) zero on fixed particles.  Returns (iterations, rel. residual)."""
    N = nbr.shape[0]
    r = b - k_apply(x, nbr, ndir, Kn, Tv)
    for i in prange(N):
        if not free[i]:
            r[i, 0] = 0.0
            r[i, 1] = 0.0
            r[i, 2] = 0.0
    bn = 0.0
    for i in prange(N):
        bn += b[i, 0] * b[i, 0] + b[i, 1] * b[i, 1] + b[i, 2] * b[i, 2]
    bn = math.sqrt(bn)
    if bn == 0.0:
        return 0, 0.0
    p = r.copy()
    rr = 0.0
    for i in prange(N):
        rr += r[i, 0] * r[i, 0] + r[i, 1] * r[i, 1] + r[i, 2] * r[i, 2]
    it = 0
    for it in range(1, maxit + 1):
        Ap = k_apply(p, nbr, ndir, Kn, Tv)
        pAp = 0.0
        for i in prange(N):
            if free[i]:
                pAp += p[i, 0] * Ap[i, 0] + p[i, 1] * Ap[i, 1] + p[i, 2] * Ap[i, 2]
        alpha = rr / pAp
        rn = 0.0
        for i in prange(N):
            if free[i]:
                for d in range(3):
                    x[i, d] += alpha * p[i, d]
                    r[i, d] -= alpha * Ap[i, d]
                rn += r[i, 0] * r[i, 0] + r[i, 1] * r[i, 1] + r[i, 2] * r[i, 2]
        if math.sqrt(rn) <= tol * bn:
            rr = rn
            break
        beta = rn / rr
        rr = rn
        for i in prange(N):
            if free[i]:
                for d in range(3):
                    p[i, d] = r[i, d] + beta * p[i, d]
    return it, math.sqrt(rr) / bn
