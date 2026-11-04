"""Aggregation AMG (graph V-cycle) for the linearised 3D SC LPM -- the 3D adaptation of gmg.py (GraphVCycle).

Kept from GraphVCycle:
  * geometric aggregation of the free particles into a x a x a cells (a = agg_size), rigid-body-mode tentative
    prolongator (per-aggregate QR of the 6 rigid modes: gmg._rbm(X, 3) and gmg._tentative), unsmoothed aggregation (UA,
    the gmg default; its smoothed variant needs the assembled fine matrix);
  * damped block-Jacobi smoother (3x3 particle blocks on the finest level, 6x6 node blocks below), nu sweeps before and
    after the coarse correction, omega = 1.32 / rho(D^-1 A) with rho from a power iteration;
  * Galerkin coarse operators P^T A P, dense Cholesky on the coarsest level (<= coarse_max unknowns);
  * the V-cycle is only the preconditioner of CG, so the stopping rule (and hence the accuracy) is the solver's.
Changed for 3D / speed (they change the cost, not the preconditioner, except the aggregate boundary rule):
  * finest level matrix-free: A x = lin3d.k_apply, block diagonal = lin3d.k_block_diag (the assembled fine K of n = 72
    would have 2e8 non-zeros and every CSR mat-vec would move ~2.5 GB);
  * level-1 Galerkin operator computed directly from the bond parameters (row blocks of K as in lin3d._csr_values,
    projected on the fly; numba, one aggregate row per thread, no K stored);
  * levels >= 1 stored as a structured 27-point block stencil (6x6 blocks) on the aggregate grid; numba mat-vec,
    restriction, prolongation and Galerkin product;
  * aggregate boundaries: chunks of a particles per axis with the remainder merged into the last chunk (no thin,
    rank-deficient aggregates); chunk="floor" reproduces gmg._cells (used for the check against GraphVCycle);
  * the first pre-smoothing step starts from x = 0 without a residual mat-vec (same result as gmg); the fine-level
    residual b - K x is one fused numba kernel (_kx_free);
  * options: W-cycle on the coarse levels (gamma = 2, used in I8b), sweeps on the coarse levels (nu_c), additive
    instead of multiplicative coarse correction on the finest level (fine_additive, tested, not faster), rho0 given
    (shared by the BC sets of one specimen);
  * PCG with the I8 stopping rule ||r|| <= tol ||b|| (x0 = 0, tested every iteration, as lin3d._pcg);
  * geometry-only parts (aggregation, tentative prolongators of all levels) are built once per lattice and BC set,
    like the lattice neighbour lists, and reused for every specimen (AMGGeom); the per-specimen numeric setup
    (block inverses, rho, Galerkin products, coarse Cholesky) is part of every timed solve.
"""
import math
import time
import numpy as np
import scipy.linalg as sla
from numba import njit, prange

import lin3d as L3
from gmg import _tentative, _rbm

OFF27 = np.array([(d % 3 - 1, (d // 3) % 3 - 1, d // 9 - 1) for d in range(27)], np.int64)   # d = (z+1)9+(y+1)3+(x+1)


def chunk_index(L, a, rule):
    """chunk id of positions 0..L-1 along one axis, number of chunks."""
    if rule == "floor":
        return np.arange(L) // a, -(-L // a)
    nch = max(1, L // a)
    return np.minimum(np.arange(L) // a, nch - 1), nch


def _members(agg_of_item, nagg):
    """CSR of the items of every aggregate (items with agg < 0 are skipped)."""
    idx = np.nonzero(agg_of_item >= 0)[0]
    order = idx[np.argsort(agg_of_item[idx], kind="stable")]
    ptr = np.zeros(nagg + 1, np.int64)
    np.cumsum(np.bincount(agg_of_item[idx], minlength=nagg), out=ptr[1:])
    return ptr, order.astype(np.int64)


def _grid_ijk(g):
    gx, gy, gz = g
    k, j, i = np.meshgrid(np.arange(gz), np.arange(gy), np.arange(gx), indexing="ij")
    return np.ascontiguousarray(np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(np.int64))


# =====================================================================================================================
# numba kernels
# =====================================================================================================================
@njit(cache=True)
def _oidx(d0, d1, d2):
    return (d2 + 2) * 25 + (d1 + 2) * 5 + (d0 + 2)


@njit(cache=True)
def _krow(p, ijk, nbr, ndir, Kn, Tv, buf):
    """3x3 blocks K_pq of row p (q = p + offset, offset index _oidx) added to buf (125, 3, 3); same algebra as
    lin3d._csr_values."""
    for c in range(2):
        k0 = 0 if c == 0 else 6
        k1 = 6 if c == 0 else 18
        T0 = 0.0
        T1 = 0.0
        T2 = 0.0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for a in range(k0, k1):
            if nbr[p, a] < 0:
                continue
            T0 += Tv[p, a] * ndir[a, 0]
            T1 += Tv[p, a] * ndir[a, 1]
            T2 += Tv[p, a] * ndir[a, 2]
            s0 += ndir[a, 0]
            s1 += ndir[a, 1]
            s2 += ndir[a, 2]
        for b in range(k0, k1):
            q = nbr[p, b]
            if q < 0:
                continue
            oq = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
            for d in range(3):
                Td = T0 if d == 0 else (T1 if d == 1 else T2)
                sd = s0 if d == 0 else (s1 if d == 1 else s2)
                w = Kn[p, b] * ndir[b, d] + 0.5 * (Td + Tv[p, b] * sd)
                for e in range(3):
                    v = w * ndir[b, e]
                    buf[62, d, e] += v
                    buf[oq, d, e] -= v
    for a in range(18):
        i = nbr[p, a]
        if i < 0:
            continue
        a2 = a ^ 1
        k0 = 0 if a < 6 else 6
        k1 = 6 if a < 6 else 18
        T0 = 0.0
        T1 = 0.0
        T2 = 0.0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for b in range(k0, k1):
            if nbr[i, b] < 0:
                continue
            T0 += Tv[i, b] * ndir[b, 0]
            T1 += Tv[i, b] * ndir[b, 1]
            T2 += Tv[i, b] * ndir[b, 2]
            s0 += ndir[b, 0]
            s1 += ndir[b, 1]
            s2 += ndir[b, 2]
        oi = _oidx(ijk[i, 0] - ijk[p, 0], ijk[i, 1] - ijk[p, 1], ijk[i, 2] - ijk[p, 2])
        for e in range(3):
            Te = T0 if e == 0 else (T1 if e == 1 else T2)
            se = s0 if e == 0 else (s1 if e == 1 else s2)
            w = Kn[i, a2] * ndir[a2, e] + 0.5 * (Te + Tv[i, a2] * se)
            for d in range(3):
                buf[oi, d, e] += ndir[a, d] * w
        for b in range(k0, k1):
            q = nbr[i, b]
            if q < 0:
                continue
            Hab = 0.5 * (Tv[i, a2] + Tv[i, b])
            if b == a2:
                Hab += Kn[i, a2]
            oq = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
            for d in range(3):
                for e in range(3):
                    buf[oq, d, e] -= Hab * ndir[a, d] * ndir[b, e]


@njit(cache=True, parallel=True)
def _galerkin0(ptr, members, agg_of, agg_ijk, ijk, nbr, ndir, Kn, Tv, site_of, nx, ny, nz, P, A1):
    """A1[I, d] = sum_{p in I, q in J} P_p^T K_pq P_q, J = I + OFF27[d] (fine level -> level 1, from the bonds)."""
    nagg = ptr.shape[0] - 1
    for I in prange(nagg):
        buf = np.zeros((125, 3, 3))
        C = np.zeros((27, 3, 6))
        touched = np.zeros(27, np.bool_)
        for d in range(27):
            for r in range(6):
                for c in range(6):
                    A1[I, d, r, c] = 0.0
        for m in range(ptr[I], ptr[I + 1]):
            p = members[m]
            buf[:] = 0.0
            _krow(p, ijk, nbr, ndir, Kn, Tv, buf)
            C[:] = 0.0
            touched[:] = False
            for o in range(125):
                if buf[o, 0, 0] == 0.0 and buf[o, 1, 1] == 0.0 and buf[o, 2, 2] == 0.0 and buf[o, 0, 1] == 0.0 \
                        and buf[o, 0, 2] == 0.0 and buf[o, 1, 2] == 0.0 and buf[o, 1, 0] == 0.0 \
                        and buf[o, 2, 0] == 0.0 and buf[o, 2, 1] == 0.0:
                    continue
                i0 = ijk[p, 0] + o % 5 - 2
                i1 = ijk[p, 1] + (o // 5) % 5 - 2
                i2 = ijk[p, 2] + o // 25 - 2
                if i0 < 0 or i0 >= nx or i1 < 0 or i1 >= ny or i2 < 0 or i2 >= nz:
                    continue
                q = site_of[(i2 * ny + i1) * nx + i0]
                if q < 0:
                    continue
                J = agg_of[q]
                if J < 0:
                    continue
                dd = (agg_ijk[J, 2] - agg_ijk[I, 2] + 1) * 9 + (agg_ijk[J, 1] - agg_ijk[I, 1] + 1) * 3 \
                    + (agg_ijk[J, 0] - agg_ijk[I, 0] + 1)
                touched[dd] = True
                for a in range(3):
                    for c in range(6):
                        C[dd, a, c] += buf[o, a, 0] * P[q, 0, c] + buf[o, a, 1] * P[q, 1, c] + buf[o, a, 2] * P[q, 2, c]
            for dd in range(27):
                if not touched[dd]:
                    continue
                for r in range(6):
                    for c in range(6):
                        A1[I, dd, r, c] += P[p, 0, r] * C[dd, 0, c] + P[p, 1, r] * C[dd, 1, c] + P[p, 2, r] * C[dd, 2, c]


@njit(cache=True, parallel=True)
def _kx_free(u, b, nbr, ndir, Kn, Tv, free, out, sub):
    """out = b - K u (sub) or K u (not sub) on free particles, 0 on fixed; same algebra as lin3d.k_apply."""
    N = nbr.shape[0]
    S, TS = L3.k_shell_sums(u, nbr, ndir, Tv)
    for i in prange(N):
        if not free[i]:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
            continue
        ox = 0.0
        oy = 0.0
        oz = 0.0
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            c = 0 if k < 6 else 1
            dl = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
            f = 2.0 * Kn[i, k] * dl + 0.5 * (TS[i, c] + TS[q, c]) + 0.5 * Tv[i, k] * (S[i, c] + S[q, c])
            ox += f * ndir[k, 0]
            oy += f * ndir[k, 1]
            oz += f * ndir[k, 2]
        if sub:
            out[i, 0] = b[i, 0] - ox
            out[i, 1] = b[i, 1] - oy
            out[i, 2] = b[i, 2] - oz
        else:
            out[i, 0] = ox
            out[i, 1] = oy
            out[i, 2] = oz


@njit(cache=True, parallel=True)
def _st_mv(A, x, gx, gy, gz, y):
    """y = A x on a 27-point 6x6-block stencil (grid gx, gy, gz)."""
    n = A.shape[0]
    for I in prange(n):
        i0 = I % gx
        i1 = (I // gx) % gy
        i2 = I // (gx * gy)
        for r in range(6):
            y[I, r] = 0.0
        for d in range(27):
            j0 = i0 + d % 3 - 1
            j1 = i1 + (d // 3) % 3 - 1
            j2 = i2 + d // 9 - 1
            if j0 < 0 or j0 >= gx or j1 < 0 or j1 >= gy or j2 < 0 or j2 >= gz:
                continue
            J = (j2 * gy + j1) * gx + j0
            for r in range(6):
                s = 0.0
                for c in range(6):
                    s += A[I, d, r, c] * x[J, c]
                y[I, r] += s


@njit(cache=True, parallel=True)
def _st_galerkin(A, gx, gy, gz, Q, agg_of, ptr, members, agg_ijk, Ac):
    """Ac[I, dd] = sum_{n in I, m in J} Q_n^T A_nm Q_m (level l -> l+1, both 27-point stencils)."""
    nagg = ptr.shape[0] - 1
    for I in prange(nagg):
        C = np.zeros((27, 6, 6))
        touched = np.zeros(27, np.bool_)
        for d in range(27):
            for r in range(6):
                for c in range(6):
                    Ac[I, d, r, c] = 0.0
        for mm in range(ptr[I], ptr[I + 1]):
            n_ = members[mm]
            i0 = n_ % gx
            i1 = (n_ // gx) % gy
            i2 = n_ // (gx * gy)
            C[:] = 0.0
            touched[:] = False
            for d in range(27):
                j0 = i0 + d % 3 - 1
                j1 = i1 + (d // 3) % 3 - 1
                j2 = i2 + d // 9 - 1
                if j0 < 0 or j0 >= gx or j1 < 0 or j1 >= gy or j2 < 0 or j2 >= gz:
                    continue
                m_ = (j2 * gy + j1) * gx + j0
                J = agg_of[m_]
                dd = (agg_ijk[J, 2] - agg_ijk[I, 2] + 1) * 9 + (agg_ijk[J, 1] - agg_ijk[I, 1] + 1) * 3 \
                    + (agg_ijk[J, 0] - agg_ijk[I, 0] + 1)
                touched[dd] = True
                for a in range(6):
                    for c in range(6):
                        s = 0.0
                        for b in range(6):
                            s += A[n_, d, a, b] * Q[m_, b, c]
                        C[dd, a, c] += s
            for dd in range(27):
                if not touched[dd]:
                    continue
                for r in range(6):
                    for c in range(6):
                        s = 0.0
                        for a in range(6):
                            s += Q[n_, a, r] * C[dd, a, c]
                        Ac[I, dd, r, c] += s


@njit(cache=True, parallel=True)
def _restrict(P, ptr, members, r, rc):
    """rc[I] = sum_{i in I} P_i^T r_i  (P (n, k, 6), r (n, k))."""
    nagg = ptr.shape[0] - 1
    k = P.shape[1]
    for I in prange(nagg):
        for c in range(6):
            s = 0.0
            for mm in range(ptr[I], ptr[I + 1]):
                i = members[mm]
                for a in range(k):
                    s += P[i, a, c] * r[i, a]
            rc[I, c] = s


@njit(cache=True, parallel=True)
def _prolong_add(P, agg_of, e, x):
    """x_i += P_i e[agg(i)] for items with agg >= 0."""
    n = P.shape[0]
    k = P.shape[1]
    for i in prange(n):
        J = agg_of[i]
        if J < 0:
            continue
        for a in range(k):
            s = 0.0
            for c in range(6):
                s += P[i, a, c] * e[J, c]
            x[i, a] += s


@njit(cache=True, parallel=True)
def _bj(M, r, w, z):
    """z = w M r (block Jacobi; M (n, k, k), zero rows on fixed items)."""
    n = M.shape[0]
    k = M.shape[1]
    for i in prange(n):
        for a in range(k):
            s = 0.0
            for b in range(k):
                s += M[i, a, b] * r[i, b]
            z[i, a] = w * s


@njit(cache=True, parallel=True)
def _bj_add(M, r, w, x):
    n = M.shape[0]
    k = M.shape[1]
    for i in prange(n):
        for a in range(k):
            s = 0.0
            for b in range(k):
                s += M[i, a, b] * r[i, b]
            x[i, a] += w * s


@njit(cache=True, parallel=True)
def _resid(Ax, b, free, r):
    """r = b - Ax on free items, 0 elsewhere."""
    n = Ax.shape[0]
    k = Ax.shape[1]
    for i in prange(n):
        if free[i]:
            for a in range(k):
                r[i, a] = b[i, a] - Ax[i, a]
        else:
            for a in range(k):
                r[i, a] = 0.0


@njit(cache=True, parallel=True)
def _sub(b, Ax, r):
    n = Ax.shape[0]
    k = Ax.shape[1]
    for i in prange(n):
        for a in range(k):
            r[i, a] = b[i, a] - Ax[i, a]


@njit(cache=True, parallel=True)
def _mask(x, free):
    for i in prange(x.shape[0]):
        if not free[i]:
            for a in range(x.shape[1]):
                x[i, a] = 0.0


@njit(cache=True, parallel=True)
def _dot(x, y):
    s = 0.0
    for i in prange(x.shape[0]):
        for a in range(x.shape[1]):
            s += x[i, a] * y[i, a]
    return s


@njit(cache=True, parallel=True)
def _cg_step(x, r, p, Ap, alpha):
    """x += alpha p; r -= alpha Ap; returns ||r||^2."""
    s = 0.0
    for i in prange(x.shape[0]):
        for a in range(x.shape[1]):
            x[i, a] += alpha * p[i, a]
            r[i, a] -= alpha * Ap[i, a]
            s += r[i, a] * r[i, a]
    return s


@njit(cache=True, parallel=True)
def _xpby(z, beta, p):
    """p = z + beta p."""
    for i in prange(z.shape[0]):
        for a in range(z.shape[1]):
            p[i, a] = z[i, a] + beta * p[i, a]


def _dense_from_stencil(A, g):
    gx, gy, gz = g
    n = A.shape[0]
    ijk = _grid_ijk(g)
    D = np.zeros((n, 6, n, 6))
    for d in range(27):
        j = ijk + OFF27[d]
        ok = np.all((j >= 0) & (j < np.array(g)), 1)
        I = np.nonzero(ok)[0]
        J = (j[ok, 2] * gy + j[ok, 1]) * gx + j[ok, 0]
        D[I, :, J, :] = A[I, d]
    return D.reshape(6 * n, 6 * n)


# =====================================================================================================================
# geometry (built once per lattice and BC set)
# =====================================================================================================================
class AMGGeom:
    """Aggregation and tentative prolongators of all levels for one lattice and one set of free particles (a box)."""

    def __init__(self, lat, free, agg=3, rule="merge", coarse_max=3000, max_levels=8):
        t0 = time.perf_counter()
        self.lat, self.free = lat, np.asarray(free, bool)
        self.agg, self.rule, self.coarse_max = agg, rule, coarse_max
        fi = np.nonzero(self.free)[0]
        ijk = lat.ijk
        lo, hi = ijk[fi].min(0), ijk[fi].max(0)
        Lx = hi - lo + 1
        assert int(np.prod(Lx)) == len(fi), "free set must be a box"
        ch = [chunk_index(int(Lx[d]), agg, rule) for d in range(3)]
        g = tuple(int(c[1]) for c in ch)
        rel = ijk[fi] - lo
        agg_of = -np.ones(lat.N, np.int64)
        agg_of[fi] = (ch[2][0][rel[:, 2]] * g[1] + ch[1][0][rel[:, 1]]) * g[0] + ch[0][0][rel[:, 0]]
        nagg = g[0] * g[1] * g[2]
        ptr, mem = _members(agg_of, nagg)
        dofs = (3 * fi[:, None] + np.arange(3)[None, :]).ravel()
        B = _rbm(lat.x, 3)[dofs]
        r_, c_, v_, Bc, ncol, colmap = _tentative(np.repeat(agg_of[fi], 3), nagg, np.ascontiguousarray(B), 6, 1e-8)
        assert ncol == 6 * nagg and np.all(colmap == np.arange(6 * nagg)), "rank-deficient aggregate"
        P = np.zeros((lat.N, 3, 6))
        P[fi[r_ // 3], r_ % 3, c_ % 6] = v_
        site_of = lat.site_of
        self.l0 = dict(agg_of=agg_of, ptr=ptr, mem=mem, P=P, g=g, agg_ijk=_grid_ijk(g), site_of=site_of)
        # levels >= 1: aggregate the node grid by chunks of `agg` nodes
        self.levels = []
        Bn = Bc.reshape(nagg, 6, 6)
        while 6 * nagg > coarse_max and len(self.levels) < max_levels:
            chn = [chunk_index(g[d], agg, rule) for d in range(3)]
            gc = tuple(int(c[1]) for c in chn)
            nij = _grid_ijk(g)
            a_of = (chn[2][0][nij[:, 2]] * gc[1] + chn[1][0][nij[:, 1]]) * gc[0] + chn[0][0][nij[:, 0]]
            nc_ = gc[0] * gc[1] * gc[2]
            ptr_, mem_ = _members(a_of, nc_)
            r_, c_, v_, Bc2, ncol, colmap = _tentative(np.repeat(a_of, 6), nc_, np.ascontiguousarray(Bn.reshape(-1, 6)),
                                                       6, 1e-8)
            assert ncol == 6 * nc_
            Q = np.zeros((nagg, 6, 6))
            Q[r_ // 6, r_ % 6, c_ % 6] = v_
            self.levels.append(dict(g=g, agg_of=a_of.astype(np.int64), ptr=ptr_, mem=mem_, Q=Q, gc=gc,
                                    agg_ijk=_grid_ijk(gc)))
            g, nagg, Bn = gc, nc_, Bc2.reshape(nc_, 6, 6)
        self.g_coarsest = g
        self.t_geom = time.perf_counter() - t0

    def describe(self):
        s = ["L0 N=%d free=%d" % (self.lat.N, int(self.free.sum())), "L1 %s n=%d" % (self.l0["g"], 6 * np.prod(self.l0["g"]))]
        for k, L in enumerate(self.levels):
            s.append("L%d %s n=%d" % (k + 2, L["gc"], 6 * np.prod(L["gc"])))
        return " | ".join(s)


# =====================================================================================================================
# per-specimen hierarchy + V-cycle + PCG
# =====================================================================================================================
def _power_rho(mv, Minv, n, k, its, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, k))
    y = np.empty_like(x)
    _bj(Minv, x, 1.0, y)                  # start in the free subspace (Minv has zero rows on fixed items)
    x, y = y, x
    lam = 1.0
    for _ in range(its):
        x /= np.linalg.norm(x)
        Ax = mv(x)
        _bj(Minv, Ax, 1.0, y)
        lam = float(np.linalg.norm(y))
        x, y = y, x
    return lam


class AMG3D:
    """Numeric hierarchy for one specimen (Kn, Tv) on an AMGGeom; vcycle() is the preconditioner, pcg() the solver."""

    def __init__(self, geom, Kn, Tv, nu=1, omega=1.32, power_its=10, Minv0=None, nu_c=None, gamma=1,
                 fine_additive=False, w_from=1, rho0=None):
        t0 = time.perf_counter()
        self.geom, self.nu, self.omega = geom, nu, omega
        self.fine_additive = fine_additive                  # additive (not multiplicative) correction on level 0
        self.nu_c = nu if nu_c is None else nu_c            # sweeps on levels >= 1
        self.gamma = gamma                                  # cycle index on levels >= w_from (1 = V, 2 = W)
        self.w_from = w_from
        lat = geom.lat
        self.Kn, self.Tv = Kn, Tv
        self.free = geom.free
        self.N = lat.N
        if Minv0 is None:
            Minv0 = L3.k_block_jacobi(L3.k_block_diag(lat.nbr, lat.ndir, Kn, Tv), self.free)
        else:
            Minv0 = Minv0 * self.free[:, None, None]
        self.M0 = Minv0
        # rho(D^-1 A) of the finest level: power iteration, or given (e.g. from another BC set of the same specimen)
        self.rho0 = _power_rho(self.apply0, self.M0, self.N, 3, power_its) if rho0 is None else rho0
        g0 = geom.l0
        nagg = int(np.prod(g0["g"]))
        A1 = np.empty((nagg, 27, 6, 6))
        _galerkin0(g0["ptr"], g0["mem"], g0["agg_of"], g0["agg_ijk"], lat.ijk, lat.nbr, lat.ndir, Kn, Tv, g0["site_of"],
                   lat.nx, lat.ny, lat.nz, g0["P"], A1)
        self.A = [A1]
        self.Minv = []
        self.rho = []
        gs = [g0["g"]]
        for k, Lg in enumerate(geom.levels):
            A = self.A[-1]
            Mi = np.linalg.inv(A[:, 13])
            self.Minv.append(Mi)
            self.rho.append(_power_rho(lambda x, A=A, g=Lg["g"]: self._st(A, g, x), Mi, A.shape[0], 6, power_its))
            Ac = np.empty((int(np.prod(Lg["gc"])), 27, 6, 6))
            _st_galerkin(A, Lg["g"][0], Lg["g"][1], Lg["g"][2], Lg["Q"], Lg["agg_of"], Lg["ptr"], Lg["mem"],
                         Lg["agg_ijk"], Ac)
            self.A.append(Ac)
            gs.append(Lg["gc"])
        self.gs = gs
        Dn = _dense_from_stencil(self.A[-1], gs[-1])
        Dn = 0.5 * (Dn + Dn.T)
        self.cho = sla.cho_factor(Dn, lower=False, check_finite=False)
        self.t_setup = time.perf_counter() - t0
        # work arrays of the finest level
        self._x0 = np.zeros((self.N, 3))
        self._r0 = np.zeros((self.N, 3))

    # ------------------------------------------------------------------------------------------- operators
    def apply0(self, x, out=None):
        """K_ff x (fixed rows zero), fused numba kernel."""
        y = np.empty_like(x) if out is None else out
        lat = self.geom.lat
        _kx_free(x, x, lat.nbr, lat.ndir, self.Kn, self.Tv, self.free, y, False)
        return y

    def resid0(self, x, b, r):
        """r = b - K_ff x (fixed rows zero)."""
        lat = self.geom.lat
        _kx_free(x, b, lat.nbr, lat.ndir, self.Kn, self.Tv, self.free, r, True)

    @staticmethod
    def _st(A, g, x):
        y = np.empty_like(x)
        _st_mv(A, x, g[0], g[1], g[2], y)
        return y

    # ------------------------------------------------------------------------------------------- V-cycle
    def _coarse(self, lvl, rc):
        """approximate solve on level lvl >= 1: one cycle (V) or gamma cycles (W) from zero."""
        ec = self._vc(lvl, rc)
        k = lvl - 1
        if k == len(self.A) - 1:
            return ec                                        # exact (Cholesky)
        for _ in range(self.gamma - 1 if lvl >= self.w_from else 0):
            Aec = np.empty_like(ec)
            _st_mv(self.A[k], ec, *self.gs[k], Aec)
            ec = ec + self._vc(lvl, rc - Aec)
        return ec

    def _vc(self, lvl, b):
        """levels >= 1 (lvl = 1 ... ), b (n, 6)."""
        k = lvl - 1                                   # index into self.A / Minv / geom.levels
        A = self.A[k]
        if k == len(self.A) - 1:
            return sla.cho_solve(self.cho, b.reshape(-1), check_finite=False).reshape(-1, 6)
        g = self.gs[k]
        Lg = self.geom.levels[k]
        w = self.omega / self.rho[k]
        Mi = self.Minv[k]
        x = np.empty_like(b)
        _bj(Mi, b, w, x)
        r = np.empty_like(b)
        Ax = np.empty_like(b)
        for _ in range(self.nu_c - 1):
            _st_mv(A, x, g[0], g[1], g[2], Ax)
            _sub(b, Ax, r)
            _bj_add(Mi, r, w, x)
        _st_mv(A, x, g[0], g[1], g[2], Ax)
        _sub(b, Ax, r)
        rc = np.empty((int(np.prod(Lg["gc"])), 6))
        _restrict(Lg["Q"], Lg["ptr"], Lg["mem"], r, rc)
        ec = self._coarse(lvl + 1, rc)
        _prolong_add(Lg["Q"], Lg["agg_of"], ec, x)
        for _ in range(self.nu_c):
            _st_mv(A, x, g[0], g[1], g[2], Ax)
            _sub(b, Ax, r)
            _bj_add(Mi, r, w, x)
        return x

    def vcycle(self, b):
        """finest level, b (N, 3) zero on fixed particles -> z (N, 3)."""
        g0 = self.geom.l0
        w = self.omega / self.rho0
        if self.fine_additive:
            # additive coarse correction on the finest level: z = w D^-1 b + P B_c P^T b (no fine mat-vec)
            x = np.empty_like(b)
            _bj(self.M0, b, w, x)
            rc = np.empty((int(np.prod(g0["g"])), 6))
            _restrict(g0["P"], g0["ptr"], g0["mem"], b, rc)
            ec = self._coarse(1, rc)
            _prolong_add(g0["P"], g0["agg_of"], ec, x)
            return x
        x = np.empty_like(b)
        _bj(self.M0, b, w, x)
        r = self._r0
        for _ in range(self.nu - 1):
            self.resid0(x, b, r)
            _bj_add(self.M0, r, w, x)
        self.resid0(x, b, r)
        rc = np.empty((int(np.prod(g0["g"])), 6))
        _restrict(g0["P"], g0["ptr"], g0["mem"], r, rc)
        ec = self._coarse(1, rc)
        _prolong_add(g0["P"], g0["agg_of"], ec, x)
        for _ in range(self.nu):
            self.resid0(x, b, r)
            _bj_add(self.M0, r, w, x)
        return x

    def pcg(self, b, tol, maxit=2000):
        """K_ff x = b (b (N,3) zero on fixed); stop at ||r|| <= tol ||b|| (x0 = 0).  Returns x, iterations, relres."""
        x = np.zeros_like(b)
        r = b.copy()
        bn = math.sqrt(_dot(b, b))
        if bn == 0.0:
            return x, 0, 0.0
        z = self.vcycle(r)
        p = z.copy()
        rz = _dot(r, z)
        it = 0
        rn = bn
        Ap = np.empty_like(b)
        for it in range(1, maxit + 1):
            self.apply0(p, Ap)
            alpha = rz / _dot(p, Ap)
            rn = math.sqrt(_cg_step(x, r, p, Ap, alpha))
            if rn <= tol * bn:
                break
            z = self.vcycle(r)
            rz2 = _dot(r, z)
            _xpby(z, rz2 / rz, p)
            rz = rz2
        return x, it, rn / bn
