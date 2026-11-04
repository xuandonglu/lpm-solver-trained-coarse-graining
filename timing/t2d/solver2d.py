"""Iterative solves of the 2D benchmark problems (fine lattice or any coarse lattice).

Sys(kind, n, st, sb): lattice of factor st (1 = fine LPM, st >= 2 = coarse model of factor st) carrying the boundary
value problem of factor sb (BC block layers of sb x sb fine sites, el_prob.prescribed; for a coarse model sb = st).
Per BC set (geometry only, cached): fixed nodes, free DOFs, CSR pattern of the free-free block and its gather index
into the element-assembly data (elem.ElemPattern), prescribed vectors of the loads.
Per solve: numba assembly -> b = -K u_presc -> PCG from a start vector with
  * 'bj'  : 2x2 particle block-Jacobi preconditioner, whole loop in one numba kernel;
  * 'amg' : the aggregation-based graph V-cycle of gmg_copy.GraphVCycle (dim = 2) as preconditioner.
Stopping rule: ||r|| <= tol ||b||, b = -K_fp u_p, tested every iteration, also with a warm start.
A descending list of tolerances returns one snapshot per tolerance (identical iterates for every tol).
Starts: 'zero', 'affine' (load-consistent affine field) or a coarse solution interpolated in lattice-index space
(bilinear, linear extrapolation at the edges; exact for affine fields on both lattices); prescribed DOFs exact.
Output: particle stress (numba, = lin2d.Lattice2D.stress for intact lattices).
"""
import a2common  # noqa: F401
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange
import el_prob as EP
import lin2d
import elem

TOLS = np.array(sorted({m * 10.0 ** k for k in range(-6, 0) for m in (1, 1.5, 2, 3, 4, 5, 7)}, reverse=True))


# ------------------------------------------------------------------------------------------------ numba kernels
@njit(cache=True)
def bj_inverse(indptr, indices, data):
    n = indptr.shape[0] - 1
    m = n // 2
    B = np.empty((m, 2, 2))
    for k in range(m):
        a = np.zeros((2, 2))
        for r in range(2):
            i = 2 * k + r
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j == 2 * k:
                    a[r, 0] = data[p]
                elif j == 2 * k + 1:
                    a[r, 1] = data[p]
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        B[k, 0, 0] = a[1, 1] / det
        B[k, 0, 1] = -a[0, 1] / det
        B[k, 1, 0] = -a[1, 0] / det
        B[k, 1, 1] = a[0, 0] / det
    return B


@njit(parallel=True, cache=True)
def csr_mv(indptr, indices, data, x, y):
    n = indptr.shape[0] - 1
    for i in prange(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        y[i] = s


@njit(parallel=True, cache=True)
def pcg_bj(indptr, indices, data, B, b, x, tols, maxit, snaps, iters):
    """PCG, 2x2 block-Jacobi; x = start (updated).  snaps[k] = iterate when ||r|| <= tols[k] ||b|| first holds."""
    n = b.shape[0]
    m = n // 2
    nt = tols.shape[0]
    r = np.empty(n)
    z = np.empty(n)
    p = np.empty(n)
    Ap = np.empty(n)
    for i in prange(n):
        s = 0.0
        for q in range(indptr[i], indptr[i + 1]):
            s += data[q] * x[indices[q]]
        r[i] = b[i] - s
    bb = 0.0
    rr = 0.0
    for i in prange(n):
        bb += b[i] * b[i]
        rr += r[i] * r[i]
    bn = math.sqrt(bb)
    k = 0
    it = 0
    while k < nt and math.sqrt(rr) <= tols[k] * bn:
        snaps[k, :] = x
        iters[k] = 0
        k += 1
    if k == nt:
        return 0
    rz = 0.0
    for c in prange(m):
        a0 = r[2 * c]
        a1 = r[2 * c + 1]
        z0 = B[c, 0, 0] * a0 + B[c, 0, 1] * a1
        z1 = B[c, 1, 0] * a0 + B[c, 1, 1] * a1
        z[2 * c] = z0
        z[2 * c + 1] = z1
        p[2 * c] = z0
        p[2 * c + 1] = z1
        rz += a0 * z0 + a1 * z1
    while it < maxit:
        pAp = 0.0
        for i in prange(n):
            s = 0.0
            for q in range(indptr[i], indptr[i + 1]):
                s += data[q] * p[indices[q]]
            Ap[i] = s
            pAp += p[i] * s
        alpha = rz / pAp
        rr = 0.0
        for i in prange(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            rr += r[i] * r[i]
        it += 1
        while k < nt and math.sqrt(rr) <= tols[k] * bn:
            snaps[k, :] = x
            iters[k] = it
            k += 1
        if k == nt:
            break
        rzn = 0.0
        for c in prange(m):
            a0 = r[2 * c]
            a1 = r[2 * c + 1]
            z0 = B[c, 0, 0] * a0 + B[c, 0, 1] * a1
            z1 = B[c, 1, 0] * a0 + B[c, 1, 1] * a1
            z[2 * c] = z0
            z[2 * c + 1] = z1
            rzn += a0 * z0 + a1 * z1
        beta = rzn / rz
        rz = rzn
        for i in prange(n):
            p[i] = z[i] + beta * p[i]
    while k < nt:                       # not reached within maxit
        snaps[k, :] = x
        iters[k] = -1
        k += 1
    return it


@njit(parallel=True, cache=True)
def _dot(x, y):
    s = 0.0
    for i in prange(x.shape[0]):
        s += x[i] * y[i]
    return s


@njit(parallel=True, cache=True)
def _resid(indptr, indices, data, x, b, r):
    n = indptr.shape[0] - 1
    for i in prange(n):
        s = 0.0
        for q in range(indptr[i], indptr[i + 1]):
            s += data[q] * x[indices[q]]
        r[i] = b[i] - s


@njit(parallel=True, cache=True)
def _upd_xr(x, r, p, Ap, alpha):
    rr = 0.0
    for i in prange(x.shape[0]):
        x[i] += alpha * p[i]
        r[i] -= alpha * Ap[i]
        rr += r[i] * r[i]
    return rr


@njit(parallel=True, cache=True)
def _xpby(z, beta, p):
    for i in prange(z.shape[0]):
        p[i] = z[i] + beta * p[i]


def pcg_prec(A, prec, b, x, tols, maxit, snaps, iters):
    """PCG with a python-level preconditioner (the AMG V-cycle); same stopping rule and snapshots as pcg_bj."""
    ip, ix, dt = A.indptr, A.indices, A.data
    n = b.shape[0]
    r = np.empty(n)
    Ap = np.empty(n)
    _resid(ip, ix, dt, x, b, r)
    bn = math.sqrt(_dot(b, b))
    rr = _dot(r, r)
    k, it, nt = 0, 0, len(tols)
    while k < nt and math.sqrt(rr) <= tols[k] * bn:
        snaps[k] = x
        iters[k] = 0
        k += 1
    if k == nt:
        return 0
    z = prec(r)
    p = z.copy()
    rz = _dot(r, z)
    while it < maxit:
        csr_mv(ip, ix, dt, p, Ap)
        alpha = rz / _dot(p, Ap)
        rr = _upd_xr(x, r, p, Ap, alpha)
        it += 1
        while k < nt and math.sqrt(rr) <= tols[k] * bn:
            snaps[k] = x
            iters[k] = it
            k += 1
        if k == nt:
            break
        z = prec(r)
        rzn = _dot(r, z)
        _xpby(z, rzn / rz, p)
        rz = rzn
    while k < nt:
        snaps[k] = x
        iters[k] = -1
        k += 1
    return it


@njit(parallel=True, cache=True)
def stress_kernel(u, nbr, valid, ndir, shell, opp, l0, Kn, Tv, V, sig):
    """lin2d.Lattice2D.stress for an intact lattice (alive = valid), fused."""
    N, NN = nbr.shape
    S = np.zeros((N, 2))
    TS = np.zeros((N, 2))
    dL = np.zeros((N, NN))
    for i in prange(N):
        for k in range(NN):
            if valid[i, k]:
                q = nbr[i, k]
                d = ndir[k, 0] * (u[2 * i] - u[2 * q]) + ndir[k, 1] * (u[2 * i + 1] - u[2 * q + 1])
                dL[i, k] = d
                c = shell[k]
                S[i, c] += d
                TS[i, c] += Tv[i, k] * d
    for i in prange(N):
        full = True
        for k in range(NN):
            if not valid[i, k]:
                full = False
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(NN):
            if valid[i, k]:
                q = nbr[i, k]
                c = shell[k]
                F = 2.0 * Kn[i, k] * dL[i, k] + 0.5 * (TS[i, c] + TS[q, c]) + 0.5 * Tv[i, k] * (S[i, c] + S[q, c])
                of = 0.5 if (full or valid[i, opp[k]]) else 1.0
                a = of / V * l0[k] * F
                sxx += a * ndir[k, 0] * ndir[k, 0]
                syy += a * ndir[k, 1] * ndir[k, 1]
                sxy += a * ndir[k, 0] * ndir[k, 1]
        sig[i, 0] = sxx
        sig[i, 1] = syy
        sig[i, 2] = sxy


def stress(lat, u, Kn, Tv):
    sig = np.empty((lat.N, 3))
    stress_kernel(np.ascontiguousarray(u.ravel()), lat.nbr, lat.valid, np.ascontiguousarray(lat.ndir),
                  lat.shell.astype(np.int64), lat.opp.astype(np.int64), np.ascontiguousarray(lat.l0),
                  np.ascontiguousarray(Kn, dtype=np.float64), np.ascontiguousarray(Tv, dtype=np.float64), float(lat.V),
                  sig)
    return sig


# ------------------------------------------------------------------------------------------------ matrix-free
@njit(parallel=True, cache=True)
def _kmv(u, nbr, valid, ndir, shell, opp, Kn, Tv, g, f):
    """f = K u (= grad of sum_i U_i), matrix-free; g (N, NN) scratch = dU_i / dL_ik."""
    N, NN = nbr.shape
    for i in prange(N):
        S0 = 0.0
        S1 = 0.0
        T0 = 0.0
        T1 = 0.0
        for k in range(NN):
            if valid[i, k]:
                q = nbr[i, k]
                d = ndir[k, 0] * (u[2 * i] - u[2 * q]) + ndir[k, 1] * (u[2 * i + 1] - u[2 * q + 1])
                g[i, k] = d
                if shell[k] == 0:
                    S0 += d
                    T0 += Tv[i, k] * d
                else:
                    S1 += d
                    T1 += Tv[i, k] * d
        for k in range(NN):
            if valid[i, k]:
                d = g[i, k]
                if shell[k] == 0:
                    g[i, k] = Kn[i, k] * d + 0.5 * (Tv[i, k] * S0 + T0)
                else:
                    g[i, k] = Kn[i, k] * d + 0.5 * (Tv[i, k] * S1 + T1)
    for j in prange(N):
        fx = 0.0
        fy = 0.0
        for k in range(NN):
            if valid[j, k]:
                t = g[j, k] + g[nbr[j, k], opp[k]]
                fx += t * ndir[k, 0]
                fy += t * ndir[k, 1]
        f[2 * j] = fx
        f[2 * j + 1] = fy


@njit(parallel=True, cache=True)
def _diag_inv(nbr, valid, ndir, shell, opp, Kn, Tv, fixed, Binv):
    """inverse 2x2 diagonal blocks of K (matrix-free); zero blocks for fixed nodes."""
    N, NN = nbr.shape
    for i in prange(N):
        if fixed[i]:
            Binv[i, 0, 0] = 0.0
            Binv[i, 0, 1] = 0.0
            Binv[i, 1, 0] = 0.0
            Binv[i, 1, 1] = 0.0
            continue
        a00 = 0.0
        a01 = 0.0
        a11 = 0.0
        m = np.zeros((2, 2))
        t = np.zeros((2, 2))
        for k in range(NN):
            if valid[i, k]:
                nx = ndir[k, 0]
                ny = ndir[k, 1]
                q = nbr[i, k]
                c = shell[k]
                w = Kn[i, k] + Kn[q, opp[k]] + Tv[q, opp[k]]
                a00 += w * nx * nx
                a01 += w * nx * ny
                a11 += w * ny * ny
                m[c, 0] += nx
                m[c, 1] += ny
                t[c, 0] += Tv[i, k] * nx
                t[c, 1] += Tv[i, k] * ny
        for c in range(2):
            a00 += t[c, 0] * m[c, 0]
            a01 += 0.5 * (t[c, 0] * m[c, 1] + m[c, 0] * t[c, 1])
            a11 += t[c, 1] * m[c, 1]
        det = a00 * a11 - a01 * a01
        Binv[i, 0, 0] = a11 / det
        Binv[i, 0, 1] = -a01 / det
        Binv[i, 1, 0] = -a01 / det
        Binv[i, 1, 1] = a00 / det


@njit(parallel=True, cache=True)
def pcg_mf(nbr, valid, ndir, shell, opp, Kn, Tv, Binv, mask, b, x, tols, maxit, snaps, iters, g):
    """matrix-free PCG on the free DOFs (full-length vectors, mask = 1 free / 0 fixed, x = 0 on fixed DOFs),
    2x2 block-Jacobi; snapshots as pcg_bj."""
    n = b.shape[0]
    N = n // 2
    nt = tols.shape[0]
    r = np.empty(n)
    z = np.empty(n)
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
    k = 0
    it = 0
    while k < nt and math.sqrt(rr) <= tols[k] * bn:
        snaps[k, :] = x
        iters[k] = 0
        k += 1
    if k == nt:
        return 0
    rz = 0.0
    for c in prange(N):
        a0 = r[2 * c]
        a1 = r[2 * c + 1]
        z0 = Binv[c, 0, 0] * a0 + Binv[c, 0, 1] * a1
        z1 = Binv[c, 1, 0] * a0 + Binv[c, 1, 1] * a1
        p[2 * c] = z0
        p[2 * c + 1] = z1
        rz += a0 * z0 + a1 * z1
    while it < maxit:
        _kmv(p, nbr, valid, ndir, shell, opp, Kn, Tv, g, Ap)
        pAp = 0.0
        for i in prange(n):
            Ap[i] *= mask[i]
            pAp += p[i] * Ap[i]
        alpha = rz / pAp
        rr = 0.0
        for i in prange(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            rr += r[i] * r[i]
        it += 1
        while k < nt and math.sqrt(rr) <= tols[k] * bn:
            snaps[k, :] = x
            iters[k] = it
            k += 1
        if k == nt:
            break
        rzn = 0.0
        for c in prange(N):
            a0 = r[2 * c]
            a1 = r[2 * c + 1]
            z0 = Binv[c, 0, 0] * a0 + Binv[c, 0, 1] * a1
            z1 = Binv[c, 1, 0] * a0 + Binv[c, 1, 1] * a1
            z[2 * c] = z0
            z[2 * c + 1] = z1
            rzn += a0 * z0 + a1 * z1
        beta = rzn / rz
        rz = rzn
        for i in prange(n):
            p[i] = z[i] + beta * p[i]
    while k < nt:
        snaps[k, :] = x
        iters[k] = -1
        k += 1
    return it


def kmv(lat, u, Kn, Tv):
    g = np.empty((lat.N, lat.NN))
    f = np.empty(2 * lat.N)
    _kmv(np.ascontiguousarray(u, dtype=np.float64), lat.nbr, lat.valid, np.ascontiguousarray(lat.ndir),
         lat.shell.astype(np.int64), lat.opp.astype(np.int64), Kn, Tv, g, f)
    return f


# ------------------------------------------------------------------------------------------------ systems
_SYS = {}


def get_sys(kind, n, st, sb=None):
    sb = st if sb is None else sb
    key = (kind, n, st, sb)
    if key not in _SYS:
        _SYS[key] = Sys(kind, n, st, sb)
    return _SYS[key]


class Sys:
    def __init__(self, kind, n, st, sb):
        t0 = time.perf_counter()
        self.kind, self.n, self.st, self.sb = kind, n, st, sb
        L = EP.bc_layers(kind)
        if st == 1:
            lat = lin2d.Lattice2D("square", n, n, 1.0)
            IJ, nc, s_ = lat.ij // sb, n // sb, sb
        else:                                               # coarse lattice; BVP of factor sb (multiple of st)
            assert sb % st == 0
            lat = EP.coarse_lattice(kind, n, st)
            IJ, nc, s_ = lat.ij // (sb // st), n // sb, sb
        self.lat, self.nl = lat, n // st                    # nl = sites per side of this lattice
        self.pat = elem.ElemPattern(lat)
        pat = self.pat
        nd = pat.nd
        Kpos = sp.csr_matrix((np.arange(1, pat.nnz + 1, dtype=np.float64), pat.indices, pat.indptr), shape=(nd, nd))
        self.sets = {}
        self.load = {}
        for bs, loads in EP.BCSETS.items():
            fixed, _, _ = EP.prescribed(loads[0], lat.x, IJ, nc, nc, s_, 1.0, L)
            fd = np.repeat(fixed, 2)
            fidx = np.nonzero(~fd)[0]
            sub = Kpos[fidx][:, fidx].tocsr()
            sub.sort_indices()
            d = dict(fidx=fidx, fixed=fixed, indptr=sub.indptr.astype(np.int64), indices=sub.indices.astype(np.int64),
                     gather=sub.data.astype(np.int64) - 1, loads=loads)
            self.sets[bs] = d
            for ld in loads:
                _, up, moving = EP.prescribed(ld, lat.x, IJ, nc, nc, s_, 1.0, L)
                ub = up.ravel().copy()
                ub[fd == 0] = 0.0
                self.load[ld] = dict(bs=bs, ub=ub, moving=moving)
        self.IJ, self.nc, self.s_, self.L = IJ, nc, s_, L
        self.t_init = time.perf_counter() - t0
        self._amg_X = lat.x

    # -------------------------------------------------------------------------------------------- starts
    def affine(self, ld):
        """load-consistent affine displacement on all sites (ring: the KUBC field; plates: linear between grips)."""
        x = self.lat.x
        ub = self.load[ld]["ub"].reshape(-1, 2)
        fixed = self.sets[self.load[ld]["bs"]]["fixed"]
        if ld.startswith("ring"):
            Lx = self.nc * self.s_
            e = {"ring_xx": [[1, 0], [0, 0]], "ring_yy": [[0, 0], [0, 1]], "ring_xy": [[0, .5], [.5, 0]]}[ld]
            return lin2d.affine_disp(x, EP.EPS * np.array(e), np.array([Lx / 2, Lx / 2])).ravel()
        ax = 1 if ld.startswith("tb") else 0           # grips are block rows / columns in lattice-index space
        ij = self.lat.ij[:, ax].astype(float)
        mov = self.load[ld]["moving"]
        a = fixed & ~mov
        xa, xb = ij[a].max(), ij[mov].min()
        t = np.clip((ij - xa) / (xb - xa), 0.0, 1.0)
        d = ub[mov][0]
        return (t[:, None] * d[None, :]).ravel()

    def interp_from(self, uc, sc):
        """bilinear (lattice-index space) interpolation of a solution on the lattice of factor sc onto this lattice."""
        key = ("ip", sc)
        if not hasattr(self, "_ip"):
            self._ip = {}
        if key not in self._ip:
            ij = self.lat.ij.astype(float)
            ncs = self.n // sc
            f = (ij * self.st + (self.st - 1) / 2.0 - (sc - 1) / 2.0) / sc
            i0 = np.clip(np.floor(f).astype(np.int64), 0, ncs - 2)
            t = f - i0
            idx = np.stack([i0[:, 1] * ncs + i0[:, 0], i0[:, 1] * ncs + i0[:, 0] + 1,
                            (i0[:, 1] + 1) * ncs + i0[:, 0], (i0[:, 1] + 1) * ncs + i0[:, 0] + 1], 1)
            w = np.stack([(1 - t[:, 0]) * (1 - t[:, 1]), t[:, 0] * (1 - t[:, 1]), (1 - t[:, 0]) * t[:, 1],
                          t[:, 0] * t[:, 1]], 1)
            self._ip[key] = (idx, w)
        idx, w = self._ip[key]
        U = uc.reshape(-1, 2)
        return (U[idx] * w[:, :, None]).sum(1).ravel()

    # -------------------------------------------------------------------------------------------- solves
    def assemble(self, Kn, Tv):
        return self.pat.assemble(Kn, Tv, self.lat.valid)

    def solve(self, K, starts, prec="bj", tols=TOLS, maxit=20000, loads=EP.LOADS, amg_opts=None, timing=None):
        """starts: {load: full start vector or None (zero)}.  Returns {load: (snaps (ntol, nd) full u, iters)}."""
        out = {}
        tm = dict(setup=0.0, rhs=0.0, iter=0.0)
        for bs, d in self.sets.items():
            lds = [ld for ld in d["loads"] if ld in loads]
            if not lds:
                continue
            ta = time.perf_counter()
            data = K.data[d["gather"]]
            if prec == "bj":
                B = bj_inverse(d["indptr"], d["indices"], data)
            else:
                from gmg_copy import GraphVCycle
                A = sp.csr_matrix((data, d["indices"], d["indptr"]), shape=(len(d["fidx"]),) * 2)
                o = dict(smoother="jacobi", nu=1, smooth_P=False, agg_size=3.0, coarse_max=1500)
                o.update(amg_opts or {})
                free = np.zeros(K.shape[0], bool)
                free[d["fidx"]] = True
                spacing = float(self.st)
                V = GraphVCycle(**o).setup(A, self._amg_X, 2, free, spacing, reduced=True)
            tb = time.perf_counter()
            tm["setup"] += tb - ta
            for ld in lds:
                tc = time.perf_counter()
                ub = self.load[ld]["ub"]
                b = -(K @ ub)[d["fidx"]]
                u0 = starts.get(ld)
                x = np.zeros(len(d["fidx"])) if u0 is None else np.ascontiguousarray(u0[d["fidx"]])
                td = time.perf_counter()
                snaps = np.empty((len(tols), len(x)))
                iters = np.zeros(len(tols), np.int64)
                if prec == "bj":
                    pcg_bj(d["indptr"], d["indices"], data, B, b, x, np.asarray(tols, float), maxit, snaps, iters)
                else:
                    pcg_prec(A, V.vcycle, b, x, np.asarray(tols, float), maxit, snaps, iters)
                te = time.perf_counter()
                tm["rhs"] += td - tc
                tm["iter"] += te - td
                U = np.repeat(ub[None, :], len(tols), 0)
                U[:, d["fidx"]] = snaps
                out[ld] = (U, iters)
        if timing is not None:
            timing.update(tm)
        return out

    def mf_arrays(self, Kn, Tv, fp32=False):
        lat = self.lat
        dt = np.float32 if fp32 else np.float64
        nbr = np.where(lat.valid, lat.nbr, 0).astype(np.int32 if fp32 else np.int64)
        return (nbr, lat.valid, np.ascontiguousarray(lat.ndir), lat.shell.astype(np.int64), lat.opp.astype(np.int64),
                np.ascontiguousarray(Kn, dtype=dt), np.ascontiguousarray(Tv, dtype=dt))

    def solve_mf(self, Kn, Tv, starts, tols=TOLS, maxit=20000, loads=EP.LOADS, fp32=False, timing=None):
        """matrix-free block-Jacobi PCG (no assembly).  Same interface as solve()."""
        arr = self.mf_arrays(Kn, Tv, fp32)
        N = self.lat.N
        g = np.empty((N, self.lat.NN))
        out = {}
        tm = dict(setup=0.0, rhs=0.0, iter=0.0)
        tols = np.asarray(tols, float)
        for bs, d in self.sets.items():
            lds = [ld for ld in d["loads"] if ld in loads]
            if not lds:
                continue
            ta = time.perf_counter()
            Binv = np.empty((N, 2, 2))
            _diag_inv(*arr, d["fixed"], Binv)
            mask = np.repeat((~d["fixed"]).astype(np.float64), 2)
            tm["setup"] += time.perf_counter() - ta
            for ld in lds:
                tc = time.perf_counter()
                ub = self.load[ld]["ub"]
                f = np.empty(2 * N)
                _kmv(ub, *arr, g, f)
                b = -f * mask
                u0 = starts.get(ld)
                x = np.zeros(2 * N) if u0 is None else (u0 - ub) * mask
                td = time.perf_counter()
                snaps = np.empty((len(tols), 2 * N))
                iters = np.zeros(len(tols), np.int64)
                pcg_mf(*arr, Binv, mask, b, x, tols, maxit, snaps, iters, g)
                snaps += ub[None, :]
                te = time.perf_counter()
                tm["rhs"] += td - tc
                tm["iter"] += te - td
                out[ld] = (snaps, iters)
        if timing is not None:
            timing.update(tm)
        return out

    def solve_direct(self, K, loads=EP.LOADS):
        """SuperLU in natural order (only for checks; the timed direct solves use el_core.Geometry)."""
        out = {}
        for bs, d in self.sets.items():
            A = sp.csc_matrix(sp.csr_matrix((K.data[d["gather"]], d["indices"], d["indptr"]),
                                            shape=(len(d["fidx"]),) * 2))
            lu = spla.splu(A)
            for ld in d["loads"]:
                if ld not in loads:
                    continue
                ub = self.load[ld]["ub"]
                u = ub.copy()
                u[d["fidx"]] = lu.solve(-(K @ ub)[d["fidx"]])
                out[ld] = u
        return out

    # -------------------------------------------------------------------------------------------- outputs
    def block_sig(self, sig, R):
        """mean particle stress over R x R fine-site blocks (this lattice: blocks of (R/st)^2 sites)."""
        k = R // self.st
        nb = self.n // R
        I = self.lat.ij[:, 0] // k
        J = self.lat.ij[:, 1] // k
        b = J * nb + I
        cnt = np.bincount(b, minlength=nb * nb)
        return np.stack([np.bincount(b, sig[:, c], minlength=nb * nb) / cnt for c in range(3)], 1)
