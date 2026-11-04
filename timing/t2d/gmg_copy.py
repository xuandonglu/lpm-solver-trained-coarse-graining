"""Graph V-cycle preconditioner for the LPM stiffness system (aggregation-based algebraic multigrid on the particle
graph):
  * level-0 graph   = LPM particles, edge (j -> i) carries the dim x dim stiffness block K_ij (bond + 2-hop terms),
  * mat-vec         = K u (csr_mv), smoothing = damped block-Jacobi or Chebyshev on the block-Jacobi operator,
  * coarsening      = geometric cell aggregation of particles, split by stiffness connectivity,
                      rigid-body-mode tentative prolongator, optionally smoothed (SA-AMG),
  * coarse operator = Galerkin P^T K P, dense Cholesky on the coarsest level.
It is used only as a preconditioner inside CG, so the converged solution satisfies the same residual test as the
plain LPM CG; the hierarchy can only change the cost, not the accuracy.
"""
import math
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from numba import njit, prange


# ----------------------------------------------------------------------------------------------- kernels
@njit(cache=True, parallel=True)
def csr_mv(indptr, indices, data, x, y):
    n = indptr.shape[0] - 1
    for i in prange(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        y[i] = s


@njit(cache=True, parallel=True)
def csr_resid(indptr, indices, data, x, b, r):
    n = indptr.shape[0] - 1
    for i in prange(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        r[i] = b[i] - s


@njit(cache=True, parallel=True)
def block_apply(binv, bptr, x, y, scale):
    """y = scale * blockdiag(binv) x ; blocks of variable size given by bptr (block start offsets)."""
    nb = bptr.shape[0] - 1
    for k in prange(nb):
        s0 = bptr[k]
        m = bptr[k + 1] - s0
        for a in range(m):
            acc = 0.0
            for b in range(m):
                acc += binv[k, a, b] * x[s0 + b]
            y[s0 + a] = scale * acc


@njit(cache=True, parallel=True)
def axpy(a, x, y):
    for i in prange(x.shape[0]):
        y[i] += a * x[i]


@njit(cache=True, parallel=True)
def dot(x, y):
    s = 0.0
    for i in prange(x.shape[0]):
        s += x[i] * y[i]
    return s


@njit(cache=True)
def _tentative(agg_of_row, nagg, B, m, tol):
    """Per-aggregate QR of the near-null-space B (rows grouped by aggregate). Returns COO of P0 and coarse B."""
    n = B.shape[0]
    counts = np.zeros(nagg, np.int64)
    for i in range(n):
        counts[agg_of_row[i]] += 1
    start = np.zeros(nagg + 1, np.int64)
    for a in range(nagg):
        start[a + 1] = start[a] + counts[a]
    order = np.empty(n, np.int64)
    fill = start[:-1].copy()
    for i in range(n):
        a = agg_of_row[i]
        order[fill[a]] = i
        fill[a] += 1
    rows = np.empty(n * m, np.int64)
    cols = np.empty(n * m, np.int64)
    vals = np.empty(n * m)
    Bc = np.zeros((nagg * m, m))
    colmap = np.full(nagg * m, -1, np.int64)
    nnz = 0
    ncol = 0
    for a in range(nagg):
        r0, r1 = start[a], start[a + 1]
        k = r1 - r0
        if k == 0:
            continue
        blk = np.empty((k, m))
        for q in range(k):
            for c in range(m):
                blk[q, c] = B[order[r0 + q], c]
        # modified Gram-Schmidt with rank detection
        Q = np.zeros((k, m))
        R = np.zeros((m, m))
        keep = np.zeros(m, np.bool_)
        scale = 0.0
        for c in range(m):
            nrm = 0.0
            for q in range(k):
                nrm += blk[q, c] ** 2
            scale = max(scale, math.sqrt(nrm))
        for c in range(m):
            v = blk[:, c].copy()
            for d in range(c):
                if keep[d]:
                    pr = 0.0
                    for q in range(k):
                        pr += Q[q, d] * v[q]
                    R[d, c] = pr
                    for q in range(k):
                        v[q] -= pr * Q[q, d]
            nrm = 0.0
            for q in range(k):
                nrm += v[q] * v[q]
            nrm = math.sqrt(nrm)
            if nrm > tol * scale and nrm > 0.0:
                keep[c] = True
                R[c, c] = nrm
                for q in range(k):
                    Q[q, c] = v[q] / nrm
        for c in range(m):
            if not keep[c]:
                continue
            colmap[a * m + c] = ncol
            for q in range(k):
                rows[nnz] = order[r0 + q]
                cols[nnz] = ncol
                vals[nnz] = Q[q, c]
                nnz += 1
            for c2 in range(m):
                Bc[ncol, c2] = R[c, c2]
            ncol += 1
    return rows[:nnz], cols[:nnz], vals[:nnz], Bc[:ncol], ncol, colmap


def _rbm(X, dim):
    """Rigid body modes of point set X (n x 3) for dim 2 (3 modes) or 3 (6 modes), rows interleaved by dof."""
    n = X.shape[0]
    c = X[:, :dim] - X[:, :dim].mean(0)
    if dim == 2:
        B = np.zeros((2 * n, 3))
        B[0::2, 0] = 1
        B[1::2, 1] = 1
        B[0::2, 2] = -c[:, 1]
        B[1::2, 2] = c[:, 0]
    else:
        B = np.zeros((3 * n, 6))
        for d in range(3):
            B[d::3, d] = 1
        B[0::3, 3] = -c[:, 1]; B[1::3, 3] = c[:, 0]
        B[1::3, 4] = -c[:, 2]; B[2::3, 4] = c[:, 1]
        B[0::3, 5] = c[:, 2];  B[2::3, 5] = -c[:, 0]
    return B


def _cells(Xn, size):
    lo = Xn.min(0)
    ijk = np.floor((Xn - lo) / size + 1e-9).astype(np.int64)
    key = ijk[:, 0] * 1000003 + ijk[:, 1]
    if ijk.shape[1] > 2:
        key = key * 1000003 + ijk[:, 2]
    _, agg = np.unique(key, return_inverse=True)
    return agg.astype(np.int64)


def _split_by_connectivity(agg, A_node):
    """Split each aggregate into connected components of the node graph restricted to the aggregate."""
    G = A_node.tocoo()
    same = (agg[G.row] == agg[G.col]) & (G.row != G.col)
    n = agg.shape[0]
    H = sp.csr_matrix((np.ones(same.sum()), (G.row[same], G.col[same])), shape=(n, n))
    ncomp, lab = sp.csgraph.connected_components(H, directed=False)
    return lab.astype(np.int64), ncomp


def _node_graph(A, bptr):
    """Node-level sparsity pattern of a block matrix (blocks given by bptr)."""
    nb = bptr.shape[0] - 1
    node_of = np.repeat(np.arange(nb), np.diff(bptr))
    C = A.tocoo()
    keep = C.data != 0
    return sp.csr_matrix((np.ones(keep.sum()), (node_of[C.row[keep]], node_of[C.col[keep]])), shape=(nb, nb))


def _block_inverses(A, bptr, mmax):
    nb = bptr.shape[0] - 1
    binv = np.zeros((nb, mmax, mmax))
    Ad = A.tocsr()
    for k in range(nb):
        s0, s1 = bptr[k], bptr[k + 1]
        blk = Ad[s0:s1, s0:s1].toarray()
        try:
            binv[k, :s1 - s0, :s1 - s0] = np.linalg.inv(blk)
        except np.linalg.LinAlgError:
            binv[k, :s1 - s0, :s1 - s0] = np.linalg.pinv(blk)
    return binv


@njit(cache=True)
def _block_inverses_fast(indptr, indices, data, bptr, mmax):
    nb = bptr.shape[0] - 1
    binv = np.zeros((nb, mmax, mmax))
    for k in range(nb):
        s0, s1 = bptr[k], bptr[k + 1]
        m = s1 - s0
        blk = np.zeros((m, m))
        for a in range(m):
            i = s0 + a
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if s0 <= j < s1:
                    blk[a, j - s0] = data[p]
        binv[k, :m, :m] = np.linalg.inv(blk)
    return binv


def _power_rho(A, binv, bptr, iters=15, seed=0):
    n = A.shape[0]
    x = np.random.default_rng(seed).standard_normal(n)
    y = np.empty(n)
    z = np.empty(n)
    lam = 1.0
    for _ in range(iters):
        x /= np.linalg.norm(x)
        csr_mv(A.indptr, A.indices, A.data, x, y)
        block_apply(binv, bptr, y, z, 1.0)
        lam = float(np.linalg.norm(z))
        x = z.copy()
    return lam


class Level:
    pass


class GraphVCycle:
    """Hierarchy + V-cycle; call setup() once per stiffness matrix, then use as a CG preconditioner."""

    def __init__(self, smoother="jacobi", nu=2, cheb_deg=3, cheb_lo=1.0 / 30, smooth_P=True, agg_size=2.0,
                 coarse_max=3000, max_levels=10, omega=None, weights=None, split=True):
        self.smoother, self.nu, self.cheb_deg, self.cheb_lo = smoother, nu, cheb_deg, cheb_lo
        self.smooth_P, self.agg_size, self.coarse_max, self.max_levels = smooth_P, agg_size, coarse_max, max_levels
        self.omega, self.weights, self.split = omega, weights, split
        self.timing = {}

    # -------------------------------------------------------------------------------------- setup
    def setup(self, A_full, X, dim, free, spacing, reduced=False):
        t0 = time.perf_counter()
        free = free.astype(bool)
        self.free_idx = np.nonzero(free)[0]
        if reduced:
            A = A_full
        else:
            A = A_full.tocsr()[self.free_idx][:, self.free_idx].tocsr()
            A.sort_indices()
        N = X.shape[0]
        node_of_dof = np.repeat(np.arange(N), dim)[self.free_idx]
        bptr = np.concatenate([[0], np.cumsum(np.bincount(node_of_dof, minlength=N))]).astype(np.int64)
        keep_nodes = np.diff(bptr) > 0
        bptr = np.concatenate([[0], np.cumsum(np.diff(bptr)[keep_nodes])]).astype(np.int64)
        Xn = X[keep_nodes, :dim]
        node_ids = np.nonzero(keep_nodes)[0]
        # near null space on free rows
        B_all = _rbm(X, dim)
        B = B_all[self.free_idx]
        m = B.shape[1]
        # floating rigid modes left free by the BCs (K z = 0): PCG keeps its iterates orthogonal to them, like
        # unpreconditioned CG from x0 = 0 does implicitly
        Q, _ = np.linalg.qr(B)
        KQ = A @ Q
        knorm = float(np.sqrt(np.mean(A.diagonal() ** 2)))
        null = np.linalg.norm(KQ, axis=0) < 1e-5 * knorm
        self.Z = None
        if null.any():
            U_, S_, _ = np.linalg.svd(Q[:, null], full_matrices=False)
            self.Z = np.ascontiguousarray(U_)
        self.levels = []
        size = spacing * self.agg_size
        for lvl in range(self.max_levels):
            L = Level()
            L.A = A
            L.bptr = bptr
            L.n = A.shape[0]
            L.mmax = int(np.max(np.diff(bptr)))
            L.binv = _block_inverses_fast(A.indptr, A.indices, A.data, bptr, L.mmax)
            L.rho = _power_rho(A, L.binv, bptr)
            L.r = np.empty(L.n); L.z = np.empty(L.n); L.t = np.empty(L.n); L.x = np.empty(L.n)
            self.levels.append(L)
            if L.n <= self.coarse_max or lvl == self.max_levels - 1:
                Ad = A.toarray()
                # rigid modes left free by the BCs (e.g. the default cube) make the coarse operator singular:
                # a tiny diagonal shift keeps the coarse solve SPD (it is only a preconditioner)
                shift, md = 0.0, float(np.mean(np.diag(Ad)))
                while True:
                    try:
                        L.coarse = sla.cho_factor(Ad + shift * md * np.eye(Ad.shape[0]), lower=False)
                        break
                    except np.linalg.LinAlgError:
                        shift = 1e-10 if shift == 0.0 else shift * 10
                L.coarse_shift = shift
                break
            nb = bptr.shape[0] - 1
            agg = _cells(Xn, size)
            if self.split:
                agg, nagg = _split_by_connectivity(agg, _node_graph(A, bptr))
            else:
                nagg = int(agg.max()) + 1
            row_agg = np.repeat(agg, np.diff(bptr))
            r_, c_, v_, Bc, ncol, colmap = _tentative(row_agg, nagg, np.ascontiguousarray(B), m, 1e-8)
            P = sp.csr_matrix((v_, (r_, c_)), shape=(L.n, ncol))
            if self.smooth_P:
                Dinv = sp.bsr_matrix((L.binv, np.arange(nb), np.arange(nb + 1)), shape=(L.n, L.n)) \
                    if np.all(np.diff(bptr) == L.mmax) else self._blockdiag(L)
                w = 4.0 / 3.0 / L.rho
                P = (P - w * (Dinv @ (A @ P))).tocsr()
            P.sort_indices()
            L.P = P
            L.PT = P.T.tocsr()
            A = (L.PT @ (A @ P)).tocsr()
            A = (A + A.T) * 0.5
            A = A.tocsr()
            A.sort_indices()
            # coarse nodes: aggregates with at least one kept column
            col_node = np.full(ncol, -1, np.int64)
            for a in range(nagg):
                for c in range(m):
                    k = colmap[a * m + c]
                    if k >= 0:
                        col_node[k] = a
            used = np.unique(col_node)
            cnt = np.bincount(col_node, minlength=nagg)[used]
            bptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
            cen = np.zeros((nagg, dim))
            np.add.at(cen, agg, Xn)
            cen /= np.maximum(np.bincount(agg, minlength=nagg), 1)[:, None]
            Xn = cen[used]
            B = Bc
            size *= self.agg_size
        self.timing["setup"] = time.perf_counter() - t0
        return self

    @staticmethod
    def _blockdiag(L):
        nb = L.bptr.shape[0] - 1
        rows, cols, vals = [], [], []
        for k in range(nb):
            s0, s1 = L.bptr[k], L.bptr[k + 1]
            mm = s1 - s0
            ii, jj = np.meshgrid(np.arange(s0, s1), np.arange(s0, s1), indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(L.binv[k, :mm, :mm].ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(L.n, L.n))

    # -------------------------------------------------------------------------------------- cycle
    def _smooth(self, L, lvl, x, b, post):
        A = L.A
        if self.smoother == "jacobi":
            w = self.omega if self.omega is not None else 2.0 / L.rho * 0.66
            if self.weights is not None:
                w = self.weights[min(lvl, len(self.weights) - 1)][0] / L.rho
            for _ in range(self.nu):
                csr_resid(A.indptr, A.indices, A.data, x, b, L.r)
                block_apply(L.binv, L.bptr, L.r, L.z, w)
                axpy(1.0, L.z, x)
        else:   # Chebyshev on D^-1 A with bounds [lo*rho, 1.1 rho] (or given per-level coefficients)
            hi = 1.1 * L.rho
            lo = self.cheb_lo * hi
            coef = None
            if self.weights is not None:
                coef = self.weights[min(lvl, len(self.weights) - 1)]
            for _ in range(self.nu):
                csr_resid(A.indptr, A.indices, A.data, x, b, L.r)
                if coef is not None:
                    # x += sum_k c_k (D^-1 A)^k D^-1 r  (polynomial with given coefficients, scaled by 1/rho)
                    block_apply(L.binv, L.bptr, L.r, L.z, 1.0)
                    acc = coef[0] / L.rho * L.z
                    v = L.z.copy()
                    for k in range(1, len(coef)):
                        csr_mv(A.indptr, A.indices, A.data, v, L.t)
                        block_apply(L.binv, L.bptr, L.t, v, 1.0 / L.rho)
                        acc += coef[k] / L.rho * v
                    axpy(1.0, acc, x)
                else:
                    self._cheb(L, x, b, lo, hi)

    def _cheb(self, L, x, b, lo, hi):
        A = L.A
        theta = 0.5 * (hi + lo)
        delta = 0.5 * (hi - lo)
        sigma = theta / delta
        rho_k = 1.0 / sigma
        csr_resid(A.indptr, A.indices, A.data, x, b, L.r)
        block_apply(L.binv, L.bptr, L.r, L.z, 1.0 / theta)
        d = L.z.copy()
        for _ in range(self.cheb_deg - 1):
            axpy(1.0, d, x)
            csr_resid(A.indptr, A.indices, A.data, x, b, L.r)
            block_apply(L.binv, L.bptr, L.r, L.z, 1.0)
            rho_new = 1.0 / (2.0 * sigma - rho_k)
            d = rho_new * rho_k * d + 2.0 * rho_new / delta * L.z
            rho_k = rho_new
        axpy(1.0, d, x)

    def vcycle(self, b, lvl=0):
        L = self.levels[lvl]
        if hasattr(L, "coarse"):
            return sla.cho_solve(L.coarse, b)
        x = np.zeros(L.n)
        self._smooth(L, lvl, x, b, False)
        csr_resid(L.A.indptr, L.A.indices, L.A.data, x, b, L.r)
        rc = np.empty(L.PT.shape[0])
        csr_mv(L.PT.indptr, L.PT.indices, L.PT.data, L.r, rc)
        ec = self.vcycle(rc, lvl + 1)
        e = np.empty(L.n)
        csr_mv(L.P.indptr, L.P.indices, L.P.data, ec, e)
        axpy(1.0, e, x)
        self._smooth(L, lvl, x, b, True)
        return x

    # -------------------------------------------------------------------------------------- PCG on free dofs
    def pcg(self, b_full, rel, absol, maxit=10000, x0_full=None):
        """Same stopping rule as LPM-C CG: ||r||^2 <= rel*||b||^2 + abs (unpreconditioned residual, x0 = 0)."""
        A = self.levels[0].A
        b = np.ascontiguousarray(b_full[self.free_idx])
        n = b.shape[0]
        x = np.zeros(n)
        r = b.copy()
        rr = dot(r, r)
        tol = rel * float(np.dot(b_full, b_full)) + absol
        it = 0
        if rr <= tol:
            out = np.zeros_like(b_full)
            return out, 0
        z = self.vcycle(r)
        if self.Z is not None:
            z -= self.Z @ (self.Z.T @ z)
        p = z.copy()
        rz = dot(r, z)
        ap = np.empty(n)
        while it < maxit:
            csr_mv(A.indptr, A.indices, A.data, p, ap)
            alpha = rz / dot(p, ap)
            axpy(alpha, p, x)
            axpy(-alpha, ap, r)
            it += 1
            rr = dot(r, r)
            if rr <= tol:
                break
            z = self.vcycle(r)
            if self.Z is not None:
                z -= self.Z @ (self.Z.T @ z)
            rz_new = dot(r, z)
            beta = rz_new / rz
            rz = rz_new
            p *= beta
            axpy(1.0, z, p)
        out = np.zeros_like(b_full)
        out[self.free_idx] = x
        return out, it

    def set_fine_data(self, data):
        """Lagged preconditioner: replace the level-0 operator values (same pattern), keep smoother data and the
        coarse hierarchy from the last setup. PCG then iterates on the current matrix."""
        self.levels[0].A.data = data

    def describe(self):
        return " | ".join("L%d n=%d nnz=%d" % (k, L.n, L.A.nnz) for k, L in enumerate(self.levels))
