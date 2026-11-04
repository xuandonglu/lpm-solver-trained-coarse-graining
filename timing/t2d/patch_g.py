"""Classical local patch homogenisation with precomputed geometry for the square lattice, any s, memory-bounded
(windows processed in chunks of identical geometry), dense or sparse window solves.

Mathematics = cg.patch_cells:
  every coarse cell gets a window = cell + s sites on each side; the domain is reflect-padded; the outer 2 site layers
  get the affine displacement of a unit strain (xx, yy, engineering xy), the interior relaxes (fine LPM);
  C_ab = (1/A_cell) u_a^T K_cell u_b with K_cell the Hessian of the energy of the cell's own particles.
What depends only on the geometry is built once: one CHUNK lattice holding `wpc` windows (all windows are identical),
its element pattern, the free DOFs of every window, the gather indices of the free-free blocks (dense) or of the
nested-dissection-ordered block-diagonal free-free matrix (sparse), the affine boundary vectors, and for every chunk the
source site of every chunk site.  Per microstructure and chunk: gather parameters -> numba assembly -> 3 solves ->
cell energies.  Dense backend = batched LAPACK solves of the d x d window blocks; sparse backend = one SuperLU
factorisation of the block-diagonal chunk matrix in ND order (cheaper for large windows, d = 2(3s-4)^2: 800 at s = 8).
"""
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import lin2d
import elem
import ndorder as ND


class PatchFastG:
    def __init__(self, kind, n, s, chunk_sites=150000, backend=None, h=1.0):
        t0 = time.perf_counter()
        self.kind, self.n, self.s = kind, n, s
        assert kind == "comp", kind
        lk = "square"
        m, ring, gap = s, 2, 2
        w = s + 2 * m
        W = w + gap
        nc = n // s
        self.nc, self.nw, self.m, self.w = nc, nc * nc, m, w
        wpc = int(max(1, min(self.nw, chunk_sites // (w * w))))
        ncol = int(math.ceil(math.sqrt(wpc)))
        nrow = int(math.ceil(wpc / ncol))
        wpc = ncol * nrow
        self.wpc = wpc
        gx, gy = ncol * W, nrow * W
        act = np.zeros((gy, gx), bool)
        for r in range(nrow):
            for c in range(ncol):
                act[r * W:r * W + w, c * W:c * W + w] = True
        lat = lin2d.Lattice2D(lk, gx, gy, h, active=act)
        self.lat = lat
        li, lj = lat.ij[:, 0] % W, lat.ij[:, 1] % W
        wloc = (lat.ij[:, 1] // W) * ncol + lat.ij[:, 0] // W          # window id inside the chunk
        self.wloc = wloc
        self.pat = elem.ElemPattern(lat)
        outer = (li < ring) | (li >= w - ring) | (lj < ring) | (lj >= w - ring)
        incell = (li >= m) & (li < m + s) & (lj >= m) & (lj < m + s)
        self.cell_parts = np.nonzero(incell)[0]
        xl = np.stack([li - (m + s / 2 - 0.5), lj - (m + s / 2 - 0.5)], 1) * h
        self.A = (s * h) ** 2
        fixed = np.nonzero(outer)[0]
        fdofs = np.concatenate([2 * fixed, 2 * fixed + 1])
        freep = np.nonzero(~outer)[0]
        order = np.lexsort((lat.ij[freep, 0], lat.ij[freep, 1], wloc[freep]))
        freep = freep[order]
        npw = len(freep) // wpc
        assert npw * wpc == len(freep)
        self.d = 2 * npw
        fd = (2 * freep[:, None] + np.arange(2)[None, :]).reshape(wpc, self.d)
        self.fd = fd
        nd = self.pat.nd
        if backend is None:
            backend = "dense" if self.d <= 200 else "sparse"
        self.backend = backend
        indptr, indices = self.pat.indptr, self.pat.indices
        if backend == "dense":
            # CSR entries that fall inside a window's free-free block -> (data index, window, row, col) in the block
            # (for s > 2 not every pair of free DOFs of a window is coupled, so the block is scattered, not gathered)
            win_of = -np.ones(nd, np.int64)
            loc_of = -np.ones(nd, np.int64)
            win_of[fd.ravel()] = np.repeat(np.arange(wpc), self.d)
            loc_of[fd.ravel()] = np.tile(np.arange(self.d), wpc)
            rowid = np.repeat(np.arange(nd, dtype=np.int64), np.diff(indptr))
            col = indices.astype(np.int64)
            ok = (win_of[rowid] >= 0) & (win_of[rowid] == win_of[col])
            self.dn_idx = np.nonzero(ok)[0]
            self.dn_w = win_of[rowid[ok]]
            self.dn_r = loc_of[rowid[ok]]
            self.dn_c = loc_of[col[ok]]
        else:
            # ND order of one window's free nodes (all windows identical), applied to every window
            Kpos = sp.csr_matrix((np.arange(1, self.pat.nnz + 1, dtype=np.float64), indices, indptr), shape=(nd, nd))
            f0 = fd[0]
            G = ND.node_graph(Kpos[f0][:, f0], len(f0) // 2)
            nodes0 = f0[0::2] // 2
            nperm = ND.nd_order(lat.x[nodes0], G, leaf=16)
            lperm = ND.dof_perm(nperm)
            self.fdall = fd[:, lperm].ravel()
            sub = Kpos[self.fdall][:, self.fdall].tocsc()
            sub.sort_indices()
            self.sp_gather = sub.data.astype(np.int64) - 1
            self.sp_indptr, self.sp_indices = sub.indptr.copy(), sub.indices.copy()
        E = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, .5], [.5, 0]])]
        UB = np.zeros((2 * lat.N, 3))
        for k, e in enumerate(E):
            ua = lin2d.affine_disp(xl, e)
            UB[fdofs, k] = np.concatenate([ua[fixed, 0], ua[fixed, 1]])
        self.UB = UB
        self.dwin = np.repeat(wloc, 2)
        # source site of every chunk site, per chunk (reflect padding as np.pad(..., mode="reflect"))
        idx = np.arange(n * n).reshape(n, n)
        ipad = np.pad(idx, ((m, m), (m, m)), mode="reflect")
        self.nchunk = int(math.ceil(self.nw / wpc))
        src = np.zeros((self.nchunk, lat.N), np.int64)
        for c in range(self.nchunk):
            g = c * wpc + wloc                               # global window id of every chunk site
            g = np.where(g < self.nw, g, 0)                  # padding windows of the last chunk: window 0 (ignored)
            J, I = g // nc, g % nc
            src[c] = ipad[J * s + lj, I * s + li]
        self.src = src
        self.t_init = time.perf_counter() - t0

    # --------------------------------------------------------------------------------------------------------
    def _params(self, prob, c):
        lat = self.lat
        sidx = self.src[c]
        return lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(prob["Cpix"][sidx]))

    def solve(self, prob, timing=None):
        """C6 per cell (nw, 6): (C11, C22, C12, C66, C16, C26) as cg.patch_cells."""
        tp = ta = ts = te = 0.0
        out = np.zeros((self.nw, 6))
        lat = self.lat
        for c in range(self.nchunk):
            t0 = time.perf_counter()
            Kn, Tv = self._params(prob, c)
            t1 = time.perf_counter()
            K = self.pat.assemble(Kn, Tv, lat.valid)
            t2 = time.perf_counter()
            R = -(K @ self.UB)                               # (2N, 3)
            U = self.UB.copy()
            if self.backend == "dense":
                blocks = np.zeros((self.wpc, self.d, self.d))
                blocks[self.dn_w, self.dn_r, self.dn_c] = K.data[self.dn_idx]
                X = np.linalg.solve(blocks, R[self.fd])      # (wpc, d, 3)
                U[self.fd.ravel()] = X.reshape(-1, 3)
            else:
                A = sp.csc_matrix((K.data[self.sp_gather], self.sp_indices, self.sp_indptr),
                                  shape=(len(self.fdall),) * 2)
                lu = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
                U[self.fdall] = lu.solve(np.ascontiguousarray(R[self.fdall]))
                del lu
            t3 = time.perf_counter()
            Hc = self.pat.assemble(Kn, Tv, lat.valid, owners=self.cell_parts)
            HU = Hc @ U
            Cm = np.zeros((self.wpc, 3, 3))
            for a in range(3):
                for b in range(a, 3):
                    v = np.bincount(self.dwin, HU[:, a] * U[:, b], minlength=self.wpc) / self.A
                    Cm[:, a, b] = v
                    Cm[:, b, a] = v
            g0 = c * self.wpc
            g1 = min(self.nw, g0 + self.wpc)
            out[g0:g1] = np.stack([Cm[:, 0, 0], Cm[:, 1, 1], Cm[:, 0, 1], Cm[:, 2, 2], Cm[:, 0, 2], Cm[:, 1, 2]],
                                  1)[:g1 - g0]
            t4 = time.perf_counter()
            tp += t1 - t0
            ta += t2 - t1
            ts += t3 - t2
            te += t4 - t3
        if timing is not None:
            timing.update(params=tp, assemble=ta, solve=ts, energy=te, total=tp + ta + ts + te)
        return out


_PF = {}


def patch_obj(kind, n, s, backend=None):
    key = (kind, n, s, backend)
    if key not in _PF:
        _PF[key] = PatchFastG(kind, n, s, backend=backend)
    return _PF[key]
