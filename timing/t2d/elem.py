"""Particle-wise ("element") assembly of the linearised LPM stiffness on a lin2d square lattice.

Energy of particle i (lin2d): U_i = 1/2 sum_k Kn_ik dL_ik^2 + 1/2 sum_c (sum_{k in c} Tv_ik dL_ik)(sum_{k in c} dL_ik)
depends only on the displacements of i and its 8 neighbours -> "element" i with 9 nodes, element matrix
K^(i) = B_i^T H_i B_i (18 x 18).  K = sum_i K^(i) is exactly lin2d.Lattice2D.stiffness (checked in check_elem.py).
Row k of K = force on particle k = sum over the elements i in {k} U nbrs(k); their nodes reach 2 lattice steps, which is
the multibody reach of LPM (F_ij needs S_j, i.e. the full neighbourhood of j).

The sparsity pattern is that of the intact lattice (fixed); broken bonds just give zeros, so every assembly is a numba
loop that writes into a preallocated CSR data array (about 10 ms for 10^4 particles).
"""
import numpy as np
import numba as nb
import scipy.sparse as sp


class ElemPattern:
    """Fixed CSR pattern of the assembled stiffness (memory-light version: the DOF pattern is
    built from the node pattern E^T E, and every element entry is addressed by indptr[2a + d] + 2 rank(a, b) + e,
    rank = position of node b among the sorted node neighbours of a)."""

    def __init__(self, lat):
        N, NN = lat.N, lat.NN
        self.NN = NN
        self.lat = lat
        nodes = np.full((N, NN + 1), -1, np.int64)
        nodes[:, 0] = np.arange(N)
        nodes[:, 1:] = np.where(lat.valid, lat.nbr, -1)
        self.nodes = nodes
        ok = nodes >= 0
        E = sp.csr_matrix((np.ones(ok.sum(), np.int8), (np.repeat(np.arange(N), NN + 1)[ok.ravel()], nodes[ok])),
                          shape=(N, N))
        Gn = (E.T @ E).tocsr()
        Gn.sort_indices()
        deg = np.diff(Gn.indptr)
        nd = 2 * N
        indptr = np.zeros(nd + 1, np.int64)
        indptr[1::2] = 2 * deg
        indptr[2::2] = 2 * deg
        indptr = np.cumsum(indptr)
        nnz = int(indptr[-1])
        indices = np.empty(nnz, np.int32)
        _fill_indices(Gn.indptr.astype(np.int64), Gn.indices.astype(np.int64), indptr, indices)
        rank = np.full((N, NN + 1, NN + 1), -1, np.int32)
        _fill_rank(nodes, Gn.indptr.astype(np.int64), Gn.indices.astype(np.int64), rank)
        self.rank = rank
        self.indices = indices
        self.indptr = indptr
        self.nnz = nnz
        self.nd = nd
        self.ndir = np.ascontiguousarray(lat.ndir)
        self.shell = lat.shell.astype(np.int64)

    def assemble(self, Kn, Tv, alive, owners=None):
        """CSR stiffness (2N x 2N) with the bonds `alive` (N, 8)."""
        data = np.zeros(self.nnz)
        if owners is None:
            owners = np.arange(self.lat.N)
        _assemble(np.ascontiguousarray(owners, dtype=np.int64), self.ndir, self.shell,
                  np.ascontiguousarray(Kn, dtype=np.float64), np.ascontiguousarray(Tv, dtype=np.float64),
                  np.ascontiguousarray(alive & self.lat.valid), self.nodes, self.rank, self.indptr, data)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.nd, self.nd))


@nb.njit(cache=True)
def _fill_indices(gp, gi, indptr, indices):
    N = gp.shape[0] - 1
    for a in range(N):
        for d in range(2):
            t = indptr[2 * a + d]
            for j in range(gp[a], gp[a + 1]):
                b = gi[j]
                indices[t] = 2 * b
                indices[t + 1] = 2 * b + 1
                t += 2


@nb.njit(cache=True)
def _fill_rank(nodes, gp, gi, rank):
    N = nodes.shape[0]
    M = nodes.shape[1]
    for i in range(N):
        for al in range(M):
            a = nodes[i, al]
            if a < 0:
                continue
            for be in range(M):
                b = nodes[i, be]
                if b < 0:
                    continue
                lo, hi = gp[a], gp[a + 1]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if gi[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                rank[i, al, be] = lo - gp[a]


@nb.njit(cache=True)
def _assemble(owners, ndir, shell, Kn, Tv, alive, nodes, rank, indptr, data):
    NN = ndir.shape[0]
    for t in range(owners.shape[0]):
        i = owners[t]
        for a in range(NN):
            if not alive[i, a]:
                continue
            for b in range(NN):
                if not alive[i, b]:
                    continue
                h = 0.0
                if a == b:
                    h += Kn[i, a]
                if shell[a] == shell[b]:
                    h += 0.5 * (Tv[i, a] + Tv[i, b])
                if h == 0.0:
                    continue
                # B row a: +n_a at node 0, -n_a at node 1+a
                for pa in range(2):
                    na = 0 if pa == 0 else 1 + a
                    sa = 1.0 if pa == 0 else -1.0
                    ga = nodes[i, na]
                    for pb in range(2):
                        nb_ = 0 if pb == 0 else 1 + b
                        sb = 1.0 if pb == 0 else -1.0
                        f = sa * sb * h
                        rk = rank[i, na, nb_]
                        for d in range(2):
                            base = indptr[2 * ga + d] + 2 * rk
                            for e in range(2):
                                data[base + e] += f * ndir[a, d] * ndir[b, e]


def bond_strain(lat, u, alive=None):
    """dL/l0 (N, 8) for displacement u (2N,) (0 for invalid / broken bonds)."""
    uu = u.reshape(-1, 2)
    q = np.where(lat.valid, lat.nbr, 0)
    dL = ((uu[:, None, :] - uu[q]) * lat.ndir[None]).sum(2)
    act = lat.valid if alive is None else (lat.valid & alive)
    return np.where(act, dL / lat.l0[None, :], 0.0)


# ---------------------------------------------------------------------------------------------------------------
# element matrices and the exact low-rank change of K when one bond breaks (for Woodbury updates)
# ---------------------------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def element_matrix(i, ndir, shell, Kn, Tv, alive_i):
    """Dense 2(NN+1) square element matrix K^(i) (nodes: i, then its NN neighbour slots) for the flags alive_i."""
    NN = ndir.shape[0]
    Ke = np.zeros((2 * NN + 2, 2 * NN + 2))
    for a in range(NN):
        if not alive_i[a]:
            continue
        for b in range(NN):
            if not alive_i[b]:
                continue
            h = 0.0
            if a == b:
                h += Kn[i, a]
            if shell[a] == shell[b]:
                h += 0.5 * (Tv[i, a] + Tv[i, b])
            if h == 0.0:
                continue
            for pa in range(2):
                na = 0 if pa == 0 else 1 + a
                sa = 1.0 if pa == 0 else -1.0
                for pb in range(2):
                    nb_ = 0 if pb == 0 else 1 + b
                    sb = 1.0 if pb == 0 else -1.0
                    f = sa * sb * h
                    for d in range(2):
                        for e in range(2):
                            Ke[2 * na + d, 2 * nb_ + e] += f * ndir[a, d] * ndir[b, e]
    return Ke


def break_delta(pat, Kn, Tv, alive_before, bonds):
    """Exact change of the assembled stiffness when the bonds (list of (p, k); both halves) break, starting from
    alive_before (N, 8).  Only the elements of the end particles change.  Returns (dofs, dK) with dK dense on dofs."""
    lat = pat.lat
    q = np.where(lat.valid, lat.nbr, 0)
    alive_after = alive_before.copy()
    owners = set()
    for (p, k) in bonds:
        qq = q[p, k]
        alive_after[p, k] = False
        alive_after[qq, k ^ 1] = False
        owners.add(int(p))
        owners.add(int(qq))
    nodes_all = []
    blocks = []
    for i in sorted(owners):
        nd = pat.nodes[i]
        ok = nd >= 0
        Kb = element_matrix(i, pat.ndir, pat.shell, Kn, Tv, (alive_before[i] & lat.valid[i]))
        Ka = element_matrix(i, pat.ndir, pat.shell, Kn, Tv, (alive_after[i] & lat.valid[i]))
        dofs = (2 * np.where(ok, nd, 0)[:, None] + np.arange(2)[None, :]).ravel()
        okd = np.repeat(ok, 2)
        blocks.append((dofs[okd], (Ka - Kb)[np.ix_(okd, okd)]))
        nodes_all.append(dofs[okd])
    D = np.unique(np.concatenate(nodes_all))
    pos = {d: j for j, d in enumerate(D)}
    dK = np.zeros((len(D), len(D)))
    for dofs, M in blocks:
        ix = np.array([pos[d] for d in dofs])
        dK[np.ix_(ix, ix)] += M
    return D, dK, alive_after


def lowrank(dK, tol=1e-12):
    """Symmetric dK = W diag(sgn) W^T (eigen-decomposition, |lambda| > tol * max)."""
    lam, vec = np.linalg.eigh(0.5 * (dK + dK.T))
    m = np.abs(lam) > tol * max(np.abs(lam).max(), 1e-300)
    return vec[:, m] * np.sqrt(np.abs(lam[m]))[None, :], np.sign(lam[m])
