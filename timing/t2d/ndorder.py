"""Geometric nested-dissection ordering for SuperLU (scipy's splu offers only MMD / COLAMD column orderings).

Nodes = particles (2 DOFs each, kept together).  Recursive coordinate bisection: split the node set at the median of
the longer bounding-box axis; the vertex separator = the nodes of the right half that have a graph neighbour in the
left half (the LPM element stiffness couples particles up to 2 lattice steps apart, so on a lattice the separator is
2 rows thick).  Order: left part, right part, separator last (recursively).  Leaves (<= leaf nodes) are ordered by
coordinate.  Works for any matrix whose nodes have coordinates (fine lattice and the coupled coarse-fine matrix).
The matrix is then factorised as P A P^T with permc_spec="NATURAL" and symmetric mode (diagonal pivots), i.e. a
sparse LU in nested-dissection order ("splu with a good ordering").
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def node_graph(A, nnode):
    """Node adjacency (CSR, nnode x nnode) of a 2-DOF-per-node matrix (DOF 2i, 2i+1 -> node i)."""
    C = A.tocoo()
    r, c = C.row // 2, C.col // 2
    m = r != c
    G = sp.csr_matrix((np.ones(m.sum(), np.int8), (r[m], c[m])), shape=(nnode, nnode))
    G = ((G + G.T) > 0).tocsr()
    return G


def nd_order(xy, G, leaf=48):
    """Nested-dissection permutation of the nodes (array of node ids, eliminated first .. last)."""
    n = len(xy)
    indptr, indices = G.indptr, G.indices

    def rec(S):
        if len(S) <= leaf:
            ax = 0 if np.ptp(xy[S, 0]) >= np.ptp(xy[S, 1]) else 1
            return [S[np.lexsort((xy[S, 1 - ax], xy[S, ax]))]]
        ax = 0 if np.ptp(xy[S, 0]) >= np.ptp(xy[S, 1]) else 1
        v = xy[S, ax]
        med = np.median(v)
        left = v < med
        if left.all() or (~left).all():
            left = v <= med
            if left.all() or (~left).all():
                return [S[np.lexsort((xy[S, 1 - ax], xy[S, ax]))]]
        inL = np.zeros(n, bool)
        inL[S[left]] = True
        R = S[~left]
        # nodes of R with a neighbour in L
        cnt = np.diff(indptr)[R]
        rows = np.repeat(R, cnt)
        # vectorised neighbour gather
        starts = indptr[R]
        idx = np.repeat(starts - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(cnt.sum())
        nb = indices[idx]
        hit = inL[nb]
        sep_nodes = np.unique(rows[hit])
        insep = np.zeros(n, bool)
        insep[sep_nodes] = True
        Rr = R[~insep[R]]
        parts = rec(S[left]) + (rec(Rr) if len(Rr) else [])
        if len(sep_nodes):
            parts.append(sep_nodes[np.lexsort((xy[sep_nodes, 1 - ax], xy[sep_nodes, ax]))])
        return parts

    import sys
    sys.setrecursionlimit(10000)
    parts = rec(np.arange(n))
    perm = np.concatenate(parts)
    assert len(perm) == n and len(np.unique(perm)) == n
    return perm


def dof_perm(node_perm):
    return (2 * node_perm[:, None] + np.arange(2)[None, :]).ravel()


class NDLU:
    """LU of A (square, 2 DOFs per node with coordinates xy) in nested-dissection order."""

    def __init__(self, A, xy, leaf=48, diag_pivot_thresh=0.0, G=None, perm=None):
        n = A.shape[0]
        assert n == 2 * len(xy)
        if perm is None:
            if G is None:
                G = node_graph(A, len(xy))
            perm = dof_perm(nd_order(xy, G, leaf))
        self.perm = perm
        self.iperm = np.empty_like(perm)
        self.iperm[perm] = np.arange(n)
        Ap = A[perm][:, perm].tocsc()
        self.lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=diag_pivot_thresh,
                            options=dict(SymmetricMode=True))
        self.shape = A.shape
        self.nnz = self.lu.L.nnz + self.lu.U.nnz

    def solve(self, b, trans="N"):
        b = np.asarray(b)
        x = self.lu.solve(np.ascontiguousarray(b[self.perm]), trans=trans)
        return x[self.iperm]
