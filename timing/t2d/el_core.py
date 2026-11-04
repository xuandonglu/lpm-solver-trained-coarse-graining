"""Direct solver of the 2D benchmark, used for every model (fine and coarse).  Problems, load cases and error measures
are in el_prob.py (numpy/scipy only).

Problems (sizes n x n with n = 48, 96, 192, 384): square lattice (LPM-C lattice 0), one particle per pixel, two phases
(random-field or disk inclusions, feature size 1.5-6 px), inclusion fraction 0.15-0.40, matrix E = 1 nu = 0.3,
inclusions E = c in {5, 10} nu = 0.2, plane strain, per-bond Kn/Tv = two-end mean; the distribution of the 2D test set.
Load cases (cg.py): KUBC ring (affine eps on the outer block ring: xx, yy, xy) and plates (tb_ty, tb_sx, lr_tx with
whole-block grips).  Boundary blocks are s x s blocks, so the fine reference depends on s; the fine model of a given s
sees exactly the loading of the coarse model of the same s.
Errors: global stiffness G (plates: grip reaction; ring: interior mean stress = apparent modulus), stress field
(relative L2 over free cells, coarse particle stress vs block-mean fine stress), interface energy density (relative L2
of the cell energy density over free cells that contain a phase boundary), displacement field.

Solver (same code for fine and coarse): numba element assembly into a fixed CSR pattern (elem.ElemPattern), per
boundary-condition set a geometric nested-dissection order of the free DOFs (ndorder) and the gather index of the
permuted free-free block (both geometry only, cached), SuperLU in that order (symmetric mode), multi-RHS solve per BC
set, then LPM stress / particle energy / reactions.  Timed per analysis (= all 6 load cases).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import common  # noqa
import lin2d
import elem
import ndorder as ND
from el_prob import *  # noqa: F401,F403
from el_prob import LOADS, BCSETS, prescribed, interface_cells, fine_bond_params, block_fields


# ----------------------------------------------------------------------------------------------------------------
# the solver
# ----------------------------------------------------------------------------------------------------------------
class Geometry:
    """Lattice + per-BC-set cached data (geometry only, reused for every microstructure / parameter set).
    IJ: block index (I, J) of every particle (fine: lat.ij // s, coarse: lat.ij)."""

    def __init__(self, lat, IJ, ncx, ncy, s, h, L):
        t0 = time.perf_counter()
        self.lat, self.IJ, self.ncx, self.ncy, self.s, self.h, self.L = lat, IJ, ncx, ncy, s, h, L
        self.pat = elem.ElemPattern(lat)
        pat = self.pat
        nd = pat.nd
        Kpos = sp.csr_matrix((np.arange(1, pat.nnz + 1, dtype=np.float64), pat.indices, pat.indptr), shape=(nd, nd))
        G = ND.node_graph(Kpos, lat.N)
        nperm = ND.nd_order(lat.x, G)
        dperm = ND.dof_perm(nperm)
        self.sets = {}
        for bs, loads in BCSETS.items():
            fixed, _, _ = prescribed(loads[0], lat.x, IJ, ncx, ncy, s, h, L)
            fd = np.zeros(nd, bool)
            fd[0::2] = fixed
            fd[1::2] = fixed
            free = ~fd
            fpos = -np.ones(nd, np.int64)
            fidx = np.nonzero(free)[0]
            fpos[fidx] = np.arange(len(fidx))
            perm = fpos[dperm]
            perm = perm[perm >= 0]                       # ND order of the free DOFs (positions in fidx)
            pidx = fidx[perm]                            # global DOF ids in elimination order
            sub = Kpos[pidx][:, pidx].tocsc()
            sub.sort_indices()
            gather = sub.data.astype(np.int64) - 1
            ubs, movs = [], []
            for ld in loads:
                _, up, moving = prescribed(ld, lat.x, IJ, ncx, ncy, s, h, L)
                ub = up.ravel().copy()
                ub[free] = 0.0
                ubs.append(ub)
                movs.append(moving)
            self.sets[bs] = dict(pidx=pidx, indptr=sub.indptr.copy(), indices=sub.indices.copy(), gather=gather,
                                 ub=np.stack(ubs, 1), moving=movs, loads=loads, fixed=fixed)
        self.t_init = time.perf_counter() - t0

    def run(self, Kn, Tv, timing=None, post=True, energy=True):
        """All 6 load cases.  Returns {load: dict(u (N,2), sig (N,3), U (N,), F (2,) or None)}."""
        lat = self.lat
        t0 = time.perf_counter()
        K = self.pat.assemble(Kn, Tv, lat.valid)
        t1 = time.perf_counter()
        tf = ts = 0.0
        out = {}
        for bs, d in self.sets.items():
            ta = time.perf_counter()
            A = sp.csc_matrix((K.data[d["gather"]], d["indices"], d["indptr"]), shape=(len(d["pidx"]),) * 2)
            lu = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            tb = time.perf_counter()
            Ub = d["ub"]
            rhs = -(K @ Ub)[d["pidx"]]
            X = lu.solve(np.ascontiguousarray(rhs))
            U = Ub.copy()
            U[d["pidx"]] = X
            tc = time.perf_counter()
            tf += tb - ta
            ts += tc - tb
            for j, ld in enumerate(d["loads"]):
                out[ld] = dict(u=U[:, j].reshape(-1, 2), moving=d["moving"][j], Ku=None)
            del lu
        t2 = time.perf_counter()
        if post:
            for ld, r in out.items():
                u = r["u"].ravel()
                r["sig"] = lat.stress(u, Kn, Tv)
                if energy:
                    r["U"] = lat.energy_per_particle(u, Kn, Tv)
                if r["moving"] is not None:
                    P = (K @ u).reshape(-1, 2)
                    r["F"] = P[r["moving"]].sum(0)
                else:
                    r["F"] = None
        t3 = time.perf_counter()
        if timing is not None:
            timing.update(assemble=t1 - t0, factor=tf, solve=ts, post=t3 - t2, total=t3 - t0)
        return out


_GEOM = {}


def fine_geometry(kind, n, s):
    """fine Geometry of a problem kind, size and s (BC blocks depend on s); cached."""
    key = ("f", kind, n, s)
    if key not in _GEOM:
        lk, L = "square", 1
        nc = n // s
        latf = lin2d.Lattice2D(lk, n, n, 1.0)
        _GEOM[key] = Geometry(latf, latf.ij // s, nc, nc, s, 1.0, L)
    return _GEOM[key]


def coarse_geometry(kind, n, s):
    key = ("c", kind, n, s)
    if key not in _GEOM:
        lk, L = "square", 1
        nc = n // s
        latc = lin2d.Lattice2D(lk, nc, nc, float(s), origin=(0.0, 0.0))
        _GEOM[key] = Geometry(latc, latc.ij.copy(), nc, nc, s, 1.0, L)
    return _GEOM[key]


def geometries(kind, n, s):
    """(fine Geometry, coarse Geometry), cached"""
    return fine_geometry(kind, n, s), coarse_geometry(kind, n, s)


def fine_reference(prob, s, timing=None, Kn=None, Tv=None):
    """Block-averaged fine fields for every load case (the reference of the coarse models of factor s)."""
    gf, _ = geometries(prob["kind"], prob["n"], s)
    lat = gf.lat
    if Kn is None:
        Kn, Tv = fine_bond_params(prob, lat)
    res = gf.run(Kn, Tv, timing=timing)
    nc = gf.ncx
    ref = {}
    for ld, r in res.items():
        ref[ld] = dict(u=block_mean(r["u"], gf.IJ, nc, nc), sig=block_mean(r["sig"], gf.IJ, nc, nc),
                       w=block_sum(r["U"], gf.IJ, nc, nc) / s ** 2, F=r["F"])
    n = prob["n"]
    iface = interface_cells(prob["phase"].reshape(n, n), s)
    return ref, iface


def coarse_run(prob, s, Kn, Tv, timing=None):
    gc_ = coarse_geometry(prob["kind"], prob["n"], s)
    res = gc_.run(Kn, Tv, timing=timing)
    out = {}
    for ld, r in res.items():
        out[ld] = dict(u=r["u"], sig=r["sig"], w=r["U"] / s ** 2, F=r["F"])
    return out


