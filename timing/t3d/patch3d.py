"""Classical local patch homogenisation for the 3D SC coarse model (mathematics of the 2D patch homogenisation, precomputed geometry).

For every coarse cell: a window of the voxel microstructure = cell + m = s voxels on every side ((3s)^3 voxels,
reflect-padded at the domain boundary); fine LPM (two-end mean bond parameters); affine displacement for the 6 unit
strains prescribed on the two outer voxel layers (two layers: with one layer the multibody term couples free particles
to prescribed particles that miss bonds); interior relaxed (dense solve, batched);
C_ab = (1/V_cell) u_a^T K_cell u_b with K_cell the Hessian of sum_{i in cell} U_i.  Then the generalised SC map
(geom3d.cell_params_from_C66) and the two-end mean give the coarse bond parameters.
Free particles per window: s = 2: 8 (the cell itself), s = 3: 125.  Exact for homogeneous material (check below).
"""
import time
import common  # noqa: F401  (thread limits before numpy)
import numpy as np
from numba import njit, prange
import geom3d as G


class PatchGeom:
    def __init__(self, s, ring=2):
        self.s = s
        self.m = s
        W = 3 * s
        self.W = W
        lat = G.LatticeGeom(W, W, W, 1.0)
        self.lat = lat
        ijk = lat.ijk
        fixed = np.any((ijk < ring) | (ijk >= W - ring), 1)
        self.fixed = fixed
        self.free_idx = -np.ones(lat.N, np.int64)
        self.free_idx[~fixed] = np.arange(int((~fixed).sum()))
        self.nfree = int((~fixed).sum())
        cell = np.all((ijk >= s) & (ijk < 2 * s), 1)
        self.cell_list = np.nonzero(cell)[0]
        self.xl = (ijk + 0.5 - 1.5 * s).astype(np.float64)        # coordinates relative to the cell centre (h = 1)
        # affine fields of the 6 unit strains on all window particles
        self.UA = np.stack([G.affine_disp(self.xl, G.unit_strain(k)) for k in range(6)], 0)   # (6, Nw, 3)


@njit(cache=True)
def _bond_params(pp, nbr):
    Nw = nbr.shape[0]
    Kn = np.zeros((Nw, 18))
    Tv = np.zeros((Nw, 18))
    for i in range(Nw):
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            c = 0 if k < 6 else 1
            Kn[i, k] = 0.5 * (pp[i, c] + pp[q, c])
            Tv[i, k] = 0.5 * (pp[i, 2] + pp[q, 2])
    return Kn, Tv


@njit(cache=True)
def _dL_all(u, nbr, ndir):
    Nw = nbr.shape[0]
    dL = np.zeros((Nw, 18))
    for i in range(Nw):
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            dL[i, k] = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
    return dL


@njit(cache=True, parallel=True)
def _assemble(PP, nbr, ndir, free_idx, UA, Kout, Bout):
    """PP (nw, Nw, 3) window particle parameters -> K_ff (nw, nf, nf) and rhs -(K u_a)_f for the 6 unit strains with
    u_a = affine on fixed particles, 0 on free ones (nw, nf, 6)."""
    nwin = PP.shape[0]
    Nw = nbr.shape[0]
    for w in prange(nwin):
        Kn, Tv = _bond_params(PP[w], nbr)
        Kf = Kout[w]
        Kf[:, :] = 0.0
        for i in range(Nw):
            for c in range(2):
                k0 = 0 if c == 0 else 6
                k1 = 6 if c == 0 else 18
                T = np.zeros(3)
                s = np.zeros(3)
                for a in range(k0, k1):
                    if nbr[i, a] < 0:
                        continue
                    for d in range(3):
                        T[d] += Tv[i, a] * ndir[a, d]
                        s[d] += ndir[a, d]
                fi = free_idx[i]
                for b in range(k0, k1):
                    qb = nbr[i, b]
                    if qb < 0:
                        continue
                    fb = free_idx[qb]
                    wb = np.zeros(3)
                    for d in range(3):
                        wb[d] = Kn[i, b] * ndir[b, d] + 0.5 * (T[d] + Tv[i, b] * s[d])
                    if fi >= 0:
                        for d in range(3):
                            for e in range(3):
                                Kf[3 * fi + d, 3 * fi + e] += wb[d] * ndir[b, e]
                        if fb >= 0:
                            for d in range(3):
                                for e in range(3):
                                    Kf[3 * fi + d, 3 * fb + e] -= wb[d] * ndir[b, e]
                                    Kf[3 * fb + e, 3 * fi + d] -= wb[d] * ndir[b, e]
                    if fb >= 0:
                        for a in range(k0, k1):
                            qa = nbr[i, a]
                            if qa < 0:
                                continue
                            fa = free_idx[qa]
                            if fa < 0:
                                continue
                            Hab = 0.5 * (Tv[i, a] + Tv[i, b])
                            if a == b:
                                Hab += Kn[i, a]
                            for d in range(3):
                                for e in range(3):
                                    Kf[3 * fa + d, 3 * fb + e] += Hab * ndir[a, d] * ndir[b, e]
        # right-hand sides: -(K u_a) at free particles, u_a = affine on fixed particles only
        for ca in range(6):
            u = np.zeros((Nw, 3))
            for i in range(Nw):
                if free_idx[i] < 0:
                    for d in range(3):
                        u[i, d] = UA[ca, i, d]
            dL = _dL_all(u, nbr, ndir)
            S = np.zeros((Nw, 2))
            TS = np.zeros((Nw, 2))
            for i in range(Nw):
                for k in range(18):
                    c = 0 if k < 6 else 1
                    S[i, c] += dL[i, k]
                    TS[i, c] += Tv[i, k] * dL[i, k]
            for i in range(Nw):
                fi = free_idx[i]
                if fi < 0:
                    continue
                for k in range(18):
                    q = nbr[i, k]
                    if q < 0:
                        continue
                    c = 0 if k < 6 else 1
                    f = 2.0 * Kn[i, k] * dL[i, k] + 0.5 * (TS[i, c] + TS[q, c]) + 0.5 * Tv[i, k] * (S[i, c] + S[q, c])
                    for d in range(3):
                        Bout[w, 3 * fi + d, ca] -= f * ndir[k, d]


@njit(cache=True, parallel=True)
def _cell_energy(PP, X, nbr, ndir, free_idx, UA, cell_list, Vc, Cout):
    """C_ab = (1/Vc) sum_{i in cell} dL_i(u_a)^T H_i dL_i(u_b), u_a = affine on fixed + solution X on free."""
    nwin = PP.shape[0]
    Nw = nbr.shape[0]
    for w in prange(nwin):
        Kn, Tv = _bond_params(PP[w], nbr)
        dLs = np.zeros((6, Nw, 18))
        for ca in range(6):
            u = np.zeros((Nw, 3))
            for i in range(Nw):
                fi = free_idx[i]
                for d in range(3):
                    u[i, d] = UA[ca, i, d] if fi < 0 else X[w, 3 * fi + d, ca]
            dLs[ca] = _dL_all(u, nbr, ndir)
        for a in range(6):
            for b in range(a, 6):
                e = 0.0
                for ii in range(cell_list.shape[0]):
                    i = cell_list[ii]
                    sa0 = 0.0
                    sa1 = 0.0
                    sb0 = 0.0
                    sb1 = 0.0
                    ta0 = 0.0
                    ta1 = 0.0
                    tb0 = 0.0
                    tb1 = 0.0
                    for k in range(18):
                        if nbr[i, k] < 0:
                            continue
                        da = dLs[a, i, k]
                        db = dLs[b, i, k]
                        e += Kn[i, k] * da * db
                        if k < 6:
                            sa0 += da
                            sb0 += db
                            ta0 += Tv[i, k] * da
                            tb0 += Tv[i, k] * db
                        else:
                            sa1 += da
                            sb1 += db
                            ta1 += Tv[i, k] * da
                            tb1 += Tv[i, k] * db
                    e += 0.5 * (ta0 * sb0 + tb0 * sa0 + ta1 * sb1 + tb1 * sa1)
                Cout[w, a, b] = e / Vc
                Cout[w, b, a] = e / Vc


@njit(cache=True, parallel=True)
def _solve_all(K, B, X):
    for w in prange(K.shape[0]):
        L = np.linalg.cholesky(K[w])
        y = np.linalg.solve(L, B[w])
        X[w] = np.linalg.solve(L.T, y)


_GEOM = {}


def patch_C(Cvox, n, s, chunk=256, timing=None):
    """Cvox (n^3, 3) voxel (C11, C12, C44) -> per-cell homogenised C (Nc, 6, 6) (Voigt, engineering shear)."""
    t0 = time.perf_counter()
    if s not in _GEOM:
        _GEOM[s] = PatchGeom(s)
    g = _GEOM[s]
    nc = n // s
    pp = G.sc_params_from_C(Cvox, 0.5).reshape(n, n, n, 3)
    pad = np.pad(pp, ((s, s), (s, s), (s, s), (0, 0)), mode="reflect")
    W = g.W
    # window particle parameters, window order = lattice order (k, j, i) of the (3s)^3 window
    kk, jj, ii = np.meshgrid(np.arange(nc), np.arange(nc), np.arange(nc), indexing="ij")
    starts = np.stack([kk.ravel(), jj.ravel(), ii.ravel()], 1) * s
    off = np.stack(np.meshgrid(np.arange(W), np.arange(W), np.arange(W), indexing="ij"), -1).reshape(-1, 3)
    Nc = nc ** 3
    C = np.zeros((Nc, 6, 6))
    lat = g.lat
    nf = 3 * g.nfree
    for c0 in range(0, Nc, chunk):
        st = starts[c0:c0 + chunk]
        idx = st[:, None, :] + off[None, :, :]
        PP = np.ascontiguousarray(pad[idx[..., 0], idx[..., 1], idx[..., 2]])       # (nw, W^3, 3)
        nw = len(st)
        K = np.zeros((nw, nf, nf))
        B = np.zeros((nw, nf, 6))
        _assemble(PP, lat.nbr, lat.ndir, g.free_idx, g.UA, K, B)
        X = np.zeros_like(B)
        _solve_all(K, B, X)
        Cc = np.zeros((nw, 6, 6))
        _cell_energy(PP, np.ascontiguousarray(X), lat.nbr, lat.ndir, g.free_idx, g.UA, g.cell_list, float(s ** 3), Cc)
        C[c0:c0 + nw] = Cc
    if timing is not None:
        timing["patch"] = time.perf_counter() - t0
    return C


if __name__ == "__main__":
    for s in (2, 3):
        n = 4 * s
        Cv = G.iso_C(np.full(n ** 3, 1.0), np.full(n ** 3, 0.2))
        C = patch_C(Cv, n, s)
        ref = G.iso_C66(G.iso_C(np.array([1.0]), np.array([0.2])))[0]
        print("s=%d homogeneous: max |C_patch - C| / C11 = %.1e" % (s, np.abs(C - ref[None]).max() / ref[0, 0]))
        rng = np.random.default_rng(0)
        Cv = G.iso_C(np.where(rng.random(n ** 3) < 0.4, 4.0, 1.0), np.full(n ** 3, 0.2))
        t0 = time.time()
        C = patch_C(Cv, n, s)
        print("   random two-phase: C symmetric %.1e, min eig %.3f, %.2f s for %d cells" %
              (np.abs(C - C.transpose(0, 2, 1)).max(), np.linalg.eigvalsh(C).min(), time.time() - t0, len(C)))
