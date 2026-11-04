"""Classical patch homogenisation with voids (numerical environment, numba): a copy of coarse3d/patch3d.py (read-only there; same
window, same two prescribed layers, same energy definition C_ab = u_a^T K_cell u_b / V_cell, same reflect padding)
where a bond exists only if both end voxels are solid (per-window neighbour table), void particles carry nothing and
the solve of a window with voids is regularised (eps = 1e-10 x mean diagonal on every free solid dof, 1 x mean diagonal
on the dofs of void particles):
fragments of the window that are not connected to its prescribed layers then stay at u = 0 and store no energy, and
zero-energy (floppy) modes of a porous window do not stop the Cholesky (they carry no bond stretch, hence no energy).
Without voids the result equals patch3d.patch_C (checked in __main__).
"""
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: F401,E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402
import geom3d as G  # noqa: E402
from patch3d import PatchGeom, _dL_all  # noqa: E402


@njit(cache=True)
def _bond_params_v(pp, nbr):
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


@njit(cache=True, parallel=True)
def _assemble_v(PP, NBR, SOL, ndir, free_idx, UA, eps, Kout, Bout):
    """as patch3d._assemble with per-window neighbour tables NBR (nw, Nw, 18) (-1 = no bond) and regularisation."""
    nwin = PP.shape[0]
    Nw = NBR.shape[1]
    for w in prange(nwin):
        nbr = NBR[w]
        Kn, Tv = _bond_params_v(PP[w], nbr)
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
        # regularisation (windows with voids only): eps x mean diagonal on solid free dofs, 1 x mean diagonal on void
        # free dofs
        anyvoid = False
        for i in range(Nw):
            if not SOL[w, i]:
                anyvoid = True
        nf = Kf.shape[0]
        dm = 0.0
        cnt = 0
        for r in range(nf):
            if Kf[r, r] > 0.0:
                dm += Kf[r, r]
                cnt += 1
        dm = dm / max(cnt, 1) if cnt > 0 else 1.0
        for i in range(Nw):
            fi = free_idx[i]
            if fi < 0 or not anyvoid:
                continue
            add = eps * dm if SOL[w, i] else dm
            for d in range(3):
                Kf[3 * fi + d, 3 * fi + d] += add
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
def _cell_energy_v(PP, NBR, X, ndir, free_idx, UA, cell_list, Vc, Cout):
    nwin = PP.shape[0]
    Nw = NBR.shape[1]
    for w in prange(nwin):
        nbr = NBR[w]
        Kn, Tv = _bond_params_v(PP[w], nbr)
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


@njit(cache=True, parallel=True)
def _window_nbr(SM, nbr, out):
    for w in prange(SM.shape[0]):
        for i in range(nbr.shape[0]):
            for k in range(18):
                q = nbr[i, k]
                out[w, i, k] = q if (q >= 0 and SM[w, i] and SM[w, q]) else -1


_GEOM = {}


def patch_C_v(Cvox, solid, n, s, chunk=256, eps=1e-10, timing=None):
    """Cvox (n^3, 3) voxel (C11, C12, C44), solid (n^3,) bool -> per-cell C (Nc, 6, 6) (zero for void cells)."""
    t0 = time.perf_counter()
    if s not in _GEOM:
        _GEOM[s] = PatchGeom(s)
    g = _GEOM[s]
    nc = n // s
    pp = G.sc_params_from_C(Cvox, 0.5).reshape(n, n, n, 3)
    pad = np.pad(pp, ((s, s), (s, s), (s, s), (0, 0)), mode="reflect")
    spad = np.pad(np.asarray(solid, bool).reshape(n, n, n), s, mode="reflect")
    W = g.W
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
        PP = np.ascontiguousarray(pad[idx[..., 0], idx[..., 1], idx[..., 2]])
        SM = np.ascontiguousarray(spad[idx[..., 0], idx[..., 1], idx[..., 2]])
        nw = len(st)
        NBR = np.empty((nw, lat.N, 18), np.int64)
        _window_nbr(SM, lat.nbr, NBR)
        K = np.zeros((nw, nf, nf))
        B = np.zeros((nw, nf, 6))
        _assemble_v(PP, NBR, SM, lat.ndir, g.free_idx, g.UA, eps, K, B)
        X = np.zeros_like(B)
        _solve_all(K, B, X)
        Cc = np.zeros((nw, 6, 6))
        _cell_energy_v(PP, NBR, np.ascontiguousarray(X), lat.ndir, g.free_idx, g.UA, g.cell_list, float(s ** 3), Cc)
        C[c0:c0 + nw] = Cc
    if timing is not None:
        timing["patch"] = time.perf_counter() - t0
    return C


if __name__ == "__main__":
    import patch3d
    rng = np.random.default_rng(0)
    for s in (2, 3):
        n = 4 * s
        Cv = G.iso_C(np.where(rng.random(n ** 3) < 0.4, 4.0, 1.0), np.full(n ** 3, 0.2))
        C0 = patch3d.patch_C(Cv, n, s)
        C1 = patch_C_v(Cv, np.ones(n ** 3, bool), n, s)
        print("s=%d no voids: max |C_v - C_patch3d| / max|C| = %.1e" % (s, np.abs(C1 - C0).max() / np.abs(C0).max()))
        sol = rng.random(n ** 3) > 0.1
        Cv2 = Cv * sol[:, None]
        C2 = patch_C_v(Cv2, sol, n, s)
        ev = np.linalg.eigvalsh(C2)
        print("   10 %% random voids: C symmetric %.1e, min eig / max eig %.3f, mean C11 ratio to no-void %.3f" % (
            np.abs(C2 - C2.transpose(0, 2, 1)).max(), ev.min() / ev.max(), C2[:, 0, 0].mean() / C0[:, 0, 0].mean()))
