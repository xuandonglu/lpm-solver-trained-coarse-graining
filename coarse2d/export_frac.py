"""Fracture data: local first-failure load factors of the fine LPM for every coarse bond (training targets of the
coarse critical stretch), plus the two classical coarse critical-stretch baselines.

Per-bond critical bond strain of the fine model (brittle, LPM-C plmode 6 rule dL/l0 >= crit):
  matrix-matrix XI_M, inclusion-inclusion XI_I, interface (phases differ) XI_INT.
For coarse bond b (direction class d = x, y, diag+, diag-) between cells I and J, its region R(b) = fine bonds of the
same direction class with at least one end in I or J.  For load case l (cg.LOADS, magnitude cg.EPS):
  lam_f(b, l) = 1 / max_{f in R(b)} (dL_f/l0_f) / crit_f      (load multiplier at which R(b) starts to break)
Baselines (no fine solve of the specimen):
  naive  : crit_C(b) = min_{f in R(b)} crit_f                                   (weakest link)
  patch  : crit_C(b) = 1 / max_{f in R(b)} A_f / crit_f, A_f = fine bond strain in the cell window under the affine
           strain n_b (x) n_b (unit coarse bond strain), window/BC as cg.patch_cells (classical localisation)
usage: python export_frac.py SPLIT [--nmax K]
"""
import argparse
import os
import time
import multiprocessing as mp
import common  # noqa
import numpy as np

XI_M, XI_I, XI_INT = 1.0e-3, 2.0e-3, 0.6e-3
CLS = np.array([0, 0, 1, 1, 2, 2, 3, 3])          # slot -> direction class
SS = (2, 3, 4)


def cell_max(lat_f, s, ncx, val, cls_of_slot=CLS):
    """m[c, d] = max over fine bonds of class d with at least one end in cell c of val (N, 8)."""
    q = np.where(lat_f.valid, lat_f.nbr, 0)
    I, J = lat_f.ij[:, 0] // s, lat_f.ij[:, 1] // s
    cp = (J * ncx + I)[:, None].repeat(8, 1)
    cq = (J * ncx + I)[q]
    d = np.broadcast_to(cls_of_slot[None, :], val.shape)
    ok = lat_f.valid
    m = np.zeros((ncx * (lat_f.ny // s), 4))
    np.maximum.at(m, (cp[ok], d[ok]), val[ok])
    np.maximum.at(m, (cq[ok], d[ok]), val[ok])
    return m


def bond_from_cells(m, b, ncx):
    """forward bonds b (n,4) [p,k,I,J] -> max(m[I,d], m[J,d])."""
    import windows as WIN
    q, _ = WIN.neighbor_of(b, ncx)
    d = CLS[b[:, 1]]
    return np.maximum(m[b[:, 0], d], m[q, d])


def patch_strength_cells(Cpix, phase, nx, ny, s, h=1.0):
    """a[c, d] = max over fine bonds f (class d, touching cell c) of A_f / crit_f under eps = n_d (x) n_d."""
    import lin2d
    m = s
    w = s + 2 * m
    ncx, ncy = nx // s, ny // s
    C2d = Cpix.reshape(ny, nx, 3)
    Cpad = np.pad(C2d, ((m, m), (m, m), (0, 0)), mode="reflect")
    Ppad = np.pad(phase.reshape(ny, nx), ((m, m), (m, m)), mode="reflect")
    gap = 2
    W = w + gap
    gx, gy = ncx * W, ncy * W
    act = np.zeros((gy, gx), bool)
    Cbig = np.zeros((gy, gx, 3))
    Pbig = np.zeros((gy, gx), np.int64)
    Cbig[:] = Cpix.mean(0)
    for J in range(ncy):
        for I in range(ncx):
            act[J * W:J * W + w, I * W:I * W + w] = True
            Cbig[J * W:J * W + w, I * W:I * W + w] = Cpad[J * s:J * s + w, I * s:I * s + w]
            Pbig[J * W:J * W + w, I * W:I * W + w] = Ppad[J * s:J * s + w, I * s:I * s + w]
    lat = lin2d.Lattice2D("square", gx, gy, h, active=act)
    Cp = Cbig.reshape(-1, 3)[lat.sites]
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cp))
    crit = brittle_crit(lat, Pbig.ravel()[lat.sites])
    K, B, H = lat.stiffness(Kn, Tv)
    li, lj = lat.ij[:, 0] % W, lat.ij[:, 1] % W
    wid = (lat.ij[:, 1] // W) * ncx + lat.ij[:, 0] // W
    outer = (li < 2) | (li >= w - 2) | (lj < 2) | (lj >= w - 2)
    incell = (li >= m) & (li < m + s) & (lj >= m) & (lj < m + s)
    xl = np.stack([li - (m + s / 2 - 0.5), lj - (m + s / 2 - 0.5)], 1) * h
    fixed = np.nonzero(outer)[0]
    fdofs = np.concatenate([2 * fixed, 2 * fixed + 1])
    free = np.ones(2 * lat.N, bool)
    free[fdofs] = False
    import scipy.sparse.linalg as spla
    lu = spla.splu(K[free][:, free].tocsc(), permc_spec="COLAMD")
    q = np.where(lat.valid, lat.nbr, 0)
    out = np.zeros((ncx * ncy, 4))
    s2 = 1 / np.sqrt(2)
    for d, n in enumerate([(1, 0), (0, 1), (s2, s2), (s2, -s2)]):
        e = np.outer(n, n)
        ua = lin2d.affine_disp(xl, e)
        u = np.zeros(2 * lat.N)
        u[fdofs] = np.concatenate([ua[fixed, 0], ua[fixed, 1]])
        u[free] = lu.solve(-K[free][:, fdofs] @ u[fdofs])
        dL, _ = lat.bond_forces(u, Kn, Tv)
        r = np.where(lat.valid & (CLS[None, :] == d), dL / lat.l0[None, :] / crit, 0.0)
        touch = incell[:, None] | incell[q]
        r = np.where(touch & lat.valid, r, 0.0)
        # owner cell = window id of either end (both ends are in the same window)
        np.maximum.at(out[:, d], wid[:, None].repeat(8, 1)[r > 0], r[r > 0])
    return out


def one(args):
    split, k = args
    import lin2d
    import cg
    import windows as WIN
    d = np.load("data/%s_n%d.npz" % (split, 96 if split == "test_big" else 48), allow_pickle=True)
    n = int(d["n"])
    ph = d["phase"][k].astype(np.int64)
    Cpix = d["Cpix"][k]
    lat = lin2d.Lattice2D("square", n, n, 1.0)
    crit = brittle_crit(lat, ph)
    out = {}
    for s in SS:
        ncx = n // s
        ref = cg.FineRef(Cpix, n, n, s, phase=ph)
        b, shell = WIN.bond_list(ncx, ncx)
        lam = np.zeros((len(b), len(cg.LOADS)))
        lmin = np.zeros(len(cg.LOADS))
        for il, ld in enumerate(cg.LOADS):
            dL, _ = lat.bond_forces(ref.U[ld].ravel(), ref.Kn, ref.Tv)
            r = np.where(lat.valid, np.maximum(dL, 0) / lat.l0[None, :] / crit, 0.0)
            m = cell_max(lat, s, ncx, r)
            lam[:, il] = 1.0 / np.maximum(bond_from_cells(m, b, ncx), 1e-30)
            lmin[il] = 1.0 / r.max()
        naive_m = cell_max(lat, s, ncx, np.where(lat.valid, 1.0 / crit, 0.0))
        naive = 1.0 / bond_from_cells(naive_m, b, ncx)
        pm = patch_strength_cells(Cpix, ph, n, n, s)
        patch = 1.0 / np.maximum(bond_from_cells(pm, b, ncx), 1e-30)
        sel0, sel1 = shell == 0, shell == 1
        for key, v in (("lam", lam), ("naive", naive), ("patch", patch)):
            out["%s_%d_0" % (key, s)] = v[sel0]
            out["%s_%d_1" % (key, s)] = v[sel1]
        out["lmin_%d" % s] = lmin
    return out


def brittle_crit(lat, ph):
    import brittle
    return brittle.crit_from_phase(lat, ph, XI_M, XI_I, XI_INT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--nmax", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = np.load("data/%s_n%d.npz" % (a.split, 96 if a.split == "test_big" else 48), allow_pickle=True)
    S = len(d["phi"]) if a.nmax == 0 else min(a.nmax, len(d["phi"]))
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(a.split, k) for k in range(S)], chunksize=4)
    data = {key: np.array([r[key] for r in res]) for key in res[0]}
    np.savez_compressed("data/frac_%s.npz" % a.split, **data)
    print("frac %s: %d samples %.1f s" % (a.split, S, time.time() - t0))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
