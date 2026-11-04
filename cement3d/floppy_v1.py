"""Numerical removal of zero-energy (floppy) modes from the solid / cell masks (torch environment, GPU), after the combinatorial
cleaning of materials.py.

A porous voxel structure can keep particles that move without stretching any bond (e.g. a pair at a pore wall or at
a free face of the cube anchored by too few bonds); K_ff is then singular (consistent: stresses, energies and
reactions are unaffected, but displacements are solver dependent and the adjoint of the training is ill-posed).
Test: for each BC set (KUBC ring, top/bottom grips, left/right grips; the s = 2 sets contain the free sets of s = 3),
x0 = random on the free solid particles, u = x0 + x with K_ff x = -K_ff x0 (block-Jacobi PCG to 1e-12) lies in
null(K_ff); particles with |u| > 1e-4 (|x0| ~ 1) are floppy -> removed, combinatorial cleaning, repeat until none.
The coarse cell masks (s = 2, 3) are rebuilt from the cleaned solid set and tested the same way on the coarse lattice
(Voigt bond parameters).  Cubes without voids are skipped (a full grid has no floppy mode).
usage: python -I floppy_v1.py SET [SET ...]   (updates data/mat_SET_{a,b}.npz in place: solid, E, nu, ph, cell_s,
       iface_s, padj_s, nrm_s; adds nfloppy_fine, nfloppy_s)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import geom3d as G  # noqa: E402
import materials as MT  # noqa: E402
import torch_v1 as TV  # noqa: E402

torch.set_num_threads(4)


def bc_free(lat, IJK, nc, s):
    out = []
    for ld in ("k_xx", "tb_tz", "lr_tx"):
        fixed, _, _ = G.prescribed(ld, lat.x, IJK, (nc,) * 3, s)
        out.append(torch.as_tensor(~fixed, device="cuda"))
    return out


def floppy(op, frees, Kn, Tv, mask, seed):
    """mask (B, N) bool tensor -> floppy (B, N) bool."""
    op.set_mask(mask)
    Kn = (Kn * op.valid[:, 0])[:, None]
    Tv = (Tv * op.valid[:, 0])[:, None]
    g = torch.Generator(device="cuda").manual_seed(seed)
    fl = torch.zeros_like(mask)
    for fb in frees:
        free = (fb[None] & mask).double()[:, None]
        x0 = torch.randn(free.shape + (3,), device="cuda", dtype=torch.float64, generator=g) * free[..., None]
        u, it, rel = op.solve(Kn, Tv, x0, free, tol=1e-12, maxit=60000)
        fl |= (u[:, 0].norm(dim=-1) > 1e-4) & (free[:, 0] > 0)
    return fl


def refine_fine(n, solids, Es, nus, bs):
    lat = G.LatticeGeom(n, n, n, 1.0)
    op = TV.SCTV(lat)
    frees = bc_free(lat, lat.ijk // 2, n // 2, 2)
    nfl = np.zeros(len(solids), np.int64)
    todo = [i for i in range(len(solids)) if not solids[i].all()]
    for rnd in range(6):
        if not todo:
            break
        nxt = []
        for k0 in range(0, len(todo), bs):
            idx = todo[k0:k0 + bs]
            KT = [G.bond_params_two_end_mean(lat, G.sc_params_from_C(G.iso_C(Es[i], nus[i]), 0.5)) for i in idx]
            Kn = torch.as_tensor(np.array([k for k, _ in KT]), device="cuda")
            Tv = torch.as_tensor(np.array([t for _, t in KT]), device="cuda")
            m = torch.as_tensor(np.array([solids[i] for i in idx]), device="cuda")
            fl = floppy(op, frees, Kn, Tv, m, 1000 * rnd + k0).cpu().numpy()
            for j, i in enumerate(idx):
                if fl[j].any():
                    nfl[i] += int(fl[j].sum())
                    s_, _ = MT.clean((solids[i] & ~fl[j]).reshape(n, n, n))
                    solids[i] = s_.ravel()
                    nxt.append(i)
        todo = nxt
    return nfl


def refine_coarse(n, s, solids, Es, nus, bs):
    nc = n // s
    lat = G.LatticeGeom(nc, nc, nc, s)
    op = TV.SCTV(lat)
    frees = bc_free(lat, lat.ijk, nc, s)
    cells, nrm = [], []
    for i in range(len(solids)):
        c, r = MT.coarse_mask(solids[i], n, s)
        cells.append(c)
        nrm.append(r)
    nfl = np.zeros(len(solids), np.int64)
    todo = [i for i in range(len(solids)) if not cells[i].all()]
    for rnd in range(6):
        if not todo:
            break
        nxt = []
        for k0 in range(0, len(todo), bs):
            idx = todo[k0:k0 + bs]
            KT = [G.bond_params_from_cells(lat, G.cell_params_from_C66(
                G.voigt_cells(G.iso_C(Es[i] * solids[i], nus[i] * solids[i]), n, s), s / 2.0)) for i in idx]
            Kn = torch.as_tensor(np.array([k for k, _ in KT]), device="cuda")
            Tv = torch.as_tensor(np.array([t for _, t in KT]), device="cuda")
            m = torch.as_tensor(np.array([cells[i] for i in idx]), device="cuda")
            fl = floppy(op, frees, Kn, Tv, m, 7 + 1000 * rnd + k0).cpu().numpy()
            for j, i in enumerate(idx):
                if fl[j].any():
                    nfl[i] += int(fl[j].sum())
                    c_, r_ = MT.clean((cells[i] & ~fl[j]).reshape(nc, nc, nc))
                    nrm[i] += r_ + int(fl[j].sum())
                    cells[i] = c_.ravel()
                    nxt.append(i)
        todo = nxt
    return np.array(cells), np.array(nrm), nfl


def run(st):
    labs = np.load(MTfile(st))["labels"]
    for opt in "ab":
        fn = common.data("mat_%s_%s.npz" % (st, opt))
        d = dict(np.load(fn))
        n = int(d["n"])
        t0 = time.time()
        bs = 16 if n <= 24 else (4 if n <= 48 else 1)
        solids = [x.copy() for x in d["solid"]]
        nfl = refine_fine(n, solids, d["E"], d["nu"], bs)
        d["solid"] = np.array(solids)
        for k in ("E", "nu"):
            d[k] = d[k] * d["solid"]
        d["ph"] = np.where(d["solid"], d["ph"], -1).astype(np.int8)
        d["nfloppy_fine"] = nfl
        msg = []
        for s in (2, 3):
            cells, nrm, nflc = refine_coarse(n, s, solids, d["E"], d["nu"], bs)
            d["cell_%d" % s] = cells
            d["nrm_%d" % s] = nrm
            d["nfloppy_%d" % s] = nflc
            ifc, pa = [], []
            for i in range(len(solids)):
                m = dict(solid=d["solid"][i], ph=d["ph"][i].astype(np.int64))
                a_, b_ = MT.cell_sets(labs[i], m, n, s)
                ifc.append(a_)
                pa.append(b_)
            d["iface_%d" % s] = np.array(ifc)
            d["padj_%d" % s] = np.array(pa)
            msg.append("s=%d floppy cells %d in %d cubes" % (s, nflc.sum(), (nflc > 0).sum()))
        d["void_frac"] = 1 - d["solid"].mean(1)
        np.savez_compressed(fn, **d)
        print("%s opt %s: floppy fine particles %d in %d cubes; %s (%.0f s)" % (
            st, opt, nfl.sum(), (nfl > 0).sum(), "; ".join(msg), time.time() - t0), flush=True)


def MTfile(st):
    return common.data("cubes_%s.npz" % st) if not st.startswith("syn") else common.data("synth_%s.npz" % st[4:])


if __name__ == "__main__":
    for st in sys.argv[1:]:
        run(st)
