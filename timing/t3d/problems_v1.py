"""The I8b s = 2 problems (timing/t3d/problems.py) for real cubes with voids (numerical environment): fine LPM (one particle per
voxel, void particles without bonds and fixed at 0) and coarse s = 2 model (void cells likewise), 9 load cases in 3 BC
sets, block-Jacobi PCG (lin3d._pcg) and AMG-PCG (amg3d_v), computeStress, the stress-field error at R = 2 (native) and
R = 4 (2^3-cell blocks), mean over the 9 load cases, against the converged fine reference (data/fine_SET_OPT.npz).
Geometry-only parts (lattice, BCs, AMG aggregation) are built once per size like in I8b; everything that depends on
the specimen (neighbour table of the solid set, bond parameters, AMG numeric setup) is inside the timed build/solve.
"""
import time
import numpy as np
import geom3d as G
import lin3d as L3
import amg3d_v as AV

S = 2
AMG_CFG = dict(agg=3, coarse_max=1000, nu=1, gamma=2, power_its=5)


class Problem:
    def __init__(self, n, coarse, amg_cfg=AMG_CFG):
        self.n, self.coarse = n, coarse
        nc = n // S
        self.nc = nc
        if coarse:
            lat = L3.Lattice3D(nc, nc, nc, float(S))
            IJK = lat.ijk
        else:
            lat = L3.Lattice3D(n, n, n, 1.0)
            IJK = lat.ijk // S
        self.lat = lat
        self.sets = []
        for name in ("ring", "tb", "lr"):
            loads = [l for l, ld in enumerate(G.LOADS) if G.BCSET[ld] == name]
            ups, fixed = [], None
            for l in loads:
                fx, up, _ = G.prescribed(G.LOADS[l], lat.x, IJK, (nc,) * 3, S)
                fixed = fx
                ups.append(np.ascontiguousarray(up))
            self.sets.append(dict(name=name, free=~fixed, fixed=fixed, loads=loads, ups=ups))
        self.cid = None if coarse else ((IJK[:, 2] * nc + IJK[:, 1]) * nc + IJK[:, 0])
        latc = G.LatticeGeom(nc, nc, nc, S)
        self.freecell = np.array([~G.prescribed(ld, latc.x, latc.ijk, (nc,) * 3, S)[0] for ld in G.LOADS])
        m = nc // 2
        ijk = latc.ijk
        self.rid = ((ijk[:, 2] // 2) * m + ijk[:, 1] // 2) * m + ijk[:, 0] // 2
        self.NR = m ** 3
        cnt = np.array([np.bincount(self.rid, weights=f.astype(float), minlength=self.NR) for f in self.freecell])
        self.freeR = cnt == 8
        self.amg_cfg = dict(amg_cfg)
        t0 = time.perf_counter()
        gkw = dict(agg=self.amg_cfg.pop("agg", 3), coarse_max=self.amg_cfg.pop("coarse_max", 1000))
        for st in self.sets:
            st["geom"] = AV.AMGGeom(lat, st["free"], **gkw)
        self.t_geom_amg = time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------------ specimen
    def specimen(self, solid):
        """per-specimen neighbour table (timed as part of the build)."""
        return AV.specimen_nbr(self.lat, np.asarray(solid, bool))

    def solve(self, solver, Kn, Tv, nbr, solid, tol):
        if solver == "cg":
            return self.solve_cg(Kn, Tv, nbr, solid, tol)
        if solver == "jac":
            return self.solve_jac(Kn, Tv, nbr, solid, tol)
        if solver == "amg":
            return self.solve_amg(Kn, Tv, nbr, solid, tol)
        raise ValueError(solver)

    def solve_cg(self, Kn, Tv, nbr, solid, tol):
        """the LPM-C workflow, unpreconditioned CG from zero (cgplain.cg3) for every load case."""
        from cgplain import cg3
        lat = self.lat
        t0 = time.perf_counter()
        us, its = [None] * 9, [0] * 9
        for st in self.sets:
            free = st["free"] & solid
            for l, up in zip(st["loads"], st["ups"]):
                upm = up * solid[:, None]
                b = -L3.k_apply(upm, nbr, lat.ndir, Kn, Tv)
                b[~free] = 0.0
                x = np.zeros((lat.N, 3))
                it, _ = cg3(nbr, lat.ndir, Kn, Tv, free, b, x, tol, 200000)
                us[l] = upm + x
                its[l] = int(it)
        return time.perf_counter() - t0, its, us, {}

    def solve_jac(self, Kn, Tv, nbr, solid, tol):
        lat = self.lat
        t0 = time.perf_counter()
        D = L3.k_block_diag(nbr, lat.ndir, Kn, Tv)
        us, its = [None] * 9, [0] * 9
        for st in self.sets:
            free = st["free"] & solid
            M = L3.k_block_jacobi(D, free)
            for l, up in zip(st["loads"], st["ups"]):
                upm = up * solid[:, None]
                b = -L3.k_apply(upm, nbr, lat.ndir, Kn, Tv)
                b[~free] = 0.0
                x = np.zeros((lat.N, 3))
                it, _ = L3._pcg(nbr, lat.ndir, Kn, Tv, M, free, b, x, tol, 20000)
                us[l] = upm + x
                its[l] = int(it)
        return time.perf_counter() - t0, its, us, {}

    def solve_amg(self, Kn, Tv, nbr, solid, tol):
        lat = self.lat
        t0 = time.perf_counter()
        D = L3.k_block_diag(nbr, lat.ndir, Kn, Tv)
        us, its = [None] * 9, [0] * 9
        t_setup = 0.0
        rho0 = None
        nreg = 0
        for st in self.sets:
            ts = time.perf_counter()
            free = st["free"] & solid
            M = L3.k_block_jacobi(D, free)
            h = AV.AMG3DV(st["geom"], Kn, Tv, nbr, solid, Minv0=M, rho0=rho0, **self.amg_cfg)
            rho0 = h.rho0
            nreg += h.nreg
            t_setup += time.perf_counter() - ts
            for l, up in zip(st["loads"], st["ups"]):
                upm = up * solid[:, None]
                b = -L3.k_apply(upm, nbr, lat.ndir, Kn, Tv)
                b[~free] = 0.0
                x, it, _ = h.pcg(b, tol)
                us[l] = upm + x
                its[l] = it
        return time.perf_counter() - t0, its, us, dict(t_setup=t_setup, nreg=nreg)

    def stresses(self, Kn, Tv, nbr, us):
        lat = self.lat
        return [L3.k_stress(np.ascontiguousarray(u), nbr, lat.ndir, lat.l0, Kn, Tv, lat.V) for u in us]

    def block_mean(self, sig):
        nc3 = self.nc ** 3
        out = np.empty((nc3, 6))
        for k in range(6):
            out[:, k] = np.bincount(self.cid, weights=sig[:, k], minlength=nc3)
        return out / S ** 3

    def field_errors(self, sigs, ref, cell):
        """stress-field error per load case at R = 2 (BC-free existing cells) and R = 4 (BC-free 2^3 blocks)."""
        e2, e4 = [], []
        for l, sg in enumerate(sigs):
            if not self.coarse:
                sg = self.block_mean(sg)
            r = ref[l].astype(np.float64)
            m = self.freecell[l] & cell
            e2.append(float(np.linalg.norm((sg - r)[m]) / np.linalg.norm(r[m])))
            a = np.stack([np.bincount(self.rid, weights=sg[:, k], minlength=self.NR) for k in range(6)], 1) / 8
            b = np.stack([np.bincount(self.rid, weights=r[:, k], minlength=self.NR) for k in range(6)], 1) / 8
            mR = self.freeR[l]
            e4.append(float(np.linalg.norm((a - b)[mR]) / np.linalg.norm(b[mR])))
        return e2, e4


# ---------------------------------------------------------------------------------------------------------------------
def fine_params(lat, Cv):
    return G.bond_params_two_end_mean(lat, G.sc_params_from_C(Cv, 0.5))


def coarse_floor(Kn, Tv, nbr, kmin=1e-3):
    """mask to existing bonds + the minimum stiffness of train_v1.finalize (all coarse models)."""
    v = nbr >= 0
    Kn = Kn * v
    Tv = Tv * v
    km = kmin * Kn.sum() / max(v.sum(), 1)
    return np.ascontiguousarray(np.where(v, np.maximum(Kn, km), Kn)), np.ascontiguousarray(Tv)


def voigt_params(latc, Cv, n):
    return G.bond_params_from_cells(latc, G.cell_params_from_C66(G.voigt_cells(Cv, n, S), S / 2.0))


def warmup():
    P = Problem(12, False)
    sol = np.ones(12 ** 3, bool)
    sol[100:140] = False
    Cv = G.iso_C(np.where(np.arange(12 ** 3) % 3 == 0, 2.0, 1.0) * sol, np.full(12 ** 3, 0.2))
    nbr = P.specimen(sol)
    Kn, Tv = fine_params(P.lat, Cv)
    for so in ("jac", "amg"):
        _, _, us, _ = P.solve(so, Kn, Tv, nbr, sol, 1e-6)
        P.stresses(Kn, Tv, nbr, us)
