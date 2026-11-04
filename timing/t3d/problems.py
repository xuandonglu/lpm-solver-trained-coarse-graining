"""Shared pieces of the I8b benchmark (numerical environment): the s = 2 problems of I8 (fine LPM and coarse s = 2 model, 9 load
cases in 3 BC sets), the three solvers (block-Jacobi PCG of I8, AMG-PCG, SuperLU), stresses and the I8 field error.

Everything is the I8 definition: block boundary conditions (geom3d.prescribed), PCG stopping rule ||r|| <= tol ||b||
from x0 = 0, stress = computeStress (lin3d.k_stress), fine stresses averaged over the s^3 blocks, error = relative L2
over the free cells of the coarse lattice, mean over the 9 load cases, reference = converged fine LPM (data/fine_*.npz).
"""
import time
import common
import numpy as np
import scipy.sparse.linalg as spla
import geom3d as G
import lin3d as L3
import amg3d as A3

S = 2
# chosen in tune_amg*.py (logs/tune_amg*.log): 3x3x3 aggregates, dense coarsest level <= 1000 unknowns, V(1,1) on the
# finest level with W-cycles below, rho from 5 power iterations
AMG_CFG = dict(agg=3, coarse_max=1000, nu=1, gamma=2, power_its=5)


class Problem:
    """Lattice + the 3 BC sets of one model: fine (n^3 voxel particles) or coarse (nc^3 block particles), s = 2."""

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
                assert fixed is None or np.array_equal(fixed, fx)
                fixed = fx
                ups.append(np.ascontiguousarray(up))
            self.sets.append(dict(name=name, free=~fixed, fixed=fixed, loads=loads, ups=ups))
        self.cid = None if coarse else ((IJK[:, 2] * nc + IJK[:, 1]) * nc + IJK[:, 0])
        latc = G.LatticeGeom(nc, nc, nc, S)
        self.freecell = np.array([~G.prescribed(ld, latc.x, latc.ijk, (nc,) * 3, S)[0] for ld in G.LOADS])
        # geometry-only, built once per lattice (like the neighbour lists): AMG aggregation, CSR pattern
        self.amg_cfg = dict(amg_cfg)
        t0 = time.perf_counter()
        gkw = dict(agg=self.amg_cfg.pop("agg", 3), coarse_max=self.amg_cfg.pop("coarse_max", 1000))
        for st in self.sets:
            st["geom"] = A3.AMGGeom(lat, st["free"], **gkw)
        self.t_geom_amg = time.perf_counter() - t0
        self._pat_done = False

    def describe_amg(self):
        return "; ".join("%s: %s" % (st["name"], st["geom"].describe()) for st in self.sets)

    def ensure_pattern(self):
        if not self._pat_done:
            t0 = time.perf_counter()
            self.lat._pat = L3.csr_pattern(self.lat.ijk, self.lat.nbr)
            self.t_pattern = time.perf_counter() - t0
            self._pat_done = True

    # ------------------------------------------------------------------------------------------------ solvers
    def solve(self, solver, Kn, Tv, tol=None, **kw):
        if solver == "cg":
            return self.solve_cg(Kn, Tv, tol)
        if solver == "jac":
            return self.solve_jac(Kn, Tv, tol)
        if solver == "amg":
            return self.solve_amg(Kn, Tv, tol)
        if solver.startswith("lu"):
            return self.solve_lu(Kn, Tv, permc=kw.get("permc", "MMD_AT_PLUS_A"))
        raise ValueError(solver)

    def solve_cg(self, Kn, Tv, tol):
        """the LPM-C workflow, unpreconditioned CG from zero (cgplain.cg3) for every load case."""
        from cgplain import cg3
        lat = self.lat
        t0 = time.perf_counter()
        us, its = [None] * 9, [0] * 9
        for st in self.sets:
            free = st["free"]
            for l, up in zip(st["loads"], st["ups"]):
                upm = np.zeros((lat.N, 3))
                upm[~free] = up[~free]
                b = -L3.k_apply(upm, lat.nbr, lat.ndir, Kn, Tv)
                b[~free] = 0.0
                x = np.zeros((lat.N, 3))
                it, _ = cg3(lat.nbr, lat.ndir, Kn, Tv, free, b, x, tol, 200000)
                us[l] = upm + x
                its[l] = int(it)
        return time.perf_counter() - t0, its, us, {}

    def solve_jac(self, Kn, Tv, tol):
        """I8 (bench_cpu.pcg_all): block-Jacobi PCG of lin3d for every load case."""
        lat = self.lat
        t0 = time.perf_counter()
        D = lat.block_diag(Kn, Tv)
        us, its = [None] * 9, [0] * 9
        for st in self.sets:
            M = L3.k_block_jacobi(D, st["free"])
            for l, up in zip(st["loads"], st["ups"]):
                us[l], info = L3.solve_pcg(lat, Kn, Tv, st["fixed"], up, tol=tol, M=M)
                its[l] = info["iters"]
        return time.perf_counter() - t0, its, us, {}

    def solve_amg(self, Kn, Tv, tol):
        """AMG-PCG (amg3d): one hierarchy per BC set (numeric setup timed), PCG for every load case."""
        lat = self.lat
        t0 = time.perf_counter()
        D = lat.block_diag(Kn, Tv)
        us, its = [None] * 9, [0] * 9
        t_setup = 0.0
        rho0 = None
        for st in self.sets:
            ts = time.perf_counter()
            M = L3.k_block_jacobi(D, st["free"])
            h = A3.AMG3D(st["geom"], Kn, Tv, Minv0=M, rho0=rho0, **self.amg_cfg)
            rho0 = h.rho0                        # the smoother weight of the first BC set is reused for the others
            t_setup += time.perf_counter() - ts
            for l, up in zip(st["loads"], st["ups"]):
                b = -L3.k_apply(up, lat.nbr, lat.ndir, Kn, Tv)
                A3._mask(b, st["free"])
                x, it, _ = h.pcg(b, tol)
                us[l] = up + x
                its[l] = it
        return time.perf_counter() - t0, its, us, dict(t_setup=t_setup)

    def solve_lu(self, Kn, Tv, permc="MMD_AT_PLUS_A"):
        """SuperLU on the assembled K_ff (one factorisation per BC set; symmetric mode, diagonal pivots)."""
        self.ensure_pattern()
        lat = self.lat
        t0 = time.perf_counter()
        K = lat.stiffness(Kn, Tv)
        us, info = [None] * 9, dict(t_fact=0.0, nnz_LU=0, t_solves=0.0)
        for st in self.sets:
            fd = np.repeat(st["free"], 3)
            A = K[fd][:, fd].tocsc()
            tf = time.perf_counter()
            lu = spla.splu(A, permc_spec=permc, diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            info["t_fact"] += time.perf_counter() - tf
            info["nnz_LU"] = max(info["nnz_LU"], lu.L.nnz + lu.U.nnz)
            for l, up in zip(st["loads"], st["ups"]):
                ts = time.perf_counter()
                b = -L3.k_apply(up, lat.nbr, lat.ndir, Kn, Tv)
                u = up.copy().reshape(-1)
                u[fd] = lu.solve(b.reshape(-1)[fd])
                us[l] = u.reshape(-1, 3)
                info["t_solves"] += time.perf_counter() - ts
            del lu
        return time.perf_counter() - t0, [0] * 9, us, info

    # ------------------------------------------------------------------------------------------------ outputs
    def stresses(self, Kn, Tv, us):
        """computeStress of the 9 solutions: (9, N, 6) (part of the end-to-end time)."""
        return [self.lat.stress(u, Kn, Tv) for u in us]

    def block_mean(self, sig):
        nc3 = self.nc ** 3
        out = np.empty((nc3, 6))
        for k in range(6):
            out[:, k] = np.bincount(self.cid, weights=sig[:, k], minlength=nc3)
        return out / S ** 3

    def field_errors(self, sigs, ref):
        """I8 stress-field error per load case (relative L2 over free cells); ref (9, Nc, 6)."""
        e = []
        for l, sg in enumerate(sigs):
            if not self.coarse:
                sg = self.block_mean(sg)
            m = self.freecell[l]
            r = ref[l].astype(np.float64)
            e.append(float(np.linalg.norm((sg - r)[m]) / np.linalg.norm(r[m])))
        return e


# ---------------------------------------------------------------------------------------------------------------------
def material(m, i):
    ph, c, nu_i = m["phase"][i], float(m["contrast"][i]), float(m["nu_i"][i])
    E = np.where(ph == 1, c, 1.0)
    nu = np.where(ph == 1, nu_i, 0.2)
    return G.iso_C(E, nu)


def fine_params(lat, Cv):
    return G.bond_params_two_end_mean(lat, G.sc_params_from_C(Cv, 0.5))


def voigt_params(latc, Cv, n):
    return G.bond_params_from_cells(latc, G.cell_params_from_C66(G.voigt_cells(Cv, n, S), S / 2.0))


def warmup():
    """compile every numba kernel on a tiny problem (not timed)."""
    P = Problem(12, False)
    m = np.ones(12 ** 3, np.int8)
    Cv = G.iso_C(np.where(m == 1, 2.0, 1.0), np.full(12 ** 3, 0.2))
    Kn, Tv = fine_params(P.lat, Cv)
    for sol in ("jac", "amg", "lu"):
        _, _, us, _ = P.solve(sol, Kn, Tv, tol=1e-6)
        P.stresses(Kn, Tv, us)
    Pc = Problem(24, True)
    Kn, Tv = voigt_params(Pc.lat, G.iso_C(np.ones(24 ** 3), np.full(24 ** 3, 0.2)), 24)
    for sol in ("jac", "amg"):
        Pc.solve(sol, Kn, Tv, tol=1e-6)
    import patch3d
    patch3d.patch_C(G.iso_C(np.ones(216), np.full(216, 0.2)), 6, S)
