"""Reference arrays from lin3d (numpy / scipy / numba environment) for check_torch3d.py: n = 12 two-phase box, s = 2 and 3, 9 load cases."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import lin3d as L3
import geom3d as G
import micro3d

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n = 12
rng = np.random.default_rng(5)
ph, E, nu, info = micro3d.sample(n, rng)
lat = L3.Lattice3D(n, n, n, 1.0)
C = G.iso_C(E, nu)
Kn, Tv = G.bond_params_two_end_mean(lat, G.sc_params_from_C(C, 0.5))
u = rng.standard_normal((lat.N, 3))
out = dict(Kn=Kn, Tv=Tv, u=u, Ku=lat.apply(u, Kn, Tv), st=lat.stress(u, Kn, Tv), U=lat.energy_per_particle(u, Kn, Tv),
           D=lat.block_diag(Kn, Tv))
K = lat.stiffness(Kn, Tv)
for s in (2, 3):
    nc = n // s
    IJK = lat.ijk // s
    cid = G.block_index(lat.ijk, s, (nc,) * 3)
    for ld in G.LOADS:
        fixed, up, mov = G.prescribed(ld, lat.x, IJK, (nc,) * 3, s)
        uu, _ = L3.solve_direct(K, fixed, up)
        st = lat.stress(uu, Kn, Tv)
        out["u_%d_%s" % (s, ld)] = np.stack([np.bincount(cid, uu[:, k]) for k in range(3)], 1) / s ** 3
        out["s_%d_%s" % (s, ld)] = np.stack([np.bincount(cid, st[:, k]) for k in range(6)], 1) / s ** 3
        out["w_%d_%s" % (s, ld)] = np.bincount(cid, lat.energy_per_particle(uu, Kn, Tv)) / s ** 3
        P = lat.apply(uu, Kn, Tv)
        out["F_%d_%s" % (s, ld)] = P[mov, G.GDIR[ld]].sum() if mov is not None else 0.0
np.savez("dev/check_ref.npz", **out)
print("saved, phi", ph.mean(), info["kind"])
