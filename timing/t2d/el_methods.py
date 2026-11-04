"""Coarse bond-parameter builders of the 2D benchmark (all timed per call):
  voigt   per-cell volume average of C (cg.voigt_cells + generalised calcKnTv), bonds = two-end mean
  patch   local patch homogenisation with precomputed geometry (patch_g.PatchFastG)
  nn      the 2D bond networks (D4-equivariant CGNet), numpy forward on one CPU thread (float32 trunk);
          ../../coarse2d/models/cgnet_sS.npz
The network inference of the timing table runs in the torch worker (infer2d.py); its numpy post-processing is nn_post.
"""
import os
import time
import numpy as np
import cg
import el_prob as EP

MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "coarse2d", "models"))


def cell_C6_voigt(prob, s):
    n = prob["n"]
    return cg.voigt_cells(prob["Cpix"], n, n, s)


def bonds_from_C6(prob, s, C6c):
    lat_c = EP.coarse_lattice(prob["kind"], prob["n"], s)
    return cg.bond_params_from_cells(lat_c, cg.cell_params_from_C6(C6c))


def build_voigt(prob, s, timing=None):
    t0 = time.perf_counter()
    Kn, Tv = bonds_from_C6(prob, s, cell_C6_voigt(prob, s))
    if timing is not None:
        timing["build"] = time.perf_counter() - t0
    return Kn, Tv


def build_patch(prob, s, timing=None, backend=None):
    from patch_g import patch_obj
    pf = patch_obj(prob["kind"], prob["n"], s, backend)
    t0 = time.perf_counter()
    tm = {}
    C6c = pf.solve(prob, timing=tm)
    Kn, Tv = bonds_from_C6(prob, s, C6c)
    if timing is not None:
        timing.update({"patch_" + k: v for k, v in tm.items()})
        timing["build"] = time.perf_counter() - t0
        timing["patch_init"] = pf.t_init
        timing["patch_backend"] = pf.backend
    return Kn, Tv


def build_nn(prob, s, timing=None):
    t0 = time.perf_counter()
    n = prob["n"]
    import coarse_params as CP
    Kn, Tv = CP.ml_elastic(prob["Cpix"], n, n, s, "%s/cgnet_s%d.npz" % (MODEL_DIR, s))
    if timing is not None:
        timing["build"] = time.perf_counter() - t0
    return Kn, Tv


def nn_post(prob, s, raw, timing=None):
    """numpy post-processing of raw network outputs (same formulas as build_nn): raw = (o0, o1)."""
    t0 = time.perf_counter()
    n = prob["n"]
    nc = n // s
    import coarse_params as CP
    import windows as WIN
    Cpix = prob["Cpix"]
    b0, b1 = CP.forward_bonds(nc, nc)
    q0, _ = WIN.neighbor_of(b0, nc)
    q1, _ = WIN.neighbor_of(b1, nc)
    cpv = cg.cell_params_from_C6(cg.voigt_cells(Cpix, n, n, s))
    kv0, tv0 = WIN.base_from_cells(cpv, b0, nc)
    kv1, tv1 = WIN.base_from_cells(cpv, b1, nc)
    kn1 = (Cpix[:, 0] - Cpix[:, 1]) / 2
    cellK = kn1.reshape(nc, s, nc, s).mean(axis=(1, 3)).ravel()
    K00 = 0.5 * (cellK[b0[:, 0]] + cellK[q0])
    K01 = 0.5 * (cellK[b1[:, 0]] + cellK[q1])
    o0, o1 = raw
    N = nc * nc
    Kn = CP.to_slots(N, nc, b0, kv0 * np.exp(o0[:, 0]), b1, kv1 * np.exp(o1[:, 0]))
    Tv = CP.to_slots(N, nc, b0, tv0 + K00 * o0[:, 1], b1, tv1 + K01 * o1[:, 1])
    if timing is not None:
        timing["post"] = time.perf_counter() - t0
    return Kn, Tv


BUILDERS = dict(voigt=build_voigt, patch=build_patch, nn=build_nn)
