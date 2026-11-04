"""Coarse bond parameters for deployment (numerical environment, numpy only): Voigt, patch, ML elastic; naive / patch / ML critical
stretch.  Every ML number here comes from the numpy forward pass (cgnet_np) - no torch at deployment."""
import time
import numpy as np
import cg
import windows as WIN
from cgnet_geom import bond_gather_index, pixel_features
from cgnet_np import CGNetNP

_NETS = {}


def net(path):
    if path not in _NETS:
        _NETS[path] = CGNetNP(path)
    return _NETS[path]


def forward_bonds(nc):
    b, shell = WIN.bond_list(nc, nc)
    return b[shell == 0], b[shell == 1]


def to_slots(N, nc, b0, v0, b1, v1):
    out = np.zeros((N, 8))
    for bb, v in ((b0, v0), (b1, v1)):
        q, kr = WIN.neighbor_of(bb, nc)
        out[bb[:, 0], bb[:, 1]] = v
        out[q, kr] = v
    return out


def elastic_from_cells(cp, nc):
    """per-cell (Nc, 5) -> Kn, Tv (Nc, 8) (two-end mean)."""
    lat = cg.CoarseModel(nc, nc, 1).lat
    return cg.bond_params_from_cells(lat, cp)


def ml_elastic(Cpix, n, s, model_path, timing=None):
    """Kn, Tv (Nc, 8) from the learned network; timing dict gets 'features', 'net', 'assemble'."""
    t0 = time.perf_counter()
    nc = n // s
    b0, b1 = forward_bonds(nc)
    X = pixel_features(Cpix, n, n, s).astype(np.float64)
    gi0 = bond_gather_index(b0, nc, nc)
    gi1 = bond_gather_index(b1, nc, nc)
    cpv = cg.cell_params_from_C6(cg.voigt_cells(Cpix, n, n, s))
    kv0, tv0 = WIN.base_from_cells(cpv, b0, nc)
    kv1, tv1 = WIN.base_from_cells(cpv, b1, nc)
    kn1 = (Cpix[:, 0] - Cpix[:, 1]) / 2
    cellK = kn1.reshape(nc, s, nc, s).mean(axis=(1, 3)).ravel()
    q0, _ = WIN.neighbor_of(b0, nc)
    q1, _ = WIN.neighbor_of(b1, nc)
    K00 = 0.5 * (cellK[b0[:, 0]] + cellK[q0])
    K01 = 0.5 * (cellK[b1[:, 0]] + cellK[q1])
    t1 = time.perf_counter()
    o0, o1 = net(model_path)(X, gi0, gi1)
    t2 = time.perf_counter()
    N = nc * nc
    Kn = to_slots(N, nc, b0, kv0 * np.exp(o0[:, 0]), b1, kv1 * np.exp(o1[:, 0]))
    Tv = to_slots(N, nc, b0, tv0 + K00 * o0[:, 1], b1, tv1 + K01 * o1[:, 1])
    t3 = time.perf_counter()
    if timing is not None:
        timing.update(features=t1 - t0, net=t2 - t1, assemble=t3 - t2)
    return Kn, Tv


def ml_crit(Cpix, n, s, model_path, naive0, naive1, timing=None):
    t0 = time.perf_counter()
    nc = n // s
    b0, b1 = forward_bonds(nc)
    X = pixel_features(Cpix, n, n, s).astype(np.float64)
    o0, o1 = net(model_path)(X, bond_gather_index(b0, nc, nc), bond_gather_index(b1, nc, nc))
    c = crit_slots(nc, naive0 * np.exp(o0[:, 0]), naive1 * np.exp(o1[:, 0]))
    if timing is not None:
        timing["crit_net"] = time.perf_counter() - t0
    return c


def crit_slots(nc, v0, v1):
    b0, b1 = forward_bonds(nc)
    c = to_slots(nc * nc, nc, b0, v0, b1, v1)
    lat = cg.CoarseModel(nc, nc, 1).lat
    return np.where(lat.valid, c, np.inf)
