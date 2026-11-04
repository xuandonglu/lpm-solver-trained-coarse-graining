"""Coarse bond parameters for deployment on rectangular specimens (numpy only, numerical environment): Voigt / patch / ML elastic,
naive / classical / ML critical stretch, and the two-parameter laws.  Generalises I2's coarse_params.py to nx != ny.
Everything is returned on the undirected forward bonds of soft.undirected(coarse lattice)."""
import time
import numpy as np
import cg
import windows as WIN
import soft
from cgnet_geom import bond_gather_index, pixel_features
from cgnet_np import CGNetNP

_NETS = {}


def net(path):
    if path not in _NETS:
        _NETS[path] = CGNetNP(path)
    return _NETS[path]


def coarse_lat(nx, ny, s):
    return cg.CoarseModel(nx // s, ny // s, s).lat


def fwd(ncx, ncy):
    b, shell = WIN.bond_list(ncx, ncy)
    return b[shell == 0], b[shell == 1]


def to_slots(N, ncx, b0, v0, b1, v1):
    out = np.zeros((N, 8))
    for bb, v in ((b0, v0), (b1, v1)):
        q, kr = WIN.neighbor_of(bb, ncx)
        out[bb[:, 0], bb[:, 1]] = v
        out[q, kr] = v
    return out


def to_und(latc, ncx, ncy, v0, v1):
    """per forward-bond values (windows order) -> per undirected bond of soft.undirected(latc)."""
    b0, b1 = fwd(ncx, ncy)
    sl = to_slots(latc.N, ncx, b0, v0, b1, v1)
    P, K, Q, KR = soft.undirected(latc)
    return sl[P, K]


def cells_elastic(latc, cp):
    return cg.bond_params_from_cells(latc, cp)


def net_out(Cpix, nx, ny, s, path, timing=None, key="net"):
    t0 = time.perf_counter()
    ncx, ncy = nx // s, ny // s
    b0, b1 = fwd(ncx, ncy)
    X = pixel_features(Cpix, nx, ny, s).astype(np.float64)
    o0, o1 = net(path)(X, bond_gather_index(b0, ncx, ncy), bond_gather_index(b1, ncx, ncy))
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return o0, o1


def ml_elastic(Cpix, nx, ny, s, path, timing=None):
    """Kn, Tv (Nc, 8) from the I2 elastic network (residual on Voigt)."""
    t0 = time.perf_counter()
    ncx, ncy = nx // s, ny // s
    b0, b1 = fwd(ncx, ncy)
    cpv = cg.cell_params_from_C6(cg.voigt_cells(Cpix, nx, ny, s))
    kv0, tv0 = WIN.base_from_cells(cpv, b0, ncx)
    kv1, tv1 = WIN.base_from_cells(cpv, b1, ncx)
    kn1 = (Cpix[:, 0] - Cpix[:, 1]) / 2
    cellK = kn1.reshape(ncy, s, ncx, s).mean(axis=(1, 3)).ravel()
    q0, _ = WIN.neighbor_of(b0, ncx)
    q1, _ = WIN.neighbor_of(b1, ncx)
    K00 = 0.5 * (cellK[b0[:, 0]] + cellK[q0])
    K01 = 0.5 * (cellK[b1[:, 0]] + cellK[q1])
    if timing is not None:
        timing["features"] = timing.get("features", 0.0) + time.perf_counter() - t0
    o0, o1 = net_out(Cpix, nx, ny, s, path, timing, "net_elastic")
    N = ncx * ncy
    Kn = to_slots(N, ncx, b0, kv0 * np.exp(o0[:, 0]), b1, kv1 * np.exp(o1[:, 0]))
    Tv = to_slots(N, ncx, b0, tv0 + K00 * o0[:, 1], b1, tv1 + K01 * o1[:, 1])
    return Kn, Tv


def naive_crit(Cpix, phase, nx, ny, s):
    """weakest-link critical stretch per forward bond (export_frac naive): (v0, v1)."""
    import lin2d
    import export_frac as EF
    import spec
    lat = lin2d.Lattice2D("square", nx, ny, 1.0)
    crit = spec.fine_model(Cpix, phase, nx, ny)[3]
    ncx = nx // s
    b, shell = WIN.bond_list(ncx, ny // s)
    m = EF.cell_max(lat, s, ncx, np.where(lat.valid, 1.0 / crit, 0.0))
    v = 1.0 / EF.bond_from_cells(m, b, ncx)
    return v[shell == 0], v[shell == 1]


def patch_crit(Cpix, phase, nx, ny, s):
    import export_frac as EF
    ncx = nx // s
    b, shell = WIN.bond_list(ncx, ny // s)
    pm = EF.patch_strength_cells(Cpix, phase, nx, ny, s)
    v = 1.0 / np.maximum(EF.bond_from_cells(pm, b, ncx), 1e-30)
    return v[shell == 0], v[shell == 1]
