"""Coarse bond parameters (numpy only) for rectangular nx x ny specimens: Voigt, ML elastic (I2 cgnet), weakest-link
(naive) and ML critical bond stretch (I2 fracnet).  Copied from coarse2d/coarse_params.py and generalised from
square n x n to nx x ny (the I2 networks are fully convolutional + local bond windows, so they apply unchanged)."""
import time
import numpy as np
import cg
import windows as WIN
from cgnet_geom import bond_gather_index, pixel_features
from cgnet_np import CGNetNP

_NETS = {}
CLS = np.array([0, 0, 1, 1, 2, 2, 3, 3])          # slot -> direction class (x, y, diag+, diag-)


PRECISION = np.float32      # default; set coarse_params.PRECISION = np.float64 to reproduce


def net(path):
    key = (path, PRECISION)
    if key not in _NETS:
        _NETS[key] = CGNetNP(path, PRECISION)
    return _NETS[key]


def forward_bonds(ncx, ncy):
    b, shell = WIN.bond_list(ncx, ncy)
    return b[shell == 0], b[shell == 1]


def to_slots(N, ncx, b0, v0, b1, v1):
    out = np.zeros((N, 8))
    for bb, v in ((b0, v0), (b1, v1)):
        q, kr = WIN.neighbor_of(bb, ncx)
        out[bb[:, 0], bb[:, 1]] = v
        out[q, kr] = v
    return out


def voigt_elastic(Cpix, nx, ny, s, lat_c):
    cp = cg.cell_params_from_C6(cg.voigt_cells(Cpix, nx, ny, s))
    return cg.bond_params_from_cells(lat_c, cp)


def ml_elastic(Cpix, nx, ny, s, model_path, timing=None):
    """Kn, Tv (Nc, 8) from the learned I2 network (numpy forward)."""
    t0 = time.perf_counter()
    ncx, ncy = nx // s, ny // s
    b0, b1 = forward_bonds(ncx, ncy)
    X = pixel_features(Cpix, nx, ny, s).astype(np.float64)
    gi0 = bond_gather_index(b0, ncx, ncy)
    gi1 = bond_gather_index(b1, ncx, ncy)
    cpv = cg.cell_params_from_C6(cg.voigt_cells(Cpix, nx, ny, s))
    kv0, tv0 = WIN.base_from_cells(cpv, b0, ncx)
    kv1, tv1 = WIN.base_from_cells(cpv, b1, ncx)
    kn1 = (Cpix[:, 0] - Cpix[:, 1]) / 2
    cellK = kn1.reshape(ncy, s, ncx, s).mean(axis=(1, 3)).ravel()
    q0, _ = WIN.neighbor_of(b0, ncx)
    q1, _ = WIN.neighbor_of(b1, ncx)
    K00 = 0.5 * (cellK[b0[:, 0]] + cellK[q0])
    K01 = 0.5 * (cellK[b1[:, 0]] + cellK[q1])
    o0, o1 = net(model_path)(X, gi0, gi1)
    N = ncx * ncy
    Kn = to_slots(N, ncx, b0, kv0 * np.exp(o0[:, 0]), b1, kv1 * np.exp(o1[:, 0]))
    Tv = to_slots(N, ncx, b0, tv0 + K00 * o0[:, 1], b1, tv1 + K01 * o1[:, 1])
    if timing is not None:
        timing["ml_elastic"] = time.perf_counter() - t0
    return Kn, Tv


def naive_crit_bonds(lat_f, crit_f, s, ncx, ncy):
    """Weakest-link coarse critical stretch on forward bonds: min crit of the fine bonds of the same direction class
    touching either end cell (export_frac.py 'naive')."""
    b0, b1 = forward_bonds(ncx, ncy)
    q = np.where(lat_f.valid, lat_f.nbr, 0)
    I, J = lat_f.ij[:, 0] // s, lat_f.ij[:, 1] // s
    cp = (J * ncx + I)[:, None].repeat(8, 1)
    cq = (J * ncx + I)[q]
    d = np.broadcast_to(CLS[None, :], crit_f.shape)
    inv = np.where(lat_f.valid & np.isfinite(crit_f), 1.0 / np.where(np.isfinite(crit_f), crit_f, 1.0), 0.0)
    ok = lat_f.valid
    m = np.zeros((ncx * ncy, 4))
    np.maximum.at(m, (cp[ok], d[ok]), inv[ok])
    np.maximum.at(m, (cq[ok], d[ok]), inv[ok])
    out = []
    for b in (b0, b1):
        qq, _ = WIN.neighbor_of(b, ncx)
        dd = CLS[b[:, 1]]
        v = np.maximum(m[b[:, 0], dd], m[qq, dd])
        out.append(np.where(v > 0, 1.0 / np.maximum(v, 1e-30), np.inf))
    return out[0], out[1]


def ml_crit(Cpix, nx, ny, s, model_path, naive0, naive1, timing=None):
    """I2 fracnet: crit_C = crit_naive * exp(o) (Nc, 8 slots; inf where invalid)."""
    t0 = time.perf_counter()
    ncx, ncy = nx // s, ny // s
    b0, b1 = forward_bonds(ncx, ncy)
    X = pixel_features(Cpix, nx, ny, s).astype(np.float64)
    o0, o1 = net(model_path)(X, bond_gather_index(b0, ncx, ncy), bond_gather_index(b1, ncx, ncy))
    c = crit_slots(ncx, ncy, naive0 * np.exp(o0[:, 0]), naive1 * np.exp(o1[:, 0]))
    if timing is not None:
        timing["ml_crit"] = time.perf_counter() - t0
    return c


def crit_slots(ncx, ncy, v0, v1):
    b0, b1 = forward_bonds(ncx, ncy)
    c = to_slots(ncx * ncy, ncx, b0, v0, b1, v1)
    lat = cg.CoarseModel(ncx, ncy, 1).lat
    return np.where(lat.valid & (c > 0), c, np.inf)
