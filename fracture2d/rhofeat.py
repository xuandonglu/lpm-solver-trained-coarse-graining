"""Torch-free helpers shared by train_rho.py (GPU) and the numpy deployment: 6-channel features, rho target."""
import numpy as np


def iface_map(ph2d):
    b = np.zeros_like(ph2d, bool)
    d = ph2d[:, 1:] != ph2d[:, :-1]
    b[:, 1:] |= d; b[:, :-1] |= d
    d = ph2d[1:, :] != ph2d[:-1, :]
    b[1:, :] |= d; b[:-1, :] |= d
    return b


def extra_channels(phase, nx, ny, s):
    """(2, ny + 4s, nx + 4s): phase, interface flag (zero outside the specimen)."""
    ph = phase.reshape(ny, nx).astype(np.float32)
    pad = 2 * s
    X = np.zeros((2, ny + 2 * pad, nx + 2 * pad), np.float32)
    X[0, pad:pad + ny, pad:pad + nx] = ph
    X[1, pad:pad + ny, pad:pad + nx] = iface_map(phase.reshape(ny, nx)).astype(np.float32)
    return X


def features6(Cpix, phase, nx, ny, s):
    from cgnet_geom import pixel_features
    return np.concatenate([pixel_features(Cpix, nx, ny, s), extra_channels(phase, nx, ny, s)], 0)


def rho_target(pt, s, sh, idx=None):
    W = pt["W_%d_s%d" % (sh, s)]
    k0 = pt["k0_%d_s%d" % (sh, s)]
    ei = pt["eps_i_%d_s%d" % (sh, s)]
    r = np.maximum(1.0, W / (0.5 * k0 * ei ** 2))
    return r if idx is None else r[idx]
