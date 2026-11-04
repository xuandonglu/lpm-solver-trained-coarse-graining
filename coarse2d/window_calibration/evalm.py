"""Unified evaluation of every rung (numpy / scipy only; same coarse solver lin2d and the same error measures as I2's
eval_elastic.py, plus a fixed-resolution measure at R = 4).

Input: per-sample coarse bond parameters Kn, Tv (Nc, 8).  The coarse LPM (cg.CoarseModel, lin2d + SuperLU) is solved for
the 6 I2 load cases and compared with the block-averaged fine LPM reference stored by I2 (data/<split>_n48.npz).

I2 measures (cg.metrics): per load case G (plate: grip reaction; RVE: interior mean stress), u, sig (relative L2 over
free cells), sig_if, w_if (interface cells), w99.  Summaries = mean |.| over samples and the 3 RVE / 3 plate loads.
Fixed resolution R = 4: coarse s-block stress / energy density and the fine reference of the
same boundary-value problem are averaged to R x R blocks (s | R, i.e. s = 2, 4); errors over the free R-blocks of the R
problem (1 R-block grip / ring layer) and, for the energy density, over the free interface R-blocks; mean over the 6 loads.
"""
import numpy as np
import common  # noqa
import cg

LOADS = cg.LOADS
KEYS = ["G", "u", "sig", "sig_if", "w_if", "w99", "sig_in", "sig_bd"]
R_FIX = 4


class Split:
    def __init__(self, split):
        self.name = split
        self.d = dict(np.load(common.split_file(split), allow_pickle=True))
        self.n = int(self.d["n"])
        self.S = len(self.d["phi"])

    def ref(self, i, s):
        d, n = self.d, self.n

        class R:
            pass
        r = R()
        r.ncx = r.ncy = n // s
        r.s, r.h, r.loads = s, 1.0, LOADS
        r.res = {}
        for ld in LOADS:
            F = d["F_%d_%s" % (s, ld)][i]
            r.res[ld] = dict(u=d["u_%d_%s" % (s, ld)][i], sig=d["sig_%d_%s" % (s, ld)][i], w=d["w_%d_%s" % (s, ld)][i],
                             F=None if np.isnan(F).all() else F, W=d["W_%d_%s" % (s, ld)][i])
        r.iface = d["iface_%d" % s][i]
        return r

    def cells(self, base, i, s):
        return self.d["%s_%d" % (base, s)][i]


def _agg(a, nc, k, op="mean"):
    ncR = nc // k
    x = a.reshape(ncR, k, ncR, k, *a.shape[1:])
    x = x.any(axis=(1, 3)) if op == "any" else x.mean(axis=(1, 3))
    return x.reshape(ncR * ncR, *a.shape[1:])


_FREE_R = {}


def free_R(n, R, ld):
    key = (n, R, ld)
    if key not in _FREE_R:
        ncR = n // R
        ii, jj = np.meshgrid(np.arange(ncR), np.arange(ncR))
        IJ = np.stack([ii.ravel(), jj.ravel()], 1)
        fixed, _, _ = cg.prescribed(ld, np.zeros((ncR * ncR, 2)), IJ, ncR, ncR, R, 1.0)
        _FREE_R[key] = ~fixed
    return _FREE_R[key]


def metrics_R(ref, res, n, s, R=R_FIX):
    """per load: stress / interface energy-density errors at output resolution R (needs s | R)."""
    nc = n // s
    k = R // s
    ifR = _agg(ref.iface.astype(bool), nc, k, "any")
    out = {}
    for ld in LOADS:
        fr = free_R(n, R, ld)
        sf, sc = _agg(ref.res[ld]["sig"], nc, k), _agg(res[ld]["sig"], nc, k)
        wf, wc = _agg(ref.res[ld]["w"], nc, k), _agg(res[ld]["w"], nc, k)
        d = dict(sigR=float(np.linalg.norm(sc[fr] - sf[fr]) / np.linalg.norm(sf[fr])))
        m = fr & ifR
        if m.any():
            d["wR_if"] = float(np.linalg.norm(wc[m] - wf[m]) / np.linalg.norm(wf[m]))
        out[ld] = d
    return out


_CM = {}


def coarse_model(n, s):
    if (n, s) not in _CM:
        _CM[(n, s)] = cg.CoarseModel(n // s, n // s, s)
    return _CM[(n, s)]


_INNER = {}


def inner_cells(nc, depth=2):
    """cells at least `depth` cells away from the specimen boundary (diagnostic: interior vs boundary-layer error)."""
    if (nc, depth) not in _INNER:
        ii, jj = np.meshgrid(np.arange(nc), np.arange(nc))
        d = np.minimum.reduce([ii, jj, nc - 1 - ii, nc - 1 - jj]).ravel()
        _INNER[(nc, depth)] = d >= depth
    return _INNER[(nc, depth)]


def eval_sample(sp, i, s, Kn, Tv):
    ref = sp.ref(i, s)
    res = coarse_model(sp.n, s).run(Kn, Tv)
    m = cg.metrics(ref, res)
    nc = sp.n // s
    inn = inner_cells(nc)
    for ld in LOADS:
        fixed, _, _ = cg.prescribed(ld, np.zeros((nc * nc, 2)), cg.b_ij(ref), nc, nc, s, 1.0)
        for key, msk in (("sig_in", inn & ~fixed), ("sig_bd", ~inn & ~fixed)):
            a, b = ref.res[ld]["sig"][msk], res[ld]["sig"][msk]
            m[ld][key] = float(np.linalg.norm(b - a) / np.linalg.norm(a))
    if R_FIX % s == 0:
        mR = metrics_R(ref, res, sp.n, s)
        for ld in LOADS:
            m[ld].update(mR[ld])
    return m


def summ(ms):
    out = {}
    for grp, loads in (("ring", LOADS[:3]), ("plate", LOADS[3:])):
        for k in KEYS:
            v = [abs(m[ld][k]) for m in ms for ld in loads if k in m[ld]]
            out["%s_%s" % (grp, k)] = float(np.mean(v))
        out["%s_G_signed" % grp] = float(np.mean([m[ld]["G"] for m in ms for ld in loads]))
    if "sigR" in ms[0][LOADS[0]]:
        out["R4_sig"] = float(np.mean([m[ld]["sigR"] for m in ms for ld in LOADS]))
        out["R4_w_if"] = float(np.mean([m[ld]["wR_if"] for m in ms for ld in LOADS if "wR_if" in m[ld]]))
    return out


def per_sample(ms, key, loads):
    """per-sample mean |error| of one key over a load group (for paired comparisons)."""
    return np.array([np.mean([abs(m[ld][key]) for ld in loads if key in m[ld]]) for m in ms])
