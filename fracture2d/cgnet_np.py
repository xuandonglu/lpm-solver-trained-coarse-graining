"""NumPy forward pass of cgnet.CGNet (deployment in the numerical environment without torch).
Weights: models/cgnet_sS[tag].npz written by train_cg.py (export_np).  Matches torch to ~1e-6 (float32 weights,
float64 arithmetic here)."""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cgnet_geom import D4, bond_gather_index, pixel_features   # noqa: F401  (pure numpy helpers)


def _erf(x):
    # vectorised erf via math.erf (exact GELU as torch.nn.GELU())
    return np.vectorize(math.erf, otypes=[np.float64])(x)


try:
    from scipy.special import erf as _sp_erf
    _erf = _sp_erf  # noqa: F811
except Exception:  # pragma: no cover
    pass


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def conv3x3(x, w, b):
    """x (C, H, W), w (Co, C, 3, 3), padding 1, stride 1 -> (Co, H, W)."""
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))           # (C, H, W, 3, 3)
    cols = win.transpose(1, 2, 0, 3, 4).reshape(H * W, C * 9)    # (HW, C*9)
    out = cols @ w.reshape(w.shape[0], -1).T + b                 # (HW, Co)
    return out.T.reshape(w.shape[0], H, W)


def conv1x1(x, w, b):
    C, H, W = x.shape
    return (w.reshape(w.shape[0], C) @ x.reshape(C, -1) + b[:, None]).reshape(w.shape[0], H, W)


def apply_g(x, g):
    t, fr, fc = D4[g]
    if t:
        x = np.swapaxes(x, -1, -2)
    if fr:
        x = x[..., ::-1, :]
    if fc:
        x = x[..., :, ::-1]
    return np.ascontiguousarray(x)


def pixel_unshuffle(x, s):
    C, H, W = x.shape
    return x.reshape(C, H // s, s, W // s, s).transpose(0, 2, 4, 1, 3).reshape(C * s * s, H // s, W // s)


def mlp(z, ws):
    for i, (w, b) in enumerate(ws):
        z = z @ w.T + b
        if i < len(ws) - 1:
            z = gelu(z)
    return z


class CGNetNP:
    def __init__(self, path):
        d = np.load(path)
        self.s = int(d["s"])
        self.trunk = [(d["trunk.%d.weight" % i].astype(np.float64), d["trunk.%d.bias" % i].astype(np.float64))
                      for i in (0, 2, 4, 6)]
        self.heads = []
        for h in ("head0", "head1"):
            self.heads.append([(d["%s.%d.weight" % (h, i)].astype(np.float64), d["%s.%d.bias" % (h, i)].astype(np.float64))
                               for i in (0, 2, 4)])

    def cell_map(self, X, g):
        x = apply_g(X, g)
        for i, (w, b) in enumerate(self.trunk):
            if i < 3:
                x = gelu(conv3x3(x, w, b))
            else:
                x = conv1x1(x, w, b)
        return pixel_unshuffle(x, self.s)

    def __call__(self, X, gi0, gi1):
        """X (4, Hp, Wp) -> o0 (n0, nout), o1 (n1, nout)."""
        maps = [self.cell_map(X, g) for g in range(8)]
        res = []
        for (G, R0, C0), kh, kw, head in ((gi0, 3, 4, self.heads[0]), (gi1, 4, 4, self.heads[1])):
            n = G.shape[0]
            acc = None
            for g in range(8):
                bi, ji = np.nonzero(G == g)
                if len(bi) == 0:
                    continue
                P = maps[g]                                       # (D, Hc, Wc)
                win = sliding_window_view(P, (kh, kw), axis=(1, 2))   # (D, Hc-kh+1, Wc-kw+1, kh, kw)
                r0, c0 = R0[bi, ji], C0[bi, ji]
                feats = win[:, r0, c0].transpose(1, 0, 2, 3).reshape(len(bi), -1)   # (nsel, D*kh*kw) order (D, kh, kw)
                o = mlp(feats, head)
                if acc is None:
                    acc = np.zeros((n, o.shape[1]))
                np.add.at(acc, bi, o)
            res.append(acc / 4.0)
        return res[0], res[1]
