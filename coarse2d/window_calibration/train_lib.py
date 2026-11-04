"""Training / evaluation pieces copied from coarse2d/train_cg.py (torch environment, GPU); the prepared tensors
data/prep2_<split>_s<S>.pt of I2 are only READ here (never rewritten)."""
import os
import numpy as np
import torch
import common
import torch_lpm as T
import windows as WIN

DEV = os.environ.get("I2C_DEV", "cuda")
LOADS = T.LOADS


def load_prep(split, s):
    fn = os.path.join(common.I2DATA, "prep2_%s_s%d.pt" % (split, s))
    return torch.load(fn, weights_only=False)


class Scatter:
    def __init__(self, P):
        nc = P["nc"]
        self.N = nc * nc
        res = {}
        for key in ("b0", "b1"):
            b = P[key]
            q, kr = WIN.neighbor_of(b, nc)
            res[key] = [torch.as_tensor(x, device=DEV) for x in (b[:, 0], b[:, 1], q, kr)]
        self.idx = res

    def __call__(self, kn0, tv0, kn1, tv1):
        B = kn0.shape[0]
        Kn = torch.zeros(B, self.N, 8, device=DEV, dtype=torch.float64)
        Tv = torch.zeros_like(Kn)
        for key, kn, tv in (("b0", kn0, tv0), ("b1", kn1, tv1)):
            p, k, q, kr = self.idx[key]
            Kn[:, p, k] = kn
            Kn[:, q, kr] = kn
            Tv[:, p, k] = tv
            Tv[:, q, kr] = tv
        return Kn, Tv


def batch_ref(P, idx):
    r = {}
    for ld in LOADS:
        a = P["ref"][ld]
        r[ld] = dict(u=a["u"][idx].to(DEV), sig=a["sig"][idx].to(DEV), w=a["w"][idx].to(DEV),
                     F=None if a["F"] is None else a["F"][idx].to(DEV))
    return r


def model_params(model, P, idx, base="voigt"):
    """Kn = Kn_base exp(o1), Tv = Tv_base + K0 o2 on the forward bonds (as train_cg.ml_params)."""
    X = P["X"][idx].to(DEV)
    gi0 = [x.to(DEV) for x in P["gi0"]]
    gi1 = [x.to(DEV) for x in P["gi1"]]
    o0, o1 = model(X, gi0, gi1)
    o0, o1 = o0.double(), o1.double()
    kv0, tv0, kv1, tv1 = [x[idx].to(DEV) for x in P[base]]
    K00, K01 = P["K00"][idx].to(DEV), P["K01"][idx].to(DEV)
    return (kv0 * torch.exp(o0[..., 0]), tv0 + K00 * o0[..., 1], kv1 * torch.exp(o1[..., 0]), tv1 + K01 * o1[..., 1])


def eval_metrics(coarse, out, ref, iface):
    res = {}
    for ld in LOADS:
        fc = coarse.bc[ld]["freecell"]
        a, b = ref[ld], out[ld]
        e_u = ((b["u"] - a["u"])[:, fc] ** 2).sum((1, 2)).sqrt() / (a["u"][:, fc] ** 2).sum((1, 2)).sqrt()
        e_s = ((b["sig"] - a["sig"])[:, fc] ** 2).sum((1, 2)).sqrt() / (a["sig"][:, fc] ** 2).sum((1, 2)).sqrt()
        fi = (fc[None, :] & iface).double()
        e_wi = (((b["w"] - a["w"]) ** 2) * fi).sum(1).sqrt() / ((a["w"] ** 2) * fi).sum(1).sqrt()
        if b["F"] is not None:
            k = 1 if ld == "tb_ty" else 0
            e_g = (b["F"][:, k] - a["F"][:, k]) / a["F"][:, k]
        else:
            k = {"ring_xx": 0, "ring_yy": 1, "ring_xy": 2}[ld]
            sa, sb = a["sig"][:, fc, k].mean(1), b["sig"][:, fc, k].mean(1)
            e_g = (sb - sa) / sa
        res[ld] = dict(G=e_g, u=e_u, sig=e_s, w_if=e_wi)
    return {ld: {k: v.detach().cpu().numpy() for k, v in d.items()} for ld, d in res.items()}


def summarize(mlist):
    out = {}
    for grp, loads in (("ring", LOADS[:3]), ("plate", LOADS[3:])):
        for k in ("G", "u", "sig", "w_if"):
            v = np.concatenate([np.abs(m[ld][k]) for m in mlist for ld in loads])
            out["%s_%s" % (grp, k)] = float(np.nanmean(v))
    return out


def run_split(P, coarse, sc, params_fn, bs=16):
    ms = []
    with torch.no_grad():
        for i0 in range(0, P["S"], bs):
            idx = torch.arange(i0, min(P["S"], i0 + bs))
            Kn, Tv = sc(*params_fn(idx))
            out = coarse.run(Kn, Tv)
            ms.append(eval_metrics(coarse, out, batch_ref(P, idx), P["iface"][idx].to(DEV)))
    return ms
