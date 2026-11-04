"""Experiment: equivariance ablation of the I8 3D super-particle network (torch environment, GPU).

Variants (same data, loss, optimiser, epochs, batch, model selection on val as I8 train3d.py):
  eq        I8 BondNet: O_h-invariant 3x3x3 kernels (trunk exactly equivariant) + first head layer averaged over the
            stabiliser of the canonical bond (16 axial / 8 diagonal elements) -> bond field exactly O_h-equivariant.
  ne        same widths / layers / windows / canonical-frame gather, but free 3x3x3 kernels (27 weights per channel
            pair) and NO stabiliser averaging (the identity coset element only) -> not equivariant (P1's A2 in 3D).
  --aug     random cube symmetry g per batch: the network sees g X, its bond outputs are pulled back to the bonds of
            the original specimen (exact for the physics: the coarse LPM, loads and references are unchanged; P1 A2).
Modes:
  python ablate.py check                         O_h pull-back check with the trained I8 net (must be ~1e-6, fp32)
  python ablate.py train VAR TAG [--aug] [--seed k] [--epochs 30] [--ntrain N]
  python ablate.py eval VAR TAG [--i8] [--orient]  errors on the I8 test splits (+ orientation scan of test_id)
Data are read from ../data (the coarse3d folder, prepared files only); everything is written under this folder.
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
I8 = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, I8)
sys.path.insert(0, HERE)
for _d in ("models", "results", "logs"):
    os.makedirs(os.path.join(HERE, _d), exist_ok=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import geom3d as G  # noqa: E402
import net3d as NT  # noqa: E402
import torch_lpm3d as T  # noqa: E402
import train3d as TR  # noqa: E402  (local copy: Scatter, errors, loss_terms, group_penalty, summarize, lin corr.)

DEV = "cuda"
S = 2


def load_prep(split, s=S):
    return torch.load(os.path.join(I8, "data", "prep_%s_s%d.pt" % (split, s)), weights_only=False)


# ------------------------------------------------------------------------------------------------ networks
class FreeConvBondNet(NT.BondNet):
    """'ne': BondNet with free 3x3x3 kernels and no stabiliser averaging (identity coset element only)."""

    def __init__(self, s, cin=3, ch=24, D=12, hid=64):
        super().__init__(s, cin, ch, D, hid)
        convs = []
        for ci, co in ((cin, ch), (ch, ch), (ch, ch)):
            c = nn.Conv3d(ci, co, 3, padding=1)
            nn.init.normal_(c.weight, 0.0, math.sqrt(2.0 / (ci * 27)))     # same per-entry scale as IsoConv3d
            nn.init.zeros_(c.bias)
            convs.append(c)
        self.trunk = nn.Sequential(convs[0], nn.GELU(), convs[1], nn.GELU(), convs[2], nn.GELU(), nn.Conv3d(ch, D, 1))

    def forward(self, X, nc):
        Fv = self.trunk(X)
        Fc = F.avg_pool3d(Fv, self.s, self.s)
        Fv = Fv.flatten(2).transpose(1, 2)
        Fc = Fc.flatten(2).transpose(1, 2)
        outs = []
        for (l1, l2, l3), g in zip(self.heads, self.windows(nc, X.device)):
            if "pf_id" not in g:                                                 # identity element of the stabiliser
                g["pf_id"] = bool((g["pf"][0] == torch.arange(g["pf"].shape[1], device=g["pf"].device)).all())
                assert g["pf_id"]
            x = torch.cat([Fv[:, g["iv"]], Fc[:, g["ic"]]], 2).flatten(2)       # (B, nb, nin) canonical order
            h = F.gelu(F.linear(x, l1.weight, l1.bias))
            outs.append(l3(F.gelu(l2(h))))
        return outs


def make_net(var):
    return NT.BondNet(S) if var == "eq" else FreeConvBondNet(S)


# ------------------------------------------------------------------------------------------------ O_h action
OH = NT.OH


def apply_g(X, gi):
    """X (B, C, Z, Y, X) -> the specimen transformed by OH[gi] about the box centre (x' = g x, centred coords)."""
    g = OH[gi]
    perm_axis = [int(np.nonzero(g[r])[0][0]) for r in range(3)]                 # x'_r = sg_r x_{p[r]}
    sg = [int(g[r, perm_axis[r]]) for r in range(3)]
    d = lambda a: 4 - a                                                          # axis a (0 x, 1 y, 2 z) -> array dim
    dims = [0, 1, 0, 0, 0]
    for r in range(3):
        dims[d(r)] = d(perm_axis[r])
    Y = X.permute(*dims)
    fl = [d(r) for r in range(3) if sg[r] < 0]
    return Y.flip(fl) if fl else Y


_BP = {}


def bond_perm(nc, gi, groups):
    """per bond group: index (in the forward-bond list of the transformed specimen) of the image of every forward bond
    of the original specimen under OH[gi]."""
    key = (nc, gi)
    if key in _BP:
        return _BP[key]
    g = OH[gi]
    fwd = {tuple(G.SC_OFFS[k]): k for k in range(0, 18, 2)}
    cen = (nc - 1) / 2.0
    out = []
    for grp in groups:
        b = grp["bonds"]
        look = {(int(p), int(k)): t for t, (p, k) in enumerate(b)}
        p = b[:, 0]
        A = np.stack([p % nc, (p // nc) % nc, p // (nc * nc)], 1)
        o = G.SC_OFFS[b[:, 1]]
        A2 = np.rint((A - cen) @ g.T + cen).astype(np.int64)
        o2 = o @ g.T
        idx = np.empty(len(b), np.int64)
        for t in range(len(b)):
            oo = tuple(int(v) for v in o2[t])
            if oo in fwd:
                a, k = A2[t], fwd[oo]
            else:
                a, k = A2[t] + o2[t], fwd[tuple(-v for v in oo)]
            idx[t] = look[(int((a[2] * nc + a[1]) * nc + a[0]), k)]
        out.append(torch.as_tensor(idx, device=DEV))
    _BP[key] = out
    return out


def raw_outputs(net, X, nc, gi=0):
    """network outputs for the original bonds, the network seeing the specimen transformed by OH[gi]."""
    if gi == 0:
        return net(X, nc)
    o = net(apply_g(X, gi), nc)
    pm = bond_perm(nc, gi, net.windows(nc, DEV))
    return [oo[:, p] for oo, p in zip(o, pm)]


def sym_outputs(net, X, nc):
    """test-time symmetrisation (frame averaging over O_h): mean over the 48 g of the pulled-back outputs of f(g X);
    exactly equivariant for any network, at 48 forward passes."""
    acc = None
    for gi in range(48):
        o = raw_outputs(net, X, nc, gi)
        acc = [x.double() for x in o] if acc is None else [a_ + x.double() for a_, x in zip(acc, o)]
    return [a_ / 48.0 for a_ in acc]


def ml_vals(net, P, idx, nc, gi=0, symm=False):
    X = P["X"][idx].to(DEV)
    o = sym_outputs(net, X, nc) if symm else raw_outputs(net, X, nc, gi)
    vals = []
    for g_, oo in enumerate(o):
        kv, tv = [x[idx].to(DEV) for x in P["voigt"][g_]]
        oo = oo.double()
        vals.append((kv * torch.exp(oo[..., 0]), tv + kv * oo[..., 1]))
    return vals


# ------------------------------------------------------------------------------------------------ modes
def check():
    net = NT.BondNet(S).to(DEV)
    net.load_state_dict(torch.load(os.path.join(I8, "models", "bondnet_s2.pt"), map_location=DEV))
    net.eval()
    net.double()
    P = load_prep("val")
    nc = P["nc"]
    X = P["X"][:2].to(DEV).double()
    with torch.no_grad():
        ref = net(X, nc)
        worst = 0.0
        for gi in range(48):
            o = raw_outputs(net, X, nc, gi)
            d = max(float((a - b).abs().max()) for a, b in zip(o, ref))
            worst = max(worst, d)
        mag = max(float(r.abs().max()) for r in ref)
        ne = FreeConvBondNet(S).to(DEV).double()
        for l1, l2, l3 in ne.heads:
            nn.init.normal_(l3.weight, 0.0, 0.1)
        dn = []
        for gi in range(1, 48):
            o = raw_outputs(ne, X, nc, gi)
            r0 = ne(X, nc)
            dn.append(max(float((a - b).abs().max()) for a, b in zip(o, r0)))
    print("I8 eq net, max |pullback f(gX) - f(X)| over 48 g: %.2e (output magnitude %.3f)" % (worst, mag))
    print("random ne net (random last layer): max |pullback f(gX) - f(X)| over 47 g != e: %.2e" % max(dn))
    nparams = {v: sum(p.numel() for p in make_net(v).parameters()) for v in ("eq", "ne")}
    print("parameters:", nparams)


def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    P = load_prep(a.train_split)
    if a.ntrain:                                    # keep only the first ntrain specimens in memory (small-N runs)
        k = a.ntrain
        P["X"] = P["X"][:k].clone()
        for b in ("voigt", "patch"):
            if b in P:
                P[b] = [(x[:k].clone(), y[:k].clone()) for x, y in P[b]]
        P["ref"] = {kk: v[:k].clone() for kk, v in P["ref"].items()}
        P["iface"] = P["iface"][:k].clone()
        P["S"] = k
        import gc
        gc.collect()
    V = load_prep("val")
    nc = P["nc"]
    cm = T.CoarseT(nc, S)
    net = make_net(a.var).to(DEV)
    gw = net.windows(nc, DEV)
    sc = TR.Scatter(nc, S, gw)
    nparam = sum(p.numel() for p in net.parameters())
    Sn = P["S"] if a.ntrain == 0 else a.ntrain
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * math.ceil(Sn / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    print("train %s var=%s aug=%s seed=%d: %d samples, %d params, %d steps" % (a.tag, a.var, a.aug, a.seed, Sn, nparam,
                                                                               steps), flush=True)
    log = dict(var=a.var, aug=a.aug, seed=a.seed, ntrain=Sn, epochs=a.epochs, hist=[], nparam=nparam)
    best = 1e9
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    every = max(1, a.epochs // 10)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(Sn)
        tl, lmins = [], []
        for i0 in range(0, Sn, a.bs):
            idx = perm[i0:i0 + a.bs]
            gi = int(rng.integers(48)) if a.aug else 0
            Kn, Tv = sc(ml_vals(net, P, idx, nc, gi))
            pen, lmin = TR.group_penalty(cm, Kn, Tv)
            out = cm.run(Kn, Tv, tol=1e-9)
            Lu, Ls, Lg = TR.loss_terms(cm, out, TR.batch_ref(P, idx))
            loss = (Lu + Ls + Lg + a.pen * pen).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(float(loss.detach()))
            lmins.append(lmin)
        if ep % every == every - 1 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                m, _ = TR.run_split(V, cm, sc, lambda idx: ml_vals(net, V, idx, nc))
            log["hist"].append(dict(ep=ep + 1, loss=float(np.mean(tl)), t=time.time() - t0, **m))
            score = m["sig_all"] + m["G_all"]
            if score < best:
                best = score
                torch.save(net.state_dict(), os.path.join(HERE, "models", "%s.pt" % a.tag))
            print("ep %3d loss %.3e minlam %.3f | val G %.4f sig %.4f (ring %.4f plate %.4f) surf %.4f w_if %.4f C %.4f "
                  "(%.0f s)" % (ep + 1, np.mean(tl), min(lmins), m["G_all"], m["sig_all"], m["sig_ring"],
                                m["sig_plate"], m["sig_surf_plate"], m["w_if_all"], m["C"], time.time() - t0), flush=True)
        else:
            print("ep %3d loss %.3e minlam %.3f (%.0f s)" % (ep + 1, np.mean(tl), min(lmins), time.time() - t0),
                  flush=True)
    log["train_time_s"] = time.time() - t0
    json.dump(log, open(os.path.join(HERE, "results", "train_%s.json" % a.tag), "w"), indent=1)


def evaluate(a):
    """test-split errors (I8 eval keys) + orientation scan on test_id (all 48 cube symmetries) + stability check."""
    net = make_net(a.var).to(DEV)
    mtag = a.model or a.tag
    fn = os.path.join(I8, "models", "bondnet_s2.pt") if a.i8 else os.path.join(HERE, "models", "%s.pt" % mtag)
    net.load_state_dict(torch.load(fn, map_location=DEV))
    net.eval()
    res = dict(tag=a.tag, var=a.var, model=fn, splits={})
    keys = ("G_all", "C", "u_all", "sig_all", "sig_ring", "sig_plate", "sig_all_p90", "sig_if_all", "w_if_all",
            "sig_surf_plate")
    for sp in a.splits:
        P = load_prep(sp)
        nc = P["nc"]
        cm = T.CoarseT(nc, S)
        sc = TR.Scatter(nc, S, net.windows(nc, DEV))
        bs = 8 if P["n"] <= 24 else (2 if P["n"] <= 48 else 1)
        with torch.no_grad():
            r, cat = TR.run_split(P, cm, sc, lambda idx: ml_vals(net, P, idx, nc, symm=a.symm), bs)
            # local stability of the deployed bond field (every (particle, shell) bond Hessian), as I8 check_stability
            lm = []
            for i0 in range(0, P["S"], bs):
                idx = torch.arange(i0, min(P["S"], i0 + bs))
                Kn, Tv = sc(ml_vals(net, P, idx, nc, symm=a.symm))
                lm.append(TR.group_penalty(cm, Kn, Tv)[1])
        r["min_rel_eig"] = float(min(lm))
        r["per_sample_sig"] = cat["sig"].mean(1).tolist()
        res["splits"][sp] = r
        print("== %s %s (%d): " % (a.tag, sp, P["S"]) + "  ".join("%s %.4f" % (k, r[k]) for k in keys)
              + "  minlam %.3f" % r["min_rel_eig"], flush=True)
    if a.orient:
        P = load_prep("test_id")
        nc = P["nc"]
        cm = T.CoarseT(nc, S)
        sc = TR.Scatter(nc, S, net.windows(nc, DEV))
        o_sig, o_C, o_dev = [], [], []
        with torch.no_grad():
            X0 = P["X"][:16].to(DEV)
            f0 = [x.double() for x in net(X0, nc)]
            for gi in range(48):
                r, cat = TR.run_split(P, cm, sc, lambda idx: ml_vals(net, P, idx, nc, gi), 8)
                o_sig.append(r["sig_all"])
                o_C.append(r["C"])
                fg = [x.double() for x in raw_outputs(net, X0, nc, gi)]
                o_dev.append(float(sum(((u - v) ** 2).sum() for u, v in zip(fg, f0)).sqrt()
                                   / sum((v ** 2).sum() for v in f0).sqrt()))
                print("  orient %2d sig %.4f C %.4f dev %.2e" % (gi, o_sig[-1], o_C[-1], o_dev[-1]), flush=True)
        res["orient"] = dict(sig=o_sig, C=o_C, dev=o_dev)
        print("orientation scan test_id: sig mean %.4f min %.4f max %.4f; C mean %.4f max %.4f; field dev mean %.2e"
              % (np.mean(o_sig), np.min(o_sig), np.max(o_sig), np.mean(o_C), np.max(o_C), np.mean(o_dev[1:])),
              flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "eval_%s.json" % a.tag), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("var", nargs="?", default="eq")
    ap.add_argument("tag", nargs="?", default="")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pen", type=float, default=100.0)
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--splits", nargs="*", default=["test_id", "test_big", "test_huge", "test_elong", "test_c8",
                                                     "test_soft"])
    ap.add_argument("--i8", action="store_true")
    ap.add_argument("--orient", action="store_true")
    ap.add_argument("--symm", action="store_true", help="eval: test-time O_h frame averaging (48 passes)")
    ap.add_argument("--model", default="", help="eval: model tag to load (default = tag)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.mode == "check":
        check()
    elif a.mode == "train":
        train(a)
    elif a.mode == "eval":
        evaluate(a)


if __name__ == "__main__":
    main()
