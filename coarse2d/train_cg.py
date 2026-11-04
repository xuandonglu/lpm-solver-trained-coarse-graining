"""Train / evaluate the learned coarse-graining (torch environment, GPU).

  python train_cg.py prep  S SPLIT...       # windows + baselines + references -> data/prep2_SPLIT_sS.pt
  python train_cg.py oracle S [--base patch|voigt] [--nfit 16]   # per-sample fitted bond parameters (upper bound)
  python train_cg.py train S [--epochs 60]  # CNN trained through the differentiable coarse LPM
  python train_cg.py eval S SPLIT...        # metrics of voigt / patch / ML on test splits
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
torch.set_num_threads(2)
import torch_lpm as T
import windows as WIN
from cgnet import CGNet, pixel_features, bond_gather_index

DEV = "cuda"
LOADS = T.LOADS


def prep(split, s, force=False):
    """Feature images + baselines + fine references -> data/prep2_SPLIT_sS.pt"""
    fn = "data/prep2_%s_s%d.pt" % (split, s)
    if os.path.exists(fn) and not force:
        return torch.load(fn, weights_only=False)
    d = dict(np.load("data/%s_n%d.npz" % (split, 96 if split == "test_big" else 48), allow_pickle=True))
    n = int(d["n"])
    nc = n // s
    S = len(d["phi"])
    b, shell = WIN.bond_list(nc, nc)
    b0, b1 = b[shell == 0], b[shell == 1]
    X = np.stack([pixel_features(d["Cpix"][i], n, n, s) for i in range(S)])
    out = dict(s=s, n=n, nc=nc, b0=b0, b1=b1, X=torch.as_tensor(X),
               gi0=[torch.as_tensor(x) for x in bond_gather_index(b0, nc, nc)],
               gi1=[torch.as_tensor(x) for x in bond_gather_index(b1, nc, nc)])
    K0a, K0b = [], []
    for i in range(S):
        kn1 = (d["Cpix"][i][:, 0] - d["Cpix"][i][:, 1]) / 2
        cellK = kn1.reshape(nc, s, nc, s).mean(axis=(1, 3)).ravel()
        q0, _ = WIN.neighbor_of(b0, nc)
        q1, _ = WIN.neighbor_of(b1, nc)
        K0a.append(0.5 * (cellK[b0[:, 0]] + cellK[q0]))
        K0b.append(0.5 * (cellK[b1[:, 0]] + cellK[q1]))
    out["K00"], out["K01"] = torch.as_tensor(np.array(K0a)), torch.as_tensor(np.array(K0b))
    for base in ("voigt", "patch"):
        kv0, tv0, kv1, tv1 = [], [], [], []
        for i in range(S):
            cp = d["%s_%d" % (base, s)][i]
            a_, b_ = WIN.base_from_cells(cp, b0, nc)
            kv0.append(a_); tv0.append(b_)
            a_, b_ = WIN.base_from_cells(cp, b1, nc)
            kv1.append(a_); tv1.append(b_)
        out[base] = [torch.as_tensor(np.array(x)) for x in (kv0, tv0, kv1, tv1)]
    ref = {}
    for ld in LOADS:
        ref[ld] = dict(u=torch.as_tensor(d["u_%d_%s" % (s, ld)]), sig=torch.as_tensor(d["sig_%d_%s" % (s, ld)]),
                       w=torch.as_tensor(d["w_%d_%s" % (s, ld)]),
                       F=None if ld.startswith("ring") else torch.as_tensor(d["F_%d_%s" % (s, ld)]))
    out["ref"] = ref
    out["iface"] = torch.as_tensor(d["iface_%d" % s])
    out["meta"] = dict(phi=d["phi"], contrast=d["contrast"], kind=d["kind"], corr=d["corr"])
    out["S"] = S
    torch.save(out, fn)
    return out


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


def export_np(model, fn):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(fn, s=model.s, **sd)


def batch_ref(P, idx):
    r = {}
    for ld in LOADS:
        a = P["ref"][ld]
        r[ld] = dict(u=a["u"][idx].to(DEV), sig=a["sig"][idx].to(DEV), w=a["w"][idx].to(DEV),
                     F=None if a["F"] is None else a["F"][idx].to(DEV))
    return r


def ml_params(model, P, idx, base="voigt"):
    X = P["X"][idx].to(DEV)
    gi0 = [x.to(DEV) for x in P["gi0"]]
    gi1 = [x.to(DEV) for x in P["gi1"]]
    o0, o1 = model(X, gi0, gi1)
    o0, o1 = o0.double(), o1.double()
    kv0, tv0, kv1, tv1 = [x[idx].to(DEV) for x in P[base]]
    K00, K01 = P["K00"][idx].to(DEV), P["K01"][idx].to(DEV)
    return (kv0 * torch.exp(o0[..., 0]), tv0 + K00 * o0[..., 1], kv1 * torch.exp(o1[..., 0]), tv1 + K01 * o1[..., 1])


def eval_metrics(coarse, out, ref, iface):
    """Absolute relative errors per sample (numpy), keys like cg.metrics, averaged over the ring and plate groups."""
    res = {}
    for ld in LOADS:
        fc = coarse.bc[ld]["freecell"]
        a, b = ref[ld], out[ld]
        e_u = ((b["u"] - a["u"])[:, fc] ** 2).sum((1, 2)).sqrt() / (a["u"][:, fc] ** 2).sum((1, 2)).sqrt()
        e_s = ((b["sig"] - a["sig"])[:, fc] ** 2).sum((1, 2)).sqrt() / (a["sig"][:, fc] ** 2).sum((1, 2)).sqrt()
        fi = (fc[None, :] & iface).double()
        e_si = (((b["sig"] - a["sig"]) ** 2).sum(2) * fi).sum(1).sqrt() / ((a["sig"] ** 2).sum(2) * fi).sum(1).sqrt()
        e_wi = (((b["w"] - a["w"]) ** 2) * fi).sum(1).sqrt() / ((a["w"] ** 2) * fi).sum(1).sqrt()
        qa = torch.quantile(a["w"][:, fc], 0.99, dim=1)
        qb = torch.quantile(b["w"][:, fc], 0.99, dim=1)
        e_w99 = (qb - qa) / qa
        if b["F"] is not None:
            k = 1 if ld == "tb_ty" else 0
            e_g = (b["F"][:, k] - a["F"][:, k]) / a["F"][:, k]
        else:
            k = {"ring_xx": 0, "ring_yy": 1, "ring_xy": 2}[ld]
            sa, sb = a["sig"][:, fc, k].mean(1), b["sig"][:, fc, k].mean(1)
            e_g = (sb - sa) / sa
        res[ld] = dict(G=e_g, u=e_u, sig=e_s, sig_if=e_si, w_if=e_wi, w99=e_w99)
    return {ld: {k: v.detach().cpu().numpy() for k, v in d.items()} for ld, d in res.items()}


def summarize(mlist):
    """list of eval_metrics dicts (batches) -> mean |error| per group/key."""
    out = {}
    for grp, loads in (("ring", LOADS[:3]), ("plate", LOADS[3:])):
        for k in ("G", "u", "sig", "sig_if", "w_if", "w99"):
            v = np.concatenate([np.abs(m[ld][k]) for m in mlist for ld in loads])
            out["%s_%s" % (grp, k)] = float(np.nanmean(v))
        v = np.concatenate([m[ld]["G"] for m in mlist for ld in loads])
        out["%s_G_signed" % grp] = float(np.nanmean(v))
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


def baseline_fn(P, base):
    def f(idx):
        return [x[idx].to(DEV) for x in P[base]]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("splits", nargs="*")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", default="voigt")
    ap.add_argument("--nfit", type=int, default=16)
    ap.add_argument("--wts", default="1,1,1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--pen", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=801)
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    wts = tuple(float(x) for x in a.wts.split(","))
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if a.mode == "prep":
        for sp in a.splits:
            t0 = time.time()
            P = prep(sp, s, force=True)
            print("prep %s s=%d: %d samples, X %s, %.1f s" % (sp, s, P["S"], tuple(P["X"].shape), time.time() - t0))
        return

    if a.mode == "oracle":
        P = prep("val", s)
        nc = P["nc"]
        coarse = T.CoarseT(nc, nc, s)
        sc = Scatter(P)
        idx = torch.arange(a.nfit)
        ref = batch_ref(P, idx)
        iface = P["iface"][idx].to(DEV)
        kv0, tv0, kv1, tv1 = [x[idx].to(DEV) for x in P[a.base]]
        K00, K01 = P["K00"][idx].to(DEV).double(), P["K01"][idx].to(DEV).double()
        th0 = torch.zeros(a.nfit, kv0.shape[1], 2, device=DEV, dtype=torch.float64, requires_grad=True)
        th1 = torch.zeros(a.nfit, kv1.shape[1], 2, device=DEV, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([th0, th1], lr=a.lr if a.lr != 1e-3 else 0.01)
        hist = []
        t0 = time.time()
        for it in range(a.iters):
            Kn, Tv = sc(kv0 * torch.exp(th0[..., 0]), tv0 + K00 * th0[..., 1], kv1 * torch.exp(th1[..., 0]),
                        tv1 + K01 * th1[..., 1])
            pen, lam = coarse.stability_penalty(Kn, Tv)
            out = coarse.run(Kn, Tv)
            Lu, Ls, Lg = T.losses(coarse, out, ref, wts=wts)
            loss = (Lu + Ls + Lg + a.pen * pen).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % a.every == 0:
                m = summarize([eval_metrics(coarse, {k: {kk: (vv.detach() if vv is not None else None) for kk, vv in v.items()}
                                                     for k, v in out.items()}, ref, iface)])
                hist.append(dict(it=it, loss=float(loss), **m))
                hist[-1]["min_lam"] = float(lam.min())
                print("oracle s=%d it %4d loss %.4e | plate G %.4f sig %.4f sig_if %.4f w_if %.4f | ring G %.4f sig %.4f "
                      "| min group eig %.3f (%.0fs)"
                      % (s, it, float(loss), m["plate_G"], m["plate_sig"], m["plate_sig_if"], m["plate_w_if"],
                         m["ring_G"], m["ring_sig"], float(lam.min()), time.time() - t0), flush=True)
        json.dump(hist, open("results/oracle_s%d_%s%s.json" % (s, a.base, a.tag), "w"), indent=1)
        return

    if a.mode == "train":
        P = prep("train", s)
        V = prep("val", s)
        nc = P["nc"]
        coarse = T.CoarseT(nc, nc, s)
        sc = Scatter(P)
        model = CGNet(s).to(DEV)
        nparam = sum(p.numel() for p in model.parameters())
        S = P["S"] if a.ntrain == 0 else a.ntrain
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        steps = a.epochs * math.ceil(S / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        print("train s=%d: %d samples, %d params" % (s, S, nparam), flush=True)
        base_v = summarize(run_split(V, coarse, sc, baseline_fn(V, "voigt")))
        base_p = summarize(run_split(V, coarse, sc, baseline_fn(V, "patch")))
        print("val voigt: %s" % {k: round(v, 4) for k, v in base_v.items()})
        print("val patch: %s" % {k: round(v, 4) for k, v in base_p.items()}, flush=True)
        log = dict(voigt=base_v, patch=base_p, hist=[])
        t0 = time.time()
        best = 1e9
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(S)
            tl = []
            for i0 in range(0, S, a.bs):
                idx = perm[i0:i0 + a.bs]
                Kn, Tv = sc(*ml_params(model, P, idx, a.base))
                out = coarse.run(Kn, Tv)
                Lu, Ls, Lg = T.losses(coarse, out, batch_ref(P, idx), wts=wts)
                loss = (Lu + Ls + Lg).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tl.append(float(loss))
            if ep % 5 == 4 or ep == a.epochs - 1:
                model.eval()
                m = summarize(run_split(V, coarse, sc, lambda idx: ml_params(model, V, idx, a.base)))
                log["hist"].append(dict(ep=ep + 1, train_loss=float(np.mean(tl)), **m))
                score = m["plate_sig"] + m["ring_sig"] + m["plate_G"] + m["ring_G"]
                if score < best:
                    best = score
                    torch.save(model.state_dict(), "models/cgnet_s%d%s.pt" % (s, a.tag))
                    export_np(model, "models/cgnet_s%d%s.npz" % (s, a.tag))
                print("ep %3d loss %.4e | val plate G %.4f u %.4f sig %.4f sig_if %.4f w_if %.4f | ring G %.4f sig %.4f (%.0fs)"
                      % (ep + 1, np.mean(tl), m["plate_G"], m["plate_u"], m["plate_sig"], m["plate_sig_if"],
                         m["plate_w_if"], m["ring_G"], m["ring_sig"], time.time() - t0), flush=True)
        log["train_time_s"] = time.time() - t0
        log["nparam"] = nparam
        json.dump(log, open("results/train_s%d%s.json" % (s, a.tag), "w"), indent=1)
        return

    if a.mode == "eval":
        model = CGNet(s).to(DEV)
        model.load_state_dict(torch.load("models/cgnet_s%d%s.pt" % (s, a.tag)))
        model.eval()
        res = {}
        for sp in a.splits:
            P = prep(sp, s)
            nc = P["nc"]
            coarse = T.CoarseT(nc, nc, s)
            sc = Scatter(P)
            bs = 4 if P["n"] > 48 else 16
            r = dict(voigt=summarize(run_split(P, coarse, sc, baseline_fn(P, "voigt"), bs)),
                     patch=summarize(run_split(P, coarse, sc, baseline_fn(P, "patch"), bs)),
                     ml=summarize(run_split(P, coarse, sc, lambda idx: ml_params(model, P, idx, a.base), bs)))
            res[sp] = r
            print("== %s s=%d" % (sp, s))
            for k in r["ml"]:
                print("  %-16s voigt %8.4f  patch %8.4f  ML %8.4f" % (k, r["voigt"][k], r["patch"][k], r["ml"][k]))
        json.dump(res, open("results/eval_s%d%s.json" % (s, a.tag), "w"), indent=1)
        return


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
