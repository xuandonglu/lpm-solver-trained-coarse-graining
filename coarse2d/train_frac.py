"""Learned coarse critical bond stretch (torch environment).

Targets (export_frac.py): for coarse bond b and load l, the fine region R(b) starts to break at the load multiplier
lam_f(b, l).  The coarse bond strain at that moment, xi_C(b, l) * lam_f(b, l) (xi_C from the coarse model with the
learned elastic bond parameters), is the critical stretch that makes the coarse bond fail exactly when its fine
region does.  One critical stretch per bond must serve all loads, so the regression is weighted towards the
(b, l) pairs that are close to the first failure of the specimen in load l:
    w(b, l) = exp(-(log lam_f(b, l) - log lam_min(l)) / tau)        (0 where xi_C <= 0)
Model: CGNet(s, nout=1), crit_C(b) = crit_naive(b) * exp(o).
  python train_frac.py train S [--epochs 60]
  python train_frac.py eval S SPLIT...
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
from cgnet import CGNet
from train_cg import prep, Scatter, ml_params, DEV

TAU = 0.25


def coarse_bond_strains(P, s, elastic, idx_all, bs=16):
    """xi_C (S, n0 / n1, nloads) of the forward bonds under the 6 unit load cases, coarse model with learned params."""
    nc = P["nc"]
    coarse = T.CoarseT(nc, nc, s)
    sc = Scatter(P)
    b0, b1 = P["b0"], P["b1"]
    out0, out1 = [], []
    with torch.no_grad():
        for i0 in range(0, len(idx_all), bs):
            idx = idx_all[i0:i0 + bs]
            Kn, Tv = sc(*ml_params(elastic, P, idx))
            res = coarse.run(Kn, Tv)
            x0, x1 = [], []
            for ld in T.LOADS:
                dL, _ = coarse.bond_forces(res[ld]["u"].reshape(len(idx), -1), Kn, Tv)
                xi = dL / coarse.l0
                x0.append(xi[:, b0[:, 0], b0[:, 1]])
                x1.append(xi[:, b1[:, 0], b1[:, 1]])
            out0.append(torch.stack(x0, -1).cpu())
            out1.append(torch.stack(x1, -1).cpu())
    return torch.cat(out0), torch.cat(out1)


def targets(P, F, s, elastic, nmax=None):
    S = len(F["lmin_%d" % s]) if nmax is None else nmax
    idx = torch.arange(S)
    x0, x1 = coarse_bond_strains(P, s, elastic, idx)
    out = {}
    lmin = torch.as_tensor(F["lmin_%d" % s][:S])                    # (S, L)
    for sh, xc in ((0, x0), (1, x1)):
        lam = torch.as_tensor(F["lam_%d_%d" % (s, sh)][:S])         # (S, n, L)
        t = lam * xc
        ok = (xc > 0) & torch.isfinite(t) & (t > 0)
        w = torch.exp(-(torch.log(lam) - torch.log(lmin)[:, None, :]) / TAU) * ok
        out["logt%d" % sh] = torch.where(ok, torch.log(torch.where(ok, t, torch.ones_like(t))), torch.zeros_like(t))
        out["w%d" % sh] = w
        out["naive%d" % sh] = torch.as_tensor(F["naive_%d_%d" % (s, sh)][:S])
        out["patch%d" % sh] = torch.as_tensor(F["patch_%d_%d" % (s, sh)][:S])
        out["xc%d" % sh] = xc
        out["lam%d" % sh] = lam
    out["lmin"] = lmin
    return out


def predict(model, P, idx, Tg):
    X = P["X"][idx].to(DEV)
    gi0 = [x.to(DEV) for x in P["gi0"]]
    gi1 = [x.to(DEV) for x in P["gi1"]]
    o0, o1 = model(X, gi0, gi1)
    return (torch.log(Tg["naive0"][idx].to(DEV)) + o0[..., 0].double(),
            torch.log(Tg["naive1"][idx].to(DEV)) + o1[..., 0].double())


def first_failure_metrics(lc0, lc1, Tg, idx):
    """lc*: log crit_C (B, n). Returns per sample and load: relative error of the predicted first-failure load
    multiplier (min_b crit_C/xi_C) vs the fine lam_min, and whether the predicted first coarse bond is one whose
    fine region fails within 5% of the fine first failure (location hit)."""
    errs, hits = [], []
    for sh_l, lc in ((0, lc0), (1, lc1)):
        pass
    xc = torch.cat([Tg["xc0"][idx], Tg["xc1"][idx]], 1).to(lc0.device)        # (B, n, L)
    lam = torch.cat([Tg["lam0"][idx], Tg["lam1"][idx]], 1).to(lc0.device)
    crit = torch.exp(torch.cat([lc0, lc1], 1))[..., None]
    lp = torch.where(xc > 0, crit / xc, torch.full_like(xc, float("inf")))
    lpm, arg = lp.min(1)                                                       # (B, L)
    lmin = Tg["lmin"][idx].to(lc0.device)
    err = (lpm - lmin) / lmin
    lam_at = torch.gather(lam, 1, arg[:, None, :])[:, 0, :]
    hit = (lam_at <= 1.05 * lmin).double()
    return err.cpu().numpy(), hit.cpu().numpy()


def wloss(lc, Tg, idx, sh):
    w = Tg["w%d" % sh][idx].to(DEV)
    t = Tg["logt%d" % sh][idx].to(DEV)
    return ((lc[..., None] - t) ** 2 * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("splits", nargs="*")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    elastic = CGNet(s).to(DEV)
    elastic.load_state_dict(torch.load("models/cgnet_s%d.pt" % s))
    elastic.eval()
    if a.mode == "train":
        P = prep("train", s)
        V = prep("val", s)
        Ft = dict(np.load("data/frac_train.npz"))
        Fv = dict(np.load("data/frac_val.npz"))
        t0 = time.time()
        Tg = targets(P, Ft, s, elastic)
        Tv_ = targets(V, Fv, s, elastic)
        S = len(Ft["lmin_%d" % s])
        print("targets: %d train samples (%.0f s)" % (S, time.time() - t0), flush=True)
        model = CGNet(s, nout=1).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        steps = a.epochs * math.ceil(S / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

        def val_report():
            res = {}
            Sv = len(Fv["lmin_%d" % s])
            idx = torch.arange(Sv)
            with torch.no_grad():
                lc0, lc1 = predict(model, V, idx, Tv_)
            for name, (c0, c1) in {"naive": (torch.log(Tv_["naive0"]).to(DEV), torch.log(Tv_["naive1"]).to(DEV)),
                                   "patch": (torch.log(Tv_["patch0"]).to(DEV), torch.log(Tv_["patch1"]).to(DEV)),
                                   "ml": (lc0, lc1)}.items():
                wl = float(((wloss(c0, Tv_, idx, 0) + wloss(c1, Tv_, idx, 1)) / 2).mean().sqrt())
                err, hit = first_failure_metrics(c0, c1, Tv_, idx)
                res[name] = dict(wrmse_log=wl, first_err_abs=float(np.abs(err).mean()),
                                 first_err_signed=float(err.mean()), first_hit=float(hit.mean()),
                                 first_err_tb_ty=float(np.abs(err[:, 3]).mean()), hit_tb_ty=float(hit[:, 3].mean()))
            return res
        r0 = val_report()
        print("val baselines: naive %s\n               patch %s" % (r0["naive"], r0["patch"]), flush=True)
        log = dict(base=r0, hist=[])
        best = 1e9
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(S)
            tl = []
            for i0 in range(0, S, a.bs):
                idx = perm[i0:i0 + a.bs]
                lc0, lc1 = predict(model, P, idx, Tg)
                loss = ((wloss(lc0, Tg, idx, 0) + wloss(lc1, Tg, idx, 1)) / 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tl.append(float(loss))
            if ep % 5 == 4 or ep == a.epochs - 1:
                model.eval()
                r = val_report()["ml"]
                log["hist"].append(dict(ep=ep + 1, train=float(np.mean(tl)), **r))
                if r["wrmse_log"] < best:
                    best = r["wrmse_log"]
                    torch.save(model.state_dict(), "models/fracnet_s%d%s.pt" % (s, a.tag))
                    from train_cg import export_np
                    export_np(model, "models/fracnet_s%d%s.npz" % (s, a.tag))
                print("ep %3d train %.4f | val ML %s (%.0fs)" % (ep + 1, np.mean(tl), {k: round(v, 4) for k, v in r.items()},
                                                                 time.time() - t0), flush=True)
        json.dump(log, open("results/train_frac_s%d%s.json" % (s, a.tag), "w"), indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
