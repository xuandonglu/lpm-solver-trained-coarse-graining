"""L2b: oracle-style global calibration (reference / upper bound; needs fine solves of the WHOLE specimen).

Per specimen, every coarse bond's (Kn, Tv) is fitted so that the coarse LPM reproduces the fine global solution of the
same specimen (I2 'oracle' mode of train_cg.py: Kn = Kn_patch exp(t1), Tv = Tv_patch + K0 t2, loss = displacement +
stress-field + global-stiffness squared relative errors, stability penalty on every particle's bond-space Hessian),
here on the TEST specimens and driven to convergence with L-BFGS instead of 400 Adam steps.
  --loads all   : the 6 I2 load cases (= the evaluation loads; a true per-sample upper bound)
  --loads ring  : only the 3 KUBC (RVE) loads = 3 global fine solves (Silling-Jafarzadeh-Yu style calibration),
                  evaluated afterwards on all 6 loads (the 3 plate loads are then unseen)
usage: python oracle.py SPLIT S [--loads all|ring] [--bs 16 --iters 60 --outer 12 --n1 100]
"""
import argparse
import json
import os
import time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("I2C_THREADS", "2")))
import common
import torch_lpm as T
import train_lib as TL

DEV = TL.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("s", type=int)
    ap.add_argument("--loads", default="all")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--outer", type=int, default=12)
    ap.add_argument("--n0", type=int, default=0)
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--pen", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--opt", default="adam")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    s = a.s
    P = TL.load_prep(a.split, s)
    nc = P["nc"]
    loads = T.LOADS if a.loads == "all" else T.LOADS[:3]
    coarse = T.CoarseT(nc, nc, s, device=DEV)
    sc = TL.Scatter(P)
    n1 = min(a.n1, P["S"])
    Kn_all = np.zeros((n1 - a.n0, nc * nc, 8))
    Tv_all = np.zeros_like(Kn_all)
    log = dict(split=a.split, s=s, loads=a.loads, batches=[])
    t_start = time.time()
    for i0 in range(a.n0, n1, a.bs):
        idx = torch.arange(i0, min(n1, i0 + a.bs))
        ref = TL.batch_ref(P, idx)
        kv0, tv0, kv1, tv1 = [x[idx].to(DEV) for x in P["patch"]]
        K00, K01 = P["K00"][idx].to(DEV).double(), P["K01"][idx].to(DEV).double()
        th0 = torch.zeros(len(idx), kv0.shape[1], 2, device=DEV, dtype=torch.float64, requires_grad=True)
        th1 = torch.zeros(len(idx), kv1.shape[1], 2, device=DEV, dtype=torch.float64, requires_grad=True)

        def params():
            return sc(kv0 * torch.exp(th0[..., 0]), tv0 + K00 * th0[..., 1], kv1 * torch.exp(th1[..., 0]),
                      tv1 + K01 * th1[..., 1])

        best = [None]

        def closure():
            opt.zero_grad()
            Kn, Tv = params()
            pen, _ = coarse.stability_penalty(Kn, Tv)
            try:
                out = coarse.run(Kn, Tv, loads=loads)
            except FloatingPointError:
                # trial point of the line search with a non positive definite coarse stiffness: report a large loss
                # (zero gradient) so that the strong-Wolfe search backtracks
                return torch.tensor(1e3 * (best[0] if best[0] is not None else 1.0), dtype=torch.float64)
            Lu, Ls, Lg = T.losses(coarse, out, ref, loads=loads)
            loss = (Lu + Ls + Lg + a.pen * pen).sum()
            loss.backward()
            lv = float(loss.detach())
            best[0] = lv if best[0] is None else min(best[0], lv)
            return loss

        t0 = time.time()
        hist = []
        if a.opt == "lbfgs":
            opt = torch.optim.LBFGS([th0, th1], lr=1.0, max_iter=a.iters, history_size=30, tolerance_grad=1e-12,
                                    tolerance_change=1e-15, line_search_fn="strong_wolfe")
            for it in range(a.outer):
                l_ = float(opt.step(closure)) / len(idx)
                hist.append(l_)
        else:
            # Adam as in the I2 oracle (lr 0.01 from the patch parameters), more steps, cosine decay to lr/100
            opt = torch.optim.Adam([th0, th1], lr=a.lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 100)
            for it in range(a.steps):
                l_ = float(closure()) / len(idx)
                opt.step()
                sch.step()
                if it % 100 == 0 or it == a.steps - 1:
                    hist.append(l_)
        with torch.no_grad():
            Kn, Tv = params()
            _, lam = coarse.stability_penalty(Kn, Tv)
        Kn_all[i0 - a.n0:i0 - a.n0 + len(idx)] = Kn.detach().cpu().numpy()
        Tv_all[i0 - a.n0:i0 - a.n0 + len(idx)] = Tv.detach().cpu().numpy()
        log["batches"].append(dict(i0=i0, hist=hist, time=time.time() - t0, min_lam=float(lam.min())))
        print("oracle(%s) %s s=%d samples %d-%d: loss/sample %s (%.0f s) min lam %.3f" % (
            a.loads, a.split, s, i0, i0 + len(idx) - 1, " ".join("%.3e" % h for h in hist[::2] + [hist[-1]]),
            time.time() - t0, float(lam.min())), flush=True)
    name = "l2b%s%s" % ("" if a.loads == "all" else "_ring", a.tag)
    np.savez(os.path.join(common.RES, "params_%s_%s_s%d.npz" % (name, a.split, s)), Kn=Kn_all, Tv=Tv_all)
    log["wall"] = time.time() - t_start
    log["time_per_sample"] = log["wall"] / (n1 - a.n0)
    json.dump(log, open(os.path.join(common.RES, "oracle_%s_%s_s%d.json" % (name, a.split, s)), "w"), indent=1)


if __name__ == "__main__":
    main()
