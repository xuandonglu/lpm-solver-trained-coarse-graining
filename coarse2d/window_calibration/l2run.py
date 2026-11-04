"""L2 driver.

  python l2run.py targets SPLIT S MF [--n0 0 --n1 100 --workers 6]      # local fine solves -> data/l2t/...
  python l2run.py fit SPLIT S MC MF MODE [--tag T --n1 100 ...]          # coarse fits -> results/params_<tag>_<SPLIT>_s<S>.npz
     MODE = central | joint | allwin;  MC = coarse margin (1 = I2 network footprint), MF = fine margin (MC: KUBC,
     MC + 1: oversampled)
The fine targets are cached per specimen (float32); the fit runs on the GPU (fp64).
"""
import argparse
import json
import math
import os
import time
import numpy as np
import common  # noqa
import l2fine as LF

TDIR = os.path.join(common.DATA, "l2t")


def tfile(split, s, mf, i, mode="aff"):
    return os.path.join(TDIR, "%s_s%d_m%d%s" % (split, s, mf, "" if mode == "aff" else "_" + mode), "%03d.npz" % i)


def _targets_one(args):
    split, s, mf, i, mode = args
    fn = tfile(split, s, mf, i, mode)
    if os.path.exists(fn):
        return i, 0.0
    d = np.load(common.split_file(split))
    n = int(d["n"])
    t0 = time.perf_counter()
    r = LF.fine_targets(d["Cpix"][i], n, s, mf, mode)
    dt = time.perf_counter() - t0
    np.savez(fn, time=dt, **{k: v for k, v in r.items() if not k.startswith("dofs")},
             dofs=np.array([r["dofs_%d" % k] for k in LF.KS]))
    return i, dt


def load_targets(split, s, mf, i, mode="aff"):
    z = np.load(tfile(split, s, mf, i, mode))
    return {k: {key: z["%s_%d" % (key, k)] for key in ("u", "sig", "w", "exists")} for k in LF.KS}, float(z["time"])


# ----------------------------------------------------------------------------------------------------------------
def spec_base(sp_d, i, s, base="patch"):
    """per-specimen forward-bond list, base (Kn, Tv) per forward bond, K0 per forward bond, and (Nc, 8) base arrays."""
    import cg
    import windows as WIN
    n = int(sp_d["n"])
    nc = n // s
    lat = cg.CoarseModel(nc, nc, s).lat
    cp = sp_d["%s_%d" % (base, s)][i]
    Kn, Tv = cg.bond_params_from_cells(lat, cp)
    kn1 = (sp_d["Cpix"][i][:, 0] - sp_d["Cpix"][i][:, 1]) / 2
    cellK = kn1.reshape(nc, s, nc, s).mean(axis=(1, 3)).ravel()
    b, _ = WIN.bond_list(nc, nc)
    q, _ = WIN.neighbor_of(b, nc)
    K0 = 0.5 * (cellK[b[:, 0]] + cellK[q])
    return dict(b=b, kn=Kn[b[:, 0], b[:, 1]], tv=Tv[b[:, 0], b[:, 1]], K0=K0, Kn=Kn, Tv=Tv, nc=nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("split")
    ap.add_argument("s", type=int)
    ap.add_argument("m1", type=int)
    ap.add_argument("m2", type=int, nargs="?")
    ap.add_argument("fitmode", nargs="?")
    ap.add_argument("--n0", type=int, default=0)
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=4, help="specimens per GPU batch")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--outer", type=int, default=8)
    ap.add_argument("--wts", default="1,1,1")
    ap.add_argument("--tik", type=float, default=1e-3)
    ap.add_argument("--pen", type=float, default=10.0)
    ap.add_argument("--base", default="patch")
    ap.add_argument("--states", default="aff", help="aff (3 affine states) | quad (+ 6 linear-gradient states)")
    a = ap.parse_args()

    if a.mode == "targets":
        import multiprocessing as mp
        mf = a.m1
        os.makedirs(os.path.dirname(tfile(a.split, a.s, mf, 0, a.states)), exist_ok=True)
        jobs = [(a.split, a.s, mf, i, a.states) for i in range(a.n0, a.n1)]
        t0 = time.time()
        if a.workers <= 1:
            res = [_targets_one(j) for j in jobs]
        else:
            with mp.Pool(a.workers) as pool:
                res = pool.map(_targets_one, jobs, chunksize=1)
        dts = [r[1] for r in res if r[1] > 0]
        print("targets %s s=%d mf=%d: %d specimens, %.0f s wall, mean %.2f s / specimen (1 thread)" % (
            a.split, a.s, mf, len(jobs), time.time() - t0, np.mean(dts) if dts else float("nan")), flush=True)
        return

    import torch
    torch.set_num_threads(2)
    import l2torch as L2
    import torch_lpm as T
    DEV = L2.DEV
    s, mc, mf, fm = a.s, a.m1, a.m2, a.fitmode
    wts = tuple(float(x) for x in a.wts.split(","))
    d = dict(np.load(common.split_file(a.split), allow_pickle=True))
    n = int(d["n"])
    nc = n // s
    tag = a.tag or "l2%s_m%d%s%s" % (fm[0], mc, "o" if mf > mc else "", "q" if a.states == "quad" else "")
    out_fn = os.path.join(common.RES, "params_%s_%s_s%d.npz" % (tag, a.split, s))
    import windows as WIN
    geo_o = {k: L2.WinGeom(k, mc, s) for k in LF.KS}
    ids = list(range(a.n0, a.n1))
    S = len(ids)
    Kn_all = np.zeros((S, nc * nc, 8))
    Tv_all = np.zeros((S, nc * nc, 8))
    log = dict(tag=tag, split=a.split, s=s, mc=mc, mf=mf, mode=fm, wts=wts, batches=[])
    t_fine = []
    t_start = time.time()
    G = LF.bond_groups(nc)

    def batch_loss(batches, Kn, Tv):
        return sum(float((wb.residuals(wb.forward(*L2.gather_local(Kn.reshape(-1), Tv.reshape(-1), wb)), wts) ** 2)
                         .sum()) for wb in batches.values())

    for j0 in range(0, S, a.bs):
        bids = ids[j0:j0 + a.bs]
        nS = len(bids)
        bases = [spec_base(d, i, s, a.base) for i in bids]
        tgs = []
        for i in bids:
            tg, dt = load_targets(a.split, s, mf, i, a.states)
            tgs.append(tg)
            t_fine.append(dt)
        batches = {}
        for kc, ks in ((0, (0, 2)), (4, (4, 6))):
            dl, K0w = [], []
            for jj in range(nS):
                for k in ks:
                    d0 = L2.window_data(geo_o[k], G[k], nc, s, mf, tgs[jj][k], i_spec=jj)
                    dl.append(L2.canonicalize(d0, k, geo_o[k], geo_o[kc]))
                    K0w.append(bases[jj]["K0"][bases[jj]["b"][:, 1] == k])
            wb = L2.WinBatch(geo_o[kc], dl, nc)
            wb.K0 = torch.as_tensor(np.concatenate(K0w), device=DEV)
            batches[kc] = wb
        nwin = sum(wb.B for wb in batches.values())
        flatKn = torch.as_tensor(np.stack([bb["Kn"] for bb in bases]).ravel(), device=DEV)
        flatTv = torch.as_tensor(np.stack([bb["Tv"] for bb in bases]).ravel(), device=DEV)
        Kn_new = [bb["Kn"].copy() for bb in bases]
        Tv_new = [bb["Tv"].copy() for bb in bases]

        def write_back(wb, kv, tvv):
            bk = wb.bk
            q, kr = WIN.neighbor_of(bk, nc)
            sp_ = wb.spec.cpu().numpy()
            for jj in range(nS):
                m = sp_ == jj
                Kn_new[jj][bk[m, 0], bk[m, 1]] = kv[m]
                Kn_new[jj][q[m], kr[m]] = kv[m]
                Tv_new[jj][bk[m, 0], bk[m, 1]] = tvv[m]
                Tv_new[jj][q[m], kr[m]] = tvv[m]

        t0 = time.time()
        rec = dict(ids=bids)
        if fm == "central":
            loss0, loss1 = [], []
            for kc, wb in batches.items():
                g = wb.g
                th, f, f0 = L2.fit_central(wb, flatKn, flatTv, wb.K0, iters=a.iters, wts=wts, pen=a.pen)
                Kl, Tl = L2.gather_local(flatKn, flatTv, wb)
                kv = (Kl[:, g.pA, g.k] * torch.exp(th[:, 0])).cpu().numpy()
                tvv = (Tl[:, g.pA, g.k] + wb.K0 * th[:, 1]).cpu().numpy()
                write_back(wb, kv, tvv)
                loss0.append(f0.cpu().numpy())
                loss1.append(f.cpu().numpy())
            rec.update(loss_patch=float(np.mean(np.concatenate(loss0))), loss_fit=float(np.mean(np.concatenate(loss1))))
        elif fm == "joint":
            bfw = bases[0]["b"]
            nb = len(bfw)
            q, kr = WIN.neighbor_of(bfw, nc)
            kn_b = torch.as_tensor(np.stack([bb["kn"] for bb in bases]), device=DEV)
            tv_b = torch.as_tensor(np.stack([bb["tv"] for bb in bases]), device=DEV)
            K0_b = torch.as_tensor(np.stack([bb["K0"] for bb in bases]), device=DEV)
            P_, K_ = torch.as_tensor(bfw[:, 0], device=DEV), torch.as_tensor(bfw[:, 1], device=DEV)
            Q_, KR_ = torch.as_tensor(q, device=DEV), torch.as_tensor(kr, device=DEV)
            coarse = T.CoarseT(nc, nc, s, loads=[])

            def to_glob(th):
                kv = kn_b * torch.exp(th[..., 0])
                tvv = tv_b + K0_b * th[..., 1]
                Kn = torch.zeros(nS, nc * nc, 8, device=DEV, dtype=torch.float64)
                Tv = torch.zeros_like(Kn)
                Kn[:, P_, K_] = kv
                Kn[:, Q_, KR_] = kv
                Tv[:, P_, K_] = tvv
                Tv[:, Q_, KR_] = tvv
                return Kn, Tv

            th = torch.zeros(nS, nb, 2, device=DEV, dtype=torch.float64, requires_grad=True)

            def closure():
                opt.zero_grad()
                Kn, Tv = to_glob(th)
                pen, _ = coarse.stability_penalty(Kn, Tv)
                lp = a.pen * pen.sum() / nS
                lp.backward(retain_graph=True)
                tot = float(lp)
                wbs = list(batches.values())
                for jb, wb in enumerate(wbs):
                    Kl, Tl = L2.gather_local(Kn.reshape(-1), Tv.reshape(-1), wb)
                    lk = (wb.residuals(wb.forward(Kl, Tl), wts) ** 2).sum() / nwin
                    lk.backward(retain_graph=(jb < len(wbs) - 1))
                    tot += float(lk)
                return torch.tensor(tot, dtype=torch.float64)

            with torch.no_grad():
                l0 = batch_loss(batches, *to_glob(th)) / nwin
            opt = torch.optim.LBFGS([th], lr=1.0, max_iter=a.iters, history_size=30, tolerance_grad=1e-12,
                                    tolerance_change=1e-14, line_search_fn="strong_wolfe")
            hist = [float(opt.step(closure)) for _ in range(a.outer)]
            with torch.no_grad():
                Kn, Tv = to_glob(th)
                l1 = batch_loss(batches, Kn, Tv) / nwin
                _, lam = coarse.stability_penalty(Kn, Tv)
            rec.update(loss_patch=l0, loss_fit=l1, hist=hist, min_lam=float(lam.min()))
            Kn, Tv = Kn.cpu().numpy(), Tv.cpu().numpy()
            for jj in range(nS):
                Kn_new[jj], Tv_new[jj] = Kn[jj], Tv[jj]
        else:   # allwin: private parameters for every bond inside each window, keep the central one
            l0s, l1s = [], []
            for kc, wb in batches.items():
                g = wb.g
                lat = g.lat
                inbox = g.layer >= 1
                uid = -np.ones((g.N, 8), np.int64)
                cnt = 0
                for p_ in range(g.N):
                    for kk in (0, 2, 4, 6):
                        qq = lat.nbr[p_, kk]
                        if lat.valid[p_, kk] and inbox[p_] and inbox[qq]:
                            uid[p_, kk] = cnt
                            uid[qq, kk ^ 1] = cnt
                            cnt += 1
                uid_t = torch.as_tensor(np.where(uid >= 0, uid, cnt), device=DEV)
                msk = torch.as_tensor(uid >= 0, device=DEV, dtype=torch.float64)
                Kb, Tb = L2.gather_local(flatKn, flatTv, wb)
                K0loc = wb.K0[:, None, None]
                thw = torch.zeros(wb.B, cnt + 1, 2, device=DEV, dtype=torch.float64, requires_grad=True)
                fpt = torch.as_tensor(g.free_p, device=DEV)

                def loc_params():
                    tt = thw[:, uid_t] * msk[None, :, :, None]
                    return Kb * torch.exp(tt[..., 0]), Tb + K0loc * tt[..., 1]

                def closure_w():
                    optw.zero_grad()
                    Kn, Tv = loc_params()
                    r = wb.residuals(wb.forward(Kn, Tv), wts)
                    lm = wb.group_lam_min(Kn, Tv, fpt)
                    loss = (r ** 2).sum() + a.tik * (thw[:, :cnt] ** 2).sum() + a.pen * (torch.relu(0.05 - lm) ** 2).sum()
                    loss.backward()
                    return loss

                with torch.no_grad():
                    l0s.append((wb.residuals(wb.forward(Kb, Tb), wts) ** 2).sum(1).cpu().numpy())
                optw = torch.optim.LBFGS([thw], lr=1.0, max_iter=a.iters, history_size=30, tolerance_grad=1e-12,
                                         tolerance_change=1e-14, line_search_fn="strong_wolfe")
                for _ in range(a.outer):
                    optw.step(closure_w)
                with torch.no_grad():
                    Kn, Tv = loc_params()
                    l1s.append((wb.residuals(wb.forward(Kn, Tv), wts) ** 2).sum(1).cpu().numpy())
                    write_back(wb, Kn[:, g.pA, g.k].cpu().numpy(), Tv[:, g.pA, g.k].cpu().numpy())
            rec.update(loss_patch=float(np.mean(np.concatenate(l0s))), loss_fit=float(np.mean(np.concatenate(l1s))))
        torch.cuda.synchronize()
        for jj in range(nS):
            Kn_all[j0 + jj], Tv_all[j0 + jj] = Kn_new[jj], Tv_new[jj]
        rec["fit_time"] = time.time() - t0
        log["batches"].append(rec)
        print("%s %s s=%d specimens %d-%d: window loss patch %.4e -> fit %.4e (%.1f s)%s" % (
            tag, a.split, s, bids[0], bids[-1], rec["loss_patch"], rec["loss_fit"], rec["fit_time"],
            (" min lam %.3f" % rec["min_lam"]) if "min_lam" in rec else ""), flush=True)
    log["t_fine_mean"] = float(np.mean(t_fine))
    log["t_fit_per_specimen"] = float(sum(r["fit_time"] for r in log["batches"]) / S)
    log["wall"] = time.time() - t_start
    np.savez(out_fn, Kn=Kn_all, Tv=Tv_all, ids=np.array(ids))
    json.dump(log, open(os.path.join(common.RES, "l2fit_%s_%s_s%d.json" % (tag, a.split, s)), "w"), indent=1)
    print("saved %s (fine targets %.2f s/specimen, fit %.2f s/specimen)" % (out_fn, log["t_fine_mean"],
                                                                         log["t_fit_per_specimen"]), flush=True)


if __name__ == "__main__":
    main()
