"""Converged L2 ('l2jc_m1'): the SAME objective as L2 (joint, m = 1, KUBC, stored local fine targets, window-normalised
residuals, stability penalty), but
L-BFGS restarted every 50 iterations until the window loss drops < 0.5 % over a restart (max MAXOUTER x 50), either
from patch or (--warm) continuing from the stored L2 parameters (4 x 50 iterations in one L-BFGS run, history kept,
which stops early).
usage: python l2conv_full.py S [--n1 32 --bs 8 --warm --maxouter 4]  -> results/params_l2jc_m1_test_id_sS.npz, results/l2fit_l2jc_m1_test_id_sS.json
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import numpy as np
import alg as A
import l2fine as LF
import l2run as LR
import l2torch as L2
import evalm as EV
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s", type=int)
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxouter", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    s = a.s
    sp = EV.Split("test_id")
    d, n = sp.d, sp.n
    nc = n // s
    G = LF.bond_groups(nc)
    geo = {k: L2.WinGeom(k, 1, s) for k in LF.KS}
    Kn_all, Tv_all, log = [], [], dict(s=s, tol=a.tol, maxouter=a.maxouter, batches=[])
    t_all = time.time()
    for b0 in range(0, a.n1, a.bs):
        ids = list(range(b0, min(a.n1, b0 + a.bs)))
        bases, dls = [], {0: [], 4: []}
        for jl, i in enumerate(ids):
            tg, _ = LR.load_targets("test_id", s, 1, i)
            bases.append(LR.spec_base(d, i, s))
            for kc, ks in ((0, (0, 2)), (4, (4, 6))):
                for k in ks:
                    d0 = L2.window_data(geo[k], G[k], nc, s, 1, tg[k], i_spec=jl)
                    d0["fixf"] = np.zeros((len(G[k]), 3, geo[k].nf))
                    dls[kc].append(A.canon_bc(d0, k, geo[k], geo[kc]))
        batches = [A.WinBatchBC(geo[kc], dls[kc], nc, norm="win") for kc in (0, 4)]
        th = np.zeros((len(ids), len(bases[0]["b"]), 2))
        if a.warm:
            z = np.load(os.path.join(common.RES, "params_l2j_m1_test_id_s%d.npz" % s))
            th = np.stack([A.theta_from(bases[jl], z["Kn"][i], z["Tv"][i]) for jl, i in enumerate(ids)])
        hist, t0 = [], time.time()
        for o in range(a.maxouter):
            th, Kn, Tv, l0, l1, _, lam = A.joint_fit(batches, bases, nc, s, th, 50, 1)
            if not hist:
                hist.append(l0)
            hist.append(l1)
            if o >= (0 if a.warm else 3) and hist[-2] - hist[-1] < a.tol * hist[-2]:
                break
        Kn_all.append(Kn); Tv_all.append(Tv)
        log["batches"].append(dict(ids=ids, hist=hist, n_iter=50 * (len(hist) - 1) + (200 if a.warm else 0), min_lam=lam, fit_time=time.time() - t0))
        print("s=%d ids %d-%d: window loss %s (%d iters, %.0fs)" % (s, ids[0], ids[-1], " ".join("%.4f" % h for h in hist),
              50 * (len(hist) - 1), time.time() - t0), flush=True)
        del batches
        A.torch.cuda.empty_cache()
    log["t_fit_per_specimen"] = (time.time() - t_all) / a.n1
    np.savez(os.path.join(common.RES, "params_l2jc_m1_test_id_s%d.npz" % s), Kn=np.concatenate(Kn_all), Tv=np.concatenate(Tv_all))
    json.dump(log, open(os.path.join(common.RES, "l2fit_l2jc_m1_test_id_s%d.json" % s), "w"), indent=0)
    print("L2C_DONE s=%d" % s, flush=True)


if __name__ == "__main__":
    main()
