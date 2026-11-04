"""Force-strain curves and broken-bond positions of one test plate: fine LPM and the three coarse models
(strength-first, energy-first, proposed) at s = 2, 3, 4.
usage: python plate_curves.py PLATE  -> results/plate_curves.npz"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
import multiprocessing as mp
import common  # noqa
import numpy as np


def job(args):
    test, split, i, s = args
    import eval2 as E
    import cg, soft, spec, cparams
    from rhofeat import features6
    from cgnet_geom import bond_gather_index
    PE = json.load(open("results/phase_energy.json"))
    d = np.load(E.split_file(split), allow_pickle=True)
    n = int(d["n"])
    ph = d["phase"][i].astype(np.int64)
    Cpix = d["Cpix"][i]
    cr = np.load("data/crit_%s.npz" % split)
    mu, kappa = spec.kf_elastic(Cpix)
    lat, Kn, Tv, crit = spec.fine_model(Cpix, ph, n, n)
    P, K, Q, KR = soft.undirected(lat)
    out = {}
    rf = E.run_one(test, lat, Kn, Tv, soft.Law(crit[P, K]), s, n, n, mu, kappa)
    out["fine_lam"], out["fine_F"] = rf["lam"], rf["F"]
    out["fine_pm"] = soft.broken_midpoints(lat, rf)
    nc = n // s
    latc = cparams.coarse_lat(n, n, s)
    Kp, Tp = cparams.cells_elastic(latc, d["patch_%d" % s][i])
    c_patch = cparams.to_und(latc, nc, nc, cr["patch_%d_0" % s][i], cr["patch_%d_1" % s][i])
    Kml, Tml = cparams.ml_elastic(Cpix, n, n, s, common.i2model("cgnet_s%d.npz" % s))
    o0, o1 = cparams.net_out(Cpix, n, n, s, common.i2model("fracnet_s%d.npz" % s))
    c_ml = cparams.to_und(latc, nc, nc, cr["naive_%d_0" % s][i] * np.exp(o0[:, 0]), cr["naive_%d_1" % s][i] * np.exp(o1[:, 0]))
    b0, b1 = cparams.fwd(nc, nc)
    X6 = features6(Cpix, ph, n, n, s).astype(np.float64)
    beta = float(np.load("models/rhonet_s%d.npz" % s)["beta"])
    q0, q1 = cparams.net("models/rhonet_s%d.npz" % s)(X6, bond_gather_index(b0, nc, nc), bond_gather_index(b1, nc, nc))
    rho = np.exp(cparams.to_und(latc, nc, nc, E.softplus(q0[:, 0] + beta), E.softplus(q1[:, 0] + beta)))
    law_c2, _ = E.classical2P_law(latc, nc, nc, s, c_patch, Kp, ph, n, n, float(d["contrast"][i]), PE)
    V = {"classical": (Kp, Tp, soft.Law(c_patch)), "classical2P": (Kp, Tp, law_c2),
         "ml2P": (Kml, Tml, soft.Law(c_ml, rho))}
    for name, (a, b, law) in V.items():
        r = E.run_one(test, latc, a, b, law, 1, n, n, mu, kappa, lam_max=(30 * rf["eps1"] if test == "kf" else np.inf))
        out[name + "_lam"], out[name + "_F"] = r["lam"], r["F"]
        out[name + "_pm"] = soft.broken_midpoints(latc, r)
    out["phase"] = ph
    return {"%s_s%d_%s" % (test, s, k): v for k, v in out.items()}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    ip = int(sys.argv[1])
    jobs = [("plate", "test_id", ip, s) for s in (2, 3, 4)]
    with mp.Pool(3) as pool:
        R = pool.map(job, jobs)
    allv = {}
    for r in R:
        allv.update(r)
    np.savez_compressed("results/plate_curves.npz", **allv)
    print("saved", len(allv))
