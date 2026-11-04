"""In-loop elastic evaluation on held-out sets (numerical environment, numpy only): coarse LPM (lin2d) with Voigt / patch / ML bond
parameters (ML = numpy forward pass of the trained network) vs the fine LPM reference (block-averaged).
usage: python eval_elastic.py S SPLIT [SPLIT ...]   -> results/eval_np_sS.json
"""
import json
import os
import sys
import time
import common  # noqa
import numpy as np
import cg
import coarse_params as CP

KEYS = ["G", "u", "sig", "sig_if", "w_if", "w99"]


def ref_from_npz(d, i, s, n):
    class R:
        pass
    r = R()
    r.ncx = r.ncy = n // s
    r.s, r.h, r.loads = s, 1.0, cg.LOADS
    r.res = {}
    for ld in cg.LOADS:
        F = d["F_%d_%s" % (s, ld)][i]
        r.res[ld] = dict(u=d["u_%d_%s" % (s, ld)][i], sig=d["sig_%d_%s" % (s, ld)][i], w=d["w_%d_%s" % (s, ld)][i],
                         F=None if np.isnan(F).all() else F, W=d["W_%d_%s" % (s, ld)][i])
    r.iface = d["iface_%d" % s][i]
    return r


def summ(ms):
    out = {}
    for grp, loads in (("ring", cg.LOADS[:3]), ("plate", cg.LOADS[3:])):
        for k in KEYS:
            v = [abs(m[ld][k]) for m in ms for ld in loads if k in m[ld]]
            out["%s_%s" % (grp, k)] = float(np.mean(v))
            out["%s_%s_p90" % (grp, k)] = float(np.quantile(v, 0.9))
        out["%s_G_signed" % grp] = float(np.mean([m[ld]["G"] for m in ms for ld in loads]))
    return out


def main():
    s = int(sys.argv[1])
    res = {}
    for split in sys.argv[2:]:
        fn = "data/%s_n%d.npz" % (split, 96 if split == "test_big" else 48)
        d = dict(np.load(fn, allow_pickle=True))
        n = int(d["n"])
        nc = n // s
        cm = cg.CoarseModel(nc, nc, s)
        ms = {"voigt": [], "patch": [], "ml": []}
        tml = []
        for i in range(len(d["phi"])):
            ref = ref_from_npz(d, i, s, n)
            for base in ("voigt", "patch"):
                Kn, Tv = cg.bond_params_from_cells(cm.lat, d["%s_%d" % (base, s)][i])
                ms[base].append(cg.metrics(ref, cm.run(Kn, Tv)))
            t = {}
            Kn, Tv = CP.ml_elastic(d["Cpix"][i], n, s, "models/cgnet_s%d.npz" % s, timing=t)
            tml.append(t)
            ms["ml"].append(cg.metrics(ref, cm.run(Kn, Tv)))
        r = {k: summ(v) for k, v in ms.items()}
        r["ml_time"] = {k: float(np.mean([t[k] for t in tml])) for k in tml[0]}
        r["n_samples"] = len(d["phi"])
        res[split] = r
        print("== %s s=%d (%d samples)  ML preprocessing %s" % (split, s, len(d["phi"]),
                                                               {k: round(v, 4) for k, v in r["ml_time"].items()}))
        for k in r["ml"]:
            if k.endswith("p90"):
                continue
            print("  %-18s voigt %8.4f  patch %8.4f  ML %8.4f" % (k, r["voigt"][k], r["patch"][k], r["ml"][k]))
    json.dump(res, open("results/eval_np_s%d.json" % s, "w"), indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
