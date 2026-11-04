"""Cost per specimen of every rung (numerical environment, numpy/scipy, ONE thread, same machine state for all; run when idle).

Timed per specimen (48 x 48 test specimens), all with the same lin2d + SuperLU solver family as the I2 patch method:
  fine      : fine LPM reference = what every rung tries to avoid (6 load cases, 3 factorisations, block means)
  coarse    : coarse LPM, 6 load cases (the same for every rung once the bond parameters exist)
  voigt     : Voigt cell parameters -> bonds
  patch     : I2 patch homogenisation (cg.patch_cells, all windows in one sparse factorisation)
  l2fine_mM : L2 local fine solves around every bond, margin M (l2fine.fine_targets)
  cnn_cpu   : I2 network, numpy forward (coarse_params.ml_elastic)
The joint fit of local window calibration is timed by l2run.py fit (results/l2fit_*.json); make_report.py
combines both into the cost per specimen.
usage: python bench_cost.py [--nspec 6] [--margins 1 2 3]
"""
import argparse
import json
import os
import time
import numpy as np
import common
import cg
import coarse_params as CP
import l2fine as LF


def tmin(f, rep=3):
    best = 1e9
    out = None
    for _ in range(rep):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nspec", type=int, default=6)
    ap.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rep", type=int, default=3)
    a = ap.parse_args()
    d = np.load(common.split_file("test_id"))
    n = int(d["n"])
    res = {}
    for s in (2, 3, 4):
        nc = n // s
        cm = cg.CoarseModel(nc, nc, s)
        rows = []
        for i in range(a.nspec):
            C = d["Cpix"][i]
            r = {}
            r["fine"], _ = tmin(lambda: cg.FineRef(C, n, n, s), a.rep)
            r["voigt"], (Kn, Tv) = tmin(lambda: cg.bond_params_from_cells(
                cm.lat, cg.cell_params_from_C6(cg.voigt_cells(C, n, n, s))), a.rep)
            r["coarse"], _ = tmin(lambda: cm.run(Kn, Tv), a.rep)
            r["patch"], _ = tmin(lambda: cg.bond_params_from_cells(
                cm.lat, cg.cell_params_from_C6(cg.patch_cells(C, n, n, s))), a.rep)
            path = os.path.join(common.I2MODELS, "cgnet_s%d.npz" % s)
            CP.ml_elastic(C, n, s, path)                         # load the weights once
            r["cnn_cpu"], _ = tmin(lambda: CP.ml_elastic(C, n, s, path), a.rep)
            for m in a.margins:
                r["l2fine_m%d" % m], _ = tmin(lambda: LF.fine_targets(C, n, s, m), 1 if m >= 3 else a.rep)
            rows.append(r)
            print("s=%d specimen %d: %s" % (s, i, {k: round(v, 4) for k, v in r.items()}), flush=True)
        res["s%d" % s] = {k: float(np.median([r_[k] for r_ in rows])) for k in rows[0]}
        print("s=%d median: %s" % (s, {k: round(v, 4) for k, v in res["s%d" % s].items()}), flush=True)
    res["threads"] = os.environ.get("OMP_NUM_THREADS")
    json.dump(res, open(os.path.join(common.RES, "bench_cost_cpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
