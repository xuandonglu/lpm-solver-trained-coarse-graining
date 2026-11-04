"""In-loop brittle fracture: fine LPM vs coarse LPM (plate in tension, tb_ty grips), held-out microstructures.

Coarse variants (elastic bond parameters + critical bond stretch):
  naive     : Voigt elastic        + weakest-link crit (min fine crit in the bond region)
  classical : patch elastic        + patch-localisation crit (export_frac.patch_strength_cells)
  ml_patch  : ML elastic           + patch-localisation crit          (ablation)
  ml        : ML elastic           + ML crit (numpy forward of fracnet)
Metrics vs fine: strain at the first bond break (exact, analytic first event), peak force, strain at peak, crack path
(chamfer distance and crack-profile distance in coarse cells) at the end of the run (F < 0.3 peak), wall time.
usage: python eval_frac.py SPLIT NSAMP [--deps 5e-6]
"""
import argparse
import json
import os
import time
import multiprocessing as mp
import common  # noqa
import numpy as np

XI_M, XI_I, XI_INT = 1.0e-3, 2.0e-3, 0.6e-3


def one(args):
    split, i, deps, fine_only = args
    import lin2d
    import cg
    import brittle
    import coarse_params as CP
    d = np.load("data/%s_n%d.npz" % (split, 96 if split == "test_big" else 48), allow_pickle=True)
    F = np.load("data/frac_%s.npz" % split)
    n = int(d["n"])
    ph = d["phase"][i].astype(np.int64)
    Cpix = d["Cpix"][i]
    lat = lin2d.Lattice2D("square", n, n, 1.0)
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cpix))
    crit = brittle.crit_from_phase(lat, ph, XI_M, XI_I, XI_INT)
    out = {}
    for s in (2, 3, 4):
        nc = n // s
        cache = "results/frac_fine/%s_%d_s%d.npz" % (split, i, s)
        import os
        if os.path.exists(cache):
            z = np.load(cache)
            fine = {k: float(z[k]) for k in ("eps1", "peak", "eps_peak", "nb", "time", "passes")}
            pf, rfF, rfe = z["pf"], z["F"], z["eps"]
        else:
            t0 = time.perf_counter()
            rf = brittle.run_tension(lat, Kn, Tv, crit, s, deps=deps, nbreak=1)
            tf = time.perf_counter() - t0
            pf = brittle.broken_midpoints(lat, rf["events"])
            ipk = int(np.argmax(rf["F"]))
            fine = dict(eps1=rf["eps_first_analytic"], peak=rf["peak"], eps_peak=rf["eps"][ipk], nb=int(rf["nb"][-1]),
                        time=tf, passes=len(rf["eps"]))
            rfF, rfe = rf["F"], rf["eps"]
            np.savez(cache, pf=pf, F=rfF, eps=rfe, **fine)
        if fine_only:
            out[s] = {"fine": fine}
            continue
        latc = cg.CoarseModel(nc, nc, s).lat
        el = {"voigt": cg.bond_params_from_cells(latc, d["voigt_%d" % s][i]),
              "patch": cg.bond_params_from_cells(latc, d["patch_%d" % s][i])}
        tml = {}
        el["ml"] = CP.ml_elastic(Cpix, n, s, "models/cgnet_s%d.npz" % s, timing=tml)
        cr = {"naive": CP.crit_slots(nc, F["naive_%d_0" % s][i], F["naive_%d_1" % s][i]),
              "patch": CP.crit_slots(nc, F["patch_%d_0" % s][i], F["patch_%d_1" % s][i])}
        cr["ml"] = CP.ml_crit(Cpix, n, s, "models/fracnet_s%d.npz" % s, F["naive_%d_0" % s][i], F["naive_%d_1" % s][i],
                              timing=tml)
        res = {"fine": fine}
        pcs = {}
        for name, (e, c) in {"naive": ("voigt", "naive"), "classical": ("patch", "patch"),
                             "ml_patch": ("ml", "patch"), "ml": ("ml", "ml")}.items():
            t0 = time.perf_counter()
            rc = brittle.run_tension(latc, el[e][0], el[e][1], cr[c], 1, deps=deps, nbreak=1)
            tc = time.perf_counter() - t0
            pc = brittle.broken_midpoints(latc, rc["events"])
            pcs["pc_" + name] = pc
            jpk = int(np.argmax(rc["F"]))
            pm = brittle.path_metrics(pf, pc, n * 1.0, s * 1.0)
            res[name] = dict(eps1=rc["eps_first_analytic"], peak=rc["peak"], eps_peak=rc["eps"][jpk],
                             nb=int(rc["nb"][-1]), time=tc, passes=len(rc["eps"]),
                             e_eps1=(rc["eps_first_analytic"] - fine["eps1"]) / fine["eps1"],
                             e_peak=(rc["peak"] - fine["peak"]) / fine["peak"],
                             e_eps_peak=(rc["eps"][jpk] - fine["eps_peak"]) / fine["eps_peak"], **pm)
        res["ml"]["pre_time"] = sum(tml.values())
        out[s] = res
        np.savez_compressed("results/frac_paths/%s_%d_s%d.npz" % (split, i, s), pf=pf, phase=ph, Ff=rfF,
                            ef=rfe, **pcs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("nsamp", type=int)
    ap.add_argument("--deps", type=float, default=5e-6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fine-only", action="store_true")
    a = ap.parse_args()
    import os
    os.makedirs("results/frac_paths", exist_ok=True)
    os.makedirs("results/frac_fine", exist_ok=True)
    os.environ["NUMBA_NUM_THREADS"] = "1"
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(a.split, i, a.deps, a.fine_only) for i in range(a.nsamp)], chunksize=1)
    if a.fine_only:
        print("fine-only done %d samples in %.0f s; mean fine time %s" % (a.nsamp, time.time() - t0,
              {s: round(np.mean([r[s]["fine"]["time"] for r in res]), 2) for s in (2, 3, 4)}))
        return
    json.dump(res, open("results/eval_frac_%s.json" % a.split, "w"), indent=1, default=float)
    print("done %d samples in %.0f s" % (a.nsamp, time.time() - t0))
    for s in (2, 3, 4):
        print("== s=%d" % s)
        for name in ("naive", "classical", "ml_patch", "ml"):
            rows = [r[s][name] for r in res]
            print("  %-10s |e_eps1| %.3f (signed %+.3f)  |e_peak| %.3f (signed %+.3f)  |e_eps_peak| %.3f  chamfer %.2f  prof %.2f  time %.2fs"
                  % (name, np.mean([abs(x["e_eps1"]) for x in rows]), np.mean([x["e_eps1"] for x in rows]),
                     np.mean([abs(x["e_peak"]) for x in rows]), np.mean([x["e_peak"] for x in rows]),
                     np.mean([abs(x["e_eps_peak"]) for x in rows]), np.nanmean([x["chamfer"] for x in rows]),
                     np.nanmean([x["prof"] for x in rows]), np.mean([x["time"] for x in rows])))
        print("  fine time %.2fs" % np.mean([r[s]["fine"]["time"] for r in res]))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
