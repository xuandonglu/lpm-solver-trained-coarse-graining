"""Evaluate rungs of the credit ladder with identical measures (numerical environment; numpy/scipy only).

usage: python eval_rungs.py SPLIT S RUNG [RUNG ...] [--nmax K]
  RUNG = voigt | patch | cnn | <name>   (<name>: bond parameters in results/params_<name>_<SPLIT>_s<S>.npz, arrays Kn, Tv
                                         of shape (n_samples, Nc, 8), written by the L2 / L2b / L3 / L4 scripts)
-> results/eval_<RUNG>_<SPLIT>_s<S>.json  (summary + per-sample per-load metrics)
"""
import argparse
import json
import os
import time
import numpy as np
import common  # noqa
import cg
import coarse_params as CP
import evalm as EV


def params_source(sp, s, rung):
    nc = sp.n // s
    if rung in ("voigt", "patch"):
        lat = EV.coarse_model(sp.n, s).lat
        return lambda i: cg.bond_params_from_cells(lat, sp.cells(rung, i, s))
    if rung == "cnn":
        path = os.path.join(common.I2MODELS, "cgnet_s%d.npz" % s)
        return lambda i: CP.ml_elastic(sp.d["Cpix"][i], sp.n, s, path)
    z = np.load(os.path.join(common.RES, "params_%s_%s_s%d.npz" % (rung, sp.name, s)))
    Kn, Tv = z["Kn"], z["Tv"]
    assert Kn.shape[1] == nc * nc
    return lambda i: (Kn[i], Tv[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("s", type=int)
    ap.add_argument("rungs", nargs="+")
    ap.add_argument("--nmax", type=int, default=0)
    a = ap.parse_args()
    sp = EV.Split(a.split)
    S = sp.S if a.nmax == 0 else min(a.nmax, sp.S)
    for rung in a.rungs:
        t0 = time.time()
        src = params_source(sp, a.s, rung)
        ms = []
        for i in range(S):
            Kn, Tv = src(i)
            ms.append(EV.eval_sample(sp, i, a.s, Kn, Tv))
        sm = EV.summ(ms)
        out = dict(rung=rung, split=a.split, s=a.s, n_samples=S, summary=sm,
                   per_sample=[{ld: {k: float(v) for k, v in m[ld].items()} for ld in EV.LOADS} for m in ms])
        json.dump(out, open(os.path.join(common.RES, "eval_%s_%s_s%d.json" % (rung, a.split, a.s)), "w"), indent=0)
        print("%-22s s=%d n=%d (%.0fs) | plate G %.4f sig %.4f w_if %.4f | ring G %.4f sig %.4f w_if %.4f%s" % (
            rung, a.s, S, time.time() - t0, sm["plate_G"], sm["plate_sig"], sm["plate_w_if"], sm["ring_G"],
            sm["ring_sig"], sm["ring_w_if"],
            (" | R4 sig %.4f w_if %.4f" % (sm["R4_sig"], sm["R4_w_if"])) if "R4_sig" in sm else ""), flush=True)


if __name__ == "__main__":
    main()
