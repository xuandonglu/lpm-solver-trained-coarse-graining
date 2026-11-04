"""Patch homogenisation (with voids) of every cube of a set, s = 2, 3, pore treatments a/b (numerical environment, numba).
usage: python -I gen_patch_v1.py SET [SET ...] [--opts ab] [--s 2,3]  ->  data/patch_SET_{opt}.npz (C_s (S, Nc, 6, 6),
       tpatch_s (S,) wall time per cube)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import geom3d as G  # noqa: E402
import patch_v as PV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--opts", default="ab")
    ap.add_argument("--s", default="2,3")
    a = ap.parse_args()
    PV.patch_C_v(G.iso_C(np.ones(216), np.full(216, 0.2)), np.ones(216, bool), 6, 2)     # compile
    PV.patch_C_v(G.iso_C(np.ones(729), np.full(729, 0.2)), np.ones(729, bool), 9, 3)
    for st in a.sets:
        for opt in a.opts:
            mf = np.load(common.data("mat_%s_%s.npz" % (st, opt)))
            n = int(mf["n"])
            S = len(mf["solid"])
            out = {}
            for s in [int(x) for x in a.s.split(",")]:
                Cs, ts = [], []
                for i in range(S):
                    t = {}
                    Cs.append(PV.patch_C_v(G.iso_C(mf["E"][i], mf["nu"][i]), mf["solid"][i], n, s, timing=t))
                    ts.append(t["patch"])
                out["C_%d" % s] = np.array(Cs)
                out["tpatch_%d" % s] = np.array(ts)
                print("%s opt %s s=%d: %d cubes, %.2f s/cube" % (st, opt, s, S, np.mean(ts)), flush=True)
            np.savez_compressed(common.data("patch_%s_%s.npz" % (st, opt)), **out)


if __name__ == "__main__":
    main()
