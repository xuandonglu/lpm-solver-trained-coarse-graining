"""Fine LPM reference (one particle per voxel, voids removed) of every cube of a set, pore treatments a/b, s = 2 and 3
(torch environment, GPU): block means of u (solid particles), stress and energy density (voids = 0), grip reactions, the 9
load cases with block boundary conditions; block-Jacobi PCG to relative residual 1e-10 (torch_v1, checked against
lin3d with the void sites removed).
usage: python -I gen_fine_v1.py SET [--opts ab] [--s 2,3]  ->  data/fine_SET_{opt}.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import geom3d as G  # noqa: E402
import torch_v1 as TV  # noqa: E402

torch.set_num_threads(4)


def fine_params(lat, Cv):
    return G.bond_params_two_end_mean(lat, G.sc_params_from_C(Cv, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--opts", default="ab")
    ap.add_argument("--s", default="2,3")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    for opt in a.opts:
        mf = np.load(common.data("mat_%s_%s.npz" % (a.set, opt)))
        n = int(mf["n"])
        S = len(mf["solid"])
        bs = a.bs if n <= 24 else (2 if n <= 48 else 1)
        lat = G.LatticeGeom(n, n, n, 1.0)
        out = {}
        t_all = time.time()
        for s in [int(x) for x in a.s.split(",")]:
            chunks = [G.LOADS] if n <= 48 else [G.RING, ["tb_tz", "tb_sx"], ["lr_tx"]]
            refs = [TV.FineRefTV(n, s, loads=ld) for ld in chunks]
            nc = n // s
            U = np.zeros((S, 9, nc ** 3, 3), np.float32)
            SG = np.zeros((S, 9, nc ** 3, 6), np.float32)
            W = np.zeros((S, 9, nc ** 3), np.float32)
            F = np.zeros((S, 9))
            its, rels, tt = [], [], []
            for i0 in range(0, S, bs):
                idx = range(i0, min(S, i0 + bs))
                KT = [fine_params(lat, G.iso_C(mf["E"][i], mf["nu"][i])) for i in idx]
                Kn = torch.as_tensor(np.array([k for k, _ in KT]), device="cuda")
                Tv = torch.as_tensor(np.array([t for _, t in KT]), device="cuda")
                sol = torch.as_tensor(np.array([mf["solid"][i] for i in idx]), device="cuda")
                torch.cuda.synchronize()
                t0 = time.time()
                l0 = 0
                for fr in refs:
                    r = fr.run(Kn, Tv, sol, tol=1e-10)
                    L = len(fr.loads)
                    U[i0:i0 + len(idx), l0:l0 + L] = r["u"].cpu().numpy()
                    SG[i0:i0 + len(idx), l0:l0 + L] = r["sig"].cpu().numpy()
                    W[i0:i0 + len(idx), l0:l0 + L] = r["w"].cpu().numpy()
                    F[i0:i0 + len(idx), l0:l0 + L] = r["F"].cpu().numpy()
                    its.append(r["iters"])
                    rels.append(r["relres"])
                    l0 += L
                torch.cuda.synchronize()
                tt.append((time.time() - t0) / len(idx))
            out.update({"u_%d" % s: U, "sig_%d" % s: SG, "w_%d" % s: W, "F_%d" % s: F, "iters_%d" % s: np.array(its),
                        "tgpu_%d" % s: np.array(tt)})
            print("%s opt %s s=%d: %d cubes, PCG its %d-%d, max relres %.1e, GPU %.3f s/cube" % (
                a.set, opt, s, S, min(its), max(its), max(rels), np.mean(tt)), flush=True)
        out["n"] = n
        np.savez_compressed(common.data("fine_%s_%s.npz" % (a.set, opt)), **out)
        print("%s opt %s done in %.0f s" % (a.set, opt, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
