"""Stage B (torch environment, GPU): fine LPM reference (one particle per voxel) of every sample of a split for s = 2 and 3:
block means of displacement and stress (computeStress), block energy density, grip reactions, 9 load cases; PCG to
relative residual 1e-10 (torch_lpm3d, checked against lin3d/SuperLU to 1e-12).
usage: python gen_fine.py SPLIT [--bs 8]  ->  data/fine_SPLIT.npz"""
import argparse
import os
import time
import numpy as np
import torch
import geom3d as G
import torch_lpm3d as T


def fine_params(lat, ph, contrast, nu_i):
    E = np.where(ph == 1, contrast, 1.0)
    nu = np.where(ph == 1, nu_i, 0.2)
    Cv = G.iso_C(E, nu)
    return G.bond_params_two_end_mean(lat, G.sc_params_from_C(Cv, 0.5)), Cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    d = dict(np.load("data/micro_%s.npz" % a.split, allow_pickle=True))
    n = int(d["n"])
    S = len(d["phase"])
    bs = a.bs if n <= 24 else (2 if n <= 48 else 1)
    lat = G.LatticeGeom(n, n, n, 1.0)
    out = {}
    t_all = time.time()
    for s in (2, 3):
        chunks = [G.LOADS] if n <= 48 else [G.RING, ["tb_tz", "tb_sx"], ["lr_tx"]]
        refs = [T.FineRefT(n, s, loads=ld) for ld in chunks]
        nc = n // s
        U = np.zeros((S, 9, nc ** 3, 3), np.float32)
        SG = np.zeros((S, 9, nc ** 3, 6), np.float32)
        W = np.zeros((S, 9, nc ** 3), np.float32)
        F = np.zeros((S, 9))
        its = []
        tt = []
        for i0 in range(0, S, bs):
            idx = range(i0, min(S, i0 + bs))
            KT = [fine_params(lat, d["phase"][i], d["contrast"][i], d["nu_i"][i])[0] for i in idx]
            Kn = torch.as_tensor(np.array([k for k, _ in KT]), device="cuda")
            Tv = torch.as_tensor(np.array([t for _, t in KT]), device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            l0 = 0
            for fr in refs:
                r = fr.run(Kn, Tv, tol=1e-10)
                L = len(fr.loads)
                U[i0:i0 + len(idx), l0:l0 + L] = r["u"].cpu().numpy()
                SG[i0:i0 + len(idx), l0:l0 + L] = r["sig"].cpu().numpy()
                W[i0:i0 + len(idx), l0:l0 + L] = r["w"].cpu().numpy()
                F[i0:i0 + len(idx), l0:l0 + L] = r["F"].cpu().numpy()
                its.append(r["iters"])
                l0 += L
            torch.cuda.synchronize()
            tt.append((time.time() - t0) / len(idx))
        out.update({"u_%d" % s: U, "sig_%d" % s: SG, "w_%d" % s: W, "F_%d" % s: F,
                    "iface_%d" % s: np.array([G.interface_cells(d["phase"][i], n, s) for i in range(S)]),
                    "iters_%d" % s: np.array(its), "tgpu_%d" % s: np.array(tt)})
        print("%s s=%d: %d samples, PCG its %d-%d, GPU %.3f s/sample" % (a.split, s, S, min(its), max(its), np.mean(tt)),
              flush=True)
    out["n"] = n
    np.savez_compressed("data/fine_%s.npz" % a.split, **out)
    print("%s done in %.0f s" % (a.split, time.time() - t_all), flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
