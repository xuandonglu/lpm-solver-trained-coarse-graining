"""Inference worker of the 3D timing table (torch environment, torch), driven over stdin/stdout by bench3d.py so
that inference is interleaved with the solver timings of the numerical environment (the I8b / V1 protocol).

Network = the trained I8 s = 2 BondNet (synthetic two-phase) or the V1 'syn_a' / 'syn_b' network (real cement paste);
weights are read-only.  Two inference variants of the SAME weights:
  cpu  I8b engineered forward (infer_fast.FastBondNet: oneDNN, fused GELU), torch intra-op threads = LPMCG_THREADS (6)
  gpu  the I8 BondNet forward on CUDA (fp32, TF32 off); features uploaded, raw outputs downloaded, synchronised
Timed per call: t_feat (voxel features + materials, numpy), t_fwd (forward incl. upload / download for gpu),
t_post (Voigt bases, Kn = Kn_V exp(o1), Tv = Tv_V + Kn_V o2, scatter to (N, 18); V1: mask + minimum stiffness).
protocol: {"op": "prep", "kind": "i8", "split": S} | {"op": "prep", "kind": "v1", "set": SET, "opt": a|b, "tag": T}
          {"op": "run", "i": i, "dev": "cpu"|"gpu", "save": path|null}     {"op": "quit"}
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import common  # noqa: E402,F401  (thread budget before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import infer_fast as IF  # noqa: E402
import net3d as NT  # noqa: E402
import geom3d as G  # noqa: E402

torch.set_num_threads(int(common.NT))
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
S = 2
KMIN = 1e-3                     # cement3d/train_v1.KMIN


def voxel_features_v(E, nu, solid, n, s):
    """copy of cement3d/train_v1.voxel_features_v: I8 voxel features with internal voids (void = padding look)."""
    solid = np.asarray(solid, bool)
    mu = np.where(solid, E / (2 * (1 + nu)), 1.0)
    lam = np.where(solid, E * nu / ((1 + nu) * (1 - 2 * np.where(solid, nu, 0.0))), 1.0)
    kref = mu[solid].mean()
    P = 2 * s
    X = np.zeros((3, n + 2 * P, n + 2 * P, n + 2 * P), np.float32)
    X[2] = 1.0
    X[0, P:P + n, P:P + n, P:P + n] = np.where(solid, np.log(mu / kref), 0.0).reshape(n, n, n)
    X[1, P:P + n, P:P + n, P:P + n] = np.where(solid, (lam - mu) / (18 * mu), 0.0).reshape(n, n, n)
    X[2, P:P + n, P:P + n, P:P + n] = (~solid).reshape(n, n, n)
    return X


class Ctx:
    def __init__(self, cmd):
        self.kind = cmd["kind"]
        if self.kind == "i8":
            self.m = dict(np.load(common.i8data("micro_%s.npz" % cmd["split"]), allow_pickle=True))
            self.n = int(self.m["n"])
            fn = os.path.join(common.I8, "models", "bondnet_s2.pt")
        else:
            self.mf = np.load(common.v1data("mat_%s_%s.npz" % (cmd["set"], cmd["opt"])))
            self.n = int(self.mf["n"])
            fn = os.path.join(common.V1, "models", "bondnet_s2_%s.pt" % cmd["tag"])
        self.nc = self.n // S
        self.net = NT.BondNet(S)
        self.net.load_state_dict(torch.load(fn, map_location="cpu"))
        self.net.eval()
        self.fast = IF.FastBondNet(self.net, chunk=4096, gemm="fused", fuse_conv=True)
        self.gnet = None
        if torch.cuda.is_available():
            self.gnet = NT.BondNet(S).cuda()
            self.gnet.load_state_dict(torch.load(fn, map_location="cuda"))
            self.gnet.eval()
        self.post = IF.Post(self.n, S, self.net)
        lat = self.post.lat_c
        self.nbr, self.gvalid = lat.nbr, lat.valid

    def features(self, i):
        if self.kind == "i8":
            ph, c, nu_i, Cv = IF.material(self.m, i)
            return NT.voxel_features(ph, c, nu_i, self.n, S), Cv, None
        E, nu, sol = self.mf["E"][i], self.mf["nu"][i], self.mf["solid"][i]
        return voxel_features_v(E, nu, sol, self.n, S), G.iso_C(E, nu), self.mf["cell_%d" % S][i]

    def params(self, i, dev):
        t0 = time.perf_counter()
        X, Cv, cell = self.features(i)
        t1 = time.perf_counter()
        if dev == "cpu":
            with torch.inference_mode():
                outs = self.fast.forward(torch.as_tensor(X)[None], self.nc)
        else:
            with torch.inference_mode():
                o = self.gnet(torch.as_tensor(X)[None].cuda(), self.nc)
                outs = [x.cpu() for x in o]
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        Kn, Tv = self.post.scatter(outs, self.post.bases(Cv))
        if cell is not None:
            v = self.gvalid & cell[:, None] & cell[np.where(self.nbr >= 0, self.nbr, 0)]
            Kn = Kn * v
            Tv = Tv * v
            km = KMIN * Kn.sum() / max(v.sum(), 1)
            Kn = np.where(v, np.maximum(Kn, km), Kn)
        t3 = time.perf_counter()
        return Kn, Tv, dict(t_feat=t1 - t0, t_fwd=t2 - t1, t_post=t3 - t2)


def main():
    ctx = None
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "quit":
            break
        if op == "prep":
            ctx = Ctx(cmd)
            for dev in ("cpu", "gpu") if ctx.gnet is not None else ("cpu",):
                ctx.params(0, dev)
                ctx.params(0, dev)
            print(json.dumps(dict(ok=1, n=ctx.n, threads=torch.get_num_threads(), gpu=ctx.gnet is not None,
                                  gpu_name=torch.cuda.get_device_name(0) if ctx.gnet is not None else None)),
                  flush=True)
        elif op == "run":
            Kn, Tv, t = ctx.params(int(cmd["i"]), cmd.get("dev", "cpu"))
            if cmd.get("save"):
                np.savez(cmd["save"], Kn=Kn, Tv=Tv)
            print(json.dumps(t), flush=True)
        else:
            print(json.dumps(dict(error="unknown op %s" % op)), flush=True)


if __name__ == "__main__":
    main()
