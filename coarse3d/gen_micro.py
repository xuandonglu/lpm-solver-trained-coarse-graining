"""Stage A (numerical environment, CPU): microstructures of every split + classical patch homogenisation (s = 2, 3) of the
evaluation splits.  usage: python gen_micro.py SPLIT [--workers 6]  ->  data/micro_SPLIT.npz
Splits: train (n=24, 1200), val (100), cal (20, calibration set of the linear correction), test_id (100),
test_elong (50, elongated aggregates, aspect 3-4), test_c8 (50, contrast 8), test_soft (50, contrast 0.5),
test_big (n=48, 20), test_huge (n=72, 4)."""
import argparse
import os
import time
import multiprocessing as mp
import numpy as np

SPLITS = {
    "train": dict(seed=11, n=24, S=1200, fam="train", patch=False),
    "val": dict(seed=12, n=24, S=100, fam="train", patch=True),
    "cal": dict(seed=13, n=24, S=20, fam="train", patch=True),
    "test_id": dict(seed=14, n=24, S=100, fam="train", patch=True),
    "test_elong": dict(seed=15, n=24, S=50, fam="elong", patch=True),
    "test_c8": dict(seed=16, n=24, S=50, fam="c8", patch=True),
    "test_soft": dict(seed=17, n=24, S=50, fam="soft", patch=True),
    "test_big": dict(seed=18, n=48, S=20, fam="train", patch=True),
    "test_huge": dict(seed=19, n=72, S=4, fam="train", patch=True),
}


def one(args):
    split, k = args
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import common  # noqa
    import micro3d
    import geom3d as G
    cfg = SPLITS[split]
    rng = np.random.default_rng([cfg["seed"], k])
    n = cfg["n"]
    ph, E, nu, info = micro3d.sample(n, rng, cfg["fam"])
    out = dict(phase=ph.astype(np.int8), contrast=info["contrast"], nu_i=info["nu_i"], phi=info["phi"],
               kind=info["kind"])
    if cfg["patch"]:
        import patch3d
        Cv = G.iso_C(E, nu)
        for s in (2, 3):
            t = {}
            out["patch_%d" % s] = patch3d.patch_C(Cv, n, s, timing=t).astype(np.float64)
            out["tpatch_%d" % s] = t["patch"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = SPLITS[a.split]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(a.split, k) for k in range(cfg["S"])], chunksize=2)
    data = {key: np.array([r[key] for r in res]) for key in res[0]}
    data["n"] = cfg["n"]
    np.savez_compressed("data/micro_%s.npz" % a.split, **data)
    kinds, cnt = np.unique(data["kind"], return_counts=True)
    print("%s: %d samples n=%d, %.0f s; phi %.3f-%.3f; kinds %s" % (a.split, cfg["S"], cfg["n"], time.time() - t0,
          data["phi"].min(), data["phi"].max(), dict(zip(kinds, cnt))), flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
