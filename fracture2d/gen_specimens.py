"""Critical stretches of the classical coarse models for the plates of a split.
  python gen_specimens.py SPLIT [NMAX]   weakest-link (naive) and patch-localised (classical) critical stretch per
                                         coarse bond, s = 2, 3, 4 -> data/crit_SPLIT.npz
The plates are the microstructures of ../coarse2d/data/SPLIT_n48.npz (96 x 96 for test_big), written by
../coarse2d/export_data.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import multiprocessing as mp
import common  # noqa
from common import i2data
import numpy as np

SS = (2, 3, 4)


def split_file(split):
    return i2data("%s_n%d.npz" % (split, 96 if split == "test_big" else 48))


def crit_one(args):
    split, k = args
    import cparams
    d = np.load(split_file(split), allow_pickle=True)
    n = int(d["n"])
    ph = d["phase"][k].astype(np.int64)
    Cpix = d["Cpix"][k]
    out = {}
    for s in SS:
        out["naive_%d_0" % s], out["naive_%d_1" % s] = cparams.naive_crit(Cpix, ph, n, n, s)
        t0 = time.perf_counter()
        out["patch_%d_0" % s], out["patch_%d_1" % s] = cparams.patch_crit(Cpix, ph, n, n, s)
        out["tpcrit_%d" % s] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    split = sys.argv[1]
    t0 = time.time()
    d = np.load(split_file(split), allow_pickle=True)
    N = len(d["phase"]) if len(sys.argv) < 3 else min(int(sys.argv[2]), len(d["phase"]))
    with mp.Pool(6) as pool:
        res = pool.map(crit_one, [(split, k) for k in range(N)], chunksize=1)
    data = {key: np.array([r[key] for r in res]) for key in res[0]}
    np.savez_compressed("data/crit_%s.npz" % split, **data)
    print("critical stretches %s: %d plates in %.0f s" % (split, N, time.time() - t0), flush=True)
