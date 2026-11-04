"""Generate fine-LPM training/test data for the coarse-graining model (square lattice, elastic).

For every microstructure and s in (2, 3, 4): fine reference (lin2d) block-averaged fields for the 6 load cases,
plus Voigt and patch-homogenised cell parameters (baselines).
usage: python export_data.py SPLIT NSAMP [--n 48] [--workers 8]
splits: train, val, test_id, test_phi (phi = 0.5), test_c7 (contrast 7.5), test_c15 (contrast 15),
        test_soft (soft inclusions, contrast 0.2), test_big (n = 96)
"""
import argparse
import os
import time
import multiprocessing as mp
import common  # noqa
import numpy as np

SPLITS = {
    "train": dict(seed=1, phi=(0.15, 0.40), contrast=[5.0, 10.0]),
    "val": dict(seed=2, phi=(0.15, 0.40), contrast=[5.0, 10.0]),
    "test_id": dict(seed=3, phi=(0.15, 0.40), contrast=[5.0, 10.0]),
    "test_phi": dict(seed=4, phi=(0.50, 0.50), contrast=[5.0, 10.0]),
    "test_c7": dict(seed=5, phi=(0.15, 0.40), contrast=[7.5]),
    "test_c15": dict(seed=6, phi=(0.15, 0.40), contrast=[15.0]),
    "test_soft": dict(seed=7, phi=(0.15, 0.40), contrast=[0.2]),
    "test_big": dict(seed=8, phi=(0.15, 0.40), contrast=[5.0, 10.0]),
}
SS = (2, 3, 4)


def one(args):
    split, k, n = args
    import lin2d
    import micro
    import cg
    cfg = SPLITS[split]
    rng = np.random.default_rng([cfg["seed"], k])
    ph, info = micro.random_two_phase(n, n, rng, corr_range=(1.5, 6.0), phi_range=cfg["phi"])
    c = float(rng.choice(cfg["contrast"]))
    E = np.where(ph == 1, c, 1.0)
    nu = np.where(ph == 1, 0.2, 0.3)
    Cpix = lin2d.iso_C(E, nu)
    out = dict(phase=ph.astype(np.int8), Cpix=Cpix, contrast=c, phi=ph.mean(), kind=info[0], corr=info[1])
    times = {}
    for s in SS:
        t0 = time.time()
        ref = cg.FineRef(Cpix, n, n, s, phase=ph)
        times["fine%d" % s] = time.time() - t0
        for ld in cg.LOADS:
            r = ref.res[ld]
            out["u_%d_%s" % (s, ld)] = r["u"]
            out["sig_%d_%s" % (s, ld)] = r["sig"]
            out["w_%d_%s" % (s, ld)] = r["w"]
            out["F_%d_%s" % (s, ld)] = np.full(2, np.nan) if r["F"] is None else r["F"]
            out["W_%d_%s" % (s, ld)] = r["W"]
            out["wmax_%d_%s" % (s, ld)] = r["wmax"]
        out["iface_%d" % s] = ref.iface
        out["voigt_%d" % s] = cg.cell_params_from_C6(cg.voigt_cells(Cpix, n, n, s))
        t0 = time.time()
        out["patch_%d" % s] = cg.cell_params_from_C6(cg.patch_cells(Cpix, n, n, s))
        times["patch%d" % s] = time.time() - t0
    out["times"] = np.array([times[k] for k in sorted(times)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("nsamp", type=int)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    n = a.n or (96 if a.split == "test_big" else 48)
    os.environ["NUMBA_NUM_THREADS"] = "1"
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(a.split, k, n) for k in range(a.nsamp)], chunksize=4)
    keys = res[0].keys()
    data = {}
    for key in keys:
        vals = [r[key] for r in res]
        data[key] = np.array(vals)
    data["n"] = n
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/%s_n%d.npz" % (a.split, n), **data)
    print("%s: %d samples, n=%d, %.1f s; mean phi %.3f; mean times %s" % (
        a.split, a.nsamp, n, time.time() - t0, data["phi"].mean(), np.round(data["times"].mean(0), 3)))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
