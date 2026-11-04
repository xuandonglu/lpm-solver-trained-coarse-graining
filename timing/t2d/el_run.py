"""Fine reference of the larger 2D benchmark specimens: for one problem instance (kind, n, seed) and every coarsening
factor s, the fine reference (SuperLU of el_core, timed) and the coarse models of every method (Voigt, patch, NN), with
errors and timings.  Single-threaded worker processes.

usage: python el_run.py comp N SEED0 NSEED --tag T [--s 2 3 4 6 8] [--methods voigt patch nn] [--ref lu|file]
                        [--workers W] [--mem-need GB]
  --ref lu   fine reference with el_core's SuperLU (timed: the fine-LPM cost of this sample)
  --ref file reuse the reference stored by an earlier run (fields npz), e.g. to add a method later
Outputs: results/el_{T}.jsonl (one line per instance, s and method: errors per load case, global-measure values,
timings) and data/fields/{T}/... (block fields of the reference and of every coarse model, float32), read by
bench2d.py.  Networks exist for s = 2, 3, 4; the NN rows of other s are skipped.
"""
import argparse
import json
import os
import time
import common  # noqa
import numpy as np
import el_core as E
import el_methods as M
from memguard import status as mem_status


def mem_gate(need_gb, margin=8.0, max_wait=7200):
    """shared machine: wait until the free commit charge leaves room for this factorisation and a margin."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        _, cm = mem_status()
        if cm > need_gb + margin:
            return time.time() - t0
        time.sleep(15)
    return time.time() - t0


NEED_GB = {("comp", 96): 0.5, ("comp", 192): 3.2, ("comp", 384): 13.0}


def prob_of(kind, n, seed):
    assert kind == "comp", kind
    return E.make_comp(n, seed)


def fdir(tag):
    d = os.path.join("data", "fields", tag)
    os.makedirs(d, exist_ok=True)
    return d


def ffile(tag, kind, n, seed, s, method):
    return os.path.join(fdir(tag), "%s_%d_%d_s%d_%s.npz" % (kind, n, seed, s, method))


def save_fields(fn, res, iface=None):
    d = {}
    for ld, r in res.items():
        d["sig_" + ld] = r["sig"].astype(np.float32)
        d["w_" + ld] = r["w"].astype(np.float32)
        d["u_" + ld] = r["u"].astype(np.float32)
        d["F_" + ld] = np.full(2, np.nan) if r["F"] is None else np.asarray(r["F"], float)
    if iface is not None:
        d["iface"] = iface
    np.savez_compressed(fn, **d)


def load_ref(fn):
    z = np.load(fn)
    ref = {}
    for ld in E.LOADS:
        F = z["F_" + ld]
        ref[ld] = dict(u=z["u_" + ld].astype(float), sig=z["sig_" + ld].astype(float), w=z["w_" + ld].astype(float),
                       F=None if np.isnan(F).all() else F)
    return ref, z["iface"]


def gvals(kind, n, s, res):
    return {ld: E.gval(ld, res[ld], E.free_cells(kind, n, s, ld)) for ld in E.LOADS}


def one(kind, n, seed, ss, methods, refmode, tag, mem_need):
    prob = prob_of(kind, n, seed)
    lines = []
    for s in ss:
        base = dict(kind=kind, n=n, seed=seed, s=s, tag=tag)
        ffn = ffile(tag, kind, n, seed, s, "fine")
        if refmode == "file":
            ref, iface = load_ref(ffn)
            ref_lu = None
        else:
            tm = {}
            wait = mem_gate(mem_need)
            ref, iface = E.fine_reference(prob, s, timing=tm)
            tm["mem_wait"] = wait
            save_fields(ffn, ref, iface)
            lines.append(dict(base, method="fine", ref=refmode, timing=tm, G=gvals(kind, n, s, ref)))
        for m in methods:
            if m == "nn":
                mp = "%s/cgnet_s%d.npz" % (M.MODEL_DIR, s)
                if not os.path.exists(mp):
                    continue
            tb = {}
            Kn, Tv = M.BUILDERS[m](prob, s, timing=tb)
            tsol = {}
            out = E.coarse_run(prob, s, Kn, Tv, timing=tsol)
            met = E.metrics(kind, n, s, ref, iface, out)
            rec = dict(base, method=m, build=tb, solve=tsol, metrics=met, G=gvals(kind, n, s, out))
            if m == "nn":
                tp = {}
                import coarse_params as CP
                b0, b1 = CP.forward_bonds(n // s, n // s)
                raw = [np.zeros((len(b0), 2)), np.zeros((len(b1), 2))]
                M.nn_post(prob, s, raw, timing=tp)
                rec["gpu_post"] = tp["post"]
            save_fields(ffile(tag, kind, n, seed, s, m), out)
            lines.append(rec)
    return lines


_A = {}


def _init(a):
    _A.update(a)
    import random
    time.sleep(random.Random(os.getpid()).uniform(0, a["stagger"]))


def _job(seed):
    a = _A
    t0 = time.perf_counter()
    try:
        lines = one(a["kind"], a["n"], seed, a["s"], a["methods"], a["ref"], a["tag"], a["mem_need"])
    except Exception as e:
        import traceback
        return [dict(seed=seed, error=repr(e), tb=traceback.format_exc())]
    for l in lines:
        l["t_job"] = time.perf_counter() - t0
        l["pid"] = os.getpid()
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("n", type=int)
    ap.add_argument("seed0", type=int)
    ap.add_argument("nseed", type=int)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--s", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--methods", nargs="+", default=["voigt", "patch", "nn"])
    ap.add_argument("--ref", default="lu", choices=["lu", "file"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stagger", type=float, default=10.0)
    ap.add_argument("--mem-need", type=float, default=None)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    mem_need = a.mem_need if a.mem_need is not None else NEED_GB.get((a.kind, a.n), 25.0)
    args = dict(kind=a.kind, n=a.n, s=a.s, methods=a.methods, ref=a.ref, tag=a.tag, mem_need=mem_need,
                stagger=a.stagger)
    seeds = list(range(a.seed0, a.seed0 + a.nseed))
    fn = "results/el_%s.jsonl" % a.tag
    T0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init, initargs=(args,)) as pool:
            for lines in pool.imap_unordered(_job, seeds, chunksize=1):
                with open(fn, "a") as f:
                    for l in lines:
                        f.write(json.dumps(l, default=float) + "\n")
                print("[%.0fs] seed %s: %d lines%s" % (time.time() - T0, lines[0].get("seed"), len(lines),
                                                       (" ERROR " + lines[0]["error"]) if "error" in lines[0] else ""),
                      flush=True)
    else:
        _A.update(args)
        for sd in seeds:
            lines = _job(sd)
            with open(fn, "a") as f:
                for l in lines:
                    f.write(json.dumps(l, default=float) + "\n")
            msg = []
            for l in lines:
                if "metrics" in l:
                    mm = l["metrics"]
                    msg.append("s%d %s: plate G %.2f%% sig %.2f%%" % (
                        l["s"], l["method"], 100 * np.mean([abs(mm[ld]["G"]) for ld in E.LOADS[3:]]),
                        100 * np.mean([mm[ld]["sig"] for ld in E.LOADS[3:]])))
                elif "error" in l:
                    msg.append("ERROR " + l["error"])
            print("[%.0fs] seed %d | %s" % (time.time() - T0, sd, " | ".join(msg)), flush=True)
    print("DONE %.0f s" % (time.time() - T0), flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
