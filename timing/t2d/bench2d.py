"""Unified timing table, 2D (numerical environment + torch inference worker): wall time per analysis (two-phase
composite on the square lattice, 6 load cases -> particle stresses), all rows interleaved on the same machine, >= 2
rounds, LPMCG_THREADS (6) CPU threads.

Rows:
  F-trad   fine LPM (s = 2 block BCs), LPM-C workflow: unpreconditioned CG from zero, ||r|| <= 1e-4 ||b|| (matrix-free)
  F-lu     fine LPM, direct solver (el_core: nested-dissection SuperLU, 3 factorisations, 6 back-solves)
  F-amg    fine LPM, AMG-PCG (aggregation-based graph V-cycle, gmg_copy.py), tolerance 1e-4, assembled K
  NNs-trad coarse s (2, 4) + 2D bond network, the same traditional CG 1e-4; inference CPU (torch, 6 threads) or GPU
  Vs-trad / Ps-trad  coarse s Voigt / patch (patch_g, timed) + traditional CG 1e-4
End-to-end = parameters (fine bond parameters; NN: worker inference + numpy post-processing; Voigt / patch build)
+ solve + particle stress of the 6 cases; geometry-only set-up (lattices, BC sets, ND orders, patch geometry,
numba compilation) in a warm-up.  Errors (untimed, round 0): stress field at the method's own
resolution R = s (fine rows: R = 2), mean of the 6 cases, vs the exact fine LPM of the same boundary-value problem
(block fields stored by el_run.py for n = 96 / 192 / 384; computed here by SuperLU for n = 48).
usage: python bench2d.py N --seeds 0 1 2 --reps 2   -> results/T2_comp_N.json
"""
import argparse
import json
import os
import subprocess
import time
import a2common as AC
import numpy as np
import el_prob as EP
import el_core as E
import el_methods as M
import solver2d as S2
import ref2d as TL
import cg2plain as CG

TRAD = 1e-4
SS = (2, 4)


class Worker:
    def __init__(self, n):
        self.p = subprocess.Popen([AC.PT, "-B", AC.here("infer2d.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.info = self.call(dict(op="prep", n=n, s=list(SS)))

    def call(self, cmd):
        time.sleep(0.3)
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        time.sleep(0.3)
        return out

    def close(self):
        self.p.stdin.write(json.dumps(dict(op="quit")) + "\n")
        self.p.stdin.flush()
        self.p.wait(timeout=60)


def refs(tag, prob, n):
    """{s: {load: (n/s)^2 x 3 block stress of the exact fine LPM of the s-problem}}"""
    out = {}
    for s in SS:
        if tag != "comp_48":
            out[s] = TL.ref_R(tag, prob["seed"], s, s)
            continue
        gf = E.fine_geometry("comp", n, s)
        Kn, Tv = EP.fine_bond_params(prob, gf.lat)
        r = gf.run(Kn, Tv, post=False)
        sy = S2.get_sys("comp", n, 1, s)
        out[s] = {ld: sy.block_sig(S2.stress(gf.lat, r[ld]["u"].ravel(), Kn, Tv), s) for ld in EP.LOADS}
    return out


def rule_s(lab):
    head = lab.split("-")[0]
    rule = head.rstrip("0123456789")
    return rule, int(head[len(rule):])


def run(lab, prob, tag, n, tm, nnraw=None):
    """one end-to-end analysis; returns (stress fields {load: particle stress}, system, resolution R)."""
    t0 = time.perf_counter()
    if lab.startswith("F-"):
        if lab == "F-lu":
            gf = E.fine_geometry("comp", n, 2)
            Kn, Tv = EP.fine_bond_params(prob, gf.lat)
            t1 = time.perf_counter()
            r = gf.run(Kn, Tv, post=False)
            U = {ld: r[ld]["u"].ravel() for ld in EP.LOADS}
            lat = gf.lat
            its = 0
        else:
            sy = S2.get_sys("comp", n, 1, 2)
            Kn, Tv = EP.fine_bond_params(prob, sy.lat)
            t1 = time.perf_counter()
            if lab == "F-trad":
                U, its = CG.solve_cg(sy, Kn, Tv, TRAD)
            else:
                K = sy.assemble(Kn, Tv)
                out = sy.solve(K, {}, prec="amg", tols=np.array([TRAD]))
                U = {ld: out[ld][0][0] for ld in EP.LOADS}
                its = int(sum(out[ld][1][0] for ld in EP.LOADS))
            lat = sy.lat
        t2 = time.perf_counter()
        sig = {ld: S2.stress(lat, U[ld], Kn, Tv) for ld in EP.LOADS}
        t3 = time.perf_counter()
        tm.update(params=t1 - t0, solve=t2 - t1, stress=t3 - t2, its=its)
        return sig, S2.get_sys("comp", n, 1, 2), 2
    rule, s = rule_s(lab)
    sy = S2.get_sys("comp", n, s)
    if rule == "NN":
        M.nn_post(prob, s, nnraw)                                   # timed numpy post-processing (dummy raw, same cost)
        tp = time.perf_counter()
        Kn, Tv = TL.rule_params(prob, tag, "nn", s, cache=True)     # the real network parameters (cached, untimed)
        t0 += time.perf_counter() - tp
    elif rule == "V":
        Kn, Tv = M.build_voigt(prob, s)
    else:
        Kn, Tv = M.build_patch(prob, s)
    t1 = time.perf_counter()
    U, its = CG.solve_cg(sy, Kn, Tv, TRAD)
    t2 = time.perf_counter()
    sig = {ld: S2.stress(sy.lat, U[ld], Kn, Tv) for ld in EP.LOADS}
    t3 = time.perf_counter()
    tm.update(params=t1 - t0, solve=t2 - t1, stress=t3 - t2, its=its)
    return sig, sy, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rows", default="F-trad,F-lu,F-amg,NN2-trad,NN4-trad,V2-trad,P2-trad,V4-trad,P4-trad")
    a = ap.parse_args()
    n = a.n
    tag = "comp_%d" % n
    rows = a.rows.split(",")
    free = {R: TL.free_R("comp", n, R) for R in SS}
    import coarse_params as CP
    raw = {s: [np.zeros((len(b), 2)) for b in CP.forward_bonds(n // s, n // s)] for s in SS}
    # warm-up on another instance: geometry, patterns, ND orders, patch geometry, numba compilation, network caches
    tw = time.perf_counter()
    pw = AC.prob_of(tag, 1000)
    for lab in rows:
        run(lab, pw, "warm_%d" % n, n, {}, raw[rule_s(lab)[1]] if lab.startswith("NN") else None)
    print("warm-up %.1f s" % (time.perf_counter() - tw), flush=True)
    W = Worker(n)
    print("worker:", W.info, flush=True)
    out = AC.here("results", "T2_%s.json" % tag)
    res = dict(tag=tag, n=n, seeds=a.seeds, reps=a.reps, threads=AC.NT, rows=rows, worker=W.info, samples=[])
    for sd in a.seeds:
        prob = AC.prob_of(tag, sd)
        ref = refs(tag, prob, n)
        for s in SS:
            TL.rule_params(prob, tag, "nn", s, cache=True)            # network parameters cached before timing
        rec = dict(seed=sd, infer={"%d_%s" % (s, d): [] for s in SS for d in ("cpu", "gpu")},
                   runs={lab: [] for lab in rows}, err={}, its={})
        for r in range(a.reps):
            for s in SS:
                for dev in (("cpu", "gpu") if r % 2 == 0 else ("gpu", "cpu")):
                    rec["infer"]["%d_%s" % (s, dev)].append(W.call(dict(op="run", seed=sd, s=s, dev=dev)))
            order = rows[r % len(rows):] + rows[:r % len(rows)]
            for lab in order:
                tm = {}
                sig, sy, R = run(lab, prob, tag, n, tm, raw[rule_s(lab)[1]] if lab.startswith("NN") else None)
                rec["runs"][lab].append(tm)
                if r == 0:
                    e = [TL.rel(sy.block_sig(sig[ld], R)[free[R][ld]], ref[R][ld][free[R][ld]]) for ld in EP.LOADS]
                    rec["err"][lab] = float(np.mean(e))
                    rec["its"][lab] = int(tm["its"])
                print("  seed %d r %d %-9s params %7.3f solve %8.3f stress %6.3f its %6d err %.2f%%" % (
                    sd, r, lab, tm["params"], tm["solve"], tm["stress"], tm["its"], 100 * rec["err"][lab]), flush=True)
            print("%s seed %d round %d done" % (tag, sd, r), flush=True)
        res["samples"].append(rec)
        json.dump(res, open(out, "w"))
    W.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
