"""Unified timing table, 3D (numerical environment + torch inference worker): wall time per analysis (9 load cases ->
particle stresses), all methods interleaved on the same machine, >= 2 rounds, LPMCG_THREADS (6) CPU threads.

Rows (method, solver, tol):
  F-trad    fine LPM, LPM-C workflow: unpreconditioned CG from zero, ||r|| <= 1e-4 ||b||   ("traditional LPM")
  F-amg     fine LPM, tuned AMG-PCG (I8b / V1 amg3d), same tolerance 1e-4                 (same answer, strong solver)
  F-tuned   fine LPM, AMG-PCG at the loosest tolerance that meets the I8b / V1 target       (strict-appendix reference)
  NN-trad   coarse s = 2 + network, the same traditional CG 1e-4 on the coarse model; inference CPU or GPU
  NN-tuned  coarse s = 2 + network, the I8b / V1 tuned solver and tolerance for the same target
  V-trad    coarse s = 2 Voigt, traditional CG 1e-4;  P-trad: patch homogenisation (timed) + traditional CG 1e-4
End-to-end = build (fine: bond parameters [+ V1 neighbour table of the solid set]; NN: worker inference incl. features
and post-processing [+ V1 coarse neighbour table]; Voigt / patch: homogenisation) + solve (9 cases, AMG numeric setup
included) + computeStress of the 9 solutions.  Geometry-only set-up (lattices, BC sets, AMG aggregation) is built once
per size and not counted, as in I8b.  Errors (untimed, round 0): I8 stress-field error vs the converged fine reference
(mean of the 9 cases; V1: R = 2, existing BC-free cells).
usage: python bench3d.py i8 SPLIT K [--reps R]        -> results/T3_i8_SPLIT.json
       python bench3d.py v1 SET OPT K [--reps R]      -> results/T3_v1_SET_OPT.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import geom3d as G  # noqa: E402

TRAD = 1e-4
# tuned (solver, tol) for the 5 % target (I8b phaseB configs; V1 phaseB configs: (b) 5 %, (a) 10 % / floor)
TUNED_I8 = {24: dict(fine=("amg", 2e-2), ml=("jac", 5e-3)), 48: dict(fine=("amg", 5e-3), ml=("amg", 1e-2)),
            72: dict(fine=("amg", 5e-3), ml=("amg", 5e-3))}
TUNED_V1 = {("w30", 48, "b"): dict(fine=("amg", 5e-3), ml=("amg", 5e-3)),
            ("w30", 72, "b"): dict(fine=("amg", 5e-3), ml=("amg", 5e-3)),
            ("w40", 48, "b"): dict(fine=("amg", 5e-3), ml=("amg", 5e-3)),
            ("w40", 72, "b"): dict(fine=("amg", 5e-3), ml=("amg", 3e-3)),
            ("w30", 48, "a"): dict(fine=("amg", 1e-2), ml=("amg", 5e-3)),
            ("w30", 72, "a"): dict(fine=("amg", 1e-2), ml=("amg", 5e-3)),
            ("w40", 48, "a"): dict(fine=("amg", 1e-2), ml=("amg", 2e-3)),
            ("w40", 72, "a"): dict(fine=("amg", 1e-2), ml=("amg", 2e-3))}


def configs(tuned):
    return [("F-trad", "fine", "cg", TRAD), ("F-amg", "fine", "amg", TRAD),
            ("F-tuned", "fine") + tuned["fine"],
            ("NN-trad", "ml", "cg", TRAD), ("NN-tuned", "ml") + tuned["ml"],
            ("V-trad", "voigt", "cg", TRAD), ("P-trad", "patch", "cg", TRAD)]


class Worker:
    def __init__(self, prep):
        self.p = subprocess.Popen([common.PT, "-B", os.path.join(common.HERE, "infer3d.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.info = self.call(prep)

    def call(self, cmd):
        time.sleep(0.3)          # idle gap: spinning OpenMP threads of the other process stop (I8b protocol)
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        time.sleep(0.3)
        return out

    def close(self):
        self.p.stdin.write(json.dumps(dict(op="quit")) + "\n")
        self.p.stdin.flush()
        self.p.wait(timeout=120)


def run_cfg(a, v1, PB, Pf, Pc, n, mth, sv, tol, i, Kml, Tml, data):
    """one end-to-end analysis; returns (t_build, t_solve, t_stress, iterations, info, stresses, problem)."""
    t0 = time.perf_counter()
    if v1:
        import patch_v as PV
        mf = data
        sol, cell = mf["solid"][i], mf["cell_2"][i]
        Cv = G.iso_C(mf["E"][i], mf["nu"][i])
        if mth == "fine":
            P, so = Pf, sol
            nbr = Pf.specimen(sol)
            Kn, Tv = PB.fine_params(Pf.lat, Cv)
        else:
            P, so = Pc, cell
            nbr = Pc.specimen(cell)
            if mth == "ml":
                Kn, Tv = (Kml, Tml) if Kml is not None else PB.coarse_floor(*PB.voigt_params(Pc.lat, Cv, n), nbr)
            elif mth == "voigt":
                Kn, Tv = PB.coarse_floor(*PB.voigt_params(Pc.lat, Cv, n), nbr)
            else:
                Cp = PV.patch_C_v(Cv, sol, n, PB.S)
                Kn, Tv = PB.coarse_floor(*G.bond_params_from_cells(Pc.lat, G.cell_params_from_C66(Cp, 1.0)), nbr)
        t_build = time.perf_counter() - t0
        t_sol, its, us, info = P.solve(sv, Kn, Tv, nbr, so, tol)
        t0 = time.perf_counter()
        sigs = P.stresses(Kn, Tv, nbr, us)
        t_st = time.perf_counter() - t0
    else:
        import patch3d
        m = data
        if mth == "fine":
            P = Pf
            Kn, Tv = PB.fine_params(Pf.lat, PB.material(m, i))
        elif mth == "ml":
            P = Pc
            Kn, Tv = (Kml, Tml) if Kml is not None else PB.voigt_params(Pc.lat, PB.material(m, i), n)
        elif mth == "voigt":
            P = Pc
            Kn, Tv = PB.voigt_params(Pc.lat, PB.material(m, i), n)
        else:
            P = Pc
            Cp = patch3d.patch_C(PB.material(m, i), n, PB.S)
            Kn, Tv = G.bond_params_from_cells(Pc.lat, G.cell_params_from_C66(Cp, PB.S / 2.0))
        t_build = time.perf_counter() - t0
        t_sol, its, us, info = P.solve(sv, Kn, Tv, tol=tol)
        t0 = time.perf_counter()
        sigs = P.stresses(Kn, Tv, us)
        t_st = time.perf_counter() - t0
    return t_build, t_sol, t_st, its, info, sigs, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("args", nargs="+")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list of row labels (debug)")
    a = ap.parse_args()
    v1 = a.kind == "v1"
    if v1:
        import problems_v1 as PB
        import patch_v as PV
        st, opt, K = a.args[0], a.args[1], int(a.args[2])
        mf = np.load(common.v1data("mat_%s_%s.npz" % (st, opt)))
        ref_all = np.load(common.v1data("fine_%s_%s.npz" % (st, opt)))["sig_2"]
        n = int(mf["n"])
        tuned = TUNED_V1[(st.split("_")[0], n, opt)]
        tag = "syn_" + opt
        prep = dict(op="prep", kind="v1", set=st, opt=opt, tag=tag)
        out = os.path.join(common.HERE, "results", "T3_v1_%s_%s.json" % (st, opt))
        PB.warmup()
        PV.patch_C_v(G.iso_C(np.ones(216), np.full(216, 0.2)), np.ones(216, bool), 6, 2)
    else:
        import problems as PB
        import patch3d
        split, K = a.args[0], int(a.args[1])
        m = dict(np.load(common.i8data("micro_%s.npz" % split), allow_pickle=True))
        ref_all = np.load(common.i8data("fine_%s.npz" % split))["sig_2"]
        n = int(m["n"])
        tuned = TUNED_I8[n]
        prep = dict(op="prep", kind="i8", split=split)
        out = os.path.join(common.HERE, "results", "T3_i8_%s.json" % split)
        PB.warmup()
    cfgs = configs(tuned)
    if a.only:
        cfgs = [c for c in cfgs if c[0] in a.only.split(",")]
    Pf = PB.Problem(n, False)
    Pc = PB.Problem(n, True)
    W = Worker(prep)
    print("worker:", W.info, flush=True)
    tmp = tempfile.mkdtemp(prefix="t3_", dir=os.path.join(common.HERE, "results"))
    res = dict(kind=a.kind, args=a.args, n=n, K=K, reps=a.reps, threads=common.NT, worker=W.info,
               configs=[dict(label=c[0], method=c[1], solver=c[2], tol=c[3]) for c in cfgs],
               t_geom_amg=dict(fine=Pf.t_geom_amg, coarse=Pc.t_geom_amg), samples=[])
    # warm-up (untimed): every configuration once on specimen 0 at a loose tolerance (numba / BLAS / allocations)
    tw = time.perf_counter()
    for (lab, mth, sv, tol) in cfgs:
        run_cfg(a, v1, PB, Pf, Pc, n, mth, sv, 0.5, 0, None, None, mf if v1 else m)
    print("warm-up %.1f s" % (time.perf_counter() - tw), flush=True)
    for i in range(K):
        rec = dict(i=i, infer=dict(cpu=[], gpu=[]), runs={c[0]: [] for c in cfgs}, err={}, its={})
        pfile = os.path.join(tmp, "nn_%d.npz" % i)
        W.call(dict(op="run", i=i, dev="cpu", save=pfile))
        d = np.load(pfile)
        Kml, Tml = np.ascontiguousarray(d["Kn"]), np.ascontiguousarray(d["Tv"])
        d.close()
        os.remove(pfile)
        if v1:
            sol, cell = mf["solid"][i], mf["cell_2"][i]
        for r in range(a.reps):
            for dev in (("cpu", "gpu") if r % 2 == 0 else ("gpu", "cpu")):
                if dev == "gpu" and not W.info.get("gpu"):
                    continue
                rec["infer"][dev].append(W.call(dict(op="run", i=i, dev=dev, save=None)))
            order = cfgs[r % len(cfgs):] + cfgs[:r % len(cfgs)]
            for (lab, mth, sv, tol) in order:
                t_build, t_sol, t_st, its, info, sigs, P = run_cfg(a, v1, PB, Pf, Pc, n, mth, sv, tol, i, Kml, Tml,
                                                                   mf if v1 else m)
                if v1:
                    cell = mf["cell_2"][i]
                rec["runs"][lab].append(dict(t_build=t_build, t_solve=t_sol, t_stress=t_st, t_setup=info.get("t_setup")))
                if r == 0:
                    e = P.field_errors(sigs, ref_all[i], cell) if v1 else P.field_errors(sigs, ref_all[i])
                    if v1:
                        e = e[0]
                    rec["err"][lab] = float(np.mean(e))
                    rec["its"][lab] = int(sum(its))
                print("  i %d r %d %-9s build %7.3f solve %8.3f stress %6.3f  its %6d  err %.2f%%" % (
                    i, r, lab, t_build, t_sol, t_st, sum(its), 100 * rec["err"][lab]), flush=True)
            print("%s specimen %d round %d done; infer cpu %s gpu %s" % (" ".join(a.args), i, r, rec["infer"]["cpu"][-1:],
                                                                          rec["infer"]["gpu"][-1:]), flush=True)
        res["samples"].append(rec)
        json.dump(res, open(out, "w"))
    W.close()
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
