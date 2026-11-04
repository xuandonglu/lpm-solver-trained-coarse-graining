"""Summary of the timing runs (t3d/results/T3_*.json, t2d/results/T2_*.json).
Per specimen: median over rounds of the end-to-end time (NN rows: the same round's worker inference added, CPU or GPU
variant); then median over specimens.  Errors: mean over specimens (round 0).
usage: python summ_timing.py  ->  results/timing_tables.md, results/timing_summary.json"""
import glob
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def rows3(res):
    labs = [c["label"] for c in res["configs"]]
    out = {}
    for lab in labs:
        per = {"cpu": [], "gpu": [], "plain": []}
        parts = {"build": [], "solve": [], "stress": [], "infer_cpu": [], "infer_gpu": []}
        errs = []
        for smp in res["samples"]:
            runs = smp["runs"][lab]
            if not runs:
                continue
            base = [r["t_build"] + r["t_solve"] + r["t_stress"] for r in runs]
            if lab.startswith("NN"):
                for dev in ("cpu", "gpu"):
                    inf = [x["t_feat"] + x["t_fwd"] + x["t_post"] for x in smp["infer"][dev]]
                    k = min(len(inf), len(base))
                    if k:
                        per[dev].append(med([base[j] + inf[j] for j in range(k)]))
                        parts["infer_" + dev].append(med(inf))
            per["plain"].append(med(base))
            parts["build"].append(med([r["t_build"] for r in runs]))
            parts["solve"].append(med([r["t_solve"] for r in runs]))
            parts["stress"].append(med([r["t_stress"] for r in runs]))
            errs.append(smp["err"][lab])
        out[lab] = dict(t=med(per["plain"]), t_cpu=med(per["cpu"]), t_gpu=med(per["gpu"]),
                        err=float(np.mean(errs)), its=float(np.median([s["its"][lab] for s in res["samples"]])),
                        **{k: med(v) for k, v in parts.items()}, nspec=len(errs))
    return out


def rows2(res):
    out = {}
    for lab in res["rows"]:
        per = {"cpu": [], "gpu": [], "plain": []}
        parts = {"params": [], "solve": [], "stress": [], "infer_cpu": [], "infer_gpu": []}
        errs = []
        for smp in res["samples"]:
            runs = smp["runs"][lab]
            base = [r["params"] + r["solve"] + r["stress"] for r in runs]
            if lab.startswith("NN"):
                s = int(lab.split("-")[0][2:])
                for dev in ("cpu", "gpu"):
                    inf = [x["t_feat"] + x["t_fwd"] for x in smp["infer"]["%d_%s" % (s, dev)]]
                    k = min(len(inf), len(base))
                    per[dev].append(med([base[j] + inf[j] for j in range(k)]))
                    parts["infer_" + dev].append(med(inf))
            per["plain"].append(med(base))
            for k in ("params", "solve", "stress"):
                parts[k].append(med([r[k] for r in runs]))
            errs.append(smp["err"][lab])
        out[lab] = dict(t=med(per["plain"]), t_cpu=med(per["cpu"]), t_gpu=med(per["gpu"]), err=float(np.mean(errs)),
                        its=float(np.median([s["its"][lab] for s in res["samples"]])),
                        **{k: med(v) for k, v in parts.items()}, nspec=len(errs))
    return out


NAMES3 = {"F-trad": "Fine LPM, unpreconditioned CG from zero, 1e-4", "F-amg": "Fine LPM, tuned AMG-PCG, 1e-4",
          "F-tuned": "Fine LPM, AMG-PCG at the loosest tolerance that meets the target error",
          "NN-trad": "Coarse s=2, proposed, unpreconditioned CG 1e-4",
          "NN-tuned": "Coarse s=2, proposed, tuned solver and tolerance for the target error",
          "V-trad": "Coarse s=2, Voigt, unpreconditioned CG 1e-4",
          "P-trad": "Coarse s=2, patch (incl. homogenization), unpreconditioned CG 1e-4"}


def fmt_t(x):
    return "–" if not np.isfinite(x) else ("%.3f" % x if x < 1 else ("%.2f" % x if x < 10 else "%.1f" % x))


def table3(name, R):
    ref = R["F-trad"]["t"]
    L = ["| Model | Stress-field error | Iterations | Wall time (s) | Speed-up |", "|---|---|---|---|---|"]
    for lab, r in R.items():
        if lab.startswith("NN"):
            for dev in ("cpu", "gpu"):
                t = r["t_" + dev]
                L.append("| %s (%s inference %s s) | %.2f %% | %d | %s | %.1f |" % (
                    NAMES3[lab], dev.upper(), fmt_t(r["infer_" + dev]), 100 * r["err"], r["its"], fmt_t(t), ref / t))
        else:
            L.append("| %s | %.2f %% | %d | %s | %.1f |" % (NAMES3[lab], 100 * r["err"], r["its"], fmt_t(r["t"]),
                                                          ref / r["t"]))
    return "\n".join(L)


NAMES2 = {"F-trad": "Fine LPM, unpreconditioned CG from zero, 1e-4", "F-lu": "Fine LPM, direct solver (SuperLU)",
          "F-amg": "Fine LPM, AMG-PCG, 1e-4"}


def table2(R):
    ref = R["F-trad"]["t"]
    L = ["| Model | Stress-field error (own resolution) | Iterations | Wall time (s) | Speed-up |", "|---|---|---|---|---|"]
    for lab, r in R.items():
        nm = NAMES2.get(lab)
        if nm is None:
            rule, s = lab.split("-")[0].rstrip("0123456789"), lab.split("-")[0][-1]
            nm = {"NN": "Coarse s=%s, proposed" % s, "V": "Coarse s=%s, Voigt" % s,
                  "P": "Coarse s=%s, patch (incl. homogenization)" % s}[rule] + ", unpreconditioned CG 1e-4"
        if lab.startswith("NN"):
            for dev in ("cpu", "gpu"):
                t = r["t_" + dev]
                L.append("| %s (%s inference %s s) | %.2f %% | %d | %s | %.1f |" % (
                    nm, dev.upper(), fmt_t(r["infer_" + dev]), 100 * r["err"], r["its"], fmt_t(t), ref / t))
        else:
            L.append("| %s | %.2f %% | %d | %s | %.1f |" % (nm, 100 * r["err"], r["its"], fmt_t(r["t"]), ref / r["t"]))
    return "\n".join(L)


def main():
    md, js = ["# Timing tables (summ_timing.py); speed-up over the fine LPM with unpreconditioned CG", ""], {}
    for fn in sorted(glob.glob(os.path.join(HERE, "t3d", "results", "T3_*.json"))):
        res = json.load(open(fn))
        R = rows3(res)
        name = os.path.basename(fn)[3:-5]
        js[name] = R
        md += ["## %s (n = %d, %d specimens, %d rounds, %s threads)" % (name, res["n"], len(res["samples"]), res["reps"],
                                                                        res["threads"]), "", table3(name, R), ""]
    for fn in sorted(glob.glob(os.path.join(HERE, "t2d", "results", "T2_*.json"))):
        res = json.load(open(fn))
        R = rows2(res)
        name = os.path.basename(fn)[3:-5]
        js[name] = R
        md += ["## %s (n = %d, %d specimens, %d rounds, %s threads)" % (name, res["n"], len(res["samples"]), res["reps"],
                                                                        res["threads"]), "", table2(R), ""]
    open(os.path.join(OUT, "timing_tables.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(js, open(os.path.join(OUT, "timing_summary.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
