"""Summary of the fracture evaluation from results/eval2_plate_SPLIT.json
-> results/tables2.md, results/summary2.json.
Per coarsening factor s and coarse model: mean absolute error (and mean signed error) of the strain at the first bond
failure, the peak force and the dissipated energy, the mean force-strain curve error, and the speed-up = fine time /
(coarse time + time to compute the coarse bond parameters), mean over the plates of the split.
usage: python make_tables2.py
"""
import json
import os
import numpy as np

SS = (2, 3, 4)
SPLITS = ("test_id", "test_phi", "test_c15", "test_big")
V = ["classical", "classical2P", "ml2P"]
NAME = {"classical": "Classical, strength first", "classical2P": "Classical, energy first", "ml2P": "Proposed"}
PRE = {"classical": ["patch_elastic", "patch_crit"], "classical2P": ["patch_elastic", "patch_crit"],
       "ml2P": ["ml_elastic", "naive_crit", "ml_crit", "ml_rho"]}


def agg(R, s, v, key):
    """(mean |e|, mean e) over the plates."""
    x = np.array([r[str(s)][v].get(key, np.nan) for r in R if v in r[str(s)]], float)
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.nanmean(np.abs(x))), float(np.nanmean(x))


def speedup(R, s, v):
    out = [r[str(s)]["fine"]["time"] / (r[str(s)][v]["time"] + sum(r[str(s)]["timing_pre"].get(k, 0.0) for k in PRE[v]))
           for r in R if v in r[str(s)]]
    return float(np.mean(out)) if out else np.nan


def plate_table(split, R, lines, summ):
    lines += ["", "## Plates %s (%d specimens)" % (split, len(R)), "",
              "| s | Coarse model | Initiation strain | Peak force | Dissipated energy | Curve | Speed-up |",
              "|---|---|---|---|---|---|---|"]
    for s in SS:
        for v in V:
            if not any(v in r[str(s)] for r in R):
                continue
            a = {k: agg(R, s, v, k) for k in ("e_init", "e_peak", "e_D")}
            c, cp, h, ch = (agg(R, s, v, k)[1] for k in ("curve", "curve_post", "haus95", "chamfer"))
            sp = speedup(R, s, v)
            lines.append("| %d | %s | %s | %s | %s | %.1f%% | %.1f |" % (
                s, NAME[v], *("%.1f%% (%+.1f)" % (100 * a[k][0], 100 * a[k][1]) for k in ("e_init", "e_peak", "e_D")),
                100 * c, sp))
            summ["plate_%s_s%d_%s" % (split, s, v)] = dict(init=a["e_init"], peak=a["e_peak"], D=a["e_D"], curve=c,
                                                            curve_post=cp, haus95=h, chamfer=ch, speedup=sp)
    lines.append("")
    lines.append("Mean absolute errors, mean signed errors in brackets (percentage points); fine time per plate %s s."
                 % " / ".join("%.2f" % np.mean([r[str(s)]["fine"]["time"] for r in R]) for s in SS))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    lines, summ = ["# Fracture of unnotched plates (make_tables2.py)"], {}
    for split in SPLITS:
        fn = os.path.join("results", "eval2_plate_%s.json" % split)
        if os.path.exists(fn):
            plate_table(split, json.load(open(fn)), lines, summ)
    open(os.path.join("results", "tables2.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(summ, open(os.path.join("results", "summary2.json"), "w"), indent=1)
    print("\n".join(lines))
