"""Summary of local window calibration and the per-specimen fit on test_id.
Stress-field errors of the plates and under KUBC: mean over the specimens and over the three load cases of each kind.
  errors     patch homogenization, local window calibration (l2j_m1) and the proposed network (cnn) on the 100 test
             specimens; per-specimen fit (l2b) on the first 32
  reduction  of the plate error of patch homogenization by local window calibration
  cost       (local fine window solves + joint fit + coarse solve) / fine solution, per specimen
             (results/bench_cost_cpu.json of bench_cost.py, results/l2fit_l2j_m1_test_id_sS.json of l2run.py fit)
  converged  local window calibration with 4 x 50 L-BFGS iterations (l2j_m1) against the converged fit (l2jc_m1) on
             the first 32 test specimens
usage: python make_report.py  -> results/report.md, results/report.json
"""
import json
import os

import numpy as np
import common

R = common.RES
SS = (2, 3, 4)
LOADS = {"kubc": ["ring_xx", "ring_yy", "ring_xy"], "plate": ["tb_ty", "tb_sx", "lr_tx"]}
RUNGS = [("patch", "Patch"), ("l2j_m1", "Local window calibration"), ("cnn", "Proposed"),
         ("l2b", "Per-specimen fit"), ("l2jc_m1", "Local window calibration, converged")]


def load(name):
    fn = os.path.join(R, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


def err(per_sample, kind, nmax=None):
    ps = per_sample[:nmax] if nmax else per_sample
    return float(np.mean([abs(m[ld]["sig"]) for m in ps for ld in LOADS[kind]]))


def main():
    out = dict(errors={}, reduction={}, cost={}, converged={})
    md = ["# Local window calibration and per-specimen fit (make_report.py)", "",
          "Stress-field errors on test_id (%; plates / KUBC)", "", "| Model | Specimens | s = 2 | s = 3 | s = 4 |",
          "|---|---|---|---|---|"]
    ev = {(r, s): load("eval_%s_test_id_s%d.json" % (r, s)) for r, _ in RUNGS for s in SS}
    for r, name in RUNGS:
        cells, nsp = [], set()
        for s in SS:
            e = ev[(r, s)]
            if e is None:
                cells.append("-")
                continue
            v = {k: err(e["per_sample"], k) for k in LOADS}
            out["errors"]["%s_s%d" % (r, s)] = dict(v, n=len(e["per_sample"]))
            nsp.add(len(e["per_sample"]))
            cells.append("%.1f / %.1f" % (100 * v["plate"], 100 * v["kubc"]))
        md.append("| %s | %s | %s |" % (name, "/".join(map(str, sorted(nsp))) or "-", " | ".join(cells)))
    md += ["", "Reduction of the plate error of patch homogenization by local window calibration: " + ", ".join(
        "s = %d %s" % (s, "-" if ev[("patch", s)] is None or ev[("l2j_m1", s)] is None else "%.0f%%" % (
            100 * (1 - err(ev[("l2j_m1", s)]["per_sample"], "plate") / err(ev[("patch", s)]["per_sample"], "plate"))))
        for s in SS)]
    for s in SS:
        if ev[("patch", s)] is not None and ev[("l2j_m1", s)] is not None:
            out["reduction"]["s%d" % s] = 1 - (err(ev[("l2j_m1", s)]["per_sample"], "plate")
                                               / err(ev[("patch", s)]["per_sample"], "plate"))
    bc = load("bench_cost_cpu.json")
    cells = []
    for s in SS:
        f = load("l2fit_l2j_m1_test_id_s%d.json" % s)
        if bc is None or f is None:
            cells.append("s = %d -" % s)
            continue
        c = bc["s%d" % s]
        t = c["l2fine_m1"] + f["t_fit_per_specimen"] + c["coarse"]
        out["cost"]["s%d" % s] = dict(t=t, fine=c["fine"], ratio=t / c["fine"])
        cells.append("s = %d %.2f s = %.1f fine solutions" % (s, t, t / c["fine"]))
    md += ["", "Cost of local window calibration per specimen: " + ", ".join(cells)]
    cells = []
    for s in SS:
        a, b = ev[("l2j_m1", s)], ev[("l2jc_m1", s)]
        if a is None or b is None:
            cells.append("s = %d -" % s)
            continue
        n = min(len(a["per_sample"]), len(b["per_sample"]))
        d = {k: err(a["per_sample"], k, n) - err(b["per_sample"], k, n) for k in LOADS}
        out["converged"]["s%d" % s] = dict(d, n=n)
        cells.append("s = %d %.2f / %.2f" % (s, 100 * d["plate"], 100 * d["kubc"]))
    md += ["", "Error of local window calibration minus that of the converged fit (percentage points, plates / KUBC, "
           "first 32 specimens): " + ", ".join(cells), ""]
    open(os.path.join(R, "report.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(out, open(os.path.join(R, "report.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
