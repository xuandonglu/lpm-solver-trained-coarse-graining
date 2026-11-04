"""Materials of every cube set and pore treatment (numpy/scipy, either env).
usage: python -I prep_mat.py SET [SET ...]      SET = w30_test24, w40_train24, syn_train, ...
  -> data/mat_SET_{a,b}.npz: solid (S, n^3) bool, E, nu (S, n^3) [units of E_C-S-H], ph (S, n^3) int8,
     cell_s / iface_s / padj_s (S, nc^3) bool and nrm_s (S,) for s = 2, 3, ndrop, void_frac, (b: phi_ph, E_ph, nu_ph)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import materials as MT  # noqa: E402


def cube_file(st):
    return common.data("cubes_%s.npz" % st) if not st.startswith("syn") else common.data("synth_%s.npz" % st[4:])


def run(st):
    d = np.load(cube_file(st))
    L = d["labels"]
    n = int(d["n"])
    props = d["props"] if "props" in d else None             # synthetic sets: per-sample phase properties
    for opt in "ab":
        t0 = time.time()
        out = {k: [] for k in ("solid", "E", "nu", "ph", "ndrop", "void_frac", "phi_ph", "E_ph", "nu_ph")}
        for s in (2, 3):
            for k in ("cell", "iface", "padj", "nrm"):
                out["%s_%d" % (k, s)] = []
        for i in range(len(L)):
            pr = None if props is None else {k: (float(props[i, k - 1, 0]), float(props[i, k - 1, 1])) for k in (1, 2, 3)}
            m = MT.make_material(L[i], opt, pr)
            for k in ("solid", "E", "nu"):
                out[k].append(m[k])
            out["ph"].append(m["ph"].astype(np.int8))
            out["ndrop"].append(m["info"]["ndrop"])
            out["void_frac"].append(m["info"]["void_frac"])
            for k in ("phi_ph", "E_ph", "nu_ph"):
                out[k].append(m["info"].get(k, np.nan))
            for s in (2, 3):
                c, nrm = MT.coarse_mask(m["solid"], n, s)
                ifc, padj = MT.cell_sets(L[i], m, n, s)
                out["cell_%d" % s].append(c)
                out["nrm_%d" % s].append(nrm)
                out["iface_%d" % s].append(ifc)
                out["padj_%d" % s].append(padj)
        out = {k: np.array(v) for k, v in out.items()}
        out["n"] = n
        np.savez_compressed(common.data("mat_%s_%s.npz" % (st, opt)), **out)
        print("%s opt %s: %d cubes n=%d, void %.3f, dropped vox %.1f, missing cells s2 %.1f s3 %.1f, coarse-cleaned "
              "s2 %d s3 %d, iface cells s2 %.2f, pore-adj cells s2 %.2f (%.0f s)" % (
                  st, opt, len(L), n, out["void_frac"].mean(), out["ndrop"].mean(),
                  (~out["cell_2"]).sum(1).mean(), (~out["cell_3"]).sum(1).mean(), out["nrm_2"].sum(), out["nrm_3"].sum(),
                  out["iface_2"].mean(), out["padj_2"].mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        run(st)
