"""Summary of the cement-paste evaluation from the result files of train_v1.py (numpy only).
Stress-field errors (sig_all: free cells of the coarse lattice, mean over the 9 load cases) on the test cubes of n = 24
at s = 2 for both pore treatments (a: empty pores, b: porous hydrate) and both scans; the proposed model on the larger
cubes (n = 48, 72); the share of the gap between patch homogenization and the per-specimen fit that the proposed model
recovers.
usage: python report_v1.py  -> results/tables_v1.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

R = os.path.join(common.HERE, "results")
COLS = [("a", "w30"), ("a", "w40"), ("b", "w30"), ("b", "w40")]
HEAD = ["Empty pores, w/c 0.30", "Empty pores, w/c 0.40", "Porous hydrate, w/c 0.30", "Porous hydrate, w/c 0.40"]
OTHER = {"w30": "r40", "w40": "r30"}          # network trained on real cubes of the other scan


def load(fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def err(opt, sc, model, size="24"):
    ev = load("eval_%s_s2_all_%s.json" % (opt, sc))
    try:
        return ev["%s_test%s" % (sc, size)][model]["sig_all"]
    except (KeyError, TypeError):
        return None


def fit(opt, sc):
    o = load("oracle_%s_test24_%s_s2_from_syn_%s.json" % (sc, opt, opt))
    return None if o is None else o["best"]["sig_all"]


def pct(x):
    return "-" if x is None else "%.1f" % (100 * x)


def main():
    rows = [("Voigt", lambda o, sc: err(o, sc, "voigt")), ("Patch", lambda o, sc: err(o, sc, "patch")),
            ("Network, two-phase training", lambda o, sc: err(o, sc, "nn_i8")),
            ("Proposed, cement-like training", lambda o, sc: err(o, sc, "nn_syn_" + o)),
            ("Network, real-scan training", lambda o, sc: err(o, sc, "nn_%s_%s" % (OTHER[sc], o))),
            ("Per-specimen fit (12 cubes)", fit)]
    md = ["# Cement-paste test cubes (report_v1.py)", "", "Stress-field errors (%; n = 24, s = 2, 40 cubes per scan)", "",
          "| Coarse model | " + " | ".join(HEAD) + " |", "|---" * 5 + "|"]
    for name, f in rows:
        md.append("| %s | %s |" % (name, " | ".join(pct(f(o, sc)) for o, sc in COLS)))
    md += ["", "Proposed model on the larger cubes (%; n = 48 / n = 72)", "", "| | " + " | ".join(HEAD) + " |",
           "|---" * 5 + "|",
           "| Proposed, cement-like training | %s |" % " | ".join(
               "%s / %s" % (pct(err(o, sc, "nn_syn_" + o, "48")), pct(err(o, sc, "nn_syn_" + o, "72"))) for o, sc in COLS)]
    share = []
    for o, sc in COLS:
        p, n, b = err(o, sc, "patch"), err(o, sc, "nn_syn_" + o), fit(o, sc)
        share.append("-" if None in (p, n, b) else "%.0f%%" % (100 * (p - n) / (p - b)))
    md += ["", "Share of the gap between patch homogenization and the per-specimen fit recovered by the proposed model: "
           + ", ".join("%s %s" % (h, s) for h, s in zip(HEAD, share)), ""]
    open(os.path.join(R, "tables_v1.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
