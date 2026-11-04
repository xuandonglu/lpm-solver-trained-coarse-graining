"""Pieces of the 2D benchmark: coarse bond parameters of a method (Voigt, patch, NN; cached on disk for the accuracy
evaluation) and the fine reference averaged to the output resolution R, with its free cells."""
import os
import a2common as AC
import numpy as np
import el_prob as EP
import el_methods as M

M.MODEL_DIR = AC.I2MODELS               # networks of ../../coarse2d, read-only


def pdir(tag):
    d = AC.here("data", "params", tag)
    os.makedirs(d, exist_ok=True)
    return d


def rule_params(prob, tag, rule, s, timing=None, cache=True):
    """(Kn, Tv) of a coarse rule at factor s.  NN / patch parameters are cached on disk for the accuracy stages
    (cache=True); the timing stage calls the builders (cache=False)."""
    if rule == "voigt":
        return M.build_voigt(prob, s, timing=timing)
    fn = os.path.join(pdir(tag), "%s_%d_s%d.npz" % (rule, prob["seed"], s))
    if cache and os.path.exists(fn):
        z = np.load(fn)
        return z["Kn"], z["Tv"]
    Kn, Tv = (M.build_nn if rule == "nn" else M.build_patch)(prob, s, timing=timing)
    if cache:
        np.savez(fn, Kn=Kn, Tv=Tv)
    return Kn, Tv


def agg(a, n, s, R):
    """s-block field (ncs^2, ...) -> R-block field"""
    k, ncR = R // s, n // R
    x = a.reshape(ncR, k, ncR, k, *a.shape[1:]).mean(axis=(1, 3))
    return x.reshape(ncR * ncR, *a.shape[1:])


_REF = {}


def ref_R(tag, seed, s, R):
    """fine reference of the s-problem (block means stored by el_run.py) averaged to R."""
    key = (tag, seed, s, R)
    if key not in _REF:
        kind, n = AC.TAGS[tag]
        z = np.load(AC.field_file(tag, seed, s, "fine"))
        _REF[key] = {ld: agg(z["sig_" + ld].astype(float), n, s, R) for ld in EP.LOADS}
    return _REF[key]


def free_R(kind, n, R):
    return {ld: EP.free_cells(kind, n, R, ld) for ld in EP.LOADS}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
