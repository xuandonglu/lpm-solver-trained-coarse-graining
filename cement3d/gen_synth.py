"""Synthetic cement-paste-like voxel microstructures (numpy only): 3 solids + capillary pores, labels as the scans
(0 pore, 1 hydrate, 2 CH, 3 clinker).  Only the statistics of the two scans are used (phase fractions and mean chord
lengths of the scans, printed by `gen_synth.py chords`), no scan voxels.

  clinker   union of randomly oriented, polydisperse ellipsoids (random sequential addition, overlaps allowed, the
            micro3d machinery of I8) and of a thresholded Gaussian random field blob phase (irregular grains); mixture
            weight random per sample; mean radius 4-8 voxels
  CH        thresholded Gaussian random field (correlation 0.45-1.1 voxel) plus a long-wave (correlation 4) modulation
            that clusters the small crystals, inside the non-clinker space
  pores     the same with correlation 0.7-1.6, inside the remaining hydrate space
  hydrate   the rest
Fractions per sample: clinker U(0.11, 0.27), CH U(0.05, 0.13), pore U(0.03, 0.12) (both scans: clinker 15-22 %,
CH 7-12 %, pore 5-9 %).
usage: python -I gen_synth.py chords                      (phase fractions and mean chord lengths of
                                                           synthetic cubes and of the scan test cubes, n = 48)
       python -I gen_synth.py make NAME S n seed           -> data/synth_NAME.npz (labels (S, n, n, n) uint8)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import micro3d as M3  # noqa: E402


def grf_field(n, corr, rng):
    p = int(4 * corr) + 4
    f = rng.standard_normal((n + 2 * p,) * 3)
    k0 = np.fft.fftfreq(f.shape[0])[:, None, None]
    k1 = np.fft.fftfreq(f.shape[1])[None, :, None]
    k2 = np.fft.rfftfreq(f.shape[2])[None, None, :]
    g = np.exp(-2 * (np.pi * corr) ** 2 * (k0 ** 2 + k1 ** 2 + k2 ** 2))
    return np.fft.irfftn(np.fft.rfftn(f) * g, s=f.shape, axes=(0, 1, 2))[p:p + n, p:p + n, p:p + n]


def pick(field, allowed, frac_total, n3):
    """the frac_total * n3 voxels with the largest field values inside `allowed`."""
    k = int(round(frac_total * n3))
    k = min(k, int(allowed.sum()))
    v = np.where(allowed, field, -np.inf).ravel()
    idx = np.argpartition(-v, k - 1)[:k] if k > 0 else np.array([], np.int64)
    m = np.zeros(n3, bool)
    m[idx] = True
    return m.reshape(field.shape)


def sample(n, rng):
    f_clk = rng.uniform(0.11, 0.27)
    f_ch = rng.uniform(0.05, 0.13)
    f_po = rng.uniform(0.03, 0.12)
    n3 = n ** 3
    w_part = rng.uniform(0.3, 0.8)                            # share of the clinker volume made of ellipsoids
    rmean = rng.uniform(4.0, 8.0)
    img, _ = M3._rsa(n, n, n, f_clk * w_part, rng,
                     lambda r_: (rmean * (1 + 0.6 * (2 * r_.random() - 1)) *
                                 np.array([r_.uniform(1.0, 1.6), 1.0, r_.uniform(0.6, 1.0)]), M3._rand_rot(r_)))
    clk = img
    rest = f_clk - clk.mean()
    if rest > 0:
        fld = grf_field(n, rng.uniform(2.5, 4.0), rng)
        clk = clk | pick(fld, ~clk, rest, n3)
    # small CH / pore features, clustered by a long-wave modulation (scan: both are concentrated in parts of the
    # hydrate space, which keeps the hydrate chords long)
    def clustered(corr):
        a = grf_field(n, corr, rng)
        b = grf_field(n, 4.0, rng)
        return a / a.std() + rng.uniform(0.6, 1.6) * b / b.std()
    ch = pick(clustered(rng.uniform(0.45, 1.1)), ~clk, f_ch, n3)
    po = pick(clustered(rng.uniform(0.7, 1.6)), ~clk & ~ch, f_po, n3)
    lab = np.ones((n, n, n), np.uint8)
    lab[clk] = 3
    lab[ch] = 2
    lab[po] = 0
    return lab


def chords(m):
    out = []
    for ax in range(3):
        a = np.moveaxis(m, ax, -1).reshape(-1, m.shape[ax]).astype(np.int8)
        d = np.diff(np.pad(a, ((0, 0), (1, 1))), axis=1)
        L = np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1]
        out.append(L)
    L = np.concatenate(out)
    return L.mean(), np.median(L)


def chord_table(title, labs):
    print(title)
    for k, name in ((0, "pore"), (1, "hydrate"), (2, "CH"), (3, "clinker")):
        st = [chords(l == k) for l in labs]
        fr = [(l == k).mean() for l in labs]
        print("  %-8s fraction %.3f  mean chord %.2f  median %.1f" % (name, np.mean(fr), np.mean([s[0] for s in st]),
                                                                     np.mean([s[1] for s in st])))


def main():
    if sys.argv[1] == "chords":
        rng = np.random.default_rng(0)
        chord_table("synthetic, 12 cubes of n = 48", [sample(48, rng) for _ in range(12)])
        for sc, wc in (("w30", "0.30"), ("w40", "0.40")):      # test cubes of n = 48 cut by cubes.py
            fn = common.data("cubes_%s_test48.npz" % sc)
            if os.path.exists(fn):
                chord_table("scan w/c %s, 10 test cubes of n = 48" % wc, np.load(fn)["labels"])
        return
    name, S, n, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(seed)
    labs = np.stack([sample(n, rng) for _ in range(S)])
    frac = np.stack([np.bincount(l.ravel(), minlength=5) / l.size for l in labs])
    np.savez_compressed(common.data("synth_%s.npz" % name), labels=labs, n=n, frac=frac)
    print("synth_%s: %d cubes n=%d, mean fractions %% pore %.1f hyd %.1f CH %.1f clk %.1f" % (
        name, S, n, *(100 * frac.mean(0)[:4])), flush=True)


if __name__ == "__main__":
    main()
