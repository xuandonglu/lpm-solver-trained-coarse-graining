"""Cut non-overlapping interior sub-volumes from the two segmented cement-paste scans (numerical environment, numpy only).

Data: Limtong et al., Zenodo 10.5281/zenodo.8112442 (CC BY 4.0); 1 um voxels; labels 0 pore, 1 hydrates (C-S-H
dominated), 2 portlandite (CH), 3 clinker, 4 large air void (w/c 0.40 only).  Loaded with allow_pickle=False, never
written.  Array axes (a0, a1, a2) are used as (z, y, x), so a C-order ravel of a cube is the lattice order of
geom3d/lin3d (p = (k ny + j) nx + i).

For every cube size n the interior [M, L - M)^3 (M = 20-voxel margin) is tiled by n^3 tiles; tiles are drawn at random
without replacement, so all cubes of one size and scan are disjoint.  Tiles with more than 20 % large air void (label 4,
entrapped air of ~60 um chord: such a tile is a piece of an air bubble, not of paste) are skipped; tiles with a smaller
part of an air void are kept (internal free surface).  Splits (per scan):
  n = 24: test 40, cal 10 (calibration of the linear correction), train 600 + val 50 (real -> real transfer only)
  n = 48: test 10;   n = 72: test 4
usage: python -I cubes.py  ->  data/cubes_{w30,w40}_{split}.npz (labels uint8 (S, n, n, n), origin (S, 3))
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402

SCANS = {"w30": "segmented_030_412_upload.npy", "w40": "segmented_040_550_void_extracted.npy"}
MARGIN = 20
MAX_AIR = 0.2
SPLITS = {24: [("test24", 40), ("cal24", 10), ("train24", 600), ("val24", 50)], 48: [("test48", 10)],
          72: [("test72", 4)]}


def main():
    for k, (scan, fn) in enumerate(SCANS.items()):
        a = np.load(os.path.join(common.SCANS, fn), allow_pickle=False)
        assert a.dtype == np.uint8 and a.ndim == 3
        L = a.shape[0]
        for n, splits in SPLITS.items():
            nt = (L - 2 * MARGIN) // n
            tiles = np.stack(np.meshgrid(np.arange(nt), np.arange(nt), np.arange(nt), indexing="ij"), -1).reshape(-1, 3)
            rng = np.random.default_rng([2026, k, n])
            order = rng.permutation(len(tiles))
            pos = 0
            nskip = 0
            for name, S in splits:
                org, cubes = [], []
                while len(cubes) < S:
                    o = MARGIN + tiles[order[pos]] * n
                    pos += 1
                    c = a[o[0]:o[0] + n, o[1]:o[1] + n, o[2]:o[2] + n]
                    if (c == 4).mean() > MAX_AIR:
                        nskip += 1
                        continue
                    org.append(o)
                    cubes.append(c)
                org = np.array(org)
                cubes = np.stack(cubes).astype(np.uint8)
                frac = np.stack([np.bincount(c.ravel(), minlength=5) / c.size for c in cubes])
                np.savez_compressed(common.data("cubes_%s_%s.npz" % (scan, name)), labels=cubes, origin=org, n=n,
                                    frac=frac)
                print("%s %-8s n=%d S=%3d of %4d tiles | mean fractions %% pore %.1f hyd %.1f CH %.1f clk %.1f air %.1f"
                      " | cubes with air void: %d (skipped so far %d)" % (scan, name, n, S, len(tiles),
                                                                         *(100 * frac.mean(0)),
                                                                         int((frac[:, 4] > 0).sum()), nskip), flush=True)


if __name__ == "__main__":
    main()
