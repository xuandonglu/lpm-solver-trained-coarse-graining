"""Pure-numpy geometry helpers shared by cgnet.py (torch) and cgnet_np.py (numpy deployment)."""
import numpy as np

# D4 on (row, col) arrays of shape (H, W): (transpose?, flip rows?, flip cols?) applied as: T then flips
D4 = [(False, False, False), (False, False, True), (False, True, False), (False, True, True),
      (True, False, False), (True, False, True), (True, True, False), (True, True, True)]



def map_cell(r, c, H, W, g):
    """Cell (r, c) of an (H, W) grid -> its coordinates after transform g (numpy arrays)."""
    t, fr, fc = D4[g]
    if t:
        r, c, H, W = c, r, W, H
    if fr:
        r = H - 1 - r
    if fc:
        c = W - 1 - c
    return r, c, H, W


def bond_gather_index(b, ncx, ncy):
    """For forward bonds b (n, 4) [p, k, I, J] of one shell: (n, 4) arrays of transform id and patch start (r0, c0)
    in the transformed padded cell grid (padding = 2 cells)."""
    H, W = ncy + 4, ncx + 4
    d = {0: (0, 1), 2: (1, 0), 4: (1, 1), 6: (-1, 1)}          # (drow, dcol) of the second cell
    rA = b[:, 3] + 2
    cA = b[:, 2] + 2
    dr = np.array([d[k][0] for k in b[:, 1]])
    dc = np.array([d[k][1] for k in b[:, 1]])
    rB, cB = rA + dr, cA + dc
    axial = b[0, 1] in (0, 2)
    G = np.zeros((len(b), 4), np.int64)
    R0 = np.zeros((len(b), 4), np.int64)
    C0 = np.zeros((len(b), 4), np.int64)
    cnt = np.zeros(len(b), np.int64)
    for g in range(8):
        ra, ca, Hg, Wg = map_cell(rA, cA, H, W, g)
        rb, cb, _, _ = map_cell(rB, cB, H, W, g)
        if axial:
            ok = (ra == rb) & (np.abs(ca - cb) == 1)
            r0, c0 = ra - 1, np.minimum(ca, cb) - 1
        else:
            ok = ((rb - ra) == (cb - ca)) & (np.abs(rb - ra) == 1)
            r0, c0 = np.minimum(ra, rb) - 1, np.minimum(ca, cb) - 1
        idx = np.nonzero(ok)[0]
        G[idx, cnt[idx]] = g
        R0[idx, cnt[idx]] = r0[idx]
        C0[idx, cnt[idx]] = c0[idx]
        cnt[idx] += 1
    assert (cnt == 4).all()
    return G, R0, C0


def pixel_features(Cpix, nx, ny, s):
    """(4, ny + 4s, nx + 4s) float32 feature image."""
    C11, C12, C44 = Cpix[:, 0], Cpix[:, 1], Cpix[:, 2]
    kn1, kn2, tv = (C11 - C12) / 2, C44 / 2, (C12 - C44) / 12
    kref = kn1.mean()
    f = np.stack([np.log(kn1 / kref), kn2 / kn1, tv / kn1, np.zeros_like(kn1)], 0).reshape(4, ny, nx)
    pad = 2 * s
    X = np.zeros((4, ny + 2 * pad, nx + 2 * pad), np.float32)
    X[3] = 1.0
    X[:, pad:pad + ny, pad:pad + nx] = f
    return X


