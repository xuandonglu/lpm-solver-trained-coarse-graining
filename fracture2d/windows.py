"""Bond windows of the fine microstructure for every coarse bond (numpy only; used for training and deployment).

Coarse square lattice ncx x ncy, cell = s x s pixels.  Undirected bonds are enumerated by the forward slots
  0: (+1, 0)   2: (0, +1)   4: (+1, +1)   6: (+1, -1)        (lin2d.SQ_OFFS order; reverse slot = k ^ 1)
Window = the two end cells plus one cell all around:
  shell 0 (axial): 3 x 4 cells -> canonical array (3s, 4s), bond along +x (columns)
  shell 1 (diagonal): 4 x 4 cells -> canonical array (4s, 4s), bond along (+1,+1) (col, row)
Canonicalisation: slot 2 -> transpose; slot 6 -> flip rows.  Any element of the coset works because the network
output is averaged over the stabiliser of the canonical bond (see STAB0/STAB1), which makes the predicted bond
field exactly equivariant under the D4 symmetry of the square lattice.
Pixel features: [Kn1, Kn2, Tv] / K0 (standard per-pixel LPM parameters, K0 = mean Kn1 of the two end cells) and a
void flag (outside the specimen).
"""
import numpy as np


def pixel_params(Cpix):
    C11, C12, C44 = Cpix[..., 0], Cpix[..., 1], Cpix[..., 2]
    return np.stack([(C11 - C12) / 2, C44 / 2, (C12 - C44) / 12], -1)


def bond_list(ncx, ncy):
    """Forward bonds: arrays (p, k, I, J, shell)."""
    I, J = np.meshgrid(np.arange(ncx), np.arange(ncy))
    I, J = I.ravel(), J.ravel()
    p = J * ncx + I
    out = []
    for k, (di, dj) in zip((0, 2, 4, 6), ((1, 0), (0, 1), (1, 1), (1, -1))):
        ok = (I + di >= 0) & (I + di < ncx) & (J + dj >= 0) & (J + dj < ncy)
        out.append(np.stack([p[ok], np.full(ok.sum(), k), I[ok], J[ok]], 1))
    b = np.concatenate(out, 0)
    shell = (b[:, 1] >= 4).astype(np.int64)
    return b, shell


def extract(Cpix, nx, ny, s):
    """Returns dict with
       w0: (n0, 4, 3s, 4s) float32 windows of axial bonds, w1: (n1, 4, 4s, 4s) diagonal bonds,
       b0/b1: (n, 4) [p, k, I, J] of those bonds, K0_0/K0_1: reference scale per bond."""
    ncx, ncy = nx // s, ny // s
    P = pixel_params(Cpix).reshape(ny, nx, 3)
    pad = 2 * s
    F = np.zeros((ny + 2 * pad, nx + 2 * pad, 4), np.float64)
    F[..., 3] = 1.0
    F[pad:pad + ny, pad:pad + nx, :3] = P
    F[pad:pad + ny, pad:pad + nx, 3] = 0.0
    Kn1 = P[..., 0]
    cellK = Kn1.reshape(ncy, s, ncx, s).mean(axis=(1, 3))          # (ncy, ncx)
    b, shell = bond_list(ncx, ncy)
    W0, W1, B0, B1, K00, K01 = [], [], [], [], [], []
    for (p, k, I, J) in b:
        if k == 0:
            r0, c0 = (J - 1) * s + pad, (I - 1) * s + pad
            w = F[r0:r0 + 3 * s, c0:c0 + 4 * s]
            K0 = 0.5 * (cellK[J, I] + cellK[J, I + 1])
        elif k == 2:
            r0, c0 = (J - 1) * s + pad, (I - 1) * s + pad
            w = F[r0:r0 + 4 * s, c0:c0 + 3 * s].transpose(1, 0, 2)
            K0 = 0.5 * (cellK[J, I] + cellK[J + 1, I])
        elif k == 4:
            r0, c0 = (J - 1) * s + pad, (I - 1) * s + pad
            w = F[r0:r0 + 4 * s, c0:c0 + 4 * s]
            K0 = 0.5 * (cellK[J, I] + cellK[J + 1, I + 1])
        else:  # k == 6, bond (I,J)->(I+1,J-1)
            r0, c0 = (J - 2) * s + pad, (I - 1) * s + pad
            w = F[r0:r0 + 4 * s, c0:c0 + 4 * s][::-1]
            K0 = 0.5 * (cellK[J, I] + cellK[J - 1, I + 1])
        w = w.copy()
        w[..., :3] /= K0
        if k < 4:
            W0.append(w); B0.append((p, k, I, J)); K00.append(K0)
        else:
            W1.append(w); B1.append((p, k, I, J)); K01.append(K0)
    t = lambda W: np.ascontiguousarray(np.array(W, np.float32).transpose(0, 3, 1, 2))
    return dict(w0=t(W0), w1=t(W1), b0=np.array(B0), b1=np.array(B1), K0_0=np.array(K00), K0_1=np.array(K01))


def stab0(x):
    """Stabiliser of an axial bond along the columns: id, flip cols, flip rows, rot180 (x: (..., H, W))."""
    return [x, x[..., ::-1], x[..., ::-1, :], x[..., ::-1, ::-1]]


def stab1(x):
    """Stabiliser of a diagonal bond along (+1,+1): id, transpose, rot180, anti-transpose."""
    xt = np.swapaxes(x, -1, -2)
    return [x, xt, x[..., ::-1, ::-1], xt[..., ::-1, ::-1]]


def neighbor_of(b, ncx):
    """q (coarse particle at the other end) and reverse slot for forward bonds b = [p, k, I, J]."""
    d = {0: (1, 0), 2: (0, 1), 4: (1, 1), 6: (1, -1)}
    di = np.array([d[k][0] for k in b[:, 1]])
    dj = np.array([d[k][1] for k in b[:, 1]])
    q = (b[:, 3] + dj) * ncx + (b[:, 2] + di)
    return q, b[:, 1] ^ 1


def scatter_bonds(N, ncx, b, vals):
    """vals (n,) on forward bonds b -> (N, 8) array, same value on (p, k) and (q, k^1)."""
    out = np.zeros((N, 8))
    q, kr = neighbor_of(b, ncx)
    out[b[:, 0], b[:, 1]] = vals
    out[q, kr] = vals
    return out


SLOT_COL = np.array([0, 0, 1, 1, 2, 2, 3, 3])


def base_from_cells(cp, b, ncx):
    """Two-end mean of per-cell parameters cp (Nc, 5: knx, kny, knp, knm, tv) on forward bonds -> (Kn, Tv) (n,)."""
    q, _ = neighbor_of(b, ncx)
    col = SLOT_COL[b[:, 1]]
    kn = 0.5 * (cp[b[:, 0], col] + cp[q, col])
    tv = 0.5 * (cp[b[:, 0], 4] + cp[q, 4])
    return kn, tv
