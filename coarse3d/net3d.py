"""Exactly cubic-symmetric (O_h, 48 elements) bond-parameter network for the coarse SC lattice (torch).

Input X (B, 3, n + 4s, n + 4s, n + 4s): voxel features of the fine microstructure padded by 2s void voxels per side:
  [log(Kn1 / Kref), Tv / Kn1, void]   (standard SC voxel parameters, Kref = mean Kn1 of the sample; isotropic phases).
1. Trunk: 3x3x3 convolutions whose kernels are O_h-invariant (4 weights per channel pair: centre / 6 faces / 12 edges /
   8 corners), GELU, 1x1 reduction to D channels -> scalar feature fields, trunk(g X) = g trunk(X) exactly.
2. Every coarse bond (3 axial + 6 face-diagonal directions) reads, in its canonical frame (bond along +x, resp. along
   (1,1,0); midpoint at the origin), the D features of all voxels of its two end cells and the cell-averaged features of
   the other cells of a 4x3x3 (axial) / 4x4x3 (diagonal) cell window.
3. First head layer averaged over the stabiliser of the canonical bond (16 / 8 signed permutations that map the window
   onto itself): h = mean_g GELU(W P_g x + b)  -> invariant; then MLP -> (o1, o2).
   => the predicted bond field is exactly equivariant under the 48 cube symmetries.
4. Kn_b = Kn_b^Voigt exp(o1), Tv_b = Tv_b^Voigt + Kn_b^Voigt o2 (last layer zero-initialised: starts at Voigt).
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geom3d import SC_OFFS


def signed_perms():
    out = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), np.int64)
            for r in range(3):
                M[r, p[r]] = sg[r]
            out.append(M)
    return out


OH = signed_perms()                                                  # 48 signed permutation matrices


class IsoConv3d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        off = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        cls = (off != 0).sum(1)                                         # 0 centre, 1 face, 2 edge, 3 corner
        m = np.zeros((4, 27))
        m[cls, np.arange(27)] = 1.0
        self.register_buffer("masks", torch.as_tensor(m.reshape(4, 3, 3, 3), dtype=torch.float32))
        self.w = nn.Parameter(torch.randn(cout, cin, 4) * np.sqrt(2.0 / (cin * 27)))
        self.b = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        k = torch.einsum("oim,mabc->oiabc", self.w, self.masks)
        return F.conv3d(x, k, self.b, padding=1)


# ---------------------------------------------------------------------------------------------------------------------
# bond windows (numpy, geometry only)
# ---------------------------------------------------------------------------------------------------------------------
def canonical_window(s, axial):
    """Slots in the canonical frame (voxel units, midpoint of the bond at 0): end-cell voxel centres and the centres of
    the other window cells.  Returns (pos_vox (nv,3) float, pos_cell (nc,3) float, stabiliser list)."""
    c = np.array([1, 0, 0]) if axial else np.array([1, 1, 0])
    t = np.arange(s) - (s - 1) / 2.0
    tv = np.array(list(itertools.product(t, t, t)))
    pv = np.concatenate([tv - c * s / 2.0, tv + c * s / 2.0], 0)
    xr = [-1, 0, 1, 2]
    yr = [-1, 0, 1, 2] if not axial else [-1, 0, 1]
    zr = [-1, 0, 1]
    cells = [np.array(e) for e in itertools.product(xr, yr, zr)]
    cells = [e for e in cells if not (np.all(e == 0) or np.all(e == c))]
    pc = np.array([(e - c / 2.0) * s for e in cells])
    stab = [g for g in OH if np.all(g @ c == c) or np.all(g @ c == -c)]
    return pv, pc, stab


def _perm(P, g):
    """Index j' with P[j'] = g P[j] for every j (P: (m,3) positions)."""
    Q = P @ g.T
    key = {tuple(np.round(p * 2).astype(int)): i for i, p in enumerate(P)}
    return np.array([key[tuple(np.round(q * 2).astype(int))] for q in Q])


class BondWindows:
    """Gather indices of every forward coarse bond of an nc^3 box (fine box n = nc s padded by P = 2s)."""

    def __init__(self, nc, s):
        self.nc, self.s = nc, s
        P = 2 * s
        npd = nc * s + 2 * P
        ncp = nc + 4
        self.npd, self.ncp = npd, ncp
        self.groups = []
        for axial in (True, False):
            pv, pc, stab = canonical_window(s, axial)
            c = np.array([1, 0, 0]) if axial else np.array([1, 1, 0])
            perms = []
            for g in stab:
                pvv = _perm(pv, g)
                pcc = _perm(pc, g)
                perms.append(np.concatenate([pvv, len(pv) + pcc]))
            perms = np.array(perms)
            iv, ic, bonds = [], [], []
            slots = range(0, 6, 2) if axial else range(6, 18, 2)
            for k in slots:
                o = SC_OFFS[k]
                R = [g for g in OH if np.all(g @ c == o)][0]
                I = np.stack(np.meshgrid(np.arange(nc), np.arange(nc), np.arange(nc), indexing="ij"), -1).reshape(-1, 3)
                I = I[:, ::-1]                                                  # (i, j, k) with i fastest order
                B = I + o
                ok = np.all((B >= 0) & (B < nc), 1)
                A = I[ok]
                p = (A[:, 2] * nc + A[:, 1]) * nc + A[:, 0]
                m = (A + 2) * s + s / 2.0 + o * s / 2.0                          # midpoint, padded voxel coordinates
                vox = m[:, None, :] + (pv @ R.T)[None] - 0.5                     # voxel indices (x, y, z)
                vox = np.round(vox).astype(np.int64)
                cel = np.round((m[:, None, :] + (pc @ R.T)[None]) / s - 0.5).astype(np.int64)
                iv.append((vox[..., 2] * npd + vox[..., 1]) * npd + vox[..., 0])
                ic.append((cel[..., 2] * ncp + cel[..., 1]) * ncp + cel[..., 0])
                bonds.append(np.stack([p, np.full(len(p), k)], 1))
            self.groups.append(dict(iv=np.concatenate(iv), ic=np.concatenate(ic), bonds=np.concatenate(bonds),
                                    perms=perms, nv=len(pv), nc=len(pc)))


class BondNet(nn.Module):
    def __init__(self, s, cin=3, ch=24, D=12, hid=64):
        super().__init__()
        self.s, self.D = s, D
        self.trunk = nn.Sequential(IsoConv3d(cin, ch), nn.GELU(), IsoConv3d(ch, ch), nn.GELU(), IsoConv3d(ch, ch),
                                   nn.GELU(), nn.Conv3d(ch, D, 1))
        self.heads = nn.ModuleList()
        for axial in (True, False):
            pv, pc, _ = canonical_window(s, axial)
            nin = (len(pv) + len(pc)) * D
            l1 = nn.Linear(nin, hid)
            l3 = nn.Linear(hid, 2)
            nn.init.zeros_(l3.weight)
            nn.init.zeros_(l3.bias)
            self.heads.append(nn.ModuleList([l1, nn.Linear(hid, hid), l3]))
        self._idx = {}

    def windows(self, nc, device):
        key = (nc, str(device))
        if key not in self._idx:
            bw = BondWindows(nc, self.s)
            gs = []
            for g in bw.groups:
                pf = (g["perms"][:, :, None] * self.D + np.arange(self.D)[None, None, :]).reshape(len(g["perms"]), -1)
                gs.append(dict(iv=torch.as_tensor(g["iv"], device=device), ic=torch.as_tensor(g["ic"], device=device),
                               pf=torch.as_tensor(pf, device=device), bonds=g["bonds"]))
            self._idx[key] = gs
        return self._idx[key]

    def forward(self, X, nc):
        """X (B, cin, npd, npd, npd) -> list over (axial, diagonal) of raw outputs (B, nbonds, 2)."""
        Fv = self.trunk(X)                                                       # (B, D, npd^3)
        Fc = F.avg_pool3d(Fv, self.s, self.s)
        Fv = Fv.flatten(2).transpose(1, 2)                                       # (B, npd^3, D)
        Fc = Fc.flatten(2).transpose(1, 2)
        outs = []
        for (l1, l2, l3), g in zip(self.heads, self.windows(nc, X.device)):
            xv = Fv[:, g["iv"]]                                                  # (B, nb, nv, D)
            xc = Fc[:, g["ic"]]                                                  # (B, nb, nc, D)
            x = torch.cat([xv, xc], 2).flatten(2)                                # (B, nb, nin)
            Wst = l1.weight[:, g["pf"]].permute(1, 0, 2).reshape(-1, l1.weight.shape[1])   # (H*hid, nin)
            z = F.linear(x, Wst).view(x.shape[0], x.shape[1], g["pf"].shape[0], -1) + l1.bias
            h = F.gelu(z).mean(2)
            outs.append(l3(F.gelu(l2(h))))
        return outs


def voxel_features(phase, contrast, nu_i, n, s):
    """(3, n + 4s, ...) float32: log(Kn1/Kref), Tv/Kn1, void.  For isotropic phases Kn1 = Kn2 = r mu."""
    E = np.where(phase == 1, contrast, 1.0)
    nu = np.where(phase == 1, nu_i, 0.2)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    kref = mu.mean()
    P = 2 * s
    X = np.zeros((3, n + 2 * P, n + 2 * P, n + 2 * P), np.float32)
    X[2] = 1.0
    X[0, P:P + n, P:P + n, P:P + n] = np.log(mu / kref).reshape(n, n, n)
    X[1, P:P + n, P:P + n, P:P + n] = ((lam - mu) / (18 * mu)).reshape(n, n, n)
    X[2, P:P + n, P:P + n, P:P + n] = 0.0
    return X
