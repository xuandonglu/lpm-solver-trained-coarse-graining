"""Fully convolutional bond-parameter network for the coarse square lattice (torch).

Input image X (B, 4, Hp, Wp): the fine microstructure padded by 2s void pixels on every side; per pixel
  [log(Kn1/Kref), Kn2/Kn1, Tv/Kn1, void]   (standard LPM pixel parameters, Kref = sample mean of Kn1).
For each of the 8 elements g of D4: trunk(g X) (3x3 convs, stride 1) -> 1x1 reduction -> pixel-unshuffle by s
  -> cell feature map P_g (B, D, Hc, Wc), D = cred*s*s (keeps all sub-cell positions).
Coarse bond b (cells A, B) is read from the 4 transformed maps in which it has the canonical orientation
(axial: horizontal, 3 x 4 cells around it; diagonal: main diagonal, 4 x 4 cells) and the outputs of the shell head
are averaged over those 4 transforms (= the coset of the bond's stabiliser) => the predicted bond field is exactly
D4-equivariant.
Bond parameters: Kn_b = Kn_b^Voigt exp(o1), Tv_b = Tv_b^Voigt + Kn1_b^Voigt o2  (nout = 2), or a single output
for the critical stretch head (nout = 1).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgnet_geom import D4, map_cell, bond_gather_index, pixel_features  # noqa: F401


def apply_g(x, g):
    t, fr, fc = D4[g]
    if t:
        x = x.transpose(-1, -2)
    dims = []
    if fr:
        dims.append(-2)
    if fc:
        dims.append(-1)
    return torch.flip(x, dims) if dims else x


class CGNet(nn.Module):
    def __init__(self, s, cin=4, ch=32, cred=8, hid=64, nout=2):
        super().__init__()
        self.s, self.nout = s, nout
        self.trunk = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, cred, 1))
        D = cred * s * s
        self.D = D
        self.head0 = nn.Sequential(nn.Linear(D * 12, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, nout))
        self.head1 = nn.Sequential(nn.Linear(D * 16, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, nout))
        for h in (self.head0, self.head1):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def cell_maps(self, X):
        """X (B, cin, Hp, Wp) -> list over g of unfolded patch features: (B, D*12, L) and (B, D*16, L)."""
        out = []
        for g in range(8):
            P = F.pixel_unshuffle(self.trunk(apply_g(X, g)), self.s)     # (B, D, Hc, Wc)
            out.append(P)
        return out

    def forward(self, X, gi0, gi1):
        """gi0/gi1 = (G, R0, C0) torch long tensors (n, 4) for the axial / diagonal forward bonds.
        Returns o0 (B, n0, nout), o1 (B, n1, nout)."""
        maps = self.cell_maps(X)
        res = []
        for (G, R0, C0), kh, kw, head in ((gi0, 3, 4, self.head0), (gi1, 4, 4, self.head1)):
            n = G.shape[0]
            acc = 0.0
            for g in range(8):
                m = G == g                                    # (n, 4)
                if not bool(m.any()):
                    continue
                bi, ji = torch.nonzero(m, as_tuple=True)
                P = maps[g]
                U = F.unfold(P, (kh, kw))                     # (B, D*kh*kw, L)
                Wc = P.shape[-1] - kw + 1
                l = R0[bi, ji] * Wc + C0[bi, ji]
                feats = U[:, :, l].transpose(1, 2)            # (B, nsel, D*kh*kw)
                o = head(feats)                               # (B, nsel, nout)
                full = torch.zeros(X.shape[0], n, self.nout, device=X.device, dtype=o.dtype)
                full.index_add_(1, bi, o)
                acc = acc + full
            res.append(acc / 4.0)
        return res[0], res[1]
