"""CPU inference of the I8 bond network (torch environment, torch): the I8 forward and an engineered forward with the SAME
weights and the SAME architecture/maths (only the execution changes).

Engineering (each switchable, see FastBondNet):
  * torch.inference_mode instead of no_grad;
  * the O_h-invariant 3x3x3 kernels and the stacked stabiliser weights of the first head layer are built once at load
    time (I8 rebuilt them in every forward);
  * channels_last_3d memory format for the conv trunk (oneDNN NDHWC path);
  * the bond heads are evaluated in chunks of bonds (the (bonds x 16*64) first-layer activations of n = 72 are ~0.55 GB
    per direction class in one piece; chunks stay in cache), one gather from a joint voxel/cell feature table, GELU
    in place;
  * the first head layer (the stacked stabiliser GEMM, ~90 % of the flops) on oneDNN instead of MKL: MKL runs AVX2 code
    on this AMD CPU, oneDNN AVX-512; optionally with the GELU fused into the GEMM (gemm="fused");
  * all bonds of one direction class are already batched in I8 (3 axial / 6 diagonal directions per call);
  * duplicate bond windows: counted (exact input equality of the head), see count_duplicates().
The post-processing to per-slot Kn, Tv (N, 18) (Voigt base x exp(o1), Tv base + Kn base o2, written to both ends) is the
I8 maths of train3d.ml_vals + Scatter, in numpy.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
I8 = os.path.normpath(os.path.join(HERE, "..", "..", "coarse3d"))     # read-only: model weights only
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.mkldnn  # noqa: E402,F401
import torch.nn.functional as F  # noqa: E402
import geom3d as G  # noqa: E402
import net3d as NT  # noqa: E402


def load_net(s):
    net = NT.BondNet(s)
    net.load_state_dict(torch.load(os.path.join(I8, "models", "bondnet_s%d.pt" % s), map_location="cpu"))
    net.eval()
    return net


class FastBondNet:
    """Same parameters as a BondNet; forward(X, nc) returns the same list of (1, nbonds, 2) outputs.

    gemm = "mkl"    first head layer by torch.addmm (MKL sgemm: AVX2 kernels on this AMD CPU, ~0.5 TFLOP/s)
           "dnn"    oneDNN inner product (MkldnnLinear, with format conversions)
           "fused"  oneDNN inner product with prepacked weights and the exact (erf) GELU fused as a post-op
                    (torch.ops.mkldnn._linear_pointwise, the kernel torch.compile uses on CPU; ~1.6 TFLOP/s here)
    fuse_conv: oneDNN convolution with fused GELU for the trunk (torch.ops.mkldnn._convolution_pointwise)."""

    def __init__(self, net, channels_last=True, chunk=4096, gemm="fused", fuse_conv=False):
        self.net, self.s, self.D = net, net.s, net.D
        self.channels_last, self.chunk, self.gemm, self.fuse_conv = channels_last, chunk, gemm, fuse_conv
        with torch.inference_mode():
            self.convs = []
            mods = list(net.trunk)
            for j, m in enumerate(mods):
                gelu_next = j + 1 < len(mods) and isinstance(mods[j + 1], torch.nn.GELU)
                if isinstance(m, NT.IsoConv3d):
                    k = torch.einsum("oim,mabc->oiabc", m.w, m.masks).contiguous()
                    if channels_last:
                        k = k.to(memory_format=torch.channels_last_3d)
                    self.convs.append(("iso", k, m.b.detach().clone(), gelu_next))
                elif isinstance(m, torch.nn.Conv3d):
                    w = m.weight.detach().clone()
                    if channels_last:
                        w = w.to(memory_format=torch.channels_last_3d)
                    self.convs.append(("c1", w, m.bias.detach().clone(), gelu_next))
                else:
                    self.convs.append(("gelu", None, None, False))
        self._heads = {}

    def heads(self, nc):
        if nc not in self._heads:
            gs = self.net.windows(nc, "cpu")
            hs = []
            with torch.inference_mode():
                for (l1, l2, l3), g in zip(self.net.heads, gs):
                    H = g["pf"].shape[0]
                    hid = l1.weight.shape[0]
                    Wst = l1.weight[:, g["pf"]].permute(1, 0, 2).reshape(-1, l1.weight.shape[1]).contiguous()  # (H*hid, nin)
                    bst = l1.bias.repeat(H).contiguous()
                    h = dict(iv=g["iv"], ic=g["ic"], Wst=Wst, bst=bst, H=H, hid=hid, l2=l2, l3=l3)
                    if self.gemm == "mkl":
                        h["WstT"] = Wst.t().contiguous()
                    elif self.gemm == "dnn":
                        lin = torch.nn.Linear(Wst.shape[1], Wst.shape[0])
                        lin.weight.data.copy_(Wst)
                        lin.bias.data.copy_(bst)
                        h["mk"] = torch.utils.mkldnn.to_mkldnn(lin)
                    else:
                        h["Wp"] = torch.ops.mkldnn._reorder_linear_weight(Wst, None)
                    hs.append(h)
            self._heads[nc] = hs
        return self._heads[nc]

    def trunk(self, X):
        x = X.contiguous(memory_format=torch.channels_last_3d) if self.channels_last else X
        skip = False
        for kind, w, b, gelu_next in self.convs:
            if kind == "gelu":
                if not skip:
                    x = F.gelu(x)
                skip = False
                continue
            pad = [1, 1, 1] if kind == "iso" else [0, 0, 0]
            if self.fuse_conv and gelu_next:
                x = torch.ops.mkldnn._convolution_pointwise(x, w, b, pad, [1, 1, 1], [1, 1, 1], 1, "gelu", [], "none")
                skip = True
            else:
                x = F.conv3d(x, w, b, padding=pad[0])
        return x

    def forward(self, X, nc):
        """X (1, 3, npd, npd, npd) float32."""
        Fv = self.trunk(X)                                                # (1, D, npd^3), maybe channels_last
        Fc = F.avg_pool3d(Fv, self.s, self.s)
        Fv = Fv.permute(0, 2, 3, 4, 1).reshape(-1, self.D)               # (npd^3, D) (free view if channels_last)
        Fc = Fc.permute(0, 2, 3, 4, 1).reshape(-1, self.D)
        T = torch.cat([Fv, Fc], 0)                                        # one feature table: voxels, then cells
        nv0 = Fv.shape[0]
        outs = []
        for h in self.heads(nc):
            if "idx" not in h:
                h["idx"] = torch.cat([h["iv"], h["ic"] + nv0], 1).contiguous()     # (nb, nslots), one gather
            idx = h["idx"]
            nb = idx.shape[0]
            out = torch.empty(nb, 2)
            H, hid = h["H"], h["hid"]
            for a in range(0, nb, self.chunk):
                b = min(nb, a + self.chunk)
                x = T[idx[a:b]].reshape(b - a, -1)                                     # (m, nin)
                if self.gemm == "fused":
                    z = torch.ops.mkldnn._linear_pointwise(x, h["Wp"], h["bst"], "gelu", [], "none")
                else:
                    if self.gemm == "dnn":
                        z = h["mk"](x.to_mkldnn()).to_dense()
                    else:
                        z = torch.addmm(h["bst"], x, h["WstT"])                       # (m, H*hid)
                    z = torch.ops.aten.gelu_(z)
                z = z.view(b - a, H, hid).mean(1)
                out[a:b] = h["l3"](F.gelu(h["l2"](z)))
            outs.append(out[None])
        return outs


# ---------------------------------------------------------------------------------------------------------------------
class Post:
    """Voigt bond bases of the canonical bond list and the scatter to per-slot (N, 18) arrays (numpy, I8 maths)."""

    def __init__(self, n, s, net):
        self.n, self.s, self.nc = n, s, n // s
        self.lat_c = G.LatticeGeom(self.nc, self.nc, self.nc, s)
        gw = net.windows(self.nc, "cpu")
        self.bonds = [g["bonds"] for g in gw]
        self.q = [self.lat_c.nbr[b[:, 0], b[:, 1]] for b in self.bonds]

    def bases(self, Cv):
        cp = G.cell_params_from_C66(G.voigt_cells(Cv, self.n, self.s), self.s / 2.0)
        Kn, Tv = G.bond_params_from_cells(self.lat_c, cp)
        return [(Kn[b[:, 0], b[:, 1]], Tv[b[:, 0], b[:, 1]]) for b in self.bonds]

    def scatter(self, outs, base):
        N = self.lat_c.N
        Kn = np.zeros((N, 18))
        Tv = np.zeros((N, 18))
        for b, q, o, (kv, tv) in zip(self.bonds, self.q, outs, base):
            o = o[0].double().numpy()
            kn = kv * np.exp(o[:, 0])
            tvv = tv + kv * o[:, 1]
            Kn[b[:, 0], b[:, 1]] = kn
            Kn[q, b[:, 1] ^ 1] = kn
            Tv[b[:, 0], b[:, 1]] = tvv
            Tv[q, b[:, 1] ^ 1] = tvv
        return Kn, Tv


def material(m, i):
    ph, c, nu_i = m["phase"][i], float(m["contrast"][i]), float(m["nu_i"][i])
    E = np.where(ph == 1, c, 1.0)
    nu = np.where(ph == 1, nu_i, 0.2)
    return ph, c, nu_i, G.iso_C(E, nu)


def count_duplicates(net, X, nc):
    """number of distinct head inputs per direction class (exact equality of the gathered window features)."""
    with torch.inference_mode():
        Fv = net.trunk(X)
        Fc = F.avg_pool3d(Fv, net.s, net.s)
        Fv = Fv.flatten(2).transpose(1, 2)[0]
        Fc = Fc.flatten(2).transpose(1, 2)[0]
        res = []
        for g in net.windows(nc, "cpu"):
            x = torch.cat([Fv[g["iv"]], Fc[g["ic"]]], 1).flatten(1)
            res.append((x.shape[0], torch.unique(x, dim=0).shape[0]))
    return res
