"""Differentiable small-strain LPM (square lattice, per-bond Kn/Tv) in torch, batched, dense.

Same energy as lin2d.py (agrees with lin2d to round-off):
  U_i = 1/2 sum_j Kn_ij dL_ij^2 + 1/2 sum_c (sum_{j in c} Tv_ij dL_ij)(sum_{j in c} dL_ij),  dL = n.(u_i - u_j)
K is assembled from 10x10 local matrices of every (particle, shell) group and scattered into a dense matrix.
Runs in the torch environment (torch, numpy; no scipy/numba).
"""
import math
import numpy as np
import torch

SQ_OFFS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)])
SQ_SHELL = np.array([0, 0, 0, 0, 1, 1, 1, 1])
EPS = 1e-3
CHUNK = 4          # max batch per dense solve (see solve())
LOADS = ["ring_xx", "ring_yy", "ring_xy", "tb_ty", "tb_sx", "lr_tx"]
BCSET = {"ring_xx": "ring", "ring_yy": "ring", "ring_xy": "ring", "tb_ty": "tb", "tb_sx": "tb", "lr_tx": "lr"}


def _chol_chunks(K):
    Ls = []
    for i in range(0, K.shape[0], CHUNK):
        L, info = torch.linalg.cholesky_ex(K[i:i + CHUNK])
        if bool((info > 0).any()):
            raise FloatingPointError("coarse stiffness not positive definite (unstable bond parameters)")
        Ls.append(L)
    return Ls


def _cs_chunks(b, Ls):
    return torch.cat([torch.cholesky_solve(b[i * CHUNK:(i + 1) * CHUNK], L) for i, L in enumerate(Ls)], 0)


class SPDSolve(torch.autograd.Function):
    """x = K^{-1} b for SPD K (batched), adjoint backward: lam = K^{-1} g, dK = -lam x^T, db = lam.
    Avoids differentiating through the Cholesky factorisation (5-8x cheaper) and the batched-solver issue."""

    @staticmethod
    def forward(ctx, K, b):
        Ls = _chol_chunks(K)
        x = _cs_chunks(b, Ls)
        ctx.Ls = Ls
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        lam = _cs_chunks(g.contiguous(), ctx.Ls)
        gK = -lam @ x.transpose(-1, -2)
        return gK, lam


HX_OFFS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1),
                    (1, 1), (-1, -1), (-1, 2), (1, -2), (-2, 1), (2, -1)])
HX_SHELL = np.array([0] * 6 + [1] * 6)


class SquareLattice:
    """Same numbering/geometry as lin2d.Lattice2D(kind, nx, ny, h, origin) (square or hex)."""

    def __init__(self, nx, ny, h, kind="square", origin=(0.0, 0.0)):
        self.nx, self.ny, self.h, self.kind = nx, ny, h, kind
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
        ii, jj = ii.ravel(), jj.ravel()
        self.ij = np.stack([ii, jj], 1)
        if kind == "square":
            offs, shell = SQ_OFFS, SQ_SHELL
            x, y = (ii + 0.5) * h, (jj + 0.5) * h
            self.V = h * h
            dirs = offs.astype(float)
        else:
            offs, shell = HX_OFFS, HX_SHELL
            x, y = h * (ii + 0.5 * jj), h * (math.sqrt(3) / 2) * jj
            self.V = math.sqrt(3) / 2 * h * h
            dirs = np.stack([offs[:, 0] + 0.5 * offs[:, 1], math.sqrt(3) / 2 * offs[:, 1]], 1)
        self.x = np.stack([x + origin[0], y + origin[1]], 1)
        self.N = nx * ny
        self.NN = len(offs)
        L = np.linalg.norm(dirs, axis=1)
        self.l0 = L * h
        self.ndir = -dirs / L[:, None]
        self.shell = shell
        self.opp = np.array([k ^ 1 for k in range(self.NN)])
        ni = ii[:, None] + offs[None, :, 0]
        nj = jj[:, None] + offs[None, :, 1]
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        self.nbr = np.where(ok, nj * nx + ni, -1)
        self.valid = ok


def prescribed(load, x, IJ, ncx, ncy, s, h, L=1):
    Lx, Ly = ncx * s * h, ncy * s * h
    xc = np.array([Lx / 2, Ly / 2])
    I, J = IJ[:, 0], IJ[:, 1]
    u = np.zeros((len(x), 2))
    if load.startswith("ring"):
        e = {"ring_xx": [[1, 0], [0, 0]], "ring_yy": [[0, 0], [0, 1]], "ring_xy": [[0, .5], [.5, 0]]}[load]
        fixed = (I < L) | (I >= ncx - L) | (J < L) | (J >= ncy - L)
        u[fixed] = (x[fixed] - xc) @ (EPS * np.array(e)).T
        return fixed, u, None
    if load.startswith("tb"):
        a, b = J < L, J >= ncy - L
        dist = (ncy - L) * s * h
        d = np.array([0.0, EPS * dist]) if load == "tb_ty" else np.array([EPS * dist, 0.0])
    else:
        a, b = I < L, I >= ncx - L
        dist = (ncx - L) * s * h
        d = np.array([EPS * dist, 0.0])
    u[b] = d
    return a | b, u, b


class CoarseT:
    """Torch coarse model on a ncx x ncy square lattice of spacing s*h."""

    def __init__(self, ncx, ncy, s, h=1.0, device="cuda", dtype=torch.float64, loads=LOADS, kind="square",
                 origin=(0.0, 0.0), bc_layers=1):
        self.lat = lat = SquareLattice(ncx, ncy, s * h, kind=kind, origin=origin)
        self.ncx, self.ncy, self.s, self.h = ncx, ncy, s, h
        self.dev, self.dt = device, dtype
        N = lat.N
        self.N = N
        # groups (particle p, shell c): local particles [p, q1..q4], local B (4 bonds x 10 dofs)
        npg = lat.NN // 2                                         # bonds per (particle, shell) group
        self.npg = npg
        g_parts = np.zeros((N, 2, npg + 1), np.int64)
        g_B = np.zeros((N, 2, npg, 2 * (npg + 1)))
        g_slots = np.zeros((2, npg), np.int64)
        for c in (0, 1):
            ks = np.nonzero(lat.shell == c)[0]
            g_slots[c] = ks
            for a, k in enumerate(ks):
                q = np.where(lat.valid[:, k], lat.nbr[:, k], np.arange(N))
                g_parts[:, c, 0] = np.arange(N)
                g_parts[:, c, 1 + a] = q
                v = lat.valid[:, k][:, None]
                n = lat.ndir[k]
                g_B[:, c, a, 0] = n[0] * v[:, 0]
                g_B[:, c, a, 1] = n[1] * v[:, 0]
                g_B[:, c, a, 2 + 2 * a] = -n[0] * v[:, 0]
                g_B[:, c, a, 3 + 2 * a] = -n[1] * v[:, 0]
        nd = 2 * (npg + 1)
        dofs = np.stack([2 * g_parts, 2 * g_parts + 1], -1).reshape(N, 2, nd)
        rows = np.repeat(dofs[:, :, :, None], nd, 3)
        cols = np.repeat(dofs[:, :, None, :], nd, 2)
        self.flat_idx = torch.as_tensor((rows * (2 * N) + cols).reshape(-1), device=device)
        self.gB = torch.as_tensor(g_B, device=device, dtype=dtype)
        self.g_slots = g_slots
        self.valid = torch.as_tensor(lat.valid, device=device, dtype=dtype)
        self.nbr = torch.as_tensor(np.where(lat.valid, lat.nbr, 0), device=device)
        self.ndir = torch.as_tensor(lat.ndir, device=device, dtype=dtype)
        self.shell = torch.as_tensor(lat.shell, device=device)
        self.l0 = torch.as_tensor(lat.l0, device=device, dtype=dtype)
        full = lat.valid.sum(1) == lat.NN
        of = np.where(full[:, None], 0.5, np.where(lat.valid[:, lat.opp], 0.5, 1.0))
        self.opp_w = torch.as_tensor(of * lat.valid / lat.V * lat.l0[None, :], device=device, dtype=dtype)
        # boundary conditions
        self.loads = loads
        self.bc = {}
        for ld in loads:
            fixed, up, moving = prescribed(ld, lat.x, lat.ij, ncx, ncy, s, h, bc_layers)
            fd = np.zeros(2 * N, bool)
            fd[0::2] = fixed
            fd[1::2] = fixed
            self.bc[ld] = dict(free=torch.as_tensor(np.nonzero(~fd)[0], device=device),
                               fix=torch.as_tensor(np.nonzero(fd)[0], device=device),
                               up=torch.as_tensor(up.ravel()[fd], device=device, dtype=dtype),
                               moving=None if moving is None else torch.as_tensor(np.nonzero(moving)[0], device=device),
                               freecell=torch.as_tensor(~fixed, device=device))

    # --------------------------------------------------------------------------------------------
    def assemble(self, Kn, Tv):
        """Kn, Tv: (B, N, 8) -> K (B, 2N, 2N)."""
        Bsz = Kn.shape[0]
        N = self.N
        Kn = Kn * self.valid
        Tv = Tv * self.valid
        Hs = []
        for c in (0, 1):
            ks = self.g_slots[c]
            kn = Kn[:, :, ks]                       # (B, N, 4)
            tv = Tv[:, :, ks]
            H = torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :])
            Hs.append(H)
        H = torch.stack(Hs, 2)                      # (B, N, 2, 4, 4)
        Kloc = torch.einsum("ncad,bncae,ncef->bncdf", self.gB, H, self.gB)   # (B, N, 2, 10, 10)
        K = torch.zeros(Bsz, (2 * N) * (2 * N), device=Kn.device, dtype=Kn.dtype)
        K.index_add_(1, self.flat_idx, Kloc.reshape(Bsz, -1))
        K = K.view(Bsz, 2 * N, 2 * N)
        return 0.5 * (K + K.transpose(1, 2))

    def group_hessians(self, Kn, Tv):
        """Bond-space Hessians H_pc = diag(Kn) + (t 1^T + 1 t^T)/2 of every (particle, shell) group, (B, N, 2, 4, 4);
        rows/cols of missing bonds replaced by a large diagonal so that they do not affect the spectrum."""
        Hs = []
        for c in (0, 1):
            ks = self.g_slots[c]
            v = self.valid[:, ks]                              # (N, 4)
            kn = Kn[:, :, ks] * v
            tv = Tv[:, :, ks] * v
            H = torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :])
            H = H * (v[:, :, None] * v[:, None, :])
            big = (1 - v) * 1e3 * Kn.abs().mean()
            Hs.append(H + torch.diag_embed(big.expand_as(kn)))
        return torch.stack(Hs, 2)

    def stability_penalty(self, Kn, Tv, frac=0.05):
        """sum relu(frac * mean Kn_group - lambda_min)^2 / scale^2 (local stability: every particle's bond energy PD)."""
        H = self.group_hessians(Kn, Tv)
        lam = torch.linalg.eigvalsh(H)[..., 0]                # (B, N, 2)
        sc = Kn.abs().mean((1, 2))[:, None, None]
        viol = torch.relu(frac * sc - lam) / sc
        return (viol ** 2).sum((1, 2)), lam / sc

    def solve(self, K, ld):
        bc = self.bc[ld]
        fr, fx = bc["free"], bc["fix"]
        Kff = K[:, fr][:, :, fr]
        Kfp = K[:, fr][:, :, fx]
        rhs = -(Kfp @ bc["up"])
        # Cholesky (K_ff is SPD), in chunks of CHUNK matrices.  NOTE: on this machine (torch 2.9, CUDA 13,
        # RTX 5090) batched fp64 solves with batch > 8 and n ~ 1000 are WRONG (Cholesky: NaN values; LU: NaN or
        # wrong gradients), while batches <= 8 agree with the CPU to 1e-15.
        uf = SPDSolve.apply(Kff, rhs.unsqueeze(-1)).squeeze(-1)
        u = torch.zeros(K.shape[0], 2 * self.N, device=K.device, dtype=K.dtype)
        u[:, fr] = uf
        u[:, fx] = bc["up"]
        return u

    def bond_forces(self, u, Kn, Tv):
        """u (B, 2N) -> dL, F (B, N, 8)."""
        uu = u.view(u.shape[0], self.N, 2)
        du = uu[:, :, None, :] - uu[:, self.nbr]                # (B, N, 8, 2)
        dL = (du * self.ndir).sum(-1) * self.valid
        sh = self.shell
        S = torch.stack([dL[:, :, sh == c].sum(-1) for c in (0, 1)], -1)          # (B, N, 2)
        TS = torch.stack([(Tv * dL)[:, :, sh == c].sum(-1) for c in (0, 1)], -1)
        Si = S[:, :, sh]                                        # (B, N, 8)
        TSi = TS[:, :, sh]
        Sq = S[:, self.nbr, sh[None, :].expand(self.N, self.lat.NN)]
        TSq = TS[:, self.nbr, sh[None, :].expand(self.N, self.lat.NN)]
        F = (2 * Kn * dL + 0.5 * (TSi + TSq) + 0.5 * Tv * (Si + Sq)) * self.valid
        return dL, F

    def stress(self, u, Kn, Tv):
        dL, F = self.bond_forces(u, Kn, Tv)
        a = self.opp_w * F
        n = self.ndir
        return torch.stack([(a * n[:, 0] ** 2).sum(-1), (a * n[:, 1] ** 2).sum(-1), (a * n[:, 0] * n[:, 1]).sum(-1)], -1)

    def energy_density(self, u, Kn, Tv):
        dL, _ = self.bond_forces(u, Kn, Tv)
        U = 0.5 * (Kn * dL * dL).sum(-1)
        for c in (0, 1):
            m = self.shell == c
            U = U + 0.5 * (Tv[:, :, m] * dL[:, :, m]).sum(-1) * dL[:, :, m].sum(-1)
        return U / (self.s * self.h) ** 2

    def run(self, Kn, Tv, loads=None):
        loads = loads or self.loads
        K = self.assemble(Kn, Tv)
        out = {}
        for ld in loads:
            u = self.solve(K, ld)
            sig = self.stress(u, Kn, Tv)
            w = self.energy_density(u, Kn, Tv)
            bc = self.bc[ld]
            F = None
            if bc["moving"] is not None:
                P = torch.einsum("bij,bj->bi", K, u).view(u.shape[0], self.N, 2)
                F = P[:, bc["moving"]].sum(1)
            out[ld] = dict(u=u.view(u.shape[0], self.N, 2), sig=sig, w=w, F=F)
        return out


def losses(model, out, ref, loads=None, wts=(1.0, 1.0, 1.0)):
    """Squared relative errors (per sample, mean over loads) of displacement, stress field and the global measure.
    ref[ld]: dict of tensors u (B,N,2), sig (B,N,3), F (B,2) or None."""
    loads = loads or model.loads
    Lu = Ls = Lg = 0.0
    for ld in loads:
        fc = model.bc[ld]["freecell"]
        a, b = ref[ld], out[ld]
        du = ((b["u"] - a["u"])[:, fc] ** 2).sum((1, 2)) / (a["u"][:, fc] ** 2).sum((1, 2))
        ds = ((b["sig"] - a["sig"])[:, fc] ** 2).sum((1, 2)) / (a["sig"][:, fc] ** 2).sum((1, 2))
        if b["F"] is not None:
            k = 1 if ld == "tb_ty" else 0
            dg = ((b["F"][:, k] - a["F"][:, k]) / a["F"][:, k]) ** 2
        else:
            k = {"ring_xx": 0, "ring_yy": 1, "ring_xy": 2}[ld]
            sa = a["sig"][:, fc, k].mean(1)
            sb = b["sig"][:, fc, k].mean(1)
            dg = ((sb - sa) / sa) ** 2
        Lu = Lu + du
        Ls = Ls + ds
        Lg = Lg + dg
    n = len(loads)
    return wts[0] * Lu / n, wts[1] * Ls / n, wts[2] * Lg / n
