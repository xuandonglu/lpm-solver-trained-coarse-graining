"""Batched, differentiable linearised 3D SC LPM in torch (GPU, matrix-free), torch environment.

Same energy / force law / computeStress / particle energy as lin3d.py (checked in check_torch3d.py):
  dL_ik = n_k.(u_i - u_q),  F_ik = 2 Kn dL + 1/2 (TS_i^c + TS_q^c) + 1/2 Tv (S_i^c + S_q^c),  K u = sum_k F_ik n_k.
Linear solves: block-Jacobi PCG on all systems of a batch at once (no dense factorisation: the fp64 batched dense
solvers of this GPU are wrong for batch > 8).  Training uses LinSolve: forward PCG, backward = one adjoint PCG
(K_ff lam = g_f) plus autograd of -lam^T K(Kn, Tv) u (u, lam fixed), i.e. the exact gradient of the linear solve.
"""
import numpy as np
import torch
from geom3d import (LatticeGeom, SC_SHELL, LOADS, PLATES, GDIR, prescribed, cell_ijk, free_surface_cells)

NN6 = None


def _nn6(ndir):
    n = ndir
    return torch.stack([n[:, 0] * n[:, 0], n[:, 1] * n[:, 1], n[:, 2] * n[:, 2], n[:, 1] * n[:, 2], n[:, 0] * n[:, 2],
                        n[:, 0] * n[:, 1]], 1)                       # (18, 6) xx yy zz yz xz xy


def inv3(D):
    a, b, c = D[..., 0, 0], D[..., 0, 1], D[..., 0, 2]
    d, e, f = D[..., 1, 0], D[..., 1, 1], D[..., 1, 2]
    g, h, i = D[..., 2, 0], D[..., 2, 1], D[..., 2, 2]
    co00 = e * i - f * h
    co01 = -(d * i - f * g)
    co02 = d * h - e * g
    det = a * co00 + b * co01 + c * co02
    r0 = torch.stack([co00, -(b * i - c * h), b * f - c * e], -1)
    r1 = torch.stack([co01, a * i - c * g, -(a * f - c * d)], -1)
    r2 = torch.stack([co02, -(a * h - b * g), a * e - b * d], -1)
    return torch.stack([r0, r1, r2], -2) / det[..., None, None]


class SCT:
    """Matrix-free SC operator on a fixed lattice geometry (LatticeGeom)."""

    def __init__(self, lat, device="cuda", dtype=torch.float64):
        self.lat, self.dev, self.dt = lat, device, dtype
        N = lat.N
        self.N = N
        nbr = lat.nbr.copy()
        nbr[nbr < 0] = N                                             # ghost row (zeros)
        self.nbr = torch.as_tensor(nbr, device=device)
        self.valid = torch.as_tensor(lat.valid, device=device, dtype=dtype)
        self.ndir = torch.as_tensor(lat.ndir, device=device, dtype=dtype)
        self.slot = torch.arange(18, device=device)[None, :]
        self.opp = torch.as_tensor(np.arange(18) ^ 1, device=device)[None, :]
        self.shell = torch.as_tensor(SC_SHELL, device=device)
        full = lat.valid.sum(1) == 18
        of = np.where(full[:, None], 0.5, np.where(lat.valid[:, np.arange(18) ^ 1], 0.5, 1.0))
        self.stw = torch.as_tensor(of / lat.V * lat.l0[None, :] * lat.valid, device=device, dtype=dtype)
        self.nn6 = _nn6(self.ndir)
        self.s0 = self.valid[:, :6] @ self.ndir[:6]                 # (N, 3) sum of n over existing bonds per shell
        self.s1 = self.valid[:, 6:] @ self.ndir[6:]

    # --------------------------------------------------------------------------------------------- kernels
    def _pad(self, x):
        z = torch.zeros(x.shape[:-2] + (1, x.shape[-1]), device=x.device, dtype=x.dtype)
        return torch.cat([x, z], -2)

    def state(self, u, Kn, Tv):
        """dL (..., N, 18), per-slot own shell sums Si, TSi and neighbour sums Sq, TSq."""
        up = self._pad(u)
        dL = ((u.unsqueeze(-2) - up[..., self.nbr, :]) * self.ndir).sum(-1) * self.valid
        TdL = Tv * dL
        S = torch.stack([dL[..., :6].sum(-1), dL[..., 6:].sum(-1)], -1)          # (..., N, 2)
        TS = torch.stack([TdL[..., :6].sum(-1), TdL[..., 6:].sum(-1)], -1)
        Si = S[..., self.shell]
        TSi = TS[..., self.shell]
        Sq = self._pad(Si)[..., self.nbr, self.slot]
        TSq = self._pad(TSi)[..., self.nbr, self.slot]
        return dL, Si, TSi, Sq, TSq

    def forces(self, u, Kn, Tv):
        dL, Si, TSi, Sq, TSq = self.state(u, Kn, Tv)
        F = (2 * Kn * dL + 0.5 * (TSi + TSq) + 0.5 * Tv * (Si + Sq)) * self.valid
        return dL, F

    def apply(self, u, Kn, Tv):
        _, F = self.forces(u, Kn, Tv)
        return F @ self.ndir                                                      # (..., N, 3)

    def stress(self, u, Kn, Tv):
        _, F = self.forces(u, Kn, Tv)
        return (self.stw * F) @ self.nn6                                          # (..., N, 6)

    def energy(self, u, Kn, Tv):
        dL, Si, TSi, _, _ = self.state(u, Kn, Tv)
        U = 0.5 * (Kn * dL * dL).sum(-1)
        return U + 0.5 * (Si[..., 0] * TSi[..., 0] + Si[..., 6] * TSi[..., 6])

    def block_diag(self, Kn, Tv):
        Knp = self._pad(Kn)[..., self.nbr, self.opp]
        Tvp = self._pad(Tv)[..., self.nbr, self.opp]
        coef = (Kn + Knp + Tvp) * self.valid
        D = torch.einsum("...na,ad,ae->...nde", coef, self.ndir, self.ndir)
        for sl, s in ((slice(0, 6), self.s0), (slice(6, 18), self.s1)):
            T = (Tv[..., sl] * self.valid[:, sl]) @ self.ndir[sl]
            D = D + 0.5 * (T[..., :, None] * s[:, None, :] + s[:, :, None] * T[..., None, :])
        return D

    def precond(self, Kn, Tv, free):
        """Block-Jacobi inverse (..., N, 3, 3), zero on fixed particles (free: (..., N) float)."""
        return inv3(self.block_diag(Kn, Tv)) * free[..., None, None]

    # --------------------------------------------------------------------------------------------- PCG
    def pcg(self, b, Kn, Tv, M, free, tol=1e-10, maxit=20000, x0=None, check=10):
        """Solve K_ff x = b for all systems of the batch.  b, x: (..., N, 3), zero on fixed; free (..., N) float
        (broadcastable).  Stops per system at ||r|| <= tol ||b||.  Returns x, iterations, max relative residual."""
        fr = free[..., None]
        x = torch.zeros_like(b) if x0 is None else x0 * fr
        r = b - self.apply(x, Kn, Tv) * fr if x0 is not None else b.clone()
        bn = b.flatten(-2).norm(dim=-1)
        bn = torch.where(bn > 0, bn, torch.ones_like(bn))
        z = (M @ r.unsqueeze(-1)).squeeze(-1)
        p = z.clone()
        rz = (r * z).flatten(-2).sum(-1)
        act = torch.ones_like(rz)
        it = 0
        rel = torch.zeros_like(bn)
        for it in range(1, maxit + 1):
            Ap = self.apply(p, Kn, Tv) * fr
            pAp = (p * Ap).flatten(-2).sum(-1)
            alpha = torch.where(pAp != 0, rz / pAp, torch.zeros_like(rz)) * act
            x = x + alpha[..., None, None] * p
            r = r - alpha[..., None, None] * Ap
            if it % check == 0 or it == maxit:
                rel = r.flatten(-2).norm(dim=-1) / bn
                act = (rel > tol).to(b.dtype)
                if not bool(act.any()):
                    break
            z = (M @ r.unsqueeze(-1)).squeeze(-1)
            rz2 = (r * z).flatten(-2).sum(-1)
            beta = torch.where(rz != 0, rz2 / rz, torch.zeros_like(rz))
            rz = rz2
            p = z + beta[..., None, None] * p
        rel = r.flatten(-2).norm(dim=-1) / bn
        return x, it, float(rel.max())

    def solve(self, Kn, Tv, up, free, tol=1e-10, maxit=20000, M=None):
        """u with u = up on fixed particles and (K u)_free = 0.  up: (..., N, 3) zero on free; free (..., N)."""
        b = -self.apply(up, Kn, Tv) * free[..., None]
        if M is None:
            M = self.precond(Kn, Tv, free)
        x, it, rel = self.pcg(b, Kn, Tv, M, free, tol, maxit)
        return up + x, it, rel


class LinSolve(torch.autograd.Function):
    """u = solve(Kn, Tv) with exact adjoint gradient w.r.t. Kn, Tv (op: SCT; up, free fixed)."""

    @staticmethod
    def forward(ctx, Kn, Tv, op, up, free, tol):
        with torch.no_grad():
            M = op.precond(Kn, Tv, free)
            u, it, rel = op.solve(Kn, Tv, up, free, tol, M=M)
        ctx.save_for_backward(Kn, Tv, u, M)
        ctx.op, ctx.free, ctx.tol = op, free, tol
        ctx.info = (it, rel)
        return u

    @staticmethod
    def backward(ctx, g):
        Kn, Tv, u, M = ctx.saved_tensors
        op, free = ctx.op, ctx.free
        with torch.no_grad():
            lam, _, _ = op.pcg(g * free[..., None], Kn, Tv, M, free, ctx.tol)
        with torch.enable_grad():
            Kn_ = Kn.detach().requires_grad_(True)
            Tv_ = Tv.detach().requires_grad_(True)
            e = (lam * op.apply(u, Kn_, Tv_)).sum()
            gK, gT = torch.autograd.grad(e, (Kn_, Tv_))
        return -gK, -gT, None, None, None, None


# =====================================================================================================================
# coarse model with the 9 load cases (training / evaluation) and the fine reference (data generation)
# =====================================================================================================================
class LoadSet:
    """BCs of the 9 load cases on a lattice whose particles belong to blocks IJK (nc blocks per direction)."""

    def __init__(self, lat, IJK, nc, s, h, device, dtype, loads=LOADS):
        self.loads = loads
        fr, ups, mov = [], [], []
        for ld in loads:
            fixed, up, moving = prescribed(ld, lat.x, IJK, nc, s, h)
            fr.append(~fixed)
            ups.append(up)
            mov.append(np.zeros(lat.N, bool) if moving is None else moving)
        self.free = torch.as_tensor(np.array(fr), device=device, dtype=dtype)       # (L, N)
        self.up = torch.as_tensor(np.array(ups), device=device, dtype=dtype)        # (L, N, 3)
        self.moving = torch.as_tensor(np.array(mov), device=device, dtype=dtype)    # (L, N)


class CoarseT:
    """Coarse SC model: nc^3 particles at the block centres, spacing s h."""

    def __init__(self, nc, s, h=1.0, device="cuda", dtype=torch.float64, loads=LOADS):
        self.nc, self.s, self.h = nc, s, h
        self.lat = LatticeGeom(nc, nc, nc, s * h)
        self.op = SCT(self.lat, device, dtype)
        self.N = self.lat.N
        self.loads = loads
        self.bc = LoadSet(self.lat, self.lat.ijk, (nc, nc, nc), s, h, device, dtype, loads)
        self.freecell = self.bc.free.bool()                                        # (L, N)
        surf = [free_surface_cells(ld, (nc, nc, nc)) if ld in PLATES else np.zeros(self.N, bool) for ld in loads]
        self.surf = torch.as_tensor(np.array(surf), device=device)                 # (L, N) free-surface cells
        self.V = (s * h) ** 3

    def run(self, Kn, Tv, tol=1e-10, differentiable=True):
        """Kn, Tv (B, N, 18) -> dict u (B,L,N,3), sig (B,L,N,6), w (B,L,N) energy density, F (B,L) reaction."""
        Kn4, Tv4 = Kn[:, None], Tv[:, None]
        up, free = self.bc.up[None], self.bc.free[None]
        if differentiable:
            u = LinSolve.apply(Kn4, Tv4, self.op, up.expand(Kn.shape[0], -1, -1, -1), free.expand(Kn.shape[0], -1, -1), tol)
        else:
            with torch.no_grad():
                u, _, _ = self.op.solve(Kn4, Tv4, up, free, tol)
        sig = self.op.stress(u, Kn4, Tv4)
        w = self.op.energy(u, Kn4, Tv4) / self.V
        P = self.op.apply(u, Kn4, Tv4)                                            # (B, L, N, 3)
        F = torch.stack([(P[:, l, :, GDIR[ld]] * self.bc.moving[l]).sum(-1) if ld in PLATES
                         else torch.zeros(Kn.shape[0], device=Kn.device, dtype=Kn.dtype)
                         for l, ld in enumerate(self.loads)], 1)
        return dict(u=u, sig=sig, w=w, F=F)


class FineRefT:
    """Fine reference (no grad) for one n^3 voxel box and block size s: block means of u, stress, energy density,
    grip reactions for the 9 load cases (all samples of a batch at once)."""

    def __init__(self, n, s, h=1.0, device="cuda", dtype=torch.float64, loads=LOADS):
        self.n, self.s, self.h = n, s, h
        self.nc = n // s
        self.lat = LatticeGeom(n, n, n, h)
        self.op = SCT(self.lat, device, dtype)
        IJK = self.lat.ijk // s
        self.bc = LoadSet(self.lat, IJK, (self.nc,) * 3, s, h, device, dtype, loads)
        self.loads = loads
        cid = (IJK[:, 2] * self.nc + IJK[:, 1]) * self.nc + IJK[:, 0]
        self.cid = torch.as_tensor(cid, device=device)
        self.Nc = self.nc ** 3

    def bmean(self, f):
        """(..., N, k) -> (..., Nc, k) block mean."""
        out = torch.zeros(f.shape[:-2] + (self.Nc, f.shape[-1]), device=f.device, dtype=f.dtype)
        out.index_add_(f.dim() - 2, self.cid, f)
        return out / self.s ** 3

    @torch.no_grad()
    def run(self, Kn, Tv, tol=1e-10):
        Kn4, Tv4 = Kn[:, None], Tv[:, None]
        u, it, rel = self.op.solve(Kn4, Tv4, self.bc.up[None], self.bc.free[None], tol)
        sig = self.op.stress(u, Kn4, Tv4)
        U = self.op.energy(u, Kn4, Tv4)
        P = self.op.apply(u, Kn4, Tv4)
        F = torch.stack([(P[:, l, :, GDIR[ld]] * self.bc.moving[l]).sum(-1) if ld in PLATES
                         else torch.zeros(Kn.shape[0], device=Kn.device, dtype=Kn.dtype)
                         for l, ld in enumerate(self.loads)], 1)
        return dict(u=self.bmean(u), sig=self.bmean(sig), w=self.bmean(U[..., None])[..., 0] / self.h ** 3,
                    F=F, iters=it, relres=rel)
