"""Batched, differentiable linearised 3D SC LPM with per-specimen voids (torch, GPU, torch environment).

Copy of the I8 operator (coarse3d/torch_lpm3d.SCT, read-only there) generalised to removed particles: the lattice is the
full n^3 grid, every specimen of a batch carries its own solid mask, and a bond exists only if both ends are solid
(per-specimen validity (B, 1, N, 18) replaces the geometric one).  This is exactly the LPM-C model with the void
particles removed: shell sums S, TS run over existing bonds only, computeStress uses the opp_flag surface rule of the
removed neighbours (1/2 if the particle keeps all 18 bonds or the opposite bond exists, else 1), void particles are
fixed at u = 0 and have no bonds (checked against lin3d with an active-site lattice).
Without a mask (solid = None) everything reduces to the I8 operator bit for bit.
LinSolve (exact adjoint gradient) is the I8 class: it only calls op.precond / solve / pcg / apply.
"""
import numpy as np
import torch
from geom3d import LatticeGeom, SC_SHELL, LOADS, PLATES, GDIR, free_surface_cells
from torch_lpm3d import inv3, _nn6, LinSolve, LoadSet


class SCTV:
    def __init__(self, lat, device="cuda", dtype=torch.float64):
        self.lat, self.dev, self.dt = lat, device, dtype
        N = lat.N
        self.N = N
        nbr = lat.nbr.copy()
        nbr[nbr < 0] = N
        self.nbr = torch.as_tensor(nbr, device=device)
        self.gvalid = torch.as_tensor(lat.valid, device=device)
        self.ndir = torch.as_tensor(lat.ndir, device=device, dtype=dtype)
        self.slot = torch.arange(18, device=device)[None, :]
        self.opp = torch.as_tensor(np.arange(18) ^ 1, device=device)[None, :]
        self.shell = torch.as_tensor(SC_SHELL, device=device)
        self.l0 = torch.as_tensor(lat.l0, device=device, dtype=dtype)
        self.Vp = lat.V
        self.nn6 = _nn6(self.ndir)
        self.eye = torch.eye(3, device=device, dtype=dtype)
        self.set_mask(None)

    def set_mask(self, solid):
        """solid: None (no voids) or (B, N) bool tensor.  Sets valid (B|1, 1, N, 18), stress weights, shell sums."""
        g = self.gvalid
        if solid is None:
            v = g[None, None]
        else:
            sp = torch.cat([solid, torch.zeros_like(solid[:, :1])], 1)
            v = (g[None] & solid[:, :, None] & sp[:, self.nbr])[:, None]
        self.vb = v
        self.valid = v.to(self.dt)
        full = v.sum(-1, keepdim=True) == 18
        oppv = v[..., self.opp[0]]
        of = torch.where(full | oppv, torch.full_like(self.valid, 0.5), torch.ones_like(self.valid))
        self.stw = of / self.Vp * self.l0 * self.valid
        self.s0 = self.valid[..., :6] @ self.ndir[:6]
        self.s1 = self.valid[..., 6:] @ self.ndir[6:]

    def _pad(self, x):
        z = torch.zeros(x.shape[:-2] + (1, x.shape[-1]), device=x.device, dtype=x.dtype)
        return torch.cat([x, z], -2)

    def state(self, u, Kn, Tv):
        up = self._pad(u)
        dL = ((u.unsqueeze(-2) - up[..., self.nbr, :]) * self.ndir).sum(-1) * self.valid
        TdL = Tv * dL
        S = torch.stack([dL[..., :6].sum(-1), dL[..., 6:].sum(-1)], -1)
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
        return F @ self.ndir

    def stress(self, u, Kn, Tv):
        _, F = self.forces(u, Kn, Tv)
        return (self.stw * F) @ self.nn6

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
            T = (Tv[..., sl] * self.valid[..., sl]) @ self.ndir[sl]
            D = D + 0.5 * (T[..., :, None] * s[..., None, :] + s[..., :, None] * T[..., None, :])
        return D

    def precond(self, Kn, Tv, free):
        """Block-Jacobi inverse, zero on fixed / void particles (their identity-padded blocks are never used)."""
        D = self.block_diag(Kn, Tv)
        D = D + (1 - free)[..., None, None] * self.eye
        return inv3(D) * free[..., None, None]

    def pcg(self, b, Kn, Tv, M, free, tol=1e-10, maxit=20000, x0=None, check=10):
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
        self.last_iters = it
        return x, it, float(rel.max())

    def solve(self, Kn, Tv, up, free, tol=1e-10, maxit=20000, M=None):
        b = -self.apply(up, Kn, Tv) * free[..., None]
        if M is None:
            M = self.precond(Kn, Tv, free)
        x, it, rel = self.pcg(b, Kn, Tv, M, free, tol, maxit)
        return up + x, it, rel


# =====================================================================================================================
class CoarseTV:
    """Coarse SC model (nc^3 grid, spacing s h) with per-specimen cell masks; 9 load cases of I8."""

    def __init__(self, nc, s, h=1.0, device="cuda", dtype=torch.float64, loads=LOADS):
        self.nc, self.s, self.h = nc, s, h
        self.lat = LatticeGeom(nc, nc, nc, s * h)
        self.op = SCTV(self.lat, device, dtype)
        self.N = self.lat.N
        self.loads = loads
        self.bc = LoadSet(self.lat, self.lat.ijk, (nc, nc, nc), s, h, device, dtype, loads)
        self.freecell = self.bc.free.bool()                                        # (L, N) BC-free cells
        surf = [free_surface_cells(ld, (nc, nc, nc)) if ld in PLATES else np.zeros(self.N, bool) for ld in loads]
        self.surf = torch.as_tensor(np.array(surf), device=device)
        self.V = (s * h) ** 3
        self.dt = dtype

    def masks(self, cell):
        """cell (B, N) bool or None -> free (B|1, L, N), up (B|1, L, N, 3)."""
        if cell is None:
            return self.bc.free[None], self.bc.up[None]
        cm = cell.to(self.dt)[:, None]
        return self.bc.free[None] * cm, self.bc.up[None] * cm[..., None]

    def run(self, Kn, Tv, cell=None, tol=1e-10, differentiable=True, maxit=20000):
        """Kn, Tv (B, N, 18) (zero on non-existing bonds after masking), cell (B, N) bool -> u, sig, w, F."""
        self.op.set_mask(cell)
        Kn = Kn * self.op.valid[:, 0]
        Tv = Tv * self.op.valid[:, 0]
        Kn4, Tv4 = Kn[:, None], Tv[:, None]
        free, up = self.masks(cell)
        B = Kn.shape[0]
        if differentiable:
            u = LinSolve.apply(Kn4, Tv4, self.op, up.expand(B, -1, -1, -1), free.expand(B, -1, -1), tol)
        else:
            with torch.no_grad():
                u, it, rel = self.op.solve(Kn4, Tv4, up, free, tol, maxit=maxit)
                self.info = (it, rel)
        sig = self.op.stress(u, Kn4, Tv4)
        w = self.op.energy(u, Kn4, Tv4) / self.V
        P = self.op.apply(u, Kn4, Tv4)
        F = torch.stack([(P[:, l, :, GDIR[ld]] * self.bc.moving[l]).sum(-1) if ld in PLATES
                         else torch.zeros(B, device=Kn.device, dtype=Kn.dtype)
                         for l, ld in enumerate(self.loads)], 1)
        return dict(u=u, sig=sig, w=w, F=F)


class FineRefTV:
    """Fine reference with voids: block means for the s^3 blocks (stress and energy: sum / s^3, voids count as zero;
    displacement: mean over the solid particles of the block), grip reactions.  No grad."""

    def __init__(self, n, s, h=1.0, device="cuda", dtype=torch.float64, loads=LOADS):
        self.n, self.s, self.h = n, s, h
        self.nc = n // s
        self.lat = LatticeGeom(n, n, n, h)
        self.op = SCTV(self.lat, device, dtype)
        IJK = self.lat.ijk // s
        self.bc = LoadSet(self.lat, IJK, (self.nc,) * 3, s, h, device, dtype, loads)
        self.loads = loads
        cid = (IJK[:, 2] * self.nc + IJK[:, 1]) * self.nc + IJK[:, 0]
        self.cid = torch.as_tensor(cid, device=device)
        self.Nc = self.nc ** 3
        self.dt = dtype

    def bsum(self, f):
        out = torch.zeros(f.shape[:-2] + (self.Nc, f.shape[-1]), device=f.device, dtype=f.dtype)
        out.index_add_(f.dim() - 2, self.cid, f)
        return out

    @torch.no_grad()
    def run(self, Kn, Tv, solid=None, tol=1e-10, maxit=40000):
        self.op.set_mask(solid)
        Kn = Kn * self.op.valid[:, 0]
        Tv = Tv * self.op.valid[:, 0]
        Kn4, Tv4 = Kn[:, None], Tv[:, None]
        if solid is None:
            free, up = self.bc.free[None], self.bc.up[None]
            sm = torch.ones(1, 1, self.lat.N, device=Kn.device, dtype=self.dt)
        else:
            sm = solid.to(self.dt)[:, None]
            free, up = self.bc.free[None] * sm, self.bc.up[None] * sm[..., None]
        u, it, rel = self.op.solve(Kn4, Tv4, up, free, tol, maxit=maxit)
        u = u * sm[..., None]
        sig = self.op.stress(u, Kn4, Tv4)
        U = self.op.energy(u, Kn4, Tv4)
        P = self.op.apply(u, Kn4, Tv4)
        F = torch.stack([(P[:, l, :, GDIR[ld]] * self.bc.moving[l]).sum(-1) if ld in PLATES
                         else torch.zeros(Kn.shape[0], device=Kn.device, dtype=Kn.dtype)
                         for l, ld in enumerate(self.loads)], 1)
        cnt = self.bsum(sm[..., None])                                          # (B|1, 1, Nc, 1) solid voxels / block
        um = self.bsum(u) / cnt.clamp_min(1.0)
        return dict(u=um, sig=self.bsum(sig) / self.s ** 3, w=self.bsum(U[..., None])[..., 0] / self.s ** 3 / self.h ** 3,
                    F=F, iters=it, relres=rel)
