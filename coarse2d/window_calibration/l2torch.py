"""L2 part 2: the coarse LPM on every bond window (torch, batched dense, fp64, GPU) and the per-bond fits.

Coarse window of bond b with coarse margin mc: the box of l2fine (margin mc) plus one GHOST layer of cells, i.e. a local
(W+2) x (H+2) square lattice of coarse particles at the cell centres (spacing s h).  Ghost + box ring are prescribed, the
box interior is free (m = 1: only the two end cells A, B of an axial bond; the 2 x 2 block of a diagonal bond).  The ghost
layer completes the bond sets (multibody sums S, TS) of the prescribed ring particles, so the free particles see the
same environment as in the global coarse model.  Cells outside the specimen do not exist (free surface).
Prescribed values: fine block means where the fine box (margin mf >= mc) has them, otherwise the affine field.
  mf == mc      : "KUBC"   (ring = affine, the same BC as the fine window)
  mf == mc + 1  : "oversampled" (fine solved on a box one cell larger; the coarse ring gets the non-affine fine block
                  means, the coarse ghost gets the affine fine ring)
Targets at the free particles: fine block-mean displacement, block-mean stress, block energy density (3 states).
Fit modes (all by least squares on relative residuals, the same three kinds of quantities as the I2 training loss):
  central : only the central bond's (Kn, Tv) per window (Levenberg-Marquardt, 2 unknowns), all other bonds = patch.
  joint   : one set of per-bond parameters for the whole specimen, fitted to ALL windows at once (L-BFGS); consistent
            combination of the overlapping windows (no double counting).
  allwin  : every bond inside the window has its own unknowns (Tikhonov to patch); the central bond's value is kept.
"""
import math
import numpy as np
import torch
import common  # noqa
import torch_lpm as T
import l2fine as LF
import windows as WIN

DEV = "cuda"
DT = torch.float64
NMAX_BATCHED = 256          # batched fp64 Cholesky on this GPU checked correct up to n = 200 for batch 4096


class SPDSolveB(torch.autograd.Function):
    """x = K^{-1} b, batched Cholesky without chunking (small matrices only, see NMAX_BATCHED), adjoint backward."""

    @staticmethod
    def forward(ctx, K, b):
        assert K.shape[-1] <= NMAX_BATCHED
        L, info = torch.linalg.cholesky_ex(K)
        if bool((info > 0).any()):
            raise FloatingPointError("window stiffness not positive definite")
        x = torch.cholesky_solve(b, L)
        ctx.save_for_backward(L, x)
        return x

    @staticmethod
    def backward(ctx, g):
        L, x = ctx.saved_tensors
        lam = torch.cholesky_solve(g.contiguous(), L)
        return -lam @ x.transpose(-1, -2), lam


class WinGeom:
    """Local lattice geometry of one orientation group k and coarse margin mc (shared by its windows)."""

    def __init__(self, k, mc, s, dev=DEV):
        Wc, Hc = LF.box_dims(k, mc)
        self.k, self.mc, self.s, self.Wc, self.Hc = k, mc, s, Wc, Hc
        nx, ny = Wc + 2, Hc + 2
        self.nx, self.ny = nx, ny
        lat = T.SquareLattice(nx, ny, float(s))
        self.lat, self.N = lat, lat.N
        a, b = lat.ij[:, 0], lat.ij[:, 1]
        layer = np.minimum.reduce([a, b, nx - 1 - a, ny - 1 - b])
        self.layer = layer
        self.free_p = np.nonzero(layer >= 2)[0]
        self.fix_p = np.nonzero(layer < 2)[0]
        self.nf = len(self.free_p)
        self.xrel = lat.x - np.array([s + Wc * s / 2.0, s + Hc * s / 2.0])
        aA, bA = mc + 1, mc + 1 + (1 if k == 6 else 0)
        di, dj = LF.DIJ[k]
        self.pA, self.pB = bA * nx + aA, (bA + dj) * nx + (aA + di)
        assert lat.nbr[self.pA, k] == self.pB and lat.nbr[self.pB, k ^ 1] == self.pA
        # (particle, shell) groups: slot lists and local B matrices (geometric validity only)
        npg = 4
        g_slots = np.zeros((2, npg), np.int64)
        g_parts = np.zeros((self.N, 2, npg + 1), np.int64)
        g_B = np.zeros((self.N, 2, npg, 2 * (npg + 1)))
        for c in (0, 1):
            ks = np.nonzero(lat.shell == c)[0]
            g_slots[c] = ks
            for i_, kk in enumerate(ks):
                q = np.where(lat.valid[:, kk], lat.nbr[:, kk], np.arange(self.N))
                g_parts[:, c, 0] = np.arange(self.N)
                g_parts[:, c, 1 + i_] = q
                n = lat.ndir[kk]
                g_B[:, c, i_, 0], g_B[:, c, i_, 1] = n[0], n[1]
                g_B[:, c, i_, 2 + 2 * i_], g_B[:, c, i_, 3 + 2 * i_] = -n[0], -n[1]
        nd = 2 * (npg + 1)
        dofs = np.stack([2 * g_parts, 2 * g_parts + 1], -1).reshape(self.N, 2, nd)
        rows = np.repeat(dofs[:, :, :, None], nd, 3)
        cols = np.repeat(dofs[:, :, None, :], nd, 2)
        self.flat_idx = torch.as_tensor((rows * (2 * self.N) + cols).reshape(-1), device=dev)
        self.g_slots = g_slots
        self.gB = torch.as_tensor(g_B, device=dev, dtype=DT)
        self.vgeo = torch.as_tensor(lat.valid, device=dev)
        self.nbr = torch.as_tensor(np.where(lat.valid, lat.nbr, 0), device=dev)
        self.ndir = torch.as_tensor(lat.ndir, device=dev, dtype=DT)
        self.l0 = torch.as_tensor(lat.l0, device=dev, dtype=DT)
        self.shell = torch.as_tensor(lat.shell, device=dev)
        self.opp = torch.as_tensor(lat.opp, device=dev)
        fd = np.zeros(2 * self.N, bool)
        fd[0::2][self.free_p] = True
        fd[1::2][self.free_p] = True
        self.fdof = torch.as_tensor(np.nonzero(fd)[0], device=dev)
        self.pdof = torch.as_tensor(np.nonzero(~fd)[0], device=dev)
        self.free_t = torch.as_tensor(self.free_p, device=dev)
        self.V = lat.V
        self.dev = dev


def window_data(geom, bk, nc, s, mf, tg, i_spec=0):
    """numpy data of the windows of one group for one specimen.
    bk: forward bonds of the group; tg: fine targets of the group at margin mf (l2fine.fine_group output)."""
    mc, N = geom.mc, geom.N
    nw = len(bk)
    Ilo, Jlo = LF.box_origin(bk, mc)
    a, b = geom.lat.ij[:, 0], geom.lat.ij[:, 1]
    gI = Ilo[:, None] - 1 + a[None, :]
    gJ = Jlo[:, None] - 1 + b[None, :]
    exists = (gI >= 0) & (gI < nc) & (gJ >= 0) & (gJ < nc)
    gcell = np.where(exists, gJ * nc + gI, 0)
    # fine box cell of every local particle
    dlt = mf - mc
    Wf, Hf = LF.box_dims(geom.k, mf)
    fa, fb = a - 1 + dlt, b - 1 + dlt
    inbox = (fa >= 0) & (fa < Wf) & (fb >= 0) & (fb < Hf)
    fcell = np.where(inbox, fb * Wf + fa, 0)
    u_f = tg["u"][:, :, fcell]                                    # (NS, nw, N, 2)
    have = inbox[None, :] & tg["exists"][:, fcell]                 # (nw, N)
    NS = tg["u"].shape[0]
    mode = "aff" if NS == 3 else "quad"
    Lh = max(Wf, Hf) * s / 2.0
    up = np.empty((NS, nw, N, 2))
    for e in range(NS):
        aff = LF.state_disp(e, geom.xrel, mode, Lh, s=s)            # block means of the prescribed field (N, 2)
        up[e] = np.where(have[:, :, None], u_f[e], aff[None])
    up[:, ~exists] = 0.0
    fp = geom.free_p
    assert inbox[fp].all()
    fcf = fcell[fp]
    return dict(exists=exists, gcell=gcell, up=up, u_t=tg["u"][:, :, fcf].astype(np.float64),
                sig_t=tg["sig"][:, :, fcf].astype(np.float64), w_t=tg["w"][:, :, fcf].astype(np.float64),
                fex=exists[:, fp], spec=np.full(nw, i_spec), bk=bk)


TSLOT = np.array([2, 3, 0, 1, 4, 5, 7, 6])     # slot map of the transpose x <-> y   (lin2d.SQ_OFFS order)
FSLOT = np.array([0, 1, 3, 2, 6, 7, 4, 5])     # slot map of the reflection y -> -y
CANON = {0: 0, 2: 0, 4: 4, 6: 4}               # orientation group -> canonical group (one batch per shell)


def canonicalize(dl, k, geom_o, geom_c):
    """Window data of group k (2: vertical bonds, 6: anti-diagonal bonds) in the frame of the canonical group
    (0 / 4): k = 2 is transposed, k = 6 reflected (y -> -y).  Particles are permuted, vectors / stresses transformed,
    and smap gives, for every canonical slot, the slot of the original (global) lattice."""
    nw = len(dl["bk"])
    if k in (0, 4):
        dl["smap"] = np.tile(np.arange(8), (nw, 1))
        return dl
    ac, bc = geom_c.lat.ij[:, 0], geom_c.lat.ij[:, 1]
    if k == 2:
        ao, bo = bc, ac
    else:
        ao, bo = ac, geom_o.ny - 1 - bc
    perm = bo * geom_o.nx + ao
    pos = -np.ones(geom_o.N, np.int64)
    pos[geom_o.free_p] = np.arange(geom_o.nf)
    fperm = pos[perm[geom_c.free_p]]
    assert (fperm >= 0).all()
    out = dict(exists=dl["exists"][:, perm], gcell=dl["gcell"][:, perm], spec=dl["spec"], bk=dl["bk"],
               fex=dl["fex"][:, fperm], w_t=dl["w_t"][:, :, fperm])
    up, ut, st = dl["up"][:, :, perm].copy(), dl["u_t"][:, :, fperm].copy(), dl["sig_t"][:, :, fperm].copy()
    if k == 2:
        up, ut = up[..., ::-1].copy(), ut[..., ::-1].copy()
        st = st[..., [1, 0, 2]].copy()
        out["smap"] = np.tile(TSLOT, (nw, 1))
    else:
        up[..., 1] *= -1
        ut[..., 1] *= -1
        st[..., 2] *= -1
        out["smap"] = np.tile(FSLOT, (nw, 1))
    out.update(up=up, u_t=ut, sig_t=st)
    return out


class WinBatch:
    """Torch tensors of a batch of windows of one geometry (possibly several specimens and, after
    canonicalize(), both orientation groups of a shell)."""

    def __init__(self, geom, dl, nc, dev=DEV):
        g = geom
        cat = lambda key: np.concatenate([d[key] for d in dl], 0) if key not in ("up", "u_t", "sig_t", "w_t") \
            else np.concatenate([d[key] for d in dl], 1)
        self.g, self.nc = g, nc
        exists = torch.as_tensor(cat("exists"), device=dev)
        self.B = exists.shape[0]
        self.exists = exists
        nbr = g.nbr
        self.valid = (g.vgeo[None] & exists[:, :, None] & exists[:, nbr]).to(DT)          # (B, N, 8)
        spec = torch.as_tensor(cat("spec"), device=dev)
        gcell = torch.as_tensor(cat("gcell"), device=dev)
        # flat index of each local directed bond into the stacked global (S, Nc, 8) arrays
        smap = torch.as_tensor(cat("smap"), device=dev) if "smap" in dl[0] else             torch.arange(8, device=dev)[None].expand(self.B, 8)
        self.gidx = ((spec[:, None, None] * (nc * nc) + gcell[:, :, None]) * 8 + smap[:, None, :])
        self.spec = spec
        self.up = torch.as_tensor(cat("up"), device=dev, dtype=DT).permute(1, 0, 2, 3).contiguous()        # (B, 3, N, 2)
        self.u_t = torch.as_tensor(cat("u_t"), device=dev, dtype=DT).permute(1, 0, 2, 3).contiguous()      # (B, 3, nf, 2)
        self.sig_t = torch.as_tensor(cat("sig_t"), device=dev, dtype=DT).permute(1, 0, 2, 3).contiguous()
        self.w_t = torch.as_tensor(cat("w_t"), device=dev, dtype=DT).permute(1, 0, 2).contiguous()
        fex = torch.as_tensor(cat("fex"), device=dev)
        self.fex = fex.to(DT)                                                                 # (B, nf)
        self.u_t = torch.nan_to_num(self.u_t) * self.fex[:, None, :, None]
        self.sig_t = torch.nan_to_num(self.sig_t) * self.fex[:, None, :, None]
        self.w_t = torch.nan_to_num(self.w_t) * self.fex[:, None, :]
        nrm = lambda x, dims: torch.sqrt((x ** 2).sum(dims)).clamp_min(1e-300)
        # displacement errors relative to the displacement increment across one cell (eps s h): the block-mean
        # displacement itself can vanish (e.g. eyy at the two cells of a horizontal bond)
        self.NS = self.up.shape[1]
        self.nu = (LF.EPS * g.s * torch.sqrt(fex.to(DT).sum(1)))[:, None].expand(-1, self.NS).clamp_min(1e-300)
        self.ns = nrm(self.sig_t, (2, 3))
        self.nw_ = nrm(self.w_t, (2,))
        # dofs of non-existing free particles get an identity row (decoupled, u = 0)
        nofree = (1 - self.fex)
        self.diag_fix = torch.stack([nofree, nofree], -1).reshape(self.B, -1)                # (B, 2 nf)
        full = self.valid.sum(-1) == 8
        oppv = self.valid[:, :, g.opp]
        of = torch.where(full[..., None], 0.5, torch.where(oppv > 0, 0.5, 1.0))
        self.opp_w = of * self.valid / g.V * g.l0                                           # (B, N, 8)
        self.bk = np.concatenate([d["bk"] for d in dl], 0)

    # ------------------------------------------------------------------------------------------
    def assemble(self, Kn, Tv):
        g = self.g
        Kn = Kn * self.valid
        Tv = Tv * self.valid
        Hs, Ms = [], []
        for c in (0, 1):
            ks = g.g_slots[c]
            kn, tv = Kn[:, :, ks], Tv[:, :, ks]
            Hs.append(torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :]))
            Ms.append(self.valid[:, :, ks])
        H = torch.stack(Hs, 2)                                       # (B, N, 2, 4, 4)
        M = torch.stack(Ms, 2)                                       # (B, N, 2, 4)
        gB = g.gB[None] * M[..., None]                               # (B, N, 2, 4, 10)
        Kloc = torch.einsum("bncad,bncae,bncef->bncdf", gB, H, gB)
        K = torch.zeros(self.B, (2 * g.N) ** 2, device=Kn.device, dtype=Kn.dtype)
        K.index_add_(1, g.flat_idx, Kloc.reshape(self.B, -1))
        K = K.view(self.B, 2 * g.N, 2 * g.N)
        return 0.5 * (K + K.transpose(1, 2))

    def forward(self, Kn, Tv):
        """Kn, Tv (B, N, 8) local bond parameters -> u (B,3,nf,2), sig (B,3,nf,3), w (B,3,nf) at the free particles."""
        g = self.g
        NS = self.NS
        K = self.assemble(Kn, Tv)
        Kff = K[:, g.fdof][:, :, g.fdof] + torch.diag_embed(self.diag_fix)
        Kfp = K[:, g.fdof][:, :, g.pdof]
        upd = self.up.reshape(self.B, NS, -1)[:, :, g.pdof]          # (B, 3, n_pdof)
        rhs = -(Kfp @ upd.transpose(1, 2))                           # (B, n_fdof, 3)
        # windows whose stiffness is not positive definite (trial parameters of a fit): solved with an identity
        # matrix and flagged; residuals() gives them a large constant residual so that the trial is rejected
        _, info = torch.linalg.cholesky_ex(Kff.detach())
        bad = info > 0
        self.last_bad = bad
        if bool(bad.any()):
            eye = torch.eye(Kff.shape[-1], device=Kff.device, dtype=Kff.dtype)
            Kff = torch.where(bad[:, None, None], eye, Kff)
            rhs = torch.where(bad[:, None, None], torch.zeros_like(rhs), rhs)
        uf = SPDSolveB.apply(Kff, rhs)                               # (B, n_fdof, 3)
        u = self.up.reshape(self.B, NS, -1).clone()
        u[:, :, g.fdof] = uf.transpose(1, 2)
        u = u.view(self.B * NS, g.N, 2)
        rep = lambda x: x.repeat_interleave(NS, 0)
        val, kn, tv = rep(self.valid), rep(Kn * self.valid), rep(Tv * self.valid)
        du = u[:, :, None, :] - u[:, g.nbr]
        dL = (du * g.ndir).sum(-1) * val
        sh = g.shell
        S = torch.stack([dL[:, :, sh == c].sum(-1) for c in (0, 1)], -1)
        TS = torch.stack([(tv * dL)[:, :, sh == c].sum(-1) for c in (0, 1)], -1)
        Si, TSi = S[:, :, sh], TS[:, :, sh]
        shx = sh[None, :].expand(g.N, 8)
        Sq, TSq = S[:, g.nbr, shx], TS[:, g.nbr, shx]
        F = (2 * kn * dL + 0.5 * (TSi + TSq) + 0.5 * tv * (Si + Sq)) * val
        fp = g.free_t
        a = rep(self.opp_w)[:, fp] * F[:, fp]
        n = g.ndir
        sig = torch.stack([(a * n[:, 0] ** 2).sum(-1), (a * n[:, 1] ** 2).sum(-1), (a * n[:, 0] * n[:, 1]).sum(-1)], -1)
        U = 0.5 * (kn[:, fp] * dL[:, fp] ** 2).sum(-1)
        for c in (0, 1):
            m = sh == c
            U = U + 0.5 * (tv[:, fp][:, :, m] * dL[:, fp][:, :, m]).sum(-1) * dL[:, fp][:, :, m].sum(-1)
        w = U / float(g.s) ** 2
        return (u.view(self.B, NS, g.N, 2)[:, :, fp], sig.view(self.B, NS, -1, 3), w.view(self.B, NS, -1))

    def residuals(self, out, wts=(1.0, 1.0, 1.0)):
        """relative residual vector per window (B, n_res); squared norm / 3 = mean-over-states loss."""
        u, sig, w = out
        fm = self.fex[:, None, :, None]
        ru = (u - self.u_t) * fm / self.nu[:, :, None, None] * math.sqrt(wts[0])
        rs = (sig - self.sig_t) * fm / self.ns[:, :, None, None] * math.sqrt(wts[1])
        rw = (w - self.w_t) * self.fex[:, None, :] / self.nw_[:, :, None] * math.sqrt(wts[2])
        r = torch.cat([ru.reshape(self.B, -1), rs.reshape(self.B, -1), rw.reshape(self.B, -1)], 1) / math.sqrt(self.NS)
        bad = getattr(self, "last_bad", None)
        if bad is not None and bool(bad.any()):
            r = torch.where(bad[:, None], torch.full_like(r, 10.0), r)
        return r

    def group_lam_min(self, Kn, Tv, parts):
        """min eigenvalue / mean Kn of the bond-space Hessians of the given local particles (both shells)."""
        g = self.g
        res = []
        for c in (0, 1):
            ks = g.g_slots[c]
            v = self.valid[:, parts][:, :, ks]
            kn, tv = Kn[:, parts][:, :, ks] * v, Tv[:, parts][:, :, ks] * v
            H = torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :])
            H = H * (v[..., :, None] * v[..., None, :])
            sc = (kn.abs().sum(-1) / v.sum(-1).clamp_min(1)).clamp_min(1e-30)
            big = (1 - v) * 1e3 * sc[..., None]
            lam = torch.linalg.eigvalsh(H + torch.diag_embed(big))[..., 0]
            res.append(lam / sc)
        return torch.stack(res, -1)                                  # (B, len(parts), 2)


# ----------------------------------------------------------------------------------------------------------------
def gather_local(flatKn, flatTv, batch):
    """global stacked (S*Nc*8,) parameters -> local (B, N, 8)."""
    return flatKn[batch.gidx] * batch.valid, flatTv[batch.gidx] * batch.valid


def fit_central(batch, Kn_g, Tv_g, K0, iters=20, wts=(1.0, 1.0, 1.0), pen=10.0, lam_floor=0.05, verbose=False):
    """Levenberg-Marquardt on the central bond's (log Kn factor, Tv offset / K0) of every window.
    Kn_g, Tv_g: stacked global base parameters (flat); K0 (B,) scale.  Returns theta (B, 2), final loss (B,)."""
    g = batch.g
    Kb, Tb = gather_local(Kn_g, Tv_g, batch)
    k, kr = g.k, g.k ^ 1
    kn0 = Kb[:, g.pA, k].clone()
    tv0 = Tb[:, g.pA, k].clone()
    B = batch.B
    th = torch.zeros(B, 2, device=batch.exists.device, dtype=DT)
    lam = torch.full((B,), 1e-2, device=th.device, dtype=DT)
    parts = torch.as_tensor([g.pA, g.pB], device=th.device)

    def resid(t):
        Kn, Tv = Kb.clone(), Tb.clone()
        kv = kn0 * torch.exp(t[:, 0])
        tvv = tv0 + K0 * t[:, 1]
        Kn[:, g.pA, k] = kv
        Kn[:, g.pB, kr] = kv
        Tv[:, g.pA, k] = tvv
        Tv[:, g.pB, kr] = tvv
        r = batch.residuals(batch.forward(Kn, Tv), wts)
        lm = batch.group_lam_min(Kn, Tv, parts)
        rp = math.sqrt(pen) * torch.relu(lam_floor - lm).reshape(B, -1)
        return torch.cat([r, rp], 1)

    with torch.no_grad():
        r = resid(th)
        f = (r ** 2).sum(1)
        f_init = f.clone()
        for it in range(iters):
            J = []
            for j in range(2):
                d = torch.zeros_like(th)
                d[:, j] = 1e-6
                J.append((resid(th + d) - r) / 1e-6)
            J = torch.stack(J, 2)                                    # (B, nr, 2)
            A = J.transpose(1, 2) @ J
            gvec = J.transpose(1, 2) @ r[:, :, None]
            Ad = A + lam[:, None, None] * torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2) + 1e-12)
            step = -torch.linalg.solve(Ad, gvec)[..., 0]
            tn = (th + step).clamp(-4.0, 4.0)
            rn = resid(tn)
            fn = (rn ** 2).sum(1)
            ok = fn < f
            th = torch.where(ok[:, None], tn, th)
            r = torch.where(ok[:, None], rn, r)
            f = torch.where(ok, fn, f)
            lam = torch.where(ok, lam * 0.3, lam * 10.0).clamp(1e-8, 1e8)
            if verbose:
                print("  LM it %d: mean loss %.4e (init %.4e), accepted %.2f" % (it, float(f.mean()), float(f_init.mean()),
                                                                            float(ok.double().mean())), flush=True)
    return th, f, f_init
