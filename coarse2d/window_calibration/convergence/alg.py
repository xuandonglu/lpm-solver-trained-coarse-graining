"""Skeptic test of L2: a better NON-ML per-bond local calibration -- adaptive local-global (ALG) upscaling
(Chen & Durlofsky 2003 style), only LOCAL fine solves + cheap GLOBAL COARSE solves, no learning, no global fine solve.

ALG iteration (start: the L2 parameters, or patch):
  1. global COARSE solve of the specimen with the current per-bond parameters, for the load states (6 I2 loads or the
     3 KUBC ones)
  2. local FINE solve on every bond window (margin m): window-ring pixels get the global coarse displacement
     interpolated (bilinear) to the pixels, pixels of global grip / KUBC-ring cells get the exact global BC
  3. joint per-bond fit (as L2 'joint': one parameter set per specimen, all windows, L-BFGS, the same stability
     penalty) of the coarse window model (ring + ghost + global-BC cells prescribed with the same data) to the fine
     block means u, sig, w of the free cells (optionally only the central cells: buffered)
usage: python alg.py S N1 [--m 1 --nit 3 --start l2j_m1 --loads all --norm spec --match all]
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import common  # noqa
import cg
import lin2d
import l2fine as LF
import windows as WIN
import evalm as EV
import l2run as LR
import torch

torch.set_num_threads(2)
import l2torch as L2
import torch_lpm as T

DEV, DT = L2.DEV, L2.DT
BCSET = cg.BCSET


# ------------------------------------------------------------------------------------------------------------ fine
def pixel_fields(n, s, U, loads):
    """prescribed pixel displacement per state: global BC exact on global BC cells, else the coarse field interpolated.
    Returns P (NS, n*n, 2), Gpix {bcset: (n*n,) bool}, Pc (NS, Nc, 2) cell block means of P, Gc {bcset: (Nc,)}."""
    nc = n // s
    jj, ii = np.mgrid[0:n, 0:n]
    x = np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], 1)
    IJ = np.stack([ii.ravel() // s, jj.ravel() // s], 1)
    xc = (np.arange(nc) + 0.5) * s
    P = np.zeros((len(loads), n * n, 2))
    Gpix, Gc = {}, {}
    Ic, Jc = np.meshgrid(np.arange(nc), np.arange(nc))
    IJc = np.stack([Ic.ravel(), Jc.ravel()], 1)
    for e, ld in enumerate(loads):
        fixed, up, _ = cg.prescribed(ld, x, IJ, nc, nc, s, 1.0)
        Ug = U[ld].reshape(nc, nc, 2)
        for d in range(2):
            f = RegularGridInterpolator((xc, xc), Ug[:, :, d], method="linear", bounds_error=False, fill_value=None)
            P[e, :, d] = f(np.stack([x[:, 1], x[:, 0]], 1))
        P[e, fixed] = up[fixed]
        Gpix[BCSET[ld]] = fixed
        Gc[BCSET[ld]] = cg.prescribed(ld, np.zeros((nc * nc, 2)), IJc, nc, nc, s, 1.0)[0]
    Pc = np.stack([cg.block_mean(P[e], IJ, nc, nc) for e in range(len(loads))])
    return P, Gpix, Pc, Gc


def fine_group_bc(Cpix, n, s, k, bk, m, P, Gpix, bcsets, h=1.0):
    """l2fine.fine_group with general prescribed data: ring pixels <- P, global BC pixels <- P (exact BC)."""
    Wc, Hc = LF.box_dims(k, m)
    Wp, Hp = Wc * s, Hc * s
    Ilo, Jlo = LF.box_origin(bk, m)
    nw = len(bk)
    gap = 2
    SW, SH = Wp + gap, Hp + gap
    ncol = int(np.ceil(np.sqrt(nw)))
    nrow = int(np.ceil(nw / ncol))
    GX, GY = ncol * SW, nrow * SH
    lj, li = np.mgrid[0:Hp, 0:Wp]
    li, lj = li.ravel(), lj.ravel()
    gi = Ilo[:, None] * s + li[None, :]
    gj = Jlo[:, None] * s + lj[None, :]
    inside = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
    wcol, wrow = np.arange(nw) % ncol, np.arange(nw) // ncol
    site = (wrow[:, None] * SH + lj[None, :]) * GX + wcol[:, None] * SW + li[None, :]
    act = np.zeros(GX * GY, bool)
    act[site[inside]] = True
    lat = lin2d.Lattice2D("square", GX, GY, h, active=act)
    pid = lat.site_of[site]
    gpi = (gj * n + gi)
    Cp = np.zeros((lat.N, 3))
    Cp[pid[inside]] = Cpix[gpi[inside]]
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cp))
    K, _, _ = lat.stiffness(Kn, Tv)
    Kc = K.tocsr()
    gp = np.full(lat.N, -1, np.int64)
    gp[pid[inside]] = gpi[inside]
    ci, cj = li // s, lj // s
    ring = (ci == 0) | (ci == Wc - 1) | (cj == 0) | (cj == Hc - 1)
    cell = (np.arange(nw)[:, None] * (Hc * Wc) + (cj * Wc + ci)[None, :])
    cid = np.full(lat.N, -1, np.int64)
    cid[pid[inside]] = cell[inside]
    ncl = nw * Hc * Wc
    cnt = np.bincount(cid, minlength=ncl).astype(float)
    exists = cnt > 0
    NS = len(bcsets)
    U = np.zeros((NS, ncl, 2))
    SG = np.zeros((NS, ncl, 3))
    WE = np.zeros((NS, ncl))
    c = np.maximum(cnt, 1)
    for bset in dict.fromkeys(bcsets):
        fixed_mask = np.zeros(lat.N, bool)
        fixed_mask[pid[inside & ring[None, :]]] = True
        fixed_mask |= Gpix[bset][np.maximum(gp, 0)] & (gp >= 0)
        fidx = np.nonzero(fixed_mask)[0]
        fdofs = np.concatenate([2 * fidx, 2 * fidx + 1])
        free = np.ones(2 * lat.N, bool)
        free[fdofs] = False
        Kff = Kc[free][:, free].tocsc()
        Kfp = Kc[free][:, fdofs]
        lu = spla.splu(Kff, permc_spec="COLAMD")
        for e in [e for e in range(NS) if bcsets[e] == bset]:
            ua = P[e][gp[fidx]]
            u = np.zeros(2 * lat.N)
            u[fdofs] = np.concatenate([ua[:, 0], ua[:, 1]])
            u[free] = lu.solve(-(Kfp @ u[fdofs]))
            uu = u.reshape(-1, 2)
            sg = lat.stress(u, Kn, Tv)
            en = lat.energy_per_particle(u, Kn, Tv)
            for d in range(2):
                U[e, :, d] = np.bincount(cid, uu[:, d], minlength=ncl) / c
            for d in range(3):
                SG[e, :, d] = np.bincount(cid, sg[:, d], minlength=ncl) / c
            WE[e] = np.bincount(cid, en, minlength=ncl) / (s * h) ** 2
    U[:, ~exists] = np.nan
    SG[:, ~exists] = np.nan
    WE[:, ~exists] = np.nan
    sh = (nw, Hc * Wc)
    return dict(u=U.reshape(NS, *sh, 2), sig=SG.reshape(NS, *sh, 3), w=WE.reshape(NS, *sh), exists=exists.reshape(sh))


# ---------------------------------------------------------------------------------------------------------- coarse
def window_data_bc(geom, bk, nc, s, m, tg, Pc, Gc, bcsets, i_spec):
    N = geom.N
    nw = len(bk)
    Ilo, Jlo = LF.box_origin(bk, m)
    a, b = geom.lat.ij[:, 0], geom.lat.ij[:, 1]
    gI = Ilo[:, None] - 1 + a[None, :]
    gJ = Jlo[:, None] - 1 + b[None, :]
    exists = (gI >= 0) & (gI < nc) & (gJ >= 0) & (gJ < nc)
    gcell = np.where(exists, gJ * nc + gI, 0)
    Wf, Hf = LF.box_dims(geom.k, m)
    fa, fb = a - 1, b - 1
    inbox = (fa >= 0) & (fa < Wf) & (fb >= 0) & (fb < Hf)
    fcell = np.where(inbox, fb * Wf + fa, 0)
    NS = len(bcsets)
    up = np.stack([Pc[e][gcell] for e in range(NS)])               # (NS, nw, N, 2)
    up[:, ~exists] = 0.0
    fp = geom.free_p
    fcf = fcell[fp]
    fixf = np.stack([Gc[bcsets[e]][gcell[:, fp]] & exists[:, fp] for e in range(NS)], 1).astype(float)
    return dict(exists=exists, gcell=gcell, up=up, u_t=tg["u"][:, :, fcf], sig_t=tg["sig"][:, :, fcf],
                w_t=tg["w"][:, :, fcf], fex=exists[:, fp], spec=np.full(nw, i_spec), bk=bk, fixf=fixf)


def canon_bc(d0, k, go, gc):
    fx = d0.pop("fixf")
    out = L2.canonicalize(d0, k, go, gc)
    if k in (0, 4):
        out["fixf"] = fx
        return out
    ac, bc_ = gc.lat.ij[:, 0], gc.lat.ij[:, 1]
    ao, bo = (bc_, ac) if k == 2 else (ac, go.ny - 1 - bc_)
    perm = bo * go.nx + ao
    pos = -np.ones(go.N, np.int64)
    pos[go.free_p] = np.arange(go.nf)
    out["fixf"] = fx[:, :, pos[perm[gc.free_p]]]
    return out


class WinBatchBC(L2.WinBatch):
    """WinBatch with per-state fixed free particles (global BC cells) and a residual mask; stress / energy residuals
    normalised per (specimen, state) ('spec', ~ the global relative field error) or per window ('win', as L2)."""

    def __init__(self, geom, dl, nc, center=None, norm="spec"):
        super().__init__(geom, dl, nc)
        g = self.g
        self.fixf = torch.as_tensor(np.concatenate([d["fixf"] for d in dl], 0), device=DEV, dtype=DT)  # (B,NS,nf)
        rm = self.fex[:, None, :] * (1 - self.fixf)
        if center is not None:
            rm = rm * torch.as_tensor(center, device=DEV, dtype=DT)[None, None, :]
        self.rm = rm
        self.u_t = self.u_t * rm[..., None]
        self.sig_t = self.sig_t * rm[..., None]
        self.w_t = self.w_t * rm
        cnt = rm.sum(2)                                                          # (B, NS)
        self.nu = (LF.EPS * g.s * torch.sqrt(cnt)).clamp_min(1e-300)
        s2 = (self.sig_t ** 2).sum((2, 3))
        w2 = (self.w_t ** 2).sum(2)
        if norm == "win":
            self.ns, self.nw_ = s2.sqrt().clamp_min(1e-300), w2.sqrt().clamp_min(1e-300)
        else:
            nsp = int(self.spec.max()) + 1
            z = lambda: torch.zeros(nsp, self.NS, device=DEV, dtype=DT)
            C = z().index_add_(0, self.spec, cnt)
            S2 = z().index_add_(0, self.spec, s2)
            W2 = z().index_add_(0, self.spec, w2)
            self.ns = (torch.sqrt(cnt) * torch.sqrt(S2 / C.clamp_min(1))[self.spec]).clamp_min(1e-300)
            self.nw_ = (torch.sqrt(cnt) * torch.sqrt(W2 / C.clamp_min(1))[self.spec]).clamp_min(1e-300)

    def forward(self, Kn, Tv):
        g = self.g
        NS, B = self.NS, self.B
        K = self.assemble(Kn, Tv)
        Kff = K[:, g.fdof][:, :, g.fdof] + torch.diag_embed(self.diag_fix)
        Kfp = K[:, g.fdof][:, :, g.pdof]
        upall = self.up.reshape(B, NS, -1)
        upd, uff = upall[:, :, g.pdof], upall[:, :, g.fdof]
        rhs0 = -(Kfp @ upd.transpose(1, 2))                                     # (B, nfd, NS)
        Fd = self.fixf.repeat_interleave(2, -1)                                 # (B, NS, nfd)
        xs = []
        bad = torch.zeros(B, dtype=torch.bool, device=DEV)
        eye = torch.eye(Kff.shape[-1], device=DEV, dtype=DT)
        for e in range(NS):
            F = Fd[:, e]
            Fm = 1 - F
            Ke = Kff * Fm[:, :, None] * Fm[:, None, :] + torch.diag_embed(F)
            re = Fm * (rhs0[:, :, e] - (Kff @ (F * uff[:, e]).unsqueeze(-1)).squeeze(-1)) + F * uff[:, e]
            _, info = torch.linalg.cholesky_ex(Ke.detach())
            be = info > 0
            if bool(be.any()):
                Ke = torch.where(be[:, None, None], eye, Ke)
                re = torch.where(be[:, None], torch.zeros_like(re), re)
            bad = bad | be
            xs.append(L2.SPDSolveB.apply(Ke, re.unsqueeze(-1)).squeeze(-1))
        self.last_bad = bad
        u = upall.clone()
        u[:, :, g.fdof] = torch.stack(xs, 1)
        u = u.view(B * NS, g.N, 2)
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
        Fb = (2 * kn * dL + 0.5 * (TSi + TSq) + 0.5 * tv * (Si + Sq)) * val
        fp = g.free_t
        a = rep(self.opp_w)[:, fp] * Fb[:, fp]
        n = g.ndir
        sig = torch.stack([(a * n[:, 0] ** 2).sum(-1), (a * n[:, 1] ** 2).sum(-1), (a * n[:, 0] * n[:, 1]).sum(-1)], -1)
        Uen = 0.5 * (kn[:, fp] * dL[:, fp] ** 2).sum(-1)
        for c in (0, 1):
            msk = sh == c
            Uen = Uen + 0.5 * (tv[:, fp][:, :, msk] * dL[:, fp][:, :, msk]).sum(-1) * dL[:, fp][:, :, msk].sum(-1)
        w = Uen / float(g.s) ** 2
        return (u.view(B, NS, g.N, 2)[:, :, fp], sig.view(B, NS, -1, 3), w.view(B, NS, -1))

    def residuals(self, out, wts=(1.0, 1.0, 1.0)):
        u, sig, w = out
        rm = self.rm
        ru = (u - self.u_t) * rm[..., None] / self.nu[:, :, None, None] * math.sqrt(wts[0])
        rs = (sig - self.sig_t) * rm[..., None] / self.ns[:, :, None, None] * math.sqrt(wts[1])
        rw = (w - self.w_t) * rm / self.nw_[:, :, None] * math.sqrt(wts[2])
        r = torch.cat([ru.reshape(self.B, -1), rs.reshape(self.B, -1), rw.reshape(self.B, -1)], 1) / math.sqrt(self.NS)
        bad = getattr(self, "last_bad", None)
        if bad is not None and bool(bad.any()):
            r = torch.where(bad[:, None], torch.full_like(r, 10.0), r)
        return r


# ------------------------------------------------------------------------------------------------------------- fit
def joint_fit(batches, bases, nc, s, th0, iters, outer, pen=10.0, wts=(1.0, 1.0, 1.0)):
    nS = len(bases)
    bfw = bases[0]["b"]
    q, kr = WIN.neighbor_of(bfw, nc)
    kn_b = torch.as_tensor(np.stack([bb["kn"] for bb in bases]), device=DEV)
    tv_b = torch.as_tensor(np.stack([bb["tv"] for bb in bases]), device=DEV)
    K0_b = torch.as_tensor(np.stack([bb["K0"] for bb in bases]), device=DEV)
    P_, K_ = torch.as_tensor(bfw[:, 0], device=DEV), torch.as_tensor(bfw[:, 1], device=DEV)
    Q_, KR_ = torch.as_tensor(q, device=DEV), torch.as_tensor(kr, device=DEV)
    coarse = T.CoarseT(nc, nc, s, loads=[])
    nwin = sum(wb.B for wb in batches)

    def to_glob(th):
        kv = kn_b * torch.exp(th[..., 0])
        tvv = tv_b + K0_b * th[..., 1]
        Kn = torch.zeros(nS, nc * nc, 8, device=DEV, dtype=DT)
        Tv = torch.zeros_like(Kn)
        Kn[:, P_, K_] = kv
        Kn[:, Q_, KR_] = kv
        Tv[:, P_, K_] = tvv
        Tv[:, Q_, KR_] = tvv
        return Kn, Tv

    th = torch.as_tensor(th0, device=DEV, dtype=DT).clone().requires_grad_(True)

    def wloss(Kn, Tv):
        return sum(float((wb.residuals(wb.forward(*L2.gather_local(Kn.reshape(-1), Tv.reshape(-1), wb)), wts) ** 2)
                         .sum()) for wb in batches) / nwin

    def closure():
        opt.zero_grad()
        Kn, Tv = to_glob(th)
        p_, _ = coarse.stability_penalty(Kn, Tv)
        lp = pen * p_.sum() / nS
        lp.backward(retain_graph=True)
        tot = float(lp)
        for jb, wb in enumerate(batches):
            Kl, Tl = L2.gather_local(Kn.reshape(-1), Tv.reshape(-1), wb)
            lk = (wb.residuals(wb.forward(Kl, Tl), wts) ** 2).sum() / nwin
            lk.backward(retain_graph=(jb < len(batches) - 1))
            tot += float(lk)
        return torch.tensor(tot, dtype=DT)

    with torch.no_grad():
        l0 = wloss(*to_glob(th))
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, history_size=30, tolerance_grad=1e-12,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")
    hist = [float(opt.step(closure)) for _ in range(outer)]
    with torch.no_grad():
        Kn, Tv = to_glob(th)
        l1 = wloss(Kn, Tv)
        _, lam = coarse.stability_penalty(Kn, Tv)
    return th.detach().cpu().numpy(), Kn.cpu().numpy(), Tv.cpu().numpy(), l0, l1, hist, float(lam.min())


def theta_from(bb, Kn, Tv):
    b = bb["b"]
    kv, tv = Kn[b[:, 0], b[:, 1]], Tv[b[:, 0], b[:, 1]]
    return np.stack([np.log(kv / bb["kn"]), (tv - bb["tv"]) / bb["K0"]], -1)


def summarize(ms):
    out = {}
    for grp, lds in (("ring", EV.LOADS[:3]), ("plate", EV.LOADS[3:])):
        for k in ("sig", "G", "sig_in", "sig_bd", "w_if"):
            out[grp + "_" + k] = 100 * float(np.mean([abs(m[ld][k]) for m in ms for ld in lds if k in m[ld]]))
    return out


def fmt(sm):
    return "plate sig %.2f G %.2f in %.2f bd %.2f w_if %.1f | ring sig %.2f G %.2f" % (
        sm["plate_sig"], sm["plate_G"], sm["plate_sig_in"], sm["plate_sig_bd"], sm["plate_w_if"], sm["ring_sig"],
        sm["ring_G"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s", type=int)
    ap.add_argument("n1", type=int)
    ap.add_argument("--n0", type=int, default=0)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--nit", type=int, default=3)
    ap.add_argument("--start", default="l2j_m1")
    ap.add_argument("--loads", default="all")
    ap.add_argument("--norm", default="spec")
    ap.add_argument("--match", default="all", help="all free cells | center (layer >= m+1: the two end cells)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--outer", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s, m = a.s, a.m
    loads = EV.LOADS if a.loads == "all" else EV.LOADS[:3]
    bcsets = [BCSET[ld] for ld in loads]
    sp = EV.Split("test_id")
    d = sp.d
    n = sp.n
    nc = n // s
    ids = list(range(a.n0, a.n1))
    G = LF.bond_groups(nc)
    geo = {k: L2.WinGeom(k, m, s) for k in LF.KS}
    center = None
    if a.match == "center":
        center = (geo[0].layer[geo[0].free_p] >= m + 1).astype(float)
        c4 = (geo[4].layer[geo[4].free_p] >= m + 1).astype(float)
    tag = a.tag or "alg_m%d_%s_%s_%s" % (m, a.loads, a.norm, a.match)
    log = dict(args=vars(a), iters=[])
    # start parameters
    if a.start == "patch":
        lat = EV.coarse_model(n, s).lat
        cur = [cg.bond_params_from_cells(lat, sp.cells("patch", i, s)) for i in ids]
    else:
        z = np.load(os.path.join(common.RES, "params_%s_test_id_s%d.npz" % (a.start, s)))
        cur = [(z["Kn"][i].copy(), z["Tv"][i].copy()) for i in ids]
    ms = [EV.eval_sample(sp, i, s, *cur[j]) for j, i in enumerate(ids)]
    sm = summarize(ms)
    print("%s s=%d it 0 (start %s): %s" % (tag, s, a.start, fmt(sm)), flush=True)
    log["iters"].append(dict(it=0, summ=sm))
    t_fine_tot, t_fit_tot = 0.0, 0.0
    for it in range(1, a.nit + 1):
        newcur = list(cur)
        wl = []
        for j0 in range(0, len(ids), a.bs):
            jj_ = list(range(j0, min(len(ids), j0 + a.bs)))
            bases, dls = [], {0: [], 4: []}
            for jl, j in enumerate(jj_):
                i = ids[j]
                t0 = time.perf_counter()
                res = EV.coarse_model(n, s).run(*cur[j], loads=loads)
                U = {ld: res[ld]["u"] for ld in loads}
                P, Gpix, Pc, Gc = pixel_fields(n, s, U, loads)
                tg = {k: fine_group_bc(d["Cpix"][i], n, s, k, G[k], m, P, Gpix, bcsets) for k in LF.KS}
                t_fine_tot += time.perf_counter() - t0
                bases.append(LR.spec_base(d, i, s))
                for kc, ks in ((0, (0, 2)), (4, (4, 6))):
                    for k in ks:
                        d0 = window_data_bc(geo[k], G[k], nc, s, m, tg[k], Pc, Gc, bcsets, jl)
                        dls[kc].append(canon_bc(d0, k, geo[k], geo[kc]))
            batches = [WinBatchBC(geo[kc], dls[kc], nc, center=None if center is None else (center if kc == 0 else c4),
                                  norm=a.norm) for kc in (0, 4)]
            th0 = np.stack([theta_from(bases[jl], *cur[j]) for jl, j in enumerate(jj_)])
            t0 = time.time()
            th, Kn, Tv, l0, l1, hist, lam = joint_fit(batches, bases, nc, s, th0, a.iters, a.outer)
            t_fit_tot += time.time() - t0
            wl.append((l0, l1, hist, lam))
            for jl, j in enumerate(jj_):
                newcur[j] = (Kn[jl], Tv[jl])
            del batches
            torch.cuda.empty_cache()
        cur = newcur
        ms = [EV.eval_sample(sp, i, s, *cur[j]) for j, i in enumerate(ids)]
        sm = summarize(ms)
        print("%s s=%d it %d: %s | window loss %s" % (tag, s, it, fmt(sm), " ".join(
            "%.4f->%.4f (lam %.3f)" % (w[0], w[1], w[3]) for w in wl)), flush=True)
        log["iters"].append(dict(it=it, summ=sm, wl=[(w[0], w[1], w[2], w[3]) for w in wl],
                                 per_sample=[{ld: {k: float(v) for k, v in mm[ld].items()} for ld in EV.LOADS}
                                             for mm in ms]))
        np.savez(os.path.join(HERE, "params_%s_s%d.npz" % (tag, s)), Kn=np.stack([c[0] for c in cur]),
                 Tv=np.stack([c[1] for c in cur]), ids=np.array(ids))
    log["t_fine_per_specimen_per_it"] = t_fine_tot / max(1, a.nit) / len(ids)
    log["t_fit_per_specimen_per_it"] = t_fit_tot / max(1, a.nit) / len(ids)
    print("time per specimen per ALG iteration: fine+coarse %.2f s (1 thread), fit %.2f s (GPU)" % (
        log["t_fine_per_specimen_per_it"], log["t_fit_per_specimen_per_it"]), flush=True)
    json.dump(log, open(os.path.join(HERE, "log_%s_s%d.json" % (tag, s)), "w"), indent=0)


if __name__ == "__main__":
    main()
