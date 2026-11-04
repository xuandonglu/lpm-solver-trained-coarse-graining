"""Cement-paste engine (torch environment, GPU): coarse models with voids, I8 network zero-shot / retrained, baselines, errors, training,
per-sample oracle fit.

  python -I train_v1.py prep  S SET [SET ...] OPT          tensors -> data/prep_SET_OPT_sS.pt
  python -I train_v1.py eval  S OPT TAG CAL SET [SET ...]   CAL = calibration set of the linear correction (same scan)
                                                             TAG = i8 (I8 weights) or the tag of a V1 model
  python -I train_v1.py train S OPT TAG --train SET[,SET] --val SET [--init i8|none] [--epochs E] [--lr LR]
  python -I train_v1.py oracle S OPT SET [--nfit 12] [--its 400]

Everything is the I8 definition (train3d.py), generalised to voids: the coarse lattice is the full nc^3 grid with a
per-specimen cell mask (cell exists if its block holds a solid voxel; coarse cleaning in materials.py), a coarse bond
exists if both cells exist; network input = I8 voxel features [log(mu/mu_ref), (lambda - mu)/(18 mu), void] with the
void flag also set on internal void voxels (mu_ref = mean mu of the solid voxels); bases (Voigt, patch) and the NN
output map Kn = Kn_V exp(o1), Tv = Tv_V + Kn_V o2 are unchanged; every coarse model (all methods) gets the same minimum
stiffness Kn >= 1e-3 x (mean Kn of the specimen's bonds).
Errors (coarse vs block-averaged fine LPM): stress field on BC-free existing cells at the native resolution R = s and
on R = 2s blocks (mean over the 2^3 cells, BC-free blocks only), mean over the 9 load cases; KUBC apparent stiffness
(6x6, mean stress over the BC-free cells incl. void cells as zero); global measure; stress on interface cells and on
pore-adjacent cells; energy density on interface cells; free-surface cells of the plates; displacement field.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import geom3d as G  # noqa: E402
import net3d as NT  # noqa: E402
import torch_v1 as TV  # noqa: E402

torch.set_num_threads(4)
DEV = "cuda"
LOADS = G.LOADS
KMIN = 1e-3


# ---------------------------------------------------------------------------------------------------------------------
def voxel_features_v(E, nu, solid, n, s):
    """I8 voxel features (net3d.voxel_features) with internal voids: void voxels look like the padding."""
    solid = np.asarray(solid, bool)
    mu = np.where(solid, E / (2 * (1 + nu)), 1.0)
    lam = np.where(solid, E * nu / ((1 + nu) * (1 - 2 * np.where(solid, nu, 0.0))), 1.0)
    kref = mu[solid].mean()
    P = 2 * s
    X = np.zeros((3, n + 2 * P, n + 2 * P, n + 2 * P), np.float32)
    X[2] = 1.0
    X[0, P:P + n, P:P + n, P:P + n] = np.where(solid, np.log(mu / kref), 0.0).reshape(n, n, n)
    X[1, P:P + n, P:P + n, P:P + n] = np.where(solid, (lam - mu) / (18 * mu), 0.0).reshape(n, n, n)
    X[2, P:P + n, P:P + n, P:P + n] = (~solid).reshape(n, n, n)
    return X


def bases_from_cells(lat_c, cp, gw):
    Kn, Tv = G.bond_params_from_cells(lat_c, cp)
    return [(Kn[g["bonds"][:, 0], g["bonds"][:, 1]], Tv[g["bonds"][:, 0], g["bonds"][:, 1]]) for g in gw]


def prep(st, opt, s, force=False):
    fn = common.data("prep_%s_%s_s%d.pt" % (st, opt, s))
    if os.path.exists(fn) and not force:
        return torch.load(fn, weights_only=False)
    mf = np.load(common.data("mat_%s_%s.npz" % (st, opt)))
    ffn = common.data("fine_%s_%s.npz" % (st, opt))
    pfn = common.data("patch_%s_%s.npz" % (st, opt))
    ff = np.load(ffn) if os.path.exists(ffn) else None
    pf = np.load(pfn) if os.path.exists(pfn) else None
    n = int(mf["n"])
    nc = n // s
    S = len(mf["solid"])
    gw = NT.BondNet(s).windows(nc, "cpu")
    lat_c = G.LatticeGeom(nc, nc, nc, s)
    X = np.zeros((S, 3) + (n + 4 * s,) * 3, np.float32)
    bases = {"voigt": [], "patch": []}
    for i in range(S):
        E, nu, sol = mf["E"][i], mf["nu"][i], mf["solid"][i]
        X[i] = voxel_features_v(E, nu, sol, n, s)
        Cv = G.iso_C(E, nu)
        bases["voigt"].append(bases_from_cells(lat_c, G.cell_params_from_C66(G.voigt_cells(Cv, n, s), s / 2.0), gw))
        if pf is not None and "C_%d" % s in pf:
            bases["patch"].append(bases_from_cells(lat_c, G.cell_params_from_C66(pf["C_%d" % s][i], s / 2.0), gw))
    out = dict(s=s, n=n, nc=nc, S=S, X=torch.as_tensor(X), cell=torch.as_tensor(mf["cell_%d" % s]),
               iface=torch.as_tensor(mf["iface_%d" % s]), padj=torch.as_tensor(mf["padj_%d" % s]))
    for b, L in bases.items():
        if L:
            out[b] = [(torch.as_tensor(np.array([x[gi][0] for x in L])), torch.as_tensor(np.array([x[gi][1] for x in L])))
                      for gi in range(2)]
    if ff is not None and "sig_%d" % s in ff:
        out["ref"] = dict(u=torch.as_tensor(ff["u_%d" % s]), sig=torch.as_tensor(ff["sig_%d" % s]),
                          w=torch.as_tensor(ff["w_%d" % s]), F=torch.as_tensor(ff["F_%d" % s]))
    out["meta"] = dict(void_frac=mf["void_frac"], ndrop=mf["ndrop"], nrm=mf["nrm_%d" % s])
    if pf is not None and "tpatch_%d" % s in pf:
        out["tpatch"] = pf["tpatch_%d" % s]
    torch.save(out, fn)
    return out


# ---------------------------------------------------------------------------------------------------------------------
class Scatter:
    """train3d.Scatter: forward-bond values of the canonical bond lists -> per-slot (B, N, 18), both ends."""

    def __init__(self, nc, s, groups, device=DEV):
        lat = G.LatticeGeom(nc, nc, nc, s)
        self.N = lat.N
        self.idx = []
        for g in groups:
            b = g["bonds"]
            q = lat.nbr[b[:, 0], b[:, 1]]
            self.idx.append([torch.as_tensor(x, device=device) for x in (b[:, 0], b[:, 1], q, b[:, 1] ^ 1)])

    def __call__(self, vals):
        B = vals[0][0].shape[0]
        Kn = torch.zeros(B, self.N, 18, device=vals[0][0].device, dtype=torch.float64)
        Tv = torch.zeros_like(Kn)
        ar = torch.arange(B, device=Kn.device)[:, None]
        for (p, k, q, kr), (kn, tv) in zip(self.idx, vals):
            Kn = Kn.index_put((ar, p[None], k[None]), kn)
            Kn = Kn.index_put((ar, q[None], kr[None]), kn)
            Tv = Tv.index_put((ar, p[None], k[None]), tv)
            Tv = Tv.index_put((ar, q[None], kr[None]), tv)
        return Kn, Tv


def ml_vals(net, P, idx, nc, X=None):
    X = P["X"][idx].to(DEV) if X is None else X
    o = net(X, nc)
    vals = []
    for gi, oo in enumerate(o):
        kv, tv = [x[idx].to(DEV) for x in P["voigt"][gi]]
        oo = oo.double()
        vals.append((kv * torch.exp(oo[..., 0]), tv + kv * oo[..., 1]))
    return vals


def base_vals(P, idx, base):
    return [tuple(x[idx].to(DEV) for x in P[base][gi]) for gi in range(2)]


def batch_ref(P, idx):
    return {k: v[idx].to(DEV).double() for k, v in P["ref"].items()}


def finalize(cm, Kn, Tv, cell):
    """mask to existing bonds of the specimens, minimum stiffness."""
    cm.op.set_mask(cell)
    v = cm.op.valid[:, 0]
    Kn = Kn * v
    Tv = Tv * v
    km = KMIN * Kn.sum((1, 2)) / v.sum((1, 2)).clamp_min(1.0)
    Kn = torch.where(v > 0, torch.maximum(Kn, km[:, None, None]), Kn)
    return Kn, Tv


def group_penalty(cm, Kn, Tv, frac=0.05):
    """train3d.group_penalty with per-specimen bond validity (cm.op.set_mask must have been called)."""
    pen = 0.0
    lmin = []
    v = cm.op.valid[:, 0]
    for sl in (slice(0, 6), slice(6, 18)):
        vv = v[..., sl]
        kn = Kn[..., sl] * vv
        tv = Tv[..., sl] * vv
        H = torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :])
        H = H * (vv[..., :, None] * vv[..., None, :])
        sc = Kn.abs().mean((1, 2))[:, None, None]
        H = H + torch.diag_embed((1 - vv) * 1e3) * sc[..., None]
        lam = torch.linalg.eigvalsh(H)[..., 0] / sc[..., 0]
        pen = pen + (torch.relu(frac - lam) ** 2).sum(1)
        lmin.append(lam.min())
    return pen, float(min(lmin).detach())


@torch.no_grad()
def unstable_fraction(cm, Kn, Tv):
    """fraction of existing (particle, shell) groups whose bond Hessian is not positive definite."""
    v = cm.op.valid[:, 0]
    bad, tot = 0, 0
    for sl in (slice(0, 6), slice(6, 18)):
        vv = v[..., sl]
        H = torch.diag_embed(Kn[..., sl] * vv) + 0.5 * ((Tv[..., sl] * vv)[..., :, None] + (Tv[..., sl] * vv)[..., None, :])
        H = H * (vv[..., :, None] * vv[..., None, :]) + torch.diag_embed(1 - vv) * Kn.abs().max()
        lam = torch.linalg.eigvalsh(H)[..., 0]
        ex = vv.sum(-1) > 0
        bad += int(((lam <= 0) & ex).sum())
        tot += int(ex.sum())
    return bad / max(tot, 1)


# ---------------------------------------------------------------------------------------------------------------------
class Agg:
    """R = 2s aggregation of coarse cell fields (2^3 cells), BC-free blocks per load case."""

    def __init__(self, cm):
        nc = cm.nc
        assert nc % 2 == 0
        ijk = cm.lat.ijk
        m = nc // 2
        rid = ((ijk[:, 2] // 2) * m + ijk[:, 1] // 2) * m + ijk[:, 0] // 2
        self.rid = torch.as_tensor(rid, device=DEV)
        self.NR = m ** 3
        fr = cm.freecell.double()                                            # (L, Nc)
        cnt = torch.zeros(fr.shape[0], self.NR, device=DEV, dtype=torch.float64).index_add_(1, self.rid, fr)
        self.free = cnt == 8                                                 # (L, NR)

    def __call__(self, x):
        """(..., Nc, k) -> (..., NR, k) mean over the 8 cells."""
        out = torch.zeros(x.shape[:-2] + (self.NR, x.shape[-1]), device=x.device, dtype=x.dtype)
        out.index_add_(x.dim() - 2, self.rid, x)
        return out / 8.0


def errors(cm, agg, out, ref, cell, iface, padj):
    """per sample (B, L) relative errors and (B,) apparent stiffness error."""
    bcf = cm.freecell[None]                                                  # (1, L, N) BC-free
    ex = cell[:, None, :]                                                    # (B, 1, N)
    fc = (bcf & ex)[..., None].double()
    nrm = lambda x, m: (x * x * m).sum((-1, -2)).sqrt()
    rel = lambda a, b, m: nrm(a - b, m) / nrm(b, m).clamp_min(1e-300)
    e = {}
    e["u"] = rel(out["u"], ref["u"], fc)
    e["sig"] = rel(out["sig"], ref["sig"], fc)
    fR = agg.free[None, :, :, None].double()
    e["sig_R2"] = rel(agg(out["sig"]), agg(ref["sig"]), fR)
    fi = fc * iface[:, None, :, None].double()
    e["sig_if"] = rel(out["sig"], ref["sig"], fi)
    wi = fi[..., 0]
    e["w_if"] = ((out["w"] - ref["w"]) ** 2 * wi).sum(-1).sqrt() / ((ref["w"] ** 2) * wi).sum(-1).sqrt().clamp_min(1e-300)
    fp = fc * padj[:, None, :, None].double()
    e["sig_pa"] = rel(out["sig"], ref["sig"], fp)
    fsu = fc * cm.surf[None, :, :, None].double()
    e["sig_surf"] = rel(out["sig"], ref["sig"], fsu)
    fb = bcf[..., None].double()
    msig_c = (out["sig"] * fb).sum(2) / fb.sum(2)
    msig_f = (ref["sig"] * fb).sum(2) / fb.sum(2)
    g = []
    for l, ld in enumerate(LOADS):
        if ld in G.PLATES:
            g.append((out["F"][:, l] - ref["F"][:, l]) / ref["F"][:, l])
        else:
            k = G.RING.index(ld)
            g.append((msig_c[:, l, k] - msig_f[:, l, k]) / msig_f[:, l, k])
    e["G"] = torch.stack(g, 1)
    Cc = msig_c[:, :6].transpose(1, 2)
    Cf = msig_f[:, :6].transpose(1, 2)
    e["C"] = (Cc - Cf).flatten(1).norm(dim=1) / Cf.flatten(1).norm(dim=1)
    return e


def global_of(cm, out):
    fb = cm.freecell[None, :, :, None].double()
    msig = (out["sig"] * fb).sum(2) / fb.sum(2)
    g = []
    for l, ld in enumerate(LOADS):
        g.append(out["F"][:, l] if ld in G.PLATES else msig[:, l, G.RING.index(ld)])
    return torch.stack(g, 1)


def loss_terms(cm, out, ref, cell):
    fc = (cm.freecell[None] & cell[:, None, :])[..., None].double()
    nrm2 = lambda x: (x * x * fc).sum((-1, -2))
    Lu = (nrm2(out["u"] - ref["u"]) / nrm2(ref["u"])).mean(1)
    Ls = (nrm2(out["sig"] - ref["sig"]) / nrm2(ref["sig"])).mean(1)
    Lg = ((global_of(cm, out) - global_of(cm, ref)) / global_of(cm, ref)) ** 2
    return Lu, Ls, Lg.mean(1)


def summarize(elist):
    cat = {k: torch.cat([e[k] for e in elist], 0).abs().cpu().numpy() for k in elist[0]}
    out = {}
    for k, v in cat.items():
        if v.ndim == 1:
            out[k] = float(np.nanmean(v))
            continue
        out[k + "_ring"] = float(np.nanmean(v[:, :6]))
        out[k + "_plate"] = float(np.nanmean(v[:, 6:]))
        out[k + "_all"] = float(np.nanmean(v))
    out["sig_all_p90"] = float(np.quantile(cat["sig"].mean(1), 0.9))
    return out, cat


def run_split(P, cm, agg, sc, vals_fn, bs=8, tol=1e-10, post=None, stab=None):
    es = []
    with torch.no_grad():
        for i0 in range(0, P["S"], bs):
            idx = torch.arange(i0, min(P["S"], i0 + bs))
            cell = P["cell"][idx].to(DEV)
            Kn, Tv = finalize(cm, *sc(vals_fn(idx)), cell)
            if stab is not None:
                stab.append(unstable_fraction(cm, Kn, Tv))
            out = cm.run(Kn, Tv, cell, tol=tol, differentiable=False)
            if post is not None:
                out = post(out, idx)
            es.append(errors(cm, agg, out, batch_ref(P, idx), cell, P["iface"][idx].to(DEV), P["padj"][idx].to(DEV)))
    return summarize(es)


def lin_correction(Pcal, cm, sc, base, bs=8):
    """Proportional correction G_fine = b G_coarse per load case, fitted on the calibration cubes."""
    gc, gf = [], []
    with torch.no_grad():
        for i0 in range(0, Pcal["S"], bs):
            idx = torch.arange(i0, min(Pcal["S"], i0 + bs))
            cell = Pcal["cell"][idx].to(DEV)
            Kn, Tv = finalize(cm, *sc(base_vals(Pcal, idx, base)), cell)
            out = cm.run(Kn, Tv, cell, differentiable=False)
            gc.append(global_of(cm, out))
            ref = batch_ref(Pcal, idx)
            gf.append(global_of(cm, dict(sig=ref["sig"], F=ref["F"])))
    gc = torch.cat(gc).cpu().numpy()
    gf = torch.cat(gf).cpu().numpy()
    return torch.as_tensor((gc * gf).sum(0) / (gc * gc).sum(0), device=DEV)


def make_post(cm, b):
    def post(out, idx):
        r = b[None, :].expand(out["F"].shape[0], -1)
        o = dict(out)
        o["sig"] = out["sig"] * r[..., None, None]
        o["w"] = out["w"] * r[..., None]
        o["F"] = out["F"] * r
        return o
    return post


def load_net(s, tag):
    net = NT.BondNet(s).to(DEV)
    fn = os.path.join(common.I8, "models", "bondnet_s%d.pt" % s) if tag == "i8" else \
        os.path.join(common.HERE, "models", "bondnet_s%d_%s.pt" % (s, tag))
    net.load_state_dict(torch.load(fn, map_location=DEV))
    net.eval()
    return net


def bs_of(n):
    return 8 if n <= 24 else (2 if n <= 48 else 1)


# ---------------------------------------------------------------------------------------------------------------------
def cmd_eval(a):
    s = a.s
    nets = {t: load_net(s, t) for t in a.tags.split(",")}
    Pcal = prep(a.cal, a.opt, s)
    res = {}
    rfn = common.HERE + "/results/eval_%s_s%d_%s.json" % (a.opt, s, a.out)
    if os.path.exists(rfn):
        res = json.load(open(rfn))
    for st in a.sets:
        P = prep(st, a.opt, s)
        nc = P["nc"]
        cm = TV.CoarseTV(nc, s)
        agg = Agg(cm)
        gw = NT.BondNet(s).windows(nc, DEV)
        sc = Scatter(nc, s, gw)
        cmc = TV.CoarseTV(Pcal["nc"], s)
        scc = Scatter(Pcal["nc"], s, NT.BondNet(s).windows(Pcal["nc"], DEV))
        bs = bs_of(P["n"])
        r, cats, stabs = {}, {}, {}
        t0 = time.time()
        for base in ("voigt", "patch"):
            if base not in P:
                continue
            stabs[base] = []
            r[base], cats[base] = run_split(P, cm, agg, sc, lambda idx: base_vals(P, idx, base), bs, stab=stabs[base])
            b = lin_correction(Pcal, cmc, scc, base)
            r[base + "+lin"], cats[base + "+lin"] = run_split(P, cm, agg, sc, lambda idx: base_vals(P, idx, base), bs,
                                                              post=make_post(cm, b))
        for t, net in nets.items():
            stabs["nn_" + t] = []
            with torch.no_grad():
                r["nn_" + t], cats["nn_" + t] = run_split(P, cm, agg, sc, lambda idx: ml_vals(net, P, idx, nc), bs,
                                                          stab=stabs["nn_" + t])
        for k, v in stabs.items():
            r[k]["unstable_groups"] = float(np.mean(v))
        res[st] = r
        np.savez(common.HERE + "/results/errcat_%s_%s_s%d_%s.npz" % (st, a.opt, s, a.out),
                 **{"%s__%s" % (mth, k): v for mth, d in cats.items() for k, v in d.items()})
        print("== %s opt %s s=%d (%d cubes, %.0f s)" % (st, a.opt, s, P["S"], time.time() - t0))
        for k in ("sig_all", "sig_R2_all", "C", "G_all", "sig_if_all", "sig_pa_all", "w_if_all", "sig_surf_plate",
                  "u_all", "sig_all_p90"):
            print("  %-15s " % k + "  ".join("%s %6.2f" % (m, 100 * r[m][k]) for m in r), flush=True)
        print("  unstable groups " + "  ".join("%s %.1e" % (m, r[m]["unstable_groups"]) for m in r
                                               if "unstable_groups" in r[m]), flush=True)
        json.dump(res, open(rfn, "w"), indent=1)


class Joined:
    """several prepared sets as one training set (same n, s)."""

    def __init__(self, Ps):
        self.P = dict(Ps[0])
        for k in ("X", "cell", "iface", "padj"):
            self.P[k] = torch.cat([p[k] for p in Ps], 0)
        self.P["voigt"] = [tuple(torch.cat([p["voigt"][gi][j] for p in Ps], 0) for j in range(2)) for gi in range(2)]
        self.P["ref"] = {k: torch.cat([p["ref"][k] for p in Ps], 0) for k in Ps[0]["ref"]}
        self.P["S"] = sum(p["S"] for p in Ps)


def cmd_train(a):
    s = a.s
    torch.manual_seed(a.seed)
    P = Joined([prep(t, a.opt, s) for t in a.train.split(",")]).P
    V = prep(a.val, a.opt, s)
    nc = P["nc"]
    cm = TV.CoarseTV(nc, s)
    agg = Agg(cm)
    net = NT.BondNet(s).to(DEV)
    if a.init == "i8":
        net.load_state_dict(torch.load(os.path.join(common.I8, "models", "bondnet_s%d.pt" % s), map_location=DEV))
    gw = net.windows(nc, DEV)
    sc = Scatter(nc, s, gw)
    S = P["S"] if a.ntrain == 0 else min(a.ntrain, P["S"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * math.ceil(S / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    net.eval()
    m0, _ = run_split(V, cm, agg, sc, lambda idx: ml_vals(net, V, idx, nc))
    print("train s=%d opt %s tag %s: %d samples (%s), init %s; val at start: sig %.4f G %.4f" % (
        s, a.opt, a.tag, S, a.train, a.init, m0["sig_all"], m0["G_all"]), flush=True)
    log = dict(args=vars(a), start=m0, hist=[])
    best = 1e9
    t0 = time.time()
    fn = os.path.join(common.HERE, "models", "bondnet_s%d_%s.pt" % (s, a.tag))
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(S)
        tl, lmins = [], []
        for i0 in range(0, S, a.bs):
            idx = perm[i0:i0 + a.bs]
            cell = P["cell"][idx].to(DEV)
            Kn, Tv = finalize(cm, *sc(ml_vals(net, P, idx, nc)), cell)
            pen, lmin = group_penalty(cm, Kn, Tv)
            out = cm.run(Kn, Tv, cell, tol=1e-9)
            Lu, Ls, Lg = loss_terms(cm, out, batch_ref(P, idx), cell)
            loss = (Lu + Ls + Lg + a.pen * pen).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(float(loss))
            lmins.append(lmin)
        if ep % 2 == 1 or ep == a.epochs - 1:
            net.eval()
            m, _ = run_split(V, cm, agg, sc, lambda idx: ml_vals(net, V, idx, nc))
            log["hist"].append(dict(ep=ep + 1, loss=float(np.mean(tl)), **m))
            score = m["sig_all"] + m["G_all"]
            if score < best:
                best = score
                torch.save(net.state_dict(), fn)
            print("ep %3d loss %.3e minlam %.3f | val sig %.4f R2 %.4f G %.4f C %.4f pa %.4f (%.0f s)" % (
                ep + 1, np.mean(tl), min(lmins), m["sig_all"], m["sig_R2_all"], m["G_all"], m["C"], m["sig_pa_all"],
                time.time() - t0), flush=True)
        else:
            print("ep %3d loss %.3e minlam %.3f (%.0f s)" % (ep + 1, np.mean(tl), min(lmins), time.time() - t0),
                  flush=True)
    log["train_time_s"] = time.time() - t0
    json.dump(log, open(os.path.join(common.HERE, "results", "train_s%d_%s.json" % (s, a.tag)), "w"), indent=1)


def cmd_oracle(a):
    """per-sample fit of every coarse bond's (Kn, Tv) (approximate upper bound of the coarse SC model), started from
    patch (floored), parametrised on the Voigt base: Kn = Kn_V exp(t0), Tv = Tv_V + Kn_V t1; the best locally stable
    state (all bond Hessians positive definite) is kept (I8 oracle)."""
    s = a.s
    P = prep(a.set, a.opt, s)
    nc = P["nc"]
    cm = TV.CoarseTV(nc, s)
    agg = Agg(cm)
    sc = Scatter(nc, s, NT.BondNet(s).windows(nc, DEV))
    idx = torch.arange(min(a.nfit, P["S"]))
    cell = P["cell"][idx].to(DEV)
    vb = base_vals(P, idx, "voigt")
    th = []
    if a.init_tag:
        # start from a trained network's prediction (same parametrisation: o1, o2 on the Voigt base)
        net = load_net(s, a.init_tag)
        with torch.no_grad():
            o = net(P["X"][idx].to(DEV), nc)
        th = [oo.double().clone().requires_grad_(True) for oo in o]
    else:
        pb = base_vals(P, idx, "patch")
        for (kv, tv), (kp, tp) in zip(vb, pb):
            kvs = kv.clamp_min(1e-12)
            t0_ = torch.log(torch.maximum(kp, 1e-3 * kvs) / kvs)
            t1_ = (tp - tv) / kvs
            th.append(torch.stack([t0_, t1_], -1).clone().requires_grad_(True))
    opt = torch.optim.Adam(th, lr=a.lr)
    ref = batch_ref(P, idx)
    best = None
    hist = []
    t0 = time.time()
    for it in range(a.its):
        vals = [(kv * torch.exp(t[..., 0]), tv + kv * t[..., 1]) for (kv, tv), t in zip(vb, th)]
        Kn, Tv = finalize(cm, *sc(vals), cell)
        pen, lmin = group_penalty(cm, Kn, Tv)
        out = cm.run(Kn, Tv, cell, tol=1e-9)
        Lu, Ls, Lg = loss_terms(cm, out, ref, cell)
        loss = (Lu + Ls + Lg + a.pen * pen).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == a.its - 1:
            with torch.no_grad():
                e = errors(cm, agg, {k: v.detach() for k, v in out.items()}, ref, cell, P["iface"][idx].to(DEV),
                           P["padj"][idx].to(DEV))
                m, _ = summarize([e])
            hist.append(dict(it=it, loss=float(loss), lmin=lmin, **m))
            if lmin >= 0 and (best is None or m["sig_all"] < best["sig_all"]):
                best = dict(it=it, lmin=lmin, **m)
            print("oracle %s %s s=%d it %4d loss %.3e | sig %.4f R2 %.4f G %.4f C %.4f pa %.4f minlam %.3f (%.0f s)" % (
                a.set, a.opt, s, it, float(loss), m["sig_all"], m["sig_R2_all"], m["G_all"], m["C"], m["sig_pa_all"],
                lmin, time.time() - t0), flush=True)
    print("best locally stable fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()
                                       if k in ("it", "lmin", "sig_all", "sig_R2_all", "G_all", "C", "sig_pa_all",
                                                "sig_if_all", "w_if_all")}, flush=True)
    json.dump(dict(hist=hist, best=best, nfit=len(idx), time_s=time.time() - t0, init=a.init_tag or "patch", lr=a.lr),
              open(os.path.join(common.HERE, "results", "oracle_%s_%s_s%d%s.json" % (
                  a.set, a.opt, s, "_from_" + a.init_tag if a.init_tag else "")), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--train", default="")
    ap.add_argument("--val", default="")
    ap.add_argument("--init", default="i8")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pen", type=float, default=100.0)
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfit", type=int, default=12)
    ap.add_argument("--its", type=int, default=400)
    ap.add_argument("--out", default="")
    ap.add_argument("--init-tag", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(common.HERE, "results"), exist_ok=True)
    if a.mode == "prep":
        opt = a.rest[-1]
        for st in a.rest[:-1]:
            t0 = time.time()
            P = prep(st, opt, a.s, force=True)
            print("prep %s %s s=%d: %d cubes, %.0f s" % (st, opt, a.s, P["S"], time.time() - t0), flush=True)
    elif a.mode == "eval":
        a.opt, a.tags, a.cal, a.sets = a.rest[0], a.rest[1], a.rest[2], a.rest[3:]
        a.out = a.out or a.tags.replace(",", "+")
        cmd_eval(a)
    elif a.mode == "train":
        a.opt, a.tag = a.rest[0], a.rest[1]
        cmd_train(a)
    elif a.mode == "oracle":
        a.opt, a.set = a.rest[0], a.rest[1]
        a.lr = 0.005 if a.lr == 5e-4 else a.lr
        cmd_oracle(a)


if __name__ == "__main__":
    main()
