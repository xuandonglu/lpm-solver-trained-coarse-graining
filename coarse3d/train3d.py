"""Train / evaluate the 3D super-particle network (torch environment, GPU).

  python train3d.py prep S SPLIT ...      # features, Voigt / patch bond bases, fine references -> data/prep_SPLIT_sS.pt
  python train3d.py train S [--epochs 30] # BondNet trained through the differentiable coarse LPM (adjoint PCG)
  python train3d.py eval S SPLIT ...      # errors of voigt / voigt+lin / patch / patch+lin / ML on held-out splits

Loss (I2): per load case, squared relative L2 errors of the displacement and stress fields over free cells (coarse vs
block-averaged fine) + squared relative error of the global measure (grip reaction / interior mean stress); mean over
the 9 load cases; + stability penalty (every particle's two shell-group bond Hessians must stay positive definite).
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import geom3d as G
import torch_lpm3d as T
import net3d as NT

torch.set_num_threads(4)
DEV = "cuda"
LOADS = G.LOADS
NL = len(LOADS)


# ---------------------------------------------------------------------------------------------------------------------
def cell_bases(Cv, n, s, C66=None):
    """Per-cell generalised SC parameters (Nc, 10) from Voigt-averaged (or given patch) cell stiffness."""
    if C66 is None:
        C66 = G.voigt_cells(Cv, n, s)
    return G.cell_params_from_C66(C66, s / 2.0)


def prep(split, s, force=False):
    fn = "data/prep_%s_s%d.pt" % (split, s)
    if os.path.exists(fn) and not force:
        return torch.load(fn, weights_only=False)
    m = dict(np.load("data/micro_%s.npz" % split, allow_pickle=True))
    f = dict(np.load("data/fine_%s.npz" % split, allow_pickle=True))
    n = int(m["n"])
    nc = n // s
    S = len(m["phase"])
    net = NT.BondNet(s)
    gw = net.windows(nc, "cpu")
    lat_c = G.LatticeGeom(nc, nc, nc, s)
    X = np.zeros((S, 3) + (n + 4 * s,) * 3, np.float32)
    bases = {"voigt": [], "patch": []}
    for i in range(S):
        ph, c, nu_i = m["phase"][i], float(m["contrast"][i]), float(m["nu_i"][i])
        X[i] = NT.voxel_features(ph, c, nu_i, n, s)
        E = np.where(ph == 1, c, 1.0)
        nu = np.where(ph == 1, nu_i, 0.2)
        Cv = G.iso_C(E, nu)
        for b in ("voigt", "patch"):
            if b == "patch" and "patch_%d" % s not in m:
                continue
            cp = cell_bases(Cv, n, s, None if b == "voigt" else m["patch_%d" % s][i])
            Kn, Tv = G.bond_params_from_cells(lat_c, cp)
            bases[b].append([(Kn[g["bonds"][:, 0], g["bonds"][:, 1]], Tv[g["bonds"][:, 0], g["bonds"][:, 1]]) for g in gw])
    out = dict(s=s, n=n, nc=nc, S=S, X=torch.as_tensor(X))
    for b, L in bases.items():
        if L:
            out[b] = [(torch.as_tensor(np.array([x[gi][0] for x in L])), torch.as_tensor(np.array([x[gi][1] for x in L])))
                      for gi in range(2)]
    out["ref"] = dict(u=torch.as_tensor(f["u_%d" % s]), sig=torch.as_tensor(f["sig_%d" % s]),
                      w=torch.as_tensor(f["w_%d" % s]), F=torch.as_tensor(f["F_%d" % s]))
    out["iface"] = torch.as_tensor(f["iface_%d" % s])
    out["meta"] = dict(phi=m["phi"], contrast=m["contrast"], kind=m["kind"])
    if "tpatch_%d" % s in m:
        out["tpatch"] = m["tpatch_%d" % s]
    torch.save(out, fn)
    return out


class Scatter:
    def __init__(self, nc, s, groups, device=DEV):
        lat = G.LatticeGeom(nc, nc, nc, s)
        self.N = lat.N
        self.idx = []
        for g in groups:
            b = g["bonds"]
            q = lat.nbr[b[:, 0], b[:, 1]]
            self.idx.append([torch.as_tensor(x, device=device) for x in (b[:, 0], b[:, 1], q, b[:, 1] ^ 1)])

    def __call__(self, vals):
        """vals: list over groups of (kn (B, nb), tv (B, nb)) -> Kn, Tv (B, N, 18)."""
        B = vals[0][0].shape[0]
        Kn = torch.zeros(B, self.N, 18, device=vals[0][0].device, dtype=torch.float64)
        Tv = torch.zeros_like(Kn)
        for (p, k, q, kr), (kn, tv) in zip(self.idx, vals):
            Kn = Kn.index_put((torch.arange(B, device=Kn.device)[:, None], p[None], k[None]), kn)
            Kn = Kn.index_put((torch.arange(B, device=Kn.device)[:, None], q[None], kr[None]), kn)
            Tv = Tv.index_put((torch.arange(B, device=Kn.device)[:, None], p[None], k[None]), tv)
            Tv = Tv.index_put((torch.arange(B, device=Kn.device)[:, None], q[None], kr[None]), tv)
        return Kn, Tv


def ml_vals(net, P, idx, nc):
    X = P["X"][idx].to(DEV)
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


def group_penalty(cm, Kn, Tv, frac=0.05):
    """relu(frac mean Kn - lambda_min)^2 of every (particle, shell) bond Hessian diag(Kn) + (t1^T + 1t^T)/2."""
    pen = 0.0
    lmin = []
    v = cm.op.valid
    for sl in (slice(0, 6), slice(6, 18)):
        vv = v[:, sl]
        kn = Kn[..., sl] * vv
        tv = Tv[..., sl] * vv
        H = torch.diag_embed(kn) + 0.5 * (tv[..., :, None] + tv[..., None, :])
        H = H * (vv[:, :, None] * vv[:, None, :])
        sc = Kn.abs().mean((1, 2))[:, None, None]
        H = H + torch.diag_embed((1 - vv) * 1e3).unsqueeze(0) * sc[..., None]
        lam = torch.linalg.eigvalsh(H)[..., 0] / sc[..., 0]
        pen = pen + (torch.relu(frac - lam) ** 2).sum(1)
        lmin.append(lam.min())
    return pen, float(min(lmin).detach())


# ---------------------------------------------------------------------------------------------------------------------
def errors(cm, out, ref, iface):
    """Per sample and load: relative errors (B, L) of global measure G, displacement u, stress sig (free cells),
    stress / energy density on interface cells, stress on free-surface cells (plates); apparent stiffness (B,)."""
    fc = cm.freecell[None, :, :, None].double()                                  # (1, L, N, 1)
    nrm = lambda x, m: (x * x * m).sum((-1, -2)).sqrt()
    e_u = nrm(out["u"] - ref["u"], fc) / nrm(ref["u"], fc)
    e_s = nrm(out["sig"] - ref["sig"], fc) / nrm(ref["sig"], fc)
    fi = fc * iface[:, None, :, None].double()
    e_si = nrm(out["sig"] - ref["sig"], fi) / nrm(ref["sig"], fi)
    wi = fi[..., 0]
    e_wi = ((out["w"] - ref["w"]) ** 2 * wi).sum(-1).sqrt() / ((ref["w"] ** 2) * wi).sum(-1).sqrt()
    fsu = (cm.surf[None, :, :, None] & cm.freecell[None, :, :, None]).double()
    e_ss = nrm(out["sig"] - ref["sig"], fsu) / nrm(ref["sig"], fsu).clamp_min(1e-300)
    G_ = []
    msig_c = (out["sig"] * fc).sum(2) / fc.sum(2)                               # (B, L, 6)
    msig_f = (ref["sig"] * fc).sum(2) / fc.sum(2)
    for l, ld in enumerate(LOADS):
        if ld in G.PLATES:
            G_.append((out["F"][:, l] - ref["F"][:, l]) / ref["F"][:, l])
        else:
            k = G.RING.index(ld)
            G_.append((msig_c[:, l, k] - msig_f[:, l, k]) / msig_f[:, l, k])
    e_G = torch.stack(G_, 1)
    Cc = msig_c[:, :6].transpose(1, 2)                                           # columns = load cases
    Cf = msig_f[:, :6].transpose(1, 2)
    e_C = (Cc - Cf).flatten(1).norm(dim=1) / Cf.flatten(1).norm(dim=1)
    return dict(G=e_G, u=e_u, sig=e_s, sig_if=e_si, w_if=e_wi, sig_surf=e_ss, C=e_C)


def loss_terms(cm, out, ref):
    fc = cm.freecell[None, :, :, None].double()
    nrm2 = lambda x: (x * x * fc).sum((-1, -2))
    Lu = (nrm2(out["u"] - ref["u"]) / nrm2(ref["u"])).mean(1)
    Ls = (nrm2(out["sig"] - ref["sig"]) / nrm2(ref["sig"])).mean(1)
    msig_c = (out["sig"] * fc).sum(2) / fc.sum(2)
    msig_f = (ref["sig"] * fc).sum(2) / fc.sum(2)
    g = []
    for l, ld in enumerate(LOADS):
        if ld in G.PLATES:
            g.append((out["F"][:, l] - ref["F"][:, l]) / ref["F"][:, l])
        else:
            k = G.RING.index(ld)
            g.append((msig_c[:, l, k] - msig_f[:, l, k]) / msig_f[:, l, k])
    Lg = (torch.stack(g, 1) ** 2).mean(1)
    return Lu, Ls, Lg


def summarize(elist):
    """list of errors() dicts -> mean |error| per key and group (ring = KUBC cases, plate, all)."""
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


def run_split(P, cm, sc, vals_fn, bs=8, tol=1e-10, post=None):
    es = []
    with torch.no_grad():
        for i0 in range(0, P["S"], bs):
            idx = torch.arange(i0, min(P["S"], i0 + bs))
            Kn, Tv = sc(vals_fn(idx))
            out = cm.run(Kn, Tv, tol=tol, differentiable=False)
            if post is not None:
                out = post(out, idx)
            es.append(errors(cm, out, batch_ref(P, idx), P["iface"][idx].to(DEV)))
    return summarize(es)


def global_of(cm, out):
    fc = cm.freecell[None, :, :, None].double()
    msig = (out["sig"] * fc).sum(2) / fc.sum(2)
    g = []
    for l, ld in enumerate(LOADS):
        g.append(out["F"][:, l] if ld in G.PLATES else msig[:, l, G.RING.index(ld)])
    return torch.stack(g, 1)


def lin_correction(Pcal, cm, sc, base, bs=8):
    """Linear correction of the global measure per load case, fitted on the calibration split."""
    gc, gf = [], []
    with torch.no_grad():
        for i0 in range(0, Pcal["S"], bs):
            idx = torch.arange(i0, min(Pcal["S"], i0 + bs))
            out = cm.run(*sc(base_vals(Pcal, idx, base)), differentiable=False)
            gc.append(global_of(cm, out))
            ref = batch_ref(Pcal, idx)
            gf.append(global_of(cm, dict(sig=ref["sig"], F=ref["F"])))
    gc = torch.cat(gc).cpu().numpy()
    gf = torch.cat(gf).cpu().numpy()
    # proportional fit G_fine = b G_coarse (scale-free, so the factor of the n = 24 calibration set transfers to other
    # specimen sizes; an intercept would carry the size of the calibration specimens)
    b = (gc * gf).sum(0) / (gc * gc).sum(0)
    ab = np.stack([b, np.zeros_like(b)], 1)
    return torch.as_tensor(ab, device=DEV)


def make_post(cm, ab):
    def post(out, idx):
        g = global_of(cm, out)
        gcor = ab[:, 0] * g + ab[:, 1]
        r = (gcor / g)                                                           # (B, L)
        o = dict(out)
        o["sig"] = out["sig"] * r[..., None, None]
        o["w"] = out["w"] * r[..., None]
        o["F"] = out["F"] * r
        return o
    return post


# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("splits", nargs="*")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pen", type=float, default=100.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--train-split", default="train")
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)

    if a.mode == "prep":
        for sp in a.splits:
            t0 = time.time()
            P = prep(sp, s, force=True)
            print("prep %s s=%d: %d samples, X %s, %.0f s" % (sp, s, P["S"], tuple(P["X"].shape), time.time() - t0),
                  flush=True)
        return

    if a.mode == "train":
        P = prep(a.train_split, s)
        V = prep("val", s)
        nc = P["nc"]
        cm = T.CoarseT(nc, s)
        net = NT.BondNet(s).to(DEV)
        gw = net.windows(nc, DEV)
        sc = Scatter(nc, s, gw)
        nparam = sum(p.numel() for p in net.parameters())
        S = P["S"] if a.ntrain == 0 else a.ntrain
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
        steps = a.epochs * math.ceil(S / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        bv, _ = run_split(V, cm, sc, lambda idx: base_vals(V, idx, "voigt"))
        bp, _ = run_split(V, cm, sc, lambda idx: base_vals(V, idx, "patch"))
        print("train s=%d: %d samples, %d params" % (s, S, nparam))
        print("val voigt: G %.4f sig %.4f | patch: G %.4f sig %.4f" % (bv["G_all"], bv["sig_all"], bp["G_all"],
                                                                         bp["sig_all"]), flush=True)
        log = dict(voigt=bv, patch=bp, hist=[], nparam=nparam)
        best = 1e9
        t0 = time.time()
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(S)
            tl = []
            lmins = []
            for i0 in range(0, S, a.bs):
                idx = perm[i0:i0 + a.bs]
                Kn, Tv = sc(ml_vals(net, P, idx, nc))
                pen, lmin = group_penalty(cm, Kn, Tv)
                out = cm.run(Kn, Tv, tol=1e-9)
                Lu, Ls, Lg = loss_terms(cm, out, batch_ref(P, idx))
                loss = (Lu + Ls + Lg + a.pen * pen).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tl.append(float(loss))
                lmins.append(lmin)
            if ep % 3 == 2 or ep == a.epochs - 1:
                net.eval()
                with torch.no_grad():
                    m, _ = run_split(V, cm, sc, lambda idx: ml_vals(net, V, idx, nc))
                log["hist"].append(dict(ep=ep + 1, loss=float(np.mean(tl)), **m))
                score = m["sig_all"] + m["G_all"]
                if score < best:
                    best = score
                    torch.save(net.state_dict(), "models/bondnet_s%d%s.pt" % (s, a.tag))
                print("ep %3d loss %.3e minlam %.3f | val G %.4f u %.4f sig %.4f (ring %.4f plate %.4f) surf %.4f "
                      "w_if %.4f C %.4f (%.0f s)" % (ep + 1, np.mean(tl), min(lmins), m["G_all"], m["u_all"],
                                                       m["sig_all"], m["sig_ring"], m["sig_plate"], m["sig_surf_plate"],
                                                       m["w_if_all"], m["C"], time.time() - t0), flush=True)
            else:
                print("ep %3d loss %.3e minlam %.3f (%.0f s)" % (ep + 1, np.mean(tl), min(lmins), time.time() - t0),
                      flush=True)
        log["train_time_s"] = time.time() - t0
        json.dump(log, open("results/train_s%d%s.json" % (s, a.tag), "w"), indent=1)
        return

    if a.mode == "oracle":
        # per-sample fitted bond parameters (approximate upper bound of the per-bond coarse SC model), from patch
        V = prep("val", s)
        nc = V["nc"]
        cm = T.CoarseT(nc, s)
        gw = NT.BondNet(s).windows(nc, DEV)
        sc = Scatter(nc, s, gw)
        nfit = 16
        idx = torch.arange(nfit)
        base = base_vals(V, idx, "patch")
        th = [torch.zeros(nfit, b[0].shape[1], 2, device=DEV, dtype=torch.float64, requires_grad=True) for b in base]
        opt = torch.optim.Adam(th, lr=a.lr)
        ref = batch_ref(V, idx)
        hist = []
        best = None
        for it in range(a.epochs):
            vals = [(kv * torch.exp(t[..., 0]), tv + kv * t[..., 1]) for (kv, tv), t in zip(base, th)]
            Kn, Tv = sc(vals)
            pen, lmin = group_penalty(cm, Kn, Tv)
            out = cm.run(Kn, Tv, tol=1e-9)
            Lu, Ls, Lg = loss_terms(cm, out, ref)
            loss = (Lu + Ls + Lg + a.pen * pen).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 50 == 0 or it == a.epochs - 1:
                with torch.no_grad():
                    m, _ = summarize([errors(cm, {k: v.detach() for k, v in out.items()}, ref, V["iface"][idx].to(DEV))])
                hist.append(dict(it=it, loss=float(loss), lmin=lmin, **m))
                if lmin >= 0 and (best is None or m["sig_all"] < best["sig_all"]):
                    best = dict(it=it, lmin=lmin, **m)
                print("oracle s=%d it %4d loss %.3e | G %.4f sig %.4f (ring %.4f plate %.4f) surf %.4f w_if %.4f "
                      "minlam %.3f" % (s, it, float(loss), m["G_all"], m["sig_all"], m["sig_ring"], m["sig_plate"],
                                       m["sig_surf_plate"], m["w_if_all"], lmin), flush=True)
        print("best locally stable fit: %s" % {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()
                                               if k in ("it", "lmin", "G_all", "sig_all", "sig_ring", "sig_plate",
                                                        "sig_surf_plate", "w_if_all", "C")}, flush=True)
        json.dump(dict(hist=hist, best=best), open("results/oracle_s%d.json" % s, "w"), indent=1)
        return

    if a.mode == "eval":
        net = NT.BondNet(s).to(DEV)
        net.load_state_dict(torch.load("models/bondnet_s%d%s.pt" % (s, a.tag)))
        net.eval()
        Pcal = prep("cal", s)
        res = {}
        for sp in a.splits:
            P = prep(sp, s)
            nc = P["nc"]
            cm = T.CoarseT(nc, s)
            gw = net.windows(nc, DEV)
            sc = Scatter(nc, s, gw)
            cmc = T.CoarseT(Pcal["nc"], s)
            scc = Scatter(Pcal["nc"], s, net.windows(Pcal["nc"], DEV))
            bs = 8 if P["n"] <= 24 else (2 if P["n"] <= 48 else 1)
            r, cats = {}, {}
            for base in ("voigt", "patch"):
                r[base], cats[base] = run_split(P, cm, sc, lambda idx: base_vals(P, idx, base), bs)
                ab = lin_correction(Pcal, cmc, scc, base)
                r[base + "+lin"], cats[base + "+lin"] = run_split(P, cm, sc, lambda idx: base_vals(P, idx, base), bs,
                                                                  post=make_post(cm, ab))
            with torch.no_grad():
                r["ml"], cats["ml"] = run_split(P, cm, sc, lambda idx: ml_vals(net, P, idx, nc), bs)
            res[sp] = r
            np.savez("results/errcat_%s_s%d%s.npz" % (sp, s, a.tag),
                     **{"%s__%s" % (mth, k): v for mth, d in cats.items() for k, v in d.items()})
            print("== %s s=%d (%d samples)" % (sp, s, P["S"]))
            for k in ("G_all", "C", "u_all", "sig_all", "sig_ring", "sig_plate", "sig_all_p90", "sig_if_all",
                      "w_if_all", "sig_surf_plate"):
                print("  %-16s " % k + "  ".join("%s %7.4f" % (mth, r[mth][k]) for mth in r), flush=True)
        json.dump(res, open("results/eval_s%d%s.json" % (s, a.tag), "w"), indent=1)
        return


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
