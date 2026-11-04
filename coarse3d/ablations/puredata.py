"""Ablation "pure-data model" (torch environment, GPU): a 3D CNN that maps the voxel microstructure DIRECTLY to the
block-averaged stress fields of the 9 load cases (no coarse LPM, no solver, no bond parameters) -- the usual
"image -> field" surrogate of the AI + lattice / PD literature -- trained on the same 1 200 I8 specimens and the same
fine references (s = 2 blocks) as the solver-in-the-loop BondNet, evaluated with the same stress-field error.

Network: fine-level trunk (3 x [3x3x3 conv, GELU], 24 channels, free kernels) -> 2x2x2 average pool to the coarse
cells -> coarse-level context (3x3x3 convs with dilation 1, 2, 4, 48 channels) + global mean feature (broadcast) ->
1x1 convs -> 9 x 6 stress channels per cell; output = net x EPS x kref (kref = mean shear modulus of the specimen, the
same normalisation as the BondNet voxel features).  Fully convolutional: applies to n = 48 / 72 unchanged.
Loss = mean over the 9 cases of the squared relative L2 stress error over free cells (the BondNet loss without the
displacement / global terms, which a stress-only model has no access to).
usage: python puredata.py train [--epochs 60]   |   python puredata.py eval   -> results/eval_pure.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import geom3d as G  # noqa: E402
import torch_lpm3d as T  # noqa: E402
import ablate as A  # noqa: E402

DEV = "cuda"
S = 2
SPLITS = ["test_id", "test_big", "test_huge", "test_elong", "test_c8", "test_soft"]


def kref_of(split):
    m = np.load(os.path.join(A.I8, "data", "micro_%s.npz" % split), allow_pickle=True)
    out = []
    for i in range(len(m["phase"])):
        ph, c, nu_i = m["phase"][i], float(m["contrast"][i]), float(m["nu_i"][i])
        E = np.where(ph == 1, c, 1.0)
        nu = np.where(ph == 1, nu_i, 0.2)
        out.append((E / (2 * (1 + nu))).mean())
    return torch.as_tensor(np.array(out))


class PureNet(nn.Module):
    def __init__(self, cin=3, ch=24, C=48, nout=54):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv3d(cin, ch, 3, padding=1), nn.GELU(), nn.Conv3d(ch, ch, 3, padding=1),
                                   nn.GELU(), nn.Conv3d(ch, ch, 3, padding=1), nn.GELU())
        self.ctx = nn.Sequential(nn.Conv3d(ch, C, 3, padding=1), nn.GELU(), nn.Conv3d(C, C, 3, padding=2, dilation=2),
                                 nn.GELU(), nn.Conv3d(C, C, 3, padding=4, dilation=4), nn.GELU())
        self.head = nn.Sequential(nn.Conv3d(2 * C, C, 1), nn.GELU(), nn.Conv3d(C, nout, 1))

    def forward(self, X):
        f = F.avg_pool3d(self.trunk(X), S, S)                    # (B, ch, nc + 4, ...) incl. 2 padding cells per side
        h = self.ctx(f)
        g = h.mean((2, 3, 4), keepdim=True).expand_as(h)
        o = self.head(torch.cat([h, g], 1))[:, :, 2:-2, 2:-2, 2:-2]          # specimen cells only
        B, _, nz, ny, nx = o.shape
        return o.reshape(B, 9, 6, nz * ny * nx).transpose(2, 3)            # (B, 9, Nc, 6), cell order z-major


def predict(net, P, idx, kref):
    o = net(P["X"][idx].to(DEV)).double()
    return o * (G.EPS * kref[idx].to(DEV).double())[:, None, None, None]


def field_err(cm, sig, ref):
    fc = cm.freecell[None, :, :, None].double()
    nrm = lambda x: (x * x * fc).sum((-1, -2)).sqrt()
    return nrm(sig - ref) / nrm(ref)                                      # (B, 9)


def stiff_err(cm, sig, ref):
    fc = cm.freecell[None, :, :, None].double()
    mc = (sig * fc).sum(2) / fc.sum(2)
    mf = (ref * fc).sum(2) / fc.sum(2)
    Cc, Cf = mc[:, :6].transpose(1, 2), mf[:, :6].transpose(1, 2)
    return (Cc - Cf).flatten(1).norm(dim=1) / Cf.flatten(1).norm(dim=1)


def run_eval(net, P, kref, bs):
    cm = T.CoarseT(P["nc"], S)
    es, ec = [], []
    with torch.no_grad():
        for i0 in range(0, P["S"], bs):
            idx = torch.arange(i0, min(P["S"], i0 + bs))
            sig = predict(net, P, idx, kref)
            ref = P["ref"]["sig"][idx].to(DEV).double()
            es.append(field_err(cm, sig, ref).cpu())
            ec.append(stiff_err(cm, sig, ref).cpu())
    e = torch.cat(es).numpy()
    c = torch.cat(ec).numpy()
    return dict(sig_all=float(e.mean()), sig_ring=float(e[:, :6].mean()), sig_plate=float(e[:, 6:].mean()),
                sig_all_p90=float(np.quantile(e.mean(1), 0.9)), C=float(c.mean()), per_sample_sig=e.mean(1).tolist())


def train(a):
    torch.manual_seed(0)
    P = A.load_prep("train")
    V = A.load_prep("val")
    kP, kV = kref_of("train"), kref_of("val")
    cm = T.CoarseT(P["nc"], S)
    net = PureNet().to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    steps = a.epochs * math.ceil(P["S"] / a.bs)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    print("pure-data net: %d params, %d steps" % (npar, steps), flush=True)
    log = dict(nparam=npar, epochs=a.epochs, hist=[])
    best, t0 = 1e9, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(P["S"])
        tl = []
        for i0 in range(0, P["S"], a.bs):
            idx = perm[i0:i0 + a.bs]
            sig = predict(net, P, idx, kP)
            ref = P["ref"]["sig"][idx].to(DEV).double()
            loss = (field_err(cm, sig, ref) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(float(loss.detach()))
        net.eval()
        m = run_eval(net, V, kV, 8)
        log["hist"].append(dict(ep=ep + 1, loss=float(np.mean(tl)), val_sig=m["sig_all"], t=time.time() - t0))
        if m["sig_all"] < best:
            best = m["sig_all"]
            torch.save(net.state_dict(), os.path.join(HERE, "models", "pure.pt"))
        print("ep %3d loss %.3e | val sig %.4f (ring %.4f plate %.4f) C %.4f (%.0f s)" % (
            ep + 1, np.mean(tl), m["sig_all"], m["sig_ring"], m["sig_plate"], m["C"], time.time() - t0), flush=True)
    log["train_time_s"] = time.time() - t0
    json.dump(log, open(os.path.join(HERE, "results", "train_pure.json"), "w"), indent=1)


def evaluate(a):
    net = PureNet().to(DEV)
    net.load_state_dict(torch.load(os.path.join(HERE, "models", "pure.pt"), map_location=DEV))
    net.eval()
    res = {}
    for sp in SPLITS:
        P = A.load_prep(sp)
        bs = 8 if P["n"] <= 24 else (2 if P["n"] <= 48 else 1)
        r = run_eval(net, P, kref_of(sp), bs)
        res[sp] = r
        print("== pure %s (%d): sig %.4f ring %.4f plate %.4f p90 %.4f C %.4f" % (
            sp, P["S"], r["sig_all"], r["sig_ring"], r["sig_plate"], r["sig_all_p90"], r["C"]), flush=True)
    json.dump(dict(splits=res), open(os.path.join(HERE, "results", "eval_pure.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.mode == "train":
        train(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
