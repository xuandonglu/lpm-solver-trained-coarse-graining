"""Second bond-law parameter: per-bond ductility rho (fracture energy / brittle energy) learned from patch fracture
tests (torch environment, GPU).

Target (patch_frac.py): rho_W(b) = W_patch / (1/2 k0 eps_i^2) >= 1, the work needed to separate the fine patch of
coarse bond b divided by its elastic energy at the first fine break (dimensionless; transfers to the coarse bond
whose onset is the I2 in-situ critical stretch).  Loss: MSE on log rho over all bonds.
Network: I2's CGNet (D4-equivariant CNN + per-bond heads) with 6 input channels
  [log Kn1/Kref, Kn2/Kn1, Tv/Kn1, void, phase (inclusion = 1), interface flag]
initialised from the I2 critical-stretch network (first 4 input channels and all hidden layers; the new channels and
the last layer start at zero, bias = mean target), output log rho = softplus(o + beta).
  python train_rho.py train S [--epochs 40]
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
torch.set_num_threads(2)
from cgnet import CGNet
import common  # noqa
from common import i2data

DEV = "cuda"


from rhofeat import iface_map, extra_channels, features6, rho_target  # noqa: E402


def load_split(split, s, nmax):
    P = torch.load(i2data("prep2_%s_s%d.pt" % (split, s)), weights_only=False)
    d = np.load(i2data("%s_n48.npz" % split), allow_pickle=True)
    pt = np.load("data/patch_%s.npz" % split)
    S = min(nmax, len(pt["idx"]))
    idx = pt["idx"][:S]
    n = int(d["n"])
    Xe = np.stack([extra_channels(d["phase"][i], n, n, s) for i in idx])
    X = torch.cat([P["X"][idx], torch.as_tensor(Xe)], 1)
    y0 = torch.as_tensor(np.log(rho_target(pt, s, 0)[:S]))
    y1 = torch.as_tensor(np.log(rho_target(pt, s, 1)[:S]))
    return dict(X=X, gi0=P["gi0"], gi1=P["gi1"], y0=y0, y1=y1, S=S)


def init_from_frac(model, s):
    sd = torch.load(common.i2model("fracnet_s%d.pt" % s))
    own = model.state_dict()
    for k, v in sd.items():
        if k == "trunk.0.weight":
            w = torch.zeros_like(own[k])
            w[:, :4] = v
            own[k] = w
        elif k.endswith(".4.weight") or k.endswith(".4.bias"):
            continue                                   # last head layer: zero (fresh)
        else:
            own[k] = v
    model.load_state_dict(own)


class RhoNet(torch.nn.Module):
    def __init__(self, s, beta):
        super().__init__()
        self.net = CGNet(s, cin=6, nout=1)
        self.beta = beta

    def forward(self, X, gi0, gi1):
        o0, o1 = self.net(X, gi0, gi1)
        f = torch.nn.functional.softplus
        return f(o0[..., 0] + self.beta), f(o1[..., 0] + self.beta)


def export_np(model, fn, beta):
    sd = {k: v.detach().cpu().numpy() for k, v in model.net.state_dict().items()}
    np.savez(fn, s=model.net.s, beta=beta, **sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ntrain", type=int, default=400)
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    T = load_split("train", s, a.ntrain)
    V = load_split("val", s, 100)
    m = float(torch.cat([T["y0"].ravel(), T["y1"].ravel()]).mean())
    beta = math.log(math.expm1(max(m, 1e-3)))
    model = RhoNet(s, beta).to(DEV)
    init_from_frac(model.net, s)
    gi = lambda D: ([x.to(DEV) for x in D["gi0"]], [x.to(DEV) for x in D["gi1"]])
    gT, gV = gi(T), gi(V)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * math.ceil(T["S"] / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    def evaluate(D, g):
        model.eval()
        p0, p1 = [], []
        with torch.no_grad():
            for i0 in range(0, D["S"], 16):
                q0, q1 = model(D["X"][i0:i0 + 16].to(DEV), *g)
                p0.append(q0.cpu()); p1.append(q1.cpu())
        p0, p1 = torch.cat(p0).double(), torch.cat(p1).double()
        y = torch.cat([D["y0"].ravel(), D["y1"].ravel()])
        p = torch.cat([p0.ravel(), p1.ravel()])
        rmse = float(((p - y) ** 2).mean().sqrt())
        r2 = 1 - float(((p - y) ** 2).mean() / ((y - y.mean()) ** 2).mean())
        const = float(((y - m) ** 2).mean().sqrt())
        return dict(rmse_log=rmse, r2=r2, rmse_const=const, mean_pred=float(p.mean()), mean_true=float(y.mean()),
                    rel_err_rho=float((torch.exp(p) / torch.exp(y) - 1).abs().mean()))
    log = dict(hist=[], s=s, beta=beta)
    t0 = time.time()
    best = 1e9
    print("train %d / val %d samples, mean log rho %.3f" % (T["S"], V["S"], m), flush=True)
    print("val before training:", evaluate(V, gV), flush=True)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(T["S"])
        tl = []
        for i0 in range(0, T["S"], a.bs):
            idx = perm[i0:i0 + a.bs]
            p0, p1 = model(T["X"][idx].to(DEV), *gT)
            loss = 0.5 * (((p0.double() - T["y0"][idx].to(DEV)) ** 2).mean() + ((p1.double() - T["y1"][idx].to(DEV)) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(float(loss.detach()))
        if ep % 5 == 4 or ep == a.epochs - 1:
            r = evaluate(V, gV)
            log["hist"].append(dict(ep=ep + 1, train=float(np.mean(tl)), **r))
            if r["rmse_log"] < best:
                best = r["rmse_log"]
                torch.save(model.state_dict(), "models/rhonet_s%d.pt" % s)
                export_np(model, "models/rhonet_s%d.npz" % s, beta)
            print("ep %3d train %.5f | val %s (%.0fs)" % (ep + 1, np.mean(tl), {k: round(v, 4) for k, v in r.items()},
                                                          time.time() - t0), flush=True)
    log["time"] = time.time() - t0
    json.dump(log, open("results/train_rho_s%d.json" % s, "w"), indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
