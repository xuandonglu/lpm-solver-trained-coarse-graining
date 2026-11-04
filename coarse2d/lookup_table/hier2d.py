"""Ablation, lookup table with backoff (2D): exact end-cell residual on top of the trained inclusion-count table.

    o = frac[key_frac] (lut_frac_s{S}.pt, frozen) + res[key_exact];   exact key not seen in training -> res = 0 -> frac

Why: the plain exact table (lut2d.py exact) trains unstably with Adam + one-cycle because most exact keys are rare
(a row seen once still moves ~25 lr through Adam's momentum and global bias correction; s=4 crashed with a non-positive
definite coarse stiffness, the 3D tables diverged).  Here only the residual is trained, with Adagrad (a row moves only
when its key is in the batch, first step = lr), constant lr, and model selection includes epoch 0 (= the frac table),
so the result is never worse than frac on validation.  Same data, loss, penalty, views and splits as lut2d.py.

  python hier2d.py train S [--epochs 30] [--lr 1e-2]
  python hier2d.py eval  S
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import lut2d as L                                                                            # noqa: E402
from lut2d import T, Scatter, batch_ref, summarize, run_split, DEV                           # noqa: E402


class HierLUT(torch.nn.Module):
    """i (B, nb, 8): 4 frac view rows | 4 exact view rows; both parts are means over the stabiliser views."""

    def __init__(self, base, V0, V1):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.res = L.LUT(V0, V1)

    def forward(self, i0, i1):
        b = self.base(i0[..., :4], i1[..., :4])
        r = self.res(i0[..., 4:], i1[..., 4:])
        return [x + y for x, y in zip(b, r)]


def lookup(K, V):
    pos = np.clip(np.searchsorted(V, K), 0, len(V) - 1)
    return np.where(V[pos] == K, pos, -1)


def build_index(split, s, P, vocab):
    Ph, c = L.load_phase(split)
    out, cov = [], {}
    for sh, key in ((0, "b0"), (1, "b1")):
        fr = lookup(L.view_keys(Ph, c, s, P[key], sh, "frac"), vocab["frac"][sh])
        ex = lookup(L.view_keys(Ph, c, s, P[key], sh, "exact"), vocab["exact"][sh])
        out.append(torch.as_tensor(np.concatenate([fr, ex], -1)))
        cov["shell%d_view_seen" % sh] = float((ex >= 0).mean())
        cov["shell%d_frac_seen" % sh] = float((fr >= 0).mean())
    return out[0], out[1], cov


def new_model(vocab):
    return HierLUT(L.LUT(len(vocab["frac"][0]), len(vocab["frac"][1])), len(vocab["exact"][0]),
                   len(vocab["exact"][1])).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("s", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--pen", type=float, default=10.0)
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    tag = "hier_s%d" % s

    if a.mode == "train":
        t0 = time.time()
        ck = torch.load(os.path.join(HERE, "models", "lut_frac_s%d.pt" % s), weights_only=False)
        vocab = dict(frac=[ck["vocab0"], ck["vocab1"]], exact=L.make_vocab(s, "exact"))
        model = new_model(vocab)
        model.base.load_state_dict(ck["state"])
        P, V = L.load_prep("train", s), L.load_prep("val", s)
        I0, I1, _ = build_index("train", s, P, vocab)
        J0, J1, cov_v = build_index("val", s, V, vocab)
        nc = P["nc"]
        coarse = T.CoarseT(nc, nc, s)
        sc = Scatter(P)
        opt = torch.optim.Adagrad(model.res.parameters(), lr=a.lr)
        S = P["S"]
        print("train %s: exact vocab %d / %d, val coverage %s, %.0f s prep" % (
            tag, len(vocab["exact"][0]), len(vocab["exact"][1]), cov_v, time.time() - t0), flush=True)
        log = dict(vocab=[len(v) for v in vocab["exact"]], cov_val=cov_v, hist=[], nonpd_steps=0)
        best = [1e9]

        def validate(ep, tl):
            with torch.no_grad():
                m = summarize(run_split(V, coarse, sc, lambda idx: L.lut_params(model, V, J0, J1, idx)))
            score = m["plate_sig"] + m["ring_sig"] + m["plate_G"] + m["ring_G"]
            log["hist"].append(dict(ep=ep, train_loss=tl, **m))
            if score < best[0]:
                best[0] = score
                torch.save(dict(state=model.state_dict(), vocab=vocab, ep=ep),
                           os.path.join(HERE, "models", "lut_%s.pt" % tag))
            print("ep %3d loss %.4e | val plate sig %.4f G %.4f | ring sig %.4f G %.4f (%.0fs)"
                  % (ep, tl, m["plate_sig"], m["plate_G"], m["ring_sig"], m["ring_G"], time.time() - t0), flush=True)

        validate(0, float("nan"))
        for ep in range(a.epochs):
            perm = torch.randperm(S)
            tl = []
            for i0 in range(0, S, a.bs):
                idx = perm[i0:i0 + a.bs]
                Kn, Tv = sc(*L.lut_params(model, P, I0, I1, idx))
                pen, _ = coarse.stability_penalty(Kn, Tv)
                try:
                    out = coarse.run(Kn, Tv)
                    Lu, Ls, Lg = T.losses(coarse, out, batch_ref(P, idx))
                    loss = (Lu + Ls + Lg + a.pen * pen).mean()
                except FloatingPointError:                     # non-PD coarse stiffness: step on the penalty only
                    loss = (a.pen * pen).mean()
                    log["nonpd_steps"] += 1
                if not loss.requires_grad:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl.append(float(loss.detach()))
            if ep % 5 == 4 or ep == a.epochs - 1:
                validate(ep + 1, float(np.mean(tl)))
        log["train_time_s"] = time.time() - t0
        json.dump(log, open(os.path.join(HERE, "results", "train_lut_%s.json" % tag), "w"), indent=1)
        return

    if a.mode == "eval":
        ck = torch.load(os.path.join(HERE, "models", "lut_%s.pt" % tag), weights_only=False)
        vocab = ck["vocab"]
        model = new_model(vocab)
        model.load_state_dict(ck["state"])
        print("%s: checkpoint from epoch %d" % (tag, ck["ep"]), flush=True)
        res = {}
        for sp in L.EVAL_SPLITS:
            P = L.load_prep(sp, s)
            nc = P["nc"]
            coarse = T.CoarseT(nc, nc, s)
            sc = Scatter(P)
            bs = 4 if P["n"] > 48 else 16
            I0, I1, cov = build_index(sp, s, P, vocab)
            with torch.no_grad():
                r = dict(lut=summarize(run_split(P, coarse, sc, lambda idx: L.lut_params(model, P, I0, I1, idx), bs)),
                         coverage=cov, ep=ck["ep"])
            res[sp] = r
            print("%s %s: %s | coverage %s" % (tag, sp, {k: round(v, 4) for k, v in r["lut"].items() if k in
                                                          ("plate_sig", "ring_sig", "plate_G", "ring_G")}, cov), flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "eval_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
