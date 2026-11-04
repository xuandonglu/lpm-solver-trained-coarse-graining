"""Ablation: lookup-table (LUT) coarse bond parameters vs the I2 CNN (2D square lattice, torch environment, GPU).

Library in the style of data-driven FE coarsening (Chen, Levin, Sueda, Matusik 2015): the parameters of a coarse
bond are looked up by the phase configuration of its two END CELLS only (no ring of neighbours) and the contrast.
  exact : key = exact pixel pattern of the two end cells (2 s^2 bits) + contrast
  frac  : key = (inclusion pixel count of A, of B) + contrast   (clustered library)
Table entries are trained END-TO-END through the differentiable coarse LPM with the same parameterisation (Voigt
start, Kn = Kn_V exp(o1), Tv = Tv_V + K_AB o2), data, loss and schedule as the I2 CNN, and the output is averaged
over the 4 views of the bond's stabiliser like the CNN (exact D4 equivariance).  Keys not seen in training -> Voigt.
Contrasts outside {5, 10} use the nearest contrast in log (test_c7: c = 7.5 -> 10, test_c15 -> 10).

  python lut2d.py train {exact,frac} S [--epochs 60] [--lr 1e-2]
  python lut2d.py eval  {exact,frac,cnn} S          (cnn = I2 CNN + Voigt + patch on the same splits)
Data and models of the parent folder (coarse2d) are only read; outputs go to this folder.
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
I2 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, I2)
import torch_lpm as T                                                                      # noqa: E402
from train_cg import Scatter, batch_ref, summarize, run_split, baseline_fn, ml_params, DEV  # noqa: E402
from cgnet import CGNet                                                                    # noqa: E402

DATA = os.path.join(I2, "data")
EVAL_SPLITS = ["test_id", "test_phi", "test_big", "test_c7", "test_c15"]
CONTRASTS = np.array([5.0, 10.0])
DIRS = {0: (1, 0), 2: (0, 1), 4: (1, 1), 6: (1, -1)}
for _d in ("models", "results", "logs"):
    os.makedirs(os.path.join(HERE, _d), exist_ok=True)


def load_prep(split, s):
    return torch.load(os.path.join(DATA, "prep2_%s_s%d.pt" % (split, s)), weights_only=False)


def load_phase(split):
    n = 96 if split == "test_big" else 48
    d = np.load(os.path.join(DATA, "%s_n%d.npz" % (split, n)), allow_pickle=True)
    return d["phase"].reshape(-1, n, n).astype(np.int64), np.asarray(d["contrast"], float)


def contrast_id(c):
    return np.abs(np.log(c)[:, None] - np.log(CONTRASTS)[None, :]).argmin(1)


def end_cells(Ph, s, b):
    """Ph (S, n, n) [sample, j, i]; forward bonds b = [p, k, I, J] -> A, B (S, nb, s, s), canonical orientation."""
    S, n, _ = Ph.shape
    nc = n // s
    C = Ph.reshape(S, nc, s, nc, s).transpose(0, 1, 3, 2, 4)            # (S, J, I, r, c)
    k, I, J = b[:, 1], b[:, 2], b[:, 3]
    di = np.array([DIRS[x][0] for x in k])
    dj = np.array([DIRS[x][1] for x in k])
    A = C[:, J, I]
    B = C[:, J + dj, I + di]
    t = (k == 2)[None, :, None, None]                                    # +y: transpose (windows.extract, slot 2)
    A = np.where(t, np.swapaxes(A, -1, -2), A)
    B = np.where(t, np.swapaxes(B, -1, -2), B)
    f = (k == 6)[None, :, None, None]                                    # (+1,-1): flip rows (slot 6)
    A = np.where(f, A[..., ::-1, :], A)
    B = np.where(f, B[..., ::-1, :], B)
    return A, B


def _bits(x):
    """(..., m) 0/1 -> int64 key."""
    w = (np.int64(1) << np.arange(x.shape[-1], dtype=np.int64))
    return (x.astype(np.int64) * w).sum(-1)


def view_keys(Ph, contrast, s, b, shell, variant):
    """(S, nb, 4) int64 keys, one per stabiliser view (axial: id, flip cols, flip rows, rot180;
    diagonal: id, transpose, rot180, anti-transpose)."""
    A, B = end_cells(Ph, s, b)
    S, nb = A.shape[:2]
    cid = contrast_id(contrast)[:, None, None]
    if variant == "frac":
        a, bb = A.sum((-1, -2)), B.sum((-1, -2))
        if shell == 0:
            pairs = [(a, bb), (bb, a), (a, bb), (bb, a)]
        else:
            pairs = [(a, bb), (a, bb), (bb, a), (bb, a)]
        keys = np.stack([x * (s * s + 1) + y for x, y in pairs], -1)
    else:
        if shell == 0:
            E = np.concatenate([A, B], -1)                                # (S, nb, s, 2s): A | B
            views = [E, E[..., ::-1], E[..., ::-1, :], E[..., ::-1, ::-1]]
            keys = np.stack([_bits(v.reshape(S, nb, -1)) for v in views], -1)
        else:
            tr = lambda x: np.swapaxes(x, -1, -2)                          # noqa: E731
            rot = lambda x: x[..., ::-1, ::-1]                             # noqa: E731
            views = [(A, B), (tr(A), tr(B)), (rot(B), rot(A)), (tr(rot(B)), tr(rot(A)))]
            keys = np.stack([_bits(np.concatenate([x.reshape(S, nb, -1), y.reshape(S, nb, -1)], -1))
                             for x, y in views], -1)
    return keys * len(CONTRASTS) + cid


class LUT(torch.nn.Module):
    def __init__(self, V0, V1):
        super().__init__()
        self.t0 = torch.nn.Parameter(torch.zeros(V0, 2, dtype=torch.float64))
        self.t1 = torch.nn.Parameter(torch.zeros(V1, 2, dtype=torch.float64))

    def forward(self, i0, i1):
        """i: (B, nb, 4) table rows, -1 = unseen; mean over the seen views (no seen view -> 0 = Voigt)."""
        out = []
        for t, i in ((self.t0, i0), (self.t1, i1)):
            m = i >= 0
            v = t[torch.where(m, i, torch.zeros_like(i))] * m[..., None]
            out.append(v.sum(-2) / m.sum(-1, keepdim=True).clamp(min=1))
        return out


def build_index(split, s, P, vocab, variant):
    Ph, c = load_phase(split)
    out, cov = [], {}
    for sh, key in ((0, "b0"), (1, "b1")):
        K = view_keys(Ph, c, s, P[key], sh, variant)
        V = vocab[sh]
        pos = np.clip(np.searchsorted(V, K), 0, len(V) - 1)
        idx = np.where(V[pos] == K, pos, -1)
        out.append(torch.as_tensor(idx))
        cov["shell%d_view_seen" % sh] = float((idx >= 0).mean())
        cov["shell%d_bond_any_seen" % sh] = float((idx >= 0).any(-1).mean())
    return out[0], out[1], cov


def make_vocab(s, variant):
    P = load_prep("train", s)
    Ph, c = load_phase("train")
    return [np.unique(view_keys(Ph, c, s, P[key], sh, variant)) for sh, key in ((0, "b0"), (1, "b1"))]


def lut_params(model, P, I0, I1, idx):
    o0, o1 = model(I0[idx].to(DEV), I1[idx].to(DEV))
    kv0, tv0, kv1, tv1 = [x[idx].to(DEV) for x in P["voigt"]]
    K00, K01 = P["K00"][idx].to(DEV), P["K01"][idx].to(DEV)
    return (kv0 * torch.exp(o0[..., 0]), tv0 + K00 * o0[..., 1], kv1 * torch.exp(o1[..., 0]), tv1 + K01 * o1[..., 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("variant")
    ap.add_argument("s", type=int)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--pen", type=float, default=10.0)
    a = ap.parse_args()
    s = a.s
    torch.manual_seed(0)
    tag = "%s_s%d" % (a.variant, s)

    if a.mode == "train":
        t0 = time.time()
        vocab = make_vocab(s, a.variant)
        P, V = load_prep("train", s), load_prep("val", s)
        I0, I1, cov_tr = build_index("train", s, P, vocab, a.variant)
        J0, J1, cov_v = build_index("val", s, V, vocab, a.variant)
        nc = P["nc"]
        coarse = T.CoarseT(nc, nc, s)
        sc = Scatter(P)
        model = LUT(len(vocab[0]), len(vocab[1])).to(DEV)
        S = P["S"]
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        steps = a.epochs * math.ceil(S / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        print("train %s: vocab %d / %d, val coverage %s, %.0f s prep" % (tag, len(vocab[0]), len(vocab[1]), cov_v,
                                                                      time.time() - t0), flush=True)
        log = dict(vocab=[len(vocab[0]), len(vocab[1])], cov_val=cov_v, hist=[])
        best = 1e9
        for ep in range(a.epochs):
            perm = torch.randperm(S)
            tl = []
            for i0 in range(0, S, a.bs):
                idx = perm[i0:i0 + a.bs]
                Kn, Tv = sc(*lut_params(model, P, I0, I1, idx))
                pen, _ = coarse.stability_penalty(Kn, Tv)
                out = coarse.run(Kn, Tv)
                Lu, Ls, Lg = T.losses(coarse, out, batch_ref(P, idx))
                loss = (Lu + Ls + Lg + a.pen * pen).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tl.append(float(loss))
            if ep % 5 == 4 or ep == a.epochs - 1:
                m = summarize(run_split(V, coarse, sc, lambda idx: lut_params(model, V, J0, J1, idx)))
                score = m["plate_sig"] + m["ring_sig"] + m["plate_G"] + m["ring_G"]
                log["hist"].append(dict(ep=ep + 1, train_loss=float(np.mean(tl)), **m))
                if score < best:
                    best = score
                    torch.save(dict(state=model.state_dict(), vocab0=vocab[0], vocab1=vocab[1]),
                               os.path.join(HERE, "models", "lut_%s.pt" % tag))
                print("ep %3d loss %.4e | val plate sig %.4f G %.4f | ring sig %.4f G %.4f (%.0fs)"
                      % (ep + 1, np.mean(tl), m["plate_sig"], m["plate_G"], m["ring_sig"], m["ring_G"],
                         time.time() - t0), flush=True)
        log["train_time_s"] = time.time() - t0
        json.dump(log, open(os.path.join(HERE, "results", "train_lut_%s.json" % tag), "w"), indent=1)
        return

    if a.mode == "eval":
        res = {}
        if a.variant == "cnn":
            model = CGNet(s).to(DEV)
            model.load_state_dict(torch.load(os.path.join(I2, "models", "cgnet_s%d.pt" % s)))
            model.eval()
        else:
            ck = torch.load(os.path.join(HERE, "models", "lut_%s.pt" % tag), weights_only=False)
            vocab = [ck["vocab0"], ck["vocab1"]]
            model = LUT(len(vocab[0]), len(vocab[1])).to(DEV)
            model.load_state_dict(ck["state"])
        for sp in EVAL_SPLITS:
            P = load_prep(sp, s)
            nc = P["nc"]
            coarse = T.CoarseT(nc, nc, s)
            sc = Scatter(P)
            bs = 4 if P["n"] > 48 else 16
            if a.variant == "cnn":
                r = dict(voigt=summarize(run_split(P, coarse, sc, baseline_fn(P, "voigt"), bs)),
                         patch=summarize(run_split(P, coarse, sc, baseline_fn(P, "patch"), bs)),
                         cnn=summarize(run_split(P, coarse, sc, lambda idx: ml_params(model, P, idx, "voigt"), bs)))
            else:
                I0, I1, cov = build_index(sp, s, P, vocab, a.variant)
                with torch.no_grad():
                    r = dict(lut=summarize(run_split(P, coarse, sc, lambda idx: lut_params(model, P, I0, I1, idx), bs)),
                             coverage=cov)
            res[sp] = r
            print("%s %s: %s" % (tag, sp, {k: {kk: round(vv, 4) for kk, vv in v.items() if kk in
                                                 ("plate_sig", "ring_sig", "plate_G", "ring_G", "shell0_view_seen")}
                                            for k, v in r.items()}), flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "eval_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
