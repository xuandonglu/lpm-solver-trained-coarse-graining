"""Finite-difference check of d(loss)/d(theta) through torch_lpm (adjoint SPDSolve), small lattice, batch 6 (GPU)."""
import torch, numpy as np
import torch_lpm as T
nc, s = 8, 2
m = T.CoarseT(nc, nc, s)
N = nc * nc; lat = m.lat
rng = np.random.default_rng(0)
B = 6
Kn0 = np.where(lat.shell[None, None, :] == 0, 0.38, 0.19) * (1 + 0.3 * rng.random((B, N, 8)))
q = np.where(lat.valid, lat.nbr, 0); op = lat.opp[None, :].repeat(N, 0)
Kn0 = 0.5 * (Kn0 + Kn0[:, q, op]) * lat.valid
Kn0 = torch.as_tensor(Kn0).cuda(); Tv0 = torch.full((B, N, 8), 0.03, dtype=torch.float64).cuda() * torch.as_tensor(lat.valid).cuda()
ref = m.run(Kn0 * 1.1, Tv0)
ref = {k: {kk: (vv.detach() if vv is not None else None) for kk, vv in v.items()} for k, v in ref.items()}
qt = torch.as_tensor(q).cuda(); opt_ = torch.as_tensor(op).cuda()
def L(th):
    Kn = Kn0 * torch.exp(0.5 * (th + th[:, qt, opt_]))
    a, b, c = T.losses(m, m.run(Kn, Tv0), ref)
    return (a + b + c).sum()
th = torch.zeros(B, N, 8, dtype=torch.float64, device="cuda", requires_grad=True)
L(th).backward(); g = th.grad.cpu().numpy()
for (bb, p, k) in [(0, 10, 0), (3, 27, 4), (5, 35, 2), (2, 0, 4)]:
    e = torch.zeros_like(th); e[bb, p, k] = 1e-6
    fd = (float(L(th.detach() + e)) - float(L(th.detach() - e))) / 2e-6
    print("b %d p %d k %d autograd %.6e fd %.6e" % (bb, p, k, g[bb, p, k], fd))
