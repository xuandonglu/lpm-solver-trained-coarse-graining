"""Checks of torch_lpm3d.py (torch environment, GPU) against lin3d (reference arrays from dev/export_check.py):
operator, stress, particle energy, block diagonal, PCG fine reference (block means, reactions) for s = 2, 3; adjoint
gradient of the differentiable coarse solve vs central finite differences.  Output: logs/check_torch3d.log"""
import os
import time
import numpy as np
import torch
import geom3d as G
import torch_lpm3d as T

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)

out = []


def log(*a):
    s = " ".join(str(v) for v in a)
    print(s, flush=True)
    out.append(s)


dev = "cuda"
d = dict(np.load("dev/check_ref.npz"))
n = 12
lat = G.LatticeGeom(n, n, n, 1.0)
op = T.SCT(lat, dev)
t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float64)
Kn, Tv, u = t(d["Kn"]), t(d["Tv"]), t(d["u"])
rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
log("operator vs lin3d (n=12 two-phase): K u %.1e | stress %.1e | particle energy %.1e | block diagonal %.1e"
    % (rel(op.apply(u, Kn, Tv), t(d["Ku"])), rel(op.stress(u, Kn, Tv), t(d["st"])), rel(op.energy(u, Kn, Tv), t(d["U"])),
       rel(op.block_diag(Kn, Tv), t(d["D"]))))
for s in (2, 3):
    fr = T.FineRefT(n, s, device=dev)
    t0 = time.time()
    r = fr.run(Kn[None], Tv[None], tol=1e-12)
    torch.cuda.synchronize()
    e = []
    for l, ld in enumerate(G.LOADS):
        e.append((rel(r["u"][0, l], t(d["u_%d_%s" % (s, ld)])), rel(r["sig"][0, l], t(d["s_%d_%s" % (s, ld)])),
                  rel(r["w"][0, l] * 1.0, t(d["w_%d_%s" % (s, ld)])),
                  0.0 if ld not in G.PLATES else abs(float(r["F"][0, l]) / float(d["F_%d_%s" % (s, ld)]) - 1)))
    e = np.array(e)
    log("PCG fine reference s=%d (9 loads, tol 1e-12, %d its, %.2f s) vs lin3d SuperLU: max rel. diff block u %.1e, "
        "stress %.1e, energy %.1e, reaction %.1e" % (s, r["iters"], time.time() - t0, *e.max(0)))

# ------------------------------------------------------------------------------------------- adjoint gradient
torch.manual_seed(0)
nc, s = 5, 2
cm = T.CoarseT(nc, s, device=dev)
lat_c = cm.lat
rng = np.random.default_rng(1)
Kn0 = np.zeros((lat_c.N, 18))
Tv0 = np.zeros((lat_c.N, 18))
for i in range(lat_c.N):
    for k in range(0, 18, 2):
        q = lat_c.nbr[i, k]
        if q >= 0:
            Kn0[i, k] = Kn0[q, k ^ 1] = rng.uniform(0.8, 1.2)
            Tv0[i, k] = Tv0[q, k ^ 1] = rng.uniform(-0.03, 0.03)
Wsig = torch.randn(1, len(G.LOADS), lat_c.N, 6, device=dev, dtype=torch.float64)
Wu = torch.randn(1, len(G.LOADS), lat_c.N, 3, device=dev, dtype=torch.float64)


def loss_of(kn, tv):
    o = cm.run(kn, tv, tol=1e-13)
    return (o["sig"] * Wsig).sum() + 1e3 * (o["u"] * Wu).sum() + o["F"].sum() + (o["w"]).sum()


kn = t(Kn0)[None].requires_grad_(True)
tv = t(Tv0)[None].requires_grad_(True)
L = loss_of(kn, tv)
L.backward()
errs = []
for _ in range(6):
    i = int(rng.integers(lat_c.N))
    k = int(rng.choice(np.nonzero(lat_c.valid[i])[0]))
    q = lat_c.nbr[i, k]
    for which in ("kn", "tv"):
        hstep = 1e-6
        A = [t(Kn0)[None].clone(), t(Tv0)[None].clone()]
        idx = 0 if which == "kn" else 1
        g_auto = float((kn.grad if which == "kn" else tv.grad)[0, i, k] + (kn.grad if which == "kn" else tv.grad)[0, q, k ^ 1])
        vals = []
        for sg in (1, -1):
            B = [a.clone() for a in A]
            B[idx][0, i, k] += sg * hstep
            B[idx][0, q, k ^ 1] += sg * hstep
            with torch.no_grad():
                vals.append(float(loss_of(B[0], B[1])))
        g_fd = (vals[0] - vals[1]) / (2 * hstep)
        errs.append(abs(g_auto - g_fd) / max(abs(g_fd), 1e-12))
log("adjoint gradient (coarse 5^3, s=2, all 9 loads, symmetric bond perturbation) vs central FD: max rel. error %.1e "
    "over %d checks" % (max(errs), len(errs)))
open("logs/check_torch3d.log", "w").write("\n".join(out) + "\n")
