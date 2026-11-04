"""Homogeneous-material floor of the coarse SC model (I2 sec. 4.1 in 3D): for a homogeneous specimen Voigt = patch =
the exact cell parameters, so any coarse-vs-fine difference is the coarse-graining floor (free-surface softening of the
LPM, which has no surface correction: the soft surface layer is one particle thick, i.e. s voxels in the coarse model).
usage (torch environment): python floor3d.py  ->  logs/floor3d.log"""
import os
import numpy as np
import torch
import geom3d as G
import torch_lpm3d as T
import train3d as TR

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)

out = []
for n in (24, 48):
    lat = G.LatticeGeom(n, n, n, 1.0)
    Cv = G.iso_C(np.ones(lat.N), np.full(lat.N, 0.2))
    Kf, Tf = G.bond_params_two_end_mean(lat, G.sc_params_from_C(Cv, 0.5))
    Kf = torch.as_tensor(Kf, device="cuda")[None]
    Tf = torch.as_tensor(Tf, device="cuda")[None]
    for s in (2, 3):
        nc = n // s
        loads = G.LOADS
        fr = [T.FineRefT(n, s, loads=ld) for ld in ([loads] if n <= 24 else [G.RING, G.PLATES])]
        rs = [f.run(Kf, Tf) for f in fr]
        ref = {k: torch.cat([r[k] for r in rs], 1) for k in ("u", "sig", "w", "F")}
        cm = T.CoarseT(nc, s)
        latc = cm.lat
        Kc, Tc = G.bond_params_from_cells(latc, G.cell_params_from_C66(G.voigt_cells(Cv, n, s), s / 2.0))
        o = cm.run(torch.as_tensor(Kc, device="cuda")[None], torch.as_tensor(Tc, device="cuda")[None],
                   differentiable=False)
        e = TR.errors(cm, o, ref, torch.zeros(1, latc.N, dtype=torch.bool, device="cuda"))
        g = e["G"][0].cpu().numpy()
        sg = e["sig"][0].cpu().numpy()
        msg = ("n=%d s=%d homogeneous: reaction error tb_tz %+.1f%% tb_sx %+.1f%% lr_tx %+.1f%% | KUBC apparent modulus "
               "%+.2f%% | stress field ring %.1f%% plate %.1f%% | free-surface cells %.1f%% | apparent C %.2f%%"
               % (n, s, 100 * g[6], 100 * g[7], 100 * g[8], 100 * np.abs(g[:6]).mean(), 100 * sg[:6].mean(),
                  100 * sg[6:].mean(), 100 * e["sig_surf"][0, 6:].mean().item(), 100 * e["C"][0].item()))
        print(msg, flush=True)
        out.append(msg)
open("logs/floor3d.log", "w").write("\n".join(out) + "\n")
