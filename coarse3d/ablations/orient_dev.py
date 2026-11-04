"""Orientation consistency of the bond field (torch environment, GPU): for 16 test_id specimens, the bond stiffness / Tv predicted
when the network sees the specimen transformed by each of the 48 cube symmetries (pulled back to the original bonds),
compared with the identity orientation:  |dKn| / Kn = |exp(o1_g - o1_e) - 1|,  |dTv| / Kn_V = |o2_g - o2_e|.
usage: python orient_dev.py  -> results/orient_dev.json"""
import json
import os
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import ablate as A

torch.set_num_threads(1)
res = {}
P = A.load_prep("test_id")
nc = P["nc"]
X = P["X"][:16].to(A.DEV)
for tag, var, fn in (("eq_i8", "eq", os.path.join(A.I8, "models", "bondnet_s2.pt")),
                     ("ne_aug", "ne", os.path.join(A.HERE, "models", "ne_aug.pt")),
                     ("ne_noaug", "ne", os.path.join(A.HERE, "models", "ne_noaug.pt"))):
    net = A.make_net(var).to(A.DEV)
    net.load_state_dict(torch.load(fn, map_location=A.DEV))
    net.eval()
    dk, dt = [], []
    with torch.no_grad():
        o0 = [x.double() for x in net(X, nc)]
        for gi in range(1, 48):
            og = [x.double() for x in A.raw_outputs(net, X, nc, gi)]
            dk.append(torch.cat([(torch.exp(a[..., 0] - b[..., 0]) - 1).abs().flatten() for a, b in zip(og, o0)]).cpu())
            dt.append(torch.cat([(a[..., 1] - b[..., 1]).abs().flatten() for a, b in zip(og, o0)]).cpu())
    dk = torch.cat(dk).numpy()
    dt = torch.cat(dt).numpy()
    res[tag] = dict(dKn_mean=float(dk.mean()), dKn_p95=float(np.quantile(dk, 0.95)), dKn_max=float(dk.max()),
                    dTv_mean=float(dt.mean()), dTv_p95=float(np.quantile(dt, 0.95)))
    print(tag, {k: round(v, 5) for k, v in res[tag].items()}, flush=True)
json.dump(res, open(os.path.join(A.HERE, "results", "orient_dev.json"), "w"), indent=1)
