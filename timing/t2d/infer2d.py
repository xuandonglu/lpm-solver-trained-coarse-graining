"""2D inference worker (torch environment) for the timing table: the 2D bond networks (cgnet_s2, s4 of
../../coarse2d/models, read-only), features + forward (+ upload / download on GPU, synchronised) timed per call; the
numpy post-processing is timed in bench2d.py.  CPU: torch with LPMCG_THREADS (6) intra-op threads.
protocol: {"op": "prep", "n": n, "s": [2, 4]} | {"op": "run", "seed": k, "s": s, "dev": "cpu"|"gpu"} | {"op": "quit"}"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
NT = int(os.environ.get("LPMCG_THREADS", "6"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import el_prob as EP  # noqa: E402
from cgnet_geom import bond_gather_index, pixel_features  # noqa: E402
from cgnet_torch import CGNet  # noqa: E402
import windows as WIN  # noqa: E402

torch.set_num_threads(NT)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
COARSE2D = os.path.normpath(os.path.join(HERE, "..", "..", "coarse2d"))


def main():
    nets = {}
    n = None
    probs = {}
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "quit":
            break
        if op == "prep":
            n = int(cmd["n"])
            for s in cmd["s"]:
                nc = n // s
                d = np.load(os.path.join(COARSE2D, "models", "cgnet_s%d.npz" % s))
                for dev in ("cpu", "cuda"):
                    net = CGNet(s, nout=2)
                    net.load_state_dict({k: torch.from_numpy(np.asarray(d[k])) for k in net.state_dict().keys()})
                    b, shell = WIN.bond_list(nc, nc)
                    gi0 = tuple(torch.as_tensor(x, device=dev) for x in bond_gather_index(b[shell == 0], nc, nc))
                    gi1 = tuple(torch.as_tensor(x, device=dev) for x in bond_gather_index(b[shell == 1], nc, nc))
                    nets[(s, dev)] = (net.to(dev).eval(), gi0, gi1)
            p0 = EP.make_comp(n, 0)
            for (s, dev), (net, gi0, gi1) in nets.items():
                with torch.no_grad():
                    for _ in range(2):
                        X = torch.from_numpy(np.stack([pixel_features(p0["Cpix"], n, n, s)])).to(dev)
                        [x.double().cpu().numpy() for x in net(X, gi0, gi1)]
                if dev == "cuda":
                    torch.cuda.synchronize()
            print(json.dumps(dict(ok=1, n=n, threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(0))),
                  flush=True)
        elif op == "run":
            s, dev = int(cmd["s"]), ("cuda" if cmd["dev"] == "gpu" else "cpu")
            sd = int(cmd["seed"])
            if sd not in probs:
                probs[sd] = EP.make_comp(n, sd)          # problem generation is not part of any method's time
            net, gi0, gi1 = nets[(s, dev)]
            t0 = time.perf_counter()
            X = np.stack([pixel_features(probs[sd]["Cpix"], n, n, s)])
            t1 = time.perf_counter()
            with torch.no_grad():
                o = [x.double().cpu().numpy() for x in net(torch.from_numpy(X).to(dev), gi0, gi1)]
            if dev == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(json.dumps(dict(t_feat=t1 - t0, t_fwd=t2 - t1)), flush=True)
        else:
            print(json.dumps(dict(error="unknown op")), flush=True)


if __name__ == "__main__":
    main()
