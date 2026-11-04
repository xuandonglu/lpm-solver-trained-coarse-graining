"""Fracture of unnotched plates in tension: fine LPM (brittle reference) against coarse LPMs.
  python eval2.py SPLIT NSAMP [--workers W] [--variants ...]     plates of ../coarse2d/data/SPLIT_n48.npz (96 x 96 for
                                                                  test_big) -> results/eval2_plate_SPLIT.json
Coarse models (per s = 2, 3, 4):
  classical    strength first: patch elasticity + patch-localised critical stretch, brittle bonds
  classical2P  energy first: patch elasticity + patch strength + rule-of-mixtures fracture energy; crack-band rule: if
               this energy is below that of a brittle coarse bond, the onset stretch is lowered to match it,
               otherwise bilinear softening carries the extra energy
  ml2P         proposed: network elasticity + network critical stretch + network ductility rho (bilinear softening)
The fine reference of each plate is cached in results/fine/; broken-bond paths go to results/paths/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import argparse
import json
import time
import multiprocessing as mp
import common  # noqa
from common import i2data
import numpy as np

SS = (2, 3, 4)
VARIANTS = ["classical", "classical2P", "ml2P"]


def split_file(split):
    return i2data("%s_n%d.npz" % (split, 96 if split == "test_big" else 48))


def softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def bond_type_counts(phase, nx, ny, s):
    """per forward coarse bond (windows order): number of fine bonds of type mm, ii, int touching its two cells."""
    import lin2d
    import windows as WIN
    lat = lin2d.Lattice2D("square", nx, ny, 1.0)
    q = np.where(lat.valid, lat.nbr, 0)
    ph = phase.astype(np.int64)
    t = np.where((ph[:, None] == 0) & (ph[q] == 0), 0, np.where((ph[:, None] == 1) & (ph[q] == 1), 1, 2))
    ncx = nx // s
    cell = (lat.ij[:, 1] // s) * ncx + lat.ij[:, 0] // s
    cnt = np.zeros((ncx * (ny // s), 3))
    for typ in range(3):
        m = lat.valid & (t == typ)
        np.add.at(cnt[:, typ], np.broadcast_to(cell[:, None], m.shape)[m], 1.0)
    b, shell = WIN.bond_list(ncx, ny // s)
    qq, _ = WIN.neighbor_of(b, ncx)
    c = cnt[b[:, 0]] + cnt[qq]
    return c[shell == 0], c[shell == 1]


def classical2P_law(latc, ncx, ncy, s, xi, Kn, phase, nx, ny, contrast, PE):
    """crack-band rule with rule-of-mixtures fracture energy (per bond)."""
    import cparams
    import soft
    import lin2d
    c0, c1 = bond_type_counts(phase, nx, ny, s)
    key = str(float(contrast)) if str(float(contrast)) in PE["G_i"] else min(PE["G_i"], key=lambda k: abs(float(k) - contrast))
    Gm, Gi, Gint = PE["G_m"], PE["G_i"][key], PE["G_int"][key]
    G = [(c[:, 0] * Gm + c[:, 1] * Gi + c[:, 2] * Gint) / c.sum(1) for c in (c0, c1)]
    Grom = cparams.to_und(latc, ncx, ncy, G[0], G[1])
    lam_s = PE["Lambda"][str(s)]
    Cm = lin2d.iso_C(1.0, 0.3)
    knm = {0: (Cm[0] - Cm[1]) / 2, 1: Cm[2] / 2}
    P, K, Q, KR = soft.undirected(latc)
    sh = latc.shell[K]
    kn_ref = np.where(sh == 0, knm[0], knm[1])
    kn_b = Kn[P, K]
    Gcb = lam_s * Gm * (kn_b / kn_ref) * (xi / 1e-3) ** 2
    rho = Grom / Gcb
    xi2 = np.where(rho < 1, xi * np.sqrt(np.maximum(rho, 1e-6)), xi)
    return soft.Law(xi2, np.maximum(rho, 1.0)), float(np.mean(rho < 1))


def run_one(test, lat, Kn, Tv, law, s_bc, nx, ny, mu, kappa, lam_max=np.inf):
    import spec
    t0 = time.perf_counter()
    r = spec.run_test(test, lat, Kn, Tv, law, s_bc, nx, ny, mu=mu, kappa=kappa, lam_max=lam_max)
    r["time"] = time.perf_counter() - t0
    r["T"] = law.T
    return r


def metrics_plate(rf, rc, s):
    import spec
    out = dict(e_init=(rc["eps1"] - rf["eps1"]) / rf["eps1"], e_peak=(rc["peak"] - rf["peak"]) / rf["peak"])
    jf, jc = int(np.argmax(rf["F"])), int(np.argmax(rc["F"]))
    out["e_eps_peak"] = (rc["lam"][jc] - rf["lam"][jf]) / rf["lam"][jf]
    out["e_D"] = (rc["D"] - rf["D"]) / rf["D"]
    lam_end = 1.5 * rf["lam"][jf]
    grid = np.linspace(0, lam_end, 600)
    Ff, Fc = spec.F_at(rf, grid), spec.F_at(rc, grid)
    out["curve"] = float(np.abs(Fc - Ff).mean() / rf["peak"])
    post = grid >= rf["lam"][jf]
    out["curve_post"] = float(np.abs(Fc[post] - Ff[post]).mean() / rf["peak"])
    out["resid_post"] = float(Fc[post].mean() / rf["peak"])
    return out


def one(args):
    split, i, variants = args
    test = "plate"
    import cg, soft, spec, cparams
    from rhofeat import features6
    from cgnet_geom import bond_gather_index
    PE = json.load(open("results/phase_energy.json"))
    d = np.load(split_file(split), allow_pickle=True)
    n = int(d["n"])
    nx = ny = n
    ph = d["phase"][i].astype(np.int64)
    Cpix = d["Cpix"][i]
    contrast = float(d["contrast"][i])
    cr = np.load("data/crit_%s.npz" % split)
    mu, kappa = spec.kf_elastic(Cpix)
    lat, Kn, Tv, crit = spec.fine_model(Cpix, ph, nx, ny)
    P, K, Q, KR = soft.undirected(lat)
    out = dict(i=i, contrast=contrast, phi=float(ph.mean()))
    os.makedirs("results/fine", exist_ok=True)
    os.makedirs("results/paths", exist_ok=True)
    for s in SS:
        # ---------------- fine reference (cached)
        cache = "results/fine/%s_%s_%d_s%d.npz" % (test, split, i, s)
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            rf = {k: z[k] for k in z.files}
            for k in ("eps1", "peak", "D", "W", "Dm", "Dr", "time", "passes"):
                rf[k] = float(rf[k])
            rf["events"] = [(int(e[0]), int(e[1]), int(e[2]), float(e[3])) for e in rf["events"]]
            rf["T"] = np.ones(len(P), np.int64)
        else:
            rf = run_one(test, lat, Kn, Tv, soft.Law(crit[P, K]), s, nx, ny, mu, kappa)
            pf = soft.broken_midpoints(lat, rf)
            np.savez(cache, lam=rf["lam"], F=rf["F"], eps1=rf["eps1"], peak=rf["peak"], D=rf["D"], W=rf["W"], Dm=rf["Dm"],
                     Dr=rf["Dr"], time=rf["time"], passes=rf["passes"], pf=pf, events=np.array(rf["events"], float),
                     broken=rf["broken"], wb=rf["wb"], dead0=rf["dead0"])
            rf["pf"] = pf
        pf = rf["pf"]
        res = {"fine": dict(eps1=rf["eps1"], peak=rf["peak"], D=rf["D"], Dm=rf["Dm"], Dr=rf["Dr"], time=rf["time"],
                            passes=rf["passes"], nbroken=int(len(pf)))}
        # ---------------- coarse parameters (timed: they are part of the cost of each coarse model)
        ncx = ncy = n // s
        latc = cparams.coarse_lat(nx, ny, s)
        Pc, Kc, Qc, KRc = soft.undirected(latc)
        tim = {}
        t0 = time.perf_counter()
        cg.patch_cells(Cpix, nx, ny, s)            # timing only (identical cells are stored in the data file)
        tim["patch_elastic"] = time.perf_counter() - t0
        Kp, Tp = cparams.cells_elastic(latc, d["patch_%d" % s][i])
        t0 = time.perf_counter()
        cparams.naive_crit(Cpix, ph, nx, ny, s)
        tim["naive_crit"] = time.perf_counter() - t0
        tim["patch_crit"] = float(cr["tpcrit_%d" % s][i])
        c_patch = cparams.to_und(latc, ncx, ncy, cr["patch_%d_0" % s][i], cr["patch_%d_1" % s][i])
        t0 = time.perf_counter()
        Kml, Tml = cparams.ml_elastic(Cpix, nx, ny, s, common.i2model("cgnet_s%d.npz" % s))
        tim["ml_elastic"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        o0, o1 = cparams.net_out(Cpix, nx, ny, s, common.i2model("fracnet_s%d.npz" % s))
        c_ml = cparams.to_und(latc, ncx, ncy, cr["naive_%d_0" % s][i] * np.exp(o0[:, 0]), cr["naive_%d_1" % s][i] * np.exp(o1[:, 0]))
        tim["ml_crit"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        X6 = features6(Cpix, ph, nx, ny, s).astype(np.float64)
        b0, b1 = cparams.fwd(ncx, ncy)
        netr = cparams.net("models/rhonet_s%d.npz" % s)
        beta = float(np.load("models/rhonet_s%d.npz" % s)["beta"])
        q0, q1 = netr(X6, bond_gather_index(b0, ncx, ncy), bond_gather_index(b1, ncx, ncy))
        rho_ml = np.exp(cparams.to_und(latc, ncx, ncy, softplus(q0[:, 0] + beta), softplus(q1[:, 0] + beta)))
        tim["ml_rho"] = time.perf_counter() - t0
        law_c2, frac_cb = classical2P_law(latc, ncx, ncy, s, c_patch, Kp, ph, nx, ny, contrast, PE)
        V = {"classical": (Kp, Tp, lambda: soft.Law(c_patch)),
             "classical2P": (Kp, Tp, lambda: law_c2),
             "ml2P": (Kml, Tml, lambda: soft.Law(c_ml, rho_ml))}
        res["timing_pre"] = tim
        res["classical2P_frac_reduced"] = frac_cb
        paths = {}
        for name in variants:
            a, b, mk = V[name]
            rc = run_one(test, latc, a, b, mk(), 1, nx, ny, mu, kappa)
            pc = soft.broken_midpoints(latc, rc)
            paths[name] = pc
            m = dict(time=rc["time"], passes=rc["passes"], eps1=rc["eps1"], peak=rc["peak"], D=rc["D"], Dm=rc["Dm"],
                     Dr=rc["Dr"], nbroken=int(len(pc)), ledger=abs(rc["W"] - rc["U"] - rc["Dm"] - rc["Dr"]) / max(rc["W"], 1e-300))
            m.update(metrics_plate(rf, rc, s))
            m.update(spec.path_metrics(pf, pc, float(s)))
            res[name] = m
        np.savez_compressed("results/paths/%s_%s_%d_s%d.npz" % (test, split, i, s), pf=pf, phase=ph,
                            lam_f=rf["lam"], F_f=rf["F"], **{"pc_" + k: v for k, v in paths.items()})
        out[s] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("nsamp", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(a.split, i, a.variants) for i in range(a.start, a.start + a.nsamp)]
    with mp.Pool(a.workers) as pool:
        R = pool.map(one, jobs, chunksize=1)
    fn = "results/eval2_plate_%s.json" % a.split
    json.dump(R, open(fn, "w"), default=float)
    print("plates %s: %d samples in %.0f s -> %s" % (a.split, len(R), time.time() - t0, fn), flush=True)
    summarize(R)


def summarize(R):
    keys = ["e_init", "e_peak", "e_D", "curve", "haus95", "chamfer"]
    for s in SS:
        print("== s=%d  fine time %.2fs" % (s, np.mean([r[s]["fine"]["time"] for r in R])))
        for v in VARIANTS:
            rows = [r[s][v] for r in R if v in r[s]]
            if not rows:
                continue
            txt = []
            for k in keys:
                x = np.array([row.get(k, np.nan) for row in rows], float)
                if k.startswith("e_"):
                    txt.append("%s %.3f(%+.3f)" % (k[2:], np.nanmean(np.abs(x)), np.nanmean(x)))
                else:
                    txt.append("%s %.3f" % (k, np.nanmean(x)))
            print("  %-12s %s  t %.2fs" % (v, "  ".join(txt), np.mean([row["time"] for row in rows])))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
