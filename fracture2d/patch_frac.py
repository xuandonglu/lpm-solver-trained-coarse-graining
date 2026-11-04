"""Patch fracture tests: for every coarse bond, a small fine-LPM tension bar cut out of the microstructure along the
bond, pulled until it separates (traditional brittle LPM: critical bond stretch, one break per pass, exact
event-driven load = the deps -> 0 limit of the plmode-6 loop).  All bonds of one sample and class are solved as one
batch of small dense systems (numpy).

Patch of coarse bond b (direction n, length l_b = s or s*sqrt2, lateral spacing of parallel bonds l_perp = s or
s/sqrt2), in the bond frame (p along n, t across), origin at the bond midpoint:
    free part   |p| <= l_b            (axial bonds: exactly the two end cells)
    grips       l_b < |p| <= l_b + g  (g = 2 lattice layers: 2 for axial, 2 sqrt2 for diagonal pulls)
    width       |t| <= (l_perp + 2m)/2,  m = ceil(s/2) pixels of lateral margin, lateral faces free
    loading     grip '-' fixed, grip '+' displaced by eps * 2 l_b * n  (both components prescribed)
Specimen boundary: free sites outside the specimen are removed (real free surface); grip sites outside are kept with
reflected material (so every bond can be pulled).
Outputs per forward coarse bond (order of windows.bond_list): eps_i (nominal strain at the first fine break), F_i,
eps_pk, F_pk (peak), k0 (initial stiffness F/eps), W (work to separation), Dm (material release of the broken fine
bonds at fixed displacement), U_end (stored energy left), nbrk, sep (separated?), eps_end.
Energies in units of F*eps (only ratios are used: rho_W = W / (1/2 k0 eps_i^2) etc.).
usage: python patch_frac.py SPLIT [--nmax K] [--start K0] [--workers 8] [--ss 2 3 4]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import math
import time
import multiprocessing as mp
import common  # noqa
from common import i2data
import numpy as np
import scipy.sparse as sp
import lin2d

XI_M, XI_I, XI_INT = 1.0e-3, 2.0e-3, 0.6e-3
S2 = math.sqrt(2.0)
# forward slot k -> (direction n, base cell offset) ; base = fine site (bi*s, bj*s) + (I, J)*s
CLS_DIR = {0: (np.array([1.0, 0.0]), (1, 0), (0.5, 0.5)),       # x bond: base ((I+1)s, J s), rel = (di+.5, dj+.5-s/2)
           2: (np.array([0.0, 1.0]), (0, 1), (0.5, 0.5)),       # y bond: base (I s, (J+1)s), rel = (di+.5-s/2, dj+.5)
           4: (np.array([1.0, 1.0]) / S2, (1, 1), (0.5, 0.5)),  # (1,1): base ((I+1)s, (J+1)s)
           6: (np.array([1.0, -1.0]) / S2, (1, 0), (0.5, 0.5))}  # (1,-1) from (I,J) to (I+1,J-1): base ((I+1)s, J s)


class Template:
    def __init__(self, s, k):
        n, (bi, bj), _ = CLS_DIR[k]
        self.s, self.k, self.n = s, k, n
        diag = k >= 4
        lb = s * S2 if diag else float(s)
        lperp = s / S2 if diag else float(s)
        m = int(math.ceil(s / 2))
        g = 2 * S2 if diag else 2.0
        W = lperp + 2 * m
        self.L = 2 * lb
        R = int(math.ceil(lb + g + W)) + 2
        di, dj = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
        di, dj = di.ravel(), dj.ravel()
        ox = 0.5 - (s / 2 if k == 2 else 0.0)
        oy = 0.5 - (s / 2 if k == 0 else 0.0)
        rx, ry = di + ox, dj + oy
        t = np.array([-n[1], n[0]])
        p = rx * n[0] + ry * n[1]
        q = rx * t[0] + ry * t[1]
        eps = 1e-9
        sel = (np.abs(p) <= lb + g + eps) & (np.abs(q) <= W / 2 + eps)
        di, dj, p = di[sel], dj[sel], p[sel]
        self.bi, self.bj = bi, bj
        i0, j0 = di.min(), dj.min()
        wx, wy = di.max() - i0 + 1, dj.max() - j0 + 1
        act = np.zeros((wy, wx), bool)
        act[dj - j0, di - i0] = True
        self.lat = lin2d.Lattice2D("square", wx, wy, 1.0, active=act)
        # template site order = lat.sites order (row-major over the bounding box)
        site_di = self.lat.ij[:, 0] + i0
        site_dj = self.lat.ij[:, 1] + j0
        self.di, self.dj = site_di, site_dj
        pp = (site_di + ox) * n[0] + (site_dj + oy) * n[1]
        qq = (site_di + ox) * t[0] + (site_dj + oy) * t[1]
        self.gm = pp < -lb - eps
        self.gp = pp > lb + eps
        self.free_site = ~(self.gm | self.gp)
        self.pp, self.qq = pp, qq
        N, NN = self.lat.N, self.lat.NN
        # dofs: D = free dofs then grip+ dofs (grip- is zero)
        fdof = np.concatenate([2 * np.nonzero(self.free_site)[0], 2 * np.nonzero(self.free_site)[0] + 1])
        gdof = np.concatenate([2 * np.nonzero(self.gp)[0], 2 * np.nonzero(self.gp)[0] + 1])
        fdof.sort(); gdof.sort()
        self.fdof, self.gdof = fdof, gdof
        D = np.concatenate([fdof, gdof])
        self.nf, self.ng = len(fdof), len(gdof)
        nD = len(D)
        dmap = -np.ones(2 * N, np.int64)
        dmap[D] = np.arange(nD)
        # unit-strain grip displacement: delta = L (eps = 1) along n
        ubar = np.zeros(2 * N)
        ubar[2 * np.nonzero(self.gp)[0]] = self.L * n[0]
        ubar[2 * np.nonzero(self.gp)[0] + 1] = self.L * n[1]
        self.ubar_full = ubar
        self.ubar_g = ubar[gdof]
        # H entries and the map to the dense K over D
        valid = self.lat.valid
        rows = np.arange(N * NN).reshape(N, NN)
        R_, C_ = [], []
        for c in (0, 1):
            ks = np.nonzero(self.lat.shell == c)[0]
            for a in ks:
                for b in ks:
                    mm = valid[:, a] & valid[:, b]
                    R_.append(rows[mm, a]); C_.append(rows[mm, b])
        self.eR, self.eC = np.concatenate(R_), np.concatenate(C_)
        self.ediag = self.eR == self.eC
        p_of = np.repeat(np.arange(N), NN)
        k_of = np.tile(np.arange(NN), N)
        q_of = np.where(valid.ravel(), self.lat.nbr.ravel(), 0)
        nd = self.lat.ndir

        def rowdofs(r):
            pr, kr, qr = p_of[r], k_of[r], q_of[r]
            return ([2 * pr, 2 * pr + 1, 2 * qr, 2 * qr + 1], [nd[kr, 0], nd[kr, 1], -nd[kr, 0], -nd[kr, 1]])
        dr, vr = rowdofs(self.eR)
        dc, vc = rowdofs(self.eC)
        I, J, E, V = [], [], [], []
        eid = np.arange(len(self.eR))
        for a in range(4):
            for b in range(4):
                I.append(dr[a]); J.append(dc[b]); E.append(eid); V.append(vr[a] * vc[b])
        I, J, E, V = np.concatenate(I), np.concatenate(J), np.concatenate(E), np.concatenate(V)
        keep = (dmap[I] >= 0) & (dmap[J] >= 0)
        lin = dmap[I[keep]] * nD + dmap[J[keep]]
        self.MD = sp.csr_matrix((V[keep], (lin, E[keep])), shape=(nD * nD, len(self.eR)))
        self.nD = nD
        self.B = self.lat.bond_matrix()                   # (N*NN, 2N)
        self.l0 = np.broadcast_to(self.lat.l0[None, :], (N, NN)).ravel()
        self.valid_flat = valid.ravel()
        self.nbr_flat = np.where(valid, self.lat.nbr, 0).ravel()
        self.opp_slot = (q_of * NN + (k_of ^ 1))           # flat index of the reverse directed slot
        self.N, self.NN = N, NN

    def global_sites(self, b, ncx, P, npad):
        """padded-lattice site index (nb, N_T) of every template site for forward bonds b [p,k,I,J]."""
        I, J = b[:, 2], b[:, 3]
        bi = (I + self.bi) * self.s + P
        bj = (J + self.bj) * self.s + P
        gi = bi[:, None] + self.di[None, :]
        gj = bj[:, None] + self.dj[None, :]
        return gj * npad + gi, gi - P, gj - P


def padded_specimen(Cpix, phase, nx, ny, P):
    C2 = np.pad(Cpix.reshape(ny, nx, 3), ((P, P), (P, P), (0, 0)), mode="symmetric")
    ph = np.pad(phase.reshape(ny, nx), ((P, P), (P, P)), mode="symmetric")
    lat = lin2d.Lattice2D("square", nx + 2 * P, ny + 2 * P, 1.0)
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(C2.reshape(-1, 3)))
    import brittle
    crit = brittle.crit_from_phase(lat, ph.ravel().astype(np.int64), XI_M, XI_I, XI_INT)
    return lat, Kn, Tv, crit


def run_patches(T, Knb, Tvb, critb, alive, chunk=96, max_iter=None, sep_tol=1e-3):
    """Batched exact event-driven brittle pull of nb patches (per-patch per-slot arrays (nb, N*NN)).
    Returns dict of per-patch arrays."""
    nb = Knb.shape[0]
    nf, nD = T.nf, T.nD
    out = {k: np.zeros(nb) for k in ("eps_i", "F_i", "eps_pk", "F_pk", "k0", "W", "Dm", "U_end", "nbrk", "sep",
                                     "eps_end")}
    max_iter = max_iter or int(T.valid_flat.sum() // 2 + 2)
    l0 = T.l0
    for c0 in range(0, nb, chunk):
        idx = np.arange(c0, min(nb, c0 + chunk))
        al = alive[idx].copy()
        Kn, Tv, cr = Knb[idx], Tvb[idx], critb[idx]
        hv0 = 0.5 * (Tv[:, T.eR] + Tv[:, T.eC]) + np.where(T.ediag[None, :], Kn[:, T.eR], 0.0)
        nb_c = len(idx)
        lam = np.zeros(nb_c)
        W = np.zeros(nb_c)
        Dm = np.zeros(nb_c)
        peak = np.zeros(nb_c)
        epk = np.zeros(nb_c)
        ei = np.full(nb_c, np.nan)
        Fi = np.full(nb_c, np.nan)
        k0 = np.full(nb_c, np.nan)
        nbrk = np.zeros(nb_c)
        done = np.zeros(nb_c, bool)
        sep = np.zeros(nb_c, bool)
        uold = None
        Ubefore = None
        Fend = np.zeros(nb_c)
        for it in range(max_iter):
            act = np.nonzero(~done)[0]
            if len(act) == 0:
                break
            w = al[act].astype(float)
            hv = hv0[act] * w[:, T.eR] * w[:, T.eC]
            KD = (T.MD @ hv.T).T.reshape(len(act), nD, nD)
            if uold is not None:
                # material release of the previous event at fixed displacement
                ua = uold[act]
                Ufix = 0.5 * np.einsum("bi,bij,bj->b", ua, KD, ua)
                Dm[act] += Ubefore[act] - Ufix
            Kff = KD[:, :nf, :nf]
            Kfg = KD[:, :nf, nf:]
            Kgg = KD[:, nf:, nf:]
            dg = np.einsum("bii->bi", Kff)
            dmean = np.maximum(dg.mean(1, keepdims=True), 1e-30)
            reg = np.where(dg > 1e-14 * dmean, 1e-12 * dmean, 1.0 * dmean)
            Kff = Kff + reg[:, :, None] * np.eye(nf)[None]
            rhs = -Kfg @ T.ubar_g
            uf = np.linalg.solve(Kff, rhs[..., None])[..., 0]
            F1 = np.einsum("g,bgf,bf->b", T.ubar_g, np.transpose(Kfg, (0, 2, 1)), uf) + np.einsum("g,bgh,h->b", T.ubar_g, Kgg, T.ubar_g)
            u = np.zeros((len(act), 2 * T.N))
            u[:, T.fdof] = uf
            u[:, T.gdof] = T.ubar_g[None, :]
            dL = (T.B @ u.T).T
            if it == 0:
                k0[act] = F1
            # separation: stiffness lost
            sepnow = F1 < sep_tol * k0[act]
            r = np.where(al[act] & (dL > 0), dL / l0[None, :] / cr[act], 0.0)
            j = np.argmax(r, 1)
            rmax = r[np.arange(len(act)), j]
            fin = sepnow | (rmax <= 0)
            # finish
            for a_, flag in ((act[fin], True),):
                Fend[a_] = F1[fin] * lam[a_]
                sep[a_] = sepnow[fin]
                done[a_] = True
            go = ~fin
            ag = act[go]
            if len(ag) == 0:
                break
            F1g = F1[go]
            lam_e = np.maximum(lam[ag], 1.0 / rmax[go])
            W[ag] += 0.5 * F1g * (lam_e ** 2 - lam[ag] ** 2)
            lam[ag] = lam_e
            Fb = F1g * lam_e
            newpk = Fb > peak[ag]
            peak[ag] = np.where(newpk, Fb, peak[ag])
            epk[ag] = np.where(newpk, lam_e, epk[ag])
            firstm = np.isnan(ei[ag])
            ei[ag[firstm]] = lam_e[firstm]
            Fi[ag[firstm]] = Fb[firstm]
            # break bond j (both directed copies)
            jj = j[go]
            al[ag, jj] = False
            al[ag, T.opp_slot[jj]] = False
            nbrk[ag] += 1
            if uold is None:
                uold = np.zeros((nb_c, nD))
                Ubefore = np.zeros(nb_c)
            ugD = np.concatenate([uf[go], np.broadcast_to(T.ubar_g, (len(ag), T.ng))], 1) * lam_e[:, None]
            uold[ag] = ugD
            Ubefore[ag] = 0.5 * Fb * lam_e
        U_end = 0.5 * Fend * lam
        for key, v in (("eps_i", ei), ("F_i", Fi), ("eps_pk", epk), ("F_pk", peak), ("k0", k0), ("W", W),
                       ("Dm", Dm), ("U_end", U_end), ("nbrk", nbrk), ("sep", sep.astype(float)), ("eps_end", lam)):
            out[key][idx] = v
    return out


_TPL = {}


def template(s, k):
    if (s, k) not in _TPL:
        _TPL[(s, k)] = Template(s, k)
    return _TPL[(s, k)]


def sample_patches(Cpix, phase, nx, ny, s, chunk=96):
    """All coarse bonds of one specimen -> dict of arrays for shell 0 (axial) and shell 1 (diagonal) in
    windows.bond_list order."""
    import windows as WIN
    ncx, ncy = nx // s, ny // s
    b, shell = WIN.bond_list(ncx, ncy)
    P = 3 * s + 6
    latp, Knp, Tvp, critp = padded_specimen(Cpix, phase, nx, ny, P)
    npad = nx + 2 * P
    NN = 8
    res = {}
    for k in (0, 2, 4, 6):
        T = template(s, k)
        sel = np.nonzero(b[:, 1] == k)[0]
        bb = b[sel]
        gs, gi, gj = T.global_sites(bb, ncx, P, npad)
        inside = (gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
        active = inside | (~T.free_site[None, :])
        flat = (gs[:, :, None] * NN + np.arange(NN)[None, None, :]).reshape(len(bb), -1)
        Knb = Knp.ravel()[flat]
        Tvb = Tvp.ravel()[flat]
        crb = critp.ravel()[flat]
        nbr = T.nbr_flat
        site_of_slot = np.repeat(np.arange(T.N), NN)
        alive = T.valid_flat[None, :] & active[:, site_of_slot] & active[:, nbr]
        Knb = np.where(alive, Knb, 0.0)
        Tvb = np.where(alive, Tvb, 0.0)
        crb = np.where(alive, crb, np.inf)
        o = run_patches(T, Knb, Tvb, crb, alive, chunk=chunk)
        o["sel"] = sel
        res[k] = o
    out = {}
    for sh, ks in ((0, (0, 2)), (1, (4, 6))):
        idx = np.nonzero(shell == sh)[0]
        pos = {g: i for i, g in enumerate(idx)}
        for key in res[0]:
            if key == "sel":
                continue
            v = np.zeros(len(idx))
            for k in ks:
                v[[pos[g] for g in res[k]["sel"]]] = res[k][key]
            out["%s_%d" % (key, sh)] = v
    return out


def _one(args):
    path, k, ss = args
    d = np.load(path, allow_pickle=True)
    ph = d["phase"][k].astype(np.int64)
    Cpix = d["Cpix"][k]
    nx = int(d["nx"]) if "nx" in d else int(d["n"])
    ny = int(d["ny"]) if "ny" in d else int(d["n"])
    out = {}
    t = {}
    for s in ss:
        t0 = time.perf_counter()
        o = sample_patches(Cpix, ph, nx, ny, s)
        t[s] = time.perf_counter() - t0
        for key, v in o.items():
            out["%s_s%d" % (key, s)] = v
        out["time_s%d" % s] = t[s]
    return out


def split_path(split):
    if os.path.exists(os.path.join("data", "%s.npz" % split)):
        return os.path.join("data", "%s.npz" % split)
    n = 96 if split == "test_big" else 48
    return i2data("%s_n%d.npz" % (split, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--nmax", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ss", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = split_path(a.split)
    d = np.load(path, allow_pickle=True)
    S = len(d["phase"])
    stop = S if a.nmax == 0 else min(S, a.start + a.nmax)
    jobs = [(path, k, a.ss) for k in range(a.start, stop)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    data = {key: np.array([r[key] for r in res]) for key in res[0]}
    data["idx"] = np.arange(a.start, stop)
    fn = "data/patch_%s%s.npz" % (a.split, a.tag)
    np.savez_compressed(fn, **data)
    print("patch %s [%d:%d] %.0f s; mean time per sample %s" % (a.split, a.start, stop, time.time() - t0,
          {s: round(float(data["time_s%d" % s].mean()), 1) for s in a.ss}), flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _d in ("data", "models", "results", "logs"):
        os.makedirs(_d, exist_ok=True)
    main()
