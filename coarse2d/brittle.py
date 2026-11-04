"""Quasi-static brittle LPM (plmode 6 logic of LPM-C) on lin2d lattices, with per-bond critical bond strain.

Algorithm = LPM-C main loop for plmode 6:
  for every displacement step: equilibrium -> bonds with dL/l0 >= crit are candidates -> break at most `nbreak`
  (largest dL/(l0 crit) first, both directions of a bond) -> re-equilibrate at the same load (goto) -> repeat until
  no candidate -> next step.
Differences (documented): small-strain linear equilibrium solved exactly (sparse LU) instead of modified Newton + CG
(the elastic problem between two break events is linear, so this is the same equilibrium up to the 1e-4 Newton
tolerance and O(strain) geometric non-linearity); the first step is jumped to the load
where the first bond reaches its critical strain (no bond can break before, so the history is identical);
particles that lose all bonds are fixed (as updateCrack does).
Loading: tension in y with whole-block grips (load case tb_ty of cg.py), macroscopic strain step `deps`.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import lin2d


def grips(lat, s):
    J = lat.ij[:, 1] // s
    ncy = lat.ny // s
    return J == 0, J == ncy - 1


def run_tension(lat, Kn, Tv, crit, s, deps=2e-5, nbreak=1, max_steps=400, stop_frac=0.3, max_breaks=4000,
                record_fields=False):
    """Returns dict: eps (applied strain per equilibrium), F (reaction), nb (broken bonds), events (list of
    (pass index, p, k)), first-break strain, peak."""
    bot, top = grips(lat, s)
    Lg = lat.x[top, 1].mean() - lat.x[bot, 1].mean()
    fixed = np.nonzero(bot | top)[0]
    fdofs = np.concatenate([2 * fixed, 2 * fixed + 1])
    uy_top = np.concatenate([np.zeros(len(fixed)), np.where(top[fixed], 1.0, 0.0)])   # unit top displacement
    alive = lat.valid.copy()
    pinned = np.zeros(lat.N, bool)
    q = np.where(lat.valid, lat.nbr, 0)
    l0 = lat.l0[None, :]
    hist_eps, hist_F, hist_nb = [], [], []
    events = []
    nbroken = 0

    def solve(alive, delta):
        K, _, _ = lat.stiffness(Kn, Tv, alive)
        extra = np.nonzero(pinned & ~(bot | top))[0]
        fd = np.concatenate([fdofs, 2 * extra, 2 * extra + 1])
        uf = np.concatenate([uy_top * delta, np.zeros(2 * len(extra))])
        n = K.shape[0]
        free = np.ones(n, bool)
        free[fd] = False
        u = np.zeros(n)
        u[fd] = uf
        Kff = K[free][:, free].tocsc()
        # tiny shift against loose fragments (only affects singular directions)
        d = Kff.diagonal().mean()
        Kff = Kff + sp.identity(Kff.shape[0], format="csc") * (1e-12 * d)
        u[free] = spla.splu(Kff, permc_spec="COLAMD").solve(-K[free][:, fd] @ uf)
        F = (K @ u).reshape(-1, 2)[top, 1].sum()
        return u, F

    # unit solution -> load step of the first break
    u1, F1 = solve(alive, Lg)                         # delta = Lg  <=> macroscopic strain 1
    dL, _ = lat.bond_forces(u1, Kn, Tv, alive)
    ratio1 = np.where(alive, dL / l0 / crit, 0.0)
    lam1 = 1.0 / ratio1.max()                         # strain at which the first bond reaches crit
    k0 = max(1, int(np.ceil(lam1 / deps - 1e-9)))
    peak = 0.0
    first = None
    for step in range(k0, k0 + max_steps):
        eps = step * deps
        delta = eps * Lg
        while True:
            u, F = solve(alive, delta)
            dL, _ = lat.bond_forces(u, Kn, Tv, alive)
            r = np.where(alive, dL / l0 / crit, 0.0)
            cand = np.argwhere(r >= 1.0)
            hist_eps.append(eps)
            hist_F.append(F)
            hist_nb.append(nbroken)
            peak = max(peak, F)
            if len(cand) == 0:
                break
            if first is None:
                first = (eps, F)
            # one entry per undirected bond (both directions have the same strain)
            vals = r[cand[:, 0], cand[:, 1]]
            order = np.argsort(-vals)
            seen = set()
            nb_this = 0
            for o in order:
                p, k = cand[o]
                qq = q[p, k]
                key = (min(p, qq), max(p, qq))
                if key in seen:
                    continue
                seen.add(key)
                alive[p, k] = False
                alive[qq, k ^ 1] = False
                events.append((len(hist_eps) - 1, p, k))
                nbroken += 1
                nb_this += 1
                if nb_this >= nbreak:
                    break
            dead = alive.sum(1) == 0
            pinned |= dead
            if nbroken >= max_breaks:
                break
        if nbroken > 0 and F < stop_frac * peak:
            break
        if nbroken >= max_breaks:
            break
    return dict(eps=np.array(hist_eps), F=np.array(hist_F), nb=np.array(hist_nb), events=events,
                first=first, peak=peak, eps_first_analytic=lam1, alive=alive, Lg=Lg)


def crit_from_phase(lat, phase, xi_m, xi_i, xi_int):
    """Per-bond critical strain from the phases of the two ends."""
    q = np.where(lat.valid, lat.nbr, 0)
    pi, pj = phase[:, None], phase[q]
    c = np.where((pi == 0) & (pj == 0), xi_m, np.where((pi == 1) & (pj == 1), xi_i, xi_int))
    return np.where(lat.valid, c, np.inf)


def broken_midpoints(lat, events):
    pts = []
    for (_, p, k) in events:
        qq = lat.nbr[p, k]
        pts.append(0.5 * (lat.x[p] + lat.x[qq]))
    return np.array(pts).reshape(-1, 2)


def path_metrics(pf, pc, L, cell, nbins=None):
    """Crack path comparison between fine (pf) and coarse (pc) broken-bond midpoints.
    chamfer: symmetric mean nearest distance (in coarse cell units); prof: mean |y_f(x) - y_c(x)| of the
    crack profile (median y of broken bonds per x-bin of one cell width), over bins where both have breaks."""
    if len(pf) == 0 or len(pc) == 0:
        return dict(chamfer=np.nan, prof=np.nan, cover=np.nan)
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(pc).query(pf)
    d2, _ = cKDTree(pf).query(pc)
    ch = 0.5 * (d1.mean() + d2.mean()) / cell
    nb = int(round(L / cell)) if nbins is None else nbins
    edges = np.linspace(0, L, nb + 1)
    bf = np.digitize(pf[:, 0], edges) - 1
    bc = np.digitize(pc[:, 0], edges) - 1
    diffs = []
    for b in range(nb):
        a, c = pf[bf == b, 1], pc[bc == b, 1]
        if len(a) and len(c):
            diffs.append(abs(np.median(a) - np.median(c)))
    cover = len(diffs) / nb
    return dict(chamfer=ch, prof=np.mean(diffs) / cell if diffs else np.nan, cover=cover)
