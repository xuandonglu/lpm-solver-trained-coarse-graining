"""Quasi-static LPM fracture with a per-bond softening (cohesive) bond law, on lin2d lattices (square / hex).

Bond law (per undirected bond b, parameters xi0_b = onset/peak stretch, rho_b >= 1 = ductility):
  envelope (pair force / pair stiffness, in stretch units):  f(xi) = xi                                   xi <= xi0
                                                            f(xi) = xi0 (xi_f - xi) / (xi_f - xi0)       xi0 <= xi <= xi_f
  fracture energy of the bond = area under f = 1/2 k (xi0 l)(xi_f l) = rho * (brittle energy 1/2 k (xi0 l)^2),
  i.e. xi_f = rho xi0.  rho = 1 is the brittle bond of LPM-C plmode 6 (critical stretch xi0).
Damage enters exactly like LPM-C's damage_w (kernels.compute_dL / force_from_state): the bond stretch is multiplied
by w_b in the pair term AND in the multibody sums, so K = B^T W H W B and the pair secant stiffness is w^2 * 2Kn.
Irreversible damage + secant unloading: w only decreases; unloading/reloading is linear through the origin.

Realisation = sequentially linear 'saw-tooth' (event driven, same loop as the brittle code): the envelope is replaced
by T_b teeth; tooth t has secant factor phi_t = w^2 (phi_0 = 1, phi_T = 0) and fails when the bond stretch reaches
xi_t (on the envelope).  A tooth failure is an EVENT handled exactly like a bond break in plmode 6 (one event per
pass, largest stretch / threshold first, re-equilibrate at the same load = LPM-C goto).  The tooth stretches are
chosen so that the energy dissipated by the saw-tooth equals the envelope energy exactly (bisection per bond).
No iterations, no convergence issue, snap-backs handled as in the brittle code.

Energy ledger (all exact for the piecewise-linear history):
  W   external work  (elastic increments between events are linear: trapezoid is exact)
  U   stored elastic energy = 1/2 lam F (only prescribed displacements)
  Dm  'material' release: energy drop of an event at FIXED displacement (the bond's own release, per bond Db)
  Dr  'structural/unstable' release: energy drop while re-equilibrating at fixed load after an event
  closure: W - U = Dm + Dr
Two loading modes: 'step' = fixed load steps + goto (identical to brittle.run_tension, used for the plates as in I2),
                   'exact' = load jumps exactly to the next event (the deps -> 0 limit; used for the DCB).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ------------------------------------------------------------------------------------------------------------
# bond law
# ------------------------------------------------------------------------------------------------------------
def _sawtooth_energy(x, T):
    """Normalised saw-tooth energy (units xi0^2) for envelope end x = xi_e/xi0 and T teeth (arrays, same shape)."""
    T = np.asarray(T)
    tmax = int(T.max())
    E = np.zeros_like(x, dtype=float)
    for i in range(tmax):
        act = i < T
        Ti = np.maximum(T, 1)
        f_i = 1.0 - i / Ti
        xi_i = 1.0 + (i / Ti) * (x - 1.0)
        phi_i = f_i / xi_i
        f_n = 1.0 - (i + 1) / Ti
        xi_n = 1.0 + ((i + 1) / Ti) * (x - 1.0)
        phi_n = np.where(i + 1 < T, f_n / xi_n, 0.0)
        E += np.where(act, 0.5 * (phi_i - phi_n) * xi_i ** 2, 0.0)
    return E


class Law:
    """Per undirected bond: thr (nb, Tmax) failure stretch of tooth t (inf for t >= T_b), phi (nb, Tmax+1) secant
    factor of tooth t (1 for t = 0, 0 for t >= T_b), T (nb,) number of teeth, xi0, rho, G (= rho, energy in units of
    the brittle energy 1/2 k (xi0 l)^2)."""

    def __init__(self, xi0, rho=None, nmax=12, drho=0.1, rho_brittle=1.02, xi_tip=None):
        """xi_tip: optional per-bond critical stretch used (brittle, full break) once the bond is at a crack tip,
        i.e. one of its two particles already has a fully broken bond (crack = broken bonds)."""
        self.xi_tip = None if xi_tip is None else np.asarray(xi_tip, float)
        xi0 = np.asarray(xi0, float)
        rho = np.ones_like(xi0) if rho is None else np.maximum(np.asarray(rho, float), 1.0)
        T = np.where(rho <= rho_brittle, 1, np.clip(1 + np.ceil((rho - 1.0) / drho), 2, nmax)).astype(np.int64)
        T = np.where(np.isfinite(xi0), T, 1)
        # bisection for x = xi_e / xi0 such that sawtooth energy = rho / 2
        lo = np.ones_like(rho)
        hi = np.maximum(2.0, 4.0 * rho * T)
        tgt = 0.5 * np.where(T > 1, rho, 1.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            e = _sawtooth_energy(mid, T)
            big = e > tgt
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        x = 0.5 * (lo + hi)
        Tm = int(T.max())
        thr = np.full((len(xi0), Tm), np.inf)
        phi = np.zeros((len(xi0), Tm + 1))
        for i in range(Tm):
            act = i < T
            f_i = 1.0 - i / T
            xr = 1.0 + (i / T) * (x - 1.0)
            thr[:, i] = np.where(act, xi0 * xr, np.inf)
            phi[:, i] = np.where(act, f_i / xr, 0.0)
        phi[:, 0] = 1.0
        self.thr, self.phi, self.T, self.xi0, self.rho, self.xe = thr, phi, T, xi0, rho, xi0 * x
        self.E = _sawtooth_energy(x, T) * 2.0          # realised energy / brittle energy (= rho up to 1e-12)


def undirected(lat):
    """Forward copies (even slot) of all valid bonds: P, K, Q, KR (= K ^ 1)."""
    m = lat.valid & (np.arange(lat.NN)[None, :] % 2 == 0)
    P, K = np.nonzero(m)
    return P, K, lat.nbr[P, K], K ^ 1


# ------------------------------------------------------------------------------------------------------------
# fast assembly of K_ff = (W B)^T H (W B) restricted to the free dofs
# ------------------------------------------------------------------------------------------------------------
class Assembler:
    def __init__(self, lat, Kn, Tv, free):
        N, NN = lat.N, lat.NN
        self.lat = lat
        valid = lat.valid
        self.B = lat.bond_matrix()                      # all valid slots
        self.H = lat.bond_hessian(Kn, Tv)               # all valid slots
        rows = np.arange(N * NN).reshape(N, NN)
        # H entries e = (r, c, h)
        R, C, Hv = [], [], []
        for c in (0, 1):
            ks = np.nonzero(lat.shell == c)[0]
            for a in ks:
                for b in ks:
                    m = valid[:, a] & valid[:, b]
                    R.append(rows[m, a]); C.append(rows[m, b])
                    v = 0.5 * (Tv[m, a] + Tv[m, b])
                    if a == b:
                        v = v + Kn[m, a]
                    Hv.append(v)
        R, C, Hv = np.concatenate(R), np.concatenate(C), np.concatenate(Hv)
        self.eR, self.eC, self.eH = R, C, Hv
        # B row r = (p, k): +n_k on dofs of p, -n_k on dofs of q
        p_of = np.repeat(np.arange(N), NN)
        k_of = np.tile(np.arange(NN), N)
        q_of = np.where(valid.ravel(), lat.nbr.ravel(), 0)
        nd = lat.ndir

        def rowdofs(r):
            p, k, q = p_of[r], k_of[r], q_of[r]
            d = [2 * p, 2 * p + 1, 2 * q, 2 * q + 1]
            v = [nd[k, 0], nd[k, 1], -nd[k, 0], -nd[k, 1]]
            return d, v
        dr, vr = rowdofs(R)
        dc, vc = rowdofs(C)
        I, J, E, V = [], [], [], []
        ne = len(R)
        eid = np.arange(ne)
        for a in range(4):
            for b in range(4):
                I.append(dr[a]); J.append(dc[b]); E.append(eid); V.append(vr[a] * vc[b])
        I, J, E, V = np.concatenate(I), np.concatenate(J), np.concatenate(E), np.concatenate(V)
        self.free = free
        fidx = -np.ones(2 * N, np.int64)
        fidx[free] = np.arange(free.sum())
        keep = free[I] & free[J]
        fi, fj, E, V = fidx[I[keep]], fidx[J[keep]], E[keep], V[keep]
        nf = int(free.sum())
        key = fi * nf + fj
        uk, inv = np.unique(key, return_inverse=True)
        self.M = sp.csr_matrix((V, (inv, E)), shape=(len(uk), ne))
        self.ind = (uk % nf).astype(np.int32)
        cnt = np.bincount(uk // nf, minlength=nf)
        self.ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
        self.nf = nf

    def Kff(self, wdir):
        """wdir: (N*NN,) damage weight per directed slot."""
        data = self.M @ (self.eH * wdir[self.eR] * wdir[self.eC])
        return sp.csr_matrix((data, self.ind, self.ptr), shape=(self.nf, self.nf))

    def internal(self, u, wdir):
        """P = B^T W H W B u, bond-space z = W B u and H z."""
        z = wdir * (self.B @ u)
        Hz = self.H @ z
        return self.B.T @ (wdir * Hz), z, Hz


# ------------------------------------------------------------------------------------------------------------
# boundary conditions
# ------------------------------------------------------------------------------------------------------------
def bc_plate(lat, s):
    """Plate in tension in y, whole-block grips (as brittle.grips / cg 'tb_ty'): lam = macroscopic strain,
    displacement of the top grip = lam * Lg, reported force = sum of the top-grip reactions (y)."""
    J = lat.ij[:, 1] // s
    ncy = lat.ny // s
    bot, top = J == 0, J == ncy - 1
    Lg = lat.x[top, 1].mean() - lat.x[bot, 1].mean()
    g = np.zeros((lat.N, 2))
    g[top, 1] = Lg
    return dict(fixed=bot | top, g=g, Fscale=1.0 / Lg, name="plate")


def bc_dcb(lat, s, ncy_half):
    """Double cantilever beam: left block column of the upper arm moved (0, +lam/2), lower arm (0, -lam/2);
    lam = opening, reported force = (P_up,y - P_low,y)/2 (conjugate to the opening)."""
    I = lat.ij[:, 0] // s
    J = lat.ij[:, 1] // s
    up = (I == 0) & (J >= ncy_half)
    lo = (I == 0) & (J < ncy_half)
    g = np.zeros((lat.N, 2))
    g[up, 1] = 0.5
    g[lo, 1] = -0.5
    return dict(fixed=up | lo, g=g, Fscale=1.0, name="dcb")


# ------------------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------------------
def run(lat, Kn, Tv, law, bc, mode="step", deps=5e-6, max_steps=400, stop_frac=0.02, max_events=200000,
        stop_fn=None, lam_max=np.inf, dead0=None, record_u=False, max_passes=10 ** 7, unbreakable=None):
    """law: Law over the undirected bonds of `undirected(lat)` (forward order).  dead0: optional (nb,) bool of bonds
    removed from the start (pre-crack).  Returns a dict with the history (lam, F per pass), events, ledger."""
    N, NN = lat.N, lat.NN
    P, Kf, Q, KR = undirected(lat)
    nb = len(P)
    slot_f = P * NN + Kf
    slot_r = Q * NN + KR
    l0 = lat.l0[Kf]
    fixed = bc["fixed"]
    fdofs = np.concatenate([2 * np.nonzero(fixed)[0], 2 * np.nonzero(fixed)[0] + 1])
    free = np.ones(2 * N, bool)
    free[fdofs] = False
    g = bc["g"].ravel()
    A = Assembler(lat, Kn, Tv, free)
    tooth = np.zeros(nb, np.int64)
    wb = np.ones(nb)
    if dead0 is not None:
        tooth[dead0] = law.T[dead0]
        wb[dead0] = 0.0
    wdir = np.zeros(N * NN)

    def set_w():
        wdir[:] = 0.0
        wdir[slot_f] = wb
        wdir[slot_r] = wb
    set_w()
    shift_cache = {}

    def solve(lam):
        Kff = A.Kff(wdir)
        d = Kff.diagonal()
        dm = d.mean()
        Kff = Kff + sp.diags(np.where(d > 1e-14 * dm, 1e-12 * dm, dm))     # tiny shift; isolated dofs -> u = 0
        up = np.zeros(2 * N)
        up[~free] = g[~free] * lam
        Pp, _, _ = A.internal(up, wdir)
        u = up.copy()
        u[free] = spla.splu(Kff.tocsc(), permc_spec="COLAMD").solve(-Pp[free])
        Pin, z, Hz = A.internal(u, wdir)
        F = float(g @ Pin)                       # conjugate to lam
        xi = (A.B @ u)[slot_f] / l0              # geometric bond stretch of the forward copies
        return u, F, xi, z, Hz

    nbrk_at = np.zeros(N, np.int64)          # fully broken bonds per particle (crack topology)
    if dead0 is not None:
        np.add.at(nbrk_at, P[dead0], 1)
        np.add.at(nbrk_at, Q[dead0], 1)
    tipfull = np.zeros(nb, bool)

    def thr_now():
        t = np.minimum(tooth, law.thr.shape[1] - 1)
        th = law.thr[np.arange(nb), t]
        th = np.where((tooth < law.T) & (wb > 0), th, np.inf)
        if unbreakable is not None:
            th = np.where(unbreakable, np.inf, th)
        if law.xi_tip is not None:
            at_tip = ((nbrk_at[P] > 0) | (nbrk_at[Q] > 0)) & (wb > 0)
            use = at_tip & (law.xi_tip < th) & (np.isfinite(th) | (tooth < law.T))
            if unbreakable is not None:
                use &= ~unbreakable
            tipfull[:] = use
            th = np.where(use, law.xi_tip, th)
        return th

    H_lam, H_F, H_ev = [], [], []
    events = []                     # (pass index, bond, new tooth, lam, Dm)
    Db = np.zeros(nb)
    led = dict(W=0.0, Dm=0.0, Dr=0.0)
    nev = 0
    peak = 0.0
    first = None
    passes = 0

    def event(u, lam, F, xi, z, Hz, b):
        """apply one tooth failure on bond b at the equilibrium (u, lam); returns material release."""
        nonlocal nev
        U0 = 0.5 * lam * F
        tooth[b] = law.T[b] if tipfull[b] else tooth[b] + 1
        wb[b] = np.sqrt(law.phi[b, tooth[b]]) if tooth[b] < law.T[b] else 0.0
        if wb[b] == 0.0:
            nbrk_at[P[b]] += 1
            nbrk_at[Q[b]] += 1
        set_w()
        z2 = wdir * (A.B @ u)
        U1 = 0.5 * float(z2 @ (A.H @ z2))
        dm = U0 - U1
        Db[b] += dm
        led["Dm"] += dm
        nev += 1
        return U1

    if mode == "step":
        u1, F1, xi1, _, _ = solve(1.0)
        passes += 1
        r1 = np.where(xi1 > 0, xi1 / thr_now(), 0.0)
        lam1 = 1.0 / r1.max()
        k0 = max(1, int(np.ceil(lam1 / deps - 1e-9)))
        lam_prev, F_prev = 0.0, 0.0
        for step in range(k0, k0 + max_steps):
            lam = step * deps
            Ufix = None
            first_in_step = True
            while True:
                u, F, xi, z, Hz = solve(lam)
                passes += 1
                if first_in_step:
                    led["W"] += 0.5 * (F_prev + F) * (lam - lam_prev)
                    first_in_step = False
                if Ufix is not None:
                    led["Dr"] += Ufix - 0.5 * lam * F
                H_lam.append(lam); H_F.append(F); H_ev.append(nev)
                peak = max(peak, F)
                r = np.where(xi > 0, xi / thr_now(), 0.0)
                b = int(np.argmax(r))
                if r[b] < 1.0 or nev >= max_events or passes >= max_passes:
                    break
                if first is None:
                    first = (lam, F)
                Ufix = event(u, lam, F, xi, z, Hz, b)
                events.append((len(H_lam) - 1, b, int(tooth[b]), lam))
            lam_prev, F_prev = lam, F
            if nev > 0 and F < stop_frac * peak:
                break
            if nev >= max_events or lam > lam_max or passes >= max_passes:
                break
            if stop_fn is not None and stop_fn(events, wb):
                break
        eps1 = lam1
    else:   # exact event-driven
        lam = 0.0
        eps1 = None
        F_after = 0.0
        Ufix = None
        while True:
            u1, F1, xi1, z1, Hz1 = solve(1.0)
            passes += 1
            if nev > 0:
                F_after = F1 * lam
                led["Dr"] += Ufix - 0.5 * lam * F_after
                H_lam.append(lam); H_F.append(F_after); H_ev.append(nev)
                if F_after < stop_frac * peak or (stop_fn is not None and stop_fn(events, wb)):
                    break
            r = np.where(xi1 > 0, xi1 / thr_now(), 0.0)
            b = int(np.argmax(r))
            if r[b] <= 0 or nev >= max_events or passes >= max_passes:
                break
            lam_e = max(lam, 1.0 / r[b])
            if lam_e > lam_max:
                led["W"] += 0.5 * F1 * (lam_max ** 2 - lam ** 2)
                lam = lam_max
                H_lam.append(lam); H_F.append(F1 * lam); H_ev.append(nev)
                break
            led["W"] += 0.5 * F1 * (lam_e ** 2 - lam ** 2)
            lam = lam_e
            Fb = F1 * lam
            H_lam.append(lam); H_F.append(Fb); H_ev.append(nev)
            peak = max(peak, Fb)
            if eps1 is None:
                eps1 = lam
                first = (lam, Fb)
            Ufix = event(u1 * lam, lam, Fb, xi1 * lam, z1 * lam, Hz1 * lam, b)
            events.append((len(H_lam) - 1, b, int(tooth[b]), lam))
    H_lam, H_F = np.array(H_lam), np.array(H_F)
    U_end = 0.5 * H_lam[-1] * H_F[-1] if len(H_F) else 0.0
    fs = bc.get("Fscale", 1.0)
    first = None if first is None else (first[0], first[1] * fs)
    return dict(lam=H_lam, F=H_F * fs, nev=np.array(H_ev), events=events, first=first, eps1=eps1,
                peak=peak * fs, W=led["W"], Dm=led["Dm"], Dr=led["Dr"], U=U_end, D=led["W"] - U_end, Db=Db,
                wb=wb.copy(), tooth=tooth.copy(), passes=passes, nb=nb, P=P, K=Kf,
                broken=(wb == 0) & ~(dead0 if dead0 is not None else np.zeros(nb, bool)),
                dead0=(dead0 if dead0 is not None else np.zeros(nb, bool)))


def broken_midpoints(lat, res):
    P, K = res["P"], res["K"]
    m = res["broken"]
    q = lat.nbr[P[m], K[m]]
    return 0.5 * (lat.x[P[m]] + lat.x[q])


def damaged_midpoints(lat, res, wmax=0.999):
    P, K = res["P"], res["K"]
    m = res["wb"] < wmax
    q = lat.nbr[P[m], K[m]]
    return 0.5 * (lat.x[P[m]] + lat.x[q]), res["wb"][m]


def williams_unit(dx, dy, mu, kappa):
    """Mode-I LEFM displacement for K = 1 (plane strain kappa = 3 - 4 nu); crack along -x from the tip."""
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    c = np.sqrt(r / (2 * np.pi)) / (2 * mu)
    return np.stack([c * np.cos(th / 2) * (kappa - np.cos(th)), c * np.sin(th / 2) * (kappa - np.cos(th))], -1)


def bc_kfield(lat, s, tip, mu, kappa, layers=1):
    """K-field (modified boundary layer) test: outer ring of whole blocks prescribed u = lam * u_K(x - tip);
    lam = K_app; reported force = conjugate generalised force dU/dlam (U = lam F / 2)."""
    I = lat.ij[:, 0] // s
    J = lat.ij[:, 1] // s
    ncx, ncy = lat.nx // s, lat.ny // s
    ring = (I < layers) | (I >= ncx - layers) | (J < layers) | (J >= ncy - layers)
    g = np.zeros((lat.N, 2))
    g[ring] = williams_unit(lat.x[ring, 0] - tip[0], lat.x[ring, 1] - tip[1], mu, kappa)
    return dict(fixed=ring, g=g, Fscale=1.0, name="kfield")


class CrackTracker:
    """Main crack = cluster of fully broken bonds connected (through shared particles) to the pre-crack
    (union-find over particles).  advance() = largest distance from the initial tip point tip0 among main-cluster
    bonds lying ahead of it (midpoint x >= tip0_x - 1), so a deflected crack also counts."""

    def __init__(self, lat, P, Q, dead0, tip0):
        self.parent = np.arange(lat.N)
        mid = 0.5 * (lat.x[P] + lat.x[Q])
        self.adv = np.where(mid[:, 0] >= tip0[0] - 1.0, np.hypot(mid[:, 0] - tip0[0], mid[:, 1] - tip0[1]), 0.0)
        self.P, self.Q = P, Q
        self.cx = {}
        idx = np.nonzero(dead0)[0]
        for b in idx:
            self.add(int(b))
        self.seed = int(P[idx[0]]) if len(idx) else None

    def find(self, a):
        r = a
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[a] != r:
            self.parent[a], a = r, self.parent[a]
        return r

    def add(self, b):
        ra, rb = self.find(int(self.P[b])), self.find(int(self.Q[b]))
        m = max(self.cx.get(ra, 0.0), self.cx.get(rb, 0.0), self.adv[b])
        if ra != rb:
            self.parent[rb] = ra
            self.cx.pop(rb, None)
        self.cx[ra] = m

    def advance(self):
        return self.cx.get(self.find(self.seed), 0.0) if self.seed is not None else 0.0
