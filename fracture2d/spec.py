"""Specimens (plate in tension, DCB), fine LPM references and coarse models with any bond law; metrics.

Plate (as I2): n x n pixels, whole-block grips (tb_ty), fixed load steps deps = 5e-6 + goto, 1 event per pass.
DCB: nx x ny = 96 x 48 pixels, pre-crack on the mid-plane y = ny/2 for x < A0 = 24 (bonds crossing it removed, fine
     and coarse alike), the left block column of each arm moved +-lam/2 (rigid, x fixed); exact event-driven loading
     (deps -> 0 limit of the same loop); stopped when the crack (any broken bond) reaches x >= X_STOP = 72.
"""
import math
import numpy as np
import lin2d
import brittle
import soft

XI_M, XI_I, XI_INT = 1.0e-3, 2.0e-3, 0.6e-3
A0, X_STOP = 24.0, 72.0


def fine_model(Cpix, phase, nx, ny):
    lat = lin2d.Lattice2D("square", nx, ny, 1.0)
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cpix))
    crit = brittle.crit_from_phase(lat, phase.astype(np.int64), XI_M, XI_I, XI_INT)
    return lat, Kn, Tv, crit


def precrack_mask(lat, ycut, a0):
    """undirected bonds (soft.undirected order) crossing y = ycut with midpoint x < a0."""
    P, K, Q, KR = soft.undirected(lat)
    y1, y2 = lat.x[P, 1], lat.x[Q, 1]
    xm = 0.5 * (lat.x[P, 0] + lat.x[Q, 0])
    return ((y1 - ycut) * (y2 - ycut) < 0) & (xm < a0)


def crack_stop(lat, P, Q, dead0, tip0, adv_stop):
    """stop when the MAIN crack (broken bonds connected to the pre-crack) has advanced adv_stop from tip0."""
    tr = soft.CrackTracker(lat, P, Q, dead0, tip0)
    seen = [0]

    def f(events, wb):
        for e in events[seen[0]:]:
            b = e[1]
            if wb[b] == 0:
                tr.add(b)
        seen[0] = len(events)
        return tr.advance() >= adv_stop
    return f


def frame_mask(lat, P, Q, nx, ny, width):
    """bonds with an end within `width` (fine units) of the specimen boundary: elastic frame (cannot break)."""
    x = lat.x
    near = (x[:, 0] < width) | (x[:, 0] > nx - width) | (x[:, 1] < width) | (x[:, 1] > ny - width)
    return near[P] | near[Q]


def kf_elastic(Cpix):
    """effective isotropic mu, kappa (plane strain) of a specimen for the K-field shape (Voigt mean)."""
    C11, C12, C44 = Cpix[:, 0].mean(), Cpix[:, 1].mean(), Cpix[:, 2].mean()
    mu = C44
    nu = C12 / (2 * (C12 + mu))
    return mu, 3 - 4 * nu


KF_GROW = 24.0       # stop when the main crack has advanced 24 h
KF_LAMCAP = 0.02     # cap of the applied K (about 10 x the matrix K_c): beyond it the crack counts as arrested
KF_MAXEV = 1500
KF_FRAME = 8.0       # elastic frame next to the loaded ring (no bond breaks there, fine and coarse alike)


def run_test(test, lat, Kn, Tv, law, s, nx, ny, deps=5e-6, stop_frac=0.02, max_steps=600, mu=None, kappa=None,
             lam_max=np.inf):
    """test 'plate' or 'dcb'.  lat may be fine (s = block size for the grips) or coarse (pass s=1)."""
    if test == "plate":
        return soft.run(lat, Kn, Tv, law, soft.bc_plate(lat, s), mode="step", deps=deps, stop_frac=stop_frac,
                        max_steps=max_steps)
    P, K, Q, KR = soft.undirected(lat)
    if test == "dcb":
        dead0 = precrack_mask(lat, ny / 2.0, A0)
        ncy_half = (lat.ny // s) // 2
        bc = soft.bc_dcb(lat, s, ncy_half)
        return soft.run(lat, Kn, Tv, law, bc, mode="exact", stop_frac=0.0, dead0=dead0,
                        stop_fn=crack_stop(lat, P, Q, dead0, (A0, ny / 2.0), X_STOP - A0), lam_max=1.0)
    # K-field: pre-crack from the left edge to the centre, crack grows to the right until x >= KF_STOP
    tip = (nx / 2.0, ny / 2.0)
    dead0 = precrack_mask(lat, tip[1], tip[0])
    bc = soft.bc_kfield(lat, s, tip, mu, kappa)
    return soft.run(lat, Kn, Tv, law, bc, mode="exact", stop_frac=0.0, dead0=dead0,
                    stop_fn=crack_stop(lat, P, Q, dead0, tip, KF_GROW), lam_max=min(lam_max, KF_LAMCAP),
                    max_events=KF_MAXEV, unbreakable=frame_mask(lat, P, Q, nx, ny, KF_FRAME))


# --------------------------------------------------------------------------------------------------------------
# curves and metrics
# --------------------------------------------------------------------------------------------------------------
def curve_state(r):
    """(lam_k, F_k after all events at lam_k) for the plate (step) or DCB (exact) history: between two recorded
    equilibria with the same state the response is linear through the origin (secant)."""
    lam, F = r["lam"], r["F"]
    return lam, F


def F_at(r, grid):
    """Force on a grid of load values, using the secant stiffness of the last state reached at or before lam."""
    lam, F = r["lam"], r["F"]
    # last record per distinct lam (= after all events at that lam)
    out = np.zeros_like(grid)
    if len(lam) == 0:
        return out
    k_after = F / np.maximum(lam, 1e-300)
    j = np.searchsorted(lam, grid, side="right") - 1
    ok = j >= 0
    # before the first record: intact stiffness of the first record
    out[~ok] = k_after[0] * grid[~ok]
    out[ok] = k_after[j[ok]] * grid[ok]
    # beyond the end of the run: no information -> keep the last state
    return out


def energy_at(r, lam_t):
    """dissipated energy D(lam_t) = W - U at load lam_t (in lam-F units of the run, F unscaled by Fscale)."""
    lam, F = r["lam"], r["F"]
    W = 0.0
    lp, kp = 0.0, None
    # reconstruct W along the path: linear secant segments between records, vertical drops at fixed lam
    for l, f in zip(lam, F):
        if l > lam_t:
            break
        if kp is None:
            kp = f / l
        if l > lp:
            W += 0.5 * kp * (l * l - lp * lp)
        kp = f / l if l > 0 else kp
        lp = l
    if kp is None:
        return 0.0, 0.0
    if lam_t > lp:
        W += 0.5 * kp * (lam_t ** 2 - lp ** 2)
    U = 0.5 * kp * lam_t ** 2
    return W - U, W


def path_metrics(pf, pc, cell):
    """Fine (pf) vs coarse (pc) broken-bond midpoints, both mapped to coarse cell units: symmetric Hausdorff (max),
    95% Hausdorff, chamfer (mean)."""
    if len(pf) == 0 or len(pc) == 0:
        return dict(haus=np.nan, haus95=np.nan, chamfer=np.nan)
    from scipy.spatial import cKDTree
    a, b = pf / cell, pc / cell
    d1, _ = cKDTree(b).query(a)
    d2, _ = cKDTree(a).query(b)
    return dict(haus=float(max(d1.max(), d2.max())), haus95=float(max(np.quantile(d1, .95), np.quantile(d2, .95))),
                chamfer=float(0.5 * (d1.mean() + d2.mean())))


def summary(r, lat, test):
    """scalars of one run: initiation (first event) load/force, peak, dissipated energy, broken midpoints."""
    pm = soft.broken_midpoints(lat, r)
    lam, F = r["lam"], r["F"]
    ipk = int(np.argmax(F)) if len(F) else 0
    out = dict(lam1=r["eps1"], F1=(r["first"][1] * (1.0 if test == "dcb" else 1.0)) if r["first"] else np.nan,
               peak=float(r["peak"]), lam_pk=float(lam[ipk]) if len(lam) else np.nan, D=float(r["D"]), W=float(r["W"]),
               Dm=float(r["Dm"]), Dr=float(r["Dr"]), passes=int(r["passes"]), nbroken=int(r["broken"].sum()),
               lam_end=float(lam[-1]) if len(lam) else np.nan)
    return out, pm
