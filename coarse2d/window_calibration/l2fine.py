"""L2 (non-ML per-bond local calibration), part 1: local FINE solves in a window around every coarse bond.

Window of coarse bond b = (cells A, B) with margin m (cells): the two end cells plus m cells all round
  axial  bond: (1+2m) x (2+2m) cells,  diagonal bond: (2+2m) x (2+2m) cells.
m = 1 is exactly the footprint read by the I2 network (windows.extract); m = 2, 3 are the "+1", "+2 cells" variants.
Fine model on the window = the actual microstructure of the specimen inside the box (pixels outside the specimen are
absent, so windows at the free surface contain the free surface).  BC: the outer ring of cells (1 cell = s pixel layers,
>= 2 layers so the multibody term of the free pixels is complete) is prescribed with an affine displacement
u = eps.(x - x_c) (x_c = box centre) for 3 strain states (exx, eyy, gxy); interior pixels relaxed.
All windows of one orientation group are solved at once as disjoint sub-lattices of one big lattice (one SuperLU
factorisation, 3 right-hand sides): the same solver family as the I2 patch homogenisation (cg.patch_cells).
Output per window and box cell (out-of-specimen cells: exists = False): block-mean displacement (2), block-mean
stress (3, computeStress), block energy density (sum of particle energies / (s h)^2), for the 3 states.
"""
import numpy as np
import scipy.sparse.linalg as spla
import common  # noqa
import lin2d
import windows as WIN

EPS = 1e-3
STATES = [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1.0]]), np.array([[0, 0.5], [0.5, 0]])]
KS = (0, 2, 4, 6)
DIJ = {0: (1, 0), 2: (0, 1), 4: (1, 1), 6: (1, -1)}


def n_states(mode):
    return 3 if mode == "aff" else 9


def state_disp(e, x, mode, Lh, s=None, h=1.0):
    """displacement of load state e at local points x (n, 2) (w.r.t. the box centre); with s: block mean over the
    s x s pixels of a cell centred at x.  States 0-2: affine (exx, eyy, gxy); 3-8 ('quad'): u_x or u_y = c x^2, c y^2,
    c x y with c = EPS / (2 Lh) (strain of order EPS at the window boundary; linear strain gradients)."""
    if e < 3:
        return x @ (EPS * STATES[e]).T
    c = EPS / (2.0 * Lh)
    comp, kind = (e - 3) // 3, (e - 3) % 3
    var = 0.0 if s is None else (s * s - 1) / 12.0 * h * h
    X, Y = x[:, 0], x[:, 1]
    u = np.zeros_like(x, dtype=float)
    u[:, comp] = c * [X * X + var, Y * Y + var, X * Y][kind]
    return u


def box_dims(k, m):
    """(W, H) of the window in cells (columns, rows)."""
    if k == 0:
        return 2 + 2 * m, 1 + 2 * m
    if k == 2:
        return 1 + 2 * m, 2 + 2 * m
    return 2 + 2 * m, 2 + 2 * m


def box_origin(bk, m):
    """lower-left cell (I_lo, J_lo) of the windows of forward bonds bk = [p, k, I, J]."""
    I, J, k = bk[:, 2], bk[:, 3], bk[:, 1]
    return I - m, J - m - (k == 6).astype(np.int64)


def bond_groups(nc):
    b, _ = WIN.bond_list(nc, nc)
    return {k: b[b[:, 1] == k] for k in KS}


def fine_group(Cpix, n, s, k, bk, m, h=1.0, mode="aff"):
    """Local fine solves for all windows of orientation group k.  Returns dict of float32 arrays
    u (3, nw, ncell, 2), sig (3, nw, ncell, 3), w (3, nw, ncell), exists (nw, ncell); cells row-major in the box."""
    Wc, Hc = box_dims(k, m)
    Wp, Hp = Wc * s, Hc * s
    Ilo, Jlo = box_origin(bk, m)
    nw = len(bk)
    gap = 2
    SW, SH = Wp + gap, Hp + gap
    ncol = int(np.ceil(np.sqrt(nw)))
    nrow = int(np.ceil(nw / ncol))
    GX, GY = ncol * SW, nrow * SH
    lj, li = np.mgrid[0:Hp, 0:Wp]
    li, lj = li.ravel(), lj.ravel()
    gi = Ilo[:, None] * s + li[None, :]
    gj = Jlo[:, None] * s + lj[None, :]
    inside = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
    wcol, wrow = np.arange(nw) % ncol, np.arange(nw) // ncol
    site = (wrow[:, None] * SH + lj[None, :]) * GX + wcol[:, None] * SW + li[None, :]
    act = np.zeros(GX * GY, bool)
    act[site[inside]] = True
    lat = lin2d.Lattice2D("square", GX, GY, h, active=act)
    pid = lat.site_of[site]
    Cp = np.zeros((lat.N, 3))
    Cp[pid[inside]] = Cpix[(gj * n + gi)[inside]]
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cp))
    K, _, _ = lat.stiffness(Kn, Tv)
    ci, cj = li // s, lj // s
    ring = (ci == 0) | (ci == Wc - 1) | (cj == 0) | (cj == Hc - 1)
    xl = np.stack([li + 0.5 - Wp / 2, lj + 0.5 - Hp / 2], 1) * h            # local coords w.r.t. box centre
    fixed_mask = np.zeros(lat.N, bool)
    fixed_mask[pid[inside & ring[None, :]]] = True
    fdofs = np.concatenate([2 * np.nonzero(fixed_mask)[0], 2 * np.nonzero(fixed_mask)[0] + 1])
    free = np.ones(2 * lat.N, bool)
    free[fdofs] = False
    Kc = K.tocsr()
    Kff = Kc[free][:, free].tocsc()
    Kfp = Kc[free][:, fdofs]
    lu = spla.splu(Kff, permc_spec="COLAMD")
    # position of every particle in its window frame
    xloc = np.zeros((lat.N, 2))
    xloc[pid[inside]] = np.broadcast_to(xl[None], (nw, len(li), 2))[inside]
    cell = (np.arange(nw)[:, None] * (Hc * Wc) + (cj * Wc + ci)[None, :])
    cid = np.full(lat.N, -1, np.int64)
    cid[pid[inside]] = cell[inside]
    ncl = nw * Hc * Wc
    cnt = np.bincount(cid, minlength=ncl).astype(float)
    exists = cnt > 0
    NS = n_states(mode)
    Lh = max(Wp, Hp) / 2.0 * h
    U = np.zeros((NS, ncl, 2), np.float32)
    SG = np.zeros((NS, ncl, 3), np.float32)
    WE = np.zeros((NS, ncl), np.float32)
    for e in range(NS):
        ua = state_disp(e, xloc, mode, Lh)
        u = np.zeros(2 * lat.N)
        u[fdofs] = np.concatenate([ua[fixed_mask, 0], ua[fixed_mask, 1]])
        u[free] = lu.solve(-(Kfp @ u[fdofs]))
        uu = u.reshape(-1, 2)
        sg = lat.stress(u, Kn, Tv)
        en = lat.energy_per_particle(u, Kn, Tv)
        c = np.maximum(cnt, 1)
        for d in range(2):
            U[e, :, d] = np.bincount(cid, uu[:, d], minlength=ncl) / c
        for d in range(3):
            SG[e, :, d] = np.bincount(cid, sg[:, d], minlength=ncl) / c
        WE[e] = np.bincount(cid, en, minlength=ncl) / (s * h) ** 2
    U[:, ~exists] = np.nan
    SG[:, ~exists] = np.nan
    WE[:, ~exists] = np.nan
    sh = (nw, Hc * Wc)
    return dict(u=U.reshape(NS, *sh, 2), sig=SG.reshape(NS, *sh, 3), w=WE.reshape(NS, *sh),
                exists=exists.reshape(sh), dofs=int(free.sum()))


def fine_targets(Cpix, n, s, m, mode="aff"):
    nc = n // s
    G = bond_groups(nc)
    out = {}
    for k in KS:
        r = fine_group(Cpix, n, s, k, G[k], m, mode=mode)
        for key, v in r.items():
            out["%s_%d" % (key, k)] = v
    return out
