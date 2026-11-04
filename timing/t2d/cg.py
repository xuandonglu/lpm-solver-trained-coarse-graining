"""Coarse-graining framework for the square lattice (LPM-C lattice 0).

Fine lattice: one particle per pixel (nx x ny, spacing h).  Coarse lattice: one particle per s x s block,
spacing s h, particle at the block centre.  Boundary conditions are always prescribed on whole blocks, so the
fine and the coarse model see the same loading (a coarse particle in a grip/ring block gets the value of the
affine/rigid field at the block centre = block mean of the fine values).

Load cases (strain magnitude EPS, linear problems):
  ring_xx, ring_yy, ring_xy : outer ring of blocks prescribed affine u = eps.(x - x_c)          (RVE, KUBC)
  tb_ty, tb_sx              : bottom block row clamped, top block row translated (0, d) / (d, 0) (plate)
  lr_tx                     : left block column clamped, right block column translated (d, 0)   (plate)
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import lin2d

EPS = 1e-3
LOADS = ["ring_xx", "ring_yy", "ring_xy", "tb_ty", "tb_sx", "lr_tx"]
BCSET = {"ring_xx": "ring", "ring_yy": "ring", "ring_xy": "ring", "tb_ty": "tb", "tb_sx": "tb", "lr_tx": "lr"}


def block_ij(lat, s):
    return lat.ij // s


BC_LAYERS = 1      # block layers of the grips / KUBC ring (hex lattice needs 2: its 2nd-shell bonds reach 2 rows)


def bc_masks(IJ, ncx, ncy):
    I, J = IJ[:, 0], IJ[:, 1]
    L = BC_LAYERS
    ring = (I < L) | (I >= ncx - L) | (J < L) | (J >= ncy - L)
    return {"ring": (ring, None),
            "tb": (J < L, J >= ncy - L),
            "lr": (I < L, I >= ncx - L)}


def prescribed(load, x, IJ, ncx, ncy, s, h):
    """(fixed particle mask, prescribed displacement (N,2)) for a load case."""
    Lx, Ly = ncx * s * h, ncy * s * h
    xc = np.array([Lx / 2, Ly / 2])
    m = bc_masks(IJ, ncx, ncy)
    u = np.zeros((len(x), 2))
    if load.startswith("ring"):
        e = {"ring_xx": [[1, 0], [0, 0]], "ring_yy": [[0, 0], [0, 1]], "ring_xy": [[0, .5], [.5, 0]]}[load]
        fixed = m["ring"][0]
        u[fixed] = lin2d.affine_disp(x[fixed], EPS * np.array(e), xc)
        return fixed, u, None
    if load.startswith("tb"):
        a, b = m["tb"]
        dist = (ncy - BC_LAYERS) * s * h
        d = np.array([0.0, EPS * dist]) if load == "tb_ty" else np.array([EPS * dist, 0.0])
    else:
        a, b = m["lr"]
        dist = (ncx - BC_LAYERS) * s * h
        d = np.array([EPS * dist, 0.0])
    fixed = a | b
    u[b] = d
    return fixed, u, b


def solve_cases(lat, K, IJ, ncx, ncy, s, h, loads=LOADS):
    """Solve all load cases; factorisation reused per BC set. Returns dict load -> u (N,2)."""
    out = {}
    lu = {}
    for ld in loads:
        fixed, up, moving = prescribed(ld, lat.x, IJ, ncx, ncy, s, h)
        bcs = BCSET[ld]
        fdofs = np.concatenate([2 * np.nonzero(fixed)[0], 2 * np.nonzero(fixed)[0] + 1])
        uf = np.concatenate([up[fixed, 0], up[fixed, 1]])
        u, fac = lin2d.solve_disp(K, fdofs, uf, lu_cache=lu.get(bcs))
        lu[bcs] = fac
        out[ld] = u.reshape(-1, 2)
    return out


def block_mean(f, IJ, ncx, ncy):
    """Mean of a per-fine-particle field over each block (blocks in coarse row-major order)."""
    b = IJ[:, 1] * ncx + IJ[:, 0]
    cnt = np.bincount(b, minlength=ncx * ncy)
    if f.ndim == 1:
        return np.bincount(b, f, minlength=ncx * ncy) / np.maximum(cnt, 1)
    return np.stack([np.bincount(b, f[:, k], minlength=ncx * ncy) / np.maximum(cnt, 1) for k in range(f.shape[1])], 1)


def block_sum(f, IJ, ncx, ncy):
    b = IJ[:, 1] * ncx + IJ[:, 0]
    return np.bincount(b, f, minlength=ncx * ncy)


def interface_cells(phase2d, s):
    """Cells whose block contains a pixel that has a 4-neighbour of another phase (or grain)."""
    p = phase2d
    b = np.zeros_like(p, bool)
    b[:, 1:] |= p[:, 1:] != p[:, :-1]
    b[:, :-1] |= p[:, 1:] != p[:, :-1]
    b[1:, :] |= p[1:, :] != p[:-1, :]
    b[:-1, :] |= p[1:, :] != p[:-1, :]
    ny, nx = p.shape
    return b.reshape(ny // s, s, nx // s, s).any(axis=(1, 3)).ravel()


class FineRef:
    """Fine LPM reference (lin2d, verified against the port) for one microstructure and coarse factor s."""

    def __init__(self, Cpix, nx, ny, s, h=1.0, loads=LOADS, phase=None, lat=None, Kn=None, Tv=None):
        """Square lattice from pixel stiffness Cpix, or (lat, Kn, Tv) given."""
        self.nx, self.ny, self.s, self.h = nx, ny, s, h
        self.ncx, self.ncy = nx // s, ny // s
        self.Cpix = Cpix
        if lat is None:
            self.lat = lin2d.Lattice2D("square", nx, ny, h)
            ppar = lin2d.square_params_from_C(Cpix)
            self.Kn, self.Tv = lin2d.bond_params_two_end_mean(self.lat, ppar)
        else:
            self.lat, self.Kn, self.Tv = lat, Kn, Tv
        self.K, _, _ = self.lat.stiffness(self.Kn, self.Tv)
        self.IJ = block_ij(self.lat, s)
        self.U = solve_cases(self.lat, self.K, self.IJ, self.ncx, self.ncy, s, h, loads)
        self.loads = loads
        self.res = {}
        for ld, u in self.U.items():
            sig = self.lat.stress(u.ravel(), self.Kn, self.Tv)
            Ui = self.lat.energy_per_particle(u.ravel(), self.Kn, self.Tv)
            fixed, _, moving = prescribed(ld, self.lat.x, self.IJ, self.ncx, self.ncy, s, h)
            P = (self.K @ u.ravel()).reshape(-1, 2)
            self.res[ld] = dict(
                u=block_mean(u, self.IJ, self.ncx, self.ncy),
                sig=block_mean(sig, self.IJ, self.ncx, self.ncy),
                w=block_sum(Ui, self.IJ, self.ncx, self.ncy) / (s * h) ** 2,
                W=Ui.sum(),
                F=None if moving is None else P[moving].sum(0),
                wmax=Ui.max() / h ** 2)
        if phase is not None:
            self.iface = interface_cells(phase.reshape(ny, nx), s)


class CoarseModel:
    def __init__(self, ncx, ncy, s, h=1.0, kind="square", origin=(0.0, 0.0)):
        self.ncx, self.ncy, self.s, self.h = ncx, ncy, s, h
        self.lat = lin2d.Lattice2D(kind, ncx, ncy, s * h, origin=origin)
        self.IJ = self.lat.ij.copy()

    def run(self, Kn, Tv, loads=LOADS):
        K, _, _ = self.lat.stiffness(Kn, Tv)
        U = solve_cases(self.lat, K, self.IJ, self.ncx, self.ncy, self.s, self.h, loads)
        res = {}
        for ld, u in U.items():
            sig = self.lat.stress(u.ravel(), Kn, Tv)
            Ui = self.lat.energy_per_particle(u.ravel(), Kn, Tv)
            fixed, _, moving = prescribed(ld, self.lat.x, self.IJ, self.ncx, self.ncy, self.s, self.h)
            P = (K @ u.ravel()).reshape(-1, 2)
            res[ld] = dict(u=u, sig=sig, w=Ui / (self.s * self.h) ** 2, W=Ui.sum(),
                           F=None if moving is None else P[moving].sum(0))
        return res


def metrics(ref, res, loads=None):
    """Errors of a coarse result against the block-averaged fine reference."""
    loads = loads or ref.loads
    m = {}
    for ld in loads:
        a, b = ref.res[ld], res[ld]
        fixed, _, _ = prescribed(ld, np.zeros((ref.ncx * ref.ncy, 2)), b_ij(ref), ref.ncx, ref.ncy, ref.s, ref.h)
        free = ~fixed
        e_W = (b["W"] - a["W"]) / a["W"]
        e_u = np.linalg.norm(b["u"][free] - a["u"][free]) / np.linalg.norm(a["u"][free])
        e_s = np.linalg.norm(b["sig"][free] - a["sig"][free]) / np.linalg.norm(a["sig"][free])
        d = dict(W=e_W, u=e_u, sig=e_s)
        if hasattr(ref, "iface"):
            fi = free & ref.iface
            if fi.any():
                d["sig_if"] = np.linalg.norm(b["sig"][fi] - a["sig"][fi]) / np.linalg.norm(a["sig"][fi])
                d["w_if"] = np.linalg.norm(b["w"][fi] - a["w"][fi]) / np.linalg.norm(a["w"][fi])
                # strain-energy concentration: 99th percentile of the cell energy density (interior cells)
                q_ref = np.quantile(a["w"][free], 0.99)
                q_c = np.quantile(b["w"][free], 0.99)
                d["w99"] = (q_c - q_ref) / q_ref
        if a["F"] is not None:
            k = 1 if ld == "tb_ty" else 0
            d["F"] = (b["F"][k] - a["F"][k]) / a["F"][k]
            d["G"] = d["F"]                       # global stiffness measure of the plate cases
        else:
            k = {"ring_xx": 0, "ring_yy": 1, "ring_xy": 2}[ld]
            sa, sb = a["sig"][free, k].mean(), b["sig"][free, k].mean()
            d["G"] = (sb - sa) / sa               # apparent modulus = interior mean stress (KUBC)
        m[ld] = d
    return m


def b_ij(ref):
    ii, jj = np.meshgrid(np.arange(ref.ncx), np.arange(ref.ncy))
    return np.stack([ii.ravel(), jj.ravel()], 1)


# -------------------------------------------------------------------------------------------------
# classical coarse parameters
# -------------------------------------------------------------------------------------------------
def cell_params_from_C6(C6):
    """Per-cell coarse parameters from an effective C = (C11, C22, C12, C66, C16, C26) (Voigt, tensor shear
    C66 = C1212): per-direction Kn (x, y, diag+, diag-) and one Tv (generalised calcKnTv lattice 0)."""
    C11, C22, C12, C66, C16, C26 = [C6[:, k] for k in range(6)]
    knx = (C11 - C12) / 2
    kny = (C22 - C12) / 2
    knp = C66 / 2 + (C16 + C26) / 4
    knm = C66 / 2 - (C16 + C26) / 4
    tv = (C12 - C66) / 12
    return np.stack([knx, kny, knp, knm, tv], 1)


# slot -> which per-cell column (square offsets: (1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))
SLOT_COL = np.array([0, 0, 1, 1, 2, 2, 3, 3])


def bond_params_from_cells(lat_c, cp):
    """Two-end mean of per-cell parameters (cp: (Nc, 5) = knx, kny, knp, knm, tv) -> Kn, Tv (Nc, 8)."""
    q = np.where(lat_c.valid, lat_c.nbr, 0)
    kn_i = cp[:, SLOT_COL]
    kn_j = cp[q, SLOT_COL[None, :].repeat(lat_c.N, 0)]
    Kn = 0.5 * (kn_i + kn_j) * lat_c.valid
    Tv = 0.5 * (cp[:, 4][:, None] + cp[q, 4]) * lat_c.valid
    return Kn, Tv


def C6_from_C3(C3):
    return np.stack([C3[:, 0], C3[:, 0], C3[:, 1], C3[:, 2], 0 * C3[:, 0], 0 * C3[:, 0]], 1)


def voigt_cells(Cpix, nx, ny, s):
    """Volume (Voigt) average of the pixel stiffness over each block -> C6 per cell."""
    C = Cpix.reshape(ny // s, s, nx // s, s, 3).mean(axis=(1, 3)).reshape(-1, 3)
    return C6_from_C3(C)


def patch_cells(Cpix, nx, ny, s, margin=None, h=1.0, ring=2):
    """Local patch homogenisation by fine LPM: for every cell, a (s + 2m)^2 window of the microstructure
    (reflect-padded at the domain boundary) with affine displacement on its outer `ring` pixel layers (2 layers:
    with 1 layer the multibody term couples the free particles to prescribed particles that miss bonds, and even a
    homogeneous patch comes out 1.5-3.4 % off; with 2 layers it is exact to 1e-15) and
    relaxed interior; C_ab = (1/A_cell) u_a^T K_cell u_b, K_cell = Hessian of sum_{i in cell} U_i.
    All windows are solved at once as disjoint sub-lattices of one big lattice (one sparse factorisation).
    Returns C6 per cell (C11, C22, C12, C66, C16, C26)."""
    m = s if margin is None else margin
    w = s + 2 * m
    ncx, ncy = nx // s, ny // s
    C2d = Cpix.reshape(ny, nx, 3)
    Cpad = np.pad(C2d, ((m, m), (m, m), (0, 0)), mode="reflect")
    gap = 2
    W = w + gap
    gx, gy = ncx * W, ncy * W
    act = np.zeros((gy, gx), bool)
    Cbig = np.zeros((gy, gx, 3))
    Cbig[:] = Cpix.mean(0)
    for J in range(ncy):
        for I in range(ncx):
            act[J * W:J * W + w, I * W:I * W + w] = True
            Cbig[J * W:J * W + w, I * W:I * W + w] = Cpad[J * s:J * s + w, I * s:I * s + w]
    lat = lin2d.Lattice2D("square", gx, gy, h, active=act)
    Cp = Cbig.reshape(-1, 3)[lat.sites]
    Kn, Tv = lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(Cp))
    K, B, H = lat.stiffness(Kn, Tv)
    li, lj = lat.ij[:, 0] % W, lat.ij[:, 1] % W
    wid = (lat.ij[:, 1] // W) * ncx + lat.ij[:, 0] // W
    outer = (li < ring) | (li >= w - ring) | (lj < ring) | (lj >= w - ring)
    incell = (li >= m) & (li < m + s) & (lj >= m) & (lj < m + s)
    # window-local coordinates relative to the cell centre
    xl = np.stack([li - (m + s / 2 - 0.5), lj - (m + s / 2 - 0.5)], 1) * h
    fixed = np.nonzero(outer)[0]
    fdofs = np.concatenate([2 * fixed, 2 * fixed + 1])
    free = np.ones(2 * lat.N, bool)
    free[fdofs] = False
    Kff = K[free][:, free].tocsc()
    lu = spla.splu(Kff, permc_spec="COLAMD")
    E = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, .5], [.5, 0]])]
    Us = []
    for e in E:
        ua = lin2d.affine_disp(xl, e)
        u = np.zeros(2 * lat.N)
        u[fdofs] = np.concatenate([ua[fixed, 0], ua[fixed, 1]])
        rhs = -K[free][:, fdofs] @ u[fdofs]
        u[free] = lu.solve(rhs)
        Us.append(u)
    # bilinear form of the cell energy: sum_{i in cell} U_i(u) is quadratic; use bond-space Hessian rows of
    # cell particles only: U_cell(u) = 1/2 (B u)^T H_cell (B u), H_cell = rows/cols of bonds owned by cell particles
    owner = np.repeat(np.arange(lat.N), lat.NN)
    sel = incell[owner].astype(float)
    Dsel = sp.diags(sel)
    Hc = Dsel @ H @ Dsel
    BU = [B @ u for u in Us]
    A = (s * h) ** 2
    Cmat = np.zeros((ncx * ncy, 3, 3))
    bw = wid[owner]
    for a in range(3):
        HBa = Hc @ BU[a]
        for b in range(a, 3):
            v = np.bincount(bw, HBa * BU[b], minlength=ncx * ncy) / A
            Cmat[:, a, b] = v
            Cmat[:, b, a] = v
    # Voigt with engineering shear for the third column: C_ab with eps3 = gamma -> C66(tensor) = Cmat[2,2]
    C11, C22, C12 = Cmat[:, 0, 0], Cmat[:, 1, 1], Cmat[:, 0, 1]
    C66 = Cmat[:, 2, 2]
    C16, C26 = Cmat[:, 0, 2], Cmat[:, 1, 2]
    return np.stack([C11, C22, C12, C66, C16, C26], 1)
