"""Geometry, load cases and parameter maps of the 3D SC super-particle study (numpy only: used by the numba code
(lin3d, numerical environment) and by the torch code (torch environment)).

Fine model: one SC particle per voxel (spacing h = 2r).  Coarse model: one particle per s x s x s block (spacing s h),
placed at the block centre; same lattice, same two shells, per-bond Kn/Tv.  Boundary conditions are always applied on
whole blocks, so the fine model of a given s sees exactly the loading of the coarse model (I2 design).

Load cases (strain magnitude EPS, linear problems), BC sets in brackets:
  k_xx k_yy k_zz k_yz k_xz k_xy  [ring]  outer layer of blocks prescribed affine u = eps.(x - x_c)  (KUBC, RVE)
  tb_tz tb_sx                    [tb]    bottom block layer (z) clamped, top layer translated (0,0,d) / (d,0,0)
  lr_tx                          [lr]    left block layer (x) clamped, right layer translated (d,0,0)
The plates have four free lateral faces (the LPM free surface, no surface correction).
"""
import numpy as np

SC_OFFS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)], np.int64)
SC_SHELL = np.array([0] * 6 + [1] * 12, np.int64)
NSLOT = 18
DIR_OF_SLOT = np.arange(18) // 2          # 9 undirected direction classes: x y z xy+ xy- xz+ xz- yz+ yz-
FWD_SLOTS = np.arange(0, 18, 2)

EPS = 1e-3
LOADS = ["k_xx", "k_yy", "k_zz", "k_yz", "k_xz", "k_xy", "tb_tz", "tb_sx", "lr_tx"]
RING = LOADS[:6]
PLATES = LOADS[6:]
BCSET = {ld: "ring" for ld in RING}
BCSET.update(tb_tz="tb", tb_sx="tb", lr_tx="lr")
GDIR = {"tb_tz": 2, "tb_sx": 0, "lr_tx": 0}          # reaction component of the plate cases
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]      # xx yy zz yz xz xy (code / Voigt order)


class LatticeGeom:
    """Regular SC lattice of nx x ny x nz sites, spacing h, optional active mask (full-grid order)."""

    def __init__(self, nx, ny, nz, h=1.0, origin=(0.0, 0.0, 0.0), active=None):
        self.nx, self.ny, self.nz, self.h = nx, ny, nz, h
        kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
        nsite = nx * ny * nz
        active = np.ones(nsite, bool) if active is None else np.asarray(active, bool).ravel()
        self.site_of = -np.ones(nsite, np.int64)
        self.site_of[active] = np.arange(int(active.sum()))
        self.sites = np.nonzero(active)[0]
        ii, jj, kk = ii[active], jj[active], kk[active]
        self.ijk = np.ascontiguousarray(np.stack([ii, jj, kk], 1))
        self.x = (self.ijk + 0.5) * h + np.asarray(origin, float)[None, :]
        self.N = len(ii)
        self.V = h ** 3
        self.offs, self.shell, self.NN = SC_OFFS, SC_SHELL, NSLOT
        self.opp = np.arange(NSLOT) ^ 1
        L = np.linalg.norm(SC_OFFS, axis=1)
        self.l0 = L * h
        self.ndir = np.ascontiguousarray(-SC_OFFS / L[:, None])
        ni = ii[:, None] + SC_OFFS[None, :, 0]
        nj = jj[:, None] + SC_OFFS[None, :, 1]
        nk = kk[:, None] + SC_OFFS[None, :, 2]
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny) & (nk >= 0) & (nk < nz)
        q = self.site_of[np.where(ok, (nk * ny + nj) * nx + ni, 0)]
        ok &= q >= 0
        self.nbr = np.ascontiguousarray(np.where(ok, q, -1).astype(np.int64))
        self.valid = ok
        self.nb_initial = ok.sum(1)


# ---------------------------------------------------------------------------------------------------------------------
# load cases
# ---------------------------------------------------------------------------------------------------------------------
def unit_strain(k):
    """Tensor strain of Voigt component k with engineering shear (gamma = 1 -> eps_ij = 1/2)."""
    e = np.zeros((3, 3))
    a, b = VOIGT_PAIRS[k]
    if a == b:
        e[a, a] = 1.0
    else:
        e[a, b] = e[b, a] = 0.5
    return e


def affine_disp(x, eps, x0=(0.0, 0.0, 0.0)):
    return (x - np.asarray(x0)[None, :]) @ np.asarray(eps, float).T


def prescribed(load, x, IJK, nc, s, h=1.0):
    """(fixed (N,) bool, prescribed u (N,3), moving grip (N,) bool or None).  IJK: block index of every particle,
    nc = (ncx, ncy, ncz) blocks."""
    ncx, ncy, ncz = nc
    I, J, K = IJK[:, 0], IJK[:, 1], IJK[:, 2]
    u = np.zeros((len(x), 3))
    if load in RING:
        fixed = (I < 1) | (I >= ncx - 1) | (J < 1) | (J >= ncy - 1) | (K < 1) | (K >= ncz - 1)
        xc = 0.5 * np.array([ncx, ncy, ncz]) * s * h
        u[fixed] = affine_disp(x[fixed], EPS * unit_strain(RING.index(load)), xc)
        return fixed, u, None
    if load.startswith("tb"):
        a, b = K < 1, K >= ncz - 1
        dist = (ncz - 1) * s * h
        d = np.array([0.0, 0.0, EPS * dist]) if load == "tb_tz" else np.array([EPS * dist, 0.0, 0.0])
    else:
        a, b = I < 1, I >= ncx - 1
        dist = (ncx - 1) * s * h
        d = np.array([EPS * dist, 0.0, 0.0])
    u[b] = d
    return a | b, u, b


def free_surface_cells(load, nc):
    """Free cells of a plate case that touch a free lateral face (the LPM free surface)."""
    ncx, ncy, ncz = nc
    kk, jj, ii = np.meshgrid(np.arange(ncz), np.arange(ncy), np.arange(ncx), indexing="ij")
    I, J, K = ii.ravel(), jj.ravel(), kk.ravel()
    if load.startswith("tb"):
        grip = (K < 1) | (K >= ncz - 1)
        face = (I < 1) | (I >= ncx - 1) | (J < 1) | (J >= ncy - 1)
    else:
        grip = (I < 1) | (I >= ncx - 1)
        face = (J < 1) | (J >= ncy - 1) | (K < 1) | (K >= ncz - 1)
    return face & ~grip


def cell_ijk(nc):
    ncx, ncy, ncz = nc
    kk, jj, ii = np.meshgrid(np.arange(ncz), np.arange(ncy), np.arange(ncx), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)


def block_index(ijk, s, nc):
    B = ijk // s
    return (B[:, 2] * nc[1] + B[:, 1]) * nc[0] + B[:, 0]


def interface_cells(phase, n, s):
    """Cells whose block contains a voxel with a 6-neighbour of the other phase."""
    p = phase.reshape(n, n, n)
    b = np.zeros_like(p, bool)
    for ax in range(3):
        d = np.diff(p, axis=ax) != 0
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(1, None)
        sl1[ax] = slice(None, -1)
        b[tuple(sl0)] |= d
        b[tuple(sl1)] |= d
    nc = n // s
    return b.reshape(nc, s, nc, s, nc, s).any(axis=(1, 3, 5)).ravel()


# ---------------------------------------------------------------------------------------------------------------------
# materials: isotropic C -> SC parameters (calcKnTv lattice 2, mapping {1,-1,-1; 0,0,1; 0,1/18,-1/18} * radius)
# ---------------------------------------------------------------------------------------------------------------------
def iso_C(E, nu):
    E = np.asarray(E, float)
    nu = np.asarray(nu, float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.stack([lam + 2 * mu, lam, mu], -1)          # C11, C12, C44


def sc_params_from_C(C, r):
    """(Kn1, Kn2, Tv) = r (C11 - C12 - C44, C44, (C12 - C44)/18); r = h/2."""
    C11, C12, C44 = C[..., 0], C[..., 1], C[..., 2]
    return np.stack([r * (C11 - C12 - C44), r * C44, r * (C12 - C44) / 18.0], -1)


def bond_params_two_end_mean(lat, ppar):
    """ppar (N, 3) = (Kn1, Kn2, Tv) per particle -> Kn, Tv (N, 18): mean of the two ends (LPM-C SC calcKnTv)."""
    q = np.where(lat.valid, lat.nbr, 0)
    sh = SC_SHELL[None, :]
    kn_i = np.where(sh == 0, ppar[:, None, 0], ppar[:, None, 1])
    kn_j = np.where(sh == 0, ppar[q, 0], ppar[q, 1])
    Kn = 0.5 * (kn_i + kn_j) * lat.valid
    Tv = 0.5 * (ppar[:, None, 2] + ppar[q, 2]) * lat.valid
    return np.ascontiguousarray(Kn), np.ascontiguousarray(Tv)


def C21_to_mat(C):
    """(..., 6, 6) Voigt (engineering shear) matrices are used directly; helper kept for clarity."""
    return C


def cell_params_from_C66(C, r):
    """Generalised SC calcKnTv for an anisotropic cell stiffness C (Ncell, 6, 6) (Voigt, engineering shear):
    per-direction Kn (x, y, z, xy+, xy-, xz+, xz-, yz+, yz-) and one Tv (both shells), r = cell spacing / 2.
    Representable by the SC bond set: C12 - C66 = C13 - C55 = C23 - C44 = T (= 18 Tv / r), C16 = C26, C15 = C35,
    C24 = C34, other couplings 0 (derived from C V = sum_b Kn_b l^2 n^4 + sum_c Tv (sum l n n)(sum l n n)); the rest
    of C is projected out:
      T = mean(C12 - C66, C13 - C55, C23 - C44),   Tv = r T / 18
      Kn_xy+- = r (C66 +- (C16 + C26)/2),  Kn_xz+- = r (C55 +- (C15 + C35)/2),  Kn_yz+- = r (C44 +- (C24 + C34)/2)
      Kn_x = r (C11 - C55 - C66 - T),  Kn_y = r (C22 - C44 - C66 - T),  Kn_z = r (C33 - C44 - C55 - T)
    For an isotropic C this is calcKnTv lattice 2 exactly (Kn1 = Kn2 = r mu, Tv = r (lam - mu)/18)."""
    c = lambda a, b: C[:, a - 1, b - 1]
    T = (c(1, 2) - c(6, 6) + c(1, 3) - c(5, 5) + c(2, 3) - c(4, 4)) / 3.0
    out = np.stack([r * (c(1, 1) - c(5, 5) - c(6, 6) - T), r * (c(2, 2) - c(4, 4) - c(6, 6) - T),
                    r * (c(3, 3) - c(4, 4) - c(5, 5) - T),
                    r * (c(6, 6) + 0.5 * (c(1, 6) + c(2, 6))), r * (c(6, 6) - 0.5 * (c(1, 6) + c(2, 6))),
                    r * (c(5, 5) + 0.5 * (c(1, 5) + c(3, 5))), r * (c(5, 5) - 0.5 * (c(1, 5) + c(3, 5))),
                    r * (c(4, 4) + 0.5 * (c(2, 4) + c(3, 4))), r * (c(4, 4) - 0.5 * (c(2, 4) + c(3, 4))),
                    r * T / 18.0], 1)
    return out


def iso_C66(C3):
    """(N, 3) (C11, C12, C44) -> (N, 6, 6)."""
    N = len(C3)
    C = np.zeros((N, 6, 6))
    C[:, :3, :3] = C3[:, 1][:, None, None]
    C[:, [0, 1, 2], [0, 1, 2]] = C3[:, 0][:, None]
    C[:, [3, 4, 5], [3, 4, 5]] = C3[:, 2][:, None]
    return C


def bond_params_from_cells(lat_c, cp):
    """Two-end mean of per-cell parameters cp (Nc, 10: 9 direction Kn + Tv) -> Kn, Tv (Nc, 18)."""
    q = np.where(lat_c.valid, lat_c.nbr, 0)
    col = DIR_OF_SLOT[None, :].repeat(lat_c.N, 0)
    Kn = 0.5 * (cp[:, DIR_OF_SLOT] + cp[q, col]) * lat_c.valid
    Tv = 0.5 * (cp[:, 9][:, None] + cp[q, 9]) * lat_c.valid
    return np.ascontiguousarray(Kn), np.ascontiguousarray(Tv)


def voigt_cells(Cvox, n, s):
    """Volume average of the voxel (C11, C12, C44) over every block -> (Nc, 6, 6)."""
    nc = n // s
    C = Cvox.reshape(nc, s, nc, s, nc, s, 3).mean(axis=(1, 3, 5)).reshape(-1, 3)
    return iso_C66(C)
