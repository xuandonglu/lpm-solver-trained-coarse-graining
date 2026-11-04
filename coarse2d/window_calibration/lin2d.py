"""Small-strain (linearised) 2D LPM on regular lattices, with per-bond Kn/Tv.

This is the LPM-C elastic law (constitutive.c computeBondForceElastic) linearised about the reference
configuration.  Energy of particle i:
    U_i = 1/2 sum_j Kn_ij dL_ij^2 + 1/2 sum_c (sum_{j in c} Tv_ij dL_ij)(sum_{j in c} dL_ij)
with dL_ij = n_ij . (u_i - u_j), n_ij = (x_i - x_j)/|x_i - x_j|.  For Kn_ij = Kn_ji, Tv_ij = Tv_ji the
force F_ij = 2 Kn dL + 1/2(TS_i + TS_j) + 1/2 Tv (S_i + S_j) of the code is dU/d(dL_ij), so the stiffness is
the exact Hessian K = B^T H B (B: bond incidence, H: bond-space Hessian).

Lattices:
  square (LPM-C lattice 0): pixel (i, j) -> particle at ((i+.5)h, (j+.5)h); shell 0 = 4 nearest (h),
                            shell 1 = 4 diagonal (sqrt2 h).   V = h^2 (unit thickness).
  hex    (LPM-C lattice 1): axial site (i, j) -> (h(i + j/2), h sqrt3/2 j); shell 0 = 6 at h,
                            shell 1 = 6 at sqrt3 h.          V = sqrt3/2 h^2.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SQ_OFFS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)])
SQ_SHELL = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# hex, axial offsets (di, dj); cartesian = (di + dj/2, sqrt3/2 dj)
HX_OFFS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1),          # shell 0 (0,180,60,240,120,300 deg)
                    (1, 1), (-1, -1), (-1, 2), (1, -2), (-2, 1), (2, -1)])      # shell 1 (30,210,90,270,150,330 deg)
HX_SHELL = np.array([0] * 6 + [1] * 6)


class Lattice2D:
    """Regular 2D lattice of nx x ny sites with spacing h (nearest-neighbour distance)."""

    def __init__(self, kind, nx, ny, h, origin=(0.0, 0.0), active=None):
        """active: optional (ny, nx) bool mask of occupied sites (holes, notches, several disjoint windows)."""
        self.kind, self.nx, self.ny, self.h = kind, nx, ny, h
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))          # (ny, nx), row-major p = j*nx + i
        ii, jj = ii.ravel(), jj.ravel()
        if active is None:
            active = np.ones(nx * ny, bool)
        active = np.asarray(active, bool).ravel()
        self.site_of = -np.ones(nx * ny, np.int64)                   # full-grid site -> particle id
        self.site_of[active] = np.arange(active.sum())
        self.sites = np.nonzero(active)[0]
        ii, jj = ii[active], jj[active]
        self.ij = np.stack([ii, jj], 1)
        if kind == "square":
            offs, shell = SQ_OFFS, SQ_SHELL
            x = (ii + 0.5) * h
            y = (jj + 0.5) * h
            self.V = h * h
            dirs = offs.astype(float)
        elif kind == "hex":
            offs, shell = HX_OFFS, HX_SHELL
            x = h * (ii + 0.5 * jj)
            y = h * (math.sqrt(3) / 2) * jj
            self.V = math.sqrt(3) / 2 * h * h
            dirs = np.stack([offs[:, 0] + 0.5 * offs[:, 1], math.sqrt(3) / 2 * offs[:, 1]], 1)
        else:
            raise ValueError(kind)
        self.x = np.stack([x + origin[0], y + origin[1]], 1)
        self.N = len(ii)
        self.offs, self.shell = offs, shell
        self.NN = len(offs)
        self.opp = np.array([k ^ 1 for k in range(self.NN)])         # opposite slot (pairs 0-1, 2-3, ...)
        L = np.linalg.norm(dirs, axis=1)
        self.l0 = L * h                                               # bond lengths per slot
        self.ndir = -dirs / L[:, None]                                # n_ij = (x_i - x_j)/|.|, j = i + off
        ni = ii[:, None] + offs[None, :, 0]
        nj = jj[:, None] + offs[None, :, 1]
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        q = self.site_of[np.where(ok, nj * nx + ni, 0)]
        ok = ok & (q >= 0)
        self.nbr = np.where(ok, q, -1)
        self.valid = ok
        self.nb_initial = ok.sum(1)

    # ------------------------------------------------------------------------------------------
    def bond_matrix(self, alive=None):
        """B (8N or 12N rows x 2N): dL[i,k] = n_k . (u_i - u_nbr)."""
        N, NN = self.N, self.NN
        act = self.valid if alive is None else (self.valid & alive)
        rows = np.arange(N * NN).reshape(N, NN)
        q = np.where(self.valid, self.nbr, 0)
        r_list, c_list, v_list = [], [], []
        for d in range(2):
            nd = np.broadcast_to(self.ndir[None, :, d], (N, NN))
            r_list += [rows[act], rows[act]]
            c_list += [2 * np.broadcast_to(np.arange(N)[:, None], (N, NN))[act] + d, 2 * q[act] + d]
            v_list += [nd[act], -nd[act]]
        B = sp.csr_matrix((np.concatenate(v_list), (np.concatenate(r_list), np.concatenate(c_list))),
                          shape=(N * NN, 2 * N))
        return B

    def bond_hessian(self, Kn, Tv, alive=None):
        """Bond-space Hessian H (NN*N square) of sum_i U_i."""
        N, NN = self.N, self.NN
        act = self.valid if alive is None else (self.valid & alive)
        rows = np.arange(N * NN).reshape(N, NN)
        R = [rows[act]]
        C = [rows[act]]
        V = [Kn[act]]
        for c in (0, 1):
            ks = np.nonzero(self.shell == c)[0]
            for a in ks:
                for b in ks:
                    m = act[:, a] & act[:, b]
                    # 1/2 (t s^T + s t^T): entry (a,b) = 1/2 (Tv_a + Tv_b)
                    R.append(rows[m, a])
                    C.append(rows[m, b])
                    V.append(0.5 * (Tv[m, a] + Tv[m, b]))
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(N * NN, N * NN))
        return H

    def stiffness(self, Kn, Tv, alive=None):
        B = self.bond_matrix(alive)
        H = self.bond_hessian(Kn, Tv, alive)
        K = (B.T @ H @ B).tocsr()
        K = 0.5 * (K + K.T)
        return K.tocsr(), B, H

    # ------------------------------------------------------------------------------------------
    def bond_forces(self, u, Kn, Tv, alive=None):
        """dL (N,NN) and code bond force F (N,NN) (eq. constitutive.c:270)."""
        N, NN = self.N, self.NN
        act = self.valid if alive is None else (self.valid & alive)
        uu = u.reshape(N, 2)
        q = np.where(self.valid, self.nbr, 0)
        du = uu[:, None, :] - uu[q]
        dL = (du * self.ndir[None]).sum(2) * act
        S = np.zeros((N, 2))
        TS = np.zeros((N, 2))
        for c in (0, 1):
            m = self.shell == c
            S[:, c] = dL[:, m].sum(1)
            TS[:, c] = (Tv[:, m] * dL[:, m]).sum(1)
        sh = self.shell[None, :].repeat(N, 0)
        Sq = S[q, sh]
        TSq = TS[q, sh]
        Si = S[np.arange(N)[:, None], sh]
        TSi = TS[np.arange(N)[:, None], sh]
        F = (2 * Kn * dL + 0.5 * (TSi + TSq) + 0.5 * Tv * (Si + Sq)) * act
        return dL, F

    def stress(self, u, Kn, Tv, alive=None):
        """computeStress(): sigma_i = sum_j opp/V l0 F n n  (xx, yy, xy)."""
        act = self.valid if alive is None else (self.valid & alive)
        dL, F = self.bond_forces(u, Kn, Tv, alive)
        full = act.sum(1) == self.NN
        opp_ok = act[:, self.opp]
        of = np.where(full[:, None], 0.5, np.where(self.valid[:, self.opp] & opp_ok, 0.5, 1.0))
        a = of / self.V * self.l0[None, :] * F * act
        n = self.ndir
        s = np.stack([(a * n[None, :, 0] ** 2).sum(1), (a * n[None, :, 1] ** 2).sum(1),
                      (a * n[None, :, 0] * n[None, :, 1]).sum(1)], 1)
        return s

    def energy_per_particle(self, u, Kn, Tv, alive=None):
        act = self.valid if alive is None else (self.valid & alive)
        dL, _ = self.bond_forces(u, Kn, Tv, alive)
        U = 0.5 * (Kn * dL * dL).sum(1)
        for c in (0, 1):
            m = self.shell == c
            U += 0.5 * (Tv[:, m] * dL[:, m]).sum(1) * dL[:, m].sum(1)
        return U


def solve_disp(K, fixed_dofs, u_fixed, f_ext=None, lu_cache=None):
    """Solve K u = f with prescribed u on fixed_dofs. Returns u (full) and the factor (for reuse)."""
    n = K.shape[0]
    fixed = np.zeros(n, bool)
    fixed[fixed_dofs] = True
    free = ~fixed
    u = np.zeros(n)
    u[np.asarray(fixed_dofs)] = u_fixed
    rhs = -K[free][:, fixed] @ u[fixed]
    if f_ext is not None:
        rhs += f_ext[free]
    if lu_cache is None:
        lu_cache = spla.splu(K[free][:, free].tocsc(), permc_spec="COLAMD")
    u[free] = lu_cache.solve(rhs)
    return u, lu_cache


# ---------------------------------------------------------------------------------------------------
# material maps (plane strain, isotropic phases) -> per-particle standard parameters -> per-bond means
# ---------------------------------------------------------------------------------------------------
def iso_C(E, nu, plane="strain"):
    E = np.asarray(E, float)
    nu = np.asarray(nu, float)
    if plane == "strain":
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    else:
        lam = E * nu / (1 - nu * nu)
    mu = E / (2 * (1 + nu))
    return np.stack([lam + 2 * mu, lam, mu], -1)          # C11, C12, C44(=C66)


def square_params_from_C(C):
    """calcKnTv lattice 0: (Kn1, Kn2, Tv) = M (C11, C12, C44)."""
    C11, C12, C44 = C[..., 0], C[..., 1], C[..., 2]
    return np.stack([(C11 - C12) / 2, C44 / 2, (C12 - C44) / 12], -1)


def hex_params_from_C(C):
    """calcKnTv lattice 1 (isotropic, both shells): (Kn, Tv)."""
    C11, C12 = C[..., 0], C[..., 1]
    s3 = math.sqrt(3)
    return np.stack([s3 * (C11 - C12) / 12, s3 * (3 * C12 - C11) / 144], -1)


def bond_params_two_end_mean(lat, ppar):
    """Square: ppar (N,3) = (Kn1, Kn2, Tv) per particle -> Kn, Tv per directed bond (mean of the two ends,
    i.e. the interface treatment of LPM-C)."""
    q = np.where(lat.valid, lat.nbr, 0)
    sh = lat.shell[None, :]
    kn_i = np.where(sh == 0, ppar[:, None, 0], ppar[:, None, 1])
    kn_j = np.where(sh == 0, ppar[q, 0], ppar[q, 1])
    Kn = 0.5 * (kn_i + kn_j) * lat.valid
    Tv = 0.5 * (ppar[:, None, 2] + ppar[q, 2]) * lat.valid
    return Kn, Tv


def affine_disp(x, eps, x0=(0.0, 0.0)):
    """u = eps . (x - x0), eps = [[exx, exy], [exy, eyy]]."""
    d = x - np.asarray(x0)[None, :]
    e = np.asarray(eps, float)
    return d @ e.T
