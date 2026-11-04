"""Small-strain (linearised) 3D LPM on the simple-cubic lattice (LPM-C lattice 2) with per-bond Kn/Tv.

This is the LPM-C elastic law (constitutive.c computeBondForceElastic; port kernels.force_from_state) linearised about
the reference configuration.  Energy of particle i:
    U_i = 1/2 sum_j Kn_ij dL_ij^2 + 1/2 sum_c (sum_{j in c} Tv_ij dL_ij)(sum_{j in c} dL_ij),   dL_ij = n_ij.(u_i - u_j),
n_ij = (x_i - x_j)/|x_i - x_j| (the port's csx/csy/csz).  For symmetric bond parameters (Kn_ij = Kn_ji, Tv_ij = Tv_ji;
LPM-C's SC calcKnTv takes the two-end mean) the code bond force
    F_ij = 2 Kn dL + 1/2 (TS_i^c + TS_j^c) + 1/2 Tv (S_i^c + S_j^c),   S_i^c = sum_{k in c} dL_ik, TS_i^c = sum Tv_ik dL_ik
is dU/d(dL_ij), so  K u = sum_j F_ij n_ij  is the exact Hessian-vector product and K is symmetric.

SC lattice: spacing h = 2r; shell 0 = 6 axial neighbours (distance h), shell 1 = 12 face diagonals (sqrt2 h); particle
volume V = h^3 = 8 r^3.  Voxel (i, j, k) -> particle at ((i+.5)h, (j+.5)h, (k+.5)h) + origin, id = (k ny + j) nx + i
(only active sites are numbered).  Slots come in reverse pairs (k, k^1); shell 0 = slots 0-5, shell 1 = slots 6-17.

Contents: lattice geometry, numba kernels (matrix-free K u, block diagonal, stress = computeStress incl. the opp_flag
surface rule, particle energy), CSR assembly of K (row-wise, direct, no COO), direct (SuperLU) and block-Jacobi PCG
solvers, isotropic -> SC (Kn1, Kn2, Tv) map (calcKnTv lattice 2) and the two-end mean.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

from geom3d import (SC_OFFS, SC_SHELL, NSLOT, DIR_OF_SLOT, FWD_SLOTS, LatticeGeom, iso_C, sc_params_from_C,  # noqa
                    bond_params_two_end_mean, affine_disp, unit_strain, VOIGT_PAIRS)


class Lattice3D(LatticeGeom):
    """SC lattice geometry (geom3d.LatticeGeom) + numba kernels."""

    def __init__(self, nx, ny, nz, h=1.0, origin=(0.0, 0.0, 0.0), active=None):
        super().__init__(nx, ny, nz, h, origin, active)
        self._pat = None

    # ---------------------------------------------------------------- kernels (thin wrappers)
    def apply(self, u, Kn, Tv):
        return k_apply(np.ascontiguousarray(u.reshape(self.N, 3)), self.nbr, self.ndir, Kn, Tv)

    def bond_forces(self, u, Kn, Tv):
        return k_bond_forces(np.ascontiguousarray(u.reshape(self.N, 3)), self.nbr, self.ndir, Kn, Tv)

    def stress(self, u, Kn, Tv):
        """computeStress(): (N, 6) in the code order xx, yy, zz, yz, xz, xy."""
        return k_stress(np.ascontiguousarray(u.reshape(self.N, 3)), self.nbr, self.ndir, self.l0, Kn, Tv, self.V)

    def energy_per_particle(self, u, Kn, Tv):
        return k_energy(np.ascontiguousarray(u.reshape(self.N, 3)), self.nbr, self.ndir, Kn, Tv)

    def block_diag(self, Kn, Tv):
        return k_block_diag(self.nbr, self.ndir, Kn, Tv)

    def stiffness(self, Kn, Tv):
        """Assembled K (3N x 3N CSR)."""
        if self._pat is None:
            self._pat = csr_pattern(self.ijk, self.nbr)
        rowptr, colblk = self._pat
        indptr, indices, data = csr_values(self.ijk, self.nbr, self.ndir, Kn, Tv, rowptr, colblk)
        return sp.csr_matrix((data, indices, indptr), shape=(3 * self.N, 3 * self.N))

    # ---------------------------------------------------------------- reference: K = B^T H B (slow, small lattices)
    def bond_matrix(self):
        N = self.N
        rows = np.arange(N * NSLOT).reshape(N, NSLOT)
        q = np.where(self.valid, self.nbr, 0)
        r_l, c_l, v_l = [], [], []
        for d in range(3):
            nd = np.broadcast_to(self.ndir[None, :, d], (N, NSLOT))
            r_l += [rows[self.valid], rows[self.valid]]
            c_l += [3 * np.broadcast_to(np.arange(N)[:, None], (N, NSLOT))[self.valid] + d, 3 * q[self.valid] + d]
            v_l += [nd[self.valid], -nd[self.valid]]
        return sp.csr_matrix((np.concatenate(v_l), (np.concatenate(r_l), np.concatenate(c_l))), shape=(N * NSLOT, 3 * N))

    def bond_hessian(self, Kn, Tv):
        N = self.N
        act = self.valid
        rows = np.arange(N * NSLOT).reshape(N, NSLOT)
        R, C, V = [rows[act]], [rows[act]], [Kn[act]]
        for c in (0, 1):
            ks = np.nonzero(SC_SHELL == c)[0]
            for a in ks:
                for b in ks:
                    m = act[:, a] & act[:, b]
                    R.append(rows[m, a])
                    C.append(rows[m, b])
                    V.append(0.5 * (Tv[m, a] + Tv[m, b]))
        return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(N * NSLOT, N * NSLOT))

    def stiffness_BHB(self, Kn, Tv):
        B = self.bond_matrix()
        H = self.bond_hessian(Kn, Tv)
        return (B.T @ H @ B).tocsr()


# =====================================================================================================================
# numba kernels
# =====================================================================================================================
@njit(cache=True, parallel=True)
def k_shell_sums(u, nbr, ndir, Tv):
    N = nbr.shape[0]
    S = np.zeros((N, 2))
    TS = np.zeros((N, 2))
    for i in prange(N):
        a0 = 0.0
        a1 = 0.0
        b0 = 0.0
        b1 = 0.0
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            dl = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
            if k < 6:
                a0 += dl
                b0 += Tv[i, k] * dl
            else:
                a1 += dl
                b1 += Tv[i, k] * dl
        S[i, 0] = a0
        S[i, 1] = a1
        TS[i, 0] = b0
        TS[i, 1] = b1
    return S, TS


@njit(cache=True, parallel=True)
def k_apply(u, nbr, ndir, Kn, Tv):
    """Pin = K u (N, 3)."""
    N = nbr.shape[0]
    S, TS = k_shell_sums(u, nbr, ndir, Tv)
    out = np.zeros((N, 3))
    for i in prange(N):
        ox = 0.0
        oy = 0.0
        oz = 0.0
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            c = 0 if k < 6 else 1
            dl = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
            f = 2.0 * Kn[i, k] * dl + 0.5 * (TS[i, c] + TS[q, c]) + 0.5 * Tv[i, k] * (S[i, c] + S[q, c])
            ox += f * ndir[k, 0]
            oy += f * ndir[k, 1]
            oz += f * ndir[k, 2]
        out[i, 0] = ox
        out[i, 1] = oy
        out[i, 2] = oz
    return out


@njit(cache=True, parallel=True)
def k_bond_forces(u, nbr, ndir, Kn, Tv):
    N = nbr.shape[0]
    S, TS = k_shell_sums(u, nbr, ndir, Tv)
    dL = np.zeros((N, 18))
    F = np.zeros((N, 18))
    for i in prange(N):
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            c = 0 if k < 6 else 1
            dl = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
            dL[i, k] = dl
            F[i, k] = 2.0 * Kn[i, k] * dl + 0.5 * (TS[i, c] + TS[q, c]) + 0.5 * Tv[i, k] * (S[i, c] + S[q, c])
    return dL, F


@njit(cache=True, parallel=True)
def k_stress(u, nbr, ndir, l0, Kn, Tv, V):
    """computeStress (lpm_basic.c; port kernels.compute_stress) for intact bonds: opp_flag = 1/2 if the particle has all
    18 bonds or the opposite bond exists, else 1."""
    N = nbr.shape[0]
    dL, F = k_bond_forces(u, nbr, ndir, Kn, Tv)
    st = np.zeros((N, 6))
    for i in prange(N):
        nb = 0
        for k in range(18):
            if nbr[i, k] >= 0:
                nb += 1
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        for k in range(18):
            if nbr[i, k] < 0:
                continue
            of = 0.5
            if nb != 18 and nbr[i, k ^ 1] < 0:
                of = 1.0
            a = of / V * l0[k] * F[i, k]
            n0 = ndir[k, 0]
            n1 = ndir[k, 1]
            n2 = ndir[k, 2]
            s0 += a * n0 * n0
            s1 += a * n1 * n1
            s2 += a * n2 * n2
            s3 += a * n1 * n2
            s4 += a * n0 * n2
            s5 += a * n0 * n1
        st[i, 0] = s0
        st[i, 1] = s1
        st[i, 2] = s2
        st[i, 3] = s3
        st[i, 4] = s4
        st[i, 5] = s5
    return st


@njit(cache=True, parallel=True)
def k_energy(u, nbr, ndir, Kn, Tv):
    N = nbr.shape[0]
    U = np.zeros(N)
    for i in prange(N):
        e = 0.0
        a0 = 0.0
        a1 = 0.0
        b0 = 0.0
        b1 = 0.0
        for k in range(18):
            q = nbr[i, k]
            if q < 0:
                continue
            dl = ndir[k, 0] * (u[i, 0] - u[q, 0]) + ndir[k, 1] * (u[i, 1] - u[q, 1]) + ndir[k, 2] * (u[i, 2] - u[q, 2])
            e += 0.5 * Kn[i, k] * dl * dl
            if k < 6:
                a0 += dl
                b0 += Tv[i, k] * dl
            else:
                a1 += dl
                b1 += Tv[i, k] * dl
        U[i] = e + 0.5 * (a0 * b0 + a1 * b1)
    return U


@njit(cache=True, parallel=True)
def k_block_diag(nbr, ndir, Kn, Tv):
    """3x3 diagonal blocks of K: D_p = sum_a (Kn_pa + Kn_qa' + Tv_qa') n_a n_a^T + sum_c 1/2 (T_pc s_pc^T + s_pc T_pc^T),
    T_pc = sum_{a in c} Tv_pa n_a, s_pc = sum_{a in c} n_a (valid bonds)."""
    N = nbr.shape[0]
    D = np.zeros((N, 3, 3))
    for p in prange(N):
        for c in range(2):
            k0 = 0 if c == 0 else 6
            k1 = 6 if c == 0 else 18
            T = np.zeros(3)
            s = np.zeros(3)
            for a in range(k0, k1):
                if nbr[p, a] < 0:
                    continue
                for d in range(3):
                    T[d] += Tv[p, a] * ndir[a, d]
                    s[d] += ndir[a, d]
            for d in range(3):
                for e in range(3):
                    D[p, d, e] += 0.5 * (T[d] * s[e] + s[d] * T[e])
            for a in range(k0, k1):
                q = nbr[p, a]
                if q < 0:
                    continue
                a2 = a ^ 1
                coef = Kn[p, a] + Kn[q, a2] + Tv[q, a2]
                for d in range(3):
                    for e in range(3):
                        D[p, d, e] += coef * ndir[a, d] * ndir[a, e]
    return D


@njit(cache=True)
def _inv3(A):
    a, b, c = A[0, 0], A[0, 1], A[0, 2]
    d, e, f = A[1, 0], A[1, 1], A[1, 2]
    g, h, i = A[2, 0], A[2, 1], A[2, 2]
    co00 = e * i - f * h
    co01 = -(d * i - f * g)
    co02 = d * h - e * g
    det = a * co00 + b * co01 + c * co02
    out = np.empty((3, 3))
    out[0, 0] = co00 / det
    out[1, 0] = co01 / det
    out[2, 0] = co02 / det
    out[0, 1] = -(b * i - c * h) / det
    out[1, 1] = (a * i - c * g) / det
    out[2, 1] = -(a * h - b * g) / det
    out[0, 2] = (b * f - c * e) / det
    out[1, 2] = -(a * f - c * d) / det
    out[2, 2] = (a * e - b * d) / det
    return out


@njit(cache=True, parallel=True)
def k_block_jacobi(D, free):
    N = D.shape[0]
    M = np.zeros((N, 3, 3))
    for p in prange(N):
        if free[p]:
            M[p] = _inv3(D[p])
    return M


# ---------------------------------------------------------------------------------------------- CSR assembly
@njit(cache=True)
def _oidx(d0, d1, d2):
    return (d2 + 2) * 25 + (d1 + 2) * 5 + (d0 + 2)


@njit(cache=True, parallel=True)
def _pattern_count(ijk, nbr):
    N = nbr.shape[0]
    cnt = np.zeros(N, np.int64)
    for p in prange(N):
        mark = np.zeros(125, np.bool_)
        mark[62] = True
        for a in range(18):
            i = nbr[p, a]
            if i < 0:
                continue
            mark[_oidx(ijk[i, 0] - ijk[p, 0], ijk[i, 1] - ijk[p, 1], ijk[i, 2] - ijk[p, 2])] = True
            k0 = 0 if a < 6 else 6
            k1 = 6 if a < 6 else 18
            for b in range(k0, k1):
                q = nbr[i, b]
                if q >= 0:
                    mark[_oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])] = True
        c = 0
        for m in range(125):
            if mark[m]:
                c += 1
        cnt[p] = c
    return cnt


@njit(cache=True, parallel=True)
def _pattern_fill(ijk, nbr, rowptr, colblk):
    N = nbr.shape[0]
    for p in prange(N):
        ids = np.empty(125, np.int64)
        mark = np.zeros(125, np.bool_)
        mark[62] = True
        ids[62] = p
        for a in range(18):
            i = nbr[p, a]
            if i < 0:
                continue
            o = _oidx(ijk[i, 0] - ijk[p, 0], ijk[i, 1] - ijk[p, 1], ijk[i, 2] - ijk[p, 2])
            mark[o] = True
            ids[o] = i
            k0 = 0 if a < 6 else 6
            k1 = 6 if a < 6 else 18
            for b in range(k0, k1):
                q = nbr[i, b]
                if q >= 0:
                    o = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
                    mark[o] = True
                    ids[o] = q
        c = 0
        tmp = np.empty(rowptr[p + 1] - rowptr[p], np.int64)
        for m in range(125):
            if mark[m]:
                tmp[c] = ids[m]
                c += 1
        tmp = np.sort(tmp)
        for m in range(c):
            colblk[rowptr[p] + m] = tmp[m]


def csr_pattern(ijk, nbr):
    cnt = _pattern_count(ijk, nbr)
    rowptr = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=rowptr[1:])
    colblk = np.zeros(rowptr[-1], np.int64)
    _pattern_fill(ijk, nbr, rowptr, colblk)
    return rowptr, colblk


@njit(cache=True, parallel=True)
def _csr_values(ijk, nbr, ndir, Kn, Tv, rowptr, colblk, indptr, indices, data):
    N = nbr.shape[0]
    for p in prange(N):
        buf = np.zeros((125, 3, 3))
        # (A) the two groups (p, c) centred at p
        for c in range(2):
            k0 = 0 if c == 0 else 6
            k1 = 6 if c == 0 else 18
            T = np.zeros(3)
            s = np.zeros(3)
            for a in range(k0, k1):
                if nbr[p, a] < 0:
                    continue
                for d in range(3):
                    T[d] += Tv[p, a] * ndir[a, d]
                    s[d] += ndir[a, d]
            for b in range(k0, k1):
                q = nbr[p, b]
                if q < 0:
                    continue
                oq = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
                for d in range(3):
                    w = Kn[p, b] * ndir[b, d] + 0.5 * (T[d] + Tv[p, b] * s[d])
                    for e in range(3):
                        v = w * ndir[b, e]
                        buf[62, d, e] += v
                        buf[oq, d, e] -= v
        # (B) groups (i, c) centred at the neighbours i = q_a (p sits in slot a^1 of i, c = shell of a)
        for a in range(18):
            i = nbr[p, a]
            if i < 0:
                continue
            a2 = a ^ 1
            k0 = 0 if a < 6 else 6
            k1 = 6 if a < 6 else 18
            T = np.zeros(3)
            s = np.zeros(3)
            for b in range(k0, k1):
                if nbr[i, b] < 0:
                    continue
                for d in range(3):
                    T[d] += Tv[i, b] * ndir[b, d]
                    s[d] += ndir[b, d]
            oi = _oidx(ijk[i, 0] - ijk[p, 0], ijk[i, 1] - ijk[p, 1], ijk[i, 2] - ijk[p, 2])
            for e in range(3):
                w = Kn[i, a2] * ndir[a2, e] + 0.5 * (T[e] + Tv[i, a2] * s[e])
                for d in range(3):
                    buf[oi, d, e] += ndir[a, d] * w
            for b in range(k0, k1):
                q = nbr[i, b]
                if q < 0:
                    continue
                Hab = 0.5 * (Tv[i, a2] + Tv[i, b])
                if b == a2:
                    Hab += Kn[i, a2]
                oq = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
                for d in range(3):
                    for e in range(3):
                        buf[oq, d, e] -= Hab * ndir[a, d] * ndir[b, e]
        nblk = rowptr[p + 1] - rowptr[p]
        for m in range(nblk):
            q = colblk[rowptr[p] + m]
            o = _oidx(ijk[q, 0] - ijk[p, 0], ijk[q, 1] - ijk[p, 1], ijk[q, 2] - ijk[p, 2])
            for d in range(3):
                base = indptr[3 * p + d] + 3 * m
                for e in range(3):
                    indices[base + e] = 3 * q + e
                    data[base + e] = buf[o, d, e]


def csr_values(ijk, nbr, ndir, Kn, Tv, rowptr, colblk):
    N = nbr.shape[0]
    nblk = np.diff(rowptr)
    rowlen = np.repeat(3 * nblk, 3)
    indptr = np.zeros(3 * N + 1, np.int64)
    np.cumsum(rowlen, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.zeros(nnz, np.int32)
    data = np.zeros(nnz)
    _csr_values(ijk, nbr, ndir, np.ascontiguousarray(Kn), np.ascontiguousarray(Tv), rowptr, colblk, indptr, indices, data)
    return indptr, indices, data


# ---------------------------------------------------------------------------------------------- solvers
@njit(cache=True, parallel=True)
def _pcg(nbr, ndir, Kn, Tv, M, free, b, x, tol, maxit):
    """Block-Jacobi PCG for K_ff x_f = b_f; free: (N,) bool (all 3 dofs of a particle free or fixed); b, x (N,3) with
    zeros on fixed particles.  Stops when ||r|| <= tol ||b||.  Returns (iterations, relative residual)."""
    N = nbr.shape[0]
    r = b - k_apply(x, nbr, ndir, Kn, Tv)
    for i in prange(N):
        if not free[i]:
            r[i, 0] = 0.0
            r[i, 1] = 0.0
            r[i, 2] = 0.0
    bn = 0.0
    for i in prange(N):
        bn += b[i, 0] * b[i, 0] + b[i, 1] * b[i, 1] + b[i, 2] * b[i, 2]
    bn = math.sqrt(bn)
    if bn == 0.0:
        return 0, 0.0
    z = np.zeros((N, 3))
    for i in prange(N):
        for d in range(3):
            z[i, d] = M[i, d, 0] * r[i, 0] + M[i, d, 1] * r[i, 1] + M[i, d, 2] * r[i, 2]
    p = z.copy()
    rz = 0.0
    for i in prange(N):
        rz += r[i, 0] * z[i, 0] + r[i, 1] * z[i, 1] + r[i, 2] * z[i, 2]
    it = 0
    rn = 0.0
    for it in range(1, maxit + 1):
        Ap = k_apply(p, nbr, ndir, Kn, Tv)
        pAp = 0.0
        for i in prange(N):
            if free[i]:
                pAp += p[i, 0] * Ap[i, 0] + p[i, 1] * Ap[i, 1] + p[i, 2] * Ap[i, 2]
        alpha = rz / pAp
        rn = 0.0
        for i in prange(N):
            if free[i]:
                for d in range(3):
                    x[i, d] += alpha * p[i, d]
                    r[i, d] -= alpha * Ap[i, d]
                rn += r[i, 0] * r[i, 0] + r[i, 1] * r[i, 1] + r[i, 2] * r[i, 2]
        rn = math.sqrt(rn)
        if rn <= tol * bn:
            break
        rz2 = 0.0
        for i in prange(N):
            for d in range(3):
                z[i, d] = M[i, d, 0] * r[i, 0] + M[i, d, 1] * r[i, 1] + M[i, d, 2] * r[i, 2]
            rz2 += r[i, 0] * z[i, 0] + r[i, 1] * z[i, 1] + r[i, 2] * z[i, 2]
        beta = rz2 / rz
        rz = rz2
        for i in prange(N):
            for d in range(3):
                p[i, d] = z[i, d] + beta * p[i, d]
    return it, rn / bn


def solve_pcg(lat, Kn, Tv, fixed, u_presc, tol=1e-10, maxit=20000, M=None, x0=None):
    """u with u = u_presc on fixed particles (fixed: (N,) bool) and K_ff u_f = -K_fp u_p.  Returns u (N,3), info."""
    free = ~np.asarray(fixed, bool)
    up = np.zeros((lat.N, 3))
    up[~free] = u_presc[~free]
    b = -k_apply(up, lat.nbr, lat.ndir, Kn, Tv)
    b[~free] = 0.0
    if M is None:
        M = k_block_jacobi(k_block_diag(lat.nbr, lat.ndir, Kn, Tv), free)
    x = np.zeros((lat.N, 3)) if x0 is None else np.where(free[:, None], x0, 0.0)
    it, rr = _pcg(lat.nbr, lat.ndir, Kn, Tv, M, free, b, x, tol, maxit)
    return up + x, dict(iters=int(it), relres=float(rr))


def solve_direct(K, fixed, u_presc, lu=None):
    """Same problem with the assembled K and SuperLU (factorisation returned for reuse)."""
    n = K.shape[0]
    fd = np.repeat(np.asarray(fixed, bool), 3)
    free = ~fd
    u = np.zeros(n)
    u[fd] = np.asarray(u_presc).reshape(-1)[fd]
    rhs = -(K[free][:, fd] @ u[fd])
    if lu is None:
        lu = spla.splu(K[free][:, free].tocsc(), permc_spec="COLAMD")
    u[free] = lu.solve(rhs)
    return u.reshape(-1, 3), lu
