"""2D benchmark problems, numpy/scipy-only pieces (importable from the torch environment): problems, boundary
conditions, block averages, error measures.  See el_core.py for the problem definitions and the solver."""
import numpy as np
import lin2d
import micro

LOADS = ["ring_xx", "ring_yy", "ring_xy", "tb_ty", "tb_sx", "lr_tx"]
BCSETS = {"ring": ["ring_xx", "ring_yy", "ring_xy"], "tb": ["tb_ty", "tb_sx"], "lr": ["lr_tx"]}
EPS = 1e-3


# ----------------------------------------------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------------------------------------------
def make_comp(n, seed):
    rng = np.random.default_rng([7007, n, seed])
    ph, info = micro.random_two_phase(n, n, rng, corr_range=(1.5, 6.0), phi_range=(0.15, 0.40))
    c = float(rng.choice([5.0, 10.0]))
    E = np.where(ph == 1, c, 1.0)
    nu = np.where(ph == 1, 0.2, 0.3)
    Cpix = lin2d.iso_C(E, nu)
    return dict(kind="comp", n=n, seed=seed, phase=ph, Cpix=Cpix, contrast=c, phi=float(ph.mean()), micro=info[0],
                corr=float(info[1]))


def fine_bond_params(prob, lat):
    return lin2d.bond_params_two_end_mean(lat, lin2d.square_params_from_C(prob["Cpix"]))


def interface_cells(phase2d, s):
    p = phase2d
    b = np.zeros_like(p, bool)
    b[:, 1:] |= p[:, 1:] != p[:, :-1]
    b[:, :-1] |= p[:, 1:] != p[:, :-1]
    b[1:, :] |= p[1:, :] != p[:-1, :]
    b[:-1, :] |= p[1:, :] != p[:-1, :]
    ny, nx = p.shape
    return b.reshape(ny // s, s, nx // s, s).any(axis=(1, 3)).ravel()


# ----------------------------------------------------------------------------------------------------------------
# boundary conditions (I2 cg.prescribed with an explicit number of block layers)
# ----------------------------------------------------------------------------------------------------------------
def prescribed(load, x, IJ, ncx, ncy, s, h, L):
    Lx, Ly = ncx * s * h, ncy * s * h
    I, J = IJ[:, 0], IJ[:, 1]
    u = np.zeros((len(x), 2))
    if load.startswith("ring"):
        xc = np.array([Lx / 2, Ly / 2])
        e = {"ring_xx": [[1, 0], [0, 0]], "ring_yy": [[0, 0], [0, 1]], "ring_xy": [[0, .5], [.5, 0]]}[load]
        fixed = (I < L) | (I >= ncx - L) | (J < L) | (J >= ncy - L)
        u[fixed] = lin2d.affine_disp(x[fixed], EPS * np.array(e), xc)
        return fixed, u, None
    if load.startswith("tb"):
        a, b = J < L, J >= ncy - L
        dist = (ncy - L) * s * h
        d = np.array([0.0, EPS * dist]) if load == "tb_ty" else np.array([EPS * dist, 0.0])
    else:
        a, b = I < L, I >= ncx - L
        dist = (ncx - L) * s * h
        d = np.array([EPS * dist, 0.0])
    u[b] = d
    return a | b, u, b


def block_mean(f, IJ, ncx, ncy):
    b = IJ[:, 1] * ncx + IJ[:, 0]
    cnt = np.bincount(b, minlength=ncx * ncy)
    if f.ndim == 1:
        return np.bincount(b, f, minlength=ncx * ncy) / np.maximum(cnt, 1)
    return np.stack([np.bincount(b, f[:, k], minlength=ncx * ncy) / np.maximum(cnt, 1) for k in range(f.shape[1])], 1)


def block_sum(f, IJ, ncx, ncy):
    b = IJ[:, 1] * ncx + IJ[:, 0]
    return np.bincount(b, f, minlength=ncx * ncy)


_LATC = {}


def coarse_lattice(kind, n, s):
    key = (kind, n, s)
    if key not in _LATC:
        nc = n // s
        _LATC[key] = lin2d.Lattice2D("square", nc, nc, float(s), origin=(0.0, 0.0))
    return _LATC[key]


def bc_layers(kind):
    return 1


_FREE = {}


def free_cells(kind, n, s, ld):
    key = (kind, n, s, ld)
    if key not in _FREE:
        lat = coarse_lattice(kind, n, s)
        nc = n // s
        fixed, _, _ = prescribed(ld, lat.x, lat.ij, nc, nc, s, 1.0, bc_layers(kind))
        _FREE[key] = ~fixed
    return _FREE[key]


def gval(ld, r, free):
    """global stiffness measure of a load case: grip reaction (plates) or interior mean stress (ring)."""
    if r["F"] is not None:
        return float(r["F"][1 if ld == "tb_ty" else 0])
    k = {"ring_xx": 0, "ring_yy": 1, "ring_xy": 2}[ld]
    return float(r["sig"][free, k].mean())


def metrics(kind, n, s, ref, iface, out):
    """I2 error measures per load case (signed G, others relative L2 over free cells)."""
    m = {}
    for ld in LOADS:
        a, b = ref[ld], out[ld]
        free = free_cells(kind, n, s, ld)
        ga, gb = gval(ld, a, free), gval(ld, b, free)
        d = dict(G=(gb - ga) / ga,
                 u=float(np.linalg.norm(b["u"][free] - a["u"][free]) / np.linalg.norm(a["u"][free])),
                 sig=float(np.linalg.norm(b["sig"][free] - a["sig"][free]) / np.linalg.norm(a["sig"][free])))
        fi = free & iface
        if fi.any():
            d["w_if"] = float(np.linalg.norm(b["w"][fi] - a["w"][fi]) / np.linalg.norm(a["w"][fi]))
            d["sig_if"] = float(np.linalg.norm(b["sig"][fi] - a["sig"][fi]) / np.linalg.norm(a["sig"][fi]))
        else:
            d["w_if"] = d["sig_if"] = float("nan")
        m[ld] = d
    return m


def block_fields(lat, IJ, nc, s, res):
    """fine results {load: u, sig, U, F} -> block-averaged reference {load: u, sig, w, F}."""
    ref = {}
    for ld, r in res.items():
        ref[ld] = dict(u=block_mean(r["u"], IJ, nc, nc), sig=block_mean(r["sig"], IJ, nc, nc),
                       w=block_sum(r["U"], IJ, nc, nc) / s ** 2, F=r["F"])
    return ref