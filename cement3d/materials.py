"""Labels -> voxel materials of the fine LPM, pore treatments, connectivity cleaning, cell masks (numpy + scipy; both
environments).

Phase properties (Haecker et al. 2005, Cem. Concr. Res. 35:1948, the commonly used CEMHYD3D/ELAS3D inputs):
  C-S-H / hydrates 22.4 GPa, nu 0.25;  CH 42.3 GPa, nu 0.324;  clinker (as C3S) 117.6 GPa, nu 0.314.
Moduli are used in units of E_C-S-H (linear problems; all errors are relative).

Pore treatments (labels 0 = capillary pore, 4 = large air void):
  "a"  void: pore and air voxels are removed (no particle), the network sees the void flag inside the specimen;
  "b"  porous hydrate: pore voxels are merged with the hydrate voxels of the same cube into one porous-hydrate phase,
       Mori-Tanaka with spherical voids at the cube's porosity phi = n_pore / (n_pore + n_hydrate) (the standard
       "hydrate foam" estimate); large air voids (label 4) stay void (a 60 um air bubble is not porous hydrate).
Cleaning of the solid set (both treatments): keep the largest 18-connected component (the SC bonds of both shells), then
iteratively remove particles with fewer than 4 bonds or whose bonds do not span R^3 (singular 3x3 diagonal block of K,
a mechanism; 3-bond particles form floppy pairs), until nothing changes.  Remaining zero-energy (floppy) modes of
K_ff are removed numerically in floppy_v1.py (GPU null-space test).  Coarse cells exist if their s^3 block contains a
solid voxel; the same cleaning is applied on the coarse grid (cells removed there are excluded from all metrics and
counted).
"""
import numpy as np
from scipy import ndimage as ndi
from geom3d import SC_OFFS, iso_C

E_CSH = 22.4
PROPS = {1: (22.4, 0.25), 2: (42.3, 0.324), 3: (117.6, 0.314)}      # GPa, nu
ST18 = ndi.generate_binary_structure(3, 2)
_ND = SC_OFFS / np.linalg.norm(SC_OFFS, axis=1)[:, None]
_NN = np.einsum("ka,kb->kab", _ND, _ND)


def mt_porous(E, nu, phi):
    """Mori-Tanaka, spherical voids at volume fraction phi in an isotropic matrix (E, nu) -> (E*, nu*)."""
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    Ks = K * 4 * G * (1 - phi) / (3 * phi * K + 4 * G)
    z = G * (9 * K + 8 * G) / (6 * (K + 2 * G))
    Gs = G * z * (1 - phi) / (phi * G + z)
    return 9 * Ks * Gs / (3 * Ks + Gs), (3 * Ks - 2 * Gs) / (2 * (3 * Ks + Gs))


def _shift(a, o):
    """b[p] = a[p + o] (zero outside), a (nz, ny, nx), o = (dx, dy, dz)."""
    b = np.zeros_like(a)
    n = a.shape
    src = [slice(max(0, d), n[ax] + min(0, d)) for ax, d in zip((2, 1, 0), o)]
    dst = [slice(max(0, -d), n[ax] - max(0, d)) for ax, d in zip((2, 1, 0), o)]
    b[dst[2], dst[1], dst[0]] = a[src[2], src[1], src[0]]
    return b


def degenerate(s):
    """Solid sites whose bonds to solid neighbours do not span R^3 (s: (nz, ny, nx) bool)."""
    M = np.zeros(s.shape + (3, 3))
    for k, o in enumerate(SC_OFFS):
        M += _shift(s, o)[..., None, None] * _NN[k]
    lmin = np.linalg.eigvalsh(M[s])[:, 0]
    out = np.zeros_like(s)
    out[s] = lmin < 1e-8
    return out


def nbonds(s):
    c = np.zeros(s.shape, np.int64)
    for o in SC_OFFS:
        c += _shift(s, o)
    return c


def clean(s, max_it=50, kmin=4):
    """Largest 18-connected component + iterative removal of degenerate / under-coordinated sites.  Returns cleaned
    mask, #removed."""
    s = s.copy()
    n0 = int(s.sum())
    for _ in range(max_it):
        lab, nl = ndi.label(s, structure=ST18)
        if nl > 1:
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            s = lab == sz.argmax()
        d = degenerate(s) | (s & (nbonds(s) < kmin))
        if not d.any() and nl <= 1:
            break
        s &= ~d
    return s, n0 - int(s.sum())


def block_count(m, n, s):
    nc = n // s
    return m.reshape(nc, s, nc, s, nc, s).sum(axis=(1, 3, 5))


def make_material(lab, option, props=None):
    """lab (n,n,n) uint8 labels -> dict: solid (n^3,) bool, E, nu (n^3,) in units of E_C-S-H (0 on void), Cv (n^3, 3),
    ph (n^3,) phase id used for interfaces (void = -1), info."""
    props = PROPS if props is None else props
    n = lab.shape[0]
    lab = np.asarray(lab)
    if option == "a":
        void = (lab == 0) | (lab == 4)
    elif option == "b":
        void = lab == 4
    else:
        raise ValueError(option)
    solid, ndrop = clean(~void)
    E = np.zeros(lab.shape)
    nu = np.zeros(lab.shape)
    ph = np.full(lab.shape, -1, np.int64)
    info = dict(option=option, ndrop=ndrop, void_frac=float(1 - solid.mean()))
    for k in (1, 2, 3):
        Ek, nk = props[k]
        m = lab == k
        if option == "b" and k == 1:
            n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
            phi = n0 / max(1, n0 + n1)
            Ek, nk = mt_porous(Ek, nk, phi)
            m = (lab == 1) | (lab == 0)
            info.update(phi_ph=phi, E_ph=Ek / E_CSH, nu_ph=nk)
        E[m] = Ek / E_CSH
        nu[m] = nk
        ph[m] = k
    E[~solid] = 0.0
    nu[~solid] = 0.0
    ph[~solid] = -1
    Cv = iso_C(E.ravel(), nu.ravel())
    return dict(solid=solid.ravel(), E=E.ravel(), nu=nu.ravel(), Cv=Cv, ph=ph.ravel(), info=info)


def coarse_mask(solid, n, s):
    """Coarse cells: block contains a solid voxel, then cleaned on the coarse grid.  Returns (cell mask (nc^3,),
    #cells with solid voxels that were removed by the coarse cleaning)."""
    cnt = block_count(solid.reshape(n, n, n), n, s)
    c0 = cnt > 0
    c, nrm = clean(c0)
    return c.ravel(), nrm


def cell_sets(lab, mat, n, s):
    """Interface cells (block has a solid voxel with a 6-neighbour of another solid phase) and pore-adjacent cells
    (block contains a pore/air voxel (labels 0, 4) or a voxel 6-adjacent to one; the same geometric set for both pore
    treatments)."""
    ph = mat["ph"].reshape(n, n, n)
    sol = mat["solid"].reshape(n, n, n)
    poreg = (lab == 0) | (lab == 4)
    iface = np.zeros((n, n, n), bool)
    padj = poreg.copy()
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(1, None)
        b[ax] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        d = (ph[a] != ph[b]) & sol[a] & sol[b]
        iface[a] |= d
        iface[b] |= d
        padj[a] |= poreg[b]
        padj[b] |= poreg[a]
    return block_count(iface, n, s).ravel() > 0, block_count(padj, n, s).ravel() > 0
