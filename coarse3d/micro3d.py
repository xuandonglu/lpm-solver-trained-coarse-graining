"""Synthetic 3D two-phase voxel microstructures (numpy only; usable in both environments).

Voxel order = lattice order of lin3d: index p = (k ny + j) nx + i, arrays shaped (nz, ny, nx) in C order.
  spheres     random sequential addition of polydisperse spheres (concrete aggregates in mortar), cut by the box
  ellipsoids  same with randomly oriented ellipsoids (aspect ratios a:b:c), e.g. crushed / elongated aggregates
  grf         thresholded Gaussian random field (non-periodic: generated on a padded grid and cropped)
Overlap rule: the first 70 % of the attempts must not touch existing inclusions (aggregates separated by mortar); after
that overlaps are accepted so that high volume fractions (0.5) are reachable.
"""
import numpy as np


def _grid(nx, ny, nz):
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return np.stack([i, j, k], -1) + 0.5                       # (nz, ny, nx, 3) voxel centres


def _rand_rot(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rsa(nx, ny, nz, phi, rng, sample_shape, max_tries=20000):
    """sample_shape(rng) -> (semi-axes a (3,), rotation R (3,3)); returns phase (nz,ny,nx) bool and #inclusions."""
    img = np.zeros((nz, ny, nx), bool)
    G = _grid(nx, ny, nz)
    ninc = 0
    tries = 0
    target = phi * img.size
    occ = 0
    while occ < target and tries < max_tries:
        tries += 1
        a, R = sample_shape(rng)
        rmax = a.max()
        c = rng.uniform([-rmax * 0.5] * 3, [nx + rmax * 0.5, ny + rmax * 0.5, nz + rmax * 0.5])
        lo = np.maximum(np.floor(c - rmax - 1).astype(int), 0)
        hi = np.minimum(np.ceil(c + rmax + 1).astype(int), [nx, ny, nz])
        if np.any(hi <= lo):
            continue
        sub = G[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] - c
        loc = sub @ R                                         # coordinates in the inclusion frame
        m = ((loc / a) ** 2).sum(-1) <= 1.0
        if not m.any():
            continue
        cur = img[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        if tries < 0.7 * max_tries:
            # no contact: dilate the candidate by one voxel (6-neighbourhood) inside the sub-box
            md = m.copy()
            md[1:] |= m[:-1]
            md[:-1] |= m[1:]
            md[:, 1:] |= m[:, :-1]
            md[:, :-1] |= m[:, 1:]
            md[:, :, 1:] |= m[:, :, :-1]
            md[:, :, :-1] |= m[:, :, 1:]
            if (md & cur).any():
                continue
        new = m & ~cur
        if not new.any():
            continue
        cur |= m
        occ += int(new.sum())
        ninc += 1
    return img, ninc


def spheres(nx, ny, nz, phi, rmean, rng, rjit=0.4):
    def shape(r_):
        r = rmean * (1 + rjit * (2 * r_.random() - 1))
        return np.array([r, r, r]), np.eye(3)
    img, ninc = _rsa(nx, ny, nz, phi, rng, shape)
    return img.astype(np.int64).ravel(), dict(kind="spheres", rmean=rmean, ninc=ninc)


def ellipsoids(nx, ny, nz, phi, rmean, rng, aspect=(1.0, 2.0), rjit=0.4):
    """Semi-axes r*(al, 1, 1/al)^(..) with al in aspect: a = r al^(2/3), b = r al^(-1/3), c = r al^(-1/3) (volume of a
    sphere of radius r, elongated); random orientation."""
    def shape(r_):
        r = rmean * (1 + rjit * (2 * r_.random() - 1))
        al = r_.uniform(*aspect)
        be = r_.uniform(1.0, np.sqrt(al))                     # mild flattening too
        a = r * np.array([al ** (2 / 3) * be ** (1 / 3), al ** (-1 / 3) * be ** (1 / 3), al ** (-1 / 3) * be ** (-2 / 3)])
        return a, _rand_rot(r_)
    img, ninc = _rsa(nx, ny, nz, phi, rng, shape)
    return img.astype(np.int64).ravel(), dict(kind="ellipsoids", rmean=rmean, ninc=ninc, aspect=aspect)


def grf(nx, ny, nz, corr, phi, rng):
    p = int(4 * corr) + 4
    f = rng.standard_normal((nz + 2 * p, ny + 2 * p, nx + 2 * p))
    kz = np.fft.fftfreq(f.shape[0])[:, None, None]
    ky = np.fft.fftfreq(f.shape[1])[None, :, None]
    kx = np.fft.rfftfreq(f.shape[2])[None, None, :]
    g = np.exp(-2 * (np.pi * corr) ** 2 * (kx ** 2 + ky ** 2 + kz ** 2))
    f = np.fft.irfftn(np.fft.rfftn(f) * g, s=f.shape)[p:p + nz, p:p + ny, p:p + nx]
    thr = np.quantile(f, 1 - phi)
    return (f > thr).astype(np.int64).ravel(), dict(kind="grf", corr=corr)


# ---------------------------------------------------------------------------------------------------------------------
# families (training distribution and held-out sets)
# ---------------------------------------------------------------------------------------------------------------------
FAMILIES = {
    # name: phi range, contrast range (E_incl / E_matrix), morphology mixture
    "train": dict(phi=(0.30, 0.50), contrast=(2.0, 5.0), morph="mix"),
    "elong": dict(phi=(0.30, 0.50), contrast=(2.0, 5.0), morph="elong"),
    "c8": dict(phi=(0.30, 0.50), contrast=(8.0, 8.0), morph="mix"),
    "soft": dict(phi=(0.30, 0.50), contrast=(0.5, 0.5), morph="mix"),
}
NU_M = 0.2


def sample(n, rng, family="train"):
    """One microstructure of an n^3 box: phase (n^3,), E, nu per voxel, info."""
    f = FAMILIES[family]
    phi = rng.uniform(*f["phi"])
    c = rng.uniform(*f["contrast"])
    nu_i = rng.uniform(0.15, 0.25)
    morph = f["morph"]
    u = rng.random()
    if morph == "elong":
        ph, info = ellipsoids(n, n, n, phi, rng.uniform(2.0, 3.5), rng, aspect=(3.0, 4.0))
    elif u < 0.25:
        ph, info = spheres(n, n, n, phi, rng.uniform(2.0, 4.0), rng)
    elif u < 0.5:
        ph, info = ellipsoids(n, n, n, phi, rng.uniform(2.0, 4.0), rng, aspect=(1.0, 2.0))
    else:
        ph, info = grf(n, n, n, rng.uniform(1.0, 3.0), phi, rng)
    E = np.where(ph == 1, c, 1.0)
    nu = np.where(ph == 1, nu_i, NU_M)
    info.update(phi_target=phi, phi=float(ph.mean()), contrast=c, nu_i=nu_i, family=family)
    return ph, E, nu, info
