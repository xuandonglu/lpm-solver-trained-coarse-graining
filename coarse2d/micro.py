"""Random microstructure generators on pixel grids (row-major: index p = j*nx + i)."""
import numpy as np


def _gauss_filter_fft(noise, corr):
    ny, nx = noise.shape
    kx = np.fft.fftfreq(nx)[None, :]
    ky = np.fft.fftfreq(ny)[:, None]
    g = np.exp(-2 * (np.pi * corr) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * g))


def grf_two_phase(nx, ny, corr, phi, seed, pad=None):
    """Thresholded Gaussian random field: phase 1 with area fraction ~phi, correlation length ~corr pixels.
    Generated on a padded grid and cropped, so the result is not periodic."""
    rng = np.random.default_rng(seed)
    p = int(4 * corr) + 4 if pad is None else pad
    f = _gauss_filter_fft(rng.standard_normal((ny + 2 * p, nx + 2 * p)), corr)[p:p + ny, p:p + nx]
    thr = np.quantile(f, 1 - phi)
    return (f > thr).astype(np.int64).ravel()


def disks_two_phase(nx, ny, radius, phi, seed, rjit=0.3, max_tries=20000):
    """Random (non-overlapping where possible) disks of radius ~radius pixels until area fraction phi."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    img = np.zeros((ny, nx), bool)
    cen = []
    tries = 0
    while img.mean() < phi and tries < max_tries:
        tries += 1
        r = radius * (1 + rjit * (2 * rng.random() - 1))
        c = rng.uniform([-r, -r], [nx + r, ny + r])
        if cen and min(np.hypot(c[0] - a, c[1] - b) - r - rr for a, b, rr in cen) < 0.5 and tries < max_tries // 2:
            continue
        cen.append((c[0], c[1], r))
        img |= (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
    return img.astype(np.int64).ravel()


def random_two_phase(nx, ny, rng, corr_range=(1.5, 6.0), phi_range=(0.2, 0.45)):
    """Mixture used for training: GRF or disks with random feature size and fraction."""
    phi = rng.uniform(*phi_range)
    seed = int(rng.integers(1 << 30))
    if rng.random() < 0.5:
        corr = rng.uniform(*corr_range)
        return grf_two_phase(nx, ny, corr, phi, seed), ("grf", corr, phi)
    rad = rng.uniform(1.5 * corr_range[0], 1.2 * corr_range[1])
    return disks_two_phase(nx, ny, rad, phi, seed), ("disk", rad, phi)


def voronoi_grains(xy, n_grains, box, seed):
    """Grain id of each point (Voronoi of uniform random seeds) and seed positions."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    seeds = rng.uniform([box[0], box[2]], [box[1], box[3]], size=(n_grains, 2))
    _, gid = cKDTree(seeds).query(xy)
    return gid, seeds
