# Cement-paste scans (not included)

The cement-paste case uses two segmented X-ray computed tomography scans of hardened cement paste with water-to-cement ratios of
0.30 and 0.40. They are published by Limtong et al. under the CC BY 4.0 license at
https://doi.org/10.5281/zenodo.8112442, and the underlying images come from the NIST Visible Cement Data Set.

Download the two files below from the Zenodo record and place them in this folder:

| File | Size | Content |
|---|---|---|
| `segmented_030_412_upload.npy` | 69,934,656 bytes | w/c 0.30, 412³ voxels of 1 µm, uint8 |
| `segmented_040_550_void_extracted.npy` | 166,375,128 bytes | w/c 0.40, 550³ voxels of 1 µm, uint8 |

Labels: 0 pore, 1 hydrate (dominated by calcium silicate hydrate), 2 portlandite, 3 unhydrated clinker, 4 large air
void (w/c 0.40 only). `cement3d/cubes.py` reads the files with `np.load(..., allow_pickle=False)` and checks that each
is a three-dimensional uint8 array.
