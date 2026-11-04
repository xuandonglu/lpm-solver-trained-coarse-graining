"""Environment of the V1 scripts (real cement-paste morphology).  Works in both Python environments:
the numerical environment (numpy / scipy / numba) and the torch environment.

* the 3D code of ../coarse3d is imported read-only: no .pyc is written anywhere (sys.dont_write_bytecode) and
  every numba cache goes to ./_numba_cache (also for the patch3d kernels);
* at most 8 threads (numba + BLAS + torch), shared machine;
* the downloaded scans (../data, see ../data/README.md) are read with np.load(..., allow_pickle=False) only;
  derived arrays live in ./data.
This module shadows ../coarse3d/common.py for the modules imported from there (patch3d does `import common`): same
thread limits, cache directory moved here.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
I8 = os.path.normpath(os.path.join(HERE, "..", "coarse3d"))
SCANS = os.path.normpath(os.path.join(HERE, "..", "data"))
sys.dont_write_bytecode = True
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if I8 not in sys.path:
    sys.path.append(I8)
for _d in ("data", "models", "results", "logs"):
    os.makedirs(os.path.join(HERE, _d), exist_ok=True)


def data(name):
    return os.path.join(HERE, "data", name)


def i8data(name):
    return os.path.join(I8, "data", name)
