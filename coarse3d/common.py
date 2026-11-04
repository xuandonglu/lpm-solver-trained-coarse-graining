"""Environment for the I8 (3D SC) scripts run in the numerical environment (numpy/scipy/numba).

* numba cache inside this folder (no .pyc);
* at most 8 threads (numba + BLAS), shared machine;
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
