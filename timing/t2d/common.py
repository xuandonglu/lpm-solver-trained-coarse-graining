"""Environment defaults of the copied 2D modules (a2common sets the real values first)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "6")
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
