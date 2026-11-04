"""Environment of the 3D timing scripts, both Python environments.

* local copies of the 3D modules (geom3d, lin3d, patch3d, net3d, amg3d, gmg, problems, infer_fast, problems_v1,
  amg3d_v, patch_v) live here; ../../coarse3d and ../../cement3d are only READ (data, model weights);
* no .pyc anywhere (sys.dont_write_bytecode); numba cache in ./_numba_cache;
* thread budget LPMCG_THREADS (default 6: numba + BLAS + torch), set before numpy loads;
* the inference worker runs in the torch environment LPMCG_TORCH_PYTHON (default: this interpreter).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
I8 = os.path.join(ROOT, "coarse3d")
V1 = os.path.join(ROOT, "cement3d")
EQ = os.path.join(ROOT, "coarse3d", "ablations")
sys.dont_write_bytecode = True
NT = os.environ.get("LPMCG_THREADS", "6")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "_numba_cache")
os.environ["NUMBA_NUM_THREADS"] = NT
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = NT
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PT = os.environ.get("LPMCG_TORCH_PYTHON", sys.executable)
for _d in ("results", "logs"):
    os.makedirs(os.path.join(HERE, _d), exist_ok=True)


def i8data(name):
    return os.path.join(I8, "data", name)


def v1data(name):
    return os.path.join(V1, "data", name)


def data(name):          # used by copied I8b code (problems / bench): the I8 data folder
    return i8data(name)
