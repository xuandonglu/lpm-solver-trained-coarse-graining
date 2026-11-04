"""Environment of the 2D timing scripts (numpy / scipy / numba, and torch for inference).  The fine reference block
fields are written by el_run.py to ./data/fields; the network weights are read from ../../coarse2d/models.  Thread
budget LPMCG_THREADS (default 6); numba cache -> ./_numba_cache; no .pyc anywhere."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
I2MODELS = os.path.normpath(os.path.join(HERE, "..", "..", "coarse2d", "models"))
sys.dont_write_bytecode = True
NT = os.environ.get("LPMCG_THREADS", "6")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "_numba_cache")
os.environ["NUMBA_NUM_THREADS"] = NT
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = NT
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
import common  # noqa: E402,F401

TAGS = {"comp_48": ("comp", 48), "comp_96": ("comp", 96), "comp_192": ("comp", 192), "comp_384": ("comp", 384)}
PT = os.environ.get("LPMCG_TORCH_PYTHON", sys.executable)
for _d in ("results", "logs", "data"):
    os.makedirs(os.path.join(HERE, _d), exist_ok=True)


def here(*p):
    return os.path.join(HERE, *p)


def field_file(tag, seed, s, method):
    kind, n = TAGS[tag]
    return here("data", "fields", tag, "%s_%d_%d_s%d_%s.npz" % (kind, n, seed, s, method))


def prob_of(tag, seed):
    import el_prob as EP
    kind, n = TAGS[tag]
    return EP.make_comp(n, seed)
