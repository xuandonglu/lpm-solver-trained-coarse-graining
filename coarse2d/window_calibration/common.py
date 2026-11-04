"""I2c credit ladder: environment and paths.

The I2 data (fine references, Voigt / patch cell parameters, prepared torch tensors) and the I2 network weights are
READ from ../{data,models} (the coarse2d folder); nothing there is written.  All new files go to this folder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
I2 = os.path.normpath(os.path.join(HERE, ".."))
I2DATA = os.path.join(I2, "data")
I2MODELS = os.path.join(I2, "models")
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
for _d in (DATA, RES, MODELS, os.path.join(HERE, "logs")):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def split_file(split):
    return os.path.join(I2DATA, "%s_n%d.npz" % (split, 96 if split == "test_big" else 48))
