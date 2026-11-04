"""Environment of the fracture scripts.  Data and networks of the elastic step are read from ../coarse2d/{data,models}
(read-only); new data goes to ./data."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
I2 = os.path.normpath(os.path.join(HERE, "..", "coarse2d"))
I2DATA = os.path.join(I2, "data")
I2MODELS = os.path.join(I2, "models")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np


def i2data(name):
    return os.path.join(I2DATA, name)


def i2model(name):
    return os.path.join(I2MODELS, name)
