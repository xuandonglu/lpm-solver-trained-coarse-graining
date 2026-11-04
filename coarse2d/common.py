"""Environment of the 2D scripts."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
