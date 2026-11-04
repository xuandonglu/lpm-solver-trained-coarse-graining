"""run a script with torch / BLAS limited to 1 CPU thread: python run1.py SCRIPT.py args..."""
import os, sys, runpy
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["I2C_THREADS"] = "1"
import torch
_o = torch.set_num_threads
torch.set_num_threads = lambda n: _o(1)
torch.set_num_threads(1)
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
