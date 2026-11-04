# Solver-trained bond parameters for coarse-grained lattice particle models

The lattice particle model (LPM) represents a heterogeneous material with one particle per pixel or voxel. Neighbouring
particles are connected by bonds with a pair coefficient (Kn) and a multibody coefficient (Tv), and fracture is
followed by breaking bonds. The cost grows with the number of particles. This code coarse-grains the LPM: one coarse
particle represents an s × s block (2D) or an s × s × s block (3D) of fine particles, and the coarse lattice is again
an LPM.

Every coarse bond receives its own Kn and Tv from a bond network. The network reads the fine microstructure in a
window around the bond, made of the two end blocks and one ring of neighbouring blocks, and it is exactly equivariant
under the symmetries of the lattice. It predicts corrections to the Voigt values and is trained through a
differentiable coarse LPM: the coarse model is solved for several load cases, its displacements, stresses and global
stiffness are compared with the block-averaged fine LPM solution, and the gradients reach the network through the
adjoint of the coarse solve. No target values of the bond parameters are needed. Because the coarse model remains an
LPM, it can be loaded in any way and carried into fracture with learned bond strengths.

## Cases

The coarse models are compared with the Voigt rule, patch homogenization and a per-specimen fit, in which the bond
parameters are fitted to the fine solution of the same specimen. The reference is always the block-averaged solution
of the fine LPM.

| Folder | Case |
|---|---|
| `coarse2d` | Two-phase composites on the square lattice: random-field and disk inclusions, inclusion fraction 0.15–0.40, stiffness contrast 5 and 10, coarsening factors s = 2, 3, 4. Plates in tension and shear and kinematic uniform boundary conditions (KUBC). Held-out sets with inclusion fraction 0.5, contrast 15 and twice the size. |
| `coarse2d/window_calibration`, `coarse2d/lookup_table` | Two other ways to obtain the coarse bond parameters: local window calibration, in which each bond is fitted on an isolated window, and a configuration lookup table trained through the coarse solver. |
| `coarse3d` | Two-phase materials on the simple cubic lattice with spherical and ellipsoidal inclusions, s = 2, 3, nine load cases. Transfer to specimens of 48³ and 72³ voxels, elongated inclusions and other contrasts. |
| `coarse3d/ablations` | A stress-field surrogate that maps the voxel image directly to the coarse stress field without an LPM solver, and bond networks without built-in equivariance. |
| `cement3d` | Hardened cement paste from two X-ray tomography scans (water-to-cement ratios 0.30 and 0.40; hydrate, portlandite, clinker and pores). Pores are either empty or treated as porous hydrate. The networks are trained on synthetic cement-like microstructures, on real cubes of the other scan, or not retrained. |
| `fracture2d` | Brittle fracture of unnotched two-phase plates in tension. The coarse bonds follow a two-parameter law (critical elongation and ductility, both predicted by networks; the critical-elongation network is in `coarse2d`). Strength-first and energy-first classical coarse models serve as baselines. |
| `timing` | Wall time of the coarse model and the fine LPM in 2D and 3D, with CPU and GPU inference, the unpreconditioned conjugate gradient solver of the LPM and tuned algebraic multigrid solvers. |

Each script changes into its own folder, so it can be started from any directory. Data, results and logs are written
to `data/`, `results/` and `logs/` next to the script, and `models/` holds the trained networks. The trained networks
are included (`coarse2d/models`, `coarse2d/lookup_table/models`, `coarse3d/models`, `coarse3d/ablations/models`,
`cement3d/models`, `fracture2d/models`), so the evaluations run without retraining. The data sets are not included;
the scripts below generate them from fixed random seeds. The cement-paste case needs two published tomography scans
(see `data/README.md`).

## Requirements

Two Python environments are used, listed in `requirements.txt`: a numerical environment (NumPy, SciPy, Numba) for
data generation, fine reference solutions in 2D, patch homogenization, evaluation, fracture and timing, and a torch
environment (PyTorch with CUDA) for training, the 3D fine reference solutions and network inference. One environment
with all packages also works.

```bash
pip install -r requirements.txt
```

Training and the 3D reference solutions need a CUDA GPU. GPU computations are not bitwise reproducible, so trained
networks and fitted bond parameters can differ slightly from run to run; evaluations with the included networks are
reproducible. Thread counts can be set with `LPMCG_THREADS` (timing scripts) and the usual `OMP_NUM_THREADS` /
`NUMBA_NUM_THREADS` variables. The timing scripts start the network inference in a separate process, and
`LPMCG_TORCH_PYTHON` names its Python interpreter (default: the interpreter that runs the benchmark).

## Running the cases

"(num)" marks the numerical environment and "(torch)" the torch environment.

### Two-dimensional composites

```bash
# data: microstructures, fine reference solutions, Voigt and patch cell parameters (num)
for split in "train 1200" "val 100" "test_id 100" "test_phi 100" "test_c7 100" "test_c15 100" "test_big 40"; do
  python coarse2d/export_data.py $split
done
# bond networks: prepared tensors and training through the coarse LPM, s = 2, 3, 4 (torch, GPU)
for s in 2 3 4; do
  python coarse2d/train_cg.py prep $s train val test_id test_phi test_c7 test_c15 test_big
  python coarse2d/train_cg.py train $s --epochs 40 --bs 16 --lr 1e-3      # writes models/cgnet_s$s.{pt,npz}
done
# Voigt, patch and bond network on the test and held-out sets (num)
for s in 2 3 4; do python coarse2d/eval_elastic.py $s test_id test_phi test_c15 test_big; done
python coarse2d/check_gradient.py                                          # adjoint vs finite differences (torch)
# per-specimen fit on the first 32 test specimens (torch) and its errors (num)
for s in 2 3 4; do
  python coarse2d/window_calibration/oracle.py test_id $s --n1 32
  python coarse2d/window_calibration/eval_rungs.py test_id $s l2b --nmax 32
done
```

### Local window calibration and configuration lookup table

```bash
# local window calibration: window targets (num), joint fit (torch), errors (num)
for s in 2 3 4; do
  python coarse2d/window_calibration/l2run.py targets test_id $s 1 --n1 100 --workers 5
  python coarse2d/window_calibration/l2run.py fit test_id $s 1 1 joint --iters 50 --outer 4 --bs 8
  python coarse2d/window_calibration/eval_rungs.py test_id $s voigt patch cnn l2j_m1
done
python coarse2d/window_calibration/bench_cost.py --nspec 5                 # cost per specimen, one thread (num)
# fit continued to convergence on the first 32 specimens (torch, then num)
for s in 2 3 4; do
  python coarse2d/window_calibration/convergence/run1.py coarse2d/window_calibration/convergence/l2conv_full.py $s --n1 32 --warm --maxouter 10
  python coarse2d/window_calibration/eval_rungs.py test_id $s l2jc_m1 --nmax 32
done
python coarse2d/window_calibration/make_report.py      # errors, cost per specimen and the gap to the converged fit (num)
# configuration lookup table trained through the coarse LPM (torch)
for s in 2 3 4; do
  python coarse2d/lookup_table/lut2d.py train frac $s
  python coarse2d/lookup_table/hier2d.py train $s
  python coarse2d/lookup_table/hier2d.py eval $s
  python coarse2d/lookup_table/lut2d.py eval cnn $s
done
```

### Three-dimensional two-phase materials

```bash
for split in train val cal test_id test_elong test_c8 test_soft test_big test_huge; do
  python coarse3d/gen_micro.py $split          # microstructures and patch homogenization (num)
  python coarse3d/gen_fine.py $split           # fine reference, PCG to 1e-10 (torch, GPU)
done
for s in 2 3; do
  python coarse3d/train3d.py prep $s train val cal test_id test_elong test_c8 test_soft test_big test_huge
  python coarse3d/train3d.py train $s --epochs 30                          # writes models/bondnet_s$s.pt
  python coarse3d/train3d.py eval $s test_id test_elong test_c8 test_soft test_big test_huge
done
python coarse3d/train3d.py oracle 2 --epochs 400 --lr 0.005                # per-specimen fit, 16 validation cubes
python coarse3d/train3d.py oracle 3 --epochs 300 --lr 0.005
python coarse3d/floor3d.py                                                  # homogeneous cube: free-surface layer
python coarse3d/dev/export_check.py && python coarse3d/check_torch3d.py     # operator and adjoint checks (num, torch)

# stress-field surrogate and networks without built-in equivariance (torch)
python coarse3d/ablations/puredata.py train --epochs 200
python coarse3d/ablations/puredata.py eval
python coarse3d/ablations/ablate.py train ne ne_aug --aug
python coarse3d/ablations/ablate.py train ne ne_noaug
python coarse3d/ablations/ablate.py eval eq eq_i8 --i8 --orient
python coarse3d/ablations/ablate.py eval ne ne_aug --orient
python coarse3d/ablations/ablate.py eval ne ne_noaug --orient
python coarse3d/ablations/orient_dev.py
```

### Hardened cement paste

Download the two scans into `data/` first (see `data/README.md`).

```bash
python cement3d/cubes.py                                   # test, calibration and training cubes of both scans (num)
python cement3d/gen_synth.py make train 1000 24 101        # synthetic cement-like microstructures (num)
python cement3d/gen_synth.py make val 50 24 102
python cement3d/gen_synth.py chords                        # phase fractions and mean chord lengths, synthetic and scan cubes
SETS="w30_test24 w30_test48 w30_test72 w30_cal24 w40_test24 w40_test48 w40_test72 w40_cal24"
TRAIN="syn_train syn_val w30_train24 w30_val24 w40_train24 w40_val24"
python cement3d/prep_mat.py $SETS $TRAIN                   # materials, pore treatments a (empty) and b (porous hydrate) (num)
python cement3d/floppy_v1.py $SETS $TRAIN                  # removal of zero-energy modes (torch, GPU)
python cement3d/gen_patch_v1.py $SETS                      # patch homogenization (num)
for set in $SETS; do python cement3d/gen_fine_v1.py $set; done            # fine reference (torch, GPU)
for set in $TRAIN; do python cement3d/gen_fine_v1.py $set --s 2; done
for opt in a b; do
  python cement3d/train_v1.py prep 2 $SETS $TRAIN $opt
  python cement3d/train_v1.py train 2 $opt syn_$opt --train syn_train --val syn_val
  python cement3d/train_v1.py train 2 $opt r30_$opt --train w30_train24 --val w30_val24
  python cement3d/train_v1.py train 2 $opt r40_$opt --train w40_train24 --val w40_val24
  python cement3d/train_v1.py eval 2 $opt i8,syn_$opt,r30_$opt,r40_$opt w30_cal24 w30_test24 w30_test48 w30_test72 --out all_w30
  python cement3d/train_v1.py eval 2 $opt i8,syn_$opt,r30_$opt,r40_$opt w40_cal24 w40_test24 w40_test48 w40_test72 --out all_w40
  for set in w30_test24 w40_test24; do
    python cement3d/train_v1.py oracle 2 $opt $set --nfit 12 --its 400 --init-tag syn_$opt --lr 0.002
  done
done
python cement3d/report_v1.py                               # summary of the errors (num)
```

### Brittle fracture

```bash
# critical elongation: in-situ targets (num), network (torch), elastic network with classical strength (num)
for split in train val test_id; do python coarse2d/export_frac.py $split; done
for s in 2 3 4; do python coarse2d/train_frac.py train $s --epochs 60; done   # writes models/fracnet_s$s.{pt,npz}
python coarse2d/eval_frac.py test_id 30
# ductility: patch fracture tests (num) and network (torch)
python fracture2d/patch_frac.py train --nmax 400
python fracture2d/patch_frac.py val
for s in 2 3 4; do python fracture2d/train_rho.py train $s --epochs 40; done # writes models/rhonet_s$s.{pt,npz}
# plates: strength-first and energy-first classical models and the two-parameter law (num)
for split in "test_id 30" "test_phi 30" "test_c15 30" "test_big 10"; do
  python fracture2d/gen_specimens.py $split
  python fracture2d/eval2.py $split
done
python fracture2d/make_tables2.py                          # errors and speed-ups of all splits
python fracture2d/plate_curves.py 0                        # force-strain curves and broken bonds of plate 0
```

`fracture2d/results/phase_energy.json` holds the constants of the energy-first model: the fracture energies of the
matrix, the inclusions and the matrix-inclusion interface, measured with crack-growth tests in a K-field on
homogeneous and bimaterial fine lattices of 96 × 96 particles, and the ratio of the fracture energy of a brittle coarse
lattice to that of the fine lattice for each coarsening factor.

### Timing

```bash
# 2D: fine reference fields of the larger specimens (num), then the benchmark
for n in "96 0 6" "192 0 4" "384 0 2"; do python timing/t2d/el_run.py comp $n --tag comp_${n%% *} --ref lu; done
python timing/t2d/bench2d.py 48 --seeds 0 1 2 3 4 5 6 7 8 9 --reps 3
python timing/t2d/bench2d.py 96 --seeds 0 1 2 3 4 5 --reps 3
python timing/t2d/bench2d.py 192 --seeds 0 1 2 3 --reps 2
python timing/t2d/bench2d.py 384 --seeds 0 1 --reps 2 --rows F-trad,F-amg,NN2-trad,NN4-trad,V2-trad,P2-trad,V4-trad,P4-trad
# 3D two-phase cubes and cement-paste cubes (num; the inference worker uses LPMCG_TORCH_PYTHON)
python timing/t3d/bench3d.py i8 test_id 10 --reps 3
python timing/t3d/bench3d.py i8 test_big 5 --reps 3
python timing/t3d/bench3d.py i8 test_huge 4 --reps 2
for opt in a b; do
  for set in "w30_test48 5" "w40_test48 5" "w30_test72 4" "w40_test72 4"; do
    python timing/t3d/bench3d.py v1 ${set% *} $opt ${set#* } --reps 2
  done
done
python timing/summ_timing.py                               # summary (results/timing_tables.md, timing_summary.json)
```

Timings depend on the machine. The reference timings were measured on one workstation (AMD Ryzen 9 9950X3D, NVIDIA
GeForce RTX 5090) with six threads for the solvers and the CPU inference.

## Names used in the code

Comments and some variable names use internal names of the study steps: I2 (two-dimensional coarse-graining,
`coarse2d`), I2B (fracture, `fracture2d`), I2C and L2 (window calibration, `coarse2d/window_calibration`), I8 and I8b
(three-dimensional coarse-graining and its solvers, `coarse3d`, `timing/t3d`), V1 (cement paste, `cement3d`).
"Oracle" is the per-specimen fit, "patch" is patch homogenization, "ring" denotes the load cases with kinematic
uniform boundary conditions (KUBC), "tb" and "lr" the plate load cases, and "super-particle" is a coarse particle.
