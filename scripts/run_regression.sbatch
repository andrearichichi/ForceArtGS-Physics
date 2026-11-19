#!/bin/bash
#SBATCH --job-name=fagp_regression
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=01:00:00
# Full regression of ForceArtGS-Physics on one GPU node (account/partition/output on the command line):
#   sbatch -A <account> -p <gpu partition> -o <log> scripts/run_regression.sbatch <python> <out_dir> [original_root]
# MODE=cpu (parts 1-2, no GPU needed: e.g. -p <serial partition> --gres=none -c 4) | gpu (part 3) | all (explicit opt-in)
# JOBS = max concurrent CPU processes for part 2 (default: SLURM_CPUS_PER_TASK)
# 1. fast suite (schemas, dynamics, forces, benchmark scoring on the original fits, identifiability, ForceArtGS e2e,
#    provenance - text identity with the originals when original_root is given)
# 2. CPU re-fits (Backprop, AKF, AKF-benchmark) of the 10 A3 objects, one process per object
# 3. GPU re-fits (PINN seed 0, MLP seed 0) of $GPU_OBJECTS (default Knife_103706), concurrently on the GPU
set -uo pipefail
PY="$1"; OUT="$2"; ORIG="${3:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; [ -f "$REPO/pyproject.toml" ] || REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"; mkdir -p "$OUT"
MODE="${MODE:-cpu}"; JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# src first on the path: works with an editable install AND with a plain interpreter (e.g. the reference environment)
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
[ -n "$ORIG" ] && export FORCEARTGS_PHYSICS_ORIGINAL_ROOT="$ORIG"
echo "host $(hostname) start $(date -Is) python $PY repo $REPO"; nvidia-smi -L || true
"$PY" -c "import sys, torch, numpy, scipy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'cuda', torch.cuda.is_available())"
P0=""; G1=""; G2=""; pids=()
if [ "$MODE" != gpu ]; then
  "$PY" -m pytest -p no:cacheprovider -m "not slow and not gpu" -ra --junitxml="$OUT/fast.xml" > "$OUT/fast.log" 2>&1 &
  P0=$!
fi
export FORCEARTGS_PHYSICS_REFIT_OUT="$OUT/refit"
GPU_OBJECTS="${GPU_OBJECTS:-Knife_103706}"
GPU_METHODS="${GPU_METHODS:-PINN MLP}"      # subset, e.g. GPU_METHODS=MLP
if [ "$MODE" != cpu ]; then
  if [[ " $GPU_METHODS " == *" PINN "* ]]; then
    FORCEARTGS_PHYSICS_OBJECTS="$GPU_OBJECTS" "$PY" -m pytest -p no:cacheprovider -m gpu -k PINN -ra --junitxml="$OUT/gpu_pinn.xml" > "$OUT/gpu_pinn.log" 2>&1 &
    G1=$!
  fi
  if [[ " $GPU_METHODS " == *" MLP "* ]]; then
    FORCEARTGS_PHYSICS_OBJECTS="$GPU_OBJECTS" "$PY" -m pytest -p no:cacheprovider -m gpu -k MLP -ra --junitxml="$OUT/gpu_mlp.xml" > "$OUT/gpu_mlp.log" 2>&1 &
    G2=$!
  fi
fi
if [ "$MODE" != gpu ]; then
  for s in FoldingChair_102255 Knife_103706 Laptop_10211 Oven_101917 Refrigerator_10905 Scissors_11100 Stapler_103111 StorageFurniture_45135 USB_100109 WashingMachine_103776; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    FORCEARTGS_PHYSICS_OBJECTS="$s" "$PY" -m pytest -p no:cacheprovider -m slow -ra --junitxml="$OUT/slow_$s.xml" > "$OUT/slow_$s.log" 2>&1 &
    pids+=("$!")
  done
fi
rc=0
for p in $P0 "${pids[@]}" $G1 $G2; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
for f in "$OUT"/*.log; do echo "== $f"; tail -3 "$f"; done
echo "END $(date -Is) rc=$rc"
exit $rc
