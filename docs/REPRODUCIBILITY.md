# Reproducibility and regression tolerances

## Reference environment
The reference results were produced on CINECA Leonardo:
- Python 3.11.13, torch 2.5.1+cu121, numpy 1.26.4, scipy 1.17.1;
- `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`;
- Backprop and AKF on CPU; PINN-V2 and MLP-Joint on NVIDIA A100-SXM-64GB (10 fits sharing one GPU).

`pip install -e .[reference]` pins those library versions. The estimators set one thread by default
(`EstimatorConfig.threads=1`).

Observed stability (2026-11-19). The 30 CPU re-fits (Backprop, AKF, AKF-benchmark × 10 objects) were **bitwise**
identical to the original fits in two environments:
- the reference environment;
- a plain `pip install -e .`, which pulled torch 2.15.0 and numpy 2.2.6.

In the reference environment the GPU re-fits of PINN and MLP (seed 0, Knife, A100) were also bitwise identical. Under
torch 2.15.0, PINN differed by −0.028 % (I) and −0.014 % (c).
See VALIDATION_REPORT.md.

## What is compared, and with which tolerance

| check | test | tolerance | motivation |
|---|---|---|---|
| B1–B10 trajectories from the original (I, c) | `test_benchmark_reference` | 1e-12 × range (observed: identical) | same solver code, deterministic |
| per-object scores, κ, A3_RESULTS, AKF summary | `test_benchmark_reference` | rel 1e-12 (observed: identical) | same arithmetic |
| nondimensional scales, integral-LS start | `test_benchmark_reference` | exact equality | same code, same inputs |
| generalized effort vs stored effort | `test_forces` | 1e-12 × max\|τ\| | axis re-normalisation can change the last bit |
| Backprop / AKF / AKF-benchmark re-fit (10 objects) | `test_refit -m slow` | rel 1e-9 on I, c, q_hat (bitwise expected and recorded) | CPU, deterministic; margin only for a different BLAS/libm build |
| PINN / MLP re-fit (seed 0) | `test_refit -m gpu` | \|ΔI/I\|, \|Δc/c\| ≤ max(1e-3, ½ × the original across-seed range) | GPU results depend on the torch/CUDA build (bitwise in the reference env, −0.03 % under torch 2.15); a difference smaller than half the seed spread cannot change any reported median |
| vendored text vs originals | `test_provenance` (original tree) | exact text identity | definition of "unchanged" |

The headline numbers are object medians of seed medians. A difference in one GPU seed that stays within the seed
spread cannot move them by more than that spread.

## Re-running

Benchmark checks require an authorized external bundle via `FORCEARTGS_PHYSICS_BENCHMARK`.
No benchmark payload is included in this repository. Without it those checks explicitly skip.

```bash
pytest                                    # fast suite
pytest -m slow                            # CPU re-fits (≈ 3-5 min per object on one core)
pytest -m gpu                             # GPU re-fits (FORCEARTGS_PHYSICS_OBJECTS=<slug,...>)
sbatch -A <account> -p <partition> scripts/run_regression.sbatch <python> <out> [original_root]
```
`FORCEARTGS_PHYSICS_REFIT_OUT=<dir>` writes one JSON per re-fit with the reference values and the differences.
`docs/VALIDATION_REPORT.md` summarises them.
