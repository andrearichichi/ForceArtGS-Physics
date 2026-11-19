# ForceSAPIEN A3 → B1–B10 benchmark

## Protocol (preserved exactly; `validation/forcesapien_a3.py` mirrors `force_estimators/a3/post_a3.py`)

- **Objects (10):** FoldingChair 102255, Knife 103706, Laptop 10211, Oven 101917, Refrigerator 10905,
  Scissors 11100, Stapler 103111, StorageFurniture 45135, USB 100109, WashingMachine 103776.
- **A3 (identification).**
  - Physics: one square push on [0, 2 s) from rest near the closed end, on a viscous joint (c = I / 2 s, gravity off,
    no friction).
  - Kinematics: captured by 8 fixed cameras and reconstructed with the joint multicamera FreeArtGS fork. The joint and
    q(t) passed a pre-registered geometry gate (type correct, axis < 1°, q nRMSE < 1 %, hinge ≤ 5 mm).
  - WashingMachine uses a GT-free pivot-initialisation fix.
  - Estimators see only `inputs/<slug>.A3.estimator_input.json`: FreeArtGS joint and q(t), and the simulator force
    record treated as measured. No initial state and no GT are given.
- **Freeze:** (I, c) from A3.
- **B1–B10 (prediction):**
  - ten further interactions per object (same I and c, different push profiles), from the 1+10 benchmark;
  - force model τ_B = |κ_A3|·|F_B|, where κ is the least-squares ratio of the A3 generalised effort (FreeArtGS
    geometry) to |F_A3| over the active samples;
  - forward ZOH, rest at t = 0, exact solver. The B kinematics are never used.
- **Score (evaluation only):** nRMSE of the predicted displacement against push-sign·q_rel of the B simulator truth,
  normalised by its range. A3: I/c relative errors, and the q_hat nRMSE against the sign-aligned truth.
- **Aggregation:**
  - per B interaction, the median over seeds;
  - per object, the median over B1–B10;
  - the table reports the median over the 10 objects;
  - |I err| and |c err| are the medians over objects of |median over seeds|;
  - an interaction passes if its nRMSE ≤ 10 %.

The force model is sign-blind: it drops the force direction. That is valid for these B sets (single-direction pushes)
but fails for reversing pushes (laptop stress test v2: 30/100 gross failures). The general API therefore predicts with
the **signed** projection; the lever-factor model is available only as `api.predict_lever_factor` /
`validation.forcesapien_a3`.

## External bundle (not distributed)

The licensed data are excluded from this repository, history and package artifacts.
Set `FORCEARTGS_PHYSICS_BENCHMARK=/absolute/path/to/authorized/forcesapien_a3` for an independently authorized copy.
Without it, benchmark tests explicitly skip and benchmark commands explain the missing external dependency.
See [redistribution review](REDISTRIBUTION.md). The maintainer builder requires `--out` outside the repository.

External directory layout (about 6 MB):

```
inputs/<slug>.A3.estimator_input.json      byte-identical copies of the frozen A3 inputs
forces_B/<slug>.B<j>.json.gz               B force records (time, force, point, active)
truth/<slug>.{A3,B<j>}.truth.json.gz       simulator truth - EVALUATION ONLY, never opened by estimators (tested)
reference/a3_fits.json.gz                  original fits: I, c, q_hat, scales, LS start, device, runtime (all seeds)
reference/akf_improved_fits.json.gz        original AKF-improved fits incl. the selection grid
reference/b_predictions.json.gz            original B1-B10 predicted trajectories (every fit)
reference/a3_scores.json.gz, A3_RESULTS.json, AKF_STUDY_SUMMARY_B.json, akf_study_scores_B.json.gz
MANIFEST.json                              SHA-256 of all 543 source files and of every bundle file
```

## Results

| estimator | \|I err\| % | \|c err\| % | B1–B10 nRMSE % (object median) | PASS |
|---|---|---|---|---|
| PINN | 0.129 | 0.224 | **0.147** | 100/100 |
| Backprop | 0.156 | 0.247 | **0.176** | 100/100 |
| AKF (AKF-improved) | 0.155 | 0.256 | **0.184** | 100/100 |
| MLP | 0.822 | 0.394 | **0.512** | 100/100 |
| AKF-benchmark (original A3 row) | 2.862 | 2.982 | 2.350 | 100/100 |

Reproduce: `python examples/02_forcesapien_benchmark.py` scores the original fits;
`--refit Backprop AKF` re-fits on CPU; `--refit PINN MLP --device cuda` re-fits on GPU.

**Interpretation limits** (from the original reports):
- The inter-interaction spread is tiny, so the effective sample size is the 10 objects.
- The paired differences between PINN, Backprop and AKF-improved were not shown to be significant in earlier
  benchmarks (paired CIs contain 0).
- The B sets had been used for evaluation before. B results are therefore exploratory for AKF-improved, whose
  selection rule was written afterwards.
