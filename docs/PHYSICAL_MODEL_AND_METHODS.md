# Physical model and the four estimators

## 1. Model (ForceSAPIEN model, used by every validated estimator)

One degree of freedom q (rad for a revolute joint, m for a prismatic joint):

    I q''(t) + c q'(t) = τ(t)

| symbol | revolute | prismatic |
|---|---|---|
| q | rad | m |
| I | kg·m², inertia of the moving part about the joint axis | kg, effective mass |
| c | N·m·s/rad, viscous damping | N·s/m |
| τ | N·m, τ = a·((p − o) × F) (+ a·M) | N, τ = a·F |

Here a is the unit joint axis, o a point on the axis, p the application point, F the force on the moving part and M an
optional pure couple. τ is a forward zero-order hold: sample k is held on [t_k, t_{k+1}), and the first sample is
also held back to t = 0. Positive τ drives positive q.

The decay time is T = I/c. The model has **no** gravity, Coulomb or static friction, joint limits or contact model:
the ForceSAPIEN simulations were run viscous-only with gravity off. These terms are separate extensions for
real data (§4).

The exact solution for piecewise-constant τ (`dynamics.viscous`) is used for prediction by every method:

    v(h) = v e^{−x} + (τ/I) h φ1(x),   q(h) = q + v h φ1(x) + (τ/I) h² φ2(x),   x = (c/I) h
    φ1 = (1 − e^{−x})/x,   φ2 = (x − 1 + e^{−x})/x²   (series for x < 1e-2, cancellation-free)

## 2. Common preprocessing of the validated pipeline

Every estimator sees the same inputs, and none receives the initial state: offset and initial velocity are nuisance
unknowns. Steps:
1. **Inputs:** observation times and q (all samples), the force record, joint type, axis and origin.
2. **Nondimensionalisation** (`pinn_nd.nondimensionalize`):
   - t ← (t − t0)/T with T = the observation span;
   - q ← (q − q0)/Q with Q = range(q);
   - force ← F/F_s with F_s = max|τ| over the observation window.

   Parameters map back as I = I_nd F_s T²/Q and c = c_nd F_s T/Q. The force after the last observation is discarded.
3. **Shared integral-LS initializer** (`common.ls`): integrate the equation twice, which needs no differentiation of noisy q:

       I (q − q0) + c ∫(q − q0) − I v0 (t − t0) + const = ∬ τ

   This is linear in (I, c, I v0, const). It is solved by column-scaled least squares, with a positivity-constrained
   fallback.

## 3. Estimators

### PINN (PINN-V2)
A trajectory network q_θ(t) is used: 4 × 64 tanh, with hard q(t0) = q_obs(t0). It is trained jointly with log I and
log c on

    L = mean (q_θ(t_i) − q_i)² + mean_j [ (I (q_θ'(t_j) − q_θ'(t0)) + c (q_θ(t_j) − q_θ(t0)) − J(t_j)) / rms(J) ]² + IC

where J(t) = ∫ τ is exact for the ZOH force. This first-integral residual needs only q', which is continuous; the
strong form needs a q'' that jumps at the force edges, and that was the PINN-V1 failure mode. Schedule:
- parameters start at the integral-LS estimate;
- 2000 data-only Adam steps, then 20 000 joint Adam steps (lr 1e-3);
- up to 10 000 L-BFGS iterations (chunks of 100, strong Wolfe) with the V1 stopping rule;
- 4096 uniform collocation points, float64.

Seeds 0, 1, 2; the reported (I, c) are the medians over seeds. Frozen in robust_v2 `METHODS_FREEZE.json`.

### Backprop
Least squares on q through the exact ZOH solution, with offset and initial velocity solved in closed form at every
evaluation (`expB_cells.zoh_basis` + `_ls_offset_velocity`). Optimisation runs in (log I, log T):
- 35 starts on the SUPPORT grid (log10 I_nd ∈ [−3, 3], log10 T_nd ∈ [−2, 2]);
- 3000 Adam steps (lr 0.05) with bound clamping;
- L-BFGS-B refinement of the best 8 (500 iterations);
- selection by training loss.

CPU, deterministic.

### MLP (MLP-Joint)
Amortised and label-free, trained **per interaction**. 20 000 synthetic trajectories are generated with:
- the same measured effort;
- the exact ZOH model;
- parameters from the generic SUPPORT prior;
- a random initial velocity;
- none/white/AR(1) noise.

Network: an N-256-256-128 SiLU trunk, a parameter head (bounded log T, nondimensional log I) and a J-invariant motion
head. Loss: L_par + L_obs + 0.3 L_phys. These are reconstruction losses through the ZOH projection; no parameter
labels are used. Training: AdamW 1e-3, cosine schedule, 200 epochs, early stopping on synthetic validation (patience 40).
Seeds 0–4, median. There are no external pre-trained weights: the "pretraining" *is* this per-interaction synthetic
training. Checkpoints go to `workdir`.

### AKF (AKF-improved, definitive)
An augmented extended Kalman filter with state [q, q', log I, log c]:
- RK4 transition with the analytic variational Jacobian (step ≤ 1/240 s);
- Joseph-form update.

Configuration:
- process noise: position 0.1 σ, velocity σ, parameters 0;
- σ: robust local-interpolation noise of q (floor 1e-4 nondimensional).

The filter starts at the integral-LS (I, c); q0 is the first observation. The measurement std m_R·σ and the
initial log-parameter std P0 are chosen over m_R ∈ {1, 2, 5, 10, 20, 50, 100} × P0 ∈ {ln 100, ln 10} by the
open-loop replay error of the final (I, c) on the **training** interaction. The output is the final filtered (I, c).

`AKF-benchmark` (the original A3 row) uses a 35-start innovation-NLL search instead, with R = σ and P0 = ln 100.

### IntegralLS (reference)
The shared closed-form initializer, exposed as a fast baseline. `fit_many` stacks several interactions with shared
(I, c) and per-interaction nuisance terms.

## 4. Real-data extensions (separate, not used by the four estimators)

`dynamics.extensions`:

    I q'' + c q' + Fc s(q') + g(q) + b = τ

The terms are:
- Coulomb friction Fc, with s = sign or tanh(q'/ε);
- static (Karnopp) breakaway level Fs;
- a configuration load g(q), for example lid gravity, as a user callable;
- a constant effort b (sensor bias or a constant load).

`simulate` integrates the model. `identify_extended_ls` fits it by integral LS and returns a standard error, a
correlation matrix and an explicit `identifiable` flag per term. It reports a term as not identifiable instead of
returning a number for it.

## 5. Identifiability (`validation.identifiability.diagnose`)

This is a GT-free diagnostic computed at the integral-LS estimate. It is the robust_v2 `estimator_side` diagnostic, with
the reliability threshold pre-registered in `METHODS_FREEZE.json`:
- conditioning of the integral regressor;
- the Fisher information of q(t_obs) with respect to (log I, log c, v0, q0) for white noise at the LS residual level;
- the Cramér–Rao relative std of I and c, and corr(log I, log c).

Verdicts:

| condition | verdict |
|---|---|
| kinematic status ≠ identified | not identifiable (no estimate) |
| no applied effort while q is observed | unobservable |
| CRLB(I) ≤ 10 % and CRLB(c) ≤ 10 % | I and c separately identifiable |
| otherwise | NOT separately identifiable for the failing parameter(s); "only a combination" if \|corr\| > 0.95 |

The bound is local and assumes white q noise. Correlated visual errors and model errors are not in it, so passing it
is necessary, not sufficient, for an accurate estimate. The estimators still run when it fails; the verdict is
attached to the diagnostic result (`parameters.identifiable`, `parameters.uncertainty`). Physical parameter export
and PArtTwin export raise `NotIdentifiableError` unless both result and parameter verdicts pass. Diagnostic JSON
reports retain candidate estimates for research inspection; they are not deployable physics.
