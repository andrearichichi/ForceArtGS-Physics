# Real data: Hoi! and REAL

ForceSAPIEN is the reference benchmark. The ForceSAPIEN model and the four estimators are **not** modified for real data.
Real-data effects are handled by explicit preprocessing (`forces.preprocessing`), separate model extensions
(`dynamics.extensions`) and the identifiability diagnostic. What follows summarises what the earlier work on these
datasets measured. Those findings set what the library may and may not claim.

## Hoi! (gripper sessions with a wrist F/T sensor)

- **Data.** Hoi! has **no q(t) ground truth**: only Leica axis/origin/type, evaluation only, and open/close windows.
  22 of 25 gripper sessions record a wrist F/T sensor: raw `wrench`, driver-compensated `wrench_ext` and an IMU, at 100 Hz.
- **Existing pipeline.** Our `freeartgs-hoi` force pipeline (`multicam/force/`) builds the sensor-to-world chain and
  estimates the lag (−33 to +11 ms). It validates the sign rule (force on object = −ext) by Newton's third law and
  positive work, 6/6 interactions. It writes per-interaction CSVs with q from the gripper TCP on the packaged joint
  and the signed generalized force with an uncertainty envelope (axis ±2°, hinge ±22 mm, lag ±5 ms).
  `adapters.hoi.load_force_csv` reads them into an `Interaction`. It refuses the sealed test locations (wardrobe_1,
  bedroom_10, kitchen_14, livingroom_1).
- **Measured limits:**
  - tool inertia is not compensated: 5–14 % of the applied force;
  - bias drifts 0.36–1.2 N per interaction;
  - damping is not identifiable, even on clean synthetic data of these interactions;
  - in one-direction interactions, Coulomb friction, sensor bias and any constant force share one regressor
    (condition number 10³–10¹⁴);
  - open-to-close forward prediction was credible on 0 of 6 objects.
- **Previously identified with extended models:**
  - wardrobe mass 9.6 kg and Coulomb 8.4 N;
  - dishwasher inertia 2.95 kg·m² plus a gravity term 16.1 N·m;
  - breakaway torques 24–26 N·m.

  These used models outside the ForceSAPIEN family; their code was not ported. `dynamics.extensions` provides the same
  model terms, with explicit identifiability flags.

- **Check with this library (2026-11-19, drawer_open7, session kitchen_15, not sealed).** The adapter reads the CSV.
  - The viscous-model CRLB diagnostic **passes** (I 1.2 %, c 3.7 %). IntegralLS gives m ≈ 30.3 kg and
    c ≈ 11.8 N·s/m.
  - The extended integral LS (Coulomb + bias) gives a **negative damping** (−16.6 N·s/m), a Coulomb term of 15.8 N
    and corr(Coulomb, bias) = −0.94. The extended fit is now marked physically inconsistent, with no term identified.

  Lesson: the CRLB verdict is *conditional on the model*. Passing it does not show that the viscous model describes a
  real drawer. Real-data numbers from the viscous estimators must be read with the model-adequacy caveat; this library
  does not yet provide a model-selection test.

## REAL (Doosan robot pushes, LeRobot parquet)

- **Recording.** 50 Hz:
  - `observation.state[:3]`: EE position in base_link (m); `[3:6]`: rotation vector;
  - `observation.wrench`: 6 values (N, N·m), frame declared as the EE axes, torque origin undocumented;
  - no per-sensor timestamps.

  The robot pushes in straight lines at 0.05–0.10 m/s and never grasps.
- **Kinematics.** Only 6/58 episodes were kinematically verified (light_blue_box drawer). Revolute objects were
  systematically predicted as prismatic by the legacy pipeline. No ForceArtGS joint-authority run exists yet.
- **Forces.** The axial wrench while opening is −1.75 to −2.15 N in 12/12 light_blue_box episodes, and about 0 on the
  return. This holds only if the wrench is in the tool frame. It is compatible with Coulomb-like friction of about
  1.9 N but **cannot be separated from a sensor bias** (same regressor). The script that computed this number is not
  saved, so it is not reproducible yet.
- **Identifiability.** At constant speed, mass is not identifiable and viscous vs Coulomb friction are collinear. The
  library's diagnostic reports exactly this (`tests/test_identifiability.py::test_steady_push_mass_not_identifiable`).

## Rules the library enforces

1. No implicit unit or frame conversion. Joint and forces must be in the same named frame.
2. Ambiguous, unobservable, static or rejected kinematics are never identified.
3. The identifiability verdict is attached to every result. Physical export raises `NotIdentifiableError` when the diagnostic fails or is missing; diagnostic reports may retain candidate estimates.
4. Extension terms (Coulomb, bias, load) are reported only with `identifiable: true`. Otherwise their value is not a result.
5. Bias removal needs a *declared* contact-free window. Contact masking is explicit.
