# Interface between ForceArtGS and ForceArtGS-Physics

ForceArtGS (Samuele Mara, <https://github.com/SamueleMara/ForceArtGS>, Apache-2.0) discovers the articulation and
estimates q(t). ForceArtGS-Physics **consumes** those kinematics together with the robot force record and identifies
the joint dynamics. It never re-estimates the joint. This page describes the contract as implemented at ForceArtGS commit
`886d49eed435e06babed8b69723082facdd4d3fb` (audited from the code, not only the README; file:line references are
to that commit).

## 1. Which ForceArtGS output to use

| level | files | frame / units | use here |
|---|---|---|---|
| **A: joint authority** (joint-aware reconstruction) | `<branch>/joint_authority/v<NNN>/authority.json` + `joint_states.npz` (`integration_contract.JointAuthority.write`, `:242`) | W = `metadata/cameras.json` world (OpenCV c2w) = robot `base_link` on real scenes; m, rad, s | **preferred**, `adapters.forceartgs.load_authority` |
| **B: joint discovery run** | `<run>/joint_candidates.json` (+ `joint_states.npz`) (`joint_discovery/cli.py:146`, `exports.py:47`) | same W | `adapters.forceartgs.load_discovery_run` |
| legacy FreeArtGS (`twopart_blend/estimated_joint.json`, `splatfacto-art/articulation.json`) | normalised nerfstudio frame (scale ≈ 1.19 + rotation), frame indices only | not metric | **refused** (`refuse_legacy`); register it to a metric world first (as done for the ForceSAPIEN inputs) |

### Level A keys (verbatim)
`authority.json`: `schema_version` (`forceartgs.joint_authority/1`), `version`, `joint_type`, `axis_world`,
`origin_world`, `reference_time_index`, `q_units`, `world_frame`, `decision_status`, `fit_covariance_status`,
`support_centroid_world`, `cameras`, `calibration_hash`, `source_hashes`, `derived_from`, `parent_authority_id`,
`authority_id`, `frame_map_hash`, `joint_states_sha256`.
`joint_states.npz`: `q`, `q_valid`, `rgbd_frame_index`, `source_frame_index`, `timestamp_s`, optional `q_sigma`.
The loader checks the schema and the `joint_states.npz` SHA-256 recorded in `authority.json`, as `JointAuthority.load` does (fail closed).

### Level B keys (verbatim)
`joint_candidates.json`: `candidates{revolute,prismatic}`, `decision`, `coordinate_conventions`. Each candidate has
`joint_type, joint_axis_world, joint_origin_world (null for prismatic), origin_convention, sign_convention,
q_units, q, q_valid, q_uncertainty, reference_time_index, timestamps_s, rgbd_frame_index, source_frame_index,
n_moving_tracks, world_frame, transform_convention, diagnostics, observability[, oracle_diagnostic_only]`.

## 2. Mapping to the library schema

| ForceArtGS | library (`schemas.py`) | notes |
|---|---|---|
| `joint_type` | `JointKinematics.joint_type` | revolute / prismatic only |
| `axis_world` / `joint_axis_world` | `JointKinematics.axis` | unit vector; the sign is canonicalised **together with q** (`(u, q) ~ (-u, -q)`, largest component positive). Always take axis and q from the **same file**. `joint_states.npz` of a Level-B run stores the *raw*, uncanonicalised q (audit hazard G7), so the loader reads q from `joint_candidates.json`. |
| `origin_world` | `JointKinematics.origin` | revolute: the point on the hinge line closest to the moving-support centroid. Its position along the axis is a gauge and does not affect the torque. Prismatic: `null` is replaced by a zero vector, flagged `origin_is_placeholder`; it is unused by the prismatic projection. |
| `q`, `q_valid`, `q_uncertainty`/`q_sigma` | `Trajectory.q`, invalid samples dropped, `q_std` | rad / m; `q[reference_time_index] = 0` |
| `timestamps_s` / `timestamp_s` | `Trajectory.t_s` | real scenes: `source_frame_index / 50`, not hardware stamps (gap G2) |
| `decision` (Level B) | `KinematicStatus` | `revolute`/`prismatic` → IDENTIFIED; `static/no_motion` → STATIC; `ambiguous/unobservable` → AMBIGUOUS |
| physical selection v2 | `KinematicStatus` | `IDENTIFIED_*` → IDENTIFIED; `TYPE_UNOBSERVABLE` → UNOBSERVABLE; `MODEL_INCONSISTENT` → REJECTED; `NO_SIGNIFICANT_MOTION` → STATIC |
| `decision_status` (Level A) | `KinematicStatus` | `identified` → IDENTIFIED; `ambiguous` → AMBIGUOUS; `candidate`/`provisional` → UNKNOWN (`allow_provisional=True` opts in, and the choice is recorded) |
| `observability.axis_sigma_deg`, `hinge_line_sigma_m` | `axis_std_rad`, `origin_std_m` | informative |

Every status other than IDENTIFIED makes `Interaction.require_identifiable()` raise `NotIdentifiableError`. An
ambiguous or unobservable joint therefore never produces physical parameters. This is tested on the actual ForceArtGS
**synthetic-scene** output of `cli synthetic --kind prismatic`, which ForceArtGS itself labels `ambiguous/unobservable`.

## 3. Robot data → force on the object

LeRobot episode parquet (`<scene>/metadata/raw/episode_*.parquet`, 50 Hz):
- `observation.state[:3]`: EE position in `base_link` [m]; `[3:6]`: rotation vector [rad].
- `observation.wrench`: `Fx..Tz` [N, N·m] in the sensor axes, declared equal to the EE axes.
- `frame_index` and `timestamp`.

`adapters.forceartgs.load_lerobot_parquet` (or `robot_forces_from_arrays`) computes:

```
R_ee        = Rotation.from_rotvec(state[3:6])                   base <- EE
F_object    = object_side_multiplier * R_ee @ R_ee_sensor @ F_sensor     (multiplier = -1: the recorded force is the
                                                                          reaction on the robot, configs/real_robot_tools.json)
p_contact   = p_ee + R_ee @ t_ee_tcp                             TCP, e.g. t_ee_tcp = (0, 0, 0.1876) m (pointed tip)
t           = frame_index / fps                                  the clock ForceArtGS uses for timestamps_s
```

- **Torques are dropped.** The reference point of the recorded torques is undocumented (gap G4). The joint torque is
  rebuilt by the projection as `a · ((p − o) × F)`.
- **Bias / tare is not removed automatically.** Use `forces.preprocessing.remove_constant_bias` with a declared
  contact-free window. The calibration log of the `ForceRecord` records every step.
- **Contact validity:** `ForceRecord.contact_valid`. `apply_contact_mask` zeroes off-contact samples, and only when called explicitly.

The joint frame and the force frame must have the **same name** (`Interaction` refuses otherwise). On real scenes, name
both `base_link` only if every camera of `metadata/cameras.json` declares `base_link`. That is the case in which
ForceArtGS sets `T_WB = I` (`tcp_contact.py:64-82`). Otherwise transform explicitly with `forces.SimilarityTransform`.

## 4. Generalized force

`forces.projection.generalized_effort` computes `Q_P = u · F` [N] and `Q_R = u · ((p − o) × F) (+ u · M)` [N·m].
These are the same formulas as ForceArtGS's `Q_H = f_obj · J_c,H(p_tcp)`, with `J_R = u × (p − o)` and `J_P = u`
(`wrench_physics.py:88`, `rigid_kinematics.py:277`), and as the validated benchmark code. The library does not
re-use ForceArtGS's `contact_force_diagnostics.csv` `Q_R/Q_P`: those use the raw-sign axis and the untared force
(gap G7).

## 5. Gaps found in the ForceArtGS interface (for Samuele)

| id | gap | consequence here | suggested change on the ForceArtGS side |
|---|---|---|---|
| G1 | no Level-A/B output exists yet on the shared cluster for the real scenes (all runs there are legacy FreeArtGS) | real-scene identification cannot be run yet | run the joint-aware pipeline on the verified REAL episodes |
| G2 | `timestamp_s = source_frame_index / 50`, not the parquet `timestamp` and not a hardware stamp; sync never verified | possible q-force lag | export the parquet `timestamp` per model time and a measured camera-robot lag |
| G3 | the discovery path uses the raw force (rotated, sign-flipped, no tare or gravity compensation) | user must tare | export the tared object-side force with its calibration gates |
| G4 | torque reference point undocumented | measured torques unusable | document the sensor torque origin in `robot_tool.json` |
| G5 | TCP point not re-estimated; camera↔base registration has cm-level unverified error | lever-arm error in revolute torque | export TCP-to-surface residuals and the registration covariance |
| G6 | no joint limits; no gravity/Coulomb model | biased I, c for lids / doors with non-vertical axes | export observed-range vs mechanical limits separately (already in the spec) |
| G7 | `joint_states.npz` (Level B) stores raw-sign q; `contact_force_diagnostics.csv` uses the raw-sign axis | sign hazard | canonicalise `joint_states.npz` too |
| G8 | `estimated_joint.json` / `splatfacto-art/articulation.json` of the joint-aware path are in the blend frame `B = N_BW · W`; the doc claims `N_BW = I` on real scenes, contradicting the base_link cameras | wrong frame if used | always read the authority (W) |
| G9 | `reconstruction/nerfstudio` submodule (joint-aware writer) not in the public clone | the `articulation.json` writer cannot be audited | push the submodule commit |

No change to ForceArtGS is *required* to use this library: it reads the existing Level-A/B files. The changes above
would remove the manual steps (tare, sync, torque origin) and the sign hazard. `docs/forceartgs_suggested_changes.patch`
is a documentation-only patch proposal. It has **not** been applied to Samuele's repository.

## 6. PArtTwin1D export

`export.parttwin.write_dynamics_yaml` writes `physics/dynamics.yaml` (`parttwin.dynamics/v1`, spec §6 of
`FORCEARTGS_RIGEULER_PARTTWIN1D_ASSET_EXPORT_CODEX_SPEC.md`). A 1-DoF experiment determines the inertia **about
the axis** (or the effective mass) and the damping. It does not determine the link mass, CoM or inertia tensor the schema
asks for. Those fields therefore stay `null`, including `links.moving.mass_kg` for prismatic joints; `joint.damping` is filled. The axis inertia goes in the extension block `joint_space_identification`.
`mode: identified` is written only for successful, finite positive parameters with a positive diagnostic and parameter verdict. Otherwise export raises `NotIdentifiableError` before writing a file.
