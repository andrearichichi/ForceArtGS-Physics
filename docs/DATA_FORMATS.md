# Data formats and conventions

## Units (no implicit conversion anywhere)

| quantity | revolute | prismatic |
|---|---|---|
| q | rad | m |
| q' | rad/s | m/s |
| generalized effort τ | N·m | N |
| I | kg·m² (about the joint axis) | kg (effective mass) |
| c | N·m·s/rad | N·s/m |
| time | s | s |
| positions / points / origin | m | m |
| force / couple | N / N·m | N / N·m |

## Frames

Every geometric quantity carries a frame **name**. `Interaction` requires `joint.frame == forces.frame` and raises
otherwise. Use `forces.SimilarityTransform(R, t, s, src, dst)` for explicit changes: points x' = s R x + t;
directions, forces and couples v' = R v. Couples cannot pass through a scaled transform. Prismatic q in metres is never
rescaled implicitly: an adapter that applies a scale records it.

Sign conventions:
- Positive q is a right-hand rotation about +axis (revolute) or a translation along +axis (prismatic).
- Positive τ drives positive q.
- Forces are the force applied **on the moving part**.
- The low-level LeRobot helper defaults to reaction sign −1 (`object_side_multiplier`); verify this on the recording.
  The measured JSON reader applies no sign or calibration defaults: forces must already be on the object.

## Input schema (`forceartgs_physics.schemas`)

### `JointKinematics`
| field | type | unit | notes |
|---|---|---|---|
| joint_type | `"revolute"`/`"prismatic"` | – | |
| axis | [3] | – | normalised on construction |
| origin | [3] | m | point on the axis (revolute); informative for prismatic |
| frame | str | – | |
| status | `KinematicStatus` | – | identified, ambiguous, unobservable, static, rejected, unknown |
| limits | (lo, hi) or None | rad / m | physical limits only; never the observed range |
| axis_std_rad, origin_std_m | float or None | rad, m | upstream uncertainties |
| source | dict | – | provenance (file, decision, conventions) |

### `Trajectory`
| field | type | unit | notes |
|---|---|---|---|
| t_s | [N] strictly increasing | s | same clock as the forces |
| q | [N] | rad / m | offset arbitrary unless `zero_reference` says otherwise |
| q_std | [N] or None | rad / m | |
| zero_reference | str | – | e.g. "q = 0 at ForceArtGS reference_time_index 0" |
| initial_state | `InitialState(t_s, q, qdot, source)` or None | s, rad\|m, rad/s\|m/s | DECLARED condition; the validated estimators do not use it |

### `ForceRecord`
| field | type | unit | notes |
|---|---|---|---|
| t_s | [M] strictly increasing | s | forward ZOH: sample k holds on [t_k, t_k+1); the first sample also before t_0 |
| force_N | [M, 3] | N | on the moving part |
| frame | str | – | |
| point_m | [M, 3] or None | m | application points (required for revolute) |
| torque_Nm | [M, 3] or None | N·m | pure couples (adds a·M to the revolute effort) |
| contact_valid | [M] bool or None | – | applied only through `apply_contact_mask` |
| force_std_N, point_std_m | scalar or [M] | N, m | |
| wrench_frame, tcp | str, dict | – | sensor axes and TCP convention |
| calibration | dict | – | log of every preprocessing step |

## Outputs

`IdentificationResult` (JSON via `.save()` / `.load()`):
```
estimator, status (ok | numerical_failure | failed), interactions[]
parameters: {joint_type, inertia, damping, units{inertia, damping}, decay_time_s, model: "viscous_1dof",
             identifiable, uncertainty{method, inertia_rel_std, damping_rel_std}, extra}
q_fit: {t_s[], q[], seed}                  estimator's state estimate on the training data
fit_metrics: {q_fit_rmse, q_fit_nrmse_pct, normalisation, units}
identifiability: {kinematic_status, verdict, identifiable, identifiable_inertia, identifiable_damping,
                  crlb_rel_std_I_pct, crlb_rel_std_c_pct, sens_corr_logI_logc, integral_cond, ...}
per_seed[]: {seed, status, inertia, damping, convergence, runtime_s, diagnostics}
diagnostics: {scales (nondimensionalisation), shared_ls (initializer), encoding, aggregation}
config: {seeds, device, encoding, threads, workdir, frozen_hyperparameters}
provenance: {library_version, implementation{code, ...}, environment{torch, numpy, threads, cuda}, input_fingerprint}
```

`.save()` is a diagnostic report and may retain candidate parameters after a failed identifiability check.
`api.save_parameters`, `api.load_parameters` and PArtTwin physical export require successful status, identified
kinematics, positive diagnostic/parameter verdicts and finite positive I/c; otherwise they raise
`NotIdentifiableError`. Effective prismatic mass is never written as link mass. All link mass/CoM/tensor fields
remain null; the identified joint quantity is in `joint_space_identification`.
`mode: identified` describes that joint-space result; it does not certify a complete rigid-body asset.
A simulator requiring link inertial properties needs independent evidence for those missing fields.

`Prediction.to_dict()`: `t_s, q, parameters, effort_model, initial_state, effort_t_s, effort, units`.

## Dataset adapters

| adapter | reads | frame name | status source |
|---|---|---|---|
| `forcesapien.load_estimator_input` | `<slug>.*.estimator_input.json` (Protocol C / A3) | `forcesapien_world` | inputs passed the geometry gate → identified |
| `forcesapien.load_force_record` | `simulation.json`, `*.force_record.json`, bundle `forces_B/*.json.gz` | `forcesapien_world` | – |
| `forceartgs.load_authority` | `authority.json` + `joint_states.npz` (Level A) | caller (`base_link`) | `decision_status` |
| `forceartgs.load_discovery_run` | `joint_candidates.json` (Level B) | caller | `decision` |
| `forceartgs.load_lerobot_parquet` | LeRobot episode parquet (needs `pyarrow`) | caller | – |
| `hoi.load_force_csv` | freeartgs-hoi force CSV + `report.json` | `joint_aligned_effort_frame` | identified (packaged joint) |
| `real.load_observed_forces_json` | preprocessed measured forces with calibration provenance | caller-verified | – |

## JSON delle forze osservate per gli episodi misurati

Dal tuo codice puoi salvare il JSON direttamente dagli array già calibrati:

```python
import json
from pathlib import Path

# t_s, force_on_object_world_N, contact_points_world_m e calibration_log
# provengono dal tuo episodio e dal tuo preprocessing verificato.
payload = dict(t_s=t_s.tolist(), force_N=force_on_object_world_N.tolist(),
               point_m=contact_points_world_m.tolist(), frame=verified_world_frame,
               source=dict(kind="measured", recording=recording_path), calibration=calibration_log)
Path("forze_osservate.json").write_text(json.dumps(payload))
```

Il tuo preprocessing deve scrivere queste chiavi; tutti gli array devono contenere i campioni reali:

| Chiave | Contenuto |
|---|---|
| `t_s` | N timestamp in secondi, strettamente crescenti, nello stesso clock di q(t) |
| `force_N` | N × 3 forze **sull'oggetto**, in newton, già nel frame del joint |
| `point_m` | N × 3 punti di applicazione in metri nello stesso frame; obbligatori per revolute |
| `frame` | nome esatto del frame metrico verificato; `base_link` solo se realmente coincidente con W |
| `source.kind` | stringa `measured` |
| `source.recording` | percorso/identificatore del record misurato originale |
| `calibration.frame_registration` | riferimento alla registrazione camera/mondo/robot verificata |
| `calibration.force_sign` | riferimento alla verifica del segno della forza sull'oggetto |
| `calibration.sensor_axes` | trasformazione sensore → frame del joint verificata |
| `calibration.application_point` | TCP/punto di contatto calibrato e relativo riferimento |
| `calibration.bias` | procedura di tare/compensazione e finestra libera dal contatto usata |
| `calibration.time_alignment` | procedura di sincronizzazione e lag misurato applicato |

Le sei voci di `calibration` sono stringhe descrittive non vuote; il loader richiede la provenienza,
ma non può verificare che una calibrazione dichiarata sia corretta. Non riempirle con valori ipotizzati.
`torque_Nm` è facoltativo: includi solo coppie pure correttamente trasportate al punto di applicazione,
con `calibration.torque_reference`. Altrimenti omettilo: il momento della forza viene calcolato da `point_m`.

Il reader LeRobot `FA.load_lerobot_parquet` è una funzione di basso livello con convenzioni storiche
(segno −1, frame EE, clock `frame_index / 50`); non è il quickstart reale.
Nel preprocessing passa esplicitamente TCP, rotazione sensore/EE, segno e clock verificati e applica tare/sync.
[Contratto completo e gap noti](FORCEARTGS_INTERFACE.md), [limiti dei dati reali](REAL_DATA.md).
