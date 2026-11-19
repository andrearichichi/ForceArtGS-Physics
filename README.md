# ForceArtGS-Physics

Identify effective joint inertia/mass and viscous damping from a ForceArtGS joint trajectory and observed forces:

```text
I q'' + c q' = tau
```

Version **0.1.0**, prepared for private handoff to Samuele Mara.
**Start here: [Samuele's quickstart](docs/QUICKSTART_SAMUELE.md).**

## What has been tested

| Input / integration | Evidence | Scope |
|---|---|---|
| ForceSAPIEN A3 → B1–B10 | frozen synthetic benchmark; stored-fit scoring and historical re-fits | simulated viscous 1-DoF model; AKF-improved B results exploratory |
| ForceArtGS Level B | actual outputs of ForceArtGS's **synthetic** generator plus synthetic robot forces | loading, sign pairing, identification, prediction and export |
| ForceArtGS Level A | schema/hash loader checks | no measured-scene end-to-end validation |
| Hoi! measured drawer | adapter and diagnostic spot check | model mismatch observed; no validated physical result |
| REAL robot / Samuele's measured scenes | adapter/format support | **end-to-end physical identification still to be tested** |

Synthetic robot records and dataset simulations are not measured force data.
See [validation evidence](docs/VALIDATION_REPORT.md), [real-data limits](docs/REAL_DATA.md) and
[delivery audit](docs/DELIVERY_AUDIT.md). No real-data accuracy claim is made.

## Install

Python ≥ 3.10, numpy, scipy and torch; no Gaussian Splatting, nerfstudio or CUDA requirement.
After the private repository has been created and access granted:

```bash
git clone git@github.com:andrearichichi/ForceArtGS-Physics.git
cd ForceArtGS-Physics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python examples/03_minimal_numpy.py
```

The last command is a **synthetic CPU demonstration**. `.[real]` adds pyarrow for the low-level LeRobot reader;
`.[test]` adds pytest. [Reproducibility](docs/REPRODUCIBILITY.md) documents the reference stack.

## Methods

| API name | Frozen implementation | Device |
|---|---|---|
| `PINN` | PINN-V2, integral physics residual, LS start, Adam + L-BFGS; seeds 0–2 | CPU/GPU |
| `Backprop` | exact-ZOH least squares; 35 starts; Adam + L-BFGS-B | CPU |
| `MLP` | MLP-Joint, synthetic training per interaction; seeds 0–4 | CPU/GPU |
| `AKF` | AKF-improved, augmented EKF; R/P0 selected by training replay | CPU |
| `AKF-benchmark` | original A3-row AKF, retained for reproduction | CPU |
| `IntegralLS` | closed-form shared initializer, fast baseline | CPU |

The vendored estimator bodies and frozen hyperparameters are unchanged.
[Methods](docs/PHYSICAL_MODEL_AND_METHODS.md) and [provenance](docs/PROVENANCE.md) give details.
Only IntegralLS has a stacked multi-interaction fit; other `fit_many` methods aggregate individual fits.

## Pipeline contract

Read **Level A** `authority.json` + `joint_states.npz` (preferred), or **Level B** `joint_candidates.json`.
Read axis and q from the same authority/candidate: their signs are paired. Legacy normalised FreeArtGS outputs
require explicit metric registration and are refused. Geometry and forces must share a verified frame and clock.

```python
from forceartgs_physics import api
from forceartgs_physics.adapters import forceartgs as FA
from forceartgs_physics.adapters.real import load_observed_forces_json

joint, trajectory = FA.load_authority(authority_dir, frame=verified_world_frame)
forces = load_observed_forces_json(observed_forces_json)
x = FA.interaction("measured_episode", joint, trajectory, forces)
result = api.identify(x, "AKF", device="cpu")
result.save("diagnostic_result.json")
```

`diagnostic_result.json` is a research report: candidate estimates may be retained even if not identifiable.
`api.save_parameters`, `api.load_parameters` and `export.write_dynamics_yaml` require a successful result,
positive identifiability evidence and finite positive parameters. A failed or missing diagnostic refuses physical
export. Direct estimator calls and `identify_many` without a diagnostic therefore cannot export physics.
A positive CRLB verdict remains conditional on the viscous model; assess model adequacy on measured data before use.

The output identifies **axis inertia** [kg·m²] for revolute joints or **effective translating mass** [kg] for
prismatic joints, plus viscous damping. Link mass, CoM and the inertia tensor remain unknown and null in PArtTwin1D.
Effective mass is written only in `joint_space_identification.effective_mass_kg`.

## Examples and tests

- `examples/03_minimal_numpy.py`: standalone synthetic CPU example.
- `examples/01_forceartgs_end_to_end.py`: bundled synthetic ForceArtGS Level-B fixture and synthetic robot record;
  accepts only that fixture, to prevent mixing measured kinematics with synthetic forces.
- `examples/04_samuele_pipeline.py`: measured authority/discovery and prepared observed forces; diagnostic report only.
- `examples/02_forcesapien_benchmark.py`: optional external licensed benchmark, stored-fit scoring by default.

```bash
python -m pip install '.[test]'
python -m pytest -m 'not slow and not gpu'
```

Benchmark tests skip explicitly without `FORCEARTGS_PHYSICS_BENCHMARK`; source-identity tests require the original
research trees. No GPU training is needed for installation or handoff. Historical GPU evidence is limited to one
object and one seed per method. The maintainer regression script defaults to CPU.

## Optional benchmark and rights

**No ForceSAPIEN bundle is distributed in this repository or wheel.** If you have an independently authorized copy,
point `FORCEARTGS_PHYSICS_BENCHMARK` at its external directory. The code never downloads it automatically.
See [benchmark](docs/BENCHMARK.md) and [redistribution review](docs/REDISTRIBUTION.md).

No open-source license has been granted for the original research code or the vendored AKF_PINN subset.
The private handoff does not establish redistribution permission. AKF_PINN originates from
[SamueleMara/AKF_PINN](https://github.com/SamueleMara/AKF_PINN), with recorded contributions by Andrea Richichi and
Samuele Mara. See [LICENSE](LICENSE), [NOTICE](NOTICE), [provenance](docs/PROVENANCE.md) and the
[publication checklist](docs/PUBLICATION_CHECKLIST.md) for unresolved rights clearance.

ForceArtGS synthetic fixtures are attributed separately; its code is not vendored.
