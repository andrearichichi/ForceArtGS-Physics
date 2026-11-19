# Validation report (2026-11-19)

All runs on CINECA Leonardo:
- CPU jobs on the serial partition;
- GPU jobs on one NVIDIA A100-SXM-64GB.

Three environments were used:

| env | Python | torch | numpy | scipy | how |
|---|---|---|---|---|---|
| **reference** | 3.11 (3.11.13 login, 3.11.2 compute) | 2.5.1+cu121 | 1.26.4 | 1.17.1 | the interpreter and site-packages that produced the original A3 fits; package from `src/` |
| **editable install** | 3.11.13 | 2.15.0 | 2.2.6 | 1.17.1 | venv with system site-packages enabled + `pip install -e .[test,real]` (pip resolved the newest wheels) |
| **ForceArtGS-like** | 3.10.21 | 2.4.0+cu121 | 1.26.4 | 1.15.3 | venv on the FreeArtGS conda env + `pip install -e .[test,real]` (only pytest, pyarrow added) |

The earlier editable-install environment was not isolated (`include-system-site-packages = true`).
A separate clean wheel installation with system site-packages disabled is covered by [DELIVERY_AUDIT.md](DELIVERY_AUDIT.md).
Historical runs below predate the delivery guard changes. No GPU run was repeated for this audit.

## 1. Benchmark protocol on the original fits (`tests/test_benchmark_reference.py`, all envs)

| check | result |
|---|---|
| A3_RESULTS.json (PINN, Backprop, MLP, AKF-benchmark): \|I err\|, \|c err\|, B object median, PASS | identical (rel 1e-12) |
| AKF-improved row of SUMMARY_B.json | identical |
| 1 100 stored B1–B10 predicted trajectories (10 objects × 10 B × 11 fits) | identical (≤ 1e-12 × range) |
| per-object, per-interaction scores; κ | identical |
| nondimensional scales and integral-LS start of every object | exactly equal |
| no file under `truth/` opened while fitting | pass |

| estimator | B1–B10 nRMSE (object median) | reference value |
|---|---|---|
| PINN | 0.1471 % | 0.147 % |
| Backprop | 0.1762 % | 0.176 % |
| AKF (AKF-improved) | 0.1835 % | 0.184 % |
| MLP | 0.5123 % | 0.512 % |

## 2. Re-fits of the estimators against the original fits (`tests/test_refit.py`)

| re-fit | reference env | editable install (torch 2.15) |
|---|---|---|
| Backprop, 10 objects | **10/10 bitwise** (I, c, q_hat) | **10/10 bitwise** |
| AKF (AKF-improved), 10 objects, incl. the selected (P0, m_R) | **10/10 bitwise** | **10/10 bitwise** |
| AKF-benchmark, 10 objects | **10/10 bitwise** | **10/10 bitwise** |
| PINN seed 0, Knife (GPU) | **bitwise** (I = 0.019592090897008075) | ΔI −0.028 %, Δc −0.014 % (tolerance 0.1 %) |
| MLP seed 0, Knife (GPU) | **bitwise** (I = 0.019148677541958647) | not run (job time limit) |

The GPU re-fits covered one object and one seed for budget reasons: about 6 min (PINN) and 2.5 min (MLP) of A100
time in the reference environment. The original 3-seed / 5-seed fits of all objects are in the external licensed bundle and are scored
exactly in §1.

## 3. Fast suite (`pytest`)

| env | result |
|---|---|
| reference (with `FORCEARTGS_PHYSICS_ORIGINAL_ROOT`) | 74 passed, 1 skipped (Hoi! data path not set) |
| ForceArtGS-like, Python 3.10 | 47 passed, 28 skipped (original trees not given → provenance text checks skipped; Hoi!) |
| Hoi! adapter, with the data path | 1 passed |

The provenance checks in the reference run confirm that the copied source text of every vendored name is identical to
the original files. They also confirm that those files still have the SHA-256 recorded at vendoring.
The first fresh-install run had one failure, in `test_state_solver_reduces_to_validated_solver`: the general-start
solver inserted a redundant knot (differences ~1e-16). It was fixed, and the later runs pass.

## 4. ForceArtGS integration (synthetic scenes, genuine pipeline output)

Inputs: Level-B outputs of ForceArtGS `cli synthetic` at commit 886d49e (`tests/fixtures`), and a robot record in
LeRobot parquet form generated for I = 0.05 kg·m², c = 0.15 N·m·s/rad (reaction sign, EE-axis wrench, TCP offset
0.1876 m). Python 3.10:

| step | result |
|---|---|
| decision → status | revolute → identified; prismatic run → `ambiguous/unobservable` → `NotIdentifiableError` (refused) |
| robot record → force on object at TCP | reconstruction error < 1e-15 N, < 1e-15 m |
| IntegralLS | I −0.10 %, c +0.36 % |
| Backprop | I −0.10 %, c −0.29 % (40 s) |
| AKF | I +0.53 %, c −0.28 % (33 s) |
| identifiability | separately identifiable (CRLB ≤ 10 %) |
| new interaction (25° in 0.8 s), frozen AKF parameters, signed projection | prediction nRMSE 0.109 % |
| PArtTwin1D `physics/dynamics.yaml` | written, `mode: identified`, damping filled, tensors null |

The error comes from ForceArtGS's q estimate (12 samples at 10 Hz) and the 50 Hz force record. The estimators' code
is the same as in §2.

## 5. Real-data check (Hoi! drawer_open7, kitchen_15, not sealed)

The adapter reads the CSV. The viscous CRLB check passes (I 1.2 %, c 3.7 %), and IntegralLS gives m ≈ 30.3 kg,
c ≈ 11.8 N·s/m. The extended integral LS (Coulomb + bias) returns a negative damping. It is now reported as physically
inconsistent, with no term identified. This is a model-adequacy warning; see docs/REAL_DATA.md.

## Raw outputs
The run directories (logs, junit XML, one JSON per re-fit) were kept outside the repository in the build scratch area.
They are not part of the repository.


## 6. Final handoff wheel verification

The final v0.1.0 wheel was installed non-editably in a new isolated Python 3.11.13 venv with torch 2.5.1+cpu,
NumPy 1.26.4, SciPy 1.17.1 and PyArrow 25.0.1. Outside-source execution: **108 passed, 1 skipped** (Hoi! data absent),
**32 slow/GPU tests deselected**. The synthetic ForceArtGS demo exercised the LeRobot parquet reader, identification,
frozen prediction and guarded export. No measured-scene end-to-end test or GPU training was performed.
See [DELIVERY_AUDIT.md](DELIVERY_AUDIT.md) for package, licensing, safety and publication-history checks.
