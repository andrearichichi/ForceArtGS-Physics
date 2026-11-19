# Delivery audit — 2026-11-19 — v0.1.0

Prepared for the **private** `andrearichichi/ForceArtGS-Physics` handoff to Samuele Mara.
No remote repository, push, tag or release was created by this audit. Rights clearance remains pending;
technical readiness does not establish permission to share unlicensed co-authored code.

## File inventory and purpose

| Included path | Reason to retain |
|---|---|
| `src/forceartgs_physics/` | public interface, adapters, projection, diagnostics, guarded export |
| `src/forceartgs_physics/_validated/` | frozen execution chain needed by the estimators; original hashes/names retained |
| `configs/` | frozen estimator settings and benchmark protocol, no trajectories or truth |
| `examples/` | standalone synthetic demo, synthetic ForceArtGS integration, measured-input diagnostic and optional external benchmark |
| `tests/` | contract, physics, export, provenance and regression checks; two small generated synthetic fixtures |
| `tools/` | reproducible vendoring, configuration inventory and external-only bundle construction |
| `scripts/run_regression.sbatch` | maintainer regression entry point, CPU by default, GPU explicit opt-in |
| `docs/` | methods, conventions, limitations, validation evidence, handoff and publication instructions |
| `docs/forceartgs_suggested_changes.patch` | clearly marked documentation-only suggestion for upstream; not applied |
| `LICENSE`, `NOTICE`, `third_party/` | accurate rights status and upstream fixture attribution/license |
| packaging files / `.gitignore` | installable source/wheel and exclusion of build artifacts, data and outputs |

Excluded: ForceSAPIEN inputs/forces/truth/reference-fit payloads/predictions/manifest, robot recordings,
Hoi! recordings, model checkpoints, caches, editable-install metadata, build environments and logs.
There is no vendored ForceArtGS/FreeArtGS reconstruction stack and no dependency on it at runtime.

## Code and rights

All 33 tracked files under `_validated/` are byte-identical to the pre-review commit.
The original-tree provenance tests verify source hashes and copied names separately.
The audit changes public I/O boundaries and documentation; estimator algorithms and hyperparameters remain frozen.

AKF_PINN's audited local HEAD is `d0830c4fcd23791b2a8d4f6ab45b1184910d94ca`, originating from
`SamueleMara/AKF_PINN`. Its history records Andrea Richichi and samuelemara (including initial implementation
commit `4751b11`). No license grant was found. Rights/consent and final code license require confirmation.
The repository's LICENSE reports that status rather than inventing a grant.

ForceArtGS-generated synthetic fixtures are attributed to Samuele Mara and upstream commit
`886d49eed435e06babed8b69723082facdd4d3fb`; the upstream Apache-2.0 text is preserved.
ForceSAPIEN bundle redistribution was not clearly cleared; see [REDISTRIBUTION.md](REDISTRIBUTION.md).

## Git and artifact separation

The only pre-review local commit contained the bundle. The complete original Git metadata and bundle were
preserved in a **separate private sibling directory**, outside the deliverable. The publication repository was
initialized with a fresh Git object database. No original branches, tags, reflogs, alternates or remotes were
imported. The old backup must not be used as the source for GitHub, attached to a release or sent to Samuele.

The optional benchmark accessor now requires an explicit external directory. Its builder refuses an output
inside the repository. Tests skip with an explicit reason when an authorized external bundle is not configured.
Both wheel and source distribution exclude the bundle.

## Safety and claims

- All non-identified kinematic statuses block fitting. Missing/failed/conflicting identifiability evidence
  blocks physical parameter and PArtTwin export, including numbers inside the joint-space extension.
- Nonfinite, zero or negative I/c values block export. Invalid identified extension coefficients and an
  explicitly supplied inconsistent/unassessed extension fit also block asset export.
- `IdentificationResult.save` writes a diagnostic research report, which may retain invalid candidate estimates.
  It is not a physical asset export; `api.load_parameters` refuses such reports for physical use.
- Effective translating mass is **not** exported as link mass. Both link mass fields, CoM and inertia tensors
  remain null. Axis inertia/effective mass is exported only in joint space.
- The synthetic ForceArtGS demo accepts only its included synthetic fixture. Measured examples require actual
  observed force arrays and declared calibration provenance; they have no synthetic-force fallback.
- Calibration provenance is a required declaration, not an automatic metrology verification. Frame registration,
  TCP/contact, sign, sensor axes, bias and time alignment remain experiment responsibilities.
- CRLB passing is model-conditional. Measured-scene integration, model adequacy and held-out measured prediction
  remain to be tested. Historical Hoi! mismatch and REAL constant-speed limitations remain explicit.

## Verification

Final checks and artifact details are recorded below. Build/test logs and environments stay outside the repository.
No new GPU training or GPU re-fit was launched in this audit.

The optional parquet dependency is constrained to `pyarrow>=14,<26`: Arrow 26.0.1 failed to import with
NumPy 1.26.4 despite `pip check` passing, because its runtime requires NumPy >=2. This was reproduced directly.
Arrow 25.0.1 imports successfully with NumPy 1.26.4. An integration test now verifies that an installed Arrow
package really imports, so the synthetic array fallback cannot hide this incompatibility.

The final wheel was installed non-editably into a fresh local temporary venv with
`include-system-site-packages = false`, Python 3.11.13, torch 2.5.1+cpu, NumPy 1.26.4,
SciPy 1.17.1 and PyArrow 25.0.1. User site-packages and PYTHONPATH were disabled; the import path was
verified to be the venv's `site-packages`, with no source-tree import. A shared-filesystem staging attempt
was superseded by this local-disk verification. Build tools ran in a separate build environment.

| Final check | Outcome |
|---|---|
| Wheel full suite, outside the source tree, CPU only | **108 passed, 1 skipped, 32 deselected**, 71.20 s |
| Reason for the one skip | Hoi! measured data directory not supplied |
| Training/re-fit selection | all 32 slow/GPU tests deselected; no GPU training or re-fit run |
| Vendored-source checks | 44 passed, included in the wheel suite; all 33 vendored files unchanged from pre-review commit |
| Delivery safety checks | 33 passed, included in the wheel suite |
| `pip check` | no broken requirements |
| Optional benchmark absent | 5 reference tests explicitly skipped with the external-bundle explanation |
| Standalone NumPy quickstart | effective mass 1.2000 kg, damping 0.8000 N·s/m; identifiable |
| Synthetic ForceArtGS quickstart | actual LeRobot parquet path exercised; IntegralLS; frozen prediction nRMSE 0.107%; guarded PArtTwin export succeeds |
| Measured example and CLI entry points | help/argument paths checked; no measured-scene validation claimed |
| Git payload audit | all 228 external bundle blob IDs absent, old data-bearing commit absent, no alternates |
| Artifact content | no benchmark payload; LICENSE, NOTICE and upstream ForceArtGS license preserved |

Final wheel: `forceartgs_physics-0.1.0-py3-none-any.whl`.
SHA-256: `13961be2adca0d36eebb7c93ed77d7424ffa0de9ceb709235d0858a3bd018199`.
The source archive is built from the same reviewed files; final checksums and detailed logs are kept with the
private build artifacts outside Git. The delivery tree contains 111 selected files.

**Remaining gates:** contributor/rights-holder consent for the unlicensed code and an agreed licensing decision.
Measured-scene model adequacy and held-out measured prediction remain future experiments, not release validation claims.
See [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md). Do not publish or share the private Git backup.
