# Provenance of the implementations and of the reference results

## 1. Where the definitive results come from

The definitive ForceSAPIEN benchmark is **A3 → B1–B10**, finalised on 2026-11-02:
- 10 objects;
- one large-motion training interaction A3 per object, captured with 8 fixed cameras and reconstructed with
  Samuele Mara's joint multicamera FreeArtGS fork;
- ten held-out interactions B1–B10 per object.

| estimator | reference results (frozen research tree) | B1–B10 nRMSE, object median |
|---|---|---|
| PINN | `results/forcesapien_a3` fits `PINN-V2_primary_s{0,1,2}`; table `A3_RESULTS.json` | **0.1471 %** |
| Backprop | `results/forcesapien_a3` fit `Backprop_primary_s0` | **0.1762 %** |
| AKF | `results/forcesapien_a3_akf` fit `est/AKF-improved`; table `SUMMARY_B.json` | **0.1835 %** |
| MLP | `results/forcesapien_a3` fits `MLP-Joint_primary_s{0..4}` | **0.5123 %** |
| (AKF, original A3 row) | `results/forcesapien_a3` fit `AKF_primary_s0` | 2.3496 % |

These numbers were re-read from those files (and are re-checked by `tests/test_benchmark_reference.py`).

**Definitive AKF.** The AKF in the original `A3_RESULTS.md` row (native NLL start, R = σ) is *not* the definitive
one. The definitive variant is **AKF-improved** from the follow-up study `results/forcesapien_a3_akf`
(`AKF_A3_PROTOCOL.md`). That study found the filter's own error persists with oracle q; R = σ collapses the parameter
covariance. AKF-improved starts from the shared integral-LS estimate and chooses (R, P0) by the open-loop replay error
on the training interaction. The final A3 visualizer and its `PROVENANCE.md` show this variant. The library's `AKF` is
AKF-improved; `AKF-benchmark` reproduces the original row.
Caveat recorded by that protocol: the selection rule was written after B1–B10 had already been used in earlier
benchmarks, so the AKF-improved B result is *exploratory*. The confirmation set C1–C10 was generated but never scored.

Discarded experiments that are **not** used here: Protocol C/J/F tables, the FS4 overnight table, the 1+10 A2 table,
the PINN-V1/EKF-V2 variants, MLP-SS/MLP-SUP, the "ultimate estimator" (VP-LM) campaign (abandoned) and the
late-fusion multicam pilot.

## 2. Code chain that produced the A3 fits

```
a3/a3_stage3.sbatch  (per object, AKF_PINN conda env: Python 3.11.13, torch 2.5.1+cu121, numpy 1.26.4, scipy 1.17.1)
  └─ robust_v2/run_heldout.py --method {AKF,PINN-V2,Backprop,MLP-Joint}
       ├─ PINN-V2      : robust_v2/pinn_v2.py::run(V2Config(seed, **run_v2.VARIANTS['V2']))
       └─ others       : controlled_fairness/run.py::execute(method, 'primary')
                           ├─ Backprop  : controlled_fairness/cpu.py::backprop(d, 'native', SUPPORT)
                           ├─ AKF       : controlled_fairness/cpu.py::akf(d, sc, 'native')   (original A3 row)
                           └─ MLP-Joint : controlled_fairness/mlp.py::train(d, seed, 'primary')
                                            └─ mlp_joint/mlp_joint.py
       shared: controlled_fairness/common.py  (load, normalize, ls = integral-LS initializer, restore)
               ├─ freeartgs_only/cells_C.py  (observation, compress_zoh)
               │    └─ improved/methods.py (force_knots) ─ expB/expB_cells.py (zoh_basis, LS offset/velocity)
               │         └─ estimators.py (Window, _features) ─ AKF_PINN evaluation_v2 (Observation)
               ├─ final_benchmark/pinn_nd.py (nondimensionalize)
               └─ AKF_PINN/src/akf_pinn (config, physics, data.interaction, filters.augmented_ekf, pinn.network/derivatives)
akf_a3/akf_variants.sbatch → akf_a3/akf_variants.py (AKF-improved: fit_grid) + akf_a3/dev_probe.py (base_cfg, run_variant, replay)
a3/post_a3.py evaluate|score|aggregate  → B1–B10 prediction with oneplus10/replay.py (zoh_levels, propagate)
akf_a3/evaluate_bc.py + tables.py       → the same for the AKF variants
```

The earlier dedicated repository (`AKF_PINN`, package `akf_pinn`, history up to 2026-09-14) supplies the filter,
the integrators and the PINN network. Its own evaluation protocol (`evaluation_v2`) and its PINN-V1 were superseded,
and the later work called the library *unchanged*.

## 3. How the code was carried over

`tools/vendor_validated.py` (re-runnable, given the two original roots) generates `src/forceartgs_physics/_validated`:
- the `akf_pinn` modules used are copied whole. The only edits are the rewrite of `from akf_pinn.` imports into
  relative imports and a Python-3.10 guard around `import tomllib`;
- for every research-script module, only the names the A3 chain executes are copied, as **exact source text**
  (`ast` segments). A generated import header recreates the original namespace (for example `H.C.B.zoh_basis`).
  The function bodies are therefore unchanged.
- `_validated/PROVENANCE.json` lists every original file (path relative to its root), its SHA-256 and the copied names.

`tests/test_provenance.py` re-extracts the same names from the original files and asserts that the text is identical
and that the original files still have the recorded hash. It runs when `FORCEARTGS_PHYSICS_ORIGINAL_ROOT` is set.

The public wrappers (`estimators/*.py`) call the vendored functions exactly as the drivers did:
- the same arguments;
- the same validity rule;
- the median over seeds;
- one thread, as in the validated runs.

The only replaced step is the *file read* of `common.load`: the window is built from a `schemas.Interaction`.
For ForceSAPIEN inputs it is field-for-field the same.

## 4. What is new (not part of the validated benchmark)

| module | content | validated how |
|---|---|---|
| `schemas`, `units`, adapters | typed interface, frames, units, status handling | unit tests |
| `forces.projection.generalized_effort` | signed projection (same formula as the validated `replay.effort`) | equals the stored effort of all 10 A3 inputs to 1e-12 |
| `dynamics.viscous.propagate_from_state` | exact solver from any (t0, q0, v0) | equals the validated solver for rest at t=0 (bitwise) and an independent solver (1e-10) |
| `validation.identifiability` | robust_v2 `estimator_side` diagnostic + the pre-registered 10 % CRLB rule | synthetic identifiable / non-identifiable cases |
| `dynamics.extensions` | Coulomb/static friction, bias, load; extended integral LS with identifiability flags | synthetic tests; **not** used by the four estimators |
| `estimators.IntegralLS.fit_many`, `Estimator.fit_many` | multi-interaction identification (stacked LS / median aggregation) | unit tests only |
| `export.parttwin` | PArtTwin1D `dynamics.yaml` | format test |

## 5. Rights and attribution audit (2026-11-19)

| Component | Provenance | Rights status | Delivery |
|---|---|---|---|
| Research scripts | original comparison tree; file hashes in `_validated/PROVENANCE.json` | no declared grant; contributor clearance pending | selected frozen code |
| AKF_PINN subset | `https://github.com/SamueleMara/AKF_PINN`; local HEAD `d0830c4fcd23791b2a8d4f6ab45b1184910d94ca` | no LICENSE/NOTICE or pyproject license; history records Andrea Richichi and samuelemara | vendored with provenance; no inferred license |
| ForceArtGS synthetic outputs | Samuele Mara; commit `886d49eed435e06babed8b69723082facdd4d3fb` | upstream Apache-2.0; generated outputs attributed separately | two small synthetic fixtures |
| ForceArtGS source | upstream only | Apache-2.0 | not included |
| ForceSAPIEN derived bundle | SAPIEN simulations using PartNet-Mobility objects | redistribution not clearly cleared | excluded from tree/history/artifacts |
| Hoi! recordings | Dataverse doi:10.60507/FK2/QODWTV | dataset terms | not included |
| REAL recordings | Samuele's robot episodes | no redistribution grant assumed | not included |

The earlier description “same authors” was insufficient: Git records authorship, not exclusive ownership.
AKF_PINN includes commits by both Andrea and Samuele. No blanket open-source license has been assigned.
See [LICENSE](../LICENSE), [NOTICE](../NOTICE), [redistribution review](REDISTRIBUTION.md) and
[publication checklist](PUBLICATION_CHECKLIST.md). Private repository visibility does not settle these rights.
