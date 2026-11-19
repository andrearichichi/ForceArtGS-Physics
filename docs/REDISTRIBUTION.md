# Redistribution review — 2026-11-19

The ForceSAPIEN A3/B bundle is **excluded** from the repository, its publication Git object database,
wheel and source distribution. No automatic download is provided. This includes the manifest, input
trajectories, forces, simulator truth, reference fits, predictions and scores.

The official [PartNet-Mobility terms](https://huggingface.co/datasets/sapien-sim/PartNetMobility)
(accessed 2026-11-19) restrict database use to non-commercial research/education, require colleagues to
accept the terms before access, and incorporate the
[ShapeNet terms](https://huggingface.co/datasets/ShapeNet/ShapeNetCore), which also condition redistribution.
Those pages do not explicitly grant unrestricted redistribution of this derived numerical simulation bundle.
The fact that it contains no meshes or images is insufficient evidence of permission. Exclusion is the release
decision; it is not a claim that all derived trajectories are necessarily prohibited.

An authorized researcher may configure an external copy:

```bash
export FORCEARTGS_PHYSICS_BENCHMARK=/absolute/path/to/authorized/forcesapien_a3
python examples/02_forcesapien_benchmark.py
```

Do not place that copy inside the repository, attach it to a GitHub release or send the original Git backup.
Sharing it separately requires confirmation of applicable rights and recipients' acceptance where required.
The simulator software license does not replace dataset permissions.

AKF_PINN is a separate outstanding rights question: no license was declared in the audited local repository.
Its history contains contributions by Andrea Richichi and Samuele Mara. The vendored provenance identifies
source bytes and transformations; it does not grant rights. Confirm the relevant contributors' consent for
private sharing and decide a license before broader distribution. No license was invented during this audit.
