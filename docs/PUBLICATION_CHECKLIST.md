# Private publication checklist — andrearichichi/ForceArtGS-Physics v0.1.0

**Status: prepared locally; nothing published.** Technical checks are recorded in
[DELIVERY_AUDIT.md](DELIVERY_AUDIT.md). Rights clearance remains a release gate, including for private sharing.

## Before sharing

- [x] Scope limited to identification library, adapters, synthetic examples, tests and documentation.
- [x] Frozen estimator code/hyperparameters preserved; no new algorithms or GPU training for this review.
- [x] ForceSAPIEN payload separated from code; external opt-in path, no automatic download.
- [x] Publication history rebuilt without the old data-bearing commit or its objects/reflogs.
- [x] Synthetic, measured and tested integration evidence distinguished.
- [x] Physical export refuses failed, unknown, non-identifiable and invalid-parameter results.
- [x] Link mass, CoM and tensor stay unknown; effective mass/axis inertia remain in joint space.
- [x] Wheel installed in a genuinely isolated venv; 108 tests pass, one optional data skip; artifacts exclude the bundle.
- [x] Simple CPU quickstart and measured-input contract prepared for Samuele.
- [ ] Andrea and Samuele confirm rights/consent for vendored AKF_PINN and research-script code, including any institutional obligations.
- [ ] Agree the intended code license, update LICENSE/NOTICE/metadata consistently, rebuild artifacts if necessary.
- [ ] Review the exact handoff diff and final audit with the rights holders.

AKF_PINN has no declared license. Recorded authorship is not a licensing grant.
Do not tick the rights items on the basis of a successful test run or private repository visibility.

## Publication operations — to run only after explicit authorization

These commands are instructions, **not executed by this audit**. `main` is the clean local root commit.
Do not import branches/tags from the backup or use `git push --mirror`.

```bash
cd ForceArtGS-Physics
git status --short
git log --all --oneline
git rev-list --objects --all | rg 'benchmark/|forces_B/|/truth/|b_predictions|a3_fits'
# The rg command above must have no matches (exit 1 is expected).

gh repo create andrearichichi/ForceArtGS-Physics --private --source=. --remote=origin
# Confirm visibility before pushing; the value must be PRIVATE:
gh repo view andrearichichi/ForceArtGS-Physics --json visibility --jq .visibility
git push -u origin main
git tag -a v0.1.0 -m 'ForceArtGS-Physics v0.1.0: private research handoff'
git push origin v0.1.0
```

- [ ] Confirm `andrearichichi` account, exact name and **private** visibility before first push.
- [ ] Push clean `main` only, then tag `v0.1.0` on the reviewed commit.
- [ ] Invite Samuele using his verified GitHub account, with agreed access.
- [ ] Clone into a new directory and execute the documented synthetic quickstart.
- [ ] If creating a private GitHub release, attach only the reviewed wheel/source archive and checksums.
- [ ] Do not attach the external dataset, raw robot data, checkpoints, original Git backup or build environments.
- [ ] Describe measured-scene integration and real-data model adequacy as pending experiments.

## First measured episode (validation still pending)

- [ ] Export a metric Level-A authority or Level-B discovery result for the episode.
- [ ] Verify frame registration, TCP/contact, force sign/axes, tare and q-force synchronization.
- [ ] Supply measured forces with calibration provenance; run the diagnostic and CPU identification.
- [ ] Check model adequacy and predict a separate measured interaction before using a physical asset export.
- [ ] Record failure/non-identifiability as an experimental outcome, without filling unknown link properties.
