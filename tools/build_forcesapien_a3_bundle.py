#!/usr/bin/env python3
"""Build ``benchmark/forcesapien_a3`` from the frozen ForceSAPIEN A3 results (maintainer tool).

    python tools/build_forcesapien_a3_bundle.py --results-root <.../comparison/freeartgs_all_objects/results>

Reads (never writes) the frozen trees:
  forcesapien_a3/inputs/<slug>.G1.estimator_input.json      A3 estimator inputs (FreeArtGS N8 joint, q(t), forces)
  forcesapien_a3/est/raw/<slug>/<method>_primary_s<k>.json  official A3 fits (PINN-V2, Backprop, MLP-Joint, AKF)
  forcesapien_a3/evaluation/<slug>/B<j>.json                official B1-B10 open-loop predictions
  forcesapien_a3/scores/<slug>.json, A3_RESULTS.json        official scores
  forcesapien_a3/gt_sealed/<slug>.A3.evaluation_truth.json  A3 truth (EVALUATION ONLY)
  forcesapien_a3_akf/est/AKF-improved/raw/...               definitive AKF (AKF-improved) fits
  forcesapien_a3_akf/evaluation/B/<slug>/B<j>.json          AKF-improved B predictions
  forcesapien_a3_akf/SUMMARY_B.json, SCORES_B.json          AKF study scores
  force_estimators_1plus10_cvpr/dataset/measured/<slug>.B<j>.force_record.json   B forces (estimator input)
  force_estimators_1plus10_cvpr/dataset/gt_sealed/<slug>.B<j>.evaluation_truth.json B truth (EVALUATION ONLY)

Writes to an explicitly supplied directory outside the repository:
  benchmark/forcesapien_a3/inputs/      A3 inputs, byte-identical copies
  benchmark/forcesapien_a3/forces_B/    B force records reduced to the fields the replay reads (gzip JSON)
  benchmark/forcesapien_a3/truth/       A3 + B truth, EVALUATION ONLY (gzip JSON); estimators never read it
  benchmark/forcesapien_a3/reference/   original fits, predictions and scores (gzip JSON)
  benchmark/forcesapien_a3/MANIFEST.json   sha256 of every source file and every written file
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = None
SLUGS = ["FoldingChair_102255", "Knife_103706", "Laptop_10211", "Oven_101917", "Refrigerator_10905", "Scissors_11100",
         "Stapler_103111", "StorageFurniture_45135", "USB_100109", "WashingMachine_103776"]
SEEDS = {"AKF": [0], "PINN-V2": [0, 1, 2], "Backprop": [0], "MLP-Joint": [0, 1, 2, 3, 4]}
B = [f"B{j}" for j in range(1, 11)]
rd = lambda p: json.loads(Path(p).read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wgz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, separators=(",", ":")).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:      # mtime=0: reproducible bytes
        f.write(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    global OUT
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="authorized directory outside the repository")
    a = ap.parse_args()
    OUT = a.out.resolve()
    if OUT == REPO or REPO in OUT.parents:
        ap.error("licensed benchmark data must remain outside the repository")
    R = a.results_root.resolve()
    A3, AKF, P10 = R / "forcesapien_a3", R / "forcesapien_a3_akf", R / "force_estimators_1plus10_cvpr"
    sources: dict[str, str] = {}

    def src(p: Path) -> Path:
        sources[str(p.relative_to(R))] = sha(p)
        return p

    if OUT.exists():
        shutil.rmtree(OUT)
    fits, preds, akf_imp = {}, {}, {}
    for s in SLUGS:
        p = src(A3 / "inputs" / f"{s}.G1.estimator_input.json")
        (OUT / "inputs").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, OUT / "inputs" / f"{s}.A3.estimator_input.json")
        # A3 truth, evaluation only
        t = rd(src(A3 / "gt_sealed" / f"{s}.A3.evaluation_truth.json"))
        wgz(OUT / "truth" / f"{s}.A3.truth.json.gz", t)
        fits[s] = {}
        for m, seeds in SEEDS.items():
            for k in seeds:
                r = rd(src(A3 / "est/raw" / s / f"{m}_primary_s{k}.json"))
                e = r.get("estimate") or {}
                fits[s][f"{m}/s{k}"] = dict(status=r["status"], inertia=e.get("inertia"), damping=e.get("damping"),
                                            q_hat=e.get("q_hat"), convergence=e.get("convergence"), runtime_s=e.get("runtime_s"),
                                            scales=r.get("scales"), shared_ls=r.get("shared_ls"), device=r.get("device"),
                                            gpu=r.get("gpu"), torch_version=r.get("torch_version"), slurm_job=r.get("slurm_job"),
                                            input_sha256=(r.get("input") or {}).get("sha256"))
        r = rd(src(AKF / "est/AKF-improved/raw" / s / "AKF-improved_primary_s0.json"))
        e = r["estimate"]
        akf_imp[s] = dict(status=r["status"], inertia=e["inertia"], damping=e["damping"], q_hat=e["q_hat"],
                          q_replay_A=e.get("q_replay_A"), selected=r["selected"], sigma=r["sigma"], scales=r["scales"],
                          shared_ls=r["shared_ls"], convergence=e["convergence"],
                          grid=[{k: g.get(k) for k in ("P0", "m_R", "ok", "I", "c", "replay_A_nrmse_pct", "nis_mean")} for g in r["grid"]],
                          input_sha256=r["input"]["sha256"])
        preds[s] = {}
        for b in B:
            fr = rd(src(P10 / "dataset/measured" / f"{s}.{b}.force_record.json"))
            smp = fr["samples"]
            wgz(OUT / "forces_B" / f"{s}.{b}.json.gz", dict(
                slug=s, interaction=b, units=dict(time="s", force="N", point="m"), frame="ForceSAPIEN world (simulator)",
                note="B force record reduced to the fields read by the replay (source: 1+10 benchmark measured record)",
                time_s=[x["time_s"] for x in smp],
                force_world=[x.get("force_vector_world", x.get("applied_force_world")) for x in smp],
                application_point_world=[x["application_point_world"] for x in smp],
                force_active=[bool(x["force_active"]) for x in smp]))
            bt = rd(src(P10 / "dataset/gt_sealed" / f"{s}.{b}.evaluation_truth.json"))
            wgz(OUT / "truth" / f"{s}.{b}.truth.json.gz", bt)
            ev = rd(src(A3 / "evaluation" / s / f"{b}.json"))
            ev2 = rd(src(AKF / "evaluation/B" / s / f"{b}.json"))
            row = dict(kappa=ev["kappa"], t_eval=ev["t_eval"], predictions={})
            for p in ev["predictions"]:
                row["predictions"][f"{p['method']}/s{p['seed']}"] = p.get("q_pred_primary")
            for p in ev2["predictions"]:
                if p["method"] == "AKF-improved":
                    row["predictions"]["AKF-improved/s0"] = p.get("q_pred")
                    assert abs(ev2["kappa"] - ev["kappa"]) <= 1e-15 * abs(ev["kappa"]), "kappa mismatch between A3 and AKF study"
            preds[s][b] = row
    wgz(OUT / "reference" / "a3_fits.json.gz", fits)
    wgz(OUT / "reference" / "akf_improved_fits.json.gz", akf_imp)
    wgz(OUT / "reference" / "b_predictions.json.gz", preds)
    scores = {s: rd(src(A3 / "scores" / f"{s}.json")) for s in SLUGS}
    wgz(OUT / "reference" / "a3_scores.json.gz", scores)
    (OUT / "reference").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src(A3 / "A3_RESULTS.json"), OUT / "reference" / "A3_RESULTS.json")
    summ = rd(src(AKF / "SUMMARY_B.json"))
    (OUT / "reference" / "AKF_STUDY_SUMMARY_B.json").write_text(json.dumps(summ, indent=1) + "\n")
    rows = [r for r in rd(src(AKF / "SCORES_B.json")) if r["method"] in ("AKF-improved", "AKF")]
    wgz(OUT / "reference" / "akf_study_scores_B.json.gz", rows)
    written = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    (OUT / "MANIFEST.json").write_text(json.dumps(dict(
        description="ForceSAPIEN A3 (identify) + B1-B10 (predict) benchmark bundle, built from the frozen results",
        source_root="comparison/freeartgs_all_objects/results (paths below are relative to it)",
        sources_sha256=sources, files_sha256=written), indent=1) + "\n")
    print("files", len(written), "sources", len(sources))


if __name__ == "__main__":
    main()
