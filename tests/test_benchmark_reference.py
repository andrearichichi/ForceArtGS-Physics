"""Regression of the ForceSAPIEN A3 -> B1-B10 protocol on the ORIGINAL fits (no re-fitting, CPU, ~1 min).

Checks that the library's force model, solver, scoring and aggregation reproduce, from the stored original (I, c):
  * the stored B1-B10 predicted trajectories                         (exact up to 1e-12 of the range)
  * the lever factor kappa of every object                           (exact)
  * every per-object, per-interaction score of results/forcesapien_a3/scores
  * the headline table A3_RESULTS.json and the AKF-improved row of forcesapien_a3_akf/SUMMARY_B.json
"""
import numpy as np
import pytest

from forceartgs_physics.validation import forcesapien_a3 as P

HEADLINE = {"PINN": 0.147, "Backprop": 0.176, "AKF": 0.184, "MLP": 0.512}       # brief values (rounded)
A3_NAMES = {"PINN": "PINN-V2", "Backprop": "Backprop", "MLP": "MLP-Joint", "AKF-benchmark": "AKF"}


@pytest.fixture(scope="module")
def evaluation(bench):
    return P.evaluate(P.reference_estimates(bench), bench, with_predictions=True)


def test_headline_numbers(evaluation, bench):
    tab = evaluation["per_method"]
    off = bench.reference("A3_RESULTS.json")["per_method"]
    for ours, orig in A3_NAMES.items():
        for k in ("I_err_median", "c_err_median", "B_object_median"):
            assert tab[ours][k] == pytest.approx(off[orig][k], rel=1e-12, abs=0), (ours, k)
        assert tab[ours]["B_pass"] == off[orig]["B_pass"] == 100
    akf = bench.reference("AKF_STUDY_SUMMARY_B.json")["summary"]["AKF-improved"]
    assert tab["AKF"]["B_object_median"] == pytest.approx(akf["B"], rel=1e-12)
    assert tab["AKF"]["I_err_median"] == pytest.approx(akf["I"], rel=1e-12)
    for m, v in HEADLINE.items():
        assert round(tab[m]["B_object_median"], 3) == v, m


def test_predicted_trajectories_identical(evaluation, bench):
    ref = bench.reference("b_predictions.json.gz")
    rename = {"PINN-V2": "PINN", "MLP-Joint": "MLP", "AKF": "AKF-benchmark", "Backprop": "Backprop", "AKF-improved": "AKF"}
    n = 0
    for slug, per_b in ref.items():
        for b, row in per_b.items():
            for key, q in row["predictions"].items():
                m, s = key.split("/s")
                ours = evaluation["predictions"][slug][b][f"{rename[m]}/s{s}"]
                assert np.max(np.abs(np.asarray(q) - ours)) <= 1e-12 * np.ptp(q), (slug, b, key)
                n += 1
    assert n == 10 * 10 * 11


def test_kappa_and_per_object_scores(evaluation, bench):
    scores = bench.reference("a3_scores.json.gz")
    preds = bench.reference("b_predictions.json.gz")
    for slug in bench.objects:
        for ours, orig in A3_NAMES.items():
            s = scores[slug]["summary"][orig]
            o = evaluation["per_object"][slug][ours]
            for b in bench.interactions_B:
                assert o["B"][b] == pytest.approx(s["B"][b], rel=1e-12), (slug, ours, b)
            assert o["I_err_pct"] == pytest.approx(s["I_err_pct"], rel=1e-12)
            assert o["qA3_nrmse_pct"] == pytest.approx(s["qA3_nrmse_pct"], rel=1e-12)
        assert evaluation["per_object"][slug]["PINN"]["kappa"] == preds[slug]["B1"]["kappa"]


def test_normalisation_and_shared_initializer_identical(bench):
    """common.normalize + common.ls on the bundled inputs give the scales / LS start stored in every original fit."""
    from forceartgs_physics.estimators import _bridge
    fits = bench.reference("a3_fits.json.gz")
    for slug in bench.objects:
        _, d, sc, init = _bridge.prepare(bench.interaction_A3(slug), "cartesian")
        ref = fits[slug]["Backprop/s0"]
        assert sc == ref["scales"], slug
        for k in ("inertia", "damping", "q0", "v0"):
            assert init[k] == ref["shared_ls"][k], (slug, k)


def test_estimators_never_open_truth(bench, monkeypatch):
    """Structural leak guard: fitting through the API on a bundled input opens no file under truth/."""
    import builtins
    import gzip
    opened = []
    real_open, real_gz = builtins.open, gzip.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(gzip, "open", lambda f, *a, **k: (opened.append(str(f)), real_gz(f, *a, **k))[1])
    from forceartgs_physics import api
    x = bench.interaction_A3("USB_100109")
    api.identify(x, "IntegralLS")
    assert not [p for p in opened if "/truth/" in p]
