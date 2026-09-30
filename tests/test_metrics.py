import numpy as np
import pytest

from dmroute.metrics import calibration, cost_quality, stats


def test_hull_drops_dominated_and_concave_points():
    pts = [(1, 0.5), (2, 0.55), (3, 0.9), (4, 0.8), (5, 0.95)]
    # (2, .55) sits under the segment (1,.5)-(3,.9); (4,.8) is dominated by (3,.9).
    assert cost_quality.hull(pts) == [(1, 0.5), (3, 0.9), (5, 0.95)]


def test_hull_value_interpolates_and_is_flat_past_end():
    h = [(1.0, 0.5), (3.0, 0.9)]
    assert cost_quality.hull_value(h, 2.0) == pytest.approx(0.7)
    assert cost_quality.hull_value(h, 10.0) == pytest.approx(0.9)
    assert cost_quality.hull_value(h, 0.5) == 0.0


def test_aiq_of_a_flat_hull_is_its_quality():
    h = [(1.0, 0.8)]
    assert cost_quality.aiq_linear(h, 1.0, 5.0) == pytest.approx(0.8)
    assert cost_quality.aiq(h, 1.0, 5.0) == pytest.approx(0.8)


def test_apgr_perfect_and_reversed_scores():
    # Weak fails exactly on the first half; strong always succeeds.
    q = np.array([[0, 1]] * 50 + [[1, 1]] * 50, dtype=float)
    perfect = np.column_stack([q[:, 0], np.ones(100)])
    reversed_ = np.column_stack([1 - q[:, 0], np.ones(100)])
    good = cost_quality.apgr(perfect, q, 0, 1)
    bad = cost_quality.apgr(reversed_, q, 0, 1)
    assert good["apgr"] > 0.7
    assert bad["apgr"] < 0.3
    assert good["cpt50"] == pytest.approx(0.25, abs=0.01)


def test_tier_metrics_traj_pass():
    choice = np.array([0, 1, 0, 3])
    target = np.array([0, 1, 1, 2])
    groups = ["a", "a", "b", "b"]
    m = cost_quality.tier_metrics(choice, target, groups)
    assert m["row_pass"] == 0.75
    assert m["row_exact"] == 0.5
    assert m["traj_pass"] == 0.5


def test_ece_zero_when_calibrated_and_brier():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert calibration.ece(p, y, n_bins=2) == 0.0
    assert calibration.brier(np.full(4, 0.5), y) == 0.25


def test_aurc_prefers_confident_correct():
    correct = np.array([1, 1, 0, 0], dtype=float)
    good = calibration.aurc(np.array([0.9, 0.8, 0.2, 0.1]), correct)
    bad = calibration.aurc(np.array([0.1, 0.2, 0.8, 0.9]), correct)
    assert good < bad


def test_paired_stats():
    a = np.array([1, 1, 1, 0] * 25, dtype=float)
    b = np.array([0, 1, 1, 0] * 25, dtype=float)
    boot = stats.paired_bootstrap(a, b)
    assert boot["mean_diff"] == pytest.approx(0.25)
    assert boot["ci95"][0] > 0
    assert stats.mcnemar_exact(a, b)["p_value"] < 0.001


def test_oracle_sweep_dominates_single_models():
    rng = np.random.default_rng(0)
    q = (rng.random((300, 3)) < [0.3, 0.6, 0.9]).astype(float)
    c = np.tile([0.1, 1.0, 10.0], (300, 1))
    pts = cost_quality.oracle_sweep(q, c)
    h = cost_quality.hull([(p["cost"], p["quality"]) for p in pts])
    for cost, qual in cost_quality.single_model_points(q, c):
        assert cost_quality.hull_value(h, cost) >= qual - 1e-9


def test_bootstrap_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.random(200) < 0.4
    p = np.round(rng.random(200) + 0.3 * y, 1)  # rounding creates ties
    out = stats.bootstrap_auroc(p, y)
    assert out["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert out["ci95"][0] < out["auroc"] < out["ci95"][1]
