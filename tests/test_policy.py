import numpy as np

from dmroute.policy import Recalibrator, cascade, pick
from dmroute.schema import RouterOutput, step_p


def test_pick_cheapest_above_threshold_else_strongest():
    p = np.array([[0.9, 0.95, 1.0], [0.1, 0.6, 0.9], [0.1, 0.2, 0.3]])
    assert pick(p, 0.5).tolist() == [0, 1, 2]
    assert pick(p, -np.inf).tolist() == [0, 0, 0]
    assert pick(p, np.inf).tolist() == [2, 2, 2]


def test_step_p_reproduces_discrete_choice_for_any_interior_tau():
    p = np.array([step_p(1, 4), step_p(3, 4)])
    for tau in (0.01, 0.5, 1.0):
        assert pick(p, tau).tolist() == [1, 3]


def test_recalibrator_is_monotone_across_models():
    rng = np.random.default_rng(0)
    p = rng.random((200, 3))
    q = (rng.random((200, 3)) < np.array([0.3, 0.6, 0.9])).astype(float)
    out = Recalibrator("platt").fit(p, q).transform(p)
    assert (np.diff(out, axis=1) >= -1e-12).all()


def test_cascade_defers_low_confidence_and_adds_cost():
    a = [
        RouterOutput(p=[1, 1], confidence=0.9, router_cost=1.0),
        RouterOutput(p=[1, 1], confidence=0.1, router_cost=1.0),
    ]
    b = [RouterOutput(p=[0, 1], choice=1, router_cost=5.0)] * 2
    out = cascade(a, b, 0.5)
    assert out[0] is a[0]
    assert out[1].p == [0, 1] and out[1].router_cost == 6.0
