import math

import numpy as np

from physica.engine import BallisticSimulator


def test_gravity_vertical_shot_has_peak_and_returns():
    sim = BallisticSimulator(drag_coeff=0.0)
    res = sim.simulate(v0=10.0, angle_deg=90.0, max_time=5.0, steps=400)

    y = res.y_pos
    assert float(np.max(y)) > 0.0
    # Eventually comes back to (or below) ground.
    assert res.impact is not None
    assert abs(float(res.impact["y"])) < 1e-6


def test_no_drag_range_matches_analytic_45deg():
    g = 9.81
    v0 = 50.0
    sim = BallisticSimulator(gravity=g, drag_coeff=0.0)
    res = sim.simulate(v0=v0, angle_deg=45.0, max_time=20.0, steps=2000)
    assert res.impact is not None

    expected = (v0 * v0) / g
    got = float(res.impact["x"])
    assert math.isfinite(got)
    # Numerical integration tolerance: tight but reasonable.
    assert abs(got - expected) / expected < 0.01


def test_higher_drag_reduces_range():
    sim_low = BallisticSimulator(drag_coeff=0.0)
    sim_high = BallisticSimulator(drag_coeff=0.5)

    res_low = sim_low.simulate(v0=50.0, angle_deg=45.0, max_time=20.0, steps=2000)
    res_high = sim_high.simulate(v0=50.0, angle_deg=45.0, max_time=20.0, steps=2000)

    assert res_low.impact is not None
    assert res_high.impact is not None
    assert float(res_low.impact["x"]) > float(res_high.impact["x"])
