from __future__ import annotations

"""Simple optimization "agent" layer.

This stays intentionally lightweight: it provides a clean objective interface
and a robust SciPy-based optimizer to choose an initial speed that hits a
target distance.

You can build richer agent/critic loops on top of these building blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .engine import BallisticSimulator, TrajectoryResult


@dataclass(frozen=True)
class TargetDistanceProblem:
    """Define a target-distance problem for the ballistic simulator."""

    target_distance: float
    angle_deg: float = 45.0
    max_time: float = 20.0
    steps: int = 800
    tolerance: float = 1.0
    v_min: float = 0.0
    v_max: float = 250.0


def _impact_x(result: TrajectoryResult) -> float:
    if result.impact is not None:
        return float(result.impact["x"])
    # If no impact in window, fall back to last simulated x
    return float(result.x[-1])


@dataclass
class PhysicaAgent:
    """Find launch parameters that best match a problem objective.

    Current implementation optimizes only the initial speed (v0) for a fixed
    angle. This is already useful and production-safe, while remaining easy to
    extend.
    """

    simulator: BallisticSimulator = field(default_factory=BallisticSimulator)

    def solve_target_distance(
        self,
        problem: TargetDistanceProblem,
        *,
        method: str = "bounded",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Solve for v0 that hits the target distance.

        Parameters
        ----------
        problem:
            Target definition and constraints.
        method:
            Optimization method (passed to scipy.optimize.minimize_scalar).
        """

        from scipy.optimize import minimize_scalar

        td = float(problem.target_distance)

        def objective(v0: float) -> float:
            res = self.simulator.simulate(
                v0=v0,
                angle_deg=problem.angle_deg,
                max_time=problem.max_time,
                steps=problem.steps,
            )
            x = _impact_x(res)
            return abs(x - td)

        bounds = (float(problem.v_min), float(problem.v_max))
        opt = minimize_scalar(objective, bounds=bounds, method=method)

        best_v = float(opt.x)
        best_res = self.simulator.simulate(
            v0=best_v,
            angle_deg=problem.angle_deg,
            max_time=problem.max_time,
            steps=problem.steps,
        )

        best_x = _impact_x(best_res)
        err = abs(best_x - td)
        success = bool(err <= float(problem.tolerance))

        return success, {
            "velocity": best_v,
            "impact": best_x,
            "error": float(err),
            "optimizer_success": bool(opt.success),
            "optimizer_message": str(opt.message),
            "result": best_res,
        }


# Backwards compatible name
ScientistAgent = PhysicaAgent
