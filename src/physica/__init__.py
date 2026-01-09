"""Project Physica (Physics-World-Model).

Public API exports the main simulator and agent.
"""

from .engine import BallisticSimulator, TrajectoryResult
from .agent import PhysicaAgent, ScientistAgent, TargetDistanceProblem

__all__ = [
    "BallisticSimulator",
    "TrajectoryResult",
    "PhysicaAgent",
    "ScientistAgent",
    "TargetDistanceProblem",
]
