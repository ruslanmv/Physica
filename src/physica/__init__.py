"""Project Physica: The Physics World Model for Agentic AI.

A revolutionary neuro-physical AI system that unifies:
- Large Language Models (Cognitive Layer)
- Differentiable Physics Simulators (Physical Layer)
- Physics-Informed Neural Networks (Learning Layer)

This creates AI systems that compute reality, not hallucinate it.
"""

# Core physics engine
from .engine import BallisticSimulator, TrajectoryResult

# Classic agent
from .agent import PhysicaAgent, ScientistAgent, TargetDistanceProblem

# Physics-Informed Neural Networks (Learning Layer)
from .pinn import (
    PINN,
    PINNConfig,
    MechanicsPINN,
    ThermodynamicsPINN,
    PINNTrainer,
)

# Cognitive Layer (LLM Integration)
from .cognitive import (
    IntentParser,
    PhysicsIntent,
    PlanningAgent,
    LLMBackend,
    MockLLM,
    get_llm_backend,
)

# Conservation Laws
from .conservation import (
    ConservationLaw,
    EnergyConservation,
    MomentumConservation,
    ConservationValidator,
)

# Neuro-Physical Loop (Core Innovation)
from .neuro_physical_loop import NeuroPhysicalLoop, PhysicsValidation

# Advanced Agentic AI
from .agentic import AutonomousPhysicist, AdaptiveOptimizer

__version__ = "1.0.0"

__all__ = [
    # Classic API
    "BallisticSimulator",
    "TrajectoryResult",
    "PhysicaAgent",
    "ScientistAgent",
    "TargetDistanceProblem",
    # PINNs
    "PINN",
    "PINNConfig",
    "MechanicsPINN",
    "ThermodynamicsPINN",
    "PINNTrainer",
    # Cognitive
    "IntentParser",
    "PhysicsIntent",
    "PlanningAgent",
    "LLMBackend",
    "MockLLM",
    "get_llm_backend",
    # Conservation
    "ConservationLaw",
    "EnergyConservation",
    "MomentumConservation",
    "ConservationValidator",
    # Neuro-Physical Loop
    "NeuroPhysicalLoop",
    "PhysicsValidation",
    # Agentic AI
    "AutonomousPhysicist",
    "AdaptiveOptimizer",
]
