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
    ElectrostaticsPINN,
    MagnetostaticsPINN,
    MaxwellPINN,
    PINNTrainer,
)

# Advanced Physics Domains (Phase II)
from .domains import (
    # Electromagnetism
    ElectromagneticField,
    ChargedParticle,
    ParticleInFieldSimulator,
    CyclotronMotion,
    # Hamiltonian mechanics
    HamiltonianSystem,
    HamiltonianSimulator,
    PoissonBracket,
    # Lagrangian mechanics
    LagrangianSystem,
    LagrangianSimulator,
    ActionPrinciple,
    # Thermodynamics
    ThermodynamicState,
    IdealGasEOS,
    VanDerWaalsEOS,
    CarnotCycle,
    OttoCycle,
    HeatEngine,
    EntropyCalculator,
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

# Phase III: Production-Ready Applications
# Phase III-A: Physics-Constrained Autonomous AI
from .autonomous_control import (
    ActionProposal,
    ActionStatus,
    AutonomousController,
    ConstraintViolationFeedback,
    PhysicsConstraintValidator,
)

# Phase III-B: Semiconductor Thermal & Power Digital Twin
from .semiconductor_twin import (
    ChipLayer,
    HeatDiffusion2D,
    MaterialProperties,
    MultiLayerChip,
    PowerDensityMap,
)

# Phase III-C: PINN Surrogates for Browser Deployment
from .surrogate_models import (
    BrowserDeployment,
    PhysicsSurrogate,
    SurrogateTrainer,
)

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
    "ElectrostaticsPINN",
    "MagnetostaticsPINN",
    "MaxwellPINN",
    "PINNTrainer",
    # Phase II Domains
    "ElectromagneticField",
    "ChargedParticle",
    "ParticleInFieldSimulator",
    "CyclotronMotion",
    "HamiltonianSystem",
    "HamiltonianSimulator",
    "PoissonBracket",
    "LagrangianSystem",
    "LagrangianSimulator",
    "ActionPrinciple",
    "ThermodynamicState",
    "IdealGasEOS",
    "VanDerWaalsEOS",
    "CarnotCycle",
    "OttoCycle",
    "HeatEngine",
    "EntropyCalculator",
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
    # Phase III-A: Autonomous Control
    "ActionProposal",
    "ActionStatus",
    "AutonomousController",
    "ConstraintViolationFeedback",
    "PhysicsConstraintValidator",
    # Phase III-B: Semiconductor Thermal Twin
    "ChipLayer",
    "HeatDiffusion2D",
    "MaterialProperties",
    "MultiLayerChip",
    "PowerDensityMap",
    # Phase III-C: Surrogate Models
    "BrowserDeployment",
    "PhysicsSurrogate",
    "SurrogateTrainer",
]
