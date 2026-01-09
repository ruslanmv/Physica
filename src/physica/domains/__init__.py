"""Physics domains: specialized simulators for different physical systems.

This module provides production-ready implementations of:
- Electromagnetism: Maxwell's equations, Lorentz force, field simulations
- Hamiltonian mechanics: Energy-preserving symplectic integrators
- Lagrangian mechanics: Euler-Lagrange equations, variational methods
- Thermodynamics: State equations, entropy, thermodynamic cycles
"""

from .electromagnetism import (
    ChargedParticle,
    CyclotronMotion,
    ElectromagneticField,
    ParticleInFieldSimulator,
    oscillating_E_field,
    uniform_B_field,
    uniform_E_field,
)
from .hamiltonian import (
    HamiltonianSimulator,
    HamiltonianSystem,
    PoissonBracket,
    harmonic_oscillator,
    kepler_problem,
    simple_pendulum_hamiltonian,
)
from .lagrangian import (
    ActionPrinciple,
    LagrangianSimulator,
    LagrangianSystem,
    brachistochrone_problem,
    double_pendulum_lagrangian,
    simple_pendulum_lagrangian,
)
from .thermodynamics import (
    AdiabaticProcess,
    CarnotCycle,
    EntropyCalculator,
    HeatEngine,
    IdealGasEOS,
    IsobaricProcess,
    IsochoricProcess,
    IsothermalProcess,
    OttoCycle,
    StateEquation,
    ThermodynamicCycle,
    ThermodynamicState,
    VanDerWaalsEOS,
    boltzmann_entropy,
    maxwell_boltzmann_distribution,
    phase_transition_entropy,
)

__all__ = [
    # Electromagnetism
    "ChargedParticle",
    "CyclotronMotion",
    "ElectromagneticField",
    "ParticleInFieldSimulator",
    "oscillating_E_field",
    "uniform_B_field",
    "uniform_E_field",
    # Hamiltonian mechanics
    "HamiltonianSimulator",
    "HamiltonianSystem",
    "PoissonBracket",
    "harmonic_oscillator",
    "kepler_problem",
    "simple_pendulum_hamiltonian",
    # Lagrangian mechanics
    "ActionPrinciple",
    "LagrangianSimulator",
    "LagrangianSystem",
    "brachistochrone_problem",
    "double_pendulum_lagrangian",
    "simple_pendulum_lagrangian",
    # Thermodynamics
    "AdiabaticProcess",
    "CarnotCycle",
    "EntropyCalculator",
    "HeatEngine",
    "IdealGasEOS",
    "IsobaricProcess",
    "IsochoricProcess",
    "IsothermalProcess",
    "OttoCycle",
    "StateEquation",
    "ThermodynamicCycle",
    "ThermodynamicState",
    "VanDerWaalsEOS",
    "boltzmann_entropy",
    "maxwell_boltzmann_distribution",
    "phase_transition_entropy",
]
