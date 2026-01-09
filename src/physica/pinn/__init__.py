"""Physics-Informed Neural Networks (PINNs) module.

This module provides the learning layer that enables AI systems to learn
from physical constraints and correct predictions based on causal relationships.
"""

from .base import PINN, PINNConfig, PhysicsLoss
from .electromagnetism import (
    ElectrostaticsPINN,
    MagnetostaticsPINN,
    MaxwellPINN,
)
from .mechanics import MechanicsPINN
from .thermodynamics import ThermodynamicsPINN
from .trainer import PINNTrainer

__all__ = [
    "PINN",
    "PINNConfig",
    "PhysicsLoss",
    "MechanicsPINN",
    "ThermodynamicsPINN",
    "ElectrostaticsPINN",
    "MagnetostaticsPINN",
    "MaxwellPINN",
    "PINNTrainer",
]
