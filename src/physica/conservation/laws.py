"""Conservation law implementations.

Conservation laws are fundamental constraints that no physical system can violate.
These are used to validate and correct AI predictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConservationViolation:
    """Represents a violation of a conservation law.

    Attributes
    ----------
    law_name:
        Name of the violated conservation law.
    expected:
        Expected conserved value.
    actual:
        Actual computed value.
    tolerance:
        Tolerance threshold.
    relative_error:
        Relative error magnitude.
    """

    law_name: str
    expected: float
    actual: float
    tolerance: float
    relative_error: float

    def __str__(self) -> str:
        return (
            f"{self.law_name} violation: "
            f"expected={self.expected:.6e}, "
            f"actual={self.actual:.6e}, "
            f"error={self.relative_error:.2%}"
        )


class ConservationLaw(ABC):
    """Abstract base class for conservation laws.

    Conservation laws provide hard constraints that physical systems must satisfy.
    They are used to validate AI predictions and guide corrections.
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize conservation law.

        Parameters
        ----------
        tolerance:
            Relative tolerance for conservation violations.
        """
        self.tolerance = tolerance

    @abstractmethod
    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute the conserved quantity for a given state.

        Parameters
        ----------
        state:
            Physical state dictionary.
        parameters:
            System parameters.

        Returns
        -------
        quantity:
            Value of the conserved quantity.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this conservation law."""
        pass

    def validate(
        self,
        initial_state: Dict[str, np.ndarray],
        final_state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[bool, Optional[ConservationViolation]]:
        """Validate conservation between initial and final states.

        Parameters
        ----------
        initial_state:
            Initial physical state.
        final_state:
            Final physical state.
        parameters:
            System parameters.

        Returns
        -------
        is_valid:
            True if conservation is satisfied within tolerance.
        violation:
            Violation details if not valid, None otherwise.
        """
        q_initial = self.compute_conserved_quantity(initial_state, parameters)
        q_final = self.compute_conserved_quantity(final_state, parameters)

        if abs(q_initial) < 1e-12:
            # Handle near-zero case
            abs_error = abs(q_final - q_initial)
            is_valid = abs_error < self.tolerance
            rel_error = abs_error
        else:
            rel_error = abs((q_final - q_initial) / q_initial)
            is_valid = rel_error < self.tolerance

        if not is_valid:
            violation = ConservationViolation(
                law_name=self.name(),
                expected=q_initial,
                actual=q_final,
                tolerance=self.tolerance,
                relative_error=rel_error,
            )
            return False, violation

        return True, None


class EnergyConservation(ConservationLaw):
    """Conservation of energy.

    In an isolated system, total energy (kinetic + potential) remains constant.
    """

    def name(self) -> str:
        return "Energy Conservation"

    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute total energy (kinetic + potential).

        Expected state keys:
        - 'position': array of shape (n_particles, n_dims)
        - 'velocity': array of shape (n_particles, n_dims)

        Expected parameters:
        - 'mass': float or array
        - 'gravity': float (default 9.81)
        """
        position = state.get("position", np.array([]))
        velocity = state.get("velocity", np.array([]))
        mass = parameters.get("mass", 1.0)
        gravity = parameters.get("gravity", 9.81)

        # Kinetic energy: (1/2) * m * v²
        if velocity.size > 0:
            v_squared = np.sum(velocity ** 2, axis=-1)
            if isinstance(mass, np.ndarray):
                kinetic = 0.5 * np.sum(mass * v_squared)
            else:
                kinetic = 0.5 * mass * np.sum(v_squared)
        else:
            kinetic = 0.0

        # Potential energy: m * g * h (height is last coordinate)
        if position.size > 0 and position.shape[-1] > 1:
            heights = position[..., -1]  # Last coordinate is vertical
            if isinstance(mass, np.ndarray):
                potential = np.sum(mass * gravity * heights)
            else:
                potential = mass * gravity * np.sum(heights)
        else:
            potential = 0.0

        return float(kinetic + potential)


class MomentumConservation(ConservationLaw):
    """Conservation of momentum.

    In an isolated system with no external forces, total momentum is conserved.
    """

    def name(self) -> str:
        return "Momentum Conservation"

    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute total momentum magnitude.

        Expected state keys:
        - 'velocity': array of shape (n_particles, n_dims)

        Expected parameters:
        - 'mass': float or array
        """
        velocity = state.get("velocity", np.array([]))
        mass = parameters.get("mass", 1.0)

        if velocity.size == 0:
            return 0.0

        # Total momentum: sum(m * v)
        if isinstance(mass, np.ndarray):
            momentum = np.sum(mass[:, np.newaxis] * velocity, axis=0)
        else:
            momentum = mass * np.sum(velocity, axis=0)

        # Return magnitude
        return float(np.linalg.norm(momentum))


class AngularMomentumConservation(ConservationLaw):
    """Conservation of angular momentum.

    In an isolated system, total angular momentum L = r × p is conserved.
    """

    def name(self) -> str:
        return "Angular Momentum Conservation"

    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute total angular momentum magnitude.

        Expected state keys:
        - 'position': array of shape (n_particles, 3)
        - 'velocity': array of shape (n_particles, 3)

        Expected parameters:
        - 'mass': float or array
        - 'origin': center of rotation (default: origin)
        """
        position = state.get("position", np.array([]))
        velocity = state.get("velocity", np.array([]))
        mass = parameters.get("mass", 1.0)
        origin = parameters.get("origin", np.zeros(3))

        if position.size == 0 or velocity.size == 0:
            return 0.0

        if position.shape[-1] != 3:
            raise ValueError("Angular momentum requires 3D positions")

        # r = position - origin
        r = position - origin

        # L = sum(m * r × v)
        if isinstance(mass, np.ndarray):
            L = np.sum(mass[:, np.newaxis] * np.cross(r, velocity), axis=0)
        else:
            L = mass * np.sum(np.cross(r, velocity), axis=0)

        return float(np.linalg.norm(L))


class MassConservation(ConservationLaw):
    """Conservation of mass.

    In a closed system, total mass remains constant.
    """

    def name(self) -> str:
        return "Mass Conservation"

    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute total mass.

        Expected parameters:
        - 'mass': float or array
        """
        mass = parameters.get("mass", 1.0)

        if isinstance(mass, np.ndarray):
            return float(np.sum(mass))
        else:
            return float(mass)


class ChargeConservation(ConservationLaw):
    """Conservation of electric charge.

    Total electric charge in an isolated system is conserved.
    """

    def name(self) -> str:
        return "Charge Conservation"

    def compute_conserved_quantity(
        self,
        state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> float:
        """Compute total charge.

        Expected parameters:
        - 'charge': float or array
        """
        charge = parameters.get("charge", 0.0)

        if isinstance(charge, np.ndarray):
            return float(np.sum(charge))
        else:
            return float(charge)


class ConservationValidator:
    """Validates multiple conservation laws simultaneously.

    This class orchestrates conservation law checking across a simulation
    or AI-generated plan.
    """

    def __init__(self, laws: Optional[List[ConservationLaw]] = None):
        """Initialize validator with conservation laws.

        Parameters
        ----------
        laws:
            List of conservation laws to enforce. If None, uses default set.
        """
        if laws is None:
            laws = [
                EnergyConservation(),
                MomentumConservation(),
            ]
        self.laws = laws

    def validate_trajectory(
        self,
        trajectory: List[Dict[str, np.ndarray]],
        parameters: Dict[str, Any],
    ) -> Tuple[bool, List[ConservationViolation]]:
        """Validate conservation laws along a trajectory.

        Parameters
        ----------
        trajectory:
            List of states representing a time evolution.
        parameters:
            System parameters.

        Returns
        -------
        is_valid:
            True if all laws are satisfied.
        violations:
            List of all violations found.
        """
        if len(trajectory) < 2:
            return True, []

        initial_state = trajectory[0]
        violations = []

        for state in trajectory[1:]:
            for law in self.laws:
                is_valid, violation = law.validate(
                    initial_state, state, parameters
                )
                if not is_valid and violation is not None:
                    violations.append(violation)

        return len(violations) == 0, violations

    def validate_single(
        self,
        initial_state: Dict[str, np.ndarray],
        final_state: Dict[str, np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[bool, List[ConservationViolation]]:
        """Validate conservation between two states.

        Parameters
        ----------
        initial_state:
            Initial state.
        final_state:
            Final state.
        parameters:
            System parameters.

        Returns
        -------
        is_valid:
            True if all laws satisfied.
        violations:
            List of violations.
        """
        violations = []

        for law in self.laws:
            is_valid, violation = law.validate(
                initial_state, final_state, parameters
            )
            if not is_valid and violation is not None:
                violations.append(violation)

        return len(violations) == 0, violations
