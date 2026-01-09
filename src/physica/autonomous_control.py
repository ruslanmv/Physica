"""Phase III-A: Physics-Constrained Autonomous AI.

Implements an autonomous control system where AI decisions are validated
against physical laws before execution, ensuring safety and trustworthiness.

This is the cornerstone of "AI that is physically trustworthy" for:
- Industry automation
- Robotics
- Energy systems
- Manufacturing optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from .conservation import ConservationLaw


class ActionStatus(str, Enum):
    """Status of a proposed action."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class ActionProposal:
    """A proposed action from the AI decision layer.

    Attributes
    ----------
    action_id:
        Unique identifier for this action.
    action_type:
        Type of action (e.g., "move", "apply_force", "set_temperature").
    parameters:
        Dictionary of action parameters.
    expected_state:
        Expected physical state after action execution.
    priority:
        Action priority (0-1, higher is more important).
    """

    action_id: str
    action_type: str
    parameters: dict[str, Any]
    expected_state: dict[str, float]
    priority: float = 0.5
    status: ActionStatus = ActionStatus.PROPOSED
    rejection_reason: str | None = None


@dataclass
class ConstraintViolation:
    """Record of a physics constraint violation.

    Attributes
    ----------
    action_id:
        ID of the action that caused the violation.
    constraint_type:
        Type of constraint violated (e.g., "energy", "momentum").
    violation_magnitude:
        How much the constraint was violated.
    description:
        Human-readable description.
    """

    action_id: str
    constraint_type: str
    violation_magnitude: float
    description: str
    timestamp: float


class PhysicsConstraintValidator:
    """Validates proposed actions against physical laws.

    This is the core safety layer that prevents AI from proposing
    physically impossible or dangerous actions.
    """

    def __init__(
        self,
        conservation_laws: list[ConservationLaw] | None = None,
        custom_constraints: list[Callable] | None = None,
    ):
        """Initialize physics validator.

        Parameters
        ----------
        conservation_laws:
            List of conservation laws to enforce.
        custom_constraints:
            Additional custom constraint functions.
        """
        self.conservation_laws = conservation_laws or []
        self.custom_constraints = custom_constraints or []
        self.violation_history: list[ConstraintViolation] = []

    def validate_action(
        self,
        proposal: ActionProposal,
        current_state: dict[str, float],
    ) -> tuple[bool, str | None]:
        """Validate if an action is physically feasible.

        Parameters
        ----------
        proposal:
            The proposed action to validate.
        current_state:
            Current physical state of the system.

        Returns
        -------
        is_valid:
            True if action passes all constraints.
        rejection_reason:
            Description if rejected, None otherwise.
        """
        # Check conservation laws
        for law in self.conservation_laws:
            is_valid, reason = self._check_conservation(
                proposal, current_state, law
            )
            if not is_valid:
                self._record_violation(proposal, law.__class__.__name__, reason)
                return False, reason

        # Check custom constraints
        for constraint in self.custom_constraints:
            is_valid, reason = constraint(proposal, current_state)
            if not is_valid:
                self._record_violation(proposal, "custom", reason)
                return False, reason

        # Check physical bounds
        is_valid, reason = self._check_physical_bounds(proposal)
        if not is_valid:
            self._record_violation(proposal, "bounds", reason)
            return False, reason

        return True, None

    def _check_conservation(
        self,
        proposal: ActionProposal,
        current_state: dict[str, float],
        law: ConservationLaw,
    ) -> tuple[bool, str | None]:
        """Check if action violates a conservation law."""
        expected = proposal.expected_state

        # Get conserved quantity from current and expected states
        current_value = current_state.get(law.__class__.__name__.lower(), 0.0)
        expected_value = expected.get(law.__class__.__name__.lower(), 0.0)

        # Check if conservation is maintained
        violation = abs(expected_value - current_value)
        tolerance = 1e-6

        if violation > tolerance:
            return (
                False,
                f"Violates {law.__class__.__name__}: "
                f"change of {violation:.2e} exceeds tolerance",
            )

        return True, None

    def _check_physical_bounds(
        self, proposal: ActionProposal
    ) -> tuple[bool, str | None]:
        """Check if action parameters are within physical bounds."""
        params = proposal.parameters

        # Check for negative energy
        if "energy" in params and params["energy"] < 0:
            return False, "Negative energy is unphysical"

        # Check for superluminal velocities
        if "velocity" in params:
            v = np.array(params["velocity"])
            speed = np.linalg.norm(v)
            c = 3e8  # Speed of light
            if speed >= c:
                return False, f"Velocity {speed:.2e} m/s exceeds speed of light"

        # Check temperature bounds
        if "temperature" in params:
            T = params["temperature"]
            if T < 0:
                return False, f"Negative absolute temperature: {T} K"
            if T > 1e7:  # Reasonable upper bound for most applications
                return False, f"Temperature {T:.2e} K exceeds practical limits"

        return True, None

    def _record_violation(
        self, proposal: ActionProposal, constraint_type: str, description: str
    ):
        """Record a constraint violation for learning."""
        import time

        violation = ConstraintViolation(
            action_id=proposal.action_id,
            constraint_type=constraint_type,
            violation_magnitude=1.0,  # Could be calculated
            description=description,
            timestamp=time.time(),
        )
        self.violation_history.append(violation)

    def get_violation_statistics(self) -> dict[str, Any]:
        """Get statistics about constraint violations."""
        if not self.violation_history:
            return {"total_violations": 0}

        by_type = {}
        for v in self.violation_history:
            by_type[v.constraint_type] = by_type.get(v.constraint_type, 0) + 1

        return {
            "total_violations": len(self.violation_history),
            "by_type": by_type,
            "recent_violations": self.violation_history[-10:],
        }


class ConstraintViolationFeedback:
    """Learning system that improves action proposals based on violations.

    This implements the feedback loop where the AI learns from
    rejected actions to propose better ones in the future.
    """

    def __init__(self, learning_rate: float = 0.1):
        """Initialize feedback system.

        Parameters
        ----------
        learning_rate:
            How quickly to adapt based on violations.
        """
        self.learning_rate = learning_rate
        self.violation_weights: dict[str, float] = {}
        self.action_success_rates: dict[str, float] = {}

    def process_violation(self, violation: ConstraintViolation):
        """Process a violation and update learning weights.

        Parameters
        ----------
        violation:
            The constraint violation to learn from.
        """
        # Increase penalty weight for this constraint type
        current_weight = self.violation_weights.get(violation.constraint_type, 1.0)
        self.violation_weights[violation.constraint_type] = (
            current_weight * (1 + self.learning_rate)
        )

    def adjust_proposal(
        self, proposal: ActionProposal, violation_history: list[ConstraintViolation]
    ) -> ActionProposal:
        """Adjust a proposal based on past violations.

        Parameters
        ----------
        proposal:
            Original proposal to adjust.
        violation_history:
            History of past violations.

        Returns
        -------
        adjusted_proposal:
            Modified proposal more likely to succeed.
        """
        # Find similar past violations
        similar_violations = [
            v for v in violation_history if v.action_id.startswith(proposal.action_type)
        ]

        if not similar_violations:
            return proposal

        # Adjust parameters to avoid common violations
        adjusted_params = proposal.parameters.copy()

        # Conservative adjustment: reduce magnitudes by 10%
        for key, value in adjusted_params.items():
            if isinstance(value, (int, float)):
                adjusted_params[key] = value * 0.9

        proposal.parameters = adjusted_params
        return proposal

    def update_success_rate(self, action_type: str, succeeded: bool):
        """Update success rate statistics for an action type.

        Parameters
        ----------
        action_type:
            Type of action.
        succeeded:
            Whether the action succeeded.
        """
        current_rate = self.action_success_rates.get(action_type, 0.5)
        # Exponential moving average
        new_rate = current_rate * (1 - self.learning_rate) + (
            1.0 if succeeded else 0.0
        ) * self.learning_rate
        self.action_success_rates[action_type] = new_rate


class AutonomousController:
    """Physics-constrained autonomous control system.

    Orchestrates the entire decision-validate-execute-learn loop.
    This is the top-level API for trustworthy autonomous AI.
    """

    def __init__(
        self,
        validator: PhysicsConstraintValidator,
        feedback: ConstraintViolationFeedback | None = None,
    ):
        """Initialize autonomous controller.

        Parameters
        ----------
        validator:
            Physics constraint validator.
        feedback:
            Feedback learning system (optional).
        """
        self.validator = validator
        self.feedback = feedback or ConstraintViolationFeedback()
        self.action_queue: list[ActionProposal] = []
        self.execution_history: list[tuple[ActionProposal, bool]] = []

    def propose_action(
        self,
        action_type: str,
        parameters: dict[str, Any],
        expected_state: dict[str, float],
        priority: float = 0.5,
    ) -> ActionProposal:
        """Propose a new action for validation.

        Parameters
        ----------
        action_type:
            Type of action to propose.
        parameters:
            Action parameters.
        expected_state:
            Expected physical state after execution.
        priority:
            Action priority (0-1).

        Returns
        -------
        proposal:
            The action proposal.
        """
        import uuid

        action_id = f"{action_type}_{uuid.uuid4().hex[:8]}"

        proposal = ActionProposal(
            action_id=action_id,
            action_type=action_type,
            parameters=parameters,
            expected_state=expected_state,
            priority=priority,
        )

        # Learn from past violations
        proposal = self.feedback.adjust_proposal(
            proposal, self.validator.violation_history
        )

        self.action_queue.append(proposal)
        return proposal

    def validate_and_execute(
        self,
        proposal: ActionProposal,
        current_state: dict[str, float],
        execute_fn: Callable[[ActionProposal], bool] | None = None,
    ) -> tuple[bool, str | None]:
        """Validate and optionally execute an action.

        Parameters
        ----------
        proposal:
            Action to validate and execute.
        current_state:
            Current physical state.
        execute_fn:
            Optional function to actually execute the action.

        Returns
        -------
        success:
            True if action was executed successfully.
        message:
            Status message.
        """
        # Validate
        is_valid, reason = self.validator.validate_action(proposal, current_state)

        if not is_valid:
            proposal.status = ActionStatus.REJECTED
            proposal.rejection_reason = reason
            self.feedback.update_success_rate(proposal.action_type, succeeded=False)
            return False, f"Action rejected: {reason}"

        proposal.status = ActionStatus.VALIDATED

        # Execute if function provided
        if execute_fn is not None:
            try:
                success = execute_fn(proposal)
                proposal.status = (
                    ActionStatus.EXECUTED if success else ActionStatus.FAILED
                )
                self.feedback.update_success_rate(proposal.action_type, succeeded=success)
                self.execution_history.append((proposal, success))
                return success, "Action executed" if success else "Execution failed"
            except Exception as e:
                proposal.status = ActionStatus.FAILED
                self.feedback.update_success_rate(proposal.action_type, succeeded=False)
                return False, f"Execution error: {str(e)}"

        return True, "Action validated (not executed)"

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics.

        Returns
        -------
        stats:
            Dictionary of statistics.
        """
        total_actions = len(self.execution_history)
        successful = sum(1 for _, success in self.execution_history if success)

        return {
            "total_actions": total_actions,
            "successful_actions": successful,
            "success_rate": successful / total_actions if total_actions > 0 else 0.0,
            "queued_actions": len(self.action_queue),
            "violation_stats": self.validator.get_violation_statistics(),
            "action_success_rates": self.feedback.action_success_rates,
        }
