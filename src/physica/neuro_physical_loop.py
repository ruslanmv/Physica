"""The Neuro-Physical Intelligence Loop.

This is the core innovation of Project Physica: a closed-loop system where:
1. Cognitive Layer (LLM) interprets intent and proposes solutions
2. Physical Layer (simulators) validates against natural laws
3. Learning Layer (PINNs) corrects and optimizes based on physics

The loop iterates until solutions satisfy all physical constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cognitive.intent import IntentParser, PhysicsIntent, PlanningAgent
from .cognitive.llm import LLMBackend, MockLLM
from .conservation.laws import ConservationValidator, ConservationViolation
from .engine import BallisticSimulator

logger = logging.getLogger(__name__)


@dataclass
class PhysicsValidation:
    """Result of physics validation.

    Attributes
    ----------
    is_valid:
        Whether the proposal satisfies physical constraints.
    violations:
        List of conservation law violations.
    simulation_result:
        Result from physics simulation (if applicable).
    error_message:
        Human-readable error description.
    """

    is_valid: bool
    violations: List[ConservationViolation] = field(default_factory=list)
    simulation_result: Optional[Any] = None
    error_message: str = ""

    def feedback_message(self) -> str:
        """Generate feedback message for cognitive layer."""
        if self.is_valid:
            return "All physical constraints satisfied."

        msg = "Physical constraint violations detected:\n"
        for v in self.violations:
            msg += f"  - {str(v)}\n"

        if self.error_message:
            msg += f"\nAdditional errors: {self.error_message}"

        return msg


@dataclass
class LoopIteration:
    """Record of one iteration through the neuro-physical loop.

    Attributes
    ----------
    iteration:
        Iteration number.
    intent:
        Cognitive layer proposal.
    validation:
        Physical layer validation result.
    correction:
        Learning layer correction (if any).
    """

    iteration: int
    intent: PhysicsIntent
    validation: PhysicsValidation
    correction: Optional[Dict[str, Any]] = None


class NeuroPhysicalLoop:
    """The Neuro-Physical Intelligence Loop.

    This is the main orchestrator that implements the three-layer architecture:

    ┌─────────────────┐
    │ Cognitive Layer │  (LLM: Intent → Parameters)
    └────────┬────────┘
             │ proposes
             ▼
    ┌─────────────────┐
    │ Physical Layer  │  (Simulator: Validates against laws)
    └────────┬────────┘
             │ validates
             ▼
    ┌─────────────────┐
    │ Learning Layer  │  (PINN: Corrects via gradients)
    └────────┬────────┘
             │ corrects
             └──────────► (loop until valid)

    Key Features:
    - Self-correcting: violations feedback to cognitive layer
    - Physically grounded: all outputs respect conservation laws
    - Explainable: full trace of reasoning and corrections
    """

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        max_iterations: int = 5,
        enable_pinn: bool = False,
    ):
        """Initialize neuro-physical loop.

        Parameters
        ----------
        llm:
            LLM backend for cognitive layer.
        max_iterations:
            Maximum correction iterations.
        enable_pinn:
            Whether to use PINN-based corrections (requires training).
        """
        self.llm = llm or MockLLM()
        self.max_iterations = max_iterations
        self.enable_pinn = enable_pinn

        # Initialize components
        self.intent_parser = IntentParser(llm=self.llm)
        self.planner = PlanningAgent(llm=self.llm)
        self.validator = ConservationValidator()
        self.simulator = BallisticSimulator()

        # Iteration history
        self.history: List[LoopIteration] = []

    def execute(
        self,
        user_request: str,
        verbose: bool = True,
    ) -> Tuple[PhysicsIntent, Any, List[LoopIteration]]:
        """Execute the full neuro-physical loop.

        Parameters
        ----------
        user_request:
            Natural language physics problem.
        verbose:
            Whether to log progress.

        Returns
        -------
        final_intent:
            Validated and corrected physics intent.
        result:
            Simulation or optimization result.
        history:
            Full iteration history for explainability.
        """
        if verbose:
            logger.info(f"[Neuro-Physical Loop] Processing: {user_request}")

        self.history = []

        # Step 1: Cognitive Layer - Parse intent
        if verbose:
            logger.info("[Cognitive Layer] Parsing intent...")

        intent = self.intent_parser.parse(user_request)

        if verbose:
            logger.info(f"[Cognitive Layer] Parsed intent: {intent.intent}")
            logger.info(f"[Cognitive Layer] Parameters: {intent.parameters}")

        # Iterative refinement loop
        for iteration in range(self.max_iterations):
            if verbose:
                logger.info(f"\n[Loop Iteration {iteration + 1}/{self.max_iterations}]")

            # Step 2: Physical Layer - Validate
            if verbose:
                logger.info("[Physical Layer] Validating against physics...")

            validation = self._validate_physics(intent)

            # Step 3: Learning Layer - Correct if needed
            correction = None
            if not validation.is_valid:
                if verbose:
                    logger.warning(
                        f"[Physical Layer] Violations detected: {len(validation.violations)}"
                    )
                    for v in validation.violations:
                        logger.warning(f"  - {v}")

                if verbose:
                    logger.info("[Learning Layer] Attempting correction...")

                correction = self._apply_correction(intent, validation)

                # Update intent with correction
                if correction:
                    intent.parameters.update(correction)
                    if verbose:
                        logger.info(f"[Learning Layer] Applied correction: {correction}")

            # Record iteration
            loop_iter = LoopIteration(
                iteration=iteration,
                intent=intent,
                validation=validation,
                correction=correction,
            )
            self.history.append(loop_iter)

            # Check if we're done
            if validation.is_valid:
                if verbose:
                    logger.info("[Success] All physical constraints satisfied!")
                break

            if iteration == self.max_iterations - 1 and verbose:
                logger.warning(
                    "[Warning] Max iterations reached. Constraints may not be fully satisfied."
                )

        # Execute final simulation
        result = self._execute_simulation(intent)

        return intent, result, self.history

    def _validate_physics(self, intent: PhysicsIntent) -> PhysicsValidation:
        """Validate intent against physical laws.

        Parameters
        ----------
        intent:
            Physics intent to validate.

        Returns
        -------
        validation:
            Validation result with any violations.
        """
        try:
            # Simulate based on intent
            if intent.intent == "simulate_projectile" or intent.intent == "optimize_trajectory":
                params = intent.parameters

                # Run simulation
                result = self.simulator.simulate(
                    v0=params.get("initial_velocity", 50.0),
                    angle_deg=params.get("angle_degrees", 45.0),
                    max_time=params.get("max_time", 20.0),
                    steps=params.get("steps", 800),
                )

                # Extract trajectory
                trajectory = []
                for i in range(len(result.t)):
                    state = {
                        "position": np.array([[result.x[i], result.y_pos[i]]]),
                        "velocity": np.array([[result.vx[i], result.vy[i]]]),
                    }
                    trajectory.append(state)

                # Validate conservation laws
                system_params = {
                    "mass": params.get("mass", 1.0),
                    "gravity": self.simulator.gravity,
                }

                is_valid, violations = self.validator.validate_trajectory(
                    trajectory, system_params
                )

                return PhysicsValidation(
                    is_valid=is_valid,
                    violations=violations,
                    simulation_result=result,
                )

            else:
                # For unsupported intents, just accept for now
                return PhysicsValidation(is_valid=True)

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return PhysicsValidation(
                is_valid=False,
                error_message=str(e),
            )

    def _apply_correction(
        self,
        intent: PhysicsIntent,
        validation: PhysicsValidation,
    ) -> Optional[Dict[str, Any]]:
        """Apply corrections based on physics violations.

        Parameters
        ----------
        intent:
            Current intent.
        validation:
            Validation result with violations.

        Returns
        -------
        corrections:
            Dictionary of parameter corrections.
        """
        corrections = {}

        # Simple heuristic corrections (can be enhanced with PINN)
        for violation in validation.violations:
            if "Energy" in violation.law_name:
                # Energy violation: adjust drag coefficient
                current_drag = intent.parameters.get("drag_coefficient", 0.0)
                if violation.relative_error > 0:
                    # Losing too much energy: reduce drag
                    corrections["drag_coefficient"] = max(0.0, current_drag * 0.8)
                else:
                    # Gaining energy (impossible): increase drag
                    corrections["drag_coefficient"] = current_drag * 1.2

            elif "Momentum" in violation.law_name:
                # Momentum violation: likely numerical issue
                # Increase simulation resolution
                current_steps = intent.parameters.get("steps", 800)
                corrections["steps"] = int(current_steps * 1.5)

        return corrections if corrections else None

    def _execute_simulation(self, intent: PhysicsIntent) -> Any:
        """Execute final simulation with validated parameters.

        Parameters
        ----------
        intent:
            Validated physics intent.

        Returns
        -------
        result:
            Simulation result.
        """
        params = intent.parameters

        if intent.intent == "simulate_projectile":
            return self.simulator.simulate(
                v0=params.get("initial_velocity", 50.0),
                angle_deg=params.get("angle_degrees", 45.0),
                max_time=params.get("max_time", 20.0),
                steps=params.get("steps", 800),
            )

        elif intent.intent == "optimize_trajectory":
            from .agent import PhysicaAgent, TargetDistanceProblem

            agent = PhysicaAgent(simulator=self.simulator)
            problem = TargetDistanceProblem(
                target_distance=params.get("target_distance", 300.0),
                angle_deg=params.get("angle_degrees", 45.0),
                tolerance=params.get("tolerance", 2.0),
            )

            success, result = agent.solve_target_distance(problem)
            return result

        else:
            raise ValueError(f"Unsupported intent: {intent.intent}")

    def explain(self) -> str:
        """Generate human-readable explanation of the loop execution.

        Returns
        -------
        explanation:
            Natural language description of what happened.
        """
        if not self.history:
            return "No execution history available."

        explanation = f"Neuro-Physical Loop executed {len(self.history)} iterations:\n\n"

        for i, iteration in enumerate(self.history, 1):
            explanation += f"Iteration {i}:\n"
            explanation += f"  Intent: {iteration.intent.intent}\n"
            explanation += f"  Parameters: {iteration.intent.parameters}\n"

            if iteration.validation.is_valid:
                explanation += "  ✓ All physical constraints satisfied\n"
            else:
                explanation += f"  ✗ {len(iteration.validation.violations)} violations detected\n"
                for v in iteration.validation.violations:
                    explanation += f"    - {v.law_name}: {v.relative_error:.2%} error\n"

            if iteration.correction:
                explanation += f"  Corrections applied: {iteration.correction}\n"

            explanation += "\n"

        final = self.history[-1]
        if final.validation.is_valid:
            explanation += "Result: Successfully converged to physically valid solution."
        else:
            explanation += "Result: Did not fully converge (max iterations reached)."

        return explanation
