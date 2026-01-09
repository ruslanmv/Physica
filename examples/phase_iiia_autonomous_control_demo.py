"""Phase III-A Demonstration: Physics-Constrained Autonomous AI.

Showcases autonomous decision-making with physics validation,
demonstrating "AI that is physically trustworthy" for:
- Industry automation
- Robotics control
- Energy systems
- Manufacturing optimization
"""

import numpy as np

from physica.autonomous_control import (
    ActionProposal,
    AutonomousController,
    ConstraintViolationFeedback,
    PhysicsConstraintValidator,
)
from physica.conservation import EnergyConservation, MomentumConservation


def demo_basic_validation():
    """Demonstrate basic action validation against physics constraints."""
    print("=" * 80)
    print("Demo 1: Basic Physics Validation")
    print("=" * 80)
    print()

    # Create validator with conservation laws
    validator = PhysicsConstraintValidator(
        conservation_laws=[
            EnergyConservation(tolerance=1e-3),
            MomentumConservation(tolerance=1e-3),
        ]
    )

    # Current system state
    current_state = {
        "energy": 100.0,  # Joules
        "momentum": 50.0,  # kg·m/s
        "position": np.array([0.0, 0.0, 0.0]),
    }

    print("Current system state:")
    print(f"  Energy: {current_state['energy']} J")
    print(f"  Momentum: {current_state['momentum']} kg·m/s")
    print()

    # Test 1: Valid action (conserves energy and momentum)
    print("Test 1: Valid action (energy and momentum conserved)")
    print("-" * 80)
    valid_action = ActionProposal(
        action_id="move_001",
        action_type="move",
        parameters={"velocity": [1.0, 0.0, 0.0], "duration": 1.0},
        expected_state={"energy": 100.0, "momentum": 50.0},  # Conserved
    )

    is_valid, reason = validator.validate_action(valid_action, current_state)
    print(f"  Result: {'✓ ACCEPTED' if is_valid else '✗ REJECTED'}")
    if reason:
        print(f"  Reason: {reason}")
    print()

    # Test 2: Invalid action (violates energy conservation)
    print("Test 2: Invalid action (creates energy from nothing)")
    print("-" * 80)
    invalid_action = ActionProposal(
        action_id="accelerate_001",
        action_type="accelerate",
        parameters={"force": [100.0, 0.0, 0.0], "duration": 1.0},
        expected_state={"energy": 200.0, "momentum": 50.0},  # Energy violation!
    )

    is_valid, reason = validator.validate_action(invalid_action, current_state)
    print(f"  Result: {'✓ ACCEPTED' if is_valid else '✗ REJECTED'}")
    if reason:
        print(f"  Reason: {reason}")
    print()

    # Test 3: Physically impossible parameters
    print("Test 3: Physically impossible parameters (superluminal velocity)")
    print("-" * 80)
    impossible_action = ActionProposal(
        action_id="warp_001",
        action_type="warp",
        parameters={"velocity": [4e8, 0.0, 0.0]},  # Faster than light!
        expected_state={"energy": 100.0, "momentum": 50.0},
    )

    is_valid, reason = validator.validate_action(impossible_action, current_state)
    print(f"  Result: {'✓ ACCEPTED' if is_valid else '✗ REJECTED'}")
    if reason:
        print(f"  Reason: {reason}")
    print()


def demo_feedback_learning():
    """Demonstrate learning from constraint violations."""
    print("=" * 80)
    print("Demo 2: Learning from Violations")
    print("=" * 80)
    print()

    validator = PhysicsConstraintValidator(
        conservation_laws=[EnergyConservation(tolerance=1e-3)]
    )
    feedback = ConstraintViolationFeedback(learning_rate=0.2)

    current_state = {"energy": 100.0}

    # Simulate multiple action proposals with violations
    print("Training phase: Proposing actions and learning from rejections")
    print("-" * 80)

    for i in range(5):
        # Propose increasingly aggressive energy changes
        energy_change = 10.0 * (i + 1)
        action = ActionProposal(
            action_id=f"test_{i}",
            action_type="energy_transfer",
            parameters={"delta_energy": energy_change},
            expected_state={"energy": 100.0 + energy_change},  # Violates conservation
        )

        is_valid, reason = validator.validate_action(action, current_state)

        print(f"  Iteration {i + 1}: ΔE = {energy_change:.1f} J")
        print(f"    Result: {'✓' if is_valid else '✗ REJECTED'}")

        if not is_valid:
            # Learn from violation
            violation = validator.violation_history[-1]
            feedback.process_violation(violation)
            print("    Learning: Updated constraint weight")

        print()

    # Show learned weights
    print("Learned constraint weights:")
    for constraint, weight in feedback.violation_weights.items():
        print(f"  {constraint}: {weight:.3f}")
    print()


def demo_autonomous_controller():
    """Demonstrate full autonomous control loop."""
    print("=" * 80)
    print("Demo 3: Autonomous Control Loop")
    print("=" * 80)
    print()

    # Set up controller
    validator = PhysicsConstraintValidator(
        conservation_laws=[
            EnergyConservation(tolerance=1e-3),
            MomentumConservation(tolerance=1e-3),
        ]
    )
    controller = AutonomousController(validator=validator)

    # System state
    current_state = {
        "energy": 100.0,
        "momentum": 50.0,
        "temperature": 300.0,  # Kelvin
    }

    print("Autonomous controller initialized")
    print(f"Initial state: E={current_state['energy']} J, "
          f"p={current_state['momentum']} kg·m/s, "
          f"T={current_state['temperature']} K")
    print()

    # Define execution function (simulated)
    def execute_action(proposal: ActionProposal) -> bool:
        """Simulate action execution."""
        # In real system, this would actually execute the action
        print(f"    Executing: {proposal.action_type}")
        return True  # Simulate success

    # Propose and execute actions
    print("Action sequence:")
    print("-" * 80)

    # Action 1: Valid movement
    print("1. Propose valid movement")
    action1 = controller.propose_action(
        action_type="move",
        parameters={"velocity": [1.0, 0.0, 0.0], "duration": 1.0},
        expected_state={"energy": 100.0, "momentum": 50.0, "temperature": 300.0},
        priority=0.8,
    )
    success, msg = controller.validate_and_execute(
        action1, current_state, execute_fn=execute_action
    )
    print(f"  Status: {msg}")
    print()

    # Action 2: Invalid energy creation
    print("2. Propose impossible energy creation")
    action2 = controller.propose_action(
        action_type="create_energy",
        parameters={"amount": 50.0},
        expected_state={"energy": 150.0, "momentum": 50.0, "temperature": 300.0},
        priority=0.5,
    )
    success, msg = controller.validate_and_execute(action2, current_state)
    print(f"  Status: {msg}")
    print()

    # Action 3: Invalid temperature
    print("3. Propose negative absolute temperature")
    action3 = controller.propose_action(
        action_type="cool",
        parameters={"target_temperature": -10.0},
        expected_state={"energy": 100.0, "momentum": 50.0, "temperature": -10.0},
        priority=0.3,
    )
    success, msg = controller.validate_and_execute(action3, current_state)
    print(f"  Status: {msg}")
    print()

    # Show statistics
    print("=" * 80)
    print("Controller Statistics")
    print("=" * 80)
    stats = controller.get_statistics()
    print(f"Total actions proposed: {stats['total_actions']}")
    print(f"Successful executions: {stats['successful_actions']}")
    print(f"Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"Total violations: {stats['violation_stats']['total_violations']}")
    print()

    if stats['violation_stats'].get('by_type'):
        print("Violations by type:")
        for vtype, count in stats['violation_stats']['by_type'].items():
            print(f"  {vtype}: {count}")
    print()


def demo_industrial_application():
    """Demonstrate industrial robotic arm control."""
    print("=" * 80)
    print("Demo 4: Industrial Application - Robotic Arm Control")
    print("=" * 80)
    print()

    # Custom constraint: Maximum torque
    def max_torque_constraint(
        proposal: ActionProposal, current_state: dict
    ) -> tuple[bool, str | None]:
        """Ensure torque doesn't exceed motor limits."""
        if "torque" in proposal.parameters:
            torque = proposal.parameters["torque"]
            max_torque = 100.0  # N·m
            if abs(torque) > max_torque:
                return False, f"Torque {torque:.1f} N·m exceeds limit {max_torque} N·m"
        return True, None

    # Custom constraint: Workspace bounds
    def workspace_constraint(
        proposal: ActionProposal, current_state: dict
    ) -> tuple[bool, str | None]:
        """Ensure end effector stays within safe workspace."""
        if "position" in proposal.expected_state:
            pos = np.array(proposal.expected_state["position"])
            # Spherical workspace with 2m radius
            distance = np.linalg.norm(pos)
            if distance > 2.0:
                return (
                    False,
                    f"Position {distance:.2f}m exceeds workspace radius 2.0m",
                )
        return True, None

    # Create validator with custom constraints
    validator = PhysicsConstraintValidator(
        conservation_laws=[EnergyConservation(tolerance=1e-2)],
        custom_constraints=[max_torque_constraint, workspace_constraint],
    )
    controller = AutonomousController(validator=validator)

    # Robot state
    robot_state = {
        "energy": 50.0,
        "position": np.array([0.5, 0.5, 0.5]),
        "joint_angles": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    print("Industrial robotic arm controller")
    print("  Workspace: 2.0m radius sphere")
    print("  Max torque: 100.0 N·m")
    print(f"  Current position: {robot_state['position']}")
    print()

    print("Test sequence:")
    print("-" * 80)

    # Test 1: Valid movement within workspace
    print("1. Move to position [1.0, 1.0, 0.5] (within workspace)")
    action1 = controller.propose_action(
        action_type="move_to",
        parameters={"target": [1.0, 1.0, 0.5], "torque": 50.0},
        expected_state={
            "energy": 55.0,
            "position": [1.0, 1.0, 0.5],
        },
    )
    success, msg = controller.validate_and_execute(action1, robot_state)
    print(f"  Result: {msg}")
    print()

    # Test 2: Movement outside workspace
    print("2. Move to position [3.0, 3.0, 0.0] (outside workspace)")
    action2 = controller.propose_action(
        action_type="move_to",
        parameters={"target": [3.0, 3.0, 0.0], "torque": 50.0},
        expected_state={
            "energy": 60.0,
            "position": [3.0, 3.0, 0.0],
        },
    )
    success, msg = controller.validate_and_execute(action2, robot_state)
    print(f"  Result: {msg}")
    print()

    # Test 3: Excessive torque
    print("3. Apply excessive torque (150 N·m)")
    action3 = controller.propose_action(
        action_type="apply_force",
        parameters={"torque": 150.0},
        expected_state={"energy": 70.0},
    )
    success, msg = controller.validate_and_execute(action3, robot_state)
    print(f"  Result: {msg}")
    print()

    print("Safety verification complete!")
    print("  All unsafe actions were rejected before execution")
    print()


def main():
    """Run all Phase III-A demonstrations."""
    print("\n" + "=" * 80)
    print("PHASE III-A: PHYSICS-CONSTRAINED AUTONOMOUS AI")
    print("Trustworthy AI through Physics Validation")
    print("=" * 80 + "\n")

    demo_basic_validation()
    demo_feedback_learning()
    demo_autonomous_controller()
    demo_industrial_application()

    print("=" * 80)
    print("All Phase III-A demonstrations completed!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("✓ AI actions are validated against physical laws before execution")
    print("✓ Impossible or dangerous actions are automatically rejected")
    print("✓ System learns from violations to propose better actions")
    print("✓ Custom constraints enable domain-specific safety rules")
    print("✓ Applicable to robotics, automation, energy systems, manufacturing")
    print()


if __name__ == "__main__":
    main()
