"""Demo: Conservation Law Validation.

Shows how Physica enforces fundamental conservation laws to validate
and correct AI predictions.
"""

import numpy as np

from physica import (
    BallisticSimulator,
    ConservationValidator,
    EnergyConservation,
    MomentumConservation,
)


def main():
    print("=" * 70)
    print("CONSERVATION LAW VALIDATION DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows how conservation laws act as hard constraints")
    print("that AI systems cannot violate.")
    print()

    # Create simulator
    sim = BallisticSimulator(drag_coeff=0.0)  # No drag for ideal conservation

    # Simulate trajectory
    print("🚀 Simulating projectile motion...")
    result = sim.simulate(v0=50.0, angle_deg=45.0, max_time=10.0, steps=1000)

    # Extract trajectory
    trajectory = []
    for i in range(len(result.t)):
        state = {
            "position": np.array([[result.x[i], result.y_pos[i]]]),
            "velocity": np.array([[result.vx[i], result.vy[i]]]),
        }
        trajectory.append(state)

    print(f"   ✓ Simulated {len(trajectory)} timesteps")

    # Setup conservation validator
    validator = ConservationValidator(
        laws=[
            EnergyConservation(tolerance=1e-3),
            MomentumConservation(tolerance=1e-3),
        ]
    )

    system_params = {
        "mass": 1.0,
        "gravity": sim.gravity,
    }

    print("\n" + "─" * 70)
    print("VALIDATING CONSERVATION LAWS")
    print("─" * 70)

    # Validate trajectory
    is_valid, violations = validator.validate_trajectory(
        trajectory, system_params
    )

    if is_valid:
        print("\n✅ ALL CONSERVATION LAWS SATISFIED!")
    else:
        print(f"\n❌ {len(violations)} VIOLATIONS DETECTED:")
        for v in violations:
            print(f"\n{v}")

    # Detailed analysis
    print("\n" + "─" * 70)
    print("CONSERVATION ANALYSIS")
    print("─" * 70)

    # Energy at different time points
    energy_law = EnergyConservation()

    print("\nEnergy throughout flight:")
    sample_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]

    for idx in sample_indices:
        state = trajectory[idx]
        E = energy_law.compute_conserved_quantity(state, system_params)
        t = result.t[idx]
        print(f"  t = {t:5.2f}s : E = {E:8.2f} J")

    # Momentum conservation (horizontal component)
    print("\nHorizontal momentum throughout flight:")
    for idx in sample_indices:
        state = trajectory[idx]
        px = state["velocity"][0, 0] * system_params["mass"]
        t = result.t[idx]
        print(f"  t = {t:5.2f}s : p_x = {px:6.2f} kg·m/s")

    # Example: Introduce a violation artificially
    print("\n" + "─" * 70)
    print("TESTING VIOLATION DETECTION")
    print("─" * 70)

    print("\nArtificially adding energy to the system...")

    # Corrupt final state
    corrupted_trajectory = trajectory.copy()
    corrupted_state = corrupted_trajectory[-1].copy()
    corrupted_state["velocity"] = corrupted_state["velocity"] * 1.5  # Boost velocity
    corrupted_trajectory[-1] = corrupted_state

    is_valid, violations = validator.validate_trajectory(
        corrupted_trajectory, system_params
    )

    if not is_valid:
        print("\n✅ VIOLATION SUCCESSFULLY DETECTED!")
        for v in violations:
            print(f"\n{v}")
    else:
        print("\n⚠️  Violation not detected (increase sensitivity)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Conservation laws provide:")
    print("  • Hard constraints that cannot be violated")
    print("  • Automatic validation of AI predictions")
    print("  • Detection of physically impossible proposals")
    print("  • Feedback for self-correction")
    print()
    print("This ensures AI outputs are not just plausible, but physically valid.")
    print()


if __name__ == "__main__":
    main()
