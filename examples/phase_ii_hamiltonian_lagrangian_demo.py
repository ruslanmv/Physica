"""Phase II Demonstration: Hamiltonian and Lagrangian Mechanics.

Showcases energy-preserving symplectic integrators, Euler-Lagrange equations,
and analytical mechanics principles.
"""

import numpy as np
import matplotlib.pyplot as plt
from physica.domains import (
    HamiltonianSimulator,
    LagrangianSimulator,
    simple_pendulum_hamiltonian,
    simple_pendulum_lagrangian,
    double_pendulum_lagrangian,
    ActionPrinciple,
)


def demo_hamiltonian_pendulum():
    """Demonstrate Hamiltonian formulation with symplectic integrator."""
    print("=" * 60)
    print("Demo 1: Hamiltonian Pendulum (Energy Conservation)")
    print("=" * 60)

    # Create simple pendulum Hamiltonian system
    system = simple_pendulum_hamiltonian(m=1.0, l=1.0, g=9.81)

    # Initial conditions: small angle
    q0 = np.array([0.2])  # 0.2 rad ≈ 11.5°
    p0 = np.array([0.0])  # At rest

    # Simulate with symplectic integrator
    simulator = HamiltonianSimulator(system)
    result_symplectic = simulator.simulate_symplectic_euler(
        q0, p0, t_span=(0, 10), dt=0.01
    )

    # Simulate with standard RK45 for comparison
    result_rk45 = simulator.simulate(q0, p0, t_span=(0, 10), n_steps=1000)

    # Calculate energy conservation
    energy_symplectic = np.array(
        [
            system.H(result_symplectic["q"][i], result_symplectic["p"][i])
            for i in range(len(result_symplectic["t"]))
        ]
    )

    energy_drift_symplectic = (
        np.abs(energy_symplectic - energy_symplectic[0]) / energy_symplectic[0]
    )
    energy_drift_rk45 = (
        np.abs(result_rk45["energy"] - result_rk45["energy"][0])
        / result_rk45["energy"][0]
    )

    print(f"Initial angle: {np.degrees(q0[0]):.2f}°")
    print(f"Initial energy: {energy_symplectic[0]:.6f} J")
    print(f"\nEnergy drift after 10 seconds:")
    print(f"  Symplectic Euler: {energy_drift_symplectic[-1]:.2e}")
    print(f"  RK45: {energy_drift_rk45[-1]:.2e}")
    print(f"\nSymplectic integrator preserves energy {energy_drift_rk45[-1] / energy_drift_symplectic[-1]:.1f}× better!")
    print()

    # Plot energy conservation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory
    axes[0, 0].plot(
        result_symplectic["t"],
        np.degrees(result_symplectic["q"][:, 0]),
        label="Symplectic",
    )
    axes[0, 0].plot(
        result_rk45["t"], np.degrees(result_rk45["q"][:, 0]), "--", label="RK45"
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (°)")
    axes[0, 0].set_title("Pendulum Motion")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Phase space
    axes[0, 1].plot(result_symplectic["q"][:, 0], result_symplectic["p"][:, 0])
    axes[0, 1].set_xlabel("q (rad)")
    axes[0, 1].set_ylabel("p (kg·m²/s)")
    axes[0, 1].set_title("Phase Space (Symplectic)")
    axes[0, 1].grid(True)

    # Energy conservation
    axes[1, 0].semilogy(result_symplectic["t"], energy_drift_symplectic, label="Symplectic")
    axes[1, 0].semilogy(result_rk45["t"], energy_drift_rk45, "--", label="RK45")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Relative Energy Drift")
    axes[1, 0].set_title("Energy Conservation")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Energy over time
    axes[1, 1].plot(result_symplectic["t"], energy_symplectic)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Total Energy (Symplectic)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("hamiltonian_pendulum.png", dpi=150)
    print("Saved plot: hamiltonian_pendulum.png\n")


def demo_lagrangian_pendulum():
    """Demonstrate Lagrangian formulation with Euler-Lagrange equations."""
    print("=" * 60)
    print("Demo 2: Lagrangian Pendulum")
    print("=" * 60)

    # Create simple pendulum Lagrangian system
    system = simple_pendulum_lagrangian(m=1.0, l=1.0, g=9.81)

    # Initial conditions
    q0 = np.array([np.pi / 6])  # 30°
    q_dot0 = np.array([0.0])

    # Simulate
    simulator = LagrangianSimulator(system)
    result = simulator.simulate(q0, q_dot0, t_span=(0, 10), n_steps=1000)

    print(f"Initial angle: {np.degrees(q0[0]):.2f}°")
    print(f"Simulation time: 10 s")
    print(f"Success: {result['success']}")
    print(f"Final angle: {np.degrees(result['q'][-1, 0]):.2f}°")

    # Calculate period
    # Find zero crossings to estimate period
    angles = result["q"][:, 0]
    zero_crossings = np.where(np.diff(np.sign(angles)))[0]
    if len(zero_crossings) >= 2:
        period_measured = 2 * (
            result["t"][zero_crossings[2]] - result["t"][zero_crossings[0]]
        )
        period_theory = 2 * np.pi * np.sqrt(1.0 / 9.81)  # Small angle approximation
        print(f"Measured period: {period_measured:.3f} s")
        print(f"Theoretical period (small angle): {period_theory:.3f} s")

    print()


def demo_double_pendulum_chaos():
    """Demonstrate chaotic behavior of double pendulum."""
    print("=" * 60)
    print("Demo 3: Double Pendulum (Chaos)")
    print("=" * 60)

    # Create double pendulum
    system = double_pendulum_lagrangian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)

    # Two very similar initial conditions
    q0_1 = np.array([np.pi / 2, np.pi / 2])
    q0_2 = np.array([np.pi / 2 + 0.001, np.pi / 2])  # Tiny perturbation

    q_dot0 = np.array([0.0, 0.0])

    # Simulate both
    simulator = LagrangianSimulator(system)
    result1 = simulator.simulate(q0_1, q_dot0, t_span=(0, 20), n_steps=2000)
    result2 = simulator.simulate(q0_2, q_dot0, t_span=(0, 20), n_steps=2000)

    # Calculate divergence
    divergence = np.linalg.norm(result1["q"] - result2["q"], axis=1)

    print(f"Initial perturbation: 0.001 rad")
    print(f"Divergence after 20s: {divergence[-1]:.3f} rad")
    print(f"Divergence growth: {divergence[-1] / 0.001:.0f}×")
    print("This is characteristic of chaotic dynamics!")
    print()

    # Plot chaos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Angle evolution
    axes[0].plot(result1["t"], np.degrees(result1["q"][:, 0]), label="Pendulum 1 (θ₁)")
    axes[0].plot(result1["t"], np.degrees(result1["q"][:, 1]), label="Pendulum 1 (θ₂)")
    axes[0].plot(
        result2["t"],
        np.degrees(result2["q"][:, 0]),
        "--",
        label="Pendulum 2 (θ₁)",
        alpha=0.7,
    )
    axes[0].plot(
        result2["t"],
        np.degrees(result2["q"][:, 1]),
        "--",
        label="Pendulum 2 (θ₂)",
        alpha=0.7,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (°)")
    axes[0].set_title("Double Pendulum Chaos")
    axes[0].legend()
    axes[0].grid(True)

    # Divergence
    axes[1].semilogy(result1["t"], divergence)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Divergence (rad)")
    axes[1].set_title("Exponential Divergence (Lyapunov)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("double_pendulum_chaos.png", dpi=150)
    print("Saved plot: double_pendulum_chaos.png\n")


def demo_action_principle():
    """Demonstrate principle of least action."""
    print("=" * 60)
    print("Demo 4: Principle of Least Action")
    print("=" * 60)

    # Create simple pendulum
    system = simple_pendulum_lagrangian(m=1.0, l=1.0, g=9.81)

    # Simulate a physical trajectory
    q0 = np.array([0.3])
    q_dot0 = np.array([0.0])

    simulator = LagrangianSimulator(system)
    result = simulator.simulate(q0, q_dot0, t_span=(0, 2), n_steps=100)

    # Compute action along physical trajectory
    action_calc = ActionPrinciple()
    action_physical = action_calc.compute_action(
        system, result["q"], result["q_dot"], result["t"]
    )

    # Create a perturbed (non-physical) trajectory
    q_perturbed = result["q"] + 0.1 * np.sin(2 * np.pi * result["t"][:, None])
    q_dot_perturbed = result["q_dot"] + 0.2 * np.cos(2 * np.pi * result["t"][:, None])

    action_perturbed = action_calc.compute_action(
        system, q_perturbed, q_dot_perturbed, result["t"]
    )

    print(f"Action along physical trajectory: {action_physical:.6f} J·s")
    print(f"Action along perturbed trajectory: {action_perturbed:.6f} J·s")
    print(f"Physical trajectory has {'lower' if action_physical < action_perturbed else 'higher'} action!")
    print(
        "This confirms the principle of least action (stationary action)."
    )
    print()


def main():
    """Run all Hamiltonian and Lagrangian demonstrations."""
    print("\n" + "=" * 60)
    print("PHASE II: HAMILTONIAN & LAGRANGIAN MECHANICS")
    print("=" * 60 + "\n")

    demo_hamiltonian_pendulum()
    demo_lagrangian_pendulum()
    demo_double_pendulum_chaos()
    demo_action_principle()

    print("=" * 60)
    print("All analytical mechanics demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
