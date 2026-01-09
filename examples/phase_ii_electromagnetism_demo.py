"""Phase II Demonstration: Electromagnetism.

Showcases electromagnetic field calculations, particle motion in fields,
and cyclotron motion analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

from physica.domains import (
    ChargedParticle,
    CyclotronMotion,
    ElectromagneticField,
    ParticleInFieldSimulator,
    uniform_B_field,
    uniform_E_field,
)


def demo_lorentz_force():
    """Demonstrate Lorentz force calculation."""
    print("=" * 60)
    print("Demo 1: Lorentz Force")
    print("=" * 60)

    em_field = ElectromagneticField()

    # Create an electron
    e = 1.602e-19  # Elementary charge (C)
    m_e = 9.109e-31  # Electron mass (kg)

    particle = ChargedParticle(
        charge=-e,
        mass=m_e,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1e6, 0.0, 0.0]),  # 1 Mm/s in x-direction
    )

    # Electric field in y-direction
    E = np.array([0.0, 1e3, 0.0])  # 1 kV/m

    # Magnetic field in z-direction
    B = np.array([0.0, 0.0, 0.1])  # 0.1 Tesla

    # Calculate Lorentz force
    F = em_field.lorentz_force(particle, E, B)

    print("Particle: electron")
    print(f"Velocity: {particle.velocity} m/s")
    print(f"Electric field: {E} V/m")
    print(f"Magnetic field: {B} T")
    print(f"Lorentz force: {F} N")
    print(f"Force magnitude: {np.linalg.norm(F):.2e} N")
    print()


def demo_cyclotron_motion():
    """Demonstrate cyclotron motion analysis."""
    print("=" * 60)
    print("Demo 2: Cyclotron Motion")
    print("=" * 60)

    # Proton in uniform magnetic field
    e = 1.602e-19
    m_p = 1.673e-27  # Proton mass (kg)

    particle = ChargedParticle(
        charge=e,
        mass=m_p,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1e5, 1e5, 1e5]),  # 100 km/s
    )

    # Uniform B field in z-direction
    B = np.array([0.0, 0.0, 1.0])  # 1 Tesla

    cyclotron = CyclotronMotion(particle, B)

    # Calculate cyclotron parameters
    f_c = cyclotron.cyclotron_frequency()
    v_perp = np.sqrt(particle.velocity[0] ** 2 + particle.velocity[1] ** 2)
    r_c = cyclotron.cyclotron_radius(v_perp)
    alpha = cyclotron.pitch_angle(particle.velocity[2], v_perp)

    print("Particle: proton")
    print(f"Magnetic field: {B} T")
    print(f"Cyclotron frequency: {f_c:.2e} Hz ({f_c / 1e6:.2f} MHz)")
    print(f"Cyclotron period: {1 / f_c * 1e6:.2f} μs")
    print(f"Cyclotron radius: {r_c * 100:.2f} cm")
    print(f"Pitch angle: {np.degrees(alpha):.2f}°")
    print()


def demo_particle_in_crossed_fields():
    """Simulate particle motion in crossed E and B fields (E×B drift)."""
    print("=" * 60)
    print("Demo 3: Particle in Crossed E and B Fields")
    print("=" * 60)

    # Electron
    e = 1.602e-19
    m_e = 9.109e-31

    particle = ChargedParticle(
        charge=-e,
        mass=m_e,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),  # Start at rest
    )

    # Crossed fields: E in y-direction, B in z-direction
    E_field = uniform_E_field(np.array([0.0, 1e3, 0.0]))  # 1 kV/m
    B_field = uniform_B_field(np.array([0.0, 0.0, 0.01]))  # 10 mT

    # Simulate
    simulator = ParticleInFieldSimulator()
    result = simulator.simulate_particle(
        particle,
        E_field,
        B_field,
        t_span=(0, 1e-6),  # 1 microsecond
        n_steps=1000,
    )

    # Analyze drift velocity
    # E×B drift velocity: v_d = E×B / |B|²
    E = np.array([0.0, 1e3, 0.0])
    B = np.array([0.0, 0.0, 0.01])
    v_drift_theory = np.cross(E, B) / np.dot(B, B)

    # Calculate average drift from simulation
    v_avg = np.mean(result["velocity"], axis=0)

    print(f"Electric field: {E} V/m")
    print(f"Magnetic field: {B} T")
    print(f"Theoretical E×B drift: {v_drift_theory} m/s")
    print(f"Simulated average velocity: {v_avg} m/s")
    print(f"Final position: {result['position'][-1]} m")
    print(f"Simulation success: {result['success']}")
    print()

    # Plot trajectory
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax1.plot(result["position"][:, 0] * 1e3, result["position"][:, 1] * 1e3)
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title("Trajectory (x-y plane)")
    ax1.grid(True)

    ax2 = fig.add_subplot(132)
    ax2.plot(result["t"] * 1e9, result["position"][:, 0] * 1e3, label="x")
    ax2.plot(result["t"] * 1e9, result["position"][:, 1] * 1e3, label="y")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Position (mm)")
    ax2.set_title("Position vs Time")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(133)
    ax3.plot(result["t"] * 1e9, result["kinetic_energy"] * 1e19)
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Kinetic Energy (×10⁻¹⁹ J)")
    ax3.set_title("Kinetic Energy")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("electromagnetism_crossed_fields.png", dpi=150)
    print("Saved plot: electromagnetism_crossed_fields.png")
    print()


def demo_electric_field_point_charge():
    """Calculate electric field from point charge."""
    print("=" * 60)
    print("Demo 4: Electric Field from Point Charge")
    print("=" * 60)

    em_field = ElectromagneticField()

    # 1 nC charge at origin
    q = 1e-9  # Coulombs
    source_pos = np.array([0.0, 0.0, 0.0])

    # Measure field at various distances
    distances = [0.01, 0.1, 1.0, 10.0]  # meters

    print(f"Point charge: {q * 1e9:.1f} nC at origin")
    print(f"{'Distance (m)':<15} {'E-field (V/m)':<20} {'Direction'}")
    print("-" * 60)

    for d in distances:
        field_pos = np.array([d, 0.0, 0.0])
        E = em_field.electric_field_point_charge(q, source_pos, field_pos)
        E_mag = np.linalg.norm(E)
        E_dir = E / E_mag if E_mag > 0 else np.zeros(3)

        print(
            f"{d:<15.2f} {E_mag:<20.2e} [{E_dir[0]:.2f}, {E_dir[1]:.2f}, {E_dir[2]:.2f}]"
        )

    print()


def main():
    """Run all electromagnetic demonstrations."""
    print("\n" + "=" * 60)
    print("PHASE II: ELECTROMAGNETISM DEMONSTRATIONS")
    print("=" * 60 + "\n")

    demo_lorentz_force()
    demo_cyclotron_motion()
    demo_electric_field_point_charge()
    demo_particle_in_crossed_fields()

    print("=" * 60)
    print("All electromagnetic demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
