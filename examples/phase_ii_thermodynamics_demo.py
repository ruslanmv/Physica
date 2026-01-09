"""Phase II Demonstration: Thermodynamics.

Showcases thermodynamic state equations, entropy calculations,
heat engine cycles, and the Second Law.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R

from physica.domains import (
    CarnotCycle,
    EntropyCalculator,
    HeatEngine,
    IdealGasEOS,
    OttoCycle,
    ThermodynamicState,
    VanDerWaalsEOS,
    maxwell_boltzmann_distribution,
)


def demo_ideal_gas():
    """Demonstrate ideal gas law."""
    print("=" * 60)
    print("Demo 1: Ideal Gas Law")
    print("=" * 60)

    eos = IdealGasEOS()

    # Standard conditions
    T = 273.15  # K (0°C)
    P = 101325  # Pa (1 atm)
    n = 1.0  # mol

    V = eos.volume(P, T, n)

    print("Standard conditions (STP):")
    print(f"  Temperature: {T} K ({T - 273.15}°C)")
    print(f"  Pressure: {P} Pa ({P / 101325:.2f} atm)")
    print(f"  Amount: {n} mol")
    print(f"  Molar volume: {V * 1000:.2f} L/mol")
    print("  (Theoretical: 22.4 L/mol)")
    print()


def demo_van_der_waals():
    """Demonstrate Van der Waals equation and critical point."""
    print("=" * 60)
    print("Demo 2: Van der Waals Equation (Real Gas)")
    print("=" * 60)

    # CO₂ parameters
    a_CO2 = 0.3658  # Pa·m⁶/mol²
    b_CO2 = 4.267e-5  # m³/mol

    eos = VanDerWaalsEOS(a=a_CO2, b=b_CO2)

    # Calculate critical point
    Tc, Pc, Vc = eos.critical_point()

    print("Van der Waals equation for CO₂:")
    print(f"  a = {a_CO2} Pa·m⁶/mol²")
    print(f"  b = {b_CO2} m³/mol")
    print("\nCritical point:")
    print(f"  Tc = {Tc:.2f} K ({Tc - 273.15:.2f}°C)")
    print(f"  Pc = {Pc / 1e6:.2f} MPa")
    print(f"  Vc = {Vc * 1000:.2f} L/mol")
    print("\n(Experimental for CO₂: Tc ≈ 304 K, Pc ≈ 7.4 MPa)")
    print()


def demo_carnot_cycle():
    """Demonstrate Carnot cycle and maximum efficiency."""
    print("=" * 60)
    print("Demo 3: Carnot Cycle (Maximum Efficiency)")
    print("=" * 60)

    # Hot and cold reservoirs
    Th = 600  # K (327°C)
    Tc = 300  # K (27°C)

    # Create Carnot cycle
    V1 = 0.001  # m³ (1 L)
    V2 = 0.005  # m³ (5 L)
    gamma = 1.4  # Air

    cycle = CarnotCycle(Th=Th, Tc=Tc, V1=V1, V2=V2, gamma=gamma)

    # Calculate efficiency
    eta = cycle.efficiency()

    print("Carnot heat engine:")
    print(f"  Hot reservoir: {Th} K ({Th - 273.15}°C)")
    print(f"  Cold reservoir: {Tc} K ({Tc - 273.15}°C)")
    print(f"  Carnot efficiency: η = 1 - Tc/Th = {eta:.4f} ({eta * 100:.2f}%)")
    print("\nThis is the maximum possible efficiency for any heat engine")
    print("operating between these temperatures (Second Law).")

    # Calculate work output
    cv = (5 / 2) * R  # Ideal gas
    W_net = cycle.work_output(cv)

    print(f"\nWork output per cycle: {W_net:.2f} J")
    print()

    # Plot P-V diagram
    states = cycle.states
    volumes = [s.V * 1000 for s in states] + [states[0].V * 1000]  # Close loop
    pressures = [s.P / 1000 for s in states] + [states[0].P / 1000]

    plt.figure(figsize=(8, 6))
    plt.plot(volumes, pressures, "o-", linewidth=2, markersize=8)
    plt.xlabel("Volume (L)")
    plt.ylabel("Pressure (kPa)")
    plt.title("Carnot Cycle P-V Diagram")
    plt.grid(True)

    # Label states
    labels = ["1 (Th, V1)", "2 (Th, V2)", "3 (Tc, V3)", "4 (Tc, V4)"]
    for v, p, label in zip(volumes[:4], pressures[:4], labels):
        plt.annotate(label, (v, p), textcoords="offset points", xytext=(10, 10))

    plt.tight_layout()
    plt.savefig("carnot_cycle.png", dpi=150)
    print("Saved plot: carnot_cycle.png\n")


def demo_otto_cycle():
    """Demonstrate Otto cycle (gasoline engine)."""
    print("=" * 60)
    print("Demo 4: Otto Cycle (Internal Combustion Engine)")
    print("=" * 60)

    # Typical compression ratio for gasoline engine
    r = 9.0
    gamma = 1.4

    cycle = OttoCycle(compression_ratio=r, gamma=gamma)

    eta = cycle.efficiency()

    print("Otto cycle (gasoline engine):")
    print(f"  Compression ratio: r = {r}")
    print(f"  Heat capacity ratio: γ = {gamma}")
    print(f"  Otto efficiency: η = 1 - 1/r^(γ-1) = {eta:.4f} ({eta * 100:.2f}%)")
    print("\nReal engines achieve ~25-30% due to friction and heat losses.")
    print()


def demo_entropy_second_law():
    """Demonstrate entropy and Second Law."""
    print("=" * 60)
    print("Demo 5: Entropy and the Second Law")
    print("=" * 60)

    cv = (5 / 2) * R  # Diatomic gas

    # Irreversible free expansion
    state1 = ThermodynamicState(P=200000, V=0.001, T=300, n=1.0)
    # Free expansion to double volume (T constant for ideal gas)
    state2 = ThermodynamicState(P=100000, V=0.002, T=300, n=1.0)

    # Calculate entropy change
    entropy_calc = EntropyCalculator()
    delta_S_system = entropy_calc.entropy_change_ideal_gas(state1, state2, cv)

    # For free expansion, Q = 0, so surroundings entropy change = 0
    delta_S_universe = delta_S_system

    print("Free expansion (irreversible):")
    print(f"  Initial state: P = {state1.P / 1000:.0f} kPa, V = {state1.V * 1000:.0f} L")
    print(f"  Final state: P = {state2.P / 1000:.0f} kPa, V = {state2.V * 1000:.0f} L")
    print(f"  System entropy change: ΔS = {delta_S_system:.4f} J/K")
    print(f"  Universe entropy change: ΔS_univ = {delta_S_universe:.4f} J/K")
    print("\nΔS_universe > 0: Second Law satisfied! ✓")
    print("Process is irreversible.")
    print()


def demo_maxwell_boltzmann():
    """Demonstrate Maxwell-Boltzmann distribution."""
    print("=" * 60)
    print("Demo 6: Maxwell-Boltzmann Speed Distribution")
    print("=" * 60)

    # Nitrogen gas at room temperature
    T = 300  # K
    m_N2 = 28.014e-3 / 6.022e23  # kg (molecular mass)

    # Speed range
    v = np.linspace(0, 1500, 1000)  # m/s

    # Calculate distribution
    f_v = maxwell_boltzmann_distribution(v, T, m_N2)

    # Find most probable speed
    v_mp = v[np.argmax(f_v)]

    # Calculate mean speed (theoretical)
    from scipy.constants import k as k_B

    v_mean_theory = np.sqrt(8 * k_B * T / (np.pi * m_N2))
    v_rms_theory = np.sqrt(3 * k_B * T / m_N2)

    print(f"Nitrogen (N₂) at T = {T} K:")
    print(f"  Most probable speed: {v_mp:.1f} m/s")
    print(f"  Mean speed: {v_mean_theory:.1f} m/s")
    print(f"  RMS speed: {v_rms_theory:.1f} m/s")
    print()

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.plot(v, f_v, linewidth=2)
    plt.axvline(v_mp, color="r", linestyle="--", label=f"Most probable: {v_mp:.0f} m/s")
    plt.axvline(
        v_mean_theory, color="g", linestyle="--", label=f"Mean: {v_mean_theory:.0f} m/s"
    )
    plt.axvline(
        v_rms_theory, color="b", linestyle="--", label=f"RMS: {v_rms_theory:.0f} m/s"
    )
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Probability Density")
    plt.title("Maxwell-Boltzmann Speed Distribution (N₂ at 300 K)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("maxwell_boltzmann.png", dpi=150)
    print("Saved plot: maxwell_boltzmann.png\n")


def demo_heat_engine_power():
    """Demonstrate heat engine power output."""
    print("=" * 60)
    print("Demo 7: Heat Engine Power Output")
    print("=" * 60)

    # Create Carnot cycle
    Th = 500  # K
    Tc = 300  # K
    cycle = CarnotCycle(Th=Th, Tc=Tc, V1=0.001, V2=0.005)

    # Create heat engine running at 3000 RPM
    rpm = 3000
    frequency = rpm / 60  # Hz

    engine = HeatEngine(cycle, operating_frequency=frequency)

    # Heat input per cycle
    Q_in = 1000  # J

    # Calculate power
    P = engine.power_output(Q_in)
    Q_out = engine.heat_rejected(Q_in)

    print(f"Heat engine operating at {rpm} RPM:")
    print(f"  Efficiency: {cycle.efficiency() * 100:.1f}%")
    print(f"  Heat input: {Q_in} J/cycle")
    print(f"  Work output: {Q_in * cycle.efficiency():.1f} J/cycle")
    print(f"  Heat rejected: {Q_out:.1f} J/cycle")
    print(f"  Power output: {P / 1000:.2f} kW")
    print()


def main():
    """Run all thermodynamics demonstrations."""
    print("\n" + "=" * 60)
    print("PHASE II: THERMODYNAMICS DEMONSTRATIONS")
    print("=" * 60 + "\n")

    demo_ideal_gas()
    demo_van_der_waals()
    demo_carnot_cycle()
    demo_otto_cycle()
    demo_entropy_second_law()
    demo_maxwell_boltzmann()
    demo_heat_engine_power()

    print("=" * 60)
    print("All thermodynamics demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
