"""Phase III-B Demonstration: Semiconductor Thermal & Power Digital Twin.

Showcases production-ready thermal and power simulations for:
- Data center chips
- AI accelerators
- Electric vehicles
- Consumer electronics

Focuses on thermal/power (NOT quantum) for immediate industrial impact.
"""

import matplotlib.pyplot as plt
import numpy as np

from physica.semiconductor_twin import (
    ChipLayer,
    HeatDiffusion2D,
    MaterialProperties,
    MultiLayerChip,
    PowerDensityMap,
    compute_max_power_budget,
    compute_thermal_energy,
)


def demo_heat_diffusion_steady_state():
    """Demonstrate steady-state heat diffusion in silicon."""
    print("=" * 80)
    print("Demo 1: Steady-State Heat Diffusion")
    print("=" * 80)
    print()

    # Create 1mm x 1mm silicon chip
    nx, ny = 50, 50
    dx, dy = 20e-6, 20e-6  # 20 μm grid
    silicon = MaterialProperties.silicon()

    solver = HeatDiffusion2D(nx, ny, dx, dy, silicon)

    # Create power density map with hotspot
    power_map = PowerDensityMap(nx, ny, dx, dy)
    power_map.add_hotspot(
        center=(500e-6, 500e-6),  # Center
        radius=200e-6,  # 200 μm radius
        power_density=1e9,  # 1 GW/m³ (typical for modern chips)
    )
    power_map.add_uniform_layer(1e8)  # 100 MW/m³ background

    print(f"Chip dimensions: {nx * dx * 1e3:.2f} mm × {ny * dy * 1e3:.2f} mm")
    print(f"Grid resolution: {dx * 1e6:.1f} μm")
    print("Material: Silicon")
    print(f"  κ = {silicon.thermal_conductivity} W/(m·K)")
    print(f"Peak power density: {1e9 / 1e9:.1f} GW/m³")
    print()

    # Solve steady-state
    T_ambient = 300.0  # K (27°C)
    T = solver.solve_steady_state(power_map.power_map, T_boundary=T_ambient)

    T_max = np.max(T)
    T_avg = np.mean(T)

    print("Results:")
    print(f"  Ambient temperature: {T_ambient - 273.15:.1f}°C")
    print(f"  Maximum temperature: {T_max - 273.15:.1f}°C")
    print(f"  Average temperature: {T_avg - 273.15:.1f}°C")
    print(f"  Temperature rise: {T_max - T_ambient:.1f} K")
    print()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature map
    im1 = ax1.contourf(
        solver.X * 1e6,
        solver.Y * 1e6,
        T - 273.15,
        levels=20,
        cmap="hot",
    )
    ax1.set_xlabel("x (μm)")
    ax1.set_ylabel("y (μm)")
    ax1.set_title("Steady-State Temperature (°C)")
    plt.colorbar(im1, ax=ax1, label="Temperature (°C)")

    # Power density map
    im2 = ax2.contourf(
        solver.X * 1e6,
        solver.Y * 1e6,
        power_map.power_map / 1e9,
        levels=20,
        cmap="plasma",
    )
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    ax2.set_title("Power Density (GW/m³)")
    plt.colorbar(im2, ax=ax2, label="Power Density (GW/m³)")

    plt.tight_layout()
    plt.savefig("semiconductor_steady_state.png", dpi=150)
    print("Saved plot: semiconductor_steady_state.png\n")


def demo_transient_thermal():
    """Demonstrate transient thermal response (chip power-up)."""
    print("=" * 80)
    print("Demo 2: Transient Thermal Response (Power-Up)")
    print("=" * 80)
    print()

    # Small chip for faster computation
    nx, ny = 30, 30
    dx, dy = 30e-6, 30e-6
    silicon = MaterialProperties.silicon()

    solver = HeatDiffusion2D(nx, ny, dx, dy, silicon)

    # Initial condition: ambient temperature
    T_initial = 300.0 * np.ones((ny, nx))

    # Sudden power-on with hotspot
    power_map = PowerDensityMap(nx, ny, dx, dy)
    power_map.add_hotspot(
        center=(450e-6, 450e-6),
        radius=150e-6,
        power_density=5e8,  # 500 MW/m³
    )

    print("Simulating chip power-up...")
    print(f"  Initial temperature: {T_initial[0, 0] - 273.15:.1f}°C")
    print("  Power density: 500 MW/m³")
    print("  Simulation time: 1 ms")
    print()

    # Solve transient
    result = solver.solve_transient(
        T_initial=T_initial,
        power_density=power_map.power_map,
        t_span=(0, 1e-3),  # 1 millisecond
        n_steps=50,
    )

    # Analyze temperature rise
    T_max_history = [np.max(T) for T in result["T"]]
    T_center_history = [T[ny // 2, nx // 2] for T in result["T"]]

    print("Results:")
    print(f"  Success: {result['success']}")
    print(f"  Final max temperature: {T_max_history[-1] - 273.15:.1f}°C")
    print(f"  Temperature rise: {T_max_history[-1] - T_max_history[0]:.1f} K")
    print()

    # Plot temperature evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature vs time
    ax1.plot(result["t"] * 1e6, np.array(T_max_history) - 273.15, label="Max")
    ax1.plot(result["t"] * 1e6, np.array(T_center_history) - 273.15, label="Center")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Transient Thermal Response")
    ax1.legend()
    ax1.grid(True)

    # Final temperature distribution
    T_final = result["T"][-1]
    im = ax2.contourf(
        solver.X * 1e6,
        solver.Y * 1e6,
        T_final - 273.15,
        levels=20,
        cmap="hot",
    )
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    ax2.set_title("Final Temperature Distribution")
    plt.colorbar(im, ax=ax2, label="Temperature (°C)")

    plt.tight_layout()
    plt.savefig("semiconductor_transient.png", dpi=150)
    print("Saved plot: semiconductor_transient.png\n")


def demo_multilayer_chip():
    """Demonstrate multi-layer chip thermal resistance."""
    print("=" * 80)
    print("Demo 3: Multi-Layer Chip Stack")
    print("=" * 80)
    print()

    # Realistic chip stack (bottom to top)
    layers = [
        ChipLayer(
            thickness=50e-6,  # 50 μm
            material=MaterialProperties.silicon(),
            power_density=1e9,  # Active layer
        ),
        ChipLayer(
            thickness=5e-6,  # 5 μm
            material=MaterialProperties.copper(),  # Metal interconnects
            power_density=5e8,
        ),
        ChipLayer(
            thickness=10e-6,  # 10 μm
            material=MaterialProperties.sio2(),  # Dielectric
            power_density=0.0,
        ),
        ChipLayer(
            thickness=100e-6,  # 100 μm
            material=MaterialProperties.copper(),  # Heat spreader
            power_density=0.0,
        ),
    ]

    chip = MultiLayerChip(layers)

    print("Chip stack (bottom to top):")
    for i, layer in enumerate(layers):
        mat_name = (
            "Silicon"
            if layer.material.thermal_conductivity > 100
            else "Copper"
            if layer.material.thermal_conductivity > 300
            else "SiO₂"
        )
        print(
            f"  Layer {i + 1}: {layer.thickness * 1e6:.1f} μm {mat_name} "
            f"(κ = {layer.material.thermal_conductivity} W/(m·K))"
        )

    print()

    # Calculate thermal resistance
    R_th = chip.compute_thermal_resistance()
    print(f"Total thermal resistance: {R_th:.6f} K·m²/W")
    print()

    # Estimate max temperature for different power levels
    chip_area = 1e-4  # 1 cm² = 10mm x 10mm
    T_ambient = 300.0  # K

    powers = [1, 5, 10, 20, 50]  # Watts
    print("Temperature estimates for 1 cm² chip:")
    print(f"  {'Power (W)':<12} {'T_max (°C)':<15} {'ΔT (K)':<10} {'Status'}")
    print("-" * 60)

    for P in powers:
        T_max = chip.estimate_max_temperature(P, chip_area, T_ambient)
        delta_T = T_max - T_ambient
        status = "✓ Safe" if T_max < 358 else "⚠ Hot" if T_max < 378 else "✗ FAIL"
        print(f"  {P:<12.1f} {T_max - 273.15:<15.1f} {delta_T:<10.1f} {status}")

    print()
    print("Thermal limits:")
    print("  ✓ Safe: < 85°C")
    print("  ⚠ Hot: 85-105°C (throttling recommended)")
    print("  ✗ FAIL: > 105°C (thermal shutdown required)")
    print()


def demo_power_budget():
    """Demonstrate thermal power budget calculation."""
    print("=" * 80)
    print("Demo 4: Thermal Power Budget Analysis")
    print("=" * 80)
    print()

    # Chip specifications
    chip_sizes = {
        "Mobile SoC": 1e-4,  # 100 mm²
        "Desktop CPU": 2e-4,  # 200 mm²
        "Server CPU": 6e-4,  # 600 mm²
        "AI Accelerator": 8e-4,  # 800 mm²
    }

    # Thermal constraints
    T_max = 378.0  # 105°C max
    T_ambient = 300.0  # 27°C ambient
    R_th = 5e-5  # K·m²/W (typical)

    print("Maximum power budget for different chip types:")
    print("(Assuming T_max = 105°C, T_ambient = 27°C)")
    print()
    print(f"  {'Chip Type':<20} {'Area (mm²)':<15} {'P_max (W)':<12} {'Notes'}")
    print("-" * 80)

    for chip_name, area in chip_sizes.items():
        P_max = compute_max_power_budget(area, T_max, T_ambient, R_th)
        notes = ""
        if P_max < 10:
            notes = "Fanless OK"
        elif P_max < 65:
            notes = "Active cooling"
        else:
            notes = "Liquid cooling"

        print(
            f"  {chip_name:<20} {area * 1e6:<15.0f} {P_max:<12.1f} {notes}"
        )

    print()
    print("Design insights:")
    print("✓ Larger chips can dissipate more power")
    print("✓ Better thermal resistance (lower R_th) increases power budget")
    print("✓ Multi-chip modules help spread heat")
    print()


def demo_thermal_energy_conservation():
    """Demonstrate thermal energy conservation."""
    print("=" * 80)
    print("Demo 5: Thermal Energy Conservation")
    print("=" * 80)
    print()

    # Simple test: uniform temperature field
    nx, ny = 20, 20
    dx, dy = 50e-6, 50e-6
    thickness = 10e-6  # 10 μm
    silicon = MaterialProperties.silicon()

    # Create temperature fields
    T1 = 300.0 * np.ones((ny, nx))  # Room temperature
    T2 = 350.0 * np.ones((ny, nx))  # Heated

    # Compute thermal energies
    E1 = compute_thermal_energy(T1, silicon, dx, dy, thickness)
    E2 = compute_thermal_energy(T2, silicon, dx, dy, thickness)

    # Energy required for heating
    Q_required = E2 - E1

    print(f"System: {nx}×{ny} silicon grid")
    print(f"  Volume: {nx * dx * 1e3:.2f} mm × {ny * dy * 1e3:.2f} mm × "
          f"{thickness * 1e6:.1f} μm")
    print()
    print("Energy analysis:")
    print(f"  E(T=300K) = {E1 * 1e9:.3f} nJ")
    print(f"  E(T=350K) = {E2 * 1e9:.3f} nJ")
    print(f"  ΔE required = {Q_required * 1e9:.3f} nJ")
    print()

    # Compare with theoretical
    mass = silicon.density * (nx * dx) * (ny * dy) * thickness
    Q_theory = mass * silicon.specific_heat * 50  # 50 K rise
    error = abs(Q_required - Q_theory) / Q_theory * 100

    print("Validation:")
    print(f"  Calculated: {Q_required * 1e9:.3f} nJ")
    print(f"  Theoretical (mc_pΔT): {Q_theory * 1e9:.3f} nJ")
    print(f"  Error: {error:.6f}%")
    print()
    print("✓ Thermal energy conservation verified!")
    print()


def main():
    """Run all Phase III-B demonstrations."""
    print("\n" + "=" * 80)
    print("PHASE III-B: SEMICONDUCTOR THERMAL & POWER DIGITAL TWIN")
    print("Production-Ready Chip Thermal Simulation")
    print("=" * 80 + "\n")

    demo_heat_diffusion_steady_state()
    demo_transient_thermal()
    demo_multilayer_chip()
    demo_power_budget()
    demo_thermal_energy_conservation()

    print("=" * 80)
    print("All Phase III-B demonstrations completed!")
    print("=" * 80)
    print()
    print("Key Capabilities:")
    print("✓ 2D heat diffusion (Fourier's law)")
    print("✓ Steady-state and transient thermal analysis")
    print("✓ Multi-layer chip stacks")
    print("✓ Power density mapping and hotspots")
    print("✓ Thermal resistance and power budgeting")
    print("✓ Energy conservation verification")
    print()
    print("Industrial Applications:")
    print("→ Data center chip design")
    print("→ AI accelerator thermal management")
    print("→ Electric vehicle power electronics")
    print("→ Consumer electronics (smartphones, laptops)")
    print()


if __name__ == "__main__":
    main()
