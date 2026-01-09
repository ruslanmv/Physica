"""Phase III-B: Semiconductor Thermal & Power Digital Twin.

Production-ready thermal and power simulations for semiconductor devices:
- Heat diffusion (Fourier's law)
- Power density mapping
- Thermal energy conservation
- Multi-layer chip modeling

Targets: Data centers, AI accelerators, EVs, consumer electronics.
Avoids full quantum simulation (Phase IV) - focuses on thermal/power only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import convolve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass
class MaterialProperties:
    """Thermal and electrical properties of a material.

    Attributes
    ----------
    thermal_conductivity:
        κ in W/(m·K).
    specific_heat:
        c_p in J/(kg·K).
    density:
        ρ in kg/m³.
    electrical_resistivity:
        ρ_e in Ω·m.
    """

    thermal_conductivity: float
    specific_heat: float
    density: float
    electrical_resistivity: float = 1e-6

    @classmethod
    def silicon(cls) -> MaterialProperties:
        """Silicon thermal properties at room temperature."""
        return cls(
            thermal_conductivity=150.0,  # W/(m·K)
            specific_heat=700.0,  # J/(kg·K)
            density=2330.0,  # kg/m³
            electrical_resistivity=2.3e3,  # Ω·m (intrinsic)
        )

    @classmethod
    def copper(cls) -> MaterialProperties:
        """Copper (interconnect) thermal properties."""
        return cls(
            thermal_conductivity=400.0,  # W/(m·K)
            specific_heat=385.0,  # J/(kg·K)
            density=8960.0,  # kg/m³
            electrical_resistivity=1.68e-8,  # Ω·m
        )

    @classmethod
    def sio2(cls) -> MaterialProperties:
        """Silicon dioxide (insulator) thermal properties."""
        return cls(
            thermal_conductivity=1.4,  # W/(m·K)
            specific_heat=730.0,  # J/(kg·K)
            density=2200.0,  # kg/m³
            electrical_resistivity=1e16,  # Ω·m (insulator)
        )


class HeatDiffusion2D:
    """2D heat diffusion solver using Fourier's law.

    Solves: ρc_p ∂T/∂t = ∇·(κ∇T) + Q

    Where:
    - T: temperature field
    - κ: thermal conductivity
    - ρ: density
    - c_p: specific heat
    - Q: heat source term (power density)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        material: MaterialProperties,
    ):
        """Initialize 2D heat diffusion solver.

        Parameters
        ----------
        nx, ny:
            Grid points in x and y directions.
        dx, dy:
            Grid spacing in meters.
        material:
            Material thermal properties.
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.material = material

        # Thermal diffusivity α = κ/(ρc_p)
        self.alpha = material.thermal_conductivity / (
            material.density * material.specific_heat
        )

        # Create grid
        self.x = np.linspace(0, (nx - 1) * dx, nx)
        self.y = np.linspace(0, (ny - 1) * dy, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def solve_steady_state(
        self,
        power_density: np.ndarray,
        T_boundary: float = 300.0,
    ) -> np.ndarray:
        """Solve steady-state heat equation ∇²T = -Q/κ.

        Parameters
        ----------
        power_density:
            Heat generation Q(x,y) in W/m³.
        T_boundary:
            Boundary temperature in Kelvin.

        Returns
        -------
        T:
            Steady-state temperature field (K).
        """
        # Build Laplacian matrix using finite differences
        N = self.nx * self.ny

        # Second derivatives with Dirichlet boundary conditions
        dx2 = self.dx**2
        dy2 = self.dy**2

        # Create sparse Laplacian
        main_diag = -2 * (1 / dx2 + 1 / dy2) * np.ones(N)
        x_off_diag = (1 / dx2) * np.ones(N - 1)
        y_off_diag = (1 / dy2) * np.ones(N - self.nx)

        # Handle boundaries
        for i in range(self.ny):
            x_off_diag[i * self.nx - 1] = 0  # Don't wrap around rows

        L = diags(
            [main_diag, x_off_diag, x_off_diag, y_off_diag, y_off_diag],
            [0, -1, 1, -self.nx, self.nx],
            shape=(N, N),
            format="csr",
        )

        # Right-hand side: -Q/κ
        Q_flat = power_density.flatten()
        rhs = -Q_flat / self.material.thermal_conductivity

        # Apply boundary conditions (Dirichlet: fix edges to T_boundary)
        # This is simplified - just solve interior
        T_flat = spsolve(L, rhs)

        # Reshape and add boundary temperature offset
        T = T_flat.reshape((self.ny, self.nx)) + T_boundary

        # Force boundary conditions
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary

        return T

    def solve_transient(
        self,
        T_initial: np.ndarray,
        power_density: np.ndarray | Callable,
        t_span: tuple[float, float],
        n_steps: int = 100,
        T_boundary: float = 300.0,
    ) -> dict:
        """Solve transient heat equation.

        Parameters
        ----------
        T_initial:
            Initial temperature field (K).
        power_density:
            Heat generation Q(x,y,t) as array or function.
        t_span:
            Time interval (t_start, t_end) in seconds.
        n_steps:
            Number of time steps.
        T_boundary:
            Boundary temperature (K).

        Returns
        -------
        result:
            Dictionary with 't', 'T' (temperature history).
        """
        # Flatten initial condition
        T0 = T_initial.flatten()

        # Time points
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        # Define time derivative
        def dT_dt(t, T_flat):
            T = T_flat.reshape((self.ny, self.nx))

            # Compute Laplacian using convolution
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / self.dx**2
            laplacian = convolve(T, kernel, mode="constant", cval=T_boundary)

            # Add heat source
            Q = power_density(t) if callable(power_density) else power_density

            # Heat equation: ∂T/∂t = α∇²T + Q/(ρc_p)
            dT = self.alpha * laplacian + Q / (
                self.material.density * self.material.specific_heat
            )

            return dT.flatten()

        # Solve ODE
        sol = solve_ivp(
            dT_dt,
            t_span,
            T0,
            t_eval=t_eval,
            method="BDF",  # Good for stiff problems
        )

        # Reshape solutions
        T_history = [sol.y[:, i].reshape((self.ny, self.nx)) for i in range(len(sol.t))]

        return {
            "t": sol.t,
            "T": T_history,
            "success": sol.success,
        }


class PowerDensityMap:
    """Power density distribution for semiconductor devices.

    Models Joule heating: P = I²R = J²ρ_e
    Where J is current density and ρ_e is resistivity.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        """Initialize power density mapper.

        Parameters
        ----------
        nx, ny:
            Grid dimensions.
        dx, dy:
            Grid spacing (m).
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.power_map = np.zeros((ny, nx))

    def add_hotspot(
        self,
        center: tuple[float, float],
        radius: float,
        power_density: float,
    ):
        """Add a circular hotspot.

        Parameters
        ----------
        center:
            (x, y) center position in meters.
        radius:
            Hotspot radius in meters.
        power_density:
            Power density in W/m³.
        """
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(x, y)

        # Distance from center
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        # Add Gaussian hotspot
        sigma = radius / 3  # 3-sigma rule
        hotspot = power_density * np.exp(-0.5 * (dist / sigma) ** 2)

        self.power_map += hotspot

    def add_uniform_layer(self, power_density: float):
        """Add uniform background power density.

        Parameters
        ----------
        power_density:
            Uniform power in W/m³.
        """
        self.power_map += power_density

    def from_current_density(
        self,
        current_density: np.ndarray,
        resistivity: float,
    ) -> np.ndarray:
        """Calculate power density from current density.

        P = J²ρ_e (Joule heating)

        Parameters
        ----------
        current_density:
            Current density J in A/m².
        resistivity:
            Electrical resistivity ρ_e in Ω·m.

        Returns
        -------
        power_density:
            Power density in W/m³.
        """
        return current_density**2 * resistivity


@dataclass
class ChipLayer:
    """Single layer in multi-layer chip stack.

    Attributes
    ----------
    thickness:
        Layer thickness in meters.
    material:
        Material properties.
    power_density:
        Heat generation in W/m³.
    """

    thickness: float
    material: MaterialProperties
    power_density: float = 0.0


class MultiLayerChip:
    """Multi-layer semiconductor chip thermal model.

    Models realistic chip stacks:
    - Active silicon layer (transistors)
    - Metal interconnects
    - Dielectric layers
    - Heat spreader
    """

    def __init__(self, layers: list[ChipLayer], nx: int = 50, ny: int = 50):
        """Initialize multi-layer chip.

        Parameters
        ----------
        layers:
            List of chip layers from bottom to top.
        nx, ny:
            In-plane grid resolution.
        """
        self.layers = layers
        self.nx = nx
        self.ny = ny

        # Calculate total thickness
        self.total_thickness = sum(layer.thickness for layer in layers)

    def compute_thermal_resistance(self) -> float:
        """Calculate total thermal resistance R_th = Σ(t_i / κ_i).

        Returns
        -------
        R_th:
            Thermal resistance in K/W (for unit area).
        """
        R_th = 0.0
        for layer in self.layers:
            R_th += layer.thickness / layer.material.thermal_conductivity
        return R_th

    def estimate_max_temperature(
        self,
        total_power: float,
        chip_area: float,
        T_ambient: float = 300.0,
    ) -> float:
        """Estimate maximum chip temperature using 1D thermal resistance.

        Parameters
        ----------
        total_power:
            Total chip power in Watts.
        chip_area:
            Chip area in m².
        T_ambient:
            Ambient/base temperature in K.

        Returns
        -------
        T_max:
            Estimated maximum temperature in K.
        """
        R_th = self.compute_thermal_resistance()
        # T_max = T_ambient + P * R_th (per unit area)
        Q_per_area = total_power / chip_area
        delta_T = Q_per_area * R_th
        return T_ambient + delta_T


def compute_thermal_energy(
    T: np.ndarray,
    material: MaterialProperties,
    dx: float,
    dy: float,
    thickness: float = 1e-6,
) -> float:
    """Compute total thermal energy in a temperature field.

    E = ∫ρc_p T dV

    Parameters
    ----------
    T:
        Temperature field (K).
    material:
        Material properties.
    dx, dy:
        Grid spacing (m).
    thickness:
        Layer thickness (m).

    Returns
    -------
    E_thermal:
        Total thermal energy (J).
    """
    dV = dx * dy * thickness  # Volume element
    E = np.sum(material.density * material.specific_heat * T) * dV
    return E


def compute_max_power_budget(
    chip_area: float,
    max_temperature: float,
    ambient_temperature: float,
    thermal_resistance: float,
) -> float:
    """Calculate maximum allowable power for thermal constraints.

    Parameters
    ----------
    chip_area:
        Chip area in m².
    max_temperature:
        Maximum allowed temperature in K.
    ambient_temperature:
        Ambient temperature in K.
    thermal_resistance:
        Thermal resistance in K/W per m².

    Returns
    -------
    P_max:
        Maximum power in Watts.
    """
    delta_T_max = max_temperature - ambient_temperature
    P_max = (delta_T_max / thermal_resistance) * chip_area
    return P_max
