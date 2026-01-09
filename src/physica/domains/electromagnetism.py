"""Electromagnetic physics domain.

Implements Maxwell's equations, electric and magnetic fields, Lorentz force,
and electromagnetic wave propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.constants import c, epsilon_0, mu_0


@dataclass
class ChargedParticle:
    """Represents a charged particle.

    Attributes
    ----------
    charge:
        Electric charge (Coulombs).
    mass:
        Mass (kg).
    position:
        3D position vector (m).
    velocity:
        3D velocity vector (m/s).
    """

    charge: float
    mass: float
    position: np.ndarray
    velocity: np.ndarray


class ElectromagneticField:
    """Electromagnetic field simulator.

    Implements Maxwell's equations in vacuum and computes Lorentz force
    on charged particles.
    """

    def __init__(self):
        """Initialize electromagnetic field simulator."""
        self.c = c  # Speed of light
        self.epsilon_0 = epsilon_0  # Permittivity of free space
        self.mu_0 = mu_0  # Permeability of free space

    def electric_field_point_charge(
        self,
        q: float,
        source_position: np.ndarray,
        field_position: np.ndarray,
    ) -> np.ndarray:
        """Calculate electric field from a point charge.

        Parameters
        ----------
        q:
            Charge (Coulombs).
        source_position:
            Position of the charge (m).
        field_position:
            Position where field is evaluated (m).

        Returns
        -------
        E:
            Electric field vector (V/m).
        """
        r_vec = field_position - source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        # Coulomb's law: E = (1/4πε₀) * q/r² * r̂
        E = (q / (4 * np.pi * self.epsilon_0)) * r_vec / (r ** 3)
        return E

    def magnetic_field_current_element(
        self,
        I: float,
        dl: np.ndarray,
        source_position: np.ndarray,
        field_position: np.ndarray,
    ) -> np.ndarray:
        """Calculate magnetic field from a current element (Biot-Savart law).

        Parameters
        ----------
        I:
            Current (Amperes).
        dl:
            Current element vector (m).
        source_position:
            Position of current element (m).
        field_position:
            Position where field is evaluated (m).

        Returns
        -------
        B:
            Magnetic field vector (Tesla).
        """
        r_vec = field_position - source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        # Biot-Savart: B = (μ₀/4π) * I * (dl × r̂) / r²
        B = (self.mu_0 / (4 * np.pi)) * I * np.cross(dl, r_vec) / (r ** 3)
        return B

    def lorentz_force(
        self,
        particle: ChargedParticle,
        E: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Calculate Lorentz force on a charged particle.

        Parameters
        ----------
        particle:
            Charged particle.
        E:
            Electric field at particle position (V/m).
        B:
            Magnetic field at particle position (Tesla).

        Returns
        -------
        F:
            Lorentz force vector (N).
        """
        # F = q(E + v × B)
        return particle.charge * (E + np.cross(particle.velocity, B))

    def poynting_vector(self, E: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate Poynting vector (electromagnetic energy flux).

        Parameters
        ----------
        E:
            Electric field (V/m).
        B:
            Magnetic field (Tesla).

        Returns
        -------
        S:
            Poynting vector (W/m²).
        """
        # S = (1/μ₀) * E × B
        return np.cross(E, B) / self.mu_0

    def energy_density(self, E: np.ndarray, B: np.ndarray) -> float:
        """Calculate electromagnetic energy density.

        Parameters
        ----------
        E:
            Electric field (V/m).
        B:
            Magnetic field (Tesla).

        Returns
        -------
        u:
            Energy density (J/m³).
        """
        # u = (ε₀/2)|E|² + (1/2μ₀)|B|²
        E_energy = 0.5 * self.epsilon_0 * np.dot(E, E)
        B_energy = 0.5 * np.dot(B, B) / self.mu_0
        return E_energy + B_energy


class ParticleInFieldSimulator:
    """Simulate charged particle motion in electromagnetic fields.

    Uses the Lorentz force and numerical integration to compute trajectories.
    """

    def __init__(self, em_field: Optional[ElectromagneticField] = None):
        """Initialize particle simulator.

        Parameters
        ----------
        em_field:
            Electromagnetic field instance. If None, creates new one.
        """
        self.em_field = em_field or ElectromagneticField()

    def simulate_particle(
        self,
        particle: ChargedParticle,
        E_func,
        B_func,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
    ) -> dict:
        """Simulate charged particle in electromagnetic field.

        Parameters
        ----------
        particle:
            Initial particle state.
        E_func:
            Function E(r, t) returning electric field at position r, time t.
        B_func:
            Function B(r, t) returning magnetic field at position r, time t.
        t_span:
            Time span (t_start, t_end).
        n_steps:
            Number of timesteps.

        Returns
        -------
        result:
            Dictionary with keys:
            - 't': time array
            - 'position': position history (n_steps, 3)
            - 'velocity': velocity history (n_steps, 3)
            - 'kinetic_energy': KE history
        """
        from scipy.integrate import solve_ivp

        def equations_of_motion(t, y):
            # y = [x, y, z, vx, vy, vz]
            pos = y[:3]
            vel = y[3:6]

            # Get fields at current position
            E = E_func(pos, t)
            B = B_func(pos, t)

            # Lorentz force
            F = particle.charge * (E + np.cross(vel, B))

            # Acceleration: a = F/m
            acc = F / particle.mass

            # dy/dt = [vx, vy, vz, ax, ay, az]
            return np.concatenate([vel, acc])

        # Initial conditions
        y0 = np.concatenate([particle.position, particle.velocity])

        # Solve ODE
        sol = solve_ivp(
            equations_of_motion,
            t_span,
            y0,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], n_steps),
            rtol=1e-8,
            atol=1e-10,
        )

        # Extract results
        positions = sol.y[:3, :].T
        velocities = sol.y[3:6, :].T

        # Compute kinetic energy
        ke = 0.5 * particle.mass * np.sum(velocities ** 2, axis=1)

        return {
            't': sol.t,
            'position': positions,
            'velocity': velocities,
            'kinetic_energy': ke,
            'success': sol.success,
        }


def uniform_E_field(E0: np.ndarray):
    """Create uniform electric field function.

    Parameters
    ----------
    E0:
        Constant electric field vector.

    Returns
    -------
    E_func:
        Function E(r, t) returning E0.
    """
    return lambda r, t: E0


def uniform_B_field(B0: np.ndarray):
    """Create uniform magnetic field function.

    Parameters
    ----------
    B0:
        Constant magnetic field vector.

    Returns
    -------
    B_func:
        Function B(r, t) returning B0.
    """
    return lambda r, t: B0


def oscillating_E_field(E0: np.ndarray, omega: float, k: np.ndarray):
    """Create plane wave electric field.

    Parameters
    ----------
    E0:
        Amplitude vector.
    omega:
        Angular frequency (rad/s).
    k:
        Wave vector (1/m).

    Returns
    -------
    E_func:
        Function E(r, t) for plane wave.
    """
    def E_func(r, t):
        phase = np.dot(k, r) - omega * t
        return E0 * np.cos(phase)
    return E_func


class CyclotronMotion:
    """Analyze cyclotron motion of charged particles in magnetic field.

    Provides analytical solutions for circular motion in uniform B field.
    """

    def __init__(self, particle: ChargedParticle, B: np.ndarray):
        """Initialize cyclotron analyzer.

        Parameters
        ----------
        particle:
            Charged particle.
        B:
            Uniform magnetic field vector (Tesla).
        """
        self.particle = particle
        self.B = B
        self.B_magnitude = np.linalg.norm(B)

    def cyclotron_frequency(self) -> float:
        """Calculate cyclotron frequency.

        Returns
        -------
        f:
            Cyclotron frequency (Hz).
        """
        # f = qB / (2πm)
        return abs(self.particle.charge) * self.B_magnitude / (2 * np.pi * self.particle.mass)

    def cyclotron_radius(self, v_perp: float) -> float:
        """Calculate cyclotron radius (Larmor radius).

        Parameters
        ----------
        v_perp:
            Velocity component perpendicular to B (m/s).

        Returns
        -------
        r:
            Cyclotron radius (m).
        """
        # r = mv_perp / (qB)
        return self.particle.mass * v_perp / (abs(self.particle.charge) * self.B_magnitude)

    def pitch_angle(self, v_parallel: float, v_perp: float) -> float:
        """Calculate pitch angle.

        Parameters
        ----------
        v_parallel:
            Velocity parallel to B (m/s).
        v_perp:
            Velocity perpendicular to B (m/s).

        Returns
        -------
        alpha:
            Pitch angle (radians).
        """
        return np.arctan2(v_perp, v_parallel)
