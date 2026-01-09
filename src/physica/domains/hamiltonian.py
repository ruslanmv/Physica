"""Hamiltonian mechanics domain.

Implements Hamiltonian formulation of classical mechanics with symplectic
integrators and canonical transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class HamiltonianSystem:
    """Represents a Hamiltonian dynamical system.

    Attributes
    ----------
    n_dof:
        Number of degrees of freedom.
    H:
        Hamiltonian function H(q, p, t).
    dH_dq:
        Partial derivative ∂H/∂q function.
    dH_dp:
        Partial derivative ∂H/∂p function.
    """

    n_dof: int
    H: Callable[[np.ndarray, np.ndarray, float], float]
    dH_dq: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    dH_dp: Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class HamiltonianSimulator:
    """Simulate Hamiltonian systems using symplectic integrators.

    Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
    """

    def __init__(self, system: HamiltonianSystem):
        """Initialize Hamiltonian simulator.

        Parameters
        ----------
        system:
            Hamiltonian system definition.
        """
        self.system = system

    def hamilton_equations(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Hamilton's equations.

        Parameters
        ----------
        t:
            Time.
        state:
            State vector [q, p] of length 2*n_dof.

        Returns
        -------
        dstate_dt:
            Time derivative [dq/dt, dp/dt].
        """
        n = self.system.n_dof
        q = state[:n]
        p = state[n:]

        dq_dt = self.system.dH_dp(q, p, t)
        dp_dt = -self.system.dH_dq(q, p, t)

        return np.concatenate([dq_dt, dp_dt])

    def simulate_symplectic_euler(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float,
    ) -> dict:
        """Simulate using symplectic Euler method.

        This method preserves the symplectic structure and energy
        better than standard integrators.

        Parameters
        ----------
        q0:
            Initial generalized coordinates.
        p0:
            Initial generalized momenta.
        t_span:
            Time span (t_start, t_end).
        dt:
            Timestep.

        Returns
        -------
        result:
            Dictionary with 't', 'q', 'p', 'energy'.
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        t = np.zeros(n_steps + 1)
        q = np.zeros((n_steps + 1, self.system.n_dof))
        p = np.zeros((n_steps + 1, self.system.n_dof))
        energy = np.zeros(n_steps + 1)

        # Initial conditions
        t[0] = t_start
        q[0] = q0
        p[0] = p0
        energy[0] = self.system.H(q0, p0, t_start)

        # Symplectic Euler: p_{n+1} = p_n + dt*(-∂H/∂q_n)
        #                   q_{n+1} = q_n + dt*(∂H/∂p_{n+1})
        for i in range(n_steps):
            t[i + 1] = t[i] + dt

            # Update momentum using old position
            p[i + 1] = p[i] - dt * self.system.dH_dq(q[i], p[i], t[i])

            # Update position using new momentum
            q[i + 1] = q[i] + dt * self.system.dH_dp(q[i], p[i + 1], t[i + 1])

            # Compute energy
            energy[i + 1] = self.system.H(q[i + 1], p[i + 1], t[i + 1])

        return {
            't': t,
            'q': q,
            'p': p,
            'energy': energy,
        }

    def simulate_verlet(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float,
    ) -> dict:
        """Simulate using Velocity Verlet (symplectic, 2nd order).

        Parameters
        ----------
        q0:
            Initial positions.
        p0:
            Initial momenta.
        t_span:
            Time span.
        dt:
            Timestep.

        Returns
        -------
        result:
            Simulation results.
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        t = np.zeros(n_steps + 1)
        q = np.zeros((n_steps + 1, self.system.n_dof))
        p = np.zeros((n_steps + 1, self.system.n_dof))
        energy = np.zeros(n_steps + 1)

        t[0] = t_start
        q[0] = q0
        p[0] = p0
        energy[0] = self.system.H(q0, p0, t_start)

        for i in range(n_steps):
            t[i + 1] = t[i] + dt

            # Half-step momentum update
            p_half = p[i] - 0.5 * dt * self.system.dH_dq(q[i], p[i], t[i])

            # Full-step position update
            q[i + 1] = q[i] + dt * self.system.dH_dp(q[i], p_half, t[i] + 0.5 * dt)

            # Half-step momentum update
            p[i + 1] = p_half - 0.5 * dt * self.system.dH_dq(q[i + 1], p_half, t[i + 1])

            energy[i + 1] = self.system.H(q[i + 1], p[i + 1], t[i + 1])

        return {
            't': t,
            'q': q,
            'p': p,
            'energy': energy,
        }


def simple_pendulum_hamiltonian() -> HamiltonianSystem:
    """Create simple pendulum Hamiltonian system.

    H = p²/(2m) + mgl(1 - cos(q))

    Returns
    -------
    system:
        Hamiltonian system for simple pendulum.
    """
    m = 1.0  # mass
    l = 1.0  # length
    g = 9.81  # gravity

    def H(q, p, t):
        return (p[0] ** 2) / (2 * m) + m * g * l * (1 - np.cos(q[0]))

    def dH_dq(q, p, t):
        return np.array([m * g * l * np.sin(q[0])])

    def dH_dp(q, p, t):
        return np.array([p[0] / m])

    return HamiltonianSystem(n_dof=1, H=H, dH_dq=dH_dq, dH_dp=dH_dp)


def harmonic_oscillator_hamiltonian(m: float = 1.0, k: float = 1.0) -> HamiltonianSystem:
    """Create harmonic oscillator Hamiltonian.

    H = p²/(2m) + (k/2)q²

    Parameters
    ----------
    m:
        Mass.
    k:
        Spring constant.

    Returns
    -------
    system:
        Hamiltonian system.
    """
    def H(q, p, t):
        return (p[0] ** 2) / (2 * m) + 0.5 * k * (q[0] ** 2)

    def dH_dq(q, p, t):
        return np.array([k * q[0]])

    def dH_dp(q, p, t):
        return np.array([p[0] / m])

    return HamiltonianSystem(n_dof=1, H=H, dH_dq=dH_dq, dH_dp=dH_dp)


def kepler_problem_hamiltonian(mu: float = 1.0) -> HamiltonianSystem:
    """Create Kepler problem (planetary motion) Hamiltonian.

    H = |p|²/(2m) - μ/|q|

    Parameters
    ----------
    mu:
        Gravitational parameter (G*M).

    Returns
    -------
    system:
        2D Kepler problem.
    """
    m = 1.0

    def H(q, p, t):
        r = np.linalg.norm(q)
        return np.dot(p, p) / (2 * m) - mu / r

    def dH_dq(q, p, t):
        r = np.linalg.norm(q)
        return mu * q / (r ** 3)

    def dH_dp(q, p, t):
        return p / m

    return HamiltonianSystem(n_dof=2, H=H, dH_dq=dH_dq, dH_dp=dH_dp)


class PoissonBracket:
    """Compute Poisson brackets for Hamiltonian systems."""

    @staticmethod
    def compute(
        f: Callable,
        g: Callable,
        q: np.ndarray,
        p: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Compute Poisson bracket {f, g}.

        {f, g} = Σᵢ (∂f/∂qᵢ ∂g/∂pᵢ - ∂f/∂pᵢ ∂g/∂qᵢ)

        Parameters
        ----------
        f:
            First function f(q, p).
        g:
            Second function g(q, p).
        q:
            Generalized coordinates.
        p:
            Generalized momenta.
        eps:
            Finite difference step.

        Returns
        -------
        bracket:
            Poisson bracket value.
        """
        n = len(q)
        bracket = 0.0

        for i in range(n):
            # ∂f/∂qᵢ
            q_plus = q.copy()
            q_plus[i] += eps
            df_dqi = (f(q_plus, p) - f(q, p)) / eps

            # ∂g/∂pᵢ
            p_plus = p.copy()
            p_plus[i] += eps
            dg_dpi = (g(q, p_plus) - g(q, p)) / eps

            # ∂f/∂pᵢ
            df_dpi = (f(q, p_plus) - f(q, p)) / eps

            # ∂g/∂qᵢ
            dg_dqi = (g(q_plus, p) - g(q, p)) / eps

            bracket += df_dqi * dg_dpi - df_dpi * dg_dqi

        return bracket
