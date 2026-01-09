"""Lagrangian mechanics domain.

Implements Lagrangian formulation with Euler-Lagrange equations,
action principles, and variational methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class LagrangianSystem:
    """Represents a Lagrangian dynamical system.

    Attributes
    ----------
    n_dof:
        Number of degrees of freedom.
    L:
        Lagrangian function L(q, q_dot, t).
    dL_dq:
        Partial derivative ∂L/∂q.
    dL_dqdot:
        Partial derivative ∂L/∂q̇.
    """

    n_dof: int
    L: Callable[[np.ndarray, np.ndarray, float], float]
    dL_dq: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    dL_dqdot: Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class LagrangianSimulator:
    """Simulate Lagrangian systems using Euler-Lagrange equations.

    Euler-Lagrange equation:
        d/dt(∂L/∂q̇) - ∂L/∂q = 0
    """

    def __init__(self, system: LagrangianSystem):
        """Initialize Lagrangian simulator.

        Parameters
        ----------
        system:
            Lagrangian system definition.
        """
        self.system = system

    def euler_lagrange_equations(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Euler-Lagrange equations as first-order ODEs.

        Parameters
        ----------
        t:
            Time.
        state:
            State vector [q, q_dot] of length 2*n_dof.

        Returns
        -------
        dstate_dt:
            Time derivative [q_dot, q_ddot].
        """
        n = self.system.n_dof
        q = state[:n]
        q_dot = state[n:]

        # We need to compute q_ddot from:
        # d/dt(∂L/∂q̇) = ∂L/∂q
        # This requires solving for q_ddot, which generally requires
        # inverting the mass matrix. For simple cases, we can use
        # numerical differentiation.

        # Simplified approach: compute using finite differences
        eps = 1e-8
        dL_dq_val = self.system.dL_dq(q, q_dot, t)

        # Compute d/dt(∂L/∂q̇) numerically
        dL_dqdot_val = self.system.dL_dqdot(q, q_dot, t)

        # For second-order system, we need mass matrix
        # M(q) * q_ddot = F(q, q_dot, t)
        # where M_ij = ∂²L/∂q̇ᵢ∂q̇ⱼ

        # Compute mass matrix using finite differences
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                q_dot_plus = q_dot.copy()
                q_dot_plus[j] += eps
                dL_dqdot_plus = self.system.dL_dqdot(q, q_dot_plus, t)

                M[i, j] = (dL_dqdot_plus[i] - dL_dqdot_val[i]) / eps

        # Compute forcing term
        F = dL_dq_val

        # Solve M * q_ddot = F
        try:
            q_ddot = np.linalg.solve(M, F)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudo-inverse
            q_ddot = np.linalg.lstsq(M, F, rcond=None)[0]

        return np.concatenate([q_dot, q_ddot])

    def simulate(
        self,
        q0: np.ndarray,
        q_dot0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
    ) -> dict:
        """Simulate Lagrangian system.

        Parameters
        ----------
        q0:
            Initial generalized coordinates.
        q_dot0:
            Initial generalized velocities.
        t_span:
            Time span (t_start, t_end).
        n_steps:
            Number of timesteps.

        Returns
        -------
        result:
            Dictionary with 't', 'q', 'q_dot', 'energy'.
        """
        from scipy.integrate import solve_ivp

        # Initial state
        y0 = np.concatenate([q0, q_dot0])

        # Solve ODE
        sol = solve_ivp(
            self.euler_lagrange_equations,
            t_span,
            y0,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], n_steps),
            rtol=1e-8,
            atol=1e-10,
        )

        n = self.system.n_dof
        q = sol.y[:n, :].T
        q_dot = sol.y[n:, :].T

        # Compute energy (if conservative)
        energy = np.zeros(len(sol.t))
        for i in range(len(sol.t)):
            # E = T + V, where L = T - V for conservative systems
            # For general case, we compute kinetic energy
            energy[i] = self.system.L(q[i], q_dot[i], sol.t[i])

        return {
            't': sol.t,
            'q': q,
            'q_dot': q_dot,
            'energy': energy,
            'success': sol.success,
        }


def simple_pendulum_lagrangian(m: float = 1.0, l: float = 1.0, g: float = 9.81) -> LagrangianSystem:
    """Create simple pendulum Lagrangian.

    L = (1/2)ml²θ̇² - mgl(1 - cos(θ))

    Parameters
    ----------
    m:
        Mass (kg).
    l:
        Length (m).
    g:
        Gravity (m/s²).

    Returns
    -------
    system:
        Lagrangian system for simple pendulum.
    """
    def L(q, q_dot, t):
        # T = (1/2)ml²θ̇²
        T = 0.5 * m * l ** 2 * q_dot[0] ** 2
        # V = mgl(1 - cos(θ))
        V = m * g * l * (1 - np.cos(q[0]))
        return T - V

    def dL_dq(q, q_dot, t):
        # ∂L/∂θ = mgl sin(θ)
        return np.array([-m * g * l * np.sin(q[0])])

    def dL_dqdot(q, q_dot, t):
        # ∂L/∂θ̇ = ml²θ̇
        return np.array([m * l ** 2 * q_dot[0]])

    return LagrangianSystem(n_dof=1, L=L, dL_dq=dL_dq, dL_dqdot=dL_dqdot)


def double_pendulum_lagrangian(
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> LagrangianSystem:
    """Create double pendulum Lagrangian (chaotic system).

    Parameters
    ----------
    m1, m2:
        Masses (kg).
    l1, l2:
        Lengths (m).
    g:
        Gravity (m/s²).

    Returns
    -------
    system:
        Double pendulum Lagrangian system.
    """
    def L(q, q_dot, t):
        theta1, theta2 = q
        theta1_dot, theta2_dot = q_dot

        # Kinetic energy
        T1 = 0.5 * m1 * l1 ** 2 * theta1_dot ** 2
        T2 = 0.5 * m2 * (
            l1 ** 2 * theta1_dot ** 2
            + l2 ** 2 * theta2_dot ** 2
            + 2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2)
        )
        T = T1 + T2

        # Potential energy
        V1 = -m1 * g * l1 * np.cos(theta1)
        V2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
        V = V1 + V2

        return T - V

    def dL_dq(q, q_dot, t):
        theta1, theta2 = q
        theta1_dot, theta2_dot = q_dot

        # ∂L/∂θ₁
        dL_dtheta1 = (
            -m2 * l1 * l2 * theta1_dot * theta2_dot * np.sin(theta1 - theta2)
            - (m1 + m2) * g * l1 * np.sin(theta1)
        )

        # ∂L/∂θ₂
        dL_dtheta2 = (
            m2 * l1 * l2 * theta1_dot * theta2_dot * np.sin(theta1 - theta2)
            - m2 * g * l2 * np.sin(theta2)
        )

        return np.array([dL_dtheta1, dL_dtheta2])

    def dL_dqdot(q, q_dot, t):
        theta1, theta2 = q
        theta1_dot, theta2_dot = q_dot

        # ∂L/∂θ̇₁
        dL_dtheta1_dot = (
            (m1 + m2) * l1 ** 2 * theta1_dot
            + m2 * l1 * l2 * theta2_dot * np.cos(theta1 - theta2)
        )

        # ∂L/∂θ̇₂
        dL_dtheta2_dot = (
            m2 * l2 ** 2 * theta2_dot
            + m2 * l1 * l2 * theta1_dot * np.cos(theta1 - theta2)
        )

        return np.array([dL_dtheta1_dot, dL_dtheta2_dot])

    return LagrangianSystem(n_dof=2, L=L, dL_dq=dL_dq, dL_dqdot=dL_dqdot)


class ActionPrinciple:
    """Implement principle of least action and variational calculus."""

    @staticmethod
    def compute_action(
        system: LagrangianSystem,
        q_trajectory: np.ndarray,
        q_dot_trajectory: np.ndarray,
        t_trajectory: np.ndarray,
    ) -> float:
        """Compute action integral S = ∫L dt along a trajectory.

        Parameters
        ----------
        system:
            Lagrangian system.
        q_trajectory:
            Position trajectory (n_steps, n_dof).
        q_dot_trajectory:
            Velocity trajectory (n_steps, n_dof).
        t_trajectory:
            Time array (n_steps,).

        Returns
        -------
        S:
            Action value.
        """
        action = 0.0
        for i in range(len(t_trajectory) - 1):
            L_i = system.L(q_trajectory[i], q_dot_trajectory[i], t_trajectory[i])
            L_ip1 = system.L(q_trajectory[i + 1], q_dot_trajectory[i + 1], t_trajectory[i + 1])

            dt = t_trajectory[i + 1] - t_trajectory[i]
            # Trapezoidal rule
            action += 0.5 * (L_i + L_ip1) * dt

        return action


def brachistochrone_problem():
    """Classic brachistochrone problem (fastest descent curve).

    Returns the Lagrangian for finding the curve of fastest descent
    under gravity.
    """
    g = 9.81

    def L(q, q_dot, t):
        # y is height (downward positive)
        # x is horizontal position
        # q = [x], q_dot = [dx/dt]
        # For brachistochrone: minimize time
        # Functional: T = ∫ ds/v = ∫ √(1 + y'²) / √(2gy) dx
        # This is equivalent to Lagrangian formulation
        y = q[0]  # In this simplified version
        dy_dx = q_dot[0]

        if y <= 0:
            return float('inf')

        return np.sqrt((1 + dy_dx ** 2) / (2 * g * y))

    def dL_dq(q, q_dot, t):
        y = q[0]
        dy_dx = q_dot[0]

        if y <= 1e-10:
            return np.array([0.0])

        return np.array([-0.5 * np.sqrt((1 + dy_dx ** 2) / (2 * g)) * y ** (-1.5)])

    def dL_dqdot(q, q_dot, t):
        y = q[0]
        dy_dx = q_dot[0]

        if y <= 0:
            return np.array([0.0])

        return np.array([dy_dx / np.sqrt(2 * g * y * (1 + dy_dx ** 2))])

    return LagrangianSystem(n_dof=1, L=L, dL_dq=dL_dq, dL_dqdot=dL_dqdot)
