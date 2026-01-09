"""Physics-Informed Neural Networks for classical mechanics.

This module implements PINNs that respect Newton's laws, conservation of
energy and momentum, and other fundamental mechanical principles.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import PINN, PINNConfig


class MechanicsPINN(PINN):
    """PINN for classical mechanics problems.

    Enforces Newton's laws and conservation principles:
    - F = ma (Newton's second law)
    - Energy conservation
    - Momentum conservation (when applicable)

    Input: (t, x, y) or (t, x, y, z) for time and spatial coordinates
    Output: velocity and/or position predictions
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        config: Optional[PINNConfig] = None,
        gravity: float = 9.81,
        mass: float = 1.0,
    ):
        """Initialize mechanics PINN.

        Parameters
        ----------
        spatial_dim:
            Spatial dimensionality (1, 2, or 3).
        config:
            PINN configuration.
        gravity:
            Gravitational acceleration (m/s²).
        mass:
            Mass of the object (kg).
        """
        # Input: time + spatial coordinates
        # Output: spatial coordinates (network learns position as function of time)
        input_dim = 1 + spatial_dim
        output_dim = spatial_dim

        super().__init__(input_dim, output_dim, config)

        self.spatial_dim = spatial_dim
        self.gravity = gravity
        self.mass = mass

    def physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics loss enforcing Newton's second law.

        For a projectile: d²r/dt² = -g (vertical component)
        Plus any drag or other forces.

        Parameters
        ----------
        x:
            Input tensor with shape (N, 1+spatial_dim) containing [t, spatial_coords].
        y_pred:
            Predicted positions with shape (N, spatial_dim).

        Returns
        -------
        loss:
            Physics violation penalty.
        """
        x = x.requires_grad_(True)

        # Get position predictions
        r = self.forward(x)

        # Compute first derivatives (velocity)
        v = []
        for i in range(self.spatial_dim):
            grad = torch.autograd.grad(
                r[:, i:i+1],
                x,
                grad_outputs=torch.ones_like(r[:, i:i+1]),
                create_graph=True,
                retain_graph=True,
            )[0]
            v.append(grad[:, 0:1])  # Time derivative only

        v = torch.cat(v, dim=-1)

        # Compute second derivatives (acceleration)
        a = []
        for i in range(self.spatial_dim):
            grad = torch.autograd.grad(
                v[:, i:i+1],
                x,
                grad_outputs=torch.ones_like(v[:, i:i+1]),
                create_graph=True,
                retain_graph=True,
            )[0]
            a.append(grad[:, 0:1])  # Time derivative only

        a = torch.cat(a, dim=-1)

        # Physics: For free fall, a_y = -g, a_x = 0 (no drag in this simple version)
        if self.spatial_dim == 2:
            # 2D: x, y coordinates
            a_expected = torch.zeros_like(a)
            a_expected[:, 1] = -self.gravity  # Gravity in y direction
        elif self.spatial_dim == 3:
            # 3D: x, y, z coordinates (z is vertical)
            a_expected = torch.zeros_like(a)
            a_expected[:, 2] = -self.gravity  # Gravity in z direction
        else:
            # 1D vertical motion
            a_expected = torch.full_like(a, -self.gravity)

        # Physics loss: predicted acceleration should match expected
        physics_residual = a - a_expected
        return torch.mean(physics_residual ** 2)


class HamiltonianPINN(PINN):
    """PINN that preserves Hamiltonian structure.

    Enforces energy conservation through Hamiltonian mechanics.
    Input: (t, q, p) where q are generalized coordinates, p are momenta
    Output: time derivatives (dq/dt, dp/dt)
    """

    def __init__(
        self,
        n_dof: int,
        config: Optional[PINNConfig] = None,
    ):
        """Initialize Hamiltonian PINN.

        Parameters
        ----------
        n_dof:
            Number of degrees of freedom.
        config:
            PINN configuration.
        """
        # Input: time + q + p
        input_dim = 1 + 2 * n_dof
        # Output: q and p (state)
        output_dim = 2 * n_dof

        super().__init__(input_dim, output_dim, config)
        self.n_dof = n_dof

        # Learnable Hamiltonian network
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(2 * n_dof, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(q, p).

        Parameters
        ----------
        q:
            Generalized coordinates.
        p:
            Generalized momenta.

        Returns
        -------
        H:
            Hamiltonian value (total energy).
        """
        qp = torch.cat([q, p], dim=-1)
        return self.hamiltonian_net(qp)

    def physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce Hamiltonian dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.

        Parameters
        ----------
        x:
            Input tensor [t, q, p].
        y_pred:
            Predicted state [q, p].

        Returns
        -------
        loss:
            Hamiltonian structure violation.
        """
        x = x.requires_grad_(True)

        # Split state into q and p
        n = self.n_dof
        q = y_pred[:, :n]
        p = y_pred[:, n:]

        # Compute Hamiltonian
        H = self.compute_hamiltonian(q, p)

        # Compute ∂H/∂q and ∂H/∂p
        dH_dq = torch.autograd.grad(
            H, q, grad_outputs=torch.ones_like(H),
            create_graph=True, retain_graph=True
        )[0]

        dH_dp = torch.autograd.grad(
            H, p, grad_outputs=torch.ones_like(H),
            create_graph=True, retain_graph=True
        )[0]

        # Compute time derivatives dq/dt and dp/dt
        dq_dt = []
        dp_dt = []

        for i in range(n):
            grad_q = torch.autograd.grad(
                q[:, i:i+1], x,
                grad_outputs=torch.ones_like(q[:, i:i+1]),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]  # Time component
            dq_dt.append(grad_q)

            grad_p = torch.autograd.grad(
                p[:, i:i+1], x,
                grad_outputs=torch.ones_like(p[:, i:i+1]),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]
            dp_dt.append(grad_p)

        dq_dt = torch.cat(dq_dt, dim=-1)
        dp_dt = torch.cat(dp_dt, dim=-1)

        # Hamilton's equations
        residual_q = dq_dt - dH_dp
        residual_p = dp_dt + dH_dq

        return torch.mean(residual_q ** 2 + residual_p ** 2)
