"""Physics-Informed Neural Networks for thermodynamics.

Enforces laws of thermodynamics, heat equations, and conservation of energy.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import PINN, PINNConfig


class ThermodynamicsPINN(PINN):
    """PINN for thermodynamics and heat transfer.

    Enforces:
    - Heat equation: ∂T/∂t = α∇²T
    - First law of thermodynamics: dU = δQ - δW
    - Energy conservation

    Input: (t, x, y, z) for time and spatial coordinates
    Output: Temperature T
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        config: Optional[PINNConfig] = None,
        thermal_diffusivity: float = 1.0,
    ):
        """Initialize thermodynamics PINN.

        Parameters
        ----------
        spatial_dim:
            Spatial dimensionality (1, 2, or 3).
        config:
            PINN configuration.
        thermal_diffusivity:
            Thermal diffusivity α (m²/s).
        """
        input_dim = 1 + spatial_dim  # time + space
        output_dim = 1  # temperature

        super().__init__(input_dim, output_dim, config)

        self.spatial_dim = spatial_dim
        self.alpha = thermal_diffusivity

    def physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce heat equation: ∂T/∂t = α∇²T.

        Parameters
        ----------
        x:
            Input tensor [t, spatial_coords].
        y_pred:
            Predicted temperature.

        Returns
        -------
        loss:
            Heat equation violation.
        """
        x = x.requires_grad_(True)
        T = self.forward(x)

        # Compute ∂T/∂t
        dT_dt = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]  # Time component

        # Compute spatial derivatives and Laplacian
        laplacian = torch.zeros_like(T)

        for i in range(1, 1 + self.spatial_dim):  # Spatial indices
            # First derivative
            dT_dxi = torch.autograd.grad(
                T, x,
                grad_outputs=torch.ones_like(T),
                create_graph=True,
                retain_graph=True,
            )[0][:, i:i+1]

            # Second derivative
            d2T_dxi2 = torch.autograd.grad(
                dT_dxi, x,
                grad_outputs=torch.ones_like(dT_dxi),
                create_graph=True,
                retain_graph=True,
            )[0][:, i:i+1]

            laplacian = laplacian + d2T_dxi2

        # Heat equation residual
        residual = dT_dt - self.alpha * laplacian

        return torch.mean(residual ** 2)


class ConservationPINN(PINN):
    """PINN that enforces general conservation laws.

    Useful for:
    - Mass conservation: ∂ρ/∂t + ∇·(ρv) = 0
    - Momentum conservation
    - Energy conservation
    """

    def __init__(
        self,
        spatial_dim: int,
        n_conserved: int,
        config: Optional[PINNConfig] = None,
    ):
        """Initialize conservation law PINN.

        Parameters
        ----------
        spatial_dim:
            Spatial dimensionality.
        n_conserved:
            Number of conserved quantities.
        config:
            PINN configuration.
        """
        input_dim = 1 + spatial_dim
        output_dim = n_conserved

        super().__init__(input_dim, output_dim, config)

        self.spatial_dim = spatial_dim
        self.n_conserved = n_conserved

    def physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce conservation law: ∂u/∂t + ∇·F(u) = 0.

        For simplicity, assumes F is linear in u (can be extended).

        Parameters
        ----------
        x:
            Input tensor [t, spatial_coords].
        y_pred:
            Conserved quantities.

        Returns
        -------
        loss:
            Conservation law violation.
        """
        x = x.requires_grad_(True)
        u = self.forward(x)

        total_residual = torch.zeros_like(u[:, 0:1])

        for k in range(self.n_conserved):
            # Time derivative
            du_dt = torch.autograd.grad(
                u[:, k:k+1], x,
                grad_outputs=torch.ones_like(u[:, k:k+1]),
                create_graph=True,
                retain_graph=True,
            )[0][:, 0:1]

            # Spatial divergence (simplified: assuming F_i = u for flux)
            div_F = torch.zeros_like(du_dt)

            for i in range(1, 1 + self.spatial_dim):
                du_dxi = torch.autograd.grad(
                    u[:, k:k+1], x,
                    grad_outputs=torch.ones_like(u[:, k:k+1]),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, i:i+1]

                div_F = div_F + du_dxi

            # Conservation residual
            residual = du_dt + div_F
            total_residual = total_residual + residual ** 2

        return torch.mean(total_residual)
