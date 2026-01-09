"""Physics-Informed Neural Networks for Electromagnetism.

Implements PINNs that learn solutions to Maxwell's equations
while respecting electromagnetic conservation laws.
"""

from __future__ import annotations

import torch
from scipy.constants import c, epsilon_0, mu_0

from .base import PINN


class MaxwellPINN(PINN):
    """PINN for Maxwell's equations in vacuum.

    Enforces:
    - ∇·E = ρ/ε₀ (Gauss's law)
    - ∇·B = 0 (No magnetic monopoles)
    - ∇×E = -∂B/∂t (Faraday's law)
    - ∇×B = μ₀ε₀∂E/∂t (Ampère's law, vacuum)

    Network inputs: (t, x, y, z)
    Network outputs: (Ex, Ey, Ez, Bx, By, Bz)
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
    ):
        """Initialize Maxwell PINN.

        Parameters
        ----------
        hidden_layers:
            List of hidden layer sizes.
        activation:
            Activation function name.
        """
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]
        super().__init__(
            input_dim=4,  # (t, x, y, z)
            output_dim=6,  # (Ex, Ey, Ez, Bx, By, Bz)
            hidden_layers=hidden_layers,
            activation=activation,
        )
        self.c = c
        self.epsilon_0 = epsilon_0
        self.mu_0 = mu_0

    def physics_loss(
        self,
        x: torch.Tensor,
        rho: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute physics loss enforcing Maxwell's equations.

        Parameters
        ----------
        x:
            Input tensor (batch_size, 4) of (t, x, y, z).
        rho:
            Charge density (batch_size, 1). If None, assumes vacuum (rho=0).

        Returns
        -------
        losses:
            Dictionary of physics loss components.
        """
        x = x.requires_grad_(True)

        # Forward pass: predict E and B fields
        fields = self.forward(x)
        Ex = fields[:, 0:1]
        Ey = fields[:, 1:2]
        Ez = fields[:, 2:3]
        Bx = fields[:, 3:4]
        By = fields[:, 4:5]
        Bz = fields[:, 5:6]

        # Compute gradients
        # ∂E/∂t, ∂E/∂x, ∂E/∂y, ∂E/∂z
        Ex_grads = self._compute_all_gradients(Ex, x)
        Ey_grads = self._compute_all_gradients(Ey, x)
        Ez_grads = self._compute_all_gradients(Ez, x)

        # ∂B/∂t, ∂B/∂x, ∂B/∂y, ∂B/∂z
        Bx_grads = self._compute_all_gradients(Bx, x)
        By_grads = self._compute_all_gradients(By, x)
        Bz_grads = self._compute_all_gradients(Bz, x)

        # Gauss's law: ∇·E = ρ/ε₀
        div_E = Ex_grads["dx"] + Ey_grads["dy"] + Ez_grads["dz"]
        if rho is None:
            rho = torch.zeros_like(div_E)
        gauss_electric = div_E - rho / self.epsilon_0

        # No magnetic monopoles: ∇·B = 0
        div_B = Bx_grads["dx"] + By_grads["dy"] + Bz_grads["dz"]

        # Faraday's law: ∇×E = -∂B/∂t
        curl_E_x = Ez_grads["dy"] - Ey_grads["dz"]
        curl_E_y = Ex_grads["dz"] - Ez_grads["dx"]
        curl_E_z = Ey_grads["dx"] - Ex_grads["dy"]

        faraday_x = curl_E_x + Bx_grads["dt"]
        faraday_y = curl_E_y + By_grads["dt"]
        faraday_z = curl_E_z + Bz_grads["dt"]

        # Ampère's law (vacuum): ∇×B = μ₀ε₀∂E/∂t
        curl_B_x = Bz_grads["dy"] - By_grads["dz"]
        curl_B_y = Bx_grads["dz"] - Bz_grads["dx"]
        curl_B_z = By_grads["dx"] - Bx_grads["dy"]

        ampere_x = curl_B_x - self.mu_0 * self.epsilon_0 * Ex_grads["dt"]
        ampere_y = curl_B_y - self.mu_0 * self.epsilon_0 * Ey_grads["dt"]
        ampere_z = curl_B_z - self.mu_0 * self.epsilon_0 * Ez_grads["dt"]

        return {
            "gauss_electric": torch.mean(gauss_electric**2),
            "gauss_magnetic": torch.mean(div_B**2),
            "faraday_x": torch.mean(faraday_x**2),
            "faraday_y": torch.mean(faraday_y**2),
            "faraday_z": torch.mean(faraday_z**2),
            "ampere_x": torch.mean(ampere_x**2),
            "ampere_y": torch.mean(ampere_y**2),
            "ampere_z": torch.mean(ampere_z**2),
        }

    def _compute_all_gradients(
        self, field: torch.Tensor, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute all first derivatives of a field component.

        Parameters
        ----------
        field:
            Field component (batch_size, 1).
        x:
            Input coordinates (batch_size, 4) = (t, x, y, z).

        Returns
        -------
        grads:
            Dictionary with keys "dt", "dx", "dy", "dz".
        """
        grad = torch.autograd.grad(
            field,
            x,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True,
        )[0]

        return {
            "dt": grad[:, 0:1],
            "dx": grad[:, 1:2],
            "dy": grad[:, 2:3],
            "dz": grad[:, 3:4],
        }

    def energy_density(self, fields: torch.Tensor) -> torch.Tensor:
        """Calculate electromagnetic energy density u = (ε₀/2)|E|² + (1/2μ₀)|B|².

        Parameters
        ----------
        fields:
            Field tensor (batch_size, 6) = (Ex, Ey, Ez, Bx, By, Bz).

        Returns
        -------
        u:
            Energy density (batch_size, 1).
        """
        E = fields[:, 0:3]
        B = fields[:, 3:6]

        E_energy = 0.5 * self.epsilon_0 * torch.sum(E**2, dim=1, keepdim=True)
        B_energy = 0.5 * torch.sum(B**2, dim=1, keepdim=True) / self.mu_0

        return E_energy + B_energy


class ElectrostaticsPINN(PINN):
    """PINN for electrostatics (Poisson's equation).

    Enforces:
    - ∇²φ = -ρ/ε₀ (Poisson's equation)
    - E = -∇φ (Electric field from potential)

    Network inputs: (x, y, z)
    Network outputs: (φ,) - electric potential
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
    ):
        """Initialize electrostatics PINN.

        Parameters
        ----------
        hidden_layers:
            List of hidden layer sizes.
        activation:
            Activation function name.
        """
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]
        super().__init__(
            input_dim=3,  # (x, y, z)
            output_dim=1,  # φ
            hidden_layers=hidden_layers,
            activation=activation,
        )
        self.epsilon_0 = epsilon_0

    def physics_loss(
        self,
        x: torch.Tensor,
        rho: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute physics loss enforcing Poisson's equation.

        Parameters
        ----------
        x:
            Input tensor (batch_size, 3) of (x, y, z).
        rho:
            Charge density (batch_size, 1). If None, assumes vacuum (Laplace eq).

        Returns
        -------
        losses:
            Dictionary of physics loss components.
        """
        x = x.requires_grad_(True)

        # Forward pass: predict potential φ
        phi = self.forward(x)

        # Compute ∇φ
        grad_phi = torch.autograd.grad(
            phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]

        dphi_dx = grad_phi[:, 0:1]
        dphi_dy = grad_phi[:, 1:2]
        dphi_dz = grad_phi[:, 2:3]

        # Compute ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx,
            x,
            grad_outputs=torch.ones_like(dphi_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2phi_dy2 = torch.autograd.grad(
            dphi_dy,
            x,
            grad_outputs=torch.ones_like(dphi_dy),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        d2phi_dz2 = torch.autograd.grad(
            dphi_dz,
            x,
            grad_outputs=torch.ones_like(dphi_dz),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        laplacian_phi = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

        # Poisson equation: ∇²φ = -ρ/ε₀
        if rho is None:
            rho = torch.zeros_like(laplacian_phi)

        poisson_residual = laplacian_phi + rho / self.epsilon_0

        return {
            "poisson": torch.mean(poisson_residual**2),
        }

    def electric_field(self, x: torch.Tensor) -> torch.Tensor:
        """Compute electric field E = -∇φ.

        Parameters
        ----------
        x:
            Input coordinates (batch_size, 3).

        Returns
        -------
        E:
            Electric field (batch_size, 3).
        """
        x = x.requires_grad_(True)
        phi = self.forward(x)

        grad_phi = torch.autograd.grad(
            phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]

        return -grad_phi


class MagnetostaticsPINN(PINN):
    """PINN for magnetostatics (vector potential formulation).

    Enforces:
    - ∇²A = -μ₀J (Poisson equation for vector potential)
    - B = ∇×A (Magnetic field from vector potential)
    - ∇·A = 0 (Coulomb gauge)

    Network inputs: (x, y, z)
    Network outputs: (Ax, Ay, Az) - magnetic vector potential
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
    ):
        """Initialize magnetostatics PINN.

        Parameters
        ----------
        hidden_layers:
            List of hidden layer sizes.
        activation:
            Activation function name.
        """
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]
        super().__init__(
            input_dim=3,  # (x, y, z)
            output_dim=3,  # (Ax, Ay, Az)
            hidden_layers=hidden_layers,
            activation=activation,
        )
        self.mu_0 = mu_0

    def physics_loss(
        self,
        x: torch.Tensor,
        J: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute physics loss enforcing vector Poisson equation.

        Parameters
        ----------
        x:
            Input tensor (batch_size, 3) of (x, y, z).
        J:
            Current density (batch_size, 3). If None, assumes no current.

        Returns
        -------
        losses:
            Dictionary of physics loss components.
        """
        x = x.requires_grad_(True)

        # Forward pass: predict vector potential A
        A = self.forward(x)
        Ax = A[:, 0:1]
        Ay = A[:, 1:2]
        Az = A[:, 2:3]

        # Compute Laplacian for each component
        laplacian_Ax = self._compute_laplacian(Ax, x)
        laplacian_Ay = self._compute_laplacian(Ay, x)
        laplacian_Az = self._compute_laplacian(Az, x)

        # Vector Poisson: ∇²A = -μ₀J
        if J is None:
            J = torch.zeros_like(A)

        poisson_x = laplacian_Ax + self.mu_0 * J[:, 0:1]
        poisson_y = laplacian_Ay + self.mu_0 * J[:, 1:2]
        poisson_z = laplacian_Az + self.mu_0 * J[:, 2:3]

        # Coulomb gauge: ∇·A = 0
        grad_A = torch.autograd.grad(
            A,
            x,
            grad_outputs=torch.ones_like(A),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Reshape gradient properly for divergence
        dAx_dx = grad_A[:, 0]  # ∂Ax/∂x
        dAy_dy = grad_A[:, 4]  # ∂Ay/∂y (index 1*3 + 1)
        dAz_dz = grad_A[:, 8]  # ∂Az/∂z (index 2*3 + 2)

        div_A = (dAx_dx + dAy_dy + dAz_dz).unsqueeze(1)

        return {
            "poisson_x": torch.mean(poisson_x**2),
            "poisson_y": torch.mean(poisson_y**2),
            "poisson_z": torch.mean(poisson_z**2),
            "gauge": torch.mean(div_A**2),
        }

    def _compute_laplacian(
        self, field: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian ∇²field.

        Parameters
        ----------
        field:
            Scalar field (batch_size, 1).
        x:
            Coordinates (batch_size, 3).

        Returns
        -------
        laplacian:
            ∇²field (batch_size, 1).
        """
        grad_field = torch.autograd.grad(
            field,
            x,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True,
        )[0]

        d2_dx2 = torch.autograd.grad(
            grad_field[:, 0:1],
            x,
            grad_outputs=torch.ones_like(grad_field[:, 0:1]),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2_dy2 = torch.autograd.grad(
            grad_field[:, 1:2],
            x,
            grad_outputs=torch.ones_like(grad_field[:, 1:2]),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        d2_dz2 = torch.autograd.grad(
            grad_field[:, 2:3],
            x,
            grad_outputs=torch.ones_like(grad_field[:, 2:3]),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        return d2_dx2 + d2_dy2 + d2_dz2

    def magnetic_field(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnetic field B = ∇×A.

        Parameters
        ----------
        x:
            Input coordinates (batch_size, 3).

        Returns
        -------
        B:
            Magnetic field (batch_size, 3).
        """
        x = x.requires_grad_(True)
        A = self.forward(x)

        # Compute curl of A
        grad_A = torch.autograd.grad(
            A,
            x,
            grad_outputs=torch.ones_like(A),
            create_graph=True,
        )[0]

        # B = ∇×A = (∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x, ∂Ay/∂x - ∂Ax/∂y)
        # grad_A has shape (batch, 9) flattened as [dAx/dx, dAx/dy, dAx/dz, dAy/dx, ...]
        dAz_dy = grad_A[:, 7:8]  # Index 2*3 + 1
        dAy_dz = grad_A[:, 5:6]  # Index 1*3 + 2
        dAx_dz = grad_A[:, 2:3]  # Index 0*3 + 2
        dAz_dx = grad_A[:, 6:7]  # Index 2*3 + 0
        dAy_dx = grad_A[:, 3:4]  # Index 1*3 + 0
        dAx_dy = grad_A[:, 1:2]  # Index 0*3 + 1

        Bx = dAz_dy - dAy_dz
        By = dAx_dz - dAz_dx
        Bz = dAy_dx - dAx_dy

        return torch.cat([Bx, By, Bz], dim=1)
