"""Base classes for Physics-Informed Neural Networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Networks.

    Attributes
    ----------
    hidden_layers:
        List of hidden layer sizes.
    activation:
        Activation function name ('tanh', 'relu', 'gelu', 'silu').
    learning_rate:
        Learning rate for optimizer.
    physics_weight:
        Weight for physics loss term.
    data_weight:
        Weight for data fitting loss term.
    boundary_weight:
        Weight for boundary condition loss term.
    """

    hidden_layers: list[int] = None
    activation: str = "tanh"
    learning_rate: float = 1e-3
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 1.0

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64]


class PhysicsLoss(ABC):
    """Abstract base class for physics-based loss functions.

    Physics losses encode the differential equations and conservation laws
    that govern the system behavior.
    """

    @abstractmethod
    def compute(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the physics-based loss.

        Parameters
        ----------
        inputs:
            Input tensor (typically spatial-temporal coordinates).
        outputs:
            Network predictions.
        model:
            The neural network model (needed for computing derivatives).

        Returns
        -------
        loss:
            Physics loss scalar.
        """
        pass


class PINN(nn.Module):
    """Base Physics-Informed Neural Network.

    A PINN is a neural network trained to satisfy:
    1. Data fitting loss (if training data available)
    2. Physics loss (differential equations, conservation laws)
    3. Boundary/initial condition loss

    This creates a network that not only fits data but respects physical laws.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[PINNConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or PINNConfig()

        # Build network architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization (good for tanh)
        self._initialize_weights()

    def _get_activation(self) -> nn.Module:
        """Get activation function from config."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(self.config.activation, nn.Tanh())

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        output:
            Network predictions of shape (batch_size, output_dim).
        """
        return self.network(x)

    def compute_gradients(
        self,
        x: torch.Tensor,
        order: int = 1,
        output_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients of network outputs w.r.t. inputs.

        This is essential for enforcing differential equations in PINNs.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, input_dim).
        order:
            Maximum order of derivatives to compute (1 or 2).
        output_idx:
            Index of output to compute gradients for.

        Returns
        -------
        gradients:
            Dictionary containing first and second order derivatives.
        """
        x = x.requires_grad_(True)
        y = self.forward(x)

        if y.shape[-1] == 1:
            y_selected = y
        else:
            y_selected = y[:, output_idx:output_idx+1]

        # First order derivatives
        grads_1 = []
        for i in range(x.shape[-1]):
            grad = torch.autograd.grad(
                y_selected,
                x,
                grad_outputs=torch.ones_like(y_selected),
                create_graph=True,
                retain_graph=True,
            )[0]
            grads_1.append(grad[:, i:i+1])

        grad_dict = {"dy_dx": torch.cat(grads_1, dim=-1)}

        # Second order derivatives (if requested)
        if order >= 2:
            grads_2 = []
            for i in range(x.shape[-1]):
                grad_2 = torch.autograd.grad(
                    grads_1[i],
                    x,
                    grad_outputs=torch.ones_like(grads_1[i]),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grads_2.append(grad_2[:, i:i+1])
            grad_dict["d2y_dx2"] = torch.cat(grads_2, dim=-1)

        return grad_dict

    @abstractmethod
    def physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics-based loss.

        This method must be implemented by subclasses to encode the specific
        physical laws governing the system.

        Parameters
        ----------
        x:
            Input tensor (coordinates).
        y_pred:
            Network predictions.

        Returns
        -------
        loss:
            Physics loss value.
        """
        pass

    def data_loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute data fitting loss.

        Parameters
        ----------
        x:
            Input tensor.
        y_true:
            Ground truth values.

        Returns
        -------
        loss:
            Mean squared error between predictions and ground truth.
        """
        y_pred = self.forward(x)
        return torch.mean((y_pred - y_true) ** 2)

    def boundary_loss(
        self,
        x_boundary: torch.Tensor,
        y_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary condition loss.

        Parameters
        ----------
        x_boundary:
            Boundary points.
        y_boundary:
            Boundary values.

        Returns
        -------
        loss:
            Boundary condition violation penalty.
        """
        y_pred = self.forward(x_boundary)
        return torch.mean((y_pred - y_boundary) ** 2)

    def total_loss(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        x_boundary: Optional[torch.Tensor] = None,
        y_boundary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total weighted loss.

        Parameters
        ----------
        x:
            Input points for physics loss.
        y_true:
            True values for data loss (optional).
        x_boundary:
            Boundary points (optional).
        y_boundary:
            Boundary values (optional).

        Returns
        -------
        total_loss:
            Weighted sum of all loss components.
        loss_dict:
            Dictionary with individual loss values for logging.
        """
        y_pred = self.forward(x)

        # Physics loss (always computed)
        phys_loss = self.physics_loss(x, y_pred)
        total = self.config.physics_weight * phys_loss

        loss_dict = {"physics": phys_loss.item()}

        # Data loss (if training data provided)
        if y_true is not None:
            d_loss = self.data_loss(x, y_true)
            total = total + self.config.data_weight * d_loss
            loss_dict["data"] = d_loss.item()

        # Boundary loss (if boundary conditions provided)
        if x_boundary is not None and y_boundary is not None:
            b_loss = self.boundary_loss(x_boundary, y_boundary)
            total = total + self.config.boundary_weight * b_loss
            loss_dict["boundary"] = b_loss.item()

        loss_dict["total"] = total.item()

        return total, loss_dict
