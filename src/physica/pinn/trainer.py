"""Training utilities for Physics-Informed Neural Networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .base import PINN


@dataclass
class TrainingConfig:
    """Configuration for PINN training.

    Attributes
    ----------
    n_epochs:
        Number of training epochs.
    batch_size:
        Batch size for mini-batch training.
    n_physics_points:
        Number of collocation points for physics loss.
    learning_rate:
        Learning rate.
    scheduler_gamma:
        Learning rate decay factor.
    scheduler_step:
        Steps between learning rate decay.
    device:
        Device to train on ('cpu', 'cuda', 'mps').
    verbose:
        Whether to show progress bar.
    """

    n_epochs: int = 1000
    batch_size: int = 256
    n_physics_points: int = 10000
    learning_rate: float = 1e-3
    scheduler_gamma: float = 0.95
    scheduler_step: int = 100
    device: str = "cpu"
    verbose: bool = True


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks.

    Handles:
    - Training loop with physics and data losses
    - Learning rate scheduling
    - Adaptive sampling of physics points
    - Loss tracking and visualization
    """

    def __init__(
        self,
        model: PINN,
        config: Optional[TrainingConfig] = None,
    ):
        """Initialize PINN trainer.

        Parameters
        ----------
        model:
            The PINN model to train.
        config:
            Training configuration.
        """
        self.model = model
        self.config = config or TrainingConfig()

        # Setup device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step,
            gamma=self.config.scheduler_gamma,
        )

        # Loss history
        self.history: Dict[str, List[float]] = {
            "total": [],
            "physics": [],
            "data": [],
            "boundary": [],
        }

    def sample_physics_points(
        self,
        bounds: np.ndarray,
        n_points: int,
    ) -> torch.Tensor:
        """Sample collocation points for physics loss.

        Parameters
        ----------
        bounds:
            Array of shape (n_dims, 2) with [min, max] for each dimension.
        n_points:
            Number of points to sample.

        Returns
        -------
        points:
            Sampled points with shape (n_points, n_dims).
        """
        n_dims = bounds.shape[0]
        points = np.random.uniform(
            low=bounds[:, 0],
            high=bounds[:, 1],
            size=(n_points, n_dims),
        )
        return torch.tensor(points, dtype=torch.float32, device=self.device)

    def train_epoch(
        self,
        physics_bounds: np.ndarray,
        data_x: Optional[torch.Tensor] = None,
        data_y: Optional[torch.Tensor] = None,
        boundary_x: Optional[torch.Tensor] = None,
        boundary_y: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        physics_bounds:
            Bounds for sampling physics points.
        data_x:
            Training data inputs (optional).
        data_y:
            Training data outputs (optional).
        boundary_x:
            Boundary condition points (optional).
        boundary_y:
            Boundary condition values (optional).

        Returns
        -------
        losses:
            Dictionary of loss values.
        """
        self.model.train()

        # Sample physics collocation points
        x_physics = self.sample_physics_points(
            physics_bounds,
            self.config.n_physics_points,
        )

        # Move data to device if provided
        if data_x is not None:
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

        if boundary_x is not None:
            boundary_x = boundary_x.to(self.device)
            boundary_y = boundary_y.to(self.device)

        # Compute loss
        self.optimizer.zero_grad()

        total_loss, loss_dict = self.model.total_loss(
            x=x_physics,
            y_true=data_y if data_x is not None else None,
            x_boundary=boundary_x,
            y_boundary=boundary_y,
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return loss_dict

    def train(
        self,
        physics_bounds: np.ndarray,
        data_x: Optional[torch.Tensor] = None,
        data_y: Optional[torch.Tensor] = None,
        boundary_x: Optional[torch.Tensor] = None,
        boundary_y: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Train the PINN.

        Parameters
        ----------
        physics_bounds:
            Bounds for each input dimension, shape (n_dims, 2).
        data_x:
            Training data inputs (optional).
        data_y:
            Training data outputs (optional).
        boundary_x:
            Boundary points (optional).
        boundary_y:
            Boundary values (optional).

        Returns
        -------
        history:
            Training history with loss curves.
        """
        iterator = range(self.config.n_epochs)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Training PINN")

        for epoch in iterator:
            loss_dict = self.train_epoch(
                physics_bounds=physics_bounds,
                data_x=data_x,
                data_y=data_y,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )

            # Store history
            for key in ["total", "physics", "data", "boundary"]:
                if key in loss_dict:
                    self.history[key].append(loss_dict[key])

            # Update learning rate
            self.scheduler.step()

            # Log progress
            if self.config.verbose and epoch % 100 == 0:
                lr = self.scheduler.get_last_lr()[0]
                iterator.set_postfix({
                    "loss": f"{loss_dict['total']:.4e}",
                    "physics": f"{loss_dict['physics']:.4e}",
                    "lr": f"{lr:.4e}",
                })

        return self.history

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions with trained model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        predictions:
            Model predictions as numpy array.
        """
        self.model.eval()
        with torch.no_grad():
            x_device = x.to(self.device)
            y_pred = self.model(x_device)
            return y_pred.cpu().numpy()
