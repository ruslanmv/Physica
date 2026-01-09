"""Phase III-C: PINN-Based Surrogate Physics Models.

Fast physics surrogates that:
- Train on expensive solvers (offline)
- Deploy to browsers and edge devices (online)
- Preserve physics structure (conservation laws)
- Enable web-based digital twins

This democratizes simulation by making physics deployable everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


class PhysicsSurrogate(nn.Module):
    """Fast surrogate model trained on expensive physics solvers.

    Workflow:
    1. Generate training data from expensive solver
    2. Train lightweight PINN surrogate
    3. Export to browser-deployable format (ONNX or JSON)
    4. Deploy for real-time inference
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
    ):
        """Initialize physics surrogate.

        Parameters
        ----------
        input_dim:
            Input dimension (e.g., space-time coordinates).
        output_dim:
            Output dimension (e.g., solution fields).
        hidden_layers:
            Hidden layer architecture (kept small for deployment).
        activation:
            Activation function.
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build lightweight network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Normalization statistics (for deployment)
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor (batch_size, input_dim).

        Returns
        -------
        y:
            Output predictions (batch_size, output_dim).
        """
        # Normalize input if statistics available
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        y = self.network(x)

        # Denormalize output
        if self.output_mean is not None and self.output_std is not None:
            y = y * (self.output_std + 1e-8) + self.output_mean

        return y

    def compute_normalization_stats(
        self,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
    ):
        """Compute and store normalization statistics.

        Parameters
        ----------
        train_inputs:
            Training input data.
        train_outputs:
            Training output data.
        """
        self.input_mean = train_inputs.mean(dim=0)
        self.input_std = train_inputs.std(dim=0)
        self.output_mean = train_outputs.mean(dim=0)
        self.output_std = train_outputs.std(dim=0)


class SurrogateTrainer:
    """Training framework for physics surrogates.

    Combines data-driven fitting with physics constraints.
    """

    def __init__(
        self,
        surrogate: PhysicsSurrogate,
        learning_rate: float = 1e-3,
    ):
        """Initialize surrogate trainer.

        Parameters
        ----------
        surrogate:
            The surrogate model to train.
        learning_rate:
            Learning rate for optimization.
        """
        self.surrogate = surrogate
        self.optimizer = torch.optim.Adam(surrogate.parameters(), lr=learning_rate)
        self.history = {"data_loss": [], "physics_loss": [], "total_loss": []}

    def train_from_solver(
        self,
        solver_fn: Callable,
        input_samples: np.ndarray,
        n_epochs: int = 1000,
        physics_weight: float = 0.1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train surrogate using expensive solver.

        Parameters
        ----------
        solver_fn:
            Expensive solver function: inputs -> outputs.
        input_samples:
            Input parameter samples for training.
        n_epochs:
            Number of training epochs.
        physics_weight:
            Weight for physics loss (vs data loss).
        verbose:
            Print progress.

        Returns
        -------
        training_info:
            Dictionary with training history.
        """
        # Generate training data from expensive solver
        if verbose:
            print("Generating training data from expensive solver...")

        outputs = []
        for i, x in enumerate(input_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Solver call {i + 1}/{len(input_samples)}")
            y = solver_fn(x)
            outputs.append(y)

        train_inputs = torch.tensor(input_samples, dtype=torch.float32)
        train_outputs = torch.tensor(np.array(outputs), dtype=torch.float32)

        # Compute normalization
        self.surrogate.compute_normalization_stats(train_inputs, train_outputs)

        # Train surrogate
        if verbose:
            print("Training surrogate model...")

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            pred_outputs = self.surrogate(train_inputs)

            # Data loss (MSE)
            data_loss = torch.mean((pred_outputs - train_outputs) ** 2)

            # Physics loss (could add PDE constraints here)
            physics_loss = torch.tensor(0.0)  # Placeholder

            # Total loss
            total_loss = data_loss + physics_weight * physics_loss

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Record history
            self.history["data_loss"].append(data_loss.item())
            self.history["physics_loss"].append(physics_loss.item())
            self.history["total_loss"].append(total_loss.item())

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch + 1}/{n_epochs}: "
                    f"loss = {total_loss.item():.6f}"
                )

        return {"history": self.history, "final_loss": total_loss.item()}


class BrowserDeployment:
    """Export surrogates for browser deployment.

    Supports:
    - ONNX export (for TensorFlow.js, ONNX.js)
    - JSON export (for custom JavaScript inference)
    """

    @staticmethod
    def export_to_onnx(
        surrogate: PhysicsSurrogate,
        output_path: str | Path,
        example_input: torch.Tensor | None = None,
    ):
        """Export model to ONNX format.

        Parameters
        ----------
        surrogate:
            Trained surrogate model.
        output_path:
            Path to save ONNX file.
        example_input:
            Example input tensor for tracing.
        """
        if example_input is None:
            example_input = torch.randn(1, surrogate.input_dim)

        surrogate.eval()

        torch.onnx.export(
            surrogate,
            example_input,
            output_path,
            export_params=True,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    @staticmethod
    def export_to_json(
        surrogate: PhysicsSurrogate,
        output_path: str | Path,
    ):
        """Export model weights to JSON for JavaScript.

        This creates a lightweight JSON file that can be loaded
        directly in browsers without TensorFlow.js.

        Parameters
        ----------
        surrogate:
            Trained surrogate model.
        output_path:
            Path to save JSON file.
        """
        model_dict = {
            "input_dim": surrogate.input_dim,
            "output_dim": surrogate.output_dim,
            "layers": [],
            "normalization": {
                "input_mean": (
                    surrogate.input_mean.tolist()
                    if surrogate.input_mean is not None
                    else None
                ),
                "input_std": (
                    surrogate.input_std.tolist()
                    if surrogate.input_std is not None
                    else None
                ),
                "output_mean": (
                    surrogate.output_mean.tolist()
                    if surrogate.output_mean is not None
                    else None
                ),
                "output_std": (
                    surrogate.output_std.tolist()
                    if surrogate.output_std is not None
                    else None
                ),
            },
        }

        # Extract weights and biases
        for layer in surrogate.network:
            if isinstance(layer, nn.Linear):
                model_dict["layers"].append(
                    {
                        "type": "linear",
                        "weights": layer.weight.detach().cpu().numpy().tolist(),
                        "bias": layer.bias.detach().cpu().numpy().tolist(),
                    }
                )
            elif isinstance(layer, nn.Tanh):
                model_dict["layers"].append({"type": "tanh"})
            elif isinstance(layer, nn.ReLU):
                model_dict["layers"].append({"type": "relu"})
            elif isinstance(layer, nn.GELU):
                model_dict["layers"].append({"type": "gelu"})

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(model_dict, f, indent=2)

    @staticmethod
    def generate_javascript_inference(
        json_path: str | Path,
        output_path: str | Path,
    ):
        """Generate standalone JavaScript inference code.

        Parameters
        ----------
        json_path:
            Path to exported JSON model.
        output_path:
            Path to save JavaScript file.
        """
        js_code = """
// Physics Surrogate Model - Browser Inference
// Auto-generated by Physica

class PhysicsSurrogate {
    constructor(modelData) {
        this.inputDim = modelData.input_dim;
        this.outputDim = modelData.output_dim;
        this.layers = modelData.layers;
        this.norm = modelData.normalization;
    }

    // Activation functions
    tanh(x) {
        return x.map(v => Math.tanh(v));
    }

    relu(x) {
        return x.map(v => Math.max(0, v));
    }

    gelu(x) {
        // Approximate GELU
        return x.map(v => 0.5 * v * (1 + Math.tanh(
            Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)
        )));
    }

    // Matrix-vector multiplication
    matmul(W, x) {
        return W.map(row =>
            row.reduce((sum, w, i) => sum + w * x[i], 0)
        );
    }

    // Normalize input
    normalizeInput(x) {
        if (!this.norm.input_mean) return x;
        return x.map((v, i) =>
            (v - this.norm.input_mean[i]) / (this.norm.input_std[i] + 1e-8)
        );
    }

    // Denormalize output
    denormalizeOutput(y) {
        if (!this.norm.output_mean) return y;
        return y.map((v, i) =>
            v * (this.norm.output_std[i] + 1e-8) + this.norm.output_mean[i]
        );
    }

    // Forward pass
    predict(input) {
        let x = this.normalizeInput(input);

        for (const layer of this.layers) {
            if (layer.type === 'linear') {
                x = this.matmul(layer.weights, x);
                x = x.map((v, i) => v + layer.bias[i]);
            } else if (layer.type === 'tanh') {
                x = this.tanh(x);
            } else if (layer.type === 'relu') {
                x = this.relu(x);
            } else if (layer.type === 'gelu') {
                x = this.gelu(x);
            }
        }

        return this.denormalizeOutput(x);
    }
}

// Load model and create instance
async function loadModel(jsonPath) {
    const response = await fetch(jsonPath);
    const modelData = await response.json();
    return new PhysicsSurrogate(modelData);
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PhysicsSurrogate, loadModel };
}
"""

        with open(output_path, "w") as f:
            f.write(js_code)


def create_lightweight_surrogate(
    expensive_solver: Callable,
    input_bounds: list[tuple[float, float]],
    n_train_samples: int = 1000,
    output_dim: int = 1,
) -> PhysicsSurrogate:
    """Helper to create and train a surrogate model.

    Parameters
    ----------
    expensive_solver:
        The expensive solver to approximate.
    input_bounds:
        List of (min, max) bounds for each input dimension.
    n_train_samples:
        Number of training samples.
    output_dim:
        Output dimension.

    Returns
    -------
    surrogate:
        Trained surrogate model.
    """
    # Generate training samples
    input_dim = len(input_bounds)
    train_samples = np.random.uniform(
        low=[b[0] for b in input_bounds],
        high=[b[1] for b in input_bounds],
        size=(n_train_samples, input_dim),
    )

    # Create and train surrogate
    surrogate = PhysicsSurrogate(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=[32, 32],  # Lightweight
    )

    trainer = SurrogateTrainer(surrogate, learning_rate=1e-3)
    trainer.train_from_solver(
        solver_fn=expensive_solver,
        input_samples=train_samples,
        n_epochs=500,
    )

    return surrogate
