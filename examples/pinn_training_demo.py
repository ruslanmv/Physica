"""Demo: Training a Physics-Informed Neural Network.

Shows how PINNs learn to solve physics problems by respecting
differential equations and conservation laws.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from physica.pinn import MechanicsPINN, PINNConfig, PINNTrainer


def main():
    print("=" * 70)
    print("PHYSICS-INFORMED NEURAL NETWORK (PINN) DEMONSTRATION")
    print("=" * 70)
    print()
    print("Training a neural network to solve projectile motion")
    print("by enforcing Newton's second law: F = ma")
    print()

    # Configure PINN
    config = PINNConfig(
        hidden_layers=[32, 32, 32],
        activation="tanh",
        learning_rate=1e-3,
        physics_weight=1.0,
        data_weight=0.1,
    )

    # Create mechanics PINN for 2D projectile motion
    pinn = MechanicsPINN(
        spatial_dim=2,
        config=config,
        gravity=9.81,
        mass=1.0,
    )

    print(f"✓ Created PINN with {sum(p.numel() for p in pinn.parameters())} parameters")

    # Generate some training data (optional - PINN can learn from physics alone!)
    t = np.linspace(0, 5, 50)
    v0, angle = 50.0, 45.0
    v0x = v0 * np.cos(np.deg2rad(angle))
    v0y = v0 * np.sin(np.deg2rad(angle))

    # Ideal trajectory (no drag)
    x_true = v0x * t
    y_true = v0y * t - 0.5 * 9.81 * t ** 2

    # Prepare training data
    X_data = torch.tensor(
        np.column_stack([t, x_true, y_true]),
        dtype=torch.float32,
    )
    y_data = torch.tensor(
        np.column_stack([x_true, y_true]),
        dtype=torch.float32,
    )

    # Setup trainer
    trainer = PINNTrainer(
        model=pinn,
        config=None,  # Use defaults
    )

    print("\n🎓 Training PINN...")
    print("   (This enforces physics laws, not just data fitting)")

    # Train
    # Define physics domain bounds: [t_min, x_min, y_min] to [t_max, x_max, y_max]
    physics_bounds = np.array([
        [0.0, 10.0],    # time
        [0.0, 300.0],   # x
        [0.0, 150.0],   # y
    ])

    history = trainer.train(
        physics_bounds=physics_bounds,
        data_x=X_data,
        data_y=y_data,
    )

    print("\n✓ Training complete!")

    # Test predictions
    print("\n📊 Testing predictions...")

    t_test = np.linspace(0, 5, 100)
    x_test_input = torch.tensor(
        np.column_stack([t_test, np.zeros_like(t_test), np.zeros_like(t_test)]),
        dtype=torch.float32,
    )

    predictions = trainer.predict(x_test_input)

    print(f"   Predicted trajectory shape: {predictions.shape}")
    print(f"   Final physics loss: {history['physics'][-1]:.6f}")
    if history['data']:
        print(f"   Final data loss: {history['data'][-1]:.6f}")

    # Visualize (if matplotlib available)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curves
        ax1.semilogy(history['total'], label='Total Loss')
        ax1.semilogy(history['physics'], label='Physics Loss')
        if history['data']:
            ax1.semilogy(history['data'], label='Data Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('PINN Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Trajectory
        ax2.plot(x_true, y_true, 'b--', label='True Trajectory', linewidth=2)
        ax2.plot(predictions[:, 0], predictions[:, 1], 'r-', label='PINN Prediction', linewidth=2)
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Learned Trajectory (Physics-Constrained)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pinn_demo_results.png', dpi=150)
        print("\n📈 Plots saved to 'pinn_demo_results.png'")

    except Exception as e:
        print(f"\n⚠ Visualization skipped: {e}")

    print("\n✅ Demo complete!")
    print("\nKey insights:")
    print("  • PINN learns by enforcing F = ma, not just fitting data")
    print("  • Physics loss ensures predictions respect natural laws")
    print("  • Can learn from minimal or no data (physics-only training)")
    print("  • Provides physically meaningful predictions beyond training data")


if __name__ == "__main__":
    main()
