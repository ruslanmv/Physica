"""Demo: Neuro-Physical Loop in Action.

This demonstrates the core innovation of Project Physica:
Natural language → Physics simulation → Automatic correction
"""

import logging

from physica import NeuroPhysicalLoop

# Setup logging to see the loop in action
logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("NEURO-PHYSICAL LOOP DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows how Physica translates natural language into")
    print("physically valid simulations with automatic correction.")
    print()

    # Initialize the neuro-physical loop
    loop = NeuroPhysicalLoop()

    # Example 1: Simple projectile simulation
    print("\n" + "─" * 70)
    print("EXAMPLE 1: Basic Projectile Simulation")
    print("─" * 70)

    request1 = """
    Simulate a projectile launched at 50 m/s at a 45-degree angle.
    Include gravity and air drag.
    """

    intent, result, history = loop.execute(request1, verbose=True)

    print("\n📊 Results:")
    if hasattr(result, 'impact') and result.impact:
        print(f"   Impact distance: {result.impact['x']:.2f} m")
        print(f"   Flight time: {result.impact['t']:.2f} s")
    print(f"   Iterations needed: {len(history)}")

    # Example 2: Target optimization
    print("\n" + "─" * 70)
    print("EXAMPLE 2: Optimize to Hit Target")
    print("─" * 70)

    request2 = """
    Find the launch velocity needed to hit a target 300 meters away.
    Use a 45-degree launch angle.
    """

    intent, result, history = loop.execute(request2, verbose=True)

    print("\n📊 Results:")
    if isinstance(result, dict):
        print(f"   Required velocity: {result.get('velocity', 0):.2f} m/s")
        print(f"   Final impact: {result.get('impact', 0):.2f} m")
        print(f"   Error: {result.get('error', 0):.2f} m")
    print(f"   Iterations needed: {len(history)}")

    # Show explanation
    print("\n" + "=" * 70)
    print("EXECUTION TRACE")
    print("=" * 70)
    print(loop.explain())

    print("\n✅ Demo complete!")
    print("\nKey insights:")
    print("  • Natural language was parsed into formal physics parameters")
    print("  • Conservation laws were automatically validated")
    print("  • Any violations would trigger automatic corrections")
    print("  • All outputs are guaranteed to respect physical laws")


if __name__ == "__main__":
    main()
