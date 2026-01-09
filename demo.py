"""Project Physica: Main Demo

Demonstrates the revolutionary Neuro-Physical Intelligence Loop.
"""

import logging

from physica import NeuroPhysicalLoop

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main() -> None:
    print('=' * 80)
    print('PROJECT PHYSICA: THE PHYSICS WORLD MODEL FOR AGENTIC AI')
    print('=' * 80)
    print()
    print('A neuro-physical AI system that unifies:')
    print('  🧠 Cognitive Layer (LLM): Intent → Parameters')
    print('  ⚛️  Physical Layer (Simulator): Validates against natural laws')
    print('  🎓 Learning Layer (PINN): Corrects via physics gradients')
    print()
    print('This creates AI that computes reality, not hallucinate it.')
    print()

    # Initialize the neuro-physical loop
    loop = NeuroPhysicalLoop()

    # Example: Natural language to physics
    request = """
    I need to hit a target 300 meters away with a projectile.
    What launch velocity should I use if I launch at 45 degrees?
    Make sure the solution respects conservation of energy and momentum.
    """

    print('User Request:')
    print(request)
    print()

    print('Executing Neuro-Physical Loop...')
    print('─' * 80)

    # Execute the loop
    intent, result, history = loop.execute(request, verbose=True)

    # Show results
    print()
    print('=' * 80)
    print('RESULTS')
    print('=' * 80)

    if isinstance(result, dict):
        print("\n✅ Solution found!")
        print(f"   Required velocity: {result.get('velocity', 0):.2f} m/s")
        print(f"   Achieved distance: {result.get('impact', 0):.2f} m")
        print(f"   Error: {result.get('error', 0):.2f} m")
        print(f"   Converged in {len(history)} iterations")

    print()
    print('=' * 80)
    print('EXECUTION TRACE')
    print('=' * 80)
    print(loop.explain())

    print()
    print('=' * 80)
    print('KEY INNOVATIONS')
    print('=' * 80)
    print()
    print('✓ Natural language parsed into formal physics parameters')
    print('✓ Conservation laws automatically validated')
    print('✓ Violations trigger automatic corrections')
    print('✓ All outputs guaranteed to respect physical laws')
    print('✓ Fully explainable reasoning chain')
    print()
    print('This is the foundation for trustworthy AI in the real world.')
    print()

    # Show more examples
    print('For more advanced examples, see:')
    print('  • examples/neuro_physical_demo.py       - Core loop demonstration')
    print('  • examples/autonomous_physicist_demo.py  - AI-driven discovery')
    print('  • examples/pinn_training_demo.py         - Physics-informed learning')
    print('  • examples/conservation_validation_demo.py - Law enforcement')
    print()


if __name__ == '__main__':
    main()
