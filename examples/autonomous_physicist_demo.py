"""Demo: Autonomous Physicist - AI-driven Scientific Discovery.

Shows how an AI agent can autonomously generate and test hypotheses
about physical systems.
"""

import logging

from physica import AutonomousPhysicist

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("AUTONOMOUS PHYSICIST DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows an AI agent autonomously exploring physics,")
    print("generating hypotheses, and learning from experiments.")
    print()

    # Create autonomous physicist
    physicist = AutonomousPhysicist()

    # Research question
    research_question = """
    Investigate how launch angle affects projectile range.
    Assume launch velocity of 50 m/s with realistic air drag.
    """

    print("Research Question:")
    print(research_question)
    print()

    # Run autonomous exploration
    print("🔬 Starting autonomous exploration...")
    print()

    results = physicist.explore(
        research_question=research_question,
        n_experiments=3,
    )

    # Display results
    print("\n" + "=" * 70)
    print("EXPERIMENTAL RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\nExperiment {i}:")
        print(f"  Hypothesis: {result.hypothesis.statement}")
        print(f"  Parameters: {result.hypothesis.parameters}")
        print(f"  Expected: {result.hypothesis.expected_outcome}")
        print(f"  Status: {'✓ CONFIRMED' if result.is_confirmed else '✗ REJECTED'}")
        if result.actual_outcome:
            print(f"  Observed: {result.actual_outcome}")

    # Summary
    print("\n" + "=" * 70)
    print(physicist.summarize_findings())

    print("\n✅ Autonomous exploration complete!")
    print("\nKey capabilities demonstrated:")
    print("  • Autonomous hypothesis generation")
    print("  • Automated experimental design")
    print("  • Learning from experimental outcomes")
    print("  • Building scientific knowledge base")
    print("  • Self-correcting based on physics constraints")


if __name__ == "__main__":
    main()
