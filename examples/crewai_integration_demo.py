"""Demo: CrewAI Integration with Physica.

Shows how to use Physica's physics capabilities with CrewAI agents for
advanced agentic workflows.

Note: This demo requires CrewAI to be installed:
  pip install 'physica[crewai]'
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    print("=" * 80)
    print("CREWAI INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print()

    # Check if CrewAI is available
    try:
        from crewai import Agent, Crew, Process, Task

        from physica.cognitive import build_crewai_llm, get_settings
    except ImportError as e:
        print(f"❌ CrewAI not installed: {e}")
        print()
        print("To run this demo, install CrewAI:")
        print("  pip install 'physica[crewai]'")
        print()
        print("And configure a provider (mock LLM won't work with CrewAI):")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export PHYSICA_PROVIDER='openai'")
        return

    print("✓ CrewAI installed")
    print()

    # Show settings
    settings = get_settings()
    print(f"Active Provider: {settings.provider.value}")

    if settings.provider.value == "mock":
        print()
        print("⚠️  Mock LLM cannot be used with CrewAI.")
        print("Please configure a real provider:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export PHYSICA_PROVIDER='openai'")
        return

    print()

    # Create CrewAI LLM
    print("Creating CrewAI LLM from Physica settings...")
    try:
        llm = build_crewai_llm()
        print(f"✓ Created CrewAI LLM: {llm.model}")
    except Exception as e:
        print(f"❌ Failed to create LLM: {e}")
        print()
        print("Make sure you've configured API keys for your provider.")
        return

    print()

    # Create physics-focused agents
    print("=" * 80)
    print("CREATING PHYSICS AGENTS")
    print("=" * 80)
    print()

    physicist = Agent(
        role="Expert Physicist",
        goal="Analyze physical systems and validate calculations against natural laws",
        backstory=(
            "You are a world-class physicist with deep expertise in classical mechanics, "
            "thermodynamics, and conservation laws. You ensure all predictions respect "
            "fundamental physics principles."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    print("✓ Created Physicist agent")

    engineer = Agent(
        role="Physics Engineer",
        goal="Design practical solutions constrained by physical laws",
        backstory=(
            "You are an experienced engineer who designs systems that are not just "
            "theoretically sound, but also practically feasible. You work with physicists "
            "to ensure designs respect conservation laws and real-world constraints."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    print("✓ Created Engineer agent")
    print()

    # Create physics task
    print("=" * 80)
    print("DEFINING PHYSICS TASK")
    print("=" * 80)
    print()

    task = Task(
        description="""
        Analyze a projectile motion problem:

        A projectile is launched at 50 m/s at a 45-degree angle.
        Assume no air resistance (ideal conditions).

        Your analysis must include:
        1. Calculate the maximum height reached
        2. Calculate the total range (horizontal distance)
        3. Verify energy conservation (initial KE = PE at peak + KE at peak)
        4. Verify the trajectory satisfies Newton's laws

        Use these physics principles:
        - Initial kinetic energy: KE = (1/2)mv²
        - Potential energy: PE = mgh
        - Range formula: R = v²sin(2θ)/g
        - Max height: H = v²sin²(θ)/(2g)
        - g = 9.81 m/s²

        Provide detailed calculations showing all conservation laws are satisfied.
        """,
        expected_output=(
            "A comprehensive physics analysis including:\n"
            "- Maximum height calculation\n"
            "- Range calculation\n"
            "- Energy conservation verification\n"
            "- Confirmation that all values respect physical laws"
        ),
        agent=physicist,
    )

    print("✓ Defined physics analysis task")
    print()

    # Create crew
    print("=" * 80)
    print("ASSEMBLING CREW")
    print("=" * 80)
    print()

    _crew = Crew(
        agents=[physicist, engineer],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    print("✓ Crew assembled")
    print()

    # Execute (this would actually run if configured properly)
    print("=" * 80)
    print("EXECUTION")
    print("=" * 80)
    print()

    print("To execute this crew:")
    print("  result = _crew.kickoff()")
    print()
    print("This would:")
    print("  1. Physicist analyzes the projectile problem")
    print("  2. Calculates all physics quantities")
    print("  3. Verifies conservation laws")
    print("  4. Returns physics-validated results")
    print()

    # Show CrewAI advantages
    print("=" * 80)
    print("CREWAI + PHYSICA ADVANTAGES")
    print("=" * 80)
    print()

    print("✓ Multi-agent collaboration on physics problems")
    print("✓ Task decomposition and specialization")
    print("✓ Physics constraints enforced throughout")
    print("✓ Explainable agent reasoning")
    print("✓ Support for all major LLM providers")
    print("✓ Production-ready agentic workflows")
    print()

    print("=" * 80)
    print("EXAMPLE USE CASES")
    print("=" * 80)
    print()

    print("1. Autonomous Physics Research")
    print("   • AI generates hypotheses")
    print("   • Simulates experiments")
    print("   • Validates against conservation laws")
    print()

    print("2. Engineering Design Optimization")
    print("   • AI proposes designs")
    print("   • Physics agent validates feasibility")
    print("   • Iterates until physically optimal")
    print()

    print("3. Educational Tutoring")
    print("   • AI explains physics concepts")
    print("   • Works through problems step-by-step")
    print("   • Checks student work for physical correctness")
    print()

    print("=" * 80)
    print("✅ Demo complete!")
    print()


if __name__ == "__main__":
    main()
