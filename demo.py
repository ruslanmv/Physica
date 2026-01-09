from physica.agent import PhysicaAgent, TargetDistanceProblem


def main() -> None:
    print('--- Project Physica Demo ---')

    target = 300.0
    agent = PhysicaAgent()
    problem = TargetDistanceProblem(target_distance=target, tolerance=2.0)

    success, result = agent.solve_target_distance(problem)

    status = 'SUCCESS' if success else 'FAILED'
    print(f"{status}: target={target} m")
    print(f"Best v0={result['velocity']:.1f} m/s, impact={result['impact']:.2f} m, error={result['error']:.2f} m")


if __name__ == '__main__':
    main()
