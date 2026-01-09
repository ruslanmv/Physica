# 🌌 Physica

Physica is a portable ballistic physics simulator (gravity + quadratic drag) with a small optimization layer for
hitting target distances.

## Highlights

- **Structured simulation results**: time grid, state history, solver metadata, and ground-impact details.
- **Accurate ground impact** on the NumPy/SciPy backend via solver events.
- **Optional JAX/Diffrax backend** for acceleration and differentiable workflows.
- Production-friendly project setup: dependency declaration, tests, lint/type-check configuration, and CI workflow.

---

## Install

Editable install:

```bash
pip install -e .
```

Optional JAX backend:

```bash
pip install -e ".[jax]"
```

Developer tooling:

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## Quickstart

### Simulate a trajectory

```python
from physica import BallisticSimulator

sim = BallisticSimulator(drag_coeff=0.05)
res = sim.simulate(v0=50.0, angle_deg=45.0, max_time=20.0, steps=1000)

print(res.backend)
print(res.y.shape)  # (N, 4) with columns [x, y, vx, vy]
print(res.impact)   # {"t": ..., "x": ..., "y": 0.0, "state": ...} or None
```

### Optimize to hit a target distance

```python
from physica import PhysicaAgent, TargetDistanceProblem

agent = PhysicaAgent()
problem = TargetDistanceProblem(target_distance=300.0, angle_deg=45.0, tolerance=2.0)

success, out = agent.solve_target_distance(problem)
print(success, out["velocity"], out["impact"], out["error"])
```

---

## Model

State is `[x, y, vx, vy]` with gravity and quadratic drag:

- `dx/dt = vx`
- `dy/dt = vy`
- `dv/dt = -c * |v| * v  + [0, -g]`

Where `g` is gravity and `c` is `drag_coeff`.

---

## Development

Run tests:

```bash
pytest
```

Lint and type-check:

```bash
ruff check .
mypy src
```

CI is provided in `.github/workflows/ci.yml`.
