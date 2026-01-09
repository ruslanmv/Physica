# Project Physica: Teaching AI to Understand Gravity
*LLMs can write poetry — but they can't design a bridge (yet).*

> Ask a modern LLM to "simulate a pendulum."
> It writes code that **looks** correct.
> Run it long enough… and the pendulum starts gaining (or losing) energy from nowhere.
> The bob drifts. The orbit precesses. Reality breaks.

That failure isn't just "a bug in the code."
It's a **category error**.

Language models are trained to predict *tokens*, not to obey *laws*. They can recall Newton's laws, but they don't naturally **live inside** them.

**Project Physica** is an attempt to change that: a *Physics World Model* where outputs are constrained by physical structure, not by what "sounds right."

---

## The Observation: Token Fluency ≠ Physical Truth

A simple pendulum is governed by:

$$
\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin(\theta) = 0
$$

If there is no friction, total energy should be conserved:

$$
E(t) = \frac{1}{2}m(L\dot{\theta})^2 + mgL(1-\cos\theta) \quad,\quad \frac{dE}{dt}=0
$$

A classic "LLM-generated simulation" failure mode is **energy drift**: the system slowly gains or loses energy due to numerical mistakes, unstable integrators, hidden unit mismatches, or step-size issues.

Here are the kinds of questions a physics-aware agent should ask *before* trusting results:

$$
\boxed{\text{Is energy conserved when the model says it should be?}}
$$

$$
\boxed{\text{Do units balance? i.e., do we ever add meters to seconds?}}
$$

$$
\boxed{\text{What invariants are implied by the equations, and are they preserved numerically?}}
$$

This is the gap Physica is targeting: **from "looks plausible" to "must be true."**

---

## The Grand Unification: Why Physics Beats Prompts

As a physicist-engineer, you learn a harsh truth early:

> **Reality is governed by differential equations, not tokens.**

Nature does not autocomplete.
It constrains.

Many real-world systems are not "best-effort predictions." They are **structures**:

- conservation laws (energy, momentum, charge)
- symmetries (Noether)
- stability conditions
- boundary conditions
- causality

When you build software that touches the physical world, you stop asking "does this compile?" and start asking:

$$
\boxed{\text{Does this violate a conservation law?}}
$$

Physica's thesis is that "world-model AI" should be trained and evaluated in that spirit.

---

## The Methodology (Plain English): PINNs and the "Neuro-Physical Loop"

Physica includes Physics-Informed Neural Network (PINN) components **and** a broader "Neuro-Physical Loop."

### What a PINN is (without the hype)
A PINN is a neural network trained not only to match data, but also to minimize equation residuals. If a system is governed by:

$$
\mathcal{N}[u](x,t)=0
$$

then the model is trained to keep:

$$
\|\mathcal{N}[u_\theta](x,t)\|
$$

small across the domain, in addition to fitting any measured points.

In the simplest Newton form:

$$
\mathbf{F} = m\mathbf{a}
$$

Physica can treat violations as *loss*, not as "oops."

### What the "Neuro-Physical Loop" means in Physica
From the codebase, Physica's core loop is explicitly framed as:

**Natural language → formal physics parameters → simulation/checks → correction → validated output**

In other words:
- parse intent
- map it into a physical task
- run a physically grounded solver or surrogate
- validate against constraints
- repair if violated
- return something that is *physically coherent*

This is aiming at a different kind of "trust": not "the model sounds confident," but "the model can't output non-physical states without being penalized/corrected."

---

## What's Actually in the Repo

The project contains three main tracks:

1. **Python package (`physica`)**
   - Core loop (`NeuroPhysicalLoop`) and agent orchestration
   - Conservation law checks (`src/physica/conservation`)
   - Domain modules (mechanics, electromagnetism, thermo, Hamiltonian/Lagrangian)
   - PINN framework (`src/physica/pinn`)
   - Phase III production systems (autonomous control, semiconductor twins, surrogates)
   - Multiple demos in `examples/`

2. **WebXR demo app (`apps/webxr-demos`)**
   - A local Three.js + WebXR interactive sandbox
   - A minimal physics step loop with projectile spawning
   - A simple "digital twin" thermal tile (heat diffusion grid)
   - Designed to be readable and swappable with a real rigid-body engine later

3. **Documentation (`docs/`)**
   - This tutorial
   - Installation guides
   - Usage examples

This matters because it shows Physica is not only "math on paper."
It is built to be **experienced**: a world you can interact with.

---

# Installation & Running Physica (Complete, Practical)

Below are install/run steps that match the repository structure (Python + optional WebXR).

## Prerequisites

### For Python (core Physica)
- **Python 3.9+** (3.11 recommended)
- `pip` (recommended: newest pip)
- Virtual environment (venv/conda) - **highly recommended**

### For WebXR demos (3D)
- **Node.js 18+** (Node 20+ is also fine)
- `npm` (comes with Node) or `pnpm`/`yarn`

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/ruslanmv/Physica.git
cd Physica
```

**Verify you're in the correct directory:**

```bash
ls -la
# You should see: README.md, pyproject.toml, src/, examples/, apps/, docs/
```

---

## Step 2: Create a Python Virtual Environment (Strongly Recommended)

### Option A: Using `venv` (Built-in, works everywhere)

```bash
python3 -m venv .venv
```

**Activate it:**

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**Upgrade pip (important):**
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Option B: Using `conda`

```bash
conda create -n physica python=3.11 -y
conda activate physica
python -m pip install --upgrade pip
```

**Verify your environment:**

```bash
which python  # Should show path to .venv or conda env
python --version  # Should be 3.9+
```

---

## Step 3: Install Physica Core Package

### Basic Installation (Editable Mode)

```bash
pip install -e .
```

This installs:
- `numpy`, `scipy`, `torch`, `matplotlib`, `pydantic`, `tqdm`
- Makes `physica` importable
- Allows you to edit code and see changes immediately

**Verify installation:**

```bash
python -c "import physica; print(f'✓ Physica version: {physica.__version__}')"
```

Expected output:
```
✓ Physica version: 1.0.0
```

### Developer Installation (With Testing Tools)

If you plan to contribute or run tests:

```bash
pip install -e ".[dev]"
```

This additionally installs:
- `pytest`, `pytest-cov`
- `ruff` (linter)
- `mypy` (type checker)

**Or use the Makefile:**

```bash
make install
```

---

## Step 4: Optional Extras (Advanced Features)

### JAX Backend (Accelerated/Differentiable Components)

For GPU-accelerated physics and automatic differentiation:

```bash
pip install -e ".[jax]"
```

### LLM Integration (Multi-Provider Support)

For OpenAI, Claude, Watsonx, Ollama backends:

```bash
pip install -e ".[llm]"
```

This installs:
- `openai`
- `anthropic`
- `ibm-watsonx-ai`
- Additional LLM client libraries

### Visualization Tools

```bash
pip install -e ".[viz]"
```

### CrewAI Integration

For multi-agent workflows:

```bash
pip install -e ".[crewai]"
```

### Install Everything

```bash
pip install -e ".[dev,llm,viz,crewai,jax]"
```

---

## Step 5: Run the Main Demo

### Using the Demo Script

```bash
python demo.py
```

**Expected Output:**

```
================================================================================
PROJECT PHYSICA: THE PHYSICS WORLD MODEL FOR AGENTIC AI
================================================================================

A neuro-physical AI system that unifies:
  🧠 Cognitive Layer (LLM): Intent → Parameters
  ⚛️  Physical Layer (Simulator): Validates against natural laws
  🎓 Learning Layer (PINN): Corrects via physics gradients

This creates AI that computes reality, not hallucinate it.

User Request:

    I need to hit a target 300 meters away with a projectile.
    What launch velocity should I use if I launch at 45 degrees?
    Make sure the solution respects conservation of energy and momentum.

Executing Neuro-Physical Loop...
────────────────────────────────────────────────────────────────────────────────

[Iteration logs showing convergence...]

================================================================================
RESULTS
================================================================================

✅ Solution found!
   Required velocity: 54.25 m/s
   Achieved distance: 300.00 m
   Error: 0.02 m
   Converged in 3 iterations
```

### Using the Makefile

```bash
make run
```

---

## Step 6: Run Example Demonstrations

The `examples/` directory contains comprehensive demos for all phases:

### Phase I: Foundational Mechanics

```bash
# Neuro-physical loop demonstration
python examples/neuro_physical_demo.py

# Conservation law validation
python examples/conservation_validation_demo.py

# PINN training example
python examples/pinn_training_demo.py

# Autonomous physicist agent
python examples/autonomous_physicist_demo.py

# Multi-provider LLM support
python examples/multi_provider_demo.py

# CrewAI integration
python examples/crewai_integration_demo.py
```

### Phase II: Energy & Fields

```bash
# Electromagnetism (Maxwell's equations, Lorentz force)
python examples/phase_ii_electromagnetism_demo.py

# Hamiltonian & Lagrangian mechanics
python examples/phase_ii_hamiltonian_lagrangian_demo.py

# Thermodynamics (Carnot cycles, entropy)
python examples/phase_ii_thermodynamics_demo.py
```

### Phase III: Production Applications

```bash
# Physics-constrained autonomous control
python examples/phase_iiia_autonomous_control_demo.py

# Semiconductor thermal & power digital twin
python examples/phase_iiib_semiconductor_twin_demo.py

# PINN surrogates for browser deployment
python examples/phase_iiic_surrogate_models_demo.py
```

### What to Expect from These Demos

Each demo is self-contained and will:
1. Print a banner explaining the demonstration
2. Show the physics being simulated
3. Display validation checks (conservation laws, constraints)
4. Produce visualizations (saved as PNG files)
5. Print results and statistics

**Example output from electromagnetism demo:**

```
================================================================================
PHASE II: ELECTROMAGNETISM DEMONSTRATIONS
================================================================================

Demo 1: Lorentz Force
────────────────────────────────────────────────────────────────────────────────
Particle: electron
Velocity: [1000000.0, 0.0, 0.0] m/s
Electric field: [0.0, 1000.0, 0.0] V/m
Magnetic field: [0.0, 0.0, 0.1] T
Lorentz force: [-1.602e-14, 1.602e-16, 0.0] N
Force magnitude: 1.60e-14 N

[... additional demos ...]

✓ All electromagnetic demonstrations completed!
Saved plots:
  - electromagnetism_crossed_fields.png
```

---

## Step 7: Configure LLM Providers (Optional)

Physica supports multiple LLM providers via environment variables.

### Mock Provider (Default - No API Key Required)

By default, Physica uses a mock provider, so you can run all demos without API keys:

```bash
# No configuration needed - works out of the box!
python examples/multi_provider_demo.py
```

### OpenAI Configuration

```bash
export OPENAI_API_KEY="sk-..."
export PHYSICA_PROVIDER="openai"
export PHYSICA_OPENAI_MODEL="gpt-4o-mini"

python examples/multi_provider_demo.py
```

### Claude (Anthropic) Configuration

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export PHYSICA_PROVIDER="claude"
export PHYSICA_CLAUDE_MODEL="claude-sonnet-4-5"

python examples/multi_provider_demo.py
```

### Watsonx (IBM) Configuration

```bash
export WATSONX_API_KEY="..."
export WATSONX_PROJECT_ID="..."
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export PHYSICA_PROVIDER="watsonx"
export PHYSICA_WATSONX_MODEL="meta-llama/llama-3-1-70b-instruct"

python examples/multi_provider_demo.py
```

### Ollama (Local Models) Configuration

```bash
# Ensure Ollama is running locally
# Install from: https://ollama.ai

export PHYSICA_PROVIDER="ollama"
export PHYSICA_OLLAMA_MODEL="llama2"

python examples/multi_provider_demo.py
```

### Persistent Configuration

Create a settings file at `~/.physica/settings.json`:

```json
{
  "provider": "openai",
  "openai_api_key": "sk-...",
  "openai_model": "gpt-4o-mini"
}
```

---

## Step 8: Run Tests (Sanity Check)

### Using pytest

```bash
pytest
```

### Using the Makefile

```bash
make test
```

### Expected Output

```
========================= test session starts ==========================
collected 15 items

tests/test_engine.py .....                                       [ 33%]
tests/test_conservation.py .....                                 [ 66%]
tests/test_pinn.py .....                                         [100%]

========================= 15 passed in 2.34s ===========================
```

---

## Step 9: Run Linting and Type Checks

### Check code style

```bash
ruff check .
```

### Run type checker

```bash
mypy src
```

### Using the Makefile

```bash
make lint
make typecheck
```

---

# Running the WebXR 3D Demo (Interactive Physics Visualization)

Physica includes a separate, additive 3D visualization app in `apps/webxr-demos/`.

## Features

- **Desktop Mode**: Orbit controls + click to spawn physics spheres
- **VR Mode**: WebXR support for VR headsets (Quest, Vive, etc.)
- **Real-time Physics**: Gravity, collisions, bouncing
- **Digital Twin Tile**: Live heat diffusion visualization
- **Multi-Physics Coupling**: Interactions affect both physics domains

## Installation and Launch

### Step 1: Navigate to WebXR Directory

```bash
cd apps/webxr-demos
```

### Step 2: Install Dependencies

```bash
npm install
```

This installs:
- `three` (Three.js for 3D rendering)
- `vite` (dev server and build tool)

### Step 3: Start Development Server

```bash
npm run dev
```

**Expected Output:**

```
  VITE v5.4.0  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Step 4: Open in Browser

Open [http://localhost:5173](http://localhost:5173)

## Desktop Controls

- **Left mouse drag**: Rotate camera (orbit)
- **Right mouse drag**: Pan camera
- **Scroll wheel**: Zoom in/out
- **Click anywhere**: Shoot a physics sphere
- **HUD**: Shows FPS, body count, XR status

## VR Controls

1. **Click "ENTER VR" button** (appears if WebXR is available)
2. **Put on your VR headset**
3. **Controller triggers**: Shoot spheres
4. **White rays**: Visual indicators of controller direction

## What You'll See

- **Blue physics spheres**: Spawn with gravity, bounce on floor
- **Thermal tile**: Color-coded heatmap (blue→cyan→yellow→red)
- **Grid floor**: 24×24 meter environment
- **Real-time stats**: FPS counter, body count

## Building for Production

```bash
npm run build
```

Output in `dist/` folder can be deployed to any static host.

---

# The Vision: Scientific AI (Not Just Chatbots)

Imagine an agent that can design a system, simulate it, detect non-physical behavior, and refine the design before a human checks the math.

That's the trajectory Physica points toward.

This is not "AI that talks about physics."
It is "AI that is *forced to respect* physics."

Here are the questions that define the roadmap:

$$
\boxed{\text{Can we make hallucinations impossible by constraining outputs to physical manifolds?}}
$$

$$
\boxed{\text{Can an AI be trained to minimize equation residuals as aggressively as it minimizes token loss?}}
$$

$$
\boxed{\text{Can we build an agent that designs experiments, then rejects its own results when invariants break?}}
$$

If the answer is "yes," the implications are enormous:

- safer robotics
- trustworthy industrial automation
- faster scientific iteration
- "design in simulation first" workflows that actually hold up

---

# The Reveal: Embodied Physics (Why WebXR Matters)

Most "AI physics" demos stop at plots.

Physica goes one step further: an environment you can inhabit.

The moment you see:

- a naïve agent throw a ball that flies forever
  versus
- a constrained model that produces a parabola

…you understand the difference instantly.

Physics becomes a *constraint surface* the agent must live on, not a paragraph it can paraphrase.

---

# Why This Is a Moat

LLMs scale with data.

Physica's direction scales with **truth**:

- conservation laws
- symmetries
- invariants
- stable integration
- constraint checking and correction

Anyone can scrape the web.
Not everyone can encode **reality** as a first-class primitive.

---

# Project Structure

```
Physica/
├── src/physica/              # Core Python package
│   ├── __init__.py          # Main exports
│   ├── engine.py            # Ballistic simulator
│   ├── agent.py             # Agent framework
│   ├── neuro_physical_loop.py  # Core innovation
│   ├── conservation/        # Conservation law validators
│   ├── cognitive/           # LLM integration layer
│   ├── pinn/               # Physics-Informed Neural Networks
│   ├── domains/            # Physics domains (EM, thermo, etc.)
│   ├── autonomous_control.py   # Phase III-A
│   ├── semiconductor_twin.py   # Phase III-B
│   └── surrogate_models.py     # Phase III-C
├── examples/               # Demonstration scripts
│   ├── neuro_physical_demo.py
│   ├── phase_ii_*.py
│   ├── phase_iiia_*.py
│   ├── phase_iiib_*.py
│   └── phase_iiic_*.py
├── apps/webxr-demos/      # 3D interactive visualization
│   ├── src/
│   │   ├── main.js        # Main application loop
│   │   ├── physics/       # Physics simulation
│   │   └── render/        # Three.js rendering
│   └── package.json
├── docs/                  # Documentation
│   └── tutorial.md        # This file
├── tests/                 # Test suite
├── pyproject.toml         # Python project config
├── demo.py               # Main demo entry point
└── README.md             # Project overview
```

---

# Quick Troubleshooting

## Import Errors (`physica` not found)

**Problem:**
```python
ModuleNotFoundError: No module named 'physica'
```

**Solution:**
1. Ensure you're in the repo root: `cd Physica`
2. Activate your virtual environment
3. Install in editable mode: `pip install -e .`
4. Verify: `python -c "import physica; print(physica.__version__)"`

## Dependency Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution:**
1. Use a fresh virtual environment
2. Upgrade pip first: `python -m pip install --upgrade pip`
3. Install again: `pip install -e .`

## LLM API Errors (Missing Keys)

**Problem:**
```
Error: OpenAI API key not found
```

**Solution:**
- Use mock provider (default): No configuration needed
- Or set environment variables:
  ```bash
  export OPENAI_API_KEY="sk-..."
  export PHYSICA_PROVIDER="openai"
  ```

## WebXR Not Working in VR Headset

**Problem:**
VR mode doesn't activate

**Solution:**
1. Use a WebXR-enabled browser (Chrome, Firefox, Edge)
2. For remote deployment, use HTTPS (WebXR requires secure context)
3. Desktop mode should always work

## Slow Installation

**Problem:**
`pip install -e .` takes a long time

**Solution:**
- PyTorch installation is large (~2GB)
- Use `--no-cache-dir` if needed: `pip install -e . --no-cache-dir`
- Or install without PyTorch extras initially

## Plots Not Showing

**Problem:**
Example demos run but no PNG files created

**Solution:**
- Check current directory: `pwd`
- Plots are saved in the same directory where you run the script
- Look for `*.png` files: `ls -la *.png`

---

# Advanced Usage

## Custom Physics Simulations

```python
from physica import BallisticSimulator

# Create simulator
sim = BallisticSimulator()

# Run simulation
result = sim.simulate(
    initial_velocity=50.0,  # m/s
    angle=45.0,            # degrees
    height=10.0            # m
)

print(f"Impact distance: {result.impact_distance:.2f} m")
print(f"Flight time: {result.flight_time:.2f} s")
print(f"Max height: {result.max_height:.2f} m")
```

## Conservation Law Validation

```python
from physica import EnergyConservation, ConservationValidator

# Create validator
validator = ConservationValidator([
    EnergyConservation(tolerance=1e-6)
])

# Check trajectory
is_valid, violations = validator.validate_trajectory(
    positions=positions,
    velocities=velocities,
    times=times,
    mass=1.0
)

if not is_valid:
    print(f"Conservation violations detected: {violations}")
```

## Training a PINN

```python
from physica import MechanicsPINN, PINNTrainer
import torch

# Create PINN
pinn = MechanicsPINN(
    input_dim=2,   # (x, t)
    output_dim=2,  # (x_pred, v_pred)
    hidden_layers=[64, 64, 64]
)

# Create trainer
trainer = PINNTrainer(pinn, learning_rate=1e-3)

# Define physics bounds
physics_bounds = torch.tensor([
    [0.0, 0.0],  # min (x, t)
    [100.0, 10.0]  # max (x, t)
])

# Train
history = trainer.train(
    physics_bounds=physics_bounds,
    epochs=1000,
    physics_weight=0.1
)
```

---

# Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-physics`
3. **Make your changes**
4. **Run tests**: `make test`
5. **Run linters**: `make lint`
6. **Commit**: `git commit -m "feat: Add amazing physics feature"`
7. **Push**: `git push origin feature/amazing-physics`
8. **Create Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Physica.git
cd Physica

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check linting
ruff check .

# Check types
mypy src
```

---

# Call to Action

If you're tired of "looks right" code that drifts over time—
and you want AI that can't violate the laws of the universe without being caught—

**Fork the repo. Run the demos. Break it. Improve it.**

Welcome to the era of Scientific AI.

---

## Resources

- **Repository**: https://github.com/ruslanmv/Physica
- **Issues**: https://github.com/ruslanmv/Physica/issues
- **Discussions**: https://github.com/ruslanmv/Physica/discussions

---

*LLMs can write poetry.*
**Physica is teaching them to respect gravity.**

---

## License

Apache License 2.0 - See [LICENSE](../LICENSE) for details.
