# 🌌 Project Physica: The Physics World Model

**A Revolutionary Neuro-Physical AI System**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Artificial Intelligence has reached a critical limitation: it can generate eloquent language and reason symbolically, yet **fundamentally does not understand the physical world** it describes. Today's large language models predict *what sounds right*, not *what must be true*.

**Project Physica** solves this by introducing the world's first **Physics World Model for AI**—a system where generative intelligence is constrained, corrected, and optimized by the immutable laws of physics.

### The Core Innovation: The Neuro-Physical Loop

```
┌─────────────────────────┐
│  🧠 Cognitive Layer     │  LLM interprets intent
│     (Intent → Params)   │  and proposes solutions
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ⚛️  Physical Layer     │  Simulator validates
│     (Reality Engine)    │  against natural laws
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  🎓 Learning Layer      │  PINN corrects via
│     (Causal Correct)    │  physics gradients
└───────────┬─────────────┘
            │
            └──► Loop until valid
```

This creates AI systems that **compute reality, not hallucinate it**.

---

## Key Features

### 🧠 Cognitive Layer (LLM Integration)
- Natural language intent parsing
- Multi-step planning and decomposition
- Self-correcting based on physics feedback
- Explainable reasoning chains

### ⚛️ Physical Layer (Differentiable Simulators)
- Classical mechanics (projectile motion, collisions)
- Thermodynamics (heat transfer, energy conservation)
- Electromagnetism (fields, forces)
- Conservation law enforcement (energy, momentum, mass)

### 🎓 Learning Layer (Physics-Informed Neural Networks)
- Learns from differential equations, not just data
- Respects conservation laws by design
- Provides physically meaningful predictions
- Enables gradient-based optimization in physics space

### 🤖 Advanced Agentic AI
- **Autonomous Physicist**: Generates and tests hypotheses
- **Adaptive Optimization**: Learns from physics constraints
- **Self-Correction**: Violations trigger automatic refinement
- **Causal Reasoning**: Understands cause-effect relationships

---

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Features

```bash
# JAX backend (accelerated, differentiable physics)
pip install -e ".[jax]"

# LLM integration (OpenAI, Claude, Watsonx, Ollama)
pip install -e ".[llm]"

# CrewAI integration (agentic workflows)
pip install -e ".[crewai]"

# Visualization
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"
pre-commit install
```

### LLM Provider Configuration

Physica supports multiple LLM providers. Configure via environment variables:

```bash
# OpenAI
export OPENAI_API_KEY='sk-...'
export PHYSICA_PROVIDER='openai'
export PHYSICA_OPENAI_MODEL='gpt-4o-mini'

# Claude (Anthropic)
export ANTHROPIC_API_KEY='sk-ant-...'
export PHYSICA_PROVIDER='claude'
export PHYSICA_CLAUDE_MODEL='claude-sonnet-4-5'

# Watsonx (IBM)
export WATSONX_API_KEY='...'
export WATSONX_PROJECT_ID='...'
export PHYSICA_PROVIDER='watsonx'
export PHYSICA_WATSONX_MODEL='meta-llama/llama-3-3-70b-instruct'

# Ollama (local)
export OLLAMA_BASE_URL='http://localhost:11434'
export PHYSICA_PROVIDER='ollama'
export PHYSICA_OLLAMA_MODEL='llama3'

# Mock (no API required - for demos)
export PHYSICA_PROVIDER='mock'
```

---

## Quick Start

### 1. Neuro-Physical Loop (Natural Language → Validated Physics)

```python
from physica import NeuroPhysicalLoop

# Initialize the loop
loop = NeuroPhysicalLoop()

# Natural language request
request = """
Find the launch velocity to hit a target 300m away at 45 degrees.
Ensure energy and momentum conservation.
"""

# Execute (automatically validates and corrects)
intent, result, history = loop.execute(request, verbose=True)

print(f"Required velocity: {result['velocity']:.2f} m/s")
print(f"Error: {result['error']:.2f} m")
```

### 2. Physics-Informed Neural Networks

```python
from physica.pinn import MechanicsPINN, PINNTrainer
import numpy as np

# Create a PINN that learns F = ma
pinn = MechanicsPINN(spatial_dim=2, gravity=9.81)

# Train on physics alone (no data required!)
trainer = PINNTrainer(pinn)

physics_bounds = np.array([
    [0.0, 10.0],    # time
    [0.0, 300.0],   # x
    [0.0, 150.0],   # y
])

history = trainer.train(physics_bounds=physics_bounds)

# PINN now predicts trajectories that satisfy Newton's laws
```

### 3. Autonomous Physicist

```python
from physica import AutonomousPhysicist

# Create AI scientist
physicist = AutonomousPhysicist()

# Let it explore autonomously
results = physicist.explore(
    research_question="How does launch angle affect range?",
    n_experiments=5,
)

# It generates hypotheses, runs experiments, and learns!
print(physicist.summarize_findings())
```

### 4. Conservation Law Validation

```python
from physica import ConservationValidator, EnergyConservation

# Enforce fundamental laws
validator = ConservationValidator(laws=[
    EnergyConservation(tolerance=1e-3),
])

# Validate a trajectory
is_valid, violations = validator.validate_trajectory(
    trajectory, system_params
)

if not is_valid:
    print("Physics violations detected:")
    for v in violations:
        print(f"  - {v}")
```

---

## Architecture

### Three-Layer System

1. **Cognitive Layer** (`physica.cognitive`)
   - `IntentParser`: Natural language → Physics intent
   - `PlanningAgent`: Multi-step problem decomposition
   - `LLMBackend`: Claude, GPT, or mock for testing

2. **Physical Layer** (`physica.engine`, `physica.conservation`)
   - `BallisticSimulator`: Projectile motion with drag
   - `ConservationValidator`: Energy, momentum, mass laws
   - Extensible to more physics domains

3. **Learning Layer** (`physica.pinn`)
   - `PINN`: Base physics-informed network
   - `MechanicsPINN`: Classical mechanics
   - `ThermodynamicsPINN`: Heat transfer
   - `HamiltonianPINN`: Energy-preserving dynamics

4. **Integration** (`physica.neuro_physical_loop`)
   - `NeuroPhysicalLoop`: Orchestrates all three layers
   - Automatic validation and correction
   - Full explainability and tracing

5. **Agentic AI** (`physica.agentic`)
   - `AutonomousPhysicist`: Scientific discovery agent
   - `AdaptiveOptimizer`: Physics-constrained optimization

---

## Examples

Run comprehensive demos:

```bash
# Main demo
python demo.py

# Phase I examples
python examples/neuro_physical_demo.py
python examples/autonomous_physicist_demo.py
python examples/pinn_training_demo.py
python examples/conservation_validation_demo.py

# Phase II examples (Energy & Fields)
python examples/phase_ii_electromagnetism_demo.py
python examples/phase_ii_hamiltonian_lagrangian_demo.py
python examples/phase_ii_thermodynamics_demo.py

# Phase III examples (Production Applications)
python examples/phase_iiia_autonomous_control_demo.py
python examples/phase_iiib_semiconductor_twin_demo.py
python examples/phase_iiic_surrogate_models_demo.py

# Multi-provider LLM
python examples/multi_provider_demo.py
python examples/crewai_integration_demo.py
```

### WebXR 3D Demos (Interactive Physics Visualization)

Experience Physica's physics in 3D with WebXR support:

```bash
cd apps/webxr-demos
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to interact with:
- **Desktop mode**: Orbit camera, click to shoot physics spheres
- **VR mode**: "ENTER VR" button, use controller triggers
- **Real-time thermal digital twin**: Live heat diffusion visualization
- **Multi-physics coupling**: Interactions affect both physics domains

This demonstrates the full "digital twin runtime loop" with Three.js + WebXR.

---

## Development

### Run Tests

```bash
pytest
```

### Lint and Type Check

```bash
ruff check .
mypy src
```

### Use Makefile

```bash
make install   # Install package
make test      # Run tests
make run       # Run main demo
make clean     # Clean build artifacts
```

---

## Strategic Impact

### Industries Transformed

- **🚀 Aerospace & Defense**: Physically verified mission planning
- **⚡ Energy & Nuclear**: AI-designed reactors constrained by thermodynamics
- **🚗 Automotive & Robotics**: Simulation-native autonomy
- **🌍 Climate & Infrastructure**: Predictive models governed by fluid dynamics
- **🔬 Scientific Discovery**: Automated hypothesis testing

### Competitive Advantages

- **Trustworthy AI**: All outputs verifiable against physical law
- **No Hallucinations**: Physics constraints eliminate impossible proposals
- **Explainable**: Full trace of reasoning and corrections
- **Self-Correcting**: Violations trigger automatic refinement
- **Foundational Moat**: Requires deep physics expertise, not just software

---

## Vision

> **Language taught machines to speak.**
> **Physics will teach them to understand.**

Project Physica is not a product—it is a **paradigm shift**. It transforms AI from a probabilistic storyteller into a **law-abiding participant in reality**.

The organizations that adopt this framework will not just build better AI. They will define **the next era of intelligence itself**.

---

## Citation

If you use Project Physica in your research, please cite:

```bibtex
@software{physica2026,
  title = {Project Physica: The Physics World Model for Agentic AI},
  author = {Magana Vsevolodovna, Ruslan},
  year = {2026},
  url = {https://github.com/ruslanmv/Physica}
}
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Roadmap

### Phase I – Classical Reality ✅
- Foundational mechanics: motion, force, energy
- Conservation law enforcement
- Neuro-physical loop integration
- Multi-provider LLM support (OpenAI, Claude, Watsonx, Ollama)
- CrewAI compatibility

### Phase II – Energy & Fields ✅
- **Hamiltonian mechanics**: Symplectic integrators, energy-preserving dynamics
- **Lagrangian mechanics**: Euler-Lagrange equations, action principles, variational methods
- **Electromagnetism**: Maxwell's equations, Lorentz force, particle-in-field simulations
- **Thermodynamics**: State equations, entropy, Carnot/Otto cycles, Second Law
- **Electromagnetic PINNs**: Neural networks learning Maxwell's equations

### Phase III – Production-Ready Applications ✅
- **Phase III-A: Physics-Constrained Autonomous AI** - Trustworthy autonomous control with physics validation
- **Phase III-B: Semiconductor Thermal & Power Digital Twin** - Heat diffusion, power density mapping, multi-layer chips
- **Phase III-C: PINN Surrogates** - Fast, browser-deployable models for web and edge devices
- Industrial applications: Robotics, data centers, AI accelerators, manufacturing

### Phase IV – Advanced Applications (Future)
- Full quantum device simulation
- Multi-physics coupling at scale
- XR visualization and interaction
- Real-time industrial deployment

---

## Contact

For questions, collaborations, or inquiries:

- **Email**: ruslan@example.com
- **GitHub**: [github.com/ruslanmv/Physica](https://github.com/ruslanmv/Physica)

---

**Project Physica** – Where AI meets the laws of nature. 🌌
