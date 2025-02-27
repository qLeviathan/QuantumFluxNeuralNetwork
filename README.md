# Quantum Flux Neural Network

A PyTorch implementation of a quantum-inspired neural network with Hebbian learning and geometric integration.

![Quantum States Visualization](docs/images/quantum_states.png)

## Overview

Quantum Flux is a novel neural network architecture that draws inspiration from quantum mechanics, diffusion theory, and Hebbian learning principles. Unlike traditional transformers, this model:

- Represents tokens as quantum states with magnitude (radius) and phase (angle)
- Evolves states through physical equations rather than self-attention
- Implements Hebbian learning ("neurons that fire together, wire together")
- Uses geometric integration methods that preserve important physical properties

This framework bridges quantum physics and deep learning, creating a neural network that follows physical laws in its operations.

## Key Features

- **Quantum State Representation**: Tokens are represented as points in a quantum state space with both magnitude and phase
- **Physics-Inspired Evolution**: States evolve through equations derived from quantum mechanics and diffusion theory
- **Hebbian Learning**: Connections strengthen based on co-activation patterns, similar to neuroplasticity
- **Geometric Integration**: Numerical methods preserve important geometric properties of the system
- **Latent Radius**: Token importance evolves dynamically through the "radius" property
- **Visualization Tools**: Comprehensive tools for visualizing quantum states, attention patterns, and training metrics

## Installation

```bash
pip install quantum-flux
```

Or install from source:

```bash
git clone https://github.com/yourusername/quantum-flux.git
cd quantum-flux
pip install -e .
```

## Quick Start

```python
import torch
from quantum_flux import QuantumFluxConfig, QuantumFluxModel
from quantum_flux.visualization import visualize_quantum_states

# Create configuration
config = QuantumFluxConfig(
    embed_dim=64,
    max_seq_length=32,
    num_layers=3,
    vocab_size=10000
)

# Create model
model = QuantumFluxModel(config)

# Process input
input_ids = torch.randint(0, config.vocab_size, (1, 10))
outputs = model(input_ids)

# Generate text
generated = model.generate(input_ids, max_length=50)

# Visualize quantum states
states = outputs['quantum_states'][0].cpu().numpy()
fig = visualize_quantum_states(states)
fig.show()
```

## Core Concepts

### Quantum Geometry

In traditional transformers, tokens are represented as high-dimensional vectors. In Quantum Flux, tokens exist in a 2D quantum state space defined by:

- **Radius (r)**: Represents token importance or amplitude
- **Angle (θ)**: Represents token phase or semantic meaning

These are transformed to Cartesian coordinates:
- **rx = r·cos(θ)**
- **ry = r·sin(θ)**

The first two dimensions of each token embedding encode these quantum states, while additional dimensions can capture more complex relationships.

### Physical Evolution

Instead of self-attention, Quantum Flux evolves token states through physical equations:

1. States interact based on their geometric proximity
2. Evolution follows the Heun-Euler integration method:
   ```
   k1 = f(y_n)
   k2 = f(y_n + dt·k1)
   y_{n+1} = y_n + 0.5·dt·(k1 + k2)
   ```
3. Radius adjustment controls token importance dynamically

### Hebbian Learning

Connections between tokens strengthen when they co-activate, following Hebb's principle that "neurons that fire together, wire together." This creates a form of memory that preserves important relationships throughout training.

## Documentation

For detailed documentation, see the [full reference guide](docs/reference.md).

For a step-by-step introduction to quantum geometry concepts, see the [training module](docs/training/quantum_geometry.md).

## Advanced Usage

### Training

For production training, use the optimized training script:

```bash
python scripts/train.py --config configs/base.yaml --data path/to/data --output path/to/output
```

### Customization

Quantum Flux can be customized with different:
- Integration methods
- Quantum state initializations
- Hebbian learning parameters
- Radius evolution strategies

See the [customization guide](docs/guides/customization.md) for details.

## Performance

Quantum Flux offers several advantages over traditional transformers:

- **Efficiency**: Direct geometric calculations eliminate the need for query-key-value projections
- **Interpretability**: States have clear physical meanings and visualizations
- **Memory**: Hebbian learning preserves important relationships over time
- **Emergent behavior**: Complex patterns emerge from simple physical rules

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib (for visualization)
- tqdm (for progress bars)

## Citation

If you use Quantum Flux in your research, please cite:

```
@software{quantum_flux,
  author = {Marc, Castillo},
  title = {Quantum Flux Neural Network},
  year = {2025},
  url = {https://github.com/qLeviathan/quantum-flux}
}
```

## License

MIT License

## Future Directions

This package lays the groundwork for the Quantum Flux Reality Engine, an extended framework that will incorporate:

- Quantum field theory principles for multi-modal data
- Simulation of complex physical systems
- Reality modeling through quantum-inspired transformations
- Integration with other physics-based AI systems

Stay tuned for updates on this exciting extension.
