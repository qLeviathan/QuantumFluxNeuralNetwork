# Quantum Flux Reference Guide

This comprehensive guide explains the theoretical foundations, implementation details, and connections between physics and neural networks in the Quantum Flux model.

## Table of Contents

1. [Physics Foundations](#physics-foundations)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Hebbian Learning Infrastructure](#hebbian-learning-infrastructure)
4. [Magnetic Flux Analogy](#magnetic-flux-analogy)
5. [Module Reference](#module-reference)
6. [Future Research Directions](#future-research-directions)

## Physics Foundations

### Quantum Mechanics Foundation

The Quantum Flux model draws inspiration from several quantum mechanical principles:

#### 1. Wavefunctions and States

In quantum mechanics, particles are described by wavefunctions:

```
ψ(r,θ) = R(r)e^(iθ)
```

Where:
- `R(r)` is the radial component (amplitude)
- `e^(iθ)` is the phase component

In our model, tokens are represented as points in a 2D quantum state space, where:
- The radial component (`r`) represents token importance/relevance
- The angular component (`θ`) represents semantic relationships

#### 2. Schrödinger Equation

The evolution of quantum states is governed by the Schrödinger equation:

```
iℏ ∂ψ/∂t = Ĥψ
```

Where:
- `Ĥ` is the Hamiltonian operator
- `ψ` is the wavefunction
- `ℏ` is the reduced Planck constant

Our model uses an imaginary-time variation:

```
∂ψ/∂τ = -Ĥψ
```

This formulation has mathematical similarities to diffusion equations, which we exploit in our integration methods.

### Diffusion Equations

The model also draws from diffusion theory, particularly the radial diffusion equation:

```
∂u/∂t = α·[1/r·∂/∂r(r·∂u/∂r)]
```

This equation describes how quantities spread out over time in systems with radial symmetry, providing a mathematical foundation for how token states evolve.

### Geometric Integration

To solve these differential equations numerically, we employ the Heun-Euler method, a second-order Runge-Kutta method:

```
k1 = f(y_n)
k2 = f(y_n + dt·k1)
y_{n+1} = y_n + 0.5·dt·(k1 + k2)
```

This method offers a good balance between accuracy and computational efficiency while preserving important geometric properties of the system.

## Neural Network Architecture

### Token Embedding

Unlike traditional transformers that use linear embedding layers, Quantum Flux embeds tokens into quantum states:

1. Each token is assigned a unique position in a circular arrangement
2. The position determines both angle (θ) and radius (r)
3. These are converted to Cartesian coordinates (rx, ry)
4. These coordinates are stored in the first two dimensions of the embedding

### Quantum Attention

Traditional attention computes scores using query-key-value projections and softmax. Quantum Flux instead:

1. Computes direct geometric relationships between token states
2. Uses dot products to measure similarity: `r_i·r_j·cos(θ_i - θ_j)`
3. Applies adaptive thresholding instead of softmax
4. Develops connection strengths over time through Hebbian learning

### Integration Layer

Each layer in the network:

1. Evolves quantum states through physical equations
2. Implements dynamic skip connections (quantum tunneling)
3. Projects to higher dimensions
4. Normalizes for stability

### Full Pipeline

The complete forward pass:

1. Tokens → Quantum States
2. Multiple Quantum Layers (integration + projection)
3. Final Projection → Vocabulary
4. Loss Computation (cross-entropy + geometric)
5. Hebbian Weight Updates

## Hebbian Learning Infrastructure

### Hebbian Principle

The core principle, "neurons that fire together, wire together," is implemented through:

1. Connection Strength Matrix
   - Tracks the strength of connections between tokens
   - Evolves over time through decay and reinforcement

2. Update Rules
   - Existing connections decay: `strength *= decay_factor`
   - Active connections strengthen: `strength += hebbian_strength * score`
   - Strength is clamped to prevent runaway values

3. Memory Effect
   - Connection strengths influence future attention scores
   - This creates a form of memory that persists throughout training

### Implementation Details

The Hebbian learning implementation includes:

1. Adaptive Thresholding
   - Connections form when similarity exceeds a threshold
   - Threshold adapts based on the statistical properties of the attention matrix

2. Causal Masking
   - Ensures information only flows forward in time
   - Maintains causality constraints in the attention mechanism

3. Weight Updates
   - The output projection weights update using a Hebbian-inspired rule
   - Updates occur outside the standard backpropagation

## Magnetic Flux Analogy

The Quantum Flux model has interesting parallels to magnetic flux concepts in physics:

### Quantum Field Analogy

1. Token States as Field Excitations
   - Tokens can be viewed as excitations in a quantum field
   - Their interactions follow field-theoretic principles

2. Connection Strengths as Flux Lines
   - The Hebbian connection strengths can be viewed as flux lines connecting tokens
   - Stronger connections = stronger flux

3. Radius as Field Amplitude
   - The token radius represents the amplitude of the field excitation
   - Larger radius = stronger excitation

### Mathematical Connection

The evolution equations have similarities to magnetic flux equations:

1. Poisson Equation
   ```
   ∇²A = -μ₀J
   ```
   Where:
   - `A` is the magnetic vector potential
   - `J` is the current density
   - `μ₀` is the magnetic permeability

2. In our model:
   - Token states evolve in a way analogous to magnetic potentials
   - Connection strengths evolve analogously to magnetic flux densities

This analogy provides a rich interpretative framework for understanding the model's dynamics.

## Module Reference

### config.py

The configuration module defines hyperparameters with physics-inspired meanings:

- `embed_dim`: Dimensionality of the quantum state space (Hilbert space dimensions)
- `phase_shift`: Controls initial quantum state angle displacements (phase factors)
- `dt_scale`, `dt_max`: Govern integration time steps (energy-time uncertainty constraints)
- `hebbian_decay`, `hebbian_strength`: Control neural connection dynamics (Hebbian principle)
- `min_radius`, `max_radius`: Define the bounds of token importance (probability amplitude constraints)

### geometry.py

Core geometric transformations:

- `polar_to_cartesian`: Converts (r, θ) to (rx, ry)
- `cartesian_to_polar`: Converts (rx, ry) to (r, θ)
- `normalize_embeddings`: Normalizes vectors to unit length
- `negative_distance_matrix`: Computes similarity based on proximity
- `normalize_matrix`: Scales matrix values to [0, 1]

### encoder.py

Transforms token indices to quantum states:

- `encode`: Maps token IDs to (rx, ry) coordinates
- `create_sinusoidal_embeddings`: Alternative initialization with sinusoidal patterns

### attention.py

Implements quantum-inspired attention:

- `compute_score`: Calculates attention based on geometric relationships
- `get_causal_mask`: Ensures causality constraints

### integration.py

Implements numerical integration of physical equations:

- `forward`: Performs one integration step with Heun-Euler method
- `sample_timestep`: Samples adaptive time steps
- `radial_diffusion_step`: Implements radial diffusion equation
- `imaginary_time_schrodinger_step`: Implements imaginary-time Schrödinger equation

### layers.py

Neural network layers:

- `QuantumLayer`: Processes quantum states through integration and projection
- `OutputProjection`: Projects to vocabulary space with Hebbian learning

### model.py

Complete model implementation:

- `forward`: Full processing pipeline
- `generate`: Text generation through auto-regressive sampling

### visualization.py

Visualization utilities:

- `visualize_quantum_states`: Shows tokens in quantum state space
- `visualize_attention`: Shows attention patterns
- `visualize_training_metrics`: Shows training progress
- `visualize_state_evolution`: Shows state evolution through layers

## Future Research Directions

The Quantum Flux model opens several exciting research directions:

### 1. Extended Physical Models

Future versions could incorporate:
- Quantum field theory principles for multi-modal data
- Relativistic effects for modeling context dependencies
- Quantum entanglement for modeling complex relationships

### 2. Applications

Promising application areas include:
- Language modeling with deeper semantic understanding
- Physical system simulation
- Quantum-inspired reinforcement learning

### 3. Integration with Quantum Flux Reality Engine

This model serves as a foundation for the larger Quantum Flux Reality Engine, which will extend these principles to:
- Multi-modal reality modeling
- Physics-informed simulations
- Emergent complex behaviors from simple physical rules

### 4. Optimization Strategies

Future research will explore:
- Specialized hardware acceleration for physics-based neural networks
- Sparse attention implementations for efficiency
- Adaptive radius strategies based on context importance
