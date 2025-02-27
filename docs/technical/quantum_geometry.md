# Quantum Geometry: A Developer's Guide

This training module introduces the quantum geometry concepts underlying the Quantum Flux Neural Network. By the end, you'll understand how tokens are represented as quantum states and how these states evolve through the network.

## Table of Contents

1. [Introduction to Quantum States](#introduction-to-quantum-states)
2. [Polar vs. Cartesian Coordinates](#polar-vs-cartesian-coordinates)
3. [Token Embedding in Quantum Space](#token-embedding-in-quantum-space)
4. [Geometric Relationships Between Tokens](#geometric-relationships-between-tokens)
5. [State Evolution Through Integration](#state-evolution-through-integration)
6. [Radial Adjustments and Token Importance](#radial-adjustments-and-token-importance)
7. [Hands-on Examples](#hands-on-examples)

## Introduction to Quantum States

### Traditional Embeddings vs. Quantum States

In traditional neural networks like transformers, tokens are embedded as high-dimensional vectors without clear geometric meaning. In Quantum Flux, tokens exist as points in a quantum state space with clear physical interpretations.

**Traditional Embedding:**
```
token → [0.1, -0.3, 0.5, 0.2, -0.4, ...]  (high-dimensional vector)
```

**Quantum State Embedding:**
```
token → (r, θ) → (rx, ry, 0, 0, ...)  (first 2 dimensions have geometric meaning)
```

Where:
- `r` is the radius (importance/amplitude)
- `θ` is the angle (semantic meaning/phase)
- `(rx, ry)` are the Cartesian coordinates derived from `(r, θ)`

### Advantages of Quantum States

This representation offers several advantages:

1. **Interpretability:** States have clear geometric meanings
2. **Visualization:** 2D representation is easy to visualize
3. **Physics-inspired operations:** Can apply equations from quantum mechanics and diffusion theory
4. **Emergent behavior:** Complex patterns emerge from simple physical rules

## Polar vs. Cartesian Coordinates

### Polar Coordinates

Polar coordinates represent a point using:
- `r`: Distance from the origin
- `θ`: Angle from the positive x-axis (counterclockwise)

This is particularly useful for expressing quantum states where magnitude and phase are natural properties.

### Cartesian Coordinates

Cartesian coordinates represent a point using:
- `x`: Horizontal distance from the origin
- `y`: Vertical distance from the origin

These are more convenient for direct calculations and tensor operations.

### Conversion Formulas

Converting between these coordinate systems:

**Polar → Cartesian:**
```
rx = r * cos(θ)
ry = r * sin(θ)
```

**Cartesian → Polar:**
```
r = sqrt(rx² + ry²)
θ = atan2(ry, rx)
```

### Implementation in Quantum Flux

The model internally uses Cartesian coordinates for efficiency but conceptually thinks in polar coordinates. The first two dimensions of each token embedding store the `(rx, ry)` components.

```python
# Example from geometry.py
def polar_to_cartesian(r, theta):
    if isinstance(r, torch.Tensor):
        rx = r * torch.cos(theta)
        ry = r * torch.sin(theta)
    else:
        rx = r * math.cos(theta)
        ry = r * math.sin(theta)
    return rx, ry
```

## Token Embedding in Quantum Space

### Initial Embedding Strategy

When tokens are first embedded into quantum states, we position them in a circular arrangement:

1. Each token `i` in a sequence of length `D` is assigned an angle:
   ```
   θ_i = 2π * (i/D)
   ```

2. Radii increase with position to create a spiral-like pattern:
   ```
   r_i = 0.3 + 0.6 * (i/(D-1))  # from 0.3 to 0.9
   ```

3. These `(r, θ)` coordinates are converted to `(rx, ry)` and stored in the first two dimensions of the embedding.

### Visualizing the Embedding

This creates a distinctive spiral pattern where:
- The first token is at angle 0 with radius 0.3
- The last token is at angle `2π*(D-1)/D` with radius 0.9
- Consecutive tokens are adjacent in both angle and radius

![Token Embedding Visualization](../images/token_embedding.png)

### Physics Interpretation

This embedding has a quantum physics interpretation:
- Tokens are like particles in a vortex
- Their position encodes both order (angle) and importance (radius)
- This creates a natural geometric structure for processing information

## Geometric Relationships Between Tokens

### Direct Similarity Calculation

In traditional transformers, token relationships are computed through query-key attention scores. In Quantum Flux, we directly compute geometric relationships:

```
similarity(i, j) = r_i * r_j * cos(θ_i - θ_j)
```

This is equivalent to the dot product of the position vectors and has a natural interpretation:
- Maximum when tokens have the same angle
- Scaled by the product of their radii
- Negative when tokens are in opposite directions

### Adaptive Thresholding

Instead of softmax, we use adaptive thresholding to determine which tokens interact:

1. Calculate mean and standard deviation of similarity scores
2. Set threshold based on statistics: `threshold = mean + 0.5 * std`
3. Only connections above this threshold are considered

This creates a sparse attention pattern that adapts to the data distribution.

### Hebbian Connection Strengthening

Connections that fire together wire together:

1. Active connections strengthen over time
2. Inactive connections decay
3. This creates a form of memory in the attention mechanism

## State Evolution Through Integration

### Physical Equations

Token states evolve through equations inspired by quantum mechanics and diffusion theory:

**Radial Diffusion Equation:**
```
∂u/∂t = α·[1/r·∂/∂r(r·∂u/∂r)]
```

**Imaginary-time Schrödinger Equation:**
```
∂ψ/∂τ = α·[1/r·∂/∂r(r·∂ψ/∂r)] - V(r)·ψ
```

### Heun-Euler Integration

We solve these equations numerically using the Heun-Euler method:

1. First stage (k1): Calculate initial derivative
   ```
   k1 = f(y_n)
   ```

2. Intermediate state: 
   ```
   y_mid = y_n + dt·k1
   ```

3. Second stage (k2): Calculate derivative at intermediate state
   ```
   k2 = f(y_mid)
   ```

4. Final update:
   ```
   y_{n+1} = y_n + 0.5·dt·(k1 + k2)
   ```

### Implementation in Quantum Flux

In our model, this integration process acts on token states:

1. States interact based on their geometric relationships
2. The context vector represents the derivative: `context = sum(score * states)`
3. Integration follows the Heun-Euler steps
4. States evolve following physical principles

## Radial Adjustments and Token Importance

### Dynamic Importance

Token importance (radius) evolves dynamically through the network:

1. Initial radii are set based on position
2. Integration influences radii based on interactions
3. Explicit radius adjustment based on connection influence

### Radius Adjustment Mechanism

After integration, we adjust radii:

1. Calculate current radius: `new_radius = norm(r_new)`
2. Compute adjustment from connections: `adjustment = sum(score) * radius_update_rate`
3. Calculate target radius: `adjusted_radius = clamp(new_radius + adjustment, min_radius, max_radius)`
4. Scale vectors to match target radius: `r_final = r_new * (adjusted_radius / new_radius)`

### Physics Interpretation

This has a natural interpretation in quantum mechanics:
- Radius represents probability amplitude
- Important tokens (with many connections) develop larger amplitudes
- Less important tokens have smaller amplitudes
- This creates a dynamic flow of importance through the network

## Hands-on Examples

### Example 1: Creating and Visualizing Quantum States

```python
import torch
import matplotlib.pyplot as plt
from quantum_flux.geometry import polar_to_cartesian
from quantum_flux.visualization import visualize_quantum_states

# Create quantum states for a sequence of 10 tokens
seq_length = 10
angles = torch.linspace(0, 2*torch.pi*(seq_length-1)/seq_length, seq_length)
radii = 0.3 + 0.6 * torch.linspace(0, 1, seq_length)

# Convert to Cartesian coordinates
rx, ry = polar_to_cartesian(radii, angles)
states = torch.stack([rx, ry], dim=-1)

# Visualize
fig = visualize_quantum_states(states, title="Example Token States")
plt.show()
```

### Example 2: Computing Geometric Relationships

```python
import torch
import matplotlib.pyplot as plt
from quantum_flux.geometry import negative_distance_matrix, normalize_matrix
from quantum_flux.visualization import visualize_attention

# Using the states from Example 1
# Compute negative distance matrix
neg_dist = negative_distance_matrix(states)

# Normalize to [0, 1]
norm_dist = normalize_matrix(neg_dist)

# Visualize attention pattern
fig = visualize_attention(norm_dist, title="Token Relationships")
plt.show()
```

### Example 3: Evolving States Through Integration

```python
from quantum_flux.config import QuantumFluxConfig
from quantum_flux.integration import QuantumIntegrator

# Create configuration
config = QuantumFluxConfig(
    embed_dim=64,
    max_seq_length=16,
    dt_scale=1e-5,
    dt_max=0.05
)

# Create integrator
integrator = QuantumIntegrator(config)

# Evolve states for a batch of sequences
batch_size = 1
states_batch = states.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, seq_len, 2]

# Perform integration step
dt = torch.tensor([0.01], device=states.device)
evolved_states, attention_scores, metrics = integrator(
    states_batch, 
    causal=True,
    dt=dt
)

# Visualize before and after
fig = visualize_quantum_states(states, title="Before Integration")
plt.figure()
fig = visualize_quantum_states(evolved_states[0], title="After Integration")
plt.show()
```

### Example 4: Full Layer Processing

```python
from quantum_flux.layers import QuantumLayer

# Create quantum layer
layer = QuantumLayer(config)

# Process states through layer
final_states, embeddings, metrics = layer(states_batch, causal=True)

# Embeddings are the higher-dimensional projections of the quantum states
print(f"Quantum states shape: {final_states.shape}")  # [batch, seq_len, 2]
print(f"Embeddings shape: {embeddings.shape}")       # [batch, seq_len, embed_dim]
```

## Conclusion

This concludes our introduction to quantum geometry in the Quantum Flux Neural Network. You now understand:

1. How tokens are represented as quantum states with radius and angle
2. How geometric relationships between tokens are computed
3. How states evolve through physical equations
4. How token importance (radius) changes dynamically
5. How to work with the Quantum Flux API to create and manipulate quantum states

For more advanced topics, see the [full reference guide](../reference.md) and [customization guide](../guides/customization.md).
