"""
Quantum Flux Neural Network Package
===================================

A PyTorch implementation of a quantum-inspired neural network with Hebbian learning
and geometric integration methods.

This package implements a neural network architecture based on quantum physics principles,
where tokens are represented as quantum states with radial coordinates, and transformed
through attention mechanisms and integrators inspired by quantum mechanics.

Core modules:
------------
- config: Configuration settings for the model
- geometry: Quantum geometric transformations and utilities
- encoder: Token to quantum state encoding
- attention: Quantum attention mechanism
- integration: ODE integration methods
- layers: Neural network layer implementations
- model: Complete model implementation
- visualization: Visualization utilities

Physics background:
-----------------
The neural network incorporates concepts from:
1. Quantum mechanics (wavefunctions, imaginary-time Schrödinger equation)
2. Diffusion equations (radial diffusion)
3. Geometric integration (Heun-Euler method)
4. Hebbian learning ("neurons that fire together, wire together")
"""

from .config import QuantumFluxConfig
from .geometry import polar_to_cartesian, cartesian_to_polar
from .encoder import QuantumEncoder
from .attention import QuantumAttention
from .integration import QuantumIntegrator
from .layers import QuantumLayer
from .model import QuantumFluxModel
from .visualization import visualize_quantum_states

__version__ = '0.1.0'
