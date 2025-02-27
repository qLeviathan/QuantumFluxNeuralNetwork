"""
Quantum Encoder Module
====================

This module implements token encoding into quantum states with latent radius properties.

The encoder maps token indices to quantum states represented in a 2D space,
where tokens are arranged in a circular pattern with varying radii to encode
both semantic meaning (angle) and importance (radius).

Theoretical Background:
---------------------
In quantum mechanics, a particle's state is described by a wavefunction:
    ψ(r,θ) = R(r)e^(iθ)

Where:
- R(r) represents the radial component (magnitude)
- e^(iθ) represents the phase component

In our neural network context:
- Each token is assigned a unique phase angle
- Token importance is encoded in the radial component
- The resulting 2D vector (rx, ry) captures both aspects

This creates a geometric embedding space where:
- Similar tokens have similar angles
- More important tokens have larger radii
- The embedding preserves both direction and magnitude information
"""

import torch
import math
from .geometry import polar_to_cartesian


class QuantumEncoder:
    """
    Encodes token indices into quantum states with latent radius.
    
    This encoder creates a mapping from token IDs to 2D vectors that
    represent quantum states, where each token has a unique position
    in a circular arrangement with a specific radius.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    device : torch.device
        Device where computations are performed
    r_embed : torch.Tensor
        Token embedding matrix of shape (vocab_size, 2)
        
    Physics interpretation:
    ---------------------
    The encoder creates a mapping from discrete tokens to continuous
    quantum states, similar to how a quantum operator maps basis states
    to superposition states. Each token gets a unique position in a 
    2D quantum phase space.
    """
    
    def __init__(self, config):
        """
        Initialize the quantum encoder with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize quantum embeddings with latent radius
        with torch.no_grad():
            # Calculate normalized positions in [0,1] range
            positions = torch.arange(config.vocab_size, dtype=torch.float32, device=self.device)
            r_norm = positions / config.vocab_size  # Normalized position [0,1]
            
            # Calculate angles (2π·r) with phase shift
            theta = 2 * math.pi * r_norm + config.phase_shift
            
            # Initialize radii (using simple formula)
            radii = 0.5 + 0.5 * torch.sin(theta)  # Range [0.0, 1.0]
            
            # Create 2D quantum state vectors with radius
            r_embed = torch.zeros((config.vocab_size, 2), dtype=torch.float32, device=self.device)
            r_embed[:, 0] = radii * torch.cos(theta)  # r·cos(θ)
            r_embed[:, 1] = radii * torch.sin(theta)  # r·sin(θ)
            
            # Store for lookup
            self.register_buffer('r_embed', r_embed)
    
    def encode(self, tokens):
        """
        Encode token indices to quantum states with radius.
        
        Parameters:
        ----------
        tokens : torch.Tensor
            Integer tensor of token indices, shape (..., seq_len)
            
        Returns:
        -------
        torch.Tensor
            Quantum state embeddings, shape (..., seq_len, 2)
            
        Physics interpretation:
        ---------------------
        This is analogous to a quantum measurement operator that maps
        from the discrete basis states (tokens) to continuous quantum states.
        """
        return self.r_embed[tokens]
    
    def register_buffer(self, name, tensor):
        """
        Helper method for registering buffers.
        
        Parameters:
        ----------
        name : str
            Name of the buffer
        tensor : torch.Tensor
            Tensor to register
        """
        setattr(self, name, tensor)
    
    def create_sinusoidal_embeddings(self):
        """
        Alternative initialization using sinusoidal embeddings.
        
        This creates embeddings similar to positional encodings in
        the transformer architecture, but adapted to our polar coordinate system.
        
        Returns:
        -------
        torch.Tensor
            Sinusoidal embeddings for tokens
            
        Physics interpretation:
        ---------------------
        This creates a mapping where tokens are arranged in a pattern
        that preserves locality - nearby tokens have similar phases,
        which is analogous to continuous wavefunctions in quantum mechanics.
        """
        positions = torch.arange(self.config.vocab_size, dtype=torch.float32, device=self.device)
        
        # Create sinusoidal patterns with different frequencies
        freqs = torch.exp(
            torch.arange(0, 2, 2, device=self.device) * 
            -(math.log(10000.0) / 2)
        )
        
        # Outer product of positions and frequencies
        args = positions[:, None] * freqs[None, :]
        
        # Apply sinusoidal functions
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding[:, :2]  # Only keep first two dimensions
