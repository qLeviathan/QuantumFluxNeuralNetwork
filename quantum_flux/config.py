"""
Quantum Flux Configuration Module
===============================

This module defines the configuration settings for the Quantum Flux model,
encapsulating hyperparameters and model architecture specifications.

The configuration combines several physics-inspired parameters:
1. Embedding dimensions and sequence handling
2. Quantum dynamics parameters (phase shifts, time steps)
3. Hebbian learning parameters
4. Latent radius parameters

Theoretical Background:
---------------------
Many parameters have direct analogies to physical concepts:

- embed_dim: Dimensionality of the quantum state space
  (analogous to Hilbert space dimensions in quantum mechanics)

- phase_shift: Controls initial quantum state angle displacements
  (similar to phase factors in quantum wavefunctions)

- dt_scale, dt_max: Govern integration time steps 
  (analogous to energy-time uncertainty relation constraints)

- hebbian_decay, hebbian_strength: Control neural connection dynamics
  (implementing Hebb's "neurons that fire together, wire together" principle)

- min_radius, max_radius: Define the bounds of token importance in the geometric space
  (analogous to probability amplitude constraints in quantum mechanics)
"""

import math
import torch


class QuantumFluxConfig:
    """
    Configuration class for the Quantum Flux model.
    
    This class encapsulates all the hyperparameters and settings needed
    for a Quantum Flux model instance, including dimensions, learning
    parameters, and physics-inspired constants.
    
    Attributes:
    ----------
    embed_dim : int
        Embedding dimension for token representations
    max_seq_length : int
        Maximum sequence length the model can process
    num_layers : int
        Number of quantum layers in the model
    vocab_size : int
        Size of the vocabulary
    phase_shift : float
        Initial phase shift for token embeddings (in radians)
    dt_scale : float
        Time step scale factor for integration
    dt_max : float
        Maximum allowed time step
    hebbian_decay : float
        Decay factor for Hebbian learning connections
    hebbian_strength : float
        Strength of Hebbian learning updates
    min_radius : float
        Minimum token radius in the latent space
    max_radius : float
        Maximum token radius in the latent space
    radius_update_rate : float
        Rate at which token radii are updated
    skip_beta : float
        Controls skip connection dynamics
    learning_rate : float
        Learning rate for weight updates
    epsilon : float
        Small value for numerical stability
    connection_clamp : float
        Maximum absolute value for connection strengths
        
    Physics interpretation:
    ---------------------
    - embed_dim: In quantum physics, corresponds to the dimensionality of the
      Hilbert space that represents possible states
      
    - phase_shift: Analogous to initial phase offsets in quantum wavefunctions
      that affect interference patterns
      
    - dt_scale, dt_max: In quantum simulations, the time step must be small
      enough to resolve the fastest oscillations in the system (Nyquist criterion)
      
    - Hebbian parameters: Implement a form of neural plasticity where
      connections strengthen when neurons fire together (resembling
      quantum entanglement where particles develop correlated states)
      
    - Radius parameters: Control the "importance" or "amplitude" of each
      token, similar to probability amplitudes in quantum states
    """
    
    def __init__(
        self,
        embed_dim=64,        
        max_seq_length=64,   
        num_layers=2,        
        vocab_size=10000,    
        phase_shift=math.pi/4,  
        dt_scale=1e-5,       
        dt_max=0.05,         
        hebbian_decay=0.99,  
        hebbian_strength=0.1,
        min_radius=0.1,      
        max_radius=2.0,      
        radius_update_rate=0.01,  
        skip_beta=5.0,       
        learning_rate=1e-4,
        epsilon=1e-6,
        connection_clamp=0.5
    ):
        """
        Initialize the configuration with specified or default parameters.
        
        Parameters:
        ----------
        embed_dim : int
            Embedding dimension for token representations
        max_seq_length : int
            Maximum sequence length the model can process
        num_layers : int
            Number of quantum layers in the model
        vocab_size : int
            Size of the vocabulary
        phase_shift : float
            Initial phase shift for token embeddings (in radians)
        dt_scale : float
            Time step scale factor for integration
        dt_max : float
            Maximum allowed time step
        hebbian_decay : float
            Decay factor for Hebbian learning connections
        hebbian_strength : float
            Strength of Hebbian learning updates
        min_radius : float
            Minimum token radius in the latent space
        max_radius : float
            Maximum token radius in the latent space
        radius_update_rate : float
            Rate at which token radii are updated
        skip_beta : float
            Controls skip connection dynamics
        learning_rate : float
            Learning rate for weight updates
        epsilon : float
            Small value for numerical stability
        connection_clamp : float
            Maximum absolute value for connection strengths
        """
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.phase_shift = phase_shift
        self.dt_scale = dt_scale
        self.dt_max = dt_max
        self.hebbian_decay = hebbian_decay
        self.hebbian_strength = hebbian_strength
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_update_rate = radius_update_rate
        self.skip_beta = skip_beta
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.connection_clamp = connection_clamp
    
    @property
    def optimal_batch_size(self):
        """
        Calculate optimal batch size based on dimensions.
        
        Returns:
        -------
        int
            Recommended batch size
            
        Notes:
        -----
        This is a heuristic based on the ratio of embedding dimension
        to sequence length, which affects memory and computation efficiency.
        
        Physics interpretation:
        ---------------------
        This is similar to optimizing computational resources based on 
        problem dimensions in numerical physics simulations.
        """
        return max(1, int((self.embed_dim / self.max_seq_length)**0.5 * 16))
    
    def get_default_time_step(self, batch_size=1):
        """
        Sample time step from a Beta distribution with parameters based on
        embedding dimension and sequence length.
        
        Parameters:
        ----------
        batch_size : int
            Number of time steps to sample
            
        Returns:
        -------
        torch.Tensor
            Sampled time steps
            
        Physics interpretation:
        ---------------------
        In numerical integration of physical systems, adaptive time stepping
        is often used to balance stability and computational efficiency.
        
        The Beta distribution parameters are chosen based on the system dimensions:
        - α = embed_dim/2: Higher dimensional systems need smaller time steps
        - β = max_seq_length/2: Longer sequences need smaller time steps
        
        This ensures the integration is stable across various model sizes.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set alpha, beta based on dimensions
        alpha = float(self.embed_dim) / 2.0
        beta = float(self.max_seq_length) / 2.0
        
        # Sample from beta distribution
        x = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        
        # Convert to time step (1/x with clamping for stability)
        dt = (1.0 / x.clamp(min=1e-3, max=0.999)) * self.dt_scale
        dt = torch.clamp(dt, min=1e-6, max=self.dt_max)
        
        return dt
