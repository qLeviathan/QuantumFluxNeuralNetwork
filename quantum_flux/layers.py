"""
Quantum Layer Module
=================

This module implements neural network layers that process quantum states
using integration, projection, and normalization operations.

The quantum layer combines several operations:
1. Quantum state integration using physical equations
2. Dynamic skip connections based on time step
3. Projection to higher dimensional space
4. Layer normalization for stability

Theoretical Background:
---------------------
The quantum layer implements a form of neural processing inspired by:

1. Quantum dynamics: States evolve according to physical equations

2. Skip connections: Information can bypass the integration step,
   similar to quantum tunneling through potential barriers

3. Dimensionality expansion: Quantum states are projected to a higher
   dimensional space, analogous to embedding a lower-dimensional system
   into a higher-dimensional Hilbert space
"""

import torch
import torch.nn as nn
from .integration import QuantumIntegrator


class QuantumLayer(nn.Module):
    """
    Complete quantum layer with dynamic skip connections.
    
    This layer processes quantum states through integration,
    applies skip connections, and projects to embedding space.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    integrator : QuantumIntegrator
        Quantum integration module
    proj : nn.Linear
        Projection from quantum state to embedding
    layer_norm : nn.LayerNorm
        Layer normalization
        
    Physics interpretation:
    ---------------------
    The layer implements a quantum state transformation with:
    
    1. Physical evolution: States evolve according to quantum-inspired equations
    
    2. Tunneling: Skip connections allow information to "tunnel" through
       the transformation, similar to quantum tunneling
       
    3. Dimensionality expansion: Projecting from 2D quantum states to
       higher dimensions is similar to embedding a system in a larger
       Hilbert space to reveal more complex patterns
    """
    
    def __init__(self, config):
        """
        Initialize the quantum layer with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        
        # Quantum integration
        self.integrator = QuantumIntegrator(config)
        
        # Projection to embedding space
        self.proj = nn.Linear(2, config.embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.epsilon)
    
    def forward(self, states, causal=True):
        """
        Process quantum states through the layer.
        
        Parameters:
        ----------
        states : torch.Tensor
            Quantum state embeddings, shape (batch_size, seq_len, 2)
        causal : bool
            Whether to apply causal masking in attention
            
        Returns:
        -------
        final_states : torch.Tensor
            Updated quantum states after integration
        normalized : torch.Tensor
            Normalized embeddings in higher-dimensional space
        metrics : dict
            Metrics about the layer processing
            
        Physics interpretation:
        ---------------------
        The forward pass implements:
        
        1. Quantum state evolution through integration
        
        2. Skip connections that allow information to bypass the integrator
           (similar to quantum tunneling through a potential barrier)
           
        3. Projection to higher dimensions (expanding the representation space)
           
        4. Normalization for numerical stability (preserving the overall
           scale of the system)
        """
        batch_size = states.shape[0]
        
        # Sample timestep
        dt = self.integrator.sample_timestep(batch_size)
        
        # Evolve quantum states with latent radius
        evolved_states, attention_scores, metrics = self.integrator(
            states, 
            causal=causal,
            dt=dt
        )
        
        # Dynamic skip connection based on timestep
        # Larger dt → more evolution, less skip connection
        # Smaller dt → less evolution, more skip connection
        skip_ratio = torch.sigmoid(self.config.skip_beta * (dt - self.config.dt_scale))
        skip_ratio = skip_ratio.view(-1, 1, 1)
        
        # Apply skip connection
        final_states = skip_ratio * states + (1 - skip_ratio) * evolved_states
        
        # Track skip ratio
        metrics['skip_ratio'] = skip_ratio.mean().item()
        
        # Project to embedding dimension
        embeddings = self.proj(final_states)
        
        # Apply layer normalization
        normalized = self.layer_norm(embeddings)
        
        return final_states, normalized, metrics


class OutputProjection(nn.Module):
    """
    Output projection with Hebbian learning.
    
    This module projects from embedding space to vocabulary space,
    and includes Hebbian-inspired weight updates.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    weight : nn.Parameter
        Weight matrix for projection
    bias : nn.Parameter
        Bias vector
        
    Physics interpretation:
    ---------------------
    The output projection implements:
    
    1. Dimensionality reduction from embedding space to vocabulary space
    
    2. Hebbian learning ("neurons that fire together, wire together"),
       which has parallels to how synaptic connections strengthen
       in biological neural networks based on co-activation patterns
    """
    
    def __init__(self, config):
        """
        Initialize the output projection with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        self.in_features = config.embed_dim
        self.out_features = config.vocab_size
        
        # Initialize weights
        stdv = 1. / (self.in_features ** 0.5)
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input embeddings
            
        Returns:
        -------
        torch.Tensor
            Output logits
        """
        return nn.functional.linear(x, self.weight, self.bias)
    
    def update_weights(self, x, targets, learning_rate=None):
        """
        Update weights using Hebbian learning rule.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input embeddings
        targets : torch.Tensor
            Target token indices
        learning_rate : float or None
            Learning rate. If None, use config value
            
        Returns:
        -------
        dict
            Metrics about the update
            
        Physics interpretation:
        ---------------------
        Hebbian learning implements the principle that "neurons that fire
        together wire together" - the connections between neurons strengthen
        when they activate simultaneously.
        
        In our context:
        1. The error between prediction and target drives the update
        2. The update strengthens connections that reduce this error
        3. This is similar to how physical systems evolve towards
           minimum energy states
        """
        # Use config learning rate if not specified
        lr = learning_rate or self.config.learning_rate
        
        with torch.no_grad():
            # Create one-hot target representation
            y_onehot = nn.functional.one_hot(targets, num_classes=self.out_features).float()
            
            # Current predictions
            current_out = self.forward(x)
            
            # Error
            error = y_onehot - current_out
            
            # Weight update (Hebbian rule)
            weight_update = lr * torch.matmul(error.t(), x) / x.size(0)
            bias_update = lr * error.mean(dim=0)
            
            # Apply updates
            self.weight.add_(weight_update)
            self.bias.add_(bias_update)
            
            # Calculate metrics
            mean_error = error.abs().mean().item()
            
            return {'mean_error': mean_error}
