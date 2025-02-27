"""
Quantum Attention Module
======================

This module implements a quantum-inspired attention mechanism that processes
token embeddings based on their geometric relationships in the quantum state space.

Unlike traditional attention mechanisms that use linear projections and softmax,
this quantum attention computes relationships directly based on the geometric
properties of the quantum states, including their radii and relative angles.

Theoretical Background:
---------------------
In quantum mechanics, the interaction between particles depends on their
relative positions and momenta. Similarly, in our model:

1. Interaction strength is determined by geometric proximity
   (dot product of position vectors)

2. Hebbian learning ("neurons that fire together, wire together") is implemented
   through adaptive connection strengths

3. Causal masking ensures information only flows in the forward direction,
   similar to causality constraints in physics

The attention mechanism uses direct quantum state similarity instead of
the standard query-key-value mechanism of traditional transformers.
"""

import torch
import torch.nn as nn


class QuantumAttention(nn.Module):
    """
    Quantum attention with latent radius.
    
    This attention mechanism computes token interactions based on their
    geometric relationships in the quantum state space, using direct
    inner products between state vectors.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    device : torch.device
        Device where computations are performed
    causal_mask : torch.Tensor or None
        Causal attention mask (lower triangular)
    connection_strength : torch.Tensor or None
        Hebbian connection strength matrix
        
    Physics interpretation:
    ---------------------
    The attention mechanism mimics quantum particle interactions where:
    - Particle interaction strength depends on spatial proximity
    - Interactions strengthen over time (Hebbian learning)
    - Future states cannot influence past states (causality)
    """
    
    def __init__(self, config):
        """
        Initialize the quantum attention with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize buffers
        self.register_buffer('causal_mask', None)
        self.register_buffer('connection_strength', None)
    
    def get_causal_mask(self, seq_len):
        """
        Get or compute causal attention mask.
        
        Parameters:
        ----------
        seq_len : int
            Sequence length
            
        Returns:
        -------
        torch.Tensor
            Boolean causal mask of shape (seq_len, seq_len)
            
        Physics interpretation:
        ---------------------
        This implements causality constraints, ensuring that information
        only flows forward in time, similar to how physical interactions
        respect causality in spacetime.
        """
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create lower triangular mask (future cannot influence past)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
            self.register_buffer('causal_mask', mask)
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def compute_score(self, r_embed, causal=True):
        """
        Compute attention scores using direct dot product for latent radius states.
        
        Parameters:
        ----------
        r_embed : torch.Tensor
            Quantum state embeddings, shape (batch_size, seq_len, 2)
        causal : bool
            Whether to apply causal masking
            
        Returns:
        -------
        score : torch.Tensor
            Attention scores, shape (batch_size, seq_len, seq_len)
        metrics : dict
            Metrics about the attention patterns
            
        Physics interpretation:
        ---------------------
        This computes interaction strengths between quantum states based on:
        
        1. Spatial proximity (dot product of position vectors)
           r_i·r_j·cos(θ_i - θ_j)
           
        2. Adaptive thresholding to create sparse interactions (similar to
           activation thresholds in physical systems)
           
        3. Hebbian learning where "neurons that fire together wire together"
           (analogous to quantum entanglement developing over time)
        """
        batch_size, seq_len, _ = r_embed.shape
        
        # Use direct dot product to compute r_i*r_j*cos(θ_i - θ_j)
        r_embed_i = r_embed.unsqueeze(2)                   # [batch, seq, 1, 2]
        r_embed_j = r_embed.unsqueeze(1)                   # [batch, 1, seq, 2]
        similarity = torch.sum(r_embed_i * r_embed_j, dim=-1)  # [batch, seq, seq]
        
        # Calculate radii for metrics only
        radii = torch.norm(r_embed, dim=-1)  # [batch, seq]
        
        # Scale to [0, 1] range
        similarity = (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Apply adaptive threshold for optimal sparsity
        epsilon = self.config.epsilon
        mean = similarity.mean(dim=-1, keepdim=True) 
        var = ((similarity - mean) ** 2).mean(dim=-1, keepdim=True) + epsilon
        std = torch.sqrt(var)
        
        # Set threshold based on mean and std to achieve target sparsity
        threshold = mean + 0.5 * std
        
        # Initialize or update connection strength tracker
        if self.connection_strength is None or self.connection_strength.size(0) != batch_size or self.connection_strength.size(1) < seq_len:
            # Reinitialize with correct batch size and sequence length
            self.connection_strength = torch.zeros(
                batch_size, seq_len, seq_len, device=self.device
            )
        
        # Create score tensor
        score = torch.zeros_like(similarity)
        
        # Apply causal masking if needed
        if causal:
            # Get causal mask (lower triangular)
            mask = self.get_causal_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Find neurons that should fire together (high similarity and causal)
            valid_connections = (similarity >= threshold) & mask
        else:
            valid_connections = similarity >= threshold
        
        # Neurons that fire together wire together
        score[valid_connections] = similarity[valid_connections]
        
        # Clip scores for stability
        score = torch.clamp(score, 0.0, 1.0)
        

        # Update connection strength using Hebbian learning
        with torch.no_grad():
            # Check if we're in inference mode
            in_inference = torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
            
            if in_inference:
                # Create clones to avoid in-place operations during inference
                temp_connection = self.connection_strength.clone()
                temp_connection[:, :seq_len, :seq_len] *= self.config.hebbian_decay
                
                # Strengthen active connections
                update_mask = valid_connections & (score > 0.1)
                if update_mask.any():
                    temp_connection[:, :seq_len, :seq_len][update_mask] += \
                        self.config.hebbian_strength * score[update_mask]
                
                # Clip connection strength
                temp_connection = torch.clamp(
                    temp_connection, 
                    -self.config.connection_clamp,
                    self.config.connection_clamp
                )
                
                # Update the reference
                self.connection_strength = temp_connection
            else:
                # Original in-place operations for training mode
                self.connection_strength[:, :seq_len, :seq_len] *= self.config.hebbian_decay
                
                # Strengthen active connections
                update_mask = valid_connections & (score > 0.1)
                if update_mask.any():
                    self.connection_strength[:, :seq_len, :seq_len][update_mask] += \
                        self.config.hebbian_strength * score[update_mask]
                
                # Clip connection strength
                self.connection_strength = torch.clamp(
                    self.connection_strength, 
                    -self.config.connection_clamp,
                    self.config.connection_clamp
                )

            # Add connection strength to score (memory effect)
            if causal:
                memory_effect = 0.3 * self.connection_strength[:, :seq_len, :seq_len] * mask
            else:
                memory_effect = 0.3 * self.connection_strength[:, :seq_len, :seq_len]
            
            score += memory_effect
            score = torch.clamp(score, 0.0, 1.0)
        
        # Metrics
        metrics = {
            'sparsity': valid_connections.float().mean().item(),
            'mean_score': score[score > 0].mean().item() if (score > 0).any() else 0.0,
            'hebbian_strength': self.connection_strength[:, :seq_len, :seq_len].mean().item(),
            'mean_radius': radii.mean().item()
        }
        
        return score, metrics
