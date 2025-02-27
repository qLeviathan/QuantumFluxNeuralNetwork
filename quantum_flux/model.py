"""
Quantum Flux Model Module
======================

This module implements the complete Quantum Flux neural network model,
combining all components into a coherent architecture.

The model implements a transformer-like architecture with quantum-inspired
components, including token encoding, quantum state evolution, and
Hebbian learning mechanisms.

Theoretical Background:
---------------------
The Quantum Flux model draws inspiration from several physics concepts:

1. Quantum mechanics: Tokens are represented as quantum states with
   magnitude (radius) and phase (angle)

2. Diffusion equations: States evolve according to physics-inspired
   differential equations

3. Hebbian learning: Connections strengthen based on co-activation patterns

4. Geometric integration: Numerical methods preserve important geometric
   properties of the system

Unlike traditional transformers, this model:
- Uses direct geometric relationships instead of attention scores
- Evolves states through integration of physical equations
- Updates weights through Hebbian learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import QuantumEncoder
from .layers import QuantumLayer, OutputProjection


class QuantumFluxModel(nn.Module):
    """
    Complete Quantum Flux model with latent radius.
    
    This model implements a transformer-like architecture with
    quantum-inspired components for natural language processing.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    device : torch.device
        Device where computations are performed
    encoder : QuantumEncoder
        Token encoding module
    layers : nn.ModuleList
        List of quantum layers
    output_proj : OutputProjection
        Output projection module
        
    Physics interpretation:
    ---------------------
    The model implements a quantum-inspired neural network where:
    
    1. Tokens are encoded as quantum states with magnitude and phase
    
    2. States evolve through physical equations in each layer
    
    3. Evolution follows principles from quantum mechanics and diffusion theory
    
    4. Information flows causally (past to future) in the attention mechanism
    
    5. Weights adapt through Hebbian learning principles
    """
    
    def __init__(self, config):
        """
        Initialize the Quantum Flux model with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Quantum token encoding with latent radius
        self.encoder = QuantumEncoder(config)
        
        # Quantum layers
        self.layers = nn.ModuleList([
            QuantumLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = OutputProjection(config)
    
    def forward(self, input_ids, targets=None, is_training=True, causal=True):
        """
        Forward pass with latent radius.
        
        Parameters:
        ----------
        input_ids : torch.Tensor
            Token indices, shape (batch_size, seq_len)
        targets : torch.Tensor or None
            Target token indices for loss computation
        is_training : bool
            Whether the model is in training mode
        causal : bool
            Whether to apply causal masking in attention
            
        Returns:
        -------
        dict
            Dictionary containing model outputs and metrics
            
        Physics interpretation:
        ---------------------
        The forward pass implements:
        
        1. Token encoding into quantum states
        
        2. Sequential processing through quantum layers, where each layer:
           - Evolves states according to physical equations
           - Projects to higher dimensions
           - Normalizes for stability
           
        3. Loss computation combines:
           - Cross-entropy loss (information-theoretic)
           - Geometric loss (from physics principles)
           
        4. Hebbian weight updates during training
        """
        # Encode tokens to quantum states with radius
        quantum_states = self.encoder.encode(input_ids)
        
        # Process through quantum layers
        all_layer_metrics = []
        all_quantum_states = [quantum_states]
        
        current_states = quantum_states
        current_embeddings = None
        
        # Track geometric loss
        geometric_loss = 0.0
        
        for layer in self.layers:
            # Process through layer
            current_states, current_embeddings, layer_metrics = layer(
                current_states, 
                causal=causal
            )
            
            # Track states
            all_quantum_states.append(current_states)
            
            # Track geometric coherence loss
            if 'geometric_coherence' in layer_metrics:
                geometric_loss += 1.0 - layer_metrics['geometric_coherence']
            
            all_layer_metrics.append(layer_metrics)
        
        # Final output projection
        logits = self.output_proj(current_embeddings)
        
        # Compute loss if targets provided
        loss = None
        ce_loss = None
        if targets is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            
            # Combined loss (cross-entropy + geometric)
            loss = ce_loss + 0.01 * geometric_loss
            
            # Update output projection weights if in training mode
            if is_training:
                # Flatten tensors for update
                valid_mask = targets.view(-1) != -100
                if valid_mask.any():
                    valid_inputs = current_embeddings.view(-1, current_embeddings.size(-1))[valid_mask]
                    valid_targets = targets.view(-1)[valid_mask]
                    
                    # Update weights without backpropagation
                    update_metrics = self.output_proj.update_weights(
                        valid_inputs, 
                        valid_targets, 
                        self.config.learning_rate
                    )
                    all_layer_metrics.append(update_metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_layer_metrics[0].keys():
            if isinstance(all_layer_metrics[0][key], (int, float)):
                avg_metrics[key] = sum(m.get(key, 0) for m in all_layer_metrics) / len(all_layer_metrics)
        
        # Add loss metrics
        if loss is not None:
            avg_metrics['loss'] = loss.item()
            avg_metrics['ce_loss'] = ce_loss.item()
            avg_metrics['geometric_loss'] = geometric_loss if isinstance(geometric_loss, float) else geometric_loss.item()
            avg_metrics['perplexity'] = torch.exp(torch.clamp(ce_loss, 0, 20)).item()
        
        # Results
        return {
            'logits': logits,
            'loss': loss,
            'ce_loss': ce_loss,
            'geometric_loss': geometric_loss,
            'quantum_states': current_states,
            'all_quantum_states': all_quantum_states,
            'hidden_states': current_embeddings,
            'metrics': avg_metrics,
            'all_layer_metrics': all_layer_metrics,
        }
    
    @torch.inference_mode()
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50):
        """
        Generate text auto-regressively.
        
        Parameters:
        ----------
        input_ids : torch.Tensor
            Initial token indices, shape (batch_size, seq_len)
        max_length : int
            Maximum length of generated sequence
        temperature : float
            Temperature for sampling (higher = more random)
        top_k : int
            Number of top tokens to consider for sampling
            
        Returns:
        -------
        torch.Tensor
            Generated token indices
            
        Physics interpretation:
        ---------------------
        Text generation can be viewed as a physical process where:
        
        1. The system evolves from an initial state (prompt)
        
        2. Each step follows a trajectory influenced by:
           - Current state (context)
           - Learned dynamics (model weights)
           
        3. Randomness from sampling introduces quantum-like uncertainty
        
        4. Temperature controls the "energy" of the system:
           - High temperature: More random exploration (high energy)
           - Low temperature: More deterministic (low energy)
        """
        # Start with the provided context
        generated = input_ids.clone()
        
        # Generate tokens auto-regressively
        for _ in range(max_length - generated.size(1)):
            # Use only the last tokens if context is too long
            if generated.size(1) > self.config.max_seq_length:
                context = generated[:, -self.config.max_seq_length:]
            else:
                context = generated
            
            # Forward pass
            outputs = self.forward(
                context, 
                is_training=False
            )
            logits = outputs['logits']
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
