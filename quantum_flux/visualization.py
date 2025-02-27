"""
Quantum Flux Visualization Module
==============================

This module provides visualization utilities for the Quantum Flux model,
including functions for plotting quantum states, attention patterns,
and training metrics.

The visualizations help to understand:
1. How tokens are arranged in the quantum state space
2. How attention patterns develop
3. How states evolve through the layers
4. How training metrics change over time

Theoretical Background:
---------------------
Visualizing quantum states and their evolution helps to understand
the physics-inspired dynamics of the model:

1. Polar plots show the magnitude (radius) and phase (angle) of quantum states

2. Attention visualizations reveal how information flows between tokens

3. Layer-wise visualizations show how states transform through the network

4. Training curves illustrate how the system energy (loss) decreases over time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualize_quantum_states(states, title="Quantum States", show_unit_circle=True, colormap='viridis'):
    """
    Visualize quantum states in polar coordinates.
    
    Parameters:
    ----------
    states : torch.Tensor or numpy.ndarray
        Quantum states with shape (..., 2)
    title : str
        Plot title
    show_unit_circle : bool
        Whether to show the unit circle
    colormap : str
        Matplotlib colormap name
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure object
        
    Physics interpretation:
    ---------------------
    This visualization shows tokens as points in a 2D quantum state space where:
    - Distance from origin (radius) represents token importance/amplitude
    - Angle represents token phase/semantic meaning
    
    This is analogous to how quantum particles are represented in phase space.
    """
    # Validate input
    if states is None:
        raise ValueError("states cannot be None")
    
    # Convert to numpy if tensor
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    
    # Ensure states is 2D
    if states.ndim == 1:
        states = states.reshape(1, -1)
    
    # Validate shape
    if states.shape[-1] < 2:
        raise ValueError(f"Last dimension of states must be at least 2, got {states.shape[-1]}")
    
    # Extract x and y coordinates
    rx = states[..., 0]
    ry = states[..., 1]
    
    # Calculate radius and angle
    radius = np.sqrt(rx**2 + ry**2)
    angles = np.arctan2(ry, rx)
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot reference unit circle if requested
    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
    
    # Plot states with varying radius
    scatter = ax.scatter(angles, radius, c=np.arange(len(states)), cmap=colormap, 
                         s=100, alpha=0.7)
    
    # Add colorbar for token sequence
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Token Position')
    
    # Add token indices as labels
    for i, (angle, r) in enumerate(zip(angles, radius)):
        ax.text(angle, r+0.1, str(i), fontsize=8)
    
    # Set title with y parameter to position it higher
    ax.set_title(title, fontsize=16, y=1.1)
    
    # Adjust figure to make room for the title
    plt.subplots_adjust(top=0.85)
    
    return fig


def visualize_attention(attention_scores, title="Attention Pattern"):
    """
    Visualize attention scores as a heatmap.
    
    Parameters:
    ----------
    attention_scores : torch.Tensor or numpy.ndarray
        Attention scores with shape (seq_len, seq_len)
    title : str
        Plot title
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure object
        
    Physics interpretation:
    ---------------------
    The attention heatmap shows interaction strengths between tokens,
    similar to how interaction potentials between particles are visualized
    in physics. Stronger interactions (brighter colors) indicate
    stronger coupling between tokens.
    """
    # Validate input
    if attention_scores is None:
        raise ValueError("attention_scores cannot be None")
    
    # Convert to numpy if tensor
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.detach().cpu().numpy()
    
    # Validate shape
    if len(attention_scores.shape) != 2:
        raise ValueError(f"attention_scores must be 2D, got shape {attention_scores.shape}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate blue heatmap
    cmap = LinearSegmentedColormap.from_list(
        'blue_gradient', [(0, 'white'), (1, 'blue')]
    )
    
    # Plot heatmap
    im = ax.imshow(attention_scores, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Strength')
    
    # Set axis labels
    ax.set_xlabel('Token Position (Destination)')
    ax.set_ylabel('Token Position (Source)')
    
    # Add grid
    ax.grid(False)
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    # Add tick marks
    seq_len = attention_scores.shape[0]
    if seq_len <= 20:  # Only add ticks for reasonably sized sequences
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(np.arange(seq_len))
        ax.set_yticklabels(np.arange(seq_len))
    
    return fig


def visualize_training_metrics(metrics_history, title="Training Metrics"):
    """
    Visualize training metrics over time.
    
    Parameters:
    ----------
    metrics_history : list
        List of dictionaries containing metrics
    title : str
        Plot title
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure object
        
    Physics interpretation:
    ---------------------
    Training curves show how the system energy (loss) decreases over time,
    similar to how physical systems evolve towards minimum energy states.
    Other metrics reveal properties like:
    - Coherence: How ordered/disordered the system is
    - Radius: The scale of token representations
    - Hebbian strength: How connections develop over time
    """
    # Validate input
    if not metrics_history or not isinstance(metrics_history, list):
        raise ValueError("metrics_history must be a non-empty list of dictionaries")
    
    # Extract metrics
    metrics = {}
    for key in metrics_history[0].keys():
        metrics[key] = [m.get(key, 0) for m in metrics_history]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot loss if available
    if 'loss' in metrics:
        axes[0].plot(metrics['loss'], 'r-', label='Total Loss')
        if 'ce_loss' in metrics:
            axes[0].plot(metrics['ce_loss'], 'b--', label='CE Loss')
        if 'geometric_loss' in metrics:
            axes[0].plot(metrics['geometric_loss'], 'g--', label='Geometric Loss')
        axes[0].set_title('Loss Evolution')
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Plot radius metrics if available
    radius_keys = [k for k in metrics.keys() if 'radius' in k]
    if radius_keys:
        for key in radius_keys:
            axes[1].plot(metrics[key], label=key)
        axes[1].set_title('Radius Evolution')
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Radius')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    # Plot attention metrics if available
    if 'sparsity' in metrics or 'mean_score' in metrics:
        if 'sparsity' in metrics:
            axes[2].plot(metrics['sparsity'], 'b-', label='Sparsity')
        if 'mean_score' in metrics:
            axes[2].plot(metrics['mean_score'], 'g-', label='Mean Score')
        axes[2].set_title('Attention Metrics')
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    
    # Plot hebbian metrics if available
    if 'hebbian_strength' in metrics:
        axes[3].plot(metrics['hebbian_strength'], 'r-')
        axes[3].set_title('Hebbian Connection Strength')
        axes[3].set_xlabel('Training Steps')
        axes[3].set_ylabel('Strength')
        axes[3].grid(alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Use manual layout adjustment instead of tight_layout
    # This avoids warnings with polar plots
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.92, hspace=0.35, wspace=0.35)
    
    return fig


def visualize_state_evolution(all_states, title="Quantum State Evolution"):
    """
    Visualize how quantum states evolve through the layers.
    
    Parameters:
    ----------
    all_states : list
        List of torch.Tensor or numpy.ndarray with quantum states
    title : str
        Plot title
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure object
        
    Physics interpretation:
    ---------------------
    This visualization shows how quantum states evolve through the network,
    similar to how particle trajectories are plotted in physics simulations.
    Each layer transformation can be viewed as a time step in the evolution
    of the quantum system.
    """
    # Validate input
    if not all_states or not isinstance(all_states, list):
        raise ValueError("all_states must be a non-empty list of tensors or arrays")
    
    num_layers = len(all_states)
    num_tokens = all_states[0].shape[0]
    
    # Convert to numpy if tensors
    states_np = []
    for states in all_states:
        if isinstance(states, torch.Tensor):
            states_np.append(states.detach().cpu().numpy())
        else:
            states_np.append(states)
    
    # Create figure with subplots - add extra width for colorbar
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers*5 + 1, 5), 
                            subplot_kw={'projection': 'polar'})
    
    # Handle single layer case
    if num_layers == 1:
        axes = [axes]
    
    # Plot each layer's states
    for i, (ax, states) in enumerate(zip(axes, states_np)):
        # Extract coordinates
        rx = states[:, 0]
        ry = states[:, 1]
        
        # Calculate radius and angle
        radius = np.sqrt(rx**2 + ry**2)
        angles = np.arctan2(ry, rx)
        
        # Plot reference unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
        
        # Plot states
        scatter = ax.scatter(angles, radius, c=np.arange(num_tokens), 
                           cmap='viridis', s=100, alpha=0.7)
        
        # Set title with y parameter to position it higher
        layer_name = f"Layer {i}" if i > 0 else "Input"
        ax.set_title(layer_name, y=1.1)
    
    # Add colorbar for the last subplot
    # Position it at the right edge of the figure with proper padding
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Token Position')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Use manual layout adjustment instead of tight_layout
    # This avoids warnings with polar plots
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    
    return fig
