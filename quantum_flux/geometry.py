"""
Quantum Geometry Module
======================

Implements geometric transformations between polar and Cartesian coordinates,
as well as utility functions for geometric operations used in the quantum flux model.

The module provides the mathematical foundation for representing tokens in a
polar coordinate system, which allows for more meaningful geometric relationships
between embeddings by encoding both magnitude (radius) and direction (angle).

Theoretical Background:
---------------------
In quantum mechanics, complex wavefunctions are often represented in polar form:
    ψ(r,θ) = R(r)e^(iθ)

Where:
- R(r) is the radial component (amplitude)
- θ is the phase angle
- r is the radial distance

In our neural network context:
- Tokens are embedded as points with (r,θ) coordinates
- The radial component (r) can represent token importance/relevance
- The angular component (θ) can represent semantic relationships

Key mathematical relationships:
- (rx, ry) = (r·cos(θ), r·sin(θ))  [Polar to Cartesian]
- r = sqrt(rx² + ry²)              [Radius calculation]
- θ = atan2(ry, rx)                [Angle calculation]
"""

import torch
import math


def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates (r, theta) to Cartesian coordinates (rx, ry).
    
    In the quantum embedding space, this transforms from magnitude-phase
    representation to a 2D vector representation that can be processed by
    neural network layers.
    
    Parameters:
    ----------
    r : torch.Tensor or float
        Radial distance, typically in [0,1] or [0, some max]
    theta : torch.Tensor or float
        Angle in radians, typically in [0, 2π)
        
    Returns:
    -------
    rx, ry : torch.Tensor or float
        Cartesian coordinates
        
    Physics interpretation:
    ---------------------
    This conversion maps quantum state magnitude (r) and phase (theta) to
    real and imaginary parts of a complex wavefunction:
        ψ = r·e^(iθ) = r·cos(θ) + i·r·sin(θ) = rx + i·ry
    """
    if isinstance(r, torch.Tensor):
        rx = r * torch.cos(theta)
        ry = r * torch.sin(theta)
    else:
        rx = r * math.cos(theta)
        ry = r * math.sin(theta)
    return rx, ry


def cartesian_to_polar(rx, ry):
    """
    Convert Cartesian coordinates (rx, ry) to polar coordinates (r, theta).
    
    This transformation allows us to interpret token embeddings in terms
    of magnitude (importance) and direction (semantic meaning).
    
    Parameters:
    ----------
    rx : torch.Tensor or float
        X coordinate
    ry : torch.Tensor or float
        Y coordinate
        
    Returns:
    -------
    r : torch.Tensor or float
        Radial distance (magnitude)
    theta : torch.Tensor or float
        Angle in radians [0, 2π)
        
    Physics interpretation:
    ---------------------
    This extracts the magnitude and phase from a complex wavefunction:
        r = |ψ| = sqrt(rx² + ry²)
        θ = arg(ψ) = atan2(ry, rx)
    """
    if isinstance(rx, torch.Tensor):
        r = torch.sqrt(rx * rx + ry * ry)
        theta = torch.atan2(ry, rx)
        # Ensure theta is in [0, 2π)
        theta = torch.where(theta < 0, theta + 2.0 * math.pi, theta)
    else:
        r = math.sqrt(rx * rx + ry * ry)
        theta = math.atan2(ry, rx)
        # Ensure theta is in [0, 2π)
        if theta < 0:
            theta += 2.0 * math.pi
    return r, theta


def normalize_embeddings(embs):
    """
    Normalize embeddings to unit vectors.
    
    Parameters:
    ----------
    embs : torch.Tensor
        Embeddings tensor of shape (batch_size, seq_len, embed_dim)
        
    Returns:
    -------
    torch.Tensor
        Normalized embeddings
        
    Physics interpretation:
    ---------------------
    In quantum mechanics, wavefunctions are normalized so that
    the probability of finding the particle somewhere in space is 1:
        ∫|ψ|² dr = 1
        
    Here, we normalize each embedding vector to unit length, which
    preserves directional information while standardizing magnitudes.
    """
    if isinstance(embs, torch.Tensor):
        # Safe division with small epsilon
        norms = torch.norm(embs, dim=-1, keepdim=True)
        return torch.where(norms > 1e-12, embs / norms, embs)
    else:
        # Handle numpy arrays
        import numpy as np
        normed = np.copy(embs)
        norms = np.linalg.norm(normed, axis=-1, keepdims=True)
        mask = norms > 1e-12
        normed[mask] = normed[mask] / norms[mask]
        return normed


def negative_distance_matrix(embs):
    """
    Compute negative Euclidean distance matrix between all pairs of embeddings.
    
    Parameters:
    ----------
    embs : torch.Tensor
        Embeddings tensor of shape (batch_size, seq_len, embed_dim)
        
    Returns:
    -------
    torch.Tensor
        Negative distance matrix of shape (batch_size, seq_len, seq_len)
        
    Physics interpretation:
    ---------------------
    This computes a similarity matrix based on spatial proximity, similar to
    a potential energy matrix in physics where:
        - Lower potential (more negative) for more distant particles
        - Higher potential (less negative) for closer particles
        
    This can be interpreted as "particles that are close have stronger interactions".
    """
    if isinstance(embs, torch.Tensor):
        # Compute squared norm of each embedding
        x_norm = torch.sum(embs**2, dim=-1, keepdim=True)
        
        # Compute pairwise distances using the identity:
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*<x,y>
        y_norm = x_norm.transpose(-1, -2)
        dist = x_norm + y_norm - 2.0 * torch.bmm(embs, embs.transpose(-1, -2))
        
        # Ensure no negative values due to numerical issues
        dist = torch.clamp(dist, min=0.0)
        return -torch.sqrt(dist + 1e-12)  # Negative distance
    else:
        # Handle numpy arrays
        import numpy as np
        D = embs.shape[0]
        M = np.zeros((D, D), dtype=np.float32)
        for i in range(D):
            for j in range(D):
                diff = embs[i] - embs[j]
                dist = np.linalg.norm(diff)
                M[i, j] = -dist
        return M


def normalize_matrix(M, min_val=None, max_val=None):
    """
    Normalize matrix values to the range [0, 1] using min-max scaling.
    
    Parameters:
    ----------
    M : torch.Tensor
        Input matrix
    min_val : float or None
        Minimum value for scaling. If None, uses M.min()
    max_val : float or None
        Maximum value for scaling. If None, uses M.max()
        
    Returns:
    -------
    torch.Tensor
        Normalized matrix
        
    Physics interpretation:
    ---------------------
    This is analogous to rescaling a potential field to a specified range,
    making it easier to interpret and compare relative values.
    """
    if isinstance(M, torch.Tensor):
        m_min = M.min() if min_val is None else min_val
        m_max = M.max() if max_val is None else max_val
        
        if torch.abs(m_max - m_min) < 1e-12:
            return torch.ones_like(M)
        
        return (M - m_min) / (m_max - m_min)
    else:
        # Handle numpy arrays
        import numpy as np
        m_min = M.min() if min_val is None else min_val
        m_max = M.max() if max_val is None else max_val
        
        if abs(m_max - m_min) < 1e-12:
            return np.ones_like(M)
        
        return (M - m_min) / (m_max - m_min)


def sentence_to_polar_embeddings(sentence_length, embedding_dim, device=None):
    """
    Create polar embeddings for a sequence of tokens.
    
    Parameters:
    ----------
    sentence_length : int
        Length of the sequence
    embedding_dim : int
        Dimension of embedding space
    device : torch.device or None
        Device to place tensors on
        
    Returns:
    -------
    torch.Tensor
        Embeddings tensor of shape (sentence_length, embedding_dim)
        
    Physics interpretation:
    ---------------------
    This arranges tokens in a circular pattern where:
    - Each token has a unique angle θ_i = 2π·(i/D)
    - Radial distance increases with token position: r_i = 0.3 + 0.6·(i/(D-1))
    
    This creates a spiral-like pattern that encodes sequence order geometrically,
    similar to how particles in a vortex follow a spiral pattern with both
    angular and radial components of motion.
    """
    embs = torch.zeros((sentence_length, embedding_dim), device=device)
    
    # Skip if sentence is empty
    if sentence_length == 0:
        return embs
        
    # Compute theta and r for each position
    positions = torch.arange(sentence_length, device=device, dtype=torch.float32)
    theta = 2.0 * math.pi * positions / sentence_length
    
    # Calculate radius - increasing from 0.3 to 0.9
    if sentence_length > 1:
        r = 0.3 + 0.6 * positions / (sentence_length - 1)
    else:
        r = torch.tensor([0.3], device=device)
    
    # Convert to Cartesian coordinates
    rx = r * torch.cos(theta)
    ry = r * torch.sin(theta)
    
    # Place in first two dimensions
    embs[:, 0] = rx
    embs[:, 1] = ry
    # The rest (embedding_dim - 2) remain zeros
    
    return embs
