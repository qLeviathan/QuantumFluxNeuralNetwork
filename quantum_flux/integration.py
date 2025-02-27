"""
Quantum Integration Module
=======================

This module implements numerical integration methods for evolving quantum states
through time, based on physical equations from quantum mechanics and diffusion theory.

The integrator implements a Heun-Euler method (a second-order Runge-Kutta method)
for solving the differential equations that govern quantum state evolution.

Theoretical Background:
---------------------
In quantum mechanics, the time evolution of a wavefunction is governed by:
    i·ℏ·∂ψ/∂t = H·ψ

In our neural network context, we use an imaginary-time Schrödinger equation:
    ∂ψ/∂τ = -H·ψ

Which has mathematical similarities to diffusion equations:
    ∂u/∂t = α·∇²u

Our integration scheme combines these ideas with:
1. Heun-Euler integration for numerical stability
2. Adaptive time stepping based on system dimensions
3. Radius adjustment for token importance evolution
"""

import torch
import torch.nn as nn
from .attention import QuantumAttention


class QuantumIntegrator(nn.Module):
    """
    Quantum integrator with latent radius support.
    
    This module evolves quantum states through time using numerical
    integration methods inspired by quantum mechanics and diffusion equations.
    
    Attributes:
    ----------
    config : QuantumFluxConfig
        Configuration object with model parameters
    device : torch.device
        Device where computations are performed
    attention : QuantumAttention
        Attention mechanism for computing state interactions
        
    Physics interpretation:
    ---------------------
    The integrator solves differential equations that describe how
    quantum states evolve over time, using:
    
    1. Heun-Euler method (second-order Runge-Kutta) for numerical stability
       dy/dt = f(y) → y_{n+1} = y_n + 0.5·dt·(k1 + k2)
       where k1 = f(y_n) and k2 = f(y_n + dt·k1)
       
    2. Adaptive time stepping based on system dimensions
       dt ~ 1/Beta(embed_dim/2, seq_len/2)
       
    3. Radius adjustment to control token importance
       r_new = r_old·scale_factor
    """
    
    def __init__(self, config):
        """
        Initialize the quantum integrator with configuration parameters.
        
        Parameters:
        ----------
        config : QuantumFluxConfig
            Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = QuantumAttention(config)
    
    def sample_timestep(self, batch_size):
        """
        Sample integration timestep from distribution.
        
        Parameters:
        ----------
        batch_size : int
            Number of time steps to sample
            
        Returns:
        -------
        torch.Tensor
            Sampled time steps of shape (batch_size,)
            
        Physics interpretation:
        ---------------------
        Adaptive time stepping is crucial in physical simulations to balance
        accuracy and efficiency. The time step is sampled from a Beta distribution:
        
        - α = (embed_dim-2)/2: Higher dimensional systems need smaller time steps
        - β = (seq_len-1)/2: Longer sequences need smaller time steps
        
        The -2 and -1 adjustments account for the fact that we need at least
        2 dimensions for the quantum state and 1 token in the sequence.
        """
        a = torch.tensor(float(self.config.embed_dim - 2) / 2.0, device=self.device)
        b = torch.tensor(float(self.config.max_seq_length - 1) / 2.0, device=self.device)
        
        x = torch.distributions.Beta(a, b).sample((batch_size,)).to(self.device)
        
        dt = (1.0 / x.clamp(min=1e-3, max=0.999)) * self.config.dt_scale
        dt = torch.clamp(dt, min=1e-6, max=self.config.dt_max)
        
        return dt
    
    def forward(self, r_embed, causal=True, dt=None):
        """
        Perform one integration step with latent radius using Heun-Euler method.
        
        Parameters:
        ----------
        r_embed : torch.Tensor
            Quantum state embeddings, shape (batch_size, seq_len, 2)
        causal : bool
            Whether to apply causal masking in attention
        dt : torch.Tensor or None
            Time step. If None, will be sampled automatically
            
        Returns:
        -------
        r_final : torch.Tensor
            Updated quantum states after integration
        score : torch.Tensor
            Attention scores
        metrics : dict
            Metrics about the integration process
            
        Physics interpretation:
        ---------------------
        This implements numerical integration of the quantum state evolution equation:
        
        1. First stage (k1): Calculate initial state derivatives based on current positions
        
        2. Intermediate state: r_mid = r_embed + dt·k1
        
        3. Second stage (k2): Calculate derivatives at the intermediate state
        
        4. Final update: r_new = r_embed + 0.5·dt·(k1 + k2)
        
        5. Radius adjustment: scale the vectors to maintain proper token importance
        """
        batch_size = r_embed.shape[0]
        
        # Sample timestep if not provided
        if dt is None:
            dt = self.sample_timestep(batch_size)
        
        # Calculate radius for metrics only (avoid excessive calculations)
        radii = torch.norm(r_embed, dim=-1)
        
        # First step (k1)
        score, metrics = self.attention.compute_score(r_embed, causal=causal)
        
        # Compute context vectors (weighted sum)
        context = torch.bmm(score, r_embed)
        
        # Scale by timestep
        dt_expanded = dt.view(batch_size, 1, 1)
        k1 = dt_expanded * context
        
        # Handle numerical stability
        k1 = torch.where(torch.isnan(k1) | torch.isinf(k1), torch.zeros_like(k1), k1)
        
        # Intermediate state
        r_mid = r_embed + k1
        
        # Handle numerical stability
        r_mid = torch.where(torch.isnan(r_mid) | torch.isinf(r_mid), r_embed, r_mid)
        
        # Second step (k2)
        score_mid, _ = self.attention.compute_score(r_mid, causal=causal)
        context_mid = torch.bmm(score_mid, r_mid)
        k2 = dt_expanded * context_mid
        
        # Handle numerical stability
        k2 = torch.where(torch.isnan(k2) | torch.isinf(k2), torch.zeros_like(k2), k2)
        
        # Final state (Heun-Euler integration)
        r_new = r_embed + 0.5 * (k1 + k2)
        
        # Calculate current radius
        new_radius = torch.norm(r_new, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # Compute radius adjustment from connection influence
        radius_adjustment = torch.sum(score, dim=-1, keepdim=True) * self.config.radius_update_rate
        
        # Calculate target radius (within bounds)
        adjusted_radius = (new_radius + radius_adjustment).clamp(
            self.config.min_radius, self.config.max_radius
        )
        
        # Calculate scaling factor
        scale_factor = adjusted_radius / new_radius
        
        # Apply scaling directly to vectors (no angle calculation needed)
        r_final = r_new * scale_factor
        
        # Handle numerical issues
        if torch.isnan(r_final).any() or torch.isinf(r_final).any():
            r_final = torch.where(torch.isnan(r_final) | torch.isinf(r_final), r_embed, r_final)
            metrics['integration_unstable'] = 1.0
        else:
            metrics['integration_unstable'] = 0.0
        
        # Track radius statistics
        final_radius = torch.norm(r_final, dim=-1)
        metrics['final_radius_mean'] = final_radius.mean().item()
        metrics['final_radius_std'] = final_radius.std().item()
        
        # Calculate geometric coherence using direct dot product (no angles)
        # Use batch matrix multiplication for efficiency
        r_normed = r_final / torch.norm(r_final, dim=-1, keepdim=True).clamp(min=1e-6)
        coherence = torch.bmm(r_normed, r_normed.transpose(1, 2)).mean()
        metrics['geometric_coherence'] = coherence.item()
        
        return r_final, score, metrics
    
    def radial_diffusion_step(self, u, r_grid, alpha, dt):
        """
        Perform a radial diffusion step.
        
        Parameters:
        ----------
        u : torch.Tensor
            Radial function values
        r_grid : torch.Tensor
            Radial grid points
        alpha : float
            Diffusion coefficient
        dt : float
            Time step
            
        Returns:
        -------
        torch.Tensor
            Updated values after diffusion step
            
        Physics interpretation:
        ---------------------
        This implements the radial diffusion equation:
            ∂u/∂t = α·[1/r·∂/∂r(r·∂u/∂r)]
            
        This equation describes how a quantity (like heat or concentration)
        spreads out over time in a system with radial symmetry, which is
        mathematically similar to quantum state evolution in certain cases.
        """
        # Calculate dr
        dr = r_grid[1] - r_grid[0]
        
        # Initialize derivative
        du_dt = torch.zeros_like(u)
        
        # Loop over interior points
        for i in range(1, len(r_grid)-1):
            r = r_grid[i]
            
            # Finite difference approximation
            d_u_dr_plus = (u[i+1] - u[i]) / dr
            d_u_dr_minus = (u[i] - u[i-1]) / dr
            
            # Radial flux
            flux_plus = r * d_u_dr_plus
            flux_minus = r * d_u_dr_minus
            
            # Update using the diffusion equation
            du_dt[i] = alpha * (1.0 / r) * (flux_plus - flux_minus) / dr
        
        # Update u using forward Euler
        u_new = u + dt * du_dt
        
        return u_new
    
    def imaginary_time_schrodinger_step(self, u, r_grid, alpha, dt, potential=None):
        """
        Perform an imaginary-time Schrödinger equation step.
        
        Parameters:
        ----------
        u : torch.Tensor
            Wavefunction values
        r_grid : torch.Tensor
            Radial grid points
        alpha : float
            Coefficient (related to ℏ/2m)
        dt : float
            Time step
        potential : callable or None
            Potential energy function V(r)
            
        Returns:
        -------
        torch.Tensor
            Updated wavefunction after step
            
        Physics interpretation:
        ---------------------
        This implements the imaginary-time Schrödinger equation:
            ∂ψ/∂τ = α·[1/r·∂/∂r(r·∂ψ/∂r)] - V(r)·ψ
            
        This equation is used in quantum mechanics to find the ground
        state of a system, by evolving an initial state in imaginary time
        until it converges to the lowest energy state.
        """
        # Default potential is zero (free particle)
        if potential is None:
            potential = lambda r: torch.zeros_like(r)
        
        # Diffusion part
        u_diffusion = self.radial_diffusion_step(u, r_grid, alpha, dt)
        
        # Potential part
        u_potential = u - dt * potential(r_grid) * u
        
        # Combine (approximation of operator splitting)
        u_new = 0.5 * (u_diffusion + u_potential)
        
        return u_new
