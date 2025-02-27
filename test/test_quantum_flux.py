"""
Quantum Flux Test Script
======================

This script tests the functionality of the Quantum Flux model,
including individual components and the full model pipeline.

The tests include:
1. Geometric transformations
2. Encoder functionality
3. Attention mechanism
4. Integration methods
5. Full model training and generation
6. Visualization utilities

This serves as both a validation tool and a usage example.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_flux.config import QuantumFluxConfig
from quantum_flux.geometry import polar_to_cartesian, cartesian_to_polar, normalize_embeddings
from quantum_flux.encoder import QuantumEncoder
from quantum_flux.attention import QuantumAttention
from quantum_flux.integration import QuantumIntegrator
from quantum_flux.layers import QuantumLayer
from quantum_flux.model import QuantumFluxModel
from quantum_flux.visualization import (
    visualize_quantum_states, 
    visualize_attention,
    visualize_training_metrics,
    visualize_state_evolution
)


def test_geometry():
    """Test geometric transformations."""
    print("\n=== Testing Geometry Module ===")
    
    # Test polar to Cartesian
    r = torch.tensor([1.0, 2.0, 0.5])
    theta = torch.tensor([0.0, np.pi/2, np.pi])
    rx, ry = polar_to_cartesian(r, theta)
    
    print(f"Polar to Cartesian:")
    print(f"r = {r}, theta = {theta}")
    print(f"rx = {rx}, ry = {ry}")
    
    # Test Cartesian to polar
    r_new, theta_new = cartesian_to_polar(rx, ry)
    
    print(f"Cartesian to Polar:")
    print(f"rx = {rx}, ry = {ry}")
    print(f"r = {r_new}, theta = {theta_new}")
    
    # Test normalization
    embs = torch.randn(5, 3)
    normed = normalize_embeddings(embs)
    norms = torch.norm(normed, dim=1)
    
    print(f"Normalization:")
    print(f"Original norms: {torch.norm(embs, dim=1)}")
    print(f"Normalized norms: {norms}")
    
    return "Geometry tests passed!"


def test_encoder():
    """Test quantum encoder."""
    print("\n=== Testing Encoder Module ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=64,
        vocab_size=100,
        phase_shift=np.pi/4
    )
    
    # Create encoder
    encoder = QuantumEncoder(config)
    
    # Test encoding
    tokens = torch.randint(0, config.vocab_size, (2, 10))
    encoded = encoder.encode(tokens)
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded sample: {encoded[0, 0]}")
    
    # Test radii and angles
    radii = torch.norm(encoded, dim=-1)
    print(f"Encoded radii range: {radii.min().item():.3f} to {radii.max().item():.3f}")
    
    return "Encoder tests passed!"


def test_attention():
    """Test attention mechanism."""
    print("\n=== Testing Attention Module ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=64,
        max_seq_length=16,
        hebbian_decay=0.99,
        hebbian_strength=0.1
    )
    
    # Create attention
    attention = QuantumAttention(config)
    
    # Create random quantum states
    batch_size = 2
    seq_len = 5
    r_embed = torch.randn(batch_size, seq_len, 2)
    
    # Test attention computation
    score, metrics = attention.compute_score(r_embed, causal=True)
    
    print(f"Attention score shape: {score.shape}")
    print(f"Attention metrics: {metrics}")
    
    # Check causal mask
    mask = attention.get_causal_mask(seq_len)
    print(f"Causal mask:\n{mask}")
    
    return "Attention tests passed!"


def test_integrator():
    """Test quantum integrator."""
    print("\n=== Testing Integrator Module ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=64,
        max_seq_length=16,
        dt_scale=1e-5,
        dt_max=0.05,
        min_radius=0.1,
        max_radius=2.0,
        radius_update_rate=0.01
    )
    
    # Create integrator
    integrator = QuantumIntegrator(config)
    
    # Create random quantum states
    batch_size = 2
    seq_len = 5
    r_embed = torch.randn(batch_size, seq_len, 2)
    
    # Test time step sampling
    dt = integrator.sample_timestep(batch_size)
    print(f"Sampled time steps: {dt}")
    
    # Test integration
    r_final, score, metrics = integrator.forward(r_embed, causal=True, dt=dt)
    
    print(f"Original states shape: {r_embed.shape}")
    print(f"Evolved states shape: {r_final.shape}")
    print(f"Score shape: {score.shape}")
    print(f"Integration metrics: {metrics}")
    
    # Check radius adjustment
    orig_radii = torch.norm(r_embed, dim=-1).mean().item()
    final_radii = torch.norm(r_final, dim=-1).mean().item()
    print(f"Original mean radius: {orig_radii:.3f}")
    print(f"Final mean radius: {final_radii:.3f}")
    
    return "Integrator tests passed!"


def test_layer():
    """Test quantum layer."""
    print("\n=== Testing Layer Module ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=64,
        max_seq_length=16,
        skip_beta=5.0
    )
    
    # Create layer
    layer = QuantumLayer(config)
    
    # Create random quantum states
    batch_size = 2
    seq_len = 5
    states = torch.randn(batch_size, seq_len, 2)
    
    # Test layer processing
    final_states, normalized, metrics = layer(states, causal=True)
    
    print(f"Input states shape: {states.shape}")
    print(f"Output states shape: {final_states.shape}")
    print(f"Normalized shape: {normalized.shape}")
    print(f"Layer metrics: {metrics}")
    
    return "Layer tests passed!"


def test_model():
    """Test full model."""
    print("\n=== Testing Full Model ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=32,
        max_seq_length=16,
        num_layers=2,
        vocab_size=100
    )
    
    # Create model
    model = QuantumFluxModel(config)
    
    # Create random input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    outputs = model(input_ids, targets=targets, is_training=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.3f}")
    print(f"Model metrics: {outputs['metrics']}")
    
    # Test generation
    generated = model.generate(input_ids[:, :3], max_length=15, temperature=0.8)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated sample: {generated[0]}")
    
    return "Full model tests passed!"


def train_on_synthetic_data():
    """Train the model on synthetic data for demonstration."""
    print("\n=== Training on Synthetic Data ===")
    
    # Create configuration
    config = QuantumFluxConfig(
        embed_dim=32,
        max_seq_length=16,
        num_layers=2,
        vocab_size=100,
        min_radius=0.1,
        max_radius=2.0,
        radius_update_rate=0.02,
        learning_rate=1e-4
    )
    
    # Create model
    model = QuantumFluxModel(config)
    
    # Training parameters
    batch_size = 8
    seq_length = 16
    num_steps = 50
    
    # Track metrics
    all_metrics = []
    
    # Training loop
    for step in tqdm(range(num_steps), desc="Training"):
        # Generate synthetic data
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        # Simple pattern: predict token n from token n-1
        targets = torch.zeros_like(input_ids)
        targets[:, 1:] = input_ids[:, :-1]
        targets[:, 0] = input_ids[:, -1]
        
        # Add noise for more interesting learning
        mask = torch.rand_like(targets.float()) > 0.3
        targets = torch.where(mask, targets, torch.randint(0, config.vocab_size, targets.shape))
        
        # Forward pass
        outputs = model(input_ids, targets=targets, is_training=True)
        
        # Store metrics
        metrics = outputs['metrics']
        all_metrics.append(metrics)
        
        # Occasional stabilization
        if step % 10 == 0:
            with torch.no_grad():
                # Apply simple stabilization
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param.data = torch.where(
                            torch.isnan(param) | torch.isinf(param),
                            torch.zeros_like(param),
                            param.data.clamp(-5.0, 5.0)
                        )
    
    # Final metrics
    print(f"Final loss: {all_metrics[-1]['loss']:.4f}")
    print(f"Final perplexity: {all_metrics[-1]['perplexity']:.4f}")
    
    # Generate with trained model
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(input_ids, max_length=15)
    
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Visualization
    # Plot training metrics
    fig = visualize_training_metrics(all_metrics, title="Training on Synthetic Data")
    fig.savefig("training_metrics.png")
    
    # Get states and attention for visualization
    with torch.no_grad():
        outputs = model(input_ids)
        
        # Get quantum states and metrics
        states = outputs['quantum_states'][0].cpu().numpy()
        
        # Get all states through layers
        all_states = [s[0].cpu().numpy() for s in outputs['all_quantum_states']]
        
        # Visualize final states with radius
        fig = visualize_quantum_states(states, title="Quantum States with Latent Radius")
        fig.savefig("quantum_states.png")
        
        # Visualize state evolution
        fig = visualize_state_evolution(all_states, title="Quantum State Evolution Through Layers")
        fig.savefig("state_evolution.png")
    
    return model, all_metrics


def test_visualization():
    """Test visualization utilities."""
    print("\n=== Testing Visualization Module ===")
    
    # Create random quantum states
    states = torch.randn(10, 2)
    
    # Test quantum state visualization
    fig = visualize_quantum_states(states, title="Random Quantum States")
    fig.savefig("random_states.png")
    
    # Create random attention scores
    attention = torch.rand(10, 10)
    attention = torch.tril(attention)  # Make it causal
    
    # Test attention visualization
    fig = visualize_attention(attention, title="Random Attention Pattern")
    fig.savefig("random_attention.png")
    
    # Create random metrics
    metrics_history = [
        {'loss': 2.0 - i*0.1, 'ce_loss': 1.8 - i*0.09, 'geometric_loss': 0.2 - i*0.01,
         'sparsity': 0.3 + i*0.01, 'mean_score': 0.5 + i*0.01, 
         'hebbian_strength': 0.1 + i*0.02, 'final_radius_mean': 0.5 + i*0.03}
        for i in range(20)
    ]
    
    # Test metrics visualization
    fig = visualize_training_metrics(metrics_history, title="Random Training Metrics")
    fig.savefig("random_metrics.png")
    
    # Create random state evolution
    all_states = [torch.randn(10, 2) for _ in range(3)]
    
    # Test state evolution visualization
    fig = visualize_state_evolution(all_states, title="Random State Evolution")
    fig.savefig("random_evolution.png")
    
    return "Visualization tests passed!"


def run_all_tests():
    """Run all tests in sequence."""
    print("=== Running All Quantum Flux Tests ===")
    
    start_time = time.time()
    
    try:
        # Run component tests
        test_geometry()
        test_encoder()
        test_attention()
        test_integrator()
        test_layer()
        test_model()
        test_visualization()
        
        # Run training demo
        model, metrics = train_on_synthetic_data()
        
        print("\n=== All Tests Completed Successfully! ===")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        return model, metrics
    except Exception as e:
        print(f"\n=== Error During Tests ===")
        print(f"Error: {str(e)}")
        
        raise


if __name__ == "__main__":
    run_all_tests()
