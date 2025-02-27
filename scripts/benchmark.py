"""
Quantum Flux Benchmarking Script
==============================

This script provides comprehensive benchmarking for the Quantum Flux model,
including throughput, latency, memory usage, and comparison with transformers.

Features:
- Throughput testing with various batch sizes
- Latency measurement for different sequence lengths
- Memory usage profiling
- FLOP counting and efficiency analysis
- Comparison with transformer models of similar size
- Industry standard benchmark metrics

Usage:
```
python scripts/benchmark.py \
    --model_path /path/to/model.pt \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output_file results/benchmark_results.json \
    --compare_transformer
```
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import gc
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel


def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, seq_length, batch_size=1):
    """
    Estimate FLOPs for Quantum Flux model.
    
    Parameters:
    ----------
    model : QuantumFluxModel
        Model to analyze
    seq_length : int
        Sequence length
    batch_size : int
        Batch size
        
    Returns:
    -------
    dict
        FLOP estimates for different components
    """
    config = model.config
    
    # FLOP calculations
    flops = {
        'encoder': 0,
        'attention': 0,
        'integration': 0,
        'projection': 0,
        'layer_norm': 0,
        'output': 0,
        'total': 0
    }
    
    # Encoding FLOPs (embeddings lookup)
    flops['encoder'] = batch_size * seq_length * 2  # Just copy 2D vectors
    
    # Per layer FLOPs
    for _ in range(config.num_layers):
        # Attention FLOPs
        # - Computing dot products: 2 * seq_length^2 * batch_size
        # - Computing attention scores: 3 * seq_length^2 * batch_size
        # - Context vectors: 2 * seq_length^2 * batch_size
        flops['attention'] += 7 * batch_size * seq_length**2
        
        # Integration FLOPs
        # - Heun-Euler steps: 2 * (attention + context calculation)
        # - Radius adjustments: 5 * batch_size * seq_length
        flops['integration'] += 2 * (7 * batch_size * seq_length**2) + 5 * batch_size * seq_length
        
        # Projection FLOPs
        # - Linear from 2 to embed_dim: 2 * embed_dim * batch_size * seq_length
        flops['projection'] += 2 * config.embed_dim * batch_size * seq_length
        
        # Layer Norm FLOPs
        # - Mean, variance, normalization: 4 * embed_dim * batch_size * seq_length
        flops['layer_norm'] += 4 * config.embed_dim * batch_size * seq_length
    
    # Output Projection FLOPs
    # - Linear from embed_dim to vocab_size: embed_dim * vocab_size * batch_size * seq_length
    flops['output'] = config.embed_dim * config.vocab_size * batch_size * seq_length
    
    # Calculate total FLOPs
    flops['total'] = sum(flops.values())
    
    return flops


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the Quantum Flux model from a checkpoint.
    
    Parameters:
    ----------
    model_path : str
        Path to the model checkpoint
    device : str
        Device to load the model on
        
    Returns:
    -------
    tuple
        (model, config, tokenizer)
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get configuration
        config_dict = checkpoint.get('config', None)
        if config_dict is None:
            # Try to load from separate config file
            config_path = Path(model_path).parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError("No configuration found in checkpoint or config.yaml")
        
        # Create configuration
        config = QuantumFluxConfig(**config_dict)
        
        # Create model
        model = QuantumFluxModel(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # Try to load tokenizer if available
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            
            # Check if tokenizer files exist
            tokenizer_path = Path(model_path).parent / "tokenizer"
            if tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Use default GPT-2 tokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            print("Tokenizer loaded successfully")
        except ImportError:
            print("Transformers not available, tokenizer not loaded")
        
        return model, config, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_transformer_model(config, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a transformer model with similar configuration.
    
    Parameters:
    ----------
    config : QuantumFluxConfig
        Configuration to match
    device : str
        Device to load the model on
        
    Returns:
    -------
    tuple
        (model, tokenizer)
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
        
        # Create GPT-2 config
        tf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_seq_length,
            n_embd=config.embed_dim,
            n_layer=config.num_layers,
            n_head=8  # Standard number of heads
        )
        
        # Create model
        model = GPT2LMHeadModel(tf_config)
        
        # Move to device
        model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print(f"Transformer model loaded successfully on {device}")
        
        return model, tokenizer
    
    except ImportError:
        print("Transformers library not available")
        return None, None


def load_dataset(dataset_name, subset=None, split="test", tokenizer=None, max_samples=1000):
    """
    Load dataset for benchmarking.
    
    Parameters:
    ----------
    dataset_name : str
        Dataset name from Hugging Face
    subset : str
        Dataset subset
    split : str
        Dataset split
    tokenizer : tokenizer or None
        Tokenizer to use
    max_samples : int
        Maximum number of samples to load
        
    Returns:
    -------
    List
        List of tokenized examples
    """
    try:
        from datasets import load_dataset
        
        # Load dataset
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Limit samples
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        # Tokenize if tokenizer is available
        if tokenizer:
            # Get text column
            text_column = next((col for col in dataset.column_names if col in ["text", "sentence", "content"]), dataset.column_names[0])
            
            # Tokenize
            examples = []
            for example in dataset:
                input_ids = tokenizer.encode(example[text_column], return_tensors="pt")
                examples.append(input_ids)
            
            return examples
        
        else:
            # Return raw text
            text_column = next((col for col in dataset.column_names if col in ["text", "sentence", "content"]), dataset.column_names[0])
            return [example[text_column] for example in dataset]
    
    except ImportError:
        print("Datasets library not available, using dummy data")
        
        # Create dummy data
        dummy_data = []
        for _ in range(max_samples):
            if tokenizer:
                # Create random token IDs
                input_ids = torch.randint(0, 1000, (1, 128))
                dummy_data.append(input_ids)
            else:
                # Create random text
                dummy_data.append("This is a dummy example for benchmarking. " * 16)
        
        return dummy_data


def benchmark_latency(model, examples, batch_sizes, sequence_lengths, device, tokenizer=None, num_runs=3):
    """
    Benchmark model latency across different batch sizes and sequence lengths.
    
    Parameters:
    ----------
    model : torch.nn.Module
        Model to benchmark
    examples : List
        List of examples
    batch_sizes : List[int]
        List of batch sizes to test
    sequence_lengths : List[int]
        List of sequence lengths to test
    device : torch.device
        Device to run benchmark on
    tokenizer : tokenizer or None
        Tokenizer to use
    num_runs : int
        Number of runs for each configuration
        
    Returns:
    -------
    Dict
        Benchmark results
    """
    results = {
        'batch_size': [],
        'sequence_length': [],
        'latency_ms': [],
        'throughput_tokens_per_sec': [],
        'memory_mb': []
    }
    
    # Ensure model is in eval mode
    model.eval()
    
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            # Prepare inputs
            if tokenizer:
                # Use tokenized examples
                batch_examples = examples[:batch_size]
                # Truncate or pad to desired sequence length
                input_ids_list = []
                for example in batch_examples:
                    if example.shape[1] > seq_length:
                        input_ids_list.append(example[:, :seq_length])
                    else:
                        # Pad
                        padding = torch.zeros((1, seq_length - example.shape[1]), dtype=torch.long)
                        input_ids_list.append(torch.cat([example, padding], dim=1))
                
                input_ids = torch.cat(input_ids_list, dim=0).to(device)
            else:
                # Create dummy input
                input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
            
            # Empty cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Warmup
            with torch.no_grad():
                _ = model(input_ids)
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Benchmark
            latencies = []
            for _ in range(num_runs):
                # Run forward pass
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_ids)
                torch.cuda.synchronize()
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            throughput = (batch_size * seq_length) / (mean_latency / 1000)  # tokens per second
            
            # Measure memory
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory_mb = 0
            
            # Record results
            results['batch_size'].append(batch_size)
            results['sequence_length'].append(seq_length)
            results['latency_ms'].append(mean_latency)
            results['throughput_tokens_per_sec'].append(throughput)
            results['memory_mb'].append(memory_mb)
            
            print(f"Batch Size: {batch_size}, Sequence Length: {seq_length}, "
                  f"Latency: {mean_latency:.2f} ms, Throughput: {throughput:.2f} tokens/sec, "
                  f"Memory: {memory_mb:.2f} MB")
    
    return results


def benchmark_generation(model, examples, batch_sizes, sequence_lengths, device, max_new_tokens=50, 
                         tokenizer=None, num_runs=3, model_type="quantum_flux"):
    """
    Benchmark text generation.
    
    Parameters:
    ----------
    model : torch.nn.Module
        Model to benchmark
    examples : List
        List of examples
    batch_sizes : List[int]
        List of batch sizes to test
    sequence_lengths : List[int]
        List of sequence lengths to test
    device : torch.device
        Device to run benchmark on
    max_new_tokens : int
        Maximum number of new tokens to generate
    tokenizer : tokenizer or None
        Tokenizer to use
    num_runs : int
        Number of runs for each configuration
    model_type : str
        Type of model ("quantum_flux" or "transformer")
        
    Returns:
    -------
    Dict
        Benchmark results
    """
    results = {
        'batch_size': [],
        'sequence_length': [],
        'generation_time_sec': [],
        'tokens_per_sec': [],
        'memory_mb': []
    }
    
    # Ensure model is in eval mode
    model.eval()
    
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            # Prepare inputs
            if tokenizer:
                # Use tokenized examples
                batch_examples = examples[:batch_size]
                # Truncate or pad to desired sequence length
                input_ids_list = []
                for example in batch_examples:
                    if example.shape[1] > seq_length:
                        input_ids_list.append(example[:, :seq_length])
                    else:
                        # Pad
                        padding = torch.zeros((1, seq_length - example.shape[1]), dtype=torch.long)
                        input_ids_list.append(torch.cat([example, padding], dim=1))
                
                input_ids = torch.cat(input_ids_list, dim=0).to(device)
            else:
                # Create dummy input
                input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
            
            # Empty cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Benchmark
            generation_times = []
            for _ in range(num_runs):
                # Run generation
                start_time = time.time()
                with torch.no_grad():
                    if model_type == "quantum_flux":
                        # Quantum Flux generate method
                        generated_ids = model.generate(
                            input_ids,
                            max_length=seq_length + max_new_tokens
                        )
                    else:
                        # Transformers generate method
                        generated_ids = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.8,
                            top_k=50
                        )
                torch.cuda.synchronize()
                generation_times.append(time.time() - start_time)
            
            # Calculate statistics
            mean_generation_time = np.mean(generation_times)
            tokens_per_sec = (batch_size * max_new_tokens) / mean_generation_time
            
            # Measure memory
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory_mb = 0
            
            # Record results
            results['batch_size'].append(batch_size)
            results['sequence_length'].append(seq_length)
            results['generation_time_sec'].append(mean_generation_time)
            results['tokens_per_sec'].append(tokens_per_sec)
            results['memory_mb'].append(memory_mb)
            
            print(f"Batch Size: {batch_size}, Sequence Length: {seq_length}, "
                  f"Generation Time: {mean_generation_time:.2f} sec, Tokens/sec: {tokens_per_sec:.2f}, "
                  f"Memory: {memory_mb:.2f} MB")
    
    return results


def compare_models(qf_model, tf_model, examples, batch_sizes, sequence_lengths, device, 
                   tokenizer=None, num_runs=3, max_new_tokens=50):
    """
    Compare Quantum Flux and Transformer models.
    
    Parameters:
    ----------
    qf_model : QuantumFluxModel
        Quantum Flux model
    tf_model : GPT2LMHeadModel
        Transformer model
    examples : List
        List of examples
    batch_sizes : List[int]
        List of batch sizes to test
    sequence_lengths : List[int]
        List of sequence lengths to test
    device : torch.device
        Device to run benchmark on
    tokenizer : tokenizer or None
        Tokenizer to use
    num_runs : int
        Number of runs for each configuration
    max_new_tokens : int
        Maximum number of new tokens to generate
        
    Returns:
    -------
    Dict
        Comparison results
    """
    results = {
        'quantum_flux': {
            'latency': benchmark_latency(qf_model, examples, batch_sizes, sequence_lengths, device, tokenizer, num_runs),
            'generation': benchmark_generation(qf_model, examples, batch_sizes, sequence_lengths, device, max_new_tokens, tokenizer, num_runs, "quantum_flux")
        }
    }
    
    if tf_model is not None:
        results['transformer'] = {
            'latency': benchmark_latency(tf_model, examples, batch_sizes, sequence_lengths, device, tokenizer, num_runs),
            'generation': benchmark_generation(tf_model, examples, batch_sizes, sequence_lengths, device, max_new_tokens, tokenizer, num_runs, "transformer")
        }
    
    # Calculate ratios
    if tf_model is not None:
        results['ratios'] = {
            'latency': [],
            'throughput': [],
            'memory': [],
            'generation_time': [],
            'generation_throughput': []
        }
        
        for i in range(len(results['quantum_flux']['latency']['batch_size'])):
            qf_latency = results['quantum_flux']['latency']['latency_ms'][i]
            tf_latency = results['transformer']['latency']['latency_ms'][i]
            
            qf_throughput = results['quantum_flux']['latency']['throughput_tokens_per_sec'][i]
            tf_throughput = results['transformer']['latency']['throughput_tokens_per_sec'][i]
            
            qf_memory = results['quantum_flux']['latency']['memory_mb'][i]
            tf_memory = results['transformer']['latency']['memory_mb'][i]
            
            results['ratios']['latency'].append(qf_latency / tf_latency)
            results['ratios']['throughput'].append(qf_throughput / tf_throughput)
            results['ratios']['memory'].append(qf_memory / tf_memory)
        
        for i in range(len(results['quantum_flux']['generation']['batch_size'])):
            qf_gen_time = results['quantum_flux']['generation']['generation_time_sec'][i]
            tf_gen_time = results['transformer']['generation']['generation_time_sec'][i]
            
            qf_gen_throughput = results['quantum_flux']['generation']['tokens_per_sec'][i]
            tf_gen_throughput = results['transformer']['generation']['tokens_per_sec'][i]
            
            results['ratios']['generation_time'].append(qf_gen_time / tf_gen_time)
            results['ratios']['generation_throughput'].append(qf_gen_throughput / tf_gen_throughput)
    
    return results


def count_flops_comparison(qf_model, tf_model, sequence_lengths, batch_size=1):
    """
    Compare FLOP counts between Quantum Flux and Transformer models.
    
    Parameters:
    ----------
    qf_model : QuantumFluxModel
        Quantum Flux model
    tf_model : GPT2LMHeadModel
        Transformer model
    sequence_lengths : List[int]
        List of sequence lengths to test
    batch_size : int
        Batch size
        
    Returns:
    -------
    Dict
        FLOP comparison results
    """
    results = {
        'sequence_length': sequence_lengths,
        'quantum_flux': [],
        'transformer': [],
        'ratio': []
    }
    
    for seq_length in sequence_lengths:
        # Calculate FLOPs for Quantum Flux
        qf_flops = calculate_flops(qf_model, seq_length, batch_size)
        qf_total_flops = qf_flops['total']
        
        # Calculate FLOPs for Transformer if available
        if tf_model is not None:
            try:
                tf_config = tf_model.config
                
                # Embedding FLOPs
                tf_embedding_flops = batch_size * seq_length * tf_config.n_embd
                
                # Self-attention FLOPs
                # - QKV projections: 3 * n_embd^2 * seq_length * batch_size
                # - Attention scores: 2 * n_embd * seq_length^2 * batch_size
                # - Attention weights: seq_length^2 * batch_size
                # - Context vectors: n_embd * seq_length^2 * batch_size
                # - Output projection: n_embd^2 * seq_length * batch_size
                attn_flops_per_layer = (
                    3 * tf_config.n_embd**2 * seq_length * batch_size +
                    2 * tf_config.n_embd * seq_length**2 * batch_size +
                    seq_length**2 * batch_size +
                    tf_config.n_embd * seq_length**2 * batch_size +
                    tf_config.n_embd**2 * seq_length * batch_size
                )
                
                # MLP FLOPs
                # - First linear: n_embd * 4*n_embd * seq_length * batch_size
                # - GELU: 8 * 4*n_embd * seq_length * batch_size
                # - Second linear: 4*n_embd * n_embd * seq_length * batch_size
                mlp_flops_per_layer = (
                    tf_config.n_embd * 4*tf_config.n_embd * seq_length * batch_size +
                    8 * 4*tf_config.n_embd * seq_length * batch_size +
                    4*tf_config.n_embd * tf_config.n_embd * seq_length * batch_size
                )
                
                # Layer norm FLOPs
                layer_norm_flops = 4 * tf_config.n_embd * seq_length * batch_size
                
                # Total FLOPs per layer
                layer_flops = attn_flops_per_layer + mlp_flops_per_layer + 2 * layer_norm_flops
                
                # Total FLOPs for all layers
                tf_layers_flops = layer_flops * tf_config.n_layer
                
                # Output FLOPs
                tf_output_flops = tf_config.n_embd * tf_config.vocab_size * seq_length * batch_size
                
                # Total transformer FLOPs
                tf_total_flops = tf_embedding_flops + tf_layers_flops + tf_output_flops
                
                # Calculate ratio
                ratio = qf_total_flops / tf_total_flops
                
            except Exception as e:
                print(f"Error calculating transformer FLOPs: {e}")
                tf_total_flops = 0
                ratio = 0
        else:
            tf_total_flops = 0
            ratio = 0
        
        # Record results
        results['quantum_flux'].append(qf_total_flops)
        results['transformer'].append(tf_total_flops)
        results['ratio'].append(ratio)
        
        print(f"Sequence Length: {seq_length}, Quantum Flux FLOPs: {qf_total_flops:,}, "
              f"Transformer FLOPs: {tf_total_flops:,}, Ratio: {ratio:.6f}")
    
    return results


def plot_results(benchmark_results, output_dir):
    """
    Plot benchmark results.
    
    Parameters:
    ----------
    benchmark_results : Dict
        Benchmark results
    output_dir : str
        Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    
    # Plot latency comparison
    if 'quantum_flux' in benchmark_results and 'transformer' in benchmark_results:
        # Prepare data for latency comparison
        batch_sizes = benchmark_results['quantum_flux']['latency']['batch_size']
        seq_lengths = benchmark_results['quantum_flux']['latency']['sequence_length']
        
        qf_latency = benchmark_results['quantum_flux']['latency']['latency_ms']
        tf_latency = benchmark_results['transformer']['latency']['latency_ms']
        
        # Create DataFrame for easy plotting
        df_latency = pd.DataFrame({
            'Batch Size': batch_sizes,
            'Sequence Length': seq_lengths,
            'Quantum Flux': qf_latency,
            'Transformer': tf_latency
        })
        
        # Reshape for heatmap
        unique_batch_sizes = sorted(set(batch_sizes))
        unique_seq_lengths = sorted(set(seq_lengths))
        
        qf_latency_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        tf_latency_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        ratio_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        
        for i, bs in enumerate(unique_batch_sizes):
            for j, sl in enumerate(unique_seq_lengths):
                idx = [k for k, (b, s) in enumerate(zip(batch_sizes, seq_lengths)) if b == bs and s == sl]
                if idx:
                    qf_latency_matrix[i, j] = qf_latency[idx[0]]
                    tf_latency_matrix[i, j] = tf_latency[idx[0]]
                    ratio_matrix[i, j] = qf_latency[idx[0]] / tf_latency[idx[0]]
        
        # Plot latency heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Quantum Flux latency
        sns.heatmap(qf_latency_matrix, annot=True, fmt=".1f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="Blues", ax=axes[0])
        axes[0].set_title("Quantum Flux Latency (ms)")
        axes[0].set_xlabel("Sequence Length")
        axes[0].set_ylabel("Batch Size")
        
        # Transformer latency
        sns.heatmap(tf_latency_matrix, annot=True, fmt=".1f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="Blues", ax=axes[1])
        axes[1].set_title("Transformer Latency (ms)")
        axes[1].set_xlabel("Sequence Length")
        axes[1].set_ylabel("Batch Size")
        
        # Ratio
        sns.heatmap(ratio_matrix, annot=True, fmt=".2f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="RdBu_r", center=1.0, ax=axes[2])
        axes[2].set_title("Quantum Flux / Transformer Latency Ratio")
        axes[2].set_xlabel("Sequence Length")
        axes[2].set_ylabel("Batch Size")
        
        plt.tight_layout()
        plt.savefig(output_path / "latency_comparison.png", dpi=300)
        plt.close()
        
        # Plot throughput comparison
        qf_throughput = benchmark_results['quantum_flux']['latency']['throughput_tokens_per_sec']
        tf_throughput = benchmark_results['transformer']['latency']['throughput_tokens_per_sec']
        
        qf_throughput_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        tf_throughput_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        throughput_ratio_matrix = np.zeros((len(unique_batch_sizes), len(unique_seq_lengths)))
        
        for i, bs in enumerate(unique_batch_sizes):
            for j, sl in enumerate(unique_seq_lengths):
                idx = [k for k, (b, s) in enumerate(zip(batch_sizes, seq_lengths)) if b == bs and s == sl]
                if idx:
                    qf_throughput_matrix[i, j] = qf_throughput[idx[0]]
                    tf_throughput_matrix[i, j] = tf_throughput[idx[0]]
                    throughput_ratio_matrix[i, j] = qf_throughput[idx[0]] / tf_throughput[idx[0]]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Quantum Flux throughput
        sns.heatmap(qf_throughput_matrix, annot=True, fmt=".0f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="Greens", ax=axes[0])
        axes[0].set_title("Quantum Flux Throughput (tokens/sec)")
        axes[0].set_xlabel("Sequence Length")
        axes[0].set_ylabel("Batch Size")
        
        # Transformer throughput
        sns.heatmap(tf_throughput_matrix, annot=True, fmt=".0f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="Greens", ax=axes[1])
        axes[1].set_title("Transformer Throughput (tokens/sec)")
        axes[1].set_xlabel("Sequence Length")
        axes[1].set_ylabel("Batch Size")
        
        # Ratio
        sns.heatmap(throughput_ratio_matrix, annot=True, fmt=".2f", 
                    xticklabels=unique_seq_lengths, yticklabels=unique_batch_sizes,
                    cmap="RdBu", center=1.0, ax=axes[2])
        axes[2].set_title("Quantum Flux / Transformer Throughput Ratio")
        axes[2].set_xlabel("Sequence Length")
        axes[2].set_ylabel("Batch Size")
        
        plt.tight_layout()
        plt.savefig(output_path / "throughput_comparison.png", dpi=300)
        plt.close()
    
    # Plot FLOP comparison if available
    if 'flops' in benchmark_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        seq_lengths = benchmark_results['flops']['sequence_length']
        qf_flops = benchmark_results['flops']['quantum_flux']
        tf_flops = benchmark_results['flops']['transformer']
        ratios = benchmark_results['flops']['ratio']
        
        # FLOPs
        ax1.plot(seq_lengths, qf_flops, marker='o', label='Quantum Flux')
        ax1.plot(seq_lengths, tf_flops, marker='s', label='Transformer')
        ax1.set_title('FLOPs Comparison')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('FLOPs')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        # Ratio
        ax2.plot(seq_lengths, ratios, marker='D', color='purple')
        ax2.set_title('Quantum Flux / Transformer FLOP Ratio')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Ratio')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / "flops_comparison.png", dpi=300)
        plt.close()
    
    print(f"Plots saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Flux Benchmarking")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run benchmark on")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for benchmarking")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16", 
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--sequence_lengths", type=str, default="128,256,512,1024", 
                        help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--compare_transformer", action="store_true", help="Compare with transformer model")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--plot_results", action="store_true", help="Generate plots of benchmark results")
    parser.add_argument("--output_dir", type=str, default="benchmark_plots", help="Output directory for plots")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Convert string lists to integers
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(",")]
    
    # Set device
    device = torch.device(args.device)
    
    # Load Quantum Flux model
    print(f"Loading Quantum Flux model from {args.model_path}")
    qf_model, config, tokenizer = load_model(args.model_path, device)
    
    # Load transformer model for comparison if requested
    tf_model = None
    if args.compare_transformer:
        print("Loading transformer model for comparison")
        tf_model, _ = load_transformer_model(config, device)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}")
    examples = load_dataset(args.dataset, args.subset, args.split, tokenizer, args.max_samples)
    
    # Print model information
    qf_params = count_parameters(qf_model)
    print(f"Quantum Flux model parameters: {qf_params:,}")
    
    if tf_model:
        tf_params = count_parameters(tf_model)
        print(f"Transformer model parameters: {tf_params:,}")
        print(f"Parameter ratio (QF/TF): {qf_params/tf_params:.4f}")
    
    # Benchmark results
    results = {}
    
    # Compare models
    if tf_model:
        print("\nComparing models...")
        results['comparison'] = compare_models(qf_model, tf_model, examples, batch_sizes, sequence_lengths, 
                                               device, tokenizer, args.num_runs, args.max_new_tokens)
    else:
        # Benchmark latency
        print("\nBenchmarking latency...")
        results['latency'] = benchmark_latency(qf_model, examples, batch_sizes, sequence_lengths, 
                                               device, tokenizer, args.num_runs)
        
        # Benchmark generation
        print("\nBenchmarking generation...")
        results['generation'] = benchmark_generation(qf_model, examples, batch_sizes, sequence_lengths, 
                                                    device, args.max_new_tokens, tokenizer, args.num_runs)
    
    # Compare FLOPs
    print("\nComparing FLOPs...")
    results['flops'] = count_flops_comparison(qf_model, tf_model, sequence_lengths)
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Generate plots if requested
    if args.plot_results:
        print("\nGenerating plots...")
        plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
