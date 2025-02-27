"""
Quantum Flux Training Script with W&B Integration
================================================

This script provides a production-ready training pipeline for the Quantum Flux
Neural Network with Weights & Biases (W&B) integration and Hugging Face datasets.

Features:
- Weights & Biases experiment tracking
- Hugging Face datasets integration
- Model comparison with traditional transformers
- Multi-GPU training with distributed data parallel
- Mixed precision training
- Performance benchmarking and FLOP calculation
- Customizable model configurations

Usage:
```
python scripts/train_wandb.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --wandb-project quantum-flux \
    --wandb-name "qf-base-wikitext" \
    --output ./output/wikitext
```
"""

import os
import sys
import time
import argparse
import yaml
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import quantum flux modules
from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel
from quantum_flux.visualization import (
    visualize_quantum_states,
    visualize_attention,
    visualize_training_metrics
)

# No need to import custom collate function

# Import transformers and datasets
try:
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from datasets import load_dataset, DatasetDict
    import wandb
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Please install required packages: pip install transformers datasets wandb")
    sys.exit(1)


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


def benchmark_comparison(quantum_flux_model, transformer_model, seq_length, batch_size=1):
    """
    Compare Quantum Flux and Transformer models in terms of FLOPs and parameters.
    
    Parameters:
    ----------
    quantum_flux_model : QuantumFluxModel
        Quantum Flux model
    transformer_model : nn.Module
        Transformer model (e.g., GPT2)
    seq_length : int
        Sequence length
    batch_size : int
        Batch size
        
    Returns:
    -------
    dict
        Comparison metrics
    """
    # Count parameters
    qf_params = count_parameters(quantum_flux_model)
    tf_params = count_parameters(transformer_model)
    
    # Calculate FLOPs
    qf_flops = calculate_flops(quantum_flux_model, seq_length, batch_size)
    
    # Transformer FLOPs (based on standard calculations)
    tf_config = transformer_model.config
    tf_flops = {}
    
    # Embedding FLOPs
    tf_flops['embedding'] = batch_size * seq_length * tf_config.n_embd
    
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
    tf_flops['attention'] = attn_flops_per_layer * tf_config.n_layer
    tf_flops['mlp'] = mlp_flops_per_layer * tf_config.n_layer
    tf_flops['layer_norm'] = 2 * layer_norm_flops * tf_config.n_layer
    
    # Output FLOPs
    tf_flops['output'] = tf_config.n_embd * tf_config.vocab_size * seq_length * batch_size
    
    # Total transformer FLOPs
    tf_flops['total'] = tf_flops['embedding'] + (layer_flops * tf_config.n_layer) + tf_flops['output']
    
    # Comparison metrics
    comparison = {
        'quantum_flux': {
            'params': qf_params,
            'flops': qf_flops['total'],
            'flops_breakdown': qf_flops
        },
        'transformer': {
            'params': tf_params,
            'flops': tf_flops['total'],
            'flops_breakdown': tf_flops
        },
        'ratio': {
            'params': qf_params / tf_params,
            'flops': qf_flops['total'] / tf_flops['total']
        }
    }
    
    return comparison


def get_tokenized_datasets(args):
    """
    Load and tokenize dataset from Hugging Face.
    
    Parameters:
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    -------
    tuple
        (train_dataset, val_dataset, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if args.local_dataset:
        # Load from local files
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
        
        extension = args.train_file.split(".")[-1] if args.train_file else "txt"
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        # Load from Hugging Face
        raw_datasets = load_dataset(args.dataset, args.subset)
    
    # Rename columns for consistency
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Tokenizing datasets",
    )
    
    # Concatenation and chunking function
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop small remainder
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        
        # Split by chunks of max_len
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        
        # Create labels
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Apply concatenation and chunking
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {args.max_seq_length}",
    )
    
    # Get train and validation splits
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if "validation" in lm_datasets else lm_datasets["test"]
    
    return train_dataset, eval_dataset, tokenizer


class PerformanceCallback:
    """
    Callback to track performance metrics during training.
    
    Attributes:
    ----------
    start_time : float
        Training start time
    batch_times : list
        List of batch processing times
    batch_sizes : list
        List of batch sizes
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.batch_times = []
        self.batch_sizes = []
    
    def on_batch_begin(self):
        """Mark batch start time."""
        self.batch_start = time.time()
    
    def on_batch_end(self, batch_size):
        """
        Record batch processing time.
        
        Parameters:
        ----------
        batch_size : int
            Batch size
        """
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)
    
    def get_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
        -------
        dict
            Performance metrics
        """
        total_time = time.time() - self.start_time
        total_samples = sum(self.batch_sizes)
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        throughput = total_samples / total_time if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "throughput_samples_per_second": throughput,
            "total_samples": total_samples
        }


def collate_fn(batch):
    """
    Custom collate function to convert batch data into a dictionary format.
    
    Parameters:
    ----------
    batch : list
        List of batch items
        
    Returns:
    -------
    dict
        Dictionary containing input_ids, labels, and attention_mask
    """
    # Separate inputs and labels
    inputs, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = torch.tensor([len(x) for x in inputs])
    max_len = lengths.max().item()
    
    # Prepare batched tensors with padding
    batch_size = len(inputs)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    label_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    # Fill in values
    for i, (input_seq, label_seq) in enumerate(zip(inputs, labels)):
        input_len = len(input_seq)
        input_ids[i, :input_len] = input_seq
        label_ids[i, :input_len] = label_seq
    
    # Create attention mask
    attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    # Return as dictionary instead of list
    return {
        'input_ids': input_ids,
        'labels': label_ids,
        'attention_mask': attention_mask
    }


def train(rank, world_size, args):
    """
    Training function for a single GPU.
    
    Parameters:
    ----------
    rank : int
        Process rank
    world_size : int
        Number of processes
    args : argparse.Namespace
        Command line arguments
    """
    # Setup distributed training if using multiple GPUs
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb (only on master process)
    if rank == 0 and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration
    config = QuantumFluxConfig(**config_dict)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tokenized datasets
    train_dataset, val_dataset, tokenizer = get_tokenized_datasets(args)
    
    # Update config with vocab size from tokenizer
    config.vocab_size = len(tokenizer)
    
    # Create Quantum Flux model
    quantum_flux_model = QuantumFluxModel(config).to(device)
    
    # Create transformer model for comparison if requested
    transformer_model = None
    if args.compare_transformer:
        tf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_seq_length,
            n_embd=config.embed_dim,
            n_layer=config.num_layers,
            n_head=8  # Standard number of heads
        )
        transformer_model = GPT2LMHeadModel(tf_config).to(device)
    
    # Print model statistics
    if rank == 0:
        qf_params = count_parameters(quantum_flux_model)
        print(f"Quantum Flux model parameters: {qf_params:,}")
        
        if transformer_model:
            tf_params = count_parameters(transformer_model)
            print(f"Transformer model parameters: {tf_params:,}")
            
            # Compare FLOPs
            comparison = benchmark_comparison(
                quantum_flux_model, 
                transformer_model, 
                config.max_seq_length
            )
            
            print("\nModel Comparison:")
            print(f"Parameters - QF: {comparison['quantum_flux']['params']:,}, "
                  f"TF: {comparison['transformer']['params']:,}, "
                  f"Ratio: {comparison['ratio']['params']:.2f}")
            print(f"FLOPs - QF: {comparison['quantum_flux']['flops']:,}, "
                  f"TF: {comparison['transformer']['flops']:,}, "
                  f"Ratio: {comparison['ratio']['flops']:.2f}")
            
            # Log to wandb
            if args.wandb:
                wandb.log({
                    "model/qf_params": qf_params,
                    "model/tf_params": tf_params,
                    "model/param_ratio": qf_params / tf_params,
                    "model/qf_flops": comparison['quantum_flux']['flops'],
                    "model/tf_flops": comparison['transformer']['flops'],
                    "model/flop_ratio": comparison['ratio']['flops']
                })
    
    # Distributed Data Parallel
    if world_size > 1:
        quantum_flux_model = DDP(quantum_flux_model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        quantum_flux_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 100
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    # Create data loaders with pin memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        # No custom collate_fn - use default
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        # No custom collate_fn - use default
        pin_memory=True
    )
    
    # Create TensorBoard writer (only on master process)
    if rank == 0:
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_dir)
    else:
        writer = None
    
    # Training state
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Load checkpoint if exists
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        
        # Handle DDP loading
        if world_size > 1:
            # Remove 'module.' prefix if loading from non-DDP checkpoint
            if not any(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        else:
            # Remove 'module.' prefix if loading from DDP checkpoint
            if any(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        quantum_flux_model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Performance tracking
    perf_callback = PerformanceCallback()
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training phase
        quantum_flux_model.train()
        train_loss = 0.0
        train_metrics = {}
        
        # Progress bar (only on master process)
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=rank != 0)
        
        for i, batch in enumerate(train_iter):
            # Record batch start time
            perf_callback.on_batch_begin()
            
            # Get inputs and labels
            try:
                if isinstance(batch, dict):
                    # Handle dictionary-style batches
                    # Check what type input_ids is and handle accordingly
                    if isinstance(batch['input_ids'], torch.Tensor):
                        input_ids = batch['input_ids'].to(device)
                    elif isinstance(batch['input_ids'], list):
                        # Print debug info about the list
                        print(f"DEBUG - input_ids is a list with length {len(batch['input_ids'])}")
                        if len(batch['input_ids']) > 0:
                            print(f"DEBUG - First element type: {type(batch['input_ids'][0])}")
                        
                        # Try to convert to tensor if it's a simple list
                        try:
                            input_ids = torch.tensor(batch['input_ids']).to(device)
                        except Exception as e:
                            print(f"ERROR converting input_ids to tensor: {e}")
                            # If conversion fails, try a different approach
                            if all(isinstance(item, torch.Tensor) for item in batch['input_ids']):
                                input_ids = torch.stack(batch['input_ids']).to(device)
                            else:
                                raise TypeError(f"Cannot convert input_ids to tensor: {e}")
                    else:
                        raise TypeError(f"Unexpected input_ids type: {type(batch['input_ids'])}")
                    
                    # Check if labels is a list or tensor
                    if isinstance(batch['labels'], list):
                        labels = torch.tensor(batch['labels']).to(device)
                    else:
                        labels = batch['labels'].to(device)
                    
                    # Handle attention_mask if present
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        if isinstance(attention_mask, list):
                            attention_mask = torch.tensor(attention_mask).to(device)
                        else:
                            attention_mask = attention_mask.to(device)
                elif isinstance(batch, list):
                    # Handle list-type batches
                    if isinstance(batch[0], torch.Tensor):
                        input_ids = batch[0].to(device)
                    elif isinstance(batch[0], list):
                        input_ids = torch.tensor(batch[0]).to(device)
                    else:
                        raise TypeError(f"Unexpected batch[0] type: {type(batch[0])}")
                    
                    if isinstance(batch[1], torch.Tensor):
                        labels = batch[1].to(device)
                    elif isinstance(batch[1], list):
                        labels = torch.tensor(batch[1]).to(device)
                    else:
                        raise TypeError(f"Unexpected batch[1] type: {type(batch[1])}")
                    
                    attention_mask = None
                    if len(batch) > 2:
                        if isinstance(batch[2], torch.Tensor):
                            attention_mask = batch[2].to(device)
                        elif isinstance(batch[2], list):
                            attention_mask = torch.tensor(batch[2]).to(device)
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, list):
                    print(f"Batch length: {len(batch)}")
                    print(f"Batch[0] type: {type(batch[0]) if len(batch) > 0 else 'N/A'}")
                raise
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = quantum_flux_model(
                    input_ids,
                    targets=labels,
                    is_training=True
                )
                
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % args.gradient_accumulation == 0 or (i + 1) == len(train_loader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(quantum_flux_model.parameters(), args.max_grad_norm)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                global_step += 1
            
            # Record batch end time
            perf_callback.on_batch_end(input_ids.size(0))
            
            # Update metrics
            train_loss += loss.item() * args.gradient_accumulation
            
            batch_metrics = outputs['metrics']
            for key, value in batch_metrics.items():
                if key not in train_metrics:
                    train_metrics[key] = 0.0
                train_metrics[key] += value
            
            # Update progress bar
            if rank == 0:
                train_iter.set_postfix({
                    'loss': train_loss / (i + 1),
                    'perplexity': np.exp(min(train_loss / (i + 1), 20))
                })
            
            # Log to wandb and tensorboard
            if rank == 0 and global_step % args.log_interval == 0:
                # Current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Performance metrics
                perf_metrics = perf_callback.get_metrics()
                
                # Log to tensorboard
                writer.add_scalar('train/loss', loss.item() * args.gradient_accumulation, global_step)
                writer.add_scalar('train/perplexity', np.exp(min(loss.item() * args.gradient_accumulation, 20)), global_step)
                writer.add_scalar('train/lr', current_lr, global_step)
                writer.add_scalar('performance/throughput', perf_metrics['throughput_samples_per_second'], global_step)
                writer.add_scalar('performance/batch_time', perf_metrics['avg_batch_time'], global_step)
                
                for key, value in batch_metrics.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
                
                # Log to wandb
                if args.wandb:
                    wandb_log = {
                        'train/loss': loss.item() * args.gradient_accumulation,
                        'train/perplexity': np.exp(min(loss.item() * args.gradient_accumulation, 20)),
                        'train/lr': current_lr,
                        'performance/throughput': perf_metrics['throughput_samples_per_second'],
                        'performance/batch_time': perf_metrics['avg_batch_time'],
                        'epoch': epoch
                    }
                    
                    for key, value in batch_metrics.items():
                        wandb_log[f'train/{key}'] = value
                    
                    wandb.log(wandb_log, step=global_step)
            
            # Memory optimization
            del loss, outputs
            torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        quantum_flux_model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=rank != 0):
                # Get inputs and labels
                try:
                    if isinstance(batch, dict):
                        # Handle dictionary-style batches
                        # Check if input_ids is a list or tensor
                        if isinstance(batch['input_ids'], list):
                            input_ids = torch.tensor(batch['input_ids']).to(device)
                        else:
                            input_ids = batch['input_ids'].to(device)
                        
                        # Check if labels is a list or tensor
                        if isinstance(batch['labels'], list):
                            labels = torch.tensor(batch['labels']).to(device)
                        else:
                            labels = batch['labels'].to(device)
                        
                        # Handle attention_mask if present
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            if isinstance(attention_mask, list):
                                attention_mask = torch.tensor(attention_mask).to(device)
                            else:
                                attention_mask = attention_mask.to(device)
                    elif isinstance(batch, list):
                        # Handle list-type batches
                        if isinstance(batch[0], torch.Tensor):
                            input_ids = batch[0].to(device)
                        elif isinstance(batch[0], list):
                            input_ids = torch.tensor(batch[0]).to(device)
                        else:
                            raise TypeError(f"Unexpected batch[0] type: {type(batch[0])}")
                        
                        if isinstance(batch[1], torch.Tensor):
                            labels = batch[1].to(device)
                        elif isinstance(batch[1], list):
                            labels = torch.tensor(batch[1]).to(device)
                        else:
                            raise TypeError(f"Unexpected batch[1] type: {type(batch[1])}")
                        
                        attention_mask = None
                        if len(batch) > 2:
                            if isinstance(batch[2], torch.Tensor):
                                attention_mask = batch[2].to(device)
                            elif isinstance(batch[2], list):
                                attention_mask = torch.tensor(batch[2]).to(device)
                    else:
                        raise TypeError(f"Unexpected batch type: {type(batch)}")
                except Exception as e:
                    print(f"Error processing validation batch: {e}")
                    print(f"Batch type: {type(batch)}")
                    if isinstance(batch, list):
                        print(f"Batch length: {len(batch)}")
                        print(f"Batch[0] type: {type(batch[0]) if len(batch) > 0 else 'N/A'}")
                    raise
                
                # Forward pass
                outputs = quantum_flux_model(
                    input_ids,
                    targets=labels,
                    is_training=False
                )
                
                loss = outputs['loss']
                
                # Update metrics
                val_loss += loss.item()
                
                batch_metrics = outputs['metrics']
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
                
                # Memory optimization
                del loss, outputs
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics (only on master process)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Perplexity: {np.exp(min(train_loss, 20)):.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Perplexity: {np.exp(min(val_loss, 20)):.4f}")
            
            # Log to tensorboard
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/train_perplexity', np.exp(min(train_loss, 20)), epoch)
            writer.add_scalar('epoch/val_perplexity', np.exp(min(val_loss, 20)), epoch)
            
            for key, value in train_metrics.items():
                writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            for key, value in val_metrics.items():
                writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Log epoch metrics to wandb
            if args.wandb:
                wandb_log = {
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/train_perplexity': np.exp(min(train_loss, 20)),
                    'epoch/val_perplexity': np.exp(min(val_loss, 20))
                }
                
                for key, value in train_metrics.items():
                    wandb_log[f'epoch/train_{key}'] = value
                
                for key, value in val_metrics.items():
                    wandb_log[f'epoch/val_{key}'] = value
                
                wandb.log(wandb_log, step=global_step)
            
            # Generate sample text
            if epoch % args.sample_interval == 0:
                sample_input = input_ids[0:1, :10].clone()
                
                # Get input tokens as text
                input_text = tokenizer.decode(sample_input[0])
                
                # Generate continuation
                generated = quantum_flux_model.generate(
                    sample_input,
                    max_length=50,
                    temperature=0.8,
                    top_k=50
                )
                
                # Decode generated tokens
                generated_text = tokenizer.decode(generated[0])
                
                print("\nSample generation:")
                print(f"Input: {input_text}")
                print(f"Generated: {generated_text}")
                
                # Log to tensorboard and wandb
                writer.add_text('generation/sample', generated_text, epoch)
                
                if args.wandb:
                    wandb.log({
                        'generation/sample': wandb.Html(f"<p><strong>Input:</strong> {input_text}</p><p><strong>Generated:</strong> {generated_text}</p>")
                    }, step=global_step)
            
            # Visualize quantum states
            if epoch % args.visualize_interval == 0:
                with torch.no_grad():
                    viz_input = input_ids[0:1, :10].clone()
                    outputs = quantum_flux_model(viz_input)
                    
                    # Get quantum states
                    if hasattr(quantum_flux_model, 'module'):
                        # Handle DDP case
                        states = quantum_flux_model.module.encoder.encode(viz_input)[0].cpu().numpy()
                    else:
                        states = quantum_flux_model.encoder.encode(viz_input)[0].cpu().numpy()
                    
                    # Visualize quantum states
                    try:
                        fig = visualize_quantum_states(states, title="Quantum States")
                        
                        # Save figure
                        fig_path = output_dir / f"quantum_states_epoch_{epoch}.png"
                        fig.savefig(fig_path)
                        
                        # Log to tensorboard
                        writer.add_figure('visualization/quantum_states', fig, epoch)
                        
                        # Log to wandb
                        if args.wandb:
                            wandb.log({
                                'visualization/quantum_states': wandb.Image(str(fig_path))
                            }, step=global_step)
                    except Exception as e:
                        print(f"Error visualizing quantum states: {e}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model (only on master process)
            if rank == 0:
                best_model_path = output_dir / "model_best.pt"
                
                # Get state dict
                if hasattr(quantum_flux_model, 'module'):
                    model_state_dict = quantum_flux_model.module.state_dict()
                else:
                    model_state_dict = quantum_flux_model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': config_dict
                }, best_model_path)
                
                print(f"Saved best model with val_loss: {val_loss:.4f}")
                
                # Log best model to wandb
                if args.wandb:
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_val_perplexity"] = np.exp(min(val_loss, 20))
                    wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= args.patience and args.patience > 0:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save checkpoint (only on master process)
        if rank == 0 and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / "checkpoint.pt"
            
            # Get state dict
            if hasattr(quantum_flux_model, 'module'):
                model_state_dict = quantum_flux_model.module.state_dict()
            else:
                model_state_dict = quantum_flux_model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config_dict
            }, checkpoint_path)
            
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Final performance metrics
    if rank == 0:
        perf_metrics = perf_callback.get_metrics()
        print("\nTraining complete!")
        print(f"Total time: {perf_metrics['total_time']:.2f} seconds")
        print(f"Average batch time: {perf_metrics['avg_batch_time']:.4f} seconds")
        print(f"Throughput: {perf_metrics['throughput_samples_per_second']:.2f} samples/second")
        
        # Log final metrics to wandb
        if args.wandb:
            wandb.run.summary.update({
                "total_training_time": perf_metrics['total_time'],
                "avg_batch_time": perf_metrics['avg_batch_time'],
                "throughput_samples_per_second": perf_metrics['throughput_samples_per_second'],
                "total_samples_processed": perf_metrics['total_samples'],
                "final_val_loss": val_loss,
                "final_val_perplexity": np.exp(min(val_loss, 20))
            })
            
            # Finish wandb run
            wandb.finish()
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Quantum Flux model with W&B")
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='wikitext', help='Hugging Face dataset name')
    parser.add_argument('--subset', type=str, default='wikitext-103-raw-v1', help='Dataset subset')
    parser.add_argument('--local_dataset', action='store_true', help='Use local dataset files')
    parser.add_argument('--train_file', type=str, default=None, help='Path to training file')
    parser.add_argument('--validation_file', type=str, default=None, help='Path to validation file')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer to use')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--preprocessing_num_workers', type=int, default=4, help='Number of preprocessing workers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--gpu', type=str, default='0', help='GPU indices separated by comma, or -1 for CPU')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (steps)')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint interval (epochs)')
    parser.add_argument('--sample_interval', type=int, default=1, help='Text sampling interval (epochs)')
    parser.add_argument('--visualize_interval', type=int, default=1, help='Visualization interval (epochs)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (0 to disable)')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    
    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='quantum-flux', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    
    # Comparison arguments
    parser.add_argument('--compare_transformer', action='store_true', help='Compare with transformer model')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Generate run name if not provided
    if args.wandb and not args.wandb_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.wandb_name = f"qf-{Path(args.config).stem}-{timestamp}"
    
    # Determine world size and rank
    if args.gpu == '-1':
        # CPU only
        world_size = 1
        train(0, world_size, args)
    else:
        # Get GPU indices
        gpu_indices = [int(i) for i in args.gpu.split(',')]
        world_size = len(gpu_indices)
        
        if world_size == 1:
            # Single GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            train(0, world_size, args)
        else:
            # Multiple GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
