"""
Quantum Flux Production Training Script
===================================

This script provides a production-ready training pipeline for the Quantum Flux
Neural Network, optimized for consumer-grade GPUs like RTX 4090.

Features:
- Efficient batch loading with dynamic batch sizing
- Mixed precision training
- Gradient accumulation
- Checkpointing and recovery
- Tensorboard logging
- Early stopping
- Learning rate scheduling
- Memory optimization
- Multi-GPU support via NVIDIA DDP (DistributedDataParallel)

Example usage:
```
python scripts/train.py \
    --config configs/base.yaml \
    --data path/to/data \
    --output path/to/output \
    --gpu 0
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel
from quantum_flux.visualization import (
    visualize_quantum_states,
    visualize_attention,
    visualize_training_metrics,
    visualize_state_evolution
)


class TokenizedDataset(Dataset):
    """
    Dataset class for tokenized sequences optimized for the RTX 4090.
    
    Implements memory-mapped loading for datasets too large to fit in memory
    and efficient batching with padding.
    """
    
    def __init__(self, data_path, max_seq_length=1024, split="train"):
        """
        Initialize the dataset.
        
        Parameters:
        ----------
        data_path : str
            Path to tokenized data
        max_seq_length : int
            Maximum sequence length
        split : str
            Data split ("train", "val", "test")
        """
        self.data_path = Path(data_path) / f"{split}.bin"
        self.idx_path = Path(data_path) / f"{split}.idx"
        self.max_seq_length = max_seq_length
        
        # Check if files exist
        if not self.data_path.exists() or not self.idx_path.exists():
            raise FileNotFoundError(f"Data files not found at {data_path}")
        
        # Memory map for efficient access
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        
        # Load indices
        with open(self.idx_path, 'r') as f:
            self.indices = json.load(f)
        
        self.num_samples = len(self.indices)
        print(f"Loaded {self.num_samples} samples from {self.data_path}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sequence by index.
        
        Parameters:
        ----------
        idx : int
            Sample index
            
        Returns:
        -------
        tuple
            (input_ids, labels) - both are tokenized sequences
        """
        # Get start and end indices
        start_idx = self.indices[idx]
        end_idx = self.indices[idx + 1] if idx < len(self.indices) - 1 else len(self.data)
        
        # Get sequence
        sequence = self.data[start_idx:end_idx]
        
        # Truncate if too long
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        
        # Convert to PyTorch tensor
        sequence = torch.from_numpy(sequence.astype(np.int64))
        
        # Labels are same as input (for autoregressive prediction)
        # But shifted by one position to the right
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return input_ids, labels


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences.
    
    Parameters:
    ----------
    batch : list
        List of (input_ids, labels) tuples
        
    Returns:
    -------
    tuple
        (input_ids, labels, attention_mask) - batched and padded
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
    
    return input_ids, label_ids, attention_mask


def setup(rank, world_size):
    """
    Setup distributed training.
    
    Parameters:
    ----------
    rank : int
        Process rank
    world_size : int
        Number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


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
        setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration
    config = QuantumFluxConfig(**config_dict)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = QuantumFluxModel(config).to(device)
    
    # Distributed Data Parallel
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=config.learning_rate / 100
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create datasets
    train_dataset = TokenizedDataset(
        args.data,
        max_seq_length=config.max_seq_length,
        split="train"
    )
    
    val_dataset = TokenizedDataset(
        args.data,
        max_seq_length=config.max_seq_length,
        split="val"
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create tensorboard writer (only on master process)
    writer = SummaryWriter(log_dir) if rank == 0 else None
    
    # Training state
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Load checkpoint if exists
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {}
        
        # Progress bar (only on master process)
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=rank != 0)
        
        for i, (input_ids, label_ids, attention_mask) in enumerate(train_iter):
            # Move to device
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(
                    input_ids,
                    targets=label_ids,
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                global_step += 1
            
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
                    'perplexity': np.exp(train_loss / (i + 1))
                })
            
            # Log to tensorboard
            if rank == 0 and global_step % args.log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/perplexity', np.exp(loss.item()), global_step)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
                
                for key, value in batch_metrics.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
            
            # Memory optimization
            del loss, outputs
            torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for input_ids, label_ids, attention_mask in tqdm(val_loader, desc="Validation", disable=rank != 0):
                # Move to device
                input_ids = input_ids.to(device)
                label_ids = label_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids,
                    targets=label_ids,
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
            print(f"  Train Loss: {train_loss:.4f}, Perplexity: {np.exp(train_loss):.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Perplexity: {np.exp(val_loss):.4f}")
            
            # Log to tensorboard
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/train_perplexity', np.exp(train_loss), epoch)
            writer.add_scalar('epoch/val_perplexity', np.exp(val_loss), epoch)
            
            for key, value in train_metrics.items():
                writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            for key, value in val_metrics.items():
                writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Generate sample text
            if epoch % args.sample_interval == 0:
                sample_input = input_ids[0:1, :10]
                generated = model.generate(
                    sample_input,
                    max_length=50,
                    temperature=0.8,
                    top_k=50
                )
                
                # Log generated text
                writer.add_text('generation/sample', str(generated[0].tolist()), epoch)
            
            # Visualize quantum states
            if epoch % args.visualize_interval == 0:
                with torch.no_grad():
                    outputs = model(input_ids[0:1, :10])
                    
                    # Get quantum states and metrics
                    states = outputs['quantum_states'][0].cpu().numpy()
                    
                    # Get all states through layers
                    all_states = [s[0].cpu().numpy() for s in outputs['all_quantum_states']]
                    
                    # Visualize quantum states
                    fig = visualize_quantum_states(states, title="Quantum States")
                    writer.add_figure('visualization/quantum_states', fig, epoch)
                    
                    # Visualize state evolution
                    fig = visualize_state_evolution(all_states, title="State Evolution")
                    writer.add_figure('visualization/state_evolution', fig, epoch)
                    
                    # Get attention scores from the first layer
                    attention_scores = outputs['all_layer_metrics'][0].get('attention_scores', None)
                    if attention_scores is not None:
                        fig = visualize_attention(attention_scores[0], title="Attention Pattern")
                        writer.add_figure('visualization/attention', fig, epoch)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model (only on master process)
            if rank == 0:
                best_model_path = output_dir / "model_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': config_dict
                }, best_model_path)
                
                print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save checkpoint (only on master process)
        if rank == 0 and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / "checkpoint.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config_dict
            }, checkpoint_path)
            
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Clean up distributed training
    if world_size > 1:
        cleanup()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Quantum Flux model")
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--gpu', type=str, default='0', help='GPU indices separated by comma, or -1 for CPU')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Checkpoint interval (epochs)')
    parser.add_argument('--sample-interval', type=int, default=1, help='Text sampling interval (epochs)')
    parser.add_argument('--visualize-interval', type=int, default=1, help='Visualization interval (epochs)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
