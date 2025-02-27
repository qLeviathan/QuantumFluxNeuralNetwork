# Quantum Flux: Getting Started Guide

This guide provides step-by-step instructions for validating, testing, and training Quantum Flux models. Follow these instructions to ensure a smooth setup and effective usage of the framework.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Initial Validation](#2-initial-validation)
3. [Component Testing](#3-component-testing)
4. [Training Setup](#4-training-setup)
5. [Monitoring and Evaluation](#5-monitoring-and-evaluation)
6. [Troubleshooting](#6-troubleshooting)
7. [Next Steps](#7-next-steps)

## 1. Environment Setup

### 1.1 Installation

Start by setting up a Python environment and installing the required packages:

```bash
# Create a new conda environment
conda create -n quantum-flux python=3.8
conda activate quantum-flux

# Clone the repository
git clone https://github.com/yourusername/quantum-flux.git
cd quantum-flux

# Install dependencies
pip install -e .
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers datasets wandb matplotlib tqdm pyyaml
```

### 1.2 Environment Verification

Verify your GPU setup and PyTorch installation:

```bash
# Check PyTorch and GPU availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 1.3 Configuration

Set up your configuration file by modifying the template in `configs/base.yaml`:

```bash
# Copy the base config for customization
cp configs/base.yaml configs/custom.yaml
```

Edit `configs/custom.yaml` to match your GPU capabilities and data requirements.

## 2. Initial Validation

Before running full training, validate the basic functionality of the framework:

### 2.1 Geometry Module Validation

Test the geometric transformations:

```bash
python -c "
from quantum_flux.geometry import polar_to_cartesian, cartesian_to_polar
import torch
import math

# Test with sample values
r = torch.tensor([0.5, 1.0, 1.5])
theta = torch.tensor([0.0, math.pi/4, math.pi/2])

# Convert to Cartesian
rx, ry = polar_to_cartesian(r, theta)
print(f'Polar to Cartesian: r={r}, theta={theta} → rx={rx}, ry={ry}')

# Convert back to polar
r_new, theta_new = cartesian_to_polar(rx, ry)
print(f'Cartesian to Polar: rx={rx}, ry={ry} → r={r_new}, theta={theta_new}')

# Check error
r_error = torch.abs(r - r_new).max().item()
theta_error = torch.abs(theta - theta_new).max().item()
print(f'Max errors: r_error={r_error}, theta_error={theta_error}')

assert r_error < 1e-5 and theta_error < 1e-5, 'Conversion error too large!'
print('Geometry module validation successful!')
"
```

### 2.2 Model Instantiation

Validate model creation and forward pass:

```bash
python -c "
from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel
import torch

# Create minimal config
config = QuantumFluxConfig(
    embed_dim=32,
    max_seq_length=16,
    num_layers=2,
    vocab_size=100
)

# Create model
model = QuantumFluxModel(config)
print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters')

# Test forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 10))
outputs = model(input_ids)

# Check outputs
print(f'Output logits shape: {outputs[\"logits\"].shape}')
print(f'Quantum states shape: {outputs[\"quantum_states\"].shape}')
print('Model instantiation and forward pass successful!')
"
```

## 3. Component Testing

Run the comprehensive test script to validate all components:

```bash
# Run the test script
python test_quantum_flux.py

# If you want more verbose output
python test_quantum_flux.py --verbose
```

This script tests:
- Geometric transformations
- Encoder functionality
- Attention mechanism
- Integration methods
- Full model training and generation
- Visualization utilities

Make sure all tests pass before proceeding to full training.

## 4. Training Setup

### 4.1 Download and Prepare Dataset

Set up a dataset for training:

```bash
# Create data directory
mkdir -p data

# For this example, we'll use Wikitext-103 from Hugging Face
python -c "
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
dataset.save_to_disk('data/wikitext-103')
print('Dataset downloaded and saved to data/wikitext-103')
"
```

### 4.2 Small-Scale Training

Before running a full training job, test with a small subset of data:

```bash
# Run a short training job on a small dataset
python scripts/train_wandb.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output ./output/test_run \
    --batch_size 4 \
    --epochs 2 \
    --max_seq_length 128 \
    --compare_transformer
```

This small-scale run helps verify:
- Data loading pipeline works
- Training loop executes without errors
- Forward and backward passes function correctly
- Metrics are calculated properly

### 4.3 Weights & Biases Setup

For production training, set up Weights & Biases logging:

```bash
# Login to wandb
wandb login

# Test wandb integration
python scripts/train_wandb.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output ./output/wandb_test \
    --wandb \
    --wandb_project quantum-flux \
    --wandb_name "test-run" \
    --epochs 1
```

## 5. Monitoring and Evaluation

### 5.1 Full Training

Launch a full training run:

```bash
# Start full training
python scripts/train_wandb.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output ./output/full_run \
    --wandb \
    --wandb_project quantum-flux \
    --wandb_name "qf-base-wikitext" \
    --batch_size 32 \
    --gradient_accumulation 4 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --sample_interval 1 \
    --visualize_interval 1
```

### 5.2 Multi-GPU Training

For multi-GPU training:

```bash
# Use multiple GPUs (adjust indices as needed)
python scripts/train_wandb.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output ./output/multi_gpu_run \
    --wandb \
    --wandb_project quantum-flux \
    --wandb_name "qf-multi-gpu" \
    --gpu 0,1,2,3 \
    --batch_size 8 \
    --gradient_accumulation 1
```

### 5.3 Monitoring

While training, monitor progress through:

1. **Weights & Biases Dashboard**: Track metrics, visualizations, and sample generations
   - Visit your project page at https://wandb.ai/your-username/quantum-flux

2. **TensorBoard**: For local visualization
   ```bash
   tensorboard --logdir ./output/full_run/tensorboard
   ```

3. **Log Files**: Check console output and log files in the output directory

### 5.4 Evaluation

Evaluate the trained model:

```bash
# Run evaluation on test set
python scripts/evaluate.py \
    --model_path ./output/full_run/model_best.pt \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --split test \
    --batch_size 16
```

## 6. Troubleshooting

### 6.1 Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| **CUDA out of memory** | Batch size too large | Reduce batch size or use gradient accumulation |
| **NaN loss values** | Learning rate too high or initialization issues | Reduce learning rate or check initialization |
| **Slow training** | Inefficient data loading or large sequence length | Increase num_workers or reduce sequence length |
| **Poor convergence** | Inappropriate hyperparameters | Adjust learning rate, radius parameters, or hebbian settings |

### 6.2 Debugging Tips

If you encounter issues:

1. **Enable verbose logging**:
   ```bash
   python scripts/train_wandb.py --verbose ...
   ```

2. **Check numerical stability**:
   ```python
   # Add this to model code for debugging
   if torch.isnan(tensor).any():
       print(f"NaN detected in {tensor_name}")
       # Save current state for analysis
       torch.save(..., "debug_snapshot.pt")
   ```

3. **Profile performance**:
   ```bash
   python -m torch.utils.bottleneck scripts/train_wandb.py ...
   ```

## 7. Next Steps

After successful training and evaluation:

### 7.1 Model Export

Export your model for production:

```bash
python scripts/export_model.py \
    --model_path ./output/full_run/model_best.pt \
    --output_dir ./exported_models \
    --format onnx,torchscript
```

### 7.2 Optimization

Fine-tune model performance:

```bash
# Hyperparameter optimization
python scripts/hyperparameter_search.py \
    --config configs/base.yaml \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output ./output/hyperparam_search \
    --wandb \
    --wandb_project quantum-flux-hyperopt
```

### 7.3 Scaling Up

For larger models and datasets:

1. **Increase model dimensions**:
   - Edit `configs/large.yaml` with larger embed_dim and num_layers
   - Adjust batch size and learning rate accordingly

2. **Use more efficient data loading**:
   - Implement memory-mapped datasets for large corpora
   - Pre-tokenize and cache dataset for faster loading

3. **Implement advanced techniques**:
   - Gradient checkpointing for memory efficiency
   - Mixed-precision training for speed
   - Model parallelism for very large models

## Conclusion

You've now set up, validated, and trained a Quantum Flux model. This framework offers an innovative approach to neural network design, with unique properties derived from quantum physics principles.

For more detailed information, refer to:
- [Reference Guide](reference.md) for theoretical background
- [Quantum Geometry Tutorial](training/quantum_geometry.md) for in-depth understanding
- [Transformer Comparison](transformer_comparison.md) for performance analysis
