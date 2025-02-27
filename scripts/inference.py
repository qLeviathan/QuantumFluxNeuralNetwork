"""
Quantum Flux Inference Server
============================

This script provides a FastAPI-based inference server for the Quantum Flux model.

It supports:
- Text generation with various parameters
- Model loading from checkpoints
- ONNX and TorchScript optimized inference
- Performance metrics and statistics
- Batch request handling

Usage:
```
python scripts/inference.py --model_path /path/to/model.pt --port 8000
```

API Endpoints:
- POST /generate: Generate text from a prompt
- GET /models/info: Get information about the loaded model
- GET /stats: Get inference statistics
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
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import yaml
from threading import Lock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel


# Define request and response models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text to generate from")
    max_length: int = Field(50, description="Maximum length of generated text")
    temperature: float = Field(0.8, description="Temperature for sampling (higher = more random)")
    top_k: int = Field(50, description="Number of top tokens to consider for sampling")
    top_p: float = Field(0.9, description="Cumulative probability threshold for nucleus sampling")
    num_return_sequences: int = Field(1, description="Number of sequences to generate")
    use_onnx: bool = Field(False, description="Whether to use ONNX optimized runtime")
    batch_size: Optional[int] = Field(None, description="Batch size for generation (for benchmarking)")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerationResponse(BaseModel):
    generated_text: List[str] = Field(..., description="Generated text sequences")
    prompt: str = Field(..., description="Original prompt")
    stats: Dict[str, Any] = Field(..., description="Generation statistics")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    parameters: int = Field(..., description="Number of model parameters")
    device: str = Field(..., description="Device the model is running on")
    optimizations: List[str] = Field(..., description="Applied optimizations")


class StatsResponse(BaseModel):
    total_requests: int = Field(..., description="Total number of requests processed")
    avg_latency: float = Field(..., description="Average latency in seconds")
    avg_tokens_per_second: float = Field(..., description="Average tokens generated per second")
    max_batch_size: int = Field(..., description="Maximum batch size processed")


# Initialize FastAPI app
app = FastAPI(title="Quantum Flux Inference API", 
              description="API for text generation with Quantum Flux Neural Network",
              version="1.0.0")

# Global variables
model = None
tokenizer = None
config = None
device = None
onnx_session = None
torchscript_model = None
model_lock = Lock()

# Statistics
stats = {
    "total_requests": 0,
    "total_latency": 0,
    "total_tokens": 0,
    "max_batch_size": 0,
}


def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu", use_onnx=False, use_torchscript=False):
    """
    Load the Quantum Flux model from a checkpoint.
    
    Parameters:
    ----------
    model_path : str
        Path to the model checkpoint
    device : str
        Device to load the model on
    use_onnx : bool
        Whether to load an ONNX optimized model
    use_torchscript : bool
        Whether to load a TorchScript optimized model
        
    Returns:
    -------
    tuple
        (model, config, tokenizer)
    """
    global onnx_session, torchscript_model
    
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
        
        # Load optimized models if requested
        optimizations = []
        
        if use_onnx:
            try:
                import onnx
                import onnxruntime as ort
                
                # Check if ONNX file exists, otherwise convert
                onnx_path = Path(model_path).with_suffix('.onnx')
                if not onnx_path.exists():
                    print(f"Converting model to ONNX format: {onnx_path}")
                    
                    # Export to ONNX
                    dummy_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        input_names=['input_ids'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                            'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
                        },
                        opset_version=12
                    )
                
                # Load ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                # Create ONNX runtime session
                ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.startswith('cuda') else ['CPUExecutionProvider']
                onnx_session = ort.InferenceSession(onnx_path, providers=ort_providers)
                
                optimizations.append("ONNX Runtime")
                print("ONNX model loaded successfully")
                
            except ImportError:
                print("ONNX Runtime not available, falling back to PyTorch")
                onnx_session = None
        
        if use_torchscript and not use_onnx:
            try:
                # Check if TorchScript file exists, otherwise convert
                torchscript_path = Path(model_path).with_suffix('.pt.ts')
                if not torchscript_path.exists():
                    print(f"Converting model to TorchScript format: {torchscript_path}")
                    
                    # Script the model
                    dummy_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
                    
                    # Simplified forward pass for TorchScript compatibility
                    def forward_wrapper(input_ids):
                        outputs = model(input_ids)
                        return outputs['logits']
                    
                    model.forward_script = forward_wrapper
                    scripted_model = torch.jit.trace(model.forward_script, dummy_input)
                    scripted_model.save(torchscript_path)
                
                # Load TorchScript model
                torchscript_model = torch.jit.load(torchscript_path, map_location=device)
                
                optimizations.append("TorchScript")
                print("TorchScript model loaded successfully")
                
            except Exception as e:
                print(f"TorchScript conversion failed: {e}, falling back to PyTorch")
                torchscript_model = None
        
        print(f"Model loaded successfully on {device}")
        
        # Try to load tokenizer if available
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
            tokenizer = None
        
        return model, config, tokenizer, optimizations
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def generate_text(prompt, max_length=50, temperature=0.8, top_k=50, top_p=0.9, 
                 num_return_sequences=1, use_onnx=False, batch_size=None, stream=False):
    """
    Generate text from a prompt.
    
    Parameters:
    ----------
    prompt : str
        Input text to generate from
    max_length : int
        Maximum length of generated text
    temperature : float
        Temperature for sampling
    top_k : int
        Number of top tokens to consider for sampling
    top_p : float
        Cumulative probability threshold for nucleus sampling
    num_return_sequences : int
        Number of sequences to generate
    use_onnx : bool
        Whether to use ONNX optimized runtime
    batch_size : int or None
        Batch size for generation
    stream : bool
        Whether to stream the response
        
    Returns:
    -------
    tuple
        (generated_texts, stats)
    """
    global model, tokenizer, config, device, onnx_session, torchscript_model, stats
    
    with model_lock:
        start_time = time.time()
        
        # Tokenize input
        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            # Simple fallback tokenization (character-level)
            chars = list(set(prompt))
            char_to_id = {c: i for i, c in enumerate(chars)}
            input_ids = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
        
        # Record original input length
        input_length = input_ids.shape[1]
        
        # Apply batch size if needed (for benchmarking)
        if batch_size is not None and batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            stats["max_batch_size"] = max(stats["max_batch_size"], batch_size)
        else:
            batch_size = 1
        
        # Generate text using the appropriate runtime
        if use_onnx and onnx_session is not None:
            # Generate with ONNX runtime
            generated_ids = input_ids.cpu().numpy()
            
            for _ in range(max_length - input_length):
                # Get the last token
                last_tokens = generated_ids[:, -10:] if generated_ids.shape[1] > 10 else generated_ids
                
                # Run ONNX inference
                ort_inputs = {onnx_session.get_inputs()[0].name: last_tokens}
                ort_outputs = onnx_session.run(None, ort_inputs)
                logits = ort_outputs[0][:, -1, :]
                
                # Temperature scaling
                logits = logits / temperature
                
                # Convert to PyTorch for sampling
                logits_tensor = torch.tensor(logits, device=device)
                
                # Apply top-k and top-p sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits_tensor, k=min(top_k, logits_tensor.shape[-1]))
                    logits_tensor = torch.full_like(logits_tensor, float('-inf'))
                    logits_tensor.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_tensor, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for b in range(batch_size):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        logits_tensor[b, indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits_tensor, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).cpu().numpy()
                
                # Append to generated sequence
                generated_ids = np.concatenate([generated_ids, next_tokens], axis=1)
                
                # Stream output if requested
                if stream and batch_size == 1:
                    yield {"token": int(next_tokens[0, 0])}
        
        elif torchscript_model is not None and not use_onnx:
            # Generate with TorchScript
            with torch.no_grad():
                generated_ids = input_ids
                
                for _ in range(max_length - input_length):
                    # Get the last tokens (context window)
                    last_tokens = generated_ids[:, -10:] if generated_ids.shape[1] > 10 else generated_ids
                    
                    # Run TorchScript inference
                    logits = torchscript_model(last_tokens)
                    
                    # Get logits for the last token
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Apply top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for b in range(batch_size):
                            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                            next_token_logits[b, indices_to_remove] = float('-inf')
                    
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                    
                    # Stream output if requested
                    if stream and batch_size == 1:
                        yield {"token": int(next_tokens[0, 0])}
        
        else:
            # Generate with native PyTorch model
            with torch.no_grad():
                # Use model's generate method
                generated_ids = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Implement streaming if requested
                if stream and batch_size == 1:
                    for i in range(input_length, generated_ids.shape[1]):
                        yield {"token": int(generated_ids[0, i])}
        
        # Decode generated sequences
        generated_texts = []
        
        if tokenizer:
            for i in range(min(batch_size, num_return_sequences)):
                generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                generated_texts.append(generated_text)
        else:
            # Simple fallback decoding (character-level)
            id_to_char = {i: c for c, i in char_to_id.items()}
            for i in range(min(batch_size, num_return_sequences)):
                generated_text = ''.join([id_to_char.get(int(idx), '?') for idx in generated_ids[i]])
                generated_texts.append(generated_text)
        
        # Calculate statistics
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_generated = generated_ids.shape[1] - input_length
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Update global stats
        stats["total_requests"] += 1
        stats["total_latency"] += generation_time
        stats["total_tokens"] += tokens_generated
        
        generation_stats = {
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
        }
        
        if not stream:
            return generated_texts, generation_stats


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text from a prompt."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                generate_text(
                    request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    num_return_sequences=request.num_return_sequences,
                    use_onnx=request.use_onnx,
                    batch_size=request.batch_size,
                    stream=True
                ),
                media_type="application/json"
            )
        else:
            # Return a normal response
            generated_texts, generation_stats = generate_text(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                use_onnx=request.use_onnx,
                batch_size=request.batch_size
            )
            
            return {
                "generated_text": generated_texts,
                "prompt": request.prompt,
                "stats": generation_stats
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get applied optimizations
        optimizations = []
        if onnx_session is not None:
            optimizations.append("ONNX Runtime")
        if torchscript_model is not None:
            optimizations.append("TorchScript")
        
        return {
            "model_name": "Quantum Flux Neural Network",
            "config": config.__dict__,
            "parameters": count_parameters(model),
            "device": str(device),
            "optimizations": optimizations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get inference statistics."""
    try:
        avg_latency = stats["total_latency"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
        avg_tokens_per_second = stats["total_tokens"] / stats["total_latency"] if stats["total_latency"] > 0 else 0
        
        return {
            "total_requests": stats["total_requests"],
            "avg_latency": avg_latency,
            "avg_tokens_per_second": avg_tokens_per_second,
            "max_batch_size": stats["max_batch_size"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, config, tokenizer, device, stats
    
    try:
        # Get model path from environment or use default
        model_path = os.environ.get("MODEL_PATH", args.model_path)
        device_name = os.environ.get("DEVICE", args.device)
        use_onnx = os.environ.get("USE_ONNX", "0") == "1" or args.use_onnx
        use_torchscript = os.environ.get("USE_TORCHSCRIPT", "0") == "1" or args.use_torchscript
        
        device = torch.device(device_name)
        
        # Load model
        model, config, tokenizer, optimizations = load_model(
            model_path, 
            device=device, 
            use_onnx=use_onnx, 
            use_torchscript=use_torchscript
        )
        
        print(f"Model loaded successfully on {device}")
        print(f"Applied optimizations: {optimizations}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue without a model, will return errors on API calls


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Flux Inference Server")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX runtime for inference")
    parser.add_argument("--use_torchscript", action="store_true", help="Use TorchScript for inference")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)
