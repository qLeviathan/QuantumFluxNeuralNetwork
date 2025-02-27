"""
Quantum Flux BLEU Score Evaluation Script
=====================================

This script evaluates the Quantum Flux model using BLEU scores and other
standard industry metrics for machine translation or text generation.

It supports:
- BLEU, ROUGE, METEOR, and other metrics
- Multiple language pairs for translation tasks
- Comparison with reference transformer models
- SACREBLEU industry standard implementation
- Detailed report generation

Usage:
```
python scripts/evaluate_bleu.py \
    --model_path /path/to/model.pt \
    --dataset wmt16 \
    --subset de-en \
    --direction en-de \
    --max_samples 1000 \
    --beam_size 5 \
    --output_file results/bleu_results.json
```
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_flux.config import QuantumFluxConfig
from quantum_flux.model import QuantumFluxModel


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


def load_transformer_model(config, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a transformer model for comparison.
    
    Parameters:
    ----------
    config : QuantumFluxConfig
        Configuration for parameter matching
    model_name : str
        Model name from Hugging Face
    device : str
        Device to load the model on
        
    Returns:
    -------
    tuple
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to device
        model.to(device)
        model.eval()
        
        print(f"Transformer model '{model_name}' loaded successfully on {device}")
        
        return model, tokenizer
    
    except ImportError:
        print("Transformers library not available")
        return None, None


def load_dataset(dataset_name, subset=None, split="test", direction=None, max_samples=1000):
    """
    Load dataset for evaluation.
    
    Parameters:
    ----------
    dataset_name : str
        Dataset name from Hugging Face
    subset : str
        Dataset subset (e.g., "de-en" for WMT)
    split : str
        Dataset split
    direction : str
        Translation direction (e.g., "en-de" means translate from English to German)
    max_samples : int
        Maximum number of samples to load
        
    Returns:
    -------
    tuple
        (sources, references)
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
        
        # Handle different dataset formats
        if dataset_name.startswith("wmt"):
            # WMT datasets have "translation" field with language pairs
            if direction:
                src_lang, tgt_lang = direction.split("-")
                
                if src_lang not in dataset[0]["translation"] or tgt_lang not in dataset[0]["translation"]:
                    raise ValueError(f"Invalid direction {direction}. Available fields: {dataset[0]['translation'].keys()}")
                
                sources = [example["translation"][src_lang] for example in dataset]
                references = [example["translation"][tgt_lang] for example in dataset]
            else:
                # Default to first pair in the dataset
                lang_pair = list(dataset[0]["translation"].keys())
                src_lang, tgt_lang = lang_pair[0], lang_pair[1]
                
                sources = [example["translation"][src_lang] for example in dataset]
                references = [example["translation"][tgt_lang] for example in dataset]
        else:
            # For other datasets, try to find appropriate fields
            if "source" in dataset[0] and "target" in dataset[0]:
                sources = [example["source"] for example in dataset]
                references = [example["target"] for example in dataset]
            elif "text" in dataset[0]:
                # For language modeling, use consecutive sentences
                text_field = "text"
                texts = [example[text_field] for example in dataset]
                
                # Split into sentences
                sentences = []
                for text in texts:
                    sentences.extend([s.strip() for s in text.split(".") if s.strip()])
                
                # Create source-reference pairs
                sources = sentences[:-1]
                references = sentences[1:]
                
                # Limit to max_samples
                if max_samples and max_samples < len(sources):
                    sources = sources[:max_samples]
                    references = references[:max_samples]
            else:
                # Try the first two string fields
                string_fields = [k for k, v in dataset[0].items() if isinstance(v, str)]
                
                if len(string_fields) >= 2:
                    sources = [example[string_fields[0]] for example in dataset]
                    references = [example[string_fields[1]] for example in dataset]
                else:
                    raise ValueError(f"Cannot determine source and reference fields in dataset: {dataset[0].keys()}")
        
        print(f"Loaded {len(sources)} source-reference pairs from {dataset_name}")
        
        return sources, references
    
    except ImportError:
        print("Datasets library not available, using dummy data")
        
        # Create dummy data
        sources = ["This is a test sentence." for _ in range(max_samples)]
        references = ["This is a reference sentence." for _ in range(max_samples)]
        
        return sources, references


def generate_translations(model, sources, tokenizer, batch_size=8, max_length=128, beam_size=5, 
                         num_return_sequences=1, device="cuda" if torch.cuda.is_available() else "cpu",
                         model_type="quantum_flux"):
    """
    Generate translations or continuations from the model.
    
    Parameters:
    ----------
    model : torch.nn.Module
        Model to evaluate
    sources : List[str]
        Source texts
    tokenizer : tokenizer
        Tokenizer to use
    batch_size : int
        Batch size for generation
    max_length : int
        Maximum length of generated text
    beam_size : int
        Beam size for beam search
    num_return_sequences : int
        Number of sequences to return per source
    device : str
        Device to run generation on
    model_type : str
        Type of model ("quantum_flux" or "transformer")
        
    Returns:
    -------
    List
        Generated texts
    """
    model.eval()
    translations = []
    
    for i in tqdm(range(0, len(sources), batch_size), desc="Generating"):
        batch = sources[i:i+batch_size]
        
        # Tokenize
        tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = tokenized.input_ids.to(device)
        
        with torch.no_grad():
            if model_type == "quantum_flux":
                # Generate with Quantum Flux model
                generated_ids = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    top_k=50
                )
                
                # Decode
                for j in range(len(batch)):
                    translation = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                    translations.append(translation)
            else:
                # Generate with transformer model
                try:
                    generated_ids = model.generate(
                        input_ids,
                        max_length=max_length,
                        num_beams=beam_size,
                        num_return_sequences=num_return_sequences,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    
                    # Decode
                    for j in range(len(batch)):
                        idx = j * num_return_sequences
                        translation = tokenizer.decode(generated_ids[idx], skip_special_tokens=True)
                        translations.append(translation)
                except Exception as e:
                    print(f"Error generating with transformer: {e}")
                    # Fallback to simpler generation
                    generated_ids = model.generate(
                        input_ids,
                        max_length=max_length
                    )
                    
                    # Decode
                    for j in range(len(batch)):
                        translation = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                        translations.append(translation)
    
    return translations


def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score using sacrebleu.
    
    Parameters:
    ----------
    references : List[str]
        Reference texts
    hypotheses : List[str]
        Generated texts
        
    Returns:
    -------
    float
        BLEU score
    """
    try:
        import sacrebleu
        
        # Calculate BLEU score
        refs = [references]  # sacrebleu expects multiple reference translations
        score = sacrebleu.corpus_bleu(hypotheses, refs)
        
        return {
            'score': score.score,
            'precisions': score.precisions,
            'bp': score.bp,
            'ratio': score.ratio,
            'sys_len': score.sys_len,
            'ref_len': score.ref_len
        }
    
    except ImportError:
        print("sacrebleu not available, using nltk.bleu")
        
        try:
            from nltk.translate.bleu_score import corpus_bleu
            
            # Create reference format for nltk.bleu
            refs = [[r.split()] for r in references]
            hyps = [h.split() for h in hypotheses]
            
            # Calculate BLEU score
            score = corpus_bleu(refs, hyps)
            
            return {
                'score': score * 100,  # Convert to same scale as sacrebleu
                'precisions': None,
                'bp': None,
                'ratio': None,
                'sys_len': sum(len(h) for h in hyps),
                'ref_len': sum(len(r[0]) for r in refs)
            }
        
        except ImportError:
            print("nltk not available, using simple word overlap")
            
            # Calculate simple word overlap
            total_overlap = 0
            total_words = 0
            
            for r, h in zip(references, hypotheses):
                r_words = set(r.split())
                h_words = set(h.split())
                overlap = len(r_words.intersection(h_words))
                
                total_overlap += overlap
                total_words += len(h_words)
            
            score = (total_overlap / total_words) * 100 if total_words > 0 else 0
            
            return {
                'score': score,
                'precisions': None,
                'bp': None,
                'ratio': None,
                'sys_len': total_words,
                'ref_len': sum(len(r.split()) for r in references)
            }


def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE scores.
    
    Parameters:
    ----------
    references : List[str]
        Reference texts
    hypotheses : List[str]
        Generated texts
        
    Returns:
    -------
    dict
        ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
        
        # Initialize scorer with multiple metrics
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate scores for each pair
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = scorer.score(ref, hyp)
            scores.append(score)
        
        # Calculate average scores
        avg_scores = {}
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[metric] = {
                'precision': np.mean([s[metric].precision for s in scores]),
                'recall': np.mean([s[metric].recall for s in scores]),
                'fmeasure': np.mean([s[metric].fmeasure for s in scores])
            }
        
        return avg_scores
    
    except ImportError:
        print("rouge_score not available")
        return None


def calculate_meteor(references, hypotheses):
    """
    Calculate METEOR score.
    
    Parameters:
    ----------
    references : List[str]
        Reference texts
    hypotheses : List[str]
        Generated texts
        
    Returns:
    -------
    float
        METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        
        # Ensure required resources are available
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Calculate score for each pair
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            score = meteor_score([ref_tokens], hyp_tokens)
            scores.append(score)
        
        # Calculate average
        avg_score = np.mean(scores)
        
        return avg_score
    
    except ImportError:
        print("nltk.translate.meteor_score not available")
        return None


def evaluate_model(model, sources, references, tokenizer, batch_size=8, max_length=128, beam_size=5,
                  device="cuda" if torch.cuda.is_available() else "cpu", model_type="quantum_flux"):
    """
    Evaluate the model on a dataset.
    
    Parameters:
    ----------
    model : torch.nn.Module
        Model to evaluate
    sources : List[str]
        Source texts
    references : List[str]
        Reference texts
    tokenizer : tokenizer
        Tokenizer to use
    batch_size : int
        Batch size for generation
    max_length : int
        Maximum length of generated text
    beam_size : int
        Beam size for beam search
    device : str
        Device to run evaluation on
    model_type : str
        Type of model ("quantum_flux" or "transformer")
        
    Returns:
    -------
    dict
        Evaluation results
    """
    # Generate translations
    print(f"Generating translations with {model_type} model...")
    translations = generate_translations(
        model, sources, tokenizer, batch_size, max_length, beam_size, 
        num_return_sequences=1, device=device, model_type=model_type
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    
    results = {}
    
    # BLEU score
    bleu = calculate_bleu(references, translations)
    results['bleu'] = bleu
    print(f"BLEU score: {bleu['score']:.2f}")
    
    # ROUGE score
    rouge = calculate_rouge(references, translations)
    if rouge:
        results['rouge'] = rouge
        print(f"ROUGE-1 F1: {rouge['rouge1']['fmeasure']:.4f}")
        print(f"ROUGE-2 F1: {rouge['rouge2']['fmeasure']:.4f}")
        print(f"ROUGE-L F1: {rouge['rougeL']['fmeasure']:.4f}")
    
    # METEOR score
    meteor = calculate_meteor(references, translations)
    if meteor:
        results['meteor'] = meteor
        print(f"METEOR score: {meteor:.4f}")
    
    # Save examples
    num_examples = min(10, len(translations))
    examples = []
    
    for i in range(num_examples):
        examples.append({
            'source': sources[i],
            'reference': references[i],
            'translation': translations[i]
        })
    
    results['examples'] = examples
    
    return results


def compare_models(qf_model, tf_model, sources, references, tokenizer, tf_tokenizer=None,
                  batch_size=8, max_length=128, beam_size=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compare Quantum Flux and Transformer models.
    
    Parameters:
    ----------
    qf_model : QuantumFluxModel
        Quantum Flux model
    tf_model : TransformerModel
        Transformer model
    sources : List[str]
        Source texts
    references : List[str]
        Reference texts
    tokenizer : tokenizer
        Tokenizer for Quantum Flux model
    tf_tokenizer : tokenizer or None
        Tokenizer for Transformer model
    batch_size : int
        Batch size for generation
    max_length : int
        Maximum length of generated text
    beam_size : int
        Beam size for beam search
    device : str
        Device to run evaluation on
        
    Returns:
    -------
    dict
        Comparison results
    """
    results = {}
    
    # Evaluate Quantum Flux model
    print("\nEvaluating Quantum Flux model...")
    qf_results = evaluate_model(
        qf_model, sources, references, tokenizer, batch_size, max_length, beam_size, 
        device=device, model_type="quantum_flux"
    )
    results['quantum_flux'] = qf_results
    
    # Evaluate Transformer model
    if tf_model:
        print("\nEvaluating Transformer model...")
        tf_results = evaluate_model(
            tf_model, sources, references, tf_tokenizer or tokenizer, batch_size, max_length, beam_size, 
            device=device, model_type="transformer"
        )
        results['transformer'] = tf_results
        
        # Calculate comparative metrics
        results['comparison'] = {
            'bleu_ratio': qf_results['bleu']['score'] / tf_results['bleu']['score'] if tf_results['bleu']['score'] > 0 else float('inf'),
            'bleu_difference': qf_results['bleu']['score'] - tf_results['bleu']['score']
        }
        
        if 'rouge' in qf_results and 'rouge' in tf_results:
            results['comparison']['rouge1_ratio'] = qf_results['rouge']['rouge1']['fmeasure'] / tf_results['rouge']['rouge1']['fmeasure'] if tf_results['rouge']['rouge1']['fmeasure'] > 0 else float('inf')
            results['comparison']['rouge2_ratio'] = qf_results['rouge']['rouge2']['fmeasure'] / tf_results['rouge']['rouge2']['fmeasure'] if tf_results['rouge']['rouge2']['fmeasure'] > 0 else float('inf')
            results['comparison']['rougeL_ratio'] = qf_results['rouge']['rougeL']['fmeasure'] / tf_results['rouge']['rougeL']['fmeasure'] if tf_results['rouge']['rougeL']['fmeasure'] > 0 else float('inf')
        
        if 'meteor' in qf_results and 'meteor' in tf_results:
            results['comparison']['meteor_ratio'] = qf_results['meteor'] / tf_results['meteor'] if tf_results['meteor'] > 0 else float('inf')
            results['comparison']['meteor_difference'] = qf_results['meteor'] - tf_results['meteor']
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Flux BLEU Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on")
    parser.add_argument("--dataset", type=str, default="wmt16", help="Dataset to use for evaluation")
    parser.add_argument("--subset", type=str, default="de-en", help="Dataset subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--direction", type=str, default=None, help="Translation direction (e.g., en-de)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated text")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--compare_transformer", action="store_true", help="Compare with transformer model")
    parser.add_argument("--transformer_model", type=str, default="gpt2", help="Transformer model to compare with")
    parser.add_argument("--output_file", type=str, default="bleu_results.json", help="Output file for results")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load Quantum Flux model
    print(f"Loading Quantum Flux model from {args.model_path}")
    qf_model, config, tokenizer = load_model(args.model_path, device)
    
    # Load transformer model for comparison if requested
    tf_model, tf_tokenizer = None, None
    if args.compare_transformer:
        print(f"Loading transformer model '{args.transformer_model}' for comparison")
        tf_model, tf_tokenizer = load_transformer_model(config, args.transformer_model, device)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}")
    sources, references = load_dataset(args.dataset, args.subset, args.split, args.direction, args.max_samples)
    
    # Compare models
    if tf_model:
        print("\nComparing models...")
        results = compare_models(
            qf_model, tf_model, sources, references, tokenizer, tf_tokenizer,
            args.batch_size, args.max_length, args.beam_size, device
        )
    else:
        # Evaluate Quantum Flux model only
        print("\nEvaluating Quantum Flux model...")
        results = evaluate_model(
            qf_model, sources, references, tokenizer, args.batch_size, args.max_length, args.beam_size, 
            device=device, model_type="quantum_flux"
        )
    
    # Add metadata
    results['metadata'] = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'subset': args.subset,
        'direction': args.direction,
        'samples': min(args.max_samples, len(sources)),
        'beam_size': args.beam_size,
        'max_length': args.max_length
    }
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
