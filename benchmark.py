"""
Benchmarking script for TinyMamba models

This script:
1. Compares TinyMamba against standard Transformer models
2. Measures inference speed, memory usage, perplexity and more
3. Generates comparison plots and reports
"""

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

# Import TinyMamba model
from testv2 import TinyMambaModel, Config, TextGenerator, TinyBPETokenizer

# Try to import huggingface transformers - we'll need these for comparison
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers")

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TinyMamba against Transformer models")
    parser.add_argument(
        "--tinymamba_checkpoint", 
        type=str, 
        default="./tinymamba_model_latest.pt",
        help="Path to TinyMamba checkpoint file"
    )
    parser.add_argument(
        "--models_to_compare", 
        type=str, 
        nargs="+",
        default=["gpt2", "tinymamba"],
        help="Models to include in benchmark (e.g., gpt2, gpt2-medium, tinymamba)"
    )
    parser.add_argument(
        "--benchmark_type", 
        type=str, 
        default="all",
        choices=["speed", "memory", "perplexity", "all"],
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--batch_sizes", 
        type=int, 
        nargs="+",
        default=[1, 4, 16],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--sequence_lengths", 
        type=int, 
        nargs="+",
        default=[128, 512, 1024],
        help="Sequence lengths to test"
    )
    parser.add_argument(
        "--warmup_iterations", 
        type=int, 
        default=5,
        help="Number of warmup iterations before measurement"
    )
    parser.add_argument(
        "--test_iterations", 
        type=int, 
        default=50,
        help="Number of test iterations for measurement"
    )
    parser.add_argument(
        "--eval_dataset", 
        type=str, 
        default=None,
        help="Path to evaluation dataset for perplexity testing"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for benchmarking (cuda/cpu)"
    )
    parser.add_argument(
        "--generate_plots", 
        action="store_true",
        help="Generate comparison plots"
    )
    return parser.parse_args()

def load_tinymamba_model(checkpoint_path, device):
    """Load TinyMamba model from checkpoint"""
    print(f"Loading TinyMamba model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config()
        
        # Update config with values from checkpoint
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        print("Warning: Config not found in checkpoint, using default config")
        config = Config()
    
    # Initialize model with config
    model = TinyMambaModel(config)
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Model weights not found in checkpoint")
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, config

def load_transformer_model(model_name, device):
    """Load a HuggingFace transformer model"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required to load transformer models")
    
    print(f"Loading transformer model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def create_custom_gpt2(d_model, n_layer, vocab_size, device="cuda"):
    """Create a custom GPT2 model matching TinyMamba dimensions for fair comparison"""
    config = GPT2Config(
        n_embd=d_model,
        n_layer=n_layer,
        n_head=d_model // 64,  # Standard head dimension of 64
        vocab_size=vocab_size
    )
    
    model = GPT2LMHeadModel(config).to(device)
    model.eval()
    
    return model

def measure_inference_speed(model, input_data, warmup_iterations=5, test_iterations=50, desc="Measuring speed"):
    """Measure model inference speed"""
    # Move input to same device as model
    device = next(model.parameters()).device
    if isinstance(input_data, torch.Tensor) and input_data.device != device:
        input_data = input_data.to(device)
    
    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_data)
    
    # Synchronize GPU operations
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time for inference
    times = []
    with torch.no_grad():
        for _ in tqdm(range(test_iterations), desc=desc):
            start_time = time.time()
            _ = model(input_data)
            
            # Synchronize GPU operations before stopping timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate tokens per second
    batch_size, seq_len = input_data.size()
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'tokens_per_second': tokens_per_second,
        'batch_size': batch_size,
        'sequence_length': seq_len
    }

def measure_memory_usage(model, input_sizes, device="cuda", desc="Measuring memory"):
    """Measure peak memory usage for different input sizes"""
    if device != "cuda":
        print("Memory measurement only available on CUDA devices")
        return None
    
    results = []
    
    for batch_size, seq_len in tqdm(input_sizes, desc=desc):
        # Clear cache before each test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create random input
        input_data = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # Run inference
        with torch.no_grad():
            _ = model(input_data)
        
        # Record memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        
        results.append({
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'peak_memory_mb': peak_memory
        })
        
        # Clear cache after test
        torch.cuda.empty_cache()
    
    return results

def calculate_perplexity(model, input_ids, target_ids=None, stride=512):
    """Calculate perplexity on a dataset"""
    if target_ids is None:
        target_ids = input_ids
        
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    # Initialize variables for running loss calculation
    nlls = []
    total_tokens = 0
    
    # Process in smaller windows to avoid OOM for long sequences
    for i in range(0, input_ids.size(1), stride):
        end_ix = min(i + stride, input_ids.size(1))
        window_len = end_ix - i
        
        # Get window of tokens
        window_input_ids = input_ids[:, i:end_ix]
        window_target_ids = target_ids[:, i:end_ix]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(window_input_ids)
            
            # For HuggingFace models
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            # For TinyMamba
            else:
                logits = outputs
            
            # Shift logits and targets for next token prediction
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = window_target_ids[:, 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            neg_log_likelihood = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            nlls.append(neg_log_likelihood.item())
            total_tokens += shift_labels.numel()
    
    # Calculate perplexity
    total_nll = sum(nlls)
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    
    return {
        'perplexity': ppl.item(),
        'total_tokens': total_tokens,
        'avg_nll': total_nll / total_tokens
    }

def generate_comparison_plots(results, output_dir):
    """Generate comparison plots from benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot inference speed comparison
    if 'speed' in results:
        plt.figure(figsize=(10, 6))
        
        # Group by sequence length
        for seq_len in sorted(set(item['sequence_length'] for item in results['speed'])):
            df = pd.DataFrame([item for item in results['speed'] if item['sequence_length'] == seq_len])
            
            # Create grouped bar chart by model
            models = sorted(df['model'].unique())
            batch_sizes = sorted(df['batch_size'].unique())
            x = np.arange(len(batch_sizes))
            width = 0.8 / len(models)
            
            for i, model_name in enumerate(models):
                model_df = df[df['model'] == model_name]
                plt.bar(
                    x + i * width - 0.4 + width/2, 
                    model_df['tokens_per_second'], 
                    width=width, 
                    label=f"{model_name}"
                )
            
            plt.xlabel('Batch Size')
            plt.ylabel('Tokens per Second (higher is better)')
            plt.title(f'Inference Speed (Sequence Length = {seq_len})')
            plt.xticks(x, batch_sizes)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'speed_comparison_seq_{seq_len}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot memory usage comparison
    if 'memory' in results:
        # Plot memory by sequence length
        for batch_size in sorted(set(item['batch_size'] for item in results['memory'])):
            plt.figure(figsize=(10, 6))
            df = pd.DataFrame([item for item in results['memory'] if item['batch_size'] == batch_size])
            
            for model_name in sorted(df['model'].unique()):
                model_df = df[df['model'] == model_name]
                plt.plot(
                    model_df['sequence_length'],
                    model_df['peak_memory_mb'],
                    marker='o',
                    label=model_name
                )
            
            plt.xlabel('Sequence Length')
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title(f'Memory Usage (Batch Size = {batch_size})')
            plt.legend()
            plt.grid(linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'memory_comparison_batch_{batch_size}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot perplexity comparison
    if 'perplexity' in results:
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(results['perplexity'])
        
        # Sort by perplexity for better visualization
        df = df.sort_values('perplexity')
        
        plt.bar(df['model'], df['perplexity'], color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Perplexity Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def run_speed_benchmark(models_dict, batch_sizes, sequence_lengths, args):
    """Run speed benchmark for all models with various batch sizes and sequence lengths"""
    speed_results = []
    
    for model_name, (model, _) in models_dict.items():
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Create random input ids
                if model_name == 'tinymamba':
                    vocab_size = models_dict['tinymamba'][1].vocab_size
                else:
                    vocab_size = 50000  # Default for most transformer models
                
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=args.device)
                
                # Measure inference speed
                print(f"\nBenchmarking {model_name} (batch={batch_size}, seq_len={seq_len})")
                result = measure_inference_speed(
                    model, 
                    input_ids,
                    warmup_iterations=args.warmup_iterations,
                    test_iterations=args.test_iterations,
                    desc=f"Speed test: {model_name}"
                )
                
                # Add model name to result
                result['model'] = model_name
                speed_results.append(result)
                
                print(f"  Tokens per second: {result['tokens_per_second']:.2f}")
                print(f"  Average time: {result['avg_time'] * 1000:.2f} ms")
    
    return speed_results

def run_memory_benchmark(models_dict, batch_sizes, sequence_lengths, args):
    """Run memory benchmark for all models with various batch sizes and sequence lengths"""
    memory_results = []
    
    # Create input sizes to test
    input_sizes = [(bs, sl) for bs in batch_sizes for sl in sequence_lengths]
    
    for model_name, (model, _) in models_dict.items():
        print(f"\nMeasuring memory usage for {model_name}")
        model_results = measure_memory_usage(
            model,
            input_sizes,
            device=args.device,
            desc=f"Memory test: {model_name}"
        )
        
        # Add model name to results
        for result in model_results:
            result['model'] = model_name
            memory_results.append(result)
            
            print(f"  Batch={result['batch_size']}, Seq_len={result['sequence_length']}, "
                  f"Memory: {result['peak_memory_mb']:.2f} MB")
    
    return memory_results

def run_perplexity_benchmark(models_dict, eval_dataset_path, args):
    """Run perplexity benchmark on evaluation dataset"""
    if not eval_dataset_path:
        print("No evaluation dataset provided for perplexity testing")
        return []
    
    if not os.path.exists(eval_dataset_path):
        print(f"Evaluation dataset not found at {eval_dataset_path}")
        return []
    
    perplexity_results = []
    
    # Load evaluation data
    eval_data = None
    # Choose appropriate loading method based on file type
    if eval_dataset_path.endswith('.bin'):
        # Binary format (int32 tokens)
        eval_data = np.fromfile(eval_dataset_path, dtype=np.int32)
        eval_data = torch.from_numpy(eval_data).long()
    elif eval_dataset_path.endswith('.txt'):
        # Text format - tokenize with each model's tokenizer
        with open(eval_dataset_path, 'r', encoding='utf-8') as f:
            eval_text = f.read()
    else:
        print(f"Unsupported evaluation dataset format: {eval_dataset_path}")
        return []
    
    # Run perplexity calculation for each model
    for model_name, (model, tokenizer) in models_dict.items():
        print(f"\nCalculating perplexity for {model_name}")
        
        if eval_data is None:
            # Tokenize text for this specific model
            if hasattr(tokenizer, 'encode'):
                input_ids = tokenizer.encode(eval_text, return_tensors='pt')
            else:
                input_ids = tokenizer(eval_text, return_tensors='pt')['input_ids']
        else:
            # Use pre-tokenized data
            # Create a batch dimension if needed
            if eval_data.dim() == 1:
                input_ids = eval_data.unsqueeze(0)
            else:
                input_ids = eval_data
        
        # Calculate perplexity
        result = calculate_perplexity(model, input_ids)
        
        # Add model name to result
        result['model'] = model_name
        perplexity_results.append(result)
        
        print(f"  Perplexity: {result['perplexity']:.2f}")
    
    return perplexity_results

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if transformers library is available
    if not TRANSFORMERS_AVAILABLE and any(m != 'tinymamba' for m in args.models_to_compare):
        print("Warning: transformers library is required to compare with transformer models")
        print("Only TinyMamba model will be benchmarked")
        args.models_to_compare = ['tinymamba']
    
    # Load models to benchmark
    models_dict = {}
    
    # Load TinyMamba if in compare list
    if 'tinymamba' in args.models_to_compare:
        if os.path.exists(args.tinymamba_checkpoint):
            tinymamba_model, tinymamba_config = load_tinymamba_model(args.tinymamba_checkpoint, args.device)
            tokenizer = TinyBPETokenizer()  # Initialize with default settings
            models_dict['tinymamba'] = (tinymamba_model, tokenizer)
        else:
            print(f"Warning: TinyMamba checkpoint not found at {args.tinymamba_checkpoint}")
            args.models_to_compare.remove('tinymamba')
    
    # Load transformer models for comparison
    for model_name in args.models_to_compare:
        if model_name == 'tinymamba':
            continue  # Already handled
            
        try:
            model, tokenizer = load_transformer_model(model_name, args.device)
            models_dict[model_name] = (model, tokenizer)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # Create custom-sized GPT-2 model matching TinyMamba dimensions if needed
    if 'tinymamba' in models_dict and 'gpt2-custom' in args.models_to_compare:
        try:
            tinymamba_cfg = models_dict['tinymamba'][1]
            custom_gpt2 = create_custom_gpt2(
                tinymamba_cfg.d_model,
                tinymamba_cfg.n_layer,
                tinymamba_cfg.vocab_size,
                args.device
            )
            # Use the standard GPT-2 tokenizer
            gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
            models_dict['gpt2-custom'] = (custom_gpt2, gpt2_tokenizer)
        except Exception as e:
            print(f"Failed to create custom GPT-2 model: {e}")
    
    if not models_dict:
        print("No models available for benchmarking. Exiting.")
        return
    
    # Collect results for each benchmark type
    results = defaultdict(list)
    
    # Run speed benchmark
    if args.benchmark_type in ['speed', 'all']:
        print("\n==== Running Inference Speed Benchmark ====")
        speed_results = run_speed_benchmark(
            models_dict, 
            args.batch_sizes, 
            args.sequence_lengths, 
            args
        )
        results['speed'] = speed_results
    
    # Run memory benchmark
    if args.benchmark_type in ['memory', 'all'] and args.device == 'cuda':
        print("\n==== Running Memory Usage Benchmark ====")
        memory_results = run_memory_benchmark(
            models_dict, 
            args.batch_sizes, 
            args.sequence_lengths, 
            args
        )
        results['memory'] = memory_results
    
    # Run perplexity benchmark
    if args.benchmark_type in ['perplexity', 'all'] and args.eval_dataset:
        print("\n==== Running Perplexity Benchmark ====")
        perplexity_results = run_perplexity_benchmark(
            models_dict, 
            args.eval_dataset, 
            args
        )
        results['perplexity'] = perplexity_results
    
    # Generate plots if requested
    if args.generate_plots:
        print("\nGenerating comparison plots...")
        plot_dir = os.path.join(args.output_dir, 'plots')
        generate_comparison_plots(results, plot_dir)
    
    # Save benchmark results to JSON
    print("\nSaving benchmark results...")
    results_path = os.path.join(args.output_dir, 'benchmark_results.json')
    # Convert results to serializable format
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = [
            {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray, np.float32, np.float64)) else v 
             for k, v in item.items()}
            for item in value
        ]
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Benchmark complete! Results saved to {args.output_dir}")
    
    # Print summary of results
    print("\n==== Benchmark Summary ====")
    
    if 'speed' in results:
        print("\nInference Speed (tokens per second, higher is better):")
        for model_name in sorted(set(r['model'] for r in results['speed'])):
            model_results = [r for r in results['speed'] if r['model'] == model_name]
            avg_speed = sum(r['tokens_per_second'] for r in model_results) / len(model_results)
            print(f"  {model_name}: {avg_speed:.2f}")
    
    if 'perplexity' in results:
        print("\nPerplexity (lower is better):")
        for result in sorted(results['perplexity'], key=lambda x: x['perplexity']):
            print(f"  {result['model']}: {result['perplexity']:.2f}")

if __name__ == "__main__":
    main() 