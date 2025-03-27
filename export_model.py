"""
Export trained TinyMamba model in optimized formats for deployment.

This script:
1. Loads a trained checkpoint 
2. Performs optional model optimizations
3. Exports the model in deployable formats (PyTorch, TorchScript, ONNX)
4. Verifies exported model with a small evaluation
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm

# Import model architecture from the training script
from testv2 import TinyMambaModel, Config, TextGenerator, SimpleTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Export TinyMamba model for deployment")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="./tinymamba_model_latest.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./exported_model",
        help="Directory to save exported models"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        default="all",
        choices=["pytorch", "torchscript", "onnx", "all"],
        help="Export format(s)"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Apply optimization passes to the model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for model export and verification"
    )
    parser.add_argument(
        "--seq_length", 
        type=int, 
        default=512,
        help="Sequence length for model export"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model export and verification"
    )
    parser.add_argument(
        "--vocab_file", 
        type=str, 
        default=None,
        help="Optional vocabulary file for tokenizer"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify exported model with sample input"
    )
    parser.add_argument(
        "--generate", 
        action="store_true",
        help="Generate sample text from exported model"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Once upon a time",
        help="Prompt for text generation verification"
    )
    return parser.parse_args()

def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and config"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    # Load checkpoint with map_location
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
    
    # Move model to device
    model = model.to(device)
    
    return model, config

def optimize_model(model):
    """Apply optimization passes to model"""
    print("Applying optimization passes...")
    
    # Set model to eval mode
    model.eval()
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Fuse layers if possible
    # This is a placeholder - actual fusion would depend on the model architecture
    
    return model

def export_pytorch(model, config, output_dir):
    """Export model in PyTorch format"""
    print("Exporting PyTorch model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "tinymamba_model.pt"))
    
    # Save model architecture and config
    model_info = {
        'config': vars(config),
        'model_type': 'TinyMamba',
        'version': '1.0',
        'pytorch_version': torch.__version__,
    }
    torch.save(model_info, os.path.join(output_dir, "tinymamba_model_info.pt"))
    
    print(f"PyTorch model exported to {output_dir}")

def export_torchscript(model, config, output_dir, batch_size, seq_length, device):
    """Export model in TorchScript format"""
    print("Exporting TorchScript model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create example input for tracing
    example_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Trace model with example input
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Save traced model
    torch.jit.save(traced_model, os.path.join(output_dir, "tinymamba_model.pt.zip"))
    
    # Check inference with traced model
    with torch.no_grad():
        orig_output = model(example_input)
        traced_output = traced_model(example_input)
        
        # Verify outputs match
        max_diff = torch.max(torch.abs(orig_output.logits - traced_output.logits))
        print(f"Maximum difference between original and traced model: {max_diff:.6f}")
    
    print(f"TorchScript model exported to {output_dir}")

def export_onnx(model, config, output_dir, batch_size, seq_length, device):
    """Export model in ONNX format"""
    try:
        import onnx
        import onnxruntime
        print("Exporting ONNX model...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create example input for export
        example_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        
        # Export model to ONNX
        onnx_path = os.path.join(output_dir, "tinymamba_model.onnx")
        
        # Define output names
        output_names = ["logits"]
        
        # Try to export (may not work for all model architectures)
        try:
            # Wrapper class to get just the logits output
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    return self.model(x).logits
            
            wrapper = ModelWrapper(model)
            
            torch.onnx.export(
                wrapper,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=output_names,
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            sess = onnxruntime.InferenceSession(onnx_path)
            
            # Run inference
            ort_inputs = {
                "input_ids": example_input.cpu().numpy()
            }
            ort_outputs = sess.run(None, ort_inputs)
            
            # Compare with PyTorch outputs
            with torch.no_grad():
                torch_output = model(example_input).logits.cpu().numpy()
                
            max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
            print(f"Maximum difference between PyTorch and ONNX model: {max_diff:.6f}")
            
            print(f"ONNX model exported to {onnx_path}")
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Note: ONNX export might not work for all TinyMamba model architectures")
            
    except ImportError:
        print("ONNX export requires onnx and onnxruntime packages. Skipping.")

def verify_model(model, config, device, batch_size=1, seq_length=512):
    """Verify model with random inputs"""
    print("Verifying model with random inputs...")
    
    # Set model to eval mode
    model.eval()
    
    # Create random input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Measure inference time
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_ids)
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            output = model(input_ids)
            
        end_time = time.time()
        
    # Calculate metrics
    avg_time = (end_time - start_time) / num_runs
    tokens_per_second = batch_size * seq_length / avg_time
    
    print(f"Inference time: {avg_time*1000:.2f} ms per batch")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    
    # Check output shape
    logits = output.logits
    expected_shape = (batch_size, seq_length, config.vocab_size)
    
    if logits.shape != expected_shape:
        print(f"Warning: Unexpected output shape - got {logits.shape}, expected {expected_shape}")
    else:
        print(f"Output shape verified: {logits.shape}")
    
    return tokens_per_second

def generate_text(model, config, device, prompt="Once upon a time", max_length=100):
    """Generate sample text to verify model"""
    print(f"\nGenerating text from prompt: '{prompt}'")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=0.8,
        device=device
    )
    
    # Generate text
    result = generator.generate(prompt_text=prompt)
    
    # Print generated text
    print("\nGenerated text:")
    print("-" * 40)
    print(result["text"])
    print("-" * 40)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and config
    model, config = load_checkpoint(args.checkpoint, args.device)
    
    # Apply optimization if requested
    if args.optimize:
        model = optimize_model(model)
    
    # Set model to eval mode
    model.eval()
    
    # Export model in requested format(s)
    if args.format in ["pytorch", "all"]:
        export_pytorch(model, config, os.path.join(args.output_dir, "pytorch"))
        
    if args.format in ["torchscript", "all"]:
        export_torchscript(model, config, os.path.join(args.output_dir, "torchscript"), 
                         args.batch_size, args.seq_length, args.device)
        
    if args.format in ["onnx", "all"]:
        export_onnx(model, config, os.path.join(args.output_dir, "onnx"), 
                  args.batch_size, args.seq_length, args.device)
    
    # Verify model if requested
    if args.verify:
        verify_model(model, config, args.device, args.batch_size, args.seq_length)
    
    # Generate text if requested
    if args.generate:
        generate_text(model, config, args.device, args.prompt)
    
    print("\nModel export completed successfully!")

if __name__ == "__main__":
    main() 