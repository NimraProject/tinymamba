"""
Parameter-Efficient Fine-Tuning (PEFT) for TinyMamba

This module implements:
1. LoRA (Low-Rank Adaptation) for efficient fine-tuning
2. Prefix-tuning for parameter-efficient adapters
3. Utility functions for freezing and unfreezing model components
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple
import copy
import math

# Import TinyMamba model components
from testv2 import (
    TinyMambaModel, 
    Config, 
    TinyMambaBlock,
    MambaSSM,
    AdaptiveLocalAttention,
    MLP,
    RotaryEmbedding,
)

class LoRALinear(nn.Module):
    """
    Implementation of LoRA (Low-Rank Adaptation) for linear layers.
    https://arxiv.org/abs/2106.09685
    
    Args:
        original_layer: The original linear layer to be adapted
        r: Rank of the low-rank adaptation matrices
        alpha: Scaling factor
        dropout: Dropout probability for LoRA layers
        merge_weights: Whether to merge weights during inference for faster computation
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = True
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merge_weights = merge_weights
        
        # Dimensions
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices (A: down projection, B: up projection)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        # Keep track of whether weights are merged
        self.merged = False
    
    def _init_weights(self):
        """Initialize LoRA weights for stable training"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        if self.merged:
            # If weights are merged, just use the original layer
            return self.original_layer(x)
        
        # Original output
        orig_output = self.original_layer(x)
        
        # LoRA path: x → A → dropout → B → scaled output
        lora_output = self.dropout(x @ self.lora_A.T) @ self.lora_B.T
        
        # Combined output
        return orig_output + (lora_output * self.scaling)
    
    def merge_weights(self):
        """Merge LoRA weights with the original weights for faster inference"""
        if self.merged:
            return
        
        # Ensure original layer is on the same device as LoRA matrices
        device = self.lora_A.device
        self.original_layer.weight = nn.Parameter(
            self.original_layer.weight.to(device) + 
            (self.lora_B @ self.lora_A * self.scaling)
        )
        self.merged = True
    
    def unmerge_weights(self):
        """Restore original weights by removing the LoRA contribution"""
        if not self.merged:
            return
        
        device = self.lora_A.device
        self.original_layer.weight = nn.Parameter(
            self.original_layer.weight.to(device) - 
            (self.lora_B @ self.lora_A * self.scaling)
        )
        self.merged = False


class PrefixTuning(nn.Module):
    """
    Implementation of Prefix-Tuning for TinyMamba.
    https://arxiv.org/abs/2101.00190
    
    Adds learnable prefix tokens to the attention layers.
    
    Args:
        config: TinyMamba config
        prefix_length: Number of prefix tokens to add
        prefix_dim: Dimension of prefix tokens (default: same as model dimension)
        num_layers: Number of layers to add prefixes to (default: all layers)
        init_prefix_from_vocab: Initialize prefixes from vocabulary embeddings
    """
    def __init__(
        self,
        config: Config,
        prefix_length: int = 16,
        prefix_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        init_prefix_from_vocab: bool = False
    ):
        super().__init__()
        
        self.config = config
        self.prefix_length = prefix_length
        self.prefix_dim = prefix_dim if prefix_dim is not None else config.d_model
        self.num_layers = num_layers if num_layers is not None else config.n_layer
        
        # Create prefixes for each layer
        self.prefix_tokens = nn.Parameter(
            torch.zeros(self.num_layers, 2, prefix_length, self.prefix_dim)
        )
        
        # Optional projection if prefix_dim != d_model
        if self.prefix_dim != config.d_model:
            self.projection = nn.Linear(self.prefix_dim, config.d_model)
        else:
            self.projection = nn.Identity()
        
        # Initialize weights
        self._init_weights(init_from_vocab=init_prefix_from_vocab)
    
    def _init_weights(self, init_from_vocab: bool = False):
        """Initialize prefix tokens"""
        if init_from_vocab:
            # Initialize from vocabulary embeddings (assuming wte is the embedding layer)
            # This would require access to the embedding weights which we don't have directly
            # So we use a normal distribution instead
            nn.init.normal_(self.prefix_tokens, mean=0.0, std=0.02)
        else:
            # Initialize with small random values
            nn.init.normal_(self.prefix_tokens, mean=0.0, std=0.02)
    
    def forward(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return prefix key and value for a specific layer
        
        Args:
            layer_idx: Index of the layer requesting prefix
            
        Returns:
            Tuple of (prefix_key, prefix_value) tensors
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds number of prefixes {self.num_layers}")
        
        # Get this layer's prefix tokens
        prefix_key = self.prefix_tokens[layer_idx, 0]  # [prefix_length, prefix_dim]
        prefix_value = self.prefix_tokens[layer_idx, 1]  # [prefix_length, prefix_dim]
        
        # Project if necessary
        prefix_key = self.projection(prefix_key)  # [prefix_length, d_model]
        prefix_value = self.projection(prefix_value)  # [prefix_length, d_model]
        
        return prefix_key, prefix_value


class LoRAAdapter(nn.Module):
    """
    LoRA adapter for TinyMamba model.
    
    Applies LoRA to selected linear layers in the model.
    
    Args:
        model: TinyMamba model to adapt
        lora_config: Configuration for LoRA (rank, alpha, etc.)
        target_modules: List of module types or names to apply LoRA to
    """
    def __init__(
        self,
        model: TinyMambaModel,
        lora_config: Dict[str, Union[int, float, bool, List[str]]],
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.model = model
        self.lora_config = lora_config
        
        # Default target modules if none specified
        if target_modules is None:
            target_modules = [
                "out_proj",  # Attention output
                "qkv_proj",  # Attention QKV
                "D_proj",    # SSM D parameter
                "dt_proj",   # SSM dt parameter
                "wte",       # Token embeddings
            ]
        
        # Apply LoRA to target modules
        self._apply_lora_adapters(target_modules)
        
        # Save original parameters for unmerging
        self._save_original_params()
    
    def _apply_lora_adapters(self, target_modules: List[str]):
        """
        Apply LoRA adapters to target modules in the model
        
        Args:
            target_modules: List of module types or names to apply LoRA to
        """
        self.lora_layers = {}
        
        # Extract LoRA config parameters
        r = self.lora_config.get("r", 8)
        alpha = self.lora_config.get("alpha", 16)
        dropout = self.lora_config.get("dropout", 0.0)
        
        # Function to check if a module should be adapted
        def should_adapt(name, module):
            # Skip non-linear layers
            if not isinstance(module, nn.Linear):
                return False
            
            # Check if module name contains any target name
            return any(target in name for target in target_modules)
        
        # Recursive function to adapt modules
        def adapt_module(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if should_adapt(full_name, child):
                    # Replace with LoRA layer
                    lora_layer = LoRALinear(
                        original_layer=child,
                        r=r,
                        alpha=alpha,
                        dropout=dropout
                    )
                    
                    # Store reference to the original module
                    self.lora_layers[full_name] = child
                    
                    # Replace the module
                    if "." in name:
                        parent_name, child_name = name.rsplit(".", 1)
                        parent = module.get_submodule(parent_name)
                        setattr(parent, child_name, lora_layer)
                    else:
                        setattr(module, name, lora_layer)
                else:
                    # Recurse into child modules
                    adapt_module(child, full_name)
        
        # Apply adapters starting from the top level
        adapt_module(self.model)
    
    def _save_original_params(self):
        """Save original parameters for later restoration"""
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
    
    def merge_lora_weights(self):
        """Merge LoRA weights for faster inference"""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
    
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights to restore original weights"""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()
    
    def forward(self, *args, **kwargs):
        """Forward pass (delegates to wrapped model)"""
        return self.model(*args, **kwargs)


class PrefixTuningAdapter(nn.Module):
    """
    Prefix Tuning adapter for TinyMamba model.
    
    Adds learned prefix tokens to the attention layers.
    
    Args:
        model: TinyMamba model to adapt
        prefix_config: Configuration for prefix tuning
    """
    def __init__(
        self,
        model: TinyMambaModel,
        prefix_config: Dict[str, Union[int, bool]],
    ):
        super().__init__()
        
        self.model = model
        self.prefix_config = prefix_config
        
        # Create prefix tuning module
        self.prefix_tuning = PrefixTuning(
            config=model.config,
            prefix_length=prefix_config.get("prefix_length", 16),
            prefix_dim=prefix_config.get("prefix_dim", None),
            num_layers=prefix_config.get("num_layers", None),
            init_prefix_from_vocab=prefix_config.get("init_from_vocab", False)
        )
        
        # Register forward hooks for attention layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward pre-hooks for attention layers"""
        self.hooks = []
        
        # Iterate through blocks to find attention modules
        for i, block in enumerate(self.model.blocks):
            if hasattr(block, 'attn'):
                # Define hook for this attention layer
                def make_hook(layer_idx):
                    def attention_hook(module, inputs):
                        """Add prefix to attention inputs"""
                        prefix_k, prefix_v = self.prefix_tuning(layer_idx)
                        
                        # Assume single input tensor
                        x = inputs[0]
                        
                        # Process QKV normally
                        qkv = module.qkv_proj(x)
                        qkv = qkv.reshape(qkv.size(0), qkv.size(1), 3, module.num_heads, module.head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        
                        # Apply RoPE if the module has it
                        if hasattr(module, 'rotary_emb'):
                            seq_len = x.size(1)
                            cos, sin = module.rotary_emb(seq_len, x.device)
                            q, k = apply_rotary_pos_emb(q, k, cos, sin)
                        
                        # Add prefix to k and v
                        # Need to expand prefix to match batch size and heads
                        batch_size = x.size(0)
                        prefix_k_expanded = prefix_k.unsqueeze(0).unsqueeze(0)
                        prefix_v_expanded = prefix_v.unsqueeze(0).unsqueeze(0)
                        
                        prefix_k_expanded = prefix_k_expanded.expand(batch_size, module.num_heads, -1, -1)
                        prefix_v_expanded = prefix_v_expanded.expand(batch_size, module.num_heads, -1, -1)
                        
                        # Concatenate prefix with existing k and v
                        k_with_prefix = torch.cat([prefix_k_expanded, k], dim=2)
                        v_with_prefix = torch.cat([prefix_v_expanded, v], dim=2)
                        
                        # Return modified inputs
                        return (x, q, k_with_prefix, v_with_prefix)
                    
                    return attention_hook
                
                # Register hook
                hook = block.attn.register_forward_pre_hook(make_hook(i))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, *args, **kwargs):
        """Forward pass (delegates to wrapped model)"""
        return self.model(*args, **kwargs)


def freeze_model_parameters(model, exclude_patterns=None):
    """
    Freeze all parameters in the model except those matching exclude_patterns
    
    Args:
        model: The model to freeze
        exclude_patterns: List of parameter name patterns to exclude from freezing
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    for name, param in model.named_parameters():
        # Check if parameter name matches any exclude pattern
        if any(pattern in name for pattern in exclude_patterns):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_model_parameters(model, include_patterns=None):
    """
    Unfreeze parameters in the model matching include_patterns
    
    Args:
        model: The model to partially unfreeze
        include_patterns: List of parameter name patterns to unfreeze
    """
    if include_patterns is None:
        # Unfreeze nothing if no patterns specified
        return
    
    for name, param in model.named_parameters():
        # Check if parameter name matches any include pattern
        if any(pattern in name for pattern in include_patterns):
            param.requires_grad = True


def get_trainable_parameters(model):
    """
    Get trainable parameters count and percentage
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict containing parameter counts and percentages
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.0, target_modules=None):
    """
    Apply LoRA to a TinyMamba model
    
    Args:
        model: TinyMamba model to apply LoRA to
        rank: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout rate for LoRA
        target_modules: List of module types to apply LoRA to
        
    Returns:
        LoRAAdapter wrapping the model
    """
    lora_config = {
        "r": rank,
        "alpha": alpha,
        "dropout": dropout,
    }
    
    # Create LoRA adapter
    lora_model = LoRAAdapter(
        model=model,
        lora_config=lora_config,
        target_modules=target_modules
    )
    
    # Freeze the original model parameters
    freeze_model_parameters(lora_model, exclude_patterns=["lora_A", "lora_B"])
    
    return lora_model


def apply_prefix_tuning_to_model(model, prefix_length=16, prefix_dim=None):
    """
    Apply Prefix Tuning to a TinyMamba model
    
    Args:
        model: TinyMamba model to apply prefix tuning to
        prefix_length: Number of prefix tokens
        prefix_dim: Dimension of prefix tokens
        
    Returns:
        PrefixTuningAdapter wrapping the model
    """
    prefix_config = {
        "prefix_length": prefix_length,
        "prefix_dim": prefix_dim,
        "init_from_vocab": False,
    }
    
    # Create Prefix Tuning adapter
    prefix_model = PrefixTuningAdapter(
        model=model,
        prefix_config=prefix_config
    )
    
    # Freeze the original model parameters
    freeze_model_parameters(prefix_model, exclude_patterns=["prefix_tokens"])
    
    return prefix_model


if __name__ == "__main__":
    # Example usage
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Parameter-Efficient Fine-Tuning for TinyMamba")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--method", type=str, choices=["lora", "prefix"], default="lora", help="PEFT method")
    parser.add_argument("--output_dir", type=str, default="./peft_model", help="Output directory")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--prefix_length", type=int, default=16, help="Prefix length")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        print("Warning: Using default config")
        config = Config()
    
    # Create model
    model = TinyMambaModel(config)
    
    # Load weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Apply PEFT method
    if args.method == "lora":
        print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
        peft_model = apply_lora_to_model(
            model, 
            rank=args.lora_rank, 
            alpha=args.lora_alpha
        )
    else:  # prefix
        print(f"Applying Prefix Tuning with prefix_length={args.prefix_length}")
        peft_model = apply_prefix_tuning_to_model(
            model,
            prefix_length=args.prefix_length
        )
    
    # Print parameter stats
    param_stats = get_trainable_parameters(peft_model)
    print(f"Total parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_percent']:.2f}%)")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"tinymamba_{args.method}_adapter.pt")
    
    torch.save({
        "config": config.__dict__,
        "model_state_dict": peft_model.state_dict(),
        "peft_method": args.method,
        "peft_config": {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "prefix_length": args.prefix_length
        }
    }, output_path)
    
    print(f"Model saved to {output_path}") 