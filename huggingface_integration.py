"""
HuggingFace Integration for TinyMamba

This module provides integration with the HuggingFace transformers library, enabling:
1. Loading and saving TinyMamba models with the HuggingFace model hub
2. Using TinyMamba within the transformers ecosystem
3. Converting between TinyMamba and HuggingFace compatible formats
"""

import os
import torch
import json
import copy
from typing import Optional, Tuple, Union, Dict, Any, List

# Import TinyMamba model and tokenizer
from testv2 import TinyMambaModel, Config, TextGenerator, TinyBPETokenizer

# Import transformers if available
try:
    import transformers
    from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers")

# Basic configurations
MODEL_CONFIG_NAME = "config.json"
MODEL_WEIGHTS_NAME = "pytorch_model.bin"

class TinyMambaConfig(PretrainedConfig if HF_AVAILABLE else object):
    """
    Configuration class for TinyMamba model.
    Inherits from PretrainedConfig to be compatible with HuggingFace transformers.
    """
    model_type = "tinymamba"
    
    def __init__(
        self,
        d_model=256,
        n_layer=4,
        vocab_size=50304,
        dropout=0.1,
        bias=False,
        activation='silu',
        d_state=16,
        d_conv=4,
        expand_factor=2,
        window_size=64,
        num_heads=4,
        block_size=128,
        param_method='butterfly',
        use_flash_attn=True,
        use_compile=False,
        **kwargs,
    ):
        if not HF_AVAILABLE:
            super().__init__()
        else:
            super().__init__(**kwargs)
            
        # Model architecture parameters
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        
        # SSM parameters
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        
        # Attention parameters
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Sequence parameters
        self.block_size = block_size
        
        # Implementation details
        self.param_method = param_method
        self.use_flash_attn = use_flash_attn
        self.use_compile = use_compile
    
    @classmethod
    def from_tinymamba_config(cls, config):
        """Convert standard TinyMamba Config to HuggingFace compatible config"""
        hf_config = cls(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            bias=config.bias,
            activation=config.activation,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand_factor=config.expand_factor,
            window_size=config.window_size,
            num_heads=config.num_heads,
            block_size=config.block_size,
            use_flash_attn=config.use_flash_attn,
            use_compile=config.use_compile,
        )
        return hf_config
    
    def to_tinymamba_config(self):
        """Convert HuggingFace config back to standard TinyMamba Config"""
        config = Config()
        
        # Copy parameters
        for key, value in self.__dict__.items():
            if key in [
                "d_model", "n_layer", "vocab_size", "dropout", "bias", 
                "activation", "d_state", "d_conv", "expand_factor",
                "window_size", "num_heads", "block_size", 
                "use_flash_attn", "use_compile",
            ]:
                setattr(config, key, value)
        
        return config

class TinyMambaHFModel(PreTrainedModel if HF_AVAILABLE else object):
    """
    TinyMamba model wrapped to be compatible with HuggingFace transformers.
    """
    config_class = TinyMambaConfig
    base_model_prefix = "tinymamba"
    
    def __init__(self, config):
        if not HF_AVAILABLE:
            super().__init__()
            self.config = config
        else:
            super().__init__(config)
            
        # Convert HF config to TinyMamba config
        if isinstance(config, TinyMambaConfig):
            tinymamba_config = config.to_tinymamba_config()
        else:
            tinymamba_config = Config()
            # Copy parameters
            for key, value in config.__dict__.items():
                if hasattr(tinymamba_config, key):
                    setattr(tinymamba_config, key, value)
        
        # Create TinyMamba model
        self.tinymamba = TinyMambaModel(tinymamba_config)
        self.config = config  # Store HF config
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward pass compatible with HuggingFace interface
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call the underlying TinyMamba model
        output = self.tinymamba(input_ids)
        
        if HF_AVAILABLE:
            if labels is not None:
                # Calculate loss if labels are provided
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                loss = None
                
            # Format output according to HuggingFace interface
            if return_dict:
                return CausalLMOutputWithCrossAttentions(
                    loss=loss,
                    logits=output,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                )
            else:
                return (loss, output) if loss is not None else output
        else:
            # Simple output for non-HF case
            return output
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model to a directory for HuggingFace compatibility
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_dict = self.config.to_dict()
        with open(os.path.join(save_directory, MODEL_CONFIG_NAME), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model weights
        torch.save(self.tinymamba.state_dict(), os.path.join(save_directory, MODEL_WEIGHTS_NAME))
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a model from HuggingFace style saved directory or hub
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required to load models from pretrained")
        
        # Load config
        config = TinyMambaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Create model with config
        model = cls(config)
        
        # Load weights
        weights_path = os.path.join(pretrained_model_name_or_path, MODEL_WEIGHTS_NAME)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            model.tinymamba.load_state_dict(state_dict)
        
        return model
    
    @classmethod
    def from_tinymamba_checkpoint(cls, checkpoint_path, **kwargs):
        """
        Create HF model from a standard TinyMamba checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            tinymamba_config = Config()
            
            # Update config with values from checkpoint
            for key, value in config_dict.items():
                if hasattr(tinymamba_config, key):
                    setattr(tinymamba_config, key, value)
        else:
            tinymamba_config = Config()
        
        # Convert to HF config
        hf_config = TinyMambaConfig.from_tinymamba_config(tinymamba_config)
        
        # Create model
        model = cls(hf_config)
        
        # Load weights
        if 'model' in checkpoint:
            model.tinymamba.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.tinymamba.load_state_dict(checkpoint['model_state_dict'])
        
        return model


class TinyMambaHFTokenizer(PreTrainedTokenizer if HF_AVAILABLE else object):
    """
    Tokenizer for TinyMamba compatible with HuggingFace
    """
    vocab_files_names = {"vocab_file": "vocab.txt", "merges_file": "merges.txt"}
    
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tiktoken_model=None,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        **kwargs
    ):
        if not HF_AVAILABLE:
            super().__init__()
        else:
            super().__init__(
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs
            )
        
        # Create the underlying tokenizer
        self.tokenizer = TinyBPETokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tiktoken_model=tiktoken_model,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token
        )
    
    @property
    def vocab_size(self):
        return len(self.tokenizer)
    
    def get_vocab(self):
        return self.tokenizer.vocab if hasattr(self.tokenizer, 'vocab') else {}
    
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        """
        Tokenize text
        """
        return self.tokenizer.encode(
            text, 
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Convert token IDs back to text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save tokenizer files to directory
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # If tokenizer has vocab and merges as attributes, save them
        if hasattr(self.tokenizer, 'vocab') and self.tokenizer.vocab:
            vocab_file = os.path.join(save_directory, self.vocab_files_names['vocab_file'])
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for token in self.tokenizer.vocab:
                    f.write(f"{token}\n")
        
        if hasattr(self.tokenizer, 'merges') and self.tokenizer.merges:
            merges_file = os.path.join(save_directory, self.vocab_files_names['merges_file'])
            with open(merges_file, 'w', encoding='utf-8') as f:
                for merge in self.tokenizer.merges:
                    f.write(f"{merge[0]} {merge[1]}\n")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a tokenizer from pretrained files
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required to load tokenizers from pretrained")
        
        # Vocabulary files
        vocab_file = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names['vocab_file'])
        merges_file = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names['merges_file'])
        
        # Check if files exist
        vocab_file = vocab_file if os.path.exists(vocab_file) else None
        merges_file = merges_file if os.path.exists(merges_file) else None
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_file=vocab_file,
            merges_file=merges_file,
            **kwargs
        )
        
        return tokenizer


# Function to convert TinyMamba checkpoint to HuggingFace format
def convert_tinymamba_checkpoint_to_hf(
    checkpoint_path, 
    output_dir, 
    tokenizer_file=None,
):
    """
    Convert a TinyMamba checkpoint to HuggingFace format
    
    Args:
        checkpoint_path: Path to TinyMamba checkpoint
        output_dir: Directory to save HuggingFace model
        tokenizer_file: Optional path to tokenizer vocabulary file
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library is required for this conversion")
    
    # Load TinyMamba model from checkpoint
    model = TinyMambaHFModel.from_tinymamba_checkpoint(checkpoint_path)
    
    # Save model in HuggingFace format
    model.save_pretrained(output_dir)
    
    # Also save tokenizer if provided
    if tokenizer_file:
        tokenizer = TinyMambaHFTokenizer(vocab_file=tokenizer_file)
        tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")


# Function to push model to HuggingFace Hub
def push_to_hf_hub(
    model_dir,
    repo_id, 
    commit_message="Upload TinyMamba model",
    private=False,
    token=None,
):
    """
    Push a converted TinyMamba model to HuggingFace Model Hub
    
    Args:
        model_dir: Directory with the converted model
        repo_id: HuggingFace repo ID (e.g., username/model-name)
        commit_message: Commit message
        private: Whether to create a private repo
        token: HuggingFace API token (or use HUGGINGFACE_TOKEN env var)
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library is required for Hub operations")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        
        # Upload files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            commit_message=commit_message
        )
        
        print(f"Model successfully pushed to https://huggingface.co/{repo_id}")
    except ImportError:
        print("huggingface_hub library is required. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"Error pushing to hub: {e}")


# Register model with AutoModel if transformers is available
if HF_AVAILABLE:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        
        # Register TinyMamba with Auto classes
        AutoConfig.register("tinymamba", TinyMambaConfig)
        AutoModelForCausalLM.register(TinyMambaConfig, TinyMambaHFModel)
        
        print("TinyMamba successfully registered with HuggingFace Auto classes")
    except Exception as e:
        print(f"Failed to register with Auto classes: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyMamba HuggingFace Integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert checkpoint to HF format
    convert_parser = subparsers.add_parser("convert", help="Convert TinyMamba checkpoint to HF format")
    convert_parser.add_argument("--checkpoint", required=True, help="Path to TinyMamba checkpoint")
    convert_parser.add_argument("--output_dir", required=True, help="Output directory for HF model")
    convert_parser.add_argument("--tokenizer_file", help="Path to tokenizer vocabulary file")
    
    # Push to HuggingFace Hub
    push_parser = subparsers.add_parser("push", help="Push model to HuggingFace Hub")
    push_parser.add_argument("--model_dir", required=True, help="Directory with the converted model")
    push_parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID (username/model-name)")
    push_parser.add_argument("--commit_message", default="Upload TinyMamba model", help="Commit message")
    push_parser.add_argument("--private", action="store_true", help="Create a private repo")
    push_parser.add_argument("--token", help="HuggingFace API token")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        convert_tinymamba_checkpoint_to_hf(
            args.checkpoint,
            args.output_dir,
            args.tokenizer_file
        )
    elif args.command == "push":
        push_to_hf_hub(
            args.model_dir,
            args.repo_id,
            args.commit_message,
            args.private,
            args.token
        )
    else:
        parser.print_help() 