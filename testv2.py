import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import os
import numpy as np
import math
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time

# Note: Removed huggingface_hub imports and related code

# Add a global rank variable at the top of the file (after imports)
global_rank = 0  # Default to 0

# --- Configuration ---

class Config:
    # Model Parameters
    d_model = 256         # Model dimension (D)
    n_layer = 4           # Number of layers
    vocab_size = 50304    # Vocabulary size (GPT-2 Tiktoken)
    dropout = 0.1         # Dropout rate
    bias = False          # Use bias in Linear layers?
    activation = 'silu'   # Activation function to use

    # MambaSSM Parameters (Simplified)
    d_state = 16          # SSM state dimension (N) - Mamba uses higher values
    d_conv = 4            # Local convolution width
    expand_factor = 2     # Expansion factor for MLP and SSM block

    # Attention Parameters
    window_size = 64      # Local attention window size (use -1 for full attn if needed)
    num_heads = 4         # Number of attention heads (ensure d_model % num_heads == 0)

    # Training Parameters
    block_size = 128      # Context length
    batch_size = 8        # Batch size per GPU
    gradient_accumulation_steps = 4 # Accumulate gradients for effective batch size
    num_epochs = 1        # Number of training epochs
    lr = 6e-4             # Learning rate (adjusted from original)
    warmup_steps = 1000   # Warmup steps
    grad_clip = 1.0       # Gradient clipping value
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.95

    # Runtime Parameters
    use_compile = True    # Use torch.compile (requires PyTorch 2.0+)
    use_flash_attn = True # Try to use FlashAttention if available
    amp_dtype = torch.bfloat16 # Use bfloat16 if available, else float16
    log_interval = 20     # Log frequency
    empty_cache_freq = 50 # How often to empty CUDA cache (helps on low VRAM)

    # Data and Checkpointing
    train_data_path = './data/text/train.bin' # Path to tokenized training data
    val_data_path = './data/text/val.bin'   # Path to tokenized validation data
    checkpoint_path = 'tinymamba_model_latest.pt' # Path for saving/resuming
    resume = False         # Whether to resume training from checkpoint_path

    # Validation, early stopping, and generation settings
    enable_validation = True
    validation_interval = 2000  # Steps between validation runs
    validation_steps = 50       # Number of validation steps to run
    generate_samples = True     # Whether to generate text samples during training
    generation_interval = 1000  # Steps between generation runs
    generation_length = 100     # Length of generated text
    generation_temperature = 0.8 # Temperature for generation
    
    # Early stopping
    use_early_stopping = True
    early_stopping_patience = 5 # Number of validations with no improvement before stopping
    early_stopping_min_delta = 0.01 # Minimum change to be considered an improvement
    
    # Logging options
    use_wandb = False  # Enable Weights & Biases logging
    wandb_project = "tinymamba"
    wandb_run_name = None  # Auto-generated if None
    
    # Checkpointing
    checkpoint_interval = 2000  # Save a checkpoint every N steps
    save_top_k = 3  # Number of best checkpoints to keep

# Instantiate config
config = Config()
assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"

# --- DDP Setup ---
def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Use a free port
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

# --- Hardware & Precision Setup ---
def setup_hardware_precision(rank):
    """Sets up device, AMP dtype, and backend settings."""
    if not dist.is_initialized(): # Handle non-DDP case if needed later
         world_size = 1
    else:
         world_size = dist.get_world_size()

    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device) # Ensure correct device is set for this process
        if torch.cuda.is_bf16_supported():
            config.amp_dtype = torch.bfloat16
            if rank == 0: print("Using bfloat16 for mixed precision")
        else:
            config.amp_dtype = torch.float16
            if rank == 0: print("Using float16 for mixed precision")
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True # Can improve performance if input sizes are consistent
    else:
        device = torch.device("cpu")
        config.amp_dtype = torch.float32 # AMP not used on CPU
        config.use_compile = False # torch.compile generally not beneficial on CPU
        config.use_flash_attn = False
        if rank == 0: print("Using CPU")

    return device

# --- Flash Attention Check ---
FLASH_ATTN_AVAILABLE = False
if config.use_flash_attn:
    try:
        from flash_attn import flash_attn_func
        # Check for sliding window support (available in flash-attn >= 2.0)
        # A simple check: try importing a v2 feature or check version if possible
        # For simplicity, we assume if import works, v2 features might be available
        FLASH_ATTN_AVAILABLE = True
        print("Flash Attention library found.")
        # Further check if sliding window is supported would be ideal
    except ImportError:
        print("Flash Attention library not found, falling back to PyTorch SDPA.")
        config.use_flash_attn = False # Force fallback if import fails

# --- Activation Function ---
def get_activation(name):
    if name == 'silu':
        return nn.SiLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

# --- Low-rank projection for SSM parameters ---
class LowRankProjection(nn.Module):
    def __init__(self, d_in, d_out, rank=4, bias=False):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.w2 = nn.Parameter(torch.randn(rank, d_out) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        
    def forward(self, x=None):
        # Either project input x, or just return the composed matrix
        if x is not None:
            return x @ (self.w1 @ self.w2) + (self.bias if self.bias is not None else 0)
        else:
            return self.w1 @ self.w2

# --- Updated Mamba-style SSM ---
# Based on Mamba block structure but simplified and iterative

# Add a JIT-compiled function for faster SSM recursion
@torch.jit.script
def efficient_ssm_scan(A_bar: torch.Tensor, dB: torch.Tensor, x_conv: torch.Tensor, C: torch.Tensor):
    """
    Efficient JIT-compiled function for the SSM scan operation
    Args:
        A_bar: (B, L, d_inner, d_state) - discretized state matrix
        dB: (B, L, d_inner, d_state) - discretized input matrix
        x_conv: (B, L, d_inner) - input sequence after convolution
        C: (d_inner, d_state) - output matrix
    Returns:
        y: (B, L, d_inner) - output sequence
    """
    B, L, d_inner = x_conv.shape
    d_state = A_bar.shape[-1]
    device = A_bar.device
    
    # Initialize state and output
    h = torch.zeros(B, d_inner, d_state, device=device)
    ys = torch.zeros(B, L, d_inner, device=device)
    
    # Perform scan operation
    for t in range(L):
        # h_t = A_t * h_{t-1} + B_t * x_t
        h = torch.bmm(A_bar[:, t].view(B, d_inner, d_state), h) + \
            dB[:, t] * x_conv[:, t].unsqueeze(-1)
        
        # y_t = C * h_t
        y_t = torch.einsum('b d n, d n -> b d', h, C)
        ys[:, t] = y_t
    
    return ys

class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, activation='silu', bias=False, param_method='butterfly'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Input projection with DeepNet scaling
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        # DeepNet scaling: smaller init for residual branches
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Convolution branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = get_activation(activation)
        self.act = self.activation
        
        # Add dropouts for regularization
        self.conv_dropout = nn.Dropout(0.1)  # Dropout after convolution
        self.state_dropout = nn.Dropout(0.1)  # Dropout for hidden states

        # SSM parameters with improved efficiency
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        
        # Choose parameter method based on configuration
        if param_method == 'butterfly':
            self.B_proj = ButterflyProjection(self.d_inner, self.d_state)
            self.C_proj = ButterflyProjection(self.d_inner, self.d_state)
            self.D_proj = ButterflyProjection(self.d_inner, 1)
        elif param_method == 'toeplitz':
            self.B_proj = ToeplitzProjection(self.d_inner, self.d_state)
            self.C_proj = ToeplitzProjection(self.d_inner, self.d_state)
            self.D_proj = ToeplitzProjection(self.d_inner, 1)
        else:  # Default to low-rank
            self.B_proj = LowRankProjection(self.d_inner, self.d_state, rank=4, bias=bias)
            self.C_proj = LowRankProjection(self.d_inner, self.d_state, rank=4, bias=bias)
            self.D_proj = LowRankProjection(self.d_inner, 1, rank=4, bias=bias)

        # Time step parameter
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.6)  # Initialize for stability

        # Output linear projection with DeepNet scaling
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # For streaming inference - persistent state
        self.last_state = None
        self.inference_mode = False

    def toggle_streaming(self, enabled=True):
        """Enable or disable streaming inference mode"""
        self.inference_mode = enabled
        if not enabled:
            self.last_state = None
    
    def reset_state(self):
        """Reset the hidden state for streaming inference"""
        self.last_state = None
    
    def forward(self, x, state=None):
        B, L, _ = x.shape
        d_inner = self.d_inner

        # --- Input Projections ---
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # --- Convolutional Branch with dropout ---
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)
        x_conv = self.conv_dropout(x_conv)  # Apply dropout after activation

        # --- SSM Parameters ---
        dt = F.softplus(self.dt_proj(x_in))
        A = -torch.exp(self.A_log)
        B = self.B_proj()
        C = self.C_proj()
        D = self.D_proj().squeeze(-1)
        
        # Discretized system matrices
        dA = torch.einsum('b l d, d n -> b l d n', dt, A)
        A_bar = torch.exp(dA)
        dB = torch.einsum('b l d, d n -> b l d n', dt, B)

        # Use persistent state if in streaming mode or if state is provided
        if self.inference_mode or state is not None:
            h = state if state is not None else self.last_state
            if h is None:
                h = torch.zeros(B, d_inner, self.d_state, device=x.device)
            
            # Process sequence and update state
            ys = []
            for t in range(L):
                # Update state: h_t = A_t * h_{t-1} + B_t * x_t
                h = torch.bmm(A_bar[:, t].view(B, d_inner, 1), h.view(B, d_inner, self.d_state)) + \
                    dB[:, t] * x_conv[:, t].unsqueeze(-1)
                
                # Apply state dropout if in training mode (not during inference)
                if self.training and not self.inference_mode:
                    h = self.state_dropout(h)  # Apply dropout to hidden states
                
                # Output calculation: y_t = C * h_t
                y_t = torch.einsum('b d n, d n -> b d', h, C)
                ys.append(y_t)
            
            # Save the latest state if in streaming mode
            if self.inference_mode:
                self.last_state = h.detach()
            
            y = torch.stack(ys, dim=1)
            
        else:
            # Use TorchScript-compiled function for faster processing during training
            # when we don't need to keep track of intermediate states
            if L > 64:  # Only use for longer sequences where the overhead is worth it
                try:
                    # Apply state dropout during training
                    if self.training:
                        # Add dropout logic to the scan function
                        y = efficient_ssm_scan(A_bar, dB, x_conv, C)
                        # Apply dropout after scan
                        y = self.state_dropout(y.unsqueeze(-1)).squeeze(-1)
                    else:
                        y = efficient_ssm_scan(A_bar, dB, x_conv, C)
                except Exception as e:
                    print(f"Efficient scan failed: {e}. Falling back to loop implementation.")
                    # Fall back to the original implementation
                    h = torch.zeros(B, d_inner, self.d_state, device=x.device)
                    ys = []
                    
                    for t in range(L):
                        h = torch.bmm(A_bar[:, t].view(B, d_inner, 1), h.view(B, d_inner, self.d_state)) + \
                            dB[:, t] * x_conv[:, t].unsqueeze(-1)
                        
                        # Apply state dropout during training
                        if self.training:
                            h = self.state_dropout(h)
                            
                        y_t = torch.einsum('b d n, d n -> b d', h, C)
                        ys.append(y_t)
                    
                    y = torch.stack(ys, dim=1)
            else:
                # For shorter sequences, the loop is more efficient
                h = torch.zeros(B, d_inner, self.d_state, device=x.device)
                ys = []
                
                for t in range(L):
                    h = torch.bmm(A_bar[:, t].view(B, d_inner, 1), h.view(B, d_inner, self.d_state)) + \
                        dB[:, t] * x_conv[:, t].unsqueeze(-1)
                    
                    # Apply state dropout during training
                    if self.training:
                        h = self.state_dropout(h)
                        
                    y_t = torch.einsum('b d n, d n -> b d', h, C)
                    ys.append(y_t)
                
                y = torch.stack(ys, dim=1)
        
        # Add skip connection and gating
        y = y + x_in * D.unsqueeze(0).unsqueeze(0)
        y = y * self.activation(z)
        
        # Output projection
        y = self.out_proj(y)
        return y

# --- Enhanced AdaptiveLocalAttention with position-based window scaling ---
class AdaptiveLocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, base_window_size=64, max_window_size=256, 
                 dropout=0.1, bias=False, special_token_ids=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_window_size = base_window_size
        self.max_window_size = max_window_size
        self.dropout = dropout
        
        # Special tokens that trigger full attention
        self.special_token_ids = special_token_ids or []
        
        # DeepNet scaling for residual branches
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Add RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Check Flash Attention availability
        self.use_flash_attn = False
        if config.use_flash_attn and FLASH_ATTN_AVAILABLE:
            self.use_flash_attn = True
            # Define default flash window size for initialization
            self.flash_window_size = (self.base_window_size - 1, 0) if self.base_window_size > 0 else None
            if global_rank == 0:
                print(f"AdaptiveLocalAttention: Using Flash Attention with adaptive window sizing")
        else:
            if global_rank == 0:
                print(f"AdaptiveLocalAttention: Using PyTorch SDPA with adaptive window sizing")

    def _calculate_window_sizes(self, seq_len):
        """Calculate position-dependent window sizes using Longformer-style pattern"""
        # Start with base window size for all positions
        window_sizes = torch.full((seq_len,), self.base_window_size, dtype=torch.long)
        
        # Apply stepped increases at logarithmically spaced positions
        if seq_len > 16:
            # Define logarithmically spaced breakpoints
            log_positions = [
                int(seq_len * 0.25),  # 25% into sequence
                int(seq_len * 0.5),   # 50% into sequence
                int(seq_len * 0.75)   # 75% into sequence
            ]
            
            # Apply progressively larger windows at these breakpoints
            step_size = (self.max_window_size - self.base_window_size) / 3
            
            window_sizes[log_positions[0]:] = min(self.base_window_size + step_size, self.max_window_size)
            window_sizes[log_positions[1]:] = min(self.base_window_size + 2 * step_size, self.max_window_size)
            window_sizes[log_positions[2]:] = self.max_window_size
            
        return window_sizes

    def forward(self, x, token_ids=None):
        B, L, D = x.shape
        
        # Generate position-dependent window sizes
        position_windows = self._calculate_window_sizes(L).to(x.device)
        
        # Apply token-aware attention for special tokens if token_ids are provided
        if token_ids is not None and len(self.special_token_ids) > 0:
            # Create a mask for special tokens
            special_tokens_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            for special_id in self.special_token_ids:
                special_tokens_mask = special_tokens_mask | (token_ids == special_id)
                
            # Set larger window size for positions with special tokens
            # and positions after special tokens (to allow better context flow)
            if special_tokens_mask.any():
                for b in range(B):
                    special_positions = torch.where(special_tokens_mask[b])[0]
                    if len(special_positions) > 0:
                        # Use max window size for special tokens
                        position_windows[special_positions] = self.max_window_size
                        
                        # Also increase window for tokens following special tokens
                        for pos in special_positions:
                            # Increase window size for the next few tokens
                            next_tokens = min(pos + 5, L)
                            position_windows[pos:next_tokens] = self.max_window_size
        
        # Process query, key, value projections
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        cos, sin = self.rotary_emb(L, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use Flash Attention if available
        if self.use_flash_attn:
            q_fa = q.transpose(1, 2)  # (B, L, H, D_head)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            
            # Use the maximum window size for flash attention
            # (Flash attention doesn't support per-position window sizes)
            max_window = position_windows.max().item()
            flash_window = (max_window, 0)  # (left context, right context)
            
            try:
                from flash_attn import flash_attn_func
                attn_output = flash_attn_func(
                    q_fa, k_fa, v_fa,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                    window_size=flash_window
                )
                attn_output = attn_output.transpose(1, 2)
            except Exception as e:
                print(f"Flash attention failed: {e}. Falling back to SDPA.")
                # Fall back to SDPA implementation below
                self.use_flash_attn = False
                
        if not self.use_flash_attn:
            # Custom attention mask based on position-dependent window sizes
            causal_mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril(diagonal=0)
            
            # Create position-specific masks - each position i can attend to positions (i-win_size[i]) to i
            attn_mask = torch.zeros((L, L), device=x.device, dtype=torch.bool)
            
            # For each position, create appropriate window based on its window size
            for i in range(L):
                win_size = position_windows[i].item()
                # Allow attention to tokens from (i-win_size) to i (causal window)
                attn_mask[i, :max(0, i-win_size)] = True
            
            # Convert to float mask with -inf for masked positions
            attn_mask = attn_mask.to(torch.float32).masked_fill(attn_mask, float('-inf')).masked_fill(~attn_mask, 0.0)
            
            # Apply scaled dot product attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False  # We're using our custom mask
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(attn_output)
        return output

# --- MLP Block ---
class MLP(nn.Module):
    def __init__(self, d_model, expansion_factor=2, dropout=0.1, activation='silu', bias=False):
        super().__init__()
        d_ff = int(d_model * expansion_factor)
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.activation(gate) * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

# --- SwiGLU implementation ---
class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=2, dropout=0.1, bias=False):
        super().__init__()
        d_ff = int(d_model * expansion_factor)
        # SwiGLU uses two separate projections for the gate and the value paths
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate path
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # Value path
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # SwiGLU activation: SiLU(W1x) * W2x
        x1 = F.silu(self.w1(x))
        x2 = self.w2(x)
        x = x1 * x2
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

# --- RMSNorm Implementation ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normalized = x / rms
        # Apply learned scaling
        return self.weight * x_normalized

# --- Rotary Position Embeddings (RoPE) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        inv_freq = 1. / (max_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        # Shape: (seq_len, dim/2)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        # Shape: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Complex rotation in form of (cos, sin)
        cos_cached = torch.cos(emb)
        sin_cached = torch.sin(emb)
        return cos_cached, sin_cached

def apply_rotary_pos_emb(q, k, cos, sin):
    # Split values into even and odd for rotation
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Apply rotation using einsum (batched dot product)
    q_embed = torch.cat([
        q_even * cos - q_odd * sin,
        q_odd * cos + q_even * sin
    ], dim=-1)
    
    k_embed = torch.cat([
        k_even * cos - k_odd * sin,
        k_odd * cos + k_even * sin
    ], dim=-1)
    
    return q_embed, k_embed

# --- Dynamic Token-Adaptive Gating ---
class TokenAdaptiveGating(nn.Module):
    def __init__(self, d_model, num_branches=3, reduction_factor=8):
        super().__init__()
        """
        Dynamic gating mechanism that adapts branch weights based on token content
        
        Args:
            d_model: Hidden dimension size
            num_branches: Number of branches to gate (default 3: ssm, attn, mlp)
            reduction_factor: Dimension reduction for controller efficiency
        """
        self.num_branches = num_branches
        d_reduced = max(32, d_model // reduction_factor)
        
        # Lightweight controller network
        self.controller = nn.Sequential(
            nn.Linear(d_model, d_reduced),
            nn.SiLU(),
            nn.Linear(d_reduced, num_branches)
        )
        
        # Initialize to produce roughly equal weights at the start
        with torch.no_grad():
            # Initialize last layer to produce roughly uniform outputs
            nn.init.zeros_(self.controller[-1].weight)
            nn.init.zeros_(self.controller[-1].bias)
    
    def forward(self, x):
        """
        Compute branch weights for each token in the sequence
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            weights: Softmax weights for each branch [B, L, num_branches]
        """
        # Get token-wise branch weights
        branch_logits = self.controller(x)  # [B, L, num_branches]
        
        # Apply softmax for soft routing
        branch_weights = F.softmax(branch_logits, dim=-1)  # [B, L, num_branches]
        
        return branch_weights

# --- Updated TinyMamba Block with token-adaptive branch mixing ---
class TinyMambaBlock(nn.Module):
    def __init__(self, config, param_method='butterfly'):
        super().__init__()
        # Normalization layers
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.ln3 = RMSNorm(config.d_model)
        
        # Replace global branch gate with token-adaptive gating
        self.branch_controller = TokenAdaptiveGating(config.d_model, num_branches=3)
        
        # Add residual scaling parameters for stability (T5-style)
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # Components with improved parameter efficiency
        self.ssm = MambaSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
            activation=config.activation,
            bias=config.bias,
            param_method=param_method
        )

        # Replace LocalAttention with AdaptiveLocalAttention
        self.local_attn = AdaptiveLocalAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            base_window_size=config.window_size,
            max_window_size=min(config.window_size * 4, config.block_size),  # Cap at block_size
            dropout=config.dropout,
            bias=config.bias,
            special_token_ids=[0, 1, 2]  # Common special tokens: PAD, BOS, EOS
        )

        # Replace MLP with SwiGLU for better performance
        self.mlp = SwiGLU(
            d_model=config.d_model,
            expansion_factor=config.expand_factor,
            dropout=config.dropout,
            bias=config.bias
        )
        
    def toggle_streaming(self, enabled=True):
        """Enable or disable streaming mode for all components"""
        self.ssm.toggle_streaming(enabled)
        if hasattr(self.local_attn, 'toggle_streaming'):
            self.local_attn.toggle_streaming(enabled)
        
    def reset_state(self):
        """Reset all streaming states"""
        self.ssm.reset_state()
        if hasattr(self.local_attn, 'reset_cache'):
            self.local_attn.reset_cache()

    def forward(self, x, token_ids=None):
        # Get token-specific adaptive branch mixing weights [B, L, 3]
        branch_weights = self.branch_controller(x)
        
        # Compute outputs from each branch
        ssm_out = self.ssm(self.ln1(x))
        attn_out = self.local_attn(self.ln2(x), token_ids)
        mlp_out = self.mlp(self.ln3(x))
        
        # Stack branch outputs for efficient batched multiplication
        stacked_branch_outputs = torch.stack([ssm_out, attn_out, mlp_out], dim=-2)  # [B, L, 3, D]
        
        # Apply token-specific branch weights
        # branch_weights: [B, L, 3], stacked_outputs: [B, L, 3, D]
        weighted_sum = torch.sum(branch_weights.unsqueeze(-1) * stacked_branch_outputs, dim=-2)
        
        # Apply learned residual scaling for stability in deep networks
        return x + self.residual_scale * weighted_sum

# --- Full TinyMamba Model ---
# Add µParametrization initialization for better deep network training
def _init_weights_mu_param(module, fan_in_fan_out=False, scale=1.0):
    """Initialize weights using µParametrization for better training of deep networks"""
    if isinstance(module, nn.Linear):
        # µP initialization - scale by 1/sqrt(fan_in)
        fan_in, fan_out = module.in_features, module.out_features
        std = scale / math.sqrt(fan_in)
        if fan_in_fan_out:
            # For layers where weight is transposed during forward pass
            std = scale / math.sqrt(fan_out)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.weight is not None:
            nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv1d):
        # For convolutional layers, use Kaiming initialization
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')  # SiLU/Swish is similar to ReLU
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class TinyMambaModel(nn.Module):
    def __init__(self, config, param_method='butterfly'):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TinyMambaBlock(config, param_method=param_method) for _ in range(config.n_layer)
        ])

        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # Init weights with µParametrization
        self.apply(_init_weights_mu_param)
        
        # Apply deeper initialization to residual branches for stability
        for block in self.blocks:
            # Scale output projections of residual paths by 1/sqrt(2*n_layer)
            for name, p in block.named_parameters():
                if 'out_proj.weight' in name:
                    # Further scale down residual outputs for stability in deep networks
                    p.data.mul_(1.0 / math.sqrt(2.0 * config.n_layer))
                    
        # Special handling for SSM parameters that need different initialization
        for block in self.blocks:
            if hasattr(block.ssm, 'dt_proj'):
                # Initialize dt bias to proper value for stable updates
                nn.init.constant_(block.ssm.dt_proj.bias, -4.6)

    def toggle_streaming(self, enabled=True):
        """Enable or disable streaming mode for the entire model"""
        for block in self.blocks:
            if hasattr(block, 'toggle_streaming'):
                block.toggle_streaming(enabled)
            
    def reset_state(self):
        """Reset all streaming states in the model"""
        for block in self.blocks:
            if hasattr(block, 'reset_state'):
                block.reset_state()

    def forward(self, inputs):
        B, L = inputs.shape
        assert L <= self.config.block_size, f"Input length {L} exceeds block size {self.config.block_size}"

        x = self.embedding(inputs) # (B, L, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, inputs)  # Pass token IDs for adaptive attention

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, L, vocab_size)
        return logits

# --- Simplified Streaming Dataset ---
class StreamingTokenDataset(IterableDataset):
    """
    Efficient streaming dataset for tokenized data with multi-source support
    
    Features:
    - Streaming data from disk to avoid OOM
    - Support for multiple data sources (interleaving or sampling)
    - Resumable from specific position
    - Optional shuffling with configurable buffer
    - Memory-efficient processing
    """
    def __init__(
        self, 
        data_sources, 
        block_size=512, 
        weights=None,
        shuffle=False,
        shuffle_buffer_size=1000,
        resume_position=None,
        seed=42
    ):
        """
        Initialize streaming dataset
        
        Args:
            data_sources: List of file paths or single path to data files
            block_size: Context length for each sample
            weights: Optional weights for sampling from multiple sources
            shuffle: Whether to shuffle samples
            shuffle_buffer_size: Size of buffer to use for shuffling
            resume_position: Optional dict mapping source to position for resuming
            seed: Random seed for shuffling
        """
        super().__init__()
        self.block_size = block_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Handle single source vs multiple sources
        if isinstance(data_sources, str):
            data_sources = [data_sources]
        self.data_sources = data_sources
        
        # Prepare weights for sampling
        if weights is None:
            self.weights = [1.0 / len(data_sources)] * len(data_sources)
        else:
            assert len(weights) == len(data_sources), "Weights must match number of data sources"
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        # Initialize file handles (lazily opened)
        self.file_handles = {}
        self.file_sizes = {}
        self.positions = resume_position or {}
        
        # Precompute file sizes
        for src in self.data_sources:
            if os.path.exists(src):
                self.file_sizes[src] = os.path.getsize(src) // 4  # Assuming int32 tokens
            else:
                print(f"Warning: Data source {src} not found")
                self.file_sizes[src] = 0
                
        # Buffer for shuffling
        self.buffer = []
        
        # Calculate total tokens across all sources
        self.total_tokens = sum(self.file_sizes.values())
        print(f"StreamingDataset: {len(data_sources)} sources, {self.total_tokens:,} total tokens")
        
    def _open_file(self, src):
        """Lazily open file handle when needed"""
        if src not in self.file_handles or self.file_handles[src] is None:
            self.file_handles[src] = open(src, 'rb')
            # Seek to resume position if specified
            if src in self.positions and self.positions[src] > 0:
                self.file_handles[src].seek(self.positions[src] * 4)  # Assuming int32
                
        return self.file_handles[src]
        
    def _read_block(self, src, length=None):
        """Read a block of tokens from a source file"""
        # Default to block_size + 1 (for input and target)
        if length is None:
            length = self.block_size + 1
            
        try:
            # Get file handle
            f = self._open_file(src)
            
            # Read binary data
            binary_data = f.read(length * 4)  # Assuming int32
            
            # Check if we have enough data
            if len(binary_data) < length * 4:
                # Reached EOF, rewind and try again
                f.seek(0)
                binary_data = f.read(length * 4)
                
                # Still not enough? File might be smaller than block_size
                if len(binary_data) < length * 4:
                    return None
                    
            # Convert to tensor
            tokens = np.frombuffer(binary_data, dtype=np.int32)
            sample = torch.from_numpy(tokens.astype(np.int64))
            
            # Update position
            self.positions[src] = f.tell() // 4
            
            return sample
            
        except Exception as e:
            print(f"Error reading from {src}: {e}")
            return None
            
    def _get_next_source(self):
        """Select next source according to weights"""
        return self.rng.choice(self.data_sources, p=self.weights)
        
    def _fill_buffer(self):
        """Fill shuffle buffer if needed"""
        while len(self.buffer) < self.shuffle_buffer_size:
            # Get next source
            src = self._get_next_source()
            
            # Read sample
            sample = self._read_block(src)
            
            # Add to buffer if valid
            if sample is not None and len(sample) == self.block_size + 1:
                self.buffer.append(sample)
                
            # Stop if we've filled enough
            if len(self.buffer) >= self.shuffle_buffer_size:
                break
                
    def __iter__(self):
        worker_info = get_worker_info()
        
        # Create fresh RNG for this worker
        if worker_info is not None:
            # Create different seed for each worker
            worker_seed = self.seed + worker_info.id
            self.rng = np.random.RandomState(worker_seed)
        
        # Clear buffer for fresh iteration
        self.buffer = []
        
        # Multiprocessing handling - split sources among workers
        if worker_info is not None:
            # For simplicity, each worker gets a subset of sources
            w_id = worker_info.id
            num_workers = worker_info.num_workers
            
            if num_workers > len(self.data_sources):
                # More workers than sources, split based on position
                self.weights = self.weights  # Keep original weights
            else:
                # Assign specific sources to this worker
                worker_sources = []
                worker_weights = []
                
                for i, (src, weight) in enumerate(zip(self.data_sources, self.weights)):
                    if i % num_workers == w_id:
                        worker_sources.append(src)
                        worker_weights.append(weight)
                
                if not worker_sources:
                    # Fallback if no sources assigned
                    worker_sources = [self.data_sources[w_id % len(self.data_sources)]]
                    worker_weights = [1.0]
                
                # Normalize weights
                total = sum(worker_weights)
                worker_weights = [w / total for w in worker_weights]
                
                # Update for this worker
                self.data_sources = worker_sources
                self.weights = worker_weights
        
        # Main iteration loop
        try:
            while True:
                if self.shuffle:
                    # Fill buffer and yield from it
                    self._fill_buffer()
                    
                    # If buffer is still empty, we're done
                    if not self.buffer:
                        break
                        
                    # Shuffle and yield from buffer
                    indices = torch.randperm(len(self.buffer))
                    for idx in indices:
                        yield self.buffer[idx]
                        
                    # Clear buffer for next iteration
                    self.buffer = []
                else:
                    # Simple streaming without shuffle
                    src = self._get_next_source()
                    sample = self._read_block(src)
                    
                    if sample is not None and len(sample) == self.block_size + 1:
                        yield sample
                    
        finally:
            # Close file handles when done
            for handle in self.file_handles.values():
                if handle is not None:
                    handle.close()
            self.file_handles = {}
            
    def __len__(self):
        """Estimate length for progress tracking"""
        # Estimate number of samples based on total tokens and block size
        if self.total_tokens <= self.block_size:
            return 0
        else:
            # Rough estimate - actual count may vary due to source sampling and file boundaries
            return max(0, self.total_tokens - self.block_size)

# --- Learning Rate Scheduler ---
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        step = self.step_count
        if step < self.warmup_steps:
            lr_scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, progress) # Clamp progress to 1.0
            lr_scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lr[i] * lr_scale

# --- Optimizer with parameter groups for selective weight decay ---
def create_optimizer_groups(model, lr, weight_decay, beta1, beta2):
    """Create optimizer with selective weight decay"""
    # Separate parameters into two groups: with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Skip weight decay for LayerNorm/RMSNorm, embeddings, and all biases
        if any(nd in name for nd in ['ln', 'norm', 'embedding']) or name.endswith('bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_params = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_params,
        lr=lr,
        betas=(beta1, beta2)
    )
    
    return optimizer

# --- Logging and Metrics Utilities ---
class MetricsLogger:
    def __init__(self, log_dir='./logs', use_tensorboard=True, use_csv=True):
        """
        Unified metrics logging for training and evaluation
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to log to TensorBoard
            use_csv: Whether to log to CSV files
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.writer = None
        self.csv_files = {}
        self.best_metric = float('inf')
        
        # Create log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Initialize TensorBoard if requested
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logging enabled at {log_dir}")
            except ImportError:
                print("TensorBoard not available. Install with pip install tensorboard")
                self.use_tensorboard = False
                
        # Initialize CSV files if requested
        if use_csv:
            self.train_csv_path = os.path.join(log_dir, 'train_metrics.csv')
            self.valid_csv_path = os.path.join(log_dir, 'valid_metrics.csv')
            
            # Initialize CSV files with headers
            for path, metrics in [(self.train_csv_path, ['step', 'epoch', 'loss', 'lr', 'throughput']),
                                 (self.valid_csv_path, ['step', 'epoch', 'loss', 'ppl'])]:
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        f.write(','.join(metrics) + '\n')
                        
    def log_train_metrics(self, step, epoch, loss, lr, throughput=None):
        """Log training metrics"""
        # Log to TensorBoard
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar('train/loss', loss, step)
            self.writer.add_scalar('train/lr', lr, step)
            if throughput is not None:
                self.writer.add_scalar('train/throughput', throughput, step)
                
        # Log to CSV
        if self.use_csv:
            with open(self.train_csv_path, 'a') as f:
                values = [step, epoch, loss, lr]
                if throughput is not None:
                    values.append(throughput)
                else:
                    values.append('')
                f.write(','.join(map(str, values)) + '\n')
                
    def log_valid_metrics(self, step, epoch, loss, ppl=None):
        """Log validation metrics and check if this is the best model so far"""
        # Calculate perplexity if not provided
        if ppl is None and loss is not None:
            ppl = math.exp(loss)
            
        # Log to TensorBoard
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar('valid/loss', loss, step)
            if ppl is not None:
                self.writer.add_scalar('valid/ppl', ppl, step)
                
        # Log to CSV
        if self.use_csv:
            with open(self.valid_csv_path, 'a') as f:
                values = [step, epoch, loss]
                if ppl is not None:
                    values.append(ppl)
                else:
                    values.append('')
                f.write(','.join(map(str, values)) + '\n')
                
        # Return whether this is the best model so far
        is_best = False
        if loss < self.best_metric:
            self.best_metric = loss
            is_best = True
            
        return is_best
        
    def log_generated_text(self, step, epoch, text, tag='generation'):
        """Log generated text samples (for qualitative evaluation)"""
        if self.use_tensorboard and self.writer:
            self.writer.add_text(f'{tag}/epoch_{epoch}', text, step)
            
    def close(self):
        """Close all open file handles"""
        if self.use_tensorboard and self.writer:
            self.writer.close()
            
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, metric):
        score = metric
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
                
        self.counter += 1
        return self.counter >= self.patience

class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        
    def step(self, metric):
        if self.mode == 'min':
            is_better = metric < self.best
        else:
            is_better = metric > self.best
            
        if is_better:
            self.best = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
                return True
            return False
            
    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")
            
    def state_dict(self):
        return {
            'best': self.best,
            'counter': self.counter,
        }
        
    def load_state_dict(self, state_dict):
        self.best = state_dict['best'] 
        self.counter = state_dict['counter']

# --- Text Generation ---
class TextGenerator:
    """Handles text generation from a TinyMamba model with various sampling strategies"""
    
    def __init__(self, model, tokenizer=None, max_length=100, temperature=0.8, 
                 top_k=40, top_p=0.9, repetition_penalty=1.1, device='cuda'):
        """
        Initialize text generator
        
        Args:
            model: TinyMamba model instance
            tokenizer: Optional tokenizer for encoding/decoding text
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability cutoff for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.device = device
        
    def apply_temperature(self, logits, temperature):
        """Apply temperature to logits"""
        if temperature == 0:
            # Greedy decoding
            return torch.zeros_like(logits).scatter_(
                1, torch.argmax(logits, dim=-1, keepdim=True), 1.0
            )
        logits = logits / max(temperature, 1e-8)
        return logits
        
    def apply_top_k(self, logits, k):
        """Apply top-k filtering to logits"""
        if k <= 0:
            return logits
            
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1).expand_as(logits)
        return torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
    def apply_top_p(self, logits, p):
        """Apply nucleus (top-p) sampling to logits"""
        if p <= 0.0:
            return logits
            
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create scatter tensor to map sorted indices to original
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        
        # Apply cutoff
        logits.masked_fill_(indices_to_remove, float('-inf'))
        return logits
        
    def apply_repetition_penalty(self, logits, input_ids, penalty):
        """Apply repetition penalty to logits"""
        if penalty == 1.0:
            return logits
            
        # Get score of tokens that appear in input_ids
        token_penalties = torch.ones_like(logits)
        for token_id in input_ids.tolist():
            # Get logits for this token across all positions
            token_logits = logits[0, token_id].item()
            # Apply penalty based on sign of logits
            if token_logits > 0:
                token_penalties[0, token_id] /= penalty
            else:
                token_penalties[0, token_id] *= penalty
                
        # Apply penalties to logits
        return logits * token_penalties
        
    def sample_token(self, logits):
        """Sample next token from processed logits"""
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    
    @torch.no_grad()
    def generate(self, prompt_ids=None, prompt_text=None, **kwargs):
        """
        Generate text from a prompt
        
        Args:
            prompt_ids: Token IDs for prompt (optional)
            prompt_text: Text prompt (optional, requires tokenizer)
            **kwargs: Override generation parameters
            
        Returns:
            generated_ids: List of token IDs
            generated_text: Decoded text if tokenizer available
        """
        # Override default parameters with kwargs
        max_length = kwargs.get('max_length', self.max_length)
        temperature = kwargs.get('temperature', self.temperature)
        top_k = kwargs.get('top_k', self.top_k)
        top_p = kwargs.get('top_p', self.top_p)
        repetition_penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
        
        # Prepare input tokens
        if prompt_ids is None and prompt_text is not None and self.tokenizer is not None:
            prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)
        elif prompt_ids is None:
            # Default to a special token or empty if no prompt
            prompt_ids = torch.tensor([[0]], device=self.device)
            
        # Ensure prompt_ids is on the correct device
        prompt_ids = prompt_ids.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize generation with prompt
        generated_ids = prompt_ids[0].tolist()
        input_ids = prompt_ids
        
        # Prepare model states
        past_key_values = None
        
        # Generate tokens
        for _ in range(max_length):
            # Forward pass through model
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply transformations
            next_token_logits = self.apply_temperature(next_token_logits, temperature)
            next_token_logits = self.apply_repetition_penalty(next_token_logits, input_ids, repetition_penalty)
            next_token_logits = self.apply_top_k(next_token_logits, top_k)
            next_token_logits = self.apply_top_p(next_token_logits, top_p)
            
            # Sample next token
            next_token = self.sample_token(next_token_logits)
            
            # Update input for next iteration
            generated_ids.append(next_token.item())
            input_ids = next_token.unsqueeze(0)
            
            # Stop if we generate EOS token
            if self.tokenizer is not None and next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Decode if tokenizer is available
        generated_text = None
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(generated_ids)
            
        return {
            'token_ids': generated_ids,
            'text': generated_text
        }

# --- Update main function to use the new optimizer ---
def main(rank, world_size):
    """Main training loop function for DDP."""
    # Set the global rank at the beginning of main
    global global_rank
    global_rank = rank
    
    is_ddp = world_size > 1
    if is_ddp:
        setup(rank, world_size)

    device = setup_hardware_precision(rank)

    if rank == 0:
        print("--- Configuration ---")
        for key, value in config.__class__.__dict__.items():
            if not key.startswith('__'):
                 print(f"{key}: {getattr(config, key)}")
        print("--------------------")
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

    # --- Data Loading ---
    train_dataset = StreamingTokenDataset(config.train_data_path, config.block_size)

    # Don't use a sampler with IterableDataset
    train_sampler = None  # Remove DistributedSampler for IterableDataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        # sampler=train_sampler,  # Remove this line completely
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    # Note: Validation loader setup would go here if validation is implemented

    # --- Model Initialization ---
    model = TinyMambaModel(config)
    model.to(device)

    if rank == 0:
         total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
         print(f"Model has {total_params:,} trainable parameters")

    # --- DDP Wrapping and Compilation ---
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        # Note: torch.compile should generally wrap the DDP model
        if config.use_compile:
             if rank == 0: print("Using torch.compile with DDP")
             try:
                  model = torch.compile(model, mode='default') # Or 'reduce-overhead'
                  if rank == 0: print("Model successfully compiled")
             except Exception as e:
                  if rank == 0: print(f"Torch compile failed: {e}. Continuing without compilation.")
    elif config.use_compile: # Non-DDP compilation
         if rank == 0: print("Using torch.compile (single process)")
         try:
              model = torch.compile(model, mode='default')
              if rank == 0: print("Model successfully compiled")
         except Exception as e:
              if rank == 0: print(f"Torch compile failed: {e}. Continuing without compilation.")

    # --- Use the new optimizer with selective weight decay ---
    optimizer = create_optimizer_groups(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2
    )
    
    if rank == 0:
        # Log parameter groups
        decay_params = sum(p.numel() for g in optimizer.param_groups if g['weight_decay'] > 0 for p in g['params'])
        no_decay_params = sum(p.numel() for g in optimizer.param_groups if g['weight_decay'] == 0 for p in g['params'])
        print(f"Optimizer: {decay_params:,} parameters with weight decay, {no_decay_params:,} without")

    # --- Estimate total steps ---
    # Need an estimate of dataset length for scheduler, use __len__ from one instance
    # This might be inaccurate for iterable datasets, consider setting max_steps manually
    approx_total_sequences = len(train_dataset)
    if is_ddp:
        # Each rank processes roughly total_sequences / world_size
        # Iterable loader length might be tricky, this is just an estimate!
        approx_steps_per_epoch = approx_total_sequences // (config.batch_size * config.gradient_accumulation_steps * world_size)
    else:
        approx_steps_per_epoch = approx_total_sequences // (config.batch_size * config.gradient_accumulation_steps)

    total_steps = approx_steps_per_epoch * config.num_epochs
    if rank == 0: print(f"Estimated total training steps: {total_steps}")

    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=total_steps
    )

    # --- Gradient Scaling for AMP ---
    scaler = GradScaler(enabled=(config.amp_dtype != torch.float32))

    # --- Checkpoint Loading ---
    start_epoch = 0
    global_step = 0
    if config.resume and os.path.exists(config.checkpoint_path):
        if rank == 0: print(f"Resuming training from {config.checkpoint_path}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if is_ddp else device
        checkpoint = torch.load(config.checkpoint_path, map_location=map_location)

        # Handle model state dict loading (DDP vs non-DDP)
        model_to_load = model.module if is_ddp else model
        # Adjust state dict keys if saved from DDP/non-DDP and loading into the other
        state_dict = checkpoint['model_state_dict']
        # Basic check for DDP keys (presence of 'module.')
        saved_with_ddp = any(k.startswith('module.') for k in state_dict.keys())
        if is_ddp and not saved_with_ddp:
             state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_ddp and saved_with_ddp:
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        try:
             model_to_load.load_state_dict(state_dict, strict=True) # Be strict
        except RuntimeError as e:
             print(f"Rank {rank}: State dict loading error (possibly mismatched keys): {e}")
             # Add more flexible loading logic here if needed (e.g., ignore missing keys)

        if 'optimizer_state_dict' in checkpoint:
             try:
                  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             except Exception as e:
                  if rank == 0: print(f"Warning: Could not load optimizer state: {e}")
        if 'scaler_state_dict' in checkpoint and scaler is not None:
             try:
                  scaler.load_state_dict(checkpoint['scaler_state_dict'])
             except Exception as e:
                  if rank == 0: print(f"Warning: Could not load GradScaler state: {e}")
        if 'scheduler_state_dict' in checkpoint:
             try:
                  lr_scheduler.step_count = checkpoint['scheduler_state_dict']['step_count']
             except Exception as e:
                  if rank == 0: print(f"Warning: Could not load LR Scheduler state: {e}")

        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        # Ensure scheduler step count matches global step if resuming
        # lr_scheduler.step_count = global_step # Force sync

        if rank == 0: print(f"Resumed from epoch {start_epoch-1}, global step {global_step}")
        del checkpoint # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    # Setup logging
    logger = MetricsLogger(
        log_dir=os.path.join(os.path.dirname(config.checkpoint_path), "logs"),
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
        run_config=vars(config)
    )
    
    # Setup early stopping
    early_stopping = None
    if config.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min"  # Lower validation loss is better
        )
    
    # Initialize learning rate scheduler and optimizer
    # ... existing optimizer setup code ...
    
    # Initialize learning rate scheduler
    # ... existing scheduler setup code ...
    
    # Create validation data loader
    val_loader = None
    if os.path.exists(config.val_data_path) and config.enable_validation:
        val_dataset = StreamingTokenDataset(
            config.val_data_path, 
            config.block_size,
            shuffle=False  # No need to shuffle for validation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size // 2,  # Use smaller batch for validation
            pin_memory=True,
            num_workers=1  # Use fewer workers for validation
        )
    
    # Setup tokenizer if needed for text generation
    tokenizer = None
    if config.generate_samples:
        # Look for vocab file in the same directory as the data
        vocab_path = os.path.join(os.path.dirname(config.train_data_path), "vocab.txt")
        if os.path.exists(vocab_path):
            tokenizer = SimpleTokenizer(vocab_file=vocab_path)
            print(f"Loaded tokenizer from {vocab_path} with {len(tokenizer)} tokens")
        else:
            # Create a simple tokenizer without vocabulary
            tokenizer = SimpleTokenizer()
            print("Using simple tokenizer with byte fallback (no vocabulary file found)")
    
    # Setup text generator with tokenizer
    if config.generate_samples:
        text_generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            max_length=config.generation_length,
            temperature=config.generation_temperature,
            device=device
        )
    
    # Training state tracking
    global_step = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    training_start_time = time.time()
    saved_checkpoints = []  # Track saved checkpoints for top-k
    
    # Initialize profiler if main process
    profiler = None
    # Main training loop
    for epoch in range(config.num_epochs):
        # ... existing train loader setup ...
        
        model.train()
        progress_bar = None
        if rank == 0:
            progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_loader):
            # ... existing training step code ...
            
            # Update running loss for logging
            running_loss += loss.item()
            
            # Log training metrics
            if rank == 0 and global_step % config.log_interval == 0:
                # Calculate training metrics
                avg_loss = running_loss / config.log_interval
                lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                
                # Log metrics
                logger.log_train_metrics(
                    global_step=global_step,
                    loss=avg_loss,
                    learning_rate=lr,
                    epoch=epoch,
                    throughput=config.batch_size * config.log_interval / (time.time() - training_start_time)
                )
                
                # Reset for next logging interval
                running_loss = 0.0
                training_start_time = time.time()
            
            # Run validation
            if val_loader is not None and config.enable_validation and global_step % config.validation_interval == 0:
                val_loss = validate(model, val_loader, device, config.validation_steps)
                
                if rank == 0:
                    # Log validation metrics
                    logger.log_valid_metrics(
                        global_step=global_step,
                        loss=val_loss,
                        epoch=epoch
                    )
                    
                    print(f"Step {global_step} | Validation Loss: {val_loss:.4f}")
                    
                    # Check for best model and save checkpoint
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        
                    # Save checkpoint
                    checkpoint_path = save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        val_loss=val_loss,
                        config=config,
                        is_best=is_best
                    )
                    
                    # Manage saved checkpoints (keep only top k)
                    if checkpoint_path:
                        saved_checkpoints.append((checkpoint_path, val_loss))
                        saved_checkpoints.sort(key=lambda x: x[1])  # Sort by loss
                        
                        # Remove worst checkpoints if we have more than save_top_k
                        while len(saved_checkpoints) > config.save_top_k:
                            worst_path, _ = saved_checkpoints.pop()
                            if os.path.exists(worst_path) and "best" not in worst_path:
                                os.remove(worst_path)
                                print(f"Removed checkpoint: {worst_path}")
                    
                    # Check early stopping
                    if early_stopping and early_stopping(val_loss):
                        print(f"Early stopping triggered after {global_step} steps")
                        # Break out of both loops
                        break
            
            # Generate text samples
            if rank == 0 and config.generate_samples and global_step % config.generation_interval == 0:
                generate_and_log_samples(
                    generator=text_generator, 
                    logger=logger, 
                    global_step=global_step
                )
            
            # Update progress bar
            if rank == 0 and progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})
            
            # Clear cache periodically
            if step > 0 and step % config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
            
            # Increment global step
            global_step += 1
        
        # Close progress bar for epoch
        if rank == 0 and progress_bar is not None:
            progress_bar.close()
            
        # Check if early stopping triggered
        if early_stopping and early_stopping.stopped:
            break
    
    # Cleanup
    logger.close()
    cleanup()


# --- Validation function ---
@torch.no_grad()
def validate(model, val_loader, device, max_steps=None):
    """Run validation loop and return average loss"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    for step, batch in enumerate(val_loader):
        # Stop after max_steps if provided
        if max_steps is not None and step >= max_steps:
            break
            
        # Move batch to device
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # Forward pass
        logits = model(x)
        
        # Calculate loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Update metrics
        total_loss += loss.item()
        total_steps += 1
    
    # Switch back to training mode
    model.train()
    
    # Return average loss
    return total_loss / max(1, total_steps)


# --- Checkpoint saving ---
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, config, is_best=False):
    """Save model checkpoint"""
    # Determine checkpoint directory
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint info
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'val_loss': val_loss,
        'config': vars(config),
    }
    
    # Create checkpoint path with step info
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"tinymamba_step_{global_step:07d}.pt"
    )
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
    
    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, "tinymamba_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model with val_loss {val_loss:.4f}")
    
    # Also save latest checkpoint (overwrite)
    latest_path = os.path.join(checkpoint_dir, "tinymamba_latest.pt")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


# --- Text generation helper ---
def generate_and_log_samples(generator, logger, global_step, prompts=None):
    """Generate and log text samples"""
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the distant future"
        ]
    
    # Generate samples
    samples = []
    for prompt in prompts:
        result = generator.generate(prompt_text=prompt)
        samples.append({
            "prompt": prompt,
            "generated": result["text"] if result["text"] else "N/A",
            "token_ids": result["token_ids"]
        })
    
    # Log samples
    logger.log_generated_text(global_step, samples)
    
    # Print a sample
    if samples:
        print(f"\n--- Generated Sample at Step {global_step} ---")
        print(f"Prompt: {samples[0]['prompt']}")
        print(f"Generated: {samples[0]['generated'][:200]}...")
        print("---------------------------------------------\n")

# --- DDP Launcher ---
def run_training():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs. Using DDP.")
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Found 1 or 0 GPUs. Running in single process mode.")
        main(rank=0, world_size=1) # Run directly for single GPU or CPU

if __name__ == "__main__":
    # Check if data files exist before starting
    if not os.path.exists(config.train_data_path):
         print(f"ERROR: Training data not found at {config.train_data_path}")
         print("Please ensure data is prepared and paths in Config are correct.")
         exit(1) # Exit if data is missing

    run_training()

# --- Simple Tokenizer ---
class SimpleTokenizer:
    """
    A simple vocabulary-based tokenizer with byte-level fallback
    
    Features:
    - Vocab-based tokenization with byte fallback for OOV
    - Support for special tokens
    - Handles encoding and decoding for model I/O
    """
    def __init__(self, vocab_file=None, unk_token="<unk>", pad_token="<pad>", 
                bos_token="<s>", eos_token="</s>", max_token_value=50000):
        """
        Initialize the tokenizer
        
        Args:
            vocab_file: Path to vocabulary file (one token per line)
            unk_token: Token to use for unknown words
            pad_token: Token to use for padding
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            max_token_value: Maximum value for token IDs (for safety)
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Initialize token maps
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        special_tokens = [pad_token, unk_token, bos_token, eos_token]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # Special token IDs
        self.pad_token_id = self.token_to_id[pad_token]
        self.unk_token_id = self.token_to_id[unk_token]
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
        
        self.max_token_value = max_token_value
        
        # Add additional vocabulary if file provided
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
            print(f"Loaded vocabulary with {len(self.token_to_id)} tokens")
        else:
            print(f"No vocabulary file found. Using byte encoding fallback with special tokens.")
            
    def _load_vocab(self, vocab_file):
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token and token not in self.token_to_id:
                    token_id = len(self.token_to_id)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                    
                    # Safety check
                    if token_id >= self.max_token_value:
                        print(f"Warning: Vocabulary exceeds max_token_value ({self.max_token_value}). Truncating.")
                        break
    
    def _tokenize_text(self, text):
        """Simple whitespace tokenization with fallback to bytes for unknown tokens"""
        if not text:
            return []
            
        # Basic whitespace tokenization
        tokens = text.split()
        
        # Check each token against vocabulary
        result = []
        for token in tokens:
            if token in self.token_to_id:
                result.append(token)
            else:
                # Fallback: encode unknown tokens as bytes
                for byte in token.encode('utf-8'):
                    byte_token = f"<byte_{byte}>"
                    if byte_token not in self.token_to_id:
                        # Add to vocabulary if new
                        token_id = len(self.token_to_id)
                        self.token_to_id[byte_token] = token_id
                        self.id_to_token[token_id] = byte_token
                    result.append(byte_token)
                    
        return result
    
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        """
        Encode text to token IDs
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Return format ('pt' for PyTorch tensors, None for list)
            
        Returns:
            Token IDs in specified format
        """
        if not text:
            # Return empty sequence or just special tokens if requested
            ids = []
            if add_special_tokens:
                ids = [self.bos_token_id, self.eos_token_id]
        else:
            # Tokenize and convert to IDs
            tokens = self._tokenize_text(text)
            ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
            
            # Add special tokens if requested
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
                
        # Convert to tensor if requested
        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to remove special tokens from output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Convert IDs to tokens
        tokens = []
        byte_buffer = []
        
        for token_id in token_ids:
            # Handle out-of-range token IDs
            if token_id >= len(self.id_to_token):
                token = self.unk_token
            else:
                token = self.id_to_token.get(token_id, self.unk_token)
                
            # Skip special tokens if requested
            if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
                
            # Handle byte tokens
            if token.startswith("<byte_") and token.endswith(">"):
                try:
                    byte_value = int(token[6:-1])
                    byte_buffer.append(byte_value)
                except ValueError:
                    # If parsing fails, treat as normal token
                    if byte_buffer:
                        # Flush byte buffer first
                        tokens.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
                        byte_buffer = []
                    tokens.append(token)
            else:
                # Regular token - first flush any byte buffer
                if byte_buffer:
                    tokens.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
                    byte_buffer = []
                tokens.append(token)
                
        # Handle any remaining bytes in buffer
        if byte_buffer:
            tokens.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
            
        # Join tokens with spaces
        return " ".join(tokens)
        
    def __len__(self):
        """Return vocabulary size"""
        return len(self.token_to_id)

# --- Performance Profiling ---
class ModelProfiler:
    """
    Tracks model performance metrics during training and inference
    
    Features:
    - Parameter counting by component
    - Runtime profiling by component
    - Memory usage tracking
    - FLOP estimation (basic)
    """
    def __init__(self, model, sample_input_size=(8, 512), 
                 device='cuda', detailed=True, profile_step=2000):
        """
        Initialize profiler
        
        Args:
            model: Model to profile
            sample_input_size: Input shape for profiling (batch_size, seq_len)
            device: Device to run profiling on
            detailed: Whether to collect detailed stats on model components
            profile_step: How often to run detailed profiling during training
        """
        self.model = model
        self.sample_input_size = sample_input_size
        self.device = device
        self.detailed = detailed
        self.profile_step = profile_step
        
        # Tracking data
        self.param_counts = {}
        self.layer_latencies = {}
        self.forward_times = []
        self.backward_times = []
        self.memory_stats = []
        
        # Hooks
        self.hooks = []
        
        # Analyze model structure
        self._count_parameters()
        if detailed:
            self._register_hooks()
    
    def _count_parameters(self):
        """Count parameters by model component"""
        # Total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.param_counts['total'] = total_params
        
        # Parameters by top-level component
        for name, module in self.model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            self.param_counts[name] = param_count
            
            # Get deeper component breakdown for specific modules
            if hasattr(module, '_modules'):
                for subname, submodule in module._modules.items():
                    if not submodule:
                        continue
                    params = sum(p.numel() for p in submodule.parameters())
                    self.param_counts[f"{name}.{subname}"] = params
        
    def _register_hooks(self):
        """Register forward/backward hooks for timing"""
        def forward_hook(module, input, output):
            if not hasattr(module, '_forward_start_time'):
                return
            
            elapsed = time.time() - module._forward_start_time
            module_name = module.__class__.__name__
            
            if module_name not in self.layer_latencies:
                self.layer_latencies[module_name] = {'forward': [], 'backward': []}
            
            self.layer_latencies[module_name]['forward'].append(elapsed)
        
        def forward_pre_hook(module, input):
            module._forward_start_time = time.time()
        
        def backward_hook(module, grad_input, grad_output):
            if not hasattr(module, '_backward_start_time'):
                return
                
            elapsed = time.time() - module._backward_start_time
            module_name = module.__class__.__name__
            
            if module_name not in self.layer_latencies:
                self.layer_latencies[module_name] = {'forward': [], 'backward': []}
                
            self.layer_latencies[module_name]['backward'].append(elapsed)
            
        def backward_pre_hook(module, grad_input):
            module._backward_start_time = time.time()
        
        # Register hooks on key components
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)) or \
               'MambaSSM' in module.__class__.__name__ or \
               'Attention' in module.__class__.__name__:
                fwd_pre = module.register_forward_pre_hook(forward_pre_hook)
                fwd = module.register_forward_hook(forward_hook)
                bwd_pre = module.register_full_backward_pre_hook(backward_pre_hook)
                bwd = module.register_full_backward_hook(backward_hook)
                self.hooks.extend([fwd_pre, fwd, bwd_pre, bwd])
                
    def capture_forward_time(self, start_time):
        """Record forward pass time"""
        self.forward_times.append(time.time() - start_time)
        
    def capture_backward_time(self, start_time):
        """Record backward pass time"""
        self.backward_times.append(time.time() - start_time)
        
    def capture_memory_stats(self):
        """Capture current GPU memory usage"""
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            self.memory_stats.append({
                'allocated': allocated,
                'reserved': reserved,
                'step': len(self.memory_stats)
            })
            
    def profile_forward(self):
        """Run a profiling forward pass"""
        # Create sample input
        batch_size, seq_len = self.sample_input_size
        sample_input = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Profile forward pass
        with torch.no_grad():
            start_time = time.time()
            _ = self.model(sample_input)
            self.capture_forward_time(start_time)
            
        # Capture memory usage
        self.capture_memory_stats()
        
    def estimate_flops(self):
        """Estimate model FLOPs for one forward pass (very approximate)"""
        batch_size, seq_len = self.sample_input_size
        flops = 0
        
        # Estimate for different components
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear: 2 * in_features * out_features * batch_size * seq_len
                in_features = module.in_features
                out_features = module.out_features
                flops += 2 * in_features * out_features * batch_size * seq_len
                
            elif isinstance(module, nn.Conv1d):
                # Conv1d: 2 * kernel_size * in_channels * out_channels * output_size * batch_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                # Rough output size estimate for causal conv
                output_size = seq_len
                flops += 2 * kernel_size * in_channels * out_channels * output_size * batch_size
                
            elif 'Attention' in module.__class__.__name__:
                # Rough attention FLOP estimate
                if hasattr(module, 'n_heads') and hasattr(module, 'd_model'):
                    n_heads = module.n_heads
                    d_model = module.d_model
                    # 4 * d_model^2 (QKV projections) + 2 * seq_len^2 * d_model (attention matrix) 
                    flops += 4 * d_model * d_model * batch_size * seq_len
                    flops += 2 * seq_len * seq_len * d_model * batch_size
        
        return flops
    
    def format_size(self, num_bytes):
        """Format byte size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"
    
    def get_summary(self, include_latency=True):
        """Get profiling summary as a dict"""
        summary = {
            'parameters': {
                'total': self.param_counts.get('total', 0),
                'by_component': {k: v for k, v in self.param_counts.items() if k != 'total'}
            },
            'memory': {
                'current_allocated_mb': 0,
                'peak_allocated_mb': 0
            },
            'performance': {
                'avg_forward_ms': 0,
                'avg_backward_ms': 0,
                'throughput_seq_per_sec': 0,
                'estimated_flops': self.estimate_flops()
            }
        }
        
        # Add memory statistics
        if self.device == 'cuda' and torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            summary['memory']['current_allocated_mb'] = current
            summary['memory']['peak_allocated_mb'] = peak
            
        # Add timing data
        if self.forward_times:
            avg_forward = sum(self.forward_times) / max(1, len(self.forward_times))
            summary['performance']['avg_forward_ms'] = avg_forward * 1000
            
            # Estimate throughput
            batch_size = self.sample_input_size[0]
            if avg_forward > 0:
                summary['performance']['throughput_seq_per_sec'] = batch_size / avg_forward
        
        if self.backward_times:
            avg_backward = sum(self.backward_times) / max(1, len(self.backward_times))
            summary['performance']['avg_backward_ms'] = avg_backward * 1000
            
        # Add latency breakdown if requested and available
        if include_latency and self.layer_latencies:
            layer_perf = {}
            for layer_name, times in self.layer_latencies.items():
                if times['forward']:
                    avg_fwd = sum(times['forward']) / max(1, len(times['forward']))
                    layer_perf[f"{layer_name}_forward_ms"] = avg_fwd * 1000
                
                if times['backward']:
                    avg_bwd = sum(times['backward']) / max(1, len(times['backward']))
                    layer_perf[f"{layer_name}_backward_ms"] = avg_bwd * 1000
                    
            summary['performance']['layer_breakdown'] = layer_perf
            
        return summary
        
    def log_summary(self, logger=None, global_step=None):
        """Log profiling summary"""
        summary = self.get_summary()
        
        # Print summary
        print("\n--- MODEL PROFILING SUMMARY ---")
        print(f"Parameters: {summary['parameters']['total']:,} total")
        
        # Print parameter breakdown
        print("\nParameter distribution:")
        component_params = summary['parameters']['by_component']
        for component, count in sorted(component_params.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = 100 * count / max(1, summary['parameters']['total'])
            print(f"  {component}: {count:,} ({pct:.1f}%)")
            
        # Print memory usage
        print("\nMemory usage:")
        print(f"  Current: {summary['memory']['current_allocated_mb']:.2f} MB")
        print(f"  Peak: {summary['memory']['peak_allocated_mb']:.2f} MB")
        
        # Print performance metrics
        print("\nPerformance:")
        print(f"  Forward pass: {summary['performance']['avg_forward_ms']:.2f} ms")
        print(f"  Backward pass: {summary['performance']['avg_backward_ms']:.2f} ms")
        print(f"  Throughput: {summary['performance']['throughput_seq_per_sec']:.2f} sequences/sec")
        print(f"  Estimated FLOPs: {summary['performance']['estimated_flops']:,}")
        
        # Print layer breakdown if available
        if 'layer_breakdown' in summary['performance']:
            print("\nTop 5 most expensive operations (forward):")
            forward_times = {k: v for k, v in summary['performance']['layer_breakdown'].items() if 'forward' in k}
            for op, time_ms in sorted(forward_times.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {op}: {time_ms:.2f} ms")
                
        print("----------------------------------\n")
        
        # Log to logger if provided
        if logger is not None and global_step is not None:
            # Log parameter counts
            logger.writer.add_scalar('profiler/params_total', summary['parameters']['total'], global_step)
            
            # Log top components
            for component, count in sorted(component_params.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.writer.add_scalar(f'profiler/params_{component}', count, global_step)
                
            # Log memory
            logger.writer.add_scalar('profiler/memory_current_mb', summary['memory']['current_allocated_mb'], global_step)
            logger.writer.add_scalar('profiler/memory_peak_mb', summary['memory']['peak_allocated_mb'], global_step)
            
            # Log performance
            logger.writer.add_scalar('profiler/forward_ms', summary['performance']['avg_forward_ms'], global_step)
            logger.writer.add_scalar('profiler/backward_ms', summary['performance']['avg_backward_ms'], global_step)
            logger.writer.add_scalar('profiler/throughput', summary['performance']['throughput_seq_per_sec'], global_step)
            
            # Log top 3 most expensive ops
            forward_times = {k: v for k, v in summary['performance']['layer_breakdown'].items() if 'forward' in k}
            for i, (op, time_ms) in enumerate(sorted(forward_times.items(), key=lambda x: x[1], reverse=True)[:3]):
                op_name = op.replace('_forward_ms', '')
                logger.writer.add_scalar(f'profiler/top_ops_{i}_{op_name}', time_ms, global_step)
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# --- Update main function with profiler ---
def main(rank, world_size):
    # ... existing setup code ...
    
    # Initialize profiler if main process
    profiler = None
    if rank == 0:
        profiler = ModelProfiler(
            model=model,
            sample_input_size=(config.batch_size, config.block_size),
            device=device,
            profile_step=5000  # Profile every 5000 steps
        )
        
        # Log initial profile summary
        profiler.profile_forward()
        profiler.log_summary(logger, 0)
    
    # ... existing training loop code ...
    
    for step, batch in enumerate(train_loader):
        # ... existing training step code ...
        
        # Add profiler capturing in training loop
        if rank == 0 and profiler:
            # Optionally capture forward/backward times during actual training
            forward_start = time.time()
            
        # ... existing forward pass code ...
        
        if rank == 0 and profiler:
            profiler.capture_forward_time(forward_start)
            backward_start = time.time()
            
        # ... existing backward pass code ...
        
        if rank == 0 and profiler:
            profiler.capture_backward_time(backward_start)
            
            # Run full profiling periodically
            if global_step % profiler.profile_step == 0 and global_step > 0:
                profiler.profile_forward()
                profiler.log_summary(logger, global_step)
                profiler.capture_memory_stats()
                
    # Cleanup profiler hooks at the end
    if rank == 0 and profiler:
        profiler.remove_hooks()