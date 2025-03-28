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

# Try to import optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("tiktoken not available, using simple BPE implementation")

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

# --- Butterfly projection for SSM parameters ---
class ButterflyProjection(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        # Use low-rank projection as the implementation
        # In a full implementation, this would use a butterfly pattern
        # but we're simplifying here for readability
        self.projection = LowRankProjection(d_in, d_out, rank=min(8, d_in, d_out), bias=bias)
        
    def forward(self, x=None):
        return self.projection(x)

# --- Toeplitz projection class for SSM parameters ---
class ToeplitzProjection(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        # Simplified version that uses diagonal plus low-rank
        self.diag = nn.Parameter(torch.randn(min(d_in, d_out)) * 0.02)
        self.low_rank = LowRankProjection(d_in, d_out, rank=4, bias=bias)
        
    def forward(self, x=None):
        if x is not None:
            # This is a simplified version; a true Toeplitz would have a different structure
            return self.low_rank(x)
        else:
            # For parameter generation, we'd use the true Toeplitz structure
            # But here we just use the low-rank approximation for simplicity
            return self.low_rank()

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
        h = torch.einsum('bdn,bdm->bdm', A_bar[:, t], h) + \
            dB[:, t] * x_conv[:, t].unsqueeze(-1)
        
        # y_t = C * h_t
        y_t = torch.einsum('b d n, d n -> b d', h, C)
        ys[:, t] = y_t
    
    return ys

try:
    import triton
    import triton.language as tl
    
    # Define Triton kernel for fused SSM scan operation
    @triton.jit
    def ssm_scan_kernel(
        # Pointers to matrices
        a_bar_ptr, db_ptr, x_conv_ptr, c_ptr, h_ptr, output_ptr,
        # Matrix dimensions
        batch_size, seq_len, d_inner, d_state,
        # Strides for the different dimensions
        batch_stride_a, seq_stride_a, inner_stride_a, state_stride_a,
        batch_stride_db, seq_stride_db, inner_stride_db, state_stride_db,
        batch_stride_x, seq_stride_x, inner_stride_x,
        inner_stride_c, state_stride_c,
        batch_stride_h, inner_stride_h, state_stride_h,
        batch_stride_out, seq_stride_out, inner_stride_out,
        # Optional
        BLOCK_SIZE: tl.constexpr,
    ):
        # Program ID
        batch_id = tl.program_id(0)
        inner_id = tl.program_id(1)
        
        # Initialize hidden state
        h_off = batch_id * batch_stride_h + inner_id * inner_stride_h
        h_offsets = h_off + tl.arange(0, BLOCK_SIZE) * state_stride_h
        h_mask = tl.arange(0, BLOCK_SIZE) < d_state
        h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Process sequence
        for t in range(seq_len):
            # Load A_bar for current timestep
            a_off = batch_id * batch_stride_a + t * seq_stride_a + inner_id * inner_stride_a
            a_offsets = a_off + tl.arange(0, BLOCK_SIZE) * state_stride_a
            a_mask = tl.arange(0, BLOCK_SIZE) < d_state
            a = tl.load(a_bar_ptr + a_offsets, mask=a_mask, other=0.0)
            
            # Load dB for current timestep
            db_off = batch_id * batch_stride_db + t * seq_stride_db + inner_id * inner_stride_db
            db_offsets = db_off + tl.arange(0, BLOCK_SIZE) * state_stride_db
            db = tl.load(db_ptr + db_offsets, mask=a_mask, other=0.0)
            
            # Load x_conv for current timestep
            x_off = batch_id * batch_stride_x + t * seq_stride_x + inner_id * inner_stride_x
            x = tl.load(x_conv_ptr + x_off)
            
            # Update hidden state: h = A_bar * h + dB * x
            h_new = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for i in range(d_state):
                h_val = tl.load(h_ptr + h_off + i * state_stride_h)
                h_new = h_new + a[i] * h_val
            h = h_new + db * x
            
            # Store updated hidden state
            tl.store(h_ptr + h_offsets, h, mask=h_mask)
            
            # Compute output: y = C * h
            y = tl.zeros([1], dtype=tl.float32)
            for i in range(d_state):
                c_off = inner_id * inner_stride_c + i * state_stride_c
                c_val = tl.load(c_ptr + c_off)
                h_val = h[i] if i < d_state else 0.0
                y += c_val * h_val
            
            # Store output for this timestep
            out_off = batch_id * batch_stride_out + t * seq_stride_out + inner_id * inner_stride_out
            tl.store(output_ptr + out_off, y)
    
    def fused_ssm_scan(A_bar, dB, x_conv, C):
        """
        Efficient fused SSM scan using Triton
        
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
        
        # Create output tensor
        output = torch.zeros((B, L, d_inner), device=device, dtype=torch.float32)
        
        # Create hidden state buffer
        h_buffer = torch.zeros((B, d_inner, d_state), device=device, dtype=torch.float32)
        
        # Get tensor strides for efficient memory access
        batch_stride_a, seq_stride_a, inner_stride_a, state_stride_a = A_bar.stride()
        batch_stride_db, seq_stride_db, inner_stride_db, state_stride_db = dB.stride()
        batch_stride_x, seq_stride_x, inner_stride_x = x_conv.stride()
        inner_stride_c, state_stride_c = C.stride()
        batch_stride_h, inner_stride_h, state_stride_h = h_buffer.stride()
        batch_stride_out, seq_stride_out, inner_stride_out = output.stride()
        
        # Determine block size for kernel launch
        BLOCK_SIZE = min(d_state, 32)  # Choose block size based on d_state
        
        # Launch kernel
        grid = (B, d_inner)
        ssm_scan_kernel[grid](
            A_bar, dB, x_conv, C, h_buffer, output,
            B, L, d_inner, d_state,
            batch_stride_a, seq_stride_a, inner_stride_a, state_stride_a,
            batch_stride_db, seq_stride_db, inner_stride_db, state_stride_db,
            batch_stride_x, seq_stride_x, inner_stride_x,
            inner_stride_c, state_stride_c,
            batch_stride_h, inner_stride_h, state_stride_h,
            batch_stride_out, seq_stride_out, inner_stride_out,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
except ImportError:
    # Fallback if Triton is not available
    print("Triton not available, using JIT-compiled scan")
    fused_ssm_scan = efficient_ssm_scan

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
                b = x.size(0)  # Get batch size directly from tensor dimension
                h = torch.zeros(b, d_inner, self.d_state, device=x.device)
            
            # Process sequence and update state
            ys = []
            for t in range(L):
                # Update state: h_t = A_t * h_{t-1} + B_t * x_t
                h = torch.einsum('bdn,bdm->bdm', A_bar[:, t], h) + \
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
            # Use optimized fused kernel for long sequences
            if L > 64:  # Only use for longer sequences where the overhead is worth it
                try:
                    # Use the fused kernel implementation
                    if self.training:
                        y = fused_ssm_scan(A_bar, dB, x_conv, C)
                        # Apply dropout after scan
                        y = self.state_dropout(y.unsqueeze(-1)).squeeze(-1)
                    else:
                        y = fused_ssm_scan(A_bar, dB, x_conv, C)
                except Exception as e:
                    print(f"Fused scan failed: {e}. Falling back to loop implementation.")
                    # Fall back to the original implementation
                    b = x_conv.size(0)  # Get batch size directly from tensor dimension
                    h = torch.zeros(b, d_inner, self.d_state, device=x.device)
                    ys = []
                    
                    for t in range(L):
                        h = torch.einsum('bdn,bdm->bdm', A_bar[:, t], h) + \
                            dB[:, t] * x_conv[:, t].unsqueeze(-1)
                        
                        # Apply state dropout during training
                        if self.training:
                            h = self.state_dropout(h)
                            
                        y_t = torch.einsum('b d n, d n -> b d', h, C)
                        ys.append(y_t)
                    
                    y = torch.stack(ys, dim=1)
            else:
                # For shorter sequences, the loop is more efficient
                b = x_conv.size(0)  # Get batch size directly from tensor dimension
                h = torch.zeros(b, d_inner, self.d_state, device=x.device)
                ys = []
                
                for t in range(L):
                    h = torch.einsum('bdn,bdm->bdm', A_bar[:, t], h) + \
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
                 dropout=0.1, bias=False, special_token_ids=None, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_window_size = base_window_size
        self.max_window_size = max_window_size
        self.dropout = dropout
        self.causal = causal
        
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

# --- Attention Factory to create appropriate attention mechanism ---
class AttentionFactory:
    @staticmethod
    def create_attention(attn_type, d_model, num_heads, base_window_size=64, max_window_size=256, 
                         dropout=0.1, bias=False, causal=True, special_token_ids=None):
        """
        Factory method to create different attention mechanisms
        
        Args:
            attn_type: Type of attention ("flash", "local", "full")
            d_model: Model dimension
            num_heads: Number of attention heads
            base_window_size: Base size of attention window
            max_window_size: Maximum size of attention window
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            causal: Whether to use causal attention
            special_token_ids: List of special token IDs that should have full attention
            
        Returns:
            Attention module
        """
        if attn_type == "local" or attn_type == "flash":
            # Use AdaptiveLocalAttention - it handles flash internally if available
            return AdaptiveLocalAttention(
                d_model=d_model,
                num_heads=num_heads,
                base_window_size=base_window_size,
                max_window_size=max_window_size,
                dropout=dropout,
                bias=bias,
                special_token_ids=special_token_ids,
                causal=causal
            )
        elif attn_type == "full":
            # Full attention with PyTorch SDPA
            return FullAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                causal=causal
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

# --- Full attention implementation ---
class FullAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=False, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.causal = causal
        
        # DeepNet scaling for residual branches
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Add RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, x, token_ids=None):
        B, L, D = x.shape
        
        # Process query, key, value projections
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        cos, sin = self.rotary_emb(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Apply scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal
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
    # q and k shapes: [batch, heads, seq_len, head_dim]
    # cos and sin shapes could be either:
    # 1. [seq_len, dim] - raw from RotaryEmbedding
    # 2. [1, 1, seq_len, dim] - already expanded with batch and head dims
    
    # Get dimensions
    q_dim = q.shape[-1]
    
    # Check if cos and sin already have batch and head dimensions
    if cos.ndim == 2:
        # Raw cos and sin from RotaryEmbedding
        # Ensure head_dim matches by truncating if necessary
        cos = cos[:, :q_dim].unsqueeze(0).unsqueeze(0)
        sin = sin[:, :q_dim].unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 4:
        # Already has batch and head dimensions, just ensure head_dim matches
        cos = cos[..., :q_dim]
        sin = sin[..., :q_dim]
    
    # Split values into even and odd for rotation
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Make sure dimensions match between tensors before operations
    # Get half dimension for proper reshaping
    half_dim = q_dim // 2
    cos_half = cos[..., :half_dim]
    sin_half = sin[..., :half_dim]
    
    # Apply rotation using elementwise operations for proper broadcasting
    q_embed = torch.cat([
        q_even * cos_half - q_odd * sin_half,
        q_odd * cos_half + q_even * sin_half
    ], dim=-1)
    
    k_embed = torch.cat([
        k_even * cos_half - k_odd * sin_half,
        k_odd * cos_half + k_even * sin_half
    ], dim=-1)
    
    return q_embed, k_embed

# --- Dynamic Token-Adaptive Gating ---
class TokenAdaptiveGating(nn.Module):
    def __init__(self, d_model, num_branches=3, reduction_factor=8, sparsity_k=2):
        super().__init__()
        """
        Dynamic gating mechanism that adapts branch weights based on token content
        
        Args:
            d_model: Hidden dimension size
            num_branches: Number of branches to gate (default 3: ssm, attn, mlp)
            reduction_factor: Dimension reduction for controller efficiency
            sparsity_k: Number of branches to activate (top-k) per token
        """
        self.num_branches = num_branches
        self.sparsity_k = min(sparsity_k, num_branches)  # Can't select more branches than we have
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
    
    def sparsemax(self, logits):
        """
        Sparsemax function - creates sparse distributions
        
        Args:
            logits: Input tensor of shape [..., dim]
            
        Returns:
            Sparse probability distribution with many zeros
        """
        # Sort logits in descending order
        z_sorted, _ = torch.sort(logits, dim=-1, descending=True)
        
        # Compute running sum
        dim = logits.size(-1)
        range_indices = torch.arange(1, dim+1, dtype=logits.dtype, device=logits.device)
        z_cumsum = torch.cumsum(z_sorted, dim=-1) - 1  # Subtract 1 for the threshold
        
        # Find the threshold k(x)
        k = dim - 1
        z_sorted_k = z_sorted[..., -1].unsqueeze(-1)
        cond = 1 + range_indices * z_sorted > z_cumsum
        k = torch.sum(cond, dim=-1, keepdim=True) - 1
        
        # Compute threshold tau(x)
        z_sorted_k = torch.gather(z_sorted, -1, k)
        tau = (z_cumsum.gather(-1, k) - 1) / (k.float() + 1)
        
        # Apply sparsemax transformation
        p = torch.clamp(logits - tau, min=0)
        return p
    
    def top_k_gating(self, logits, k=None):
        """
        Apply top-k gating - only keep top k values and normalize
        
        Args:
            logits: Input logits tensor [B, L, num_branches]
            k: Number of top branches to keep, defaults to self.sparsity_k
            
        Returns:
            Sparse weight tensor with only k non-zero values per token
        """
        if k is None:
            k = self.sparsity_k
            
        # Get top-k values and indices
        top_k_values, _ = torch.topk(logits, k=k, dim=-1)
        
        # Create a mask for values below the k-th value
        threshold = top_k_values[..., -1].unsqueeze(-1)
        mask = logits < threshold
        
        # Apply mask (zero out values below threshold)
        sparse_logits = logits.masked_fill(mask, float('-inf'))
        
        # Apply softmax to remaining values for normalization
        return F.softmax(sparse_logits, dim=-1)
    
    def forward(self, x, sparsity_method='top_k'):
        """
        Compute branch weights for each token in the sequence
        
        Args:
            x: Input tensor [B, L, D]
            sparsity_method: Method for sparse routing ('top_k' or 'sparsemax')
            
        Returns:
            weights: Sparse weights for each branch [B, L, num_branches]
        """
        # Get token-wise branch logits
        branch_logits = self.controller(x)  # [B, L, num_branches]
        
        # Apply sparsity based on selected method
        if sparsity_method == 'sparsemax':
            branch_weights = self.sparsemax(branch_logits)
        elif sparsity_method == 'top_k':
            branch_weights = self.top_k_gating(branch_logits)
        else:
            # Fallback to standard softmax
            branch_weights = F.softmax(branch_logits, dim=-1)
        
        return branch_weights

# --- Updated TinyMamba Block with abstracted attention mechanism ---
class TinyMambaBlock(nn.Module):
    def __init__(self, config, param_method='butterfly'):
        super().__init__()
        # Normalization layers
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.ln3 = RMSNorm(config.d_model)
        
        # Use top-2 activated branches for each token by default
        self.branch_controller = TokenAdaptiveGating(
            config.d_model, 
            num_branches=3, 
            sparsity_k=2
        )
        
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

        # Use the attention factory to create the appropriate attention mechanism
        if config.use_flash_attn and FLASH_ATTN_AVAILABLE:
            attn_type = "flash"
        else:
            attn_type = "local"
            
        self.local_attn = AttentionFactory.create_attention(
            attn_type=attn_type,
            d_model=config.d_model,
            num_heads=config.num_heads,
            base_window_size=config.window_size,
            max_window_size=min(config.window_size * 4, config.block_size),
            dropout=config.dropout,
            bias=config.bias,
            causal=True,
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
        # Use top_k gating by default for structured sparsity
        branch_weights = self.branch_controller(x, sparsity_method='top_k')
        
        # Compute outputs from each branch
        ssm_out = self.ssm(self.ln1(x))
        attn_out = self.local_attn(self.ln2(x), token_ids=token_ids)
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
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.weight is not None:
            nn.init.ones_(module.weight)
    elif isinstance(module, RMSNorm):
        # Handle RMSNorm separately since it doesn't have bias
        if hasattr(module, 'weight') and module.weight is not None:
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

# --- BPE Tokenizer Implementation ---
class TinyBPETokenizer:
    """
    An efficient BPE tokenizer implementation
    
    Features:
    - Byte-pair encoding tokenization
    - Vocabulary management
    - Special token handling
    - Fast encoding/decoding
    """
    def __init__(self, 
                vocab_file=None, 
                merges_file=None,
                tiktoken_model=None,
                unk_token="<unk>", 
                pad_token="<pad>", 
                bos_token="<s>", 
                eos_token="</s>",
                max_token_value=50304):
        """
        Initialize the BPE tokenizer
        
        Args:
            vocab_file: Path to vocabulary file (token:id mapping)
            merges_file: Path to BPE merges file (for custom BPE models)
            tiktoken_model: Name of tiktoken model to use (e.g., "gpt2")
            unk_token: Token for unknown words
            pad_token: Token for padding
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            max_token_value: Maximum token ID value
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens first
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        self.special_token_ids = {}
        
        # Initialize base vocabulary with special tokens
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            self.special_token_ids[token] = i
            
        # Special token IDs
        self.pad_token_id = self.token_to_id[pad_token]
        self.unk_token_id = self.token_to_id[unk_token]
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
        
        # Set max token value
        self.max_token_value = max_token_value
        
        # Try to use tiktoken if available
        self.use_tiktoken = False
        self.tiktoken_encoder = None
        
        if TIKTOKEN_AVAILABLE and tiktoken_model:
            try:
                self.tiktoken_encoder = tiktoken.get_encoding(tiktoken_model)
                self.use_tiktoken = True
                print(f"Using tiktoken with model: {tiktoken_model}")
                
                # Update vocabulary size based on tiktoken model
                self.max_token_value = len(self.tiktoken_encoder)
                
                # Special case for tiktoken - handle token offsets
                token_offset = len(self.special_tokens)
                
                # Add tiktoken vocab tokens to our mappings
                # but skip any that conflict with our special tokens
                for i in range(self.max_token_value):
                    try:
                        token = self.tiktoken_encoder.decode([i])
                        if token not in self.token_to_id:
                            token_id = i + token_offset
                            self.token_to_id[token] = token_id
                            self.id_to_token[token_id] = token
                    except:
                        continue  # Skip any decoding errors
            except Exception as e:
                print(f"Failed to load tiktoken model: {e}")
                self.use_tiktoken = False
        
        # Load custom vocabulary if provided
        elif vocab_file and merges_file:
            self._load_vocab(vocab_file)
            self._load_merges(merges_file)
            print(f"Loaded custom BPE vocabulary with {len(self.token_to_id)} tokens")
        else:
            # Set up a basic byte-level vocabulary as fallback
            self._setup_byte_fallback()
            print("Using byte-level fallback vocabulary")
        
        # Setup encoder/decoder
        if not self.use_tiktoken:
            # Set up BPE encoder/decoder if not using tiktoken
            self._setup_bpe()
    
    def _setup_byte_fallback(self):
        """Set up a basic byte-level vocabulary as fallback"""
        # Start from special tokens
        next_id = len(self.special_tokens)
        
        # Add all possible bytes (0-255)
        for b in range(256):
            token = f"<byte_{b}>"
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
    
    def _load_vocab(self, vocab_file):
        """Load vocabulary from file"""
        # Start from special tokens
        next_id = len(self.special_tokens)
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Check for token:id format
                    if ':' in line:
                        token, token_id = line.split(':', 1)
                        token_id = int(token_id)
                    else:
                        token = line
                        token_id = next_id
                        next_id += 1
                    
                    # Skip existing and exceeding tokens
                    if token not in self.token_to_id and token_id < self.max_token_value:
                        self.token_to_id[token] = token_id
                        self.id_to_token[token_id] = token
    
    def _load_merges(self, merges_file):
        """Load BPE merges from file"""
        self.bpe_ranks = {}
        
        with open(merges_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Skip version header
                    if i == 0 and line.startswith('#version'):
                        continue
                        
                    # Parse merge rule: token1 token2
                    # E.g., "t h" means merge 't' and 'h' into 'th'
                    try:
                        first, second = line.split()
                        self.bpe_ranks[(first, second)] = i
                    except:
                        print(f"Error parsing BPE merge rule: {line}")
    
    def _setup_bpe(self):
        """Set up the BPE encoder/decoder"""
        self.cache = {}  # Cache for BPE encoding
        
        # Byte encoder/decoder for handling UTF-8
        self.byte_encoder = {i: bytes([i]).decode('latin1') for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def _bpe_encode(self, text):
        """BPE encoding algorithm"""
        if text in self.cache:
            return self.cache[text]
            
        # Convert text to bytes then to tokens
        tokens = [self.byte_encoder[b] for b in text.encode('utf-8')]
        
        # Basic BPE merge operations
        result = []
        while tokens:
            token = tokens.pop(0)
            result.append(token)
            
        # Cache the result
        self.cache[text] = result
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
            # Use tiktoken if available
            if self.use_tiktoken:
                ids = self.tiktoken_encoder.encode(text)
                
                # Map tiktoken ids to our ids (handle offset)
                offset = len(self.special_tokens)
                ids = [id + offset for id in ids]
            else:
                # Use our BPE implementation
                tokens = self._bpe_encode(text)
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
            
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids 
                        if token_id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        
        # Use tiktoken for decoding if available
        if self.use_tiktoken:
            # Convert our ids back to tiktoken ids
            offset = len(self.special_tokens)
            tiktoken_ids = [id - offset for id in token_ids if id >= offset]
            
            # Decode with tiktoken
            try:
                text = self.tiktoken_encoder.decode(tiktoken_ids)
                return text
            except Exception as e:
                print(f"Tiktoken decoding error: {e}. Falling back to manual decoding.")
        
        # Manual decoding
        tokens = [self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids]
        
        # Combine tokens and handle byte tokens
        text = ""
        byte_buffer = []
        
        for token in tokens:
            if token.startswith("<byte_") and token.endswith(">"):
                try:
                    byte_value = int(token[6:-1])
                    byte_buffer.append(byte_value)
                except ValueError:
                    # If parsing fails, treat as normal token
                    if byte_buffer:
                        # Flush byte buffer first
                        text += bytes(byte_buffer).decode('utf-8', errors='replace')
                        byte_buffer = []
                    text += token
            else:
                # Regular token - first flush any byte buffer
                if byte_buffer:
                    text += bytes(byte_buffer).decode('utf-8', errors='replace')
                    byte_buffer = []
                text += token
                
        # Handle any remaining bytes in buffer
        if byte_buffer:
            text += bytes(byte_buffer).decode('utf-8', errors='replace')
            
        return text
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.token_to_id)

# --- Backward compatibility ---
SimpleTokenizer = TinyBPETokenizer  # For backward compatibility

def create_test_data(seq_length=32, batch_size=2, vocab_size=100):
    """Create random data for testing"""
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    return inputs

class MiniConfig:
    """Configuration for mini TinyMamba model for testing"""
    d_model = 32
    n_layer = 2
    vocab_size = 100
    dropout = 0.0
    bias = False
    activation = 'silu'
    d_state = 4
    d_conv = 2
    expand_factor = 2
    window_size = 8
    num_heads = 2
    block_size = 32
    batch_size = 2
    use_flash_attn = False
    use_compile = False
    amp_dtype = torch.float32

def test_model_mini(verbose=True):
    """Test a mini version of TinyMamba model"""
    import time
    
    # Create mini config
    config = MiniConfig()
    
    if verbose:
        print("Creating mini TinyMamba model for testing...")
    
    # Create model
    model = TinyMambaModel(config)
    
    # Move to CPU
    device = torch.device("cpu")
    model = model.to(device)
    
    # Create test data
    inputs = create_test_data(
        seq_length=config.block_size, 
        batch_size=config.batch_size, 
        vocab_size=config.vocab_size
    )
    inputs = inputs.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(inputs)
        inference_time = time.time() - start_time
    
    # Print model stats
    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model created successfully with {num_params:,} parameters")
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Inference time: {inference_time:.4f} seconds on CPU")
        
    # Validate output shapes
    assert outputs.shape == (config.batch_size, config.block_size, config.vocab_size), \
        f"Expected output shape {(config.batch_size, config.block_size, config.vocab_size)}, got {outputs.shape}"
    
    # Try autoregressive generation
    if verbose:
        print("Testing autoregressive generation...")
    
    # Test generation
    model.eval()
    
    with torch.no_grad():
        # Start with just first token of each sequence
        prompt = inputs[:, :1]
        generated = prompt.clone()
        
        # Generate sequence
        start_time = time.time()
        for i in range(1, min(16, config.block_size)):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        generation_time = time.time() - start_time
    
    if verbose:
        print(f"Generated shape: {generated.shape}")
        print(f"Generation time: {generation_time:.4f} seconds on CPU")
        print("Test passed!")
    
    return model, generated

def run_all_tests():
    """Run all unit tests"""
    print("Running TinyMamba unit tests...")
    test_model_mini()
    print("All tests passed!")

def main(rank, world_size):
    """Main training function for both single-GPU and DDP training"""
    # Set global rank for logging
    global global_rank
    global_rank = rank
    
    # Initialize distributed environment if using DDP
    if world_size > 1:
        setup(rank, world_size)
        
    # Set up device and precision
    device = setup_hardware_precision(rank)
    
    # Only log from the first process in DDP
    if rank == 0:
        print(f"Using device: {device}")
        print(f"Using AMP dtype: {config.amp_dtype}")
        
        if hasattr(torch.cuda, 'memory_summary') and device.type == 'cuda':
            print(f"CUDA memory summary before model creation:\n{torch.cuda.memory_summary()}")
    
    # Create datasets
    if rank == 0:
        print(f"Loading training data from {config.train_data_path}")
        
    train_dataset = StreamingTokenDataset(
        data_sources=[config.train_data_path],
        block_size=config.block_size,
        shuffle=True
    )
    
    # Create tokenizer
    tokenizer = None
    if rank == 0 and config.generate_samples:
        if TIKTOKEN_AVAILABLE:
            tokenizer = TinyBPETokenizer(tiktoken_model="gpt2")
        else:
            tokenizer = TinyBPETokenizer()  # Fallback to custom BPE implementation
    
    # Create model
    if rank == 0:
        print("Creating TinyMamba model...")
        
    model = TinyMambaModel(config)
    
    # Move model to device
    model = model.to(device)
    
    # Apply torch.compile if enabled
    if config.use_compile and hasattr(torch, 'compile'):
        if rank == 0:
            print("Using torch.compile for model acceleration")
        model = torch.compile(model)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Set up optimizer with parameter groups
    # Create parameter groups for weight decay following best practices
    decay_params = []
    nodecay_params = []
    emb_params = []
    
    # Check if model is wrapped in DDP
    model_to_process = model.module if isinstance(model, DDP) else model
    
    for name, param in model_to_process.named_parameters():
        if 'embedding' in name:
            emb_params.append(param)
        elif param.ndim < 2 or 'ln' in name or 'bias' in name or 'norm' in name:
            # Skip weight decay for biases, layernorms, and 1D params
            nodecay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
        {'params': emb_params, 'weight_decay': 0.0, 'lr': config.lr * 0.5}  # Lower LR for embeddings
    ]
    
    # Create optimizer
    if world_size > 1:
        # Use ZeroRedundancyOptimizer for distributed training to save memory
        optimizer = ZeroRedundancyOptimizer(
            optim_groups,
            optimizer_class=torch.optim.AdamW,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            fused=torch.cuda.is_available()  # Use fused implementation if available
        )
    else:
        # Use standard AdamW for single GPU/CPU training
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            fused=torch.cuda.is_available()  # Use fused implementation if available
        )
    
    # Set up learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=config.warmup_steps,
        max_steps=len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_epochs
    )
    
    # Set up gradient scaler for AMP
    scaler = GradScaler()
    
    # Set up logging
    logger = None
    if rank == 0:
        logger = MetricsLogger(use_tensorboard=TENSORBOARD_AVAILABLE)
        
        if config.use_wandb and WANDB_AVAILABLE:
            import wandb
            wandb_config = {k: v for k, v in vars(config).items() if not k.startswith('__')}
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=wandb_config
            )
    
    # Set up early stopping
    early_stopping = None
    if config.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
    
    # Set up profiler if on main process
    profiler = None
    if rank == 0:
        profiler = ModelProfiler(
            model if not isinstance(model, DDP) else model.module,
            sample_input_size=(config.batch_size, config.block_size),
            device=device
        )
        profiler.log_summary(logger)
    
    # Resume from checkpoint if enabled
    global_step = 0
    start_epoch = 0
    
    if config.resume and os.path.exists(config.checkpoint_path):
        if rank == 0:
            print(f"Loading checkpoint from {config.checkpoint_path}")
            
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        
        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
            
        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        # Restore training state
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, global step {global_step}")
    
    # Create validation data loader
    val_loader = None
    if config.enable_validation:
        if rank == 0:
            print(f"Loading validation data from {config.val_data_path}")
            
        val_data = StreamingTokenDataset(
            data_sources=[config.val_data_path],
            block_size=config.block_size,
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            num_workers=1
        )
    
    # Setup text generator with tokenizer
    if config.generate_samples and rank == 0 and tokenizer:
        text_generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            max_length=config.generation_length,
            temperature=config.generation_temperature,
            device=device
        )
    else:
        text_generator = None
    
    # Training state tracking
    best_val_loss = float('inf')
    training_start_time = time.time()
    samples_processed = 0
    
    # Main training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Set up train loader for this epoch
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        if rank == 0:
            print(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            
        # Train for one epoch
        try:
            # Train step
            model.train()
            running_loss = 0.0
            steps_since_accumulation = 0
            
            for step, batch in enumerate(train_loader):
                # Skip steps for other processes in DDP mode
                if step % world_size != rank:
                    continue
                
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass with mixed precision
                with autocast(dtype=config.amp_dtype):
                    # Get model predictions
                    logits = model(batch)
                    
                    # Prepare targets (shifted right for next token prediction)
                    targets = batch[:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                    
                    # Calculate loss
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1)
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update metrics
                running_loss += loss.item() * config.gradient_accumulation_steps
                samples_processed += batch.size(0)
                steps_since_accumulation += 1
                
                # Gradient accumulation
                if steps_since_accumulation == config.gradient_accumulation_steps:
                    # Gradient clipping
                    if config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.grad_clip
                        )
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Update learning rate
                    if scheduler:
                        scheduler.step()
                    
                    # Reset accumulation counter
                    steps_since_accumulation = 0
                    
                    # Update global step
                    global_step += 1
                    
                    # Log metrics
                    if rank == 0 and global_step % config.log_interval == 0:
                        # Calculate metrics
                        avg_loss = running_loss / config.log_interval
                        lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                        
                        # Calculate throughput
                        elapsed = time.time() - training_start_time
                        throughput = samples_processed / elapsed
                        
                        # Log to console
                        print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                              f"Throughput: {throughput:.2f} samples/sec")
                        
                        # Log to metrics tracker
                        if logger:
                            logger.log_train_metrics(
                                step=global_step,
                                epoch=epoch,
                                loss=avg_loss,
                                lr=lr,
                                throughput=throughput
                            )
                        
                        # Reset metrics
                        running_loss = 0.0
                        samples_processed = 0
                        training_start_time = time.time()
                    
                    # Run validation
                    if val_loader and config.enable_validation and global_step % config.validation_interval == 0:
                        val_loss, val_ppl = validate(model, val_loader, device, config.validation_steps)
                        
                        if rank == 0:
                            print(f"Validation: Loss={val_loss:.4f}, PPL={val_ppl:.2f}")
                            
                            # Log validation metrics
                            if logger:
                                logger.log_valid_metrics(
                                    step=global_step,
                                    epoch=epoch,
                                    loss=val_loss,
                                    ppl=val_ppl
                                )
                            
                            # Check for best model and save checkpoint
                            is_best = val_loss < best_val_loss
                            if is_best:
                                best_val_loss = val_loss
                                print(f"New best validation loss: {best_val_loss:.4f}")
                            
                            # Save checkpoint
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                val_loss=val_loss,
                                config=config,
                                is_best=is_best
                            )
                            
                            # Check early stopping
                            if early_stopping and early_stopping(val_loss):
                                print(f"Early stopping triggered after {global_step} steps")
                                return
                    
                    # Generate text samples
                    if rank == 0 and text_generator and config.generate_samples and global_step % config.generation_interval == 0:
                        generate_and_log_samples(
                            generator=text_generator,
                            logger=logger,
                            global_step=global_step
                        )
                    
                    # Run profiling
                    if rank == 0 and profiler and global_step % 5000 == 0:
                        profiler.profile_forward()
                        profiler.log_summary(logger, global_step)
                    
                    # Memory management - empty cache periodically
                    if device.type == 'cuda' and global_step % config.empty_cache_freq == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
        except Exception as e:
            if rank == 0:
                print(f"Error during training: {e}")
                if logger:
                    logger.close()
            raise
            
        # Save epoch checkpoint
        if rank == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                val_loss=best_val_loss,
                config=config,
                is_best=False
            )
    
    # Final validation
    if config.enable_validation and rank == 0:
        final_val_loss, final_val_ppl = validate(model, val_loader, device, config.validation_steps)
        
        print(f"Final validation loss: {final_val_loss:.4f}, perplexity: {final_val_ppl:.4f}")
    
    # Generate final samples
    if config.generate_samples and rank == 0 and text_generator:
        generate_and_log_samples(
            generator=text_generator,
            logger=logger,
            global_step=global_step
        )
    
    # Cleanup
    if rank == 0 and logger:
        logger.close()
        
    if rank == 0 and profiler:
        profiler.remove_hooks()
        
    # Clean up distributed environment
    if world_size > 1:
        cleanup()
        
    if rank == 0:
        print("Training complete!")


@torch.no_grad()
def validate(model, val_loader, device, max_steps=None):
    """Run validation loop and return average loss"""
    # Switch to eval mode
    model.eval()
    
    total_loss = 0.0
    total_steps = 0
    total_tokens = 0
    
    for step, batch in enumerate(val_loader):
        # Stop after max_steps if provided
        if max_steps is not None and step >= max_steps:
            break
            
        # Move batch to device
        batch = batch.to(device)
        
        # Get targets (shifted right)
        targets = batch[:, 1:].contiguous()
        
        # Forward pass
        with autocast(dtype=torch.float16):
            logits = model(batch)
            # Trim logits to match targets length
            logits = logits[:, :-1, :].contiguous()
            
            # Calculate loss
            B, T = targets.size()
            total_tokens += B * T
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'  # Use sum for accurate token-level loss
            )
            total_loss += loss.item()
        
        total_steps += 1
    
    # Switch back to training mode
    model.train()
    
    # Calculate average token-level loss and perplexity
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, config, is_best=False, filename=None):
    """Save model checkpoint"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    # Get model state dict (handle DDP case)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Create checkpoint data
    checkpoint = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'val_loss': val_loss,
        'config': {k: v for k, v in vars(config).items() if not k.startswith('__')},
    }
    
    # Determine checkpoint path
    if filename:
        checkpoint_path = os.path.join(os.path.dirname(config.checkpoint_path), filename)
    else:
        checkpoint_path = os.path.join(
            os.path.dirname(config.checkpoint_path),
            f"tinymamba_step_{global_step:07d}.pt"
        )
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join(os.path.dirname(config.checkpoint_path), "tinymamba_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model checkpoint to {best_path}")
    
    # Save latest checkpoint (overwrite existing)
    torch.save(checkpoint, config.checkpoint_path)
    
    return checkpoint_path


def generate_and_log_samples(generator, logger, global_step, prompts=None):
    """Generate and log text samples"""
    if generator is None:
        return
        
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the distant future",
            "The best way to learn is"
        ]
    
    # Generate samples
    model = generator.model
    model.eval()
    
    samples = []
    with torch.no_grad():
        for prompt in prompts:
            try:
                output = generator.generate(prompt_text=prompt)
                samples.append({
                    "prompt": prompt,
                    "generated": output
                })
                
                # Log to tensorboard/etc.
                if logger:
                    logger.log_generated_text(
                        global_step,
                        f"Prompt: {prompt}\n\n{output}",
                        tag=f"generation/{prompt[:10]}"
                    )
            except Exception as e:
                print(f"Error generating from prompt '{prompt}': {e}")
    
    # Print a sample to console
    if samples and logger:
        print(f"\n--- Generated Sample at Step {global_step} ---")
        print(f"Prompt: {samples[0]['prompt']}")
        print(f"Generated: {samples[0]['generated'][:200]}...")
        print("---------------------------------------------\n")
    
    model.train()
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TinyMamba training and testing script")
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--world-size', type=int, default=None, help='Number of processes for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    if args.test:
        run_all_tests()
    else:
        # Check if data files exist before starting
        if not os.path.exists(config.train_data_path):
            print(f"ERROR: Training data not found at {config.train_data_path}")
            print("Please ensure data is prepared and paths in Config are correct.")
            exit(1)  # Exit if data is missing
        
        # Choose training approach based on arguments
        if args.distributed:
            # For distributed training
            world_size = args.world_size or torch.cuda.device_count()
            if world_size > 1:
                # Launch with torch.distributed.launch
                main(args.local_rank, world_size)
            else:
                print("Warning: Distributed mode requested but only one GPU found. Falling back to single GPU training.")
                main(0, 1)
        else:
            # For single GPU/CPU training
            main(0, 1)