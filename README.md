# TinyMamba

A hybrid state space model and attention-based architecture that combines elements from Mamba and Transformer models with additional performance optimizations.

## Features

- **Hybrid Architecture**: Combines efficient state space models (SSM) with selective attention.
- **Advanced Components**:
  - Token-adaptive gating mechanism for dynamic routing
  - Rotary Position Embeddings (RoPE) for better positional awareness
  - RMSNorm for improved normalization
  - Parallel scan implementation for faster training
  - DeepNet scaling for stable training of deeper networks
  - Low-rank projections for parameter efficiency
- **Training Infrastructure**:
  - Multi-dataset support with weighted sampling
  - Efficient streaming and memory management
  - Validation, early stopping, and checkpoint management
  - Text generation for quality assessment during training
  - Support for logging to TensorBoard and Weights & Biases
  - Model profiling and parameter counting
- **Export & Deployment**:
  - Export to PyTorch, TorchScript, and ONNX formats
  - Optimization for inference
  - Simple tokenization support
- **Advanced Features**:
  - Parameter-efficient fine-tuning (LoRA, Prefix-tuning)
  - Streaming inference with proper state management
  - Quantization support for efficient deployment
  - Integration with HuggingFace Transformers ecosystem

## Installation

```bash
# Clone the repository
git clone https://github.com/NimraProject/tinymamba.git
cd tinymamba

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio numpy tqdm tensorboard
```

Optional dependencies:
```bash
# For Weights & Biases support
pip install wandb

# For ONNX export
pip install onnx onnxruntime
```

## Preparing Training Data

TinyMamba expects tokenized data in a binary format (int32 tokens).

1. Tokenize your text data using your preferred tokenizer.
2. Save the tokens as a binary file in int32 format.
3. (Optional) Create a `vocab.txt` file (one token per line) for text generation.

Example:
```python
import numpy as np
from transformers import AutoTokenizer

# Tokenize text
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("<your text content>")

# Save as int32 binary
token_array = np.array(tokens, dtype=np.int32)
token_array.tofile("data/text/train.bin")

# Save validation data similarly
val_tokens = tokenizer.encode("<your validation text>")
np.array(val_tokens, dtype=np.int32).tofile("data/text/val.bin")

# Save vocabulary (optional)
with open("data/text/vocab.txt", "w") as f:
    for token, id in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
        f.write(f"{token}\n")
```

## Training

### Basic Training

```bash
python testv2.py
```

### Distributed Training (Multiple GPUs)

```bash
python -m torch.distributed.launch --nproc_per_node=2 testv2.py
```

### Configuration

Edit the `Config` class in `testv2.py` to customize training parameters:

```python
class Config:
    # Model parameters
    d_model = 768         # Model dimension
    n_layer = 12          # Number of layers
    vocab_size = 50304    # Vocabulary size
    dropout = 0.1         # Dropout rate
    bias = True           # Whether to use bias

    # Architecture parameters
    activation = 'silu'   # Activation function
    d_state = 16          # SSM state size
    d_conv = 4            # Convolution size
    expand_factor = 2     # Expansion factor for MLP
    window_size = 64      # Local attention window size
    num_heads = 6         # Number of attention heads

    # Training parameters
    block_size = 512      # Context length
    batch_size = 8        # Batch size
    gradient_accumulation_steps = 4  # Accumulation steps
    num_epochs = 1        # Number of epochs
    lr = 6e-4             # Learning rate
    warmup_steps = 1000   # Warmup steps
    grad_clip = 1.0       # Gradient clipping
    weight_decay = 0.01   # Weight decay
    beta1 = 0.9           # Adam beta1
    beta2 = 0.95          # Adam beta2

    # Optimization options
    use_compile = True    # Use torch.compile
    use_flash_attn = True # Use flash attention
    amp_dtype = torch.bfloat16  # Mixed precision type

    # Logging and checkpointing
    log_interval = 20     # Log every N steps
    empty_cache_freq = 50 # Clear CUDA cache every N steps
    train_data_path = './data/text/train.bin'
    val_data_path = './data/text/val.bin'
    checkpoint_path = 'tinymamba_model_latest.pt'
    resume = False        # Resume from checkpoint
    
    # Validation and generation settings
    enable_validation = True
    validation_interval = 2000
    generate_samples = True
    generation_interval = 1000
    
    # Logging options
    use_wandb = False
    wandb_project = "tinymamba"
```

## Exporting a Trained Model

After training, you can export your model using the export_model.py script:

```bash
python export_model.py --checkpoint ./tinymamba_model_latest.pt --format all --verify --generate
```

Options:
- `--checkpoint`: Path to the checkpoint file
- `--output_dir`: Directory to save exported models
- `--format`: Export format (pytorch, torchscript, onnx, or all)
- `--optimize`: Apply optimization passes
- `--verify`: Run verification with random inputs
- `--generate`: Generate sample text
- `--prompt`: Custom prompt for text generation

## Model Architecture

TinyMamba uses a hybrid architecture:

1. **Token Embeddings**: Standard learned embeddings
2. **TinyMamba Blocks**: Each block contains:
   - Token-adaptive branch gating
   - State Space Model (SSM) branch
   - Local Attention branch with RoPE
   - MLP branch with SwiGLU activation
   - Residual connections with learned scaling
3. **Output Layer**: Linear projection to vocabulary size

### TokenAdaptiveGating

Instead of using fixed weights for different branches, TinyMamba uses a token-adaptive controller that predicts branch weights based on the content:

```
Token Content → Lightweight Controller → Dynamic Branch Weights
```

This allows the model to adaptively route information through different pathways (SSM, attention, MLP) based on token content.

## Parameter-Efficient Fine-Tuning (PEFT)

TinyMamba supports parameter-efficient fine-tuning methods to adapt pre-trained models to specific tasks with minimal additional parameters:

### LoRA (Low-Rank Adaptation)

```bash
python peft.py --checkpoint ./tinymamba_model_latest.pt --method lora --lora_rank 8 --lora_alpha 16.0 --output_dir ./lora_model
```

### Prefix Tuning

```bash
python peft.py --checkpoint ./tinymamba_model_latest.pt --method prefix --prefix_length 16 --output_dir ./prefix_model
```

## Streaming Inference

TinyMamba provides optimized streaming inference with proper state management for continuous text generation:

```bash
python streaming.py --checkpoint ./tinymamba_model_latest.pt --prompt "Once upon a time" --max_tokens 100 --streaming
```

### Multi-Session Support

The streaming server supports multiple concurrent sessions with independent state management:

```python
from streaming import StreamingServer, StreamingConfig

# Create server
config = StreamingConfig()
config.temperature = 0.8
config.max_new_tokens = 50

server = StreamingServer(
    model_path="./tinymamba_model_latest.pt",
    device="cuda"
)

# Create sessions
session1 = server.create_session()
session2 = server.create_session()

# Generate text in different sessions (with different contexts)
text1 = server.generate_text("Hello, my name is Alice.", session_id=session1)
text2 = server.generate_text("The weather today is", session_id=session2)

# Continue generation in a session (maintains state)
text1_continued = server.generate_text("I live in", session_id=session1)
```

## Customization

### Adding a Custom Tokenizer

Replace the SimpleTokenizer with your custom tokenizer by implementing encode/decode methods:

```python
from transformers import AutoTokenizer

class HuggingFaceTokenizerWrapper:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        tokens = self.tokenizer.encode(
            text, 
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors
        )
        return tokens
        
    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
    def __len__(self):
        return len(self.tokenizer)
```

### Using Multiple Datasets

```python
# Create datasets with weighting
train_datasets = [
    './data/text/dataset1.bin',  # General text
    './data/text/dataset2.bin',  # Code
    './data/text/dataset3.bin'   # Math
]
weights = [0.6, 0.2, 0.2]  # 60% general, 20% code, 20% math

# Create dataset
train_dataset = StreamingTokenDataset(
    train_datasets,
    block_size=config.block_size,
    weights=weights,
    shuffle=True
)
```

## Logging and Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=./logs
```

### Weights & Biases

Enable W&B logging by setting `use_wandb = True` in Config.

## TODOs

- [x] Add proper error handling for RoPE embeddings
- [x] Implement support for quantized models (int8, int4)
- [x] Create benchmarking scripts for performance measurement
- [x] Integrate with HuggingFace's transformers library
- [x] Implement parameter-efficient fine-tuning methods (LoRA, Prefix-tuning)
- [x] Add streaming inference support with proper state management

# For ONNX export

TinyMamba models can be exported to ONNX format for deployment:

```bash
python export_model.py --checkpoint ./tinymamba_model_latest.pt --format onnx --output_dir ./onnx_models
```

## License

MIT 

## TODO

The following items are planned for future development:

- ✅ Implement better error handling in RoPE embeddings to handle dimension mismatches
- ✅ Add support for quantized models (INT8, INT4) for more efficient deployment
- ✅ Create benchmarking scripts to compare against standard Transformer models
- ✅ Integrate with HuggingFace's transformers library for better ecosystem compatibility
- Implement parameter-efficient fine-tuning methods (LoRA, Prefix-tuning)
- Add streaming inference support with proper state management

## Next Steps

Immediate next development priorities:

1. **Robustness Improvements**:
   - Fix dimension handling in rotary embeddings implementation
   - Add more comprehensive test suite for all components
   - Implement better error reporting during training

2. **Performance Optimizations**:
   - Profile and optimize the SSM computation kernels
   - Reduce memory usage during training
   - Investigate faster attention implementations

3. **Deployment & Integration**:
   - Create Docker containers for easy deployment
   - Add Python package installation support
   - Develop a simple API for inference
   - Integrate with popular serving frameworks 