"""
Streaming Inference for TinyMamba

This module provides efficient streaming inference capabilities for TinyMamba models
with proper state management to handle continuous text generation.

Features:
1. Memory-efficient inference for arbitrarily long contexts
2. Proper state management between inference steps
3. Support for batched and continuous generation
4. Utilities for token-by-token generation with state caching
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple, Any
import time
import json
from collections import deque

# Import TinyMamba model components
from testv2 import (
    TinyMambaModel, 
    Config, 
    TextGenerator,
    TinyBPETokenizer
)

class StreamingConfig:
    """Configuration for streaming inference"""
    
    # State management
    state_buffer_size = 128     # How many tokens of state to maintain
    reset_on_eos = True         # Whether to reset state on end of sequence token
    
    # Optimization
    optimize_memory = True      # Apply memory optimizations
    batch_size = 1              # Batch size for streaming inference
    max_new_tokens = 20         # Default max new tokens per generation step
    
    # Generation parameters
    temperature = 0.8           # Sampling temperature
    top_k = 40                  # Top-K sampling parameter
    top_p = 0.9                 # Top-P (nucleus) sampling parameter
    repetition_penalty = 1.1    # Repetition penalty
    
    # Special token IDs (will be set by tokenizer)
    pad_token_id = 0
    eos_token_id = None
    bos_token_id = None


class StreamingInference:
    """
    Streaming inference manager for TinyMamba models
    
    Handles efficient token-by-token generation with proper state management.
    
    Args:
        model: TinyMamba model for inference
        tokenizer: Tokenizer for text encoding/decoding
        config: StreamingConfig with inference parameters
        device: Device to run inference on (cpu/cuda)
    """
    def __init__(
        self,
        model: TinyMambaModel,
        tokenizer: TinyBPETokenizer,
        config: Optional[StreamingConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()
        self.device = torch.device(device)
        
        # Set special token IDs from tokenizer if available
        if hasattr(tokenizer, "pad_token_id"):
            self.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(tokenizer, "eos_token_id"):
            self.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, "bos_token_id"):
            self.config.bos_token_id = tokenizer.bos_token_id
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Enable streaming mode in model
        self._enable_streaming_mode()
        
        # Initialize state buffers
        self.states = {}  # Dict of session_id -> state
        self.token_buffers = {}  # Dict of session_id -> recent tokens deque
        
        # Generate a default session ID if none is provided
        self.default_session_id = "default"
        self.reset_state(self.default_session_id)
    
    def _enable_streaming_mode(self):
        """Enable streaming mode in the model and all components"""
        self.model.toggle_streaming(enabled=True)
    
    def reset_state(self, session_id: Optional[str] = None):
        """
        Reset the state for a specific session
        
        Args:
            session_id: ID of the session to reset (defaults to default session)
        """
        session_id = session_id or self.default_session_id
        
        # Clear hidden state
        self.states[session_id] = None
        
        # Reset token buffer
        self.token_buffers[session_id] = deque(maxlen=self.config.state_buffer_size)
        
        # Reset model states
        self.model.reset_state()
    
    def _prepare_input_ids(self, text_or_ids, session_id: Optional[str] = None):
        """
        Prepare input IDs from text or token IDs
        
        Args:
            text_or_ids: Text string or token IDs
            session_id: ID of the session
            
        Returns:
            Tensor of input IDs
        """
        session_id = session_id or self.default_session_id
        
        if isinstance(text_or_ids, str):
            # Convert text to token IDs
            input_ids = self.tokenizer.encode(text_or_ids, return_tensors="pt")
        elif isinstance(text_or_ids, list):
            # Convert list to tensor
            input_ids = torch.tensor(text_or_ids, dtype=torch.long)
        elif isinstance(text_or_ids, torch.Tensor):
            # Use tensor as is
            input_ids = text_or_ids
        else:
            raise ValueError(f"Unsupported input type: {type(text_or_ids)}")
        
        # Ensure input_ids is 2D (add batch dimension if needed)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Move to the correct device
        input_ids = input_ids.to(self.device)
        
        return input_ids
    
    def _apply_repetition_penalty(self, logits, input_ids, penalty=1.1):
        """Apply repetition penalty to logits based on generated tokens"""
        if penalty == 1.0:
            return logits
            
        for i in range(logits.size(0)):
            for token_id in set(input_ids[i].tolist()):
                # Penalize previously generated tokens
                logits[i, token_id] /= penalty
                
        return logits
    
    def _sample_token(self, logits):
        """Sample next token from logits using temperature, top-k, and top-p"""
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        # Apply top-k if specified
        if self.config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) sampling
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            
            # Shift indices to remove first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create binary mask
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, 
                index=sorted_indices, 
                src=sorted_indices_to_remove
            )
            
            # Set removed logits to -inf
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _forward_step(self, input_ids, session_id: Optional[str] = None):
        """
        Forward pass with state management
        
        Args:
            input_ids: Input token IDs
            session_id: ID of the session
            
        Returns:
            Model output logits
        """
        session_id = session_id or self.default_session_id
        
        # Get current state
        state = self.states.get(session_id)
        
        # Forward pass
        with torch.no_grad():
            # Process each token sequentially to maintain state
            B, L = input_ids.shape
            outputs = []
            
            for pos in range(L):
                token_slice = input_ids[:, pos].unsqueeze(1)  # [B, 1]
                output = self.model(token_slice, state=state)
                
                # Update state
                state = self.model.blocks[-1].state  # Get state from the last block
                
                # Store output
                outputs.append(output)
            
            # Concatenate outputs along sequence dimension
            logits = torch.cat(outputs, dim=1)
        
        # Save updated state
        self.states[session_id] = state
        
        return logits
    
    def _update_token_buffer(self, token_id, session_id: Optional[str] = None):
        """
        Update token buffer for a session
        
        Args:
            token_id: New token ID to add
            session_id: ID of the session
        """
        session_id = session_id or self.default_session_id
        
        # Ensure session has a token buffer
        if session_id not in self.token_buffers:
            self.token_buffers[session_id] = deque(maxlen=self.config.state_buffer_size)
        
        # Add token to buffer
        self.token_buffers[session_id].append(token_id.item())
        
        # Check for EOS token to potentially reset state
        if (self.config.reset_on_eos and 
            self.config.eos_token_id is not None and 
            token_id.item() == self.config.eos_token_id):
            self.reset_state(session_id)
    
    @torch.no_grad()
    def generate_next_token(self, context, session_id: Optional[str] = None):
        """
        Generate a single next token given context
        
        Args:
            context: Text or token IDs for context
            session_id: ID of the session
            
        Returns:
            Dict with next token info
        """
        session_id = session_id or self.default_session_id
        
        # Prepare input
        input_ids = self._prepare_input_ids(context, session_id)
        
        # Get logits
        logits = self._forward_step(input_ids, session_id)
        
        # Get last token logits
        next_token_logits = logits[:, -1, :]
        
        # Apply repetition penalty based on token buffer
        if session_id in self.token_buffers and len(self.token_buffers[session_id]) > 0:
            buffer_tokens = torch.tensor([list(self.token_buffers[session_id])], 
                                       device=self.device)
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, 
                buffer_tokens, 
                self.config.repetition_penalty
            )
        
        # Sample next token
        next_token = self._sample_token(next_token_logits)
        
        # Update token buffer
        self._update_token_buffer(next_token, session_id)
        
        # Decode token
        next_token_text = self.tokenizer.decode(next_token.cpu().numpy())
        
        return {
            "token_id": next_token.item(),
            "token_text": next_token_text,
            "logits": next_token_logits.cpu().numpy(),
        }
    
    @torch.no_grad()
    def generate_stream(
        self, 
        prompt, 
        max_new_tokens=None, 
        session_id=None,
        stop_on_eos=True
    ):
        """
        Generate a stream of tokens
        
        Args:
            prompt: Initial prompt text or tokens
            max_new_tokens: Maximum number of tokens to generate
            session_id: ID of the session
            stop_on_eos: Whether to stop generation on EOS token
            
        Yields:
            Generated token text
        """
        session_id = session_id or self.default_session_id
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Ensure we have a session
        if session_id not in self.states:
            self.reset_state(session_id)
        
        # Process initial prompt if provided
        if prompt:
            input_ids = self._prepare_input_ids(prompt, session_id)
            _ = self._forward_step(input_ids, session_id)
            
            # Update token buffer with prompt tokens
            for token in input_ids[0]:
                self._update_token_buffer(token, session_id)
            
            # Yield the prompt first
            yield self.tokenizer.decode(input_ids[0].cpu().numpy())
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Use an empty token since the state contains the context
            input_ids = torch.tensor([[self.config.pad_token_id]], device=self.device)
            
            # Get next token logits
            logits = self._forward_step(input_ids, session_id)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if session_id in self.token_buffers and len(self.token_buffers[session_id]) > 0:
                buffer_tokens = torch.tensor([list(self.token_buffers[session_id])], 
                                          device=self.device)
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, 
                    buffer_tokens, 
                    self.config.repetition_penalty
                )
            
            # Sample next token
            next_token = self._sample_token(next_token_logits)
            
            # Update token buffer
            self._update_token_buffer(next_token, session_id)
            
            # Decode and yield token
            next_token_text = self.tokenizer.decode(next_token.cpu().numpy())
            yield next_token_text
            
            # Check for end of sequence
            if (stop_on_eos and 
                self.config.eos_token_id is not None and 
                next_token.item() == self.config.eos_token_id):
                break
    
    def generate_text(
        self, 
        prompt, 
        max_new_tokens=None, 
        session_id=None,
        streaming=False
    ):
        """
        Generate text from a prompt
        
        Args:
            prompt: Initial prompt text or tokens
            max_new_tokens: Maximum number of tokens to generate
            session_id: ID of the session
            streaming: Whether to return a streaming generator
            
        Returns:
            Generated text or a generator yielding token-by-token text
        """
        if streaming:
            return self.generate_stream(prompt, max_new_tokens, session_id)
        
        # Non-streaming mode concatenates all tokens
        session_id = session_id or self.default_session_id
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Process prompt
        input_ids = self._prepare_input_ids(prompt, session_id)
        
        # Generate output sequence
        generated_ids = []
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self._forward_step(current_ids, session_id)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            full_sequence = torch.cat([input_ids, torch.tensor(generated_ids, 
                                                            device=self.device).unsqueeze(0)], dim=1)
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, 
                full_sequence, 
                self.config.repetition_penalty
            )
            
            # Sample next token
            next_token = self._sample_token(next_token_logits)
            
            # Update state
            self._update_token_buffer(next_token, session_id)
            
            # Add to generated sequence
            generated_ids.append(next_token.item())
            
            # Use only the new token for next iteration
            current_ids = next_token.unsqueeze(0)
            
            # Check for EOS
            if (self.config.eos_token_id is not None and 
                next_token.item() == self.config.eos_token_id):
                break
        
        # Combine input with generated ids
        all_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=self.device)], dim=1)
        
        # Decode the complete sequence
        text = self.tokenizer.decode(all_ids[0].cpu().numpy())
        return text
    
    def get_state_info(self, session_id=None):
        """
        Get information about current state
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dict with state information
        """
        session_id = session_id or self.default_session_id
        
        # Get state for session
        state = self.states.get(session_id)
        token_buffer = list(self.token_buffers.get(session_id, []))
        
        # Prepare info
        info = {
            "session_id": session_id,
            "has_state": state is not None,
            "recent_tokens": len(token_buffer),
            "recent_text": self.tokenizer.decode(token_buffer) if token_buffer else "",
        }
        
        return info


class StreamingServer:
    """
    Simple server for TinyMamba streaming inference
    
    Provides a stateful interface for streaming text generation across multiple sessions.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer files (optional)
        device: Device to run inference on
        config: StreamingConfig with inference parameters
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[StreamingConfig] = None
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.config = config or StreamingConfig()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Create streaming inference engine
        self.engine = StreamingInference(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device
        )
        
        # Active sessions
        self.sessions = {}
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from paths"""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")
        
        # Extract config
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            model_config = Config()
            for key, value in config_dict.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        else:
            print("Warning: Using default config")
            model_config = Config()
        
        # Create model
        model = TinyMambaModel(model_config)
        
        # Load weights
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load tokenizer
        tokenizer = TinyBPETokenizer(vocab_file=self.tokenizer_path)
        
        return model, tokenizer
    
    def create_session(self, session_id=None):
        """
        Create a new session
        
        Args:
            session_id: Optional session ID (generated if None)
            
        Returns:
            Session ID
        """
        # Generate a unique session ID if none provided
        if session_id is None:
            session_id = f"session_{int(time.time())}_{len(self.sessions)}"
        
        # Initialize session
        self.engine.reset_state(session_id)
        
        # Store session
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "tokens_generated": 0
        }
        
        return session_id
    
    def generate_stream(self, prompt, session_id=None, max_new_tokens=None):
        """
        Generate a stream of text
        
        Args:
            prompt: Initial prompt
            session_id: Session ID (creates new session if None)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generator yielding text chunks
        """
        # Create new session if needed
        if session_id is None or session_id not in self.sessions:
            session_id = self.create_session(session_id)
        
        # Update session stats
        self.sessions[session_id]["last_active"] = time.time()
        
        # Generate stream
        return self.engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            session_id=session_id
        )
    
    def generate_text(self, prompt, session_id=None, max_new_tokens=None):
        """
        Generate text (non-streaming)
        
        Args:
            prompt: Initial prompt
            session_id: Session ID (creates new session if None)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Create new session if needed
        if session_id is None or session_id not in self.sessions:
            session_id = self.create_session(session_id)
        
        # Update session stats
        self.sessions[session_id]["last_active"] = time.time()
        
        # Generate text
        text = self.engine.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            session_id=session_id,
            streaming=False
        )
        
        # Update token count (estimated)
        if isinstance(prompt, str):
            prompt_tokens = len(self.tokenizer.encode(prompt))
        else:
            prompt_tokens = len(prompt) if isinstance(prompt, list) else prompt.shape[1]
        
        output_tokens = len(self.tokenizer.encode(text)) - prompt_tokens
        self.sessions[session_id]["tokens_generated"] += max(0, output_tokens)
        
        return text
    
    def get_session_info(self, session_id=None):
        """
        Get information about a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict with session information
        """
        if session_id is None:
            # Return info about all sessions
            return {
                "sessions": list(self.sessions.keys()),
                "total_sessions": len(self.sessions),
                "active_sessions": sum(1 for s in self.sessions.values() 
                                     if time.time() - s["last_active"] < 300)
            }
        
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        # Get basic session info
        session_info = self.sessions[session_id].copy()
        
        # Add state info
        state_info = self.engine.get_state_info(session_id)
        session_info.update(state_info)
        
        # Add time stats
        session_info["age_seconds"] = time.time() - session_info["created_at"]
        session_info["idle_seconds"] = time.time() - session_info["last_active"]
        
        return session_info
    
    def reset_session(self, session_id):
        """
        Reset a session's state
        
        Args:
            session_id: Session ID
            
        Returns:
            Success indicator
        """
        if session_id not in self.sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        # Reset state
        self.engine.reset_state(session_id)
        
        # Update session stats
        self.sessions[session_id]["last_active"] = time.time()
        
        return {"success": True}
    
    def delete_session(self, session_id):
        """
        Delete a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Success indicator
        """
        if session_id not in self.sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        # Delete session
        del self.sessions[session_id]
        
        # Clean up engine state
        if session_id in self.engine.states:
            del self.engine.states[session_id]
        
        if session_id in self.engine.token_buffers:
            del self.engine.token_buffers[session_id]
        
        return {"success": True}
    
    def cleanup_old_sessions(self, max_age_seconds=3600):
        """
        Clean up old inactive sessions
        
        Args:
            max_age_seconds: Maximum session age in seconds
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        sessions_to_delete = []
        
        # Find old sessions
        for session_id, info in self.sessions.items():
            if current_time - info["last_active"] > max_age_seconds:
                sessions_to_delete.append(session_id)
        
        # Delete sessions
        for session_id in sessions_to_delete:
            self.delete_session(session_id)
        
        return len(sessions_to_delete)


if __name__ == "__main__":
    # Example usage
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Streaming Inference for TinyMamba")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--streaming", action="store_true", help="Use token-by-token streaming")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()
    
    # Create config
    config = StreamingConfig()
    config.temperature = args.temperature
    config.max_new_tokens = args.max_tokens
    
    # Create streaming server
    server = StreamingServer(
        model_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        config=config
    )
    
    # Generate text
    if args.streaming:
        print(f"Generating streaming output for: {args.prompt}")
        print("-" * 40)
        print(args.prompt, end="", flush=True)
        
        for token in server.generate_stream(args.prompt):
            print(token, end="", flush=True)
            time.sleep(0.01)  # Simulate realistic typing speed
        
        print("\n" + "-" * 40)
    else:
        print(f"Generating text for: {args.prompt}")
        print("-" * 40)
        
        start_time = time.time()
        text = server.generate_text(args.prompt)
        end_time = time.time()
        
        print(text)
        print("-" * 40)
        print(f"Generated {len(text) - len(args.prompt)} characters in {end_time - start_time:.2f} seconds") 