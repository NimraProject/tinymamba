import os
import requests
import tiktoken
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def download_tinystories():
    """
    Download the TinyStories dataset from HuggingFace and save to ./data directory
    """
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    train_file = "./data/train.txt"
    val_file = "./data/val.txt"
    
    # Check if files already exist
    if os.path.exists(train_file) and os.path.exists(val_file):
        print("Dataset files already exist. Skipping download.")
        return train_file, val_file
    
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Save train and validation sets
    print("Saving training set...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset["train"]):
            f.write(item["text"] + "\n\n")
    
    print("Saving validation set...")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset["validation"]):
            f.write(item["text"] + "\n\n")
    
    print("Dataset saved to ./data directory")
    
    return train_file, val_file

def tokenize_chunk(args):
    """Helper function for parallel tokenization"""
    chunk, enc, chunk_id = args
    tokens = enc.encode(chunk)
    return chunk_id, tokens

def tokenize_data(file_path):
    """
    Tokenize a text file using GPT-2 tiktoken tokenizer with multithreading
    """
    token_file = file_path.replace(".txt", "_tokens.txt")
    
    # Check if tokenized file already exists
    if os.path.exists(token_file):
        print(f"Tokenized file {token_file} already exists. Skipping tokenization.")
        return token_file, []
    
    # Load the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"Tokenizing {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Prepare for parallel processing
    # Split the text into chunks for parallel processing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(text) // num_cores
    chunks = []
    
    for i in range(num_cores):
        start = i * chunk_size
        end = None if i == num_cores - 1 else (i + 1) * chunk_size
        chunks.append((text[start:end], enc, i))
    
    # Tokenize in parallel
    all_tokens = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(tokenize_chunk, chunks),
            total=len(chunks),
            desc="Tokenizing chunks"
        ))
    
    # Sort results by chunk_id and combine tokens
    results.sort(key=lambda x: x[0])
    for _, tokens in results:
        all_tokens.extend(tokens)
    
    # Save tokenized data
    print(f"Writing {len(all_tokens)} tokens to {token_file}...")
    with open(token_file, "w", encoding="utf-8") as f:
        for token in tqdm(all_tokens, desc="Writing tokens"):
            f.write(f"{token}\n")
    
    print(f"Tokenized data saved to {token_file}")
    print(f"Total tokens: {len(all_tokens)}")
    
    return token_file, all_tokens

def prepare_training_data(tokens, block_size=128):
    """
    Convert tokens to sequences of fixed length for transformer training
    """
    # Create sequences of block_size + 1 (input + target)
    token_sequences = []
    for i in range(0, len(tokens) - block_size, block_size):
        # Sequence includes the prediction target
        token_sequences.append(tokens[i:i + block_size + 1])
    
    return token_sequences

def save_for_training(train_tokens, val_tokens, block_size=128):
    """
    Save tokenized data in formats compatible with train.py
    """
    # Create directory structure
    os.makedirs("./data/text", exist_ok=True)
    
    # Process tokens into training sequences
    if train_tokens:
        train_sequences = prepare_training_data(train_tokens, block_size)
        val_sequences = prepare_training_data(val_tokens, block_size)
    else:
        # If tokens weren't generated (because files existed), load from token files
        train_token_file = "./data/train_tokens.txt"
        val_token_file = "./data/val_tokens.txt"
        
        train_tokens = []
        with open(train_token_file, "r") as f:
            for line in f:
                train_tokens.append(int(line.strip()))
        
        val_tokens = []
        with open(val_token_file, "r") as f:
            for line in f:
                val_tokens.append(int(line.strip()))
        
        train_sequences = prepare_training_data(train_tokens, block_size)
        val_sequences = prepare_training_data(val_tokens, block_size)
    
    print(f"Created {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    
    # Save as PyTorch tensors (.pt files)
    print("Saving PyTorch tensor files...")
    
    # Convert sequences to tensor and save
    train_tensor = torch.tensor(train_sequences, dtype=torch.long)
    val_tensor = torch.tensor(val_sequences, dtype=torch.long)
    
    train_data = {'input_ids': train_tensor}
    val_data = {'input_ids': val_tensor}
    
    torch.save(train_data, "./data/text/train.pt")
    torch.save(val_data, "./data/text/val.pt")
    
    # Save as binary files (.bin files)
    print("Saving binary files...")
    train_array = np.array(train_tokens, dtype=np.int32)
    val_array = np.array(val_tokens, dtype=np.int32)
    
    train_array.tofile("./data/text/train.bin")
    val_array.tofile("./data/text/val.bin")
    
    # Save metadata
    total_tokens = len(train_tokens) + len(val_tokens)
    metadata = {
        'total_tokens': total_tokens,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'block_size': block_size,
        'vocab_size': 50304  # GPT-2 vocabulary size
    }
    torch.save(metadata, "./data/text/metadata.pt")
    
    print(f"Data saved to ./data/text/ directory")
    print(f"Total tokens: {total_tokens:,} ({len(train_tokens):,} training, {len(val_tokens):,} validation)")

if __name__ == "__main__":
    # Download the dataset
    train_file, val_file = download_tinystories()
    
    # Tokenize the dataset
    train_token_file, train_tokens = tokenize_data(train_file)
    val_token_file, val_tokens = tokenize_data(val_file)
    
    # Save data in formats compatible with train.py
    save_for_training(train_tokens, val_tokens, block_size=128)
    
    print("Done! Files are ready for training with train.py")
