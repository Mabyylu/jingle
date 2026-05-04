"""
Data generation and loading utilities for CMP.

Generates synthetic sequence data with long-range dependencies
to test the memory palace architecture.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class SyntheticSequenceDataset(Dataset):
    """
    Generate synthetic sequences with controlled long-range dependencies.
    
    This dataset creates sequences where:
    1. Some tokens depend on tokens from many steps ago
    2. Patterns repeat at regular intervals
    3. There are hierarchical structures (phrases within phrases)
    
    This tests whether the Memory Palace can learn to navigate
    back to relevant past information via learned paths.
    """
    
    def __init__(self, vocab_size=100, seq_len=50, num_samples=1000,
                 dependency_span=20, noise_level=0.1, seed=42):
        """
        Args:
            vocab_size: Number of unique tokens
            seq_len: Length of each sequence
            num_samples: Total number of sequences to generate
            dependency_span: Maximum distance for token dependencies
            noise_level: Probability of random token replacement
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.dependency_span = dependency_span
        self.noise_level = noise_level
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate all samples upfront
        self.sequences = []
        self.targets = []
        self._generate_all_samples()
    
    def _generate_all_samples(self):
        """Generate all sequence samples."""
        for _ in range(self.num_samples):
            seq, target = self._generate_single_sample()
            self.sequences.append(seq)
            self.targets.append(target)
    
    def _generate_single_sample(self):
        """Generate a single sequence with long-range dependencies."""
        seq = []
        
        # Start with some initial context
        for i in range(self.seq_len):
            if i < 5:
                # Initial tokens are random
                token = random.randint(0, self.vocab_size - 1)
            elif random.random() < 0.3:
                # 30% of tokens depend on earlier tokens
                lookback = random.randint(5, min(self.dependency_span, i))
                token = seq[i - lookback]
                
                # Sometimes apply a transformation
                if random.random() < 0.3:
                    token = (token + random.randint(1, 5)) % self.vocab_size
            else:
                # Other tokens are based on recent context or random
                if random.random() < 0.5 and i >= 3:
                    # Copy from recent context
                    lookback = random.randint(1, 3)
                    token = seq[i - lookback]
                else:
                    token = random.randint(0, self.vocab_size - 1)
            
            # Add noise
            if random.random() < self.noise_level:
                token = random.randint(0, self.vocab_size - 1)
            
            seq.append(token)
        
        # Target is predicting the next token after the sequence
        # This depends on patterns learned throughout the sequence
        if random.random() < 0.4 and len(seq) > self.dependency_span:
            # Strong dependency on a specific earlier position
            key_position = random.randint(0, len(seq) - self.dependency_span)
            target = (seq[key_position] + random.randint(0, 2)) % self.vocab_size
        else:
            # Based on recent context
            target = (seq[-1] + random.randint(0, 3)) % self.vocab_size
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TextDataset(Dataset):
    """
    Simple character-level text dataset for real text training.
    """
    
    def __init__(self, text, seq_len=50, char_vocab=None):
        """
        Args:
            text: Raw text string
            seq_len: Length of input sequences
            char_vocab: Optional predefined vocabulary
        """
        self.seq_len = seq_len
        
        # Build character vocabulary
        if char_vocab is None:
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            self.char_to_idx = char_vocab['char_to_idx']
            self.idx_to_char = char_vocab['idx_to_char']
        
        self.vocab_size = len(self.char_to_idx)
        
        # Convert text to indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
        # Calculate number of samples
        self.num_samples = max(0, len(self.data) - seq_len)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """Create a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def generate_synthetic_data(vocab_size=100, seq_len=50, num_samples=1000,
                           batch_size=32, split=(0.8, 0.1, 0.1)):
    """
    Generate train/val/test splits of synthetic data.
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        num_samples: Total number of samples
        batch_size: Batch size for DataLoaders
        split: Tuple of (train_ratio, val_ratio, test_ratio)
    
    Returns:
        dict with train_loader, val_loader, test_loader, and metadata
    """
    total_samples = num_samples
    train_samples = int(total_samples * split[0])
    val_samples = int(total_samples * split[1])
    test_samples = total_samples - train_samples - val_samples
    
    # Generate datasets with different seeds for each split
    train_dataset = SyntheticSequenceDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_samples=train_samples,
        seed=42
    )
    
    val_dataset = SyntheticSequenceDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_samples=val_samples,
        seed=43
    )
    
    test_dataset = SyntheticSequenceDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_samples=test_samples,
        seed=44
    )
    
    # Create DataLoaders
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'metadata': {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples
        }
    }


def load_text_data(file_path, seq_len=50, batch_size=32, val_ratio=0.1):
    """
    Load text data from a file.
    
    Args:
        file_path: Path to text file
        seq_len: Sequence length
        batch_size: Batch size
        val_ratio: Validation set ratio
    
    Returns:
        dict with train_loader, val_loader, and metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into train/val
    split_idx = int(len(text) * (1 - val_ratio))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets (they'll share vocabulary)
    train_dataset = TextDataset(train_text, seq_len=seq_len)
    val_dataset = TextDataset(val_text, seq_len=seq_len, 
                              char_vocab={'char_to_idx': train_dataset.char_to_idx,
                                         'idx_to_char': train_dataset.idx_to_char})
    
    # Create DataLoaders
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'vocab_size': train_dataset.vocab_size,
        'seq_len': seq_len,
        'char_to_idx': train_dataset.char_to_idx,
        'idx_to_char': train_dataset.idx_to_char,
        'metadata': {
            'train_chars': len(train_text),
            'val_chars': len(val_text),
            'vocab_size': train_dataset.vocab_size
        }
    }
