"""
Utility functions for Chronological Memory Palaces.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load a training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def visualize_memory_palace(memory_palace, save_path=None):
    """
    Create a simple visualization of memory palace activation patterns.
    
    Returns a dictionary with visualization data.
    """
    with torch.no_grad():
        memory = memory_palace.memory
        norms = torch.norm(memory, dim=-1).cpu().numpy()
        
        # Get path trajectories
        trajectories = {}
        for p in range(memory_palace.num_paths):
            traj = memory_palace._get_path_trajectory(p, t_steps=32)
            trajectories[p] = traj.cpu().numpy()
        
        viz_data = {
            'memory_norms': norms,
            'trajectories': trajectories,
            'memory_size': memory_palace.memory_size,
            'num_paths': memory_palace.num_paths
        }
        
        if save_path:
            import json
            # Convert numpy arrays to lists for JSON serialization
            viz_data_json = {
                'memory_norms': norms.tolist(),
                'trajectories': {k: v.tolist() for k, v in trajectories.items()},
                'memory_size': memory_palace.memory_size,
                'num_paths': memory_palace.num_paths
            }
            with open(save_path, 'w') as f:
                json.dump(viz_data_json, f)
        
        return viz_data


def print_ascii_memory(memory_norms, width=60, height=30):
    """Print ASCII visualization of memory palace activations."""
    h, w = memory_norms.shape
    
    # Resize to fit terminal
    step_h = max(1, h // height)
    step_w = max(1, w // width)
    
    chars = ' .:-=+*#%@'
    
    print("\nMemory Palace Activation Map:")
    print("=" * (width + 2))
    
    for i in range(0, h, step_h):
        row = "|"
        for j in range(0, w, step_w):
            val = memory_norms[i, j]
            idx = min(int(val * len(chars)), len(chars) - 1)
            row += chars[idx]
        row += "|"
        print(row)
    
    print("=" * (width + 2))


def generate_sample_sequences(model, vocab_size, seq_len=20, num_samples=5, 
                             temperature=0.8, idx_to_char=None):
    """Generate sample sequences from a trained model."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Start with random tokens
            input_seq = torch.randint(0, vocab_size, (1, 5))
            
            generated = input_seq.tolist()[0]
            
            for _ in range(seq_len - 5):
                next_token, _ = model.predict_next(input_seq, temperature=temperature)
                generated.append(next_token.item())
                
                # Update input sequence
                input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            
            # Convert to text if vocabulary mapping provided
            if idx_to_char:
                text = ''.join([idx_to_char.get(idx, '?') for idx in generated])
                samples.append(text)
            else:
                samples.append(generated)
    
    return samples
