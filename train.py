"""
Training script for Chronological Memory Palaces.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.cmp_network import CMPNetwork, CMPLoss
from src.data.dataset import generate_synthetic_data
from src.utils.helpers import (
    set_seed, get_device, count_parameters, 
    save_checkpoint, AverageMeter, EarlyStopping,
    print_ascii_memory
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        result = model(inputs, return_memories=True)
        logits = result['logits']
        
        # Reshape for loss calculation
        # For synthetic data: targets are single tokens, so we use last prediction
        if targets.dim() == 1:
            logits_flat = logits[:, -1, :]  # Use last timestep
            targets_flat = targets
        else:
            # For character-level text: targets are sequences
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
        
        # Calculate loss
        path_weights = result.get('path_weights')
        memory = model.memory_palace.memory
        
        loss, loss_dict = criterion(logits_flat, targets_flat, path_weights, memory)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{loss_dict["cross_entropy"]:.4f}',
        })
    
    return loss_meter.avg, loss_dict


def validate(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            result = model(inputs, return_memories=True)
            logits = result['logits']
            
            # Reshape for loss calculation
            if targets.dim() == 1:
                logits_flat = logits[:, -1, :]
                targets_flat = targets
            else:
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
            
            # Calculate loss
            path_weights = result.get('path_weights')
            memory = model.memory_palace.memory
            
            loss, _ = criterion(logits_flat, targets_flat, path_weights, memory)
            
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Train CMP model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seeds
    set_seed(config['training']['seed'])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating synthetic data...")
    data = generate_synthetic_data(
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        num_samples=config['data']['num_samples'],
        batch_size=config['training']['batch_size']
    )
    
    print(f"Train samples: {data['metadata']['train_samples']}")
    print(f"Val samples: {data['metadata']['val_samples']}")
    
    # Create model
    print("\nCreating model...")
    model = CMPNetwork(
        vocab_size=config['data']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        memory_size=config['model']['memory_size'],
        num_paths=config['model']['num_paths'],
        max_seq_len=config['data']['seq_len']
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss and optimizer
    criterion = CMPLoss(
        path_diversity_weight=config['training']['path_diversity_weight'],
        memory_reg_weight=config['training']['memory_reg_weight']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, data['train_loader'], criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate(model, data['val_loader'], criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print summary
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  CE: {train_loss_dict['cross_entropy']:.4f}")
        if 'path_diversity' in train_loss_dict:
            print(f"  Path Diversity: {train_loss_dict['path_diversity']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 'best_model.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Visualize memory palace every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            memory_norms = torch.norm(model.memory_palace.memory, dim=-1).cpu().numpy()
            print_ascii_memory(memory_norms, width=40, height=20)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Show final memory palace state
    print("\nFinal Memory Palace State:")
    memory_norms = torch.norm(model.memory_palace.memory, dim=-1).cpu().numpy()
    print_ascii_memory(memory_norms, width=40, height=20)
    
    # Save final model
    save_checkpoint(model, optimizer, epoch, val_loss, 'final_model.pt')
    
    return model


if __name__ == '__main__':
    main()
