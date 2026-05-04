"""
Inference script for Chronological Memory Palaces.

Usage:
    python inference.py --model best_model.pt --config configs/config.yaml
"""

import argparse
import yaml
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.cmp_network import CMPNetwork
from src.utils.helpers import set_seed, get_device, generate_sample_sequences


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, device):
    """Load a trained model from checkpoint."""
    # Create model
    model = CMPNetwork(
        vocab_size=config['data']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        memory_size=config['model']['memory_size'],
        num_paths=config['model']['num_paths'],
        max_seq_len=config['data']['seq_len']
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Loss: {checkpoint['loss']:.4f}")
    
    return model


def interactive_generation(model, config, device, idx_to_char=None):
    """Interactive text generation session."""
    print("\n" + "="*50)
    print("Interactive Generation Mode")
    print("="*50)
    print("Type your input sequence (or 'quit' to exit)")
    print("Commands:")
    print("  /temp <value>  - Set temperature (default: 0.8)")
    print("  /len <value>   - Set generation length (default: 20)")
    print("  /viz           - Show memory palace visualization")
    print("")
    
    temperature = 0.8
    gen_length = 20
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.startswith('/temp'):
                parts = user_input.split()
                if len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                continue
            
            if user_input.startswith('/len'):
                parts = user_input.split()
                if len(parts) > 1:
                    gen_length = int(parts[1])
                    print(f"Generation length set to {gen_length}")
                continue
            
            if user_input.startswith('/viz'):
                viz_data = model.get_path_visualization(
                    torch.randint(0, config['data']['vocab_size'], (1, 10)).to(device)
                )
                print("\nPath weights (top 5 paths):")
                avg_weights = viz_data['path_weights'].mean(0)
                top_paths = sorted(range(len(avg_weights)), key=lambda i: avg_weights[i], reverse=True)[:5]
                for p in top_paths:
                    print(f"  Path {p}: {avg_weights[p]:.4f}")
                continue
            
            # Convert input to token IDs
            if idx_to_char:
                char_to_idx = {v: k for k, v in idx_to_char.items()}
                input_ids = [char_to_idx.get(ch, 0) for ch in user_input]
                if len(input_ids) < 5:
                    # Pad with random tokens
                    input_ids += [torch.randint(0, config['data']['vocab_size'], (5 - len(input_ids),)).tolist()]
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            else:
                # Parse as space-separated integers
                try:
                    input_ids = torch.tensor([[int(x) for x in user_input.split()]], 
                                           dtype=torch.long).to(device)
                except ValueError:
                    print("Please enter space-separated integers or use character mode")
                    continue
            
            # Generate continuation
            generated = input_ids.tolist()[0].copy()
            
            with torch.no_grad():
                for _ in range(gen_length):
                    next_token, confidence = model.predict_next(input_ids, temperature=temperature)
                    generated.append(next_token.item())
                    
                    # Update input
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Display output
            if idx_to_char:
                input_text = ''.join([idx_to_char.get(i, '?') for i in generated[:len(user_input)]])
                generated_text = ''.join([idx_to_char.get(i, '?') for i in generated[len(user_input):]])
                print(f"\nInput: {input_text}")
                print(f"Generated: {generated_text}")
                print(f"Avg confidence: {confidence.mean().item():.3f}\n")
            else:
                print(f"\nInput: {user_input}")
                print(f"Generated: {' '.join(map(str, generated[len(user_input.split()):]))}")
                print(f"Avg confidence: {confidence.mean().item():.3f}\n")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def run_benchmark(model, config, device, num_samples=100):
    """Run benchmark on synthetic data."""
    from src.data.dataset import generate_synthetic_data
    
    print("\nRunning benchmark...")
    
    # Generate test data
    data = generate_synthetic_data(
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        num_samples=num_samples,
        batch_size=32
    )
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data['test_loader']:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            result = model(inputs)
            logits = result['logits'][:, -1, :]  # Last prediction
            
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total * 100
    print(f"Benchmark Results:")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='CMP Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'benchmark', 'sample'],
                       help='Inference mode')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to generate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seeds
    set_seed(config['training']['seed'])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, config, device)
    
    # Run in specified mode
    if args.mode == 'interactive':
        interactive_generation(model, config, device)
    elif args.mode == 'benchmark':
        run_benchmark(model, config, device, num_samples=args.num_samples * 10)
    elif args.mode == 'sample':
        samples = generate_sample_sequences(
            model, 
            config['data']['vocab_size'],
            seq_len=30,
            num_samples=args.num_samples,
            temperature=0.8
        )
        print("\nGenerated Samples:")
        print("="*50)
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            if isinstance(sample, str):
                print(sample)
            else:
                print(' '.join(map(str, sample)))
    
    print("\nDone!")


if __name__ == '__main__':
    main()
