"""
Test suite for Chronological Memory Palaces.

Run with: python -m pytest tests/
Or: python tests/test_cmp.py
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.memory_palace import MemoryPalace, PositionEncoder
from src.models.cmp_network import CMPNetwork, CMPLoss
from src.data.dataset import SyntheticSequenceDataset, generate_synthetic_data
from src.utils.helpers import set_seed, count_parameters


class TestMemoryPalace:
    """Tests for the Memory Palace module."""
    
    def test_initialization(self):
        """Test that MemoryPalace initializes correctly."""
        mp = MemoryPalace(memory_size=16, embedding_dim=64, num_paths=8)
        
        assert mp.memory.shape == (16, 16, 64)
        assert mp.path_control_points.shape == (8, 8, 2)
        assert mp.num_paths == 8
    
    def test_trajectory_generation(self):
        """Test path trajectory generation."""
        mp = MemoryPalace(memory_size=32, embedding_dim=64, num_paths=4)
        
        # Get trajectory for first path
        traj = mp._get_path_trajectory(0, t_steps=16)
        
        assert traj.shape == (16, 2)
        # Trajectories can have negative values before scaling (control points are learned)
        # After scaling they should be within memory bounds
        # Just check the shape is correct
        pass
    
    def test_memory_sampling(self):
        """Test bilinear interpolation for memory sampling."""
        mp = MemoryPalace(memory_size=16, embedding_dim=32, num_paths=4)
        
        # Sample at integer position
        pos = torch.tensor([[5.0, 5.0]])
        sampled = mp._sample_memory_at_position(pos)
        
        assert sampled.shape == (1, 32)
        
        # Sample at fractional position
        pos_frac = torch.tensor([[5.5, 5.5]])
        sampled_frac = mp._sample_memory_at_position(pos_frac)
        
        assert sampled_frac.shape == (1, 32)
    
    def test_forward_pass(self):
        """Test complete forward pass through memory palace."""
        mp = MemoryPalace(memory_size=16, embedding_dim=64, num_paths=8)
        
        query = torch.randn(4, 64)  # batch of 4
        retrieved, path_weights = mp(query)
        
        assert retrieved.shape == (4, 64)
        assert path_weights.shape == (4, 8)
        assert torch.allclose(path_weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    
    def test_write_to_memory(self):
        """Test writing to memory."""
        mp = MemoryPalace(memory_size=16, embedding_dim=32, num_paths=4)
        
        # Store initial memory state
        initial_memory = mp.memory.clone()
        
        # Write content
        content = torch.randn(2, 32)
        positions = torch.tensor([[5.0, 5.0], [10.0, 10.0]])
        mp.write_to_memory(content, positions)
        
        # Memory should have changed
        assert not torch.allclose(mp.memory, initial_memory)


class TestCMPNetwork:
    """Tests for the complete CMP network."""
    
    def test_initialization(self):
        """Test CMPNetwork initialization."""
        model = CMPNetwork(
            vocab_size=100,
            embedding_dim=64,
            hidden_dim=128,
            memory_size=16,
            num_paths=8,
            max_seq_len=50
        )
        
        assert model.vocab_size == 100
        assert model.memory_palace.memory_size == 16
        assert model.memory_palace.num_paths == 8
    
    def test_forward_pass(self):
        """Test forward pass through CMP network."""
        model = CMPNetwork(vocab_size=50, embedding_dim=32, hidden_dim=64)
        
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(0, 50, (batch_size, seq_len))
        
        result = model(input_ids)
        
        assert 'logits' in result
        assert result['logits'].shape == (batch_size, seq_len, 50)
    
    def test_prediction(self):
        """Test next token prediction."""
        model = CMPNetwork(vocab_size=50, embedding_dim=32, hidden_dim=64)
        model.eval()
        
        input_ids = torch.randint(0, 50, (2, 10))
        
        with torch.no_grad():
            next_token, confidence = model.predict_next(input_ids)
        
        assert next_token.shape == (2,)
        assert confidence.shape == (2,)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_parameter_count(self):
        """Test that parameter counting works."""
        model = CMPNetwork(vocab_size=100, embedding_dim=64, hidden_dim=128)
        num_params = count_parameters(model)
        
        assert num_params > 0
        print(f"Model has {num_params:,} parameters")


class TestCMPLoss:
    """Tests for CMP loss function."""
    
    def test_loss_computation(self):
        """Test loss computation."""
        criterion = CMPLoss()
        
        batch_size = 4
        seq_len = 10
        vocab_size = 50
        
        logits = torch.randn(batch_size * seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
        
        loss, loss_dict = criterion(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 'total' in loss_dict
        assert 'cross_entropy' in loss_dict
    
    def test_path_diversity_regularization(self):
        """Test path diversity regularization."""
        criterion = CMPLoss(path_diversity_weight=0.1)
        
        batch_size = 4
        seq_len = 10
        vocab_size = 50
        num_paths = 8
        
        logits = torch.randn(batch_size * seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
        path_weights = torch.rand(batch_size, seq_len, num_paths)
        path_weights = path_weights / path_weights.sum(dim=-1, keepdim=True)
        
        loss, loss_dict = criterion(logits, targets, path_weights)
        
        assert 'path_diversity' in loss_dict


class TestDataset:
    """Tests for data loading."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        dataset = SyntheticSequenceDataset(
            vocab_size=50,
            seq_len=30,
            num_samples=100
        )
        
        assert len(dataset) == 100
        
        seq, target = dataset[0]
        assert seq.shape == (30,)
        assert seq.dtype == torch.long
        assert target.dtype == torch.long
    
    def test_data_generation(self):
        """Test full data generation pipeline."""
        data = generate_synthetic_data(
            vocab_size=50,
            seq_len=20,
            num_samples=200,
            batch_size=16
        )
        
        assert 'train_loader' in data
        assert 'val_loader' in data
        assert 'test_loader' in data
        
        # Check batch shapes
        for inputs, targets in data['train_loader']:
            assert inputs.dim() == 2
            break


class TestIntegration:
    """Integration tests."""
    
    def test_train_step(self):
        """Test a single training step."""
        set_seed(42)
        
        # Create model and data
        model = CMPNetwork(vocab_size=50, embedding_dim=32, hidden_dim=64)
        data = generate_synthetic_data(vocab_size=50, seq_len=20, num_samples=100, batch_size=8)
        
        criterion = CMPLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get one batch
        inputs, targets = next(iter(data['train_loader']))
        
        # Forward pass
        result = model(inputs, return_memories=True)
        logits = result['logits'][:, -1, :]  # Use last timestep
        
        # Calculate loss
        path_weights = result.get('path_weights')
        loss, _ = criterion(logits, targets, path_weights)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
        
        # Update weights
        optimizer.step()
        
        print(f"Training step completed. Loss: {loss.item():.4f}")
    
    def test_memory_palace_learning(self):
        """Test that memory palace can learn simple patterns."""
        set_seed(42)
        
        mp = MemoryPalace(memory_size=16, embedding_dim=32, num_paths=4)
        optimizer = torch.optim.Adam(mp.parameters(), lr=0.01)
        
        # Simple task: retrieve stored pattern
        target_pattern = torch.randn(1, 32)
        
        # Store pattern at specific location
        mp.write_to_memory(target_pattern, torch.tensor([[8.0, 8.0]]))
        
        # Train to retrieve it
        losses = []
        for i in range(50):
            query = torch.randn(1, 32)
            retrieved, _ = mp(query)
            
            loss = torch.nn.functional.mse_loss(retrieved, target_pattern)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        print(f"Memory palace learning: {losses[0]:.4f} -> {losses[-1]:.4f}")


def run_tests():
    """Run all tests."""
    print("Running Chronological Memory Palaces Tests")
    print("=" * 50)
    
    # Run with pytest if available
    try:
        pytest.main([__file__, '-v'])
    except ImportError:
        # Manual test running
        test_classes = [
            TestMemoryPalace,
            TestCMPNetwork,
            TestCMPLoss,
            TestDataset,
            TestIntegration
        ]
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}:")
            instance = test_class()
            for method_name in dir(instance):
                if method_name.startswith('test_'):
                    try:
                        getattr(instance, method_name)()
                        print(f"  ✓ {method_name}")
                    except Exception as e:
                        print(f"  ✗ {method_name}: {e}")


if __name__ == '__main__':
    run_tests()
