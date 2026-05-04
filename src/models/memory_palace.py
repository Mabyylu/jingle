"""
Memory Palace Module - Core novelty of CMP architecture.

This implements a differentiable memory palace where:
1. Memories are stored as vectors in a 2D spatial grid
2. Navigation happens via learned path embeddings
3. Retrieval uses soft attention along paths, not just content similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryPalace(nn.Module):
    """
    A 2D spatial memory structure with path-based navigation.
    
    Unlike traditional memory networks that use content-addressable retrieval,
    this uses spatiotemporal paths for memory access - mimicking the human
    "method of loci" memory technique.
    """
    
    def __init__(self, memory_size=64, embedding_dim=128, num_paths=16):
        super().__init__()
        self.memory_size = memory_size  # Size of each dimension (memory_size x memory_size grid)
        self.embedding_dim = embedding_dim
        self.num_paths = num_paths
        
        # The memory palace: a 2D grid of memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, memory_size, embedding_dim))
        
        # Path embeddings: each path is a sequence of positions through the palace
        # Represented as a continuous trajectory parameterized by control points
        self.path_control_points = nn.Parameter(torch.randn(num_paths, 8, 2))  # 8 control points per path
        
        # Path encoder: converts temporal context to path selection
        self.path_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_paths)
        )
        
        # Position encoder for continuous coordinates
        self.pos_encoder = PositionEncoder(embedding_dim)
        
        # Initialize memory
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.path_control_points)
    
    def _get_path_trajectory(self, path_idx, t_steps=32):
        """
        Generate continuous trajectory for a path using Bezier-like interpolation.
        
        Args:
            path_idx: Index of the path (0 to num_paths-1)
            t_steps: Number of time steps to sample along the path
            
        Returns:
            trajectory: Tensor of shape (t_steps, 2) with continuous coordinates [0, 1]
        """
        control_points = self.path_control_points[path_idx]  # (8, 2)
        
        # Use Bernstein polynomials for smooth interpolation
        t = torch.linspace(0, 1, t_steps).unsqueeze(1)  # (t_steps, 1)
        
        # Simple weighted interpolation (simplified Bezier)
        weights = self._bernstein_weights(t, len(control_points))  # (t_steps, 8)
        trajectory = torch.matmul(weights, control_points)  # (t_steps, 2)
        
        # Scale to memory coordinates
        trajectory = trajectory * (self.memory_size - 1)
        
        return trajectory
    
    def _bernstein_weights(self, t, n):
        """Compute Bernstein polynomial weights for Bezier curves."""
        weights = []
        for i in range(n):
            # Binomial coefficient
            coef = math.factorial(n - 1) / (math.factorial(i) * math.factorial(n - 1 - i))
            weight = coef * (t ** i) * ((1 - t) ** (n - 1 - i))
            weights.append(weight)
        return torch.cat(weights, dim=1)  # (t_steps, n)
    
    def _sample_memory_at_position(self, position):
        """
        Sample memory at a continuous 2D position using bilinear interpolation.
        
        Args:
            position: Tensor of shape (batch_size, 2) or (t_steps, 2) with continuous coordinates
            
        Returns:
            sampled: Tensor of shape (batch_size, embedding_dim) or (t_steps, embedding_dim)
        """
        original_shape = position.shape
        
        # Get integer coordinates and fractional parts
        pos_floor = position.floor().long()
        pos_ceil = pos_floor + 1
        frac = position - pos_floor.float()
        
        # Clamp to valid range
        pos_floor = torch.clamp(pos_floor, 0, self.memory_size - 1)
        pos_ceil = torch.clamp(pos_ceil, 0, self.memory_size - 1)
        
        # Get four corner memories
        m00 = self.memory[pos_floor[..., 0], pos_floor[..., 1]]
        m01 = self.memory[pos_floor[..., 0], pos_ceil[..., 1]]
        m10 = self.memory[pos_ceil[..., 0], pos_floor[..., 1]]
        m11 = self.memory[pos_ceil[..., 0], pos_ceil[..., 1]]
        
        # Bilinear interpolation
        m0 = m00 * (1 - frac[..., 1:2]) + m01 * frac[..., 1:2]
        m1 = m10 * (1 - frac[..., 1:2]) + m11 * frac[..., 1:2]
        sampled = m0 * (1 - frac[..., 0:1]) + m1 * frac[..., 0:1]
        
        return sampled
    
    def traverse_path(self, path_idx, t_steps=32):
        """
        Traverse a path and collect memories along it.
        
        Returns:
            path_memories: Tensor of shape (t_steps, embedding_dim)
        """
        trajectory = self._get_path_trajectory(path_idx, t_steps)  # (t_steps, 2)
        memories = self._sample_memory_at_position(trajectory)  # (t_steps, embedding_dim)
        return memories
    
    def forward(self, query, context=None):
        """
        Navigate the memory palace given a query.
        
        Args:
            query: Tensor of shape (batch_size, embedding_dim)
            context: Optional additional context for path selection
            
        Returns:
            retrieved: Tensor of shape (batch_size, embedding_dim)
            path_weights: Tensor of shape (batch_size, num_paths)
        """
        batch_size = query.shape[0]
        
        # Select paths based on query
        if context is None:
            context = query
        path_logits = self.path_encoder(context)  # (batch_size, num_paths)
        path_weights = F.softmax(path_logits, dim=-1)
        
        # Traverse all paths and aggregate
        retrieved = 0
        for p in range(self.num_paths):
            path_memories = self.traverse_path(p)  # (t_steps, embedding_dim)
            
            # Compute attention over path positions
            # path_query_sim: (batch, t_steps)
            path_query_sim = torch.einsum('be,te->bt', query, path_memories)
            path_attention = F.softmax(path_query_sim / math.sqrt(self.embedding_dim), dim=-1)
            
            # Weighted sum along path: (batch, t_steps) @ (t_steps, embed) = (batch, embed)
            path_output = torch.matmul(path_attention, path_memories)
            
            retrieved = retrieved + path_weights[:, p:p+1] * path_output
        
        return retrieved, path_weights
    
    def write_to_memory(self, content, positions=None, write_weights=None):
        """
        Write content to memory at specified positions.
        
        Args:
            content: Tensor of shape (batch_size, embedding_dim)
            positions: Tensor of shape (batch_size, 2) with continuous coordinates
            write_weights: Optional tensor of shape (batch_size,) for soft writes
        """
        if positions is None:
            # Default: distribute across memory based on content
            with torch.no_grad():
                content_proj = torch.tanh(content @ torch.randn(self.embedding_dim, 2).to(content.device))
                positions = (content_proj + 1) / 2 * (self.memory_size - 1)
        
        # Soft write using Gaussian kernel around positions
        batch_size = content.shape[0]
        for b in range(batch_size):
            pos = positions[b]
            content_vec = content[b]
            
            # Add to nearby memory locations
            center = pos.round().long()
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    idx_i = int(torch.clamp(center[0] + di, 0, self.memory_size - 1).item())
                    idx_j = int(torch.clamp(center[1] + dj, 0, self.memory_size - 1).item())
                    
                    # Distance-based weight
                    dist = abs(di) + abs(dj)
                    weight = 1.0 / (1.0 + dist)
                    
                    # Use data accessor for in-place modification
                    self.memory.data[idx_i, idx_j] = self.memory.data[idx_i, idx_j] * (1 - weight * 0.1) + content_vec.data * weight * 0.1


def squeezed_if_possible(tensor):
    """Helper to ensure tensor has correct shape."""
    # Keep tensor as-is, just ensure it's the right shape
    return tensor


class PositionEncoder(nn.Module):
    """Encode continuous 2D positions into embeddings."""
    
    def __init__(self, d_model, max_freq=100):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
    def forward(self, positions):
        """
        Args:
            positions: Tensor of shape (..., 2) with coordinates
        Returns:
            encoded: Tensor of shape (..., d_model)
        """
        # Split into x and y coordinates
        coords = positions.unbind(dim=-1)  # List of tensors
        
        encodings = []
        for coord in coords:
            # Sinusoidal encoding for each coordinate
            freqs = torch.exp(torch.linspace(0, math.log(self.max_freq), self.d_model // 4) * -1j * 2 * math.pi)
            freqs = freqs.to(coord.device)
            
            coord_scaled = coord.unsqueeze(-1) * freqs.real  # (..., d_model/4)
            sin_enc = torch.sin(coord_scaled)
            cos_enc = torch.cos(coord_scaled)
            encodings.append(torch.cat([sin_enc, cos_enc], dim=-1))
        
        # Combine x and y encodings
        combined = torch.cat(encodings, dim=-1)  # (..., d_model)
        return combined
