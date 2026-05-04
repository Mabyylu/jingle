"""
CMP Network - Full architecture combining Memory Palace with temporal processing.

This implements the complete Chronological Memory Palaces architecture for
sequence modeling and prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_palace import MemoryPalace


class CMPNetwork(nn.Module):
    """
    Complete CMP architecture for sequence learning.
    
    Architecture:
    1. Input encoder converts tokens/observations to embeddings
    2. Temporal processor maintains hidden state
    3. Memory Palace stores and retrieves information via paths
    4. Output decoder produces predictions
    
    The key innovation is that memory retrieval happens through learned
    spatiotemporal paths, not just content-addressable lookup.
    """
    
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256,
                 memory_size=64, num_paths=16, max_seq_len=100):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Input encoding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Temporal processor (simple GRU for baseline)
        self.temporal_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Memory Palace - the core innovation
        self.memory_palace = MemoryPalace(
            memory_size=memory_size,
            embedding_dim=hidden_dim,
            num_paths=num_paths
        )
        
        # Integration layer: combine temporal state with retrieved memories
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output decoder
        self.output_decoder = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode_input(self, input_ids):
        """Convert input tokens to embeddings with positional encoding."""
        batch_size, seq_len = input_ids.shape
        
        token_emb = self.token_embedding(input_ids)  # (batch, seq, embed)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq, embed)
        
        return token_emb + pos_emb
    
    def forward(self, input_ids, return_memories=False):
        """
        Process a sequence through the CMP architecture.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token IDs
            return_memories: Whether to return memory access patterns
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            memory_info: Optional dict with memory statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Encode inputs
        embedded = self.encode_input(input_ids)  # (batch, seq, embed)
        
        # Step 2: Temporal encoding
        temporal_out, hidden_state = self.temporal_encoder(embedded)  # (batch, seq, hidden)
        
        # Step 3: Memory retrieval at each timestep
        outputs = []
        path_weights_list = []
        
        for t in range(seq_len):
            # Get current temporal state
            current_state = temporal_out[:, t, :]  # (batch, hidden)
            
            # Query memory palace
            retrieved, path_weights = self.memory_palace(current_state)
            
            # Integrate temporal state with retrieved memory
            combined = torch.cat([current_state, retrieved], dim=-1)
            integrated = self.integration(combined)
            
            # Decode to vocabulary
            logits = self.output_decoder(integrated)
            outputs.append(logits)
            
            if return_memories:
                path_weights_list.append(path_weights)
            
            # Write current state back to memory (online learning)
            self.memory_palace.write_to_memory(current_state)
        
        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # (batch, seq, vocab)
        
        result = {'logits': logits}
        
        if return_memories:
            result['path_weights'] = torch.stack(path_weights_list, dim=1)
            result['memory_usage'] = self._compute_memory_usage()
        
        return result
    
    def _compute_memory_usage(self):
        """Compute statistics about memory palace usage."""
        memory_norms = torch.norm(self.memory_palace.memory, dim=-1)
        return {
            'mean_activation': memory_norms.mean().item(),
            'max_activation': memory_norms.max().item(),
            'sparsity': (memory_norms < 0.1).float().mean().item()
        }
    
    def predict_next(self, input_ids, temperature=1.0):
        """
        Predict the next token given a sequence.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            temperature: Sampling temperature
            
        Returns:
            next_token: Tensor of shape (batch_size,)
            confidence: Tensor of shape (batch_size,)
        """
        with torch.no_grad():
            result = self.forward(input_ids)
            logits = result['logits'][:, -1, :]  # Take last timestep
            
            if temperature != 1.0:
                logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            confidence = probs.max(dim=-1).values
            
            return next_token, confidence
    
    def get_path_visualization(self, input_ids):
        """
        Get visualization data for memory palace paths.
        
        Returns coordinates and weights for plotting.
        """
        with torch.no_grad():
            result = self.forward(input_ids, return_memories=True)
            
            path_weights = result['path_weights']  # (batch, seq, num_paths)
            
            # Get average path weights across sequence
            avg_path_weights = path_weights.mean(dim=1)  # (batch, num_paths)
            
            # Get trajectory coordinates for top paths
            trajectories = {}
            for p in range(self.memory_palace.num_paths):
                traj = self.memory_palace._get_path_trajectory(p, t_steps=32)
                trajectories[p] = traj.cpu().numpy()
            
            return {
                'trajectories': trajectories,
                'path_weights': avg_path_weights.cpu().numpy(),
                'memory_usage': result['memory_usage']
            }


class CMPLoss(nn.Module):
    """
    Loss function for CMP training.
    
    Combines standard cross-entropy with regularization terms for:
    1. Path diversity (encourage using multiple paths)
    2. Memory utilization (encourage spreading memories)
    """
    
    def __init__(self, path_diversity_weight=0.01, memory_reg_weight=0.001):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.path_diversity_weight = path_diversity_weight
        self.memory_reg_weight = memory_reg_weight
    
    def forward(self, logits, targets, path_weights=None, memory=None):
        """
        Compute total loss.
        
        Args:
            logits: Tensor of shape (batch * seq, vocab)
            targets: Tensor of shape (batch * seq,)
            path_weights: Optional tensor of shape (batch, seq, num_paths)
            memory: Optional memory tensor for regularization
            
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dict with individual loss components
        """
        # Base cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        loss_dict = {'cross_entropy': ce_loss.item()}
        total_loss = ce_loss
        
        # Path diversity loss (entropy regularization)
        if path_weights is not None:
            path_entropy = -(path_weights * torch.log(path_weights + 1e-10)).sum(dim=-1).mean()
            diversity_loss = -path_entropy  # Maximize entropy = minimize negative entropy
            total_loss = total_loss + self.path_diversity_weight * diversity_loss
            loss_dict['path_diversity'] = diversity_loss.item()
        
        # Memory regularization (L2 penalty to prevent explosion)
        if memory is not None:
            memory_reg = torch.norm(memory).pow(2)
            total_loss = total_loss + self.memory_reg_weight * memory_reg
            loss_dict['memory_reg'] = memory_reg.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
