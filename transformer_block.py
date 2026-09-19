"""
Megatron-LM Style Transformer Block Implementation
Includes Multi-Head Attention, Feed-Forward Network, and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional tensor parallelism"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear transformations
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, seq_length, seq_length) or None
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        query = self.query(hidden_states)  # (batch, seq_len, hidden_size)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)  # (batch, num_heads, seq_length, head_dim)
        
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)  # (batch, num_heads, seq_length, head_dim)
        
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)  # (batch, num_heads, seq_length, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # scores: (batch, num_heads, seq_length, seq_length)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Context vector
        context = torch.matmul(attention_weights, value)
        # context: (batch, num_heads, seq_length, head_dim)
        
        context = context.transpose(1, 2).contiguous()  # (batch, seq_length, num_heads, head_dim)
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Feed-forward network (Dense -> Activation -> Dense)"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # First dense layer (expand)
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swiglu":
            # SwiGLU: swish(W1 * x) * (W3 * x)
            self.dense_3 = nn.Linear(hidden_size, intermediate_size)
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Second dense layer (contract)
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_name = activation
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
        """
        # First dense layer
        hidden_states = self.dense_1(hidden_states)
        
        if self.activation_name == "swiglu":
            # SwiGLU variant
            gate = self.activation(self.dense_3(hidden_states))
            hidden_states = hidden_states * gate
        else:
            # Standard GELU/ReLU
            hidden_states = self.activation(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        
        # Second dense layer
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class TransformerBlock(nn.Module):
    """
    Megatron-style transformer block with Pre-Layer Normalization
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> Residual
        -> LayerNorm -> FeedForward -> Dropout -> Residual
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Layer normalization before attention
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
        )
        
        # Layer normalization before feed-forward
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network
        self.mlp = FeedForwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections (Pre-LN)
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, seq_length, seq_length) optional
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        
        # Self-attention with residual connection (Pre-LN)
        ln_hidden = self.ln_1(hidden_states)
        attn_output, attention_weights = self.attention(ln_hidden, attention_mask)
        hidden_states = hidden_states + attn_output
        
        # Feed-forward with residual connection (Pre-LN)
        ln_hidden = self.ln_2(hidden_states)
        mlp_output = self.mlp(ln_hidden)
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, attention_weights


class MegatronTransformer(nn.Module):
    """
    Full transformer model composed of multiple transformer blocks
    Includes embeddings and language model head
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 2048,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_blocks)
        ])
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the full transformer
        
        Args:
            input_ids: (batch_size, seq_length) - token indices
            attention_mask: (batch_size, seq_length) optional - 1 for attend, 0 for mask
        
        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        # token_embeds: (batch_size, seq_length, hidden_size)
        
        # Position embeddings
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        # position_embeds: (1, seq_length, hidden_size)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Prepare attention mask if provided
        if attention_mask is not None:
            # Convert 2D mask to 4D attention mask for broadcasting
            # attention_mask: (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
            attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        else:
            attn_mask = None
        
        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states, _ = block(hidden_states, attn_mask)
        
        # Final layer normalization
        hidden_states = self.ln_final(hidden_states)
        
        # Output projection to vocabulary
        logits = self.lm_head(hidden_states)
        # logits: (batch_size, seq_length, vocab_size)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
