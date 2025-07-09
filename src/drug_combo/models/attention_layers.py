"""
Attention mechanisms for drug combination prediction models.
Extracted from prediction_model.py and enhanced with additional attention types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for gene expression features.
    
    Allows the model to attend to different parts of the gene expression
    profile to capture gene-gene interactions and dependencies.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention layer.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            Attended output tensor
        """
        batch_size, seq_len, dim = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return self.out(out)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for drug-drug interactions.
    
    Allows one drug's representation to attend to another drug's representation
    to capture interaction patterns between different drugs.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-attention layer.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention.
        
        Args:
            query: Query tensor (drug A representation)
            key: Key tensor (drug B representation)
            value: Value tensor (drug B representation)
            mask: Optional attention mask
            
        Returns:
            Attended output tensor
        """
        batch_size, seq_len, dim = query.shape
        
        # Generate Q, K, V
        q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return self.out(out)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with residual connections and layer normalization.
    
    A complete attention block that can be used as a building block for transformer
    architectures in drug combination prediction.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.attention = SelfAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with residual connection and layer normalization.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Output tensor with residual connection
        """
        # Pre-layer normalization
        normed_x = self.norm(x)
        
        # Attention
        attended = self.attention(normed_x, mask)
        
        # Residual connection
        return x + self.dropout(attended)


class GeneAttention(nn.Module):
    """
    Specialized attention mechanism for gene expression data.
    
    Incorporates biological knowledge about gene interactions and pathways
    into the attention mechanism.
    """
    
    def __init__(
        self, 
        gene_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        use_positional_encoding: bool = False
    ):
        """
        Initialize gene attention.
        
        Args:
            gene_dim: Number of genes
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        self.gene_dim = gene_dim
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        
        # Gene-specific attention weights
        self.attention = MultiHeadAttention(gene_dim, num_heads, dropout)
        
        # Positional encoding for gene ordering (if needed)
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.randn(1, gene_dim, gene_dim))
        
        # Gene importance weights
        self.gene_importance = nn.Parameter(torch.ones(gene_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gene attention.
        
        Args:
            x: Gene expression tensor of shape (batch_size, gene_dim)
            
        Returns:
            Attended gene expression
        """
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, gene_dim)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = x + self.positional_encoding
        
        # Apply attention
        attended = self.attention(x)
        
        # Remove sequence dimension
        attended = attended.squeeze(1)
        
        # Apply gene importance weights
        attended = attended * self.gene_importance
        
        return attended


class DrugInteractionAttention(nn.Module):
    """
    Attention mechanism specifically designed for drug-drug interactions.
    
    Captures complex interaction patterns between two drugs by using
    both self-attention and cross-attention mechanisms.
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        interaction_types: int = 3  # additive, synergistic, antagonistic
    ):
        """
        Initialize drug interaction attention.
        
        Args:
            latent_dim: Latent dimension of drug representations
            num_heads: Number of attention heads
            dropout: Dropout rate
            interaction_types: Number of interaction types to model
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.interaction_types = interaction_types
        
        # Self-attention for individual drugs
        self.drug_a_attention = SelfAttention(latent_dim, num_heads, dropout)
        self.drug_b_attention = SelfAttention(latent_dim, num_heads, dropout)
        
        # Cross-attention for drug interactions
        self.cross_attention_ab = CrossAttention(latent_dim, num_heads, dropout)
        self.cross_attention_ba = CrossAttention(latent_dim, num_heads, dropout)
        
        # Interaction type classification
        self.interaction_classifier = nn.Linear(latent_dim * 2, interaction_types)
        
        # Gating mechanism for interaction strength
        self.interaction_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through drug interaction attention.
        
        Args:
            drug_a: Drug A latent representation
            drug_b: Drug B latent representation
            
        Returns:
            Tuple of (fused_representation, interaction_logits)
        """
        batch_size, latent_dim = drug_a.shape
        
        # Add sequence dimension for attention
        drug_a_seq = drug_a.unsqueeze(1)
        drug_b_seq = drug_b.unsqueeze(1)
        
        # Self-attention for individual drugs
        drug_a_attended = self.drug_a_attention(drug_a_seq).squeeze(1)
        drug_b_attended = self.drug_b_attention(drug_b_seq).squeeze(1)
        
        # Cross-attention for interactions
        drug_a_cross = self.cross_attention_ab(
            drug_a_seq, drug_b_seq, drug_b_seq
        ).squeeze(1)
        drug_b_cross = self.cross_attention_ba(
            drug_b_seq, drug_a_seq, drug_a_seq
        ).squeeze(1)
        
        # Combine attended representations
        combined = torch.cat([drug_a_cross, drug_b_cross], dim=1)
        
        # Predict interaction type
        interaction_logits = self.interaction_classifier(combined)
        
        # Apply gating mechanism
        interaction_gate = self.interaction_gate(combined)
        
        # Fuse representations with gating
        fused = (drug_a_attended + drug_b_attended) * interaction_gate
        
        return fused, interaction_logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-style attention in gene expression data.
    
    Adds positional information to gene expression vectors to help the model
    understand the relative importance of different genes.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


def create_attention_layer(
    attention_type: str,
    dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of attention layers.
    
    Args:
        attention_type: Type of attention ('self', 'cross', 'multihead', 'gene', 'drug_interaction')
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        Attention layer module
    """
    if attention_type == "self":
        return SelfAttention(dim, num_heads, dropout)
    elif attention_type == "cross":
        return CrossAttention(dim, num_heads, dropout)
    elif attention_type == "multihead":
        return MultiHeadAttention(dim, num_heads, dropout, **kwargs)
    elif attention_type == "gene":
        return GeneAttention(dim, num_heads, dropout, **kwargs)
    elif attention_type == "drug_interaction":
        return DrugInteractionAttention(dim, num_heads, dropout, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")