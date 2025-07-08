"""
Core model architecture for drug combination prediction.
Based on the interview discussion about symmetry-aware neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GeneExpressionAutoencoder(nn.Module):
    """Autoencoder for dimensionality reduction of gene expression profiles."""
    
    def __init__(
        self, 
        input_dim: int = 5000, 
        latent_dim: int = 20,
        hidden_dims: list = [1000, 200]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class SelfAttention(nn.Module):
    """Self-attention mechanism for gene expression features."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return self.out(out)


class DrugCombinationPredictor(nn.Module):
    """
    Symmetry-aware neural network for predicting drug combination effects.
    
    Handles the symmetry constraint: f(drug_A, drug_B) = f(drug_B, drug_A)
    """
    
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dims: list = [64, 128, 64],
        output_dim: int = 20,  # Will match latent_dim for autoencoder
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Input processing layers
        self.input_norm = nn.LayerNorm(latent_dim)
        
        # Symmetric fusion approaches
        self.fusion_method = "add"  # "concat", "add", "attention"
        
        if self.fusion_method == "concat":
            input_size = latent_dim * 2
        elif self.fusion_method == "add":
            input_size = latent_dim
        elif self.fusion_method == "attention":
            input_size = latent_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=latent_dim, 
                num_heads=4, 
                batch_first=True
            )
        
        # Self-attention for gene interactions
        if use_attention:
            self.self_attention = SelfAttention(latent_dim)
        
        # Main prediction network
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.predictor = nn.Sequential(*layers)
        
        # Residual connection for additive baseline
        self.use_residual = True
        if self.use_residual:
            self.residual_gate = nn.Linear(latent_dim, 1)
    
    def symmetric_fusion(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> torch.Tensor:
        """Fuse two drug representations while preserving symmetry."""
        
        if self.fusion_method == "concat":
            # Sort to ensure symmetry: smaller values first
            stacked = torch.stack([drug_a, drug_b], dim=1)
            sorted_drugs, _ = torch.sort(stacked, dim=1)
            return sorted_drugs.view(drug_a.shape[0], -1)
        
        elif self.fusion_method == "add":
            return drug_a + drug_b
        
        elif self.fusion_method == "attention":
            # Cross-attention between the two drugs
            combined = torch.stack([drug_a, drug_b], dim=1)  # (batch, 2, latent_dim)
            attended, _ = self.cross_attention(combined, combined, combined)
            return attended.mean(dim=1)  # Average over the two drugs
    
    def forward(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for drug combination prediction.
        
        Args:
            drug_a: Latent representation of drug A (batch_size, latent_dim)
            drug_b: Latent representation of drug B (batch_size, latent_dim)
            
        Returns:
            Predicted combination effect (batch_size, output_dim)
        """
        # Normalize inputs
        drug_a = self.input_norm(drug_a)
        drug_b = self.input_norm(drug_b)
        
        # Apply self-attention if enabled
        if self.use_attention:
            # Reshape for attention (treat features as sequence)
            drug_a_attn = drug_a.unsqueeze(1)  # (batch, 1, latent_dim)
            drug_b_attn = drug_b.unsqueeze(1)
            
            drug_a = self.self_attention(drug_a_attn).squeeze(1)
            drug_b = self.self_attention(drug_b_attn).squeeze(1)
        
        # Symmetric fusion
        fused = self.symmetric_fusion(drug_a, drug_b)
        
        # Main prediction
        prediction = self.predictor(fused)
        
        # Add residual connection for additive baseline
        if self.use_residual:
            additive_baseline = drug_a + drug_b
            gate = torch.sigmoid(self.residual_gate(additive_baseline))
            prediction = gate * prediction + (1 - gate) * additive_baseline
        
        return prediction


class FullDrugCombinationModel(nn.Module):
    """Complete model combining autoencoder and combination predictor."""
    
    def __init__(
        self,
        gene_dim: int = 5000,
        latent_dim: int = 20,
        autoencoder_hidden: list = [1000, 200],
        predictor_hidden: list = [64, 128, 64],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.autoencoder = GeneExpressionAutoencoder(
            input_dim=gene_dim,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden
        )
        
        self.predictor = DrugCombinationPredictor(
            latent_dim=latent_dim,
            hidden_dims=predictor_hidden,
            output_dim=latent_dim,
            use_attention=use_attention
        )
        
    def forward(
        self, 
        drug_a_expr: torch.Tensor, 
        drug_b_expr: torch.Tensor,
        return_latent: bool = False
    ) -> torch.Tensor:
        """
        Complete forward pass from gene expressions to combination prediction.
        
        Args:
            drug_a_expr: Gene expression for drug A (batch_size, gene_dim)
            drug_b_expr: Gene expression for drug B (batch_size, gene_dim)
            return_latent: Whether to return latent representations
            
        Returns:
            Predicted combination gene expression (batch_size, gene_dim)
        """
        # Encode to latent space
        drug_a_latent = self.autoencoder.encode(drug_a_expr)
        drug_b_latent = self.autoencoder.encode(drug_b_expr)
        
        # Predict combination in latent space
        combo_latent = self.predictor(drug_a_latent, drug_b_latent)
        
        # Decode back to gene expression space
        combo_expr = self.autoencoder.decode(combo_latent)
        
        if return_latent:
            return combo_expr, (drug_a_latent, drug_b_latent, combo_latent)
        
        return combo_expr


def create_model(config: dict) -> FullDrugCombinationModel:
    """Factory function to create model from configuration."""
    return FullDrugCombinationModel(
        gene_dim=config.get('gene_dim', 5000),
        latent_dim=config.get('latent_dim', 20),
        autoencoder_hidden=config.get('autoencoder_hidden', [1000, 200]),
        predictor_hidden=config.get('predictor_hidden', [64, 128, 64]),
        use_attention=config.get('use_attention', True)
    )