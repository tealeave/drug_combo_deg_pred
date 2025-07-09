"""
Core model architecture for drug combination prediction.
Based on the interview discussion about symmetry-aware neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .autoencoder import GeneExpressionAutoencoder
from .attention_layers import SelfAttention, CrossAttention, create_attention_layer


class DrugCombinationPredictor(nn.Module):
    """
    Symmetry-aware neural network for predicting drug combination effects.
    
    Handles the symmetry constraint: f(drug_A, drug_B) = f(drug_B, drug_A)
    """
    
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dims: List[int] = [64, 128, 64],
        output_dim: int = 20,  # Will match latent_dim for autoencoder
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        fusion_method: str = "add"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.fusion_method = fusion_method
        
        # Input processing layers
        self.input_norm = nn.LayerNorm(latent_dim)
        
        # Determine input size based on fusion method
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
        Forward pass through the drug combination predictor.
        
        Args:
            drug_a: Latent representation of drug A
            drug_b: Latent representation of drug B
            
        Returns:
            Predicted combination latent representation
        """
        # Normalize inputs
        drug_a = self.input_norm(drug_a)
        drug_b = self.input_norm(drug_b)
        
        # Apply self-attention if enabled
        if self.use_attention:
            # Add sequence dimension for attention
            drug_a_seq = drug_a.unsqueeze(1)
            drug_b_seq = drug_b.unsqueeze(1)
            
            # Apply self-attention
            drug_a_attended = self.self_attention(drug_a_seq).squeeze(1)
            drug_b_attended = self.self_attention(drug_b_seq).squeeze(1)
        else:
            drug_a_attended = drug_a
            drug_b_attended = drug_b
        
        # Symmetric fusion
        fused = self.symmetric_fusion(drug_a_attended, drug_b_attended)
        
        # Predict combination
        prediction = self.predictor(fused)
        
        # Apply residual connection if enabled
        if self.use_residual:
            # Calculate additive baseline
            additive_baseline = drug_a_attended + drug_b_attended
            
            # Gate mechanism
            gate = torch.sigmoid(self.residual_gate(additive_baseline))
            
            # Combine prediction with additive baseline
            prediction = gate * prediction + (1 - gate) * additive_baseline
        
        return prediction


class FullDrugCombinationModel(nn.Module):
    """
    Complete drug combination prediction model.
    
    Combines autoencoder for dimensionality reduction with combination predictor
    for end-to-end training and prediction.
    """
    
    def __init__(
        self,
        gene_dim: int = 5000,
        latent_dim: int = 20,
        autoencoder_hidden: List[int] = [1000, 200],
        predictor_hidden: List[int] = [64, 128, 64],
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        fusion_method: str = "add"
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.latent_dim = latent_dim
        
        # Autoencoder for dimensionality reduction
        self.autoencoder = GeneExpressionAutoencoder(
            input_dim=gene_dim,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout_rate=dropout_rate
        )
        
        # Drug combination predictor
        self.predictor = DrugCombinationPredictor(
            latent_dim=latent_dim,
            hidden_dims=predictor_hidden,
            output_dim=latent_dim,
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method
        )
        
    def forward(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            drug_a: Gene expression profile of drug A
            drug_b: Gene expression profile of drug B
            
        Returns:
            Predicted combination gene expression profile
        """
        # Encode to latent space
        _, latent_a = self.autoencoder(drug_a)
        _, latent_b = self.autoencoder(drug_b)
        
        # Predict combination in latent space
        combination_latent = self.predictor(latent_a, latent_b)
        
        # Decode back to gene expression space
        combination_expression = self.autoencoder.decode(combination_latent)
        
        return combination_expression
    
    def encode_drugs(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode drugs to latent representations.
        
        Args:
            drug_a: Gene expression profile of drug A
            drug_b: Gene expression profile of drug B
            
        Returns:
            Tuple of latent representations (latent_a, latent_b)
        """
        _, latent_a = self.autoencoder(drug_a)
        _, latent_b = self.autoencoder(drug_b)
        return latent_a, latent_b
    
    def predict_combination_latent(
        self, 
        latent_a: torch.Tensor, 
        latent_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict combination in latent space.
        
        Args:
            latent_a: Latent representation of drug A
            latent_b: Latent representation of drug B
            
        Returns:
            Predicted combination latent representation
        """
        return self.predictor(latent_a, latent_b)
    
    def get_attention_weights(
        self, 
        drug_a: torch.Tensor, 
        drug_b: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            drug_a: Gene expression profile of drug A
            drug_b: Gene expression profile of drug B
            
        Returns:
            Attention weights if attention is enabled, None otherwise
        """
        if not self.predictor.use_attention:
            return None
        
        # Encode to latent space
        _, latent_a = self.autoencoder(drug_a)
        _, latent_b = self.autoencoder(drug_b)
        
        # Get attention weights from predictor
        # This would need to be implemented in the attention layers
        # to return attention weights along with the output
        return None  # Placeholder
    
    def freeze_autoencoder(self):
        """Freeze autoencoder parameters for fine-tuning."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def unfreeze_autoencoder(self):
        """Unfreeze autoencoder parameters."""
        for param in self.autoencoder.parameters():
            param.requires_grad = True


def create_model(
    model_type: str = "full",
    gene_dim: int = 5000,
    latent_dim: int = 20,
    autoencoder_hidden: List[int] = [1000, 200],
    predictor_hidden: List[int] = [64, 128, 64],
    use_attention: bool = True,
    dropout_rate: float = 0.1,
    fusion_method: str = "add",
    **kwargs
) -> nn.Module:
    """
    Factory function to create drug combination prediction models.
    
    Args:
        model_type: Type of model ('full', 'autoencoder', 'predictor')
        gene_dim: Number of genes in expression profile
        latent_dim: Size of latent representation
        autoencoder_hidden: Hidden dimensions for autoencoder
        predictor_hidden: Hidden dimensions for predictor
        use_attention: Whether to use attention mechanisms
        dropout_rate: Dropout rate
        fusion_method: Method for fusing drug representations
        **kwargs: Additional arguments
        
    Returns:
        Model instance
    """
    if model_type == "full":
        return FullDrugCombinationModel(
            gene_dim=gene_dim,
            latent_dim=latent_dim,
            autoencoder_hidden=autoencoder_hidden,
            predictor_hidden=predictor_hidden,
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method
        )
    elif model_type == "autoencoder":
        return GeneExpressionAutoencoder(
            input_dim=gene_dim,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout_rate=dropout_rate,
            **kwargs
        )
    elif model_type == "predictor":
        return DrugCombinationPredictor(
            latent_dim=latent_dim,
            hidden_dims=predictor_hidden,
            output_dim=latent_dim,
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")