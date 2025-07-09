"""
Gene expression autoencoder for dimensionality reduction.
Extracted from prediction_model.py for better code organization.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class GeneExpressionAutoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of gene expression profiles.
    
    This model compresses high-dimensional gene expression data into lower-dimensional
    latent representations while preserving important biological information.
    """
    
    def __init__(
        self, 
        input_dim: int = 5000, 
        latent_dim: int = 20,
        hidden_dims: List[int] = [1000, 200],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Number of genes in input expression profile
            latent_dim: Size of latent representation
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
            
        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        prev_dim = self.latent_dim
        
        # Reverse the hidden dimensions for decoder
        for hidden_dim in reversed(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
            
        # Final layer to output dimension (no activation for regression)
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input gene expression tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to gene expression.
        
        Args:
            z: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed gene expression of shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input gene expression tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed, latent) tensors
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def reconstruction_loss(
        self, 
        x: torch.Tensor, 
        reconstructed: torch.Tensor,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss.
        
        Args:
            x: Original input
            reconstructed: Reconstructed output
            loss_type: Type of loss ('mse', 'mae', 'huber')
            
        Returns:
            Reconstruction loss
        """
        if loss_type == "mse":
            return nn.functional.mse_loss(reconstructed, x)
        elif loss_type == "mae":
            return nn.functional.l1_loss(reconstructed, x)
        elif loss_type == "huber":
            return nn.functional.huber_loss(reconstructed, x)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without gradients.
        
        Args:
            x: Input gene expression tensor
            
        Returns:
            Latent representation
        """
        with torch.no_grad():
            return self.encode(x)
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error for input.
        
        Args:
            x: Input gene expression tensor
            
        Returns:
            Reconstruction error per sample
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            return torch.mean((x - reconstructed) ** 2, dim=1)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for gene expression data.
    
    Extends the basic autoencoder with a probabilistic latent space
    for better generalization and regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 5000,
        latent_dim: int = 20,
        hidden_dims: List[int] = [1000, 200],
        dropout_rate: float = 0.1,
        beta: float = 1.0
    ):
        """
        Initialize the VAE.
        
        Args:
            input_dim: Number of genes in input expression profile
            latent_dim: Size of latent representation
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            beta: Beta parameter for KL divergence weighting
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.beta = beta
        
        # Build encoder (outputs mean and log variance)
        self.encoder = self._build_encoder()
        
        # Separate layers for mean and log variance
        encoder_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network (without final layer)."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input gene expression tensor
            
        Returns:
            Tuple of (mu, logvar) tensors
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to gene expression.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed gene expression
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input gene expression tensor
            
        Returns:
            Tuple of (reconstructed, mu, logvar) tensors
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def vae_loss(
        self, 
        x: torch.Tensor, 
        reconstructed: torch.Tensor,
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input
            reconstructed: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def create_autoencoder(
    autoencoder_type: str = "standard",
    input_dim: int = 5000,
    latent_dim: int = 20,
    hidden_dims: List[int] = [1000, 200],
    **kwargs
) -> nn.Module:
    """
    Factory function to create autoencoder models.
    
    Args:
        autoencoder_type: Type of autoencoder ('standard', 'variational')
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional arguments for specific autoencoder types
        
    Returns:
        Autoencoder model
    """
    if autoencoder_type == "standard":
        return GeneExpressionAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    elif autoencoder_type == "variational":
        return VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")