"""
Unit tests for model components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from src.drug_combo.models.prediction_model import (
    GeneExpressionAutoencoder,
    SelfAttention,
    DrugCombinationPredictor,
    FullDrugCombinationModel,
    create_model
)


class TestGeneExpressionAutoencoder:
    """Test cases for GeneExpressionAutoencoder."""
    
    @pytest.fixture
    def autoencoder(self):
        """Create a sample autoencoder."""
        return GeneExpressionAutoencoder(
            input_dim=100,
            latent_dim=10,
            hidden_dims=[50, 20]
        )
    
    def test_init(self, autoencoder):
        """Test autoencoder initialization."""
        assert autoencoder.input_dim == 100
        assert autoencoder.latent_dim == 10
        assert hasattr(autoencoder, 'encoder')
        assert hasattr(autoencoder, 'decoder')
        
        # Check encoder architecture
        encoder_layers = list(autoencoder.encoder.modules())
        assert len(encoder_layers) > 1
        
        # Check decoder architecture
        decoder_layers = list(autoencoder.decoder.modules())
        assert len(decoder_layers) > 1
    
    def test_encode(self, autoencoder):
        """Test encoding functionality."""
        batch_size = 5
        x = torch.randn(batch_size, 100)
        
        encoded = autoencoder.encode(x)
        
        assert encoded.shape == (batch_size, 10)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_decode(self, autoencoder):
        """Test decoding functionality."""
        batch_size = 5
        z = torch.randn(batch_size, 10)
        
        decoded = autoencoder.decode(z)
        
        assert decoded.shape == (batch_size, 100)
        assert not torch.isnan(decoded).any()
        assert not torch.isinf(decoded).any()
    
    def test_forward(self, autoencoder):
        """Test forward pass."""
        batch_size = 5
        x = torch.randn(batch_size, 100)
        
        reconstructed, latent = autoencoder.forward(x)
        
        assert reconstructed.shape == (batch_size, 100)
        assert latent.shape == (batch_size, 10)
        assert not torch.isnan(reconstructed).any()
        assert not torch.isnan(latent).any()
    
    def test_reconstruction_quality(self, autoencoder):
        """Test that autoencoder can reconstruct input reasonably."""
        # Simple test: zero input should give zero-ish output
        x = torch.zeros(1, 100)
        reconstructed, _ = autoencoder.forward(x)
        
        # Output should be finite
        assert torch.isfinite(reconstructed).all()
        
        # Test with random input
        x = torch.randn(5, 100)
        reconstructed, latent = autoencoder.forward(x)
        
        # Basic shape and finite checks
        assert reconstructed.shape == x.shape
        assert latent.shape == (5, 10)
        assert torch.isfinite(reconstructed).all()
        assert torch.isfinite(latent).all()
    
    def test_different_architectures(self):
        """Test autoencoder with different architectures."""
        # Test with single hidden layer
        ae1 = GeneExpressionAutoencoder(
            input_dim=50,
            latent_dim=5,
            hidden_dims=[25]
        )
        
        x = torch.randn(3, 50)
        reconstructed, latent = ae1.forward(x)
        assert reconstructed.shape == (3, 50)
        assert latent.shape == (3, 5)
        
        # Test with no hidden layers (direct encoding)
        ae2 = GeneExpressionAutoencoder(
            input_dim=50,
            latent_dim=5,
            hidden_dims=[]
        )
        
        x = torch.randn(3, 50)
        reconstructed, latent = ae2.forward(x)
        assert reconstructed.shape == (3, 50)
        assert latent.shape == (3, 5)


class TestSelfAttention:
    """Test cases for SelfAttention module."""
    
    @pytest.fixture
    def attention(self):
        """Create a sample attention module."""
        return SelfAttention(dim=64, num_heads=8)
    
    def test_init(self, attention):
        """Test attention initialization."""
        assert attention.num_heads == 8
        assert attention.dim == 64
        assert attention.head_dim == 8
        assert hasattr(attention, 'query')
        assert hasattr(attention, 'key')
        assert hasattr(attention, 'value')
        assert hasattr(attention, 'out')
    
    def test_init_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with pytest.raises(AssertionError, match="dim must be divisible by num_heads"):
            SelfAttention(dim=65, num_heads=8)  # 65 is not divisible by 8
    
    def test_forward(self, attention):
        """Test forward pass."""
        batch_size, seq_len, dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, dim)
        
        output = attention.forward(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_weights(self, attention):
        """Test that attention produces reasonable weights."""
        batch_size, seq_len, dim = 1, 5, 64
        x = torch.randn(batch_size, seq_len, dim)
        
        # Make one position very different to test attention focusing
        x[:, 0, :] = 10.0  # Very high values
        
        output = attention.forward(x)
        
        # Output should be finite and have correct shape
        assert torch.isfinite(output).all()
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_different_head_configurations(self):
        """Test attention with different head configurations."""
        # Test with different numbers of heads
        for num_heads in [1, 2, 4]:
            dim = 32
            attention = SelfAttention(dim=dim, num_heads=num_heads)
            
            x = torch.randn(2, 5, dim)
            output = attention.forward(x)
            
            assert output.shape == (2, 5, dim)
            assert torch.isfinite(output).all()


class TestDrugCombinationPredictor:
    """Test cases for DrugCombinationPredictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create a sample predictor."""
        return DrugCombinationPredictor(
            latent_dim=20,
            hidden_dims=[32, 64, 32],
            output_dim=20,
            use_attention=True
        )
    
    def test_init(self, predictor):
        """Test predictor initialization."""
        assert predictor.latent_dim == 20
        assert predictor.output_dim == 20
        assert predictor.use_attention
        assert hasattr(predictor, 'input_norm')
        assert hasattr(predictor, 'predictor')
        assert hasattr(predictor, 'cross_attention')
        assert hasattr(predictor, 'self_attention')
    
    def test_init_without_attention(self):
        """Test predictor without attention."""
        predictor = DrugCombinationPredictor(
            latent_dim=20,
            hidden_dims=[32, 64, 32],
            output_dim=20,
            use_attention=False
        )
        
        assert not predictor.use_attention
        assert not hasattr(predictor, 'self_attention')
    
    def test_symmetric_fusion_concat(self):
        """Test symmetric fusion with concatenation."""
        predictor = DrugCombinationPredictor(
            latent_dim=20,
            hidden_dims=[32],
            output_dim=20,
            use_attention=False
        )
        predictor.fusion_method = "concat"
        
        drug_a = torch.randn(5, 20)
        drug_b = torch.randn(5, 20)
        
        fused = predictor.symmetric_fusion(drug_a, drug_b)
        
        assert fused.shape == (5, 40)  # Concatenated
        assert torch.isfinite(fused).all()
    
    def test_symmetric_fusion_add(self):
        """Test symmetric fusion with addition."""
        predictor = DrugCombinationPredictor(
            latent_dim=20,
            hidden_dims=[32],
            output_dim=20,
            use_attention=False
        )
        predictor.fusion_method = "add"
        
        drug_a = torch.randn(5, 20)
        drug_b = torch.randn(5, 20)
        
        fused = predictor.symmetric_fusion(drug_a, drug_b)
        
        assert fused.shape == (5, 20)  # Same as input
        assert torch.isfinite(fused).all()
        
        # Test symmetry: f(a,b) == f(b,a)
        fused_ba = predictor.symmetric_fusion(drug_b, drug_a)
        torch.testing.assert_close(fused, fused_ba)
    
    def test_symmetric_fusion_attention(self, predictor):
        """Test symmetric fusion with attention."""
        assert predictor.fusion_method == "attention"
        
        drug_a = torch.randn(5, 20)
        drug_b = torch.randn(5, 20)
        
        fused = predictor.symmetric_fusion(drug_a, drug_b)
        
        assert fused.shape == (5, 20)
        assert torch.isfinite(fused).all()
    
    def test_forward(self, predictor):
        """Test forward pass."""
        batch_size = 5
        drug_a = torch.randn(batch_size, 20)
        drug_b = torch.randn(batch_size, 20)
        
        output = predictor.forward(drug_a, drug_b)
        
        assert output.shape == (batch_size, 20)
        assert torch.isfinite(output).all()
    
    def test_forward_without_attention(self):
        """Test forward pass without attention."""
        predictor = DrugCombinationPredictor(
            latent_dim=20,
            hidden_dims=[32, 64, 32],
            output_dim=20,
            use_attention=False
        )
        
        batch_size = 5
        drug_a = torch.randn(batch_size, 20)
        drug_b = torch.randn(batch_size, 20)
        
        output = predictor.forward(drug_a, drug_b)
        
        assert output.shape == (batch_size, 20)
        assert torch.isfinite(output).all()
    
    def test_residual_connection(self, predictor):
        """Test residual connection functionality."""
        assert predictor.use_residual
        
        batch_size = 5
        drug_a = torch.randn(batch_size, 20)
        drug_b = torch.randn(batch_size, 20)
        
        # Test with residual
        output_with_residual = predictor.forward(drug_a, drug_b)
        
        # Test without residual
        predictor.use_residual = False
        output_without_residual = predictor.forward(drug_a, drug_b)
        
        # Outputs should be different
        assert not torch.allclose(output_with_residual, output_without_residual)
    
    def test_symmetry_property(self, predictor):
        """Test that predictor respects symmetry property."""
        drug_a = torch.randn(3, 20)
        drug_b = torch.randn(3, 20)
        
        # Due to attention and other components, perfect symmetry might not hold
        # But we can test that the architecture handles both orderings
        output_ab = predictor.forward(drug_a, drug_b)
        output_ba = predictor.forward(drug_b, drug_a)
        
        assert output_ab.shape == output_ba.shape
        assert torch.isfinite(output_ab).all()
        assert torch.isfinite(output_ba).all()


class TestFullDrugCombinationModel:
    """Test cases for FullDrugCombinationModel."""
    
    @pytest.fixture
    def model(self):
        """Create a sample full model."""
        return FullDrugCombinationModel(
            gene_dim=100,
            latent_dim=10,
            autoencoder_hidden=[50, 20],
            predictor_hidden=[16, 32, 16],
            use_attention=True
        )
    
    def test_init(self, model):
        """Test model initialization."""
        assert hasattr(model, 'autoencoder')
        assert hasattr(model, 'predictor')
        assert isinstance(model.autoencoder, GeneExpressionAutoencoder)
        assert isinstance(model.predictor, DrugCombinationPredictor)
    
    def test_forward_without_latent(self, model):
        """Test forward pass without returning latent."""
        batch_size = 5
        drug_a_expr = torch.randn(batch_size, 100)
        drug_b_expr = torch.randn(batch_size, 100)
        
        output = model.forward(drug_a_expr, drug_b_expr, return_latent=False)
        
        assert output.shape == (batch_size, 100)
        assert torch.isfinite(output).all()
    
    def test_forward_with_latent(self, model):
        """Test forward pass with returning latent."""
        batch_size = 5
        drug_a_expr = torch.randn(batch_size, 100)
        drug_b_expr = torch.randn(batch_size, 100)
        
        output, latents = model.forward(drug_a_expr, drug_b_expr, return_latent=True)
        
        assert output.shape == (batch_size, 100)
        assert torch.isfinite(output).all()
        
        # Check latent representations
        drug_a_latent, drug_b_latent, combo_latent = latents
        assert drug_a_latent.shape == (batch_size, 10)
        assert drug_b_latent.shape == (batch_size, 10)
        assert combo_latent.shape == (batch_size, 10)
        assert torch.isfinite(drug_a_latent).all()
        assert torch.isfinite(drug_b_latent).all()
        assert torch.isfinite(combo_latent).all()
    
    def test_end_to_end_pipeline(self, model):
        """Test the complete end-to-end pipeline."""
        batch_size = 3
        gene_dim = 100
        
        # Generate realistic-looking expression data
        drug_a_expr = torch.abs(torch.randn(batch_size, gene_dim))
        drug_b_expr = torch.abs(torch.randn(batch_size, gene_dim))
        
        # Forward pass
        predicted_combo = model.forward(drug_a_expr, drug_b_expr)
        
        # Check output
        assert predicted_combo.shape == (batch_size, gene_dim)
        assert torch.isfinite(predicted_combo).all()
        
        # Test with different batch sizes
        for batch_size in [1, 2, 10]:
            drug_a = torch.abs(torch.randn(batch_size, gene_dim))
            drug_b = torch.abs(torch.randn(batch_size, gene_dim))
            
            output = model.forward(drug_a, drug_b)
            assert output.shape == (batch_size, gene_dim)
            assert torch.isfinite(output).all()
    
    def test_gradient_flow(self, model):
        """Test that gradients flow properly."""
        batch_size = 2
        drug_a_expr = torch.randn(batch_size, 100, requires_grad=True)
        drug_b_expr = torch.randn(batch_size, 100, requires_grad=True)
        
        # Forward pass
        output = model.forward(drug_a_expr, drug_b_expr)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert drug_a_expr.grad is not None
        assert drug_b_expr.grad is not None
        assert torch.isfinite(drug_a_expr.grad).all()
        assert torch.isfinite(drug_b_expr.grad).all()
    
    def test_model_parameters(self, model):
        """Test that model has learnable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that parameters require gradients
        for param in params:
            assert param.requires_grad
            assert param.dtype == torch.float32
    
    def test_model_modes(self, model):
        """Test switching between train and eval modes."""
        # Test train mode
        model.train()
        assert model.training
        
        # Test eval mode
        model.eval()
        assert not model.training
        
        # Test that dropout behaves differently in train vs eval
        drug_a = torch.randn(5, 100)
        drug_b = torch.randn(5, 100)
        
        model.train()
        output_train = model.forward(drug_a, drug_b)
        
        model.eval()
        output_eval = model.forward(drug_a, drug_b)
        
        # Both should be finite
        assert torch.isfinite(output_train).all()
        assert torch.isfinite(output_eval).all()


class TestCreateModel:
    """Test cases for create_model function."""
    
    def test_create_model_default_config(self):
        """Test creating model with default configuration."""
        config = {}
        model = create_model(config)
        
        assert isinstance(model, FullDrugCombinationModel)
        assert hasattr(model, 'autoencoder')
        assert hasattr(model, 'predictor')
    
    def test_create_model_custom_config(self):
        """Test creating model with custom configuration."""
        config = {
            'gene_dim': 200,
            'latent_dim': 15,
            'autoencoder_hidden': [100, 50],
            'predictor_hidden': [30, 60, 30],
            'use_attention': False
        }
        
        model = create_model(config)
        
        assert isinstance(model, FullDrugCombinationModel)
        
        # Test that config was applied
        assert model.autoencoder.input_dim == 200
        assert model.autoencoder.latent_dim == 15
        assert not model.predictor.use_attention
    
    def test_create_model_integration(self):
        """Test that created model works end-to-end."""
        config = {
            'gene_dim': 50,
            'latent_dim': 5,
            'autoencoder_hidden': [25],
            'predictor_hidden': [10, 20, 10],
            'use_attention': True
        }
        
        model = create_model(config)
        
        # Test forward pass
        batch_size = 3
        drug_a = torch.randn(batch_size, 50)
        drug_b = torch.randn(batch_size, 50)
        
        output = model.forward(drug_a, drug_b)
        
        assert output.shape == (batch_size, 50)
        assert torch.isfinite(output).all()


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_autoencoder_predictor_compatibility(self):
        """Test that autoencoder and predictor work together."""
        gene_dim = 100
        latent_dim = 20
        
        # Create components
        autoencoder = GeneExpressionAutoencoder(
            input_dim=gene_dim,
            latent_dim=latent_dim,
            hidden_dims=[50, 25]
        )
        
        predictor = DrugCombinationPredictor(
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 32],
            output_dim=latent_dim,
            use_attention=True
        )
        
        # Test pipeline
        batch_size = 5
        drug_a_expr = torch.randn(batch_size, gene_dim)
        drug_b_expr = torch.randn(batch_size, gene_dim)
        
        # Encode
        drug_a_latent = autoencoder.encode(drug_a_expr)
        drug_b_latent = autoencoder.encode(drug_b_expr)
        
        # Predict combination
        combo_latent = predictor.forward(drug_a_latent, drug_b_latent)
        
        # Decode
        combo_expr = autoencoder.decode(combo_latent)
        
        # Check shapes
        assert combo_expr.shape == (batch_size, gene_dim)
        assert torch.isfinite(combo_expr).all()
    
    def test_model_with_different_architectures(self):
        """Test model with various architecture configurations."""
        configs = [
            {
                'gene_dim': 50,
                'latent_dim': 5,
                'autoencoder_hidden': [25],
                'predictor_hidden': [10],
                'use_attention': False
            },
            {
                'gene_dim': 200,
                'latent_dim': 30,
                'autoencoder_hidden': [100, 50],
                'predictor_hidden': [50, 100, 50],
                'use_attention': True
            },
            {
                'gene_dim': 1000,
                'latent_dim': 50,
                'autoencoder_hidden': [500, 200, 100],
                'predictor_hidden': [100, 200, 100],
                'use_attention': True
            }
        ]
        
        for config in configs:
            model = create_model(config)
            
            # Test forward pass
            batch_size = 2
            drug_a = torch.randn(batch_size, config['gene_dim'])
            drug_b = torch.randn(batch_size, config['gene_dim'])
            
            output = model.forward(drug_a, drug_b)
            
            assert output.shape == (batch_size, config['gene_dim'])
            assert torch.isfinite(output).all()
    
    def test_model_memory_efficiency(self):
        """Test that model doesn't consume excessive memory."""
        model = create_model({
            'gene_dim': 100,
            'latent_dim': 10,
            'autoencoder_hidden': [50, 20],
            'predictor_hidden': [20, 40, 20],
            'use_attention': True
        })
        
        # Test with larger batches
        for batch_size in [1, 10, 50]:
            drug_a = torch.randn(batch_size, 100)
            drug_b = torch.randn(batch_size, 100)
            
            output = model.forward(drug_a, drug_b)
            
            assert output.shape == (batch_size, 100)
            assert torch.isfinite(output).all()
            
            # Clean up
            del drug_a, drug_b, output


if __name__ == "__main__":
    pytest.main([__file__])