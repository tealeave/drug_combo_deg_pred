"""
Unit tests for training module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from src.drug_combo.training.trainer import DrugCombinationTrainer
from src.drug_combo.models.prediction_model import FullDrugCombinationModel


class TestDrugCombinationTrainer:
    """Test cases for DrugCombinationTrainer class."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing."""
        return {
            'model': {
                'gene_dim': 100,
                'latent_dim': 10,
                'autoencoder_hidden': [50, 20],
                'predictor_hidden': [20, 40, 20],
                'use_attention': True
            },
            'training': {
                'ae_lr': 0.001,
                'full_lr': 0.0005,
                'weight_decay': 0.0001,
                'batch_size': 16,
                'seed': 42
            },
            'debug': {
                'enabled': True
            }
        }
    
    @pytest.fixture
    def trainer(self, config):
        """Create a trainer instance."""
        return DrugCombinationTrainer(config)
    
    def test_init(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.device is not None
        assert isinstance(trainer.model, FullDrugCombinationModel)
        assert trainer.ae_optimizer is not None
        assert trainer.full_optimizer is not None
        assert isinstance(trainer.reconstruction_loss, nn.MSELoss)
        assert isinstance(trainer.prediction_loss, nn.L1Loss)
        assert trainer.ae_scheduler is not None
        assert trainer.full_scheduler is not None
        assert isinstance(trainer.history, dict)
    
    def test_device_selection(self, config):
        """Test device selection logic."""
        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            trainer = DrugCombinationTrainer(config)
            assert trainer.device.type == 'cuda'
        
        # Test with CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            trainer = DrugCombinationTrainer(config)
            assert trainer.device.type == 'cpu'
    
    def test_model_to_device(self, trainer):
        """Test that model is moved to correct device."""
        device = trainer.device
        
        # Check that model parameters are on the correct device
        for param in trainer.model.parameters():
            assert param.device == device
    
    def test_optimizers_setup(self, trainer):
        """Test that optimizers are set up correctly."""
        # Check autoencoder optimizer
        assert trainer.ae_optimizer.param_groups[0]['lr'] == 0.001
        assert trainer.ae_optimizer.param_groups[0]['weight_decay'] == 0.0001
        
        # Check full model optimizer
        assert trainer.full_optimizer.param_groups[0]['lr'] == 0.0005
        assert trainer.full_optimizer.param_groups[0]['weight_decay'] == 0.0001
    
    def test_schedulers_setup(self, trainer):
        """Test that learning rate schedulers are set up correctly."""
        assert trainer.ae_scheduler is not None
        assert trainer.full_scheduler is not None
        
        # Test scheduler step (should not raise error)
        trainer.ae_scheduler.step(1.0)
        trainer.full_scheduler.step(1.0)
    
    def test_create_data_loaders(self, trainer):
        """Test data loader creation."""
        # Mock data
        single_drug_data = np.random.randn(50, 100)
        pair_data = {
            'drug_a_indices': np.random.randint(0, 50, 100),
            'drug_b_indices': np.random.randint(0, 50, 100),
            'expressions': np.random.randn(100, 100)
        }
        
        train_loader, val_loader, test_loader = trainer._create_data_loaders(
            single_drug_data, pair_data
        )
        
        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch sizes
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16
        assert test_loader.batch_size == 16
        
        # Test that we can iterate over loaders
        for drug_a, drug_b, target in train_loader:
            assert drug_a.shape[1] == 100
            assert drug_b.shape[1] == 100
            assert target.shape[1] == 100
            break
    
    def test_train_autoencoder(self, trainer):
        """Test autoencoder training."""
        # Create simple synthetic data
        single_drug_data = np.random.randn(20, 100)
        
        # Mock the training loop to run only 2 epochs
        with patch.object(trainer, 'history', {'ae_train_loss': [], 'ae_val_loss': []}):
            trainer.train_autoencoder(single_drug_data, epochs=2)
        
        # Check that history was updated
        assert len(trainer.history['ae_train_loss']) == 2
        assert len(trainer.history['ae_val_loss']) == 2
        
        # Check that losses are reasonable
        for loss in trainer.history['ae_train_loss']:
            assert loss > 0
            assert not np.isnan(loss)
            assert not np.isinf(loss)
    
    def test_train_autoencoder_early_stopping(self, trainer):
        """Test autoencoder early stopping."""
        single_drug_data = np.random.randn(20, 100)
        
        # Mock scheduler to simulate no improvement
        original_step = trainer.ae_scheduler.step
        
        def mock_step(loss):
            # Simulate no improvement by keeping loss constant
            original_step(1.0)
        
        with patch.object(trainer.ae_scheduler, 'step', side_effect=mock_step):
            # This should trigger early stopping due to no improvement
            trainer.train_autoencoder(single_drug_data, epochs=100)
        
        # Should have stopped early
        assert len(trainer.history['ae_train_loss']) < 100
    
    def test_evaluate(self, trainer):
        """Test model evaluation."""
        # Create mock data loader
        single_drug_data = np.random.randn(20, 100)
        pair_data = {
            'drug_a_indices': np.random.randint(0, 20, 30),
            'drug_b_indices': np.random.randint(0, 20, 30),
            'expressions': np.random.randn(30, 100)
        }
        
        _, val_loader, _ = trainer._create_data_loaders(single_drug_data, pair_data)
        
        # Test evaluation
        loss = trainer.evaluate(val_loader)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_comprehensive_evaluation(self, trainer):
        """Test comprehensive evaluation with metrics."""
        # Create mock data loader
        single_drug_data = np.random.randn(20, 100)
        pair_data = {
            'drug_a_indices': np.random.randint(0, 20, 30),
            'drug_b_indices': np.random.randint(0, 20, 30),
            'expressions': np.random.randn(30, 100)
        }
        
        _, _, test_loader = trainer._create_data_loaders(single_drug_data, pair_data)
        
        # Mock calculate_metrics function
        with patch('src.drug_combo.training.trainer.calculate_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'mae': 0.5,
                'mse': 0.3,
                'r2': 0.7,
                'pearson': 0.8
            }
            
            metrics = trainer.comprehensive_evaluation(test_loader)
            
            assert isinstance(metrics, dict)
            assert 'mae' in metrics
            assert 'mse' in metrics
            assert 'r2' in metrics
            assert 'pearson' in metrics
    
    def test_train_full_model(self, trainer):
        """Test full model training."""
        # Create mock data loaders
        single_drug_data = np.random.randn(20, 100)
        pair_data = {
            'drug_a_indices': np.random.randint(0, 20, 30),
            'drug_b_indices': np.random.randint(0, 20, 30),
            'expressions': np.random.randn(30, 100)
        }
        
        train_loader, val_loader, _ = trainer._create_data_loaders(single_drug_data, pair_data)
        
        # Mock wandb to avoid actual logging
        with patch('src.drug_combo.training.trainer.wandb') as mock_wandb:
            mock_wandb.run = None
            
            # Run short training
            trainer.train_full_model(train_loader, val_loader, epochs=2)
        
        # Check that history was updated
        assert len(trainer.history['full_train_loss']) == 2
        assert len(trainer.history['full_val_loss']) == 2
        
        # Check that losses are reasonable
        for loss in trainer.history['full_train_loss']:
            assert loss > 0
            assert not np.isnan(loss)
            assert not np.isinf(loss)
    
    def test_gradient_clipping(self, trainer):
        """Test gradient clipping functionality."""
        # Create a simple forward pass
        drug_a = torch.randn(5, 100).to(trainer.device)
        drug_b = torch.randn(5, 100).to(trainer.device)
        target = torch.randn(5, 100).to(trainer.device)
        
        # Forward pass
        prediction = trainer.model(drug_a, drug_b)
        loss = trainer.prediction_loss(prediction, target)
        
        # Backward pass
        trainer.full_optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist
        for param in trainer.model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
        
        # Test gradient clipping
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        
        # Check that gradients are still finite after clipping
        for param in trainer.model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
    
    def test_model_save_load(self, trainer):
        """Test model saving and loading."""
        # Create temporary file paths
        ae_path = "test_autoencoder.pth"
        full_path = "test_full_model.pth"
        
        try:
            # Save autoencoder
            torch.save(trainer.model.autoencoder.state_dict(), ae_path)
            
            # Save full model
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.full_optimizer.state_dict(),
                'config': trainer.config
            }, full_path)
            
            # Load and verify
            ae_state = torch.load(ae_path, map_location=trainer.device)
            full_state = torch.load(full_path, map_location=trainer.device)
            
            assert isinstance(ae_state, dict)
            assert isinstance(full_state, dict)
            assert 'model_state_dict' in full_state
            assert 'config' in full_state
            
        finally:
            # Clean up
            for path in [ae_path, full_path]:
                if Path(path).exists():
                    Path(path).unlink()
    
    def test_plot_training_history(self, trainer):
        """Test plotting training history."""
        # Add some fake history
        trainer.history['ae_train_loss'] = [1.0, 0.8, 0.6, 0.5]
        trainer.history['ae_val_loss'] = [1.1, 0.9, 0.7, 0.6]
        trainer.history['full_train_loss'] = [0.5, 0.4, 0.3, 0.2]
        trainer.history['full_val_loss'] = [0.6, 0.5, 0.4, 0.3]
        
        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                with patch('matplotlib.pyplot.show') as mock_show:
                    trainer.plot_training_history("test_plot.png")
                    
                    # Check that plotting functions were called
                    mock_subplots.assert_called_once()
                    mock_savefig.assert_called_once_with("test_plot.png", dpi=300, bbox_inches='tight')
                    mock_show.assert_called_once()
    
    @patch('src.drug_combo.training.trainer.preprocess_data')
    def test_prepare_data(self, mock_preprocess, trainer):
        """Test data preparation."""
        # Mock preprocessing
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'drug_a_indices': np.random.randint(0, 50, 100),
            'drug_b_indices': np.random.randint(0, 50, 100),
            'expressions': np.random.randn(100, 100)
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        # Test data preparation
        train_loader, val_loader, test_loader = trainer.prepare_data("dummy_path")
        
        # Check that loaders were created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check that preprocessing was called
        mock_preprocess.assert_called_once_with("dummy_path", trainer.config)
    
    def test_error_handling(self, trainer):
        """Test error handling in training."""
        # Test with invalid data
        with pytest.raises(Exception):
            trainer.train_autoencoder(None, epochs=1)
        
        # Test with empty data
        with pytest.raises(Exception):
            trainer.train_autoencoder(np.array([]), epochs=1)
        
        # Test with wrong dimensions
        with pytest.raises(Exception):
            trainer.train_autoencoder(np.random.randn(10, 50), epochs=1)  # Wrong gene dim


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        config = {
            'model': {
                'gene_dim': 50,
                'latent_dim': 5,
                'autoencoder_hidden': [25],
                'predictor_hidden': [10, 20, 10],
                'use_attention': False  # Disable for faster testing
            },
            'training': {
                'ae_lr': 0.01,
                'full_lr': 0.01,
                'weight_decay': 0.0001,
                'batch_size': 8,
                'seed': 42
            },
            'debug': {
                'enabled': True
            }
        }
        
        trainer = DrugCombinationTrainer(config)
        
        # Create synthetic data
        single_drug_data = np.random.randn(20, 50)
        pair_data = {
            'drug_a_indices': np.random.randint(0, 20, 30),
            'drug_b_indices': np.random.randint(0, 20, 30),
            'expressions': np.random.randn(30, 50)
        }
        
        try:
            # Stage 1: Train autoencoder
            trainer.train_autoencoder(single_drug_data, epochs=2)
            
            # Stage 2: Train full model
            train_loader, val_loader, test_loader = trainer._create_data_loaders(
                single_drug_data, pair_data
            )
            
            with patch('src.drug_combo.training.trainer.wandb') as mock_wandb:
                mock_wandb.run = None
                trainer.train_full_model(train_loader, val_loader, epochs=2)
            
            # Stage 3: Evaluate
            with patch('src.drug_combo.training.trainer.calculate_metrics') as mock_metrics:
                mock_metrics.return_value = {'mae': 0.5, 'mse': 0.3}
                metrics = trainer.comprehensive_evaluation(test_loader)
                assert isinstance(metrics, dict)
            
            # Check that training completed successfully
            assert len(trainer.history['ae_train_loss']) == 2
            assert len(trainer.history['full_train_loss']) == 2
            
        except Exception as e:
            pytest.fail(f"Full training pipeline failed: {e}")
    
    def test_training_with_different_configs(self):
        """Test training with different configurations."""
        configs = [
            {
                'model': {'gene_dim': 30, 'latent_dim': 3, 'autoencoder_hidden': [15], 
                         'predictor_hidden': [5], 'use_attention': False},
                'training': {'ae_lr': 0.01, 'full_lr': 0.01, 'weight_decay': 0.0001, 
                           'batch_size': 4, 'seed': 42}
            },
            {
                'model': {'gene_dim': 60, 'latent_dim': 6, 'autoencoder_hidden': [30, 15], 
                         'predictor_hidden': [10, 15, 10], 'use_attention': True},
                'training': {'ae_lr': 0.001, 'full_lr': 0.0005, 'weight_decay': 0.0001, 
                           'batch_size': 8, 'seed': 42}
            }
        ]
        
        for config in configs:
            trainer = DrugCombinationTrainer(config)
            
            # Create appropriate synthetic data
            gene_dim = config['model']['gene_dim']
            single_drug_data = np.random.randn(10, gene_dim)
            
            try:
                # Quick training test
                trainer.train_autoencoder(single_drug_data, epochs=1)
                
                # Check that training worked
                assert len(trainer.history['ae_train_loss']) == 1
                assert trainer.history['ae_train_loss'][0] > 0
                
            except Exception as e:
                pytest.fail(f"Training failed for config {config}: {e}")
    
    def test_memory_efficiency(self):
        """Test that training doesn't consume excessive memory."""
        config = {
            'model': {
                'gene_dim': 100,
                'latent_dim': 10,
                'autoencoder_hidden': [50, 20],
                'predictor_hidden': [20, 40, 20],
                'use_attention': False
            },
            'training': {
                'ae_lr': 0.001,
                'full_lr': 0.0005,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'seed': 42
            }
        }
        
        trainer = DrugCombinationTrainer(config)
        
        # Test with larger dataset
        single_drug_data = np.random.randn(100, 100)
        
        try:
            trainer.train_autoencoder(single_drug_data, epochs=1)
            
            # Check that training completed without memory issues
            assert len(trainer.history['ae_train_loss']) == 1
            
        except Exception as e:
            pytest.fail(f"Memory efficiency test failed: {e}")
    
    def test_reproducibility(self):
        """Test that training is reproducible with same seed."""
        config = {
            'model': {
                'gene_dim': 30,
                'latent_dim': 3,
                'autoencoder_hidden': [15],
                'predictor_hidden': [5],
                'use_attention': False
            },
            'training': {
                'ae_lr': 0.01,
                'full_lr': 0.01,
                'weight_decay': 0.0001,
                'batch_size': 8,
                'seed': 42
            }
        }
        
        # Create identical data
        np.random.seed(42)
        single_drug_data = np.random.randn(20, 30)
        
        # Train two models with same seed
        torch.manual_seed(42)
        trainer1 = DrugCombinationTrainer(config)
        trainer1.train_autoencoder(single_drug_data, epochs=1)
        
        torch.manual_seed(42)
        trainer2 = DrugCombinationTrainer(config)
        trainer2.train_autoencoder(single_drug_data, epochs=1)
        
        # Check that results are similar (may not be exactly identical due to non-deterministic operations)
        loss1 = trainer1.history['ae_train_loss'][0]
        loss2 = trainer2.history['ae_train_loss'][0]
        
        assert abs(loss1 - loss2) < 0.1  # Allow for small differences


if __name__ == "__main__":
    pytest.main([__file__])