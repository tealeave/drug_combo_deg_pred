"""
Pytest configuration and fixtures for drug combination prediction tests.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Standard configuration for testing."""
    return {
        'model': {
            'gene_dim': 100,
            'latent_dim': 10,
            'autoencoder_hidden': [50, 20],
            'predictor_hidden': [20, 40, 20],
            'use_attention': True
        },
        'data': {
            'normalize_method': 'standard',
            'use_differential': True,
            'min_expression_threshold': 0.1,
            'max_variance_threshold': None,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
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
        },
        'batch_size': 16,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True,
        'use_symmetry_augmentation': True,
        'max_samples': 100,
        'seed': 42
    }


@pytest.fixture
def sample_single_drug_data():
    """Generate sample single drug expression data."""
    np.random.seed(42)
    n_drugs = 50
    n_genes = 100
    
    # Generate baseline
    baseline = np.random.lognormal(mean=2, sigma=1, size=(1, n_genes))
    
    # Generate drug effects
    drug_effects = []
    for i in range(n_drugs):
        effect = np.random.normal(0, 0.5, n_genes)
        effect *= (np.random.random(n_genes) < 0.1)  # Sparse effects
        drug_expr = baseline * np.exp(effect)
        drug_effects.append(drug_expr[0])
    
    expressions = np.vstack([baseline, np.array(drug_effects)])
    drug_ids = np.array(['baseline'] + [f'drug_{i}' for i in range(n_drugs)])
    
    return expressions, drug_ids


@pytest.fixture
def sample_pair_data():
    """Generate sample drug pair data."""
    np.random.seed(42)
    n_pairs = 100
    n_drugs = 50
    n_genes = 100
    
    # Random drug pairs
    drug_a_indices = np.random.randint(1, n_drugs + 1, n_pairs)  # Skip baseline
    drug_b_indices = np.random.randint(1, n_drugs + 1, n_pairs)
    
    # Generate pair expressions
    expressions = np.random.lognormal(mean=2, sigma=1, size=(n_pairs, n_genes))
    
    pair_data = {
        'expressions': expressions,
        'drug_a_indices': drug_a_indices,
        'drug_b_indices': drug_b_indices,
        'drug_a_ids': [f'drug_{i}' for i in drug_a_indices],
        'drug_b_ids': [f'drug_{i}' for i in drug_b_indices],
        'experiment_ids': [f'pair_{i}' for i in range(n_pairs)]
    }
    
    return pair_data


@pytest.fixture
def sample_prediction_data():
    """Generate sample prediction and target data for evaluation."""
    np.random.seed(42)
    n_samples, n_genes = 100, 50
    
    # Create targets
    targets = np.random.randn(n_samples, n_genes)
    
    # Create predictions with some correlation to targets
    predictions = targets + np.random.randn(n_samples, n_genes) * 0.5
    
    return predictions, targets


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    model.train.return_value = None
    model.eval.return_value = None
    model.to.return_value = model
    
    # Mock forward pass
    def mock_forward(*args, **kwargs):
        if len(args) == 2:
            batch_size = args[0].shape[0]
            return torch.randn(batch_size, 100)
        else:
            return torch.randn(5, 100)
    
    model.forward = mock_forward
    model.__call__ = mock_forward
    
    return model


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader for testing."""
    loader = MagicMock()
    
    # Mock iteration
    def mock_iter():
        for _ in range(5):  # 5 batches
            drug_a = torch.randn(8, 100)
            drug_b = torch.randn(8, 100)
            target = torch.randn(8, 100)
            yield drug_a, drug_b, target
    
    loader.__iter__ = mock_iter
    loader.__len__ = lambda: 5
    loader.batch_size = 8
    
    return loader


@pytest.fixture
def mock_single_drug_loader():
    """Create a mock single drug data loader."""
    loader = MagicMock()
    
    # Mock iteration
    def mock_iter():
        for _ in range(3):  # 3 batches
            yield torch.randn(8, 100)
    
    loader.__iter__ = mock_iter
    loader.__len__ = lambda: 3
    loader.batch_size = 8
    
    return loader


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Get the device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA is not available
@pytest.fixture(autouse=True)
def skip_gpu_tests(request):
    """Skip GPU tests if CUDA is not available."""
    if request.node.get_closest_marker('gpu'):
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_temp_config_file(temp_dir, config_dict):
        """Create a temporary configuration file."""
        import yaml
        
        config_path = Path(temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        return str(config_path)
    
    @staticmethod
    def create_temp_data_files(temp_dir, single_drug_data, pair_data):
        """Create temporary data files for testing."""
        import pandas as pd
        
        # Create single drug file
        single_df = pd.DataFrame(
            single_drug_data[0],
            index=single_drug_data[1],
            columns=[f'gene_{i}' for i in range(single_drug_data[0].shape[1])]
        )
        single_path = Path(temp_dir) / "single_drug_expressions.csv"
        single_df.to_csv(single_path)
        
        # Create pair expression file
        pair_df = pd.DataFrame(
            pair_data['expressions'],
            index=pair_data['experiment_ids'],
            columns=[f'gene_{i}' for i in range(pair_data['expressions'].shape[1])]
        )
        pair_expr_path = Path(temp_dir) / "pair_expressions.csv"
        pair_df.to_csv(pair_expr_path)
        
        # Create pair metadata file
        meta_df = pd.DataFrame({
            'drug_a_idx': pair_data['drug_a_indices'],
            'drug_b_idx': pair_data['drug_b_indices'],
            'drug_a_id': pair_data['drug_a_ids'],
            'drug_b_id': pair_data['drug_b_ids']
        })
        meta_path = Path(temp_dir) / "pair_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        
        return str(single_path), str(pair_expr_path), str(meta_path)
    
    @staticmethod
    def assert_tensor_properties(tensor, expected_shape=None, device=None, dtype=None):
        """Assert tensor properties."""
        assert isinstance(tensor, torch.Tensor)
        
        if expected_shape is not None:
            assert tensor.shape == expected_shape
        
        if device is not None:
            assert tensor.device == device
        
        if dtype is not None:
            assert tensor.dtype == dtype
        
        # Check for NaN and Inf values
        assert torch.isfinite(tensor).all(), "Tensor contains NaN or Inf values"
    
    @staticmethod
    def assert_metrics_reasonable(metrics):
        """Assert that metrics are reasonable."""
        assert isinstance(metrics, dict)
        
        # Check that all values are finite
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"Metric {key} is not finite: {value}"
        
        # Check specific metric ranges
        if 'mae' in metrics:
            assert metrics['mae'] >= 0, "MAE should be non-negative"
        
        if 'mse' in metrics:
            assert metrics['mse'] >= 0, "MSE should be non-negative"
        
        if 'r2_overall' in metrics:
            assert metrics['r2_overall'] >= -1, "R2 should be >= -1"
        
        if 'pearson_corr' in metrics:
            assert -1 <= metrics['pearson_corr'] <= 1, "Pearson correlation should be in [-1, 1]"
        
        if 'spearman_corr' in metrics:
            assert -1 <= metrics['spearman_corr'] <= 1, "Spearman correlation should be in [-1, 1]"


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils