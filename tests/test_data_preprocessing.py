"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.drug_combo.data.preprocessing import (
    GeneExpressionPreprocessor,
    preprocess_data,
    create_data_splits
)


class TestGeneExpressionPreprocessor:
    """Test cases for GeneExpressionPreprocessor class."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing."""
        return {
            'model': {'gene_dim': 100},
            'data': {
                'normalize_method': 'standard',
                'use_differential': True,
                'min_expression_threshold': 0.1,
                'max_variance_threshold': None,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {'seed': 42}
        }
    
    @pytest.fixture
    def sample_expression_data(self):
        """Generate sample expression data for testing."""
        np.random.seed(42)
        n_samples, n_genes = 10, 100
        
        # Create realistic expression data (log-normal distribution)
        data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
        
        # Create DataFrame with sample names
        sample_names = [f'sample_{i}' for i in range(n_samples)]
        gene_names = [f'gene_{i}' for i in range(n_genes)]
        
        df = pd.DataFrame(data, index=sample_names, columns=gene_names)
        
        return df
    
    @pytest.fixture
    def preprocessor(self, config):
        """Create a preprocessor instance."""
        return GeneExpressionPreprocessor(config)
    
    def test_init(self, config):
        """Test preprocessor initialization."""
        preprocessor = GeneExpressionPreprocessor(config)
        
        assert preprocessor.config == config
        assert preprocessor.scaler is not None
        assert preprocessor.baseline_expression is None
        assert hasattr(preprocessor, 'logger')
    
    def test_init_different_scalers(self, config):
        """Test initialization with different scaler types."""
        # Test StandardScaler
        config['data']['normalize_method'] = 'standard'
        preprocessor = GeneExpressionPreprocessor(config)
        from sklearn.preprocessing import StandardScaler
        assert isinstance(preprocessor.scaler, StandardScaler)
        
        # Test MinMaxScaler
        config['data']['normalize_method'] = 'minmax'
        preprocessor = GeneExpressionPreprocessor(config)
        from sklearn.preprocessing import MinMaxScaler
        assert isinstance(preprocessor.scaler, MinMaxScaler)
        
        # Test RobustScaler
        config['data']['normalize_method'] = 'robust'
        preprocessor = GeneExpressionPreprocessor(config)
        from sklearn.preprocessing import RobustScaler
        assert isinstance(preprocessor.scaler, RobustScaler)
        
        # Test invalid scaler
        config['data']['normalize_method'] = 'invalid'
        with pytest.raises(ValueError, match="Unknown scaler type"):
            GeneExpressionPreprocessor(config)
    
    def test_validate_expression_data(self, preprocessor, sample_expression_data):
        """Test expression data validation."""
        # Should not raise any exceptions with valid data
        preprocessor._validate_expression_data(sample_expression_data)
        
        # Test with missing values
        data_with_nan = sample_expression_data.copy()
        data_with_nan.iloc[0, 0] = np.nan
        preprocessor._validate_expression_data(data_with_nan)  # Should log warning
        
        # Test with negative values
        data_with_negative = sample_expression_data.copy()
        data_with_negative.iloc[0, 0] = -1.0
        preprocessor._validate_expression_data(data_with_negative)  # Should log warning
    
    def test_calculate_differential_expression(self, preprocessor):
        """Test differential expression calculation."""
        # Create sample data
        np.random.seed(42)
        expressions = np.random.lognormal(mean=2, sigma=0.5, size=(5, 10))
        
        # Calculate differential expression
        differential = preprocessor.calculate_differential_expression(expressions, baseline_idx=0)
        
        # Check output shape (should exclude baseline)
        assert differential.shape == (4, 10)
        
        # Check that baseline was stored
        assert preprocessor.baseline_expression is not None
        assert preprocessor.baseline_expression.shape == (1, 10)
        
        # Check that differential is log2 fold change
        baseline = expressions[0:1]
        expected_diff = np.log2(expressions[1:] + 1e-8) - np.log2(baseline + 1e-8)
        np.testing.assert_allclose(differential, expected_diff, rtol=1e-10)
    
    def test_normalize_data(self, preprocessor):
        """Test data normalization."""
        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, 10)
        
        # Test fitting and transforming
        normalized = preprocessor.normalize_data(data, fit=True)
        
        # Check that scaler was fitted
        assert preprocessor.scaler is not None
        
        # Check normalization (should be standardized)
        assert normalized.shape == data.shape
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-10)
        
        # Test transforming without fitting
        new_data = np.random.randn(50, 10)
        normalized_new = preprocessor.normalize_data(new_data, fit=False)
        assert normalized_new.shape == new_data.shape
        
        # Test error when not fitted
        preprocessor.scaler = None
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.normalize_data(new_data, fit=False)
    
    def test_filter_genes(self, preprocessor):
        """Test gene filtering."""
        # Create sample data with some low-expression genes
        np.random.seed(42)
        data = np.random.lognormal(mean=2, sigma=1, size=(100, 20))
        
        # Set some genes to have low expression
        data[:, :5] = 0.05  # Below threshold
        
        # Filter genes
        filtered_data, gene_indices = preprocessor.filter_genes(data)
        
        # Check that low-expression genes were filtered
        assert filtered_data.shape[0] == data.shape[0]
        assert filtered_data.shape[1] < data.shape[1]
        assert len(gene_indices) == filtered_data.shape[1]
        
        # Check that kept genes have higher expression
        mean_expr = np.mean(data, axis=0)
        kept_genes = mean_expr >= preprocessor.config['data']['min_expression_threshold']
        assert len(gene_indices) == np.sum(kept_genes)
    
    def test_generate_synthetic_single_drug_data(self, preprocessor):
        """Test synthetic single drug data generation."""
        expressions, drug_ids = preprocessor._generate_synthetic_single_drug_data()
        
        # Check shapes
        assert expressions.shape == (1001, 1000)  # 1000 drugs + 1 baseline
        assert len(drug_ids) == 1001
        
        # Check that baseline is included
        assert drug_ids[0] == 'baseline'
        
        # Check that all drug IDs are unique
        assert len(set(drug_ids)) == len(drug_ids)
    
    def test_generate_synthetic_pair_data(self, preprocessor):
        """Test synthetic pair data generation."""
        pair_data = preprocessor._generate_synthetic_pair_data()
        
        # Check required keys
        required_keys = ['expressions', 'drug_a_indices', 'drug_b_indices', 
                        'drug_a_ids', 'drug_b_ids', 'experiment_ids']
        for key in required_keys:
            assert key in pair_data
        
        # Check shapes
        assert pair_data['expressions'].shape == (10000, 1000)
        assert len(pair_data['drug_a_indices']) == 10000
        assert len(pair_data['drug_b_indices']) == 10000
        
        # Check that indices are valid (> 0, skip baseline)
        assert np.all(pair_data['drug_a_indices'] > 0)
        assert np.all(pair_data['drug_b_indices'] > 0)
    
    def test_load_single_drug_data_missing_file(self, preprocessor):
        """Test loading single drug data when file is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should generate synthetic data when file doesn't exist
            expressions, drug_ids = preprocessor.load_single_drug_data(tmp_dir)
            
            assert expressions.shape == (1001, 1000)
            assert len(drug_ids) == 1001
    
    def test_load_single_drug_data_existing_file(self, preprocessor, sample_expression_data):
        """Test loading single drug data from existing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test file
            file_path = Path(tmp_dir) / "single_drug_expressions.csv"
            sample_expression_data.to_csv(file_path)
            
            # Load data
            expressions, drug_ids = preprocessor.load_single_drug_data(tmp_dir)
            
            # Check that data was loaded correctly
            assert expressions.shape == sample_expression_data.shape
            np.testing.assert_array_equal(drug_ids, sample_expression_data.index.values)
    
    def test_load_pair_data_missing_files(self, preprocessor):
        """Test loading pair data when files are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should generate synthetic data when files don't exist
            pair_data = preprocessor.load_pair_data(tmp_dir)
            
            required_keys = ['expressions', 'drug_a_indices', 'drug_b_indices', 
                           'drug_a_ids', 'drug_b_ids', 'experiment_ids']
            for key in required_keys:
                assert key in pair_data


class TestPreprocessingFunctions:
    """Test cases for preprocessing functions."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing."""
        return {
            'model': {'gene_dim': 100},
            'data': {
                'normalize_method': 'standard',
                'use_differential': True,
                'min_expression_threshold': 0.1,
                'max_variance_threshold': None,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {'seed': 42}
        }
    
    @patch('src.drug_combo.data.preprocessing.GeneExpressionPreprocessor')
    def test_preprocess_data(self, mock_preprocessor_class, config):
        """Test main preprocessing function."""
        # Create mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor_class.return_value = mock_preprocessor
        
        # Set up mock returns
        mock_single_data = np.random.randn(100, 50)
        mock_pair_data = {'expressions': np.random.randn(200, 50)}
        
        mock_preprocessor.load_single_drug_data.return_value = (mock_single_data, None)
        mock_preprocessor.load_pair_data.return_value = mock_pair_data
        mock_preprocessor.calculate_differential_expression.return_value = mock_single_data[1:]
        mock_preprocessor.normalize_data.side_effect = [mock_single_data[1:], mock_pair_data['expressions']]
        mock_preprocessor.filter_genes.return_value = (mock_single_data[1:], np.arange(50))
        
        # Test preprocessing
        single_data, pair_data = preprocess_data("dummy_path", config)
        
        # Check that preprocessor methods were called
        mock_preprocessor.load_single_drug_data.assert_called_once()
        mock_preprocessor.load_pair_data.assert_called_once()
        mock_preprocessor.calculate_differential_expression.assert_called_once()
        assert mock_preprocessor.normalize_data.call_count == 2
    
    def test_create_data_splits(self):
        """Test data splitting function."""
        # Create sample pair data
        n_samples = 100
        pair_data = {
            'expressions': np.random.randn(n_samples, 50),
            'drug_a_indices': np.random.randint(0, 50, n_samples),
            'drug_b_indices': np.random.randint(0, 50, n_samples),
            'drug_a_ids': [f'drug_a_{i}' for i in range(n_samples)],
            'drug_b_ids': [f'drug_b_{i}' for i in range(n_samples)]
        }
        
        config = {
            'data': {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1},
            'training': {'seed': 42}
        }
        
        # Create splits
        train_data, val_data, test_data = create_data_splits(pair_data, config)
        
        # Check that splits have correct sizes
        assert len(train_data['expressions']) == int(0.8 * n_samples)
        assert len(val_data['expressions']) == int(0.1 * n_samples)
        assert len(test_data['expressions']) == int(0.1 * n_samples)
        
        # Check that all samples are accounted for
        total_samples = (len(train_data['expressions']) + 
                        len(val_data['expressions']) + 
                        len(test_data['expressions']))
        assert total_samples == n_samples
        
        # Check that data types are preserved
        for data in [train_data, val_data, test_data]:
            assert isinstance(data['expressions'], np.ndarray)
            assert isinstance(data['drug_a_ids'], list)
            assert isinstance(data['drug_b_ids'], list)


class TestIntegration:
    """Integration tests for the preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        config = {
            'model': {'gene_dim': 100},
            'data': {
                'normalize_method': 'standard',
                'use_differential': True,
                'min_expression_threshold': 0.0,  # Don't filter for testing
                'max_variance_threshold': None,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {'seed': 42}
        }
        
        # This should work with synthetic data
        try:
            single_data, pair_data = preprocess_data("dummy_path", config)
            
            # Check outputs
            assert isinstance(single_data, np.ndarray)
            assert isinstance(pair_data, dict)
            assert single_data.shape[0] > 0
            assert single_data.shape[1] > 0
            assert len(pair_data['expressions']) > 0
            
            # Test data splits
            train_data, val_data, test_data = create_data_splits(pair_data, config)
            assert len(train_data['expressions']) > 0
            assert len(val_data['expressions']) > 0
            assert len(test_data['expressions']) > 0
            
        except Exception as e:
            pytest.fail(f"Full preprocessing pipeline failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])