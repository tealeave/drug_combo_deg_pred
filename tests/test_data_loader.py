"""
Unit tests for data loader module.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock

from src.drug_combo.data.data_loader import (
    DrugCombinationDataset,
    SingleDrugDataset,
    DrugDataLoader,
    create_debug_dataloaders
)


class TestDrugCombinationDataset:
    """Test cases for DrugCombinationDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Single drug data
        single_drug_data = np.random.randn(50, 100)  # 50 drugs, 100 genes
        
        # Pair data
        n_pairs = 20
        pair_data = {
            'drug_a_indices': np.random.randint(0, 50, n_pairs),
            'drug_b_indices': np.random.randint(0, 50, n_pairs),
            'expressions': np.random.randn(n_pairs, 100),
            'drug_a_ids': [f'drug_a_{i}' for i in range(n_pairs)],
            'drug_b_ids': [f'drug_b_{i}' for i in range(n_pairs)],
            'experiment_ids': [f'exp_{i}' for i in range(n_pairs)]
        }
        
        return single_drug_data, pair_data
    
    def test_init_without_augmentation(self, sample_data):
        """Test dataset initialization without symmetry augmentation."""
        single_drug_data, pair_data = sample_data
        
        dataset = DrugCombinationDataset(
            single_drug_data=single_drug_data,
            pair_data=pair_data,
            mode="train",
            augment_symmetry=False
        )
        
        assert len(dataset) == len(pair_data['expressions'])
        assert dataset.mode == "train"
        assert not dataset.augment_symmetry
        
        # Check tensor shapes
        assert dataset.drug_a_tensor.shape == (20, 100)
        assert dataset.drug_b_tensor.shape == (20, 100)
        assert dataset.target_tensor.shape == (20, 100)
    
    def test_init_with_augmentation(self, sample_data):
        """Test dataset initialization with symmetry augmentation."""
        single_drug_data, pair_data = sample_data
        
        dataset = DrugCombinationDataset(
            single_drug_data=single_drug_data,
            pair_data=pair_data,
            mode="train",
            augment_symmetry=True
        )
        
        # Should double the dataset size due to symmetry
        assert len(dataset) == 2 * len(pair_data['expressions'])
        assert dataset.augment_symmetry
        
        # Check tensor shapes
        assert dataset.drug_a_tensor.shape == (40, 100)
        assert dataset.drug_b_tensor.shape == (40, 100)
        assert dataset.target_tensor.shape == (40, 100)
    
    def test_init_with_indices(self, sample_data):
        """Test dataset initialization with specific indices."""
        single_drug_data, pair_data = sample_data
        indices = np.array([0, 2, 4, 6, 8])  # Select subset
        
        dataset = DrugCombinationDataset(
            single_drug_data=single_drug_data,
            pair_data=pair_data,
            mode="test",
            augment_symmetry=False,
            indices=indices
        )
        
        assert len(dataset) == len(indices)
        assert dataset.drug_a_tensor.shape == (5, 100)
    
    def test_getitem(self, sample_data):
        """Test getting items from dataset."""
        single_drug_data, pair_data = sample_data
        
        dataset = DrugCombinationDataset(
            single_drug_data=single_drug_data,
            pair_data=pair_data,
            mode="train",
            augment_symmetry=False
        )
        
        # Test getting a sample
        drug_a, drug_b, target = dataset[0]
        
        assert isinstance(drug_a, torch.Tensor)
        assert isinstance(drug_b, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        
        assert drug_a.shape == (100,)
        assert drug_b.shape == (100,)
        assert target.shape == (100,)
    
    def test_get_sample_info(self, sample_data):
        """Test getting sample information."""
        single_drug_data, pair_data = sample_data
        
        dataset = DrugCombinationDataset(
            single_drug_data=single_drug_data,
            pair_data=pair_data,
            mode="train",
            augment_symmetry=True
        )
        
        # Test regular sample
        info = dataset.get_sample_info(0)
        assert info['index'] == 0
        assert info['original_index'] == 0
        assert not info['is_symmetric']
        assert info['mode'] == "train"
        
        # Test symmetric sample
        info = dataset.get_sample_info(20)  # Second half should be symmetric
        assert info['index'] == 20
        assert info['original_index'] == 0
        assert info['is_symmetric']


class TestSingleDrugDataset:
    """Test cases for SingleDrugDataset class."""
    
    @pytest.fixture
    def sample_single_drug_data(self):
        """Create sample single drug data."""
        np.random.seed(42)
        expressions = np.random.randn(50, 100)
        drug_ids = [f'drug_{i}' for i in range(50)]
        return expressions, drug_ids
    
    def test_init_without_drug_ids(self, sample_single_drug_data):
        """Test initialization without drug IDs."""
        expressions, _ = sample_single_drug_data
        
        dataset = SingleDrugDataset(expressions)
        
        assert len(dataset) == 50
        assert dataset.drug_ids is None
        assert dataset.expressions.shape == (50, 100)
    
    def test_init_with_drug_ids(self, sample_single_drug_data):
        """Test initialization with drug IDs."""
        expressions, drug_ids = sample_single_drug_data
        
        dataset = SingleDrugDataset(expressions, drug_ids)
        
        assert len(dataset) == 50
        assert dataset.drug_ids is not None
        assert len(dataset.drug_ids) == 50
    
    def test_getitem(self, sample_single_drug_data):
        """Test getting items from single drug dataset."""
        expressions, drug_ids = sample_single_drug_data
        
        dataset = SingleDrugDataset(expressions, drug_ids)
        
        # Test getting a sample
        expr = dataset[0]
        assert isinstance(expr, torch.Tensor)
        assert expr.shape == (100,)
    
    def test_get_drug_id(self, sample_single_drug_data):
        """Test getting drug ID for a sample."""
        expressions, drug_ids = sample_single_drug_data
        
        dataset = SingleDrugDataset(expressions, drug_ids)
        
        # Test with drug IDs
        drug_id = dataset.get_drug_id(0)
        assert drug_id == 'drug_0'
        
        # Test without drug IDs
        dataset_no_ids = SingleDrugDataset(expressions)
        drug_id = dataset_no_ids.get_drug_id(0)
        assert drug_id is None


class TestDrugDataLoader:
    """Test cases for DrugDataLoader class."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing."""
        return {
            'model': {'gene_dim': 100},
            'data': {
                'normalize_method': 'standard',
                'use_differential': True,
                'min_expression_threshold': 0.0,
                'max_variance_threshold': None,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {'seed': 42},
            'batch_size': 16,
            'num_workers': 2,
            'pin_memory': True,
            'persistent_workers': True,
            'use_symmetry_augmentation': True,
            'max_samples': 100,
            'debug': {'enabled': True}
        }
    
    def test_init(self, config):
        """Test data loader initialization."""
        loader = DrugDataLoader(config)
        
        assert loader.config == config
        assert loader.batch_size == 16
        assert loader.num_workers == 2
        assert loader.pin_memory
        assert loader.persistent_workers
        assert loader.augment_symmetry
        assert loader.max_samples == 100
        
        # Check debug mode adjustments
        assert loader.debug_mode
        assert loader.batch_size <= 16
        assert loader.num_workers <= 2
    
    def test_init_debug_mode(self, config):
        """Test initialization with debug mode enabled."""
        config['batch_size'] = 128
        config['num_workers'] = 8
        
        loader = DrugDataLoader(config)
        
        # Should be reduced due to debug mode
        assert loader.batch_size == 16  # capped at 16
        assert loader.num_workers == 2   # capped at 2
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_load_data(self, mock_preprocess, config):
        """Test data loading."""
        # Mock preprocessing output
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'expressions': np.random.randn(200, 100),
            'drug_a_indices': np.random.randint(0, 50, 200),
            'drug_b_indices': np.random.randint(0, 50, 200)
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        single_data, pair_data = loader.load_data("dummy_path")
        
        # Check that data was loaded
        assert loader.single_drug_data is not None
        assert loader.pair_data is not None
        assert single_data.shape == (50, 100)
        
        # Check that preprocessing was called
        mock_preprocess.assert_called_once_with("dummy_path", config)
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_load_data_with_max_samples(self, mock_preprocess, config):
        """Test data loading with sample limit."""
        # Mock preprocessing output with many samples
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'expressions': np.random.randn(1000, 100),  # More than max_samples
            'drug_a_indices': np.random.randint(0, 50, 1000),
            'drug_b_indices': np.random.randint(0, 50, 1000),
            'drug_a_ids': [f'drug_a_{i}' for i in range(1000)],
            'drug_b_ids': [f'drug_b_{i}' for i in range(1000)]
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        single_data, pair_data = loader.load_data("dummy_path")
        
        # Check that data was limited
        assert len(pair_data['expressions']) == 100  # max_samples
        assert len(pair_data['drug_a_indices']) == 100
    
    def test_create_data_splits(self, config):
        """Test creating data splits."""
        loader = DrugDataLoader(config)
        
        # Mock some data
        loader.pair_data = {
            'expressions': np.random.randn(100, 50),
            'drug_a_indices': np.random.randint(0, 50, 100),
            'drug_b_indices': np.random.randint(0, 50, 100)
        }
        
        train_idx, val_idx, test_idx = loader.create_data_splits(random_state=42)
        
        # Check splits
        assert len(train_idx) == 80  # 80% of 100
        assert len(val_idx) == 10   # 10% of 100
        assert len(test_idx) == 10  # 10% of 100
        
        # Check that indices are valid
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert len(np.unique(all_indices)) == 100
        assert np.min(all_indices) >= 0
        assert np.max(all_indices) < 100
    
    def test_create_data_splits_no_data(self, config):
        """Test creating data splits without loaded data."""
        loader = DrugDataLoader(config)
        
        with pytest.raises(ValueError, match="Data not loaded"):
            loader.create_data_splits()
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_get_single_drug_loader(self, mock_preprocess, config):
        """Test getting single drug data loader."""
        # Mock data
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {'expressions': np.random.randn(200, 100)}
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        loader.load_data("dummy_path")
        
        single_drug_loader = loader.get_single_drug_loader()
        
        assert isinstance(single_drug_loader, DataLoader)
        assert single_drug_loader.batch_size == config['batch_size']
        
        # Test a batch
        for batch in single_drug_loader:
            assert batch.shape[0] <= config['batch_size']
            assert batch.shape[1] == 100
            break
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_get_train_val_test_loaders(self, mock_preprocess, config):
        """Test getting train/val/test loaders."""
        # Mock data
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'expressions': np.random.randn(100, 100),
            'drug_a_indices': np.random.randint(0, 50, 100),
            'drug_b_indices': np.random.randint(0, 50, 100)
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        loader.load_data("dummy_path")
        loader.create_data_splits()
        
        # Test individual loaders
        train_loader = loader.get_train_loader()
        val_loader = loader.get_val_loader()
        test_loader = loader.get_test_loader()
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Test batch shapes
        for batch in train_loader:
            drug_a, drug_b, target = batch
            assert drug_a.shape[1] == 100
            assert drug_b.shape[1] == 100
            assert target.shape[1] == 100
            break
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_get_all_loaders(self, mock_preprocess, config):
        """Test getting all loaders at once."""
        # Mock data
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'expressions': np.random.randn(100, 100),
            'drug_a_indices': np.random.randint(0, 50, 100),
            'drug_b_indices': np.random.randint(0, 50, 100)
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        single_loader, train_loader, val_loader, test_loader = loader.get_all_loaders("dummy_path")
        
        assert isinstance(single_loader, DataLoader)
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
    
    @patch('src.drug_combo.data.data_loader.preprocess_data')
    def test_get_data_info(self, mock_preprocess, config):
        """Test getting data information."""
        # Mock data
        mock_single_data = np.random.randn(50, 100)
        mock_pair_data = {
            'expressions': np.random.randn(200, 100),
            'drug_a_indices': np.random.randint(0, 50, 200),
            'drug_b_indices': np.random.randint(0, 50, 200)
        }
        mock_preprocess.return_value = (mock_single_data, mock_pair_data)
        
        loader = DrugDataLoader(config)
        
        # Test without data
        info = loader.get_data_info()
        assert "error" in info
        
        # Test with data
        loader.load_data("dummy_path")
        loader.create_data_splits()
        
        info = loader.get_data_info()
        assert info['single_drug_shape'] == (50, 100)
        assert info['pair_data_shape'] == (100, 100)  # Limited by max_samples
        assert info['n_genes'] == 100
        assert info['n_single_drugs'] == 50
        assert info['train_samples'] == 80
        assert info['val_samples'] == 10
        assert info['test_samples'] == 10


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @patch('src.drug_combo.data.data_loader.DrugDataLoader')
    def test_create_debug_dataloaders(self, mock_loader_class):
        """Test creating debug data loaders."""
        # Mock loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_loaders.return_value = (None, None, None, None)
        
        config = {'batch_size': 32, 'num_workers': 4}
        
        create_debug_dataloaders(config, "dummy_path")
        
        # Check that debug config was used
        mock_loader_class.assert_called_once()
        call_args = mock_loader_class.call_args[0][0]
        assert call_args['batch_size'] == 8  # Reduced for debug
        assert call_args['num_workers'] == 2  # Reduced for debug
        assert call_args['max_samples'] == 100
        assert call_args['use_synthetic'] is True
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    @patch('src.drug_combo.data.data_loader.DrugDataLoader')
    def test_load_config_and_create_loaders(self, mock_loader_class, mock_yaml, mock_open):
        """Test loading config and creating loaders."""
        # Mock config loading
        mock_config = {'batch_size': 16}
        mock_yaml.return_value = mock_config
        
        # Mock loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_loaders.return_value = (None, None, None, None)
        
        from src.drug_combo.data.data_loader import load_config_and_create_loaders
        load_config_and_create_loaders("config.yaml", "data_path")
        
        # Check that config was loaded and used
        mock_open.assert_called_once_with("config.yaml", 'r')
        mock_loader_class.assert_called_once_with(mock_config)


class TestIntegration:
    """Integration tests for the data loading pipeline."""
    
    def test_full_data_loading_pipeline(self):
        """Test the complete data loading pipeline."""
        config = {
            'model': {'gene_dim': 100},
            'data': {
                'normalize_method': 'standard',
                'use_differential': True,
                'min_expression_threshold': 0.0,
                'max_variance_threshold': None,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {'seed': 42},
            'batch_size': 8,
            'num_workers': 0,  # No multiprocessing for tests
            'pin_memory': False,
            'persistent_workers': False,
            'use_symmetry_augmentation': True,
            'max_samples': 50,
            'debug': {'enabled': True}
        }
        
        try:
            loader = DrugDataLoader(config)
            single_loader, train_loader, val_loader, test_loader = loader.get_all_loaders("dummy_path")
            
            # Test that loaders work
            assert len(single_loader) > 0
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            assert len(test_loader) > 0
            
            # Test batch shapes
            for batch in train_loader:
                drug_a, drug_b, target = batch
                assert drug_a.dim() == 2
                assert drug_b.dim() == 2
                assert target.dim() == 2
                break
                
        except Exception as e:
            pytest.fail(f"Full data loading pipeline failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])