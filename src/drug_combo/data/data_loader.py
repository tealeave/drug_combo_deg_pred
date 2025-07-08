"""
Data loading utilities for drug combination prediction.
Provides PyTorch DataLoader classes for efficient batch loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union
import logging
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

from .preprocessing import GeneExpressionPreprocessor, preprocess_data


class DrugCombinationDataset(Dataset):
    """
    Dataset class for drug combination prediction.
    Handles single drug expressions and drug pair combinations.
    """
    
    def __init__(
        self,
        single_drug_data: np.ndarray,
        pair_data: Dict,
        mode: str = "train",
        augment_symmetry: bool = True,
        indices: Optional[np.ndarray] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            single_drug_data: Single drug expression data (n_drugs, n_genes)
            pair_data: Dictionary containing pair information
            mode: Dataset mode ("train", "val", "test")
            augment_symmetry: Whether to augment with symmetric pairs
            indices: Specific indices to use for this dataset
        """
        self.single_drug_data = single_drug_data
        self.pair_data = pair_data
        self.mode = mode
        self.augment_symmetry = augment_symmetry
        self.logger = logging.getLogger(__name__)
        
        # Extract pair information
        self.drug_a_indices = pair_data['drug_a_indices']
        self.drug_b_indices = pair_data['drug_b_indices']
        self.pair_expressions = pair_data['expressions']
        
        # Apply indices if provided
        if indices is not None:
            self.drug_a_indices = self.drug_a_indices[indices]
            self.drug_b_indices = self.drug_b_indices[indices]
            self.pair_expressions = self.pair_expressions[indices]
        
        # Create data tensors
        self._create_tensors()
        
        self.logger.info(f"Created {mode} dataset with {len(self)} samples")
    
    def _create_tensors(self):
        """Create PyTorch tensors from data."""
        # Get single drug expressions for pairs
        drug_a_expressions = self.single_drug_data[self.drug_a_indices]
        drug_b_expressions = self.single_drug_data[self.drug_b_indices]
        
        if self.augment_symmetry:
            # Add symmetric pairs: (A,B) and (B,A)
            self.drug_a_tensor = torch.FloatTensor(
                np.concatenate([drug_a_expressions, drug_b_expressions])
            )
            self.drug_b_tensor = torch.FloatTensor(
                np.concatenate([drug_b_expressions, drug_a_expressions])
            )
            self.target_tensor = torch.FloatTensor(
                np.concatenate([self.pair_expressions, self.pair_expressions])
            )
        else:
            self.drug_a_tensor = torch.FloatTensor(drug_a_expressions)
            self.drug_b_tensor = torch.FloatTensor(drug_b_expressions)
            self.target_tensor = torch.FloatTensor(self.pair_expressions)
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.drug_a_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (drug_a_expression, drug_b_expression, target_expression)
        """
        return (
            self.drug_a_tensor[idx],
            self.drug_b_tensor[idx],
            self.target_tensor[idx]
        )
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        # Adjust index for symmetry augmentation
        if self.augment_symmetry:
            original_idx = idx % len(self.drug_a_indices)
            is_symmetric = idx >= len(self.drug_a_indices)
        else:
            original_idx = idx
            is_symmetric = False
        
        return {
            'index': idx,
            'original_index': original_idx,
            'is_symmetric': is_symmetric,
            'drug_a_idx': self.drug_a_indices[original_idx],
            'drug_b_idx': self.drug_b_indices[original_idx],
            'mode': self.mode
        }


class SingleDrugDataset(Dataset):
    """Dataset class for single drug expressions (for autoencoder training)."""
    
    def __init__(self, expressions: np.ndarray, drug_ids: Optional[np.ndarray] = None):
        """
        Initialize single drug dataset.
        
        Args:
            expressions: Single drug expression data (n_drugs, n_genes)
            drug_ids: Drug identifiers (optional)
        """
        self.expressions = torch.FloatTensor(expressions)
        self.drug_ids = drug_ids
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Created single drug dataset with {len(self)} samples")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.expressions)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single drug expression."""
        return self.expressions[idx]
    
    def get_drug_id(self, idx: int) -> Optional[str]:
        """Get drug ID for a given index."""
        if self.drug_ids is not None:
            return self.drug_ids[idx]
        return None


class DrugDataLoader:
    """
    Main data loader class that handles data loading and splitting.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data loading parameters
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)
        
        # Debug mode settings
        self.debug_mode = config.get('debug', {}).get('enabled', False)
        if self.debug_mode:
            self.batch_size = min(self.batch_size, 16)
            self.num_workers = min(self.num_workers, 2)
        
        # Data split ratios
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)
        
        # Augmentation settings
        self.augment_symmetry = config.get('use_symmetry_augmentation', True)
        
        # Dataset size limits (for debugging)
        self.max_samples = config.get('max_samples', None)
        
        # Cached data
        self.single_drug_data = None
        self.pair_data = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess data.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Tuple of (single_drug_data, pair_data)
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Use the existing preprocessing pipeline
        single_drug_data, pair_data = preprocess_data(data_path, self.config)
        
        # Limit dataset size for debugging
        if self.max_samples is not None and len(pair_data['expressions']) > self.max_samples:
            self.logger.info(f"Limiting dataset to {self.max_samples} samples for debugging")
            indices = np.random.choice(
                len(pair_data['expressions']), 
                size=self.max_samples, 
                replace=False
            )
            
            for key in pair_data:
                if isinstance(pair_data[key], np.ndarray):
                    pair_data[key] = pair_data[key][indices]
                elif isinstance(pair_data[key], list):
                    pair_data[key] = [pair_data[key][i] for i in indices]
        
        self.single_drug_data = single_drug_data
        self.pair_data = pair_data
        
        return single_drug_data, pair_data
    
    def create_data_splits(self, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation/test splits.
        
        Args:
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if self.pair_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        n_samples = len(self.pair_data['expressions'])
        indices = np.arange(n_samples)
        
        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices, 
            train_size=self.train_ratio,
            random_state=random_state
        )
        
        # Second split: val vs test
        remaining_ratio = 1 - self.train_ratio
        val_size = self.val_ratio / remaining_ratio
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=random_state
        )
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        self.logger.info(f"Data splits - Train: {len(train_indices)}, "
                        f"Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def get_single_drug_loader(self, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for single drug data (autoencoder training).
        
        Args:
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for single drug data
        """
        if self.single_drug_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        dataset = SingleDrugDataset(self.single_drug_data)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        return self._get_loader("train", self.train_indices, shuffle=True)
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        return self._get_loader("val", self.val_indices, shuffle=False)
    
    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        return self._get_loader("test", self.test_indices, shuffle=False)
    
    def _get_loader(self, mode: str, indices: np.ndarray, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader for the specified mode.
        
        Args:
            mode: Dataset mode ("train", "val", "test")
            indices: Indices to use for this dataset
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the specified mode
        """
        if self.single_drug_data is None or self.pair_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if indices is None:
            raise ValueError(f"Indices not created for {mode}. Call create_data_splits() first.")
        
        dataset = DrugCombinationDataset(
            single_drug_data=self.single_drug_data,
            pair_data=self.pair_data,
            mode=mode,
            augment_symmetry=self.augment_symmetry,
            indices=indices
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=mode == "train"  # Drop last batch only for training
        )
    
    def get_all_loaders(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Convenience method to get all data loaders.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Tuple of (single_drug_loader, train_loader, val_loader, test_loader)
        """
        # Load data
        self.load_data(data_path)
        
        # Create splits
        self.create_data_splits(random_state=self.config.get('seed', 42))
        
        # Create loaders
        single_drug_loader = self.get_single_drug_loader()
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        test_loader = self.get_test_loader()
        
        return single_drug_loader, train_loader, val_loader, test_loader
    
    def get_data_info(self) -> Dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.single_drug_data is None or self.pair_data is None:
            return {"error": "Data not loaded"}
        
        info = {
            "single_drug_shape": self.single_drug_data.shape,
            "pair_data_shape": self.pair_data['expressions'].shape,
            "n_genes": self.single_drug_data.shape[1],
            "n_single_drugs": self.single_drug_data.shape[0],
            "n_pairs": len(self.pair_data['expressions']),
            "train_samples": len(self.train_indices) if self.train_indices is not None else 0,
            "val_samples": len(self.val_indices) if self.val_indices is not None else 0,
            "test_samples": len(self.test_indices) if self.test_indices is not None else 0,
            "augment_symmetry": self.augment_symmetry,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers
        }
        
        return info


def create_debug_dataloaders(config: Dict, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create data loaders with debug settings.
    
    Args:
        config: Configuration dictionary
        data_path: Path to data directory
        
    Returns:
        Tuple of (single_drug_loader, train_loader, val_loader, test_loader)
    """
    # Override settings for debugging
    debug_config = config.copy()
    debug_config.update({
        'batch_size': 8,
        'num_workers': 2,
        'max_samples': 100,
        'use_synthetic': True
    })
    
    data_loader = DrugDataLoader(debug_config)
    return data_loader.get_all_loaders(data_path)


def load_config_and_create_loaders(config_path: str, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load configuration and create data loaders.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data directory
        
    Returns:
        Tuple of (single_drug_loader, train_loader, val_loader, test_loader)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = DrugDataLoader(config)
    return data_loader.get_all_loaders(data_path)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample configuration
    config = {
        'model': {'gene_dim': 1000},
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
        'max_samples': 100,
        'use_symmetry_augmentation': True,
        'seed': 42
    }
    
    # Test data loading
    data_loader = DrugDataLoader(config)
    
    # This would normally use real data
    print("Testing data loader with synthetic data...")
    try:
        single_drug_loader, train_loader, val_loader, test_loader = data_loader.get_all_loaders("dummy_path")
        
        print(f"Data info: {data_loader.get_data_info()}")
        print(f"Single drug loader: {len(single_drug_loader)} batches")
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Test a batch
        for batch in train_loader:
            drug_a, drug_b, target = batch
            print(f"Batch shapes - Drug A: {drug_a.shape}, Drug B: {drug_b.shape}, Target: {target.shape}")
            break
            
    except Exception as e:
        print(f"Error testing data loader: {e}")
        import traceback
        traceback.print_exc()