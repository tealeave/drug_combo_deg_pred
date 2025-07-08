"""
Data preprocessing module for drug combination prediction.
Handles loading, cleaning, and transforming gene expression data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, Union
import logging
from pathlib import Path


class GeneExpressionPreprocessor:
    """Preprocessor for gene expression data."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.baseline_expression = None
        
        # Initialize scaler based on config
        scaler_type = config['data']['normalize_method']
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.logger = logging.getLogger(__name__)
    
    def load_single_drug_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load single drug expression data.
        
        Expected format:
        - CSV file with genes as columns, samples as rows
        - First row: baseline (no drug)
        - Remaining rows: single drug treatments
        
        Returns:
            expressions: (n_samples, n_genes) array
            drug_ids: (n_samples,) array of drug identifiers
        """
        data_file = Path(data_path) / "single_drug_expressions.csv"
        
        if not data_file.exists():
            # Generate synthetic data for demo
            self.logger.warning("Single drug data not found, generating synthetic data")
            return self._generate_synthetic_single_drug_data()
        
        df = pd.read_csv(data_file, index_col=0)
        
        # Validate data
        self._validate_expression_data(df)
        
        expressions = df.values
        drug_ids = df.index.values
        
        self.logger.info(f"Loaded single drug data: {expressions.shape}")
        return expressions, drug_ids
    
    def load_pair_data(self, data_path: str) -> Dict:
        """
        Load drug pair expression data.
        
        Expected format:
        - CSV file with expression data
        - Metadata file with drug pair information
        
        Returns:
            Dictionary containing pair information and expressions
        """
        expressions_file = Path(data_path) / "pair_expressions.csv"
        metadata_file = Path(data_path) / "pair_metadata.csv"
        
        if not expressions_file.exists() or not metadata_file.exists():
            self.logger.warning("Pair data not found, generating synthetic data")
            return self._generate_synthetic_pair_data()
        
        # Load expression data
        expr_df = pd.read_csv(expressions_file, index_col=0)
        expressions = expr_df.values
        
        # Load metadata
        meta_df = pd.read_csv(metadata_file)
        
        pair_data = {
            'expressions': expressions,
            'drug_a_indices': meta_df['drug_a_idx'].values,
            'drug_b_indices': meta_df['drug_b_idx'].values,
            'drug_a_ids': meta_df['drug_a_id'].values,
            'drug_b_ids': meta_df['drug_b_id'].values,
            'experiment_ids': expr_df.index.values
        }
        
        self.logger.info(f"Loaded pair data: {expressions.shape}")
        return pair_data
    
    def _validate_expression_data(self, df: pd.DataFrame) -> None:
        """Validate expression data quality."""
        # Check for missing values
        if df.isnull().any().any():
            self.logger.warning("Found missing values in expression data")
        
        # Check for negative values
        if (df < 0).any().any():
            self.logger.warning("Found negative values in expression data")
        
        # Check for extreme outliers
        outlier_threshold = df.quantile(0.99).max() * 10
        if (df > outlier_threshold).any().any():
            self.logger.warning("Found potential outliers in expression data")
    
    def calculate_differential_expression(
        self, 
        expressions: np.ndarray,
        baseline_idx: int = 0
    ) -> np.ndarray:
        """
        Calculate differential expression relative to baseline.
        
        Args:
            expressions: (n_samples, n_genes) expression data
            baseline_idx: Index of baseline sample
            
        Returns:
            Differential expression matrix
        """
        baseline = expressions[baseline_idx:baseline_idx+1]  # Keep dimensions
        self.baseline_expression = baseline
        
        # Calculate log2 fold change
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_expressions = np.log2(expressions + epsilon)
        log_baseline = np.log2(baseline + epsilon)
        
        differential = log_expressions - log_baseline
        
        # Remove baseline from differential (it would be all zeros)
        if baseline_idx == 0:
            differential = differential[1:]
        
        self.logger.info(f"Calculated differential expression: {differential.shape}")
        return differential
    
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize expression data.
        
        Args:
            data: Expression data to normalize
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized data
        """
        if fit:
            normalized = self.scaler.fit_transform(data)
            self.logger.info("Fitted scaler on training data")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scaler.transform(data)
        
        return normalized
    
    def filter_genes(self, expressions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter genes based on expression thresholds.
        
        Args:
            expressions: Gene expression data
            
        Returns:
            Filtered expressions and gene indices kept
        """
        min_threshold = self.config['data']['min_expression_threshold']
        max_var_threshold = self.config['data']['max_variance_threshold']
        
        # Filter by minimum expression
        mean_expression = np.mean(expressions, axis=0)
        keep_min = mean_expression >= min_threshold
        
        # Filter by variance if specified
        if max_var_threshold is not None:
            variance = np.var(expressions, axis=0)
            keep_var = variance <= max_var_threshold
            keep_genes = keep_min & keep_var
        else:
            keep_genes = keep_min
        
        filtered_expressions = expressions[:, keep_genes]
        gene_indices = np.where(keep_genes)[0]
        
        self.logger.info(
            f"Filtered genes: {expressions.shape[1]} -> {filtered_expressions.shape[1]} "
            f"({np.sum(keep_genes)} genes kept)"
        )
        
        return filtered_expressions, gene_indices
    
    def _generate_synthetic_single_drug_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic single drug data for testing."""
        n_drugs = 1000
        n_genes = self.config['model']['gene_dim']
        
        # Generate baseline expression
        baseline = np.random.lognormal(mean=2, sigma=1, size=(1, n_genes))
        
        # Generate drug effects
        drug_effects = []
        for i in range(n_drugs):
            # Some genes are affected, others are not
            effect = np.random.normal(0, 0.5, n_genes)
            # Make most genes have small effects
            effect *= (np.random.random(n_genes) < 0.1)
            drug_expr = baseline * np.exp(effect)
            drug_effects.append(drug_expr[0])
        
        expressions = np.vstack([baseline, np.array(drug_effects)])
        drug_ids = np.array(['baseline'] + [f'drug_{i}' for i in range(n_drugs)])
        
        self.logger.info("Generated synthetic single drug data")
        return expressions, drug_ids
    
    def _generate_synthetic_pair_data(self) -> Dict:
        """Generate synthetic pair data for testing."""
        n_pairs = 10000
        n_drugs = 1000
        n_genes = self.config['model']['gene_dim']
        
        # Random drug pairs
        drug_a_indices = np.random.randint(1, n_drugs + 1, n_pairs)  # Skip baseline
        drug_b_indices = np.random.randint(1, n_drugs + 1, n_pairs)
        
        # Generate synthetic pair expressions
        expressions = np.random.lognormal(mean=2, sigma=1, size=(n_pairs, n_genes))
        
        pair_data = {
            'expressions': expressions,
            'drug_a_indices': drug_a_indices,
            'drug_b_indices': drug_b_indices,
            'drug_a_ids': [f'drug_{i}' for i in drug_a_indices],
            'drug_b_ids': [f'drug_{i}' for i in drug_b_indices],
            'experiment_ids': [f'pair_{i}' for i in range(n_pairs)]
        }
        
        self.logger.info("Generated synthetic pair data")
        return pair_data


def preprocess_data(data_path: str, config: dict) -> Tuple[np.ndarray, Dict]:
    """
    Main preprocessing function.
    
    Args:
        data_path: Path to data directory
        config: Configuration dictionary
        
    Returns:
        Processed single drug data and pair data
    """
    preprocessor = GeneExpressionPreprocessor(config)
    
    # Load raw data
    single_expressions, single_drug_ids = preprocessor.load_single_drug_data(data_path)
    pair_data = preprocessor.load_pair_data(data_path)
    
    # Calculate differential expression if requested
    if config['data']['use_differential']:
        single_differential = preprocessor.calculate_differential_expression(single_expressions)
        
        # Also calculate for pairs (assuming baseline is same)
        if preprocessor.baseline_expression is not None:
            baseline = preprocessor.baseline_expression
            epsilon = 1e-8
            pair_log = np.log2(pair_data['expressions'] + epsilon)
            baseline_log = np.log2(baseline + epsilon)
            pair_differential = pair_log - baseline_log
            pair_data['expressions'] = pair_differential
        
        processed_single = single_differential
    else:
        # Remove baseline from single drug data if not using differential
        processed_single = single_expressions[1:]  # Skip baseline
    
    # Normalize data
    processed_single = preprocessor.normalize_data(processed_single, fit=True)
    pair_data['expressions'] = preprocessor.normalize_data(
        pair_data['expressions'], fit=False
    )
    
    # Filter genes if requested
    if config['data']['min_expression_threshold'] > 0:
        processed_single, gene_indices = preprocessor.filter_genes(processed_single)
        pair_data['expressions'] = pair_data['expressions'][:, gene_indices]
        pair_data['gene_indices'] = gene_indices
    
    logging.info("Data preprocessing completed")
    return processed_single, pair_data


def create_data_splits(
    pair_data: Dict, 
    config: dict
) -> Tuple[Dict, Dict, Dict]:
    """
    Create train/validation/test splits from pair data.
    
    Args:
        pair_data: Dictionary containing pair information
        config: Configuration dictionary
        
    Returns:
        Train, validation, and test data dictionaries
    """
    n_samples = len(pair_data['expressions'])
    indices = np.arange(n_samples)
    
    # First split: train vs (val + test)
    train_ratio = config['data']['train_ratio']
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=config['training']['seed']
    )
    
    # Second split: val vs test
    val_ratio = config['data']['val_ratio']
    remaining_ratio = 1 - train_ratio
    val_size = val_ratio / remaining_ratio
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=config['training']['seed']
    )
    
    # Create data splits
    train_data = {key: value[train_indices] if isinstance(value, np.ndarray) 
                  else [value[i] for i in train_indices] 
                  for key, value in pair_data.items()}
    
    val_data = {key: value[val_indices] if isinstance(value, np.ndarray)
                else [value[i] for i in val_indices]
                for key, value in pair_data.items()}
    
    test_data = {key: value[test_indices] if isinstance(value, np.ndarray)
                 else [value[i] for i in test_indices]
                 for key, value in pair_data.items()}
    
    logging.info(f"Data splits - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_data, val_data, test_data