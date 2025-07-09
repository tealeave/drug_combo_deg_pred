"""
Comprehensive evaluation utilities for drug combination prediction models.
Extracted from trainer.py for better code organization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from ..utils.metrics import calculate_metrics, plot_predictions, plot_gene_performance


class ModelEvaluator:
    """
    Comprehensive evaluation class for drug combination prediction models.
    
    Provides various evaluation metrics, visualizations, and analysis tools
    for assessing model performance on drug combination prediction tasks.
    """
    
    def __init__(self, model: nn.Module, device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Store evaluation results
        self.results = {}
        
    def evaluate_basic_metrics(
        self, 
        data_loader: DataLoader, 
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate basic regression metrics.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of basic metrics
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for drug_a, drug_b, target in data_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                
                prediction = self.model(drug_a, drug_b)
                predictions.append(prediction.cpu().numpy())
                targets.append(target.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Calculate basic metrics
        metrics = {
            'mae': mean_absolute_error(targets.flatten(), predictions.flatten()),
            'mse': mean_squared_error(targets.flatten(), predictions.flatten()),
            'rmse': np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten())),
            'r2': r2_score(targets.flatten(), predictions.flatten()),
        }
        
        # Add correlation metrics
        pearson_r, pearson_p = pearsonr(targets.flatten(), predictions.flatten())
        spearman_r, spearman_p = spearmanr(targets.flatten(), predictions.flatten())
        
        metrics.update({
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        })
        
        # Store results
        self.results[f"{dataset_name}_basic"] = metrics
        self.results[f"{dataset_name}_predictions"] = predictions
        self.results[f"{dataset_name}_targets"] = targets
        
        return metrics
    
    def evaluate_gene_wise_metrics(
        self, 
        data_loader: DataLoader, 
        dataset_name: str = "test"
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate metrics for each gene individually.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of gene-wise metrics
        """
        if f"{dataset_name}_predictions" not in self.results:
            self.evaluate_basic_metrics(data_loader, dataset_name)
        
        predictions = self.results[f"{dataset_name}_predictions"]
        targets = self.results[f"{dataset_name}_targets"]
        
        num_genes = predictions.shape[1]
        gene_metrics = {
            'mae': np.zeros(num_genes),
            'mse': np.zeros(num_genes),
            'r2': np.zeros(num_genes),
            'pearson_r': np.zeros(num_genes),
            'spearman_r': np.zeros(num_genes)
        }
        
        for i in range(num_genes):
            gene_pred = predictions[:, i]
            gene_target = targets[:, i]
            
            gene_metrics['mae'][i] = mean_absolute_error(gene_target, gene_pred)
            gene_metrics['mse'][i] = mean_squared_error(gene_target, gene_pred)
            gene_metrics['r2'][i] = r2_score(gene_target, gene_pred)
            
            pearson_r, _ = pearsonr(gene_target, gene_pred)
            spearman_r, _ = spearmanr(gene_target, gene_pred)
            
            gene_metrics['pearson_r'][i] = pearson_r
            gene_metrics['spearman_r'][i] = spearman_r
        
        # Store results
        self.results[f"{dataset_name}_gene_wise"] = gene_metrics
        
        return gene_metrics
    
    def evaluate_interaction_patterns(
        self, 
        data_loader: DataLoader, 
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate drug interaction patterns (additive, synergistic, antagonistic).
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of interaction analysis results
        """
        self.model.eval()
        predictions = []
        targets = []
        drug_a_effects = []
        drug_b_effects = []
        
        with torch.no_grad():
            for drug_a, drug_b, target in data_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                
                # Get combination prediction
                prediction = self.model(drug_a, drug_b)
                predictions.append(prediction.cpu().numpy())
                targets.append(target.numpy())
                
                # Get individual drug effects (if autoencoder is available)
                if hasattr(self.model, 'autoencoder'):
                    drug_a_reconstructed, _ = self.model.autoencoder(drug_a)
                    drug_b_reconstructed, _ = self.model.autoencoder(drug_b)
                    drug_a_effects.append(drug_a_reconstructed.cpu().numpy())
                    drug_b_effects.append(drug_b_reconstructed.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        interaction_analysis = {
            'num_samples': len(predictions),
            'mean_prediction': np.mean(predictions, axis=0),
            'std_prediction': np.std(predictions, axis=0),
            'mean_target': np.mean(targets, axis=0),
            'std_target': np.std(targets, axis=0)
        }
        
        if drug_a_effects:
            drug_a_effects = np.concatenate(drug_a_effects, axis=0)
            drug_b_effects = np.concatenate(drug_b_effects, axis=0)
            
            # Calculate additive baseline
            additive_baseline = drug_a_effects + drug_b_effects
            
            # Interaction analysis
            interaction_analysis.update({
                'additive_baseline': additive_baseline,
                'synergy_score': np.mean(targets - additive_baseline, axis=0),
                'interaction_strength': np.std(targets - additive_baseline, axis=0)
            })
            
            # Classify interactions
            synergy_threshold = 0.1
            synergy_scores = targets - additive_baseline
            
            synergistic = np.mean(synergy_scores > synergy_threshold, axis=0)
            antagonistic = np.mean(synergy_scores < -synergy_threshold, axis=0)
            additive = 1 - synergistic - antagonistic
            
            interaction_analysis.update({
                'synergistic_fraction': synergistic,
                'antagonistic_fraction': antagonistic,
                'additive_fraction': additive
            })
        
        # Store results
        self.results[f"{dataset_name}_interactions"] = interaction_analysis
        
        return interaction_analysis
    
    def evaluate_model_robustness(
        self, 
        data_loader: DataLoader, 
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
        dataset_name: str = "test"
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness to input noise.
        
        Args:
            data_loader: DataLoader for evaluation
            noise_levels: List of noise levels to test
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of robustness metrics for each noise level
        """
        self.model.eval()
        robustness_results = {}
        
        for noise_level in noise_levels:
            predictions = []
            targets = []
            
            with torch.no_grad():
                for drug_a, drug_b, target in data_loader:
                    drug_a = drug_a.to(self.device)
                    drug_b = drug_b.to(self.device)
                    
                    # Add noise to inputs
                    noise_a = torch.randn_like(drug_a) * noise_level
                    noise_b = torch.randn_like(drug_b) * noise_level
                    
                    noisy_drug_a = drug_a + noise_a
                    noisy_drug_b = drug_b + noise_b
                    
                    prediction = self.model(noisy_drug_a, noisy_drug_b)
                    predictions.append(prediction.cpu().numpy())
                    targets.append(target.numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            # Calculate metrics for this noise level
            metrics = {
                'mae': mean_absolute_error(targets.flatten(), predictions.flatten()),
                'mse': mean_squared_error(targets.flatten(), predictions.flatten()),
                'r2': r2_score(targets.flatten(), predictions.flatten()),
            }
            
            pearson_r, _ = pearsonr(targets.flatten(), predictions.flatten())
            metrics['pearson_r'] = pearson_r
            
            robustness_results[f"noise_{noise_level}"] = metrics
        
        # Store results
        self.results[f"{dataset_name}_robustness"] = robustness_results
        
        return robustness_results
    
    def compare_with_baselines(
        self, 
        data_loader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance with simple baselines.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary comparing model with baselines
        """
        if f"{dataset_name}_predictions" not in self.results:
            self.evaluate_basic_metrics(data_loader, dataset_name)
        
        predictions = self.results[f"{dataset_name}_predictions"]
        targets = self.results[f"{dataset_name}_targets"]
        
        # Get individual drug effects for baselines
        drug_a_effects = []
        drug_b_effects = []
        
        self.model.eval()
        with torch.no_grad():
            for drug_a, drug_b, target in data_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                
                if hasattr(self.model, 'autoencoder'):
                    drug_a_reconstructed, _ = self.model.autoencoder(drug_a)
                    drug_b_reconstructed, _ = self.model.autoencoder(drug_b)
                    drug_a_effects.append(drug_a_reconstructed.cpu().numpy())
                    drug_b_effects.append(drug_b_reconstructed.cpu().numpy())
        
        baseline_results = {}
        
        if drug_a_effects:
            drug_a_effects = np.concatenate(drug_a_effects, axis=0)
            drug_b_effects = np.concatenate(drug_b_effects, axis=0)
            
            # Additive baseline
            additive_pred = drug_a_effects + drug_b_effects
            baseline_results['additive'] = {
                'mae': mean_absolute_error(targets.flatten(), additive_pred.flatten()),
                'mse': mean_squared_error(targets.flatten(), additive_pred.flatten()),
                'r2': r2_score(targets.flatten(), additive_pred.flatten()),
            }
            
            # Average baseline
            average_pred = (drug_a_effects + drug_b_effects) / 2
            baseline_results['average'] = {
                'mae': mean_absolute_error(targets.flatten(), average_pred.flatten()),
                'mse': mean_squared_error(targets.flatten(), average_pred.flatten()),
                'r2': r2_score(targets.flatten(), average_pred.flatten()),
            }
            
            # Maximum baseline
            max_pred = np.maximum(drug_a_effects, drug_b_effects)
            baseline_results['maximum'] = {
                'mae': mean_absolute_error(targets.flatten(), max_pred.flatten()),
                'mse': mean_squared_error(targets.flatten(), max_pred.flatten()),
                'r2': r2_score(targets.flatten(), max_pred.flatten()),
            }
        
        # Add model performance for comparison
        baseline_results['model'] = self.results[f"{dataset_name}_basic"]
        
        # Store results
        self.results[f"{dataset_name}_baselines"] = baseline_results
        
        return baseline_results
    
    def generate_evaluation_report(
        self, 
        data_loader: DataLoader,
        dataset_name: str = "test",
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset being evaluated
            save_path: Path to save report
            
        Returns:
            Formatted evaluation report
        """
        # Run all evaluations
        basic_metrics = self.evaluate_basic_metrics(data_loader, dataset_name)
        gene_metrics = self.evaluate_gene_wise_metrics(data_loader, dataset_name)
        interaction_analysis = self.evaluate_interaction_patterns(data_loader, dataset_name)
        baseline_comparison = self.compare_with_baselines(data_loader, dataset_name)
        
        # Generate report
        report = f"""
# Drug Combination Prediction Model Evaluation Report

## Dataset: {dataset_name}

### Basic Performance Metrics
- Mean Absolute Error (MAE): {basic_metrics['mae']:.4f}
- Mean Squared Error (MSE): {basic_metrics['mse']:.4f}
- Root Mean Squared Error (RMSE): {basic_metrics['rmse']:.4f}
- R² Score: {basic_metrics['r2']:.4f}
- Pearson Correlation: {basic_metrics['pearson_r']:.4f} (p={basic_metrics['pearson_p']:.4f})
- Spearman Correlation: {basic_metrics['spearman_r']:.4f} (p={basic_metrics['spearman_p']:.4f})

### Gene-wise Performance
- Mean Gene MAE: {np.mean(gene_metrics['mae']):.4f} ± {np.std(gene_metrics['mae']):.4f}
- Mean Gene R²: {np.mean(gene_metrics['r2']):.4f} ± {np.std(gene_metrics['r2']):.4f}
- Mean Gene Pearson R: {np.mean(gene_metrics['pearson_r']):.4f} ± {np.std(gene_metrics['pearson_r']):.4f}
- Best performing genes (top 10 by R²): {np.argsort(gene_metrics['r2'])[-10:][::-1]}
- Worst performing genes (bottom 10 by R²): {np.argsort(gene_metrics['r2'])[:10]}

### Interaction Analysis
- Number of samples: {interaction_analysis['num_samples']}
- Mean prediction magnitude: {np.mean(np.abs(interaction_analysis['mean_prediction'])):.4f}
- Mean target magnitude: {np.mean(np.abs(interaction_analysis['mean_target'])):.4f}
"""
        
        if 'synergistic_fraction' in interaction_analysis:
            report += f"""
- Synergistic interactions: {np.mean(interaction_analysis['synergistic_fraction']):.2%}
- Antagonistic interactions: {np.mean(interaction_analysis['antagonistic_fraction']):.2%}
- Additive interactions: {np.mean(interaction_analysis['additive_fraction']):.2%}
"""
        
        report += f"""
### Baseline Comparison
"""
        
        for baseline_name, baseline_metrics in baseline_comparison.items():
            if baseline_name != 'model':
                report += f"- {baseline_name.capitalize()} baseline MAE: {baseline_metrics['mae']:.4f}\n"
        
        report += f"""
### Model vs Best Baseline
- Model MAE: {baseline_comparison['model']['mae']:.4f}
- Best baseline MAE: {min(m['mae'] for k, m in baseline_comparison.items() if k != 'model'):.4f}
- Improvement: {(1 - baseline_comparison['model']['mae'] / min(m['mae'] for k, m in baseline_comparison.items() if k != 'model')) * 100:.1f}%
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_evaluation_results(
        self, 
        dataset_name: str = "test",
        save_dir: Optional[str] = None
    ) -> None:
        """
        Generate evaluation plots.
        
        Args:
            dataset_name: Name of dataset to plot
            save_dir: Directory to save plots
        """
        if f"{dataset_name}_predictions" not in self.results:
            raise ValueError(f"No predictions found for dataset {dataset_name}")
        
        predictions = self.results[f"{dataset_name}_predictions"]
        targets = self.results[f"{dataset_name}_targets"]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prediction vs Target scatter plot
        axes[0, 0].scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Predictions vs True Values')
        
        # Residual plot
        residuals = predictions.flatten() - targets.flatten()
        axes[0, 1].scatter(targets.flatten(), residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Gene-wise performance (if available)
        if f"{dataset_name}_gene_wise" in self.results:
            gene_metrics = self.results[f"{dataset_name}_gene_wise"]
            axes[1, 0].hist(gene_metrics['r2'], bins=30, alpha=0.7)
            axes[1, 0].set_xlabel('R² Score')
            axes[1, 0].set_ylabel('Number of Genes')
            axes[1, 0].set_title('Distribution of Gene-wise R² Scores')
        
        # Baseline comparison (if available)
        if f"{dataset_name}_baselines" in self.results:
            baseline_results = self.results[f"{dataset_name}_baselines"]
            baseline_names = list(baseline_results.keys())
            mae_values = [baseline_results[name]['mae'] for name in baseline_names]
            
            axes[1, 1].bar(baseline_names, mae_values)
            axes[1, 1].set_ylabel('Mean Absolute Error')
            axes[1, 1].set_title('Baseline Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/evaluation_summary_{dataset_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()


def evaluate_model_on_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    dataset_name: str = "test",
    device: str = "auto",
    comprehensive: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        dataset_name: Name of dataset
        device: Device to use
        comprehensive: Whether to run comprehensive evaluation
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = ModelEvaluator(model, device)
    
    # Basic evaluation
    basic_metrics = evaluator.evaluate_basic_metrics(data_loader, dataset_name)
    
    if comprehensive:
        # Comprehensive evaluation
        evaluator.evaluate_gene_wise_metrics(data_loader, dataset_name)
        evaluator.evaluate_interaction_patterns(data_loader, dataset_name)
        evaluator.compare_with_baselines(data_loader, dataset_name)
        evaluator.generate_evaluation_report(data_loader, dataset_name)
        evaluator.plot_evaluation_results(dataset_name)
    
    return evaluator.results