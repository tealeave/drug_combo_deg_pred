"""
Comprehensive evaluation script for drug combination prediction model.
Loads trained model and evaluates on test data with detailed metrics and visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Import project modules
from src.drug_combo.models.prediction_model import (
    FullDrugCombinationModel, 
    create_model
)
from src.drug_combo.data.data_loader import DrugDataLoader
from src.drug_combo.utils.metrics import (
    calculate_metrics,
    calculate_interaction_metrics,
    plot_prediction_scatter,
    plot_gene_wise_performance,
    plot_interaction_analysis,
    create_evaluation_report
)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, config: Dict, model_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
            device: Device to use for evaluation
        """
        self.config = config
        self.model_path = model_path
        self.device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize data loader
        self.data_loader = DrugDataLoader(config)
        
        # Results storage
        self.results = {}
        
        self.logger.info(f"Model evaluator initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self) -> FullDrugCombinationModel:
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
            else:
                model_config = self.config['model']
            
            # Create model
            model = create_model(model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_on_dataset(
        self, 
        data_loader: torch.utils.data.DataLoader,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: PyTorch DataLoader
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating on {dataset_name} dataset...")
        
        all_predictions = []
        all_targets = []
        all_drug_a = []
        all_drug_b = []
        
        # Loss function
        criterion = nn.L1Loss()
        total_loss = 0.0
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, (drug_a, drug_b, target) in enumerate(tqdm(data_loader)):
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                prediction = self.model(drug_a, drug_b)
                loss = criterion(prediction, target)
                total_loss += loss.item()
                
                # Store results
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_drug_a.append(drug_a.cpu().numpy())
                all_drug_b.append(drug_b.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        drug_a_data = np.concatenate(all_drug_a, axis=0)
        drug_b_data = np.concatenate(all_drug_b, axis=0)
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        basic_metrics = calculate_metrics(predictions, targets)
        
        # Calculate interaction metrics
        interaction_metrics = calculate_interaction_metrics(
            predictions, targets, drug_a_data, drug_b_data
        )
        
        # Combine metrics
        all_metrics = {
            'dataset': dataset_name,
            'avg_loss': avg_loss,
            'n_samples': len(predictions),
            'n_genes': predictions.shape[1],
            **basic_metrics,
            **interaction_metrics
        }
        
        # Store raw data for plotting
        self.results[dataset_name] = {
            'predictions': predictions,
            'targets': targets,
            'drug_a_data': drug_a_data,
            'drug_b_data': drug_b_data,
            'metrics': all_metrics
        }
        
        self.logger.info(f"Completed evaluation on {dataset_name} dataset")
        self.logger.info(f"Average loss: {avg_loss:.6f}")
        self.logger.info(f"MAE: {basic_metrics['mae']:.6f}")
        self.logger.info(f"R²: {basic_metrics['r2_overall']:.6f}")
        self.logger.info(f"Pearson correlation: {basic_metrics['pearson_corr']:.6f}")
        
        return all_metrics
    
    def evaluate_autoencoder(
        self, 
        single_drug_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Evaluate autoencoder reconstruction quality.
        
        Args:
            single_drug_loader: DataLoader for single drug data
            
        Returns:
            Dictionary with autoencoder evaluation results
        """
        self.logger.info("Evaluating autoencoder reconstruction...")
        
        all_original = []
        all_reconstructed = []
        all_latent = []
        
        criterion = nn.MSELoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(single_drug_loader):
                batch = batch.to(self.device)
                
                # Forward pass through autoencoder
                reconstructed, latent = self.model.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                total_loss += loss.item()
                
                # Store results
                all_original.append(batch.cpu().numpy())
                all_reconstructed.append(reconstructed.cpu().numpy())
                all_latent.append(latent.cpu().numpy())
        
        # Concatenate results
        original = np.concatenate(all_original, axis=0)
        reconstructed = np.concatenate(all_reconstructed, axis=0)
        latent = np.concatenate(all_latent, axis=0)
        
        # Calculate reconstruction metrics
        avg_loss = total_loss / len(single_drug_loader)
        reconstruction_metrics = calculate_metrics(reconstructed, original)
        
        # Additional autoencoder-specific metrics
        reconstruction_error = np.mean(np.abs(reconstructed - original))
        latent_variance = np.mean(np.var(latent, axis=0))
        
        ae_metrics = {
            'dataset': 'autoencoder',
            'avg_reconstruction_loss': avg_loss,
            'reconstruction_error': reconstruction_error,
            'latent_variance': latent_variance,
            'n_samples': len(original),
            'n_genes': original.shape[1],
            'latent_dim': latent.shape[1],
            **reconstruction_metrics
        }
        
        # Store results
        self.results['autoencoder'] = {
            'original': original,
            'reconstructed': reconstructed,
            'latent': latent,
            'metrics': ae_metrics
        }
        
        self.logger.info(f"Autoencoder evaluation completed")
        self.logger.info(f"Reconstruction loss: {avg_loss:.6f}")
        self.logger.info(f"Reconstruction error: {reconstruction_error:.6f}")
        
        return ae_metrics
    
    def compare_with_baselines(
        self, 
        test_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Compare model performance with simple baselines.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with baseline comparison results
        """
        self.logger.info("Comparing with baseline models...")
        
        all_predictions = []
        all_targets = []
        all_drug_a = []
        all_drug_b = []
        
        # Collect test data
        with torch.no_grad():
            for drug_a, drug_b, target in test_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                target = target.to(self.device)
                
                # Model predictions
                prediction = self.model(drug_a, drug_b)
                
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_drug_a.append(drug_a.cpu().numpy())
                all_drug_b.append(drug_b.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        drug_a_data = np.concatenate(all_drug_a, axis=0)
        drug_b_data = np.concatenate(all_drug_b, axis=0)
        
        # Baseline 1: Simple addition
        additive_baseline = drug_a_data + drug_b_data
        
        # Baseline 2: Average of inputs
        average_baseline = (drug_a_data + drug_b_data) / 2
        
        # Baseline 3: Maximum of inputs
        max_baseline = np.maximum(drug_a_data, drug_b_data)
        
        # Baseline 4: Random predictions (with same mean/std as targets)
        np.random.seed(42)
        random_baseline = np.random.normal(
            loc=targets.mean(), 
            scale=targets.std(), 
            size=targets.shape
        )
        
        # Calculate metrics for each baseline
        baselines = {
            'additive': additive_baseline,
            'average': average_baseline,
            'maximum': max_baseline,
            'random': random_baseline
        }
        
        baseline_results = {}
        model_metrics = calculate_metrics(predictions, targets)
        
        for name, baseline_pred in baselines.items():
            baseline_metrics = calculate_metrics(baseline_pred, targets)
            baseline_results[name] = baseline_metrics
        
        # Compare with model
        comparison_results = {
            'model': model_metrics,
            'baselines': baseline_results,
            'improvements': {}
        }
        
        # Calculate improvements over baselines
        for name, baseline_metrics in baseline_results.items():
            improvement = {
                'mae_improvement': (baseline_metrics['mae'] - model_metrics['mae']) / baseline_metrics['mae'] * 100,
                'r2_improvement': model_metrics['r2_overall'] - baseline_metrics['r2_overall'],
                'pearson_improvement': model_metrics['pearson_corr'] - baseline_metrics['pearson_corr']
            }
            comparison_results['improvements'][name] = improvement
        
        self.results['baseline_comparison'] = comparison_results
        
        # Log results
        self.logger.info("Baseline comparison results:")
        for name, improvement in comparison_results['improvements'].items():
            self.logger.info(f"  vs {name}: MAE improvement {improvement['mae_improvement']:.2f}%, "
                           f"R² improvement {improvement['r2_improvement']:.4f}")
        
        return comparison_results
    
    def analyze_gene_importance(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Analyze which genes are most important for predictions.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with gene importance analysis
        """
        self.logger.info("Analyzing gene importance...")
        
        all_predictions = []
        all_targets = []
        
        # Collect predictions
        with torch.no_grad():
            for drug_a, drug_b, target in test_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                target = target.to(self.device)
                
                prediction = self.model(drug_a, drug_b)
                
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate gene-wise metrics
        n_genes = predictions.shape[1]
        gene_metrics = []
        
        for gene_idx in range(n_genes):
            gene_pred = predictions[:, gene_idx]
            gene_target = targets[:, gene_idx]
            
            # Calculate metrics for this gene
            mae = np.mean(np.abs(gene_pred - gene_target))
            mse = np.mean((gene_pred - gene_target) ** 2)
            
            # Calculate R²
            ss_res = np.sum((gene_target - gene_pred) ** 2)
            ss_tot = np.sum((gene_target - np.mean(gene_target)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate correlation
            if np.std(gene_pred) > 0 and np.std(gene_target) > 0:
                correlation = np.corrcoef(gene_pred, gene_target)[0, 1]
            else:
                correlation = 0
            
            gene_metrics.append({
                'gene_idx': gene_idx,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'correlation': correlation,
                'target_variance': np.var(gene_target),
                'prediction_variance': np.var(gene_pred)
            })
        
        # Convert to DataFrame for analysis
        gene_df = pd.DataFrame(gene_metrics)
        
        # Find top/bottom performing genes
        top_genes_r2 = gene_df.nlargest(20, 'r2')
        bottom_genes_r2 = gene_df.nsmallest(20, 'r2')
        
        top_genes_corr = gene_df.nlargest(20, 'correlation')
        bottom_genes_corr = gene_df.nsmallest(20, 'correlation')
        
        # Summary statistics
        gene_analysis = {
            'summary': {
                'mean_r2': gene_df['r2'].mean(),
                'median_r2': gene_df['r2'].median(),
                'std_r2': gene_df['r2'].std(),
                'mean_correlation': gene_df['correlation'].mean(),
                'median_correlation': gene_df['correlation'].median(),
                'std_correlation': gene_df['correlation'].std(),
                'genes_with_positive_r2': (gene_df['r2'] > 0).sum(),
                'genes_with_good_r2': (gene_df['r2'] > 0.5).sum(),
                'genes_with_high_correlation': (gene_df['correlation'] > 0.7).sum()
            },
            'top_genes_r2': top_genes_r2.to_dict('records'),
            'bottom_genes_r2': bottom_genes_r2.to_dict('records'),
            'top_genes_correlation': top_genes_corr.to_dict('records'),
            'bottom_genes_correlation': bottom_genes_corr.to_dict('records'),
            'all_gene_metrics': gene_df.to_dict('records')
        }
        
        self.results['gene_analysis'] = gene_analysis
        
        self.logger.info(f"Gene analysis completed")
        self.logger.info(f"Mean gene R²: {gene_analysis['summary']['mean_r2']:.4f}")
        self.logger.info(f"Genes with positive R²: {gene_analysis['summary']['genes_with_positive_r2']}")
        self.logger.info(f"Genes with good R² (>0.5): {gene_analysis['summary']['genes_with_good_r2']}")
        
        return gene_analysis
    
    def create_visualizations(self, save_dir: str = "evaluation_results"):
        """
        Create comprehensive visualizations of evaluation results.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.logger.info("Creating visualizations...")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Test dataset visualizations
        if 'test' in self.results:
            test_data = self.results['test']
            
            # Prediction scatter plot
            plot_prediction_scatter(
                test_data['predictions'],
                test_data['targets'],
                save_path=save_path / "test_prediction_scatter.png"
            )
            
            # Gene-wise performance
            plot_gene_wise_performance(
                test_data['predictions'],
                test_data['targets'],
                save_path=save_path / "test_gene_performance.png"
            )
            
            # Interaction analysis
            plot_interaction_analysis(
                test_data['predictions'],
                test_data['targets'],
                test_data['drug_a_data'],
                test_data['drug_b_data'],
                save_path=save_path / "test_interaction_analysis.png"
            )
        
        # Autoencoder visualizations
        if 'autoencoder' in self.results:
            ae_data = self.results['autoencoder']
            
            # Reconstruction quality
            plot_prediction_scatter(
                ae_data['reconstructed'],
                ae_data['original'],
                save_path=save_path / "autoencoder_reconstruction.png"
            )
            
            # Latent space visualization
            self._plot_latent_space(ae_data['latent'], save_path / "latent_space.png")
        
        # Baseline comparison plot
        if 'baseline_comparison' in self.results:
            self._plot_baseline_comparison(save_path / "baseline_comparison.png")
        
        # Gene importance plots
        if 'gene_analysis' in self.results:
            self._plot_gene_importance(save_path / "gene_importance.png")
        
        # Training history (if available)
        if hasattr(self, 'training_history'):
            self._plot_training_history(save_path / "training_history.png")
        
        self.logger.info(f"Visualizations saved to {save_path}")
    
    def _plot_latent_space(self, latent_data: np.ndarray, save_path: Path):
        """Plot latent space visualization."""
        if latent_data.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            
            # Plot first two dimensions
            plt.scatter(latent_data[:, 0], latent_data[:, 1], alpha=0.6, s=20)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('Latent Space Visualization (First 2 Dimensions)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_baseline_comparison(self, save_path: Path):
        """Plot baseline comparison results."""
        if 'baseline_comparison' not in self.results:
            return
        
        comparison = self.results['baseline_comparison']
        
        # Extract metrics for plotting
        methods = ['model'] + list(comparison['baselines'].keys())
        mae_values = [comparison['model']['mae']]
        r2_values = [comparison['model']['r2_overall']]
        pearson_values = [comparison['model']['pearson_corr']]
        
        for baseline in comparison['baselines'].values():
            mae_values.append(baseline['mae'])
            r2_values.append(baseline['r2_overall'])
            pearson_values.append(baseline['pearson_corr'])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE comparison
        axes[0].bar(methods, mae_values)
        axes[0].set_ylabel('Mean Absolute Error')
        axes[0].set_title('MAE Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1].bar(methods, r2_values)
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('R² Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Pearson correlation comparison
        axes[2].bar(methods, pearson_values)
        axes[2].set_ylabel('Pearson Correlation')
        axes[2].set_title('Correlation Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gene_importance(self, save_path: Path):
        """Plot gene importance analysis."""
        if 'gene_analysis' not in self.results:
            return
        
        gene_analysis = self.results['gene_analysis']
        gene_df = pd.DataFrame(gene_analysis['all_gene_metrics'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² distribution
        axes[0, 0].hist(gene_df['r2'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(gene_df['r2'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {gene_df["r2"].mean():.3f}')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Number of Genes')
        axes[0, 0].set_title('Distribution of Gene-wise R² Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correlation distribution
        axes[0, 1].hist(gene_df['correlation'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(gene_df['correlation'].mean(), color='red', linestyle='--',
                          label=f'Mean: {gene_df["correlation"].mean():.3f}')
        axes[0, 1].set_xlabel('Correlation')
        axes[0, 1].set_ylabel('Number of Genes')
        axes[0, 1].set_title('Distribution of Gene-wise Correlations')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top genes by R²
        top_genes = gene_df.nlargest(20, 'r2')
        axes[1, 0].barh(range(len(top_genes)), top_genes['r2'])
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_ylabel('Gene Rank')
        axes[1, 0].set_title('Top 20 Genes by R² Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² vs Correlation scatter
        axes[1, 1].scatter(gene_df['r2'], gene_df['correlation'], alpha=0.6, s=20)
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('R² vs Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, save_dir: str = "evaluation_results"):
        """
        Save all evaluation results to files.
        
        Args:
            save_dir: Directory to save results
        """
        self.logger.info("Saving evaluation results...")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save metrics to JSON
        metrics_summary = {}
        for dataset, data in self.results.items():
            if 'metrics' in data:
                metrics_summary[dataset] = data['metrics']
        
        with open(save_path / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        # Save detailed results
        for dataset, data in self.results.items():
            if 'predictions' in data:
                # Save predictions and targets
                np.save(save_path / f"{dataset}_predictions.npy", data['predictions'])
                np.save(save_path / f"{dataset}_targets.npy", data['targets'])
            
            if 'original' in data:  # Autoencoder results
                np.save(save_path / f"{dataset}_original.npy", data['original'])
                np.save(save_path / f"{dataset}_reconstructed.npy", data['reconstructed'])
                np.save(save_path / f"{dataset}_latent.npy", data['latent'])
        
        # Save gene analysis
        if 'gene_analysis' in self.results:
            gene_df = pd.DataFrame(self.results['gene_analysis']['all_gene_metrics'])
            gene_df.to_csv(save_path / "gene_analysis.csv", index=False)
        
        # Save summary report
        self._create_summary_report(save_path / "evaluation_summary.txt")
        
        self.logger.info(f"Results saved to {save_path}")
    
    def _create_summary_report(self, save_path: Path):
        """Create a summary report of evaluation results."""
        with open(save_path, 'w') as f:
            f.write("Drug Combination Prediction - Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model information
            f.write("Model Information:\n")
            f.write(f"  - Model path: {self.model_path}\n")
            f.write(f"  - Device: {self.device}\n")
            f.write(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n\n")
            
            # Test results
            if 'test' in self.results:
                test_metrics = self.results['test']['metrics']
                f.write("Test Dataset Results:\n")
                f.write(f"  - Samples: {test_metrics['n_samples']}\n")
                f.write(f"  - Genes: {test_metrics['n_genes']}\n")
                f.write(f"  - Average Loss: {test_metrics['avg_loss']:.6f}\n")
                f.write(f"  - MAE: {test_metrics['mae']:.6f}\n")
                f.write(f"  - MSE: {test_metrics['mse']:.6f}\n")
                f.write(f"  - R²: {test_metrics['r2_overall']:.6f}\n")
                f.write(f"  - Pearson Correlation: {test_metrics['pearson_corr']:.6f}\n")
                f.write(f"  - Spearman Correlation: {test_metrics['spearman_corr']:.6f}\n\n")
            
            # Autoencoder results
            if 'autoencoder' in self.results:
                ae_metrics = self.results['autoencoder']['metrics']
                f.write("Autoencoder Results:\n")
                f.write(f"  - Reconstruction Loss: {ae_metrics['avg_reconstruction_loss']:.6f}\n")
                f.write(f"  - Reconstruction Error: {ae_metrics['reconstruction_error']:.6f}\n")
                f.write(f"  - Latent Variance: {ae_metrics['latent_variance']:.6f}\n\n")
            
            # Baseline comparison
            if 'baseline_comparison' in self.results:
                comparison = self.results['baseline_comparison']
                f.write("Baseline Comparison:\n")
                for name, improvement in comparison['improvements'].items():
                    f.write(f"  - vs {name}: MAE improvement {improvement['mae_improvement']:.2f}%\n")
                f.write("\n")
            
            # Gene analysis
            if 'gene_analysis' in self.results:
                gene_summary = self.results['gene_analysis']['summary']
                f.write("Gene Analysis:\n")
                f.write(f"  - Mean Gene R²: {gene_summary['mean_r2']:.4f}\n")
                f.write(f"  - Median Gene R²: {gene_summary['median_r2']:.4f}\n")
                f.write(f"  - Genes with positive R²: {gene_summary['genes_with_positive_r2']}\n")
                f.write(f"  - Genes with good R² (>0.5): {gene_summary['genes_with_good_r2']}\n")
                f.write(f"  - Genes with high correlation (>0.7): {gene_summary['genes_with_high_correlation']}\n\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate drug combination prediction model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--analyze-genes', action='store_true',
                        help='Perform gene importance analysis')
    parser.add_argument('--compare-baselines', action='store_true',
                        help='Compare with baseline models')
    parser.add_argument('--evaluate-ae', action='store_true',
                        help='Evaluate autoencoder separately')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config, args.model, device)
    
    # Load data
    data_loader = DrugDataLoader(config)
    single_drug_loader, train_loader, val_loader, test_loader = data_loader.get_all_loaders(args.data)
    
    # Choose dataset(s) to evaluate
    if args.dataset == 'all':
        datasets = [('train', train_loader), ('val', val_loader), ('test', test_loader)]
    else:
        dataset_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        datasets = [(args.dataset, dataset_map[args.dataset])]
    
    # Run evaluation
    results_summary = {}
    
    for dataset_name, loader in datasets:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name} dataset")
        print(f"{'='*50}")
        
        metrics = evaluator.evaluate_on_dataset(loader, dataset_name)
        results_summary[dataset_name] = metrics
    
    # Additional analyses
    if args.evaluate_ae:
        print(f"\n{'='*50}")
        print("Evaluating autoencoder")
        print(f"{'='*50}")
        
        ae_metrics = evaluator.evaluate_autoencoder(single_drug_loader)
        results_summary['autoencoder'] = ae_metrics
    
    if args.compare_baselines:
        print(f"\n{'='*50}")
        print("Comparing with baselines")
        print(f"{'='*50}")
        
        baseline_results = evaluator.compare_with_baselines(test_loader)
        results_summary['baseline_comparison'] = baseline_results
    
    if args.analyze_genes:
        print(f"\n{'='*50}")
        print("Analyzing gene importance")
        print(f"{'='*50}")
        
        gene_analysis = evaluator.analyze_gene_importance(test_loader)
        results_summary['gene_analysis'] = gene_analysis
    
    # Create visualizations
    if args.visualize:
        print(f"\n{'='*50}")
        print("Creating visualizations")
        print(f"{'='*50}")
        
        evaluator.create_visualizations(args.output)
    
    # Save results
    evaluator.save_results(args.output)
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()