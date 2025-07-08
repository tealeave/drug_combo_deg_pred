"""
Prediction script for drug combination effects using trained model.
Makes predictions for new drug combinations and saves results.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.drug_combo.models.prediction_model import (
    FullDrugCombinationModel, 
    create_model
)
from src.drug_combo.data.preprocessing import GeneExpressionPreprocessor
from src.drug_combo.utils.metrics import calculate_metrics


class DrugCombinationPredictor:
    """Prediction class for drug combination effects."""
    
    def __init__(self, model_path: str, config: Dict, device: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to use for prediction
        """
        self.model_path = model_path
        self.config = config
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
        
        # Initialize preprocessor
        self.preprocessor = GeneExpressionPreprocessor(config)
        
        # Storage for predictions
        self.predictions = {}
        self.drug_data = {}
        
        self.logger.info(f"Predictor initialized on {self.device}")
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
    
    def load_single_drug_data(self, data_path: str) -> Dict[str, np.ndarray]:
        """
        Load single drug expression data.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Dictionary mapping drug IDs to expression profiles
        """
        self.logger.info(f"Loading single drug data from {data_path}")
        
        # Load and preprocess single drug data
        expressions, drug_ids = self.preprocessor.load_single_drug_data(data_path)
        
        # Apply differential expression if configured
        if self.config['data']['use_differential']:
            expressions = self.preprocessor.calculate_differential_expression(expressions)
            drug_ids = drug_ids[1:]  # Remove baseline
        
        # Normalize data
        expressions = self.preprocessor.normalize_data(expressions, fit=True)
        
        # Create drug data dictionary
        self.drug_data = {
            drug_id: expressions[i] 
            for i, drug_id in enumerate(drug_ids)
        }
        
        self.logger.info(f"Loaded {len(self.drug_data)} drugs with {expressions.shape[1]} genes")
        return self.drug_data
    
    def predict_combination(
        self, 
        drug_a_id: str, 
        drug_b_id: str, 
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Predict combination effect for two drugs.
        
        Args:
            drug_a_id: ID of first drug
            drug_b_id: ID of second drug
            return_components: Whether to return intermediate components
            
        Returns:
            Predicted gene expression profile, optionally with components
        """
        if drug_a_id not in self.drug_data:
            raise ValueError(f"Drug {drug_a_id} not found in loaded data")
        if drug_b_id not in self.drug_data:
            raise ValueError(f"Drug {drug_b_id} not found in loaded data")
        
        # Get drug expressions
        drug_a_expr = torch.FloatTensor(self.drug_data[drug_a_id]).unsqueeze(0).to(self.device)
        drug_b_expr = torch.FloatTensor(self.drug_data[drug_b_id]).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if return_components:
                prediction, (drug_a_latent, drug_b_latent, combo_latent) = self.model(
                    drug_a_expr, drug_b_expr, return_latent=True
                )
                
                components = {
                    'drug_a_latent': drug_a_latent.cpu().numpy().squeeze(),
                    'drug_b_latent': drug_b_latent.cpu().numpy().squeeze(),
                    'combo_latent': combo_latent.cpu().numpy().squeeze(),
                    'drug_a_expr': drug_a_expr.cpu().numpy().squeeze(),
                    'drug_b_expr': drug_b_expr.cpu().numpy().squeeze()
                }
                
                return prediction.cpu().numpy().squeeze(), components
            else:
                prediction = self.model(drug_a_expr, drug_b_expr)
                return prediction.cpu().numpy().squeeze()
    
    def predict_batch(
        self, 
        drug_pairs: List[Tuple[str, str]], 
        batch_size: int = 32,
        return_components: bool = False
    ) -> Dict:
        """
        Predict combination effects for multiple drug pairs.
        
        Args:
            drug_pairs: List of (drug_a_id, drug_b_id) tuples
            batch_size: Batch size for prediction
            return_components: Whether to return intermediate components
            
        Returns:
            Dictionary with prediction results
        """
        self.logger.info(f"Predicting {len(drug_pairs)} drug combinations")
        
        predictions = []
        components_list = [] if return_components else None
        valid_pairs = []
        
        # Process in batches
        for i in tqdm(range(0, len(drug_pairs), batch_size)):
            batch_pairs = drug_pairs[i:i+batch_size]
            
            # Prepare batch data
            batch_drug_a = []
            batch_drug_b = []
            batch_valid_pairs = []
            
            for drug_a_id, drug_b_id in batch_pairs:
                if drug_a_id in self.drug_data and drug_b_id in self.drug_data:
                    batch_drug_a.append(self.drug_data[drug_a_id])
                    batch_drug_b.append(self.drug_data[drug_b_id])
                    batch_valid_pairs.append((drug_a_id, drug_b_id))
                else:
                    self.logger.warning(f"Skipping pair ({drug_a_id}, {drug_b_id}) - missing data")
            
            if not batch_drug_a:
                continue
            
            # Convert to tensors
            batch_drug_a = torch.FloatTensor(np.array(batch_drug_a)).to(self.device)
            batch_drug_b = torch.FloatTensor(np.array(batch_drug_b)).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                if return_components:
                    batch_pred, (latent_a, latent_b, latent_combo) = self.model(
                        batch_drug_a, batch_drug_b, return_latent=True
                    )
                    
                    # Store components
                    for j in range(len(batch_valid_pairs)):
                        components_list.append({
                            'drug_a_latent': latent_a[j].cpu().numpy(),
                            'drug_b_latent': latent_b[j].cpu().numpy(),
                            'combo_latent': latent_combo[j].cpu().numpy(),
                            'drug_a_expr': batch_drug_a[j].cpu().numpy(),
                            'drug_b_expr': batch_drug_b[j].cpu().numpy()
                        })
                else:
                    batch_pred = self.model(batch_drug_a, batch_drug_b)
                
                predictions.extend(batch_pred.cpu().numpy())
                valid_pairs.extend(batch_valid_pairs)
        
        # Organize results
        results = {
            'predictions': np.array(predictions),
            'drug_pairs': valid_pairs,
            'n_predictions': len(predictions),
            'n_genes': predictions[0].shape[0] if predictions else 0
        }
        
        if return_components:
            results['components'] = components_list
        
        # Store in instance
        self.predictions = results
        
        self.logger.info(f"Completed predictions for {len(valid_pairs)} valid pairs")
        return results
    
    def predict_from_file(
        self, 
        pairs_file: str, 
        batch_size: int = 32,
        return_components: bool = False
    ) -> Dict:
        """
        Predict combinations from a file of drug pairs.
        
        Args:
            pairs_file: Path to file with drug pairs (CSV format)
            batch_size: Batch size for prediction
            return_components: Whether to return intermediate components
            
        Returns:
            Dictionary with prediction results
        """
        self.logger.info(f"Loading drug pairs from {pairs_file}")
        
        # Load pairs file
        pairs_df = pd.read_csv(pairs_file)
        
        # Validate columns
        required_columns = ['drug_a_id', 'drug_b_id']
        missing_columns = [col for col in required_columns if col not in pairs_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in pairs file: {missing_columns}")
        
        # Extract pairs
        drug_pairs = list(zip(pairs_df['drug_a_id'], pairs_df['drug_b_id']))
        
        # Make predictions
        results = self.predict_batch(drug_pairs, batch_size, return_components)
        
        # Add metadata from file
        results['pairs_file'] = pairs_file
        results['original_df'] = pairs_df
        
        return results
    
    def predict_matrix(
        self, 
        drug_list: Optional[List[str]] = None,
        batch_size: int = 32,
        symmetric: bool = True
    ) -> Dict:
        """
        Predict all pairwise combinations for a list of drugs.
        
        Args:
            drug_list: List of drug IDs (uses all if None)
            batch_size: Batch size for prediction
            symmetric: Whether to treat combinations as symmetric
            
        Returns:
            Dictionary with prediction matrix results
        """
        if drug_list is None:
            drug_list = list(self.drug_data.keys())
        
        self.logger.info(f"Predicting combination matrix for {len(drug_list)} drugs")
        
        # Generate all pairs
        drug_pairs = []
        for i, drug_a in enumerate(drug_list):
            start_j = i + 1 if symmetric else 0
            for j in range(start_j, len(drug_list)):
                drug_b = drug_list[j]
                if drug_a != drug_b:  # Skip self-combinations
                    drug_pairs.append((drug_a, drug_b))
        
        # Make predictions
        results = self.predict_batch(drug_pairs, batch_size)
        
        # Organize as matrix
        n_drugs = len(drug_list)
        n_genes = results['n_genes']
        
        # Create prediction matrix (drugs x drugs x genes)
        pred_matrix = np.zeros((n_drugs, n_drugs, n_genes))
        
        for idx, (drug_a, drug_b) in enumerate(results['drug_pairs']):
            i = drug_list.index(drug_a)
            j = drug_list.index(drug_b)
            
            pred_matrix[i, j] = results['predictions'][idx]
            
            if symmetric:
                pred_matrix[j, i] = results['predictions'][idx]
        
        # Add matrix to results
        results['prediction_matrix'] = pred_matrix
        results['drug_list'] = drug_list
        results['symmetric'] = symmetric
        
        return results
    
    def calculate_baseline_predictions(self, drug_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Calculate baseline predictions for comparison.
        
        Args:
            drug_pairs: List of drug pairs
            
        Returns:
            Dictionary with baseline predictions
        """
        self.logger.info("Calculating baseline predictions")
        
        baselines = {}
        
        for baseline_name in ['additive', 'average', 'maximum']:
            baseline_preds = []
            
            for drug_a_id, drug_b_id in drug_pairs:
                if drug_a_id in self.drug_data and drug_b_id in self.drug_data:
                    drug_a_expr = self.drug_data[drug_a_id]
                    drug_b_expr = self.drug_data[drug_b_id]
                    
                    if baseline_name == 'additive':
                        baseline_pred = drug_a_expr + drug_b_expr
                    elif baseline_name == 'average':
                        baseline_pred = (drug_a_expr + drug_b_expr) / 2
                    elif baseline_name == 'maximum':
                        baseline_pred = np.maximum(drug_a_expr, drug_b_expr)
                    
                    baseline_preds.append(baseline_pred)
            
            baselines[baseline_name] = np.array(baseline_preds)
        
        return baselines
    
    def analyze_predictions(self, ground_truth: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze prediction results.
        
        Args:
            ground_truth: Ground truth values for evaluation (optional)
            
        Returns:
            Dictionary with analysis results
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_* method first.")
        
        predictions = self.predictions['predictions']
        drug_pairs = self.predictions['drug_pairs']
        
        analysis = {
            'n_predictions': len(predictions),
            'n_genes': predictions.shape[1],
            'prediction_stats': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'median': np.median(predictions)
            },
            'gene_wise_stats': {
                'mean_per_gene': np.mean(predictions, axis=0),
                'std_per_gene': np.std(predictions, axis=0),
                'min_per_gene': np.min(predictions, axis=0),
                'max_per_gene': np.max(predictions, axis=0)
            }
        }
        
        # Calculate baseline comparisons
        baselines = self.calculate_baseline_predictions(drug_pairs)
        analysis['baseline_comparisons'] = {}
        
        for baseline_name, baseline_preds in baselines.items():
            # Compare with model predictions
            diff = predictions - baseline_preds
            analysis['baseline_comparisons'][baseline_name] = {
                'mean_difference': np.mean(diff),
                'std_difference': np.std(diff),
                'mae_difference': np.mean(np.abs(diff)),
                'correlation': np.corrcoef(predictions.flatten(), baseline_preds.flatten())[0, 1]
            }
        
        # If ground truth is provided, calculate evaluation metrics
        if ground_truth is not None:
            evaluation_metrics = calculate_metrics(predictions, ground_truth)
            analysis['evaluation_metrics'] = evaluation_metrics
            
            # Compare baselines with ground truth
            analysis['baseline_evaluation'] = {}
            for baseline_name, baseline_preds in baselines.items():
                baseline_metrics = calculate_metrics(baseline_preds, ground_truth)
                analysis['baseline_evaluation'][baseline_name] = baseline_metrics
        
        return analysis
    
    def save_predictions(self, save_path: str, include_components: bool = False):
        """
        Save predictions to files.
        
        Args:
            save_path: Path to save directory
            include_components: Whether to save intermediate components
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_* method first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Saving predictions to {save_dir}")
        
        # Save predictions array
        np.save(save_dir / "predictions.npy", self.predictions['predictions'])
        
        # Save drug pairs
        pairs_df = pd.DataFrame(self.predictions['drug_pairs'], columns=['drug_a_id', 'drug_b_id'])
        pairs_df.to_csv(save_dir / "drug_pairs.csv", index=False)
        
        # Save predictions with pairs
        pred_df = pairs_df.copy()
        
        # Add prediction columns
        predictions = self.predictions['predictions']
        for gene_idx in range(predictions.shape[1]):
            pred_df[f'gene_{gene_idx}'] = predictions[:, gene_idx]
        
        pred_df.to_csv(save_dir / "predictions_with_pairs.csv", index=False)
        
        # Save metadata
        metadata = {
            'n_predictions': self.predictions['n_predictions'],
            'n_genes': self.predictions['n_genes'],
            'model_path': self.model_path,
            'prediction_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save components if requested
        if include_components and 'components' in self.predictions:
            components = self.predictions['components']
            
            # Save latent representations
            drug_a_latent = np.array([comp['drug_a_latent'] for comp in components])
            drug_b_latent = np.array([comp['drug_b_latent'] for comp in components])
            combo_latent = np.array([comp['combo_latent'] for comp in components])
            
            np.save(save_dir / "drug_a_latent.npy", drug_a_latent)
            np.save(save_dir / "drug_b_latent.npy", drug_b_latent)
            np.save(save_dir / "combo_latent.npy", combo_latent)
        
        # Save prediction matrix if available
        if 'prediction_matrix' in self.predictions:
            np.save(save_dir / "prediction_matrix.npy", self.predictions['prediction_matrix'])
            
            # Save drug list
            with open(save_dir / "drug_list.json", 'w') as f:
                json.dump(self.predictions['drug_list'], f, indent=2)
        
        self.logger.info(f"Predictions saved to {save_dir}")
    
    def create_prediction_plots(self, save_path: str, analysis_results: Optional[Dict] = None):
        """
        Create visualization plots for predictions.
        
        Args:
            save_path: Path to save plots
            analysis_results: Analysis results for plotting
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_* method first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        predictions = self.predictions['predictions']
        
        # Prediction distribution plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(predictions.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Expression')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predictions')
        plt.grid(True, alpha=0.3)
        
        # Gene-wise prediction variance
        plt.subplot(2, 2, 2)
        gene_vars = np.var(predictions, axis=0)
        plt.hist(gene_vars, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Gene-wise Variance')
        plt.ylabel('Number of Genes')
        plt.title('Distribution of Gene-wise Prediction Variance')
        plt.grid(True, alpha=0.3)
        
        # Mean prediction per gene
        plt.subplot(2, 2, 3)
        gene_means = np.mean(predictions, axis=0)
        plt.plot(gene_means)
        plt.xlabel('Gene Index')
        plt.ylabel('Mean Prediction')
        plt.title('Mean Prediction per Gene')
        plt.grid(True, alpha=0.3)
        
        # Prediction range per gene
        plt.subplot(2, 2, 4)
        gene_ranges = np.max(predictions, axis=0) - np.min(predictions, axis=0)
        plt.hist(gene_ranges, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Gene Prediction Range')
        plt.ylabel('Number of Genes')
        plt.title('Distribution of Gene Prediction Ranges')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "prediction_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Baseline comparison plot
        if analysis_results and 'baseline_comparisons' in analysis_results:
            self._plot_baseline_comparison(analysis_results['baseline_comparisons'], save_dir)
        
        # Prediction matrix heatmap if available
        if 'prediction_matrix' in self.predictions:
            self._plot_prediction_matrix(save_dir)
        
        self.logger.info(f"Plots saved to {save_dir}")
    
    def _plot_baseline_comparison(self, baseline_comparisons: Dict, save_dir: Path):
        """Plot baseline comparison results."""
        baselines = list(baseline_comparisons.keys())
        mae_diffs = [baseline_comparisons[b]['mae_difference'] for b in baselines]
        correlations = [baseline_comparisons[b]['correlation'] for b in baselines]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE difference
        axes[0].bar(baselines, mae_diffs)
        axes[0].set_ylabel('MAE Difference from Baseline')
        axes[0].set_title('Model vs Baseline MAE Differences')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Correlation with baselines
        axes[1].bar(baselines, correlations)
        axes[1].set_ylabel('Correlation with Baseline')
        axes[1].set_title('Model vs Baseline Correlations')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_matrix(self, save_dir: Path):
        """Plot prediction matrix heatmap."""
        if 'prediction_matrix' not in self.predictions:
            return
        
        pred_matrix = self.predictions['prediction_matrix']
        drug_list = self.predictions['drug_list']
        
        # Plot average expression across genes
        avg_matrix = np.mean(pred_matrix, axis=2)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            avg_matrix,
            xticklabels=drug_list,
            yticklabels=drug_list,
            cmap='viridis',
            center=0,
            cbar_kws={'label': 'Average Predicted Expression'}
        )
        plt.title('Drug Combination Prediction Matrix\n(Average across genes)')
        plt.xlabel('Drug B')
        plt.ylabel('Drug A')
        plt.tight_layout()
        plt.savefig(save_dir / "prediction_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict drug combination effects')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to single drug data directory')
    parser.add_argument('--pairs', type=str,
                        help='Path to drug pairs file (CSV)')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--matrix', action='store_true',
                        help='Generate full combination matrix')
    parser.add_argument('--drug-list', type=str, nargs='+',
                        help='List of drugs for matrix prediction')
    parser.add_argument('--components', action='store_true',
                        help='Save intermediate components')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze prediction results')
    parser.add_argument('--plot', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--ground-truth', type=str,
                        help='Path to ground truth file for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize predictor
    predictor = DrugCombinationPredictor(args.model, config, device)
    
    # Load single drug data
    predictor.load_single_drug_data(args.data)
    
    # Make predictions
    if args.matrix:
        print("Generating prediction matrix...")
        results = predictor.predict_matrix(
            drug_list=args.drug_list,
            batch_size=args.batch_size
        )
    elif args.pairs:
        print(f"Predicting combinations from {args.pairs}...")
        results = predictor.predict_from_file(
            args.pairs,
            batch_size=args.batch_size,
            return_components=args.components
        )
    else:
        raise ValueError("Must specify either --pairs or --matrix")
    
    # Save predictions
    predictor.save_predictions(args.output, include_components=args.components)
    
    # Analysis
    if args.analyze:
        print("Analyzing predictions...")
        
        # Load ground truth if provided
        ground_truth = None
        if args.ground_truth:
            ground_truth = np.load(args.ground_truth)
        
        analysis = predictor.analyze_predictions(ground_truth)
        
        # Save analysis results
        with open(Path(args.output) / "analysis_results.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Print summary
        print("\nPrediction Analysis Summary:")
        print(f"  Number of predictions: {analysis['n_predictions']}")
        print(f"  Number of genes: {analysis['n_genes']}")
        print(f"  Prediction mean: {analysis['prediction_stats']['mean']:.4f}")
        print(f"  Prediction std: {analysis['prediction_stats']['std']:.4f}")
        
        if 'evaluation_metrics' in analysis:
            metrics = analysis['evaluation_metrics']
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R²: {metrics['r2_overall']:.4f}")
            print(f"  Correlation: {metrics['pearson_corr']:.4f}")
    
    # Create plots
    if args.plot:
        print("Creating visualization plots...")
        analysis_results = predictor.analyze_predictions() if not args.analyze else analysis
        predictor.create_prediction_plots(args.output, analysis_results)
    
    print(f"\nPrediction completed! Results saved to: {args.output}")


if __name__ == "__main__":
    main()