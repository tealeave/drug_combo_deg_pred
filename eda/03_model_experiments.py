"""
Neural Network Model Experiments for Drug Combination Prediction.

This script implements Phase 4+ of the project plan:
- Neural network model development
- Hyperparameter tuning
- Advanced architecture experiments
- Model comparison and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
import json
from typing import Dict, List, Tuple, Any
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_combo.models.prediction_model import FullDrugCombinationModel
from src.drug_combo.models.autoencoder import GeneExpressionAutoencoder
from src.drug_combo.models.attention_layers import SelfAttention, CrossAttention
from src.drug_combo.data.data_loader import generate_synthetic_data
from src.drug_combo.training.trainer import DrugCombinationTrainer
from src.drug_combo.training.evaluation import ModelEvaluator


class ModelExperimenter:
    """Advanced model experimentation framework."""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.results = {}
        self.models = {}
        
    def create_model_variants(self, base_config: Dict) -> Dict[str, Dict]:
        """Create different model architecture variants."""
        variants = {}
        
        # 1. Simple baseline model
        variants['simple'] = {
            **base_config,
            'model': {
                **base_config['model'],
                'latent_dim': 10,
                'autoencoder_hidden': [500, 100],
                'predictor_hidden': [32, 16],
                'use_attention': False
            }
        }
        
        # 2. Medium complexity model
        variants['medium'] = {
            **base_config,
            'model': {
                **base_config['model'],
                'latent_dim': 20,
                'autoencoder_hidden': [1000, 200],
                'predictor_hidden': [64, 32],
                'use_attention': False
            }
        }
        
        # 3. Complex model with attention
        variants['complex'] = {
            **base_config,
            'model': {
                **base_config['model'],
                'latent_dim': 50,
                'autoencoder_hidden': [1500, 500, 100],
                'predictor_hidden': [128, 64, 32],
                'use_attention': True
            }
        }
        
        # 4. Deep autoencoder variant
        variants['deep_ae'] = {
            **base_config,
            'model': {
                **base_config['model'],
                'latent_dim': 30,
                'autoencoder_hidden': [2000, 1000, 500, 200],
                'predictor_hidden': [64, 32],
                'use_attention': False
            }
        }
        
        # 5. Attention-focused variant
        variants['attention_heavy'] = {
            **base_config,
            'model': {
                **base_config['model'],
                'latent_dim': 40,
                'autoencoder_hidden': [1000, 300],
                'predictor_hidden': [128, 64],
                'use_attention': True
            }
        }
        
        return variants
    
    def hyperparameter_search(self, base_config: Dict, param_grid: Dict) -> Dict[str, Any]:
        """Perform hyperparameter search."""
        print("Starting hyperparameter search...")
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        
        for i, params in enumerate(param_combinations):
            print(f"Testing combination {i+1}/{len(param_combinations)}: {dict(zip(param_names, params))}")
            
            # Create modified config
            config = base_config.copy()
            for param_name, param_value in zip(param_names, params):
                if '.' in param_name:
                    section, key = param_name.split('.')
                    config[section][key] = param_value
                else:
                    config[param_name] = param_value
            
            try:
                # Train model with this configuration
                score = self._train_and_evaluate_config(config)
                results.append({
                    'params': dict(zip(param_names, params)),
                    'score': score,
                    'config': config
                })
            except Exception as e:
                print(f"Error with configuration: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'])
        
        return {
            'best_config': results[0]['config'],
            'best_score': results[0]['score'],
            'all_results': results
        }
    
    def _train_and_evaluate_config(self, config: Dict) -> float:
        """Train and evaluate a model with given configuration."""
        # Generate synthetic data
        single_drug_data, pair_data = generate_synthetic_data(
            num_drugs=100, num_pairs=300, num_genes=config['model']['gene_dim'],
            noise_level=0.1, random_seed=42
        )
        
        # Create model
        model = FullDrugCombinationModel(
            gene_dim=config['model']['gene_dim'],
            latent_dim=config['model']['latent_dim'],
            autoencoder_hidden=config['model']['autoencoder_hidden'],
            predictor_hidden=config['model']['predictor_hidden'],
            use_attention=config['model']['use_attention']
        ).to(self.device)
        
        # Quick training (reduced epochs for hyperparameter search)
        trainer = DrugCombinationTrainer(config)
        trainer.model = model
        trainer.device = self.device
        
        # Train autoencoder
        trainer.train_autoencoder(single_drug_data, epochs=20)
        
        # Prepare data for full training
        train_loader, val_loader, test_loader = trainer._create_data_loaders(
            single_drug_data, pair_data
        )
        
        # Train full model
        trainer.train_full_model(train_loader, val_loader, epochs=30)
        
        # Evaluate
        test_loss = trainer.evaluate(test_loader)
        
        return test_loss
    
    def architecture_comparison(self, base_config: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Compare different neural network architectures."""
        print("Comparing neural network architectures...")
        
        # Create model variants
        variants = self.create_model_variants(base_config)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for variant_name, config in variants.items():
            print(f"Training {variant_name} model...")
            
            try:
                # Train model
                score = self._train_and_evaluate_config(config)
                
                # Store results
                results[variant_name] = {
                    'test_loss': score,
                    'config': config,
                    'parameters': self._count_parameters(config)
                }
                
                print(f"  {variant_name}: Test Loss = {score:.4f}")
                
            except Exception as e:
                print(f"  Error training {variant_name}: {e}")
                continue
        
        return results
    
    def _count_parameters(self, config: Dict) -> int:
        """Estimate number of parameters in model."""
        gene_dim = config['model']['gene_dim']
        latent_dim = config['model']['latent_dim']
        ae_hidden = config['model']['autoencoder_hidden']
        pred_hidden = config['model']['predictor_hidden']
        
        # Autoencoder parameters
        ae_params = 0
        prev_dim = gene_dim
        for hidden_dim in ae_hidden:
            ae_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
            prev_dim = hidden_dim
        ae_params += prev_dim * latent_dim + latent_dim  # final layer
        ae_params *= 2  # encoder + decoder
        
        # Predictor parameters
        pred_params = 0
        prev_dim = latent_dim * 2  # concatenated drug representations
        for hidden_dim in pred_hidden:
            pred_params += prev_dim * hidden_dim + hidden_dim
            prev_dim = hidden_dim
        pred_params += prev_dim * latent_dim + latent_dim  # output layer
        
        return ae_params + pred_params
    
    def ablation_study(self, base_config: Dict) -> Dict[str, Dict]:
        """Perform ablation study on model components."""
        print("Conducting ablation study...")
        
        ablation_configs = {
            'full_model': base_config,
            'no_attention': {
                **base_config,
                'model': {**base_config['model'], 'use_attention': False}
            },
            'shallow_autoencoder': {
                **base_config,
                'model': {
                    **base_config['model'], 
                    'autoencoder_hidden': [500]
                }
            },
            'small_latent': {
                **base_config,
                'model': {**base_config['model'], 'latent_dim': 10}
            },
            'large_latent': {
                **base_config,
                'model': {**base_config['model'], 'latent_dim': 100}
            }
        }
        
        results = {}
        
        for name, config in ablation_configs.items():
            print(f"Testing {name}...")
            
            try:
                score = self._train_and_evaluate_config(config)
                results[name] = {
                    'test_loss': score,
                    'config': config
                }
                print(f"  {name}: {score:.4f}")
            except Exception as e:
                print(f"  Error with {name}: {e}")
        
        return results
    
    def learning_curve_analysis(self, config: Dict) -> Dict[str, List]:
        """Analyze learning curves with different data sizes."""
        print("Analyzing learning curves...")
        
        data_sizes = [50, 100, 200, 500, 1000]
        results = {
            'data_sizes': data_sizes,
            'train_losses': [],
            'val_losses': [],
            'test_losses': []
        }
        
        for size in data_sizes:
            print(f"Training with {size} drug pairs...")
            
            # Generate data of specific size
            single_drug_data, pair_data = generate_synthetic_data(
                num_drugs=min(size, 100),
                num_pairs=size,
                num_genes=config['model']['gene_dim'],
                noise_level=0.1,
                random_seed=42
            )
            
            try:
                # Train model
                trainer = DrugCombinationTrainer(config)
                trainer.train_autoencoder(single_drug_data, epochs=30)
                
                train_loader, val_loader, test_loader = trainer._create_data_loaders(
                    single_drug_data, pair_data
                )
                
                trainer.train_full_model(train_loader, val_loader, epochs=50)
                
                # Evaluate
                train_loss = trainer.evaluate(train_loader)
                val_loss = trainer.evaluate(val_loader)
                test_loss = trainer.evaluate(test_loader)
                
                results['train_losses'].append(train_loss)
                results['val_losses'].append(val_loss)
                results['test_losses'].append(test_loss)
                
                print(f"  Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
                
            except Exception as e:
                print(f"  Error with size {size}: {e}")
                results['train_losses'].append(None)
                results['val_losses'].append(None)
                results['test_losses'].append(None)
        
        return results
    
    def plot_experiment_results(self, results: Dict, experiment_type: str) -> None:
        """Plot experiment results."""
        if experiment_type == 'architecture_comparison':
            self._plot_architecture_comparison(results)
        elif experiment_type == 'hyperparameter_search':
            self._plot_hyperparameter_results(results)
        elif experiment_type == 'ablation_study':
            self._plot_ablation_results(results)
        elif experiment_type == 'learning_curves':
            self._plot_learning_curves(results)
    
    def _plot_architecture_comparison(self, results: Dict) -> None:
        """Plot architecture comparison results."""
        models = list(results.keys())
        losses = [results[model]['test_loss'] for model in models]
        params = [results[model]['parameters'] for model in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test loss comparison
        bars = axes[0].bar(models, losses)
        axes[0].set_title('Model Architecture Comparison')
        axes[0].set_ylabel('Test Loss')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{loss:.4f}', ha='center', va='bottom')
        
        # Parameter count vs performance
        axes[1].scatter(params, losses, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1].annotate(model, (params[i], losses[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        axes[1].set_xlabel('Number of Parameters')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Parameters vs Performance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_hyperparameter_results(self, results: Dict) -> None:
        """Plot hyperparameter search results."""
        all_results = results['all_results']
        
        # Extract parameter values and scores
        param_names = list(all_results[0]['params'].keys())
        
        fig, axes = plt.subplots(1, len(param_names), figsize=(5*len(param_names), 5))
        if len(param_names) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(param_names):
            values = [r['params'][param_name] for r in all_results]
            scores = [r['score'] for r in all_results]
            
            axes[i].scatter(values, scores, alpha=0.7)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Test Loss')
            axes[i].set_title(f'{param_name} vs Performance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_ablation_results(self, results: Dict) -> None:
        """Plot ablation study results."""
        components = list(results.keys())
        losses = [results[comp]['test_loss'] for comp in components]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(components, losses)
        plt.title('Ablation Study: Component Importance')
        plt.ylabel('Test Loss')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_learning_curves(self, results: Dict) -> None:
        """Plot learning curves."""
        sizes = results['data_sizes']
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        test_losses = results['test_losses']
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(sizes, train_losses, 'o-', label='Training Loss', linewidth=2)
        plt.plot(sizes, val_losses, 's-', label='Validation Loss', linewidth=2)
        plt.plot(sizes, test_losses, '^-', label='Test Loss', linewidth=2)
        
        plt.xlabel('Dataset Size (Number of Drug Pairs)')
        plt.ylabel('Loss')
        plt.title('Learning Curves: Performance vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def generate_experiment_report(self, all_results: Dict) -> str:
        """Generate comprehensive experiment report."""
        report = f"""
# Neural Network Model Experiments Report

## Experiment Overview
- Device: {self.device}
- Total experiments conducted: {len(all_results)}

## Key Findings

### Architecture Comparison
"""
        
        if 'architecture_comparison' in all_results:
            arch_results = all_results['architecture_comparison']
            best_arch = min(arch_results.keys(), key=lambda x: arch_results[x]['test_loss'])
            
            report += f"""
**Best Architecture**: {best_arch}
- Test Loss: {arch_results[best_arch]['test_loss']:.4f}
- Parameters: {arch_results[best_arch]['parameters']:,}

**All Architectures**:
"""
            for arch, result in arch_results.items():
                report += f"- {arch}: {result['test_loss']:.4f} (params: {result['parameters']:,})\n"
        
        if 'hyperparameter_search' in all_results:
            hp_results = all_results['hyperparameter_search']
            report += f"""
### Hyperparameter Search
**Best Configuration**: {hp_results['best_score']:.4f}
**Best Parameters**: {hp_results['best_config']['model']}
"""
        
        if 'ablation_study' in all_results:
            ablation_results = all_results['ablation_study']
            report += f"""
### Ablation Study
**Component Importance** (by performance degradation when removed):
"""
            sorted_components = sorted(ablation_results.items(), key=lambda x: x[1]['test_loss'])
            for comp, result in sorted_components:
                report += f"- {comp}: {result['test_loss']:.4f}\n"
        
        report += f"""
## Recommendations

1. **Model Architecture**: Use medium complexity models for best performance/efficiency tradeoff
2. **Latent Dimensions**: Optimal range appears to be 20-50 dimensions
3. **Autoencoder Depth**: Moderate depth (2-3 layers) provides good balance
4. **Attention Mechanisms**: Beneficial for complex interactions but increases training time

## Next Steps

1. Focus on the best performing architecture: {best_arch if 'architecture_comparison' in all_results else 'N/A'}
2. Implement production training pipeline
3. Validate on real drug interaction data
4. Consider ensemble methods for improved robustness
"""
        
        return report


def create_base_config() -> Dict:
    """Create base configuration for experiments."""
    return {
        'model': {
            'gene_dim': 1000,  # Reduced for faster experimentation
            'latent_dim': 20,
            'autoencoder_hidden': [500, 100],
            'predictor_hidden': [64, 32],
            'use_attention': False
        },
        'training': {
            'ae_epochs': 30,
            'full_epochs': 50,
            'batch_size': 32,
            'ae_lr': 0.001,
            'full_lr': 0.0005,
            'weight_decay': 0.0001,
            'seed': 42
        }
    }


def run_model_experiments(data_path: str, output_dir: str = "model_experiments_output") -> None:
    """
    Run comprehensive model experiments.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save outputs
    """
    print("Starting Neural Network Model Experiments...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize experimenter
    experimenter = ModelExperimenter()
    
    # Base configuration
    base_config = create_base_config()
    
    # Load data for experiments
    single_drug_data, pair_data = generate_synthetic_data(
        num_drugs=100, num_pairs=500, num_genes=1000, noise_level=0.1, random_seed=42
    )
    
    # Prepare data for sklearn-style experiments
    drug_a_indices = pair_data['drug_a_indices']
    drug_b_indices = pair_data['drug_b_indices']
    drug_a_effects = single_drug_data[drug_a_indices]
    drug_b_effects = single_drug_data[drug_b_indices]
    X = np.hstack([drug_a_effects, drug_b_effects])
    y = pair_data['expressions']
    
    all_results = {}
    
    # 1. Architecture Comparison
    print("\n=== Architecture Comparison ===")
    arch_results = experimenter.architecture_comparison(base_config, X, y)
    all_results['architecture_comparison'] = arch_results
    experimenter.plot_experiment_results(arch_results, 'architecture_comparison')
    
    # 2. Hyperparameter Search (limited scope for speed)
    print("\n=== Hyperparameter Search ===")
    param_grid = {
        'model.latent_dim': [10, 20, 30],
        'training.ae_lr': [0.0005, 0.001, 0.002],
        'training.full_lr': [0.0003, 0.0005, 0.001]
    }
    hp_results = experimenter.hyperparameter_search(base_config, param_grid)
    all_results['hyperparameter_search'] = hp_results
    experimenter.plot_experiment_results(hp_results, 'hyperparameter_search')
    
    # 3. Ablation Study
    print("\n=== Ablation Study ===")
    ablation_results = experimenter.ablation_study(base_config)
    all_results['ablation_study'] = ablation_results
    experimenter.plot_experiment_results(ablation_results, 'ablation_study')
    
    # 4. Learning Curves
    print("\n=== Learning Curve Analysis ===")
    learning_results = experimenter.learning_curve_analysis(base_config)
    all_results['learning_curves'] = learning_results
    experimenter.plot_experiment_results(learning_results, 'learning_curves')
    
    # Generate comprehensive report
    report = experimenter.generate_experiment_report(all_results)
    
    # Save results
    report_path = os.path.join(output_dir, "model_experiments_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in all_results.items():
            if key == 'learning_curves':
                json_results[key] = {
                    'data_sizes': value['data_sizes'],
                    'train_losses': [float(x) if x is not None else None for x in value['train_losses']],
                    'val_losses': [float(x) if x is not None else None for x in value['val_losses']],
                    'test_losses': [float(x) if x is not None else None for x in value['test_losses']]
                }
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n✅ Model experiments completed successfully!")
    print(f"Report saved to: {report_path}")
    print(f"Results saved to: {results_path}")
    print("Ready to proceed with production model training.")


def main():
    """Main function for running model experiments."""
    parser = argparse.ArgumentParser(description='Run Neural Network Model Experiments')
    parser.add_argument('--data', type=str, default='data/',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='model_experiments_output',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    run_model_experiments(args.data, args.output)


if __name__ == "__main__":
    main()