"""
Baseline Models for Drug Combination Prediction.

This script implements Phase 3 of the project plan:
- Simple baseline models
- Evaluation framework for baselines
- Comparison with neural network approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import argparse
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_combo.data.data_loader import generate_synthetic_data
from src.drug_combo.utils.metrics import calculate_metrics


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the baseline model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y.flatten(), predictions.flatten())
        mse = mean_squared_error(y.flatten(), predictions.flatten())
        r2 = r2_score(y.flatten(), predictions.flatten())
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y.flatten(), predictions.flatten())
        spearman_r, spearman_p = spearmanr(y.flatten(), predictions.flatten())
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }


class LinearAdditionBaseline(BaselineModel):
    """Simple linear addition baseline: combo = drug_A + drug_B."""
    
    def __init__(self):
        super().__init__("Linear Addition")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No training needed for linear addition."""
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by adding the two drug effects."""
        # X should be [drug_a_features, drug_b_features] concatenated
        n_features = X.shape[1] // 2
        drug_a = X[:, :n_features]
        drug_b = X[:, n_features:]
        return drug_a + drug_b


class AverageBaseline(BaselineModel):
    """Average baseline: combo = (drug_A + drug_B) / 2."""
    
    def __init__(self):
        super().__init__("Average")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No training needed for averaging."""
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging the two drug effects."""
        n_features = X.shape[1] // 2
        drug_a = X[:, :n_features]
        drug_b = X[:, n_features:]
        return (drug_a + drug_b) / 2


class MaximumBaseline(BaselineModel):
    """Maximum baseline: combo = max(drug_A, drug_B)."""
    
    def __init__(self):
        super().__init__("Maximum")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No training needed for maximum."""
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by taking maximum of the two drug effects."""
        n_features = X.shape[1] // 2
        drug_a = X[:, :n_features]
        drug_b = X[:, n_features:]
        return np.maximum(drug_a, drug_b)


class LinearRegressionBaseline(BaselineModel):
    """Linear regression baseline on concatenated features."""
    
    def __init__(self, regularization: float = 0.0):
        super().__init__("Linear Regression")
        self.regularization = regularization
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train linear regression model."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if self.regularization > 0:
            self.model = Ridge(alpha=self.regularization)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using linear regression."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        # For multi-output, we need to reshape
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = y.reshape(y.shape[0], -1)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Random Forest."""
        return self.model.predict(X)


class MLPBaseline(BaselineModel):
    """Multi-layer Perceptron baseline."""
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100, 50)):
        super().__init__("MLP")
        self.hidden_layer_sizes = hidden_layer_sizes
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train MLP model."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # For multi-output, we need to reshape
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = y.reshape(y.shape[0], -1)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using MLP."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class BaselineEvaluator:
    """Evaluator for baseline models."""
    
    def __init__(self):
        self.results = {}
    
    def prepare_data(self, single_drug_data: np.ndarray, pair_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for baseline model training."""
        drug_a_indices = pair_data['drug_a_indices']
        drug_b_indices = pair_data['drug_b_indices']
        pair_expressions = pair_data['expressions']
        
        # Get individual drug effects
        drug_a_effects = single_drug_data[drug_a_indices]
        drug_b_effects = single_drug_data[drug_b_indices]
        
        # Concatenate features
        X = np.hstack([drug_a_effects, drug_b_effects])
        y = pair_expressions
        
        return X, y
    
    def train_and_evaluate_baselines(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all baseline models."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize baseline models
        baselines = [
            LinearAdditionBaseline(),
            AverageBaseline(),
            MaximumBaseline(),
            LinearRegressionBaseline(),
            LinearRegressionBaseline(regularization=1.0),
            RandomForestBaseline(),
            MLPBaseline()
        ]
        
        # Add regularized linear regression
        baselines[4].name = "Ridge Regression"
        
        results = {}
        
        print("Training and evaluating baseline models...")
        for baseline in baselines:
            print(f"  Processing {baseline.name}...")
            
            try:
                # Train model
                baseline.fit(X_train, y_train)
                
                # Evaluate on test set
                test_metrics = baseline.evaluate(X_test, y_test)
                results[baseline.name] = test_metrics
                
                print(f"    {baseline.name} - MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"    Error with {baseline.name}: {str(e)}")
                continue
        
        return results
    
    def plot_baseline_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Plot comparison of baseline models."""
        metrics = ['mae', 'mse', 'r2', 'pearson_r']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_prediction_quality(self, X: np.ndarray, y: np.ndarray, results: Dict[str, Dict[str, float]]) -> None:
        """Analyze prediction quality across baselines."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Get predictions from best performing models
        best_models = ['Linear Addition', 'Random Forest', 'MLP']
        
        fig, axes = plt.subplots(1, len(best_models), figsize=(15, 5))
        
        for i, model_name in enumerate(best_models):
            if model_name in results:
                # Re-train model to get predictions
                if model_name == 'Linear Addition':
                    model = LinearAdditionBaseline()
                elif model_name == 'Random Forest':
                    model = RandomForestBaseline()
                elif model_name == 'MLP':
                    model = MLPBaseline()
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Plot predictions vs actual
                axes[i].scatter(y_test.flatten(), predictions.flatten(), alpha=0.5)
                axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].set_title(f'{model_name}\nR² = {results[model_name]["r2"]:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_baseline_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate comprehensive baseline report."""
        # Find best performing models
        best_mae = min(results.values(), key=lambda x: x['mae'])
        best_r2 = max(results.values(), key=lambda x: x['r2'])
        
        best_mae_model = [k for k, v in results.items() if v['mae'] == best_mae['mae']][0]
        best_r2_model = [k for k, v in results.items() if v['r2'] == best_r2['r2']][0]
        
        report = f"""
# Baseline Models Evaluation Report

## Model Performance Summary

### Best Performing Models:
- **Lowest MAE**: {best_mae_model} (MAE: {best_mae['mae']:.4f})
- **Highest R²**: {best_r2_model} (R²: {best_r2['r2']:.4f})

### Detailed Results:
"""
        
        for model_name, metrics in results.items():
            report += f"""
#### {model_name}
- MAE: {metrics['mae']:.4f}
- MSE: {metrics['mse']:.4f}
- RMSE: {metrics['rmse']:.4f}
- R²: {metrics['r2']:.4f}
- Pearson R: {metrics['pearson_r']:.4f}
- Spearman R: {metrics['spearman_r']:.4f}
"""
        
        report += f"""
## Key Insights

1. **Simple vs Complex Models**: 
   - Linear addition baseline performance: {results.get('Linear Addition', {}).get('mae', 'N/A')} MAE
   - Best complex model performance: {best_mae['mae']:.4f} MAE
   - Improvement: {((results.get('Linear Addition', {}).get('mae', 0) - best_mae['mae']) / results.get('Linear Addition', {}).get('mae', 1) * 100):.1f}%

2. **Model Complexity Analysis**:
   - Simple baselines (addition, average, max) capture basic drug interaction patterns
   - Machine learning models show potential for capturing non-linear interactions
   - Diminishing returns with model complexity suggest careful tuning is needed

3. **Recommendations for Neural Network Development**:
   - Target performance: Beat {best_mae_model} (MAE < {best_mae['mae']:.4f})
   - Focus on capturing interaction patterns beyond simple addition
   - Consider ensemble approaches combining multiple baseline strategies

## Next Steps

1. Implement neural network architecture
2. Compare against these baseline results
3. Analyze where neural networks provide the most improvement
4. Consider hybrid approaches combining baselines with neural networks
"""
        
        return report


def load_data(data_path: str, use_synthetic: bool = True) -> Tuple[np.ndarray, Dict]:
    """Load data for baseline evaluation."""
    if use_synthetic:
        print("Generating synthetic data for baseline evaluation...")
        single_drug_data, pair_data = generate_synthetic_data(
            num_drugs=200,
            num_pairs=1000,
            num_genes=500,  # Reduced for faster baseline training
            noise_level=0.1,
            random_seed=42
        )
    else:
        # Load real data implementation
        raise NotImplementedError("Real data loading not implemented yet")
    
    return single_drug_data, pair_data


def run_baseline_experiments(data_path: str, output_dir: str = "baseline_output", use_synthetic: bool = True) -> None:
    """
    Run comprehensive baseline model experiments.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save outputs
        use_synthetic: Whether to use synthetic data
    """
    print("Starting Baseline Model Experiments...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    single_drug_data, pair_data = load_data(data_path, use_synthetic)
    
    # Initialize evaluator
    evaluator = BaselineEvaluator()
    
    # Prepare data
    X, y = evaluator.prepare_data(single_drug_data, pair_data)
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    
    # Train and evaluate baselines
    results = evaluator.train_and_evaluate_baselines(X, y)
    
    # Plot comparisons
    evaluator.plot_baseline_comparison(results)
    
    # Analyze prediction quality
    evaluator.analyze_prediction_quality(X, y, results)
    
    # Generate and save report
    report = evaluator.generate_baseline_report(results)
    
    report_path = os.path.join(output_dir, "baseline_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save results as CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "baseline_results.csv"))
    
    print(f"\n✅ Baseline experiments completed successfully!")
    print(f"Report saved to: {report_path}")
    print(f"Results saved to: {os.path.join(output_dir, 'baseline_results.csv')}")
    print("Ready to proceed with neural network model development.")


def main():
    """Main function for running baseline experiments."""
    parser = argparse.ArgumentParser(description='Run Baseline Model Experiments')
    parser.add_argument('--data', type=str, default='data/',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='baseline_output',
                        help='Output directory for results')
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                        help='Use synthetic data')
    
    args = parser.parse_args()
    
    run_baseline_experiments(args.data, args.output, args.use_synthetic)


if __name__ == "__main__":
    main()