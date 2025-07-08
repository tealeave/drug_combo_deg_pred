"""
Evaluation metrics for drug combination prediction model.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted values (n_samples, n_genes)
        targets: True values (n_samples, n_genes)
        
    Returns:
        Dictionary of metric names and values
    """
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Overall metrics
    mae = mean_absolute_error(target_flat, pred_flat)
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_flat, pred_flat)
    
    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(pred_flat, target_flat)
    spearman_corr, spearman_p = spearmanr(pred_flat, target_flat)
    
    # Gene-wise metrics
    gene_wise_r2 = []
    gene_wise_pearson = []
    
    for gene_idx in range(predictions.shape[1]):
        gene_pred = predictions[:, gene_idx]
        gene_target = targets[:, gene_idx]
        
        # R2 for this gene
        gene_r2 = r2_score(gene_target, gene_pred)
        gene_wise_r2.append(gene_r2)
        
        # Pearson correlation for this gene
        if np.std(gene_pred) > 0 and np.std(gene_target) > 0:
            gene_pearson, _ = pearsonr(gene_pred, gene_target)
            gene_wise_pearson.append(gene_pearson)
        else:
            gene_wise_pearson.append(0.0)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_overall': r2,
        'pearson_corr': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_pvalue': spearman_p,
        'r2_gene_mean': np.mean(gene_wise_r2),
        'r2_gene_std': np.std(gene_wise_r2),
        'r2_gene_median': np.median(gene_wise_r2),
        'pearson_gene_mean': np.mean(gene_wise_pearson),
        'pearson_gene_std': np.std(gene_wise_pearson),
        'pearson_gene_median': np.median(gene_wise_pearson),
        'genes_r2_positive': np.sum(np.array(gene_wise_r2) > 0),
        'genes_r2_good': np.sum(np.array(gene_wise_r2) > 0.5),
        'genes_pearson_high': np.sum(np.array(gene_wise_pearson) > 0.7)
    }
    
    return metrics


def calculate_interaction_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray,
    single_drug_a: np.ndarray,
    single_drug_b: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics specifically for drug interaction analysis.
    
    Args:
        predictions: Predicted combination effects
        targets: True combination effects
        single_drug_a: Single drug A effects
        single_drug_b: Single drug B effects
        
    Returns:
        Dictionary of interaction-specific metrics
    """
    # Calculate additive baseline
    additive_baseline = single_drug_a + single_drug_b
    
    # Metrics vs additive baseline
    additive_mae = mean_absolute_error(targets.flatten(), additive_baseline.flatten())
    model_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    improvement_over_additive = (additive_mae - model_mae) / additive_mae * 100
    
    # Interaction classification
    target_interactions = targets - additive_baseline
    pred_interactions = predictions - additive_baseline
    
    # Synergy/antagonism detection
    synergy_threshold = 0.1  # Threshold for calling an interaction synergistic
    target_synergy = target_interactions > synergy_threshold
    target_antagonism = target_interactions < -synergy_threshold
    target_additive = np.abs(target_interactions) <= synergy_threshold
    
    pred_synergy = pred_interactions > synergy_threshold
    pred_antagonism = pred_interactions < -synergy_threshold
    pred_additive = np.abs(pred_interactions) <= synergy_threshold
    
    # Classification accuracy
    interaction_accuracy = np.mean(
        (target_synergy & pred_synergy) |
        (target_antagonism & pred_antagonism) |
        (target_additive & pred_additive)
    )
    
    metrics = {
        'additive_baseline_mae': additive_mae,
        'model_mae': model_mae,
        'improvement_over_additive_pct': improvement_over_additive,
        'interaction_mae': mean_absolute_error(target_interactions.flatten(), 
                                             pred_interactions.flatten()),
        'interaction_r2': r2_score(target_interactions.flatten(), 
                                 pred_interactions.flatten()),
        'interaction_classification_accuracy': interaction_accuracy,
        'synergy_precision': precision_score(target_synergy.flatten(), 
                                           pred_synergy.flatten()),
        'antagonism_precision': precision_score(target_antagonism.flatten(), 
                                              pred_antagonism.flatten()),
        'additive_precision': precision_score(target_additive.flatten(), 
                                            pred_additive.flatten())
    }
    
    return metrics


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score, handling edge cases."""
    try:
        from sklearn.metrics import precision_score as sk_precision
        return sk_precision(y_true, y_pred, zero_division=0)
    except:
        # Fallback calculation
        tp = np.sum(y_true & y_pred)
        fp = np.sum(~y_true & y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def plot_prediction_scatter(
    predictions: np.ndarray, 
    targets: np.ndarray,
    sample_size: int = 10000,
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot of predictions vs targets.
    
    Args:
        predictions: Predicted values
        targets: True values
        sample_size: Number of points to plot (for performance)
        save_path: Path to save plot
    """
    # Flatten and sample
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    if len(pred_flat) > sample_size:
        indices = np.random.choice(len(pred_flat), sample_size, replace=False)
        pred_flat = pred_flat[indices]
        target_flat = target_flat[indices]
    
    # Calculate metrics for plot
    r2 = r2_score(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    pearson_corr, _ = pearsonr(pred_flat, target_flat)
    
    # Create plot
    plt.figure(figsize=(8, 8))
    plt.scatter(target_flat, pred_flat, alpha=0.5, s=1)
    
    # Add perfect prediction line
    min_val = min(target_flat.min(), pred_flat.min())
    max_val = max(target_flat.max(), pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs True Values\nR² = {r2:.3f}, MAE = {mae:.3f}, r = {pearson_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Make square
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_gene_wise_performance(
    predictions: np.ndarray,
    targets: np.ndarray,
    top_k: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Plot gene-wise performance metrics.
    
    Args:
        predictions: Predicted values (n_samples, n_genes)
        targets: True values (n_samples, n_genes)
        top_k: Number of top/bottom genes to highlight
        save_path: Path to save plot
    """
    n_genes = predictions.shape[1]
    gene_r2_scores = []
    gene_mae_scores = []
    
    for gene_idx in range(n_genes):
        gene_pred = predictions[:, gene_idx]
        gene_target = targets[:, gene_idx]
        
        r2 = r2_score(gene_target, gene_pred)
        mae = mean_absolute_error(gene_target, gene_pred)
        
        gene_r2_scores.append(r2)
        gene_mae_scores.append(mae)
    
    gene_r2_scores = np.array(gene_r2_scores)
    gene_mae_scores = np.array(gene_mae_scores)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R2 distribution
    axes[0, 0].hist(gene_r2_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(gene_r2_scores.mean(), color='red', linestyle='--', 
                      label=f'Mean: {gene_r2_scores.mean():.3f}')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_ylabel('Number of Genes')
    axes[0, 0].set_title('Distribution of Gene-wise R² Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE distribution
    axes[0, 1].hist(gene_mae_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(gene_mae_scores.mean(), color='red', linestyle='--',
                      label=f'Mean: {gene_mae_scores.mean():.3f}')
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_ylabel('Number of Genes')
    axes[0, 1].set_title('Distribution of Gene-wise MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top and bottom performing genes by R2
    top_genes = np.argsort(gene_r2_scores)[-top_k:][::-1]
    bottom_genes = np.argsort(gene_r2_scores)[:top_k]
    
    axes[1, 0].bar(range(top_k), gene_r2_scores[top_genes])
    axes[1, 0].set_xlabel('Gene Rank')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title(f'Top {top_k} Genes by R² Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(range(top_k), gene_r2_scores[bottom_genes])
    axes[1, 1].set_xlabel('Gene Rank')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title(f'Bottom {top_k} Genes by R² Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_interaction_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    single_drug_a: np.ndarray,
    single_drug_b: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot interaction-specific analysis.
    
    Args:
        predictions: Predicted combination effects
        targets: True combination effects
        single_drug_a: Single drug A effects
        single_drug_b: Single drug B effects
        save_path: Path to save plot
    """
    # Calculate interactions
    additive_baseline = single_drug_a + single_drug_b
    target_interactions = targets - additive_baseline
    pred_interactions = predictions - additive_baseline
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Additive vs combination scatter
    sample_size = 5000
    if len(additive_baseline.flatten()) > sample_size:
        indices = np.random.choice(len(additive_baseline.flatten()), sample_size, replace=False)
        add_sample = additive_baseline.flatten()[indices]
        target_sample = targets.flatten()[indices]
        pred_sample = predictions.flatten()[indices]
    else:
        add_sample = additive_baseline.flatten()
        target_sample = targets.flatten()
        pred_sample = predictions.flatten()
    
    axes[0, 0].scatter(add_sample, target_sample, alpha=0.5, s=1, label='True')
    axes[0, 0].scatter(add_sample, pred_sample, alpha=0.5, s=1, label='Predicted')
    axes[0, 0].plot([add_sample.min(), add_sample.max()], 
                   [add_sample.min(), add_sample.max()], 'k--', alpha=0.8)
    axes[0, 0].set_xlabel('Additive Baseline')
    axes[0, 0].set_ylabel('Combination Effect')
    axes[0, 0].set_title('Combination vs Additive Effects')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Interaction scatter
    if len(target_interactions.flatten()) > sample_size:
        target_int_sample = target_interactions.flatten()[indices]
        pred_int_sample = pred_interactions.flatten()[indices]
    else:
        target_int_sample = target_interactions.flatten()
        pred_int_sample = pred_interactions.flatten()
    
    axes[0, 1].scatter(target_int_sample, pred_int_sample, alpha=0.5, s=1)
    int_min = min(target_int_sample.min(), pred_int_sample.min())
    int_max = max(target_int_sample.max(), pred_int_sample.max())
    axes[0, 1].plot([int_min, int_max], [int_min, int_max], 'r--', alpha=0.8)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].set_xlabel('True Interaction')
    axes[0, 1].set_ylabel('Predicted Interaction')
    axes[0, 1].set_title('Interaction Effects')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Interaction distribution
    axes[1, 0].hist(target_int_sample, bins=50, alpha=0.7, label='True', density=True)
    axes[1, 0].hist(pred_int_sample, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Interaction Strength')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Interaction Effects')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement over additive
    additive_errors = np.abs(target_sample - add_sample)
    model_errors = np.abs(target_sample - pred_sample)
    improvement = additive_errors - model_errors
    
    axes[1, 1].hist(improvement, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].axvline(improvement.mean(), color='green', linestyle='--',
                      label=f'Mean: {improvement.mean():.4f}')
    axes[1, 1].set_xlabel('Error Reduction vs Additive')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Model Improvement over Additive Baseline')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    single_drug_a: Optional[np.ndarray] = None,
    single_drug_b: Optional[np.ndarray] = None,
    save_dir: str = "evaluation_results"
) -> Dict:
    """
    Create comprehensive evaluation report with plots and metrics.
    
    Args:
        predictions: Model predictions
        targets: True values
        single_drug_a: Single drug A effects (for interaction analysis)
        single_drug_b: Single drug B effects (for interaction analysis)
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing all metrics
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Calculate basic metrics
    basic_metrics = calculate_metrics(predictions, targets)
    
    # Create plots
    plot_prediction_scatter(
        predictions, targets, 
        save_path=save_path / "prediction_scatter.png"
    )
    
    plot_gene_wise_performance(
        predictions, targets,
        save_path=save_path / "gene_wise_performance.png"
    )
    
    # Interaction analysis if single drug data provided
    if single_drug_a is not None and single_drug_b is not None:
        interaction_metrics = calculate_interaction_metrics(
            predictions, targets, single_drug_a, single_drug_b
        )
        
        plot_interaction_analysis(
            predictions, targets, single_drug_a, single_drug_b,
            save_path=save_path / "interaction_analysis.png"
        )
        
        # Combine metrics
        all_metrics = {**basic_metrics, **interaction_metrics}
    else:
        all_metrics = basic_metrics
    
    # Save metrics to file
    import json
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Evaluation report saved to {save_path}")
    return all_metrics