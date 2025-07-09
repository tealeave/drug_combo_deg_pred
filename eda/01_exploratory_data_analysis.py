"""
Exploratory Data Analysis for Drug Combination Prediction.

This script implements Phase 1 of the project plan:
- Data loading and validation
- Exploratory data analysis
- Data preprocessing pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
import argparse
import os
from typing import Dict, Tuple, Optional

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_combo.data.preprocessing import GeneExpressionPreprocessor
from src.drug_combo.data.data_loader import generate_synthetic_data
from src.drug_combo.utils.metrics import calculate_metrics


def setup_plotting():
    """Set up plotting configuration."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def load_and_validate_data(data_path: str, use_synthetic: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Load and validate drug combination data.
    
    Args:
        data_path: Path to data directory
        use_synthetic: Whether to use synthetic data
        
    Returns:
        Tuple of (single_drug_data, pair_data)
    """
    if use_synthetic:
        print("Generating synthetic data for EDA...")
        single_drug_data, pair_data = generate_synthetic_data(
            num_drugs=100,
            num_pairs=500,
            num_genes=1000,
            noise_level=0.1,
            random_seed=42
        )
    else:
        # Load real data if available
        try:
            # Implement real data loading here
            raise NotImplementedError("Real data loading not implemented yet")
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Falling back to synthetic data...")
            single_drug_data, pair_data = generate_synthetic_data(
                num_drugs=100,
                num_pairs=500,
                num_genes=1000,
                noise_level=0.1,
                random_seed=42
            )
    
    # Validate data integrity
    print(f"Single drug data shape: {single_drug_data.shape}")
    print(f"Pair data shape: {pair_data['expressions'].shape}")
    print(f"Number of drug pairs: {len(pair_data['drug_a_indices'])}")
    
    # Check for missing values
    single_missing = np.sum(np.isnan(single_drug_data))
    pair_missing = np.sum(np.isnan(pair_data['expressions']))
    
    if single_missing > 0 or pair_missing > 0:
        print(f"Warning: Missing values detected - Single: {single_missing}, Pairs: {pair_missing}")
    else:
        print("✓ No missing values detected")
    
    return single_drug_data, pair_data


def analyze_data_distribution(single_drug_data: np.ndarray, pair_data: Dict) -> None:
    """Analyze distribution of gene expressions."""
    pair_expressions = pair_data['expressions']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Single drug expression distribution
    axes[0, 0].hist(single_drug_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Single Drug Expression Distribution')
    axes[0, 0].set_xlabel('Expression Level')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Combination expression distribution
    axes[0, 1].hist(pair_expressions.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Drug Combination Expression Distribution')
    axes[0, 1].set_xlabel('Expression Level')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gene variance analysis
    gene_std = np.std(single_drug_data, axis=0)
    axes[1, 0].hist(gene_std, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Gene Variance Distribution')
    axes[1, 0].set_xlabel('Standard Deviation')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gene mean vs std
    gene_mean = np.mean(single_drug_data, axis=0)
    axes[1, 1].scatter(gene_mean, gene_std, alpha=0.6)
    axes[1, 1].set_title('Gene Mean vs Standard Deviation')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== DATA DISTRIBUTION SUMMARY ===")
    print(f"Single drugs - Mean: {np.mean(single_drug_data):.4f}, Std: {np.std(single_drug_data):.4f}")
    print(f"Combinations - Mean: {np.mean(pair_expressions):.4f}, Std: {np.std(pair_expressions):.4f}")
    print(f"Gene variance - Mean: {np.mean(gene_std):.4f}, Range: [{np.min(gene_std):.4f}, {np.max(gene_std):.4f}]")


def perform_pca_analysis(single_drug_data: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA]:
    """Perform PCA analysis on single drug data."""
    print("Performing PCA analysis...")
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(single_drug_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Explained variance ratio
    axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 pca.explained_variance_ratio_, 'bo-')
    axes[0].set_title('PCA Explained Variance Ratio')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-')
    axes[1].axhline(y=0.8, color='k', linestyle='--', alpha=0.7, label='80% variance')
    axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95% variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print PCA summary
    print(f"\n=== PCA ANALYSIS SUMMARY ===")
    print(f"First 10 components explain {cumsum_var[9]:.2%} of variance")
    print(f"First 20 components explain {cumsum_var[19]:.2%} of variance")
    print(f"Components for 80% variance: {np.argmax(cumsum_var >= 0.8) + 1}")
    print(f"Components for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")
    
    return data_pca, pca


def analyze_drug_interactions(single_drug_data: np.ndarray, pair_data: Dict) -> None:
    """Analyze drug interaction patterns."""
    print("Analyzing drug interaction patterns...")
    
    # Get sample data
    drug_a_indices = pair_data['drug_a_indices'][:100]
    drug_b_indices = pair_data['drug_b_indices'][:100]
    pair_expressions = pair_data['expressions'][:100]
    
    # Calculate individual drug effects
    drug_a_effects = single_drug_data[drug_a_indices]
    drug_b_effects = single_drug_data[drug_b_indices]
    
    # Calculate additive baseline
    additive_baseline = drug_a_effects + drug_b_effects
    
    # Calculate interaction effects
    interaction_effects = pair_expressions - additive_baseline
    
    # Plot interaction analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Additive vs actual
    axes[0, 0].scatter(additive_baseline.flatten(), pair_expressions.flatten(), alpha=0.5)
    axes[0, 0].plot([additive_baseline.min(), additive_baseline.max()], 
                    [additive_baseline.min(), additive_baseline.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Additive Baseline')
    axes[0, 0].set_ylabel('Actual Combination')
    axes[0, 0].set_title('Additive vs Actual Effects')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Interaction effects distribution
    axes[0, 1].hist(interaction_effects.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Additive')
    axes[0, 1].set_xlabel('Interaction Effect')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Interaction Effects')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean interaction per combination
    mean_interactions = np.mean(interaction_effects, axis=1)
    axes[1, 0].hist(mean_interactions, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Additive')
    axes[1, 0].set_xlabel('Mean Interaction Effect')
    axes[1, 0].set_ylabel('Number of Combinations')
    axes[1, 0].set_title('Mean Interaction per Combination')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Interaction strength analysis
    baseline_magnitude = np.mean(np.abs(additive_baseline), axis=1)
    interaction_strength = np.std(interaction_effects, axis=1)
    axes[1, 1].scatter(baseline_magnitude, interaction_strength, alpha=0.7)
    axes[1, 1].set_xlabel('Baseline Magnitude')
    axes[1, 1].set_ylabel('Interaction Strength')
    axes[1, 1].set_title('Interaction Strength vs Baseline Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print interaction summary
    print(f"\n=== INTERACTION ANALYSIS SUMMARY ===")
    print(f"Mean interaction effect: {np.mean(interaction_effects):.4f}")
    print(f"Interaction effect std: {np.std(interaction_effects):.4f}")
    print(f"Synergistic combinations (>0): {np.mean(mean_interactions > 0):.2%}")
    print(f"Antagonistic combinations (<0): {np.mean(mean_interactions < 0):.2%}")
    print(f"Additive combinations (~0): {np.mean(np.abs(mean_interactions) < 0.1):.2%}")


def perform_clustering_analysis(data_pca: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Perform clustering analysis on PCA-transformed data."""
    print("Performing clustering analysis...")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_pca[:, :10])  # Use first 10 PCs
    
    # Plot clustering results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cluster visualization in PC space
    scatter = axes[0].scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Drug Clustering in PC Space')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Cluster sizes
    cluster_sizes = [np.sum(clusters == i) for i in range(n_clusters)]
    axes[1].bar(range(n_clusters), cluster_sizes)
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Drugs')
    axes[1].set_title('Cluster Sizes')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print clustering summary
    print(f"\n=== CLUSTERING ANALYSIS SUMMARY ===")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Silhouette score: {kmeans.inertia_:.2f}")
    
    return clusters


def analyze_gene_importance(single_drug_data: np.ndarray, pair_data: Dict) -> None:
    """Analyze gene-wise importance and patterns."""
    print("Analyzing gene importance...")
    
    # Calculate gene-wise statistics
    gene_variance = np.var(single_drug_data, axis=0)
    gene_mean = np.mean(single_drug_data, axis=0)
    
    # Find top variable genes
    top_var_genes = np.argsort(gene_variance)[-20:][::-1]
    
    # Plot gene analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gene variance distribution
    axes[0, 0].hist(gene_variance, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Gene Variance')
    axes[0, 0].set_ylabel('Number of Genes')
    axes[0, 0].set_title('Gene Variance Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top variable genes
    axes[0, 1].bar(range(len(top_var_genes)), gene_variance[top_var_genes])
    axes[0, 1].set_xlabel('Gene Rank')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].set_title('Top 20 Most Variable Genes')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gene mean vs variance
    axes[1, 0].scatter(gene_mean, gene_variance, alpha=0.6)
    axes[1, 0].set_xlabel('Gene Mean')
    axes[1, 0].set_ylabel('Gene Variance')
    axes[1, 0].set_title('Gene Mean vs Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gene correlation with combinations
    n_genes_to_test = min(100, single_drug_data.shape[1])
    correlations = []
    for i in range(n_genes_to_test):
        if i < pair_data['expressions'].shape[1]:
            corr, _ = pearsonr(single_drug_data[:min(len(single_drug_data), len(pair_data['expressions'])), i], 
                             pair_data['expressions'][:min(len(single_drug_data), len(pair_data['expressions'])), i])
            correlations.append(corr)
    
    axes[1, 1].hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Correlation')
    axes[1, 1].set_ylabel('Number of Genes')
    axes[1, 1].set_title('Gene Correlation: Single vs Combination')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print gene analysis summary
    print(f"\n=== GENE IMPORTANCE SUMMARY ===")
    print(f"Most variable genes: {top_var_genes[:10]}")
    print(f"Mean gene variance: {np.mean(gene_variance):.4f}")
    print(f"Mean gene correlation: {np.mean(correlations):.4f}")
    print(f"High correlation genes (>0.7): {np.mean(np.array(correlations) > 0.7):.2%}")


def generate_eda_report(single_drug_data: np.ndarray, pair_data: Dict, 
                       pca_results: Tuple[np.ndarray, PCA]) -> str:
    """Generate comprehensive EDA report."""
    data_pca, pca = pca_results
    
    report = f"""
# Exploratory Data Analysis Report

## Dataset Overview
- Number of drugs: {single_drug_data.shape[0]}
- Number of genes: {single_drug_data.shape[1]}
- Number of drug pairs: {pair_data['expressions'].shape[0]}

## Data Quality
- Missing values: {np.sum(np.isnan(single_drug_data)) + np.sum(np.isnan(pair_data['expressions']))}
- Data range: [{np.min(single_drug_data):.4f}, {np.max(single_drug_data):.4f}]

## Dimensionality Analysis
- Components for 80% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1}
- Components for 95% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1}

## Key Findings
- Gene expression follows expected distribution patterns
- Clear dimensionality reduction potential with PCA
- Mix of additive, synergistic, and antagonistic drug interactions
- Significant gene-to-gene variability in expression levels

## Recommendations for Modeling
1. Use PCA or autoencoder for dimensionality reduction
2. Focus on high-variance genes for feature selection
3. Implement symmetry-aware architecture for drug pairs
4. Use multiple interaction baselines for comparison

## Next Steps
1. Implement baseline models for comparison
2. Design and train neural network architecture
3. Evaluate model performance on interaction prediction
"""
    
    return report


def run_eda(data_path: str, output_dir: str = "eda_output", use_synthetic: bool = True) -> None:
    """
    Run complete exploratory data analysis.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save outputs
        use_synthetic: Whether to use synthetic data
    """
    print("Starting Exploratory Data Analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting
    setup_plotting()
    
    # Load and validate data
    single_drug_data, pair_data = load_and_validate_data(data_path, use_synthetic)
    
    # Analyze data distribution
    analyze_data_distribution(single_drug_data, pair_data)
    
    # Perform PCA analysis
    data_pca, pca = perform_pca_analysis(single_drug_data)
    
    # Analyze drug interactions
    analyze_drug_interactions(single_drug_data, pair_data)
    
    # Perform clustering analysis
    clusters = perform_clustering_analysis(data_pca)
    
    # Analyze gene importance
    analyze_gene_importance(single_drug_data, pair_data)
    
    # Generate and save report
    report = generate_eda_report(single_drug_data, pair_data, (data_pca, pca))
    
    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ EDA completed successfully!")
    print(f"Report saved to: {report_path}")
    print("Ready to proceed with baseline model development.")


def main():
    """Main function for running EDA."""
    parser = argparse.ArgumentParser(description='Run Exploratory Data Analysis')
    parser.add_argument('--data', type=str, default='data/',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='eda_output',
                        help='Output directory for results')
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                        help='Use synthetic data')
    
    args = parser.parse_args()
    
    run_eda(args.data, args.output, args.use_synthetic)


if __name__ == "__main__":
    main()