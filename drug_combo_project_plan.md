# Drug Combination Prediction Project Plan

## Project Overview
Predict gene expression profiles (5,000-dimensional vectors) for drug combinations based on single drug treatments using deep learning approaches in PyTorch.

## Problem Statement
- **Input**: Gene expression profiles from 1,000 single drug treatments + 10,000 drug pair experiments
- **Goal**: Predict gene expression profiles for remaining ~490,000 untested drug pairs
- **Data**: 5,000-dimensional vectors (bulk RNA-seq/gene expression profiles)
- **Type**: Regression problem with high-dimensional output

## Repository Structure (uv-managed)
```
drug-combo-prediction/
├── pyproject.toml          # uv configuration
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── drug_combo/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py
│   │   │   └── data_loader.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── autoencoder.py
│   │   │   ├── prediction_model.py
│   │   │   └── attention_layers.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   └── evaluation.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── metrics.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_model_experiments.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
└── configs/
    ├── model_config.yaml
    └── training_config.yaml
```

## Phase 1: Data Preprocessing & EDA

### 1.1 Data Loading and Validation
- Load single drug expression profiles (1,001 samples: 1,000 drugs + baseline)
- Load drug pair expression profiles (10,000 samples)
- Validate data integrity, check for missing values
- Calculate differential expression (delta) profiles against baseline

### 1.2 Exploratory Data Analysis
- Distribution analysis of differential expression values
- Correlation analysis between genes
- Clustering analysis of drug effects
- Visualization of high-variance genes
- Identify potential outliers or batch effects

### 1.3 Data Preprocessing Pipeline
- Standardization (zero mean, unit variance)
- Data augmentation for symmetry (drug A + drug B = drug B + drug A)
- Train/validation/test split (80/10/10)
- Create data loaders with proper batching

## Phase 2: Dimensionality Reduction

### 2.1 Autoencoder Development
- Design variational autoencoder for 5,000 → latent dimension reduction
- Experiment with latent dimensions (5, 10, 20, 50)
- Train on single drug differential expressions
- Evaluate reconstruction quality

### 2.2 Alternative Approaches
- Principal Component Analysis baseline
- Clustering-based dimensionality reduction
- Gene pathway-aware grouping (if permitted)

## Phase 3: Baseline Models (Week 3)

### 3.1 Simple Baselines
- Linear addition of single drug effects
- XGBoost regression on concatenated features
- Simple MLP with concatenated inputs

### 3.2 Evaluation Framework
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Correlation coefficients per gene
- R² scores for overall model performance

## Phase 4: Advanced Neural Network Models

### 4.1 Symmetry-Aware Architecture
```python
class DrugCombinationPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        # Encoder for single drugs
        # Symmetric combination layer
        # Decoder for pair prediction
        # Self-attention mechanisms
```

### 4.2 Architecture Experiments
- **Input handling approaches**:
  - Concatenation with data augmentation
  - Element-wise addition/subtraction
  - Attention-based fusion
- **Attention mechanisms**:
  - Self-attention within gene expressions
  - Cross-attention between drug representations
- **Inductive biases**:
  - Residual connections for additive effects
  - Gating mechanisms for interaction detection

### 4.3 Advanced Features
- Multi-head attention for different interaction types
- Transformer-style architecture
- Diffusion model components for generation

## Phase 5: Model Training & Optimization

### 5.1 Training Strategy
- Progressive training: start with autoencoder, then full model
- Learning rate scheduling
- Early stopping with validation monitoring
- Gradient clipping for stability

### 5.2 Hyperparameter Optimization
- Latent dimension size
- Network depth and width
- Attention head numbers
- Learning rates and batch sizes

### 5.3 Regularization
- Dropout layers
- Weight decay
- Batch normalization
- Data augmentation strategies

## Phase 6: Evaluation & Analysis

### 6.1 Comprehensive Evaluation
- Cross-validation on drug pairs
- Gene-wise performance analysis
- Interaction type classification (additive vs. synergistic vs. antagonistic)
- Comparison with biological expectations

### 6.2 Interpretability
- Attention weight visualization
- Feature importance analysis
- Latent space exploration
- Error analysis for failure cases

## Phase 7: Production & Documentation

### 7.1 Model Deployment Preparation
- Model serialization and versioning
- Inference pipeline optimization
- Batch prediction capabilities
- Performance benchmarking

### 7.2 Documentation
- Technical documentation
- Model architecture explanations
- Training procedures
- API documentation for predictions

## Key Technical Considerations

### Symmetry Handling
- Data augmentation: Include both (drug_A, drug_B) and (drug_B, drug_A)
- Architectural symmetry: Use addition/subtraction operations
- Loss function considerations for symmetric pairs

### High-Dimensional Output
- Multi-task learning framework
- Gene-wise attention mechanisms
- Structured prediction approaches
- Dimensionality reduction in output space

### Inductive Biases
- Additive baseline assumption
- Biological pathway constraints
- Sparsity assumptions for gene interactions

## Success Metrics
1. **Primary**: Mean Absolute Error < baseline linear addition
2. **Secondary**: High correlation (r > 0.7) for majority of genes
3. **Biological**: Sensible predictions for known drug interactions
4. **Computational**: Training time < 24 hours on single GPU

## Risk Mitigation
- **Overfitting**: Strong validation framework, regularization
- **Computational**: Progressive complexity, efficient architectures
- **Data quality**: Robust preprocessing, outlier detection
- **Interpretability**: Attention visualization, ablation studies

## Tools & Dependencies
- **Environment**: uv for dependency management
- **Core**: PyTorch, NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Experiment tracking**: Weights & Biases or MLflow
- **Testing**: pytest for model validation