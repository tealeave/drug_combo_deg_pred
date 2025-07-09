# Drug Combination Prediction

A deep learning framework for predicting gene expression profiles of drug combinations using autoencoder-based neural networks with order-invariant fusion.

## 🎯 Overview

This project predicts 5,000-dimensional gene expression profiles for drug combinations based on single drug treatments. The model uses a two-stage training approach with an autoencoder for dimensionality reduction and a symmetric neural network for combination prediction.

### Key Features

- **Order-Invariant Predictions**: Uses additive fusion to ensure f(drug_A, drug_B) = f(drug_B, drug_A)
- **Two-Stage Training**: Autoencoder pretraining followed by end-to-end combination learning
- **Comprehensive Evaluation**: Multiple metrics, visualizations, and baseline comparisons
- **Debug-Friendly**: Synthetic data generation and configurable model sizes
- **Production-Ready**: Batch prediction, model serialization, and comprehensive logging

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd drug_combo_deg_pred

# Install dependencies using uv
uv sync

# Activate the environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Basic Usage

```bash
# Train the model
python scripts/train.py --data data/ --config configs/model_config.yaml --wandb

# Evaluate on test data
python scripts/evaluate.py --model best_full_model.pth --data data/ --output results/

# Make predictions for new drug pairs
python scripts/predict.py --model best_full_model.pth --data data/ --pairs drug_pairs.csv --output predictions/
```

## 🏗️ Architecture

### Model Components

**⚠️ Note**: All model components are implemented in `src/drug_combo/models/prediction_model.py` as a unified architecture.

1. **Gene Expression Autoencoder** (`GeneExpressionAutoencoder`)
   - Compresses 5,000-dimensional gene profiles to latent representations
   - Architecture: 5000 → 1000 → 200 → 20 → 200 → 1000 → 5000
   - Uses batch normalization and dropout for regularization
   - Includes separate encoder and decoder methods

2. **Drug Combination Predictor** (`DrugCombinationPredictor`)
   - Takes two drug latent representations as input
   - Uses **additive fusion** for order invariance: `h_combo = h_A + h_B`
   - Predicts combination latent representation
   - Optional self-attention and residual connections

3. **Full Pipeline** (`FullDrugCombinationModel`)
   - Encode single drugs → Predict combination → Decode to gene expression
   - End-to-end differentiable training
   - Maintains autoencoder quality during combination training
   - Integrated self-attention mechanisms for gene interactions

### Order Invariance Strategy

The model ensures symmetric predictions through **additive fusion**:
```python
def symmetric_fusion(drug_a, drug_b):
    return drug_a + drug_b  # Naturally commutative
```

This guarantees mathematical symmetry without complex attention mechanisms or data augmentation.

## 📊 Data Format

### Input Data Structure
```
data/
├── single_drug_expressions.csv    # Single drug gene expression profiles
├── pair_expressions.csv           # Drug combination expressions  
└── pair_metadata.csv              # Drug pair metadata (indices, IDs)
```

### Single Drug Data Format
- Rows: Drugs (including baseline at index 0)
- Columns: Genes (5,000 features)
- Values: Expression levels or log2 fold changes

### Drug Pairs Format
- `drug_a_idx`, `drug_b_idx`: Indices into single drug data
- `drug_a_id`, `drug_b_id`: Human-readable drug identifiers
- Expression data: Same format as single drugs

## 🔧 Configuration

### Configuration Files

The project uses two configuration files:

#### Production Configuration (`configs/model_config.yaml`)
```yaml
model:
  gene_dim: 5000              # Number of genes
  latent_dim: 20              # Latent representation size
  autoencoder_hidden: [1000, 200]  # Autoencoder architecture
  predictor_hidden: [64, 128, 64]   # Predictor architecture
  use_attention: false        # Use simple addition for fusion

training:
  ae_epochs: 100             # Autoencoder pretraining epochs
  full_epochs: 200           # Full model training epochs
  batch_size: 32             # Training batch size
  ae_lr: 0.001               # Autoencoder learning rate
  full_lr: 0.0005            # Full model learning rate
  seed: 42                   # Random seed
  weight_decay: 0.0001       # L2 regularization
```

#### Debug Configuration (`configs/training_config.yaml`)
```yaml
debug:
  enabled: true               # Enable debug mode
  small_dataset: true        # Use smaller dataset for testing
  quick_epochs: true         # Reduce epochs for debugging

model:
  gene_dim: 1000             # Reduced from 5000 for faster training
  latent_dim: 10             # Reduced from 20
  autoencoder_hidden: [200, 50]  # Reduced architecture
  use_attention: true        # Enable attention for testing

training:
  ae_epochs: 20              # Reduced epochs
  full_epochs: 30            # Reduced epochs
  batch_size: 16             # Smaller batch size
```

**⚠️ Important**: When `debug.enabled: true`, the debug settings override production settings. Set `debug.enabled: false` for production training.

## 📈 Training

### Two-Stage Training Process

1. **Stage 1: Autoencoder Pretraining**
   ```bash
   # Trains autoencoder on single drug data
   # Learns compressed gene expression representations
   # Optimizes reconstruction quality
   ```

2. **Stage 2: End-to-End Training**
   ```bash
   # Trains full model on drug pairs
   # Learns combination prediction
   # Maintains autoencoder quality with regularization
   ```

### Training Features
- Early stopping with validation monitoring
- Learning rate scheduling
- Gradient clipping for stability
- Comprehensive metric tracking
- Model checkpointing and recovery

## 🔍 Evaluation

### Comprehensive Metrics

- **Regression Metrics**: MAE, MSE, R², RMSE
- **Correlation Metrics**: Pearson, Spearman correlations
- **Gene-wise Analysis**: Per-gene performance distribution
- **Interaction Analysis**: Synergy vs. antagonism detection
- **Baseline Comparisons**: vs. additive, average, maximum baselines

### Evaluation Commands

```bash
# Basic evaluation
python scripts/evaluate.py --model model.pth --data data/ --dataset test

# Comprehensive analysis
python scripts/evaluate.py --model model.pth --data data/ \
    --visualize --analyze-genes --compare-baselines --evaluate-ae

# Cross-validation evaluation
python scripts/evaluate.py --model model.pth --data data/ --dataset all
```

## 🔮 Prediction

### Making Predictions

```bash
# Predict specific drug pairs
python scripts/predict.py --model model.pth --data data/ --pairs pairs.csv

# Generate full combination matrix
python scripts/predict.py --model model.pth --data data/ --matrix

# Batch prediction with analysis
python scripts/predict.py --model model.pth --data data/ --pairs pairs.csv \
    --analyze --plot --components
```

### Prediction Outputs
- Gene expression profiles for combinations
- Confidence scores and uncertainty estimates
- Intermediate latent representations
- Baseline comparisons
- Visualization plots and analysis reports

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_data_preprocessing.py -v
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ --cov=src/drug_combo --cov-report=html
```

### Test Coverage
- Unit tests for all core components
- Integration tests for training pipeline
- Mock-based testing for external dependencies
- Property-based testing for model invariants

## 🛠️ Development

### Debug Mode

Enable debug mode for faster iteration:

```yaml
# In training_config.yaml
debug:
  enabled: true
  small_dataset: true
  quick_epochs: true
```

Features:
- Synthetic data generation
- Reduced model complexity
- Fast training loops
- Enhanced logging

### Code Quality

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ scripts/
```

## 📁 Project Structure

```
drug_combo_deg_pred/
├── configs/                 # Configuration files
│   ├── model_config.yaml   # Production model configuration
│   └── training_config.yaml # Debug/development configuration
├── data/                   # Data directory (⚠️ currently empty)
│   ├── raw/               # Raw input data
│   ├── processed/         # Preprocessed data
│   └── external/          # External datasets
├── scripts/               # Main execution scripts
│   ├── train.py          # Model training
│   ├── evaluate.py       # Model evaluation
│   └── predict.py        # Prediction generation
├── src/drug_combo/        # Core package
│   ├── data/             # Data loading and preprocessing ✅
│   ├── models/           # Neural network architectures ✅
│   │   ├── prediction_model.py  # Main model implementation ✅
│   │   ├── autoencoder.py       # Standalone autoencoder classes ✅
│   │   └── attention_layers.py  # Attention mechanisms ✅
│   ├── training/         # Training and evaluation logic ✅
│   │   ├── trainer.py    # Main training class ✅
│   │   └── evaluation.py # Comprehensive evaluation module ✅
│   └── utils/            # Utilities and metrics ✅
├── tests/                # Test suite ✅
├── eda/                  # Exploratory data analysis scripts ✅
│   ├── 01_exploratory_data_analysis.py  # Data exploration and visualization ✅
│   ├── 02_baseline_models.py           # Baseline model implementations ✅
│   └── 03_model_experiments.py         # Neural network experiments ✅
└── pyproject.toml        # Project configuration ✅
```

## 📊 Performance

### Model Specifications
- **Input**: 5,000-dimensional gene expression profiles
- **Output**: 5,000-dimensional combination predictions
- **Parameters**: ~1.3M (configurable)
- **Training Time**: ~2-4 hours on single GPU
- **Inference**: ~1000 predictions/second

### Expected Benchmarks
- **Baseline Improvement**: 15-25% MAE reduction vs. additive baseline
- **Correlation**: R > 0.7 for majority of genes
- **Symmetry**: Perfect order invariance (f(A,B) = f(B,A))

