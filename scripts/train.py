"""
Training script for drug combination prediction model.
Uses the DrugCombinationTrainer from the training module.
"""

import torch
import numpy as np
import yaml
import argparse
import wandb

from src.drug_combo.training.trainer import DrugCombinationTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train drug combination prediction model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--project', type=str, default='drug-combo-prediction',
                        help='Wandb project name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=args.project,
            config=config,
            name=f"exp_{config['model']['latent_dim']}d_{'attn' if config['model']['use_attention'] else 'no_attn'}"
        )
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Initialize trainer
    trainer = DrugCombinationTrainer(config)
    
    print(f"Using device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(args.data)
    
    # Load single drug data for autoencoder pretraining
    # Get single drug data from preprocessing
    from src.drug_combo.data.preprocessing import preprocess_data
    single_drug_data, _ = preprocess_data(args.data, config)
    
    # Stage 1: Train autoencoder
    trainer.train_autoencoder(
        single_drug_data, 
        epochs=config['training']['ae_epochs']
    )
    
    # Stage 2: Train full model
    trainer.train_full_model(
        train_loader, 
        val_loader, 
        epochs=config['training']['full_epochs']
    )
    
    # Final evaluation
    print("Evaluating on test set...")
    test_metrics = trainer.comprehensive_evaluation(test_loader)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    if args.wandb:
        wandb.log(test_metrics)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'training_history': trainer.history
    }, 'final_model.pth')
    
    print("Training completed!")


if __name__ == "__main__":
    main()