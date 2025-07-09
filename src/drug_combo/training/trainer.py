"""
Training script for drug combination prediction model.
Implements the two-stage training approach discussed in the interview.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.drug_combo.models.prediction_model import FullDrugCombinationModel
from src.drug_combo.data.preprocessing import preprocess_data
from .evaluation import ModelEvaluator


class DrugCombinationTrainer:
    """Trainer class for drug combination prediction model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FullDrugCombinationModel(
            gene_dim=config['model']['gene_dim'],
            latent_dim=config['model']['latent_dim'],
            autoencoder_hidden=config['model']['autoencoder_hidden'],
            predictor_hidden=config['model']['predictor_hidden'],
            use_attention=config['model']['use_attention']
        ).to(self.device)
        
        # Initialize optimizers for two-stage training
        self.ae_optimizer = optim.Adam(
            self.model.autoencoder.parameters(),
            lr=config['training']['ae_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.full_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['full_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_loss = nn.L1Loss()  # MAE as discussed
        
        # Learning rate schedulers
        self.ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.ae_optimizer, mode='min', patience=5, factor=0.5
        )
        self.full_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.full_optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.history = {
            'ae_train_loss': [], 'ae_val_loss': [],
            'full_train_loss': [], 'full_val_loss': []
        }
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training."""
        print("Loading and preprocessing data...")
        
        # Load data (implement based on the data format)
        single_drug_data, pair_data = preprocess_data(data_path, self.config)
        
        # Create datasets with symmetry augmentation
        train_loader, val_loader, test_loader = self._create_data_loaders(
            single_drug_data, pair_data
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_data_loaders(
        self, 
        single_drug_data: np.ndarray, 
        pair_data: Dict
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with proper train/val/test splits."""
        
        # Extract pair information
        drug_a_indices = pair_data['drug_a_indices']
        drug_b_indices = pair_data['drug_b_indices']
        pair_expressions = pair_data['expressions']
        
        # Get single drug expressions for pairs
        drug_a_expressions = single_drug_data[drug_a_indices]
        drug_b_expressions = single_drug_data[drug_b_indices]
        
        # Data augmentation: add symmetric pairs
        aug_drug_a = np.concatenate([drug_a_expressions, drug_b_expressions])
        aug_drug_b = np.concatenate([drug_b_expressions, drug_a_expressions])
        aug_targets = np.concatenate([pair_expressions, pair_expressions])
        
        # Train/val/test split
        n_samples = len(aug_drug_a)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(aug_drug_a[train_indices]),
            torch.FloatTensor(aug_drug_b[train_indices]),
            torch.FloatTensor(aug_targets[train_indices])
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(aug_drug_a[val_indices]),
            torch.FloatTensor(aug_drug_b[val_indices]),
            torch.FloatTensor(aug_targets[val_indices])
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(aug_drug_a[test_indices]),
            torch.FloatTensor(aug_drug_b[test_indices]),
            torch.FloatTensor(aug_targets[test_indices])
        )
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def train_autoencoder(
        self, 
        single_drug_data: np.ndarray, 
        epochs: int = 100
    ) -> None:
        """Pre-train the autoencoder on single drug data."""
        print("Stage 1: Training autoencoder...")
        
        # Create dataset for single drugs (including baseline)
        dataset = TensorDataset(torch.FloatTensor(single_drug_data))
        loader = DataLoader(
            dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.autoencoder.train()
            train_loss = 0.0
            
            for batch in tqdm(loader, desc=f"AE Epoch {epoch+1}"):
                x = batch[0].to(self.device)
                
                self.ae_optimizer.zero_grad()
                reconstructed, _ = self.model.autoencoder(x)
                loss = self.reconstruction_loss(reconstructed, x)
                loss.backward()
                self.ae_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(loader)
            
            # Validation (use same data for now, could split if needed)
            self.model.autoencoder.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(self.device)
                    reconstructed, _ = self.model.autoencoder(x)
                    loss = self.reconstruction_loss(reconstructed, x)
                    val_loss += loss.item()
            
            val_loss /= len(loader)
            
            # Update history
            self.history['ae_train_loss'].append(train_loss)
            self.history['ae_val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.ae_scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.autoencoder.state_dict(), 
                    'best_autoencoder.pth'
                )
            else:
                patience_counter += 1
                if patience_counter >= 15:  # Early stopping patience
                    print("Early stopping triggered for autoencoder")
                    break
    
    def train_full_model(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 200
    ) -> None:
        """Train the full model end-to-end."""
        print("Stage 2: Training full model...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for drug_a, drug_b, target in tqdm(train_loader, desc=f"Full Epoch {epoch+1}"):
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                target = target.to(self.device)
                
                self.full_optimizer.zero_grad()
                
                # Forward pass
                prediction = self.model(drug_a, drug_b)
                loss = self.prediction_loss(prediction, target)
                
                # Add reconstruction loss to maintain autoencoder quality
                drug_a_recon, _ = self.model.autoencoder(drug_a)
                drug_b_recon, _ = self.model.autoencoder(drug_b)
                recon_loss = (
                    self.reconstruction_loss(drug_a_recon, drug_a) + 
                    self.reconstruction_loss(drug_b_recon, drug_b)
                ) * 0.1  # Weight reconstruction loss lower
                
                total_loss = loss + recon_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.full_optimizer.step()
                train_loss += loss.item()  # Only track prediction loss
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = self.evaluate(val_loader)
            
            # Update history
            self.history['full_train_loss'].append(train_loss)
            self.history['full_val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.full_scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Wandb logging
            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.full_optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.full_optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss,
                    'config': self.config
                }, 'best_full_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:  # Early stopping patience
                    print("Early stopping triggered for full model")
                    break
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for drug_a, drug_b, target in data_loader:
                drug_a = drug_a.to(self.device)
                drug_b = drug_b.to(self.device)
                target = target.to(self.device)
                
                prediction = self.model(drug_a, drug_b)
                loss = self.prediction_loss(prediction, target)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def comprehensive_evaluation(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation with multiple metrics."""
        evaluator = ModelEvaluator(self.model, str(self.device))
        return evaluator.evaluate_basic_metrics(test_loader, "test")
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Autoencoder training
        if self.history['ae_train_loss']:
            axes[0].plot(self.history['ae_train_loss'], label='Train')
            axes[0].plot(self.history['ae_val_loss'], label='Validation')
            axes[0].set_title('Autoencoder Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('MSE Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Full model training
        if self.history['full_train_loss']:
            axes[1].plot(self.history['full_train_loss'], label='Train')
            axes[1].plot(self.history['full_val_loss'], label='Validation')
            axes[1].set_title('Full Model Training Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE Loss')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()