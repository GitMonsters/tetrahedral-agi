"""
Training and Optimization Framework for 64-Point Tetrahedron AI
Implements specialized training procedures for geometric deep learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
import wandb
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..neural_network.tetrahedral_network import TetrahedralAGINetwork
from ..geometry.tetrahedral_grid import TetrahedralGrid


@dataclass
class TrainingConfig:
    """Configuration for training the tetrahedral AI"""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Optimization parameters
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 10
    
    # Loss function parameters
    loss_type: str = 'geometric_mse'
    geometric_weight: float = 0.1
    attention_weight: float = 0.05
    
    # Regularization
    dropout_rate: float = 0.1
    geometric_regularization: float = 0.01
    
    # Hardware
    device: str = 'cuda'
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: str = './checkpoints'
    use_wandb: bool = True
    
    # Data
    data_dir: str = './data'
    val_split: float = 0.2
    test_split: float = 0.1


class GeometricDataset(Dataset):
    """Dataset for geometric 3D data"""
    
    def __init__(self, data_path: str, mode: str = 'train', grid: Optional[TetrahedralGrid] = None):
        self.data_path = Path(data_path)
        self.mode = mode
        self.grid = grid or TetrahedralGrid()
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load geometric data from files"""
        data = []
        
        # This is a placeholder - in practice, you'd load actual 3D data
        # For now, we generate synthetic geometric data
        num_samples = 1000 if self.mode == 'train' else 200
        
        for i in range(num_samples):
            # Generate random 3D point cloud
            num_points = np.random.randint(50, 100)
            points = torch.randn(num_points, 3)
            
            # Generate random features
            features = torch.randn(num_points, 6)
            
            # Generate target labels
            labels = torch.randint(0, 10, (num_points,))
            
            data.append({
                'points': points,
                'features': features,
                'labels': labels,
                'grid_indices': self._map_to_grid(points)
            })
        
        return data
    
    def _map_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Map arbitrary points to the tetrahedral grid"""
        # Simple nearest neighbor mapping
        grid_points = self.grid.points
        indices = []
        
        for point in points:
            distances = torch.norm(grid_points - point.unsqueeze(0), dim=1)
            nearest_idx = torch.argmin(distances)
            indices.append(nearest_idx)
        
        return torch.tensor(indices)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class GeometricLoss(nn.Module):
    """Specialized loss function for geometric deep learning"""
    
    def __init__(self, config: TrainingConfig):
        super(GeometricLoss, self).__init__()
        self.config = config
        self.geometric_weight = config.geometric_weight
        self.attention_weight = config.attention_weight
        
        # Base loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                geometric_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute geometric loss
        Args:
            predictions: Model predictions
            targets: Target values
            attention_weights: Spatial attention weights
            geometric_features: Geometric features for regularization
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Main task loss
        if self.config.loss_type == 'mse':
            main_loss = self.mse_loss(predictions, targets)
        elif self.config.loss_type == 'cross_entropy':
            main_loss = self.ce_loss(predictions, targets)
        else:
            main_loss = self.mse_loss(predictions, targets)
        
        losses['main_loss'] = main_loss
        
        # Geometric regularization
        if geometric_features is not None:
            geo_loss = self._compute_geometric_loss(geometric_features)
            losses['geometric_loss'] = self.geometric_weight * geo_loss
        
        # Attention regularization
        if attention_weights is not None:
            attention_loss = self._compute_attention_loss(attention_weights)
            losses['attention_loss'] = self.attention_weight * attention_loss
        
        # Total loss
        total_loss = main_loss
        if 'geometric_loss' in losses:
            total_loss += losses['geometric_loss']
        if 'attention_loss' in losses:
            total_loss += losses['attention_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_geometric_loss(self, geometric_features: torch.Tensor) -> torch.Tensor:
        """Compute geometric regularization loss"""
        # Encourage smooth geometric features
        smoothness_loss = torch.mean(torch.diff(geometric_features, dim=-1) ** 2)
        
        # Encourage geometric consistency
        consistency_loss = torch.std(geometric_features)
        
        return smoothness_loss + consistency_loss
    
    def _compute_attention_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute attention regularization loss"""
        # Encourage sparse attention
        sparsity_loss = torch.mean(torch.abs(attention_weights))
        
        # Encourage diverse attention patterns
        diversity_loss = -torch.mean(torch.std(attention_weights, dim=-1))
        
        return sparsity_loss + diversity_loss


class TetrahedralTrainer:
    """Specialized trainer for tetrahedral AI networks"""
    
    def __init__(self, 
                 model: TetrahedralAGINetwork,
                 config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.loss_fn = GeometricLoss(config)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Logging
        self.training_history = []
        self.best_val_loss = float('inf')
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project='tetrahedral-agi', config=config.__dict__)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def setup_data_loaders(self, dataset: GeometricDataset):
        """Setup data loaders for training"""
        # Split dataset
        total_size = len(dataset)
        val_size = int(total_size * self.config.val_split)
        test_size = int(total_size * self.config.test_split)
        train_size = total_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        total_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                # Prepare input
                points = batch['points']  # [batch_size, num_points, 3]
                features = batch['features']  # [batch_size, num_points, 6]
                
                # Combine points and features
                input_data = torch.cat([points, features], dim=-1)  # [batch_size, num_points, 9]
                input_data = input_data.transpose(1, 2)  # [batch_size, 9, num_points]
                
                # Get model predictions
                predictions = self.model(input_data)
                
                # Get attention weights for regularization
                attention_weights = self.model.get_attention_weights()
                
                # Compute loss
                targets = batch['labels'].float()
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                losses = self.loss_fn(predictions, targets, attention_weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(losses['total_loss']).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                self.optimizer.step()
            
            # Update running losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'main': f"{losses['main_loss'].item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= total_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {}
        total_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # Prepare input
                    points = batch['points']
                    features = batch['features']
                    input_data = torch.cat([points, features], dim=-1)
                    input_data = input_data.transpose(1, 2)
                    
                    # Get predictions
                    predictions = self.model(input_data)
                    attention_weights = self.model.get_attention_weights()
                    
                    # Compute loss
                    targets = batch['labels'].float()
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    
                    losses = self.loss_fn(predictions, targets, attention_weights)
                
                # Update running losses
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= total_batches
        
        return val_losses
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train epoch
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total_loss'])
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            training_history['train_loss'].append(train_losses['total_loss'])
            training_history['val_loss'].append(val_losses['total_loss'])
            training_history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss'],
                    'learning_rate': current_lr,
                    **{f'train_{k}': v for k, v in train_losses.items()},
                    **{f'val_{k}': v for k, v in val_losses.items()}
                })
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total_loss'])
            
            # Update best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, val_losses['total_loss'], is_best=True)
        
        return training_history
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        else:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']


class HyperparameterOptimizer:
    """Hyperparameter optimization for tetrahedral AI"""
    
    def __init__(self, base_config: TrainingConfig):
        self.base_config = base_config
        self.best_config = None
        self.best_score = float('inf')
    
    def objective(self, trial) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        config = TrainingConfig(
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            geometric_weight=trial.suggest_uniform('geometric_weight', 0.01, 0.5),
            attention_weight=trial.suggest_uniform('attention_weight', 0.01, 0.2),
            num_epochs=50  # Reduced for hyperparameter search
        )
        
        # Create and train model
        model = TetrahedralAGINetwork(device=config.device)
        trainer = TetrahedralTrainer(model, config)
        
        # Setup data
        dataset = GeometricDataset(config.data_dir)
        trainer.setup_data_loaders(dataset)
        
        # Train and get final validation loss
        training_history = trainer.train()
        final_val_loss = min(training_history['val_loss'])
        
        # Update best
        if final_val_loss < self.best_score:
            self.best_score = final_val_loss
            self.best_config = config
        
        return final_val_loss
    
    def optimize(self, n_trials: int = 100) -> TrainingConfig:
        """Run hyperparameter optimization"""
        import optuna
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")
        
        return self.best_config


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create model and trainer
    model = TetrahedralAGINetwork(device=config.device)
    trainer = TetrahedralTrainer(model, config)
    
    # Setup data
    dataset = GeometricDataset('./data')
    trainer.setup_data_loaders(dataset)
    
    # Train
    training_history = trainer.train()
    
    print("Training completed!")
    print(f"Final validation loss: {min(training_history['val_loss']):.4f}")