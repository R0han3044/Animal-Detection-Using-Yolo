"""
Training pipeline for YOLO model
"""
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from models.yolo import create_model, create_loss
from utils.dataset import create_dataloaders
from utils.metrics import MetricsCalculator
from utils.visualization import plot_training_history
from config import MODEL_CONFIG, TRAINING_CONFIG

class YOLOTrainer:
    """YOLO model trainer"""
    
    def __init__(self, model_config=None, training_config=None):
        self.model_config = model_config or MODEL_CONFIG
        self.training_config = training_config or TRAINING_CONFIG
        
        # Create directories
        os.makedirs(self.training_config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.training_config['log_dir'], exist_ok=True)
        
        # Initialize model
        self.device = torch.device(self.model_config['device'])
        self.model = create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = create_loss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.training_config['patience'] // 2,
            verbose=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.training_config['log_dir'])
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_map': [],
            'val_map': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'learning_rate': []
        }
        
        # Best metrics tracking
        self.best_map = 0.0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(dataloader, desc='Training')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Coord': f'{loss_dict["coord_loss"].item():.4f}',
                'Conf': f'{loss_dict["conf_obj_loss"].item():.4f}',
                'Class': f'{loss_dict["class_loss"].item():.4f}'
            })
            
            # Log batch metrics to tensorboard
            if batch_idx % 10 == 0:
                global_step = len(dataloader) * (self.current_epoch - 1) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', total_loss.item(), global_step)
                self.writer.add_scalar('Train/Coord_Loss', loss_dict['coord_loss'].item(), global_step)
                self.writer.add_scalar('Train/Conf_Loss', loss_dict['conf_obj_loss'].item(), global_step)
                self.writer.add_scalar('Train/Class_Loss', loss_dict['class_loss'].item(), global_step)
        
        avg_loss = epoch_loss / num_batches
        return avg_loss, metrics_calc.compute_metrics()
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(dataloader, desc='Validation')
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
                
                epoch_loss += total_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Val_Loss': f'{total_loss.item():.4f}'
                })
        
        avg_loss = epoch_loss / num_batches
        return avg_loss, metrics_calc.compute_metrics()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.training_config['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.training_config['checkpoint_dir'], 
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved with mAP: {self.best_map:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_map = checkpoint['best_map']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs=None, resume_from=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.model_config['epochs']
        
        # Load checkpoint if resuming
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            batch_size=self.model_config['batch_size']
        )
        
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_map'].append(train_metrics.get('mAP@0.5', 0.0))
            self.history['val_map'].append(val_metrics.get('mAP@0.5', 0.0))
            self.history['train_precision'].append(train_metrics.get('precision', 0.0))
            self.history['val_precision'].append(val_metrics.get('precision', 0.0))
            self.history['train_recall'].append(train_metrics.get('recall', 0.0))
            self.history['val_recall'].append(val_metrics.get('recall', 0.0))
            self.history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Train/mAP', train_metrics.get('mAP@0.5', 0.0), epoch)
            self.writer.add_scalar('Val/mAP', val_metrics.get('mAP@0.5', 0.0), epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train mAP: {train_metrics.get('mAP@0.5', 0.0):.4f}, Val mAP: {val_metrics.get('mAP@0.5', 0.0):.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Check for improvement
            val_map = val_metrics.get('mAP@0.5', 0.0)
            is_best = val_map > self.best_map
            
            if is_best:
                self.best_map = val_map
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % self.training_config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.training_config['patience']:
                print(f"\nEarly stopping after {self.epochs_without_improvement} epochs without improvement")
                break
        
        # Save final model
        self.save_checkpoint(epoch, False)
        
        # Plot training history
        history_plot = plot_training_history(self.history)
        history_plot.savefig(
            os.path.join(self.training_config['log_dir'], 'training_history.png'),
            dpi=300, bbox_inches='tight'
        )
        
        # Save training history
        with open(os.path.join(self.training_config['log_dir'], 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best mAP: {self.best_map:.4f}")
        print(f"Final Loss: {val_loss:.4f}")
        
        self.writer.close()
        
        return self.history

def train_model(config_overrides=None):
    """
    Main function to train YOLO model
    Args:
        config_overrides: Dictionary to override default configs
    Returns:
        Trained model and training history
    """
    # Apply config overrides
    model_config = MODEL_CONFIG.copy()
    training_config = TRAINING_CONFIG.copy()
    
    if config_overrides:
        if 'model' in config_overrides:
            model_config.update(config_overrides['model'])
        if 'training' in config_overrides:
            training_config.update(config_overrides['training'])
    
    # Create trainer
    trainer = YOLOTrainer(model_config, training_config)
    
    # Start training
    history = trainer.train()
    
    return trainer.model, history

if __name__ == "__main__":
    # Example usage
    model, history = train_model()
