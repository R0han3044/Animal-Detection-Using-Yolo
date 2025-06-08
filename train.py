#!/usr/bin/env python3
"""
Complete training script for YOLO animal detection
This script will train the model from scratch to achieve high accuracy
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from models.yolo import create_model
from utils.dataset import AnimalDataset, create_dataloaders
from utils.loss import create_loss_function
from utils.metrics import MetricsCalculator
from utils.transforms import get_transforms
from config import MODEL_CONFIG, TRAINING_CONFIG, CLASS_NAMES

def create_synthetic_dataset():
    """Create a synthetic dataset for training demonstration"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    
    # Create synthetic annotations
    annotations = []
    
    for i in range(500):  # Create 500 training samples
        annotations.append({
            'image_path': f'synthetic_image_{i:03d}.jpg',
            'width': 416,
            'height': 416,
            'objects': [
                {
                    'class': CLASS_NAMES[i % len(CLASS_NAMES)],
                    'bbox': [
                        np.random.randint(10, 150),  # x
                        np.random.randint(10, 150),  # y
                        np.random.randint(100, 250), # width
                        np.random.randint(100, 250)  # height
                    ]
                }
            ]
        })
    
    # Save annotations
    with open('data/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created synthetic dataset with {len(annotations)} samples")
    return annotations

def train_model_complete():
    """Complete training function"""
    print("Starting YOLO Animal Detection Training")
    print("=" * 50)
    
    # Create synthetic dataset
    create_synthetic_dataset()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model.to(device)
    
    # Create loss function
    criterion = create_loss_function()
    
    # Create optimizer with advanced settings for better convergence
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=MODEL_CONFIG['epochs'],
        eta_min=1e-6
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        batch_size=MODEL_CONFIG['batch_size']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(MODEL_CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{MODEL_CONFIG['epochs']}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss_dict = criterion(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{train_batches}, Loss: {total_loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                val_loss += loss_dict['total_loss'].item()
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(TRAINING_CONFIG['checkpoint_dir'], exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'history': history
            }
            
            torch.save(checkpoint, os.path.join(TRAINING_CONFIG['checkpoint_dir'], 'best_model.pth'))
            print(f"New best model saved! Loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    print("\nTraining completed successfully!")
    print(f"Best validation loss: {best_loss:.4f}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO Animal Detection Model')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['epochs'], help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=MODEL_CONFIG['learning_rate'], help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    MODEL_CONFIG['epochs'] = args.epochs
    MODEL_CONFIG['batch_size'] = args.batch_size
    MODEL_CONFIG['learning_rate'] = args.lr
    
    # Start training
    trained_model, training_history = train_model_complete()
    
    print("Training script completed successfully!")