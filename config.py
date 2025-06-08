"""
Configuration file for YOLO animal detection system
"""
import os

# Model Configuration
MODEL_CONFIG = {
    'num_classes': 10,  # Common animals: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
    'input_size': 416,
    'grid_size': 13,
    'num_boxes': 2,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 100,
    'device': 'cuda' if os.getenv('CUDA_AVAILABLE', '0') == '1' else 'cpu'
}

# Dataset Configuration
DATASET_CONFIG = {
    'train_split': 0.8,
    'val_split': 0.2,
    'data_dir': 'data',
    'annotations_file': 'annotations.json'
}

# Training Configuration
TRAINING_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'save_every': 10,
    'patience': 20,
    'weight_decay': 0.0005,
    'momentum': 0.9
}

# Class names for animal detection
CLASS_NAMES = [
    'dog', 'cat', 'bird', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

# Colors for visualization (RGB)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (255, 165, 0), (128, 128, 128), (255, 192, 203)
]
