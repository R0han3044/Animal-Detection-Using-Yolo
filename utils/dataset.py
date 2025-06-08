"""
Dataset utilities for YOLO animal detection
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from config import DATASET_CONFIG, MODEL_CONFIG, CLASS_NAMES
from utils.transforms import get_transforms

class AnimalDataset(Dataset):
    """Custom dataset for animal detection"""
    def __init__(self, data_dir, annotations_file, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.grid_size = MODEL_CONFIG['grid_size']
        self.num_classes = MODEL_CONFIG['num_classes']
        self.num_boxes = MODEL_CONFIG['num_boxes']
        self.input_size = MODEL_CONFIG['input_size']
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    def _load_annotations(self, annotations_file):
        """Load annotations from JSON file"""
        if not os.path.exists(annotations_file):
            # Create dummy annotations for demonstration
            return self._create_dummy_annotations()
        
        with open(annotations_file, 'r') as f:
            return json.load(f)
    
    def _create_dummy_annotations(self):
        """Create dummy annotations for demonstration purposes"""
        # This is a fallback when no real dataset is available
        annotations = []
        for i in range(100):  # Create 100 dummy samples
            annotations.append({
                'image_path': f'dummy_image_{i}.jpg',
                'width': 416,
                'height': 416,
                'objects': [
                    {
                        'class': CLASS_NAMES[i % len(CLASS_NAMES)],
                        'bbox': [50, 50, 200, 200]  # [x, y, width, height]
                    }
                ]
            })
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Create dummy image if file doesn't exist
        image_path = os.path.join(self.data_dir, annotation['image_path'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create dummy image
            image = Image.new('RGB', (416, 416), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Create target tensor
        target = self._create_target(annotation)
        
        return image, target
    
    def _create_target(self, annotation):
        """Create target tensor from annotation"""
        target = torch.zeros(self.grid_size, self.grid_size, self.num_boxes, 5 + self.num_classes)
        
        for obj in annotation['objects']:
            class_name = obj['class']
            if class_name not in self.class_to_idx:
                continue
            
            class_idx = self.class_to_idx[class_name]
            bbox = obj['bbox']  # [x, y, width, height]
            
            # Convert to relative coordinates
            x_center = (bbox[0] + bbox[2] / 2) / annotation['width']
            y_center = (bbox[1] + bbox[3] / 2) / annotation['height']
            width = bbox[2] / annotation['width']
            height = bbox[3] / annotation['height']
            
            # Find grid cell
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)
            
            # Ensure grid indices are within bounds
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            
            # Relative position within grid cell
            x_offset = x_center * self.grid_size - grid_x
            y_offset = y_center * self.grid_size - grid_y
            
            # Assign to first available box
            for box_idx in range(self.num_boxes):
                if target[grid_y, grid_x, box_idx, 4] == 0:  # If no object assigned yet
                    target[grid_y, grid_x, box_idx, 0] = x_offset
                    target[grid_y, grid_x, box_idx, 1] = y_offset
                    target[grid_y, grid_x, box_idx, 2] = width
                    target[grid_y, grid_x, box_idx, 3] = height
                    target[grid_y, grid_x, box_idx, 4] = 1.0  # Confidence
                    target[grid_y, grid_x, box_idx, 5 + class_idx] = 1.0  # Class
                    break
        
        return target

def create_dataloaders(data_dir=DATASET_CONFIG['data_dir'], 
                      annotations_file=DATASET_CONFIG['annotations_file'],
                      batch_size=MODEL_CONFIG['batch_size']):
    """Create training and validation dataloaders"""
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    full_dataset = AnimalDataset(data_dir, annotations_file, train_transform, is_training=True)
    
    # Split dataset
    train_size = int(DATASET_CONFIG['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)
    return images, targets
