"""
Data augmentation and transformation utilities
"""
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import random
import numpy as np
from config import MODEL_CONFIG

class Resize:
    """Resize image to target size"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return image.resize((self.size, self.size), Image.BILINEAR)

class ToTensor:
    """Convert PIL image to tensor"""
    def __call__(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

class RandomHorizontalFlip:
    """Randomly flip image horizontally"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

class ColorJitter:
    """Randomly change brightness, contrast, saturation"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, image):
        # Random brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            image = enhancer.enhance(factor)
        
        # Random saturation
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            image = enhancer.enhance(factor)
        
        return image

class RandomRotation:
    """Randomly rotate image"""
    def __init__(self, degrees=10):
        self.degrees = degrees
    
    def __call__(self, image):
        angle = random.uniform(-self.degrees, self.degrees)
        return image.rotate(angle, Image.BILINEAR)

class Normalize:
    """Normalize image with mean and std"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

def get_transforms():
    """Get training and validation transforms"""
    
    # Training transforms with augmentation
    train_transforms = Compose([
        Resize(MODEL_CONFIG['input_size']),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        RandomRotation(degrees=10),
        ToTensor(),
        Normalize()
    ])
    
    # Validation transforms without augmentation
    val_transforms = Compose([
        Resize(MODEL_CONFIG['input_size']),
        ToTensor(),
        Normalize()
    ])
    
    return train_transforms, val_transforms

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def preprocess_image(image_path, target_size=MODEL_CONFIG['input_size']):
    """Preprocess single image for inference"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Apply transforms
    transform = Compose([
        Resize(target_size),
        ToTensor(),
        Normalize()
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor, original_size

def preprocess_video_frame(frame, target_size=MODEL_CONFIG['input_size']):
    """Preprocess video frame for inference"""
    # Convert BGR to RGB (OpenCV uses BGR)
    frame_rgb = frame[:, :, ::-1]
    image = Image.fromarray(frame_rgb)
    
    # Apply transforms
    transform = Compose([
        Resize(target_size),
        ToTensor(),
        Normalize()
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor
