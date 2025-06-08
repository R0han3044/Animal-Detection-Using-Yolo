"""
Visualization utilities for YOLO model
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torch
from config import CLASS_NAMES, CLASS_COLORS

def draw_bounding_boxes(image, boxes, class_names=None, colors=None, thickness=2):
    """
    Draw bounding boxes on image
    Args:
        image: PIL Image or numpy array
        boxes: List of boxes [x1, y1, x2, y2, confidence, class_id]
        class_names: List of class names
        colors: List of colors for each class
        thickness: Line thickness
    Returns:
        Image with bounding boxes drawn
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if colors is None:
        colors = CLASS_COLORS
    
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Create a copy for drawing
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        confidence = box[4] if len(box) > 4 else 1.0
        class_id = int(box[5]) if len(box) > 5 else 0
        
        # Get class name and color
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label
        label = f'{class_name}: {confidence:.2f}'
        
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    return image_copy

def draw_bounding_boxes_cv2(image, boxes, class_names=None, colors=None, thickness=2):
    """
    Draw bounding boxes using OpenCV
    Args:
        image: numpy array (BGR format)
        boxes: List of boxes [x1, y1, x2, y2, confidence, class_id]
        class_names: List of class names
        colors: List of colors for each class (BGR format)
        thickness: Line thickness
    Returns:
        Image with bounding boxes drawn
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if colors is None:
        # Convert RGB to BGR for OpenCV
        colors = [(color[2], color[1], color[0]) for color in CLASS_COLORS]
    
    image_copy = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = box[4] if len(box) > 4 else 1.0
        class_id = int(box[5]) if len(box) > 5 else 0
        
        # Get class name and color
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f'{class_name}: {confidence:.2f}'
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            image_copy,
            (x1, y1 - text_height - baseline - 4),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_copy,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image_copy

def plot_training_history(history, save_path=None):
    """
    Plot training history
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot mAP
    if 'train_map' in history and 'val_map' in history:
        axes[0, 1].plot(history['train_map'], label='Train mAP')
        axes[0, 1].plot(history['val_map'], label='Validation mAP')
        axes[0, 1].set_title('Training and Validation mAP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot precision
    if 'train_precision' in history and 'val_precision' in history:
        axes[1, 0].plot(history['train_precision'], label='Train Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Training and Validation Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall
    if 'train_recall' in history and 'val_recall' in history:
        axes[1, 1].plot(history['train_recall'], label='Train Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Training and Validation Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_predictions(image, predictions, ground_truths=None, conf_threshold=0.5):
    """
    Visualize predictions and optionally ground truths
    Args:
        image: Input image
        predictions: Model predictions
        ground_truths: Ground truth boxes (optional)
        conf_threshold: Confidence threshold for displaying predictions
    Returns:
        Visualization image
    """
    # Filter predictions by confidence
    filtered_predictions = [
        pred for pred in predictions if pred[4] >= conf_threshold
    ]
    
    # Create visualization
    if ground_truths is not None:
        # Create side-by-side comparison
        image_pred = draw_bounding_boxes(image, filtered_predictions)
        image_gt = draw_bounding_boxes(image, ground_truths)
        
        # Convert to numpy arrays
        img_pred_array = np.array(image_pred)
        img_gt_array = np.array(image_gt)
        
        # Concatenate horizontally
        vis_image = np.concatenate([img_pred_array, img_gt_array], axis=1)
        vis_image = Image.fromarray(vis_image)
    else:
        vis_image = draw_bounding_boxes(image, filtered_predictions)
    
    return vis_image

def create_detection_grid(images, predictions_list, titles=None, max_cols=3):
    """
    Create a grid of detection results
    Args:
        images: List of input images
        predictions_list: List of predictions for each image
        titles: List of titles for each image
        max_cols: Maximum number of columns in the grid
    Returns:
        Grid visualization
    """
    num_images = len(images)
    num_cols = min(num_images, max_cols)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    if num_rows == 1:
        axes = [axes] if num_cols == 1 else axes
    elif num_cols == 1:
        axes = [[ax] for ax in axes]
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        # Draw predictions on image
        vis_image = draw_bounding_boxes(images[i], predictions_list[i])
        
        # Display image
        if num_rows == 1 and num_cols == 1:
            ax = axes
        elif num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row][col]
        
        ax.imshow(vis_image)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    # Hide empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows == 1:
            axes[col].axis('off')
        else:
            axes[row][col].axis('off')
    
    plt.tight_layout()
    return fig

def save_detection_results(image, predictions, save_path, conf_threshold=0.5):
    """
    Save detection results to file
    Args:
        image: Input image
        predictions: Model predictions
        save_path: Path to save the result
        conf_threshold: Confidence threshold
    """
    # Filter predictions
    filtered_predictions = [
        pred for pred in predictions if pred[4] >= conf_threshold
    ]
    
    # Draw bounding boxes
    result_image = draw_bounding_boxes(image, filtered_predictions)
    
    # Save image
    result_image.save(save_path)

def plot_class_distribution(annotations, save_path=None):
    """
    Plot class distribution in dataset
    Args:
        annotations: List of annotations
        save_path: Path to save the plot
    """
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    for annotation in annotations:
        for obj in annotation.get('objects', []):
            class_name = obj.get('class', 'unknown')
            if class_name in class_counts:
                class_counts[class_name] += 1
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=[f'C{i}' for i in range(len(classes))])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()
