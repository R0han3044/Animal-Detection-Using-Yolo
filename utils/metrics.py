"""
Evaluation metrics for YOLO model
"""
import torch
import numpy as np
from collections import defaultdict
from config import MODEL_CONFIG, CLASS_NAMES

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IoU value
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.4):
    """
    Apply Non-Maximum Suppression to predictions
    Args:
        predictions: List of predictions [x1, y1, x2, y2, confidence, class_id]
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
    Returns:
        Filtered predictions after NMS
    """
    if len(predictions) == 0:
        return []
    
    # Filter by confidence threshold
    predictions = [pred for pred in predictions if pred[4] >= conf_threshold]
    
    if len(predictions) == 0:
        return []
    
    # Sort by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    # Group by class
    class_predictions = defaultdict(list)
    for pred in predictions:
        class_predictions[pred[5]].append(pred)
    
    # Apply NMS for each class
    final_predictions = []
    for class_id, class_preds in class_predictions.items():
        while class_preds:
            # Take the prediction with highest confidence
            best_pred = class_preds.pop(0)
            final_predictions.append(best_pred)
            
            # Remove predictions with high IoU overlap
            class_preds = [
                pred for pred in class_preds
                if calculate_iou(best_pred[:4], pred[:4]) < nms_threshold
            ]
    
    return final_predictions

def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall for a single IoU threshold
    Args:
        predictions: List of predictions [x1, y1, x2, y2, confidence, class_id]
        ground_truths: List of ground truths [x1, y1, x2, y2, class_id]
        iou_threshold: IoU threshold for considering a prediction as correct
    Returns:
        precision, recall, true_positives, false_positives, false_negatives
    """
    if len(predictions) == 0 and len(ground_truths) == 0:
        return 1.0, 1.0, 0, 0, 0
    elif len(predictions) == 0:
        return 0.0, 0.0, 0, 0, len(ground_truths)
    elif len(ground_truths) == 0:
        return 0.0, 0.0, 0, len(predictions), 0
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)
    true_positives = 0
    false_positives = 0
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    for pred in predictions:
        pred_box = pred[:4]
        pred_class = pred[5]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx] or gt[4] != pred_class:
                continue
            
            gt_box = gt[:4]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if prediction is correct
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = sum(1 for matched in gt_matched if not matched)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall, true_positives, false_positives, false_negatives

def calculate_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) for a single class
    Args:
        predictions: List of predictions [x1, y1, x2, y2, confidence, class_id]
        ground_truths: List of ground truths [x1, y1, x2, y2, class_id]
        iou_threshold: IoU threshold
    Returns:
        Average Precision value
    """
    if len(ground_truths) == 0:
        return 0.0
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)
    
    precisions = []
    recalls = []
    
    true_positives = 0
    false_positives = 0
    
    for pred in predictions:
        pred_box = pred[:4]
        pred_class = pred[5]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx] or gt[4] != pred_class:
                continue
            
            gt_box = gt[:4]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Update counts
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            true_positives += 1
        else:
            false_positives += 1
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(ground_truths)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if len([r for r in recalls if r >= t]) == 0:
            p = 0
        else:
            p = max([precisions[i] for i, r in enumerate(recalls) if r >= t])
        ap += p / 11
    
    return ap

def calculate_map(all_predictions, all_ground_truths, iou_thresholds=[0.5]):
    """
    Calculate mean Average Precision (mAP) across all classes
    Args:
        all_predictions: Dictionary of predictions per class
        all_ground_truths: Dictionary of ground truths per class
        iou_thresholds: List of IoU thresholds
    Returns:
        mAP value and per-class AP values
    """
    class_aps = {}
    
    for iou_threshold in iou_thresholds:
        aps = []
        
        for class_idx in range(MODEL_CONFIG['num_classes']):
            class_name = CLASS_NAMES[class_idx]
            
            # Get predictions and ground truths for this class
            class_predictions = all_predictions.get(class_idx, [])
            class_ground_truths = all_ground_truths.get(class_idx, [])
            
            # Calculate AP for this class
            ap = calculate_average_precision(
                class_predictions, class_ground_truths, iou_threshold
            )
            aps.append(ap)
            class_aps[f'{class_name}_AP@{iou_threshold}'] = ap
        
        # Calculate mAP
        map_value = np.mean(aps) if aps else 0.0
        class_aps[f'mAP@{iou_threshold}'] = map_value
    
    return class_aps

class MetricsCalculator:
    """Class to calculate and track various metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, pred_boxes, true_boxes, loss=None):
        """
        Update metrics with new predictions and ground truths
        Args:
            pred_boxes: Predicted bounding boxes
            true_boxes: Ground truth bounding boxes
            loss: Loss value for this batch
        """
        # Group by class
        for pred in pred_boxes:
            class_id = int(pred[5])
            self.predictions[class_id].append(pred)
        
        for gt in true_boxes:
            class_id = int(gt[4])
            self.ground_truths[class_id].append(gt)
        
        # Update loss
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
    
    def compute_metrics(self, iou_thresholds=[0.5, 0.75], conf_threshold=0.5, nms_threshold=0.4):
        """
        Compute all metrics
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
            conf_threshold: Confidence threshold for predictions
            nms_threshold: NMS threshold
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Apply NMS to predictions
        filtered_predictions = defaultdict(list)
        for class_id, class_preds in self.predictions.items():
            filtered_preds = non_max_suppression(
                class_preds, conf_threshold, nms_threshold
            )
            filtered_predictions[class_id] = filtered_preds
        
        # Calculate mAP for different IoU thresholds
        map_results = calculate_map(
            filtered_predictions, self.ground_truths, iou_thresholds
        )
        metrics.update(map_results)
        
        # Calculate overall precision and recall
        all_preds = []
        all_gts = []
        for class_preds in filtered_predictions.values():
            all_preds.extend(class_preds)
        for class_gts in self.ground_truths.values():
            all_gts.extend(class_gts)
        
        if all_preds or all_gts:
            precision, recall, tp, fp, fn = calculate_precision_recall(
                all_preds, all_gts, iou_threshold=0.5
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            
            # F1 score
            if precision + recall > 0:
                metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
            else:
                metrics['f1_score'] = 0.0
        
        # Average loss
        if self.num_batches > 0:
            metrics['avg_loss'] = self.total_loss / self.num_batches
        
        return metrics
