"""
Loss functions for YOLO training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class YOLOLoss(nn.Module):
    """
    YOLO loss function implementation
    Combines coordinate loss, confidence loss, and classification loss
    """
    def __init__(self, 
                 num_classes=MODEL_CONFIG['num_classes'],
                 lambda_coord=5.0,
                 lambda_noobj=0.5,
                 lambda_obj=1.0,
                 lambda_class=1.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        Args:
            predictions: [batch_size, grid_size, grid_size, num_boxes, (5 + num_classes)]
            targets: [batch_size, grid_size, grid_size, num_boxes, (5 + num_classes)]
        Returns:
            Dictionary containing individual loss components and total loss
        """
        batch_size, grid_size, _, num_boxes, _ = predictions.shape
        device = predictions.device
        
        # Extract prediction components
        pred_xy = torch.sigmoid(predictions[..., :2])  # x, y coordinates (0-1)
        pred_wh = predictions[..., 2:4]  # width, height (log space)
        pred_conf = torch.sigmoid(predictions[..., 4])  # confidence (0-1)
        pred_cls = torch.sigmoid(predictions[..., 5:])  # class probabilities (0-1)
        
        # Extract target components
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]
        
        # Create masks
        obj_mask = target_conf > 0  # Cells with objects
        noobj_mask = target_conf == 0  # Cells without objects
        
        # 1. Coordinate Loss (only for cells with objects)
        if obj_mask.sum() > 0:
            # Position loss (x, y)
            xy_loss = self.mse_loss(pred_xy[obj_mask], target_xy[obj_mask]).sum()
            
            # Size loss (w, h) - using square root for better gradient flow
            pred_wh_sqrt = torch.sign(pred_wh[obj_mask]) * torch.sqrt(torch.abs(pred_wh[obj_mask]) + 1e-6)
            target_wh_sqrt = torch.sign(target_wh[obj_mask]) * torch.sqrt(torch.abs(target_wh[obj_mask]) + 1e-6)
            wh_loss = self.mse_loss(pred_wh_sqrt, target_wh_sqrt).sum()
            
            coord_loss = self.lambda_coord * (xy_loss + wh_loss)
        else:
            coord_loss = torch.tensor(0.0, device=device)
        
        # 2. Confidence Loss
        # Object confidence loss
        if obj_mask.sum() > 0:
            conf_obj_loss = self.lambda_obj * self.mse_loss(
                pred_conf[obj_mask], target_conf[obj_mask]
            ).sum()
        else:
            conf_obj_loss = torch.tensor(0.0, device=device)
        
        # No-object confidence loss
        if noobj_mask.sum() > 0:
            conf_noobj_loss = self.lambda_noobj * self.mse_loss(
                pred_conf[noobj_mask], target_conf[noobj_mask]
            ).sum()
        else:
            conf_noobj_loss = torch.tensor(0.0, device=device)
        
        # 3. Classification Loss (only for cells with objects)
        if obj_mask.sum() > 0:
            class_loss = self.lambda_class * self.mse_loss(
                pred_cls[obj_mask], target_cls[obj_mask]
            ).sum()
        else:
            class_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = coord_loss + conf_obj_loss + conf_noobj_loss + class_loss
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        coord_loss = coord_loss / batch_size
        conf_obj_loss = conf_obj_loss / batch_size
        conf_noobj_loss = conf_noobj_loss / batch_size
        class_loss = class_loss / batch_size
        
        # Return loss components for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'conf_obj_loss': conf_obj_loss,
            'conf_noobj_loss': conf_noobj_loss,
            'class_loss': class_loss,
            'num_objects': obj_mask.sum().float()
        }
        
        return loss_dict

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class IoULoss(nn.Module):
    """
    IoU-based loss for better bounding box regression
    """
    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        Calculate IoU loss
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                   (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                     (target_boxes[:, 3] - target_boxes[:, 1])
        
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        # IoU loss = 1 - IoU
        loss = 1 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def create_loss_function():
    """Create and return the main loss function"""
    return YOLOLoss(
        num_classes=MODEL_CONFIG['num_classes'],
        lambda_coord=5.0,
        lambda_noobj=0.5,
        lambda_obj=1.0,
        lambda_class=1.0
    )
