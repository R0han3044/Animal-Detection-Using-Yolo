"""
Custom YOLO model implementation from scratch using PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and leaky ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class YOLO(nn.Module):
    """Custom YOLO model for animal detection"""
    def __init__(self, num_classes=MODEL_CONFIG['num_classes'], num_boxes=MODEL_CONFIG['num_boxes']):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = MODEL_CONFIG['grid_size']
        
        # Backbone network (simplified version of Darknet)
        self.backbone = nn.Sequential(
            # Layer 0-2
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            
            # Layer 3-8
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            
            # Layer 9-16
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            
            # Layer 17-26
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            
            # Layer 27-29
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1)
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, num_boxes * (5 + num_classes), 1, 1, 0)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Detection head
        output = self.detection_head(features)
        
        # Reshape output: [batch_size, grid_size, grid_size, num_boxes, (5 + num_classes)]
        batch_size = x.size(0)
        output = output.view(batch_size, self.num_boxes, 5 + self.num_classes, self.grid_size, self.grid_size)
        output = output.permute(0, 3, 4, 1, 2).contiguous()
        
        return output

class YOLOLoss(nn.Module):
    """YOLO loss function"""
    def __init__(self, num_classes=MODEL_CONFIG['num_classes'], lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        predictions: [batch_size, grid_size, grid_size, num_boxes, (5 + num_classes)]
        targets: [batch_size, grid_size, grid_size, num_boxes, (5 + num_classes)]
        """
        batch_size, grid_size, _, num_boxes, _ = predictions.shape
        
        # Extract predictions
        pred_xy = torch.sigmoid(predictions[..., :2])  # x, y coordinates
        pred_wh = predictions[..., 2:4]  # width, height
        pred_conf = torch.sigmoid(predictions[..., 4:5])  # confidence
        pred_cls = predictions[..., 5:]  # class probabilities
        
        # Extract targets
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4:5]
        target_cls = targets[..., 5:]
        
        # Object mask
        obj_mask = target_conf > 0
        noobj_mask = target_conf == 0
        
        # Coordinate loss (only for cells with objects)
        coord_loss = self.lambda_coord * self.mse_loss(
            pred_xy[obj_mask], target_xy[obj_mask]
        ) + self.lambda_coord * self.mse_loss(
            torch.sqrt(torch.abs(pred_wh[obj_mask])), 
            torch.sqrt(target_wh[obj_mask])
        )
        
        # Confidence loss
        conf_loss_obj = self.mse_loss(pred_conf[obj_mask], target_conf[obj_mask])
        conf_loss_noobj = self.lambda_noobj * self.mse_loss(
            pred_conf[noobj_mask], target_conf[noobj_mask]
        )
        
        # Class loss (only for cells with objects)
        class_loss = self.mse_loss(pred_cls[obj_mask], target_cls[obj_mask])
        
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        return total_loss / batch_size

def create_model():
    """Create and return YOLO model"""
    model = YOLO(
        num_classes=MODEL_CONFIG['num_classes'],
        num_boxes=MODEL_CONFIG['num_boxes']
    )
    return model

def create_loss():
    """Create and return YOLO loss function"""
    return YOLOLoss(num_classes=MODEL_CONFIG['num_classes'])
