"""
Inference engine for YOLO model
"""
import torch
import cv2
import numpy as np
from PIL import Image
import time
import os
from typing import List, Tuple, Union

from models.yolo import create_model
from utils.transforms import preprocess_image, preprocess_video_frame
from utils.metrics import non_max_suppression
from utils.visualization import draw_bounding_boxes_cv2
from config import MODEL_CONFIG, CLASS_NAMES, CLASS_COLORS

class YOLOPredictor:
    """YOLO model predictor for inference"""
    
    def __init__(self, model_path=None, conf_threshold=0.5, nms_threshold=0.4, device=None):
        """
        Initialize YOLO predictor
        Args:
            model_path: Path to trained model checkpoint
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            device: Device to run inference on
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device or torch.device(MODEL_CONFIG['device'])
        
        # Load model
        self.model = create_model()
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No model checkpoint provided. Using randomly initialized weights.")
        
        self.model.eval()
        
        # Model parameters
        self.input_size = MODEL_CONFIG['input_size']
        self.grid_size = MODEL_CONFIG['grid_size']
        self.num_classes = MODEL_CONFIG['num_classes']
        self.num_boxes = MODEL_CONFIG['num_boxes']
    
    def load_model(self, model_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _decode_predictions(self, predictions, original_size):
        """
        Decode YOLO model predictions to bounding boxes
        """
        predictions = predictions.squeeze(0)
        boxes = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for b in range(self.num_boxes):
                    # Extract prediction components
                    pred_slice = predictions[i, j, b, :]
                    
                    x_offset = torch.sigmoid(pred_slice[0])
                    y_offset = torch.sigmoid(pred_slice[1])
                    width = pred_slice[2]
                    height = pred_slice[3]
                    confidence = torch.sigmoid(pred_slice[4])
                    class_probs = torch.sigmoid(pred_slice[5:])
                    
                    # Apply confidence threshold
                    if confidence.item() < self.conf_threshold:
                        continue
                    
                    # Calculate absolute coordinates
                    x_center = (j + x_offset.item()) / self.grid_size * original_size[0]
                    y_center = (i + y_offset.item()) / self.grid_size * original_size[1]
                    
                    # Calculate box dimensions
                    box_width = torch.exp(width).item() * original_size[0] / self.grid_size
                    box_height = torch.exp(height).item() * original_size[1] / self.grid_size
                    
                    # Bounding box coordinates
                    x1 = max(0, x_center - box_width / 2)
                    y1 = max(0, y_center - box_height / 2)
                    x2 = min(original_size[0], x_center + box_width / 2)
                    y2 = min(original_size[1], y_center + box_height / 2)
                    
                    # Validate box dimensions
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Get class with highest probability
                    class_id = torch.argmax(class_probs).item()
                    class_confidence = class_probs[class_id].item()
                    
                    # Final confidence score
                    final_confidence = confidence.item() * class_confidence
                    
                    if final_confidence >= self.conf_threshold:
                        boxes.append([
                            x1, y1, x2, y2,
                            final_confidence, class_id
                        ])
        
        return boxes
    
    def predict_image(self, image_path: str) -> Tuple[List, Image.Image, float]:
        """
        Predict on single image
        Args:
            image_path: Path to input image
        Returns:
            Tuple of (predictions, original_image, inference_time)
        """
        start_time = time.time()
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size
        
        # Preprocess for model
        input_tensor, _ = preprocess_image(image_path, self.input_size)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Decode predictions
        boxes = self._decode_predictions(predictions, original_size)
        
        # Apply Non-Maximum Suppression
        filtered_boxes = non_max_suppression(
            boxes, self.conf_threshold, self.nms_threshold
        )
        
        inference_time = time.time() - start_time
        
        return filtered_boxes, original_image, inference_time
    
    def predict_video(self, video_path: str, output_path: str = "", display: bool = False):
        """
        Predict on video
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            display: Whether to display video during processing
        Returns:
            List of predictions for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        out = None
        if output_path and output_path != "":
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_predictions = []
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            input_tensor = preprocess_video_frame(frame, self.input_size)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Decode predictions
            boxes = self._decode_predictions(predictions, (width, height))
            
            # Apply NMS
            filtered_boxes = non_max_suppression(
                boxes, self.conf_threshold, self.nms_threshold
            )
            
            frame_predictions.append(filtered_boxes)
            
            # Draw bounding boxes on frame
            if output_path or display:
                frame_with_boxes = draw_bounding_boxes_cv2(
                    frame, filtered_boxes, CLASS_NAMES, 
                    [(color[2], color[1], color[0]) for color in CLASS_COLORS]
                )
                
                # Add frame info
                cv2.putText(
                    frame_with_boxes,
                    f"Frame: {frame_count}/{total_frames} | Objects: {len(filtered_boxes)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                if output_path and out is not None:
                    out.write(frame_with_boxes)
                
                if display:
                    cv2.imshow('YOLO Detection', frame_with_boxes)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"Video processing completed. Processed {frame_count} frames.")
        
        return frame_predictions
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[List, Image.Image, float]]:
        """
        Predict on batch of images
        Args:
            image_paths: List of image paths
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(([], None, 0.0))
        
        return results
    
    def benchmark_inference(self, image_path: str, num_runs: int = 100):
        """
        Benchmark inference speed
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
        Returns:
            Average inference time and FPS
        """
        # Warmup
        for _ in range(10):
            self.predict_image(image_path)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            _, _, inference_time = self.predict_image(image_path)
            times.append(inference_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"Benchmark Results ({num_runs} runs):")
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Average FPS: {fps:.2f}")
        print(f"Min time: {min(times):.4f} seconds")
        print(f"Max time: {max(times):.4f} seconds")
        
        return avg_time, fps

def create_predictor(model_path=None, conf_threshold=0.5, nms_threshold=0.4):
    """
    Create YOLO predictor instance
    Args:
        model_path: Path to trained model
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
    Returns:
        YOLOPredictor instance
    """
    return YOLOPredictor(model_path, conf_threshold, nms_threshold)

if __name__ == "__main__":
    # Example usage
    predictor = create_predictor()
    
    # Test single image prediction
    if os.path.exists("test_image.jpg"):
        predictions, image, time_taken = predictor.predict_image("test_image.jpg")
        print(f"Found {len(predictions)} objects in {time_taken:.3f} seconds")
