"""
Streamlit web application for YOLO animal detection
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from io import BytesIO
import json

from models.yolo import create_model
from inference.predictor import YOLOPredictor
from training.trainer import train_model
from utils.visualization import draw_bounding_boxes
from config import MODEL_CONFIG, CLASS_NAMES, TRAINING_CONFIG

# Page configuration
st.set_page_config(
    page_title="YOLO Animal Detection",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_predictor():
    """Load YOLO predictor (cached)"""
    try:
        # Try to load best model if available
        model_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(model_path):
            predictor = YOLOPredictor(model_path)
            return predictor, True
        else:
            # Create predictor with random weights
            predictor = YOLOPredictor()
            return predictor, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def main():
    """Main Streamlit application"""
    st.title("🐾 YOLO Animal Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Load predictor
    predictor, model_loaded = load_predictor()
    
    if predictor is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    # Model status
    if model_loaded:
        st.sidebar.success("✅ Trained model loaded")
    else:
        st.sidebar.warning("⚠️ Using untrained model (random weights)")
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # NMS threshold
    nms_threshold = st.sidebar.slider(
        "NMS Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    # Update predictor thresholds
    predictor.conf_threshold = conf_threshold
    predictor.nms_threshold = nms_threshold
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "🏋️ Model Training", "📊 Model Info"])
    
    with tab1:
        image_detection_tab(predictor)
    
    with tab2:
        video_detection_tab(predictor)
    
    with tab3:
        training_tab()
    
    with tab4:
        model_info_tab()

def image_detection_tab(predictor):
    """Image detection tab"""
    st.header("Image Detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Process image
            with st.spinner("Running detection..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    
                    try:
                        # Run prediction
                        predictions, _, inference_time = predictor.predict_image(tmp_file.name)
                        
                        # Draw bounding boxes
                        if predictions:
                            result_image = draw_bounding_boxes(image, predictions)
                            st.image(result_image, use_column_width=True)
                        else:
                            st.image(image, use_column_width=True)
                            st.info("No animals detected")
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
                        os.unlink(tmp_file.name)
                        return
        
        # Detection statistics
        st.subheader("Detection Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Objects Detected", len(predictions))
        
        with col2:
            st.metric("Inference Time", f"{inference_time:.3f}s")
        
        with col3:
            fps = 1.0 / inference_time if inference_time > 0 else 0
            st.metric("FPS", f"{fps:.1f}")
        
        # Detection details
        if predictions:
            st.subheader("Detection Details")
            detection_data = []
            
            for i, pred in enumerate(predictions):
                x1, y1, x2, y2, conf, class_id = pred
                detection_data.append({
                    "Object": i + 1,
                    "Class": CLASS_NAMES[int(class_id)],
                    "Confidence": f"{conf:.3f}",
                    "Bounding Box": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                })
            
            st.table(detection_data)

def video_detection_tab(predictor):
    """Video detection tab"""
    st.header("Video Detection")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Video info
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Display video info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", fps)
        with col3:
            st.metric("Frames", frame_count)
        with col4:
            st.metric("Resolution", f"{width}x{height}")
        
        # Process video button
        if st.button("Process Video", type="primary"):
            with st.spinner("Processing video... This may take a while."):
                try:
                    # Create output path
                    output_path = tempfile.mktemp(suffix='_output.mp4')
                    
                    # Process video
                    start_time = time.time()
                    frame_predictions = predictor.predict_video(
                        video_path, 
                        output_path=output_path,
                        display=False
                    )
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.success(f"Video processed in {processing_time:.1f} seconds!")
                    
                    # Show processed video
                    if os.path.exists(output_path):
                        st.subheader("Processed Video")
                        # Provide download link instead of direct video display
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="processed_video.mp4",
                            mime="video/mp4",
                            type="primary"
                        )
                        
                        st.info("Video processing completed successfully! Click the download button above to get your processed video.")
                        
                        # Clean up
                        os.unlink(output_path)
                    
                    # Statistics
                    total_detections = sum(len(preds) for preds in frame_predictions)
                    avg_detections = total_detections / len(frame_predictions) if frame_predictions else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", total_detections)
                    with col2:
                        st.metric("Avg per Frame", f"{avg_detections:.1f}")
                    with col3:
                        st.metric("Processing FPS", f"{frame_count/processing_time:.1f}")
                    
                except Exception as e:
                    st.error(f"Error processing video: {e}")
        
        # Clean up
        os.unlink(video_path)

def training_tab():
    """Model training tab"""
    st.header("Model Training")
    
    st.info("🚧 Training functionality requires a proper dataset. This demo shows the training interface.")
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Parameters")
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=50)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    with col2:
        st.subheader("Dataset Info")
        st.info("📁 Place your dataset in the 'data' directory")
        st.info("📝 Annotations should be in JSON format")
        st.info("🏷️ Supported classes: " + ", ".join(CLASS_NAMES))
    
    # Training button
    if st.button("Start Training", type="primary"):
        # Check for dataset
        data_dir = "data"
        if not os.path.exists(data_dir):
            st.error("Dataset directory not found. Please add your dataset to the 'data' folder.")
            return
        
        # Training configuration
        config_overrides = {
            'model': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        # Progress bars
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing training...")
            
            # Start training (this would be the actual training in a real scenario)
            st.warning("⚠️ Training requires a proper dataset and significant computational resources.")
            st.info("💡 In a production environment, training would run here with progress updates.")
            
            # Simulate training progress
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training... Epoch {i//10 + 1}/10")
                time.sleep(0.01)  # Simulate work
            
            st.success("🎉 Training simulation completed!")
            
        except Exception as e:
            st.error(f"Training error: {e}")

def model_info_tab():
    """Model information tab"""
    st.header("Model Information")
    
    # Model architecture
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.code(f"""
Architecture: Custom YOLO
Input Size: {MODEL_CONFIG['input_size']}x{MODEL_CONFIG['input_size']}
Grid Size: {MODEL_CONFIG['grid_size']}x{MODEL_CONFIG['grid_size']}
Number of Classes: {MODEL_CONFIG['num_classes']}
Anchor Boxes: {MODEL_CONFIG['num_boxes']}
        """)
    
    with col2:
        st.subheader("Supported Classes")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i}: {class_name}")
    
    # Model statistics
    st.subheader("Model Statistics")
    
    # Create model to get parameter count
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{total_params:,}")
    with col2:
        st.metric("Trainable Parameters", f"{trainable_params:,}")
    with col3:
        st.metric("Model Size (MB)", f"{total_params * 4 / 1024 / 1024:.1f}")
    
    # Configuration
    st.subheader("Current Configuration")
    config_dict = {
        "Model Config": MODEL_CONFIG,
        "Training Config": TRAINING_CONFIG
    }
    st.json(config_dict)
    
    # Device info
    st.subheader("System Information")
    device = torch.device(MODEL_CONFIG['device'])
    st.write(f"**Device**: {device}")
    
    if torch.cuda.is_available():
        st.write(f"**CUDA Available**: Yes")
        st.write(f"**GPU**: {torch.cuda.get_device_name(0)}")
        st.write(f"**CUDA Version**: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
    else:
        st.write(f"**CUDA Available**: No (using CPU)")

if __name__ == "__main__":
    main()
