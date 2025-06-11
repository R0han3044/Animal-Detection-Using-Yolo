# 🐾 Animal Detection Using YOLO

A computer vision-based animal detection system using YOLOv5, ideal for farm protection and wildlife monitoring. Built with Python, Streamlit, and OpenCV.

## 🚀 Features

- 🔍 Real-time animal detection using YOLOv5
- 🎥 Live video feed integration
- 💡 Automatic alert system (email, Telegram, or console)
- 💾 Easy model management via download links
- 🌐 Web interface using Streamlit

## 🛠️ Installation

```bash
git clone https://github.com/R0han3044/Animal-Detection-Using-Yolo.git
cd Animal-Detection-Using-Yolo
pip install -r requirements.txt

Projec Structure 
Animal-Detection-Using-Yolo/
│
├── app.py                  # Streamlit web app
├── detect.py               # YOLO detection script
├── checkpoints/
│   └── best_model.pth      # (download separately)
├── utils/                  # Helper functions
├── requirements.txt
└── README.md

Running the App
streamlit run app.py

Best model path 
/checkpoints/best_model.pth