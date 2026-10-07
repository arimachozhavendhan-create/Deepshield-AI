import os

# Base Directory of the Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & Output Directories
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
HISTORY_FILE = os.path.join(BASE_DIR, 'results', 'history.json')

# Create directories if they do not exist
for folder in [UPLOADS_DIR, RESULTS_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Custom Trained Model Path
MODEL_PATH = os.path.join(MODELS_DIR, 'deep_shield_cnn.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Multi-Signal Fusion Weights (Must total 1.0 for video)
CNN_WEIGHT = 0.50
FFT_WEIGHT = 0.25
BLINK_WEIGHT = 0.25

# Forensic Localization Threshold
SUSPICION_THRESHOLD = 0.70  # Scores >= 0.70 flagged as suspicious

# Video Analysis Settings
FRAME_SAMPLE_RATE = 10     # Analyze every Nth frame for efficiency
MAX_VIDEO_DURATION = 300   # Max video duration in seconds (5 minutes)
MAX_UPLOAD_SIZE_MB = 100   # 100MB upload limit

# Image Target Size for Model Inference
IMAGE_SIZE = (224, 224)

# Allowed Media Extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# MediaPipe Thresholds
MIN_FACE_DETECTION_CONFIDENCE = 0.5
EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold for blink detection
