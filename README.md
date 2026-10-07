# DEEPSHIELD: Adaptive Explainable Deepfake Detection & Forensic Localization System

> *"Don't just detect the fake. Find where it happened. Explain why."*

DeepShield is an advanced AI-assisted forensic web application designed for deepfake detection, temporal localization, spatial feature inspection, and explainable forensic report generation.

---

## 🌟 Key Features

1. **Multi-Signal Evidence Fusion Engine**:
   - **EfficientNetB0 Visual Analysis**: Deep Convolutional Neural Network spatial anomaly scoring on facial crops.
   - **FFT Frequency Analysis**: 2D Discrete Fast Fourier Transform to analyze high-frequency spatial noise signatures and synthetic blending artifacts.
   - **Behavioral Eye Blinking Analysis**: MediaPipe FaceMesh Eye Aspect Ratio (EAR) tracking to identify unnatural blink frequencies, missing blinks, or static eye closures.
   - **Adaptive Re-normalization**: Intelligently re-normalizes signal weights for static images when temporal eye blinking signals are absent.
2. **Forensic Localization**:
   - **Peak Suspicious Frame Isolation**: Extracted frame with highest suspicion score annotated with a face bounding box (`"SUSPICIOUS FACE REGION"`).
   - **Suspicious Time-Segment Detection**: Groups consecutive anomalous frames into exact timestamp intervals (e.g. `00:04.73 → 00:05.27`).
   - **Interactive Chart.js Timeline**: Real-time visual suspicion curve across video playback with hover tooltips and threshold indicator line ($0.70$).
   - **Top Suspicious Frame Gallery**: Inspection gallery for top anomalous frames.
3. **Explainable AI & Forensic Reporting**:
   - Dynamic evidence breakdown based strictly on computed sub-signal values.
   - Automated PDF report generation powered by **ReportLab**.
   - Persistent JSON analysis history tracking.

---

## 🏗️ System Architecture

```
INPUT MEDIA (Image / Video)
        │
        ▼
Media Validation & Meta Extraction
        │
        ▼
Face Detection (MediaPipe / Haar Cascade) & Crop
        │
┌───────┴─────────────────────────────────────────────┐
│ MULTI-SIGNAL ANALYSIS PIPELINE                       │
│                                                     │
│ [1] EfficientNetB0  →  Visual Evidence Score        │
│ [2] 2D FFT Spectrum →  Frequency Evidence Score     │
│ [3] MediaPipe EAR   →  Behavioral Blinking Score    │
└───────┬─────────────────────────────────────────────┘
        │
        ▼
Weighted Signal Fusion & Weight Re-normalization
        │
        ▼
Frame-Level Suspicion Scoring & Thresholding (0.70)
        │
        ▼
Frame-Level Localization & Time-Segment Aggregation
        │
        ▼
Peak Frame Extraction + Bounding Box Annotation
        │
        ▼
Interactive Dashboard (Chart.js Timeline & Breakdown)
        │
        ▼
Downloadable PDF Forensic Report
```

---

## 🛠️ Technology Stack

- **Backend**: Python 3.11 / 3.12, Flask, OpenCV (`opencv-python`), TensorFlow / Keras (EfficientNetB0), MediaPipe, NumPy, ReportLab
- **Frontend**: HTML5, Modern CSS3 (Glassmorphism, Dark UI, CSS Variables), JavaScript (ES6+), Chart.js
- **Storage**: Local JSON analysis history log and disk storage for uploaded media and generated reports.

---

## 🚀 Quick Start & Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/DeepShield.git
cd DeepShield
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Custom Model Placement
Place your custom trained model weights inside the `models/` directory:
```
DeepShield/models/deep_shield_cnn.keras
```
*Note: If no custom model weights are detected, DeepShield automatically operates in a clearly labeled **"DEMO / MODEL NOT LOADED"** mode using EfficientNetB0 ImageNet feature heuristics.*

### 5. Run the Server
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📡 REST API Documentation

### `POST /api/analyze/image`
Uploads an image file (`.jpg`, `.png`, `.webp`) for multi-signal analysis.
- **Request Body**: `multipart/form-data` with key `file`
- **Response**: JSON result containing `verdict`, `fusion_score`, `cnn_score`, `fft_score`, `explanation`, and `result_id`.

### `POST /api/analyze/video`
Uploads a video file (`.mp4`, `.avi`, `.mov`) for frame-by-frame temporal analysis.
- **Request Body**: `multipart/form-data` with key `file`
- **Response**: JSON result containing `verdict`, `fusion_score`, `peak_frame`, `peak_timestamp`, `suspicious_segments`, `timeline_data`, and `top_gallery`.

### `GET /api/result/<result_id>`
Retrieves saved JSON result for a given analysis ID.

### `GET /api/report/<result_id>`
Downloads the generated PDF forensic report.

### `GET /api/history`
Returns list of recent local analysis records.

### `GET /api/health`
System status check.

---

## ⚠️ Disclaimers & System Limitations

- **Forensic Disclaimer**: DeepShield output represents an AI-assisted detection score derived from multi-signal feature heuristics. It does NOT provide absolute legal certainty.
- **Low Resolution / High Compression**: Heavily compressed video files or low-resolution facial crops may affect high-frequency FFT metrics and facial landmark detection.
- **Primary Face Scope**: The current pipeline isolates and analyzes the primary/largest detected face per image or video frame.

---

## 🔬 Future Scope

- Multi-face concurrent analysis across crowd scenes.
- Learned attention-based signal fusion networks.
- Vision Transformer (ViT) spatial backbones.
- Deepfake voice and audio spectrogram analysis integration.
