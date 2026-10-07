import os
import uuid
import json
import time
from datetime import datetime
import cv2

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

import config
from utils.media_utils import allowed_file
from utils.report_generator import generate_forensic_pdf

from detection.face_detector import FaceDetector
from detection.cnn_detector import CNNDetector
from detection.fft_analyzer import FFTAnalyzer
from detection.fusion_engine import FusionEngine
from detection.video_analyzer import VideoAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Helper to save JSON history
def save_to_history(result_data):
    history = []
    if os.path.exists(config.HISTORY_FILE):
        try:
            with open(config.HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    # Format item for history storage
    history_item = {
        'id': result_data.get('id'),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': result_data.get('filename'),
        'media_type': result_data.get('media_type', 'image'),
        'verdict': result_data.get('verdict'),
        'fusion_score': result_data.get('fusion_score'),
        'peak_frame': result_data.get('peak_frame'),
        'peak_timestamp': result_data.get('peak_timestamp'),
        'is_demo': result_data.get('is_demo', False)
    }

    history.insert(0, history_item)
    # Keep last 50 entries
    history = history[:50]

    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# Page Routes
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/image')
def image_page():
    return render_template('image_result.html')

@app.route('/video')
def video_page():
    return render_template('video_result.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/results/<path:filename>')
def serve_result_file(filename):
    return send_from_directory(config.RESULTS_DIR, filename)

# API Endpoints
@app.route('/api/health', methods=['GET'])
def api_health():
    cnn = CNNDetector()
    return jsonify({
        'status': 'healthy',
        'application': 'DEEPSHIELD',
        'version': '2.0',
        'demo_mode': not cnn.custom_model_loaded,
        'weights_path': config.MODEL_PATH
    })

@app.route('/api/analyze/image', methods=['POST'])
def api_analyze_image():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No media file provided in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Selected file is empty.'}), 400

    if not allowed_file(file.filename, media_type='image'):
        return jsonify({'error': f'Unsupported image format. Allowed formats: {", ".join(config.ALLOWED_IMAGE_EXTENSIONS)}'}), 400

    result_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    upload_path = os.path.join(config.UPLOADS_DIR, f"{result_id}_{filename}")
    file.save(upload_path)

    # Read image using OpenCV
    image_bgr = cv2.imread(upload_path)
    if image_bgr is None:
        return jsonify({'error': 'Corrupted or unreadable image file.'}), 400

    # Instantiate detectors
    face_detector = FaceDetector()
    cnn_detector = CNNDetector()
    fft_analyzer = FFTAnalyzer()
    fusion_engine = FusionEngine()

    try:
        # Face Detection
        face_crop, bbox, annotated_img = face_detector.detect_primary_face(image_bgr, draw_bbox=True)

        if face_crop is None:
            face_detector.close()
            return jsonify({
                'error': 'No detectable primary face was found in the image. Please upload a clearer facial image.'
            }), 400

        # Save annotated primary face crop
        annotated_crop_filename = f"{result_id}_face.jpg"
        annotated_crop_path = os.path.join(config.RESULTS_DIR, annotated_crop_filename)
        cv2.imwrite(annotated_crop_path, annotated_img)

        # CNN Analysis
        cnn_res = cnn_detector.analyze_face(face_crop)

        # FFT Analysis
        spectrum_filename = f"{result_id}_fft.jpg"
        spectrum_path = os.path.join(config.RESULTS_DIR, spectrum_filename)
        fft_res = fft_analyzer.analyze_face(face_crop, save_visualization_path=spectrum_path)

        # Multi-Signal Fusion (Normalized CNN + FFT for static image)
        fuse_res = fusion_engine.fuse(
            cnn_score=cnn_res['cnn_score'],
            fft_score=fft_res['fft_score'],
            blink_score=None
        )

        elapsed = round(time.time() - start_time, 2)

        result_data = {
            'id': result_id,
            'media_type': 'image',
            'filename': filename,
            'has_face': True,
            'verdict': fuse_res['verdict'],
            'fusion_score': fuse_res['fusion_score'],
            'cnn_score': cnn_res['cnn_score'],
            'fft_score': fft_res['fft_score'],
            'blink_score': None,
            'explanation': fuse_res['explanation'],
            'processing_time': elapsed,
            'is_demo': cnn_res['is_demo'],
            'face_crop_filename': annotated_crop_filename,
            'fft_spectrum_filename': spectrum_filename,
            'peak_frame_path': annotated_crop_path
        }

        # Generate PDF Report
        pdf_filename = f"{result_id}_report.pdf"
        pdf_path = os.path.join(config.REPORTS_DIR, pdf_filename)
        generate_forensic_pdf(result_data, pdf_path)
        result_data['pdf_filename'] = pdf_filename

        # Save to history registry
        save_to_history(result_data)
        face_detector.close()

        return jsonify(result_data)

    except Exception as e:
        face_detector.close()
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

@app.route('/api/analyze/video', methods=['POST'])
def api_analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file provided in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Selected video file is empty.'}), 400

    if not allowed_file(file.filename, media_type='video'):
        return jsonify({'error': f'Unsupported video format. Allowed formats: {", ".join(config.ALLOWED_VIDEO_EXTENSIONS)}'}), 400

    result_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    upload_path = os.path.join(config.UPLOADS_DIR, f"{result_id}_{filename}")
    file.save(upload_path)

    analyzer = VideoAnalyzer()
    try:
        result_data = analyzer.process_video(upload_path, result_id)
        
        # Generate PDF Report
        pdf_filename = f"{result_id}_report.pdf"
        pdf_path = os.path.join(config.REPORTS_DIR, pdf_filename)
        generate_forensic_pdf(result_data, pdf_path)
        result_data['pdf_filename'] = pdf_filename

        # Save to history registry
        save_to_history(result_data)
        analyzer.close()

        return jsonify(result_data)

    except Exception as e:
        analyzer.close()
        return jsonify({'error': f'An error occurred during video analysis: {str(e)}'}), 500

@app.route('/api/report/<result_id>', methods=['GET'])
def api_get_report(result_id):
    pdf_filename = f"{result_id}_report.pdf"
    pdf_path = os.path.join(config.REPORTS_DIR, pdf_filename)

    if not os.path.exists(pdf_path):
        return jsonify({'error': 'Forensic PDF report not found.'}), 404

    return send_file(pdf_path, as_attachment=True, download_name=f"DeepShield_Forensic_Report_{result_id}.pdf")

@app.route('/api/history', methods=['GET'])
def api_get_history():
    if os.path.exists(config.HISTORY_FILE):
        try:
            with open(config.HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify(history)
        except Exception:
            return jsonify([])
    return jsonify([])

# Global Error Handler for clean user experience
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File size exceeds the maximum upload limit of {config.MAX_UPLOAD_SIZE_MB}MB.'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal system error occurred during analysis. Please try again with a valid video/image.'}), 500

if __name__ == '__main__':
    print("[DEEPSHIELD] Launching Forensic Web Server on http://127.0.0.1:5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
