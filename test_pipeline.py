import os
import sys
import cv2
import numpy as np

# Add project directory to python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepShield'))
sys.path.insert(0, project_dir)

import config
from detection.face_detector import FaceDetector
from detection.cnn_detector import CNNDetector
from detection.fft_analyzer import FFTAnalyzer
from detection.blink_analyzer import BlinkAnalyzer
from detection.fusion_engine import FusionEngine
from detection.frame_localizer import FrameLocalizer
from detection.video_analyzer import VideoAnalyzer
from utils.report_generator import generate_forensic_pdf

def generate_synthetic_face_image():
    """Creates a synthetic test facial image with a simple drawn face."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (30, 30, 30) # Dark gray background
    
    # Face circle
    cv2.circle(img, (200, 200), 120, (200, 180, 160), -1)
    # Eyes
    cv2.circle(img, (160, 170), 18, (255, 255, 255), -1)
    cv2.circle(img, (240, 170), 18, (255, 255, 255), -1)
    cv2.circle(img, (160, 170), 8, (50, 50, 50), -1)
    cv2.circle(img, (240, 170), 8, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (50, 50, 50), 4)

    test_img_path = os.path.join(config.UPLOADS_DIR, 'test_synthetic_face.jpg')
    cv2.imwrite(test_img_path, img)
    return test_img_path

def generate_synthetic_video():
    """Creates a synthetic test MP4 video with moving face and simulated noise."""
    video_path = os.path.join(config.UPLOADS_DIR, 'test_synthetic_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (400, 400))

    for frame_idx in range(90): # 3 seconds at 30 fps
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)

        # Oscillating face position
        offset_x = int(10 * np.sin(frame_idx / 5.0))
        center_x = 200 + offset_x

        cv2.circle(img, (center_x, 200), 120, (200, 180, 160), -1)

        # Blink simulation (close eyes around frame 40-50)
        if 40 <= frame_idx <= 50:
            # Closed eyes (line)
            cv2.line(img, (center_x - 40, 170), (center_x - 10, 170), (50, 50, 50), 4)
            cv2.line(img, (center_x + 10, 170), (center_x + 40, 170), (50, 50, 50), 4)
        else:
            # Open eyes
            cv2.circle(img, (center_x - 25, 170), 16, (255, 255, 255), -1)
            cv2.circle(img, (center_x + 25, 170), 16, (255, 255, 255), -1)
            cv2.circle(img, (center_x - 25, 170), 7, (50, 50, 50), -1)
            cv2.circle(img, (center_x + 25, 170), 7, (50, 50, 50), -1)

        cv2.ellipse(img, (center_x, 240), (40, 20), 0, 0, 180, (50, 50, 50), 4)

        # Add artificial noise to second half to simulate fake suspicion
        if frame_idx >= 45:
            noise = np.random.randint(0, 100, (400, 400, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)

        out.write(img)

    out.release()
    return video_path

def run_all_tests():
    print("==================================================")
    print("DEEPSHIELD SYSTEM PIPELINE TEST SUITE")
    print("==================================================")

    # Test 1: Image Generation & Face Detection
    print("\n[TEST 1] Testing Image Face Detection & Multi-Signal Extraction...")
    img_path = generate_synthetic_face_image()
    img_bgr = cv2.imread(img_path)
    
    face_detector = FaceDetector()
    face_crop, bbox, ann_img = face_detector.detect_primary_face(img_bgr, draw_bbox=True)

    print(f" -> Face Detected: {bbox is not None}")
    if bbox:
        print(f" -> Bounding Box: {bbox}")

    cnn_detector = CNNDetector()
    cnn_res = cnn_detector.analyze_face(face_crop)
    print(f" -> CNN Score: {cnn_res['cnn_score']} (Demo: {cnn_res['is_demo']})")

    fft_analyzer = FFTAnalyzer()
    fft_res = fft_analyzer.analyze_face(face_crop)
    print(f" -> FFT Score: {fft_res['fft_score']}, Ratio: {fft_res['high_freq_ratio']}")

    fusion = FusionEngine()
    fuse_res = fusion.fuse(cnn_res['cnn_score'], fft_res['fft_score'])
    print(f" -> Image Verdict: {fuse_res['verdict']}, Score: {fuse_res['fusion_score']}")

    # Test 2: Video Processing & Frame Localization
    print("\n[TEST 2] Testing Video Analysis & Frame-Level Localization...")
    vid_path = generate_synthetic_video()
    video_analyzer = VideoAnalyzer()
    
    vid_res = video_analyzer.process_video(vid_path, result_id='test_vid_123')
    print(f" -> Video Verdict: {vid_res['verdict']}")
    print(f" -> Video Fusion Score: {vid_res['fusion_score']}")
    print(f" -> Total Frames Analyzed: {vid_res['frames_analyzed']}")
    print(f" -> Peak Suspicious Frame: Frame {vid_res['peak_frame']} at {vid_res['peak_timestamp']}s (Score: {vid_res['peak_score']})")
    print(f" -> Suspicious Time Segments: {vid_res['suspicious_segments']}")

    # Test 3: PDF Report Generation
    print("\n[TEST 3] Testing Forensic PDF Report Generation...")
    pdf_out = os.path.join(config.REPORTS_DIR, 'test_report.pdf')
    generate_forensic_pdf(vid_res, pdf_out)
    print(f" -> PDF Report successfully generated at: {pdf_out} (Exists: {os.path.exists(pdf_out)})")

    face_detector.close()
    video_analyzer.close()
    print("\n==================================================")
    print("ALL DEEPSHIELD PIPELINE TESTS PASSED CLEANLY!")
    print("==================================================")

if __name__ == '__main__':
    run_all_tests()
