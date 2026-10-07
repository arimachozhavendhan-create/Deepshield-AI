import os
import cv2
import numpy as np
import config

from detection.face_detector import FaceDetector
from detection.cnn_detector import CNNDetector
from detection.fft_analyzer import FFTAnalyzer
from detection.blink_analyzer import BlinkAnalyzer
from detection.fusion_engine import FusionEngine
from detection.frame_localizer import FrameLocalizer
from utils.media_utils import extract_video_metadata

class VideoAnalyzer:
    """
    Video Detection Pipeline Manager.
    Performs frame sampling, multi-signal feature extraction across frames,
    temporal blinking aggregation, frame-level localization, peak frame annotation,
    and report generation dataset creation.
    """
    def __init__(self):
        self.face_detector = FaceDetector()
        self.cnn_detector = CNNDetector()
        self.fft_analyzer = FFTAnalyzer()
        self.blink_analyzer = BlinkAnalyzer()
        self.fusion_engine = FusionEngine()
        self.localizer = FrameLocalizer()

    def process_video(self, video_path, result_id):
        """
        Executes full forensic video pipeline.
        """
        meta = extract_video_metadata(video_path)
        if meta is None:
            raise ValueError("Unable to read video file or unsupported video codec.")

        cap = cv2.VideoCapture(video_path)
        fps = meta['fps']
        total_frames = meta['total_frames']
        duration = meta['duration']

        # Determine frame sample rate
        sample_rate = config.FRAME_SAMPLE_RATE
        if total_frames > 600:
            sample_rate = max(10, total_frames // 60) # Keep around 60 analyzed samples max for speed

        frame_records = []
        ear_sequence = []
        raw_frame_cache = {}  # {frame_number: frame_bgr}

        frame_idx = 0
        analyzed_count = 0

        print(f"[VideoAnalyzer] Processing {meta['filename']} ({total_frames} frames, duration {duration}s)...")

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # 1. Temporal EAR landmark extraction
                ear = self.blink_analyzer.analyze_frame_ear(frame_bgr)
                ear_sequence.append(ear)

                # 2. Face Detection
                face_crop, bbox, _ = self.face_detector.detect_primary_face(frame_bgr, draw_bbox=False)

                if face_crop is not None:
                    # CNN Visual Analysis
                    cnn_res = self.cnn_detector.analyze_face(face_crop)
                    cnn_score = cnn_res['cnn_score']

                    # FFT Frequency Analysis
                    fft_res = self.fft_analyzer.analyze_face(face_crop)
                    fft_score = fft_res['fft_score']
                    has_face = True
                else:
                    cnn_score = 0.35  # Neutral fallback if face absent in frame
                    fft_score = 0.35
                    has_face = False

                frame_records.append({
                    'frame_number': frame_idx,
                    'timestamp': round(timestamp, 2),
                    'cnn_score': cnn_score,
                    'fft_score': fft_score,
                    'ear': ear,
                    'has_face': has_face
                })

                # Cache raw frame for peak frame export if suspicious
                raw_frame_cache[frame_idx] = frame_bgr.copy()
                analyzed_count += 1

            frame_idx += 1

        cap.release()

        # 3. Evaluate Video Eye Blinking Temporal Signal
        blink_res = self.blink_analyzer.evaluate_video_blinking(ear_sequence, duration)
        blink_score = blink_res['blink_score']

        # 4. Perform final 3-Signal Fusion for every sampled frame
        for rec in frame_records:
            fuse_res = self.fusion_engine.fuse(
                rec['cnn_score'],
                rec['fft_score'],
                blink_score
            )
            rec['blink_score'] = blink_score
            rec['fusion_score'] = fuse_res['fusion_score']

        # 5. Localize Suspicious Frames & Segments
        loc_results = self.localizer.localize(frame_records)
        peak_frame_num = loc_results['peak_frame']
        peak_score = loc_results['peak_score']

        # 6. Extract & Annotate Peak Suspicious Frame
        peak_frame_saved_path = None
        if peak_frame_num in raw_frame_cache:
            peak_raw = raw_frame_cache[peak_frame_num]
            is_suspicious_peak = peak_score >= config.SUSPICION_THRESHOLD
            _, _, annotated_peak = self.face_detector.detect_primary_face(
                peak_raw, draw_bbox=True, is_suspicious=is_suspicious_peak
            )
            
            peak_img_filename = f"{result_id}_peak_frame.jpg"
            peak_frame_saved_path = os.path.join(config.RESULTS_DIR, peak_img_filename)
            cv2.imwrite(peak_frame_saved_path, annotated_peak)

        # 7. Extract Top Suspicious Frames Gallery
        sorted_records = sorted(frame_records, key=lambda x: x['fusion_score'], reverse=True)
        top_frames_gallery = []
        for top_rec in sorted_records[:4]:  # Top 4 frames
            fn = top_rec['frame_number']
            if fn in raw_frame_cache:
                raw_img = raw_frame_cache[fn]
                _, _, ann_img = self.face_detector.detect_primary_face(
                    raw_img, draw_bbox=True, is_suspicious=(top_rec['fusion_score'] >= config.SUSPICION_THRESHOLD)
                )
                gallery_filename = f"{result_id}_frame_{fn}.jpg"
                gallery_path = os.path.join(config.RESULTS_DIR, gallery_filename)
                cv2.imwrite(gallery_path, ann_img)
                top_frames_gallery.append({
                    'frame_number': fn,
                    'timestamp': top_rec['timestamp'],
                    'score': top_rec['fusion_score'],
                    'image_filename': gallery_filename
                })

        # 8. Compute Overall Video Verdict & Aggregated Fusion Score
        # Give higher weight to top 20% suspicious frames in video
        all_fusion_scores = [r['fusion_score'] for r in frame_records]
        if all_fusion_scores:
            top_k = max(1, int(len(all_fusion_scores) * 0.25))
            top_scores = sorted(all_fusion_scores, reverse=True)[:top_k]
            overall_video_score = 0.6 * np.mean(top_scores) + 0.4 * np.median(all_fusion_scores)
        else:
            overall_video_score = 0.0

        overall_video_score = round(max(0.0, min(1.0, float(overall_video_score))), 4)
        overall_verdict = self.fusion_engine._determine_verdict(overall_video_score)

        # Calculate average sub-signal scores for display
        avg_cnn = round(float(np.mean([r['cnn_score'] for r in frame_records])), 4)
        avg_fft = round(float(np.mean([r['fft_score'] for r in frame_records])), 4)

        explanation = self.fusion_engine._generate_explanation(
            avg_cnn, avg_fft, blink_score, overall_video_score, overall_verdict
        )

        return {
            'id': result_id,
            'media_type': 'video',
            'filename': meta['filename'],
            'duration': meta['duration'],
            'total_frames': meta['total_frames'],
            'frames_analyzed': analyzed_count,
            'verdict': overall_verdict,
            'fusion_score': overall_video_score,
            'cnn_score': avg_cnn,
            'fft_score': avg_fft,
            'blink_score': blink_score,
            'blink_details': blink_res,
            'peak_frame': peak_frame_num,
            'peak_timestamp': loc_results['peak_timestamp'],
            'peak_score': peak_score,
            'peak_frame_filename': f"{result_id}_peak_frame.jpg" if peak_frame_saved_path else None,
            'peak_frame_path': peak_frame_saved_path,
            'suspicious_frames_count': loc_results['suspicious_frames_count'],
            'suspicious_segments': loc_results['suspicious_segments'],
            'top_gallery': top_frames_gallery,
            'timeline_data': frame_records,
            'explanation': explanation,
            'is_demo': self.cnn_detector.custom_model_loaded is False
        }

    def close(self):
        self.face_detector.close()
        self.blink_analyzer.close()
