import numpy as np
import cv2
import mediapipe as mp
import config

class BlinkAnalyzer:
    """
    Behavioral Evidence Analyzer using MediaPipe FaceMesh.
    Calculates Eye Aspect Ratio (EAR) across video frames to detect unnatural blinking patterns.
    """
    # MediaPipe FaceMesh eye landmark indices
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """
        Calculates Eye Aspect Ratio (EAR) for a single eye.
        """
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append(np.array([lm.x * img_w, lm.y * img_h]))

        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])

        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def analyze_frame_ear(self, frame_bgr):
        """
        Calculates average EAR for a single video frame.
        Returns EAR (float) or None if facial landmarks are undetected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES, w, h)

        avg_ear = (left_ear + right_ear) / 2.0
        return round(avg_ear, 4)

    def evaluate_video_blinking(self, ear_sequence, video_duration_sec):
        """
        Evaluates a sequence of EAR values collected across sampled frames.
        Returns:
            dict: {
                'blink_score': float (0.0 to 1.0),
                'total_blinks': int,
                'blinks_per_minute': float,
                'available': bool,
                'status': str
            }
        """
        valid_ears = [ear for ear in ear_sequence if ear is not None]
        
        if len(valid_ears) < 5:
            return {
                'blink_score': 0.50, # Neutral score when insufficient landmark data
                'total_blinks': 0,
                'blinks_per_minute': 0.0,
                'available': False,
                'status': 'Eye blinking analysis unavailable due to low landmark confidence.'
            }

        # Count blinks (EAR dropping below threshold 0.21)
        is_closed = False
        blink_count = 0

        for ear in valid_ears:
            if ear < config.EAR_THRESHOLD:
                if not is_closed:
                    blink_count += 1
                    is_closed = True
            else:
                is_closed = False

        # Calculate blinks per minute (Normal human average: 12-20 blinks/min)
        duration_min = max(0.1, video_duration_sec / 60.0)
        bpm = blink_count / duration_min

        # Calculate EAR variance across frames
        ear_variance = float(np.var(valid_ears))

        # Deepfake anomaly heuristics:
        # 1. Extremely low blink rate (< 4 blinks/min in videos > 8 sec)
        # 2. Unnatural zero variance (rigid non-blinking eyes)
        # 3. Hyperactive/flickering blinks (> 35 blinks/min)
        if bpm < 3.0 and video_duration_sec >= 8.0:
            blink_score = 0.85  # Suspicious missing blinks
        elif bpm > 35.0:
            blink_score = 0.75  # Suspicious eye flicker
        elif ear_variance < 0.0003 and video_duration_sec >= 5.0:
            blink_score = 0.80  # Suspicious static eye state
        else:
            # Normal eye behavior range
            blink_score = max(0.1, 0.40 - (ear_variance * 50.0))

        blink_score = round(max(0.0, min(1.0, float(blink_score))), 4)

        return {
            'blink_score': blink_score,
            'total_blinks': blink_count,
            'blinks_per_minute': round(bpm, 1),
            'available': True,
            'status': f'Analyzed {len(valid_ears)} frames. Blink Rate: {round(bpm, 1)} bpm.'
        }

    def close(self):
        self.face_mesh.close()
